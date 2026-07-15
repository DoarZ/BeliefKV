from __future__ import annotations

from dataclasses import dataclass

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.residency import ResidencyClass, ResidencyClassifier
from beliefkv.predictor.types import RemainingTimePrediction
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandKind,
    CommandQueueClass,
    ControlCommand,
    PhysicalResidency,
)


@dataclass(frozen=True)
class ShadowSignals:
    urgent_queue_depth: int
    pcie_utilization: float
    gpu_compute_utilization: float
    measured_inference_slowdown: float
    hbm_pressure: float
    host_free_bytes: int


@dataclass
class ShadowConfig:
    min_parked_ms: float = 25.0
    chunk_bytes: int = 64 * 1024 * 1024
    min_chunk_bytes: int = 4 * 1024 * 1024
    max_chunk_bytes: int = 128 * 1024 * 1024
    max_pcie_utilization: float = 0.55
    max_gpu_utilization: float = 0.92
    slowdown_budget: float = 0.02
    high_hbm_pressure: float = 0.95
    host_reserve_bytes: int = 1 << 30
    min_wait_survival_probability: float = 0.55


class ShadowController:
    """Non-destructive PREPARE controller with interference feedback."""

    def __init__(
        self,
        graph: RuntimeCausalContextGraph,
        page_index: PageOwnershipIndex,
        classifier: ResidencyClassifier,
        frontier: CausalFrontierScheduler,
        config: ShadowConfig | None = None,
    ) -> None:
        self.graph = graph
        self.page_index = page_index
        self.classifier = classifier
        self.frontier = frontier
        self.config = config or ShadowConfig()
        self._command_sequence = 0

    def plan(
        self,
        now_ms: float,
        signals: ShadowSignals,
        predictions: dict[str, RemainingTimePrediction] | None = None,
    ) -> ControlCommand | None:
        if not self._can_submit(signals):
            return None
        predictions = predictions or {}
        candidates: list[tuple[tuple, str, int]] = []
        for context_id, context in self.graph.contexts.items():
            assessment = self.classifier.context(context_id, now_ms)
            if assessment.residency_class != ResidencyClass.PARKED:
                continue
            parked_ms = max(0.0, now_ms - assessment.since_ms)
            if parked_ms < self.config.min_parked_ms:
                continue
            pages = [
                page
                for page in self.page_index.context_pages(context_id)
                if page.residency == PhysicalResidency.GPU_ONLY
                and page.sealed
                and page.transfer_idle
            ]
            bytes_available = sum(page.size_bytes for page in pages)
            if bytes_available == 0:
                continue
            prediction = predictions.get(context_id)
            if prediction is not None and prediction.usable:
                wait_survival = 1.0 - prediction.resume_within_transfer_probability
                if wait_survival < self.config.min_wait_survival_probability:
                    continue
                confidence_rank = prediction.confidence * wait_survival
            else:
                confidence_rank = 0.0
            stack_distance = self._continuation_distance(context_id)
            tool_wait = any(
                item.state == InvocationState.WAIT_TOOL
                for item in self.graph.context_invocations(context_id)
            )
            persistent_wait = any(
                item.state == InvocationState.WAIT_MESSAGE
                for item in self.graph.context_invocations(context_id)
            )
            rank = (
                -int(stack_distance > 0),
                -stack_distance,
                -int(tool_wait),
                -confidence_rank,
                -parked_ms,
                -int(not persistent_wait),
                -bytes_available,
                context_id,
            )
            candidates.append((rank, context_id, bytes_available))
        if not candidates:
            return None
        _, context_id, bytes_available = min(candidates)
        context = self.graph.contexts[context_id]
        target = min(self.config.chunk_bytes, bytes_available)
        command = ControlCommand(
            command_id=f"shadow-{self._command_sequence}",
            kind=CommandKind.SHADOW_CONTEXT,
            created_ts_ms=now_ms,
            context_id=context_id,
            context_epoch=context.epoch,
            target_bytes=target,
            priority=0.0,
            queue_class=CommandQueueClass.SHADOW,
            metadata={"non_destructive": True},
        )
        self._command_sequence += 1
        return command

    def observe_interference(self, measured_slowdown: float) -> None:
        if measured_slowdown > self.config.slowdown_budget:
            self.config.chunk_bytes = max(
                self.config.min_chunk_bytes, self.config.chunk_bytes // 2
            )
        elif measured_slowdown < self.config.slowdown_budget / 2:
            self.config.chunk_bytes = min(
                self.config.max_chunk_bytes,
                self.config.chunk_bytes + self.config.min_chunk_bytes,
            )

    def _can_submit(self, signals: ShadowSignals) -> bool:
        return (
            signals.urgent_queue_depth == 0
            and signals.pcie_utilization <= self.config.max_pcie_utilization
            and signals.gpu_compute_utilization <= self.config.max_gpu_utilization
            and signals.measured_inference_slowdown <= self.config.slowdown_budget
            and signals.hbm_pressure < self.config.high_hbm_pressure
            and signals.host_free_bytes
            >= self.config.host_reserve_bytes + self.config.min_chunk_bytes
        )

    def _continuation_distance(self, context_id: str) -> int:
        return max(
            (
                self.frontier.ancestor_distance_to_active_leaf(item.invocation_id)
                for item in self.graph.context_invocations(context_id)
            ),
            default=0,
        )
