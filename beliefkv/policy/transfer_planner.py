from __future__ import annotations

from dataclasses import dataclass

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.residency import ResidencyClass, ResidencyClassifier
from beliefkv.policy.shadow_controller import ShadowController, ShadowSignals
from beliefkv.predictor.types import RemainingTimePrediction
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandKind,
    CommandQueueClass,
    ControlCommand,
    PhysicalResidency,
)


@dataclass(frozen=True)
class TransferPlannerConfig:
    reserve_hbm_bytes: int = 1 << 30
    urgent_chunk_bytes: int = 256 * 1024 * 1024
    prefetch_chunk_bytes: int = 256 * 1024 * 1024
    prefetch_horizon_ms: float = 50.0
    prefetch_enabled: bool = True


class ReactiveTransferPlanner:
    """Event-driven pressure/prefetch planner; prediction is optional."""

    def __init__(
        self,
        graph: RuntimeCausalContextGraph,
        page_index: PageOwnershipIndex,
        classifier: ResidencyClassifier,
        frontier: CausalFrontierScheduler,
        shadow: ShadowController,
        config: TransferPlannerConfig | None = None,
    ) -> None:
        self.graph = graph
        self.page_index = page_index
        self.classifier = classifier
        self.frontier = frontier
        self.shadow = shadow
        self.config = config or TransferPlannerConfig()
        self._sequence = 0

    def plan_next(
        self,
        *,
        now_ms: float,
        hbm_capacity_bytes: int,
        actual_hbm_used_bytes: int | None = None,
        admission_required_bytes: int = 0,
        protected_context_id: str | None = None,
        allow_frontier_spill: bool = False,
        blocked_context_epochs: set[tuple[str, int]] | None = None,
        signals: ShadowSignals,
        predictions: dict[str, RemainingTimePrediction] | None = None,
    ) -> ControlCommand | None:
        predictions = predictions or {}
        required_free = self.config.reserve_hbm_bytes + admission_required_bytes
        physical_used = max(
            self.page_index.gpu_bytes,
            actual_hbm_used_bytes if actual_hbm_used_bytes is not None else 0,
        )
        actual_free = max(0, hbm_capacity_bytes - physical_used)
        shortage = max(0, required_free - actual_free)
        if shortage > 0:
            command = self._pressure_command(
                now_ms,
                shortage,
                protected_context_id=protected_context_id,
                allow_frontier_spill=allow_frontier_spill,
                blocked_context_epochs=blocked_context_epochs or set(),
            )
            if command is not None:
                return command

        if self.config.prefetch_enabled:
            prefetch = self._prefetch_command(
                now_ms, actual_free - self.config.reserve_hbm_bytes, predictions
            )
            if prefetch is not None:
                return prefetch
        return self.shadow.plan(now_ms, signals, predictions)

    def _pressure_command(
        self,
        now_ms: float,
        shortage: int,
        *,
        protected_context_id: str | None,
        allow_frontier_spill: bool,
        blocked_context_epochs: set[tuple[str, int]],
    ) -> ControlCommand | None:
        if any(
            not page.owner_contexts and page.gpu_resident
            for page in self.page_index.pages.values()
        ):
            return self._command(
                CommandKind.DROP_UNOWNED,
                now_ms,
                target_bytes=min(shortage, self.config.urgent_chunk_bytes),
                priority=1.0e9,
            )

        victims: list[tuple[tuple, str, int]] = []
        for context_id, context in self.graph.contexts.items():
            if (context_id, context.epoch) in blocked_context_epochs:
                continue
            assessment = self.classifier.context(context_id, now_ms)
            if assessment.residency_class not in {
                ResidencyClass.PARKED,
                ResidencyClass.DEAD_UNOWNED,
            }:
                continue
            pages = [
                page
                for page in self.page_index.context_pages(context_id)
                if page.gpu_resident
            ]
            if not pages:
                continue
            clean_bytes = sum(
                page.size_bytes
                for page in pages
                if page.residency == PhysicalResidency.DUAL_CLEAN
            )
            total_bytes = sum(page.size_bytes for page in pages)
            distance = max(
                (
                    self.frontier.ancestor_distance_to_active_leaf(item.invocation_id)
                    for item in self.graph.context_invocations(context_id)
                ),
                default=0,
            )
            dead = assessment.residency_class == ResidencyClass.DEAD_UNOWNED
            wait_tool = any(
                item.state == InvocationState.WAIT_TOOL
                for item in self.graph.context_invocations(context_id)
            )
            rank = (
                -int(dead),
                -int(clean_bytes > 0),
                -distance,
                -int(wait_tool),
                assessment.since_ms,
                -total_bytes,
                context_id,
            )
            victims.append((rank, context_id, total_bytes))
        if not victims:
            if not allow_frontier_spill or protected_context_id is None:
                return None
            return self._frontier_spill_command(
                now_ms,
                shortage,
                protected_context_id=protected_context_id,
                blocked_context_epochs=blocked_context_epochs,
            )
        _, context_id, total_bytes = min(victims)
        return self._command(
            CommandKind.OFFLOAD_CONTEXT,
            now_ms,
            context_id=context_id,
            context_epoch=self.graph.contexts[context_id].epoch,
            target_bytes=min(shortage, total_bytes, self.config.urgent_chunk_bytes),
            priority=1.0e8,
        )

    def _frontier_spill_command(
        self,
        now_ms: float,
        shortage: int,
        *,
        protected_context_id: str,
        blocked_context_epochs: set[tuple[str, int]],
    ) -> ControlCommand | None:
        protected = self.graph.contexts.get(protected_context_id)
        protected_workflow_id = protected.workflow_id if protected is not None else None
        candidates: list[tuple[tuple, str, int]] = []
        for context_id, context in self.graph.contexts.items():
            if context_id == protected_context_id:
                continue
            if (context_id, context.epoch) in blocked_context_epochs:
                continue
            assessment = self.classifier.context(context_id, now_ms)
            if assessment.residency_class != ResidencyClass.IMMINENT:
                continue
            pages = [
                page
                for page in self.page_index.context_pages(context_id)
                if page.gpu_resident
            ]
            total_bytes = sum(page.size_bytes for page in pages)
            if total_bytes == 0:
                continue
            rank = (
                int(context.workflow_id == protected_workflow_id),
                -total_bytes,
                -assessment.since_ms,
                context_id,
            )
            candidates.append((rank, context_id, total_bytes))
        if not candidates:
            return None
        _, context_id, total_bytes = min(candidates)
        return self._command(
            CommandKind.OFFLOAD_CONTEXT,
            now_ms,
            context_id=context_id,
            context_epoch=self.graph.contexts[context_id].epoch,
            target_bytes=min(shortage, total_bytes, self.config.urgent_chunk_bytes),
            priority=1.0e8,
            metadata={
                "reason": "admission_liveness_frontier_spill",
                "allow_ready_owners": True,
                "protected_context_id": protected_context_id,
            },
        )

    def _prefetch_command(
        self,
        now_ms: float,
        available_bytes: int,
        predictions: dict[str, RemainingTimePrediction],
    ) -> ControlCommand | None:
        if available_bytes <= 0:
            return None
        candidates: list[tuple[tuple, str, int]] = []
        for context_id, context in self.graph.contexts.items():
            assessment = self.classifier.context(context_id, now_ms)
            pages = [
                page
                for page in self.page_index.context_pages(context_id)
                if page.residency == PhysicalResidency.CPU_ONLY
            ]
            bytes_needed = sum(page.size_bytes for page in pages)
            if bytes_needed == 0 or bytes_needed > available_bytes:
                continue
            prediction = predictions.get(context_id)
            predicted_imminent = (
                prediction is not None
                and prediction.usable
                and prediction.p50_ms <= self.config.prefetch_horizon_ms
            )
            if assessment.residency_class != ResidencyClass.IMMINENT and not predicted_imminent:
                continue
            rank = (
                -int(assessment.residency_class == ResidencyClass.IMMINENT),
                prediction.p50_ms if prediction is not None else 0.0,
                bytes_needed,
                context_id,
            )
            candidates.append((rank, context_id, bytes_needed))
        if not candidates:
            return None
        _, context_id, bytes_needed = min(candidates)
        return self._command(
            CommandKind.PREFETCH_CONTEXT,
            now_ms,
            context_id=context_id,
            context_epoch=self.graph.contexts[context_id].epoch,
            target_bytes=min(bytes_needed, self.config.prefetch_chunk_bytes),
            priority=1.0e7,
        )

    def _command(
        self,
        kind: CommandKind,
        now_ms: float,
        *,
        context_id: str | None = None,
        context_epoch: int | None = None,
        target_bytes: int,
        priority: float,
        metadata: dict | None = None,
    ) -> ControlCommand:
        command = ControlCommand(
            command_id=f"reactive-{self._sequence}",
            kind=kind,
            created_ts_ms=now_ms,
            context_id=context_id,
            context_epoch=context_epoch,
            target_bytes=target_bytes,
            priority=priority,
            queue_class=CommandQueueClass.URGENT,
            metadata=metadata or {},
        )
        self._sequence += 1
        return command
