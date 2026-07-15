from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from beliefkv.control.causal_graph import GraphDelta, RuntimeCausalContextGraph
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.admission import (
    AdmissionController,
    AdmissionDecision,
    AdmissionRequest,
)
from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.residency import ResidencyClassifier
from beliefkv.policy.shadow_controller import ShadowConfig, ShadowController, ShadowSignals
from beliefkv.policy.transfer_cost import PCIeCostModel
from beliefkv.policy.transfer_planner import ReactiveTransferPlanner, TransferPlannerConfig
from beliefkv.policy.workflow_fairness import WorkflowFairScheduler
from beliefkv.predictor.composer import RemainingTimePredictor
from beliefkv.predictor.types import RemainingTimePrediction
from beliefkv.runtime.command_queue import TransferCommandQueue
from beliefkv.runtime.page_index import PageIndexError, PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandStatus,
    ControlCommand,
    PageHandle,
    PhysicalPageAction,
    ResolvedCommand,
    TransferDirection,
)
from beliefkv.runtime.radix_arbiter import ArbitrationConfig, RadixArbiter


@dataclass(frozen=True)
class ControllerTickResult:
    now_ms: float
    admission: AdmissionDecision | None = None
    transfer: ResolvedCommand | None = None
    cancel_command_ids: tuple[str, ...] = ()
    local_acks: tuple[CommandAck, ...] = ()
    stalled_command_ids: tuple[str, ...] = ()
    predictions: dict[str, RemainingTimePrediction] = field(default_factory=dict)


@dataclass
class _InFlightCommand:
    resolved: ResolvedCommand
    started_handles: set[PageHandle] = field(default_factory=set)


class BeliefKVController:
    """End-to-end BeliefKV control plane independent of CUDA/SGLang internals."""

    def __init__(
        self,
        config: BeliefKVConfig | None = None,
        *,
        predictor: RemainingTimePredictor | None = None,
    ) -> None:
        self.config = config or BeliefKVConfig()
        self.graph = RuntimeCausalContextGraph()
        self.page_index = PageOwnershipIndex()
        self.frontier = CausalFrontierScheduler(self.graph)
        self.classifier = ResidencyClassifier(self.graph, self.page_index)
        self.fairness = WorkflowFairScheduler()
        self.admission = AdmissionController(
            self.page_index,
            self.fairness,
            self.frontier,
            reserve_hbm_bytes=self.config.reserve_hbm_bytes,
        )
        self.shadow = ShadowController(
            self.graph,
            self.page_index,
            self.classifier,
            self.frontier,
            ShadowConfig(
                min_parked_ms=self.config.shadow_min_parked_ms,
                chunk_bytes=self.config.shadow_chunk_bytes,
                min_chunk_bytes=min(4 * 1024 * 1024, self.config.shadow_chunk_bytes),
                max_chunk_bytes=max(
                    self.config.shadow_chunk_bytes, 2 * self.config.shadow_chunk_bytes
                ),
                slowdown_budget=self.config.shadow_slowdown_budget,
                host_reserve_bytes=min(1 << 30, self.config.host_capacity_bytes // 8),
            ),
        )
        self.transfer_planner = ReactiveTransferPlanner(
            self.graph,
            self.page_index,
            self.classifier,
            self.frontier,
            self.shadow,
            TransferPlannerConfig(
                reserve_hbm_bytes=self.config.reserve_hbm_bytes,
                urgent_chunk_bytes=self.config.urgent_chunk_bytes,
                prefetch_chunk_bytes=self.config.urgent_chunk_bytes,
                prefetch_enabled=self.config.prefetch_enabled,
            ),
        )
        self.command_queue = TransferCommandQueue()
        self.arbiter = RadixArbiter(
            self.graph,
            self.page_index,
            ArbitrationConfig(
                shadow_chunk_bytes=self.config.shadow_chunk_bytes,
                urgent_chunk_bytes=self.config.urgent_chunk_bytes,
            ),
        )
        self.predictor = predictor or (
            RemainingTimePredictor.load(Path(self.config.predictor_model_path))
            if self.config.predictor_model_path
            else RemainingTimePredictor()
        )
        self.cost_model = PCIeCostModel(
            bandwidth_gbps=self.config.pcie_bandwidth_gbps,
            overhead_ms=self.config.transfer_overhead_ms,
        )
        self.now_ms = 0.0
        self.signals = ShadowSignals(
            urgent_queue_depth=0,
            pcie_utilization=0.0,
            gpu_compute_utilization=0.0,
            measured_inference_slowdown=0.0,
            hbm_pressure=0.0,
            host_free_bytes=self.config.host_capacity_bytes,
        )
        self._inflight: dict[str, _InFlightCommand] = {}
        self._queued_by_context: dict[str, str] = {}
        self._pending_cancellations: set[str] = set()
        self._blocked_context_epochs: set[tuple[str, int]] = set()
        self.command_history: list[ControlCommand] = []
        self.ack_history: list[CommandAck] = []
        self._reported_hbm_used_bytes: int | None = None
        self._engine_request_count: int | None = None
        self._external_workflow_charges: dict[str, float] = {}
        self._last_predictions: dict[str, RemainingTimePrediction] = {}

    def process_runtime_event(self, event: RuntimeEvent) -> GraphDelta:
        self.now_ms = max(self.now_ms, event.ts_ms)
        delta = self.graph.apply(event)
        self._after_runtime_event(event, delta)
        return delta

    def process_runtime_events(
        self, events: list[RuntimeEvent] | tuple[RuntimeEvent, ...]
    ) -> list[GraphDelta]:
        if not events:
            return []
        deltas = self.graph.apply_batch(events, atomic=True)
        for event, delta in zip(events, deltas):
            self.now_ms = max(self.now_ms, event.ts_ms)
            self._after_runtime_event(event, delta)
        return deltas

    def _after_runtime_event(self, event: RuntimeEvent, delta: GraphDelta) -> None:
        if self.config.predictor_enabled:
            self.predictor.observe_event(event)
        if event.kind == RuntimeEventKind.WORKFLOW_START:
            self.fairness.register(event.workflow_id)
        for context_id in delta.changed_contexts:
            context = self.graph.contexts[context_id]
            if not self.page_index.has_context(context_id):
                self.page_index.register_context(
                    context_id, context.workflow_id, context.epoch
                )
            elif self.page_index.context_epoch(context_id) != context.epoch:
                self.page_index.update_context_epoch(context_id, context.epoch)
            self._blocked_context_epochs.discard((context_id, context.epoch))
        for invocation_id in delta.awakened_invocations:
            context_id = self.graph.invocations[invocation_id].context_id
            prediction = self._last_predictions.pop(context_id, None)
            if prediction is not None and event.ts_ms >= prediction.generated_ts_ms:
                self.predictor.calibrator.observe(
                    prediction, event.ts_ms - prediction.generated_ts_ms
                )
            self._cancel_shadow_for_context(context_id)

    def submit_request(self, request: AdmissionRequest) -> None:
        context = self.graph.contexts.get(request.context_id)
        if context is None or context.epoch != request.context_epoch:
            raise ValueError("request refers to an unknown or stale context")
        invocation = self.graph.invocations.get(request.invocation_id)
        if invocation is None or invocation.workflow_id != request.workflow_id:
            raise ValueError("request refers to an unknown invocation/workflow")
        self.admission.enqueue(request)

    def acknowledge_admission(self, request_id: str) -> int:
        return self.admission.acknowledge(request_id)

    def update_signals(
        self,
        *,
        pcie_utilization: float | None = None,
        gpu_compute_utilization: float | None = None,
        measured_inference_slowdown: float | None = None,
        host_free_bytes: int | None = None,
    ) -> None:
        def bounded(value: float, field_name: str) -> float:
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be in [0, 1]")
            return value

        self.signals = ShadowSignals(
            urgent_queue_depth=self.command_queue.urgent_count,
            pcie_utilization=(
                bounded(pcie_utilization, "pcie_utilization")
                if pcie_utilization is not None
                else self.signals.pcie_utilization
            ),
            gpu_compute_utilization=(
                bounded(gpu_compute_utilization, "gpu_compute_utilization")
                if gpu_compute_utilization is not None
                else self.signals.gpu_compute_utilization
            ),
            measured_inference_slowdown=(
                max(0.0, measured_inference_slowdown)
                if measured_inference_slowdown is not None
                else self.signals.measured_inference_slowdown
            ),
            hbm_pressure=self.actual_hbm_used_bytes / self.config.hbm_capacity_bytes,
            host_free_bytes=(
                max(0, host_free_bytes)
                if host_free_bytes is not None
                else max(0, self.config.host_capacity_bytes - self.page_index.cpu_bytes)
            ),
        )
        self.shadow.observe_interference(self.signals.measured_inference_slowdown)

    def report_hbm_usage(
        self,
        used_bytes: int,
        *,
        workflow_charges: dict[str, float] | None = None,
    ) -> None:
        if not 0 <= used_bytes <= self.config.hbm_capacity_bytes:
            raise ValueError("reported HBM usage must be within configured capacity")
        self._reported_hbm_used_bytes = used_bytes
        if workflow_charges is not None:
            if any(value < 0 for value in workflow_charges.values()):
                raise ValueError("workflow HBM charges must be non-negative")
            self._external_workflow_charges = dict(workflow_charges)

    def report_engine_activity(self, request_count: int) -> None:
        if request_count < 0:
            raise ValueError("engine request count must be non-negative")
        self._engine_request_count = request_count

    @property
    def actual_hbm_used_bytes(self) -> int:
        return max(
            self.page_index.gpu_bytes,
            self._reported_hbm_used_bytes
            if self._reported_hbm_used_bytes is not None
            else 0,
        )

    def workflow_memory_charges(self) -> dict[str, float]:
        charges = self.page_index.workflow_gpu_charges()
        for workflow_id, value in self._external_workflow_charges.items():
            charges[workflow_id] = charges.get(workflow_id, 0.0) + value
        return charges

    def tick(self, now_ms: float | None = None) -> ControllerTickResult:
        if now_ms is not None:
            if now_ms < self.now_ms:
                raise ValueError("controller time cannot move backwards")
            self.now_ms = now_ms
        self.classifier.release_terminal_owners()
        self.update_signals()
        predictions = self._predictions()

        pending_requests = self.admission.pending_requests()
        liveness_target = None
        if self._engine_request_count == 0 and pending_requests:
            oldest = pending_requests[0]
            if (
                self.now_ms - oldest.submitted_ts_ms
                >= self.config.admission_liveness_timeout_ms
            ):
                liveness_target = oldest

        allow_reserve_borrow = (
            self._engine_request_count == 0
            and self.admission.reserved_bytes == 0
            and not self._inflight
            and len(self.command_queue) == 0
        )
        stalled_command_ids = self._stalled_command_ids()
        force_preferred_progress = bool(
            liveness_target is not None
            and self._engine_request_count == 0
            and self.admission.reserved_bytes == 0
            and (not self._inflight or stalled_command_ids)
            and self.now_ms - liveness_target.submitted_ts_ms
            >= self.config.admission_force_progress_timeout_ms
        )
        admission = self.admission.decide_next(
            self.config.hbm_capacity_bytes,
            actual_hbm_used_bytes=self.actual_hbm_used_bytes,
            external_workflow_charges=self._external_workflow_charges,
            allow_reserve_borrow=allow_reserve_borrow,
            preferred_request_id=(
                liveness_target.request_id if liveness_target is not None else None
            ),
            force_preferred_progress=force_preferred_progress,
        )
        required = 0
        protected_context_id = None
        if admission is not None and not admission.admitted:
            pending = {
                item.request_id: item for item in self.admission.pending_requests()
            }
            request = pending.get(admission.request_id)
            if request is not None:
                required = request.estimated_incremental_bytes
                if liveness_target is not None:
                    protected_context_id = request.context_id

        if not self._inflight:
            planned = self.transfer_planner.plan_next(
                now_ms=self.now_ms,
                hbm_capacity_bytes=self.config.hbm_capacity_bytes,
                actual_hbm_used_bytes=self.actual_hbm_used_bytes,
                admission_required_bytes=required,
                protected_context_id=protected_context_id,
                allow_frontier_spill=protected_context_id is not None,
                blocked_context_epochs=set(self._blocked_context_epochs),
                signals=self.signals,
                predictions=predictions,
            )
            if (
                planned is not None
                and (self.config.shadow_enabled or planned.kind != CommandKind.SHADOW_CONTEXT)
                and (
                planned.context_id is None
                or planned.context_epoch is None
                or (planned.context_id, planned.context_epoch)
                not in self._blocked_context_epochs
                )
            ):
                self._enqueue_if_new(planned)

        transfer, local_ack = self._dispatch_next()
        cancellations = tuple(sorted(self._pending_cancellations))
        self._pending_cancellations.clear()
        return ControllerTickResult(
            now_ms=self.now_ms,
            admission=admission,
            transfer=transfer,
            cancel_command_ids=cancellations,
            local_acks=(local_ack,) if local_ack is not None else (),
            stalled_command_ids=stalled_command_ids,
            predictions=predictions,
        )

    def _stalled_command_ids(self) -> tuple[str, ...]:
        stalled: list[str] = []
        for command_id, inflight in self._inflight.items():
            command = inflight.resolved.command
            expected_ms = self.cost_model.transfer_ms(inflight.resolved.resolved_bytes)
            timeout_ms = max(
                self.config.transfer_watchdog_floor_ms,
                expected_ms * self.config.transfer_watchdog_factor,
            )
            if self.now_ms - command.created_ts_ms >= timeout_ms:
                stalled.append(command_id)
        return tuple(sorted(stalled))

    def mark_command_started(
        self, command_id: str, handles: tuple[PageHandle, ...] | list[PageHandle]
    ) -> None:
        inflight = self._require_inflight(command_id)
        actions = {item.handle: item for item in inflight.resolved.page_actions}
        for handle in handles:
            if handle in inflight.started_handles:
                continue
            action = actions.get(handle)
            if action is None:
                raise ValueError(f"page {handle} is not part of command {command_id}")
            if action.action == PhysicalPageAction.START_D2H:
                self.page_index.begin_transfer(handle, TransferDirection.D2H)
            elif action.action == PhysicalPageAction.START_H2D:
                self.page_index.begin_transfer(handle, TransferDirection.H2D)
            inflight.started_handles.add(handle)

    def unblock_context(self, context_id: str) -> None:
        self._blocked_context_epochs = {
            item for item in self._blocked_context_epochs if item[0] != context_id
        }

    def acknowledge_command(self, ack: CommandAck) -> None:
        inflight = self._require_inflight(ack.command_id)
        actions = {item.handle: item for item in inflight.resolved.page_actions}
        completed = set(ack.page_handles)
        unknown_handles = completed - set(actions)
        if unknown_handles:
            raise ValueError(
                f"ACK contains pages outside command {ack.command_id}: "
                f"{sorted(unknown_handles)}"
            )
        if ack.status in {CommandStatus.REJECTED, CommandStatus.STALE, CommandStatus.CANCELLED}:
            completed = set()
        expected_bytes = sum(actions[handle].size_bytes for handle in completed)
        if ack.actual_bytes > expected_bytes:
            raise ValueError(
                f"ACK actual_bytes {ack.actual_bytes} exceeds selected page bytes "
                f"{expected_bytes}"
            )
        for handle, action in actions.items():
            if handle not in completed:
                if (
                    handle in inflight.started_handles
                    and action.action
                    in {
                        PhysicalPageAction.START_D2H,
                        PhysicalPageAction.START_H2D,
                    }
                ):
                    self.page_index.abort_transfer(handle)
                continue
            if action.action == PhysicalPageAction.START_D2H:
                if handle not in inflight.started_handles:
                    raise ValueError(f"D2H page {handle} completed before TRANSFER_START")
                keep_gpu = inflight.resolved.command.kind == CommandKind.SHADOW_CONTEXT
                self.page_index.complete_transfer(
                    handle, TransferDirection.D2H, keep_gpu=keep_gpu
                )
            elif action.action == PhysicalPageAction.START_H2D:
                if handle not in inflight.started_handles:
                    raise ValueError(f"H2D page {handle} completed before TRANSFER_START")
                self.page_index.complete_transfer(handle, TransferDirection.H2D)
            elif action.action == PhysicalPageAction.COMMIT_CPU:
                self.page_index.commit_cpu(handle)
            elif action.action == PhysicalPageAction.DROP:
                self.page_index.drop_page(handle)
            elif action.action == PhysicalPageAction.PIN:
                context_id = inflight.resolved.command.context_id
                if context_id is not None:
                    self.page_index.pages[handle].semantic_pin_contexts.add(context_id)
            elif action.action == PhysicalPageAction.UNPIN:
                context_id = inflight.resolved.command.context_id
                if context_id is not None:
                    self.page_index.pages[handle].semantic_pin_contexts.discard(context_id)

        command = inflight.resolved.command
        self._inflight.pop(ack.command_id)
        if command.context_id is not None:
            self._queued_by_context.pop(command.context_id, None)
        self.ack_history.append(ack)
        self.page_index.assert_consistent()
        self.update_signals()

    def _dispatch_next(
        self,
    ) -> tuple[ResolvedCommand | None, CommandAck | None]:
        if self._inflight:
            return None, None
        command = self.command_queue.pop(allow_shadow=self.config.shadow_enabled)
        if command is None:
            return None, None
        resolved = self.arbiter.resolve(command)
        self.command_history.append(command)
        if not resolved.page_actions:
            if command.context_id is not None and command.context_epoch is not None:
                self._blocked_context_epochs.add(
                    (command.context_id, command.context_epoch)
                )
                self._queued_by_context.pop(command.context_id, None)
            ack = CommandAck(
                command_id=command.command_id,
                status=(
                    CommandStatus.STALE
                    if "stale" in resolved.reason or "epoch" in resolved.reason
                    else CommandStatus.REJECTED
                ),
                completed_ts_ms=self.now_ms,
                actual_bytes=0,
                reason=resolved.reason,
            )
            self.ack_history.append(ack)
            return None, ack
        self._inflight[command.command_id] = _InFlightCommand(resolved)
        return resolved, None

    def _enqueue_if_new(self, command: ControlCommand) -> bool:
        if command.context_id is not None:
            if command.context_id in self._queued_by_context:
                return False
            self._queued_by_context[command.context_id] = command.command_id
        self.command_queue.put(command)
        return True

    def _cancel_shadow_for_context(self, context_id: str) -> None:
        queued_id = self._queued_by_context.get(context_id)
        if queued_id is not None and self.command_queue.cancel(queued_id):
            self._queued_by_context.pop(context_id, None)
            self._pending_cancellations.add(queued_id)
        for command_id, inflight in self._inflight.items():
            command = inflight.resolved.command
            if (
                command.context_id == context_id
                and command.kind == CommandKind.SHADOW_CONTEXT
            ):
                self._pending_cancellations.add(command_id)

    def _predictions(self) -> dict[str, RemainingTimePrediction]:
        if not self.config.predictor_enabled:
            return {}
        windows: dict[str, float] = {}
        for context_id in self.graph.contexts:
            size = sum(
                page.size_bytes for page in self.page_index.context_pages(context_id)
            )
            windows[context_id] = self.cost_model.transfer_ms(size)
        predictions = self.predictor.predict_all(
            self.graph, now_ms=self.now_ms, transfer_windows_ms=windows
        )
        self._last_predictions.update(predictions)
        return predictions

    def _require_inflight(self, command_id: str) -> _InFlightCommand:
        try:
            return self._inflight[command_id]
        except KeyError as exc:
            raise KeyError(f"unknown in-flight command: {command_id}") from exc

    @property
    def inflight_command_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._inflight))
