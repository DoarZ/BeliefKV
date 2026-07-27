from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from beliefkv.control.causal_graph import GraphDelta, RuntimeCausalContextGraph
from beliefkv.control.data_consumers import ObservedDataConsumerIndex
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.admission import (
    AdmissionController,
    AdmissionDecision,
    AdmissionRequest,
    AdmissionSideState,
    AdmissionTicketCompiler,
    VisibleAdmissionEntry,
    VisibleAdmissionIndex,
)
from beliefkv.policy.leases import CausalLeaseProjector
from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.residency import ResidencyClassifier
from beliefkv.policy.reference.base import (
    CapabilityReport,
    IdentityMapping,
    MetadataMode,
    MetadataValue,
    PolicyInput,
    RunnableInvocation,
)
from beliefkv.policy.reference.snapshot_builder import PolicyInputSnapshotBuilder
from beliefkv.policy.resource_snapshot import RuntimeResourceObservation
from beliefkv.policy.shadow_controller import ShadowConfig, ShadowController, ShadowSignals
from beliefkv.policy.transfer_cost import PCIeCostModel
from beliefkv.policy.transfer_guard import TransferAttemptGuard, TransferGuardEvent
from beliefkv.policy.service_curve import TransferServiceCurve
from beliefkv.policy.transfer_planner import ReactiveTransferPlanner, TransferPlannerConfig
from beliefkv.policy.workflow_fairness import WorkflowFairScheduler
from beliefkv.predictor.composer import RemainingTimePredictor
from beliefkv.predictor.types import RemainingTimePrediction
from beliefkv.runtime.command_queue import TransferCommandQueue
from beliefkv.runtime.bundles import BundlePreviewEvent, PhysicalBundleBuilder
from beliefkv.runtime.page_index import PageIndexError, PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandQueueClass,
    CommandStatus,
    ControlCommand,
    PageHandle,
    PhysicalPageAction,
    ResolvedCommand,
    TransferDirection,
    TransferBlocker,
    TransferBlockerCode,
    TransferTelemetry,
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
    transfer_guard_events: tuple[TransferGuardEvent, ...] = ()
    bundle_preview_events: tuple[BundlePreviewEvent, ...] = ()


@dataclass(frozen=True)
class RuntimeEventChangeSet:
    from_sequence: int
    to_sequence: int
    events: tuple[RuntimeEvent, ...]
    full_rebuild_required: bool = False


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
        self.data_consumers = ObservedDataConsumerIndex(self.graph)
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
        # The embedded SGLang path uses this request-ID side index. The legacy
        # AdmissionController remains available to the standalone simulator,
        # but does not own runtime request objects or reserve runtime HBM.
        self.visible_admission = VisibleAdmissionIndex()
        self.admission_ticket_compiler = AdmissionTicketCompiler()
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
        self.lease_projector = CausalLeaseProjector(self.graph)
        self.bundle_builder = PhysicalBundleBuilder(
            self.graph,
            self.page_index,
            self.lease_projector,
        )
        self.transfer_guard = TransferAttemptGuard(
            self.graph,
            self.page_index,
            enabled=self.config.transfer_retry_guard_enabled,
            max_same_snapshot_attempts=(
                self.config.transfer_retry_max_same_snapshot_attempts
            ),
            unknown_base_ms=self.config.transfer_retry_unknown_base_ms,
            unknown_max_ms=self.config.transfer_retry_unknown_max_ms,
            unknown_circuit_breaker_failures=(
                self.config.transfer_retry_unknown_circuit_breaker_failures
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
            retry_guard=self.transfer_guard,
            bundle_builder=self.bundle_builder,
        )
        self.command_queue = TransferCommandQueue()
        self.arbiter = RadixArbiter(
            self.graph,
            self.page_index,
            ArbitrationConfig(
                shadow_chunk_bytes=self.config.shadow_chunk_bytes,
                urgent_chunk_bytes=self.config.urgent_chunk_bytes,
            ),
            bundle_builder=self.bundle_builder,
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
        self.service_curve = TransferServiceCurve(
            self.cost_model,
            window=self.config.service_curve_window,
            min_samples=self.config.service_curve_min_samples,
        )
        self.policy_snapshot_builder = PolicyInputSnapshotBuilder(
            self.graph,
            self.data_consumers,
            self.page_index,
            self.admission,
            self.lease_projector,
            self.service_curve,
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
        self.command_history: list[ControlCommand] = []
        self.ack_history: list[CommandAck] = []
        self._acked_command_ids: set[str] = set()
        self.transfer_telemetry_history: list[TransferTelemetry] = []
        self._reported_hbm_used_bytes: int | None = None
        self._engine_request_count: int | None = None
        self._running_request_count: int | None = None
        self._external_workflow_charges: dict[str, float] = {}
        self._last_predictions: dict[str, RemainingTimePrediction] = {}
        self._drop_unowned_blocked = False
        self._native_admission_request_id: str | None = None
        self._native_admission_capacity_bytes = 0
        self._pcie_utilization_observed = False
        self._gpu_compute_utilization_observed = False
        self._transfer_epoch = 0
        self._transition_by_workflow: dict[str, dict[str, object]] = {}
        self._terminal_cleanup_handles: dict[str, set[PageHandle]] = {}
        self._runtime_event_sequence = 0
        self._runtime_event_journal: deque[tuple[int, RuntimeEvent]] = deque(
            maxlen=262_144
        )

    def process_runtime_event(self, event: RuntimeEvent) -> GraphDelta:
        self.now_ms = max(self.now_ms, event.ts_ms)
        delta = self.graph.apply(event)
        self.data_consumers.apply(event)
        self._observe_transition_batch((event,))
        self._after_runtime_event(event, delta)
        self._append_runtime_events((event,))
        return delta

    def process_runtime_events(
        self, events: list[RuntimeEvent] | tuple[RuntimeEvent, ...]
    ) -> list[GraphDelta]:
        if not events:
            return []
        deltas = self.graph.apply_batch(events, atomic=True)
        self.data_consumers.apply_batch(events, atomic=True)
        self._observe_transition_batch(tuple(events))
        for event, delta in zip(events, deltas):
            self.now_ms = max(self.now_ms, event.ts_ms)
            self._after_runtime_event(event, delta)
        self._append_runtime_events(tuple(events))
        return deltas

    @property
    def runtime_event_sequence(self) -> int:
        return self._runtime_event_sequence

    def runtime_events_since(self, sequence: int) -> RuntimeEventChangeSet:
        if sequence < 0 or sequence > self._runtime_event_sequence:
            raise ValueError("runtime event sequence is outside the valid range")
        if sequence == self._runtime_event_sequence:
            return RuntimeEventChangeSet(sequence, sequence, ())
        reverse_records: list[tuple[int, RuntimeEvent]] = []
        for item in reversed(self._runtime_event_journal):
            if item[0] <= sequence:
                break
            reverse_records.append(item)
        records = tuple(reversed(reverse_records))
        full_rebuild = not records or records[0][0] != sequence + 1
        return RuntimeEventChangeSet(
            from_sequence=sequence,
            to_sequence=self._runtime_event_sequence,
            events=tuple(item[1] for item in records),
            full_rebuild_required=full_rebuild,
        )

    def _append_runtime_events(self, events: tuple[RuntimeEvent, ...]) -> None:
        for event in events:
            self._runtime_event_sequence += 1
            # Detach the worker journal from adapter-owned attribute mappings.
            detached = RuntimeEvent.from_dict(copy.deepcopy(event.to_dict()))
            self._runtime_event_journal.append(
                (self._runtime_event_sequence, detached)
            )

    def _after_runtime_event(self, event: RuntimeEvent, delta: GraphDelta) -> None:
        self.notify_resource_state_changed()
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
            self.transfer_guard.invalidate_context(
                context_id, now_ms=self.now_ms, keep_epoch=context.epoch
            )
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

    def register_visible_request(
        self,
        request: AdmissionRequest,
        *,
        transition_generation: int = 0,
        bundle_generations: Mapping[str, str] | None = None,
    ) -> VisibleAdmissionEntry:
        """Track a native-queue request without taking queue or HBM ownership."""

        context = self.graph.contexts.get(request.context_id)
        if context is None or context.epoch != request.context_epoch:
            raise ValueError("request refers to an unknown or stale context")
        invocation = self.graph.invocations.get(request.invocation_id)
        if invocation is None or invocation.workflow_id != request.workflow_id:
            raise ValueError("request refers to an unknown invocation/workflow")
        self.fairness.register(request.workflow_id)
        return self.visible_admission.register(
            request,
            transition_generation=transition_generation,
            bundle_generations=bundle_generations,
        )

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

        if pcie_utilization is not None:
            self._pcie_utilization_observed = True
        if gpu_compute_utilization is not None:
            self._gpu_compute_utilization_observed = True
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

    def report_engine_activity(
        self,
        request_count: int,
        *,
        running_request_count: int | None = None,
    ) -> None:
        if request_count < 0:
            raise ValueError("engine request count must be non-negative")
        running = request_count if running_request_count is None else running_request_count
        if running < 0 or running > request_count:
            raise ValueError(
                "running request count must be between zero and engine request count"
            )
        self._engine_request_count = request_count
        self._running_request_count = running

    def notify_resource_state_changed(self) -> None:
        """Allow a previously impossible global reclaim to be reconsidered."""

        self._drop_unowned_blocked = False
        self._native_admission_request_id = None
        self._native_admission_capacity_bytes = 0

    def report_native_admission_capacity(
        self,
        request_id: str | None,
        capacity_bytes: int = 0,
    ) -> None:
        """Publish a request-specific, scheduler-verified reclaim budget."""

        if capacity_bytes < 0:
            raise ValueError("native admission capacity must be non-negative")
        if request_id is None and capacity_bytes != 0:
            raise ValueError("capacity without a request id is invalid")
        self._native_admission_request_id = request_id
        self._native_admission_capacity_bytes = capacity_bytes

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

    def external_workflow_memory_charges(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self._external_workflow_charges.items()))

    def build_policy_input(
        self,
        observation: RuntimeResourceObservation,
        *,
        additional_runnable: Sequence[RunnableInvocation] = (),
        identity_mappings: Sequence[IdentityMapping] = (),
        optional_metadata: Mapping[str, MetadataValue] | None = None,
        capabilities: CapabilityReport | None = None,
        metadata_mode: MetadataMode = MetadataMode.ONLINE,
    ) -> PolicyInput:
        """Build one read-only common-policy snapshot at a runtime safe point."""

        urgent_d2h, urgent_h2d = self.transfer_backlog_bytes()
        observation = replace(
            observation,
            urgent_d2h_bytes=urgent_d2h,
            urgent_h2d_bytes=urgent_h2d,
        )
        return self.policy_snapshot_builder.build(
            observation,
            additional_runnable=additional_runnable,
            workflow_memory_charges=self.workflow_memory_charges(),
            control_state=self.policy_control_state(observation.ts_ms),
            identity_mappings=identity_mappings,
            optional_metadata=optional_metadata,
            transfer_telemetry=tuple(self.transfer_telemetry_history),
            capabilities=capabilities,
            metadata_mode=metadata_mode,
        )

    def policy_control_state(self, now_ms: float) -> dict[str, object]:
        for workflow_id, state in self._transition_by_workflow.items():
            opened_ts_ms = state.get("opened_ts_ms")
            if (
                state.get("open")
                and isinstance(opened_ts_ms, (int, float))
                and now_ms - float(opened_ts_ms)
                >= self.config.joint_transition_settling_timeout_ms
            ):
                state["open"] = False
                state["degraded"] = True
                state["generation"] = int(state["generation"]) + 1
                state["closed_ts_ms"] = now_ms
        return {
            "transfer_epoch": self._transfer_epoch,
            "transitions": {
                workflow_id: dict(state)
                for workflow_id, state in sorted(
                    self._transition_by_workflow.items()
                )
            },
        }

    def _observe_transition_batch(
        self, events: tuple[RuntimeEvent, ...]
    ) -> None:
        relevant = {
            RuntimeEventKind.INVOCATION_CREATE,
            RuntimeEventKind.MESSAGE,
            RuntimeEventKind.HANDOFF,
            RuntimeEventKind.REACTIVATE,
            RuntimeEventKind.JOIN_SATISFIED,
        }
        by_workflow: dict[str, list[RuntimeEvent]] = {}
        for event in events:
            if (
                event.kind in relevant
                or event.attributes.get("transition_open")
                or event.attributes.get("transition_close")
            ):
                by_workflow.setdefault(event.workflow_id, []).append(event)
        for workflow_id, workflow_events in by_workflow.items():
            previous = self._transition_by_workflow.get(
                workflow_id,
                {
                    "generation": 0,
                    "open": False,
                    "degraded": False,
                    "opened_ts_ms": None,
                    "closed_ts_ms": None,
                },
            )
            state = dict(previous)
            state["generation"] = int(state["generation"]) + 1
            explicitly_open = any(
                bool(event.attributes.get("transition_open"))
                for event in workflow_events
            )
            explicitly_closed = any(
                bool(event.attributes.get("transition_close"))
                for event in workflow_events
            )
            if explicitly_open and not explicitly_closed:
                state["open"] = True
                state["degraded"] = False
                state["opened_ts_ms"] = min(
                    event.ts_ms for event in workflow_events
                )
                state["closed_ts_ms"] = None
            else:
                state["open"] = False
                state["degraded"] = False
                state["opened_ts_ms"] = None
                state["closed_ts_ms"] = max(
                    event.ts_ms for event in workflow_events
                )
            self._transition_by_workflow[workflow_id] = state

    def _bump_transfer_epoch(self) -> None:
        self._transfer_epoch += 1

    def transfer_backlog_bytes(self) -> tuple[int, int]:
        """Return urgent D2H/H2D bytes queued or awaiting ACK."""

        commands = [
            item
            for item in self.command_queue.pending_commands()
            if item.queue_class == CommandQueueClass.URGENT
        ]
        commands.extend(
            item.resolved.command
            for item in self._inflight.values()
            if item.resolved.command.queue_class == CommandQueueClass.URGENT
        )
        d2h = 0
        h2d = 0
        for command in commands:
            bundle = command.physical_bundle
            if bundle is None:
                continue
            for action in bundle.page_actions:
                if action.action == PhysicalPageAction.START_D2H:
                    d2h += action.size_bytes
                elif action.action == PhysicalPageAction.START_H2D:
                    h2d += action.size_bytes
        return d2h, h2d

    def tick(self, now_ms: float | None = None) -> ControllerTickResult:
        if now_ms is not None:
            if now_ms < self.now_ms:
                raise ValueError("controller time cannot move backwards")
            self.now_ms = now_ms
        self.classifier.release_terminal_owners(
            on_release=self._record_terminal_cleanup_handles
        )
        self._prune_terminal_cleanup_handles()
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

        reserved_liveness_target = None
        reserved_requests = self.admission.reserved_requests()
        if self._running_request_count == 0 and reserved_requests:
            oldest_reserved = reserved_requests[0]
            if (
                self.now_ms - oldest_reserved.submitted_ts_ms
                >= self.config.admission_liveness_timeout_ms
            ):
                reserved_liveness_target = oldest_reserved

        allow_reserve_borrow = (
            self._engine_request_count == 0
            and self.admission.reserved_bytes == 0
            and not self._inflight
            and len(self.command_queue) == 0
        )
        stalled_command_ids = self._stalled_command_ids()
        native_reclaim_ready = bool(
            liveness_target is not None
            and self._engine_request_count == 0
            and self.admission.reserved_bytes == 0
            and not self._inflight
            and len(self.command_queue) == 0
            and self.now_ms - liveness_target.submitted_ts_ms
            >= self.config.admission_force_progress_timeout_ms
            and self._native_admission_request_id == liveness_target.request_id
        )
        admission = self.admission.decide_next(
            self.config.hbm_capacity_bytes,
            actual_hbm_used_bytes=self.actual_hbm_used_bytes,
            external_workflow_charges=self._external_workflow_charges,
            allow_reserve_borrow=allow_reserve_borrow,
            preferred_request_id=(
                liveness_target.request_id if liveness_target is not None else None
            ),
            native_reclaim_capacity_bytes=(
                self._native_admission_capacity_bytes
                if native_reclaim_ready
                else None
            ),
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
        elif reserved_liveness_target is not None:
            required = reserved_liveness_target.estimated_incremental_bytes
            protected_context_id = reserved_liveness_target.context_id
        elif not pending_requests and not reserved_requests:
            # Runtime-visible requests have no logical reservation. Publish only
            # the oldest request's immediate pressure to the reactive transfer
            # writer; native PrefillAdder remains the final capacity authority.
            visible_pending = sorted(
                (
                    entry
                    for entry in self.visible_admission.entries()
                    if entry.state == AdmissionSideState.VISIBLE_PENDING
                ),
                key=lambda entry: (
                    entry.request.submitted_ts_ms,
                    entry.request.request_id,
                ),
            )
            if visible_pending:
                oldest_visible = visible_pending[0].request
                allocatable_free = max(
                    0,
                    self.config.hbm_capacity_bytes
                    - self.config.reserve_hbm_bytes
                    - self.actual_hbm_used_bytes,
                )
                if oldest_visible.estimated_incremental_bytes > allocatable_free:
                    required = oldest_visible.estimated_incremental_bytes
                    protected_context_id = oldest_visible.context_id

        delegated_native_reclaim = bool(
            admission is not None
            and admission.admitted
            and admission.reason == "admission_liveness_native_reclaim"
        )
        self.transfer_guard.update_resources(
            device_available_bytes=max(
                0,
                self.config.hbm_capacity_bytes
                - self.actual_hbm_used_bytes
                - self.admission.reserved_bytes,
            ),
            host_available_bytes=self.signals.host_free_bytes,
            now_ms=self.now_ms,
        )
        restore_entries = tuple(
            entry
            for entry in self.visible_admission.entries()
            if entry.state == AdmissionSideState.WAIT_RESTORE
        )
        restore_workflow_order = self.fairness.ordered(
            {entry.request.workflow_id for entry in restore_entries},
            memory_charges=self.workflow_memory_charges(),
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
        )
        restore_workflow_rank = {
            workflow_id: rank
            for rank, workflow_id in enumerate(restore_workflow_order)
        }
        ordered_restore_entries = sorted(
            restore_entries,
            key=lambda entry: (
                self.now_ms - entry.request.submitted_ts_ms
                < self.config.admission_force_progress_timeout_ms,
                restore_workflow_rank.get(entry.request.workflow_id, 1 << 30),
                entry.request.submitted_ts_ms,
                entry.request.request_id,
            ),
        )
        preferred_restore_context_ids = tuple(
            dict.fromkeys(entry.request.context_id for entry in ordered_restore_entries)
        )
        if not self._inflight and not delegated_native_reclaim:
            planned = self._plan_terminal_cleanup()
            if planned is None:
                planned = self.transfer_planner.plan_next(
                    now_ms=self.now_ms,
                    hbm_capacity_bytes=self.config.hbm_capacity_bytes,
                    actual_hbm_used_bytes=self.actual_hbm_used_bytes,
                    reserved_hbm_bytes=self.admission.reserved_bytes,
                    admission_required_bytes=required,
                    protected_context_id=protected_context_id,
                    allow_frontier_spill=protected_context_id is not None,
                    drop_unowned_enabled=not self._drop_unowned_blocked,
                    signals=self.signals,
                    predictions=predictions,
                    preferred_restore_context_ids=preferred_restore_context_ids,
                )
            if (
                planned is not None
                and (self.config.shadow_enabled or planned.kind != CommandKind.SHADOW_CONTEXT)
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
            transfer_guard_events=self.transfer_guard.drain_events(),
            bundle_preview_events=self.transfer_planner.drain_bundle_events(),
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
        if handles:
            self._bump_transfer_epoch()

    def acknowledge_command(self, ack: CommandAck) -> None:
        self.now_ms = max(self.now_ms, ack.completed_ts_ms)
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
            elif action.action == PhysicalPageAction.DROP_HOST:
                self.page_index.drop_host_copy(handle)
            elif action.action == PhysicalPageAction.PIN:
                context_id = inflight.resolved.command.context_id
                if context_id is not None:
                    self.page_index.pages[handle].semantic_pin_contexts.add(context_id)
            elif action.action == PhysicalPageAction.UNPIN:
                context_id = inflight.resolved.command.context_id
                if context_id is not None:
                    self.page_index.pages[handle].semantic_pin_contexts.discard(context_id)

        command = inflight.resolved.command
        if ack.status in {
            CommandStatus.REJECTED,
            CommandStatus.PARTIAL,
            CommandStatus.STALE,
        }:
            blockers = ack.blockers
            if not blockers:
                blockers = (
                    TransferBlocker(
                        TransferBlockerCode.STALE_GENERATION
                        if ack.status == CommandStatus.STALE
                        else TransferBlockerCode.UNKNOWN_BACKEND,
                        detail=ack.reason,
                    ),
                )
            required_retry_bytes = inflight.resolved.resolved_bytes
            if ack.status == CommandStatus.PARTIAL:
                failed_handles = {
                    item.page_handle
                    for item in blockers
                    if item.page_handle is not None
                }
                failed_action_bytes = sum(
                    action.size_bytes
                    for handle, action in actions.items()
                    if handle in failed_handles
                )
                if failed_action_bytes > 0:
                    required_retry_bytes = failed_action_bytes
            self.transfer_guard.record_failure(
                ack.command_id,
                blockers=blockers,
                required_bytes=required_retry_bytes,
                now_ms=ack.completed_ts_ms,
            )
        elif ack.status == CommandStatus.COMPLETED:
            self.transfer_guard.record_success(
                ack.command_id, now_ms=ack.completed_ts_ms
            )
        else:
            self.transfer_guard.cancel_attempt(ack.command_id)
        self._inflight.pop(ack.command_id)
        self._bump_transfer_epoch()
        if command.context_id is not None:
            self._queued_by_context.pop(command.context_id, None)
        self.ack_history.append(ack)
        self._acked_command_ids.add(ack.command_id)
        self.notify_resource_state_changed()
        self.page_index.assert_consistent()
        self.update_signals()

    def _record_terminal_cleanup_handles(
        self, context_id: str, handles: frozenset[PageHandle]
    ) -> None:
        if handles:
            self._terminal_cleanup_handles.setdefault(context_id, set()).update(
                handles
            )

    def _prune_terminal_cleanup_handles(self) -> None:
        for context_id, handles in tuple(self._terminal_cleanup_handles.items()):
            remaining = {
                handle
                for handle in handles
                if (page := self.page_index.pages.get(handle)) is not None
                and page.cpu_resident
                and not page.owner_contexts
            }
            if remaining:
                self._terminal_cleanup_handles[context_id] = remaining
            else:
                self._terminal_cleanup_handles.pop(context_id, None)

    def _plan_terminal_cleanup(self) -> ControlCommand | None:
        for context_id in sorted(self._terminal_cleanup_handles):
            if context_id in self._queued_by_context:
                continue
            context = self.graph.contexts.get(context_id)
            if context is None:
                continue
            command = self.transfer_planner.plan_terminal_cleanup(
                now_ms=self.now_ms,
                context_id=context_id,
                context_epoch=context.epoch,
                target_handles=tuple(
                    sorted(self._terminal_cleanup_handles[context_id])
                ),
            )
            if command is not None:
                return command
        return None

    def observe_transfer_telemetry(self, telemetry: TransferTelemetry) -> None:
        """Update performance models after the correctness ACK was committed."""

        if telemetry.command_id not in self._acked_command_ids:
            raise ValueError(
                f"transfer telemetry arrived before ACK: {telemetry.command_id}"
            )
        self.service_curve.observe(telemetry)
        self.transfer_telemetry_history.append(telemetry)

    def _dispatch_next(
        self,
    ) -> tuple[ResolvedCommand | None, CommandAck | None]:
        if self._inflight:
            return None, None
        while True:
            command = self.command_queue.pop(allow_shadow=self.config.shadow_enabled)
            if command is None:
                return None, None
            self._bump_transfer_epoch()
            if self.transfer_guard.command_is_eligible(command, now_ms=self.now_ms):
                break
            if command.context_id is not None:
                self._queued_by_context.pop(command.context_id, None)
        closure_fingerprint = self.transfer_guard.begin_attempt(
            command, now_ms=self.now_ms
        )
        resolved = self.arbiter.resolve(command)
        if closure_fingerprint:
            resolved = replace(
                resolved, closure_fingerprint=closure_fingerprint
            )
        self.command_history.append(command)
        if not resolved.page_actions:
            if command.kind == CommandKind.DROP_UNOWNED:
                self._drop_unowned_blocked = True
            if command.context_id is not None and command.context_epoch is not None:
                self._queued_by_context.pop(command.context_id, None)
            self.transfer_guard.record_failure(
                command.command_id,
                blockers=resolved.blockers,
                required_bytes=max(resolved.resolved_bytes, command.target_bytes),
                now_ms=self.now_ms,
            )
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
                blockers=resolved.blockers,
            )
            self.ack_history.append(ack)
            return None, ack
        self._inflight[command.command_id] = _InFlightCommand(resolved)
        return resolved, None

    def reset_transfer_attempts(self) -> None:
        self.transfer_guard.reset(now_ms=self.now_ms)
        self._bump_transfer_epoch()

    def enqueue_control_command(self, command: ControlCommand) -> bool:
        """Queue one externally compiled, versioned physical command.

        Runtime policies may use this entry point only from the scheduler safe
        point. The normal per-context de-duplication and transfer epoch rules
        remain authoritative.
        """

        return self._enqueue_if_new(command)

    def _enqueue_if_new(self, command: ControlCommand) -> bool:
        if command.context_id is not None:
            if command.context_id in self._queued_by_context:
                return False
            self._queued_by_context[command.context_id] = command.command_id
        self.command_queue.put(command)
        self._bump_transfer_epoch()
        return True

    def _cancel_shadow_for_context(self, context_id: str) -> None:
        queued_id = self._queued_by_context.get(context_id)
        if queued_id is not None and self.command_queue.cancel(queued_id):
            self._queued_by_context.pop(context_id, None)
            self._pending_cancellations.add(queued_id)
            self._bump_transfer_epoch()
        for command_id, inflight in self._inflight.items():
            command = inflight.resolved.command
            if (
                command.context_id == context_id
                and command.kind == CommandKind.SHADOW_CONTEXT
            ):
                self._pending_cancellations.add(command_id)
                self._bump_transfer_epoch()

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
