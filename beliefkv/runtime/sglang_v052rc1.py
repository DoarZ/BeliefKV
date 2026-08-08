from __future__ import annotations

import atexit
import copy
import gc
import inspect
import json
import os
import signal
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import (
    ContextMode,
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.metrics.summary import percentile
from beliefkv.policy.admission import (
    AdmissionCompileBudget,
    AdmissionRequest,
    AdmissionSideState,
    AdmissionTicket,
    AdmissionTicketEpoch,
    ObservedAdmissionCandidate,
    ObservedAdmissionScheduler,
    ObservedAdmissionSnapshot,
    ObservedAdmissionWindow,
)
from beliefkv.policy.joint_scheduler import (
    AsyncSemanticJointPlanner,
    JointPlanComponentValidation,
    JointPlanCurrentState,
    JointPlanValidation,
    JointPlannerConfig,
    JointPlannerMode,
    SemanticResidencyTarget,
    validate_joint_plan,
    validate_joint_plan_components,
)
from beliefkv.policy.online_joint import (
    ActionSlice,
    OnlineJointPlanDecision,
    OnlineJointPlanView,
    append_committed_action_slice,
    compile_bounded_seed_epoch,
    compile_online_joint_view,
)
from beliefkv.policy.reference import (
    AdmissionAction,
    CapabilityReport,
    PolicyInput,
    ResidencyAction,
    RunnableInvocation,
)
from beliefkv.policy.reference.snapshot_builder import page_handle_from_extent_id
from beliefkv.policy.retraction import (
    ObservedRetractionConfig,
    ObservedRetractionDecision,
    ObservedRetractionPlanner,
    ObservedRetractionSnapshot,
    RetractionLockedExtent,
    RetractionReplacement,
    RunningRetractionCandidate,
    RunningRetractionPlan,
)
from beliefkv.policy.resource_snapshot import RuntimeResourceObservation
from beliefkv.predictor.online_shadow import (
    build_invocation_frontier_predictions,
)
from beliefkv.runtime.audit import (
    PolicySnapshotLog,
    RequestTokenTraceLog,
    RuntimeAuditLog,
)
from beliefkv.runtime.bundles import BundleScope, PhysicalBundlePreview
from beliefkv.runtime.event_channel import (
    JsonlRuntimeEventSink,
    RuntimeEventDatagramServer,
)
from beliefkv.runtime.joint_shadow import (
    IncrementalPolicyInputAssembler,
    JointShadowDelta,
    JointShadowResult,
    JointShadowStateStamp,
    LatestWinsJointPlanWorker,
    WorkflowFairnessReplica,
)
from beliefkv.runtime.lock_service import (
    LockServiceDiagnostics,
    LockedExtentAttribution,
    RequestServiceLedger,
    TentativeUnlockPreview,
    TentativeUnlockPreviewer,
)


def close_runtime_with_signal_shield(runtime: "EmbeddedSGLangRuntime") -> None:
    """Let the scheduler persist its bounded shutdown transaction before exit."""

    previous_handlers: dict[signal.Signals, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
            try:
                previous_handlers[signum] = signal.signal(signum, signal.SIG_IGN)
            except (OSError, ValueError):
                continue
    try:
        runtime.close()
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def install_scheduler_shutdown_handler() -> Any:
    """Convert scheduler SIGTERM into a Python unwind so ``finally`` can ACK."""

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("scheduler shutdown handler requires the main thread")

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        raise SystemExit(0)

    return signal.signal(signal.SIGTERM, handle_sigterm)


def _linux_process_start_time_ticks(pid: int) -> int | None:
    """Return /proc start time so a controller can reject a reused PID."""

    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        return int(tail.split()[19])
    except (IndexError, OSError, ValueError):
        return None


from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandQueueClass,
    CommandStatus,
    ControlCommand,
    EnqueueOutcome,
    EnqueueStatus,
    PageHandle,
    PhysicalPageAction,
    PhysicalResidency,
    ResolvedCommand,
    ResolvedPageAction,
    TransferBlocker,
    TransferBlockerCode,
    TransferDirection,
    TransferTelemetry,
)
from beliefkv.runtime.restore_obligation import (
    ExternalProgressToken,
    NativeQueueLocation,
    NativeRequestPhysicalSnapshot,
    SafePointPhysicalPhase,
    SafePointPhysicalSnapshot,
    SafePointSnapshotBuildTiming,
    RestoreAuthorityMode,
    RestoreFeasibilityCertificate,
    RestoreLease,
    RestoreLeaseIndex,
    RestoreLeaseState,
    RestoreObligation,
    RestoreObligationCause,
    RestoreObligationIndex,
    RestoreObligationState,
    PersistentLivenessRevisionTracker,
    RestoreServiceGrace,
    RestorePhysicalOperation,
    RestoreTransaction,
    RestoreTransactionStage,
)
from beliefkv.runtime.sglang_adapter import (
    BASE_SGLANG_VERSION,
    BackendSubmission,
    BeliefKVRequestMetadata,
    HiCacheCapabilities,
    SGLangSchedulerBridge,
)


def _sequence_length(value: Any) -> int:
    """Return a sequence length without evaluating tensor truthiness."""

    return 0 if value is None else len(value)


class SGLangBackendError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        blocker_code: TransferBlockerCode = TransferBlockerCode.UNKNOWN_BACKEND,
        required_bytes: int = 0,
    ) -> None:
        super().__init__(message)
        self.blocker_code = blocker_code
        self.required_bytes = required_bytes


@dataclass
class SGLangNodeRegistry:
    """Generation-checked mapping from BeliefKV handles to Radix TreeNodes."""

    cache_generation: int = 0
    _nodes: dict[PageHandle, Any] = field(default_factory=dict)
    _node_state: dict[int, tuple[tuple[Any, ...], PageHandle]] = field(
        default_factory=dict
    )
    _next_generation: int = 0

    def fingerprint(self, node: Any) -> tuple[Any, ...]:
        key = getattr(node, "key", None)
        key_length = len(key) if key is not None else 0
        first = key[0] if key_length else None
        last = key[-1] if key_length else None
        return (id(node), key_length, first, last, self.cache_generation)

    def register(
        self, node: Any, fingerprint: tuple[Any, ...] | None = None
    ) -> PageHandle:
        node_id = int(node.id)
        if fingerprint is None:
            fingerprint = self.fingerprint(node)
        previous = self._node_state.get(node_id)
        if previous is not None and previous[0] == fingerprint:
            self._nodes[previous[1]] = node
            return previous[1]
        if previous is not None:
            self._nodes.pop(previous[1], None)
        handle = PageHandle(node_id, self._next_generation)
        self._next_generation += 1
        self._nodes[handle] = node
        self._node_state[node_id] = (fingerprint, handle)
        return handle

    def resolve(self, handle: PageHandle) -> Any:
        state = self._node_state.get(handle.page_id)
        if state is not None and state[1] != handle:
            raise SGLangBackendError(
                f"stale node extent generation: {handle}",
                blocker_code=TransferBlockerCode.STALE_GENERATION,
            )
        try:
            node = self._nodes[handle]
        except KeyError as exc:
            raise SGLangBackendError(
                f"unknown or stale Radix node handle: {handle}",
                blocker_code=TransferBlockerCode.STALE_GENERATION,
            ) from exc
        if int(node.id) != handle.page_id:
            raise SGLangBackendError(
                "registry/node id divergence",
                blocker_code=TransferBlockerCode.STALE_GENERATION,
            )
        return node

    def remove(self, handle: PageHandle) -> None:
        self._nodes.pop(handle, None)
        state = self._node_state.get(handle.page_id)
        if state is not None and state[1] == handle:
            self._node_state.pop(handle.page_id, None)

    def reset(self) -> None:
        self.cache_generation += 1
        self._nodes.clear()
        self._node_state.clear()

    def current_handles(self) -> set[PageHandle]:
        return {state[1] for state in self._node_state.values()}

    def current_handle(self, node_id: int) -> PageHandle | None:
        state = self._node_state.get(node_id)
        return state[1] if state is not None else None


@dataclass
class _PendingNodeCommand:
    resolved: ResolvedCommand
    submit_ts_ms: float
    start_ts_ms: float | None = None
    accepted_handles: set[PageHandle] = field(default_factory=set)
    transfer_handles: set[PageHandle] = field(default_factory=set)
    dma_completed_handles: set[PageHandle] = field(default_factory=set)
    completed_handles: set[PageHandle] = field(default_factory=set)
    rejected_handles: set[PageHandle] = field(default_factory=set)
    rejection_reasons: dict[PageHandle, str] = field(default_factory=dict)
    rejection_blockers: dict[PageHandle, TransferBlocker] = field(
        default_factory=dict
    )
    extent_fingerprints: dict[PageHandle, tuple[Any, ...]] = field(
        default_factory=dict
    )
    h2d_expected_tokens: int = 0
    h2d_allocator_available_before: int | None = None
    h2d_allocator_available_after_submit: int | None = None
    transfer_host_copy_state: dict[TransferDirection, str] = field(
        default_factory=dict
    )
    pinned_host: bool | None = None
    native_concurrent_bytes: int = 0
    allocator_submit_ms: float = 0.0
    cancel_requested: bool = False


@dataclass
class _RunningRetractionTransaction:
    transaction_id: str
    plan: RunningRetractionPlan
    created_ts_ms: float
    barrier_intent_id: str | None = None
    tentative_unlock_preview: TentativeUnlockPreview | None = None
    stage: str = "planned"
    actual_request_ids: tuple[str, ...] = ()
    native_reclaim_capacity_after: int = 0
    actual_reclaim_capacity_bytes: int = 0
    actual_engine_lock_release_bytes: int = 0
    private_reclaim_bytes: int = 0
    allocator_available_before_bytes: int = 0
    allocator_available_after_bytes: int = 0
    required_allocator_available_bytes: int = 0
    victim_context_ids: tuple[str, ...] = ()
    pending_command_id: str | None = None
    pending_command_kind: CommandKind | None = None
    residency_command_ids: list[str] = field(default_factory=list)
    explicit_reclaim_bytes: int = 0
    explicit_transfer_bytes: int = 0
    command_attempt_count: int = 0
    failure_reason: str | None = None
    source_joint_plan_id: str | None = None
    restore_path_extent_ids: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    restore_micro_gate_id: str | None = None


@dataclass
class _OnlineJointResidencyTransaction:
    transaction_id: str
    plan_id: str
    intent_index: int
    source_bundle_id: str
    action: ResidencyAction
    command_id: str
    command_kind: CommandKind
    context_id: str
    physical_bundle_id: str
    created_ts_ms: float
    stage: str = "queued"
    completed_ts_ms: float | None = None
    actual_bytes: int = 0
    failure_reason: str | None = None


@dataclass
class _RunningRetractionBarrierAttempt:
    barrier_intent_id: str
    requested_ts_ms: float
    requested_state: dict[str, Any]
    tentative_unlock_preview: TentativeUnlockPreview | None = None
    tentative_unlock_preview_status: str = "not_attempted"
    tentative_unlock_preview_compute_us: float = 0.0
    drained_ts_ms: float | None = None
    drained_state: dict[str, Any] | None = None


class HiCacheNodeCommandBackend:
    """SGLang 0.5.2rc1 backend using its scheduler-owned HiCache queues.

    A handle maps to one sealed Radix node extent. The backend intentionally
    invokes private eviction helpers only from the scheduler thread and only
    after BeliefKV's arbiter has checked locks/ownership. The pinned source
    contract prevents silently using these hooks on another SGLang release.
    """

    def __init__(
        self,
        tree_cache: Any,
        registry: SGLangNodeRegistry,
        *,
        now_ms: Any | None = None,
        h2d_context_is_busy: Any | None = None,
        kv_bytes_per_token: int = 0,
    ) -> None:
        required = (
            "write_backup",
            "check_hicache_events",
            "_evict_backuped",
            "_evict_regular",
            "load_back",
            "ready_to_load_host_cache",
        )
        missing = [name for name in required if not hasattr(tree_cache, name)]
        if missing:
            raise SGLangBackendError(
                f"tree cache lacks required HiCache methods: {missing}"
            )
        self.tree_cache = tree_cache
        self.registry = registry
        self._now_ms = now_ms or (lambda: time.monotonic() * 1000.0)
        self._h2d_context_is_busy = h2d_context_is_busy
        self._kv_bytes_per_token = max(0, int(kv_bytes_per_token))
        self._pending: dict[str, _PendingNodeCommand] = {}
        self._acks: list[CommandAck] = []
        self._telemetry: list[TransferTelemetry] = []
        self._callback_errors: list[dict[str, object]] = []
        host_layout = str(
            getattr(getattr(tree_cache, "token_to_kv_pool_host", None), "layout", "")
        )
        self._capabilities = HiCacheCapabilities(
            operation_merge=False,
            layer_completion_events=False,
            page_first_host_layout=host_layout.startswith("page_first"),
            proactive_load_trigger=True,
            max_inflight_operations=1,
            physical_unit="node_extent",
        )

    @property
    def capabilities(self) -> HiCacheCapabilities:
        return self._capabilities

    def submit(self, command: ResolvedCommand) -> BackendSubmission:
        command_id = command.command.command_id
        if command_id in self._pending:
            raise SGLangBackendError(f"duplicate command: {command_id}")
        pending = _PendingNodeCommand(command, submit_ts_ms=float(self._now_ms()))
        pending.pinned_host = self._pinned_host_state()
        self._pending[command_id] = pending
        atomic_bundle = command.command.physical_bundle is not None
        if (
            command.command.kind == CommandKind.PREFETCH_CONTEXT
            and command.command.context_id is not None
            and self._h2d_context_is_busy is not None
            and self._h2d_context_is_busy(command.command.context_id)
        ):
            handle = (
                command.page_actions[0].handle
                if command.page_actions
                else PageHandle(0, 0)
            )
            self._reject(
                pending,
                handle,
                SGLangBackendError(
                    "H2D context already has an engine-visible request",
                    blocker_code=TransferBlockerCode.ENGINE_BUSY,
                    required_bytes=command.resolved_bytes,
                ),
            )
            self._finish(
                pending,
                status=CommandStatus.REJECTED,
                reason=self._rejection_reason(
                    pending, "atomic_bundle_preflight_rejected"
                ),
            )
            return BackendSubmission(command_id=command_id, started_handles=())
        prepared: list[tuple[ResolvedPageAction, Any]] = []
        for item in command.page_actions:
            try:
                node = self.registry.resolve(item.handle)
                pending.extent_fingerprints[item.handle] = (
                    self._extent_fingerprint(node)
                )
                prepared.append((item, node))
            except (SGLangBackendError, AssertionError, RuntimeError) as error:
                self._reject(pending, item.handle, error)

        if atomic_bundle and not pending.rejected_handles:
            assert command.command.physical_bundle is not None
            action_handles = {item.handle for item, _ in prepared}
            for handle in command.command.physical_bundle.closure_handles:
                if handle in action_handles:
                    continue
                try:
                    node = self.registry.resolve(handle)
                    pending.extent_fingerprints[handle] = (
                        self._extent_fingerprint(node)
                    )
                except (SGLangBackendError, AssertionError, RuntimeError) as error:
                    self._reject(pending, handle, error)

        if atomic_bundle and not pending.rejected_handles:
            selected_actions = {
                item.handle: item.action for item, _ in prepared
            }
            for item, node in prepared:
                try:
                    self._preflight_page(
                        node,
                        item.handle,
                        item.action,
                        item.size_bytes,
                        selected_actions=selected_actions,
                        offload_bundle=(
                            command.command.kind == CommandKind.OFFLOAD_CONTEXT
                        ),
                    )
                except (SGLangBackendError, AssertionError, RuntimeError) as error:
                    self._reject(pending, item.handle, error)
            if not pending.rejected_handles:
                try:
                    self._preflight_bundle_capacity(prepared)
                except (SGLangBackendError, AssertionError, RuntimeError) as error:
                    handle = prepared[0][0].handle if prepared else PageHandle(0, 0)
                    self._reject(pending, handle, error)

        if atomic_bundle and pending.rejected_handles:
            self._finish(
                pending,
                status=CommandStatus.REJECTED,
                reason=self._rejection_reason(
                    pending, "atomic_bundle_preflight_rejected"
                ),
            )
            return BackendSubmission(command_id=command_id, started_handles=())

        if atomic_bundle:
            drop_bundle = bool(prepared) and all(
                item.action == PhysicalPageAction.DROP for item, _ in prepared
            )
            prepared.sort(
                key=lambda pair: (self._node_depth(pair[1]), pair[0].handle),
                reverse=drop_bundle,
            )

        native_h2d_closure = atomic_bundle and prepared and all(
            item.action == PhysicalPageAction.START_H2D for item, _ in prepared
        )
        if native_h2d_closure:
            try:
                pending.start_ts_ms = float(self._now_ms())
                self._submit_atomic_h2d_closure(pending, prepared)
            except (SGLangBackendError, AssertionError, RuntimeError) as error:
                self._reject(pending, prepared[-1][0].handle, error)
        else:
            for item, node in prepared:
                try:
                    observable_start_ms = float(self._now_ms())
                    self._submit_page(
                        pending,
                        node,
                        item.handle,
                        item.action,
                        item.size_bytes,
                        atomic_bundle=atomic_bundle,
                    )
                    if (
                        item.action
                        in {
                            PhysicalPageAction.START_D2H,
                            PhysicalPageAction.START_H2D,
                        }
                        and item.handle in pending.transfer_handles
                        and pending.start_ts_ms is None
                    ):
                        pending.start_ts_ms = observable_start_ms
                except (SGLangBackendError, AssertionError, RuntimeError) as error:
                    self._reject(pending, item.handle, error)
                    if atomic_bundle:
                        break

        if atomic_bundle and pending.rejected_handles:
            for handle in pending.accepted_handles - pending.transfer_handles:
                pending.rejected_handles.add(handle)
        if any(
            item.action == PhysicalPageAction.START_H2D
            and item.handle in pending.accepted_handles
            for item in command.page_actions
        ):
            # SGLang's load_back() only queues the copy. Its normal prefill path
            # raises this event when a batch is formed, but a proactive prefetch
            # may run while admission is blocked and no batch can be formed.
            self.tree_cache.ready_to_load_host_cache()
        if not pending.accepted_handles:
            self._finish(
                pending,
                status=CommandStatus.REJECTED,
                reason=self._rejection_reason(pending, "all_node_actions_rejected"),
            )
        elif atomic_bundle and drop_bundle and not pending.rejected_handles:
            self._finish(
                pending,
                status=CommandStatus.COMPLETED,
                reason="atomic_drop_bundle_completed",
            )
        return BackendSubmission(
            command_id=command_id,
            started_handles=tuple(sorted(pending.accepted_handles)),
        )

    def _submit_atomic_h2d_closure(
        self,
        pending: _PendingNodeCommand,
        prepared: list[tuple[ResolvedPageAction, Any]],
    ) -> None:
        leaf_item, leaf_node = max(
            prepared,
            key=lambda pair: (self._node_depth(pair[1]), pair[0].handle),
        )
        expected_tokens = sum(
            self._extent_tokens(getattr(node, "host_value", None))
            for _, node in prepared
        )
        required_bytes = sum(item.size_bytes for item, _ in prepared)
        allocator = self._authoritative_device_allocator()
        available_before = self._allocator_available(allocator)
        if available_before is None:
            raise SGLangBackendError(
                "HiCache device allocator does not expose available_size",
                blocker_code=TransferBlockerCode.UNKNOWN_BACKEND,
                required_bytes=required_bytes,
            )
        if expected_tokens > available_before:
            raise SGLangBackendError(
                "atomic H2D closure exceeds authoritative free device tokens",
                blocker_code=TransferBlockerCode.DEVICE_CAPACITY,
                required_bytes=required_bytes,
            )
        pending.h2d_expected_tokens = expected_tokens
        pending.h2d_allocator_available_before = available_before
        pending.transfer_host_copy_state[TransferDirection.H2D] = "present"
        allocator_submit_started_ns = time.perf_counter_ns()
        try:
            loaded = self.tree_cache.load_back(
                leaf_node,
                force=True,
                allow_eviction=False,
                beliefkv_source="explicit",
            )
        finally:
            pending.allocator_submit_ms += (
                time.perf_counter_ns() - allocator_submit_started_ns
            ) / 1_000_000.0
        pending.native_concurrent_bytes = max(
            pending.native_concurrent_bytes, self._native_inflight_bytes()
        )
        if loaded is None:
            raise SGLangBackendError(
                "HiCache device allocation failed for atomic ancestor closure",
                blocker_code=TransferBlockerCode.DEVICE_CAPACITY,
                required_bytes=required_bytes,
            )
        handles = {item.handle for item, _ in prepared}
        pending.accepted_handles.update(handles)
        pending.transfer_handles.update(handles)
        if self._extent_tokens(loaded) != expected_tokens:
            raise SGLangBackendError(
                "HiCache H2D allocation size differs from the selected closure",
                blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                required_bytes=required_bytes,
            )
        available_after = self._allocator_available(allocator)
        pending.h2d_allocator_available_after_submit = available_after
        if (
            available_after is None
            or available_before - available_after != expected_tokens
        ):
            raise SGLangBackendError(
                "HiCache H2D allocator reservation delta is inconsistent",
                blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                required_bytes=required_bytes,
            )
        if leaf_item.handle not in handles:
            raise SGLangBackendError(
                "native H2D leaf is absent from selected closure",
                blocker_code=TransferBlockerCode.EXTENT_MUTATED,
            )

    def _preflight_page(
        self,
        node: Any,
        handle: PageHandle,
        action: PhysicalPageAction,
        size_bytes: int,
        *,
        selected_actions: dict[PageHandle, PhysicalPageAction],
        offload_bundle: bool,
    ) -> None:
        if getattr(node, "lock_ref", 0) > 0:
            raise SGLangBackendError(
                "node is engine-locked",
                blocker_code=TransferBlockerCode.NODE_LOCKED,
                required_bytes=size_bytes,
            )
        if getattr(node, "loading", False):
            raise SGLangBackendError(
                "node is loading",
                blocker_code=TransferBlockerCode.NODE_LOADING,
                required_bytes=size_bytes,
            )
        selected_node_ids = {
            selected.page_id: selected_action
            for selected, selected_action in selected_actions.items()
        }
        if action == PhysicalPageAction.START_D2H:
            if getattr(node, "value", None) is None:
                raise SGLangBackendError(
                    "cannot back up an evicted node",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    required_bytes=size_bytes,
                )
            self._require_gpu_ancestor_closure(node)
            if node.id in self.tree_cache.ongoing_write_through:
                raise SGLangBackendError(
                    "D2H extent already has an active copy",
                    blocker_code=TransferBlockerCode.INFLIGHT,
                    required_bytes=size_bytes,
                )
            if offload_bundle:
                self._require_selected_gpu_descendants(node, selected_node_ids)
        elif action == PhysicalPageAction.COMMIT_CPU:
            if (
                getattr(node, "host_value", None) is None
                or getattr(node, "value", None) is None
            ):
                raise SGLangBackendError(
                    "COMMIT requires GPU+CPU clean node",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    required_bytes=size_bytes,
                )
            if node.id in self.tree_cache.ongoing_write_through:
                raise SGLangBackendError(
                    "COMMIT cannot race an active D2H copy",
                    blocker_code=TransferBlockerCode.INFLIGHT,
                    required_bytes=size_bytes,
                )
            self._require_selected_gpu_descendants(node, selected_node_ids)
        elif action == PhysicalPageAction.START_H2D:
            if (
                getattr(node, "value", None) is not None
                or getattr(node, "host_value", None) is None
            ):
                raise SGLangBackendError(
                    "H2D requires an evicted backed-up node",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    required_bytes=size_bytes,
                )
            ancestor = getattr(node, "parent", None)
            while ancestor is not None and ancestor is not self.tree_cache.root_node:
                if getattr(ancestor, "evicted", False) and (
                    selected_node_ids.get(int(ancestor.id))
                    != PhysicalPageAction.START_H2D
                ):
                    raise SGLangBackendError(
                        "H2D bundle omits an evicted Radix ancestor",
                        blocker_code=TransferBlockerCode.ANCESTOR_CLOSURE,
                        required_bytes=size_bytes,
                    )
                ancestor = getattr(ancestor, "parent", None)
        elif action == PhysicalPageAction.DROP:
            self._preflight_drop(node, size_bytes, selected_node_ids)
        elif action == PhysicalPageAction.DROP_HOST:
            self._preflight_drop_host(node, size_bytes)
        elif action not in {PhysicalPageAction.PIN, PhysicalPageAction.UNPIN}:
            raise SGLangBackendError(f"unsupported page action: {action}")

    def _preflight_bundle_capacity(
        self, prepared: list[tuple[ResolvedPageAction, Any]]
    ) -> None:
        d2h_tokens = sum(
            self._extent_tokens(getattr(node, "value", None))
            for item, node in prepared
            if item.action == PhysicalPageAction.START_D2H
            and not getattr(node, "backuped", False)
        )
        h2d_tokens = sum(
            self._extent_tokens(getattr(node, "host_value", None))
            for item, node in prepared
            if item.action == PhysicalPageAction.START_H2D
        )
        host_pool = getattr(self.tree_cache, "token_to_kv_pool_host", None)
        device_pool = self._authoritative_device_allocator()
        host_available = self._allocator_available(host_pool)
        device_available = self._allocator_available(device_pool)
        if host_available is not None and d2h_tokens > host_available:
            raise SGLangBackendError(
                "atomic D2H bundle exceeds authoritative host capacity",
                blocker_code=TransferBlockerCode.HOST_CAPACITY,
                required_bytes=sum(
                    item.size_bytes
                    for item, _ in prepared
                    if item.action == PhysicalPageAction.START_D2H
                ),
            )
        if device_available is not None and h2d_tokens > device_available:
            raise SGLangBackendError(
                "atomic H2D bundle exceeds authoritative device capacity",
                blocker_code=TransferBlockerCode.DEVICE_CAPACITY,
                required_bytes=sum(
                    item.size_bytes
                    for item, _ in prepared
                    if item.action == PhysicalPageAction.START_H2D
                ),
            )

    @staticmethod
    def _allocator_available(allocator: Any) -> int | None:
        available_size = getattr(allocator, "available_size", None)
        if not callable(available_size):
            return None
        return max(0, int(available_size()))

    def _authoritative_device_allocator(self) -> Any:
        tree_allocator = getattr(
            self.tree_cache, "token_to_kv_pool_allocator", None
        )
        controller_allocator = getattr(
            getattr(self.tree_cache, "cache_controller", None),
            "mem_pool_device_allocator",
            None,
        )
        if (
            tree_allocator is not None
            and controller_allocator is not None
            and tree_allocator is not controller_allocator
        ):
            raise SGLangBackendError(
                "HiCache tree and controller use different device allocators",
                blocker_code=TransferBlockerCode.UNKNOWN_BACKEND,
            )
        return controller_allocator or tree_allocator

    @staticmethod
    def _extent_tokens(value: Any) -> int:
        try:
            return max(0, len(value))
        except TypeError:
            return 0

    @staticmethod
    def _preflight_drop(
        node: Any,
        size_bytes: int,
        selected_node_ids: dict[int, PhysicalPageAction] | None = None,
    ) -> None:
        if getattr(node, "value", None) is None:
            if getattr(node, "host_value", None) is None:
                raise SGLangBackendError(
                    "cannot drop an extent with no KV copy",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    required_bytes=size_bytes,
                )
            if getattr(node, "host_ref_counter", 0) > 0 or node.children:
                raise SGLangBackendError(
                    "host node is protected or not a leaf",
                    blocker_code=(
                        TransferBlockerCode.NODE_LOCKED
                        if getattr(node, "host_ref_counter", 0) > 0
                        else TransferBlockerCode.DESCENDANT_CLOSURE
                    ),
                    required_bytes=size_bytes,
                )
        elif node.children:
            selected_node_ids = selected_node_ids or {}
            stack = list(node.children.values())
            seen: set[int] = set()
            while stack:
                child = stack.pop()
                identity = id(child)
                if identity in seen:
                    raise SGLangBackendError(
                        "Radix descendant cycle detected",
                        blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                        required_bytes=size_bytes,
                    )
                seen.add(identity)
                if (
                    getattr(child, "value", None) is not None
                    and selected_node_ids.get(int(child.id))
                    != PhysicalPageAction.DROP
                ):
                    raise SGLangBackendError(
                        "GPU DROP bundle omits a resident descendant",
                        blocker_code=TransferBlockerCode.DESCENDANT_CLOSURE,
                        required_bytes=size_bytes,
                    )
                if (
                    getattr(node, "host_value", None) is None
                    and getattr(child, "host_value", None) is not None
                ):
                    raise SGLangBackendError(
                        "GPU-only parent cannot be dropped above a Host descendant",
                        blocker_code=TransferBlockerCode.DESCENDANT_CLOSURE,
                        required_bytes=size_bytes,
                    )
                stack.extend(getattr(child, "children", {}).values())

    def _preflight_drop_host(self, node: Any, size_bytes: int) -> None:
        if getattr(node, "host_value", None) is None:
            raise SGLangBackendError(
                "cannot drop a missing Host KV copy",
                blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                required_bytes=size_bytes,
            )
        if getattr(node, "host_ref_counter", 0) > 0:
            raise SGLangBackendError(
                "Host KV copy is protected",
                blocker_code=TransferBlockerCode.NODE_LOCKED,
                required_bytes=size_bytes,
            )
        if node.id in self.tree_cache.ongoing_write_through:
            raise SGLangBackendError(
                "Host KV copy is still being written",
                blocker_code=TransferBlockerCode.INFLIGHT,
                required_bytes=size_bytes,
            )
        if getattr(node, "value", None) is None and node.children:
            raise SGLangBackendError(
                "CPU-only Host KV extent is not a Radix leaf",
                blocker_code=TransferBlockerCode.DESCENDANT_CLOSURE,
                required_bytes=size_bytes,
            )

    def _require_selected_gpu_descendants(
        self,
        node: Any,
        selected_node_ids: dict[int, PhysicalPageAction],
    ) -> None:
        stack = list(getattr(node, "children", {}).values())
        seen: set[int] = set()
        allowed = {
            PhysicalPageAction.START_D2H,
            PhysicalPageAction.COMMIT_CPU,
        }
        while stack:
            child = stack.pop()
            identity = id(child)
            if identity in seen:
                raise SGLangBackendError(
                    "Radix descendant cycle detected",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                )
            seen.add(identity)
            if (
                getattr(child, "value", None) is not None
                and selected_node_ids.get(int(child.id)) not in allowed
            ):
                raise SGLangBackendError(
                    "offload bundle omits a GPU-resident descendant",
                    blocker_code=TransferBlockerCode.DESCENDANT_CLOSURE,
                )
            stack.extend(getattr(child, "children", {}).values())

    def _node_depth(self, node: Any) -> int:
        depth = 0
        seen: set[int] = set()
        ancestor = getattr(node, "parent", None)
        while ancestor is not None and ancestor is not self.tree_cache.root_node:
            identity = id(ancestor)
            if identity in seen:
                raise SGLangBackendError(
                    "Radix ancestor cycle detected",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                )
            seen.add(identity)
            depth += 1
            ancestor = getattr(ancestor, "parent", None)
        return depth

    def _submit_page(
        self,
        pending: _PendingNodeCommand,
        node: Any,
        handle: PageHandle,
        action: PhysicalPageAction,
        size_bytes: int,
        *,
        atomic_bundle: bool = False,
    ) -> None:
        if getattr(node, "lock_ref", 0) > 0:
            raise SGLangBackendError(
                "node is engine-locked",
                blocker_code=TransferBlockerCode.NODE_LOCKED,
                required_bytes=size_bytes,
            )
        if getattr(node, "loading", False):
            raise SGLangBackendError(
                "node is loading",
                blocker_code=TransferBlockerCode.NODE_LOADING,
                required_bytes=size_bytes,
            )
        if action == PhysicalPageAction.START_D2H:
            if getattr(node, "value", None) is None:
                raise SGLangBackendError(
                    "cannot back up an evicted node",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    required_bytes=size_bytes,
                )
            self._require_gpu_ancestor_closure(node)
            if getattr(node, "backuped", False):
                if node.id in self.tree_cache.ongoing_write_through:
                    pending.transfer_handles.add(handle)
                elif pending.resolved.command.kind == CommandKind.OFFLOAD_CONTEXT:
                    if not atomic_bundle:
                        self._require_no_gpu_descendants(node)
                        self.tree_cache._evict_backuped(node)
                        pending.completed_handles.add(handle)
                else:
                    pending.completed_handles.add(handle)
            else:
                pending.transfer_host_copy_state[TransferDirection.D2H] = "missing"
                allocator_submit_started_ns = time.perf_counter_ns()
                try:
                    written = self.tree_cache.write_backup(
                        node, beliefkv_source="explicit"
                    )
                finally:
                    pending.allocator_submit_ms += (
                        time.perf_counter_ns() - allocator_submit_started_ns
                    ) / 1_000_000.0
                if written <= 0:
                    raise SGLangBackendError(
                        "HiCache host allocation failed",
                        blocker_code=TransferBlockerCode.HOST_CAPACITY,
                        required_bytes=size_bytes,
                    )
                pending.transfer_handles.add(handle)
                pending.native_concurrent_bytes = max(
                    pending.native_concurrent_bytes,
                    self._native_inflight_bytes(),
                )
            if getattr(node, "host_value", None) is not None:
                pending.transfer_host_copy_state.setdefault(
                    TransferDirection.D2H, "present"
                )
            pending.accepted_handles.add(handle)
        elif action == PhysicalPageAction.COMMIT_CPU:
            if (
                getattr(node, "host_value", None) is None
                or getattr(node, "value", None) is None
            ):
                raise SGLangBackendError(
                    "COMMIT requires GPU+CPU clean node",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    required_bytes=size_bytes,
                )
            if node.id in self.tree_cache.ongoing_write_through:
                raise SGLangBackendError(
                    "COMMIT cannot race an active D2H copy",
                    blocker_code=TransferBlockerCode.INFLIGHT,
                    required_bytes=size_bytes,
                )
            pending.accepted_handles.add(handle)
            if not atomic_bundle:
                self._require_no_gpu_descendants(node)
                self.tree_cache._evict_backuped(node)
                pending.completed_handles.add(handle)
        elif action == PhysicalPageAction.START_H2D:
            if (
                getattr(node, "value", None) is not None
                or getattr(node, "host_value", None) is None
            ):
                raise SGLangBackendError(
                    "H2D requires an evicted backed-up node",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    required_bytes=size_bytes,
                )
            ancestor = getattr(node, "parent", None)
            while ancestor is not None and ancestor is not self.tree_cache.root_node:
                if getattr(ancestor, "evicted", False):
                    raise SGLangBackendError(
                        "H2D selection violates HiCache ancestor closure",
                        blocker_code=TransferBlockerCode.ANCESTOR_CLOSURE,
                        required_bytes=size_bytes,
                    )
                ancestor = getattr(ancestor, "parent", None)
            pending.transfer_host_copy_state[TransferDirection.H2D] = "present"
            allocator_submit_started_ns = time.perf_counter_ns()
            try:
                loaded = self.tree_cache.load_back(
                    node,
                    force=True,
                    allow_eviction=False,
                    beliefkv_source="explicit",
                )
            finally:
                pending.allocator_submit_ms += (
                    time.perf_counter_ns() - allocator_submit_started_ns
                ) / 1_000_000.0
            if loaded is None:
                raise SGLangBackendError(
                    "HiCache device allocation failed",
                    blocker_code=TransferBlockerCode.DEVICE_CAPACITY,
                    required_bytes=size_bytes,
                )
            pending.accepted_handles.add(handle)
            pending.transfer_handles.add(handle)
            pending.native_concurrent_bytes = max(
                pending.native_concurrent_bytes, self._native_inflight_bytes()
            )
        elif action == PhysicalPageAction.DROP:
            if getattr(node, "value", None) is None:
                if getattr(node, "host_value", None) is None:
                    raise SGLangBackendError(
                        "cannot drop an extent with no KV copy",
                        blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                        required_bytes=size_bytes,
                    )
                if getattr(node, "host_ref_counter", 0) > 0 or node.children:
                    raise SGLangBackendError(
                        "host node is protected or not a leaf",
                        blocker_code=(
                            TransferBlockerCode.NODE_LOCKED
                            if getattr(node, "host_ref_counter", 0) > 0
                            else TransferBlockerCode.DESCENDANT_CLOSURE
                        ),
                        required_bytes=size_bytes,
                    )
                self.tree_cache.cache_controller.evict_host(node.host_value)
                for key, child in tuple(node.parent.children.items()):
                    if child is node:
                        del node.parent.children[key]
                        break
            elif getattr(node, "host_value", None) is not None:
                self._require_no_gpu_descendants(node)
                self.tree_cache._evict_backuped(node)
            else:
                if node.children:
                    raise SGLangBackendError(
                        "GPU-only DROP requires a Radix leaf",
                        blocker_code=TransferBlockerCode.DESCENDANT_CLOSURE,
                        required_bytes=size_bytes,
                    )
                self.tree_cache._evict_regular(node)
            pending.accepted_handles.add(handle)
            pending.completed_handles.add(handle)
        elif action == PhysicalPageAction.DROP_HOST:
            self._preflight_drop_host(node, size_bytes)
            removed = getattr(node, "value", None) is None
            released = self.tree_cache.cache_controller.evict_host(node.host_value)
            if released <= 0:
                raise SGLangBackendError(
                    "HiCache Host allocator released no tokens",
                    blocker_code=TransferBlockerCode.UNKNOWN_BACKEND,
                    required_bytes=size_bytes,
                )
            node.host_value = None
            if removed:
                for key, child in tuple(node.parent.children.items()):
                    if child is node:
                        del node.parent.children[key]
                        break
            notify = getattr(self.tree_cache, "_beliefkv_notify", None)
            if callable(notify):
                notify("on_radix_mutation", (node,), removed, removed)
            pending.accepted_handles.add(handle)
            pending.completed_handles.add(handle)
        elif action in {PhysicalPageAction.PIN, PhysicalPageAction.UNPIN}:
            # Semantic pins live in BeliefKV. Engine lock_ref is never modified
            # by a policy pin because it is an SGLang correctness reference.
            pending.accepted_handles.add(handle)
            pending.completed_handles.add(handle)
        else:
            raise SGLangBackendError(f"unsupported page action: {action}")

    def cancel(self, command_id: str) -> None:
        pending = self._pending.get(command_id)
        if pending is not None:
            pending.cancel_requested = True

    def abort_all(self, *, reason: str) -> list[CommandAck]:
        """Terminate controller bookkeeping after an authoritative cache reset."""

        acks = self._acks
        self._acks = []
        for pending in tuple(self._pending.values()):
            complete_ts_ms = float(self._now_ms())
            acks.append(
                CommandAck(
                    command_id=pending.resolved.command.command_id,
                    status=CommandStatus.CANCELLED,
                    completed_ts_ms=complete_ts_ms,
                    actual_bytes=0,
                    reason=reason,
                )
            )
            self._record_transfer_telemetry(
                pending,
                status=CommandStatus.CANCELLED,
                reason=reason,
                complete_ts_ms=complete_ts_ms,
                completed_handles=set(),
            )
        self._pending.clear()
        return acks

    def poll_acks(self) -> list[CommandAck]:
        self.tree_cache.check_hicache_events()
        take_callback_errors = getattr(
            self.tree_cache, "take_beliefkv_callback_errors", None
        )
        if callable(take_callback_errors):
            callback_errors = tuple(take_callback_errors())
            self._callback_errors.extend(callback_errors)
            self._reject_commands_with_callback_failures(callback_errors)
        for pending in tuple(self._pending.values()):
            self._refresh(pending)
            terminal = pending.completed_handles | pending.rejected_handles
            if pending.accepted_handles <= terminal:
                if pending.rejected_handles:
                    status = (
                        CommandStatus.PARTIAL
                        if pending.completed_handles
                        else CommandStatus.REJECTED
                    )
                else:
                    status = CommandStatus.COMPLETED
                reason = (
                    "current_nonpreemptible_chunk_completed_after_cancel"
                    if pending.cancel_requested
                    else self._rejection_reason(pending)
                )
                self._finish(pending, status=status, reason=reason)
        acks = self._acks
        self._acks = []
        return acks

    def _reject_commands_with_callback_failures(
        self, errors: tuple[dict[str, object], ...]
    ) -> None:
        for error in errors:
            operation_id = str(error.get("operation_id", ""))
            direction = str(error.get("direction", ""))
            expected_action = {
                "d2h": PhysicalPageAction.START_D2H,
                "h2d": PhysicalPageAction.START_H2D,
            }.get(direction)
            if expected_action is None:
                continue
            for pending in tuple(self._pending.values()):
                actions = {
                    item.handle: item for item in pending.resolved.page_actions
                }
                for handle in pending.transfer_handles:
                    action = actions.get(handle)
                    if action is None or action.action != expected_action:
                        continue
                    try:
                        node = self.registry.resolve(handle)
                    except SGLangBackendError:
                        continue
                    if str(getattr(node, "id", "")) != operation_id:
                        continue
                    self._reject(
                        pending,
                        handle,
                        SGLangBackendError(
                            "HiCache callback bookkeeping failed: "
                            f"{error.get('error', 'unknown error')}",
                            blocker_code=TransferBlockerCode.UNKNOWN_BACKEND,
                            required_bytes=action.size_bytes,
                        ),
                    )

    def poll_callback_errors(self) -> list[dict[str, object]]:
        errors = self._callback_errors
        self._callback_errors = []
        return errors

    def poll_transfer_telemetry(self) -> list[TransferTelemetry]:
        telemetry = self._telemetry
        self._telemetry = []
        return telemetry

    def _pinned_host_state(self) -> bool | None:
        value = getattr(
            getattr(self.tree_cache, "token_to_kv_pool_host", None),
            "pin_memory",
            None,
        )
        return bool(value) if value is not None else None

    def _native_inflight_bytes(self) -> int:
        metadata = getattr(self.tree_cache, "beliefkv_transfer_metadata", {})
        tokens = sum(
            max(0, int(record.get("token_count", 0)))
            for direction_records in metadata.values()
            for record in direction_records.values()
        )
        return tokens * self._kv_bytes_per_token

    def _refresh(self, pending: _PendingNodeCommand) -> None:
        if pending.resolved.command.physical_bundle is not None:
            self._refresh_atomic_bundle(pending)
            return
        command_kind = pending.resolved.command.kind
        actions = {item.handle: item.action for item in pending.resolved.page_actions}
        for handle in pending.transfer_handles - pending.completed_handles:
            if handle in pending.rejected_handles:
                continue
            try:
                node = self.registry.resolve(handle)
                action = actions[handle]
                if action == PhysicalPageAction.START_D2H:
                    still_writing = node.id in self.tree_cache.ongoing_write_through
                    if still_writing:
                        continue
                    if getattr(node, "host_value", None) is None:
                        raise SGLangBackendError(
                            "D2H ended without an authoritative host copy",
                            blocker_code=TransferBlockerCode.HOST_CAPACITY,
                        )
                    pending.dma_completed_handles.add(handle)
                    self._require_unchanged_extent(pending, handle, node)
                    if command_kind == CommandKind.OFFLOAD_CONTEXT:
                        self._require_no_gpu_descendants(node)
                        if getattr(node, "value", None) is not None:
                            self.tree_cache._evict_backuped(node)
                    pending.completed_handles.add(handle)
                elif action == PhysicalPageAction.START_H2D:
                    still_loading = node.id in self.tree_cache.ongoing_load_back
                    if still_loading or getattr(node, "loading", False):
                        continue
                    if getattr(node, "value", None) is None:
                        raise SGLangBackendError(
                            "H2D ended without an authoritative GPU copy",
                            blocker_code=TransferBlockerCode.DEVICE_CAPACITY,
                        )
                    pending.dma_completed_handles.add(handle)
                    self._require_unchanged_extent(pending, handle, node)
                    pending.completed_handles.add(handle)
            except (SGLangBackendError, AssertionError, RuntimeError) as error:
                self._reject(pending, handle, error)

    def _refresh_atomic_bundle(self, pending: _PendingNodeCommand) -> None:
        actions = {item.handle: item.action for item in pending.resolved.page_actions}
        nodes: dict[PageHandle, Any] = {}
        intent = pending.resolved.command.physical_bundle
        assert intent is not None
        for handle in intent.closure_handles:
            try:
                node = self.registry.resolve(handle)
                nodes[handle] = node
                self._require_unchanged_extent(pending, handle, node)
            except (SGLangBackendError, AssertionError, RuntimeError) as error:
                self._reject(pending, handle, error)
        waiting_for_dma = False
        for handle in sorted(pending.transfer_handles):
            try:
                node = self.registry.resolve(handle)
                nodes[handle] = node
                action = actions[handle]
                if action == PhysicalPageAction.START_D2H:
                    if node.id in self.tree_cache.ongoing_write_through:
                        waiting_for_dma = True
                        continue
                    if getattr(node, "host_value", None) is None:
                        raise SGLangBackendError(
                            "D2H ended without an authoritative host copy",
                            blocker_code=TransferBlockerCode.HOST_CAPACITY,
                        )
                elif action == PhysicalPageAction.START_H2D:
                    if (
                        node.id in self.tree_cache.ongoing_load_back
                        or getattr(node, "loading", False)
                    ):
                        waiting_for_dma = True
                        continue
                    if getattr(node, "value", None) is None:
                        raise SGLangBackendError(
                            "H2D ended without an authoritative GPU copy",
                            blocker_code=TransferBlockerCode.DEVICE_CAPACITY,
                        )
                self._require_unchanged_extent(pending, handle, node)
                pending.dma_completed_handles.add(handle)
            except (SGLangBackendError, AssertionError, RuntimeError) as error:
                self._reject(pending, handle, error)

        if waiting_for_dma:
            return
        if pending.rejected_handles:
            self._rollback_atomic_h2d(pending, actions, nodes)
            pending.rejected_handles.update(pending.accepted_handles)
            return

        try:
            for handle in sorted(pending.accepted_handles):
                node = nodes.get(handle) or self.registry.resolve(handle)
                nodes[handle] = node
                self._require_unchanged_extent(pending, handle, node)
                if getattr(node, "lock_ref", 0) > 0:
                    raise SGLangBackendError(
                        "node became engine-locked before bundle commit",
                        blocker_code=TransferBlockerCode.NODE_LOCKED,
                    )
                action = actions[handle]
                if action in {
                    PhysicalPageAction.START_D2H,
                    PhysicalPageAction.COMMIT_CPU,
                }:
                    if (
                        getattr(node, "host_value", None) is None
                        or getattr(node, "value", None) is None
                    ):
                        raise SGLangBackendError(
                            "offload bundle lost a required clean copy",
                            blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                        )
                elif (
                    action == PhysicalPageAction.START_H2D
                    and getattr(node, "value", None) is None
                ):
                    raise SGLangBackendError(
                        "prefetch bundle lost its GPU copy before commit",
                        blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    )

            if pending.resolved.command.kind == CommandKind.OFFLOAD_CONTEXT:
                selected_node_ids = {
                    int(nodes[handle].id): actions[handle]
                    for handle in pending.accepted_handles
                }
                for handle in pending.accepted_handles:
                    self._require_selected_gpu_descendants(
                        nodes[handle], selected_node_ids
                    )
                for handle in sorted(
                    pending.accepted_handles,
                    key=lambda item: (self._node_depth(nodes[item]), item),
                    reverse=True,
                ):
                    node = nodes[handle]
                    if getattr(node, "value", None) is not None:
                        self.tree_cache._evict_backuped(node)
                    pending.completed_handles.add(handle)
            else:
                pending.completed_handles.update(pending.accepted_handles)
        except (SGLangBackendError, AssertionError, RuntimeError) as error:
            failed_handle = next(
                (
                    handle
                    for handle in sorted(pending.accepted_handles)
                    if handle not in pending.completed_handles
                ),
                next(iter(pending.accepted_handles)),
            )
            self._reject(pending, failed_handle, error)
            self._rollback_atomic_h2d(pending, actions, nodes)
            pending.rejected_handles.update(pending.accepted_handles)

    def _rollback_atomic_h2d(
        self,
        pending: _PendingNodeCommand,
        actions: dict[PageHandle, PhysicalPageAction],
        nodes: dict[PageHandle, Any],
    ) -> None:
        restored = [
            handle
            for handle in pending.accepted_handles
            if actions.get(handle) == PhysicalPageAction.START_H2D
        ]
        for handle in sorted(
            restored,
            key=lambda item: (
                self._node_depth(nodes.get(item) or self.registry.resolve(item)),
                item,
            ),
            reverse=True,
        ):
            node = nodes.get(handle) or self.registry.resolve(handle)
            nodes[handle] = node
            if getattr(node, "value", None) is None:
                continue
            try:
                if getattr(node, "host_value", None) is None:
                    raise SGLangBackendError(
                        "cannot roll back H2D extent without host copy",
                        blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                    )
                self._require_no_gpu_descendants(node)
                self.tree_cache._evict_backuped(node)
            except (SGLangBackendError, AssertionError, RuntimeError) as error:
                # The GPU copy is authoritative if rollback itself fails. ACK
                # only that residual extent so the ownership mirror converges.
                pending.completed_handles.add(handle)
                self._reject(pending, handle, error)

    @staticmethod
    def _reject(
        pending: _PendingNodeCommand, handle: PageHandle, error: BaseException
    ) -> None:
        pending.rejected_handles.add(handle)
        pending.rejection_reasons[handle] = f"{type(error).__name__}: {error}"
        pending.rejection_blockers[handle] = TransferBlocker(
            code=(
                error.blocker_code
                if isinstance(error, SGLangBackendError)
                else TransferBlockerCode.UNKNOWN_BACKEND
            ),
            page_handle=handle,
            required_bytes=(
                error.required_bytes
                if isinstance(error, SGLangBackendError)
                else 0
            ),
            detail=str(error),
        )

    def _require_unchanged_extent(
        self,
        pending: _PendingNodeCommand,
        handle: PageHandle,
        node: Any,
    ) -> None:
        submitted = pending.extent_fingerprints.get(handle)
        if submitted is None or self._extent_fingerprint(node) != submitted:
            raise SGLangBackendError(
                "Radix extent mutated during transfer; residency action rejected",
                blocker_code=TransferBlockerCode.EXTENT_MUTATED,
            )

    def _extent_fingerprint(self, node: Any) -> tuple[Any, ...]:
        parent = getattr(node, "parent", None)
        children = getattr(node, "children", {})
        return (
            self.registry.fingerprint(node),
            int(parent.id) if parent is not None else None,
            tuple(sorted(int(child.id) for child in children.values())),
        )

    @staticmethod
    def _rejection_reason(
        pending: _PendingNodeCommand, fallback: str = ""
    ) -> str:
        reasons = sorted(set(pending.rejection_reasons.values()))
        if not reasons:
            return fallback
        detail = " | ".join(reasons[:3])
        if len(reasons) > 3:
            detail += f" | {len(reasons) - 3} more"
        return f"{fallback}: {detail}" if fallback else detail

    def _require_gpu_ancestor_closure(self, node: Any) -> None:
        ancestor = getattr(node, "parent", None)
        while ancestor is not None and ancestor is not self.tree_cache.root_node:
            if getattr(ancestor, "value", None) is None:
                raise SGLangBackendError(
                    "D2H target has an evicted Radix ancestor",
                    blocker_code=TransferBlockerCode.ANCESTOR_CLOSURE,
                )
            ancestor = getattr(ancestor, "parent", None)

    @staticmethod
    def _require_no_gpu_descendants(node: Any) -> None:
        stack = list(getattr(node, "children", {}).values())
        seen: set[int] = set()
        while stack:
            child = stack.pop()
            identity = id(child)
            if identity in seen:
                raise SGLangBackendError(
                    "Radix descendant cycle detected",
                    blocker_code=TransferBlockerCode.EXTENT_MUTATED,
                )
            seen.add(identity)
            if getattr(child, "value", None) is not None:
                raise SGLangBackendError(
                    "GPU eviction requires all Radix descendants off device",
                    blocker_code=TransferBlockerCode.DESCENDANT_CLOSURE,
                )
            stack.extend(getattr(child, "children", {}).values())

    def _finish(
        self,
        pending: _PendingNodeCommand,
        *,
        status: CommandStatus,
        reason: str,
    ) -> None:
        command = pending.resolved.command
        handles = tuple(sorted(pending.completed_handles))
        sizes = {item.handle: item.size_bytes for item in pending.resolved.page_actions}
        actual_bytes = sum(sizes[handle] for handle in handles)
        complete_ts_ms = float(self._now_ms())
        self._acks.append(
            CommandAck(
                command_id=command.command_id,
                status=status,
                completed_ts_ms=complete_ts_ms,
                actual_bytes=actual_bytes,
                page_handles=handles,
                reason=reason,
                blockers=self._command_blockers(pending),
            )
        )
        self._record_transfer_telemetry(
            pending,
            status=status,
            reason=reason,
            complete_ts_ms=complete_ts_ms,
            completed_handles=pending.dma_completed_handles,
        )
        self._pending.pop(command.command_id, None)

    @staticmethod
    def _command_blockers(
        pending: _PendingNodeCommand,
    ) -> tuple[TransferBlocker, ...]:
        unique = {
            (
                blocker.code,
                blocker.page_handle,
                blocker.required_bytes,
                blocker.detail,
            ): blocker
            for blocker in pending.rejection_blockers.values()
        }
        return tuple(
            unique[key]
            for key in sorted(
                unique,
                key=lambda item: (
                    item[0].value,
                    item[1] or PageHandle(0, 0),
                    item[2],
                    item[3],
                ),
            )
        )

    def _record_transfer_telemetry(
        self,
        pending: _PendingNodeCommand,
        *,
        status: CommandStatus,
        reason: str,
        complete_ts_ms: float,
        completed_handles: set[PageHandle],
    ) -> None:
        actions = {item.handle: item for item in pending.resolved.page_actions}
        direction_actions = {
            TransferDirection.D2H: PhysicalPageAction.START_D2H,
            TransferDirection.H2D: PhysicalPageAction.START_H2D,
        }
        command = pending.resolved.command
        for direction, physical_action in direction_actions.items():
            selected = {
                handle
                for handle, action in actions.items()
                if action.action == physical_action
            }
            if not selected:
                continue
            completed = selected & completed_handles
            closure_bytes = sum(actions[handle].size_bytes for handle in selected)
            actual_bytes = sum(actions[handle].size_bytes for handle in completed)
            source_tier, target_tier = (
                ("gpu", "host")
                if direction == TransferDirection.D2H
                else ("host", "gpu")
            )
            self._telemetry.append(
                TransferTelemetry(
                    command_id=command.command_id,
                    submit_ts_ms=pending.submit_ts_ms,
                    start_ts_ms=(
                        pending.start_ts_ms
                        if selected & pending.transfer_handles
                        else None
                    ),
                    first_layer_ready_ts_ms=None,
                    complete_ts_ms=complete_ts_ms,
                    compute_wait_ms=None,
                    actual_bytes=actual_bytes,
                    closure_bytes=closure_bytes,
                    merged_operation_count=0,
                    direction=direction,
                    source_tier=source_tier,
                    target_tier=target_tier,
                    status=status,
                    reason=reason,
                    page_count=len(completed),
                    context_id=command.context_id,
                    context_epoch=command.context_epoch,
                    command_kind=command.kind.value,
                    compute_phase=str(command.metadata.get("compute_phase", "unknown")),
                    host_copy_state=pending.transfer_host_copy_state.get(
                        direction, "unknown"
                    ),
                    pinned_host=pending.pinned_host,
                    native_concurrent_bytes=pending.native_concurrent_bytes,
                    allocator_submit_ms=(
                        pending.allocator_submit_ms
                        if selected & pending.transfer_handles
                        else None
                    ),
                    callback_overhead_ms=max(
                        0.0, float(self._now_ms()) - complete_ts_ms
                    ),
                    start_timestamp_semantics=(
                        "hicache_api_submit_begin"
                        if pending.start_ts_ms is not None
                        else "unavailable"
                    ),
                )
            )


_JOINT_DECISION_REUSE_INTERVAL_MS = 100.0
_JOINT_SIGNATURE_CHECK_INTERVAL_MS = 10.0
_ACK_POLL_INTERVAL_MS = 5.0
_POLICY_CHECK_INTERVAL_MS = 5.0


class EmbeddedSGLangRuntime:
    """BeliefKV control plane embedded at SGLang scheduler safe points.

    The SGLang patch calls this class; no policy is copied into SGLang itself.
    Requests without ``beliefkv_metadata`` bypass every method and retain
    upstream behavior.
    """

    def __init__(
        self,
        scheduler: Any,
        *,
        config_path: str | None = None,
        config: BeliefKVConfig | None = None,
        now_ms: Any | None = None,
    ) -> None:
        if not getattr(scheduler, "enable_hierarchical_cache", False):
            raise SGLangBackendError("BeliefKV requires SGLang HiCache")
        self.scheduler = scheduler
        self.tree_cache = scheduler.tree_cache
        self._now_ms = now_ms or (lambda: time.monotonic() * 1000.0)
        self.config = config or self._load_config(scheduler, config_path)
        if self.config.joint_policy_enabled and not (
            self.config.joint_policy_shadow_mode
            and self.config.joint_observed_mode_enabled
        ):
            raise SGLangBackendError(
                "online observed JointPlan requires its validated shadow worker"
            )
        self.controller = BeliefKVController(self.config)
        self.registry = SGLangNodeRegistry()
        self.backend = HiCacheNodeCommandBackend(
            self.tree_cache,
            self.registry,
            now_ms=self._now_ms,
            h2d_context_is_busy=self._context_has_engine_request,
            kv_bytes_per_token=self.config.kv_bytes_per_token,
        )
        self.bridge = SGLangSchedulerBridge(self.controller, self.backend)
        self._admission_epoch = 0
        self._current_ticket_epoch: AdmissionTicketEpoch | None = None
        self._current_tickets_by_request: dict[str, AdmissionTicket] = {}
        self._ticket_attempted_request_ids: set[str] = set()
        self._ticket_selected_request_ids: set[str] = set()
        self._ticket_skip_audit: set[tuple[int, str, str]] = set()
        self._ticket_selection_details: dict[str, dict[str, Any]] = {}
        self._ticket_native_rejections: dict[str, str] = {}
        self.observed_admission_scheduler = ObservedAdmissionScheduler(
            active_kv_high_watermark_ratio=(
                self.config.observed_admission_active_kv_high_watermark_ratio
            ),
            minimum_active_requests=(
                self.config.observed_admission_min_active_requests
            ),
        )
        self._current_observed_admission_window: (
            ObservedAdmissionWindow | None
        ) = None
        self._observed_admission_mode_counts: Counter[str] = Counter()
        self._observed_admission_peak_active_kv_bytes = 0
        self._observed_admission_peak_pressure = 0.0
        self._joint_retraction_solver = ObservedRetractionPlanner(
            ObservedRetractionConfig(
                minimum_admission_stall_ms=(
                    self.config.running_batch_retraction_min_stall_ms
                ),
                minimum_reclaim_bytes=(
                    self.config.running_batch_retraction_min_reclaim_bytes
                ),
                maximum_retractions_per_request=(
                    self.config.running_batch_retraction_max_per_request
                ),
            )
        )
        self._retraction_admission_stall_since_ms: float | None = None
        self._last_retraction_decision_ms: float | None = None
        self._retraction_sequence = 0
        self._retraction_barrier_sequence = 0
        self._retraction_counts_by_request: Counter[str] = Counter()
        self._retraction_cooldown_until_by_request: dict[str, float] = {}
        self._restore_service_grace_by_request: dict[
            str, RestoreServiceGrace
        ] = {}
        self._persistent_liveness_revisions = PersistentLivenessRevisionTracker()
        self._pending_selective_retraction_ids: set[str] = set()
        self._retracted_engine_request_ids: set[str] = set()
        self._retraction_priority_request_ids: tuple[str, ...] = ()
        self._running_retraction_transactions: deque[
            _RunningRetractionTransaction
        ] = deque(maxlen=65_536)
        self._pending_running_retraction_transaction: (
            _RunningRetractionTransaction | None
        ) = None
        self._pending_running_retraction_barrier: (
            _RunningRetractionBarrierAttempt | None
        ) = None
        self._running_retraction_counts: Counter[str] = Counter()
        self._running_retraction_barrier_outcomes: Counter[str] = Counter()
        self._running_retraction_preview_counts: Counter[str] = Counter()
        self._running_retraction_preview_compute_us: deque[float] = deque(
            maxlen=65_536
        )
        self._running_retraction_barrier_preview_compute_us: deque[float] = deque(
            maxlen=65_536
        )
        self._running_retraction_actual_reclaim_bytes = 0
        self._running_retraction_actual_lock_release_bytes = 0
        self._restore_micro_gate_state: dict[str, Any] = {
            "enabled": self.config.restore_micro_gate_enabled,
            "gate_id": self.config.restore_micro_gate_id,
            "stage": (
                "armed" if self.config.restore_micro_gate_enabled else "disabled"
            ),
            "victim_workflow_id": (
                self.config.restore_micro_gate_victim_workflow_id
            ),
            "replacement_workflow_id": (
                self.config.restore_micro_gate_replacement_workflow_id
            ),
        }
        self._restore_micro_gate_last_audit_signature: tuple[object, ...] | None = (
            None
        )
        self._restore_obligations = RestoreObligationIndex(
            max_active=self.config.restore_obligation_max_active
        )
        self._restore_leases = RestoreLeaseIndex(
            max_active=self.config.restore_lease_max_active
        )
        self._restore_lease_allocations: dict[str, list[Any]] = {}
        self._restore_funding_allocations: dict[str, list[Any]] = {}
        self._restore_funding_target_by_command: dict[str, dict[str, int]] = {}
        self._restore_lease_pins: dict[str, tuple[Any, Any, bool]] = {}
        self._restore_obligation_counts: Counter[str] = Counter()
        self._restore_command_to_request: dict[str, set[str]] = {}
        self._restore_transactions: dict[str, RestoreTransaction] = {}
        self._restore_authority_mode = RestoreAuthorityMode.NORMAL_JOINT
        self._restore_authority_request_id: str | None = None
        self._restore_certificate_sequence = 0
        self._restore_command_sequence = 0
        self._h2d_context_by_command: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._pending_h2d_contexts: set[str] = set()
        self._active_request_ids: set[str] = set()
        self._request_metadata_by_id: dict[str, BeliefKVRequestMetadata] = {}
        self._request_submitted_ts_by_id: dict[str, float] = {}
        self._request_physical_start_by_id: dict[str, dict[str, Any]] = {}
        self._aborted_request_physical_start_by_id: dict[
            str, dict[str, Any]
        ] = {}
        self._pending_request_physical_finish_by_id: dict[
            str, dict[str, Any]
        ] = {}
        self._terminal_cancelled_request_ids: set[str] = set()
        self._queue_timeout_request_ids: set[str] = set()
        self._execution_timeout_request_ids: set[str] = set()
        self._terminal_node_by_context: dict[str, Any] = {}
        self._event_sequence = 0
        self._linked_invocations: set[str] = set()
        self._identity_metadata: dict[str, BeliefKVRequestMetadata] = {}
        self._last_batch_selected_ms: float | None = None
        self._last_batch_workflow_counts: dict[str, int] = {}
        self._last_admission_audit: tuple[str, bool, str] | None = None
        self._last_native_admission_audit: tuple[str, int, int] | None = None
        self._stalled_command_audited: set[str] = set()
        self._last_resource_telemetry_ms: float | None = None
        self._scheduler_timing_samples: deque[tuple[float, float, int]] = deque(
            maxlen=65_536
        )
        self._native_physical_snapshot_timing_samples: deque[
            SafePointSnapshotBuildTiming
        ] = deque(maxlen=65_536)
        self._native_physical_snapshot_counts: Counter[str] = Counter()
        self._safe_point_physical_epoch_sequence = 0
        self._safe_point_physical_phase = SafePointPhysicalPhase.IDLE
        self._safe_point_physical_snapshot: SafePointPhysicalSnapshot | None = None
        self._ticket_timing_samples: dict[str, deque[float]] = {
            "compile_ms": deque(maxlen=65_536),
            "validation_ms": deque(maxlen=65_536),
        }
        self._gpu_service_launches: deque[dict[str, Any] | None] = deque()
        self._gpu_service_sequence = 0
        self._gpu_service_sample_count = 0
        self._gpu_service_sample_cap_count = 0
        self._gpu_service_previous_completion_ms: float | None = None
        self._gpu_service_prefill_chunks_by_episode: dict[str, int] = {}
        self._gpu_service_observer_timing_samples: deque[tuple[float, float]] = (
            deque(maxlen=65_536)
        )
        self._lock_service_ledger = RequestServiceLedger()
        self._lock_service_observer_error_count = 0
        self._lock_service_snapshot_count = 0
        self._lock_service_peak_bytes = {"100ms": 0, "500ms": 0}
        self._tree_dirty = True
        self._tree_full_rebuild_required = True
        self._dirty_radix_nodes: dict[int, Any] = {}
        self._removed_radix_nodes: dict[int, Any] = {}
        self._dirty_context_ids: set[str] = set()
        self._closed = False
        self._shutdown_state = "running"
        self._shutdown_prepare_ms: float | None = None
        self._shutdown_prepare_transaction_snapshot: dict[str, Any] | None = None
        self._last_runtime_summary_ms: float | None = None
        self.audit = RuntimeAuditLog(
            self.config.runtime_audit_path,
            max_pending=self.config.runtime_audit_queue_capacity,
            debug_sample_rate=self.config.runtime_audit_debug_sample_rate,
            max_debug_event_bytes=(
                self.config.runtime_audit_max_debug_event_bytes
            ),
            flush_interval_s=self.config.runtime_audit_flush_interval_s,
        )
        runtime_summary_path = self.config.runtime_summary_path
        if runtime_summary_path is None and self.config.runtime_audit_path is not None:
            runtime_summary_path = str(
                Path(self.config.runtime_audit_path).with_name(
                    "latest_runtime_summary.json"
                )
            )
        self._runtime_summary_path = (
            Path(runtime_summary_path).expanduser().resolve()
            if runtime_summary_path is not None
            else None
        )
        self._runtime_scheduler_pid_path = (
            Path(self.config.runtime_scheduler_pid_path).expanduser().resolve()
            if self.config.runtime_scheduler_pid_path is not None
            else None
        )
        self._runtime_shutdown_ack_path = (
            Path(self.config.runtime_shutdown_ack_path).expanduser().resolve()
            if self.config.runtime_shutdown_ack_path is not None
            else None
        )
        policy_snapshot_path = self.config.reference_policy_snapshot_path
        if (
            policy_snapshot_path is None
            and self.config.runtime_audit_path is not None
        ):
            policy_snapshot_path = str(
                Path(self.config.runtime_audit_path).with_name(
                    "policy_snapshots.jsonl.gz"
                )
            )
        if not self.config.reference_policy_shadow_enabled:
            policy_snapshot_path = None
        self.policy_snapshot_log = PolicySnapshotLog(
            policy_snapshot_path,
            trace_id=self.audit.run_id,
            trace_sensitivity=self.config.reference_policy_trace_sensitivity,
            max_pending=self.config.reference_policy_snapshot_max_pending,
        )
        token_trace_path = self.config.request_token_trace_path
        if self.config.request_token_trace_enabled and token_trace_path is None:
            if self.config.runtime_audit_path is None:
                raise ValueError(
                    "request token trace requires an explicit path or runtime audit"
                )
            token_trace_path = str(
                Path(self.config.runtime_audit_path).with_name(
                    "request_tokens.jsonl.gz"
                )
            )
        if not self.config.request_token_trace_enabled:
            token_trace_path = None
        if token_trace_path is not None and token_trace_path in {
            self.config.runtime_audit_path,
            policy_snapshot_path,
        }:
            raise ValueError("request_token_trace_path must use a dedicated file")
        self.request_token_trace_log = RequestTokenTraceLog(
            token_trace_path,
            run_id=self.audit.run_id,
        )
        self._request_partial_commit_count: dict[str, int] = {}
        self._last_policy_snapshot_structural_signature: (
            tuple[object, ...] | None
        ) = None
        self._last_policy_snapshot_physical_signature: (
            tuple[object, ...] | None
        ) = None
        self._last_policy_snapshot_hbm_bucket: int | None = None
        self._last_policy_snapshot_ms: float | None = None
        self._last_policy_snapshot_cheap_signature: (
            tuple[object, ...] | None
        ) = None
        self._last_persisted_policy_snapshot_ms: float | None = None
        self._shadow_event_sequence = 0
        self._shadow_page_revision = 0
        self._shadow_telemetry_sequence = 0
        self.joint_shadow_worker: LatestWinsJointPlanWorker | None = None
        self._last_joint_shadow_result_sequence = 0
        self._joint_shadow_counts: Counter[str] = Counter()
        self._joint_predictive_counts: Counter[str] = Counter()
        self._last_frontier_predictions: dict[str, dict[str, object]] = {}
        self._joint_shadow_strict_stale_reasons: Counter[str] = Counter()
        self._joint_shadow_readset_stale_reasons: Counter[str] = Counter()
        self._online_joint_result: JointShadowResult | None = None
        self._online_joint_source: PolicyInput | None = None
        self._online_joint_validation: JointPlanComponentValidation | None = None
        self._online_joint_counts: Counter[str] = Counter()
        self._last_joint_decision_plan_id: str | None = None
        self._last_joint_decision_ms: float | None = None
        self._last_joint_signature_check_ms: float | None = None
        self._last_joint_decision_state_signature: (
            tuple[object, ...] | None
        ) = None
        self._last_ack_poll_ms: float | None = None
        self._last_policy_check_ms: float | None = None
        self._online_joint_epoch_sequence = 0
        self._current_joint_plan_epoch = None
        self._current_online_joint_view: OnlineJointPlanView | None = None
        self._current_online_joint_decision: OnlineJointPlanDecision | None = None
        self._current_semantic_residency_commit: (
            tuple[str, int, SemanticResidencyTarget, PhysicalBundlePreview] | None
        ) = None
        self._online_joint_residency_sequence = 0
        self._pending_online_joint_residency: (
            _OnlineJointResidencyTransaction | None
        ) = None
        self._online_joint_residency_history: deque[
            _OnlineJointResidencyTransaction
        ] = deque(maxlen=65_536)
        self._online_joint_last_residency_action: dict[
            str, tuple[ResidencyAction, float]
        ] = {}
        self._online_joint_hysteresis_audit: set[tuple[str, str]] = set()
        self._last_joint_detailed_audit_ms: float | None = None
        self._last_joint_detailed_signature: tuple[object, ...] | None = None
        self._joint_shadow_timing_samples: dict[str, deque[float]] = {
            name: deque(maxlen=65_536)
            for name in (
                "snapshot_build_ms",
                "snapshot_delta_apply_ms",
                "snapshot_materialize_ms",
                "safe_point_delta_capture_ms",
                "snapshot_trace_enqueue_ms",
                "snapshot_enqueue_ms",
                "plan_queue_wait_ms",
                "plan_compute_ms",
                "plan_publish_to_safe_point_ms",
                "validation_ms",
                "physical_commit_ms",
                "plan_age_ms",
            )
        }
        joint_shadow_requested = (
            self.config.joint_policy_shadow_mode
            and self.config.joint_observed_mode_enabled
        )
        self._joint_shadow_disabled_reason: str | None = None
        if not joint_shadow_requested:
            self._joint_shadow_disabled_reason = "disabled_by_config"
        elif not self.audit.enabled and not self.config.joint_policy_enabled:
            self._joint_shadow_disabled_reason = "runtime_audit_disabled"
        transfer_telemetry_path = self.config.transfer_telemetry_path
        if (
            transfer_telemetry_path is None
            and self.config.runtime_audit_path is not None
        ):
            transfer_telemetry_path = str(
                Path(self.config.runtime_audit_path).with_name(
                    "transfer_telemetry.jsonl"
                )
            )
        if transfer_telemetry_path == self.config.runtime_audit_path:
            raise ValueError(
                "transfer_telemetry_path must differ from runtime_audit_path"
            )
        self.transfer_telemetry_log = RuntimeAuditLog(
            transfer_telemetry_path,
            run_id=self.audit.run_id,
        )
        self.event_log = (
            JsonlRuntimeEventSink(self.config.runtime_event_log_path)
            if self.config.runtime_event_log_path is not None
            else None
        )
        self.event_server = (
            RuntimeEventDatagramServer(
                self.config.runtime_event_socket_path,
                self._process_events,
            )
            if self.config.runtime_event_socket_path is not None
            else None
        )
        self.tree_cache.beliefkv_observer = self
        self.sync_tree()
        if self._joint_shadow_disabled_reason is None:
            planner = AsyncSemanticJointPlanner(
                JointPlannerConfig(
                    fairness_lag_budget_ms=self.config.fairness_lag_budget_ms,
                    max_workflow_candidates=(
                        self.config.max_joint_workflow_candidates
                    ),
                    max_frontier_candidates_per_workflow=(
                        self.config.max_frontier_candidates_per_workflow
                    ),
                    max_total_frontier_candidates=(
                        self.config.max_total_frontier_candidates
                    ),
                    max_package_evaluations=(
                        self.config.max_joint_package_evaluations
                    ),
                    max_planning_budget_ms=self.config.max_joint_plan_budget_ms,
                    min_planning_budget_ms=self.config.min_joint_plan_budget_ms,
                    trigger_interval_budget_fraction=(
                        self.config.joint_trigger_budget_fraction
                    ),
                    max_plan_age_ms=self.config.max_joint_plan_age_ms,
                    residency_hysteresis_ms=self.config.residency_hysteresis_ms,
                )
            )
            self.joint_shadow_worker = LatestWinsJointPlanWorker(
                planner,
                assembler=IncrementalPolicyInputAssembler(self.config),
            )
        self.audit.emit(
            "runtime_initialized",
            self._now_ms(),
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
            host_capacity_bytes=self.config.host_capacity_bytes,
            kv_bytes_per_token=self.config.kv_bytes_per_token,
            runtime_event_channel_enabled=self.event_server is not None,
            hicache_capabilities=asdict(self.backend.capabilities),
            transfer_time_semantics={
                "submit": "scheduler_backend_submit",
                "start": "hicache_api_submit_begin_not_dma_start",
                "complete": "scheduler_observed_hicache_ack",
                "first_layer_ready": "unavailable",
                "compute_wait": "unavailable",
                "allocator_submit": "synchronous_allocator_and_copy_api_staging",
            },
            telemetry_availability={
                "hbm_allocator": True,
                "host_allocator": True,
                "pcie_utilization": False,
                "copy_engine_utilization": False,
                "gpu_compute_utilization": False,
            },
            reference_policy_snapshot={
                "enabled": self.policy_snapshot_log.enabled,
                "path": (
                    str(self.policy_snapshot_log.path)
                    if self.policy_snapshot_log.path is not None
                    else None
                ),
                "format": "replay_snapshot_jsonl_gzip",
                "writer_mode": "background_single_consumer",
                "min_interval_ms": (
                    self.config.reference_policy_snapshot_min_interval_ms
                ),
                "persist_interval_ms": (
                    self.config.reference_policy_snapshot_persist_interval_ms
                ),
                "max_pending": self.config.reference_policy_snapshot_max_pending,
                "hbm_bucket_bytes": self.config.reference_policy_hbm_bucket_bytes,
                "trace_sensitivity": (
                    self.config.reference_policy_trace_sensitivity
                ),
            },
            joint_plan_shadow={
                "requested": joint_shadow_requested,
                "enabled": self.joint_shadow_worker is not None,
                "disabled_reason": self._joint_shadow_disabled_reason,
                "planner": (
                    "belief_joint_semantic_predictive"
                    if self.config.joint_predictive_enabled
                    else "belief_joint_observed"
                ),
                "worker_queue": "latest_wins_capacity_1_lossless_delta_merge",
                "snapshot_builder": "worker_owned_incremental_mirror",
                "validation": "dependency_scoped_optimistic",
                "strict_global_comparator": True,
                "application_connected": self.config.joint_policy_enabled,
                "joint_policy_enabled_requested": self.config.joint_policy_enabled,
                "joint_predictive_enabled": self.config.joint_predictive_enabled,
                "max_plan_age_ms": self.config.max_joint_plan_age_ms,
            },
            observed_admission_scheduling={
                "enabled": self.config.observed_admission_scheduling_enabled,
                "state_source": "current_batch_epoch",
                "active_kv_high_watermark_ratio": (
                    self.config.observed_admission_active_kv_high_watermark_ratio
                ),
                "minimum_active_requests": (
                    self.config.observed_admission_min_active_requests
                ),
                "prediction_used": False,
                "running_batch_retraction": (
                    self.config.running_batch_retraction_enabled
                ),
                "retraction_mode": "observed_selective_drop_or_recompute",
                "residency_control": (
                    "validated_joint_plan_single_writer"
                    if self.config.joint_policy_enabled
                    else "reactive"
                ),
            },
            restore_liveness={
                "allocator_backed_lease": self.config.restore_lease_enabled,
                "debt_owned_funding_reservation": True,
                "service_grace_decode_tokens": (
                    self.config.restore_service_grace_decode_tokens
                ),
                "normal_admission_barrier_persists_during_active_lease": True,
            },
            request_token_trace={
                "enabled": self.request_token_trace_log.enabled,
                "path": (
                    str(self.request_token_trace_log.path)
                    if self.request_token_trace_log.path is not None
                    else None
                ),
                "writer_mode": "background_single_consumer",
                "encoding": RequestTokenTraceLog.ENCODING,
                "privacy_semantics": (
                    "run-local random token symbols preserve exact equality but "
                    "do not store the token-ID mapping"
                ),
            },
        )
        self._write_scheduler_identity()
        atexit.register(self.close)

    def _write_scheduler_identity(self) -> None:
        path = getattr(self, "_runtime_scheduler_pid_path", None)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "linux_start_time_ticks": _linux_process_start_time_ticks(os.getpid()),
            "run_id": getattr(getattr(self, "audit", None), "run_id", None),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _write_shutdown_ack(self) -> None:
        path = getattr(self, "_runtime_shutdown_ack_path", None)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "linux_start_time_ticks": _linux_process_start_time_ticks(os.getpid()),
            "run_id": getattr(getattr(self, "audit", None), "run_id", None),
            "shutdown_state": getattr(self, "_shutdown_state", None),
            "final_runtime_summary": (
                str(self._runtime_summary_path)
                if getattr(self, "_runtime_summary_path", None) is not None
                else None
            ),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _runtime_summary_payload(
        self, *, now_ms: float, final: bool
    ) -> dict[str, Any]:
        scheduler = getattr(self, "scheduler", None)
        running_batch = getattr(scheduler, "running_batch", None)
        controller = getattr(self, "controller", None)
        graph = getattr(controller, "graph", None)
        online_residency = getattr(self, "_pending_online_joint_residency", None)
        retraction = getattr(
            self, "_pending_running_retraction_transaction", None
        )
        current_view = getattr(self, "_current_online_joint_view", None)
        inflight_command_ids = sorted(
            getattr(controller, "inflight_command_ids", ()) or ()
        )
        pending_transaction_ids = tuple(
            item
            for item in (
                (
                    online_residency.transaction_id
                    if online_residency is not None
                    else None
                ),
                retraction.transaction_id if retraction is not None else None,
                *(
                    item.obligation_id
                    for item in (
                        getattr(self, "_restore_obligations", None).active()
                        if getattr(self, "_restore_obligations", None) is not None
                        else ()
                    )
                ),
            )
            if item is not None
        )
        missing_source_count = int(
            getattr(self, "_online_joint_counts", {}).get(
                "online_action_missing_source_joint_plan_id", 0
            )
        )
        physical_snapshot_samples = tuple(
            getattr(self, "_native_physical_snapshot_timing_samples", ())
        )
        physical_snapshot_counts = getattr(
            self, "_native_physical_snapshot_counts", Counter()
        )
        scheduler_step_count = int(
            physical_snapshot_counts.get("scheduler_step_count", 0)
        )
        physical_snapshot_build_ms = tuple(
            item.total_ms for item in physical_snapshot_samples
        )
        max_physical_snapshot = max(
            physical_snapshot_samples,
            key=lambda item: item.total_ms,
            default=None,
        )
        obligation_index = getattr(self, "_restore_obligations", None)
        all_obligations = (
            obligation_index.all() if obligation_index is not None else ()
        )
        obligation_state_counts = Counter(
            item.state.value for item in all_obligations
        )
        obligation_reason_counts = Counter(
            item.terminal_reason or "nonterminal" for item in all_obligations
        )
        non_user_unsatisfied = tuple(
            item.obligation_id
            for item in all_obligations
            if item.state != RestoreObligationState.SATISFIED
            and not (
                item.state == RestoreObligationState.CANCELLED
                and item.terminal_reason == "request_aborted"
            )
        )
        allowed_unrecoverable_reasons = {
            "authoritative_cache_reset",
            "backend_permanent_failure",
            "context_exceeds_hbm_capacity",
            "raw_input_unavailable",
        }
        failed_unrecoverable = tuple(
            item
            for item in all_obligations
            if item.state == RestoreObligationState.FAILED
        )
        failed_unrecoverable_evidence = {
            item.obligation_id: dict(
                getattr(
                    getattr(self, "_restore_transactions", {}).get(
                        item.request_id
                    ),
                    "failure_evidence",
                    {},
                )
            )
            for item in failed_unrecoverable
        }
        failed_unrecoverable_evidence_complete = all(
            item.terminal_reason in allowed_unrecoverable_reasons
            and bool(failed_unrecoverable_evidence.get(item.obligation_id))
            for item in failed_unrecoverable
        )
        shutdown_prepare_transactions = getattr(
            self, "_shutdown_prepare_transaction_snapshot", None
        )
        shutdown_cleanup_masked_unresolved = bool(
            shutdown_prepare_transactions
            and any(bool(value) for value in shutdown_prepare_transactions.values())
        )
        return {
            "schema_version": 1,
            "run_id": getattr(getattr(self, "audit", None), "run_id", None),
            "ts_ms": now_ms,
            "final": final,
            "shutdown_state": getattr(self, "_shutdown_state", "running"),
            "queue": {
                "waiting_request_count": len(
                    tuple(getattr(scheduler, "waiting_queue", ()) or ())
                ),
                "running_request_count": len(
                    tuple(getattr(running_batch, "reqs", ()) or ())
                ),
                "chunked_request_present": bool(
                    getattr(scheduler, "chunked_req", None)
                ),
            },
            "graph": {
                "workflow_count": len(getattr(graph, "workflows", {}) or {}),
                "invocation_count": len(getattr(graph, "invocations", {}) or {}),
                "context_count": len(getattr(graph, "contexts", {}) or {}),
            },
            "transactions": {
                "inflight_command_ids": inflight_command_ids,
                "pending_online_residency_transaction_id": (
                    online_residency.transaction_id
                    if online_residency is not None
                    else None
                ),
                "pending_retraction_transaction_id": (
                    retraction.transaction_id if retraction is not None else None
                ),
                "active_restore_obligation_ids": [
                    item.obligation_id
                    for item in (
                        getattr(self, "_restore_obligations", None).active()
                        if getattr(self, "_restore_obligations", None) is not None
                        else ()
                    )
                ],
                "active_restore_lease_ids": [
                    item.lease_id
                    for item in (
                        getattr(self, "_restore_leases", None).active()
                        if getattr(self, "_restore_leases", None) is not None
                        else ()
                    )
                ],
                "restore_lease_reserved_bytes": (
                    getattr(self, "_restore_leases", None).reserved_bytes
                    if getattr(self, "_restore_leases", None) is not None
                    else 0
                ),
                "restore_funding_reserved_bytes": sum(
                    len(item) * self.config.kv_bytes_per_token
                    for allocations in getattr(
                        self, "_restore_funding_allocations", {}
                    ).values()
                    for item in allocations
                ),
                "active_restore_service_grace": {
                    request_id: {
                        "obligation_id": grace.obligation_id,
                        "served_decode_tokens": grace.served_decode_tokens,
                        "required_decode_tokens": grace.required_decode_tokens,
                    }
                    for request_id, grace in sorted(
                        getattr(
                            self, "_restore_service_grace_by_request", {}
                        ).items()
                    )
                    if grace.active
                },
                "restore_obligation_outcomes": {
                    "state_counts": dict(sorted(obligation_state_counts.items())),
                    "terminal_reason_counts": dict(
                        sorted(obligation_reason_counts.items())
                    ),
                    "non_user_unsatisfied_obligation_ids": list(
                        non_user_unsatisfied
                    ),
                    "failed_unrecoverable_evidence": (
                        failed_unrecoverable_evidence
                    ),
                },
                "shutdown_prepare_snapshot": shutdown_prepare_transactions,
            },
            "joint_control": {
                "current_plan_id": (
                    current_view.plan_id if current_view is not None else None
                ),
                "online_counts": dict(
                    sorted(getattr(self, "_online_joint_counts", {}).items())
                ),
                "retraction_counts": dict(
                    sorted(
                        getattr(self, "_running_retraction_counts", {}).items()
                    )
                ),
                "restore_micro_gate": dict(
                    getattr(self, "_restore_micro_gate_state", {})
                ),
            },
            "physical_ownership_snapshot": {
                "semantics": "native_queue_and_transfer_state_rebuild",
                "call_count": int(physical_snapshot_counts.get("call_count", 0)),
                "scheduler_step_count": scheduler_step_count,
                "calls_per_scheduler_step": (
                    physical_snapshot_counts.get("call_count", 0)
                    / scheduler_step_count
                    if scheduler_step_count
                    else 0.0
                ),
                "retained_sample_count": len(physical_snapshot_samples),
                "build_ms": {
                    "total": sum(physical_snapshot_build_ms),
                    "mean": (
                        sum(physical_snapshot_build_ms)
                        / len(physical_snapshot_build_ms)
                        if physical_snapshot_build_ms
                        else 0.0
                    ),
                    "p50": percentile(physical_snapshot_build_ms, 50),
                    "p95": percentile(physical_snapshot_build_ms, 95),
                    "p99": percentile(physical_snapshot_build_ms, 99),
                    "max": max(physical_snapshot_build_ms, default=0.0),
                },
                "queue_records_scanned": {
                    "total": int(
                        physical_snapshot_counts.get(
                            "queue_records_scanned_total", 0
                        )
                    ),
                    "max_per_call": max(
                        (
                            item.queue_record_count
                            for item in physical_snapshot_samples
                        ),
                        default=0,
                    ),
                },
                "metadata_records_scanned": {
                    "total": int(
                        physical_snapshot_counts.get(
                            "metadata_records_scanned_total", 0
                        )
                    ),
                    "max_per_call": max(
                        (
                            item.metadata_record_count
                            for item in physical_snapshot_samples
                        ),
                        default=0,
                    ),
                },
                "matched_snapshot_count": {
                    "total": int(
                        physical_snapshot_counts.get(
                            "matched_snapshot_count_total", 0
                        )
                    ),
                    "max_per_call": max(
                        (
                            item.matched_record_count
                            for item in physical_snapshot_samples
                        ),
                        default=0,
                    ),
                },
                "cache_hit_count": int(
                    physical_snapshot_counts.get("cache_hit_count", 0)
                ),
                "commit_readset": {
                    "validation_count": int(
                        physical_snapshot_counts.get(
                            "commit_readset_validation_count", 0
                        )
                    ),
                    "stale_count": int(
                        physical_snapshot_counts.get(
                            "commit_readset_stale", 0
                        )
                    ),
                    "validation_ms_total": (
                        physical_snapshot_counts.get(
                            "commit_readset_validation_us_total", 0
                        )
                        / 1000.0
                    ),
                },
                "phase_ms": {
                    name: {
                        "mean": (
                            sum(getattr(item, name) for item in physical_snapshot_samples)
                            / len(physical_snapshot_samples)
                            if physical_snapshot_samples
                            else 0.0
                        ),
                        "p99": percentile(
                            [getattr(item, name) for item in physical_snapshot_samples],
                            99,
                        ),
                        "max": max(
                            (
                                getattr(item, name)
                                for item in physical_snapshot_samples
                            ),
                            default=0.0,
                        ),
                    }
                    for name in (
                        "queue_collection_ms",
                        "metadata_indexing_ms",
                        "radix_ownership_lookup_ms",
                        "operation_indexing_ms",
                        "sorting_allocation_ms",
                    )
                },
                "amortized_cpu_ms_per_scheduler_step": (
                    sum(physical_snapshot_build_ms) / scheduler_step_count
                    if scheduler_step_count
                    else 0.0
                ),
                "mean_us_per_matched_record": (
                    physical_snapshot_counts.get("build_us_total", 0)
                    / physical_snapshot_counts.get(
                        "matched_snapshot_count_total", 1
                    )
                    if physical_snapshot_counts.get(
                        "matched_snapshot_count_total", 0
                    )
                    else 0.0
                ),
                "max_outlier": (
                    {
                        "total_ms": max_physical_snapshot.total_ms,
                        "cold_build": max_physical_snapshot.cold_build,
                        "gc_collections": max_physical_snapshot.gc_collections,
                        "dominant_phase": max(
                            (
                                (
                                    name,
                                    getattr(max_physical_snapshot, name),
                                )
                                for name in (
                                    "queue_collection_ms",
                                    "metadata_indexing_ms",
                                    "radix_ownership_lookup_ms",
                                    "operation_indexing_ms",
                                    "sorting_allocation_ms",
                                )
                            ),
                            key=lambda item: item[1],
                        )[0],
                    }
                    if max_physical_snapshot is not None
                    else None
                ),
            },
            "correctness_gates": {
                "all_online_actions_have_source_joint_plan_id": (
                    missing_source_count == 0
                ),
                "missing_source_joint_plan_id_count": missing_source_count,
                "no_pending_transactions": (
                    not inflight_command_ids and not pending_transaction_ids
                ),
                "pending_transaction_ids": list(pending_transaction_ids),
                "shutdown_summary_complete": bool(
                    final
                    and getattr(self, "_shutdown_state", "running")
                    == "acknowledged"
                ),
                "all_non_user_cancelled_obligations_satisfied": (
                    not non_user_unsatisfied
                ),
                "shutdown_cleanup_did_not_mask_unresolved_transactions": (
                    not shutdown_cleanup_masked_unresolved
                ),
                "failed_unrecoverable_has_explicit_evidence": (
                    failed_unrecoverable_evidence_complete
                ),
            },
            "audit_writer": (
                self.audit.summary()
                if callable(getattr(getattr(self, "audit", None), "summary", None))
                else None
            ),
        }

    def _write_latest_runtime_summary(
        self,
        *,
        now_ms: float,
        force: bool,
        final: bool = False,
    ) -> None:
        path = getattr(self, "_runtime_summary_path", None)
        config = getattr(self, "config", None)
        if path is None or config is None:
            return
        last_ms = getattr(self, "_last_runtime_summary_ms", None)
        if (
            not force
            and last_ms is not None
            and now_ms - last_ms < config.runtime_summary_interval_ms
        ):
            return
        payload = self._runtime_summary_payload(now_ms=now_ms, final=final)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{self.audit.run_id}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self._last_runtime_summary_ms = now_ms

    def prepare_shutdown(self) -> None:
        """Stop admitting new control plans at a scheduler-safe boundary."""

        if getattr(self, "_shutdown_state", "running") != "running":
            return
        now_ms = float(self._now_ms())
        self._shutdown_state = "preparing"
        self._shutdown_prepare_ms = now_ms
        self._shutdown_prepare_transaction_snapshot = (
            self._transaction_conservation_snapshot()
        )
        audit = getattr(self, "audit", None)
        if audit is not None:
            audit.emit(
                "shutdown_prepare",
                now_ms,
                transaction_snapshot=self._shutdown_prepare_transaction_snapshot,
                inflight_command_count=len(
                    getattr(getattr(self, "controller", None), "inflight_command_ids", ())
                ),
                pending_online_residency=bool(
                    getattr(self, "_pending_online_joint_residency", None)
                ),
                pending_retraction=bool(
                    getattr(self, "_pending_running_retraction_transaction", None)
                ),
                active_restore_obligation_count=len(
                    getattr(self, "_restore_obligations", None).active()
                    if getattr(self, "_restore_obligations", None) is not None
                    else ()
                ),
            )
        self._write_latest_runtime_summary(now_ms=now_ms, force=True)

    def _transaction_conservation_snapshot(self) -> dict[str, object]:
        controller = getattr(self, "controller", None)
        obligation_index = getattr(self, "_restore_obligations", None)
        lease_index = getattr(self, "_restore_leases", None)
        active_obligations = (
            obligation_index.active() if obligation_index is not None else ()
        )
        active_leases = lease_index.active() if lease_index is not None else ()
        nonterminal_transactions = tuple(
            sorted(
                transaction.transaction_id
                for transaction in getattr(
                    self, "_restore_transactions", {}
                ).values()
                if not transaction.stage.terminal
            )
        )
        return {
            "inflight_command_ids": list(
                getattr(controller, "inflight_command_ids", ()) or ()
            ),
            "queued_command_ids": [
                item.command_id
                for item in (
                    controller.command_queue.pending_commands()
                    if controller is not None
                    and getattr(controller, "command_queue", None) is not None
                    else ()
                )
            ],
            "active_obligation_ids": [
                item.obligation_id for item in active_obligations
            ],
            "active_lease_ids": [item.lease_id for item in active_leases],
            "active_funding_request_ids": sorted(
                getattr(self, "_restore_funding_allocations", {})
            ),
            "active_prefix_pin_request_ids": sorted(
                getattr(self, "_restore_lease_pins", {})
            ),
            "nonterminal_restore_transaction_ids": list(
                nonterminal_transactions
            ),
            "command_subscription_ids": sorted(
                getattr(self, "_restore_command_to_request", {})
            ),
            "pending_retraction_transaction_id": (
                getattr(
                    self, "_pending_running_retraction_transaction", None
                ).transaction_id
                if getattr(
                    self, "_pending_running_retraction_transaction", None
                )
                is not None
                else None
            ),
            "pending_residency_transaction_id": (
                getattr(self, "_pending_online_joint_residency", None).transaction_id
                if getattr(self, "_pending_online_joint_residency", None)
                is not None
                else None
            ),
        }

    def _drain_shutdown_acks(self) -> None:
        bridge = getattr(self, "bridge", None)
        controller = getattr(self, "controller", None)
        config = getattr(self, "config", None)
        if bridge is None or controller is None or config is None:
            return
        deadline = time.monotonic() + config.shutdown_drain_timeout_ms / 1000.0
        while getattr(controller, "inflight_command_ids", ()):
            acks = tuple(bridge.drain_acks())
            if acks:
                now_ms = float(self._now_ms())
                for ack in acks:
                    self.audit.emit(
                        "transfer_acknowledged",
                        now_ms,
                        command_id=ack.command_id,
                        status=ack.status.value,
                        actual_bytes=ack.actual_bytes,
                        page_count=len(ack.page_handles),
                        reason=ack.reason,
                        blocker_codes=sorted(
                            {item.code.value for item in ack.blockers}
                        ),
                        blockers=self._audit_blockers(ack.blockers),
                        shutdown_drain=True,
                    )
                self._retire_h2d_commands(acks)
                try:
                    self.sync_tree(force=True)
                except Exception as error:
                    self.audit.emit(
                        "shutdown_tree_sync_failed",
                        now_ms,
                        error=f"{type(error).__name__}: {error}",
                    )
                self._advance_retraction_transaction(acks, now_ms=now_ms)
                self._advance_online_joint_residency(acks, now_ms=now_ms)
                self._advance_restore_obligations(acks, now_ms=now_ms)
            if not getattr(controller, "inflight_command_ids", ()):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)

    def _abort_shutdown_transactions(self, *, now_ms: float) -> None:
        retraction = getattr(
            self, "_pending_running_retraction_transaction", None
        )
        if retraction is not None:
            retraction.stage = "aborted"
            retraction.failure_reason = "runtime_shutdown"
            self._running_retraction_counts["aborted"] += 1
            self.audit.emit(
                "running_retraction_transaction_terminal",
                now_ms,
                transaction_id=retraction.transaction_id,
                source_joint_plan_id=retraction.source_joint_plan_id,
                status="aborted",
                reason="runtime_shutdown",
                pending_command_id=retraction.pending_command_id,
            )
            self._pending_running_retraction_transaction = None
        residency = getattr(self, "_pending_online_joint_residency", None)
        if residency is not None:
            residency.stage = "aborted"
            residency.completed_ts_ms = now_ms
            residency.failure_reason = "runtime_shutdown"
            self._online_joint_counts["residency_aborted"] += 1
            self.audit.emit(
                "online_joint_residency_terminal",
                now_ms,
                transaction_id=residency.transaction_id,
                plan_id=residency.plan_id,
                intent_index=residency.intent_index,
                action=residency.action.value,
                command_id=residency.command_id,
                command_kind=residency.command_kind.value,
                source_bundle_id=residency.source_bundle_id,
                physical_bundle_id=residency.physical_bundle_id,
                context_id=residency.context_id,
                status="aborted",
                actual_bytes=residency.actual_bytes,
                reason="runtime_shutdown",
                blocker_codes=[],
            )
            self._pending_online_joint_residency = None
        index = getattr(self, "_restore_obligations", None)
        if index is not None:
            for obligation in index.active():
                self._finish_restore_obligation(
                    obligation.request_id,
                    RestoreObligationState.CANCELLED,
                    now_ms=now_ms,
                    reason="runtime_shutdown",
                )
        for request_id in tuple(
            getattr(self, "_restore_service_grace_by_request", {})
        ):
            self._cancel_restore_service_grace(
                request_id,
                now_ms=now_ms,
                reason="runtime_shutdown",
            )
        getattr(self, "_restore_command_to_request", {}).clear()
        getattr(self, "_restore_funding_target_by_command", {}).clear()

    def close(self) -> None:
        """Prepare, drain and finalize process-local runtime state once."""

        if self._closed:
            return
        self.prepare_shutdown()
        self._closed = True
        event_server = self.event_server
        self.event_server = None
        if event_server is not None:
            event_server.close()
        event_log = self.event_log
        self.event_log = None
        if event_log is not None:
            event_log.close()
        self._drain_shutdown_acks()
        self._abort_shutdown_transactions(now_ms=float(self._now_ms()))
        if getattr(self, "_pending_request_physical_finish_by_id", None):
            try:
                self.sync_tree(force=True)
                self._flush_request_physical_finishes()
            except Exception as error:
                self.audit.emit(
                    "request_physical_checkpoint_failed",
                    self._now_ms(),
                    phase="shutdown_flush",
                    error=f"{type(error).__name__}: {error}",
                    pending_request_count=len(
                        self._pending_request_physical_finish_by_id
                    ),
                )
        self._emit_controller_timing_summary()
        controller = getattr(self, "controller", None)
        transfer_guard = getattr(controller, "transfer_guard", None)
        if transfer_guard is not None:
            self.audit.emit(
                "transfer_retry_guard_summary",
                self._now_ms(),
                **transfer_guard.summary(),
            )
        config = getattr(self, "config", None)
        if getattr(config, "queue_service_observer_enabled", False):
            observer_samples = tuple(
                getattr(self, "_gpu_service_observer_timing_samples", ())
            )
            self.audit.emit(
                "gpu_service_observer_summary",
                self._now_ms(),
                sample_count=self._gpu_service_sample_count,
                sample_cap_count=getattr(
                    self, "_gpu_service_sample_cap_count", 0
                ),
                pending_launch_count=len(self._gpu_service_launches),
                max_samples=config.queue_service_observer_max_samples,
                observer_cpu_ms={
                    "count": len(observer_samples),
                    "build_p50": percentile(
                        [item[0] for item in observer_samples], 50
                    ),
                    "build_p95": percentile(
                        [item[0] for item in observer_samples], 95
                    ),
                    "build_p99": percentile(
                        [item[0] for item in observer_samples], 99
                    ),
                    "audit_enqueue_p50": percentile(
                        [item[1] for item in observer_samples], 50
                    ),
                    "audit_enqueue_p95": percentile(
                        [item[1] for item in observer_samples], 95
                    ),
                    "audit_enqueue_p99": percentile(
                        [item[1] for item in observer_samples], 99
                    ),
                },
            )
        if getattr(self, "_lock_service_ledger", None) is not None:
            self.audit.emit(
                "lock_service_observer_summary",
                self._now_ms(),
                resource_snapshot_count=getattr(
                    self, "_lock_service_snapshot_count", 0
                ),
                observer_error_count=getattr(
                    self, "_lock_service_observer_error_count", 0
                ),
                peak_locked_but_not_served_gpu_bytes_100ms=getattr(
                    self, "_lock_service_peak_bytes", {}
                ).get("100ms", 0),
                peak_locked_but_not_served_gpu_bytes_500ms=getattr(
                    self, "_lock_service_peak_bytes", {}
                ).get("500ms", 0),
                service_evidence="completed_gpu_batch",
                provenance_scope="running_request_last_node_to_radix_root",
            )
        if getattr(config, "observed_admission_scheduling_enabled", False):
            self.audit.emit(
                "observed_admission_summary",
                self._now_ms(),
                mode_counts=dict(
                    sorted(
                        getattr(
                            self, "_observed_admission_mode_counts", {}
                        ).items()
                    )
                ),
                peak_active_kv_footprint_bytes=getattr(
                    self, "_observed_admission_peak_active_kv_bytes", 0
                ),
                peak_active_kv_pressure=getattr(
                    self, "_observed_admission_peak_pressure", 0.0
                ),
                running_batch_retraction=(
                    config.running_batch_retraction_enabled
                ),
                prediction_used=False,
            )
        if getattr(config, "running_batch_retraction_enabled", False):
            self._close_pending_running_retraction_barrier(
                now_ms=float(self._now_ms())
            )
            transactions = tuple(
                getattr(self, "_running_retraction_transactions", ())
            )
            preview_compute_us = tuple(
                getattr(self, "_running_retraction_preview_compute_us", ())
            )
            barrier_preview_compute_us = tuple(
                getattr(
                    self,
                    "_running_retraction_barrier_preview_compute_us",
                    (),
                )
            )
            self.audit.emit(
                "running_retraction_summary",
                self._now_ms(),
                retained_transaction_count=len(transactions),
                transaction_history_capacity=65_536,
                transaction_count=getattr(
                    self, "_running_retraction_counts", {}
                ).get("planned", 0),
                stage_counts=dict(
                    sorted(
                        Counter(item.stage for item in transactions).items()
                    )
                ),
                decision_counts=dict(
                    sorted(
                        getattr(self, "_running_retraction_counts", {}).items()
                    )
                ),
                barrier_outcome_counts=dict(
                    sorted(
                        getattr(
                            self,
                            "_running_retraction_barrier_outcomes",
                            {},
                        ).items()
                    )
                ),
                tentative_unlock_preview_counts=dict(
                    sorted(
                        getattr(
                            self,
                            "_running_retraction_preview_counts",
                            {},
                        ).items()
                    )
                ),
                tentative_unlock_preview_compute_us_p50=(
                    percentile(preview_compute_us, 50)
                    if preview_compute_us
                    else None
                ),
                tentative_unlock_preview_compute_us_p95=(
                    percentile(preview_compute_us, 95)
                    if preview_compute_us
                    else None
                ),
                tentative_unlock_preview_compute_us_p99=(
                    percentile(preview_compute_us, 99)
                    if preview_compute_us
                    else None
                ),
                tentative_barrier_preview_compute_us_p50=(
                    percentile(barrier_preview_compute_us, 50)
                    if barrier_preview_compute_us
                    else None
                ),
                tentative_barrier_preview_compute_us_p95=(
                    percentile(barrier_preview_compute_us, 95)
                    if barrier_preview_compute_us
                    else None
                ),
                tentative_barrier_preview_compute_us_p99=(
                    percentile(barrier_preview_compute_us, 99)
                    if barrier_preview_compute_us
                    else None
                ),
                pending_barrier_intent_id=(
                    pending_barrier.barrier_intent_id
                    if (
                        pending_barrier := getattr(
                            self,
                            "_pending_running_retraction_barrier",
                            None,
                        )
                    )
                    is not None
                    else None
                ),
                request_retraction_counts=dict(
                    sorted(
                        getattr(self, "_retraction_counts_by_request", {}).items()
                    )
                ),
                actual_reclaim_capacity_bytes=getattr(
                    self, "_running_retraction_actual_reclaim_bytes", 0
                ),
                actual_engine_lock_release_bytes=getattr(
                    self, "_running_retraction_actual_lock_release_bytes", 0
                ),
                prediction_used=False,
            )
            restore_index = getattr(self, "_restore_obligations", None)
            if restore_index is not None:
                obligations = restore_index.all()
                terminal_waits = tuple(
                    max(0.0, item.terminal_ts_ms - item.created_ts_ms)
                    for item in obligations
                    if item.terminal_ts_ms is not None
                )
                self.audit.emit(
                    "restore_obligation_summary",
                    self._now_ms(),
                    obligation_count=len(obligations),
                    state_counts=dict(
                        sorted(Counter(item.state.value for item in obligations).items())
                    ),
                    active_obligation_ids=[
                        item.obligation_id for item in restore_index.active()
                    ],
                    active_restore_lease_ids=[
                        item.lease_id
                        for item in (
                            getattr(self, "_restore_leases", None).active()
                            if getattr(self, "_restore_leases", None) is not None
                            else ()
                        )
                    ],
                    restore_lease_reserved_bytes=(
                        getattr(self, "_restore_leases", None).reserved_bytes
                        if getattr(self, "_restore_leases", None) is not None
                        else 0
                    ),
                    restore_funding_reserved_bytes=sum(
                        len(item) * self.config.kv_bytes_per_token
                        for allocations in getattr(
                            self, "_restore_funding_allocations", {}
                        ).values()
                        for item in allocations
                    ),
                    active_service_grace_count=sum(
                        grace.active
                        for grace in getattr(
                            self, "_restore_service_grace_by_request", {}
                        ).values()
                    ),
                    decision_counts=dict(
                        sorted(
                            getattr(self, "_restore_obligation_counts", {}).items()
                        )
                    ),
                    terminal_wait_ms_p50=(
                        percentile(terminal_waits, 50) if terminal_waits else None
                    ),
                    terminal_wait_ms_p95=(
                        percentile(terminal_waits, 95) if terminal_waits else None
                    ),
                    terminal_wait_ms_max=max(terminal_waits, default=None),
                    total_restored_bytes=sum(
                        item.restored_bytes for item in obligations
                    ),
                    total_funding_reclaim_bytes=sum(
                        item.funding_reclaim_bytes for item in obligations
                    ),
                    liveness_escalation_count=sum(
                        item.liveness_escalated for item in obligations
                    ),
                )
        if getattr(config, "joint_policy_enabled", False):
            residency_history = tuple(
                getattr(self, "_online_joint_residency_history", ())
            )
            predictive_counts = dict(
                sorted(getattr(self, "_joint_predictive_counts", {}).items())
            )
            prediction_used = bool(
                predictive_counts
                or getattr(config, "joint_predictive_enabled", False)
            )
            terminal_stages = {"completed", "failed", "aborted"}
            pending_residency = getattr(
                self, "_pending_online_joint_residency", None
            )
            self.audit.emit(
                "online_joint_control_summary",
                self._now_ms(),
                decision_counts=dict(
                    sorted(getattr(self, "_online_joint_counts", {}).items())
                ),
                residency_transaction_count=len(residency_history),
                residency_stage_counts=dict(
                    sorted(Counter(item.stage for item in residency_history).items())
                ),
                residency_terminal_count=sum(
                    item.stage in terminal_stages for item in residency_history
                ),
                unresolved_residency_count=sum(
                    item.stage not in terminal_stages for item in residency_history
                ),
                pending_residency_transaction_id=(
                    pending_residency.transaction_id
                    if pending_residency is not None
                    else None
                ),
                current_plan_id=(
                    current_view.plan_id
                    if (
                        current_view := getattr(
                            self, "_current_online_joint_view", None
                        )
                    )
                    is not None
                    else None
                ),
                reactive_transfer_role=(
                    "disabled_as_online_policy_source_when_joint_enabled"
                ),
                prediction_used=prediction_used,
                joint_predictive_enabled=getattr(
                    config, "joint_predictive_enabled", False
                ),
                predictive_decision_counts=predictive_counts,
            )
        joint_shadow_worker = getattr(self, "joint_shadow_worker", None)
        if joint_shadow_worker is not None:
            worker_closed = joint_shadow_worker.close()
            self.joint_shadow_worker = None
            self._emit_joint_shadow_summary(
                joint_shadow_worker,
                worker_closed=worker_closed,
            )
        token_trace_log = getattr(self, "request_token_trace_log", None)
        if token_trace_log is not None:
            token_trace_error = None
            try:
                token_trace_log.close()
            except Exception as error:
                token_trace_error = f"{type(error).__name__}: {error}"
            self.audit.emit(
                "request_token_trace_summary",
                self._now_ms(),
                path=(
                    str(token_trace_log.path)
                    if token_trace_log.path is not None
                    else None
                ),
                trace_count=token_trace_log.count,
                written_trace_count=token_trace_log.written_count,
                error=token_trace_error,
            )
        policy_snapshot_log = getattr(self, "policy_snapshot_log", None)
        if policy_snapshot_log is not None:
            policy_snapshot_log.close()
            self.audit.emit(
                "policy_snapshot_summary",
                self._now_ms(),
                snapshot_count=policy_snapshot_log.count,
                written_snapshot_count=policy_snapshot_log.written_count,
                dropped_snapshot_count=policy_snapshot_log.dropped_count,
                path=(
                    str(policy_snapshot_log.path)
                    if policy_snapshot_log.path is not None
                    else None
                ),
            )
        audit_summary = getattr(self.audit, "summary", None)
        if callable(audit_summary):
            self.audit.emit(
                "runtime_audit_writer_summary",
                self._now_ms(),
                **audit_summary(),
            )
        self._shutdown_state = "acknowledged"
        self._write_latest_runtime_summary(
            now_ms=float(self._now_ms()), force=True, final=True
        )
        self.audit.emit("shutdown_ack", self._now_ms())
        self.audit.emit("runtime_shutdown", self._now_ms())
        transfer_telemetry_log = getattr(self, "transfer_telemetry_log", None)
        if transfer_telemetry_log is not None:
            transfer_telemetry_log.close()
        self.audit.close()
        self._write_shutdown_ack()

    @staticmethod
    def _load_config(scheduler: Any, config_path: str | None) -> BeliefKVConfig:
        raw: dict[str, Any] = {}
        if config_path:
            config_file = Path(config_path).resolve()
            value = json.loads(config_file.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("BeliefKV config must contain a JSON object")
            raw.update(value)
            model_path = raw.get("predictor_model_path")
            if model_path and not Path(str(model_path)).is_absolute():
                raw["predictor_model_path"] = str(
                    (config_file.parent / str(model_path)).resolve()
                )
            for field_name in (
                "runtime_audit_path",
                "runtime_summary_path",
                "transfer_telemetry_path",
                "runtime_event_socket_path",
                "runtime_event_log_path",
                "request_token_trace_path",
            ):
                value_path = raw.get(field_name)
                if value_path and not Path(str(value_path)).is_absolute():
                    raw[field_name] = str(
                        (config_file.parent / str(value_path)).resolve()
                    )
        kv_bytes = int(raw.get("kv_bytes_per_token", 57344))
        max_tokens = int(getattr(scheduler, "max_total_num_tokens", 0))
        if "hbm_capacity_bytes" not in raw:
            if max_tokens <= 0:
                raise ValueError("cannot derive KV HBM capacity from scheduler")
            raw["hbm_capacity_bytes"] = max_tokens * kv_bytes
        host_pool = getattr(scheduler.tree_cache, "token_to_kv_pool_host", None)
        host_tokens = int(getattr(host_pool, "size", 0))
        if host_tokens > 0:
            # The allocator is the physical source of truth. A stale JSON value
            # must not let policy admission exceed or underuse the real pool.
            raw["host_capacity_bytes"] = host_tokens * kv_bytes
        elif "host_capacity_bytes" not in raw:
            raw["host_capacity_bytes"] = max(1, max_tokens * 2 * kv_bytes)
        if "reserve_hbm_bytes" not in raw:
            raw["reserve_hbm_bytes"] = min(
                1 << 30, int(raw["hbm_capacity_bytes"]) // 8
            )
        return BeliefKVConfig.from_mapping(raw)

    def register_visible_request(self, req: Any) -> bool | None:
        """Register a tagged request while leaving ownership with SGLang.

        ``None`` means untagged, ``True`` means the request may enter the native
        queue, and ``False`` means a terminal request was rejected.
        """

        raw_metadata = getattr(req, "beliefkv_metadata", None)
        if raw_metadata is None:
            return None
        metadata = (
            raw_metadata
            if isinstance(raw_metadata, BeliefKVRequestMetadata)
            else BeliefKVRequestMetadata.from_wire(raw_metadata)
        )
        req.beliefkv_metadata = metadata
        if self._metadata_scope_is_terminal(metadata):
            self.scheduler.send_to_tokenizer.send_pyobj(
                self._new_abort_request(req.rid)
            )
            self.audit.emit(
                "terminal_request_rejected",
                self._now_ms(),
                request_id=req.rid,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                phase="before_native_queue",
            )
            return False
        self._ensure_causal_identity(metadata)
        if (
            req.rid in self._request_metadata_by_id
            or self.controller.visible_admission.get(str(req.rid)) is not None
        ):
            raise SGLangBackendError(f"duplicate visible request id: {req.rid}")
        max_new_tokens = int(getattr(req.sampling_params, "max_new_tokens", 0) or 0)
        req.init_next_round_input(self.tree_cache)
        estimated_cache_hit_tokens = len(getattr(req, "prefix_indices", ()))
        uncached_prompt_tokens = max(
            0, len(req.origin_input_ids) - estimated_cache_hit_tokens
        )
        submitted_ts_ms = self._now_ms()
        admission_request = AdmissionRequest(
            request_id=str(req.rid),
            workflow_id=metadata.root_workflow_id,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            context_epoch=metadata.context_epoch,
            submitted_ts_ms=submitted_ts_ms,
            uncached_prompt_tokens=uncached_prompt_tokens,
            expected_output_tokens=max_new_tokens,
            kv_bytes_per_token=self.config.kv_bytes_per_token,
            prompt_tokens=len(req.origin_input_ids),
        )
        self.controller.policy_control_state(submitted_ts_ms)
        transition_generation, transition_open = self._workflow_transition_state(
            metadata.root_workflow_id
        )
        self.controller.register_visible_request(
            admission_request,
            transition_generation=transition_generation,
            bundle_generations=self._context_bundle_generations(
                metadata.context_id
            ),
        )
        if transition_open:
            self.controller.visible_admission.set_policy_blocked(
                str(req.rid), reason="transition_open"
            )
        self._request_metadata_by_id[req.rid] = metadata
        submitted_by_id = getattr(self, "_request_submitted_ts_by_id", None)
        if submitted_by_id is None:
            submitted_by_id = {}
            self._request_submitted_ts_by_id = submitted_by_id
        submitted_by_id[req.rid] = submitted_ts_ms
        token_trace_sequence = self._record_request_token_trace(
            "request_prompt",
            submitted_ts_ms,
            req.origin_input_ids,
            request_id=str(req.rid),
            workflow_id=metadata.root_workflow_id,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            context_epoch=metadata.context_epoch,
        )
        self.audit.emit(
            "request_visible_pending",
            self._now_ms(),
            request_id=req.rid,
            workflow_id=metadata.root_workflow_id,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            context_epoch=metadata.context_epoch,
            prompt_tokens=len(req.origin_input_ids),
            estimated_cache_hit_tokens=estimated_cache_hit_tokens,
            uncached_prompt_tokens=uncached_prompt_tokens,
            expected_output_tokens=max_new_tokens,
            request_token_trace_sequence=token_trace_sequence,
            transition_generation=transition_generation,
            transition_open=transition_open,
            queue_owner="sglang_native_waiting_queue",
        )
        return True

    def on_requests_requeued(
        self,
        requests: list[Any] | tuple[Any, ...],
        *,
        is_retracted: bool,
    ) -> None:
        """Restore side state for native requests returned to the waiting queue."""

        for req in requests:
            metadata = self._metadata(req)
            request_id = str(getattr(req, "rid", ""))
            if metadata is None or not request_id:
                continue
            if self.controller.visible_admission.get(request_id) is not None:
                continue
            refresh = getattr(req, "init_next_round_input", None)
            if callable(refresh):
                refresh(self.tree_cache)
            origin_input_ids = getattr(req, "origin_input_ids", None)
            prefix_indices = getattr(req, "prefix_indices", None)
            max_new_tokens = int(
                getattr(getattr(req, "sampling_params", None), "max_new_tokens", 0)
                or 0
            )
            submitted_ts_ms = getattr(
                self, "_request_submitted_ts_by_id", {}
            ).get(request_id, self._now_ms())
            transition_generation, transition_open = self._workflow_transition_state(
                metadata.root_workflow_id
            )
            self.controller.register_visible_request(
                AdmissionRequest(
                    request_id=request_id,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    context_epoch=metadata.context_epoch,
                    submitted_ts_ms=submitted_ts_ms,
                    uncached_prompt_tokens=max(
                        0,
                        _sequence_length(origin_input_ids)
                        - _sequence_length(prefix_indices),
                    ),
                    expected_output_tokens=max(
                        0,
                        max_new_tokens
                        - _sequence_length(getattr(req, "output_ids", None)),
                    ),
                    kv_bytes_per_token=self.config.kv_bytes_per_token,
                    prompt_tokens=_sequence_length(origin_input_ids),
                ),
                transition_generation=transition_generation,
                bundle_generations=self._context_bundle_generations(
                    metadata.context_id
                ),
            )
            cooldown_until = getattr(
                self, "_retraction_cooldown_until_by_request", {}
            ).get(request_id, 0.0)
            selective_retraction = request_id in getattr(
                self, "_pending_selective_retraction_ids", set()
            )
            obligation = self._restore_obligation_index().get(request_id)
            if obligation is not None and not obligation.state.terminal:
                obligation.requeued = True
                required = self._request_path_extent_ids(
                    req, metadata.context_id, cpu_only=True
                )
                if required is None:
                    required = obligation.required_extent_ids
                obligation.set_required_extents(
                    required,
                    restore_bytes=self._extent_ids_bytes(required),
                    now_ms=self._now_ms(),
                )
                self._sync_visible_gate_state(
                    request_id, metadata, req=req
                )
            elif selective_retraction or self._now_ms() < cooldown_until:
                self.controller.visible_admission.set_policy_blocked(
                    request_id, reason="retraction_cooldown"
                )
            elif transition_open:
                self.controller.visible_admission.set_policy_blocked(
                    request_id, reason="transition_open"
                )
            self.audit.emit(
                "request_visible_requeued",
                self._now_ms(),
                request_id=request_id,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                is_retracted=is_retracted,
                transition_generation=transition_generation,
                selective_retraction=selective_retraction,
                restore_obligation_id=(
                    obligation.obligation_id if obligation is not None else None
                ),
                restore_obligation_state=(
                    obligation.state.value if obligation is not None else None
                ),
                cooldown_until_ms=(
                    cooldown_until if selective_retraction else None
                ),
            )
            getattr(self, "_pending_selective_retraction_ids", set()).discard(
                request_id
            )

    def _running_retraction_barrier_state(
        self,
        running_batch: Any,
        *,
        now_ms: float,
        replacements: tuple[RetractionReplacement, ...] | None = None,
        native_reclaim_bytes: int | None = None,
        active_kv_footprint_bytes: int | None = None,
    ) -> dict[str, Any]:
        requests = tuple(getattr(running_batch, "reqs", ()) or ())
        if replacements is None:
            replacements = self._running_retraction_replacements(now_ms=now_ms)
        if native_reclaim_bytes is None:
            native_reclaim_bytes = self._native_reclaim_capacity_bytes()
        if active_kv_footprint_bytes is None:
            active = self._observed_admission_snapshot(
                native_available_hbm_bytes=native_reclaim_bytes,
                native_max_requests=len(replacements),
            )
            active_kv_footprint_bytes = active.active_kv_footprint_bytes
        active_budget_bytes = int(
            max(
                0,
                self.config.hbm_capacity_bytes - self.config.reserve_hbm_bytes,
            )
            * self.config.observed_admission_active_kv_high_watermark_ratio
        )
        replacement_deficit_bytes = (
            max(
                0,
                replacements[0].estimated_incremental_bytes
                - native_reclaim_bytes,
            )
            if replacements
            else 0
        )

        allocator_available_bytes: int | None = None
        scheduler = getattr(self, "scheduler", None)
        if getattr(scheduler, "token_to_kv_pool_allocator", None) is not None:
            allocator_available_bytes = self._allocator_available_bytes()

        breakdown = None
        page_index = getattr(getattr(self, "controller", None), "page_index", None)
        if page_index is not None:
            breakdown = page_index.physical_kv_state_breakdown()
        return {
            "running_request_count": len(requests),
            "running_request_ids": [
                str(getattr(req, "rid", f"object:{id(req)}"))
                for req in requests
            ],
            "replacement_request_count": len(replacements),
            "replacement_request_ids": [item.request_id for item in replacements],
            "native_reclaim_capacity_bytes": native_reclaim_bytes,
            "allocator_available_bytes": allocator_available_bytes,
            "active_kv_footprint_bytes": active_kv_footprint_bytes,
            "active_kv_budget_bytes": active_budget_bytes,
            "replacement_deficit_bytes": replacement_deficit_bytes,
            "active_excess_bytes": max(
                0, active_kv_footprint_bytes - active_budget_bytes
            ),
            "gpu_kv_bytes": getattr(breakdown, "gpu_bytes", None),
            "host_kv_bytes": getattr(breakdown, "cpu_bytes", None),
            "engine_locked_bytes": getattr(
                breakdown, "engine_locked_bytes", None
            ),
            "closure_blocked_bytes": getattr(
                breakdown, "closure_blocked_bytes", None
            ),
            "migratable_bytes": getattr(breakdown, "migratable_bytes", None),
            "dual_resident_bytes": getattr(
                breakdown, "dual_resident_bytes", None
            ),
            "page_revision": getattr(page_index, "revision", None),
            "topology_revision": getattr(page_index, "topology_revision", None),
        }

    def _preview_running_retraction_barrier_unlock(
        self,
        *,
        now_ms: float,
    ) -> tuple[TentativeUnlockPreview | None, str, float]:
        started_ns = time.perf_counter_ns()
        provenance = self._lock_provenance_extents()
        if provenance is None:
            return (
                None,
                "provenance_unavailable",
                (time.perf_counter_ns() - started_ns) / 1_000.0,
            )
        extents, path_error_count = provenance
        ledger = getattr(self, "_lock_service_ledger", None)
        if ledger is None:
            return (
                None,
                "service_ledger_unavailable",
                (time.perf_counter_ns() - started_ns) / 1_000.0,
            )
        blocker_ids = {
            request_id
            for extent in extents
            for request_id in extent.blocker_request_ids
        }
        stale_ids = tuple(
            sorted(
                request_id
                for request_id in blocker_ids
                if ledger.tracks(request_id)
                and ledger.service_status(
                    request_id,
                    now_ms=now_ms,
                    window_ms=(
                        self.config.running_batch_retraction_min_stall_ms
                    ),
                )
                == "stale"
            )
        )
        if not stale_ids:
            return (
                None,
                "no_observed_stale_blocker",
                (time.perf_counter_ns() - started_ns) / 1_000.0,
            )
        preview = TentativeUnlockPreviewer(
            self.controller.page_index
        ).preview(
            extents,
            stale_ids,
            path_error_count=path_error_count,
        )
        return (
            preview,
            "previewed",
            (time.perf_counter_ns() - started_ns) / 1_000.0,
        )

    @staticmethod
    def _running_retraction_barrier_state_delta(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        for key in (
            "running_request_count",
            "replacement_request_count",
            "native_reclaim_capacity_bytes",
            "allocator_available_bytes",
            "active_kv_footprint_bytes",
            "replacement_deficit_bytes",
            "active_excess_bytes",
            "gpu_kv_bytes",
            "host_kv_bytes",
            "engine_locked_bytes",
            "closure_blocked_bytes",
            "migratable_bytes",
            "dual_resident_bytes",
        ):
            before_value = before.get(key)
            after_value = after.get(key)
            if isinstance(before_value, int) and isinstance(after_value, int):
                result[key] = after_value - before_value
        return result

    def _attribute_running_retraction_barrier(
        self,
        running_batch: Any,
        *,
        now_ms: float,
        planning_reason: str,
        replacements: tuple[RetractionReplacement, ...] | None = None,
        decision: ObservedRetractionDecision | None = None,
    ) -> str | None:
        attempt = getattr(self, "_pending_running_retraction_barrier", None)
        if attempt is None:
            return None
        decision_state = self._running_retraction_barrier_state(
            running_batch,
            now_ms=now_ms,
            replacements=replacements,
        )
        drained_state = attempt.drained_state or decision_state
        if decision is not None and decision.plan is not None:
            outcome = "plan_created"
        elif (
            attempt.requested_state["replacement_request_ids"]
            and attempt.requested_state["replacement_request_ids"][0]
            not in decision_state["replacement_request_ids"]
        ):
            outcome = "replacement_disappeared"
        elif (
            decision_state["replacement_deficit_bytes"] == 0
            and decision_state["active_excess_bytes"] == 0
        ):
            outcome = "pressure_resolved_by_drain"
        elif planning_reason == "provenance_unavailable":
            outcome = "provenance_unavailable"
        elif planning_reason == "no_eligible_stale_candidate":
            outcome = "no_eligible_stale_candidate"
        elif planning_reason == "insufficient_unlock_capacity":
            outcome = "insufficient_unlock_capacity"
        else:
            outcome = planning_reason

        outcomes = getattr(self, "_running_retraction_barrier_outcomes", None)
        if outcomes is None:
            outcomes = Counter()
            self._running_retraction_barrier_outcomes = outcomes
        outcomes[outcome] += 1
        self._running_retraction_counts[f"barrier_outcome_{outcome}"] += 1
        self.audit.emit(
            "running_retraction_overlap_barrier_outcome",
            now_ms,
            barrier_intent_id=attempt.barrier_intent_id,
            outcome=outcome,
            planning_reason=planning_reason,
            barrier_was_drained=attempt.drained_ts_ms is not None,
            drain_latency_ms=(
                max(0.0, attempt.drained_ts_ms - attempt.requested_ts_ms)
                if attempt.drained_ts_ms is not None
                else None
            ),
            decision_latency_after_drain_ms=(
                max(0.0, now_ms - attempt.drained_ts_ms)
                if attempt.drained_ts_ms is not None
                else None
            ),
            requested_state=attempt.requested_state,
            drained_state=drained_state,
            decision_state=decision_state,
            requested_to_drained_delta=(
                self._running_retraction_barrier_state_delta(
                    attempt.requested_state, drained_state
                )
            ),
            decision=(
                {
                    "reason": decision.reason,
                    "target_reclaim_bytes": decision.target_reclaim_bytes,
                    "candidate_count": decision.candidate_count,
                    "eligible_candidate_count": (
                        decision.eligible_candidate_count
                    ),
                    "fully_attributed_extent_count": (
                        decision.fully_attributed_extent_count
                    ),
                    "reclaim_capacity_bytes": decision.reclaim_capacity_bytes,
                }
                if decision is not None
                else None
            ),
            predrain_tentative_unlock_preview_status=(
                attempt.tentative_unlock_preview_status
            ),
            predrain_tentative_unlock_preview=(
                attempt.tentative_unlock_preview.to_audit_fields()
                if attempt.tentative_unlock_preview is not None
                else None
            ),
        )
        self._pending_running_retraction_barrier = None
        return attempt.barrier_intent_id

    def _close_pending_running_retraction_barrier(
        self,
        *,
        now_ms: float,
    ) -> None:
        attempt = getattr(self, "_pending_running_retraction_barrier", None)
        if attempt is None:
            return
        outcome = "runtime_closed_before_decision"
        outcomes = getattr(self, "_running_retraction_barrier_outcomes", None)
        if outcomes is None:
            outcomes = Counter()
            self._running_retraction_barrier_outcomes = outcomes
        outcomes[outcome] += 1
        self._running_retraction_counts[f"barrier_outcome_{outcome}"] += 1
        drained_state = attempt.drained_state
        self.audit.emit(
            "running_retraction_overlap_barrier_outcome",
            now_ms,
            barrier_intent_id=attempt.barrier_intent_id,
            outcome=outcome,
            planning_reason=outcome,
            barrier_was_drained=attempt.drained_ts_ms is not None,
            drain_latency_ms=(
                max(0.0, attempt.drained_ts_ms - attempt.requested_ts_ms)
                if attempt.drained_ts_ms is not None
                else None
            ),
            decision_latency_after_drain_ms=None,
            requested_state=attempt.requested_state,
            drained_state=drained_state,
            decision_state=None,
            requested_to_drained_delta=(
                self._running_retraction_barrier_state_delta(
                    attempt.requested_state, drained_state
                )
                if drained_state is not None
                else None
            ),
            decision=None,
            predrain_tentative_unlock_preview_status=(
                attempt.tentative_unlock_preview_status
            ),
            predrain_tentative_unlock_preview=(
                attempt.tentative_unlock_preview.to_audit_fields()
                if attempt.tentative_unlock_preview is not None
                else None
            ),
        )
        self._pending_running_retraction_barrier = None

    def running_batch_retraction_barrier_required(
        self,
        running_batch: Any,
    ) -> bool:
        """Return whether overlap should drain before physical retraction planning."""

        if not self.config.running_batch_retraction_enabled:
            return False
        now_ms = float(self._now_ms())
        if getattr(
            self, "_restore_authority_mode", RestoreAuthorityMode.NORMAL_JOINT
        ) != RestoreAuthorityMode.NORMAL_JOINT:
            self._running_retraction_counts[
                "barrier_restore_authority_suppressed"
            ] += 1
            return False
        overdue_restore = self._overdue_restore_obligation(now_ms=now_ms)
        if overdue_restore is not None:
            self._running_retraction_counts[
                "barrier_restore_debt_suppressed"
            ] += 1
            return False
        previous_decision_ms = getattr(self, "_last_retraction_decision_ms", None)
        if (
            previous_decision_ms is not None
            and now_ms - previous_decision_ms
            < self.config.running_batch_retraction_decision_interval_ms
        ):
            return False
        if getattr(self, "_pending_running_retraction_transaction", None) is not None:
            return False
        if getattr(self, "_pending_online_joint_residency", None) is not None:
            return False
        controller = getattr(self, "controller", None)
        if (
            controller is not None
            and controller.has_pending_transfer_work()
        ):
            return False
        if getattr(self, "_pending_running_retraction_barrier", None) is not None:
            return False
        requests = tuple(getattr(running_batch, "reqs", ()) or ())
        minimum_active = max(
            1, self.config.observed_admission_min_active_requests
        )
        if len(requests) <= minimum_active:
            return False
        replacements = self._running_retraction_replacements(now_ms=now_ms)
        if not replacements:
            self._retraction_admission_stall_since_ms = None
            return False
        stall_since = getattr(self, "_retraction_admission_stall_since_ms", None)
        if stall_since is None:
            self._retraction_admission_stall_since_ms = now_ms
            self._running_retraction_counts["stall_warming"] += 1
            return False
        admission_stall_ms = max(0.0, now_ms - stall_since)
        if admission_stall_ms < self.config.running_batch_retraction_min_stall_ms:
            return False

        native_reclaim_bytes = self._native_reclaim_capacity_bytes()
        active = self._observed_admission_snapshot(
            native_available_hbm_bytes=native_reclaim_bytes,
            native_max_requests=len(replacements),
        )
        active_budget_bytes = int(
            max(
                0,
                self.config.hbm_capacity_bytes - self.config.reserve_hbm_bytes,
            )
            * self.config.observed_admission_active_kv_high_watermark_ratio
        )
        replacement_deficit_bytes = max(
            0,
            replacements[0].estimated_incremental_bytes - native_reclaim_bytes,
        )
        active_excess_bytes = max(
            0,
            active.active_kv_footprint_bytes - active_budget_bytes,
        )
        restore_micro_gate_barrier = (
            self._restore_micro_gate_barrier_pair_visible(requests, replacements)
        )
        if (
            replacement_deficit_bytes == 0
            and active_excess_bytes == 0
            and not restore_micro_gate_barrier
        ):
            self._running_retraction_counts["barrier_no_pressure"] += 1
            return False

        sequence = getattr(self, "_retraction_barrier_sequence", 0) + 1
        self._retraction_barrier_sequence = sequence
        barrier_intent_id = f"retraction-barrier-{sequence}"
        requested_state = self._running_retraction_barrier_state(
            running_batch,
            now_ms=now_ms,
            replacements=replacements,
            native_reclaim_bytes=native_reclaim_bytes,
            active_kv_footprint_bytes=active.active_kv_footprint_bytes,
        )
        (
            tentative_unlock_preview,
            tentative_unlock_preview_status,
            tentative_unlock_preview_compute_us,
        ) = self._preview_running_retraction_barrier_unlock(now_ms=now_ms)
        preview_counts = getattr(self, "_running_retraction_preview_counts", None)
        if preview_counts is None:
            preview_counts = Counter()
            self._running_retraction_preview_counts = preview_counts
        preview_counts[
            f"barrier_status:{tentative_unlock_preview_status}"
        ] += 1
        if tentative_unlock_preview is not None:
            preview_counts[
                f"barrier_reason:{tentative_unlock_preview.reason}"
            ] += 1
            preview_counts[
                "barrier_exact"
                if tentative_unlock_preview.exact
                else "barrier_inexact"
            ] += 1
        barrier_preview_samples = getattr(
            self, "_running_retraction_barrier_preview_compute_us", None
        )
        if barrier_preview_samples is None:
            barrier_preview_samples = deque(maxlen=65_536)
            self._running_retraction_barrier_preview_compute_us = (
                barrier_preview_samples
            )
        barrier_preview_samples.append(tentative_unlock_preview_compute_us)
        self._pending_running_retraction_barrier = (
            _RunningRetractionBarrierAttempt(
                barrier_intent_id=barrier_intent_id,
                requested_ts_ms=now_ms,
                requested_state=requested_state,
                tentative_unlock_preview=tentative_unlock_preview,
                tentative_unlock_preview_status=(
                    tentative_unlock_preview_status
                ),
                tentative_unlock_preview_compute_us=(
                    tentative_unlock_preview_compute_us
                ),
            )
        )
        self._running_retraction_counts["overlap_barrier_requested"] += 1
        self.audit.emit(
            "running_retraction_overlap_barrier_requested",
            now_ms,
            barrier_intent_id=barrier_intent_id,
            running_request_count=len(requests),
            replacement_request_count=len(replacements),
            admission_stall_ms=admission_stall_ms,
            replacement_deficit_bytes=replacement_deficit_bytes,
            active_excess_bytes=active_excess_bytes,
            restore_micro_gate_forced=restore_micro_gate_barrier,
            active_kv_footprint_bytes=active.active_kv_footprint_bytes,
            active_kv_budget_bytes=active_budget_bytes,
            native_reclaim_capacity_bytes=native_reclaim_bytes,
            requested_state=requested_state,
            tentative_unlock_preview_scope=(
                "all_observed_stale_blockers_unconstrained_upper_bound"
            ),
            tentative_unlock_preview_status=tentative_unlock_preview_status,
            tentative_unlock_preview_compute_us=(
                tentative_unlock_preview_compute_us
            ),
            tentative_unlock_preview=(
                tentative_unlock_preview.to_audit_fields()
                if tentative_unlock_preview is not None
                else None
            ),
            tentative_unlock_preview_policy_effect="shadow_only",
        )
        return True

    def on_running_batch_retraction_barrier_drained(
        self,
        running_batch: Any,
    ) -> None:
        now_ms = float(self._now_ms())
        attempt = getattr(self, "_pending_running_retraction_barrier", None)
        drained_state = self._running_retraction_barrier_state(
            running_batch,
            now_ms=now_ms,
        )
        if attempt is not None:
            attempt.drained_ts_ms = now_ms
            attempt.drained_state = drained_state
        else:
            self._running_retraction_counts["overlap_barrier_orphan_drain"] += 1
        self._running_retraction_counts["overlap_barrier_drained"] += 1
        self.audit.emit(
            "running_retraction_overlap_barrier_drained",
            now_ms,
            barrier_intent_id=(
                attempt.barrier_intent_id if attempt is not None else None
            ),
            running_request_count=len(
                tuple(getattr(running_batch, "reqs", ()) or ())
            ),
            drained_state=drained_state,
            requested_to_drained_delta=(
                self._running_retraction_barrier_state_delta(
                    attempt.requested_state, drained_state
                )
                if attempt is not None
                else None
            ),
        )

    def _restore_obligation_index(self) -> RestoreObligationIndex:
        index = getattr(self, "_restore_obligations", None)
        if index is None:
            index = RestoreObligationIndex(
                max_active=int(
                    getattr(
                        getattr(self, "config", None),
                        "restore_obligation_max_active",
                        8,
                    )
                )
            )
            self._restore_obligations = index
        if not hasattr(self, "_restore_obligation_counts"):
            self._restore_obligation_counts = Counter()
        if not hasattr(self, "_restore_command_to_request"):
            self._restore_command_to_request = {}
        if not hasattr(self, "_restore_command_sequence"):
            self._restore_command_sequence = 0
        return index

    def _restore_lease_index(self) -> RestoreLeaseIndex:
        index = getattr(self, "_restore_leases", None)
        if index is None:
            index = RestoreLeaseIndex(
                max_active=int(
                    getattr(
                        getattr(self, "config", None),
                        "restore_lease_max_active",
                        1,
                    )
                )
            )
            self._restore_leases = index
        if not hasattr(self, "_restore_lease_allocations"):
            self._restore_lease_allocations = {}
        if not hasattr(self, "_restore_lease_pins"):
            self._restore_lease_pins = {}
        if not hasattr(self, "_restore_funding_allocations"):
            self._restore_funding_allocations = {}
        if not hasattr(self, "_restore_funding_target_by_command"):
            self._restore_funding_target_by_command = {}
        return index

    def _allocator_available_tokens(self) -> int:
        allocator = self.scheduler.token_to_kv_pool_allocator
        if hasattr(allocator, "full_available_size"):
            return max(
                0,
                min(
                    int(allocator.full_available_size()),
                    int(allocator.swa_available_size()),
                ),
            )
        return max(0, int(allocator.available_size()))

    def _restore_lease_tokens(self, required_bytes: int) -> int:
        allocator = self.scheduler.token_to_kv_pool_allocator
        page_size = max(1, int(getattr(allocator, "page_size", 1) or 1))
        raw_tokens = max(
            1,
            (max(0, int(required_bytes)) + self.config.kv_bytes_per_token - 1)
            // self.config.kv_bytes_per_token,
        )
        return ((raw_tokens + page_size - 1) // page_size) * page_size

    def _restore_funding_reserved_tokens(self, request_id: str) -> int:
        return sum(
            len(item)
            for item in getattr(self, "_restore_funding_allocations", {}).get(
                request_id, ()
            )
        )

    def allocator_backed_reservation_tokens(self) -> int:
        """Return KV slots intentionally held outside native request/cache state."""

        funding_tokens = sum(
            len(item)
            for allocations in getattr(
                self, "_restore_funding_allocations", {}
            ).values()
            for item in allocations
        )
        lease_tokens = sum(
            len(item)
            for allocations in getattr(
                self, "_restore_lease_allocations", {}
            ).values()
            for item in allocations
        )
        return funding_tokens + lease_tokens

    def _set_restore_funding_reservation(
        self,
        obligation: RestoreObligation,
        allocations: list[Any],
    ) -> int:
        if not hasattr(self, "_restore_funding_allocations"):
            self._restore_funding_allocations = {}
        if allocations:
            self._restore_funding_allocations[obligation.request_id] = allocations
        else:
            self._restore_funding_allocations.pop(obligation.request_id, None)
        tokens = sum(len(item) for item in allocations)
        obligation.funding_reserved_tokens = tokens
        obligation.funding_reserved_bytes = (
            tokens * self.config.kv_bytes_per_token
        )
        return tokens

    def _reserve_restore_funding_capacity(
        self,
        obligation: RestoreObligation,
        *,
        target_bytes: int,
        reclaimed_bytes: int,
        now_ms: float,
    ) -> int:
        """Turn newly reclaimed capacity into allocator-backed debt ownership."""

        target_bytes = max(0, int(target_bytes))
        reclaimed_bytes = max(0, int(reclaimed_bytes))
        if target_bytes == 0 or reclaimed_bytes == 0:
            return 0
        allocator = self.scheduler.token_to_kv_pool_allocator
        page_size = max(1, int(getattr(allocator, "page_size", 1) or 1))
        target_tokens = (
            target_bytes + self.config.kv_bytes_per_token - 1
        ) // self.config.kv_bytes_per_token
        target_tokens = (
            (target_tokens + page_size - 1) // page_size
        ) * page_size
        reclaimed_tokens = reclaimed_bytes // self.config.kv_bytes_per_token
        reclaimed_tokens = (reclaimed_tokens // page_size) * page_size
        claim_tokens = min(target_tokens, reclaimed_tokens)
        if claim_tokens <= 0:
            return 0
        allocation = allocator.alloc(claim_tokens)
        if allocation is None:
            self._restore_obligation_counts["funding_reservation_failed"] += 1
            self.audit.emit(
                "restore_funding_reservation_failed",
                now_ms,
                obligation_id=obligation.obligation_id,
                request_id=obligation.request_id,
                target_bytes=target_bytes,
                reclaimed_bytes=reclaimed_bytes,
                claim_tokens=claim_tokens,
                allocator_available_tokens=self._allocator_available_tokens(),
            )
            return 0
        allocations = list(
            getattr(self, "_restore_funding_allocations", {}).get(
                obligation.request_id, ()
            )
        )
        allocations.append(allocation)
        total_tokens = self._set_restore_funding_reservation(
            obligation, allocations
        )
        self._restore_obligation_counts["funding_capacity_reserved"] += 1
        self.audit.emit(
            "restore_funding_capacity_reserved",
            now_ms,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            target_bytes=target_bytes,
            reclaimed_bytes=reclaimed_bytes,
            newly_reserved_tokens=claim_tokens,
            total_reserved_tokens=total_tokens,
            total_reserved_bytes=obligation.funding_reserved_bytes,
            allocator_available_tokens_after=self._allocator_available_tokens(),
        )
        return claim_tokens

    def _release_restore_funding_capacity(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
        reason: str,
    ) -> int:
        allocations = getattr(self, "_restore_funding_allocations", {}).pop(
            obligation.request_id, []
        )
        if not allocations:
            self._set_restore_funding_reservation(obligation, [])
            return 0
        allocator = self.scheduler.token_to_kv_pool_allocator
        released_tokens = 0
        for allocation in allocations:
            released_tokens += len(allocation)
            allocator.free(allocation)
        self._set_restore_funding_reservation(obligation, [])
        if released_tokens:
            self.audit.emit(
                "restore_funding_capacity_released",
                now_ms,
                obligation_id=obligation.obligation_id,
                request_id=obligation.request_id,
                released_tokens=released_tokens,
                released_bytes=released_tokens * self.config.kv_bytes_per_token,
                reason=reason,
                allocator_available_tokens_after=self._allocator_available_tokens(),
            )
        return released_tokens

    def _reacquire_restore_funding_capacity(
        self,
        obligation: RestoreObligation,
        tokens: int,
        *,
        now_ms: float,
        reason: str,
    ) -> bool:
        if tokens <= 0:
            return True
        allocation = self.scheduler.token_to_kv_pool_allocator.alloc(tokens)
        if allocation is None:
            self._restore_obligation_counts[
                "funding_reservation_reacquire_failed"
            ] += 1
            self.audit.emit(
                "restore_funding_reservation_reacquire_failed",
                now_ms,
                obligation_id=obligation.obligation_id,
                request_id=obligation.request_id,
                tokens=tokens,
                reason=reason,
            )
            return False
        self._set_restore_funding_reservation(obligation, [allocation])
        return True

    def _grant_restore_lease(
        self,
        obligation: RestoreObligation,
        *,
        h2d_bytes: int,
        now_ms: float,
    ) -> RestoreLease | None:
        if not bool(getattr(self.config, "restore_lease_enabled", True)):
            return None
        index = self._restore_lease_index()
        lease = index.get(obligation.request_id)
        if lease is not None and lease.state == RestoreLeaseState.ADMISSION_COMMITTING:
            return None
        if (
            lease is None or lease.state.terminal
        ) and len(index.active()) >= index.max_active:
            return None
        required_tokens = self._restore_lease_tokens(
            obligation.required_admission_bytes
        )
        h2d_tokens = (
            max(0, int(h2d_bytes)) + self.config.kv_bytes_per_token - 1
        ) // self.config.kv_bytes_per_token
        current_tokens = sum(
            len(item)
            for item in getattr(self, "_restore_lease_allocations", {}).get(
                obligation.request_id, ()
            )
        )
        additional_tokens = max(0, required_tokens - current_tokens)
        funding_tokens = self._restore_funding_reserved_tokens(
            obligation.request_id
        )
        if (
            self._allocator_available_tokens() + funding_tokens
            < additional_tokens + h2d_tokens
        ):
            return None
        allocator = self.scheduler.token_to_kv_pool_allocator
        consumed_funding_tokens = self._release_restore_funding_capacity(
            obligation,
            now_ms=now_ms,
            reason="convert_to_restore_lease",
        )
        allocation = allocator.alloc(additional_tokens) if additional_tokens else None
        if additional_tokens and allocation is None:
            self._reacquire_restore_funding_capacity(
                obligation,
                consumed_funding_tokens,
                now_ms=now_ms,
                reason="restore_lease_allocation_failed",
            )
            return None
        if self._allocator_available_tokens() < h2d_tokens:
            if allocation is not None:
                allocator.free(allocation)
            self._reacquire_restore_funding_capacity(
                obligation,
                consumed_funding_tokens,
                now_ms=now_ms,
                reason="restore_h2d_capacity_race",
            )
            return None
        if lease is None or lease.state.terminal:
            try:
                lease = index.grant(
                    obligation=obligation,
                    granted_ts_ms=now_ms,
                    reserved_tokens=required_tokens,
                    reserved_bytes=required_tokens
                    * self.config.kv_bytes_per_token,
                    h2d_bytes=max(0, int(h2d_bytes)),
                )
            except ValueError:
                if allocation is not None:
                    allocator.free(allocation)
                self._reacquire_restore_funding_capacity(
                    obligation,
                    consumed_funding_tokens,
                    now_ms=now_ms,
                    reason="restore_lease_index_rejected",
                )
                return None
            self._restore_lease_allocations[obligation.request_id] = []
            self._restore_obligation_counts["lease_granted"] += 1
            self.audit.emit(
                "restore_lease_granted",
                now_ms,
                lease_id=lease.lease_id,
                obligation_id=obligation.obligation_id,
                request_id=obligation.request_id,
                workflow_id=obligation.workflow_id,
                context_id=obligation.context_id,
                reserved_tokens=lease.reserved_tokens,
                reserved_bytes=lease.reserved_bytes,
                h2d_bytes=lease.h2d_bytes,
                allocator_available_tokens_after=self._allocator_available_tokens(),
            )
        else:
            lease.reserved_tokens = max(lease.reserved_tokens, required_tokens)
            lease.reserved_bytes = (
                lease.reserved_tokens * self.config.kv_bytes_per_token
            )
            lease.h2d_bytes = max(lease.h2d_bytes, max(0, int(h2d_bytes)))
        if allocation is not None:
            self._restore_lease_allocations.setdefault(
                obligation.request_id, []
            ).append(allocation)
        return lease

    def _pin_restore_lease_prefix(
        self,
        obligation: RestoreObligation,
        req: Any,
        *,
        now_ms: float,
        allow_unmaterialized: bool = False,
    ) -> bool:
        lease = self._restore_lease_index().get(obligation.request_id)
        if lease is None or lease.state.terminal:
            return False
        if obligation.request_id in self._restore_lease_pins:
            return True
        node = getattr(req, "last_node", None)
        root = getattr(self.tree_cache, "root_node", None)
        if node is None or (node is root and obligation.restored_bytes > 0):
            return False
        if node is root:
            return True
        if getattr(node, "value", None) is None:
            if not allow_unmaterialized or getattr(node, "host_value", None) is None:
                return False
            self._restore_obligation_counts[
                "lease_prefix_pin_deferred_h2d"
            ] += 1
            self.audit.emit(
                "restore_lease_prefix_pin_deferred",
                now_ms,
                lease_id=lease.lease_id,
                obligation_id=obligation.obligation_id,
                request_id=obligation.request_id,
                context_id=obligation.context_id,
                radix_node_id=getattr(node, "id", None),
                reason="host_only_prefix_waiting_for_h2d",
            )
            return True
        current = node
        seen: set[int] = set()
        while current is not root:
            if current is None or id(current) in seen:
                return False
            seen.add(id(current))
            if getattr(current, "value", None) is None:
                return False
            current = getattr(current, "parent", None)
        inc_lock_ref = getattr(self.tree_cache, "inc_lock_ref", None)
        dec_lock_ref = getattr(self.tree_cache, "dec_lock_ref", None)
        if not callable(inc_lock_ref) or not callable(dec_lock_ref):
            return False
        token = inc_lock_ref(node)
        release_with_token = len(inspect.signature(dec_lock_ref).parameters) >= 2
        self._restore_lease_pins[obligation.request_id] = (
            node,
            token,
            release_with_token,
        )
        lease.pin_active = True
        self._restore_obligation_counts["lease_prefix_pinned"] += 1
        self.audit.emit(
            "restore_lease_prefix_pinned",
            now_ms,
            lease_id=lease.lease_id,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            context_id=obligation.context_id,
            radix_node_id=getattr(node, "id", None),
        )
        return True

    def _unpin_restore_lease_prefix(
        self,
        request_id: str,
        *,
        now_ms: float,
        reason: str,
    ) -> None:
        pin = getattr(self, "_restore_lease_pins", {}).pop(request_id, None)
        lease = getattr(self, "_restore_leases", None)
        lease = lease.get(request_id) if lease is not None else None
        if pin is None:
            if lease is not None:
                lease.pin_active = False
            return
        node, token, release_with_token = pin
        try:
            if int(getattr(node, "lock_ref", 1) or 0) <= 0:
                pass
            elif release_with_token:
                self.tree_cache.dec_lock_ref(node, token)
            else:
                self.tree_cache.dec_lock_ref(node)
        except Exception as error:
            self.audit.emit(
                "restore_lease_prefix_unpin_failed",
                now_ms,
                request_id=request_id,
                reason=reason,
                error=f"{type(error).__name__}: {error}",
            )
        else:
            self.audit.emit(
                "restore_lease_prefix_unpinned",
                now_ms,
                request_id=request_id,
                reason=reason,
                radix_node_id=getattr(node, "id", None),
            )
        if lease is not None:
            lease.pin_active = False

    def _free_restore_lease_allocations(self, request_id: str) -> int:
        allocations = getattr(self, "_restore_lease_allocations", {}).pop(
            request_id, []
        )
        allocator = getattr(
            getattr(self, "scheduler", None),
            "token_to_kv_pool_allocator",
            None,
        )
        released_tokens = 0
        for allocation in allocations:
            released_tokens += len(allocation)
            if allocator is not None:
                allocator.free(allocation)
        return released_tokens

    def _release_restore_lease(
        self,
        request_id: str,
        *,
        now_ms: float,
        reason: str,
        rollback: bool,
    ) -> None:
        index = getattr(self, "_restore_leases", None)
        lease = index.get(request_id) if index is not None else None
        if lease is None or lease.state.terminal:
            return
        released_tokens = self._free_restore_lease_allocations(request_id)
        self._unpin_restore_lease_prefix(
            request_id, now_ms=now_ms, reason=reason
        )
        lease.finish(
            RestoreLeaseState.ROLLED_BACK if rollback else RestoreLeaseState.RELEASED,
            now_ms=now_ms,
            reason=reason,
        )
        self._restore_obligation_counts[
            "lease_rolled_back" if rollback else "lease_released"
        ] += 1
        self.audit.emit(
            "restore_lease_terminal",
            now_ms,
            lease_id=lease.lease_id,
            obligation_id=lease.obligation_id,
            request_id=request_id,
            state=lease.state.value,
            reason=reason,
            released_tokens=released_tokens,
            reserved_tokens=lease.reserved_tokens,
            reserved_bytes=lease.reserved_bytes,
            admission_attempts=lease.admission_attempts,
        )

    def _restore_lease_credit_bytes(self) -> dict[str, int]:
        index = getattr(self, "_restore_leases", None)
        if index is None:
            return {}
        allocations = getattr(self, "_restore_lease_allocations", {})
        return {
            lease.request_id: lease.reserved_bytes
            for lease in index.active()
            if lease.capacity_held and allocations.get(lease.request_id)
        }

    def _begin_restore_lease_admission(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
    ) -> bool:
        lease = self._restore_lease_index().get(obligation.request_id)
        if lease is None or lease.state.terminal:
            return not bool(getattr(self.config, "restore_lease_enabled", True))
        if lease.state == RestoreLeaseState.ADMISSION_COMMITTING:
            return True
        resized = self._grant_restore_lease(
            obligation,
            h2d_bytes=0,
            now_ms=now_ms,
        )
        if resized is None:
            return False
        allocations = self._restore_lease_allocations.get(
            obligation.request_id, []
        )
        if sum(len(item) for item in allocations) < lease.reserved_tokens:
            return False
        released_tokens = self._free_restore_lease_allocations(
            obligation.request_id
        )
        lease.begin_admission()
        transaction = self._ensure_restore_transaction(obligation)
        transaction.stage = RestoreTransactionStage.ADMISSION_COMMITTING
        transaction.admission_state = "committing"
        self._restore_obligation_counts["lease_admission_started"] += 1
        self.audit.emit(
            "restore_lease_admission_started",
            now_ms,
            lease_id=lease.lease_id,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            released_tokens=released_tokens,
            reserved_tokens=lease.reserved_tokens,
            reserved_bytes=lease.reserved_bytes,
            admission_attempt=lease.admission_attempts,
        )
        return True

    def _reject_restore_lease_admission(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
        native_result: str,
    ) -> None:
        lease = self._restore_lease_index().get(obligation.request_id)
        if lease is None or lease.state != RestoreLeaseState.ADMISSION_COMMITTING:
            return
        allocator = self.scheduler.token_to_kv_pool_allocator
        allocation = allocator.alloc(lease.reserved_tokens)
        if allocation is None:
            self._release_restore_lease(
                obligation.request_id,
                now_ms=now_ms,
                reason="restore_admission_reservation_reacquire_failed",
                rollback=True,
            )
            obligation.block(
                blocker_codes=("restore_reservation_reacquire_failed",),
                blocker_fingerprint="restore_reservation_reacquire_failed",
                attempt_stamp=self._restore_attempt_stamp(),
                now_ms=now_ms,
            )
            self._restore_obligation_counts[
                "lease_admission_reacquire_failed"
            ] += 1
            return
        self._restore_lease_allocations[obligation.request_id] = [allocation]
        lease.admission_rejected()
        transaction = self._ensure_restore_transaction(obligation)
        transaction.stage = RestoreTransactionStage.RESTORED_RESERVED
        transaction.admission_state = "rejected_reserved"
        self._restore_obligation_counts["lease_admission_rejected"] += 1
        self.audit.emit(
            "restore_lease_admission_rejected",
            now_ms,
            lease_id=lease.lease_id,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            native_result=native_result,
            reservation_reacquired=True,
            reserved_tokens=lease.reserved_tokens,
        )

    def _commit_restore_lease_admission(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
    ) -> None:
        lease = self._restore_lease_index().get(obligation.request_id)
        if lease is None or lease.state != RestoreLeaseState.ADMISSION_COMMITTING:
            return
        lease.mark_admitted()
        transaction = self._ensure_restore_transaction(obligation)
        transaction.stage = RestoreTransactionStage.ADMITTED
        transaction.admission_state = "admitted"
        self._unpin_restore_lease_prefix(
            obligation.request_id,
            now_ms=now_ms,
            reason="native_admission_committed",
        )
        self._restore_obligation_counts["lease_admission_committed"] += 1
        self.audit.emit(
            "restore_lease_admission_committed",
            now_ms,
            lease_id=lease.lease_id,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            reserved_tokens=lease.reserved_tokens,
            reserved_bytes=lease.reserved_bytes,
        )

    def _request_path_extent_ids(
        self,
        req: Any | None,
        context_id: str,
        *,
        cpu_only: bool,
    ) -> tuple[str, ...] | None:
        """Capture the authoritative request path, not all context-owned pages."""

        page_index = self.controller.page_index
        node = getattr(req, "last_node", None) if req is not None else None
        registry = getattr(self, "registry", None)
        root = getattr(getattr(self, "tree_cache", None), "root_node", None)
        if node is None:
            return () if not page_index.has_context(context_id) else None
        if registry is None or root is None:
            if not page_index.has_context(context_id):
                return ()
            return tuple(
                sorted(
                    f"page:{page.handle.page_id}:"
                    f"{page.handle.allocation_generation}"
                    for page in page_index.context_pages(context_id)
                    if not cpu_only
                    or (page.cpu_resident and not page.gpu_resident)
                )
            )
        result: list[str] = []
        seen: set[int] = set()
        while node is not None and node is not root:
            identity = id(node)
            if identity in seen:
                return None
            seen.add(identity)
            handle = registry.current_handle(int(node.id))
            if handle is None:
                return None
            page = page_index.pages.get(handle)
            if page is None or page.residency == PhysicalResidency.DEAD:
                return None
            if not cpu_only or (page.cpu_resident and not page.gpu_resident):
                result.append(
                    f"page:{handle.page_id}:{handle.allocation_generation}"
                )
            node = getattr(node, "parent", None)
        return tuple(sorted(result))

    def _can_prepare_restore_obligations(
        self,
        plan: RunningRetractionPlan,
        requests: tuple[Any, ...],
    ) -> bool:
        request_by_id = {
            str(getattr(req, "rid", "")): req
            for req in requests
            if str(getattr(req, "rid", ""))
        }
        index = self._restore_obligation_index()
        if not index.can_create(tuple(plan.request_ids)):
            return False
        for request_id in plan.request_ids:
            req = request_by_id.get(request_id)
            metadata = self._metadata(req) if req is not None else None
            if metadata is None:
                return False
            context = self.controller.graph.contexts.get(metadata.context_id)
            if context is None or context.epoch != metadata.context_epoch:
                return False
            if self._request_path_extent_ids(
                req, metadata.context_id, cpu_only=False
            ) is None:
                return False
        return True

    def _capture_restore_paths(
        self,
        plan: RunningRetractionPlan,
        requests: tuple[Any, ...],
    ) -> dict[str, tuple[str, ...]]:
        request_by_id = {
            str(getattr(req, "rid", "")): req
            for req in requests
            if str(getattr(req, "rid", ""))
        }
        result: dict[str, tuple[str, ...]] = {}
        for request_id in plan.request_ids:
            req = request_by_id[request_id]
            metadata = self._metadata(req)
            assert metadata is not None
            path = self._request_path_extent_ids(
                req, metadata.context_id, cpu_only=False
            )
            assert path is not None
            result[request_id] = path
        return result

    def _create_restore_obligations(
        self,
        transaction: _RunningRetractionTransaction,
        requests: list[Any] | tuple[Any, ...],
        *,
        now_ms: float,
    ) -> None:
        index = self._restore_obligation_index()
        source_plan_id = (
            getattr(transaction, "source_joint_plan_id", None)
            or f"{transaction.transaction_id}:emergency"
        )
        for req in requests:
            request_id = str(getattr(req, "rid", ""))
            metadata = self._metadata(req)
            if not request_id or metadata is None:
                continue
            path_extent_ids = getattr(
                transaction, "restore_path_extent_ids", {}
            ).get(request_id)
            if path_extent_ids is None:
                path_extent_ids = self._request_path_extent_ids(
                    req, metadata.context_id, cpu_only=False
                )
            if path_extent_ids is None:
                path_extent_ids = ()
                self.audit.emit(
                    "restore_obligation_path_recapture_failed",
                    now_ms,
                    request_id=request_id,
                    source_retraction_transaction_id=transaction.transaction_id,
                    fallback="native_recompute_on_requeue",
                )
            obligation = index.create(
                request_id=request_id,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
                context_epoch=metadata.context_epoch,
                source_retraction_transaction_id=transaction.transaction_id,
                source_joint_plan_id=source_plan_id,
                created_ts_ms=now_ms,
                path_extent_ids=path_extent_ids,
                cause=RestoreObligationCause.RUNNING_RETRACTION,
            )
            self._ensure_restore_transaction(obligation)
            self._restore_obligation_counts["created"] += 1
            if getattr(transaction, "restore_micro_gate_id", None) is not None:
                self._update_restore_micro_gate(
                    "obligation_created",
                    now_ms=now_ms,
                    reason="durable_restore_obligation_created",
                    transaction_id=transaction.transaction_id,
                    obligation_id=obligation.obligation_id,
                    victim_request_id=request_id,
                )
            self.audit.emit(
                "restore_obligation_created",
                now_ms,
                obligation_id=obligation.obligation_id,
                request_id=request_id,
                workflow_id=obligation.workflow_id,
                invocation_id=obligation.invocation_id,
                context_id=obligation.context_id,
                context_epoch=obligation.context_epoch,
                source_retraction_transaction_id=(
                    obligation.source_retraction_transaction_id
                ),
                source_joint_plan_id=obligation.source_joint_plan_id,
                cause=obligation.cause.value,
                path_extent_count=len(path_extent_ids),
            )

    def _ensure_ordinary_waiting_restore_obligation(
        self,
        request_id: str,
        metadata: BeliefKVRequestMetadata,
        req: Any | None,
        required_extent_ids: tuple[str, ...],
        *,
        now_ms: float,
    ) -> RestoreObligation | None:
        """Turn an observed CPU-only waiting prefix into a durable H2D debt."""

        index = self._restore_obligation_index()
        existing = index.get(request_id)
        if existing is not None and not existing.state.terminal:
            return existing
        if not required_extent_ids:
            return None
        if not index.can_create((request_id,)):
            waiters = getattr(
                self, "_ordinary_restore_capacity_waiters", None
            )
            if waiters is None:
                waiters = set()
                self._ordinary_restore_capacity_waiters = waiters
            if request_id not in waiters:
                waiters.add(request_id)
                self._restore_obligation_counts[
                    "ordinary_waiting_capacity_blocked"
                ] += 1
                self.audit.emit(
                    "ordinary_waiting_restore_capacity_blocked",
                    now_ms,
                    request_id=request_id,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    active_obligation_count=len(index.active()),
                    max_active=index.max_active,
                )
            return None

        path_extent_ids = self._request_path_extent_ids(
            req, metadata.context_id, cpu_only=False
        )
        if path_extent_ids is None:
            path_extent_ids = required_extent_ids
        source_plan_id = (
            f"joint-restore-liveness:{request_id}:{metadata.context_epoch}"
        )
        obligation = index.create(
            request_id=request_id,
            workflow_id=metadata.root_workflow_id,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            context_epoch=metadata.context_epoch,
            source_retraction_transaction_id=(
                f"ordinary-waiting:{request_id}"
            ),
            source_joint_plan_id=source_plan_id,
            created_ts_ms=now_ms,
            path_extent_ids=path_extent_ids,
            cause=RestoreObligationCause.ORDINARY_WAITING_PREFIX,
        )
        self._ensure_restore_transaction(obligation)
        obligation.source_transaction_terminal = True
        obligation.requeued = True
        obligation.state = RestoreObligationState.PARKED_WAIT
        obligation.set_required_extents(
            required_extent_ids,
            restore_bytes=self._extent_ids_bytes(required_extent_ids),
            now_ms=now_ms,
        )
        getattr(
            self, "_ordinary_restore_capacity_waiters", set()
        ).discard(request_id)
        self._restore_obligation_counts["ordinary_waiting_created"] += 1
        self.audit.emit(
            "restore_obligation_created",
            now_ms,
            obligation_id=obligation.obligation_id,
            request_id=request_id,
            workflow_id=obligation.workflow_id,
            invocation_id=obligation.invocation_id,
            context_id=obligation.context_id,
            context_epoch=obligation.context_epoch,
            source_retraction_transaction_id=(
                obligation.source_retraction_transaction_id
            ),
            source_joint_plan_id=obligation.source_joint_plan_id,
            cause=obligation.cause.value,
            native_admission_fallback=obligation.native_admission_fallback,
            path_extent_count=len(path_extent_ids),
            required_extent_count=len(required_extent_ids),
            restore_bytes=obligation.restore_bytes,
        )
        return obligation

    def _ensure_restore_transaction(
        self, obligation: RestoreObligation
    ) -> RestoreTransaction:
        transactions = getattr(self, "_restore_transactions", None)
        if transactions is None:
            transactions = {}
            self._restore_transactions = transactions
        transaction = transactions.get(obligation.request_id)
        if (
            transaction is not None
            and transaction.obligation.obligation_id == obligation.obligation_id
        ):
            return transaction
        transaction = RestoreTransaction(
            transaction_id=f"restore-tx:{obligation.obligation_id}",
            obligation=obligation,
        )
        transactions[obligation.request_id] = transaction
        return transaction

    def _mark_restore_source_transaction_terminal(
        self,
        transaction_id: str,
        *,
        now_ms: float,
        reason: str,
    ) -> None:
        for obligation in self._restore_obligation_index().for_source_transaction(
            transaction_id
        ):
            obligation.source_transaction_terminal = True
            if obligation.state in {
                RestoreObligationState.RETRACTION_PREPARED,
                RestoreObligationState.D2H_INFLIGHT,
            }:
                obligation.state = RestoreObligationState.PARKED_WAIT
            obligation.last_progress_ts_ms = now_ms
            self.audit.emit(
                "restore_obligation_source_terminal",
                now_ms,
                obligation_id=obligation.obligation_id,
                request_id=obligation.request_id,
                source_retraction_transaction_id=transaction_id,
                source_terminal_reason=reason,
                state=obligation.state.value,
            )

    def _overdue_restore_obligation(
        self, *, now_ms: float
    ) -> RestoreObligation | None:
        if getattr(self, "_restore_obligations", None) is None:
            return None
        return next(
            (
                item
                for item in self._restore_obligation_index().active()
                if now_ms - item.created_ts_ms
                >= self.config.restore_obligation_escalation_ms
            ),
            None,
        )

    def _request_restore_drain(
        self, obligation: RestoreObligation, *, now_ms: float
    ) -> None:
        mode = getattr(
            self, "_restore_authority_mode", RestoreAuthorityMode.NORMAL_JOINT
        )
        if mode != RestoreAuthorityMode.NORMAL_JOINT:
            return
        self._restore_authority_mode = (
            RestoreAuthorityMode.RESTORE_DRAIN_REQUESTED
        )
        self._restore_authority_request_id = obligation.request_id
        self.audit.emit(
            "restore_authority_transition",
            now_ms,
            previous_mode=mode.value,
            current_mode=self._restore_authority_mode.value,
            request_id=obligation.request_id,
            obligation_id=obligation.obligation_id,
            reason="oldest_restore_liveness_escalated",
        )

    def _advance_restore_authority(self, *, now_ms: float) -> None:
        mode = getattr(
            self, "_restore_authority_mode", RestoreAuthorityMode.NORMAL_JOINT
        )
        request_id = getattr(self, "_restore_authority_request_id", None)
        obligation = (
            self._restore_obligation_index().get(request_id)
            if request_id is not None
            else None
        )
        grace = getattr(self, "_restore_service_grace_by_request", {}).get(
            request_id
        )
        owner_finished = (
            obligation is None
            or obligation.state in {
                RestoreObligationState.CANCELLED,
                RestoreObligationState.FAILED,
            }
            or (
                obligation.state == RestoreObligationState.SATISFIED
                and (grace is None or not grace.active)
            )
        )
        if mode != RestoreAuthorityMode.NORMAL_JOINT and owner_finished:
            self._restore_authority_mode = RestoreAuthorityMode.NORMAL_JOINT
            self._restore_authority_request_id = None
            self._current_online_joint_view = None
            self._current_online_joint_decision = None
            self._current_joint_plan_epoch = None
            self.audit.emit(
                "restore_authority_transition",
                now_ms,
                previous_mode=mode.value,
                current_mode=RestoreAuthorityMode.NORMAL_JOINT.value,
                request_id=request_id,
                reason="restore_owner_terminal",
            )
            return
        if mode != RestoreAuthorityMode.RESTORE_DRAIN_REQUESTED:
            return
        if self.controller.has_pending_transfer_work():
            return
        current_view = getattr(self, "_current_online_joint_view", None)
        if current_view is not None:
            self._invalidate_online_joint_plan(
                current_view.plan_id,
                reason="restore_drain_active",
                now_ms=now_ms,
            )
        self._online_joint_result = None
        self._online_joint_source = None
        self._online_joint_validation = None
        self._current_online_joint_view = None
        self._current_online_joint_decision = None
        self._current_joint_plan_epoch = None
        self._current_semantic_residency_commit = None
        self._restore_authority_mode = RestoreAuthorityMode.RESTORE_DRAIN_ACTIVE
        self.audit.emit(
            "restore_authority_transition",
            now_ms,
            previous_mode=mode.value,
            current_mode=self._restore_authority_mode.value,
            request_id=request_id,
            reason="safe_point_exclusive_authority_acquired",
        )

    def _active_restore_lease(self) -> RestoreLease | None:
        index = getattr(self, "_restore_leases", None)
        if index is None:
            return None
        return next(iter(index.active()), None)

    def _select_restore_bypass_request(
        self,
        waiting_queue: list[Any] | tuple[Any, ...],
        *,
        now_ms: float,
    ) -> str | None:
        """Permit one bounded small admission only to avoid an idle engine."""

        if getattr(
            self, "_restore_authority_mode", RestoreAuthorityMode.NORMAL_JOINT
        ) != RestoreAuthorityMode.NORMAL_JOINT:
            return None

        overdue = self._overdue_restore_obligation(now_ms=now_ms)
        if overdue is None or self._active_restore_lease() is not None:
            return None
        if overdue.bypass_count >= self.config.restore_lease_max_bypass_admissions:
            return None
        running = tuple(
            getattr(getattr(self.scheduler, "running_batch", None), "reqs", ()) or ()
        )
        if running or getattr(self.scheduler, "chunked_req", None) is not None:
            return None
        candidates: list[tuple[int, float, str]] = []
        for req in waiting_queue:
            request_id = str(getattr(req, "rid", ""))
            if not request_id or request_id == overdue.request_id:
                continue
            metadata = self._metadata(req)
            entry = self.controller.visible_admission.get(request_id)
            if (
                metadata is None
                or entry is None
                or self._metadata_scope_is_terminal(metadata)
                or self._request_restore_bundle_ids(req, metadata.context_id)
            ):
                continue
            candidates.append(
                (
                    entry.request.estimated_incremental_bytes,
                    entry.request.submitted_ts_ms,
                    request_id,
                )
            )
        if not candidates:
            return None
        request_id = min(candidates)[2]
        self.audit.emit(
            "restore_debt_bounded_bypass_selected",
            now_ms,
            obligation_id=overdue.obligation_id,
            restore_request_id=overdue.request_id,
            bypass_request_id=request_id,
            prior_bypass_count=overdue.bypass_count,
            max_bypass_admissions=(
                self.config.restore_lease_max_bypass_admissions
            ),
            reason="engine_idle_without_grantable_restore_lease",
        )
        return request_id

    def _start_restore_service_grace(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
        req: Any | None = None,
    ) -> None:
        required_tokens = int(
            getattr(self.config, "restore_service_grace_decode_tokens", 0)
        )
        if required_tokens <= 0:
            return
        graces = getattr(self, "_restore_service_grace_by_request", None)
        if graces is None:
            graces = {}
            self._restore_service_grace_by_request = graces
        previous = graces.get(obligation.request_id)
        if previous is not None and previous.active:
            previous.cancel(now_ms=now_ms, reason="superseded_by_new_restore")
        grace = RestoreServiceGrace(
            request_id=obligation.request_id,
            obligation_id=obligation.obligation_id,
            granted_ts_ms=now_ms,
            required_decode_tokens=required_tokens,
            last_observed_output_tokens=(
                _sequence_length(getattr(req, "output_ids", None))
                if req is not None
                else None
            ),
        )
        graces[obligation.request_id] = grace
        transaction = self._ensure_restore_transaction(obligation)
        transaction.stage = RestoreTransactionStage.SERVICE_GRACE
        transaction.service_grace = grace
        self._restore_obligation_counts["service_grace_started"] += 1
        self.audit.emit(
            "restore_service_grace_started",
            now_ms,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            required_decode_tokens=required_tokens,
            policy_effect="exclude_from_running_retraction",
        )

    def _cancel_restore_service_grace(
        self,
        request_id: str,
        *,
        now_ms: float,
        reason: str,
    ) -> None:
        grace = getattr(self, "_restore_service_grace_by_request", {}).pop(
            request_id, None
        )
        if grace is None:
            return
        was_active = grace.active
        grace.cancel(now_ms=now_ms, reason=reason)
        if was_active:
            self._restore_obligation_counts["service_grace_cancelled"] += 1
            self.audit.emit(
                "restore_service_grace_terminal",
                now_ms,
                obligation_id=grace.obligation_id,
                request_id=request_id,
                status="cancelled",
                reason=reason,
                served_decode_tokens=grace.served_decode_tokens,
                required_decode_tokens=grace.required_decode_tokens,
            )

    def _observe_restore_service_grace(
        self,
        batch: Any,
        *,
        now_ms: float,
        phase: str,
    ) -> None:
        graces = getattr(self, "_restore_service_grace_by_request", {})
        if not graces:
            return
        steps = max(
            1,
            int(
                getattr(
                    getattr(self.scheduler, "server_args", None),
                    "num_continuous_decode_steps",
                    1,
                )
                or 1
            ),
        )
        for req in tuple(getattr(batch, "reqs", ()) or ()):
            request_id = str(getattr(req, "rid", ""))
            grace = graces.get(request_id)
            if grace is None or not grace.active:
                continue
            output_ids = getattr(req, "output_ids", None)
            output_tokens = (
                _sequence_length(output_ids) if output_ids is not None else None
            )
            if phase != "decode":
                if output_tokens is not None:
                    grace.last_observed_output_tokens = output_tokens
                continue
            if output_tokens is None or grace.last_observed_output_tokens is None:
                served_tokens = steps
            else:
                served_tokens = max(
                    0, output_tokens - grace.last_observed_output_tokens
                )
            if output_tokens is not None:
                grace.last_observed_output_tokens = output_tokens
            if not grace.observe_decode(served_tokens, now_ms=now_ms):
                continue
            graces.pop(request_id, None)
            transaction = getattr(self, "_restore_transactions", {}).get(
                request_id
            )
            if transaction is not None:
                transaction.stage = RestoreTransactionStage.SATISFIED
            self._restore_obligation_counts["service_grace_satisfied"] += 1
            gate_state = getattr(self, "_restore_micro_gate_state", {})
            if (
                self.config.restore_micro_gate_enabled
                and gate_state.get("victim_request_id") == request_id
            ):
                obligation = self._restore_obligation_index().get(request_id)
                self._update_restore_micro_gate(
                    "completed",
                    now_ms=now_ms,
                    reason="restore_service_quantum_satisfied",
                    obligation_id=grace.obligation_id,
                    restored_h2d_bytes=(
                        obligation.restored_bytes if obligation is not None else 0
                    ),
                    served_decode_tokens=grace.served_decode_tokens,
                )
            self.audit.emit(
                "restore_service_grace_terminal",
                now_ms,
                obligation_id=grace.obligation_id,
                request_id=request_id,
                status="satisfied",
                reason=grace.terminal_reason,
                served_decode_tokens=grace.served_decode_tokens,
                required_decode_tokens=grace.required_decode_tokens,
                grace_elapsed_ms=max(0.0, now_ms - grace.granted_ts_ms),
            )

    def _finish_restore_obligation(
        self,
        request_id: str,
        state: RestoreObligationState,
        *,
        now_ms: float,
        reason: str,
    ) -> None:
        index = getattr(self, "_restore_obligations", None)
        if index is None:
            return
        obligation = index.get(request_id)
        if obligation is None or obligation.state.terminal:
            return
        funding_reserved_tokens = obligation.funding_reserved_tokens
        funding_reserved_bytes = obligation.funding_reserved_bytes
        released_funding_tokens = self._release_restore_funding_capacity(
            obligation,
            now_ms=now_ms,
            reason=reason,
        )
        self._release_restore_lease(
            request_id,
            now_ms=now_ms,
            reason=reason,
            rollback=state != RestoreObligationState.SATISFIED,
        )
        obligation.finish(state, now_ms=now_ms, reason=reason)
        transaction = self._ensure_restore_transaction(obligation)
        active_grace = getattr(self, "_restore_service_grace_by_request", {}).get(
            request_id
        )
        transaction.stage = {
            RestoreObligationState.SATISFIED: (
                RestoreTransactionStage.SERVICE_GRACE
                if active_grace is not None and active_grace.active
                else RestoreTransactionStage.SATISFIED
            ),
            RestoreObligationState.CANCELLED: RestoreTransactionStage.CANCELLED,
            RestoreObligationState.FAILED: (
                RestoreTransactionStage.FAILED_UNRECOVERABLE
            ),
        }[state]
        if state == RestoreObligationState.FAILED:
            transaction.failure_evidence = {
                "reason": reason,
                "request_cancelled": request_id
                in getattr(self, "_terminal_cancelled_request_ids", set()),
                "raw_input_reconstructable": bool(
                    getattr(self._restore_waiting_request(request_id), "origin_input_ids", None)
                ),
                "context_bytes": obligation.restore_bytes
                + obligation.required_admission_bytes,
                "hbm_capacity_bytes": self.config.hbm_capacity_bytes,
                "shutdown_state": getattr(self, "_shutdown_state", "running"),
                "blocker_codes": list(obligation.blocker_codes),
            }
        self._restore_obligation_counts[state.value] += 1
        gate_state = getattr(self, "_restore_micro_gate_state", {})
        if (
            self.config.restore_micro_gate_enabled
            and gate_state.get("victim_request_id") == request_id
        ):
            gate_stage = (
                "restore_satisfied"
                if state == RestoreObligationState.SATISFIED
                else "failed"
            )
            self._update_restore_micro_gate(
                gate_stage,
                now_ms=now_ms,
                reason=reason,
                obligation_id=obligation.obligation_id,
                restored_h2d_bytes=obligation.restored_bytes,
                obligation_state=state.value,
            )
        self.audit.emit(
            "restore_obligation_terminal",
            now_ms,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            workflow_id=obligation.workflow_id,
            invocation_id=obligation.invocation_id,
            context_id=obligation.context_id,
            state=obligation.state.value,
            reason=reason,
            total_wait_ms=max(0.0, now_ms - obligation.created_ts_ms),
            restored_bytes=obligation.restored_bytes,
            funding_reclaim_bytes=obligation.funding_reclaim_bytes,
            funding_reserved_tokens_before_terminal=funding_reserved_tokens,
            funding_reserved_bytes_before_terminal=funding_reserved_bytes,
            released_funding_tokens=released_funding_tokens,
            command_ids=list(obligation.command_ids),
            cause=obligation.cause.value,
            native_admission_fallback=obligation.native_admission_fallback,
        )

    def plan_running_batch_retraction(
        self,
        running_batch: Any,
    ) -> RunningRetractionPlan | None:
        """Plan one selective retraction at a scheduler safe point."""

        if not self.config.running_batch_retraction_enabled:
            return None
        if getattr(
            self, "_restore_authority_mode", RestoreAuthorityMode.NORMAL_JOINT
        ) != RestoreAuthorityMode.NORMAL_JOINT:
            return None
        now_ms = float(self._now_ms())
        overdue_restore = self._overdue_restore_obligation(now_ms=now_ms)
        if overdue_restore is not None:
            self._running_retraction_counts["restore_debt_barrier"] += 1
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason=(
                    f"restore_debt_barrier:{overdue_restore.obligation_id}"
                ),
            )
            return None
        previous_decision_ms = getattr(self, "_last_retraction_decision_ms", None)
        if (
            previous_decision_ms is not None
            and now_ms - previous_decision_ms
            < self.config.running_batch_retraction_decision_interval_ms
        ):
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason="decision_interval_deferred",
            )
            return None
        self._last_retraction_decision_ms = now_ms
        if getattr(self, "_pending_running_retraction_transaction", None) is not None:
            self._running_retraction_counts["transaction_inflight"] += 1
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason="transaction_inflight",
            )
            return None
        requests = tuple(getattr(running_batch, "reqs", ()) or ())
        if len(requests) <= max(
            1, self.config.observed_admission_min_active_requests
        ):
            self._running_retraction_counts["active_floor"] += 1
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason="active_floor",
            )
            return None
        replacements = self._running_retraction_replacements(now_ms=now_ms)
        if not replacements:
            self._retraction_admission_stall_since_ms = None
            self._running_retraction_counts["no_replacement"] += 1
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason="replacement_absent",
                replacements=replacements,
            )
            return None
        stall_since = getattr(self, "_retraction_admission_stall_since_ms", None)
        if stall_since is None:
            self._retraction_admission_stall_since_ms = now_ms
            self._running_retraction_counts["stall_warming"] += 1
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason="admission_stall_below_threshold",
                replacements=replacements,
            )
            return None
        provenance = self._lock_provenance_extents()
        if provenance is None:
            self._running_retraction_counts["provenance_unavailable"] += 1
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason="provenance_unavailable",
                replacements=replacements,
            )
            return None
        extents, path_error_count = provenance
        if path_error_count:
            self._running_retraction_counts["provenance_error"] += 1
        page_index = self.controller.page_index
        protected_blocker_ids = {
            request_id
            for extent in extents
            if (
                (page := page_index.pages.get(extent.handle)) is None
                or page.semantic_pin_contexts
                or page.active_reader_count > 0
                or not page.transfer_idle
                or not page.sealed
            )
            for request_id in extent.blocker_request_ids
        }

        workflow_ids = {
            metadata.root_workflow_id
            for req in requests
            if (metadata := self._metadata(req)) is not None
        }
        fair_order = self.controller.fairness.ordered(
            workflow_ids,
            memory_charges=self.controller.workflow_memory_charges(),
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
        )
        fair_rank = {
            workflow_id: index for index, workflow_id in enumerate(fair_order)
        }
        ledger = getattr(self, "_lock_service_ledger", None)
        cooldowns = getattr(self, "_retraction_cooldown_until_by_request", {})
        service_graces = getattr(
            self, "_restore_service_grace_by_request", {}
        )
        retraction_counts = getattr(self, "_retraction_counts_by_request", {})
        candidates: list[RunningRetractionCandidate] = []
        for req in requests:
            metadata = self._metadata(req)
            request_id = str(getattr(req, "rid", ""))
            if metadata is None or not request_id:
                continue
            private_bytes = self._estimated_request_private_kv_bytes(req)
            service_status = (
                ledger.service_status(
                    request_id,
                    now_ms=now_ms,
                    window_ms=self.config.running_batch_retraction_min_stall_ms,
                )
                if ledger is not None and ledger.tracks(request_id)
                else "unknown"
            )
            stale_for_ms = (
                ledger.stale_for_ms(request_id, now_ms=now_ms)
                if ledger is not None and ledger.tracks(request_id)
                else 0.0
            )
            try:
                causal = self.controller.frontier.describe_invocation(
                    metadata.invocation_id
                )
                causal_rank = int(causal.score[0])
                unblock_depth = causal.unblock_depth
                causal_known = True
            except (KeyError, ValueError):
                causal_rank = 0
                unblock_depth = 0
                causal_known = False
            candidates.append(
                RunningRetractionCandidate(
                    request_id=request_id,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    private_kv_bytes=private_bytes,
                    service_status=service_status,
                    stale_for_ms=stale_for_ms,
                    causal_rank=causal_rank,
                    unblock_depth=unblock_depth,
                    workflow_fair_rank=fair_rank.get(
                        metadata.root_workflow_id, 0
                    ),
                    prior_retraction_count=int(
                        retraction_counts.get(request_id, 0)
                    ),
                    policy_eligible=(
                        causal_known
                        and now_ms >= cooldowns.get(request_id, 0.0)
                        and not (
                            (grace := service_graces.get(request_id)) is not None
                            and grace.active
                        )
                        and request_id not in protected_blocker_ids
                        and not self._metadata_scope_is_terminal(metadata)
                    ),
                )
            )

        native_reclaim_bytes = self._native_reclaim_capacity_bytes()
        active = self._observed_admission_snapshot(
            native_available_hbm_bytes=native_reclaim_bytes,
            native_max_requests=len(replacements),
        )
        snapshot = ObservedRetractionSnapshot(
            observed_ts_ms=now_ms,
            page_revision=self.controller.page_index.revision,
            topology_revision=self.controller.page_index.topology_revision,
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
            active_kv_budget_bytes=int(
                max(
                    0,
                    self.config.hbm_capacity_bytes
                    - self.config.reserve_hbm_bytes,
                )
                * self.config.observed_admission_active_kv_high_watermark_ratio
            ),
            active_kv_footprint_bytes=active.active_kv_footprint_bytes,
            native_reclaim_capacity_bytes=native_reclaim_bytes,
            admission_stall_ms=max(0.0, now_ms - stall_since),
            running_request_count=len(requests),
            minimum_active_requests=max(
                1, self.config.observed_admission_min_active_requests
            ),
            candidates=tuple(candidates),
            locked_extents=tuple(
                RetractionLockedExtent(
                    extent_id=(
                        f"{item.handle.page_id}:{item.handle.allocation_generation}"
                    ),
                    size_bytes=item.size_bytes,
                    blocker_request_ids=item.blocker_request_ids,
                    fully_attributed=(
                        bool(item.blocker_request_ids)
                        and item.engine_lock_ref == len(item.blocker_request_ids)
                        and (
                            (page := page_index.pages.get(item.handle)) is not None
                            and not page.semantic_pin_contexts
                            and page.active_reader_count == 0
                            and page.transfer_idle
                            and page.sealed
                        )
                    ),
                )
                for item in extents
            ),
            replacements=replacements,
        )
        snapshot, restore_micro_gate_id = self._restore_micro_gate_snapshot(
            snapshot
        )
        decision, source_joint_plan_id = self._running_retraction_decision(
            snapshot,
            restore_micro_gate_id=restore_micro_gate_id,
        )
        plan = decision.plan
        if plan is None:
            self._running_retraction_counts["no_feasible_plan"] += 1
            self._running_retraction_counts[
                f"no_plan_{decision.reason}"
            ] += 1
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason=decision.reason,
                replacements=replacements,
                decision=decision,
            )
            return None

        if not self._can_prepare_restore_obligations(plan, requests):
            self._running_retraction_counts["restore_obligation_fail_closed"] += 1
            if restore_micro_gate_id is not None:
                self._update_restore_micro_gate(
                    "waiting_for_pair",
                    now_ms=now_ms,
                    reason="restore_obligation_preflight_unavailable",
                )
            self._attribute_running_retraction_barrier(
                running_batch,
                now_ms=now_ms,
                planning_reason="restore_obligation_unavailable",
                replacements=replacements,
                decision=decision,
            )
            return None

        barrier_intent_id = self._attribute_running_retraction_barrier(
            running_batch,
            now_ms=now_ms,
            planning_reason=decision.reason,
            replacements=replacements,
            decision=decision,
        )
        preview_started_ns = time.perf_counter_ns()
        tentative_unlock_preview = TentativeUnlockPreviewer(page_index).preview(
            extents,
            plan.request_ids,
            path_error_count=path_error_count,
        )
        preview_compute_us = (
            time.perf_counter_ns() - preview_started_ns
        ) / 1_000.0
        preview_counts = getattr(self, "_running_retraction_preview_counts", None)
        if preview_counts is None:
            preview_counts = Counter()
            self._running_retraction_preview_counts = preview_counts
        preview_counts[tentative_unlock_preview.reason] += 1
        preview_counts[
            "exact" if tentative_unlock_preview.exact else "inexact"
        ] += 1
        preview_samples = getattr(
            self, "_running_retraction_preview_compute_us", None
        )
        if preview_samples is None:
            preview_samples = deque(maxlen=65_536)
            self._running_retraction_preview_compute_us = preview_samples
        preview_samples.append(preview_compute_us)
        self._retraction_sequence += 1
        transaction = _RunningRetractionTransaction(
            transaction_id=f"retraction-{self._retraction_sequence}",
            plan=plan,
            created_ts_ms=now_ms,
            barrier_intent_id=barrier_intent_id,
            tentative_unlock_preview=tentative_unlock_preview,
            source_joint_plan_id=source_joint_plan_id,
            restore_path_extent_ids=self._capture_restore_paths(
                plan, requests
            ),
            restore_micro_gate_id=restore_micro_gate_id,
        )
        self._pending_running_retraction_transaction = transaction
        self._running_retraction_transactions.append(transaction)
        self._running_retraction_counts["planned"] += 1
        if restore_micro_gate_id is not None:
            self._update_restore_micro_gate(
                "transaction_created",
                now_ms=now_ms,
                reason="production_retraction_transaction_created",
                transaction_id=transaction.transaction_id,
                source_joint_plan_id=source_joint_plan_id,
            )
        self.audit.emit(
            "running_retraction_tentative_unlock_preview",
            now_ms,
            transaction_id=transaction.transaction_id,
            barrier_intent_id=barrier_intent_id,
            preview_compute_us=preview_compute_us,
            policy_effect="shadow_only",
            **tentative_unlock_preview.to_audit_fields(),
        )
        self.audit.emit(
            "running_retraction_planned",
            now_ms,
            transaction_id=transaction.transaction_id,
            barrier_intent_id=barrier_intent_id,
            request_ids=list(plan.request_ids),
            replacement_request_ids=list(plan.replacement_request_ids),
            target_reclaim_bytes=plan.target_reclaim_bytes,
            expected_private_reclaim_bytes=plan.expected_private_reclaim_bytes,
            expected_lock_release_bytes=plan.expected_lock_release_bytes,
            expected_reclaim_capacity_bytes=plan.expected_reclaim_capacity_bytes,
            native_reclaim_capacity_before=plan.native_reclaim_capacity_before,
            engine_locked_bytes_before=plan.engine_locked_bytes_before,
            page_revision=plan.page_revision,
            topology_revision=plan.topology_revision,
            admission_stall_ms=snapshot.admission_stall_ms,
            protected_blocker_request_count=len(protected_blocker_ids),
            tentative_unlock_exact=tentative_unlock_preview.exact,
            tentative_newly_migratable_bytes=(
                tentative_unlock_preview.newly_migratable_bytes
            ),
            reason=plan.reason,
            source_joint_plan_id=source_joint_plan_id,
        )
        return plan

    def on_running_batch_retracted(
        self,
        plan: RunningRetractionPlan,
        retracted_requests: list[Any] | tuple[Any, ...],
        *,
        native_reclaim_capacity_before_tokens: int,
        native_reclaim_capacity_after_tokens: int,
        native_available_before_tokens: int | None = None,
        native_available_after_tokens: int | None = None,
    ) -> None:
        """Start the residency half of a selective retraction transaction."""

        now_ms = float(self._now_ms())
        transaction = getattr(
            self, "_pending_running_retraction_transaction", None
        )
        actual_ids = tuple(
            str(getattr(req, "rid", ""))
            for req in retracted_requests
            if str(getattr(req, "rid", ""))
        )
        if transaction is None or transaction.plan is not plan:
            self.audit.emit(
                "running_retraction_callback_stale",
                now_ms,
                request_ids=list(actual_ids),
            )
            return

        before_bytes = max(
            0,
            int(native_reclaim_capacity_before_tokens)
            * self.config.kv_bytes_per_token,
        )
        after_bytes = max(
            0,
            int(native_reclaim_capacity_after_tokens)
            * self.config.kv_bytes_per_token,
        )
        self._create_restore_obligations(
            transaction,
            retracted_requests,
            now_ms=now_ms,
        )
        self._pending_selective_retraction_ids.update(actual_ids)
        self._retracted_engine_request_ids.update(actual_ids)
        cooldown_until = now_ms + self.config.running_batch_retraction_cooldown_ms
        for request_id in actual_ids:
            self._retraction_counts_by_request[request_id] += 1
            self._retraction_cooldown_until_by_request[request_id] = cooldown_until

        self._mark_full_tree_rebuild()
        self.sync_tree(force=True)
        source_joint_plan_id = getattr(
            transaction, "source_joint_plan_id", None
        )
        if source_joint_plan_id is not None:
            self._invalidate_online_joint_plan(
                source_joint_plan_id,
                reason="running_retraction_committed",
                now_ms=now_ms,
            )
        lock_after = (
            self.controller.page_index.physical_kv_state_breakdown().engine_locked_bytes
        )
        available_before_bytes = max(
            0,
            int(
                native_reclaim_capacity_before_tokens
                if native_available_before_tokens is None
                else native_available_before_tokens
            )
            * self.config.kv_bytes_per_token,
        )
        available_after_bytes = max(
            0,
            int(
                native_reclaim_capacity_after_tokens
                if native_available_after_tokens is None
                else native_available_after_tokens
            )
            * self.config.kv_bytes_per_token,
        )
        actual_reclaim = max(0, available_after_bytes - available_before_bytes)
        actual_lock_release = max(
            0, plan.engine_locked_bytes_before - lock_after
        )
        tentative_unlock_preview = getattr(
            transaction, "tentative_unlock_preview", None
        )
        if tentative_unlock_preview is not None:
            physical_after = (
                self.controller.page_index.physical_kv_state_breakdown()
            )
            actual_migratable_delta = (
                physical_after.migratable_bytes
                - tentative_unlock_preview.baseline_migratable_bytes
            )
            self.audit.emit(
                "running_retraction_tentative_unlock_realized",
                now_ms,
                transaction_id=transaction.transaction_id,
                barrier_intent_id=getattr(
                    transaction, "barrier_intent_id", None
                ),
                preview_exact=tentative_unlock_preview.exact,
                preview_page_revision=tentative_unlock_preview.page_revision,
                preview_topology_revision=(
                    tentative_unlock_preview.topology_revision
                ),
                realized_page_revision=self.controller.page_index.revision,
                realized_topology_revision=(
                    self.controller.page_index.topology_revision
                ),
                preview_newly_migratable_bytes=(
                    tentative_unlock_preview.newly_migratable_bytes
                ),
                realized_migratable_delta_bytes=actual_migratable_delta,
                preview_error_bytes=(
                    actual_migratable_delta
                    - tentative_unlock_preview.newly_migratable_bytes
                ),
                preview_engine_lock_release_bytes=(
                    tentative_unlock_preview.projected_engine_lock_release_bytes
                ),
                realized_engine_lock_release_bytes=actual_lock_release,
                request_set_matches=(
                    set(actual_ids) == set(tentative_unlock_preview.request_ids)
                ),
                policy_effect="shadow_only",
            )
        transaction.actual_request_ids = actual_ids
        transaction.native_reclaim_capacity_after = after_bytes
        transaction.actual_reclaim_capacity_bytes = actual_reclaim
        transaction.private_reclaim_bytes = actual_reclaim
        transaction.actual_engine_lock_release_bytes = actual_lock_release
        transaction.allocator_available_before_bytes = available_before_bytes
        transaction.allocator_available_after_bytes = available_after_bytes
        transaction.victim_context_ids = tuple(
            sorted(
                {
                    metadata.context_id
                    for request_id in actual_ids
                    if (
                        metadata := getattr(
                            self, "_request_metadata_by_id", {}
                        ).get(request_id)
                    )
                    is not None
                }
            )
        )
        first_replacement = next(
            iter(plan.replacement_request_ids), None
        )
        replacement_entry = (
            self.controller.visible_admission.get(first_replacement)
            if first_replacement is not None
            else None
        )
        transaction.required_allocator_available_bytes = (
            replacement_entry.request.estimated_incremental_bytes
            if replacement_entry is not None
            else 0
        )
        if getattr(transaction, "restore_micro_gate_id", None) is not None:
            self._update_restore_micro_gate(
                "retraction_committed",
                now_ms=now_ms,
                reason="native_selective_retraction_committed",
                transaction_id=transaction.transaction_id,
                actual_request_ids=list(actual_ids),
                native_reclaim_bytes=actual_reclaim,
                engine_lock_release_bytes=actual_lock_release,
            )

        exact_request_set = set(actual_ids) == set(plan.request_ids)
        reclaim_baseline_matches = (
            before_bytes == plan.native_reclaim_capacity_before
        )
        revisions_advanced = (
            self.controller.page_index.revision >= plan.page_revision
            and self.controller.page_index.topology_revision
            >= plan.topology_revision
        )
        valid_retraction = (
            exact_request_set
            and reclaim_baseline_matches
            and revisions_advanced
        )
        if not valid_retraction:
            transaction.failure_reason = "+".join(
                reason
                for condition, reason in (
                    (not exact_request_set, "request_set_changed"),
                    (
                        not reclaim_baseline_matches,
                        "reclaim_baseline_changed",
                    ),
                    (not revisions_advanced, "revision_regressed"),
                )
                if condition
            )
            self._fail_retraction_transaction(transaction, now_ms=now_ms)
        elif self._retraction_allocator_target_met(transaction):
            self._complete_retraction_transaction(transaction, now_ms=now_ms)
        else:
            transaction.stage = "residency_pending"
            self._running_retraction_counts["residency_pending"] += 1
            self._activate_retraction_admission_barrier(transaction)
            self._queue_next_retraction_residency_command(
                transaction, now_ms=now_ms
            )
        self._running_retraction_actual_reclaim_bytes += actual_reclaim
        self._running_retraction_actual_lock_release_bytes += actual_lock_release
        self.audit.emit(
            "running_retraction_committed",
            now_ms,
            transaction_id=transaction.transaction_id,
            barrier_intent_id=getattr(transaction, "barrier_intent_id", None),
            stage=transaction.stage,
            planned_request_ids=list(plan.request_ids),
            actual_request_ids=list(actual_ids),
            replacement_request_ids=(
                list(plan.replacement_request_ids)
                if transaction.stage == "reclaim_confirmed"
                else []
            ),
            target_reclaim_bytes=plan.target_reclaim_bytes,
            expected_reclaim_capacity_bytes=plan.expected_reclaim_capacity_bytes,
            actual_reclaim_capacity_bytes=actual_reclaim,
            actual_engine_lock_release_bytes=actual_lock_release,
            allocator_available_before_bytes=available_before_bytes,
            allocator_available_after_bytes=available_after_bytes,
            required_allocator_available_bytes=(
                transaction.required_allocator_available_bytes
            ),
            native_reclaim_capacity_before=before_bytes,
            native_reclaim_capacity_after=after_bytes,
            failure_reason=transaction.failure_reason,
            admission_dependency=(
                "allocator_free_confirmed"
                if transaction.stage == "reclaim_confirmed"
                else "physical_residency_ack_pending"
                if transaction.stage == "residency_pending"
                else "not_released"
            ),
        )

    def _activate_retraction_admission_barrier(
        self, transaction: _RunningRetractionTransaction
    ) -> None:
        for entry in tuple(self.controller.visible_admission.entries()):
            self.controller.visible_admission.set_policy_blocked(
                entry.request.request_id,
                reason=f"retraction_residency_pending:{transaction.transaction_id}",
            )

    @staticmethod
    def _retraction_allocator_target_met(
        transaction: _RunningRetractionTransaction,
    ) -> bool:
        return (
            transaction.actual_reclaim_capacity_bytes
            >= transaction.plan.target_reclaim_bytes
            and transaction.allocator_available_after_bytes
            >= transaction.required_allocator_available_bytes
        )

    def _complete_retraction_transaction(
        self,
        transaction: _RunningRetractionTransaction,
        *,
        now_ms: float,
    ) -> None:
        transaction.stage = "reclaim_confirmed"
        transaction.failure_reason = None
        self._retraction_priority_request_ids = (
            transaction.plan.replacement_request_ids
        )
        self._running_retraction_counts["reclaim_confirmed"] += 1
        if getattr(transaction, "restore_micro_gate_id", None) is not None:
            self._update_restore_micro_gate(
                "d2h_completed",
                now_ms=now_ms,
                reason="retraction_residency_completed",
                transaction_id=transaction.transaction_id,
                explicit_d2h_bytes=transaction.explicit_transfer_bytes,
                allocator_reclaim_bytes=(
                    transaction.actual_reclaim_capacity_bytes
                ),
            )
        self._mark_restore_source_transaction_terminal(
            transaction.transaction_id,
            now_ms=now_ms,
            reason="source_retraction_completed",
        )
        self._pending_running_retraction_transaction = None
        self._resync_retraction_barrier_entries()
        self.audit.emit(
            "running_retraction_transaction_completed",
            now_ms,
            transaction_id=transaction.transaction_id,
            barrier_intent_id=getattr(transaction, "barrier_intent_id", None),
            target_reclaim_bytes=transaction.plan.target_reclaim_bytes,
            allocator_reclaim_bytes=transaction.actual_reclaim_capacity_bytes,
            explicit_reclaim_bytes=getattr(transaction, "explicit_reclaim_bytes", 0),
            explicit_transfer_bytes=getattr(
                transaction, "explicit_transfer_bytes", 0
            ),
            residency_command_ids=list(
                getattr(transaction, "residency_command_ids", ())
            ),
            replacement_request_ids=list(
                transaction.plan.replacement_request_ids
            ),
        )

    def _fail_retraction_transaction(
        self,
        transaction: _RunningRetractionTransaction,
        *,
        now_ms: float,
        reason: str | None = None,
    ) -> None:
        transaction.stage = "residency_failed"
        transaction.failure_reason = reason or transaction.failure_reason or "unknown"
        self._running_retraction_counts["residency_failed"] += 1
        if getattr(transaction, "restore_micro_gate_id", None) is not None:
            self._update_restore_micro_gate(
                "failed",
                now_ms=now_ms,
                reason=transaction.failure_reason,
                transaction_id=transaction.transaction_id,
            )
        self._mark_restore_source_transaction_terminal(
            transaction.transaction_id,
            now_ms=now_ms,
            reason=f"source_retraction_failed:{transaction.failure_reason}",
        )
        self._pending_running_retraction_transaction = None
        self._resync_retraction_barrier_entries()
        self.audit.emit(
            "running_retraction_transaction_failed",
            now_ms,
            transaction_id=transaction.transaction_id,
            barrier_intent_id=getattr(transaction, "barrier_intent_id", None),
            failure_reason=transaction.failure_reason,
            target_reclaim_bytes=transaction.plan.target_reclaim_bytes,
            allocator_reclaim_bytes=transaction.actual_reclaim_capacity_bytes,
            explicit_reclaim_bytes=getattr(transaction, "explicit_reclaim_bytes", 0),
            residency_command_ids=list(
                getattr(transaction, "residency_command_ids", ())
            ),
        )

    def _resync_retraction_barrier_entries(self) -> None:
        scheduler = getattr(self, "scheduler", None)
        waiting = {
            str(req.rid): req
            for req in tuple(getattr(scheduler, "waiting_queue", ()) or ())
        }
        for entry in tuple(self.controller.visible_admission.entries()):
            request_id = entry.request.request_id
            metadata = getattr(self, "_request_metadata_by_id", {}).get(request_id)
            if metadata is not None:
                self._sync_visible_gate_state(
                    request_id, metadata, req=waiting.get(request_id)
                )

    def _queue_next_retraction_residency_command(
        self,
        transaction: _RunningRetractionTransaction,
        *,
        now_ms: float,
    ) -> None:
        if transaction.pending_command_id is not None:
            return
        remaining = max(
            0,
            max(
                transaction.plan.target_reclaim_bytes,
                transaction.required_allocator_available_bytes
                - transaction.allocator_available_before_bytes,
            )
            - transaction.actual_reclaim_capacity_bytes,
        )
        if remaining == 0:
            self._complete_retraction_transaction(transaction, now_ms=now_ms)
            return

        victim_contexts = frozenset(transaction.victim_context_ids)
        if not victim_contexts:
            self._fail_retraction_transaction(
                transaction,
                now_ms=now_ms,
                reason="retracted_context_identity_missing",
            )
            return
        observation = self._runtime_resource_observation(now_ms=now_ms)
        builder = self.controller.arbiter.bundle_builder
        candidates: list[tuple[tuple[object, ...], PhysicalBundlePreview]] = []

        def collect(kind: CommandKind) -> None:
            for context_id in sorted(victim_contexts):
                context = self.controller.graph.contexts.get(context_id)
                if context is None:
                    continue
                previews = builder.previews_for_context(
                    kind,
                    context_id,
                    context.epoch,
                    now_ms=now_ms,
                    bypass_owner_context_ids=victim_contexts,
                    host_available_bytes=(
                        observation.host_free_bytes
                        if kind == CommandKind.OFFLOAD_CONTEXT
                        else None
                    ),
                )
                for preview in previews:
                    bundle = preview.bundle
                    owners = frozenset(bundle.owner_context_ids)
                    if (
                        not preview.eligible
                        or bundle.marginal_reclaimable_bytes <= 0
                        or not owners.issubset(victim_contexts)
                    ):
                        continue
                    if kind == CommandKind.DROP_CONTEXT:
                        requires_recompute = any(
                            self.controller.page_index.pages[action.handle].residency
                            == PhysicalResidency.GPU_ONLY
                            for action in preview.page_actions
                        )
                        if (
                            requires_recompute
                            and not self.config.running_batch_retraction_allow_recompute_drop
                        ):
                            continue
                    rank = (
                        bundle.scope != BundleScope.EXCLUSIVE_SUFFIX,
                        preview.copy_bytes > 0,
                        bundle.marginal_reclaimable_bytes < remaining,
                        -bundle.marginal_reclaimable_bytes,
                        bundle.closure_bytes,
                        bundle.bundle_id,
                    )
                    candidates.append((rank, preview))

        collect(CommandKind.OFFLOAD_CONTEXT)
        if not candidates:
            collect(CommandKind.DROP_CONTEXT)
        if not candidates:
            self._fail_retraction_transaction(
                transaction,
                now_ms=now_ms,
                reason="no_feasible_physical_reclaim_bundle",
            )
            return

        queue_conflict = False
        for _, preview in sorted(candidates, key=lambda item: item[0]):
            transaction.command_attempt_count += 1
            command = ControlCommand(
                command_id=(
                    f"{transaction.transaction_id}-residency-"
                    f"{transaction.command_attempt_count}"
                ),
                kind=preview.command_kind,
                created_ts_ms=now_ms,
                context_id=preview.context_id,
                context_epoch=preview.context_epoch,
                target_bytes=preview.bundle.marginal_reclaimable_bytes,
                priority=3.0e9,
                queue_class=CommandQueueClass.URGENT,
                metadata={
                    "reason": "running_retraction_joint_residency",
                    "retraction_transaction_id": transaction.transaction_id,
                    "joint_plan_id": getattr(
                        transaction, "source_joint_plan_id", None
                    ),
                    "bypass_owner_context_ids": sorted(victim_contexts),
                    "physical_bundle_scope": preview.bundle.scope.value,
                    "physical_exclusive_action_bytes": (
                        preview.bundle.exclusive_action_bytes
                    ),
                    "physical_cross_context_action_bytes": (
                        preview.bundle.cross_context_action_bytes
                    ),
                    "physical_foreign_owner_context_ids": list(
                        preview.bundle.foreign_owner_context_ids
                    ),
                },
                physical_bundle=preview.intent(),
            )
            enqueue_outcome = self.controller.enqueue_control_command(command)
            if enqueue_outcome.status != EnqueueStatus.ENQUEUED:
                queue_conflict = True
                continue
            transaction.pending_command_id = command.command_id
            transaction.pending_command_kind = command.kind
            transaction.residency_command_ids.append(command.command_id)
            transaction.stage = "residency_pending"
            for obligation in self._restore_obligation_index().for_source_transaction(
                transaction.transaction_id
            ):
                obligation.state = RestoreObligationState.D2H_INFLIGHT
                obligation.last_progress_ts_ms = now_ms
            self._running_retraction_counts[
                f"residency_{command.kind.value}_queued"
            ] += 1
            self.audit.emit(
                "running_retraction_residency_queued",
                now_ms,
                transaction_id=transaction.transaction_id,
                barrier_intent_id=getattr(
                    transaction, "barrier_intent_id", None
                ),
                command_id=command.command_id,
                kind=command.kind.value,
                context_id=command.context_id,
                bundle_id=preview.bundle.bundle_id,
                bundle_scope=preview.bundle.scope.value,
                target_reclaim_bytes=transaction.plan.target_reclaim_bytes,
                remaining_reclaim_bytes=remaining,
                selected_reclaim_bytes=preview.bundle.marginal_reclaimable_bytes,
                transfer_bytes=preview.copy_bytes,
            )
            return
        if queue_conflict:
            transaction.stage = "residency_wait_queue"

    def _advance_retraction_transaction(
        self,
        acks: tuple[CommandAck, ...],
        *,
        now_ms: float,
    ) -> None:
        transaction = getattr(
            self, "_pending_running_retraction_transaction", None
        )
        if transaction is None or transaction.stage == "planned":
            return
        if (
            now_ms - transaction.created_ts_ms
            >= self.config.running_batch_retraction_transaction_timeout_ms
        ):
            self._fail_retraction_transaction(
                transaction,
                now_ms=now_ms,
                reason="residency_transaction_timeout",
            )
            return
        matching_ack = next(
            (
                ack
                for ack in acks
                if ack.command_id == transaction.pending_command_id
            ),
            None,
        )
        if transaction.pending_command_id is not None and matching_ack is None:
            return
        if matching_ack is not None:
            transaction.pending_command_id = None
            command_kind = transaction.pending_command_kind
            transaction.pending_command_kind = None
            self._record_retraction_restore_extents(
                transaction,
                matching_ack,
                command_kind=command_kind,
                now_ms=now_ms,
            )
            if matching_ack.status != CommandStatus.COMPLETED:
                self._fail_retraction_transaction(
                    transaction,
                    now_ms=now_ms,
                    reason=(
                        f"residency_{command_kind.value if command_kind else 'unknown'}_"
                        f"{matching_ack.status.value}:{matching_ack.reason}"
                    ),
                )
                return
            if command_kind == CommandKind.OFFLOAD_CONTEXT:
                transaction.explicit_transfer_bytes += matching_ack.actual_bytes
            current_available = self._allocator_available_bytes()
            transaction.allocator_available_after_bytes = current_available
            transaction.actual_reclaim_capacity_bytes = max(
                0,
                current_available - transaction.allocator_available_before_bytes,
            )
            transaction.explicit_reclaim_bytes = max(
                transaction.explicit_reclaim_bytes,
                max(
                    0,
                    transaction.actual_reclaim_capacity_bytes
                    - transaction.private_reclaim_bytes,
                ),
            )
            self.audit.emit(
                "running_retraction_residency_ack",
                now_ms,
                transaction_id=transaction.transaction_id,
                barrier_intent_id=getattr(
                    transaction, "barrier_intent_id", None
                ),
                command_id=matching_ack.command_id,
                kind=command_kind.value if command_kind is not None else None,
                actual_bytes=matching_ack.actual_bytes,
                allocator_available_bytes=current_available,
                allocator_reclaim_bytes=transaction.actual_reclaim_capacity_bytes,
                target_reclaim_bytes=transaction.plan.target_reclaim_bytes,
            )
            if self._retraction_allocator_target_met(transaction):
                self._complete_retraction_transaction(
                    transaction, now_ms=now_ms
                )
                return
        self._queue_next_retraction_residency_command(
            transaction, now_ms=now_ms
        )

    def _record_retraction_restore_extents(
        self,
        transaction: _RunningRetractionTransaction,
        ack: CommandAck,
        *,
        command_kind: CommandKind | None,
        now_ms: float,
    ) -> None:
        if command_kind != CommandKind.OFFLOAD_CONTEXT:
            return
        ack_extent_ids = {
            f"page:{handle.page_id}:{handle.allocation_generation}"
            for handle in ack.page_handles
        }
        for obligation in self._restore_obligation_index().for_source_transaction(
            transaction.transaction_id
        ):
            exact = tuple(
                sorted(ack_extent_ids.intersection(obligation.path_extent_ids))
            )
            if exact:
                merged = tuple(
                    sorted(set(obligation.required_extent_ids).union(exact))
                )
                obligation.set_required_extents(
                    merged,
                    restore_bytes=self._extent_ids_bytes(merged),
                    now_ms=now_ms,
                )
            self.audit.emit(
                "restore_obligation_source_d2h_ack",
                now_ms,
                obligation_id=obligation.obligation_id,
                request_id=obligation.request_id,
                command_id=ack.command_id,
                status=ack.status.value,
                exact_path_extent_count=len(exact),
                required_extent_count=len(obligation.required_extent_ids),
                actual_bytes=ack.actual_bytes,
            )

    def _restore_attempt_stamp(self) -> tuple[object, ...]:
        page_index = self.controller.page_index
        return (
            page_index.revision,
            page_index.topology_revision,
            self._allocator_available_bytes(),
            sum(
                len(item)
                for allocations in getattr(
                    self, "_restore_funding_allocations", {}
                ).values()
                for item in allocations
            ),
            int(getattr(self.controller, "_transfer_epoch", 0)),
        )

    def _restore_external_progress_token(
        self,
        obligation: RestoreObligation,
        *,
        required_bytes: int,
        closure_fingerprint: str,
        command: ControlCommand | None = None,
    ) -> ExternalProgressToken:
        snapshots = self._context_physical_snapshots(obligation.context_id)
        engine_epoch = tuple(
            (
                item.request_id,
                item.request_generation,
                item.queue_location.value,
                item.req_pool_slot,
                item.radix_lock_owned,
                item.terminal,
            )
            for item in snapshots
        )
        native_load_epoch = tuple(
            (item.request_id, item.native_load_operation_id)
            for item in snapshots
            if item.native_load_operation_id is not None
        )
        available = (
            self._allocator_available_bytes()
            + obligation.funding_reserved_bytes
        )
        return ExternalProgressToken(
            engine_owner_epoch=engine_epoch,
            closure_fingerprint=closure_fingerprint,
            effective_capacity_threshold_epoch=(
                max(0, int(required_bytes)),
                available >= max(0, int(required_bytes)),
            ),
            command_ownership_epoch=(
                self.controller.command_ownership_epoch(obligation.context_id)
            ),
            guard_generation=(
                self.controller.transfer_guard.generation_for(command)
                if command is not None
                else 0
            ),
            native_load_generation=native_load_epoch,
        )

    def _restore_waiting_request(self, request_id: str) -> Any | None:
        return next(
            (
                req
                for req in tuple(
                    getattr(self.scheduler, "waiting_queue", ()) or ()
                )
                if str(getattr(req, "rid", "")) == request_id
            ),
            None,
        )

    def _refresh_restore_obligation(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
    ) -> tuple[Any | None, tuple[str, ...] | None]:
        req = self._restore_waiting_request(obligation.request_id)
        if req is None:
            return None, obligation.required_extent_ids
        obligation.requeued = True
        required = self._request_path_extent_ids(
            req, obligation.context_id, cpu_only=True
        )
        if required is None:
            refresh = getattr(req, "init_next_round_input", None)
            if callable(refresh):
                refresh(self.tree_cache)
                required = self._request_path_extent_ids(
                    req, obligation.context_id, cpu_only=True
                )
        if required is None:
            return req, None
        if obligation.cause == RestoreObligationCause.ORDINARY_WAITING_PREFIX:
            current_path = self._request_path_extent_ids(
                req, obligation.context_id, cpu_only=False
            )
            if current_path is None:
                return req, None
            self._rebind_ordinary_restore_path(
                obligation,
                current_path,
                now_ms=now_ms,
            )
        obligation.set_required_extents(
            required,
            restore_bytes=self._extent_ids_bytes(required),
            now_ms=now_ms,
        )
        entry = self.controller.visible_admission.get(obligation.request_id)
        obligation.required_admission_bytes = (
            entry.request.estimated_incremental_bytes if entry is not None else 0
        )
        return req, required

    def _rebind_ordinary_restore_path(
        self,
        obligation: RestoreObligation,
        extent_ids: tuple[str, ...],
        *,
        now_ms: float,
    ) -> None:
        """Make the waiting request's current Radix path physically authoritative."""

        page_index = self.controller.page_index
        if (
            not extent_ids
            or not page_index.has_context(obligation.context_id)
            or page_index.context_epoch(obligation.context_id)
            != obligation.context_epoch
        ):
            return
        handles: list[PageHandle] = []
        for extent_id in extent_ids:
            try:
                handle = self._runtime_page_handle_from_extent_id(extent_id)
            except ValueError:
                return
            page = page_index.pages.get(handle)
            if page is None or page.residency == PhysicalResidency.DEAD:
                return
            handles.append(handle)
        desired = set(handles)
        previous = {
            page.handle for page in page_index.context_pages(obligation.context_id)
        }
        obligation.path_extent_ids = tuple(sorted(set(extent_ids)))
        if desired == previous:
            return
        page_index.bind_pages(
            obligation.context_id,
            obligation.context_epoch,
            desired,
            replace=True,
        )
        self._restore_obligation_counts["path_rebound"] += 1
        self.audit.emit(
            "restore_obligation_path_rebound",
            now_ms,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            context_id=obligation.context_id,
            context_epoch=obligation.context_epoch,
            previous_extent_count=len(previous),
            current_extent_count=len(desired),
            added_extent_count=len(desired - previous),
            removed_extent_count=len(previous - desired),
        )

    def _activate_native_restore_fallback(
        self,
        obligation: RestoreObligation,
        req: Any,
        *,
        now_ms: float,
    ) -> bool:
        """Reserve admission capacity, then use native HiCache load or prefill."""

        required_extent_count = len(obligation.required_extent_ids)
        restore_bytes = obligation.restore_bytes
        if bool(getattr(self.config, "restore_lease_enabled", True)):
            lease = self._grant_restore_lease(
                obligation,
                h2d_bytes=0,
                now_ms=now_ms,
            )
            if lease is None:
                return False
            lease.mark_restored()
        obligation.use_native_admission_fallback(now_ms=now_ms)
        self._restore_obligation_counts["native_admission_fallback"] += 1
        metadata = self._request_metadata_by_id.get(obligation.request_id)
        if metadata is not None:
            self._sync_visible_gate_state(
                obligation.request_id,
                metadata,
                req=req,
            )
        self.audit.emit(
            "restore_obligation_native_fallback_ready",
            now_ms,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            workflow_id=obligation.workflow_id,
            context_id=obligation.context_id,
            context_epoch=obligation.context_epoch,
            required_extent_count=required_extent_count,
            restore_bytes=restore_bytes,
            required_admission_bytes=obligation.required_admission_bytes,
            fallback="sglang_native_load_back_or_prefill",
        )
        return True

    def _restore_h2d_previews(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
        device_available_bytes: int,
    ) -> tuple[PhysicalBundlePreview, ...]:
        required_handles: set[PageHandle] = set()
        for extent_id in obligation.required_extent_ids:
            try:
                required_handles.add(
                    self._runtime_page_handle_from_extent_id(extent_id)
                )
            except ValueError:
                continue
        previews = self.controller.arbiter.bundle_builder.previews_for_context(
            CommandKind.PREFETCH_CONTEXT,
            obligation.context_id,
            obligation.context_epoch,
            now_ms=now_ms,
            device_available_bytes=device_available_bytes,
        )
        return tuple(
            sorted(
                (
                    preview
                    for preview in previews
                    if required_handles.intersection(
                        action.handle
                        for action in preview.page_actions
                        if action.action == PhysicalPageAction.START_H2D
                    )
                ),
                key=lambda preview: (
                    -len(
                        required_handles.intersection(
                            action.handle
                            for action in preview.page_actions
                            if action.action == PhysicalPageAction.START_H2D
                        )
                    ),
                    preview.copy_bytes,
                    preview.bundle.closure_bytes,
                    preview.bundle.bundle_id,
                ),
            )
        )

    def _restore_funding_preview(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
        deficit_bytes: int,
    ) -> tuple[PhysicalBundlePreview | None, tuple[str, ...]]:
        observation = self._runtime_resource_observation(now_ms=now_ms)
        protected_contexts = {
            item.context_id for item in self._restore_obligation_index().active()
        }
        blockers: set[str] = set()
        candidates: list[PhysicalBundlePreview] = []
        for context_id, context in sorted(self.controller.graph.contexts.items()):
            if context_id in protected_contexts:
                continue
            previews = self.controller.arbiter.bundle_builder.previews_for_context(
                CommandKind.OFFLOAD_CONTEXT,
                context_id,
                context.epoch,
                now_ms=now_ms,
                allow_ready_owners=True,
                protected_context_id=obligation.context_id,
                host_available_bytes=observation.host_free_bytes,
            )
            for preview in previews:
                if protected_contexts.intersection(
                    preview.bundle.owner_context_ids
                ):
                    continue
                if not preview.eligible:
                    blockers.update(item.code.value for item in preview.blockers)
                    continue
                if preview.bundle.marginal_reclaimable_bytes > 0:
                    candidates.append(preview)
        if not candidates:
            return None, tuple(sorted(blockers or {"no_funding_bundle"}))
        selected = min(
            candidates,
            key=lambda preview: (
                preview.bundle.scope != BundleScope.EXCLUSIVE_SUFFIX,
                preview.copy_bytes > 0,
                preview.bundle.marginal_reclaimable_bytes < deficit_bytes,
                abs(preview.bundle.marginal_reclaimable_bytes - deficit_bytes),
                preview.copy_bytes,
                preview.bundle.bundle_id,
            ),
        )
        return selected, ()

    def _restore_lease_allocator_deficit_bytes(
        self,
        obligation: RestoreObligation,
        *,
        h2d_bytes: int,
    ) -> int:
        """Return only the allocator deficit that physical reclaim can satisfy."""

        lease = self._restore_lease_index().get(obligation.request_id)
        required_tokens = self._restore_lease_tokens(
            obligation.required_admission_bytes
        )
        current_tokens = (
            lease.reserved_tokens
            if lease is not None and not lease.state.terminal
            else 0
        )
        additional_tokens = max(0, required_tokens - current_tokens)
        h2d_tokens = (
            max(0, int(h2d_bytes)) + self.config.kv_bytes_per_token - 1
        ) // self.config.kv_bytes_per_token
        missing_tokens = max(
            0,
            additional_tokens
            + h2d_tokens
            - self._allocator_available_tokens()
            - self._restore_funding_reserved_tokens(obligation.request_id),
        )
        return missing_tokens * self.config.kv_bytes_per_token

    def _try_queue_restore_lease_funding(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
        attempt_stamp: tuple[object, ...],
        h2d_bytes: int,
    ) -> tuple[bool, tuple[str, ...]]:
        """Fund a restore lease only when reclaim can close its allocator deficit."""

        deficit_bytes = self._restore_lease_allocator_deficit_bytes(
            obligation,
            h2d_bytes=h2d_bytes,
        )
        if deficit_bytes <= 0:
            return False, ("restore_lease_capacity",)
        funding, blockers = self._restore_funding_preview(
            obligation,
            now_ms=now_ms,
            deficit_bytes=deficit_bytes,
        )
        if funding is None:
            return False, tuple(
                sorted({"restore_lease_capacity", *blockers})
            )
        queued = self._queue_restore_obligation_command(
            obligation,
            funding,
            now_ms=now_ms,
            reason="restore_obligation_funding",
            attempt_stamp=attempt_stamp,
            funding_target_bytes=deficit_bytes,
        )
        if queued:
            return True, ()
        return False, ("restore_funding_queue_conflict",)

    def _queue_restore_obligation_command(
        self,
        obligation: RestoreObligation,
        preview: PhysicalBundlePreview,
        *,
        now_ms: float,
        reason: str,
        attempt_stamp: tuple[object, ...],
        funding_target_bytes: int = 0,
        command: ControlCommand | None = None,
        preflight: Any | None = None,
    ) -> Any:
        if command is None:
            command = self._build_restore_obligation_command(
                obligation,
                preview,
                now_ms=now_ms,
                reason=reason,
                funding_target_bytes=funding_target_bytes,
            )
        phase = getattr(
            self, "_safe_point_physical_phase", SafePointPhysicalPhase.IDLE
        )
        if phase != SafePointPhysicalPhase.TRANSACTIONAL_COMMIT:
            if not self._begin_physical_transactional_commit(
                command.context_id
            ):
                return EnqueueOutcome(
                    status=EnqueueStatus.STALE_CERTIFICATE,
                    canonical_command_id=None,
                    attempt_key=(
                        command.context_id,
                        command.context_epoch,
                        command.kind.value,
                        "native_ownership_readset_changed",
                    ),
                    blocker_codes=("native_ownership_readset_changed",),
                    wake_conditions=("safe_point_epoch_advanced",),
                )
        # Canonical ownership may have changed since the planning preflight.
        preflight = self.controller.preflight_control_command(command)
        outcome = self.controller.enqueue_control_command(
            command, preflight=preflight
        )
        if not outcome:
            self._restart_physical_capture_epoch()
            return outcome
        canonical_command_id = outcome.canonical_command_id
        assert canonical_command_id is not None
        obligation.start_command(
            canonical_command_id,
            command.kind,
            now_ms=now_ms,
            attempt_stamp=attempt_stamp,
        )
        if funding_target_bytes > 0:
            funding_targets = getattr(
                self, "_restore_funding_target_by_command", None
            )
            if funding_targets is None:
                funding_targets = {}
                self._restore_funding_target_by_command = funding_targets
            funding_targets.setdefault(
                canonical_command_id, {}
            )[obligation.request_id] = int(funding_target_bytes)
        if command.kind == CommandKind.PREFETCH_CONTEXT and bool(
            getattr(self.config, "restore_lease_enabled", True)
        ):
            lease = self._restore_lease_index().get(obligation.request_id)
            if lease is None or lease.state.terminal:
                raise RuntimeError(
                    "restore H2D command was queued without an active capacity lease"
                )
            lease.mark_h2d_inflight(canonical_command_id)
        subscribers = getattr(self, "_restore_command_to_request", None)
        if subscribers is None:
            subscribers = {}
            self._restore_command_to_request = subscribers
        subscribers.setdefault(
            canonical_command_id, set()
        ).add(obligation.request_id)
        transaction = self._ensure_restore_transaction(obligation)
        certificate = transaction.feasibility_certificate
        certificate_generation = (
            certificate.certificate_generation if certificate is not None else 0
        )
        transaction.add_operation(
            RestorePhysicalOperation(
                stage=command.kind.value,
                attempt_key=outcome.attempt_key,
                certificate_generation=certificate_generation,
                canonical_command_id=canonical_command_id,
                adopted=outcome.status == EnqueueStatus.ADOPT_EXISTING,
            )
        )
        transaction.stage = (
            RestoreTransactionStage.H2D_ADOPTED
            if outcome.status == EnqueueStatus.ADOPT_EXISTING
            else RestoreTransactionStage.H2D_QUEUED
            if command.kind == CommandKind.PREFETCH_CONTEXT
            else RestoreTransactionStage.WAIT_FUNDING
        )
        self._restore_obligation_counts[
            "command_adopted"
            if outcome.status == EnqueueStatus.ADOPT_EXISTING
            else f"{reason}_queued"
        ] += 1
        self.audit.emit(
            "restore_obligation_command_queued",
            now_ms,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            workflow_id=obligation.workflow_id,
            source_joint_plan_id=obligation.source_joint_plan_id,
            command_id=canonical_command_id,
            proposed_command_id=command.command_id,
            enqueue_status=outcome.status.value,
            command_kind=command.kind.value,
            reason=reason,
            context_id=command.context_id,
            physical_bundle_id=preview.bundle.bundle_id,
            closure_bytes=preview.bundle.closure_bytes,
            copy_bytes=preview.copy_bytes,
            reclaimable_bytes=preview.bundle.marginal_reclaimable_bytes,
            funding_target_bytes=max(0, int(funding_target_bytes)),
        )
        return outcome

    def _build_restore_obligation_command(
        self,
        obligation: RestoreObligation,
        preview: PhysicalBundlePreview,
        *,
        now_ms: float,
        reason: str,
        funding_target_bytes: int = 0,
    ) -> ControlCommand:
        self._restore_command_sequence += 1
        command_id = (
            f"{obligation.obligation_id}-command-{self._restore_command_sequence}"
        )
        return ControlCommand(
            command_id=command_id,
            kind=preview.command_kind,
            created_ts_ms=now_ms,
            context_id=preview.context_id,
            context_epoch=preview.context_epoch,
            target_bytes=(
                preview.copy_bytes
                if preview.command_kind == CommandKind.PREFETCH_CONTEXT
                else preview.bundle.marginal_reclaimable_bytes
            ),
            priority=5.0e9,
            deadline_ms=obligation.created_ts_ms
            + self.config.restore_obligation_escalation_ms,
            queue_class=CommandQueueClass.URGENT,
            metadata={
                "reason": reason,
                "joint_plan_id": obligation.source_joint_plan_id,
                "restore_obligation_id": obligation.obligation_id,
                "restore_request_id": obligation.request_id,
                "source_retraction_transaction_id": (
                    obligation.source_retraction_transaction_id
                ),
                "physical_bundle_scope": preview.bundle.scope.value,
                "physical_exclusive_action_bytes": (
                    preview.bundle.exclusive_action_bytes
                ),
                "physical_cross_context_action_bytes": (
                    preview.bundle.cross_context_action_bytes
                ),
                "physical_foreign_owner_context_ids": list(
                    preview.bundle.foreign_owner_context_ids
                ),
                "allow_ready_owners": (
                    preview.command_kind == CommandKind.OFFLOAD_CONTEXT
                ),
                "protected_context_id": obligation.context_id,
                "restore_funding_target_bytes": max(
                    0, int(funding_target_bytes)
                ),
            },
            physical_bundle=preview.intent(),
        )

    def _block_restore_obligation(
        self,
        obligation: RestoreObligation,
        *,
        now_ms: float,
        stamp: tuple[object, ...] | None = None,
        blocker_codes: tuple[str, ...],
        external_progress_token: ExternalProgressToken | None = None,
        wake_conditions: tuple[str, ...] = (),
    ) -> None:
        fingerprint = "|".join(blocker_codes) or "unknown"
        if (
            (
                external_progress_token is not None
                and not obligation.external_progressed(external_progress_token)
                or external_progress_token is None
                and obligation.last_attempt_stamp == stamp
            )
            and obligation.blocker_fingerprint == fingerprint
        ):
            return
        obligation.block(
            blocker_codes=blocker_codes,
            blocker_fingerprint=fingerprint,
            attempt_stamp=stamp,
            external_progress_token=external_progress_token,
            wake_conditions=wake_conditions,
            now_ms=now_ms,
        )
        transaction = self._ensure_restore_transaction(obligation)
        transaction.stage = RestoreTransactionStage.WAIT_EVENT
        transaction.wait_condition = tuple(sorted(set(wake_conditions)))
        transaction.external_progress_token = external_progress_token
        self._restore_obligation_counts["blocked"] += 1
        self.audit.emit(
            "restore_obligation_blocked",
            now_ms,
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            state=obligation.state.value,
            blocker_codes=list(obligation.blocker_codes),
            blocker_fingerprint=obligation.blocker_fingerprint,
            wake_conditions=list(obligation.wake_conditions),
            retry_count=obligation.retry_count,
            wait_ms=max(0.0, now_ms - obligation.created_ts_ms),
        )
        wait_ms = max(0.0, now_ms - obligation.created_ts_ms)
        if (
            wait_ms >= self.config.restore_obligation_max_blocked_ms
            and not obligation.liveness_escalated
        ):
            obligation.liveness_escalated = True
            self._restore_obligation_counts["liveness_escalated"] += 1
            self.audit.emit(
                "restore_obligation_liveness_escalated",
                now_ms,
                obligation_id=obligation.obligation_id,
                request_id=obligation.request_id,
                wait_ms=wait_ms,
                blocker_codes=list(obligation.blocker_codes),
                action="freeze_normal_admission_and_wait_for_physical_state_change",
            )
            oldest = next(iter(self._restore_obligation_index().active()), None)
            if oldest is not None and oldest.request_id == obligation.request_id:
                self._request_restore_drain(obligation, now_ms=now_ms)

    def _drive_restore_obligations(self, *, now_ms: float) -> None:
        phase = getattr(
            self, "_safe_point_physical_phase", SafePointPhysicalPhase.IDLE
        )
        if phase == SafePointPhysicalPhase.IDLE:
            self._begin_physical_safe_point_apply_events()
            self._begin_physical_safe_point_capture_and_plan()
        elif phase == SafePointPhysicalPhase.TRANSACTIONAL_COMMIT:
            # Direct unit/integration drivers may call this outside scheduler_step.
            self._restart_physical_capture_epoch()
        if (
            getattr(self, "_restore_obligations", None) is None
            or not self._restore_obligations.active()
        ):
            return
        if self.controller.has_pending_transfer_work():
            return
        for obligation in self._restore_obligation_index().active():
            if obligation.state == RestoreObligationState.TICKET_READY:
                continue
            if (
                not obligation.source_transaction_terminal
                or not obligation.requeued
                or obligation.pending_command_id is not None
            ):
                continue
            req, required = self._refresh_restore_obligation(
                obligation, now_ms=now_ms
            )
            stamp = (
                *self._restore_attempt_stamp(),
                obligation.context_epoch,
                obligation.required_admission_bytes,
                obligation.required_extent_ids,
            )
            if (
                obligation.blocker_codes
                and obligation.last_external_progress_token is None
                and obligation.last_attempt_stamp == stamp
            ):
                continue
            if req is None:
                self._block_restore_obligation(
                    obligation,
                    now_ms=now_ms,
                    stamp=stamp,
                    blocker_codes=("request_not_requeued",),
                )
                continue
            if required is None:
                self._block_restore_obligation(
                    obligation,
                    now_ms=now_ms,
                    stamp=stamp,
                    blocker_codes=("request_path_stale",),
                )
                continue
            if not required:
                if not self._begin_physical_transactional_commit(
                    obligation.context_id
                ):
                    self._block_restore_obligation(
                        obligation,
                        now_ms=now_ms,
                        stamp=stamp,
                        blocker_codes=("native_ownership_readset_changed",),
                        wake_conditions=("safe_point_epoch_advanced",),
                    )
                    return
                if bool(getattr(self.config, "restore_lease_enabled", True)):
                    lease = self._grant_restore_lease(
                        obligation,
                        h2d_bytes=0,
                        now_ms=now_ms,
                    )
                    if lease is None:
                        self._restart_physical_capture_epoch()
                        queued, blockers = (
                            self._try_queue_restore_lease_funding(
                                obligation,
                                now_ms=now_ms,
                                attempt_stamp=stamp,
                                h2d_bytes=0,
                            )
                        )
                        if queued:
                            return
                        self._block_restore_obligation(
                            obligation,
                            now_ms=now_ms,
                            stamp=stamp,
                            blocker_codes=blockers,
                        )
                        continue
                    if not self._pin_restore_lease_prefix(
                        obligation, req, now_ms=now_ms
                    ):
                        self._release_restore_lease(
                            obligation.request_id,
                            now_ms=now_ms,
                            reason="restore_prefix_pin_failed",
                            rollback=True,
                        )
                        self._restart_physical_capture_epoch()
                        self._block_restore_obligation(
                            obligation,
                            now_ms=now_ms,
                            stamp=stamp,
                            blocker_codes=("restore_prefix_pin_failed",),
                        )
                        continue
                    lease.mark_restored()
                    transaction = self._ensure_restore_transaction(obligation)
                    transaction.capacity_reservation_id = getattr(
                        lease, "lease_id", f"restore-lease:{obligation.request_id}"
                    )
                    transaction.prefix_pin_token = (
                        f"restore-pin:{obligation.request_id}:native"
                    )
                    transaction.stage = RestoreTransactionStage.RESTORED_RESERVED
                obligation.mark_ticket_ready(now_ms=now_ms)
                self._restore_obligation_counts["ticket_ready"] += 1
                metadata = self._request_metadata_by_id.get(obligation.request_id)
                if metadata is not None:
                    self._sync_visible_gate_state(
                        obligation.request_id, metadata, req=req
                    )
                self.audit.emit(
                    "restore_obligation_ticket_ready",
                    now_ms,
                    obligation_id=obligation.obligation_id,
                    request_id=obligation.request_id,
                    restored_bytes=obligation.restored_bytes,
                    funding_reclaim_bytes=obligation.funding_reclaim_bytes,
                )
                self._restart_physical_capture_epoch()
                continue
            available = self._allocator_available_bytes()
            available += obligation.funding_reserved_bytes
            previews = self._restore_h2d_previews(
                obligation,
                now_ms=now_ms,
                device_available_bytes=available,
            )
            eligible = next(
                (
                    item
                    for item in previews
                    if item.eligible
                    and item.copy_bytes + obligation.required_admission_bytes
                    <= available
                ),
                None,
            )
            if eligible is not None:
                command = self._build_restore_obligation_command(
                    obligation,
                    eligible,
                    now_ms=now_ms,
                    reason="restore_obligation_h2d",
                )
                required_bytes = (
                    eligible.copy_bytes + obligation.required_admission_bytes
                )
                progress_token = self._restore_external_progress_token(
                    obligation,
                    required_bytes=required_bytes,
                    closure_fingerprint=(
                        eligible.bundle.generation_fingerprint
                    ),
                    command=command,
                )
                if (
                    obligation.blocker_codes
                    and not obligation.external_progressed(progress_token)
                ):
                    continue
                preflight = self.controller.preflight_control_command(command)
                if not preflight:
                    blocker_codes = preflight.blocker_codes or (
                        preflight.status.value,
                    )
                    self._block_restore_obligation(
                        obligation,
                        now_ms=now_ms,
                        blocker_codes=blocker_codes,
                        external_progress_token=progress_token,
                        wake_conditions=preflight.wake_conditions,
                    )
                    continue
                physical_snapshots = self._context_physical_snapshots(
                    obligation.context_id
                )
                physical_blockers: set[str] = set()
                wake_conditions: set[str] = set()
                if any(item.engine_owned for item in physical_snapshots):
                    physical_blockers.add(TransferBlockerCode.ENGINE_BUSY.value)
                    wake_conditions.add("engine_owner_changed")
                if any(
                    item.native_load_operation_id is not None
                    for item in physical_snapshots
                ):
                    physical_blockers.add(TransferBlockerCode.NODE_LOADING.value)
                    wake_conditions.add("native_load_terminal")
                if any(
                    item.explicit_transfer_ids for item in physical_snapshots
                ):
                    physical_blockers.add(TransferBlockerCode.INFLIGHT.value)
                    wake_conditions.add("explicit_transfer_terminal")
                if (
                    preflight.status != EnqueueStatus.ADOPT_EXISTING
                    and physical_blockers
                ):
                    self._block_restore_obligation(
                        obligation,
                        now_ms=now_ms,
                        blocker_codes=tuple(sorted(physical_blockers)),
                        external_progress_token=progress_token,
                        wake_conditions=tuple(sorted(wake_conditions)),
                    )
                    continue
                if not self._begin_physical_transactional_commit(
                    obligation.context_id
                ):
                    self._block_restore_obligation(
                        obligation,
                        now_ms=now_ms,
                        blocker_codes=("native_ownership_readset_changed",),
                        wake_conditions=("safe_point_epoch_advanced",),
                    )
                    return
                transaction = self._ensure_restore_transaction(obligation)
                self._restore_certificate_sequence = (
                    getattr(self, "_restore_certificate_sequence", 0) + 1
                )
                certificate = RestoreFeasibilityCertificate(
                    certificate_generation=self._restore_certificate_sequence,
                    context_epoch=obligation.context_epoch,
                    attempt_key=preflight.attempt_key,
                    required_bytes=required_bytes,
                    closure_fingerprint=(
                        eligible.bundle.generation_fingerprint
                    ),
                )
                transaction.feasibility_certificate = certificate
                if bool(getattr(self.config, "restore_lease_enabled", True)):
                    lease = self._grant_restore_lease(
                        obligation,
                        h2d_bytes=eligible.copy_bytes,
                        now_ms=now_ms,
                    )
                    if lease is None:
                        self._restart_physical_capture_epoch()
                        self._block_restore_obligation(
                            obligation,
                            now_ms=now_ms,
                            stamp=stamp,
                            blocker_codes=("restore_lease_capacity",),
                            external_progress_token=progress_token,
                            wake_conditions=("capacity_threshold_satisfied",),
                        )
                        return
                    transaction.capacity_reservation_id = lease.lease_id
                    if not self._pin_restore_lease_prefix(
                        obligation,
                        req,
                        now_ms=now_ms,
                        allow_unmaterialized=True,
                    ):
                        self._release_restore_lease(
                            obligation.request_id,
                            now_ms=now_ms,
                            reason="restore_prefix_pin_prepare_failed",
                            rollback=True,
                        )
                        transaction.capacity_reservation_id = None
                        self._restart_physical_capture_epoch()
                        self._block_restore_obligation(
                            obligation,
                            now_ms=now_ms,
                            blocker_codes=("restore_prefix_pin_failed",),
                            external_progress_token=progress_token,
                            wake_conditions=("radix_path_changed",),
                        )
                        return
                    if obligation.request_id in getattr(
                        self, "_restore_lease_pins", {}
                    ):
                        transaction.prefix_pin_token = (
                            f"restore-pin:{obligation.request_id}:"
                            f"{certificate.certificate_generation}"
                        )
                transaction.stage = RestoreTransactionStage.PREPARED
                queued = self._queue_restore_obligation_command(
                    obligation,
                    eligible,
                    now_ms=now_ms,
                    reason="restore_obligation_h2d",
                    attempt_stamp=stamp,
                    command=command,
                    preflight=preflight,
                )
                if not queued:
                    self._release_restore_lease(
                        obligation.request_id,
                        now_ms=now_ms,
                        reason="restore_h2d_queue_conflict",
                        rollback=True,
                    )
                    transaction.capacity_reservation_id = None
                    transaction.prefix_pin_token = None
                    if getattr(
                        self,
                        "_safe_point_physical_phase",
                        SafePointPhysicalPhase.IDLE,
                    ) == SafePointPhysicalPhase.TRANSACTIONAL_COMMIT:
                        self._restart_physical_capture_epoch()
                    self._block_restore_obligation(
                        obligation,
                        now_ms=now_ms,
                        blocker_codes=(
                            *(queued.blocker_codes or (queued.status.value,)),
                        ),
                        external_progress_token=(
                            self._restore_external_progress_token(
                                obligation,
                                required_bytes=required_bytes,
                                closure_fingerprint=(
                                    eligible.bundle.generation_fingerprint
                                ),
                                command=command,
                            )
                        ),
                        wake_conditions=queued.wake_conditions,
                    )
                    return
                return
            h2d_blockers = {
                blocker.code.value
                for preview in previews
                for blocker in preview.blockers
            }
            capacity_previews = tuple(
                preview
                for preview in previews
                if (
                    preview.eligible
                    and preview.copy_bytes
                    + obligation.required_admission_bytes
                    > available
                )
                or (
                    preview.blockers
                    and {
                        blocker.code
                        for blocker in preview.blockers
                    }.issubset({TransferBlockerCode.DEVICE_CAPACITY})
                )
            )
            if not capacity_previews:
                if (
                    not previews
                    and obligation.cause
                    == RestoreObligationCause.ORDINARY_WAITING_PREFIX
                ):
                    if not self._begin_physical_transactional_commit(
                        obligation.context_id
                    ):
                        self._block_restore_obligation(
                            obligation,
                            now_ms=now_ms,
                            stamp=stamp,
                            blocker_codes=(
                                "native_ownership_readset_changed",
                            ),
                            wake_conditions=("safe_point_epoch_advanced",),
                        )
                        return
                    ready = self._activate_native_restore_fallback(
                        obligation,
                        req,
                        now_ms=now_ms,
                    )
                    if ready:
                        return
                    self._restart_physical_capture_epoch()
                    queued, blockers = self._try_queue_restore_lease_funding(
                        obligation,
                        now_ms=now_ms,
                        attempt_stamp=stamp,
                        h2d_bytes=0,
                    )
                    if queued:
                        return
                    self._block_restore_obligation(
                        obligation,
                        now_ms=now_ms,
                        stamp=stamp,
                        blocker_codes=blockers,
                    )
                    return
                self._block_restore_obligation(
                    obligation,
                    now_ms=now_ms,
                    stamp=stamp,
                    blocker_codes=tuple(
                        sorted(h2d_blockers or {"physical_preview_unavailable"})
                    ),
                )
                continue
            required_h2d_bytes = max(
                item.copy_bytes + obligation.required_admission_bytes
                for item in capacity_previews
            )
            deficit = max(0, required_h2d_bytes - available)
            funding, funding_blockers = self._restore_funding_preview(
                obligation,
                now_ms=now_ms,
                deficit_bytes=max(1, deficit),
            )
            if funding is not None:
                queued = self._queue_restore_obligation_command(
                    obligation,
                    funding,
                    now_ms=now_ms,
                    reason="restore_obligation_funding",
                    attempt_stamp=stamp,
                    funding_target_bytes=max(1, deficit),
                )
                if queued:
                    return
                self._block_restore_obligation(
                    obligation,
                    now_ms=now_ms,
                    stamp=stamp,
                    blocker_codes=("restore_funding_queue_conflict",),
                )
                continue
            self._block_restore_obligation(
                obligation,
                now_ms=now_ms,
                stamp=stamp,
                blocker_codes=tuple(
                    sorted(h2d_blockers.union(funding_blockers))
                ),
            )

    def _advance_restore_obligations(
        self,
        acks: tuple[CommandAck, ...] | list[CommandAck],
        *,
        now_ms: float,
    ) -> None:
        mapping = getattr(self, "_restore_command_to_request", {})
        if not mapping:
            return
        index = self._restore_obligation_index()
        for ack in acks:
            request_ids = mapping.pop(ack.command_id, None)
            funding_targets = getattr(
                self, "_restore_funding_target_by_command", {}
            ).pop(ack.command_id, {})
            if not request_ids:
                continue
            for request_id in tuple(sorted(request_ids)):
                funding_target_bytes = funding_targets.get(request_id, 0)
                obligation = index.get(request_id)
                if obligation is None or obligation.state.terminal:
                    continue
                command_kind = obligation.clear_command()
                transaction = self._ensure_restore_transaction(obligation)
                operation = next(
                    (
                        item
                        for item in reversed(transaction.physical_operations)
                        if item.canonical_command_id == ack.command_id
                        and item.terminal_status is None
                    ),
                    None,
                )
                if operation is not None:
                    operation.terminal_status = ack.status.value
                if ack.status == CommandStatus.COMPLETED:
                    if command_kind == CommandKind.PREFETCH_CONTEXT:
                        obligation.restored_bytes += ack.actual_bytes
                        obligation.state = RestoreObligationState.RESTORE_ACKED
                        lease = self._restore_lease_index().get(request_id)
                        if lease is not None and not lease.state.terminal:
                            lease.mark_restored()
                        if (
                            bool(getattr(self.config, "restore_lease_enabled", True))
                            and (
                                req := self._restore_waiting_request(request_id)
                            ) is not None
                            and not self._pin_restore_lease_prefix(
                                obligation, req, now_ms=now_ms
                            )
                        ):
                            self._release_restore_lease(
                                request_id,
                                now_ms=now_ms,
                                reason="restore_prefix_pin_after_h2d_failed",
                                rollback=True,
                            )
                            obligation.block(
                                blocker_codes=("restore_prefix_pin_failed",),
                                blocker_fingerprint="restore_prefix_pin_failed",
                                now_ms=now_ms,
                            )
                            transaction.stage = RestoreTransactionStage.WAIT_EVENT
                            continue
                        transaction.stage = (
                            RestoreTransactionStage.RESTORED_RESERVED
                        )
                    else:
                        obligation.funding_reclaim_bytes += ack.actual_bytes
                        self._reserve_restore_funding_capacity(
                            obligation,
                            target_bytes=funding_target_bytes,
                            reclaimed_bytes=ack.actual_bytes,
                            now_ms=now_ms,
                        )
                        obligation.state = RestoreObligationState.PARKED_WAIT
                        transaction.stage = RestoreTransactionStage.WAIT_FEASIBILITY
                    obligation.last_attempt_stamp = None
                    obligation.last_external_progress_token = None
                    obligation.blocker_codes = ()
                    obligation.blocker_fingerprint = None
                    obligation.last_progress_ts_ms = now_ms
                    outcome = "completed"
                else:
                    if command_kind == CommandKind.PREFETCH_CONTEXT:
                        self._release_restore_lease(
                            request_id,
                            now_ms=now_ms,
                            reason=f"restore_h2d_{ack.status.value}",
                            rollback=True,
                        )
                    blocker_codes = tuple(
                        sorted(
                            {item.code.value for item in ack.blockers}
                            or {ack.status.value}
                        )
                    )
                    obligation.block(
                        blocker_codes=blocker_codes,
                        blocker_fingerprint="|".join(blocker_codes),
                        now_ms=now_ms,
                    )
                    transaction.stage = RestoreTransactionStage.WAIT_EVENT
                    outcome = ack.status.value
                self._restore_obligation_counts[f"command_{outcome}"] += 1
                self.audit.emit(
                    "restore_obligation_command_terminal",
                    now_ms,
                    obligation_id=obligation.obligation_id,
                    request_id=obligation.request_id,
                    command_id=ack.command_id,
                    subscriber_count=len(request_ids),
                    command_kind=(
                        command_kind.value if command_kind is not None else None
                    ),
                    status=ack.status.value,
                    actual_bytes=ack.actual_bytes,
                    state=obligation.state.value,
                    blocker_codes=[item.code.value for item in ack.blockers],
                    reason=ack.reason,
                    funding_target_bytes=funding_target_bytes,
                    funding_reserved_tokens=obligation.funding_reserved_tokens,
                    funding_reserved_bytes=obligation.funding_reserved_bytes,
                )

    def _allocator_available_bytes(self) -> int:
        allocator = self.scheduler.token_to_kv_pool_allocator
        if hasattr(allocator, "full_available_size"):
            available_tokens = min(
                int(allocator.full_available_size()),
                int(allocator.swa_available_size()),
            )
        else:
            available_tokens = int(allocator.available_size())
        return max(0, available_tokens * self.config.kv_bytes_per_token)

    def _joint_retraction_planner(self) -> ObservedRetractionPlanner:
        planner = getattr(self, "_joint_retraction_solver", None)
        if planner is None:
            planner = ObservedRetractionPlanner(
                ObservedRetractionConfig(
                    minimum_admission_stall_ms=(
                        self.config.running_batch_retraction_min_stall_ms
                    ),
                    minimum_reclaim_bytes=(
                        self.config.running_batch_retraction_min_reclaim_bytes
                    ),
                    maximum_retractions_per_request=(
                        self.config.running_batch_retraction_max_per_request
                    ),
                )
            )
            self._joint_retraction_solver = planner
        return planner

    def _update_restore_micro_gate(
        self,
        stage: str,
        *,
        now_ms: float,
        **fields: object,
    ) -> None:
        if not self.config.restore_micro_gate_enabled:
            return
        state = getattr(self, "_restore_micro_gate_state", None)
        if state is None:
            state = {
                "enabled": True,
                "gate_id": self.config.restore_micro_gate_id,
            }
            self._restore_micro_gate_state = state
        state.update(fields)
        state["stage"] = stage
        state["last_update_ts_ms"] = now_ms
        signature = (
            stage,
            state.get("victim_request_id"),
            state.get("replacement_request_id"),
            state.get("transaction_id"),
            state.get("obligation_id"),
            state.get("reason"),
        )
        if signature == getattr(
            self, "_restore_micro_gate_last_audit_signature", None
        ):
            return
        self._restore_micro_gate_last_audit_signature = signature
        self.audit.emit(
            "restore_micro_gate_state",
            now_ms,
            **dict(state),
        )

    def _restore_micro_gate_barrier_pair_visible(
        self,
        requests: tuple[Any, ...],
        replacements: tuple[RetractionReplacement, ...],
    ) -> bool:
        """Request one overlap drain only for the configured deterministic pair."""

        if not self.config.restore_micro_gate_enabled:
            return False
        stage = str(self._restore_micro_gate_state.get("stage", "armed"))
        if stage not in {"armed", "waiting_for_pair", "pair_eligible"}:
            return False
        ledger = getattr(self, "_lock_service_ledger", None)
        victim_visible = any(
            (metadata := self._metadata(req)) is not None
            and metadata.root_workflow_id
            == self.config.restore_micro_gate_victim_workflow_id
            and ledger is not None
            and (record := ledger.progress_record(str(getattr(req, "rid", ""))))
            is not None
            and record.completed_service_count > 0
            and self._estimated_request_private_kv_bytes(req)
            >= self.config.restore_micro_gate_min_private_bytes
            for req in requests
        )
        if not victim_visible:
            return False
        metadata_by_id = getattr(self, "_request_metadata_by_id", {})
        return any(
            (metadata := metadata_by_id.get(item.request_id)) is not None
            and metadata.root_workflow_id
            == self.config.restore_micro_gate_replacement_workflow_id
            for item in replacements
        )

    def _estimated_request_private_kv_bytes(self, req: Any) -> int:
        """Return the scheduler-visible private suffix lower bound for a request."""

        page_size = max(
            1,
            int(
                getattr(
                    getattr(getattr(self, "scheduler", None), "server_args", None),
                    "page_size",
                    1,
                )
                or 1
            ),
        )
        prefix_tokens = _sequence_length(getattr(req, "prefix_indices", None))
        sequence_tokens = max(
            int(getattr(req, "seqlen", 0) or 0),
            _sequence_length(getattr(req, "origin_input_ids", None))
            + _sequence_length(getattr(req, "output_ids", None)),
        )
        last_uncached_pos = (prefix_tokens // page_size) * page_size
        return (
            max(0, sequence_tokens - last_uncached_pos)
            * self.config.kv_bytes_per_token
        )

    def _restore_micro_gate_snapshot(
        self,
        snapshot: ObservedRetractionSnapshot,
    ) -> tuple[ObservedRetractionSnapshot, str | None]:
        """Select one explicit test pair without relaxing physical safety."""

        if not self.config.restore_micro_gate_enabled:
            return snapshot, None
        stage = str(self._restore_micro_gate_state.get("stage", "armed"))
        if stage not in {"armed", "waiting_for_pair", "pair_eligible"}:
            return snapshot, None
        ledger = getattr(self, "_lock_service_ledger", None)
        victims = tuple(
            item
            for item in snapshot.candidates
            if item.workflow_id
            == self.config.restore_micro_gate_victim_workflow_id
            and item.private_kv_bytes
            >= self.config.restore_micro_gate_min_private_bytes
            and item.policy_eligible
            and ledger is not None
            and (
                record := ledger.progress_record(item.request_id)
            )
            is not None
            and record.completed_service_count > 0
        )
        replacements = tuple(
            item
            for item in snapshot.replacements
            if (
                metadata := getattr(self, "_request_metadata_by_id", {}).get(
                    item.request_id
                )
            )
            is not None
            and metadata.root_workflow_id
            == self.config.restore_micro_gate_replacement_workflow_id
        )
        now_ms = snapshot.observed_ts_ms
        if not victims or not replacements:
            self._update_restore_micro_gate(
                "waiting_for_pair",
                now_ms=now_ms,
                reason=(
                    "victim_not_physically_eligible"
                    if not victims
                    else "replacement_not_waiting"
                ),
                eligible_victim_count=len(victims),
                waiting_replacement_count=len(replacements),
            )
            return snapshot, None
        victim = min(victims, key=lambda item: item.request_id)
        replacement = min(replacements, key=lambda item: item.request_id)
        forced = replace(
            snapshot,
            active_kv_footprint_bytes=max(
                snapshot.active_kv_footprint_bytes,
                snapshot.active_kv_budget_bytes
                + victim.private_kv_bytes
                + self.config.running_batch_retraction_min_reclaim_bytes,
            ),
            admission_stall_ms=max(
                snapshot.admission_stall_ms,
                self.config.running_batch_retraction_min_stall_ms,
            ),
            candidates=(
                replace(
                    victim,
                    service_status="stale",
                    stale_for_ms=max(
                        victim.stale_for_ms,
                        self.config.running_batch_retraction_min_stall_ms,
                    ),
                ),
            ),
            replacements=(replacement,),
        )
        self._update_restore_micro_gate(
            "pair_eligible",
            now_ms=now_ms,
            reason="explicit_test_pair_observed",
            victim_request_id=victim.request_id,
            replacement_request_id=replacement.request_id,
            victim_private_bytes=victim.private_kv_bytes,
        )
        return forced, self.config.restore_micro_gate_id

    def _restore_micro_gate_decision(
        self,
        snapshot: ObservedRetractionSnapshot,
        gate_id: str,
    ) -> tuple[ObservedRetractionDecision, str]:
        replacement_ids = tuple(
            item.request_id for item in snapshot.replacements
        )
        victim_id = snapshot.candidates[0].request_id
        self._online_joint_epoch_sequence = (
            getattr(self, "_online_joint_epoch_sequence", 0) + 1
        )
        seed = compile_bounded_seed_epoch(
            ordered_request_ids=replacement_ids,
            visible_request_ids=replacement_ids,
            epoch_sequence=self._online_joint_epoch_sequence,
            emergency=True,
        )
        assert seed.view is not None and seed.epoch is not None
        retraction_slice = ActionSlice(
            slice_id=f"retraction:{victim_id}",
            kind="retraction",
            action_key=victim_id,
            dependency_keys=tuple(
                f"request:{request_id}" for request_id in replacement_ids
            ),
            committed=True,
        )
        epoch = append_committed_action_slice(seed.epoch, retraction_slice)
        self._current_online_joint_view = seed.view
        self._current_online_joint_decision = OnlineJointPlanDecision(
            seed.view, "applicable", epoch
        )
        self._current_joint_plan_epoch = epoch
        decision = self._joint_retraction_planner().decide(snapshot)
        if decision.plan is not None:
            decision = replace(
                decision,
                plan=replace(
                    decision.plan,
                    reason="restore_micro_gate_forced",
                ),
            )
        self._update_restore_micro_gate(
            "plan_selected" if decision.plan is not None else "pair_eligible",
            now_ms=snapshot.observed_ts_ms,
            reason=(
                "joint_retraction_plan_created"
                if decision.plan is not None
                else decision.reason
            ),
            source_joint_plan_id=epoch.source_plan_id,
            joint_epoch_id=epoch.epoch_id,
            gate_id=gate_id,
        )
        self.audit.emit(
            "online_joint_retraction_intent_committed",
            snapshot.observed_ts_ms,
            plan_id=epoch.source_plan_id,
            epoch_id=epoch.epoch_id,
            request_id=victim_id,
            intent_mode="deterministic_restore_micro_gate",
            source_joint_plan_id=epoch.source_plan_id,
            restore_micro_gate_id=gate_id,
        )
        return decision, epoch.source_plan_id

    def _running_retraction_decision(
        self,
        snapshot: ObservedRetractionSnapshot,
        *,
        restore_micro_gate_id: str | None = None,
    ) -> tuple[ObservedRetractionDecision, str | None]:
        if restore_micro_gate_id is not None:
            return self._restore_micro_gate_decision(
                snapshot, restore_micro_gate_id
            )
        if not self.config.joint_policy_enabled:
            return self._joint_retraction_planner().decide(snapshot), None

        view = getattr(self, "_current_online_joint_view", None)
        epoch = getattr(self, "_current_joint_plan_epoch", None)
        result = getattr(self, "_online_joint_result", None)
        if view is None:
            self._online_joint_epoch_sequence = (
                getattr(self, "_online_joint_epoch_sequence", 0) + 1
            )
            seed = compile_bounded_seed_epoch(
                ordered_request_ids=(
                    item.request_id for item in snapshot.replacements
                ),
                visible_request_ids=(
                    item.request_id for item in snapshot.replacements
                ),
                epoch_sequence=self._online_joint_epoch_sequence,
                emergency=True,
            )
            assert seed.view is not None and seed.epoch is not None
            view = seed.view
            epoch = seed.epoch
            self._current_online_joint_view = view
            self._current_online_joint_decision = seed
            self._current_joint_plan_epoch = epoch
            counts = getattr(self, "_online_joint_counts", None)
            if counts is not None:
                counts["emergency_retraction_seed"] += 1
            audit = getattr(self, "audit", None)
            if audit is not None:
                audit.emit(
                    "online_joint_epoch_committed",
                    snapshot.observed_ts_ms,
                    epoch_id=epoch.epoch_id,
                    plan_id=epoch.source_plan_id,
                    planner_mode=epoch.planner_mode.value,
                    commit_reason=seed.reason,
                    source_action_count=epoch.source_action_count,
                    committed_action_count=epoch.committed_action_count,
                    actionable_coverage=epoch.actionable_coverage,
                    rejected_slices=[],
                    trigger="running_retraction_emergency",
                )
        source_plan_id = (
            epoch.source_plan_id
            if epoch is not None and epoch.source_plan_id == view.plan_id
            else view.plan_id
        )

        candidate_ids = {item.request_id for item in snapshot.candidates}
        explicit_retractions = (
            tuple(
                item.request_id
                for item in getattr(result.plan, "retractions", ())
                if item.request_id in candidate_ids
            )
            if result is not None
            and result.plan is not None
            and result.plan.plan_id == view.plan_id
            else ()
        )
        if explicit_retractions:
            selected_victim = explicit_retractions[0]
            intent_mode = "optimized"
        elif epoch is not None and epoch.planner_mode in {
            JointPlannerMode.BOUNDED_SEED,
            JointPlannerMode.EMERGENCY,
        }:
            selected_victim = self._bounded_retraction_victim(snapshot)
            intent_mode = "emergency"
        else:
            selected_victim = None
            intent_mode = "optimized_no_intent"
        pause_authorized = (
            {selected_victim} if selected_victim is not None else set()
        )
        if not pause_authorized:
            return (
                ObservedRetractionDecision(
                    plan=None,
                    reason="joint_plan_has_no_pause_authorized_running_request",
                    candidate_count=len(snapshot.candidates),
                ),
                source_plan_id,
            )
        assert epoch is not None
        slice_id = f"retraction:{selected_victim}"
        if all(item.slice_id != slice_id for item in epoch.action_slices):
            retraction_slice = ActionSlice(
                slice_id=slice_id,
                kind="retraction",
                action_key=str(selected_victim),
                dependency_keys=tuple(
                    f"request:{item}" for item in view.immediate_request_ids
                ),
                committed=True,
            )
            epoch = append_committed_action_slice(epoch, retraction_slice)
            self._current_joint_plan_epoch = epoch
            self.audit.emit(
                "online_joint_retraction_intent_committed",
                snapshot.observed_ts_ms,
                plan_id=source_plan_id,
                epoch_id=epoch.epoch_id,
                request_id=selected_victim,
                intent_mode=intent_mode,
                source_joint_plan_id=source_plan_id,
            )
        restricted = replace(
            snapshot,
            candidates=tuple(
                replace(
                    candidate,
                    service_status=(
                        "stale"
                        if candidate.request_id in pause_authorized
                        else candidate.service_status
                    ),
                    policy_eligible=(
                        candidate.policy_eligible
                        and candidate.request_id in pause_authorized
                    ),
                )
                for candidate in snapshot.candidates
            ),
        )
        decision = self._joint_retraction_planner().decide(restricted)
        if decision.plan is not None:
            decision = replace(
                decision,
                plan=replace(
                    decision.plan,
                    reason="observed_joint_pause_authorized_lock_reclaim",
                ),
            )
        return decision, source_plan_id

    def _bounded_retraction_victim(
        self,
        snapshot: ObservedRetractionSnapshot,
    ) -> str | None:
        unlock_bytes: Counter[str] = Counter()
        for extent in snapshot.locked_extents:
            if not extent.fully_attributed or len(extent.blocker_request_ids) != 1:
                continue
            unlock_bytes[extent.blocker_request_ids[0]] += extent.size_bytes
        eligible = [
            item
            for item in snapshot.candidates
            if item.policy_eligible
            and item.prior_retraction_count
            < self.config.running_batch_retraction_max_per_request
        ]
        if not eligible:
            return None
        service_rank = {"stale": 0, "unknown": 1, "warming": 2, "recent": 3}
        selected = min(
            eligible,
            key=lambda item: (
                service_rank[item.service_status],
                -unlock_bytes[item.request_id],
                -item.private_kv_bytes,
                -item.causal_rank,
                -item.workflow_fair_rank,
                item.prior_retraction_count,
                item.request_id,
            ),
        )
        return selected.request_id

    def _running_retraction_replacements(
        self,
        *,
        now_ms: float,
    ) -> tuple[RetractionReplacement, ...]:
        waiting = tuple(getattr(self.scheduler, "waiting_queue", ()) or ())
        tagged: list[tuple[int, Any, BeliefKVRequestMetadata]] = []
        for native_index, req in enumerate(waiting):
            metadata = self._metadata(req)
            if metadata is None:
                continue
            entry = self.controller.visible_admission.get(str(req.rid))
            if entry is None or entry.state != AdmissionSideState.VISIBLE_PENDING:
                continue
            tagged.append((native_index, req, metadata))
        if not tagged:
            return ()
        view = getattr(self, "_current_online_joint_view", None)
        if self.config.joint_policy_enabled and view is not None:
            tagged_by_request = {
                str(req.rid): (native_index, req, metadata)
                for native_index, req, metadata in tagged
            }
            return tuple(
                RetractionReplacement(
                    request_id=request_id,
                    estimated_incremental_bytes=(
                        self.controller.visible_admission.get(
                            request_id
                        ).request.estimated_incremental_bytes
                    ),
                )
                for request_id in view.immediate_request_ids
                if request_id in tagged_by_request
            )
        workflow_ids = {item[2].root_workflow_id for item in tagged}
        fair_order = self.controller.fairness.ordered(
            workflow_ids,
            memory_charges=self.controller.workflow_memory_charges(),
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
        )
        fair_rank = {
            workflow_id: index for index, workflow_id in enumerate(fair_order)
        }
        active_workflow_ids = frozenset(
            fair_order[: self.config.joint_workflow_active_window]
        )
        frontier_rank = {
            workflow_id: {
                item.invocation_id: index
                for index, item in enumerate(
                    self.controller.frontier.candidates(workflow_id)
                )
            }
            for workflow_id in workflow_ids
        }
        ordered = sorted(
            (
                item
                for item in tagged
                if item[2].root_workflow_id in active_workflow_ids
            ),
            key=lambda item: self._observed_ticket_order_key(
                item[0],
                item[1],
                item[2],
                now_ms,
                fair_rank=fair_rank,
                frontier_rank=frontier_rank,
            ),
        )
        return tuple(
            RetractionReplacement(
                request_id=str(req.rid),
                estimated_incremental_bytes=(
                    self.controller.visible_admission.get(
                        str(req.rid)
                    ).request.estimated_incremental_bytes
                ),
            )
            for _, req, _ in ordered
        )

    def _native_reclaim_capacity_bytes(self) -> int:
        allocator = self.scheduler.token_to_kv_pool_allocator
        if hasattr(allocator, "full_available_size"):
            tokens = min(
                int(allocator.full_available_size())
                + int(self.tree_cache.full_evictable_size()),
                int(allocator.swa_available_size())
                + int(self.tree_cache.swa_evictable_size()),
            )
        else:
            tokens = int(allocator.available_size()) + int(
                self.tree_cache.evictable_size()
            )
        return max(0, tokens * self.config.kv_bytes_per_token)

    def scheduler_step(self) -> None:
        step_started_ns = time.perf_counter_ns()
        telemetry_overhead_ns = 0
        self._begin_physical_safe_point_apply_events()
        # Close the previous GPU service interval before freezing fairness state.
        self._charge_previous_batch(self._now_ms())
        if self.event_server is not None:
            for delivery in self.event_server.drain():
                self.audit.emit(
                    "runtime_event_delivery",
                    self._now_ms(),
                    message_id=delivery.message_id,
                    event_count=delivery.event_count,
                    accepted=delivery.accepted,
                    duplicate=delivery.duplicate,
                    error=delivery.error,
                )
        # ACKs are applied before the Radix mirror observes physical mutations.
        # This preserves prepare/commit/drop state-machine ordering. Transfer
        # completion is slow-moving (100ms+), so the backend poll is gated to
        # a bounded interval instead of running at event-loop rate.
        ack_now_ms = float(self._now_ms())
        last_ack_poll_ms = getattr(self, "_last_ack_poll_ms", None)
        if (
            last_ack_poll_ms is None
            or ack_now_ms - last_ack_poll_ms >= _ACK_POLL_INTERVAL_MS
        ):
            self._last_ack_poll_ms = ack_now_ms
            acks = self.bridge.drain_acks()
        else:
            acks = ()
        retired_h2d = self._retire_h2d_commands(acks)
        drain_telemetry = getattr(self.bridge, "drain_transfer_telemetry", None)
        telemetry_started_ns = time.perf_counter_ns()
        telemetry = tuple(drain_telemetry()) if drain_telemetry is not None else ()
        telemetry_overhead_ns += time.perf_counter_ns() - telemetry_started_ns
        for ack in acks:
            self.audit.emit(
                "transfer_acknowledged",
                self._now_ms(),
                command_id=ack.command_id,
                status=ack.status.value,
                actual_bytes=ack.actual_bytes,
                page_count=len(ack.page_handles),
                reason=ack.reason,
                blocker_codes=sorted(
                    {item.code.value for item in ack.blockers}
                ),
                blockers=self._audit_blockers(ack.blockers),
            )
            if ack.status in {CommandStatus.PARTIAL, CommandStatus.REJECTED}:
                self._mark_full_tree_rebuild()
        poll_callback_errors = getattr(
            getattr(self, "backend", None), "poll_callback_errors", None
        )
        if callable(poll_callback_errors):
            for error in poll_callback_errors():
                self.audit.emit(
                    "hicache_callback_error",
                    self._now_ms(),
                    **error,
                )
        if telemetry:
            telemetry_started_ns = time.perf_counter_ns()
            for observation in telemetry:
                self._emit_transfer_telemetry(observation)
            telemetry_overhead_ns += time.perf_counter_ns() - telemetry_started_ns
        self.sync_tree()
        self._flush_request_physical_finishes()
        self._report_allocator_usage()
        self._advance_retraction_transaction(
            acks,
            now_ms=float(self._now_ms()),
        )
        self._advance_online_joint_residency(
            acks,
            now_ms=float(self._now_ms()),
        )
        self._advance_restore_obligations(
            acks,
            now_ms=float(self._now_ms()),
        )
        self._advance_restore_authority(now_ms=float(self._now_ms()))
        self._enforce_queue_timeouts(now_ms=float(self._now_ms()))
        self._enforce_execution_timeouts(now_ms=float(self._now_ms()))
        for context_id, bundle_ids, status in retired_h2d:
            if status == CommandStatus.COMPLETED:
                self._release_h2d_waiters(context_id)
            else:
                self._invalidate_h2d_waiters(
                    context_id,
                    bundle_ids=bundle_ids,
                    status=status,
                )
        self._begin_physical_safe_point_capture_and_plan()
        if hasattr(self, "_last_resource_telemetry_ms"):
            self._emit_resource_snapshot(force=bool(acks or telemetry))
        policy_snapshot_log = getattr(self, "policy_snapshot_log", None)
        policy_check_now_ms = float(self._now_ms())
        last_policy_check_ms = getattr(self, "_last_policy_check_ms", None)
        if (
            last_policy_check_ms is None
            or policy_check_now_ms - last_policy_check_ms
            >= _POLICY_CHECK_INTERVAL_MS
        ):
            self._last_policy_check_ms = policy_check_now_ms
            if (
                policy_snapshot_log is not None
                and policy_snapshot_log.enabled
            ) or getattr(self, "joint_shadow_worker", None) is not None:
                self._maybe_record_policy_snapshot(
                    self._runtime_resource_observation()
                )
        restore_authority_mode = getattr(
            self, "_restore_authority_mode", RestoreAuthorityMode.NORMAL_JOINT
        )
        if restore_authority_mode != RestoreAuthorityMode.NORMAL_JOINT:
            online_joint_decision = OnlineJointPlanDecision(
                None, f"restore_authority:{restore_authority_mode.value}"
            )
        elif getattr(
            getattr(self, "config", None), "joint_policy_enabled", False
        ):
            online_joint_decision = self._online_joint_admission_decision(
                now_ms=float(self._now_ms()),
            )
            if online_joint_decision.view is None:
                online_joint_decision = self._safe_point_seed_decision(
                    now_ms=float(self._now_ms())
                )
        else:
            online_joint_decision = OnlineJointPlanDecision(None, "disabled")
        self._current_online_joint_view = online_joint_decision.view
        self._current_online_joint_decision = online_joint_decision
        self._drive_restore_obligations(now_ms=float(self._now_ms()))
        self._advance_restore_authority(now_ms=float(self._now_ms()))
        restore_authority_mode = getattr(
            self, "_restore_authority_mode", RestoreAuthorityMode.NORMAL_JOINT
        )
        if (
            restore_authority_mode == RestoreAuthorityMode.NORMAL_JOINT
            and online_joint_decision.view is not None
        ):
            self._queue_online_joint_residency(
                online_joint_decision.view,
                now_ms=float(self._now_ms()),
            )
        joint_policy_enabled = bool(
            getattr(
                getattr(self, "config", None),
                "joint_policy_enabled",
                False,
            )
        )
        online_joint_has_authority = bool(
            restore_authority_mode != RestoreAuthorityMode.NORMAL_JOINT
            or joint_policy_enabled
            and (
                online_joint_decision.view is not None
                or getattr(self, "_pending_online_joint_residency", None)
                is not None
            )
        )
        if joint_policy_enabled and not online_joint_has_authority:
            self._online_joint_counts["safe_point_missing_joint_authority"] += 1
        self._begin_physical_transactional_commit()
        tick = self.bridge.scheduler_step(
            self._now_ms(),
            drain_acks=False,
            allow_reactive_transfer=(
                not joint_policy_enabled
                and restore_authority_mode == RestoreAuthorityMode.NORMAL_JOINT
                and bool(
                    getattr(
                        getattr(self, "config", None),
                        "reactive_transfer_enabled",
                        True,
                    )
                )
            ),
        )
        for guard_event in getattr(tick, "transfer_guard_events", ()):
            self.audit.emit(
                guard_event.kind,
                guard_event.ts_ms,
                **dict(guard_event.fields),
            )
        for bundle_event in getattr(tick, "bundle_preview_events", ()):
            self.audit.emit(
                bundle_event.kind,
                bundle_event.ts_ms,
                **dict(bundle_event.fields),
            )
        for shadow_event in getattr(tick, "frontier_shadow_events", ()):
            self.audit.emit(
                "frontier_shadow",
                shadow_event.ts_ms,
                context_id=shadow_event.context_id,
                invocation_id=shadow_event.invocation_id,
                workflow_id=shadow_event.workflow_id,
                state=shadow_event.state,
                agent_definition_id=shadow_event.agent_definition_id,
                support_level=shadow_event.support_level,
                ood_reasons=list(shadow_event.ood_reasons),
                boundary_top=shadow_event.boundary_top,
                boundary_distribution=dict(shadow_event.boundary_distribution),
                remaining_decode_tokens_p50=(
                    shadow_event.remaining_decode_tokens_p50
                ),
                remaining_external_wait_ms_p50=(
                    shadow_event.remaining_external_wait_ms_p50
                ),
                prompt_growth_tokens_p50=(
                    shadow_event.prompt_growth_tokens_p50
                ),
                next_output_tokens_p50=shadow_event.next_output_tokens_p50,
                feature_source=shadow_event.feature_source,
            )
        decision = tick.admission
        if decision is not None:
            audit_state = (decision.request_id, decision.admitted, decision.reason)
            if audit_state != self._last_admission_audit:
                self.audit.emit(
                    "admission_decision",
                    tick.now_ms,
                    request_id=decision.request_id,
                    admitted=decision.admitted,
                    reason=decision.reason,
                    reserved_bytes=decision.reserved_bytes,
                    required_bytes=decision.required_bytes,
                    native_reclaim_capacity_bytes=(
                        decision.native_reclaim_capacity_bytes
                    ),
                    actual_hbm_used_bytes=self.controller.actual_hbm_used_bytes,
                )
                self._last_admission_audit = audit_state
        else:
            self._last_admission_audit = None
        if tick.transfer is not None:
            if (
                tick.transfer.command.kind == CommandKind.PREFETCH_CONTEXT
                and tick.transfer.command.context_id is not None
            ):
                context_id = tick.transfer.command.context_id
                command_id = tick.transfer.command.command_id
                h2d_commands = getattr(self, "_h2d_context_by_command", None)
                if h2d_commands is None:
                    h2d_commands = {}
                    self._h2d_context_by_command = h2d_commands
                pending_contexts = getattr(self, "_pending_h2d_contexts", None)
                if pending_contexts is None:
                    pending_contexts = set()
                    self._pending_h2d_contexts = pending_contexts
                bundle = tick.transfer.command.physical_bundle
                restored_extent_ids = tuple(
                    sorted(
                        f"page:{item.handle.page_id}:"
                        f"{item.handle.allocation_generation}"
                        for item in tick.transfer.page_actions
                        if item.action == PhysicalPageAction.START_H2D
                    )
                )
                if not restored_extent_ids and bundle is not None:
                    restored_extent_ids = tuple(
                        sorted(
                            f"page:{handle.page_id}:"
                            f"{handle.allocation_generation}"
                            for handle in bundle.closure_handles
                        )
                    )
                if not restored_extent_ids:
                    restored_extent_ids = self._context_restore_bundle_ids(
                        context_id
                    )
                h2d_commands[command_id] = (
                    context_id,
                    restored_extent_ids,
                )
                pending_contexts.add(context_id)
                self._mark_h2d_waiters(
                    context_id,
                    restored_extent_ids=restored_extent_ids,
                    reason="h2d_inflight",
                )
            action_counts: dict[str, int] = {}
            for page_action in tick.transfer.page_actions:
                action = page_action.action.value
                action_counts[action] = action_counts.get(action, 0) + 1
            source_joint_plan_id = tick.transfer.command.metadata.get(
                "joint_plan_id"
            )
            policy_reason = tick.transfer.command.metadata.get("reason")
            lifecycle_action = str(policy_reason).startswith(
                ("terminal_", "shutdown_", "host_capacity_")
            )
            action_source = (
                "joint_plan"
                if source_joint_plan_id
                else "lifecycle"
                if lifecycle_action
                else "unified_liveness"
            )
            if (
                getattr(
                    getattr(self, "config", None),
                    "joint_policy_enabled",
                    False,
                )
                and action_source == "unified_liveness"
            ):
                self._online_joint_counts[
                    "online_action_missing_source_joint_plan_id"
                ] += 1
            self.audit.emit(
                "transfer_dispatched",
                tick.now_ms,
                command_id=tick.transfer.command.command_id,
                kind=tick.transfer.command.kind.value,
                context_id=tick.transfer.command.context_id,
                context_epoch=tick.transfer.command.context_epoch,
                selected_bytes=tick.transfer.resolved_bytes,
                page_count=len(tick.transfer.page_actions),
                action_counts=action_counts,
                policy_reason=policy_reason,
                action_source=action_source,
                source_joint_plan_id=source_joint_plan_id,
                bundle_scope=tick.transfer.command.metadata.get(
                    "physical_bundle_scope"
                ),
                exclusive_action_bytes=tick.transfer.command.metadata.get(
                    "physical_exclusive_action_bytes", 0
                ),
                cross_context_action_bytes=tick.transfer.command.metadata.get(
                    "physical_cross_context_action_bytes", 0
                ),
                foreign_owner_context_ids=tick.transfer.command.metadata.get(
                    "physical_foreign_owner_context_ids", []
                ),
                closure_fingerprint=tick.transfer.closure_fingerprint,
                bundle_id=(
                    tick.transfer.command.physical_bundle.bundle_id
                    if tick.transfer.command.physical_bundle is not None
                    else ""
                ),
                expected_reclaimable_bytes=(
                    tick.transfer.command.physical_bundle.expected_reclaimable_bytes
                    if tick.transfer.command.physical_bundle is not None
                    else 0
                ),
            )
        for ack in tick.local_acks:
            self.audit.emit(
                "transfer_rejected_local",
                tick.now_ms,
                command_id=ack.command_id,
                status=ack.status.value,
                reason=ack.reason,
                blocker_codes=sorted(
                    {item.code.value for item in ack.blockers}
                ),
                blockers=self._audit_blockers(ack.blockers),
            )
        if tick.local_acks:
            self._advance_retraction_transaction(
                tuple(tick.local_acks),
                now_ms=float(self._now_ms()),
            )
            self._advance_online_joint_residency(
                tuple(tick.local_acks),
                now_ms=float(self._now_ms()),
            )
            self._advance_restore_obligations(
                tuple(tick.local_acks),
                now_ms=float(self._now_ms()),
            )
        for command_id in getattr(tick, "stalled_command_ids", ()):
            if command_id in self._stalled_command_audited:
                continue
            self._stalled_command_audited.add(command_id)
            self.audit.emit(
                "transfer_watchdog_expired",
                tick.now_ms,
                command_id=command_id,
                action="await_nonpreemptible_transfer_completion",
            )
        self._finish_physical_safe_point()
        self._record_scheduler_timing(
            total_ms=(time.perf_counter_ns() - step_started_ns) / 1_000_000.0,
            telemetry_ms=telemetry_overhead_ns / 1_000_000.0,
            telemetry_count=len(telemetry),
        )

    def _retire_h2d_commands(
        self, acks: tuple[CommandAck, ...]
    ) -> tuple[tuple[str, tuple[str, ...], CommandStatus], ...]:
        command_contexts = getattr(self, "_h2d_context_by_command", None)
        if command_contexts is None:
            return ()
        pending_contexts = getattr(self, "_pending_h2d_contexts", set())
        retired: list[tuple[str, tuple[str, ...], CommandStatus]] = []
        for ack in acks:
            command_state = command_contexts.pop(ack.command_id, None)
            if command_state is None:
                continue
            context_id, bundle_ids = command_state
            if context_id not in {
                item[0] for item in command_contexts.values()
            }:
                pending_contexts.discard(context_id)
            retired.append((context_id, bundle_ids, ack.status))
        return tuple(retired)

    def _context_has_engine_request(self, context_id: str) -> bool:
        phase = getattr(
            self, "_safe_point_physical_phase", SafePointPhysicalPhase.IDLE
        )
        if phase == SafePointPhysicalPhase.TRANSACTIONAL_COMMIT:
            # Backend submission is commit-time validation, never a new decision.
            snapshot = self._capture_safe_point_physical_snapshot(
                epoch=getattr(self, "_safe_point_physical_epoch_sequence", 0),
                record_metrics=False,
            )
            self._native_physical_snapshot_counts[
                "commit_validation_capture_count"
            ] += 1
            snapshots = snapshot.for_context(context_id)
            explicit_transfers = snapshot.explicit_transfers_by_context.get(
                context_id, ()
            )
        else:
            snapshot = self._lazy_safe_point_physical_snapshot()
            snapshots = snapshot.for_context(context_id)
            explicit_transfers = snapshot.explicit_transfers_by_context.get(
                context_id, ()
            )
        return any(
            snapshot.engine_owned
            or snapshot.native_load_operation_id is not None
            or bool(snapshot.explicit_transfer_ids)
            for snapshot in snapshots
        ) or bool(explicit_transfers)

    def _begin_physical_safe_point_apply_events(self) -> None:
        if not hasattr(self, "_native_physical_snapshot_counts"):
            self._native_physical_snapshot_counts = Counter()
        self._safe_point_physical_epoch_sequence = (
            getattr(self, "_safe_point_physical_epoch_sequence", 0) + 1
        )
        self._safe_point_physical_phase = SafePointPhysicalPhase.APPLY_EVENTS
        self._safe_point_physical_snapshot = None

    def _begin_physical_safe_point_capture_and_plan(self) -> None:
        phase = getattr(
            self, "_safe_point_physical_phase", SafePointPhysicalPhase.IDLE
        )
        if phase == SafePointPhysicalPhase.IDLE:
            self._begin_physical_safe_point_apply_events()
        elif phase == SafePointPhysicalPhase.TRANSACTIONAL_COMMIT:
            self._safe_point_physical_epoch_sequence += 1
            self._safe_point_physical_snapshot = None
        self._safe_point_physical_phase = SafePointPhysicalPhase.CAPTURE_AND_PLAN

    def _restart_physical_capture_epoch(self) -> None:
        """Start a new read epoch after a failed or completed commit attempt."""

        self._safe_point_physical_epoch_sequence = (
            getattr(self, "_safe_point_physical_epoch_sequence", 0) + 1
        )
        self._safe_point_physical_phase = SafePointPhysicalPhase.CAPTURE_AND_PLAN
        self._safe_point_physical_snapshot = None
        self._native_physical_snapshot_counts["capture_epoch_restarted"] += 1

    def _begin_physical_transactional_commit(
        self, context_id: str | None = None
    ) -> bool:
        """Revalidate a captured read-set and close the epoch before mutation."""

        phase = getattr(
            self, "_safe_point_physical_phase", SafePointPhysicalPhase.IDLE
        )
        if phase == SafePointPhysicalPhase.IDLE:
            self._begin_physical_safe_point_apply_events()
            self._begin_physical_safe_point_capture_and_plan()
            phase = SafePointPhysicalPhase.CAPTURE_AND_PLAN
        elif phase == SafePointPhysicalPhase.APPLY_EVENTS:
            self._begin_physical_safe_point_capture_and_plan()
            phase = SafePointPhysicalPhase.CAPTURE_AND_PLAN
        if phase == SafePointPhysicalPhase.TRANSACTIONAL_COMMIT:
            return True

        captured = getattr(self, "_safe_point_physical_snapshot", None)
        if captured is not None and context_id is not None:
            started_ns = time.perf_counter_ns()
            current = self._capture_safe_point_physical_snapshot(
                epoch=captured.epoch,
                record_metrics=False,
            )
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            self._native_physical_snapshot_counts[
                "commit_readset_validation_count"
            ] += 1
            self._native_physical_snapshot_counts[
                "commit_readset_validation_us_total"
            ] += int(round(elapsed_ms * 1000.0))
            if captured.context_readset(context_id) != current.context_readset(
                context_id
            ):
                self._native_physical_snapshot_counts[
                    "commit_readset_stale"
                ] += 1
                audit = getattr(self, "audit", None)
                if audit is not None:
                    audit.emit(
                        "physical_snapshot_commit_rejected",
                        self._now_ms(),
                        epoch=captured.epoch,
                        context_id=context_id,
                        reason="native_ownership_readset_changed",
                    )
                self._restart_physical_capture_epoch()
                return False

        self._safe_point_physical_phase = (
            SafePointPhysicalPhase.TRANSACTIONAL_COMMIT
        )
        self._safe_point_physical_snapshot = None
        self._native_physical_snapshot_counts["transactional_commit_entered"] += 1
        return True

    def _finish_physical_safe_point(self) -> None:
        self._safe_point_physical_phase = SafePointPhysicalPhase.IDLE
        self._safe_point_physical_snapshot = None

    @staticmethod
    def _gc_collection_count() -> int:
        return sum(int(item.get("collections", 0)) for item in gc.get_stats())

    def _capture_safe_point_physical_snapshot(
        self,
        *,
        epoch: int,
        record_metrics: bool,
    ) -> SafePointPhysicalSnapshot:
        """Capture native ownership once; callers decide whether it is cached."""

        total_started_ns = time.perf_counter_ns()
        gc_before = self._gc_collection_count()
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            empty_timing = SafePointSnapshotBuildTiming(
                total_ms=0.0,
                queue_collection_ms=0.0,
                metadata_indexing_ms=0.0,
                radix_ownership_lookup_ms=0.0,
                operation_indexing_ms=0.0,
                sorting_allocation_ms=0.0,
            )
            return SafePointPhysicalSnapshot(
                epoch=epoch,
                records=(),
                by_request={},
                by_context={},
                explicit_transfers_by_context={},
                queue_record_count=0,
                metadata_record_count=0,
                timing=empty_timing,
            )

        phase_started_ns = time.perf_counter_ns()
        waiting = tuple(getattr(scheduler, "waiting_queue", ()) or ())
        running = tuple(
            getattr(getattr(scheduler, "running_batch", None), "reqs", ()) or ()
        )
        chunked = getattr(scheduler, "chunked_req", None)
        locations: dict[str, tuple[Any, NativeQueueLocation]] = {}
        for req in waiting:
            locations[str(getattr(req, "rid", f"object:{id(req)}"))] = (
                req,
                NativeQueueLocation.WAITING,
            )
        for req in running:
            locations[str(getattr(req, "rid", f"object:{id(req)}"))] = (
                req,
                NativeQueueLocation.RUNNING,
            )
        if chunked is not None:
            locations[str(getattr(chunked, "rid", f"object:{id(chunked)}"))] = (
                chunked,
                NativeQueueLocation.CHUNKED,
            )
        queue_record_count = len(waiting) + len(running) + int(chunked is not None)
        queue_collection_ms = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0

        phase_started_ns = time.perf_counter_ns()
        request_metadata = getattr(self, "_request_metadata_by_id", {})
        metadata_items = tuple(sorted(request_metadata.items()))
        metadata_indexing_ms = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0

        phase_started_ns = time.perf_counter_ns()
        explicit_by_context: dict[str, list[str]] = {}
        for command_id, state in getattr(
            self, "_h2d_context_by_command", {}
        ).items():
            explicit_by_context.setdefault(str(state[0]), []).append(
                str(command_id)
            )
        explicit_transfers = {
            context_id: tuple(sorted(command_ids))
            for context_id, command_ids in explicit_by_context.items()
        }
        native_load_by_request: dict[str, str] = {}
        for request_id, (req, _location) in locations.items():
            for field_name in (
                "hicache_load_operation_id",
                "load_operation_id",
                "current_load_operation_id",
            ):
                value = getattr(req, field_name, None)
                if value is not None:
                    native_load_by_request[request_id] = str(value)
                    break
        operation_indexing_ms = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0

        phase_started_ns = time.perf_counter_ns()
        snapshots: list[NativeRequestPhysicalSnapshot] = []
        submitted_by_id = getattr(self, "_request_submitted_ts_by_id", {})
        terminal_cancelled = getattr(
            self, "_terminal_cancelled_request_ids", set()
        )
        has_controller = hasattr(self, "controller")
        for request_id, metadata in metadata_items:
            req, location = locations.get(
                request_id, (None, NativeQueueLocation.NONE)
            )
            slot = getattr(req, "req_pool_idx", None) if req is not None else None
            if slot is not None:
                try:
                    slot = int(slot)
                except (TypeError, ValueError):
                    slot = None
                if slot is not None and slot < 0:
                    slot = None
            node = getattr(req, "last_node", None) if req is not None else None
            lock_owned = bool(
                location
                in {NativeQueueLocation.RUNNING, NativeQueueLocation.CHUNKED}
                and node is not None
                and int(getattr(node, "lock_ref", 0) or 0) > 0
            )
            native_load_id = native_load_by_request.get(request_id)
            submitted = submitted_by_id.get(request_id, 0.0)
            generation = max(
                0,
                (int(metadata.context_epoch) << 48)
                + int(max(0.0, float(submitted)) * 1000.0),
            )
            snapshots.append(
                NativeRequestPhysicalSnapshot(
                    request_id=request_id,
                    context_id=metadata.context_id,
                    queue_location=location,
                    req_pool_slot=slot,
                    radix_lock_owned=lock_owned,
                    native_load_operation_id=native_load_id,
                    explicit_transfer_ids=explicit_transfers.get(
                        metadata.context_id, ()
                    ),
                    request_generation=generation,
                    terminal=(
                        request_id in terminal_cancelled
                        or has_controller
                        and self._metadata_scope_is_terminal(metadata)
                    ),
                )
            )
        radix_ownership_lookup_ms = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0

        phase_started_ns = time.perf_counter_ns()
        records = tuple(sorted(snapshots, key=lambda item: item.request_id))
        by_request = {item.request_id: item for item in records}
        mutable_by_context: dict[str, list[NativeRequestPhysicalSnapshot]] = {}
        for item in records:
            mutable_by_context.setdefault(item.context_id, []).append(item)
        by_context = {
            context_id: tuple(items)
            for context_id, items in mutable_by_context.items()
        }
        counts = getattr(self, "_native_physical_snapshot_counts", None)
        if counts is None:
            counts = Counter()
            self._native_physical_snapshot_counts = counts
        placeholder_timing = SafePointSnapshotBuildTiming(
            total_ms=0.0,
            queue_collection_ms=queue_collection_ms,
            metadata_indexing_ms=metadata_indexing_ms,
            radix_ownership_lookup_ms=radix_ownership_lookup_ms,
            operation_indexing_ms=operation_indexing_ms,
            sorting_allocation_ms=0.0,
            queue_record_count=queue_record_count,
            metadata_record_count=len(metadata_items),
            matched_record_count=len(records),
            cold_build=counts.get("call_count", 0) == 0,
        )
        snapshot = SafePointPhysicalSnapshot(
            epoch=epoch,
            records=records,
            by_request=by_request,
            by_context=by_context,
            explicit_transfers_by_context=explicit_transfers,
            queue_record_count=queue_record_count,
            metadata_record_count=len(metadata_items),
            timing=placeholder_timing,
        )
        sorting_allocation_ms = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0
        total_ms = (time.perf_counter_ns() - total_started_ns) / 1_000_000.0
        timing = SafePointSnapshotBuildTiming(
            total_ms=total_ms,
            queue_collection_ms=queue_collection_ms,
            metadata_indexing_ms=metadata_indexing_ms,
            radix_ownership_lookup_ms=radix_ownership_lookup_ms,
            operation_indexing_ms=operation_indexing_ms,
            sorting_allocation_ms=sorting_allocation_ms,
            queue_record_count=queue_record_count,
            metadata_record_count=len(metadata_items),
            matched_record_count=len(records),
            gc_collections=max(0, self._gc_collection_count() - gc_before),
            cold_build=counts.get("call_count", 0) == 0,
        )
        object.__setattr__(snapshot, "timing", timing)
        if record_metrics:
            samples = getattr(
                self, "_native_physical_snapshot_timing_samples", None
            )
            if samples is None:
                samples = deque(maxlen=65_536)
                self._native_physical_snapshot_timing_samples = samples
            samples.append(timing)
            counts["call_count"] += 1
            counts["queue_records_scanned_total"] += queue_record_count
            counts["metadata_records_scanned_total"] += len(metadata_items)
            counts["matched_snapshot_count_total"] += len(records)
            counts["build_us_total"] += int(round(total_ms * 1000.0))
            counts["gc_collections_during_build"] += timing.gc_collections
        return snapshot

    def _lazy_safe_point_physical_snapshot(self) -> SafePointPhysicalSnapshot:
        phase = getattr(
            self, "_safe_point_physical_phase", SafePointPhysicalPhase.IDLE
        )
        if phase == SafePointPhysicalPhase.IDLE:
            self._begin_physical_safe_point_apply_events()
            self._begin_physical_safe_point_capture_and_plan()
        elif phase == SafePointPhysicalPhase.APPLY_EVENTS:
            self._begin_physical_safe_point_capture_and_plan()
        elif phase == SafePointPhysicalPhase.TRANSACTIONAL_COMMIT:
            raise RuntimeError(
                "cannot build a planning snapshot after transactional commit"
            )
        snapshot = getattr(self, "_safe_point_physical_snapshot", None)
        epoch = getattr(self, "_safe_point_physical_epoch_sequence", 0)
        if snapshot is None:
            snapshot = self._capture_safe_point_physical_snapshot(
                epoch=epoch,
                record_metrics=True,
            )
            self._safe_point_physical_snapshot = snapshot
        else:
            self._native_physical_snapshot_counts["cache_hit_count"] += 1
        return snapshot

    def _context_physical_snapshots(
        self, context_id: str
    ) -> tuple[NativeRequestPhysicalSnapshot, ...]:
        """Return one context from the epoch-local immutable snapshot."""

        return self._lazy_safe_point_physical_snapshot().for_context(context_id)

    def _record_scheduler_timing(
        self, *, total_ms: float, telemetry_ms: float, telemetry_count: int
    ) -> None:
        samples = getattr(self, "_scheduler_timing_samples", None)
        if samples is None:
            samples = deque(maxlen=65_536)
            self._scheduler_timing_samples = samples
        samples.append((total_ms, telemetry_ms, telemetry_count))
        counts = getattr(self, "_native_physical_snapshot_counts", None)
        if counts is None:
            counts = Counter()
            self._native_physical_snapshot_counts = counts
        counts["scheduler_step_count"] += 1

    @staticmethod
    def _audit_blockers(
        blockers: tuple[TransferBlocker, ...],
    ) -> list[dict[str, object]]:
        return [
            {
                "code": blocker.code.value,
                "page_handle": (
                    {
                        "page_id": blocker.page_handle.page_id,
                        "allocation_generation": (
                            blocker.page_handle.allocation_generation
                        ),
                    }
                    if blocker.page_handle is not None
                    else None
                ),
                "required_bytes": blocker.required_bytes,
            }
            for blocker in blockers
        ]

    def _emit_controller_timing_summary(self) -> None:
        samples = tuple(getattr(self, "_scheduler_timing_samples", ()))
        ticket_samples = getattr(self, "_ticket_timing_samples", {})
        tree_samples = tuple(getattr(self, "_tree_sync_timing_samples", ()))
        if not samples and not any(ticket_samples.values()) and not tree_samples:
            return
        event_samples = [item for item in samples if item[2] > 0]

        def distribution(values: tuple[float, ...]) -> dict[str, float | int]:
            return {
                "count": len(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "max": max(values, default=0.0),
            }

        def ratio(item: tuple[float, float, int]) -> float:
            return item[1] / max(item[0], 1e-12)

        self.audit.emit(
            "controller_timing_summary",
            self._now_ms(),
            sample_window=65_536,
            scheduler_step_sample_count=len(samples),
            scheduler_step_p99_ms=percentile([item[0] for item in samples], 99),
            telemetry_overhead_p99_ms=percentile(
                [item[1] for item in samples], 99
            ),
            telemetry_overhead_ratio_p99=percentile(
                [ratio(item) for item in samples], 99
            ),
            telemetry_event_step_count=len(event_samples),
            telemetry_event_count=sum(item[2] for item in event_samples),
            telemetry_event_step_p99_ms=percentile(
                [item[0] for item in event_samples], 99
            ),
            telemetry_event_overhead_p99_ms=percentile(
                [item[1] for item in event_samples], 99
            ),
            telemetry_event_overhead_ratio_p99=percentile(
                [ratio(item) for item in event_samples], 99
            ),
            admission_ticket_timing_ms={
                name: distribution(tuple(values))
                for name, values in sorted(ticket_samples.items())
            },
            radix_sync_timing_ms={
                mode: distribution(
                    tuple(item[1] for item in tree_samples if item[0] == mode)
                )
                for mode in ("incremental", "full")
            },
        )

    def on_abort_request(self, abort_request: Any) -> int:
        """Drop side state while SGLang removes requests from native queues."""

        abort_all = bool(getattr(abort_request, "abort_all", False))
        rid_prefix = str(getattr(abort_request, "rid", ""))

        def matches(request_id: str) -> bool:
            return abort_all or request_id.startswith(rid_prefix)

        removed = 0
        visible_index = self.controller.visible_admission
        for entry in tuple(visible_index.entries()):
            request_id = entry.request.request_id
            if not matches(request_id):
                continue
            if visible_index.cancel(request_id) is not None:
                removed += 1
        # Keep the legacy simulator index clean if a test/runtime adapter used it.
        for request_id in tuple(
            item.request_id for item in self.controller.admission.pending_requests()
        ):
            if matches(request_id):
                self.controller.admission.cancel(request_id)
        for request_id in tuple(self._request_metadata_by_id):
            if matches(request_id):
                terminal_reason = (
                    "queue_timeout"
                    if request_id
                    in getattr(self, "_queue_timeout_request_ids", set())
                    else "execution_timeout"
                    if request_id
                    in getattr(self, "_execution_timeout_request_ids", set())
                    else "request_aborted"
                )
                self._finish_restore_obligation(
                    request_id,
                    RestoreObligationState.CANCELLED,
                    now_ms=float(
                        getattr(self, "_now_ms", lambda: 0.0)()
                    ),
                    reason=terminal_reason,
                )
                self._cancel_restore_service_grace(
                    request_id,
                    now_ms=float(getattr(self, "_now_ms", lambda: 0.0)()),
                    reason=terminal_reason,
                )
                self._request_metadata_by_id.pop(request_id, None)
                ledger = getattr(self, "_lock_service_ledger", None)
                if ledger is not None:
                    ledger.forget(str(request_id))
                getattr(self, "_request_submitted_ts_by_id", {}).pop(
                    request_id, None
                )
                start = getattr(self, "_request_physical_start_by_id", {}).pop(
                    request_id, None
                )
                if start is not None:
                    aborted_starts = getattr(
                        self, "_aborted_request_physical_start_by_id", None
                    )
                    if aborted_starts is None:
                        aborted_starts = {}
                        self._aborted_request_physical_start_by_id = aborted_starts
                    aborted_starts[request_id] = {
                        **start,
                        "abort_ts_ms": float(
                            getattr(self, "_now_ms", lambda: 0.0)()
                        ),
                        "abort_reason": terminal_reason,
                    }
                getattr(self, "_queue_timeout_request_ids", set()).discard(
                    request_id
                )
                getattr(
                    self, "_execution_timeout_request_ids", set()
                ).discard(request_id)
                getattr(self, "_retracted_engine_request_ids", set()).discard(
                    request_id
                )
                getattr(self, "_pending_selective_retraction_ids", set()).discard(
                    request_id
                )
                getattr(self, "_retraction_cooldown_until_by_request", {}).pop(
                    request_id, None
                )
                getattr(
                    self, "_ordinary_restore_capacity_waiters", set()
                ).discard(request_id)
        return removed

    def on_batch_selected(self, batch: Any) -> None:
        now_ms = self._now_ms()
        self._charge_previous_batch(now_ms)
        if batch is None or not getattr(batch, "reqs", None):
            self._last_batch_selected_ms = now_ms
            self._last_batch_workflow_counts = {}
            return
        self._observe_gpu_batch_launch(batch, now_ms)
        workflow_counts: dict[str, int] = {}
        for req in batch.reqs:
            metadata = self._metadata(req)
            if metadata is None:
                continue
            if self._metadata_scope_is_terminal(metadata):
                self.controller.visible_admission.cancel(str(req.rid))
                self.controller.admission.cancel(req.rid)
                self._terminal_cancelled_request_ids.add(req.rid)
                req.to_abort = True
                self.audit.emit(
                    "terminal_request_selected",
                    now_ms,
                    request_id=req.rid,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                )
                continue
            self._observe_request_selected_for_lock_service(
                req,
                metadata,
                now_ms=now_ms,
            )
            obligation = self._restore_obligation_index().get(str(req.rid))
            if obligation is not None and not obligation.state.terminal:
                self._start_restore_service_grace(
                    obligation, now_ms=now_ms, req=req
                )
            self._finish_restore_obligation(
                str(req.rid),
                RestoreObligationState.SATISFIED,
                now_ms=now_ms,
                reason="gpu_service_resumed",
            )
            workflow_counts[metadata.root_workflow_id] = (
                workflow_counts.get(metadata.root_workflow_id, 0) + 1
            )
            getattr(self, "_retracted_engine_request_ids", set()).discard(
                str(req.rid)
            )
            self.controller.visible_admission.cancel(str(req.rid))
            if req.rid not in self._active_request_ids:
                self._active_request_ids.add(req.rid)
                self._capture_request_physical_start(req, metadata, now_ms)
                self.audit.emit(
                    "request_started",
                    now_ms,
                    request_id=req.rid,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    prompt_tokens=len(req.origin_input_ids),
                    cache_hit_tokens=len(getattr(req, "prefix_indices", ())),
                )
                self._emit(
                    metadata,
                    RuntimeEventKind.LLM_SUBMIT,
                    attributes={
                        "request_id": req.rid,
                        "prompt_tokens": len(req.origin_input_ids),
                        "cache_hit_tokens": len(getattr(req, "prefix_indices", ())),
                        "expected_output_tokens": int(
                            getattr(req.sampling_params, "max_new_tokens", 0) or 0
                        ),
                        "context_tokens": len(getattr(req, "fill_ids", ())),
                        "model": str(
                            getattr(self.scheduler.model_config, "model_path", "unknown")
                        ),
                    },
                )
            last_node = getattr(req, "last_node", None)
            if last_node is not None:
                self._terminal_node_by_context[metadata.context_id] = last_node
                self._mark_context_dirty(metadata.context_id)
                self._tree_dirty = True
        self._last_batch_selected_ms = now_ms
        self._last_batch_workflow_counts = workflow_counts

    def _observe_gpu_batch_launch(self, batch: Any, now_ms: float) -> None:
        if not self.config.queue_service_observer_enabled:
            return
        observer_started_ns = time.perf_counter_ns()
        descriptor: dict[str, Any] | None = None
        reqs = tuple(getattr(batch, "reqs", ()) or ())
        metadata = tuple(self._metadata(req) for req in reqs)
        mode = getattr(batch, "forward_mode", None)
        mode_name = str(getattr(mode, "name", "unknown")).lower()
        phase: str | None = None
        tokens = 0
        if mode is not None and mode.is_decode():
            phase = "decode"
            steps = max(
                1,
                int(
                    getattr(
                        getattr(self.scheduler, "server_args", None),
                        "num_continuous_decode_steps",
                        1,
                    )
                    or 1
                ),
            )
            tokens = len(reqs) * steps
        elif mode is not None and mode.is_extend() and not mode.is_mixed():
            phase = "prefill"
            value = getattr(batch, "extend_num_tokens", None)
            tokens = int(value) if value is not None else sum(
                max(0, int(getattr(req, "extend_input_len", 0) or 0))
                for req in reqs
            )
        all_tagged = bool(reqs) and all(item is not None for item in metadata)
        tags = {
            self._service_calibration_tag(item.root_workflow_id)
            for item in metadata
            if item is not None
        }
        calibration_tag = (
            next(iter(tags))
            if all_tagged and len(tags) == 1 and None not in tags
            else None
        )
        under_limit = (
            self._gpu_service_sample_count
            < self.config.queue_service_observer_max_samples
        )
        observation_scope: str | None = None
        split = "runtime"
        calibration_kind = phase
        episode_id: str | None = None
        prefill_chunk_index = None
        prefix_tokens_before = None
        if phase is not None and tokens > 0 and under_limit and all_tagged:
            if calibration_tag is not None:
                split, calibration_kind, episode_id = calibration_tag
                if phase == calibration_kind:
                    observation_scope = "calibration"
                    if phase == "prefill" and len(reqs) == 1:
                        prefill_chunk_index = (
                            self._gpu_service_prefill_chunks_by_episode.get(
                                episode_id, 0
                            )
                        )
                        self._gpu_service_prefill_chunks_by_episode[episode_id] = (
                            prefill_chunk_index + 1
                        )
                    elif phase == "prefill":
                        observation_scope = None
            elif self.config.queue_service_observer_include_runtime_batches:
                observation_scope = "runtime"
        elif phase is not None and tokens > 0 and all_tagged and not under_limit:
            self._gpu_service_sample_cap_count = (
                getattr(self, "_gpu_service_sample_cap_count", 0) + 1
            )
        if observation_scope is not None:
            sequence_tokens_before = []
            for req in reqs:
                fill_ids = getattr(req, "fill_ids", None)
                if fill_ids is not None:
                    sequence_tokens = len(fill_ids)
                else:
                    origin = getattr(req, "origin_input_ids", ())
                    output = getattr(req, "output_ids", ())
                    sequence_tokens = (
                        len(origin) if origin is not None else 0
                    ) + (len(output) if output is not None else 0)
                if phase == "prefill":
                    extend_tokens = max(
                        0,
                        int(getattr(req, "extend_input_len", 0) or 0),
                    )
                    sequence_tokens = max(0, sequence_tokens - extend_tokens)
                sequence_tokens_before.append(sequence_tokens)
            if phase == "prefill" and len(reqs) == 1:
                prefix_tokens_before = sequence_tokens_before[0]
            self._gpu_service_sequence += 1
            sample_id = f"gpu-service-{self._gpu_service_sequence:09d}"
            request_samples = []
            decode_steps = max(1, tokens // max(1, len(reqs)))
            for req, item, sequence_tokens in zip(
                reqs, metadata, sequence_tokens_before
            ):
                assert item is not None
                output_ids = getattr(req, "output_ids", None)
                output_tokens_before = (
                    len(output_ids) if output_ids is not None else None
                )
                token_delta = (
                    decode_steps
                    if phase == "decode"
                    else max(
                        0,
                        int(getattr(req, "extend_input_len", 0) or 0),
                    )
                )
                request_samples.append(
                    {
                        "request_id": str(req.rid),
                        "workflow_id": item.root_workflow_id,
                        "invocation_id": item.invocation_id,
                        "context_id": item.context_id,
                        "context_epoch": item.context_epoch,
                        "phase": phase,
                        "token_delta": token_delta,
                        "token_delta_semantics": (
                            "scheduled_decode_steps"
                            if phase == "decode"
                            else "prefill_extend_input_len"
                        ),
                        "output_tokens_before": output_tokens_before,
                        "sequence_tokens_before": sequence_tokens,
                    }
                )
            descriptor = {
                "sample_id": sample_id,
                "observation_scope": observation_scope,
                "phase": phase,
                "tokens": tokens,
                "batch_size": len(reqs),
                "split": split,
                "calibration_kind": calibration_kind,
                "episode_id": episode_id or f"runtime:{sample_id}",
                "prefill_chunk_index": prefill_chunk_index,
                "prefix_tokens_before": prefix_tokens_before,
                "sequence_tokens_before": sequence_tokens_before,
                "max_sequence_tokens_before": max(
                    sequence_tokens_before, default=0
                ),
                "forward_mode": mode_name,
                "launch_ts_ms": now_ms,
                "workflow_ids": sorted(
                    {item.root_workflow_id for item in metadata if item is not None}
                ),
                "request_ids": sorted(str(req.rid) for req in reqs),
                "request_samples": request_samples,
            }
            descriptor["launch_observer_cpu_ms"] = (
                time.perf_counter_ns() - observer_started_ns
            ) / 1_000_000.0
        self._gpu_service_launches.append(descriptor)

    def on_batch_completed(self, batch: Any) -> None:
        mode = getattr(batch, "forward_mode", None)
        if mode is None or mode.is_idle() or mode.is_dummy_first():
            return
        now_ms = self._now_ms()
        phase = self._gpu_batch_phase(mode)
        self._observe_restore_service_grace(
            batch,
            now_ms=now_ms,
            phase=phase,
        )
        self._observe_request_service_completed(
            batch,
            now_ms=now_ms,
            phase=phase,
        )
        if not self.config.queue_service_observer_enabled:
            return
        previous_complete_ms = getattr(
            self, "_gpu_service_previous_completion_ms", None
        )
        self._gpu_service_previous_completion_ms = now_ms
        if not self._gpu_service_launches:
            self.audit.emit(
                "gpu_service_sample_failed",
                now_ms,
                error="completion has no launch descriptor",
                forward_mode=str(getattr(mode, "name", "unknown")).lower(),
            )
            return
        descriptor = self._gpu_service_launches.popleft()
        if descriptor is None:
            return
        observer_started_ns = time.perf_counter_ns()
        if descriptor["phase"] == "decode":
            current_output_tokens = {
                str(req.rid): len(output_ids)
                for req in tuple(getattr(batch, "reqs", ()) or ())
                if (output_ids := getattr(req, "output_ids", None)) is not None
            }
            for request_sample in descriptor["request_samples"]:
                before = request_sample.get("output_tokens_before")
                after = current_output_tokens.get(request_sample["request_id"])
                if before is None or after is None:
                    continue
                request_sample["token_delta"] = max(0, after - int(before))
                request_sample["token_delta_semantics"] = (
                    "observed_output_ids_delta"
                )
        launch_ts_ms = float(descriptor["launch_ts_ms"])
        service_start_ts_ms = max(
            launch_ts_ms,
            previous_complete_ms
            if previous_complete_ms is not None
            else launch_ts_ms,
        )
        service_elapsed_ms = now_ms - service_start_ts_ms
        launch_to_completion_ms = now_ms - launch_ts_ms
        if service_elapsed_ms <= 0 or launch_to_completion_ms <= 0:
            self.audit.emit(
                "gpu_service_sample_failed",
                now_ms,
                error="non-positive GPU service interval",
                sample_id=descriptor["sample_id"],
                service_elapsed_ms=service_elapsed_ms,
                launch_to_completion_ms=launch_to_completion_ms,
            )
            return
        self._gpu_service_sample_count += 1
        build_ms = (
            time.perf_counter_ns() - observer_started_ns
        ) / 1_000_000.0 + float(descriptor.get("launch_observer_cpu_ms", 0.0))
        enqueue_started_ns = time.perf_counter_ns()
        self.audit.emit(
            "gpu_service_sample",
            now_ms,
            **descriptor,
            complete_ts_ms=now_ms,
            service_start_ts_ms=service_start_ts_ms,
            service_elapsed_ms=service_elapsed_ms,
            launch_to_completion_ms=launch_to_completion_ms,
            elapsed_ms=service_elapsed_ms,
            timing_semantics_version="gpu_service_interval_v1",
            timing_semantics=(
                "max(scheduler launch, previous batch completion) to "
                "process_batch_result completion; excludes overlap queue and HTTP"
            ),
            prefill_caveat=(
                "extend timing includes result processing and first-token sampling"
                if descriptor["phase"] == "prefill"
                else None
            ),
        )
        enqueue_ms = (
            time.perf_counter_ns() - enqueue_started_ns
        ) / 1_000_000.0
        samples = getattr(self, "_gpu_service_observer_timing_samples", None)
        if samples is None:
            samples = deque(maxlen=65_536)
            self._gpu_service_observer_timing_samples = samples
        samples.append((build_ms, enqueue_ms))

    def _observe_request_selected_for_lock_service(
        self,
        req: Any,
        metadata: BeliefKVRequestMetadata,
        *,
        now_ms: float,
    ) -> None:
        ledger = getattr(self, "_lock_service_ledger", None)
        if ledger is None:
            ledger = RequestServiceLedger()
            self._lock_service_ledger = ledger
        try:
            ledger.observe_selected(
                request_id=str(req.rid),
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
                ts_ms=now_ms,
            )
        except Exception as error:
            self._record_lock_service_observer_error(
                error,
                phase="batch_selected",
                request_id=str(getattr(req, "rid", "unknown")),
                now_ms=now_ms,
            )

    def _observe_request_service_completed(
        self,
        batch: Any,
        *,
        now_ms: float,
        phase: str,
    ) -> None:
        ledger = getattr(self, "_lock_service_ledger", None)
        if ledger is None:
            return
        active_request_ids = getattr(self, "_active_request_ids", set())
        for req in tuple(getattr(batch, "reqs", ()) or ()):
            request_id = str(getattr(req, "rid", ""))
            if self._metadata(req) is None or not ledger.tracks(request_id):
                continue
            try:
                ledger.observe_completed(
                    request_id,
                    ts_ms=now_ms,
                    phase=phase,
                )
            except Exception as error:
                self._record_lock_service_observer_error(
                    error,
                    phase="batch_completed",
                    request_id=str(getattr(req, "rid", "unknown")),
                    now_ms=now_ms,
                )
            finally:
                if getattr(req, "rid", None) not in active_request_ids:
                    ledger.forget(request_id)

    def _record_lock_service_observer_error(
        self,
        error: Exception,
        *,
        phase: str,
        request_id: str,
        now_ms: float,
    ) -> None:
        self._lock_service_observer_error_count = (
            getattr(self, "_lock_service_observer_error_count", 0) + 1
        )
        self.audit.emit(
            "lock_service_observer_error",
            now_ms,
            phase=phase,
            request_id=request_id,
            error=f"{type(error).__name__}: {error}",
        )

    @staticmethod
    def _gpu_batch_phase(mode: Any) -> str:
        if mode.is_decode():
            return "decode"
        if mode.is_extend():
            return "mixed" if mode.is_mixed() else "prefill"
        return str(getattr(mode, "name", "other")).lower()

    @staticmethod
    def _service_calibration_tag(
        workflow_id: str,
    ) -> tuple[str, str, str] | None:
        prefix, separator, case_id = workflow_id.partition(":")
        if prefix != "service-calibration" or not separator:
            return None
        split, separator, case_id = case_id.partition(":")
        if split not in {"train", "holdout"} or not separator or not case_id:
            return None
        if case_id.startswith("prefill-"):
            calibration_kind = "prefill"
            episode = case_id
        elif case_id.startswith("decode-"):
            calibration_kind = "decode"
            episode, separator, request_index = case_id.rpartition("-i")
            if (
                not separator
                or not episode
                or not request_index.isdigit()
            ):
                return None
        else:
            return None
        return split, calibration_kind, f"{split}:{episode}"

    def _charge_previous_batch(self, now_ms: float) -> None:
        selected_ms = getattr(self, "_last_batch_selected_ms", None)
        workflow_counts = dict(
            getattr(self, "_last_batch_workflow_counts", {})
        )
        self._last_batch_selected_ms = None
        self._last_batch_workflow_counts = {}
        if selected_ms is None or not workflow_counts:
            return
        elapsed_ms = max(0.0, now_ms - selected_ms)
        total_reqs = sum(workflow_counts.values())
        if elapsed_ms == 0 or total_reqs == 0:
            return
        for workflow_id, count in workflow_counts.items():
            self.controller.fairness.charge_service(
                workflow_id, elapsed_ms * count / total_reqs
            )

    def begin_prefill_epoch(
        self,
        waiting_queue: list[Any],
        adder: Any,
        *,
        max_requests: int,
    ) -> tuple[Any, ...]:
        """Compile tickets and return an ordered view of the native queue."""

        if self._current_ticket_epoch is not None:
            self.end_prefill_epoch(())
        compile_started_ns = time.perf_counter_ns()
        self._admission_epoch += 1
        epoch = self._admission_epoch
        self._ticket_attempted_request_ids.clear()
        self._ticket_selected_request_ids.clear()
        self._ticket_skip_audit.clear()
        self._ticket_selection_details.clear()
        self._ticket_native_rejections.clear()
        now_ms = self._now_ms()
        self.controller.policy_control_state(now_ms)
        self._restore_bypass_request_id = self._select_restore_bypass_request(
            waiting_queue, now_ms=now_ms
        )

        tagged: list[tuple[int, Any, BeliefKVRequestMetadata]] = []
        for native_index, req in enumerate(waiting_queue):
            metadata = self._metadata(req)
            if metadata is None:
                continue
            request_id = str(req.rid)
            entry = self.controller.visible_admission.get(request_id)
            if entry is None:
                self.on_requests_requeued((req,), is_retracted=True)
            self._sync_visible_gate_state(request_id, metadata, req=req)
            tagged.append((native_index, req, metadata))

        workflow_ids = {item[2].root_workflow_id for item in tagged}
        fair_order = self.controller.fairness.ordered(
            workflow_ids,
            memory_charges=self.controller.workflow_memory_charges(),
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
        )
        active_workflow_ids = frozenset(
            fair_order[: self.config.joint_workflow_active_window]
        )
        fair_rank = {
            workflow_id: index for index, workflow_id in enumerate(fair_order)
        }
        frontier_candidates = {
            workflow_id: tuple(self.controller.frontier.candidates(workflow_id))
            for workflow_id in workflow_ids
        }
        frontier_rank = {
            workflow_id: {
                item.invocation_id: index
                for index, item in enumerate(frontier_candidates[workflow_id])
            }
            for workflow_id in workflow_ids
        }
        entries = {
            entry.request.request_id: entry
            for entry in self.controller.visible_admission.entries()
        }
        rem_input_tokens = max(0, int(getattr(adder, "rem_input_tokens", 0)))
        rem_chunk_tokens = getattr(adder, "rem_chunk_tokens", None)
        if rem_chunk_tokens is not None:
            rem_input_tokens = min(rem_input_tokens, max(0, int(rem_chunk_tokens)))
        native_hbm_tokens = max(0, int(getattr(adder, "rem_total_tokens", 0)))
        native_hbm_bytes = min(
            self.config.hbm_capacity_bytes,
            native_hbm_tokens * self.config.kv_bytes_per_token,
        )
        observed_window: ObservedAdmissionWindow | None = None
        observed_admission_error: str | None = None
        cached_online_decision = getattr(
            self, "_current_online_joint_decision", None
        )
        current_online_result = getattr(self, "_online_joint_result", None)
        if (
            cached_online_decision is not None
            and cached_online_decision.view is not None
            and current_online_result is not None
            and current_online_result.plan is not None
            and current_online_result.plan.plan_id
            == cached_online_decision.view.plan_id
        ):
            online_joint_decision = cached_online_decision
            self._online_joint_counts["safe_point_decision_reused"] += 1
        else:
            online_joint_decision = self._online_joint_admission_decision(
                now_ms=now_ms,
            )
        online_joint_view = online_joint_decision.view
        if online_joint_view is None and self.config.joint_policy_enabled:
            ordered_tagged = sorted(
                (
                    item
                    for item in tagged
                    if item[2].root_workflow_id in active_workflow_ids
                ),
                key=lambda item: self._observed_ticket_order_key(
                    item[0],
                    item[1],
                    item[2],
                    now_ms,
                    fair_rank=fair_rank,
                    frontier_rank=frontier_rank,
                ),
            )
            self._online_joint_epoch_sequence += 1
            online_joint_decision = compile_bounded_seed_epoch(
                ordered_request_ids=(
                    str(item[1].rid) for item in ordered_tagged
                ),
                visible_request_ids=(str(item[1].rid) for item in tagged),
                epoch_sequence=self._online_joint_epoch_sequence,
                emergency=(
                    self.controller.actual_hbm_used_bytes
                    / self.config.hbm_capacity_bytes
                    >= self.config.joint_emergency_hbm_ratio
                ),
                restore_requirements=(
                    (
                        request_id,
                        entry.restore_bundle_ids,
                    )
                    for request_id, entry in entries.items()
                    if entry.state == AdmissionSideState.WAIT_RESTORE
                ),
            )
            online_joint_view = online_joint_decision.view
            self._current_joint_plan_epoch = online_joint_decision.epoch
            self._online_joint_counts["bounded_seed_epoch"] += 1
            assert online_joint_decision.epoch is not None
            self.audit.emit(
                "online_joint_epoch_committed",
                now_ms,
                epoch_id=online_joint_decision.epoch.epoch_id,
                plan_id=online_joint_decision.epoch.source_plan_id,
                planner_mode=online_joint_decision.epoch.planner_mode.value,
                commit_reason=online_joint_decision.reason,
                source_action_count=(
                    online_joint_decision.epoch.source_action_count
                ),
                committed_action_count=(
                    online_joint_decision.epoch.committed_action_count
                ),
                actionable_coverage=(
                    online_joint_decision.epoch.actionable_coverage
                ),
                rejected_slices=[],
                trigger="safe_point_no_async_plan",
            )
        observed_admission_enabled = (
            self.config.observed_admission_scheduling_enabled
        )
        if online_joint_view is None and observed_admission_enabled:
            try:
                observed_candidates = self._observed_admission_candidates(
                    tagged,
                    entries=entries,
                    now_ms=now_ms,
                    fair_rank=fair_rank,
                    frontier_candidates=frontier_candidates,
                )
                observed_window = self._observed_admission_scheduler().decide(
                    observed_candidates,
                    self._observed_admission_snapshot(
                        native_available_hbm_bytes=native_hbm_bytes,
                        native_max_requests=max(0, int(max_requests)),
                    ),
                )
            except Exception as error:
                observed_admission_error = f"{type(error).__name__}: {error}"
                self.audit.emit(
                    "observed_admission_fallback",
                    now_ms,
                    epoch=epoch,
                    error=observed_admission_error,
                    fallback="observed_reactive",
                    blocker_semantics="visible_side_state_preserved",
        )
        if online_joint_view is not None:
            waiting_request_ids = {
                str(req.rid) for _, req, _ in tagged
            }
            ordered_request_ids = tuple(
                request_id
                for request_id in online_joint_view.immediate_request_ids
                if request_id in waiting_request_ids
                and request_id in entries
                and entries[request_id].request.workflow_id
                in active_workflow_ids
            )
            compile_max_requests = max(0, int(max_requests))
            compile_hbm_bytes = native_hbm_bytes
            current_epoch = getattr(self, "_current_joint_plan_epoch", None)
            planner_mode = (
                current_epoch.planner_mode.value
                if current_epoch is not None
                and current_epoch.source_plan_id == online_joint_view.plan_id
                else "optimized"
            )
            ticket_source = f"joint_{planner_mode}"
            ticket_reason = f"joint_plan:{online_joint_view.plan_id}"
            self._online_joint_counts["ticket_epoch"] += 1
        elif observed_window is not None:
            ordered_request_ids = observed_window.ordered_request_ids
            compile_max_requests = observed_window.max_new_requests
            compile_hbm_bytes = min(
                native_hbm_bytes,
                observed_window.active_growth_budget_bytes,
            )
            ticket_source = "observed_active_set"
            ticket_reason = observed_window.mode
            mode_counts = getattr(
                self, "_observed_admission_mode_counts", None
            )
            if mode_counts is None:
                mode_counts = Counter()
                self._observed_admission_mode_counts = mode_counts
            mode_counts[observed_window.mode] += 1
            self._observed_admission_peak_active_kv_bytes = max(
                getattr(self, "_observed_admission_peak_active_kv_bytes", 0),
                observed_window.active_kv_footprint_bytes,
            )
            self._observed_admission_peak_pressure = max(
                getattr(self, "_observed_admission_peak_pressure", 0.0),
                observed_window.active_kv_pressure,
            )
        else:
            ordered_tagged = sorted(
                tagged,
                key=lambda item: self._observed_ticket_order_key(
                    item[0],
                    item[1],
                    item[2],
                    now_ms,
                    fair_rank=fair_rank,
                    frontier_rank=frontier_rank,
                ),
            )
            ordered_request_ids = tuple(
                str(item[1].rid) for item in ordered_tagged
            )
            compile_max_requests = max(0, int(max_requests))
            compile_hbm_bytes = native_hbm_bytes
            ticket_source = (
                "observed_active_set_fallback"
                if observed_admission_enabled
                else "observed_reactive"
            )
            ticket_reason = (
                f"online_joint_fallback:{online_joint_decision.reason}"
                if self.config.joint_policy_enabled
                else "policy_error_reactive_fallback"
                if observed_admission_error is not None
                else "visible_bounded_fallback"
            )
        restore_ready_priority = self._restore_ready_ticket_priority(entries)
        if restore_ready_priority and self.config.joint_policy_enabled:
            base_order = (
                online_joint_view.ordered_request_ids
                if online_joint_view is not None
                else ordered_request_ids
            )
            priority_set = set(restore_ready_priority)
            restore_liveness_order = restore_ready_priority + tuple(
                request_id
                for request_id in base_order
                if request_id not in priority_set
            )
            self._online_joint_epoch_sequence += 1
            online_joint_decision = compile_bounded_seed_epoch(
                ordered_request_ids=restore_liveness_order,
                visible_request_ids=(str(item[1].rid) for item in tagged),
                epoch_sequence=self._online_joint_epoch_sequence,
                emergency=(
                    self.controller.actual_hbm_used_bytes
                    / self.config.hbm_capacity_bytes
                    >= self.config.joint_emergency_hbm_ratio
                ),
                restore_requirements=(
                    (request_id, entry.restore_bundle_ids)
                    for request_id, entry in entries.items()
                    if entry.state == AdmissionSideState.WAIT_RESTORE
                ),
            )
            assert online_joint_decision.view is not None
            assert online_joint_decision.epoch is not None
            online_joint_view = online_joint_decision.view
            self._current_joint_plan_epoch = online_joint_decision.epoch
            ordered_request_ids = online_joint_view.immediate_request_ids
            ticket_source = (
                f"joint_{online_joint_decision.epoch.planner_mode.value}"
                "+restore_liveness"
            )
            ticket_reason = (
                f"joint_plan:{online_joint_view.plan_id}:"
                "restore_obligation_ticket_ready"
            )
            self._online_joint_counts["restore_liveness_epoch"] += 1
            self.audit.emit(
                "online_joint_epoch_committed",
                now_ms,
                epoch_id=online_joint_decision.epoch.epoch_id,
                plan_id=online_joint_view.plan_id,
                planner_mode=online_joint_decision.epoch.planner_mode.value,
                commit_reason=online_joint_decision.reason,
                source_action_count=(
                    online_joint_decision.epoch.source_action_count
                ),
                committed_action_count=(
                    online_joint_decision.epoch.committed_action_count
                ),
                actionable_coverage=(
                    online_joint_decision.epoch.actionable_coverage
                ),
                rejected_slices=[],
                trigger="restore_ticket_liveness",
            )
        retraction_priority = tuple(
            request_id
            for request_id in getattr(
                self, "_retraction_priority_request_ids", ()
            )
            if request_id in entries
            and entries[request_id].state == AdmissionSideState.VISIBLE_PENDING
        )
        liveness_priority = tuple(
            dict.fromkeys((*restore_ready_priority, *retraction_priority))
        )
        if liveness_priority:
            priority_set = set(liveness_priority)
            ordered_request_ids = (
                liveness_priority
                + tuple(
                    request_id
                    for request_id in ordered_request_ids
                    if request_id not in priority_set
                )
            )
            if restore_ready_priority and not self.config.joint_policy_enabled:
                ticket_source = f"{ticket_source}+restore_liveness"
                ticket_reason = "restore_obligation_ticket_ready"
            elif not restore_ready_priority:
                ticket_source = f"{ticket_source}+retraction_replacement"
                ticket_reason = "retraction_reclaim_confirmed"
        restore_authority_mode = getattr(
            self, "_restore_authority_mode", RestoreAuthorityMode.NORMAL_JOINT
        )
        if restore_authority_mode != RestoreAuthorityMode.NORMAL_JOINT:
            restore_request_id = getattr(
                self, "_restore_authority_request_id", None
            )
            restore_entry = entries.get(restore_request_id)
            ordered_request_ids = (
                (restore_request_id,)
                if restore_request_id is not None
                and restore_entry is not None
                and restore_entry.state == AdmissionSideState.VISIBLE_PENDING
                else ()
            )
            compile_max_requests = 1 if ordered_request_ids else 0
            ticket_source = "emergency_restore_coordinator"
            ticket_reason = restore_authority_mode.value
        self._retraction_priority_request_ids = ()
        restore_lease_credits = self._restore_lease_credit_bytes()
        candidate_limit = min(
            len(ordered_request_ids),
            max(64, max(0, int(max_requests)) * 4),
        )
        ticket_epoch = self.controller.admission_ticket_compiler.compile(
            epoch=epoch,
            now_ms=now_ms,
            ordered_request_ids=ordered_request_ids,
            entries=entries,
            budget=AdmissionCompileBudget(
                max_prefill_tokens=rem_input_tokens,
                max_requests=compile_max_requests,
                max_candidates=candidate_limit,
                available_hbm_bytes=compile_hbm_bytes,
            ),
            source=ticket_source,
            reason=ticket_reason,
            reservation_credits=restore_lease_credits,
        )
        visible_pending = any(
            entry.state == AdmissionSideState.VISIBLE_PENDING
            for entry in entries.values()
        )
        if visible_pending and not ticket_epoch.tickets:
            if getattr(self, "_retraction_admission_stall_since_ms", None) is None:
                self._retraction_admission_stall_since_ms = now_ms
        elif not visible_pending:
            self._retraction_admission_stall_since_ms = None
        self._current_observed_admission_window = observed_window
        self._current_online_joint_view = online_joint_view
        self._current_online_joint_decision = online_joint_decision
        self._current_ticket_epoch = ticket_epoch
        self._current_tickets_by_request = dict(ticket_epoch.by_request_id)
        compile_ms = (
            time.perf_counter_ns() - compile_started_ns
        ) / 1_000_000.0
        self._record_ticket_timing("compile_ms", compile_ms)
        self.audit.emit(
            "admission_ticket_epoch_started",
            now_ms,
            epoch=epoch,
            native_waiting_count=len(waiting_queue),
            tagged_waiting_count=len(tagged),
            scanned_count=ticket_epoch.scanned_count,
            issued_count=len(ticket_epoch.tickets),
            skipped_count=len(ticket_epoch.skipped),
            compile_ms=compile_ms,
            source=ticket_epoch.source,
            reason=ticket_reason,
            max_requests=max_requests,
            policy_max_requests=compile_max_requests,
            max_prefill_tokens=rem_input_tokens,
            native_hbm_tokens=native_hbm_tokens,
            native_hbm_bytes=native_hbm_bytes,
            policy_hbm_bytes=compile_hbm_bytes,
            restore_lease_reserved_bytes=sum(restore_lease_credits.values()),
            active_budget_binding=(
                observed_window is not None
                and observed_window.active_growth_budget_bytes
                < native_hbm_bytes
            ),
            observed_admission_error=observed_admission_error,
            online_joint_plan_id=(
                online_joint_view.plan_id
                if online_joint_view is not None
                else None
            ),
            source_joint_plan_id=(
                online_joint_view.plan_id
                if online_joint_view is not None
                else None
            ),
            online_joint_decision=online_joint_decision.reason,
            online_joint_restore_blocked_count=(
                len(online_joint_view.restore_requirements)
                if online_joint_view is not None
                else 0
            ),
            workflow_active_window=self.config.joint_workflow_active_window,
            active_workflow_ids=sorted(active_workflow_ids),
            inactive_workflow_count=max(
                0, len(workflow_ids) - len(active_workflow_ids)
            ),
            observed_admission_window=(
                self._observed_admission_window_fields(observed_window)
                if observed_window is not None
                else None
            ),
            issued=[
                {
                    "request_id": ticket.request_id,
                    "workflow_id": ticket.workflow_id,
                    "invocation_id": ticket.invocation_id,
                    "context_id": ticket.context_id,
                    "rank": ticket.rank,
                    "estimated_prefill_tokens": ticket.estimated_prefill_tokens,
                    "estimated_incremental_bytes": (
                        ticket.estimated_incremental_bytes
                    ),
                    "reservation_credit_bytes": (
                        ticket.reservation_credit_bytes
                    ),
                }
                for ticket in ticket_epoch.tickets
            ],
            skipped_reason_counts=dict(
                sorted(Counter(reason for _, reason in ticket_epoch.skipped).items())
            ),
            skipped_non_blocker_details=[
                {"request_id": request_id, "reason": reason}
                for request_id, reason in ticket_epoch.skipped
                if reason not in {"wait_restore", "policy_blocked"}
            ],
        )

        rank_by_request = {
            ticket.request_id: ticket.rank for ticket in ticket_epoch.tickets
        }
        tagged_view = sorted(
            tagged,
            key=lambda item: (
                rank_by_request.get(str(item[1].rid), 1 << 30),
                item[0],
            ),
        )
        tagged_iter = iter(item[1] for item in tagged_view)
        return tuple(
            req if self._metadata(req) is None else next(tagged_iter)
            for req in waiting_queue
        )

    def admission_ticket_allows(self, req: Any) -> bool:
        started_ns = time.perf_counter_ns()
        try:
            return self._admission_ticket_allows(req)
        finally:
            self._record_ticket_timing(
                "validation_ms",
                (time.perf_counter_ns() - started_ns) / 1_000_000.0,
            )

    def _admission_ticket_allows(self, req: Any) -> bool:
        metadata = self._metadata(req)
        if metadata is None:
            return True
        ticket = self._ticket_for_request(str(req.rid))
        if ticket is None:
            self._record_ticket_skip(str(req.rid), "no_ticket")
            return False
        validation = self.controller.visible_admission.validate_ticket(
            ticket,
            epoch=self._admission_epoch,
        )
        if not validation.valid:
            self._record_ticket_skip(str(req.rid), "+".join(validation.reasons))
            return False
        return True

    def validate_admission_ticket_after_prefix(self, req: Any) -> bool:
        started_ns = time.perf_counter_ns()
        try:
            return self._validate_admission_ticket_after_prefix(req)
        finally:
            self._record_ticket_timing(
                "validation_ms",
                (time.perf_counter_ns() - started_ns) / 1_000_000.0,
            )

    def _validate_admission_ticket_after_prefix(self, req: Any) -> bool:
        metadata = self._metadata(req)
        if metadata is None:
            return True
        request_id = str(req.rid)
        ticket = self._ticket_for_request(request_id)
        if ticket is None:
            self._record_ticket_skip(request_id, "no_ticket_after_prefix")
            return False
        entry = self.controller.visible_admission.get(request_id)
        if entry is None:
            self._record_ticket_skip(request_id, "request_missing_after_prefix")
            return False
        origin_input_ids = getattr(req, "origin_input_ids", None)
        prefix_indices = getattr(req, "prefix_indices", None)
        uncached_prompt_tokens = max(
            0,
            _sequence_length(origin_input_ids) - _sequence_length(prefix_indices),
        )
        self.controller.visible_admission.observe_prefix(
            request_id,
            uncached_prompt_tokens=uncached_prompt_tokens,
            bundle_generations=self._context_bundle_generations(
                metadata.context_id
            ),
        )
        validation = self.controller.visible_admission.validate_ticket(
            ticket,
            epoch=self._admission_epoch,
        )
        if not validation.valid:
            self._record_ticket_skip(
                request_id,
                "prefix_rematch:" + "+".join(validation.reasons),
            )
            return False
        obligation = self._restore_obligation_index().get(request_id)
        if (
            obligation is not None
            and not obligation.state.terminal
            and obligation.state == RestoreObligationState.TICKET_READY
        ):
            refreshed = self.controller.visible_admission.get(request_id)
            if refreshed is not None:
                obligation.required_admission_bytes = (
                    refreshed.request.estimated_incremental_bytes
                )
            if not self._begin_restore_lease_admission(
                obligation, now_ms=float(self._now_ms())
            ):
                self._record_ticket_skip(
                    request_id, "restore_lease_admission_capacity"
                )
                return False
        return True

    def on_prefill_candidate_result(
        self,
        req: Any,
        *,
        admitted: bool,
        result: str,
    ) -> None:
        metadata = self._metadata(req)
        if metadata is None:
            return
        request_id = str(req.rid)
        ticket = self._ticket_for_request(request_id)
        if ticket is None:
            return
        self._ticket_attempted_request_ids.add(request_id)
        obligation = self._restore_obligation_index().get(request_id)
        if admitted:
            if obligation is not None and not obligation.state.terminal:
                self._commit_restore_lease_admission(
                    obligation, now_ms=float(self._now_ms())
                )
            self._ticket_selected_request_ids.add(request_id)
            bypass_request_id = getattr(
                self, "_restore_bypass_request_id", None
            )
            if request_id == bypass_request_id:
                overdue = self._overdue_restore_obligation(
                    now_ms=float(self._now_ms())
                )
                if overdue is not None:
                    overdue.bypass_count += 1
                    self._restore_obligation_counts["bounded_bypass_admitted"] += 1
                    self.audit.emit(
                        "restore_debt_bounded_bypass_admitted",
                        self._now_ms(),
                        obligation_id=overdue.obligation_id,
                        restore_request_id=overdue.request_id,
                        bypass_request_id=request_id,
                        bypass_count=overdue.bypass_count,
                    )
            entry = self.controller.visible_admission.cancel(request_id)
            wait_ms = (
                max(0.0, self._now_ms() - entry.request.submitted_ts_ms)
                if entry is not None
                else None
            )
            self._ticket_selection_details[request_id] = {
                "request_id": request_id,
                "workflow_id": metadata.root_workflow_id,
                "invocation_id": metadata.invocation_id,
                "rank": ticket.rank,
                "native_result": result,
                "admission_wait_ms": wait_ms,
                "reserved_bytes": ticket.reservation_credit_bytes,
            }
        else:
            if obligation is not None and not obligation.state.terminal:
                self._reject_restore_lease_admission(
                    obligation,
                    now_ms=float(self._now_ms()),
                    native_result=result,
                )
            self._ticket_native_rejections[request_id] = result

    def end_prefill_epoch(self, can_run_list: Any) -> None:
        ticket_epoch = self._current_ticket_epoch
        if ticket_epoch is None:
            return
        selected = set(self._ticket_selected_request_ids)
        if selected:
            self._retraction_admission_stall_since_ms = None
        elif ticket_epoch.tickets and any(
            entry.state == AdmissionSideState.VISIBLE_PENDING
            for entry in self.controller.visible_admission.entries()
        ):
            if getattr(self, "_retraction_admission_stall_since_ms", None) is None:
                self._retraction_admission_stall_since_ms = self._now_ms()
        expired = [
            {
                "request_id": ticket.request_id,
                "workflow_id": ticket.workflow_id,
                "rank": ticket.rank,
                "attempted": (
                    ticket.request_id in self._ticket_attempted_request_ids
                ),
            }
            for ticket in ticket_epoch.tickets
            if ticket.request_id not in selected
        ]
        for ticket in ticket_epoch.tickets:
            if ticket.request_id in selected:
                continue
            obligation = self._restore_obligation_index().get(ticket.request_id)
            lease = self._restore_lease_index().get(ticket.request_id)
            if (
                obligation is not None
                and not obligation.state.terminal
                and lease is not None
                and lease.state == RestoreLeaseState.ADMISSION_COMMITTING
            ):
                self._reject_restore_lease_admission(
                    obligation,
                    now_ms=float(self._now_ms()),
                    native_result="ticket_epoch_expired",
                )
        self.audit.emit(
            "admission_ticket_epoch_finished",
            self._now_ms(),
            epoch=ticket_epoch.epoch,
            issued_count=len(ticket_epoch.tickets),
            selected_count=len(selected),
            native_batch_size=len(tuple(can_run_list or ())),
            source=ticket_epoch.source,
            observed_admission_window=(
                self._observed_admission_window_fields(
                    self._current_observed_admission_window
                )
                if self._current_observed_admission_window is not None
                else None
            ),
            policy_skip_count=sum(
                1 for item in self._ticket_skip_audit if item[0] == ticket_epoch.epoch
            ),
            selected=[
                self._ticket_selection_details[request_id]
                for request_id in sorted(
                    self._ticket_selection_details,
                    key=lambda item: self._current_tickets_by_request[item].rank,
                )
            ],
            expired=expired,
            native_rejected=[
                {
                    "request_id": request_id,
                    "result": result,
                }
                for request_id, result in sorted(
                    self._ticket_native_rejections.items()
                )
            ],
            policy_skip_reason_counts=dict(
                sorted(
                    Counter(
                        reason
                        for event_epoch, _, reason in self._ticket_skip_audit
                        if event_epoch == ticket_epoch.epoch
                    ).items()
                )
            ),
            policy_skip_details=[
                {"request_id": request_id, "reason": reason}
                for event_epoch, request_id, reason in sorted(
                    self._ticket_skip_audit
                )
                if event_epoch == ticket_epoch.epoch and reason != "no_ticket"
            ],
        )
        self._current_ticket_epoch = None
        self._current_tickets_by_request.clear()
        self._ticket_attempted_request_ids.clear()
        self._ticket_selected_request_ids.clear()
        self._ticket_selection_details.clear()
        self._ticket_native_rejections.clear()
        self._current_observed_admission_window = None

    def _ticket_for_request(self, request_id: str) -> AdmissionTicket | None:
        ticket_epoch = self._current_ticket_epoch
        if ticket_epoch is None:
            return None
        return self._current_tickets_by_request.get(request_id)

    def _record_ticket_skip(self, request_id: str, reason: str) -> None:
        key = (self._admission_epoch, request_id, reason)
        if key in self._ticket_skip_audit:
            return
        self._ticket_skip_audit.add(key)

    def _record_ticket_timing(self, name: str, elapsed_ms: float) -> None:
        samples = getattr(self, "_ticket_timing_samples", None)
        if samples is None:
            samples = {
                "compile_ms": deque(maxlen=65_536),
                "validation_ms": deque(maxlen=65_536),
            }
            self._ticket_timing_samples = samples
        samples[name].append(elapsed_ms)

    def _observed_admission_scheduler(self) -> ObservedAdmissionScheduler:
        scheduler = getattr(self, "observed_admission_scheduler", None)
        if scheduler is None:
            scheduler = ObservedAdmissionScheduler(
                active_kv_high_watermark_ratio=(
                    self.config.observed_admission_active_kv_high_watermark_ratio
                ),
                minimum_active_requests=(
                    self.config.observed_admission_min_active_requests
                ),
            )
            self.observed_admission_scheduler = scheduler
        return scheduler

    def _observed_admission_candidates(
        self,
        tagged: list[tuple[int, Any, BeliefKVRequestMetadata]],
        *,
        entries: Mapping[str, Any],
        now_ms: float,
        fair_rank: Mapping[str, int],
        frontier_candidates: Mapping[str, tuple[Any, ...]],
    ) -> tuple[ObservedAdmissionCandidate, ...]:
        frontier_by_invocation = {
            workflow_id: {
                item.invocation_id: (index, item)
                for index, item in enumerate(items)
            }
            for workflow_id, items in frontier_candidates.items()
        }
        result: list[ObservedAdmissionCandidate] = []
        for native_index, req, metadata in tagged:
            request_id = str(req.rid)
            entry = entries.get(request_id)
            if entry is None:
                continue
            wait_ms = max(0.0, now_ms - entry.request.submitted_ts_ms)
            frontier_rank, frontier = frontier_by_invocation.get(
                metadata.root_workflow_id, {}
            ).get(metadata.invocation_id, (1 << 30, None))
            result.append(
                ObservedAdmissionCandidate(
                    request_id=request_id,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    native_index=native_index,
                    causal_rank=(
                        int(frontier.score[0]) if frontier is not None else 3
                    ),
                    unblock_depth=(
                        int(frontier.unblock_depth) if frontier is not None else 0
                    ),
                    frontier_rank=frontier_rank,
                    workflow_fair_rank=fair_rank.get(
                        metadata.root_workflow_id, 1 << 30
                    ),
                    wait_ms=wait_ms,
                    estimated_incremental_bytes=(
                        entry.request.estimated_incremental_bytes
                    ),
                    starvation=(
                        wait_ms >= self.config.admission_liveness_timeout_ms
                    ),
                    policy_eligible=(
                        entry.state == AdmissionSideState.VISIBLE_PENDING
                    ),
                )
            )
        return tuple(result)

    def _observed_admission_snapshot(
        self,
        *,
        native_available_hbm_bytes: int,
        native_max_requests: int,
    ) -> ObservedAdmissionSnapshot:
        breakdown = self.controller.page_index.physical_kv_state_breakdown()
        scheduler = getattr(self, "scheduler", None)
        running_batch = getattr(scheduler, "running_batch", None)
        requests = list(getattr(running_batch, "reqs", ()) or ())
        chunked_req = getattr(scheduler, "chunked_req", None)
        if chunked_req is not None:
            requests.append(chunked_req)
        unique_requests: dict[str, Any] = {}
        for req in requests:
            request_id = str(getattr(req, "rid", f"object:{id(req)}"))
            unique_requests.setdefault(request_id, req)

        running_private_bytes = 0
        for req in unique_requests.values():
            fill_tokens = max(
                _sequence_length(getattr(req, "fill_ids", None)),
                _sequence_length(getattr(req, "origin_input_ids", None))
                + _sequence_length(getattr(req, "output_ids", None)),
            )
            prefix_tokens = _sequence_length(
                getattr(req, "prefix_indices", None)
            )
            running_private_bytes += (
                max(0, fill_tokens - prefix_tokens)
                * self.config.kv_bytes_per_token
            )
        return ObservedAdmissionSnapshot(
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
            reserve_hbm_bytes=self.config.reserve_hbm_bytes,
            native_available_hbm_bytes=max(0, native_available_hbm_bytes),
            native_max_requests=max(0, native_max_requests),
            running_request_count=len(unique_requests),
            radix_locked_bytes=breakdown.engine_locked_bytes,
            running_private_bytes=running_private_bytes,
        )

    @staticmethod
    def _observed_admission_window_fields(
        window: ObservedAdmissionWindow,
    ) -> dict[str, object]:
        return {
            "mode": window.mode,
            "active_kv_budget_bytes": window.active_kv_budget_bytes,
            "active_kv_footprint_bytes": window.active_kv_footprint_bytes,
            "active_kv_headroom_bytes": window.active_kv_headroom_bytes,
            "active_growth_budget_bytes": window.active_growth_budget_bytes,
            "active_kv_pressure": window.active_kv_pressure,
            "radix_locked_bytes": window.radix_locked_bytes,
            "running_private_bytes": window.running_private_bytes,
            "running_request_count": window.running_request_count,
            "max_new_requests": window.max_new_requests,
        }

    def _observed_ticket_order_key(
        self,
        native_index: int,
        req: Any,
        metadata: BeliefKVRequestMetadata,
        now_ms: float,
        *,
        fair_rank: Mapping[str, int],
        frontier_rank: Mapping[str, Mapping[str, int]],
    ) -> tuple[object, ...]:
        request_id = str(req.rid)
        entry = self.controller.visible_admission.get(request_id)
        submitted_ts_ms = (
            entry.request.submitted_ts_ms if entry is not None else now_ms
        )
        wait_ms = max(0.0, now_ms - submitted_ts_ms)
        starvation = wait_ms >= self.config.admission_liveness_timeout_ms
        workflow_frontier = frontier_rank.get(metadata.root_workflow_id, {})
        workflow_fair_rank = fair_rank.get(metadata.root_workflow_id, 1 << 30)
        if starvation:
            return (
                0,
                -wait_ms,
                workflow_fair_rank,
                workflow_frontier.get(metadata.invocation_id, 1 << 30),
                native_index,
            )
        return (
            1,
            workflow_frontier.get(metadata.invocation_id, 1 << 30),
            native_index,
            workflow_fair_rank,
        )

    def _restore_ready_ticket_priority(
        self,
        entries: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Return oldest restored debts that must not be lost to plan staleness."""

        return tuple(
            obligation.request_id
            for obligation in self._restore_obligation_index().active()
            if obligation.state == RestoreObligationState.TICKET_READY
            and obligation.request_id in entries
            and entries[obligation.request_id].state
            == AdmissionSideState.VISIBLE_PENDING
        )

    def _sync_visible_gate_state(
        self,
        request_id: str,
        metadata: BeliefKVRequestMetadata,
        *,
        req: Any | None = None,
    ) -> None:
        entry = self.controller.visible_admission.get(request_id)
        if entry is None:
            return
        transition_generation, transition_open = self._workflow_transition_state(
            metadata.root_workflow_id
        )
        self.controller.visible_admission.set_transition_generation(
            request_id, transition_generation
        )
        entry = self.controller.visible_admission.get(request_id)
        if entry is not None:
            self.controller.visible_admission.observe_prefix(
                request_id,
                uncached_prompt_tokens=entry.request.uncached_prompt_tokens,
                bundle_generations=self._context_bundle_generations(
                    metadata.context_id
                ),
            )
        cooldowns = getattr(self, "_retraction_cooldown_until_by_request", {})
        cooldown_until = cooldowns.get(request_id, 0.0)
        obligation = self._restore_obligation_index().get(request_id)
        if obligation is not None and not obligation.state.terminal:
            if (
                obligation.native_admission_fallback
                and obligation.state == RestoreObligationState.TICKET_READY
            ):
                self.controller.visible_admission.set_visible(request_id)
                return
            required = self._request_path_extent_ids(
                req, metadata.context_id, cpu_only=True
            )
            if required is None:
                required = obligation.required_extent_ids
            obligation.set_required_extents(
                required,
                restore_bytes=self._extent_ids_bytes(required),
                now_ms=self._now_ms(),
            )
            if not obligation.source_transaction_terminal:
                self.controller.visible_admission.set_wait_restore(
                    request_id,
                    required or (f"obligation:{obligation.obligation_id}",),
                    reason="retraction_source_d2h_pending",
                )
            elif required or obligation.pending_command_id is not None:
                self.controller.visible_admission.set_wait_restore(
                    request_id,
                    required or (f"obligation:{obligation.obligation_id}",),
                    reason=(
                        "h2d_inflight"
                        if obligation.state == RestoreObligationState.H2D_INFLIGHT
                        else "restore_obligation_pending"
                    ),
                )
            else:
                lease = self._restore_lease_index().get(request_id)
                lease_ready = (
                    not bool(getattr(self.config, "restore_lease_enabled", True))
                    or (
                        lease is not None
                        and lease.state
                        in {
                            RestoreLeaseState.RESTORED_RESERVED,
                            RestoreLeaseState.ADMISSION_COMMITTING,
                            RestoreLeaseState.ADMITTED,
                        }
                        and (
                            obligation.restored_bytes == 0
                            or request_id in self._restore_lease_pins
                            or lease.state
                            in {
                                RestoreLeaseState.ADMISSION_COMMITTING,
                                RestoreLeaseState.ADMITTED,
                            }
                        )
                    )
                )
                if lease_ready:
                    obligation.mark_ticket_ready(now_ms=self._now_ms())
                    self.controller.visible_admission.set_visible(request_id)
                else:
                    self.controller.visible_admission.set_wait_restore(
                        request_id,
                        (f"obligation:{obligation.obligation_id}",),
                        reason="restore_lease_pending",
                    )
            return
        if self._metadata_scope_is_terminal(metadata):
            self.controller.visible_admission.set_policy_blocked(
                request_id, reason="terminal_scope"
            )
            return
        restore_bundle_ids = self._request_restore_bundle_ids(
            req, metadata.context_id
        )
        if (
            metadata.context_id in self._pending_h2d_contexts
            or restore_bundle_ids
        ):
            obligation = None
            if restore_bundle_ids:
                obligation = self._ensure_ordinary_waiting_restore_obligation(
                    request_id,
                    metadata,
                    req,
                    restore_bundle_ids,
                    now_ms=self._now_ms(),
                )
            if metadata.context_id in self._pending_h2d_contexts:
                restore_reason = "h2d_inflight"
            elif obligation is not None:
                restore_reason = "restore_obligation_pending"
            else:
                restore_reason = "restore_obligation_capacity"
            self.controller.visible_admission.set_wait_restore(
                request_id,
                restore_bundle_ids or (f"context:{metadata.context_id}",),
                reason=restore_reason,
            )
        else:
            overdue_restore = self._overdue_restore_obligation(
                now_ms=self._now_ms()
            )
            bypass_request_id = getattr(
                self, "_restore_bypass_request_id", None
            )
            if (
                overdue_restore is not None
                and request_id != overdue_restore.request_id
                and request_id != bypass_request_id
            ):
                self.controller.visible_admission.set_policy_blocked(
                    request_id,
                    reason=(
                        f"restore_debt_barrier:{overdue_restore.obligation_id}"
                    ),
                )
                return
            if self._now_ms() < cooldown_until:
                self.controller.visible_admission.set_policy_blocked(
                    request_id, reason="retraction_cooldown"
                )
                return
            cooldowns.pop(request_id, None)
            transaction = getattr(
                self, "_pending_running_retraction_transaction", None
            )
            if transaction is not None and transaction.stage in {
                "residency_pending",
                "residency_wait_queue",
            }:
                self.controller.visible_admission.set_policy_blocked(
                    request_id,
                    reason=(
                        "retraction_residency_pending:"
                        f"{transaction.transaction_id}"
                    ),
                )
                return
            if transition_open:
                self.controller.visible_admission.set_policy_blocked(
                    request_id, reason="transition_open"
                )
            else:
                self.controller.visible_admission.set_visible(request_id)

    def _workflow_transition_state(self, workflow_id: str) -> tuple[int, bool]:
        state = self.controller._transition_by_workflow.get(workflow_id, {})
        return int(state.get("generation", 0)), bool(state.get("open", False))

    def _context_restore_bundle_ids(self, context_id: str) -> tuple[str, ...]:
        page_index = self.controller.page_index
        if not page_index.has_context(context_id):
            return ()
        return tuple(
            f"page:{page.handle.page_id}:{page.handle.allocation_generation}"
            for page in page_index.context_pages(context_id)
            if page.cpu_resident and not page.gpu_resident
        )

    def _request_restore_bundle_ids(
        self,
        req: Any | None,
        context_id: str,
    ) -> tuple[str, ...]:
        """Return only CPU extents on this request's matched Radix path."""

        result = self._request_path_extent_ids(req, context_id, cpu_only=True)
        return (
            self._context_restore_bundle_ids(context_id)
            if result is None
            else result
        )

    def _extent_ids_bytes(self, extent_ids: tuple[str, ...]) -> int:
        total = 0
        for extent_id in extent_ids:
            try:
                handle = self._runtime_page_handle_from_extent_id(extent_id)
            except ValueError:
                continue
            page = self.controller.page_index.pages.get(handle)
            if page is not None and page.residency != PhysicalResidency.DEAD:
                total += page.size_bytes
        return total

    @staticmethod
    def _runtime_page_handle_from_extent_id(extent_id: str) -> PageHandle:
        parts = extent_id.split(":")
        if len(parts) == 3 and parts[0] == "page":
            try:
                return PageHandle(int(parts[1]), int(parts[2]))
            except ValueError as error:
                raise ValueError(
                    f"invalid runtime page extent ID: {extent_id}"
                ) from error
        return page_handle_from_extent_id(extent_id)

    def _context_bundle_generations(self, context_id: str) -> dict[str, str]:
        page_index = self.controller.page_index
        if not page_index.has_context(context_id):
            return {}
        result = {}
        for page in page_index.context_pages(context_id):
            bundle_id = (
                f"page:{page.handle.page_id}:{page.handle.allocation_generation}"
            )
            parent = (
                f"{page.parent.page_id}:{page.parent.allocation_generation}"
                if page.parent is not None
                else "root"
            )
            owners = ",".join(
                f"{owner}:{epoch}"
                for owner, epoch in sorted(page.owner_contexts.items())
            )
            result[bundle_id] = "|".join(
                (
                    page.residency.value,
                    str(page.engine_lock_ref),
                    str(page.active_reader_count),
                    page.transfer_direction.value
                    if page.transfer_direction is not None
                    else "idle",
                    parent,
                    owners,
                )
            )
        return result

    def _mark_context_wait_restore(
        self,
        context_id: str,
        *,
        bundle_ids: tuple[str, ...],
        reason: str,
    ) -> None:
        for entry in self.controller.visible_admission.entries():
            if entry.request.context_id == context_id:
                self.controller.visible_admission.set_wait_restore(
                    entry.request.request_id,
                    bundle_ids,
                    reason=reason,
                )

    def _mark_h2d_waiters(
        self,
        context_id: str,
        *,
        restored_extent_ids: tuple[str, ...],
        reason: str,
    ) -> None:
        restored = frozenset(restored_extent_ids)
        waiting = {
            str(req.rid): req
            for req in tuple(getattr(self.scheduler, "waiting_queue", ()) or ())
        }
        for entry in tuple(self.controller.visible_admission.entries()):
            if entry.request.context_id != context_id:
                continue
            req = waiting.get(entry.request.request_id)
            required = self._request_restore_bundle_ids(req, context_id)
            if required and (not restored or restored.intersection(required)):
                self.controller.visible_admission.set_wait_restore(
                    entry.request.request_id,
                    required,
                    reason=reason,
                )

    def _release_h2d_waiters(self, context_id: str) -> None:
        waiting = {
            str(req.rid): req
            for req in tuple(getattr(self.scheduler, "waiting_queue", ()) or ())
        }
        for entry in tuple(self.controller.visible_admission.entries()):
            if entry.request.context_id != context_id:
                continue
            request_id = entry.request.request_id
            req = waiting.get(request_id)
            if req is not None:
                req.init_next_round_input(self.tree_cache)
                prefix_indices = getattr(req, "prefix_indices", None)
                origin_input_ids = getattr(req, "origin_input_ids", None)
                self.controller.visible_admission.observe_prefix(
                    request_id,
                    uncached_prompt_tokens=max(
                        0,
                        _sequence_length(origin_input_ids)
                        - _sequence_length(prefix_indices),
                    ),
                    bundle_generations=self._context_bundle_generations(
                        context_id
                    ),
                )
            metadata = self._request_metadata_by_id.get(request_id)
            if metadata is not None:
                self._sync_visible_gate_state(request_id, metadata, req=req)
            current = self.controller.visible_admission.get(request_id)
            self.audit.emit(
                "request_restore_dependency_updated",
                self._now_ms(),
                request_id=request_id,
                context_id=context_id,
                state=(current.state.value if current is not None else "missing"),
                cache_hit_tokens=(
                    len(getattr(req, "prefix_indices", ())) if req is not None else None
                ),
            )

    def _invalidate_h2d_waiters(
        self,
        context_id: str,
        *,
        bundle_ids: tuple[str, ...],
        status: CommandStatus,
    ) -> None:
        waiting = {
            str(req.rid): req
            for req in tuple(getattr(self.scheduler, "waiting_queue", ()) or ())
        }
        for entry in tuple(self.controller.visible_admission.entries()):
            if entry.request.context_id != context_id:
                continue
            request_id = entry.request.request_id
            req = waiting.get(request_id)
            remaining = self._request_restore_bundle_ids(req, context_id)
            if remaining:
                self.controller.visible_admission.set_wait_restore(
                    request_id,
                    remaining,
                    reason=f"h2d_{status.value}",
                )
                continue
            metadata = self._request_metadata_by_id.get(request_id)
            if metadata is not None:
                self._sync_visible_gate_state(request_id, metadata, req=req)

    def _capture_request_physical_start(
        self,
        req: Any,
        metadata: BeliefKVRequestMetadata,
        now_ms: float,
    ) -> None:
        starts = getattr(self, "_request_physical_start_by_id", None)
        if starts is None:
            starts = {}
            self._request_physical_start_by_id = starts
        getattr(self, "_aborted_request_physical_start_by_id", {}).pop(
            str(req.rid), None
        )
        try:
            checkpoint = self._context_physical_checkpoint(metadata.context_id)
            origin_input_ids = getattr(req, "origin_input_ids", None)
            prefix_indices = getattr(req, "prefix_indices", None)
            prompt_tokens = (
                len(origin_input_ids) if origin_input_ids is not None else 0
            )
            cache_hit_tokens = (
                len(prefix_indices) if prefix_indices is not None else 0
            )
            page_size = max(
                1,
                int(
                    getattr(
                        getattr(
                            self.scheduler, "token_to_kv_pool_allocator", None
                        ),
                        "page_size",
                        getattr(self.scheduler, "page_size", 1),
                    )
                    or 1
                ),
            )
            value = {
                **checkpoint,
                "request_id": str(req.rid),
                "workflow_id": metadata.root_workflow_id,
                "invocation_id": metadata.invocation_id,
                "context_id": metadata.context_id,
                "context_epoch": metadata.context_epoch,
                "prompt_tokens": prompt_tokens,
                "cache_hit_tokens": cache_hit_tokens,
                "uncached_prompt_tokens": max(
                    0, prompt_tokens - cache_hit_tokens
                ),
                "allocator_page_size_tokens": page_size,
                "checkpoint_ts_ms": now_ms,
                "submitted_ts_ms": getattr(
                    self, "_request_submitted_ts_by_id", {}
                ).get(str(req.rid)),
                "queue_wait_ms": max(
                    0.0,
                    now_ms
                    - float(
                        getattr(self, "_request_submitted_ts_by_id", {}).get(
                            str(req.rid), now_ms
                        )
                    ),
                ),
            }
            starts[str(req.rid)] = value
            self.audit.emit("request_physical_start", now_ms, **value)
        except Exception as error:
            self.audit.emit(
                "request_physical_checkpoint_failed",
                now_ms,
                phase="start",
                request_id=str(req.rid),
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
                error=f"{type(error).__name__}: {error}",
            )

    def _enforce_queue_timeouts(self, *, now_ms: float) -> None:
        starts = getattr(self, "_request_physical_start_by_id", {})
        submitted = getattr(self, "_request_submitted_ts_by_id", {})
        if not submitted:
            return
        timed_out = getattr(self, "_queue_timeout_request_ids", set())
        timeout_ms = float(self.config.request_queue_timeout_ms)
        for request_id, submitted_ms in tuple(submitted.items()):
            if request_id in starts or request_id in timed_out:
                continue
            metadata = self._request_metadata_by_id.get(request_id)
            if metadata is None:
                continue
            elapsed_ms = max(0.0, now_ms - float(submitted_ms))
            if elapsed_ms < timeout_ms:
                continue
            timed_out.add(request_id)
            self._terminal_cancelled_request_ids.add(request_id)
            self.audit.emit(
                "request_queue_timeout",
                now_ms,
                request_id=request_id,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                submitted_ts_ms=submitted_ms,
                queue_wait_ms=elapsed_ms,
                queue_timeout_ms=timeout_ms,
                timeout_scope="submission_to_physical_start",
            )
            self.scheduler.abort_request(self._new_abort_request(request_id))

    def _enforce_execution_timeouts(self, *, now_ms: float) -> None:
        starts = getattr(self, "_request_physical_start_by_id", {})
        timed_out = getattr(self, "_execution_timeout_request_ids", set())
        ledger = getattr(self, "_lock_service_ledger", None)
        for request_id, checkpoint in tuple(starts.items()):
            if request_id in timed_out:
                continue
            metadata = self._request_metadata_by_id.get(request_id)
            if metadata is None or metadata.execution_timeout_s is None:
                continue
            started_ms = float(checkpoint.get("checkpoint_ts_ms", now_ms))
            elapsed_ms = max(0.0, now_ms - started_ms)
            timeout_ms = metadata.execution_timeout_s * 1000.0
            progress = (
                ledger.progress_record(request_id)
                if ledger is not None
                else None
            )
            no_progress_ms = (
                ledger.stale_for_ms(request_id, now_ms=now_ms)
                if progress is not None
                else elapsed_ms
            )
            if no_progress_ms < timeout_ms:
                continue
            timed_out.add(request_id)
            self._terminal_cancelled_request_ids.add(request_id)
            self.audit.emit(
                "request_execution_timeout",
                now_ms,
                request_id=request_id,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                execution_started_ts_ms=started_ms,
                execution_elapsed_ms=elapsed_ms,
                last_gpu_service_ts_ms=(
                    progress.last_completed_service_ts_ms
                    if progress is not None
                    else None
                ),
                completed_gpu_service_count=(
                    progress.completed_service_count
                    if progress is not None
                    else 0
                ),
                gpu_service_no_progress_ms=no_progress_ms,
                execution_timeout_ms=timeout_ms,
                timeout_kind="gpu_service_inactivity",
                timeout_scope=(
                    "last_completed_gpu_service_to_scheduler_abort"
                    if progress is not None
                    and progress.last_completed_service_ts_ms is not None
                    else "first_gpu_selection_to_scheduler_abort"
                    if progress is not None
                    else "physical_start_to_scheduler_abort_fallback"
                ),
            )
            self.scheduler.abort_request(self._new_abort_request(request_id))

    def _queue_request_physical_finish(
        self,
        req: Any,
        metadata: BeliefKVRequestMetadata,
        *,
        output_tokens: int,
        cache_commit_tokens: int,
    ) -> None:
        pending = getattr(self, "_pending_request_physical_finish_by_id", None)
        if pending is None:
            pending = {}
            self._pending_request_physical_finish_by_id = pending
        pending[str(req.rid)] = {
            "request_id": str(req.rid),
            "workflow_id": metadata.root_workflow_id,
            "invocation_id": metadata.invocation_id,
            "context_id": metadata.context_id,
            "context_epoch": metadata.context_epoch,
            "output_tokens": max(0, int(output_tokens)),
            "cache_commit_tokens": max(0, int(cache_commit_tokens)),
            "engine_finish_ts_ms": self._now_ms(),
        }

    def _flush_request_physical_finishes(self) -> None:
        pending = getattr(self, "_pending_request_physical_finish_by_id", None)
        if not pending or getattr(self, "_tree_dirty", False):
            return
        controller = getattr(self, "controller", None)
        if getattr(controller, "inflight_command_ids", ()):
            return
        starts = getattr(self, "_request_physical_start_by_id", {})
        aborted_starts = getattr(
            self, "_aborted_request_physical_start_by_id", {}
        )
        now_ms = self._now_ms()
        for request_id, finish in tuple(sorted(pending.items())):
            start = starts.get(request_id) or aborted_starts.get(request_id)
            if start is None:
                self.audit.emit(
                    "request_physical_checkpoint_failed",
                    now_ms,
                    phase="finish",
                    request_id=request_id,
                    workflow_id=finish["workflow_id"],
                    invocation_id=finish["invocation_id"],
                    context_id=finish["context_id"],
                    error="missing request_physical_start checkpoint",
                )
                pending.pop(request_id, None)
                continue
            try:
                after = self._context_physical_checkpoint(
                    str(finish["context_id"])
                )
                before_extents = {
                    str(key): int(value)
                    for key, value in dict(start["extent_bytes"]).items()
                }
                after_extents = {
                    str(key): int(value)
                    for key, value in dict(after["extent_bytes"]).items()
                }
                new_extent_ids = tuple(
                    sorted(set(after_extents) - set(before_extents))
                )
                released_extent_ids = tuple(
                    sorted(set(before_extents) - set(after_extents))
                )
                uncached_tokens = int(start["uncached_prompt_tokens"])
                page_size = int(start["allocator_page_size_tokens"])
                logical_growth_tokens = max(
                    0,
                    int(finish["cache_commit_tokens"])
                    - int(start["cache_hit_tokens"]),
                )
                allocator_growth_tokens = (
                    (logical_growth_tokens + page_size - 1) // page_size * page_size
                )
                fields = {
                    **finish,
                    "request_aborted_during_finish": request_id in aborted_starts,
                    "request_abort_reason": start.get("abort_reason"),
                    "request_abort_ts_ms": start.get("abort_ts_ms"),
                    "checkpoint_ts_ms": now_ms,
                    "prompt_tokens": int(start["prompt_tokens"]),
                    "observed_cache_hit_tokens": int(start["cache_hit_tokens"]),
                    "uncached_prompt_tokens": uncached_tokens,
                    "allocator_page_size_tokens": page_size,
                    "logical_allocator_growth_tokens": logical_growth_tokens,
                    "allocator_growth_tokens_upper_bound": (
                        allocator_growth_tokens
                    ),
                    "allocator_growth_bytes_upper_bound": (
                        allocator_growth_tokens
                        * self.config.kv_bytes_per_token
                    ),
                    "allocator_growth_exact": page_size == 1,
                    "context_path_bytes_before": int(
                        start["physical_unique_bytes"]
                    ),
                    "context_path_bytes_after": int(
                        after["physical_unique_bytes"]
                    ),
                    "context_path_growth_bytes": (
                        int(after["physical_unique_bytes"])
                        - int(start["physical_unique_bytes"])
                    ),
                    "gpu_bytes_before": int(start["gpu_bytes"]),
                    "gpu_bytes_after": int(after["gpu_bytes"]),
                    "cpu_bytes_before": int(start["cpu_bytes"]),
                    "cpu_bytes_after": int(after["cpu_bytes"]),
                    "private_bytes_after": int(after["private_bytes"]),
                    "shared_bytes_after": int(after["shared_bytes"]),
                    "new_extent_ids": list(new_extent_ids),
                    "new_extent_bytes": sum(
                        after_extents[item] for item in new_extent_ids
                    ),
                    "released_extent_ids": list(released_extent_ids),
                    "released_extent_bytes": sum(
                        before_extents[item] for item in released_extent_ids
                    ),
                    "topology_revision_before": int(
                        start["topology_revision"]
                    ),
                    "topology_revision_after": int(
                        after["topology_revision"]
                    ),
                    "extent_delta_semantics": (
                        "diagnostic only; Radix split/merge can change extent identity"
                    ),
                    "counterfactual_cache_caveat": (
                        "cache-hit outcome is policy-dependent and must be resimulated"
                    ),
                }
                self.audit.emit("request_physical_delta", now_ms, **fields)
            except Exception as error:
                self.audit.emit(
                    "request_physical_checkpoint_failed",
                    now_ms,
                    phase="finish",
                    request_id=request_id,
                    workflow_id=finish["workflow_id"],
                    invocation_id=finish["invocation_id"],
                    context_id=finish["context_id"],
                    error=f"{type(error).__name__}: {error}",
                )
            finally:
                starts.pop(request_id, None)
                aborted_starts.pop(request_id, None)
                pending.pop(request_id, None)

    def _context_physical_checkpoint(self, context_id: str) -> dict[str, Any]:
        page_index = self.controller.page_index
        if not page_index.has_context(context_id):
            raise SGLangBackendError(
                f"physical checkpoint has unknown context: {context_id}"
            )
        pages = page_index.context_pages(context_id)
        extent_bytes = {
            f"{page.handle.page_id}:{page.handle.allocation_generation}": (
                page.size_bytes
            )
            for page in pages
        }
        gpu_bytes = sum(page.size_bytes for page in pages if page.gpu_resident)
        cpu_bytes = sum(page.size_bytes for page in pages if page.cpu_resident)
        private_bytes = sum(
            page.size_bytes
            for page in pages
            if set(page.owner_contexts) == {context_id}
        )
        return {
            "physical_unique_bytes": sum(page.size_bytes for page in pages),
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "private_bytes": private_bytes,
            "shared_bytes": sum(page.size_bytes for page in pages) - private_bytes,
            "extent_count": len(pages),
            "extent_bytes": dict(sorted(extent_bytes.items())),
            "page_index_revision": page_index.revision,
            "topology_revision": page_index.topology_revision,
        }

    def on_cache_finished(self, req: Any, token_ids: list[int]) -> None:
        self._ensure_allocator_radix_consistency(reason="cache_finished")
        metadata = self._metadata(req)
        if metadata is None:
            return
        self._finish_restore_obligation(
            str(req.rid),
            RestoreObligationState.SATISFIED,
            now_ms=float(self._now_ms()),
            reason="request_finished_after_restore",
        )
        self._cancel_restore_service_grace(
            str(req.rid),
            now_ms=float(self._now_ms()),
            reason="request_finished",
        )
        last_node = self._match_terminal_node(token_ids)
        if last_node is not None:
            self._terminal_node_by_context[metadata.context_id] = last_node
            self._mark_context_dirty(metadata.context_id)
        self._tree_dirty = True
        self._active_request_ids.discard(req.rid)
        getattr(self, "_retracted_engine_request_ids", set()).discard(str(req.rid))
        getattr(self, "_pending_selective_retraction_ids", set()).discard(
            str(req.rid)
        )
        getattr(self, "_retraction_cooldown_until_by_request", {}).pop(
            str(req.rid), None
        )
        self.controller.visible_admission.cancel(str(req.rid))
        terminal_marker = req.rid in self._terminal_cancelled_request_ids
        logical_scope_terminal = self._metadata_scope_is_terminal(metadata)
        terminal_cancelled = terminal_marker or logical_scope_terminal
        self._terminal_cancelled_request_ids.discard(req.rid)
        getattr(self, "_queue_timeout_request_ids", set()).discard(req.rid)
        getattr(self, "_execution_timeout_request_ids", set()).discard(req.rid)
        self._request_metadata_by_id.pop(req.rid, None)
        getattr(self, "_request_submitted_ts_by_id", {}).pop(req.rid, None)
        if terminal_cancelled:
            getattr(self, "_request_physical_start_by_id", {}).pop(req.rid, None)
            getattr(self, "_aborted_request_physical_start_by_id", {}).pop(
                req.rid, None
            )
            getattr(self, "_pending_request_physical_finish_by_id", {}).pop(
                req.rid, None
            )
            self.audit.emit(
                "terminal_request_abort_finished",
                self._now_ms(),
                request_id=req.rid,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
                terminal_marker=terminal_marker,
                logical_scope_terminal=logical_scope_terminal,
            )
            return
        self._record_request_token_trace(
            "cache_final_commit",
            self._now_ms(),
            token_ids,
            request_id=str(req.rid),
            workflow_id=metadata.root_workflow_id,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            context_epoch=metadata.context_epoch,
        )
        self._request_partial_commit_count.pop(str(req.rid), None)
        self._queue_request_physical_finish(
            req,
            metadata,
            output_tokens=len(getattr(req, "output_ids", ())),
            cache_commit_tokens=len(token_ids),
        )
        self.audit.emit(
            "request_finished",
            self._now_ms(),
            request_id=req.rid,
            workflow_id=metadata.root_workflow_id,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            output_tokens=len(getattr(req, "output_ids", ())),
        )
        self._emit(
            metadata,
            RuntimeEventKind.LLM_RESULT,
            attributes={
                "request_id": req.rid,
                "output_tokens": len(getattr(req, "output_ids", ())),
            },
        )

    def on_cache_unfinished(self, req: Any, token_ids: list[int]) -> None:
        self._ensure_allocator_radix_consistency(reason="cache_unfinished")
        metadata = self._metadata(req)
        if metadata is None:
            return
        request_id = str(req.rid)
        chunk_index = self._request_partial_commit_count.get(request_id, 0)
        self._request_partial_commit_count[request_id] = chunk_index + 1
        self._record_request_token_trace(
            "cache_partial_commit",
            self._now_ms(),
            token_ids,
            request_id=request_id,
            workflow_id=metadata.root_workflow_id,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            context_epoch=metadata.context_epoch,
            chunk_index=chunk_index,
        )
        last_node = self._match_terminal_node(token_ids)
        if last_node is not None:
            self._terminal_node_by_context[metadata.context_id] = last_node
            self._mark_context_dirty(metadata.context_id)
        self._tree_dirty = True

    def on_cache_reset(self) -> None:
        self._record_request_token_trace(
            "cache_reset",
            self._now_ms(),
            (),
        )
        for ack in self.backend.abort_all(reason="authoritative_cache_reset"):
            self.controller.acknowledge_command(ack)
        for observation in self.backend.poll_transfer_telemetry():
            self.controller.observe_transfer_telemetry(observation)
            self._emit_transfer_telemetry(observation)
        self.controller.reset_transfer_attempts()
        self._h2d_context_by_command.clear()
        self._pending_h2d_contexts.clear()
        self.registry.reset()
        for handle, page in tuple(self.controller.page_index.pages.items()):
            if page.residency.value != "dead" and page.transfer_idle:
                self.controller.page_index.invalidate_page(handle)
        self._terminal_node_by_context.clear()
        ledger = getattr(self, "_lock_service_ledger", None)
        if ledger is not None:
            ledger.clear()
        getattr(self, "_request_physical_start_by_id", {}).clear()
        getattr(self, "_aborted_request_physical_start_by_id", {}).clear()
        getattr(self, "_pending_request_physical_finish_by_id", {}).clear()
        self._request_partial_commit_count.clear()
        index = getattr(self, "_restore_obligations", None)
        if index is not None:
            for obligation in index.active():
                self._finish_restore_obligation(
                    obligation.request_id,
                    RestoreObligationState.FAILED,
                    now_ms=float(self._now_ms()),
                    reason="authoritative_cache_reset",
                )
        for request_id in tuple(
            getattr(self, "_restore_service_grace_by_request", {})
        ):
            self._cancel_restore_service_grace(
                request_id,
                now_ms=float(self._now_ms()),
                reason="authoritative_cache_reset",
            )
        getattr(self, "_restore_command_to_request", {}).clear()
        getattr(self, "_restore_funding_target_by_command", {}).clear()
        self._mark_full_tree_rebuild()

    def _record_request_token_trace(
        self,
        event: str,
        ts_ms: float,
        token_ids: Any,
        **fields: Any,
    ) -> int:
        trace_log = getattr(self, "request_token_trace_log", None)
        if trace_log is None or not trace_log.enabled:
            return 0
        try:
            return trace_log.emit(event, ts_ms, token_ids, **fields)
        except Exception as error:
            self.audit.emit(
                "request_token_trace_failed",
                ts_ms,
                trace_event=event,
                request_id=fields.get("request_id"),
                error=f"{type(error).__name__}: {error}",
            )
            return 0

    def on_radix_mutation(
        self,
        nodes: tuple[Any, ...] | list[Any] | None = None,
        topology_changed: bool = False,
        removed: bool = False,
    ) -> None:
        if not nodes:
            self._mark_full_tree_rebuild()
            return
        dirty_nodes = getattr(self, "_dirty_radix_nodes", None)
        if dirty_nodes is None:
            dirty_nodes = {}
            self._dirty_radix_nodes = dirty_nodes
        removed_nodes = getattr(self, "_removed_radix_nodes", None)
        if removed_nodes is None:
            removed_nodes = {}
            self._removed_radix_nodes = removed_nodes
        target = removed_nodes if removed else dirty_nodes
        for node in nodes:
            target[int(node.id)] = node
            if removed:
                dirty_nodes.pop(int(node.id), None)
        if topology_changed:
            dirty_contexts = getattr(self, "_dirty_context_ids", None)
            if dirty_contexts is None:
                dirty_contexts = set()
                self._dirty_context_ids = dirty_contexts
            dirty_contexts.update(self._terminal_node_by_context)
        self._tree_dirty = True

    def on_hicache_transfer_completed(self, record: dict[str, Any]) -> None:
        """Publish native HiCache DMA without duplicating BeliefKV commands."""

        source = str(record.get("source", "native_unknown"))
        if source == "explicit":
            return
        direction = TransferDirection(str(record["direction"]))
        node_ids = tuple(int(value) for value in record.get("node_ids", ()))
        owner_context_ids: set[str] = set()
        for node_id in node_ids:
            handle = self.registry.current_handle(node_id)
            page = (
                self.controller.page_index.pages.get(handle)
                if handle is not None
                else None
            )
            if page is not None:
                owner_context_ids.update(page.owner_contexts)
        context_id = (
            next(iter(owner_context_ids)) if len(owner_context_ids) == 1 else None
        )
        context = (
            self.controller.graph.contexts.get(context_id)
            if context_id is not None
            else None
        )
        token_count = max(0, int(record.get("token_count", 0)))
        actual_bytes = token_count * self.config.kv_bytes_per_token
        submit_ts_ms = max(0.0, float(record["submit_ts_ms"]))
        complete_ts_ms = max(submit_ts_ms, float(record["complete_ts_ms"]))
        telemetry = TransferTelemetry(
            command_id=f"native-hicache-{record['operation_id']}",
            submit_ts_ms=submit_ts_ms,
            start_ts_ms=None,
            first_layer_ready_ts_ms=None,
            complete_ts_ms=complete_ts_ms,
            compute_wait_ms=None,
            actual_bytes=actual_bytes,
            closure_bytes=actual_bytes,
            merged_operation_count=0,
            direction=direction,
            source_tier="gpu" if direction == TransferDirection.D2H else "host",
            target_tier="host" if direction == TransferDirection.D2H else "gpu",
            status=CommandStatus.COMPLETED,
            reason=str(record.get("reason", "")),
            page_count=len(node_ids),
            context_id=context_id,
            context_epoch=context.epoch if context is not None else None,
            command_kind=source,
            compute_phase="native_hicache",
            host_copy_state=str(record.get("host_copy_state", "unknown")),
            pinned_host=(
                bool(record["pinned_host"])
                if record.get("pinned_host") is not None
                else None
            ),
            native_concurrent_bytes=max(
                0,
                int(record.get("native_inflight_token_count", token_count))
                * self.config.kv_bytes_per_token,
            ),
            allocator_submit_ms=(
                max(0.0, float(record["allocator_submit_ms"]))
                if record.get("allocator_submit_ms") is not None
                else None
            ),
            callback_overhead_ms=max(
                0.0, float(self._now_ms()) - complete_ts_ms
            ),
            start_timestamp_semantics="unavailable",
        )
        self._emit_transfer_telemetry(
            telemetry,
            telemetry_origin="native_hicache_callback",
            backend_operation_id=record.get("backend_operation_id"),
            radix_node_ids=node_ids,
            owner_context_ids=tuple(sorted(owner_context_ids)),
            native_inflight_operation_count_at_submit=record.get(
                "native_inflight_operation_count"
            ),
            start_timestamp_observed=False,
        )

    def _mark_context_dirty(self, context_id: str) -> None:
        dirty_contexts = getattr(self, "_dirty_context_ids", None)
        if dirty_contexts is None:
            dirty_contexts = set()
            self._dirty_context_ids = dirty_contexts
        dirty_contexts.add(context_id)

    def _mark_full_tree_rebuild(self) -> None:
        self._tree_full_rebuild_required = True
        self._tree_dirty = True

    def on_lock_change(self, node: Any) -> None:
        root = self.tree_cache.root_node
        while node is not None and node is not root:
            handle = self.registry.current_handle(int(node.id))
            page = self.controller.page_index.pages.get(handle) if handle else None
            if page is None or page.residency.value == "dead":
                self._mark_full_tree_rebuild()
                return
            previous_lock_ref = page.engine_lock_ref
            self.controller.page_index.set_engine_lock(
                handle,
                max(0, int(getattr(node, "lock_ref", 0))),
            )
            if previous_lock_ref != page.engine_lock_ref:
                self.controller.notify_resource_state_changed()
            node = getattr(node, "parent", None)

    def process_runtime_event(self, event: RuntimeEvent) -> None:
        """Entry point used by an instrumented agent-runtime adapter."""

        self._process_events((event,))

    def _process_events(self, events: tuple[RuntimeEvent, ...]) -> None:
        committed_events, adjustments = self._commit_event_times(events)
        self.controller.process_runtime_events(committed_events)
        for event in committed_events:
            if (
                event.kind == RuntimeEventKind.CONTEXT_COMPACT
                and event.context_id is not None
            ):
                previous_terminal = self._terminal_node_by_context.pop(
                    event.context_id, None
                )
                self._mark_context_dirty(event.context_id)
                self.audit.emit(
                    "context_compaction_committed",
                    event.ts_ms,
                    workflow_id=event.workflow_id,
                    invocation_id=event.invocation_id,
                    context_id=event.context_id,
                    context_epoch=event.context_epoch,
                    previous_context_epoch=event.attributes.get(
                        "previous_context_epoch"
                    ),
                    previous_terminal_present=previous_terminal is not None,
                    old_kv_disposition=event.attributes.get(
                        "old_kv_disposition"
                    ),
                )
        for event in committed_events:
            consumer_delta = self.controller.data_consumers.delta_for_event(
                event.event_id
            )
            if consumer_delta is None:
                continue
            for edge in consumer_delta.changed_edges:
                self.audit.emit(
                    "data_consumer_observed",
                    event.ts_ms,
                    source_event_id=event.event_id,
                    index_version=consumer_delta.index_version,
                    workflow_id=edge.workflow_id,
                    producer_invocation_id=edge.producer_invocation_id,
                    consumer_invocation_id=edge.consumer_invocation_id,
                    relation=edge.relation.value,
                    confidence=edge.confidence,
                    observation_count=edge.observation_count,
                )
        self._cancel_requests_for_terminal_events(committed_events)
        for source_event, committed_event, late_by_ms in adjustments:
            self.audit.emit(
                "runtime_event_time_adjusted",
                self._now_ms(),
                event_id=source_event.event_id,
                kind=source_event.kind.value,
                workflow_id=source_event.workflow_id,
                source_ts_ms=source_event.ts_ms,
                committed_ts_ms=committed_event.ts_ms,
                late_by_ms=late_by_ms,
                max_lateness_ms=self.config.runtime_event_max_lateness_ms,
            )
        if self.event_log is not None:
            try:
                self.event_log.emit_batch(committed_events)
            except Exception as error:
                self.audit.emit(
                    "runtime_event_log_error",
                    self._now_ms(),
                    error=f"{type(error).__name__}: {error}",
                )

    def _cancel_requests_for_terminal_events(
        self, events: tuple[RuntimeEvent, ...]
    ) -> None:
        terminal_invocations = {
            event.invocation_id
            for event in events
            if event.kind
            in {RuntimeEventKind.RETURN, RuntimeEventKind.INVOCATION_CANCEL}
            and event.invocation_id is not None
        }
        terminal_workflows = {
            event.workflow_id
            for event in events
            if event.kind == RuntimeEventKind.WORKFLOW_END
        }
        if not terminal_invocations and not terminal_workflows:
            return

        metadata_by_id = getattr(self, "_request_metadata_by_id", {})
        for request_id, metadata in tuple(metadata_by_id.items()):
            if (
                metadata.invocation_id not in terminal_invocations
                and metadata.root_workflow_id not in terminal_workflows
            ):
                continue
            if self.controller.visible_admission.get(request_id) is not None:
                phase = "visible_pending"
            elif request_id in self._active_request_ids:
                phase = "active"
                self._terminal_cancelled_request_ids.add(request_id)
            else:
                phase = "engine_owned"
                # The scheduler may have launched the batch after BeliefKV's active
                # set was sampled. Preserve the terminal intent so a later cache
                # callback cannot emit a second logical LLM_RESULT.
                self._terminal_cancelled_request_ids.add(request_id)
            self.audit.emit(
                "terminal_request_cancelled",
                self._now_ms(),
                request_id=request_id,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                phase=phase,
            )
            self.scheduler.abort_request(self._new_abort_request(request_id))

    def _metadata_scope_is_terminal(
        self, metadata: BeliefKVRequestMetadata
    ) -> bool:
        graph = getattr(self.controller, "graph", None)
        if graph is None:
            return False
        workflow = graph.workflows.get(metadata.root_workflow_id)
        if workflow is not None and workflow.end_ts_ms is not None:
            return True
        invocation = graph.invocations.get(metadata.invocation_id)
        return invocation is not None and invocation.state.terminal

    @staticmethod
    def _new_abort_request(request_id: str) -> Any:
        from sglang.srt.managers.io_struct import AbortReq

        return AbortReq(rid=request_id)

    def _commit_event_times(
        self, events: tuple[RuntimeEvent, ...]
    ) -> tuple[
        tuple[RuntimeEvent, ...],
        tuple[tuple[RuntimeEvent, RuntimeEvent, float], ...],
    ]:
        watermarks: dict[str, float | None] = {}
        committed: list[RuntimeEvent] = []
        adjustments: list[tuple[RuntimeEvent, RuntimeEvent, float]] = []
        for event in events:
            if event.workflow_id not in watermarks:
                watermarks[event.workflow_id] = (
                    self.controller.graph.timestamp_watermark(event.workflow_id)
                )
            watermark = watermarks[event.workflow_id]
            committed_event = event
            if watermark is not None and event.ts_ms < watermark:
                late_by_ms = watermark - event.ts_ms
                if late_by_ms > self.config.runtime_event_max_lateness_ms:
                    raise SGLangBackendError(
                        f"runtime event {event.event_id} is {late_by_ms:.3f} ms late; "
                        f"limit is {self.config.runtime_event_max_lateness_ms:.3f} ms"
                    )
                attributes = dict(event.attributes)
                attributes.update(
                    {
                        "beliefkv_event_time_adjusted": True,
                        "beliefkv_source_ts_ms": event.ts_ms,
                        "beliefkv_late_by_ms": late_by_ms,
                    }
                )
                committed_event = replace(
                    event,
                    ts_ms=watermark,
                    attributes=attributes,
                )
                adjustments.append((event, committed_event, late_by_ms))
            committed.append(committed_event)
            watermarks[event.workflow_id] = max(
                committed_event.ts_ms,
                watermark if watermark is not None else committed_event.ts_ms,
            )
        return tuple(committed), tuple(adjustments)

    def sync_tree(self, *, force: bool = False) -> None:
        if not force and not self._tree_dirty:
            return
        # A Radix split can change one logical node into multiple physical
        # extents while HiCache DMA still targets the pre-split extent. Keep the
        # mirror generation stable until the command reaches its ACK boundary;
        # the backend then rejects stale residency commits and the next sync
        # rebuilds the authoritative post-mutation topology.
        if self.controller.inflight_command_ids:
            return
        started_ns = time.perf_counter_ns()
        if force or getattr(self, "_tree_full_rebuild_required", True):
            self._sync_tree_full()
            mode = "full"
            changed_nodes = 0
            changed_contexts = 0
        else:
            changed_nodes = len(getattr(self, "_dirty_radix_nodes", {})) + len(
                getattr(self, "_removed_radix_nodes", {})
            )
            changed_contexts = len(getattr(self, "_dirty_context_ids", set()))
            self._sync_tree_incremental()
            mode = "incremental"
        samples = getattr(self, "_tree_sync_timing_samples", None)
        if samples is None:
            samples = deque(maxlen=65_536)
            self._tree_sync_timing_samples = samples
        samples.append(
            (
                mode,
                (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                changed_nodes,
                changed_contexts,
            )
        )

    def _sync_tree_full(self) -> None:
        root = self.tree_cache.root_node
        old_handles = self.registry.current_handles()
        node_handles: dict[int, PageHandle] = {}
        nodes: list[tuple[Any, int]] = []
        stack = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            children = list(getattr(node, "children", {}).values())
            stack.extend((child, depth + 1) for child in reversed(children))
            if node is root:
                continue
            nodes.append((node, depth))

        for node, _ in nodes:
            previous = self.registry.current_handle(int(node.id))
            handle = self.registry.register(node)
            node_handles[id(node)] = handle
            if previous is not None and previous != handle:
                old_page = self.controller.page_index.pages.get(previous)
                if old_page is not None and old_page.residency.value != "dead":
                    self.controller.page_index.invalidate_page(previous)

        current_handles = set(node_handles.values())
        for handle in old_handles - current_handles:
            page = self.controller.page_index.pages.get(handle)
            if page is not None and page.residency.value != "dead":
                self.controller.page_index.invalidate_page(handle)
            self.registry.remove(handle)

        for node, depth in nodes:
            handle = node_handles[id(node)]
            parent = getattr(node, "parent", None)
            parent_handle = node_handles.get(id(parent)) if parent is not root else None
            value = getattr(node, "value", None)
            host_value = getattr(node, "host_value", None)
            if getattr(node, "loading", False):
                residency = "prefetching"
            elif int(node.id) in getattr(
                self.tree_cache, "ongoing_write_through", {}
            ):
                residency = "mirroring"
            elif value is not None and host_value is not None:
                residency = "dual_clean"
            elif value is not None:
                residency = "gpu_only"
            elif host_value is not None:
                residency = "cpu_only"
            else:
                continue
            page = self.controller.page_index.pages.get(handle)
            if page is None:
                from beliefkv.runtime.protocol import PhysicalResidency

                page = self.controller.page_index.register_page(
                    handle,
                    size_bytes=max(1, len(node.key) * self.config.kv_bytes_per_token),
                    residency=PhysicalResidency(residency),
                    radix_depth=depth,
                    parent=parent_handle,
                    sealed=True,
                    last_access_ms=float(getattr(node, "last_access_time", 0.0))
                    * 1000.0,
                )
            elif page.transfer_idle:
                from beliefkv.runtime.protocol import PhysicalResidency

                self.controller.page_index.set_parent(handle, parent_handle)
                self.controller.page_index.update_runtime_state(
                    handle,
                    residency=PhysicalResidency(residency),
                    radix_depth=depth,
                    engine_lock_ref=max(0, int(getattr(node, "lock_ref", 0))),
                    last_access_ms=float(
                        getattr(node, "last_access_time", 0.0)
                    )
                    * 1000.0,
                )
            else:
                self.controller.page_index.set_engine_lock(
                    handle,
                    max(0, int(getattr(node, "lock_ref", 0))),
                )

        for context_id, terminal_node in tuple(self._terminal_node_by_context.items()):
            if not self.controller.page_index.has_context(context_id):
                continue
            path: list[PageHandle] = []
            node = terminal_node
            seen: set[int] = set()
            while node is not None and node is not root:
                if id(node) in seen:
                    raise SGLangBackendError("Radix parent cycle detected")
                seen.add(id(node))
                handle = node_handles.get(id(node))
                if handle is None:
                    break
                path.append(handle)
                node = node.parent
            self.controller.page_index.bind_pages(
                context_id,
                self.controller.page_index.context_epoch(context_id),
                path,
                replace=True,
            )
        self.controller.page_index.assert_consistent()
        self.controller.notify_resource_state_changed()
        self._tree_full_rebuild_required = False
        getattr(self, "_dirty_radix_nodes", {}).clear()
        getattr(self, "_removed_radix_nodes", {}).clear()
        getattr(self, "_dirty_context_ids", set()).clear()
        self._tree_dirty = False

    def _sync_tree_incremental(self) -> None:
        root = self.tree_cache.root_node
        page_index = self.controller.page_index
        removed_nodes = dict(getattr(self, "_removed_radix_nodes", {}))
        dirty_nodes = dict(getattr(self, "_dirty_radix_nodes", {}))

        for node_id, _node in removed_nodes.items():
            handle = self.registry.current_handle(node_id)
            if handle is None:
                continue
            page = page_index.pages.get(handle)
            if page is not None and page.residency != PhysicalResidency.DEAD:
                if not page.transfer_idle:
                    self._mark_full_tree_rebuild()
                    return
                page_index.invalidate_page(handle)
            self.registry.remove(handle)

        nodes_with_depth = []
        for node_id, node in dirty_nodes.items():
            if node_id in removed_nodes or node is root:
                continue
            nodes_with_depth.append((self._radix_node_depth(node), node))
        nodes_with_depth.sort(key=lambda item: (item[0], int(item[1].id)))

        for depth, node in nodes_with_depth:
            parent = getattr(node, "parent", None)
            if parent is None:
                self._mark_full_tree_rebuild()
                return
            parent_handle = (
                None
                if parent is root
                else self.registry.current_handle(int(parent.id))
            )
            if parent is not root and parent_handle is None:
                self._mark_full_tree_rebuild()
                return
            previous = self.registry.current_handle(int(node.id))
            handle = self.registry.register(node)
            if previous is not None and previous != handle:
                previous_page = page_index.pages.get(previous)
                if (
                    previous_page is not None
                    and previous_page.residency != PhysicalResidency.DEAD
                ):
                    if not previous_page.transfer_idle:
                        self._mark_full_tree_rebuild()
                        return
                    page_index.invalidate_page(previous)
            residency = self._radix_node_residency(node)
            if residency is None:
                page = page_index.pages.get(handle)
                if page is not None and page.residency != PhysicalResidency.DEAD:
                    if not page.transfer_idle:
                        self._mark_full_tree_rebuild()
                        return
                    page_index.invalidate_page(handle)
                self.registry.remove(handle)
                continue
            page = page_index.pages.get(handle)
            if page is None:
                page_index.register_page(
                    handle,
                    size_bytes=max(
                        1, len(node.key) * self.config.kv_bytes_per_token
                    ),
                    residency=residency,
                    radix_depth=depth,
                    parent=parent_handle,
                    sealed=True,
                    last_access_ms=float(
                        getattr(node, "last_access_time", 0.0)
                    )
                    * 1000.0,
                )
            elif page.transfer_idle:
                page_index.set_parent(handle, parent_handle)
                page_index.update_runtime_state(
                    handle,
                    residency=residency,
                    radix_depth=depth,
                    engine_lock_ref=max(0, int(getattr(node, "lock_ref", 0))),
                    last_access_ms=float(
                        getattr(node, "last_access_time", 0.0)
                    )
                    * 1000.0,
                )
            else:
                page_index.set_engine_lock(
                    handle, max(0, int(getattr(node, "lock_ref", 0)))
                )

        self._rebind_dirty_contexts(root)
        self.controller.notify_resource_state_changed()
        self._dirty_radix_nodes.clear()
        self._removed_radix_nodes.clear()
        self._dirty_context_ids.clear()
        self._tree_dirty = bool(self._tree_full_rebuild_required)

    def _rebind_dirty_contexts(self, root: Any) -> None:
        page_index = self.controller.page_index
        for context_id in tuple(sorted(self._dirty_context_ids)):
            terminal_node = self._terminal_node_by_context.get(context_id)
            if terminal_node is None or not page_index.has_context(context_id):
                continue
            path: list[PageHandle] = []
            node = terminal_node
            seen: set[int] = set()
            while node is not None and node is not root:
                if id(node) in seen:
                    raise SGLangBackendError("Radix parent cycle detected")
                seen.add(id(node))
                handle = self.registry.current_handle(int(node.id))
                if handle is None:
                    break
                page = page_index.pages.get(handle)
                if page is None or page.residency == PhysicalResidency.DEAD:
                    break
                path.append(handle)
                node = getattr(node, "parent", None)
            page_index.bind_pages(
                context_id,
                page_index.context_epoch(context_id),
                path,
                replace=True,
            )

    def _radix_node_depth(self, node: Any) -> int:
        root = self.tree_cache.root_node
        depth = 0
        seen: set[int] = set()
        while node is not root:
            if node is None or id(node) in seen:
                raise SGLangBackendError("invalid Radix parent chain")
            seen.add(id(node))
            depth += 1
            node = getattr(node, "parent", None)
        return depth

    def _radix_node_residency(self, node: Any) -> PhysicalResidency | None:
        value = getattr(node, "value", None)
        host_value = getattr(node, "host_value", None)
        if getattr(node, "loading", False):
            return PhysicalResidency.PREFETCHING
        if int(node.id) in getattr(self.tree_cache, "ongoing_write_through", {}):
            return PhysicalResidency.MIRRORING
        if value is not None and host_value is not None:
            return PhysicalResidency.DUAL_CLEAN
        if value is not None:
            return PhysicalResidency.GPU_ONLY
        if host_value is not None:
            return PhysicalResidency.CPU_ONLY
        return None

    def _report_allocator_usage(self) -> None:
        self._ensure_allocator_radix_consistency(reason="scheduler_step")
        allocator = self.scheduler.token_to_kv_pool_allocator
        available = int(allocator.available_size())
        used_tokens = max(0, int(self.scheduler.max_total_num_tokens) - available)
        used_bytes = min(
            self.config.hbm_capacity_bytes,
            used_tokens * self.config.kv_bytes_per_token,
        )
        workflow_charges: dict[str, float] = {}
        running_batch = getattr(self.scheduler, "running_batch", None)
        running_requests = tuple(getattr(running_batch, "reqs", ()) or ())
        for req in running_requests:
            metadata = self._metadata(req)
            if metadata is None:
                continue
            fill_length = len(getattr(req, "fill_ids", ()))
            prefix_length = len(getattr(req, "prefix_indices", ()))
            private_tokens = max(0, fill_length - prefix_length)
            workflow_charges[metadata.root_workflow_id] = (
                workflow_charges.get(metadata.root_workflow_id, 0.0)
                + private_tokens * self.config.kv_bytes_per_token
            )
        engine_request_ids = {
            str(getattr(req, "rid", f"object:{id(req)}"))
            for req in running_requests
        }
        for req in tuple(getattr(self.scheduler, "waiting_queue", ()) or ()):
            engine_request_ids.add(str(getattr(req, "rid", f"object:{id(req)}")))
        chunked_req = getattr(self.scheduler, "chunked_req", None)
        if chunked_req is not None:
            engine_request_ids.add(
                str(getattr(chunked_req, "rid", f"object:{id(chunked_req)}"))
            )
        engine_request_ids.update(getattr(self, "_active_request_ids", set()))
        self.controller.report_hbm_usage(
            used_bytes, workflow_charges=workflow_charges
        )
        running_request_ids = {
            str(getattr(req, "rid", f"object:{id(req)}"))
            for req in running_requests
        }
        if chunked_req is not None:
            running_request_ids.add(
                str(getattr(chunked_req, "rid", f"object:{id(chunked_req)}"))
            )
        self.controller.report_engine_activity(
            len(engine_request_ids),
            running_request_count=len(running_request_ids),
        )

    def _ensure_allocator_radix_consistency(self, *, reason: str) -> None:
        scheduler = self.scheduler
        allocator = scheduler.token_to_kv_pool_allocator
        available_before = int(allocator.available_size())
        evictable = int(self.tree_cache.evictable_size())
        protected = max(0, int(getattr(self.tree_cache, "protected_size_", 0)))
        max_tokens = int(scheduler.max_total_num_tokens)
        if available_before + evictable + protected <= max_tokens:
            return

        overlap_tokens = self._claim_live_radix_indices(allocator, max_tokens)
        available_after = int(allocator.available_size())
        accounted_after = available_after + evictable + protected
        self.audit.emit(
            "allocator_radix_resynchronized",
            self._now_ms(),
            reason=reason,
            overlap_tokens=overlap_tokens,
            available_tokens_before=available_before,
            available_tokens_after=available_after,
            evictable_tokens=evictable,
            protected_tokens=protected,
            max_total_tokens=max_tokens,
        )
        if overlap_tokens <= 0 or accounted_after > max_tokens:
            raise SGLangBackendError(
                "allocator/Radix accounting diverged and could not be repaired",
                blocker_code=TransferBlockerCode.UNKNOWN_BACKEND,
            )

    def _claim_live_radix_indices(self, allocator: Any, max_tokens: int) -> int:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - SGLang requires torch
            raise SGLangBackendError(
                "torch is required to reconcile allocator/Radix state"
            ) from error

        page_size = int(getattr(allocator, "page_size", 1) or 1)
        if page_size != 1:
            raise SGLangBackendError(
                "allocator/Radix reconciliation only supports token-granular allocators"
            )

        live_values = []
        stack = list(getattr(self.tree_cache.root_node, "children", {}).values())
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            identity = id(node)
            if identity in seen:
                raise SGLangBackendError("Radix cycle during allocator reconciliation")
            seen.add(identity)
            value = getattr(node, "value", None)
            if value is not None and len(value):
                live_values.append(
                    torch.as_tensor(value).reshape(-1).to(dtype=torch.int64)
                )
            stack.extend(getattr(node, "children", {}).values())
        if not live_values:
            return 0
        live = torch.cat(live_values)
        unique_live = torch.unique(live)
        if len(unique_live) != len(live):
            raise SGLangBackendError(
                "multiple live Radix extents reference the same device index"
            )
        if int(unique_live.min().item()) < 0 or int(unique_live.max().item()) > max_tokens:
            raise SGLangBackendError("Radix contains an out-of-range device index")

        live_bitmap = torch.zeros(
            max_tokens + 1,
            dtype=torch.bool,
            device=unique_live.device,
        )
        live_bitmap[unique_live] = True
        overlap_tokens = 0
        for attribute in ("free_pages", "release_pages"):
            pages = getattr(allocator, attribute, None)
            if pages is None or not len(pages):
                continue
            mask = live_bitmap[pages.to(dtype=torch.int64)]
            overlap_tokens += int(mask.sum().item())
            if bool(mask.any().item()):
                setattr(allocator, attribute, pages[~mask])
        return overlap_tokens

    def _emit_transfer_telemetry(
        self, telemetry: TransferTelemetry, **extra_fields: Any
    ) -> None:
        fields = {
            "command_id": telemetry.command_id,
            "submit_ts_ms": telemetry.submit_ts_ms,
            "start_ts_ms": telemetry.start_ts_ms,
            "first_layer_ready_ts_ms": telemetry.first_layer_ready_ts_ms,
            "complete_ts_ms": telemetry.complete_ts_ms,
            "compute_wait_ms": telemetry.compute_wait_ms,
            "actual_bytes": telemetry.actual_bytes,
            "closure_bytes": telemetry.closure_bytes,
            "merged_operation_count": telemetry.merged_operation_count,
            "direction": telemetry.direction.value,
            "source_tier": telemetry.source_tier,
            "target_tier": telemetry.target_tier,
            "status": telemetry.status.value,
            "reason": telemetry.reason,
            "page_count": telemetry.page_count,
            "context_id": telemetry.context_id,
            "context_epoch": telemetry.context_epoch,
            "command_kind": telemetry.command_kind,
            "compute_phase": telemetry.compute_phase,
            "host_copy_state": telemetry.host_copy_state,
            "pinned_host": telemetry.pinned_host,
            "native_concurrent_bytes": telemetry.native_concurrent_bytes,
            "allocator_wait_ms": telemetry.allocator_wait_ms,
            "allocator_submit_ms": telemetry.allocator_submit_ms,
            "callback_overhead_ms": telemetry.callback_overhead_ms,
            "start_timestamp_semantics": telemetry.start_timestamp_semantics,
            **extra_fields,
        }
        observed_ts_ms = max(float(self._now_ms()), telemetry.complete_ts_ms)
        self.audit.emit("transfer_telemetry", observed_ts_ms, **fields)
        transfer_log = getattr(self, "transfer_telemetry_log", None)
        if transfer_log is not None:
            transfer_log.emit("transfer_telemetry", observed_ts_ms, **fields)

    def _lock_service_diagnostics(
        self,
        *,
        now_ms: float,
    ) -> LockServiceDiagnostics | None:
        snapshot = self._lock_provenance_extents()
        if snapshot is None:
            return None
        extents, path_error_count = snapshot
        ledger = getattr(self, "_lock_service_ledger", None)
        if ledger is None:
            ledger = RequestServiceLedger()
            self._lock_service_ledger = ledger
        return ledger.summarize(
            extents,
            now_ms=now_ms,
            request_path_error_count=path_error_count,
        )

    def _lock_provenance_extents(
        self,
    ) -> tuple[tuple[LockedExtentAttribution, ...], int] | None:
        """Map authoritative lock refs to the tagged running request paths."""

        page_index = getattr(getattr(self, "controller", None), "page_index", None)
        provider = getattr(page_index, "engine_locked_gpu_pages", None)
        if not callable(provider):
            return None
        locked_pages = tuple(provider())
        locked_by_handle = {page.handle: page for page in locked_pages}
        blockers: dict[PageHandle, set[str]] = {
            handle: set() for handle in locked_by_handle
        }
        scheduler = getattr(self, "scheduler", None)
        running_batch = getattr(scheduler, "running_batch", None)
        requests = list(getattr(running_batch, "reqs", ()) or ())
        chunked_req = getattr(scheduler, "chunked_req", None)
        if chunked_req is not None:
            requests.append(chunked_req)
        unique_requests: dict[str, Any] = {}
        for req in requests:
            request_id = str(getattr(req, "rid", f"object:{id(req)}"))
            unique_requests.setdefault(request_id, req)

        root = getattr(getattr(self, "tree_cache", None), "root_node", None)
        registry = getattr(self, "registry", None)
        path_error_count = 0
        for request_id, req in unique_requests.items():
            if self._metadata(req) is None:
                continue
            node = getattr(req, "last_node", None)
            if node is None:
                continue
            seen: set[int] = set()
            while node is not None and node is not root:
                if id(node) in seen:
                    path_error_count += 1
                    break
                seen.add(id(node))
                try:
                    handle = (
                        registry.current_handle(int(node.id))
                        if registry is not None
                        else None
                    )
                except Exception:
                    path_error_count += 1
                    break
                if handle is None:
                    path_error_count += 1
                    break
                if handle in blockers:
                    blockers[handle].add(request_id)
                node = getattr(node, "parent", None)
            if node is None and root is not None:
                path_error_count += 1

        extents = tuple(
            LockedExtentAttribution(
                handle=handle,
                size_bytes=page.size_bytes,
                engine_lock_ref=page.engine_lock_ref,
                blocker_request_ids=tuple(sorted(blockers[handle])),
            )
            for handle, page in sorted(locked_by_handle.items())
        )
        return extents, path_error_count

    @staticmethod
    def _unavailable_lock_service_fields() -> dict[str, object]:
        fields = {
            "engine_lock_ref_gpu_bytes": None,
            "engine_lock_attributed_gpu_bytes": None,
            "engine_lock_fully_attributed_gpu_bytes": None,
            "engine_lock_partially_attributed_gpu_bytes": None,
            "engine_lock_unattributed_gpu_bytes": None,
            "engine_lock_attribution_coverage": None,
            "engine_lock_full_attribution_coverage": None,
            "engine_lock_extent_count": None,
            "engine_lock_attributed_extent_count": None,
            "engine_lock_blocker_request_count": None,
            "engine_lock_request_path_error_count": None,
            "engine_lock_ref_mismatch_extent_count": None,
            "engine_lock_logical_request_bytes": None,
            "engine_lock_provenance_scope": "unavailable",
            "engine_lock_service_evidence": "completed_gpu_batch",
        }
        for suffix in ("100ms", "500ms"):
            for prefix in (
                "lock_recently_served_request_count",
                "lock_not_served_request_count",
                "lock_warming_request_count",
                "lock_unknown_request_count",
                "lock_recently_served_gpu_bytes",
                "locked_but_not_served_gpu_bytes",
                "lock_warming_gpu_bytes",
                "lock_unknown_gpu_bytes",
                "lock_not_served_logical_request_bytes",
            ):
                fields[f"{prefix}_{suffix}"] = None
        return fields

    def _emit_resource_snapshot(self, *, force: bool) -> None:
        now_ms = float(self._now_ms())
        last_ts = getattr(self, "_last_resource_telemetry_ms", None)
        if (
            not force
            and last_ts is not None
            and now_ms - last_ts < self.config.resource_telemetry_interval_ms
        ):
            return
        observation = self._runtime_resource_observation(now_ms=now_ms)
        page_index = self.controller.page_index
        breakdown_provider = getattr(
            page_index, "physical_kv_state_breakdown", None
        )
        breakdown = breakdown_provider() if callable(breakdown_provider) else None
        page_index_gpu_bytes = (
            breakdown.gpu_bytes if breakdown is not None else page_index.gpu_bytes
        )
        page_index_cpu_bytes = (
            breakdown.cpu_bytes if breakdown is not None else page_index.cpu_bytes
        )
        lock_service = self._lock_service_diagnostics(now_ms=now_ms)
        lock_service_fields = (
            lock_service.to_audit_fields()
            if lock_service is not None
            else self._unavailable_lock_service_fields()
        )
        if lock_service is not None:
            self._lock_service_snapshot_count = (
                getattr(self, "_lock_service_snapshot_count", 0) + 1
            )
            peaks = getattr(self, "_lock_service_peak_bytes", None)
            if peaks is None:
                peaks = {"100ms": 0, "500ms": 0}
                self._lock_service_peak_bytes = peaks
            for window in lock_service.windows:
                suffix = f"{int(window.window_ms)}ms"
                peaks[suffix] = max(
                    peaks.get(suffix, 0),
                    window.locked_but_not_served_unique_bytes,
                )
        self.audit.emit(
            "resource_snapshot",
            now_ms,
            hbm_capacity_bytes=observation.hbm_capacity_bytes,
            configured_hbm_capacity_bytes=self.config.hbm_capacity_bytes,
            hbm_used_bytes=observation.hbm_used_bytes,
            hbm_free_bytes=(
                observation.hbm_capacity_bytes - observation.hbm_used_bytes
            ),
            host_capacity_bytes=observation.host_capacity_bytes,
            host_used_bytes=observation.host_used_bytes,
            host_free_bytes=observation.host_free_bytes,
            page_index_gpu_bytes=page_index_gpu_bytes,
            page_index_cpu_bytes=page_index_cpu_bytes,
            untracked_allocator_delta_bytes=max(
                0, observation.hbm_used_bytes - page_index_gpu_bytes
            ),
            restore_lease_reserved_bytes=(
                getattr(self, "_restore_leases", None).reserved_bytes
                if getattr(self, "_restore_leases", None) is not None
                else 0
            ),
            active_restore_lease_count=len(
                getattr(self, "_restore_leases", None).active()
                if getattr(self, "_restore_leases", None) is not None
                else ()
            ),
            engine_locked_gpu_bytes=(
                breakdown.engine_locked_bytes if breakdown is not None else None
            ),
            closure_blocked_gpu_bytes=(
                breakdown.closure_blocked_bytes if breakdown is not None else None
            ),
            migratable_gpu_bytes=(
                breakdown.migratable_bytes if breakdown is not None else None
            ),
            dual_resident_gpu_bytes=(
                breakdown.dual_resident_bytes if breakdown is not None else None
            ),
            kv_state_breakdown_scope=(
                "physical_radix_closure" if breakdown is not None else "unavailable"
            ),
            inflight_command_count=len(self.controller.inflight_command_ids),
            pcie_utilization=(
                self.controller.signals.pcie_utilization
                if getattr(self.controller, "_pcie_utilization_observed", False)
                else None
            ),
            copy_engine_utilization=None,
            gpu_compute_utilization=(
                self.controller.signals.gpu_compute_utilization
                if getattr(self.controller, "_gpu_compute_utilization_observed", False)
                else None
            ),
            engine_request_count=getattr(self.controller, "_engine_request_count", None),
            running_request_count=getattr(self.controller, "_running_request_count", None),
            **lock_service_fields,
        )
        self._last_resource_telemetry_ms = now_ms
        self._write_latest_runtime_summary(now_ms=now_ms, force=False)

    def _runtime_resource_observation(
        self, *, now_ms: float | None = None
    ) -> RuntimeResourceObservation:
        allocator = self.scheduler.token_to_kv_pool_allocator
        hbm_free_tokens = max(0, int(allocator.available_size()))
        hbm_capacity_tokens = max(0, int(self.scheduler.max_total_num_tokens))
        hbm_used_tokens = max(0, hbm_capacity_tokens - hbm_free_tokens)
        hbm_capacity_bytes = hbm_capacity_tokens * self.config.kv_bytes_per_token

        host_pool = getattr(self.tree_cache, "token_to_kv_pool_host", None)
        host_capacity_tokens = int(getattr(host_pool, "size", 0) or 0)
        host_free_tokens = (
            max(0, int(host_pool.available_size()))
            if host_pool is not None and hasattr(host_pool, "available_size")
            else 0
        )
        host_used_tokens = max(0, host_capacity_tokens - host_free_tokens)
        host_bytes_per_token = int(
            getattr(host_pool, "size_per_token", self.config.kv_bytes_per_token)
            or self.config.kv_bytes_per_token
        )
        return RuntimeResourceObservation(
            ts_ms=float(self._now_ms() if now_ms is None else now_ms),
            hbm_capacity_bytes=hbm_capacity_bytes,
            hbm_used_bytes=min(
                hbm_capacity_bytes,
                hbm_used_tokens * self.config.kv_bytes_per_token,
            ),
            host_capacity_bytes=host_capacity_tokens * host_bytes_per_token,
            host_used_bytes=host_used_tokens * host_bytes_per_token,
            host_free_bytes=host_free_tokens * host_bytes_per_token,
            pcie_utilization=(
                self.controller.signals.pcie_utilization
                if getattr(self.controller, "_pcie_utilization_observed", False)
                else None
            ),
            gpu_compute_utilization=(
                self.controller.signals.gpu_compute_utilization
                if getattr(
                    self.controller, "_gpu_compute_utilization_observed", False
                )
                else None
            ),
            source="sglang_allocator_safe_point",
        )

    def _maybe_record_policy_snapshot(
        self,
        observation: RuntimeResourceObservation,
    ) -> None:
        worker = getattr(self, "joint_shadow_worker", None)
        if worker is not None and worker.supports_incremental_delta:
            self._maybe_record_incremental_policy_snapshot(observation, worker)
            return
        self._maybe_record_policy_snapshot_legacy(observation)

    def _maybe_record_incremental_policy_snapshot(
        self,
        observation: RuntimeResourceObservation,
        worker: LatestWinsJointPlanWorker,
    ) -> None:
        snapshot_log = getattr(self, "policy_snapshot_log", None)
        snapshot_enabled = snapshot_log is not None and snapshot_log.enabled
        result = worker.latest(
            after_sequence=self._last_joint_shadow_result_sequence
        )
        additional_runnable = self._policy_runtime_runnable(observation.ts_ms)
        runnable_signature = self._joint_shadow_runnable_signature(
            additional_runnable
        )
        control_state = dict(
            self.controller.policy_control_state(observation.ts_ms)
        )
        restore_lease_index = getattr(self, "_restore_leases", None)
        liveness_tracker = getattr(
            self, "_persistent_liveness_revisions", None
        )
        if liveness_tracker is None:
            liveness_tracker = PersistentLivenessRevisionTracker()
            self._persistent_liveness_revisions = liveness_tracker
        liveness = liveness_tracker.observe(
            obligations=self._restore_obligation_index().all(),
            leases=(restore_lease_index.all() if restore_lease_index else ()),
            graces=tuple(
                getattr(self, "_restore_service_grace_by_request", {}).values()
            ),
        )
        control_state["persistent_liveness"] = liveness.to_dict()
        action_frontier = self.controller.action_frontier_observer
        control_state["action_frontier"] = {
            "revision": action_frontier.revision,
            # Coverage is only consumed by a captured delta and is not part of
            # any change signature; it is computed lazily at capture time.
            "coverage": None,
            "observer_error_count": getattr(
                self.controller, "_action_frontier_observer_errors", 0
            ),
        }
        stamp = JointShadowStateStamp(
            graph_version=self.controller.graph.graph_version,
            consumer_version=self.controller.data_consumers.version,
            event_sequence=self.controller.runtime_event_sequence,
            page_revision=self.controller.page_index.revision,
            topology_revision=self.controller.page_index.topology_revision,
            fairness_revision=self.controller.fairness.revision,
            transfer_epoch=int(control_state.get("transfer_epoch", 0)),
            runnable_signature=runnable_signature,
            hbm_used_bytes=observation.hbm_used_bytes,
            host_free_bytes=observation.host_free_bytes,
            obligation_revision=liveness.obligation_revision,
            lease_revision=liveness.lease_revision,
            grace_revision=liveness.grace_revision,
            parser_frontier_revision=action_frontier.revision,
        )
        elapsed = (
            float("inf")
            if self._last_policy_snapshot_ms is None
            else observation.ts_ms - self._last_policy_snapshot_ms
        )
        watchdog_due = bool(additional_runnable) and elapsed >= (
            self.config.reference_policy_snapshot_min_interval_ms
        )
        # Cheap structural pre-check: every field that can feed the full
        # structural/physical signatures is represented here. When none of
        # them changed, no snapshot capture can fire and no new shadow result
        # is pending, so the expensive signature construction and coverage
        # computation can be skipped without changing observable behavior.
        cheap_signature = (
            stamp.graph_version,
            stamp.consumer_version,
            stamp.event_sequence,
            stamp.page_revision,
            stamp.topology_revision,
            stamp.fairness_revision,
            stamp.transfer_epoch,
            stamp.runnable_signature,
            stamp.parser_frontier_revision,
            stamp.obligation_revision,
            stamp.lease_revision,
            stamp.grace_revision,
            observation.hbm_used_bytes
            // self.config.reference_policy_hbm_bucket_bytes,
            observation.host_used_bytes,
            observation.host_free_bytes,
            observation.pcie_utilization,
            observation.gpu_compute_utilization,
            self.controller.transfer_backlog_bytes(),
        )
        if (
            cheap_signature
            == getattr(self, "_last_policy_snapshot_cheap_signature", None)
            and not watchdog_due
            and result is None
        ):
            self._joint_shadow_counts["incremental_snapshot_skipped"] += 1
            return
        self._last_policy_snapshot_cheap_signature = cheap_signature
        # Queue/graph transitions publish immediately. Fairness service and
        # allocator progress change on nearly every decode quantum, so they
        # belong to the interval-gated progress signature below.
        structural_signature: tuple[object, ...] = (
            stamp.graph_version,
            stamp.consumer_version,
            stamp.transfer_epoch,
            stamp.obligation_revision,
            stamp.lease_revision,
            stamp.grace_revision,
            stamp.parser_frontier_revision,
            runnable_signature,
            tuple(
                (
                    workflow_id,
                    tuple(sorted(state.items())),
                )
                for workflow_id, state in sorted(
                    control_state.get("transitions", {}).items()
                )
                if isinstance(state, Mapping)
            ),
        )
        physical_signature: tuple[object, ...] = (
            stamp.page_revision,
            stamp.topology_revision,
            stamp.fairness_revision,
            tuple(
                (
                    workflow_id,
                    account.weight,
                    account.attained_service_ms,
                    account.dispatch_count,
                )
                for workflow_id, account in sorted(
                    self.controller.fairness.accounts.items()
                )
            ),
            self.controller.transfer_backlog_bytes(),
            observation.hbm_used_bytes
            // self.config.reference_policy_hbm_bucket_bytes,
            observation.host_used_bytes,
            observation.host_free_bytes,
            observation.pcie_utilization,
            observation.gpu_compute_utilization,
        )
        structural_changed = (
            structural_signature
            != self._last_policy_snapshot_structural_signature
        )
        physical_changed = (
            physical_signature != self._last_policy_snapshot_physical_signature
        )
        changed_snapshot_due = (
            structural_changed
            or (
                physical_changed
                and elapsed
                >= self.config.reference_policy_snapshot_min_interval_ms
            )
            or watchdog_due
        )
        trigger_parts: list[str] = []
        if changed_snapshot_due and structural_changed:
            trigger_parts.append("graph_or_queue")
        if changed_snapshot_due and physical_changed:
            trigger_parts.append("physical_pressure_or_service")
        if watchdog_due and not structural_changed and not physical_changed:
            trigger_parts.append("joint_watchdog")
        if result is not None:
            trigger_parts.append("joint_validation")
        trigger = "+".join(trigger_parts) or "joint_validation"

        if changed_snapshot_due:
            capture_started_ns = time.perf_counter_ns()
            try:
                event_delta = self.controller.runtime_events_since(
                    self._shadow_event_sequence
                )
                if event_delta.full_rebuild_required:
                    raise RuntimeError(
                        "runtime event journal gap; shadow rebuild is fail-closed"
                    )
                page_delta = self.controller.page_index.replica_delta_since(
                    self._shadow_page_revision
                )
                telemetry_history = self.controller.transfer_telemetry_history
                if self._shadow_telemetry_sequence > len(telemetry_history):
                    raise RuntimeError("transfer telemetry history moved backwards")
                telemetry = tuple(
                    telemetry_history[self._shadow_telemetry_sequence :]
                )
                urgent_d2h, urgent_h2d = self.controller.transfer_backlog_bytes()
                published_observation = replace(
                    observation,
                    urgent_d2h_bytes=urgent_d2h,
                    urgent_h2d_bytes=urgent_h2d,
                )
                control_state["action_frontier"] = {
                    **control_state["action_frontier"],
                    "coverage": action_frontier.coverage().to_dict(),
                }
                fairness_accounts = tuple(
                    WorkflowFairnessReplica(
                        workflow_id=workflow_id,
                        weight=account.weight,
                        attained_service_ms=account.attained_service_ms,
                        virtual_runtime_ms=account.virtual_runtime,
                        dispatch_count=account.dispatch_count,
                    )
                    for workflow_id, account in sorted(
                        self.controller.fairness.accounts.items()
                    )
                )
                delta = JointShadowDelta(
                    event_from_sequence=event_delta.from_sequence,
                    event_to_sequence=event_delta.to_sequence,
                    runtime_events=event_delta.events,
                    page_delta=page_delta,
                    observation=published_observation,
                    runnable_frontier=additional_runnable,
                    fairness_accounts=fairness_accounts,
                    external_workflow_charges=(
                        self.controller.external_workflow_memory_charges()
                    ),
                    control_state=copy.deepcopy(control_state),
                    transfer_telemetry=telemetry,
                    capabilities=self._policy_capabilities(),
                    stamp=stamp,
                    trigger=trigger,
                    captured_monotonic_ms=time.monotonic_ns() / 1_000_000.0,
                    frontier_predictions=dict(
                        self._last_frontier_predictions or {}
                    ),
                )
                submission = worker.submit_delta(delta)
            except Exception as error:
                self._joint_shadow_counts["submission_failed"] += 1
                self.audit.emit(
                    "joint_plan_shadow_submit_failed",
                    observation.ts_ms,
                    error=f"{type(error).__name__}: {error}",
                    trigger=trigger,
                    application_connected=self.config.joint_policy_enabled,
                )
            else:
                capture_ms = (
                    time.perf_counter_ns() - capture_started_ns
                ) / 1_000_000.0
                self._shadow_event_sequence = event_delta.to_sequence
                self._shadow_page_revision = page_delta.to_revision
                self._shadow_telemetry_sequence = len(telemetry_history)
                self._joint_shadow_counts["submitted"] += 1
                if submission.replaced_sequence is not None:
                    self._joint_shadow_counts["pending_replaced"] += 1
                self._joint_shadow_timing_samples[
                    "safe_point_delta_capture_ms"
                ].append(capture_ms)
                self._joint_shadow_timing_samples[
                    "snapshot_enqueue_ms"
                ].append(submission.enqueue_ms)
                worker_stats = worker.stats()
                self.audit.emit(
                    "joint_plan_shadow_delta_enqueued",
                    observation.ts_ms,
                    worker_sequence=submission.sequence,
                    trigger=trigger,
                    event_from_sequence=event_delta.from_sequence,
                    event_to_sequence=event_delta.to_sequence,
                    event_count=len(event_delta.events),
                    page_from_revision=page_delta.from_revision,
                    page_to_revision=page_delta.to_revision,
                    changed_page_count=len(page_delta.changed_handles),
                    full_page_record_count=len(page_delta.pages),
                    physical_state_patch_count=len(page_delta.page_states),
                    changed_context_count=len(page_delta.contexts),
                    telemetry_count=len(telemetry),
                    runnable_request_count=len(additional_runnable),
                    safe_point_delta_capture_ms=capture_ms,
                    snapshot_enqueue_ms=submission.enqueue_ms,
                    replaced_sequence=submission.replaced_sequence,
                    worker_pending_count=worker_stats.pending_count,
                    worker_busy=worker_stats.busy,
                    application_connected=self.config.joint_policy_enabled,
                )
            finally:
                self._last_policy_snapshot_structural_signature = (
                    structural_signature
                )
                self._last_policy_snapshot_physical_signature = physical_signature
                self._last_policy_snapshot_hbm_bucket = (
                    observation.hbm_used_bytes
                    // self.config.reference_policy_hbm_bucket_bytes
                )
                self._last_policy_snapshot_ms = observation.ts_ms

        if result is None:
            return
        self._last_joint_shadow_result_sequence = result.sequence
        policy_input = result.policy_input
        if policy_input is None:
            self._joint_shadow_counts["worker_failed"] += 1
            self.audit.emit(
                "joint_plan_shadow_failed",
                observation.ts_ms,
                worker_sequence=result.sequence,
                source_snapshot_id=result.snapshot_id,
                error=result.error or "worker returned no PolicyInput",
                application_connected=self.config.joint_policy_enabled,
            )
            return

        sequence = 0
        trace_enqueue_ms = 0.0
        persist_interval_ms = (
            self.config.reference_policy_snapshot_persist_interval_ms
        )
        persist_due = bool(
            getattr(self, "_last_persisted_policy_snapshot_ms", None) is None
            or persist_interval_ms == 0
            or observation.ts_ms
            - float(getattr(self, "_last_persisted_policy_snapshot_ms", 0.0))
            >= persist_interval_ms
        )
        if snapshot_enabled and persist_due:
            trace_started_ns = time.perf_counter_ns()
            dropped_before = snapshot_log.dropped_count
            try:
                sequence = snapshot_log.emit(
                    policy_input,
                    trigger=result.trigger,
                )
            except Exception as error:
                self.audit.emit(
                    "policy_snapshot_failed",
                    observation.ts_ms,
                    trigger=result.trigger,
                    error=f"{type(error).__name__}: {error}",
                )
            else:
                trace_enqueue_ms = (
                    time.perf_counter_ns() - trace_started_ns
                ) / 1_000_000.0
                if sequence:
                    self._last_persisted_policy_snapshot_ms = observation.ts_ms
                elif snapshot_log.dropped_count > dropped_before:
                    self._joint_shadow_counts[
                        "policy_snapshot_writer_dropped"
                    ] += 1
                self._joint_shadow_timing_samples[
                    "snapshot_trace_enqueue_ms"
                ].append(trace_enqueue_ms)
        elif snapshot_enabled:
            self._joint_shadow_counts["policy_snapshot_sampled_out"] += 1
        self._joint_shadow_timing_samples["snapshot_build_ms"].append(
            result.snapshot_build_ms
        )
        self._joint_shadow_timing_samples.setdefault(
            "snapshot_delta_apply_ms", deque(maxlen=65_536)
        ).append(result.snapshot_delta_apply_ms)
        self._joint_shadow_timing_samples.setdefault(
            "snapshot_materialize_ms", deque(maxlen=65_536)
        ).append(result.snapshot_materialize_ms)
        assembler = worker.assembler
        stats = assembler.last_stats if assembler is not None else None
        if sequence:
            self.audit.emit(
                "policy_snapshot_recorded",
                observation.ts_ms,
                snapshot_sequence=sequence,
                snapshot_id=policy_input.snapshot_id,
                trigger=result.trigger,
                build_ms=result.snapshot_build_ms,
                delta_apply_ms=result.snapshot_delta_apply_ms,
                materialize_ms=result.snapshot_materialize_ms,
                safe_point_build_ms=0.0,
                trace_enqueue_ms=trace_enqueue_ms,
                runnable_request_count=len(policy_input.runnable_frontier),
                physical_bundle_count=len(policy_input.physical_kv.bundles),
                tracked_hbm_bytes=(
                    stats.tracked_hbm_bytes if stats is not None else None
                ),
                untracked_hbm_bytes=(
                    stats.untracked_hbm_bytes if stats is not None else None
                ),
            )
        extra_strict, extra_readset = self._joint_shadow_stamp_conflicts(
            result,
            current=stamp,
            observation=observation,
        )
        self._record_joint_shadow_result(
            result,
            policy_input=policy_input,
            observation=observation,
            snapshot_sequence=sequence,
            snapshot_build_ms=result.snapshot_build_ms,
            validation_state_changed=bool(extra_strict or extra_readset),
            extra_strict_reasons=extra_strict,
            coarse_stamp_reasons=extra_readset,
            current_runnable=additional_runnable,
            current_control_state=control_state,
        )

    @staticmethod
    def _joint_shadow_runnable_signature(
        runnable: tuple[RunnableInvocation, ...],
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                item.request_id,
                item.workflow_id,
                item.invocation_id,
                item.context_id,
                item.context_epoch,
                item.submitted_ts_ms,
                item.startup_bytes,
                item.causal_class,
                item.program_id,
            )
            for item in runnable
        )

    @staticmethod
    def _joint_shadow_stamp_conflicts(
        result: JointShadowResult,
        *,
        current: JointShadowStateStamp,
        observation: RuntimeResourceObservation,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        source = result.state_stamp
        plan = result.plan
        if source is None or plan is None:
            return ("missing_source_stamp",), ("missing_source_stamp",)
        strict: list[str] = []
        readset: list[str] = []
        if source.graph_version != current.graph_version:
            strict.append("graph_version")
            readset.append("graph_component_changed")
        if source.topology_revision != current.topology_revision:
            strict.append("topology_version")
            if plan.residency:
                readset.append("physical_topology_changed")
        if (
            source.page_revision != current.page_revision
            or source.hbm_used_bytes != current.hbm_used_bytes
        ):
            strict.append("allocator_version")
            if plan.residency or plan.admissions:
                readset.append("physical_allocator_changed")
        if source.runnable_signature != current.runnable_signature:
            readset.append("workflow_frontier_changed")
        if source.fairness_revision != current.fairness_revision:
            readset.append("fairness_revision_changed")
        if source.transfer_epoch != current.transfer_epoch and plan.residency:
            readset.append("transfer_epoch")
        if source.host_free_bytes != current.host_free_bytes and plan.residency:
            readset.append("host_headroom_changed")
        if observation.ts_ms > plan.read_set.expires_at_ms:
            readset.append("plan_expired")
        return tuple(sorted(set(strict))), tuple(sorted(set(readset)))

    def _maybe_record_policy_snapshot_legacy(
        self,
        observation: RuntimeResourceObservation,
    ) -> None:
        snapshot_log = getattr(self, "policy_snapshot_log", None)
        snapshot_enabled = snapshot_log is not None and snapshot_log.enabled
        worker = getattr(self, "joint_shadow_worker", None)
        if not snapshot_enabled and worker is None:
            return
        result = (
            worker.latest(after_sequence=self._last_joint_shadow_result_sequence)
            if worker is not None
            else None
        )
        additional_runnable = self._policy_runtime_runnable(observation.ts_ms)
        pending_ids = tuple(
            item.request_id for item in self.controller.admission.pending_requests()
        )
        reserved_ids = tuple(
            item.request_id for item in self.controller.admission.reserved_requests()
        )
        structural_signature: tuple[object, ...] = (
            self.controller.graph.graph_version,
            self.controller.data_consumers.version,
            self.controller.admission.revision,
            tuple(
                (workflow_id, account.weight)
                for workflow_id, account in sorted(
                    self.controller.admission.fairness.accounts.items()
                )
            ),
            self.controller.policy_control_state(observation.ts_ms),
            pending_ids,
            reserved_ids,
            tuple(
                (
                    item.request_id,
                    item.workflow_id,
                    item.invocation_id,
                    item.context_id,
                    item.context_epoch,
                    item.submitted_ts_ms,
                    item.startup_bytes,
                    item.causal_class,
                    item.program_id,
                )
                for item in additional_runnable
            ),
        )
        physical_signature: tuple[object, ...] = (
            self.controller.page_index.revision,
            self.controller.page_index.topology_revision,
            self.controller.transfer_backlog_bytes(),
            observation.hbm_used_bytes
            // self.config.reference_policy_hbm_bucket_bytes,
            observation.host_used_bytes,
            observation.host_free_bytes,
            observation.pcie_utilization,
            observation.gpu_compute_utilization,
        )
        structural_changed = (
            structural_signature
            != self._last_policy_snapshot_structural_signature
        )
        physical_changed = (
            physical_signature != self._last_policy_snapshot_physical_signature
        )
        elapsed = (
            float("inf")
            if self._last_policy_snapshot_ms is None
            else observation.ts_ms - self._last_policy_snapshot_ms
        )
        watchdog_due = (
            worker is not None
            and bool(pending_ids or reserved_ids or additional_runnable)
            and elapsed
            >= self.config.reference_policy_snapshot_min_interval_ms
        )
        changed_snapshot_due = (
            structural_changed
            or (
                physical_changed
                and elapsed
                >= self.config.reference_policy_snapshot_min_interval_ms
            )
            or watchdog_due
        )
        if not changed_snapshot_due and result is None:
            return

        trigger_parts = []
        if changed_snapshot_due and structural_changed:
            trigger_parts.append("graph_or_queue")
        if changed_snapshot_due and physical_changed:
            trigger_parts.append("physical_or_pressure")
        if watchdog_due and not structural_changed and not physical_changed:
            trigger_parts.append("joint_watchdog")
        if result is not None:
            trigger_parts.append("joint_validation")
        trigger = "+".join(trigger_parts)
        started_ns = time.perf_counter_ns()
        try:
            policy_input = self.controller.build_policy_input(
                observation,
                additional_runnable=additional_runnable,
                capabilities=self._policy_capabilities(),
            )
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            if changed_snapshot_due:
                self.audit.emit(
                    "policy_snapshot_failed",
                    observation.ts_ms,
                    trigger=trigger,
                    error=error_text,
                )
            if result is not None:
                self._last_joint_shadow_result_sequence = result.sequence
                self._joint_shadow_counts["validation_snapshot_failed"] += 1
                self.audit.emit(
                    "joint_plan_shadow_validation_failed",
                    observation.ts_ms,
                    worker_sequence=result.sequence,
                    source_snapshot_id=result.snapshot_id,
                    error=error_text,
                    application_connected=self.config.joint_policy_enabled,
                )
        else:
            build_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            if worker is not None:
                self._joint_shadow_timing_samples["snapshot_build_ms"].append(
                    build_ms
                )
            sequence = 0
            snapshot_error: str | None = None
            snapshot_trace_enqueue_ms = 0.0
            validation_state_changed = structural_changed or physical_changed
            should_persist_snapshot = changed_snapshot_due or (
                result is not None and validation_state_changed
            )
            if snapshot_enabled and should_persist_snapshot:
                try:
                    trace_enqueue_started_ns = time.perf_counter_ns()
                    sequence = snapshot_log.emit(policy_input, trigger=trigger)
                    snapshot_trace_enqueue_ms = (
                        time.perf_counter_ns() - trace_enqueue_started_ns
                    ) / 1_000_000.0
                except Exception as error:
                    snapshot_error = f"{type(error).__name__}: {error}"
                    self.audit.emit(
                        "policy_snapshot_failed",
                        observation.ts_ms,
                        trigger=trigger,
                        error=snapshot_error,
                    )
                else:
                    if worker is not None:
                        self._joint_shadow_timing_samples[
                            "snapshot_trace_enqueue_ms"
                        ].append(snapshot_trace_enqueue_ms)
            stats = self.controller.policy_snapshot_builder.last_stats
            if sequence:
                self.audit.emit(
                    "policy_snapshot_recorded",
                    observation.ts_ms,
                    snapshot_sequence=sequence,
                    snapshot_id=policy_input.snapshot_id,
                    trigger=trigger,
                    build_ms=build_ms,
                    trace_enqueue_ms=snapshot_trace_enqueue_ms,
                    runnable_request_count=len(policy_input.runnable_frontier),
                    physical_bundle_count=len(policy_input.physical_kv.bundles),
                    tracked_hbm_bytes=(
                        stats.tracked_hbm_bytes if stats is not None else None
                    ),
                    untracked_hbm_bytes=(
                        stats.untracked_hbm_bytes if stats is not None else None
                    ),
                )
            if worker is not None and changed_snapshot_due:
                if snapshot_error is not None:
                    self._joint_shadow_counts["submission_skipped_trace_error"] += 1
                else:
                    try:
                        submission = worker.submit(policy_input)
                    except Exception as error:
                        self._joint_shadow_counts["submission_failed"] += 1
                        self.audit.emit(
                            "joint_plan_shadow_submit_failed",
                            observation.ts_ms,
                            snapshot_id=policy_input.snapshot_id,
                            error=f"{type(error).__name__}: {error}",
                            application_connected=self.config.joint_policy_enabled,
                        )
                    else:
                        self._joint_shadow_counts["submitted"] += 1
                        if submission.replaced_sequence is not None:
                            self._joint_shadow_counts["pending_replaced"] += 1
                        self._joint_shadow_timing_samples[
                            "snapshot_enqueue_ms"
                        ].append(submission.enqueue_ms)
                        worker_stats = worker.stats()
                        self.audit.emit(
                            "joint_plan_shadow_enqueued",
                            observation.ts_ms,
                            worker_sequence=submission.sequence,
                            snapshot_sequence=sequence,
                            snapshot_id=submission.snapshot_id,
                            trigger=trigger,
                            snapshot_build_ms=build_ms,
                            snapshot_trace_enqueue_ms=(
                                snapshot_trace_enqueue_ms
                            ),
                            snapshot_enqueue_ms=submission.enqueue_ms,
                            replaced_sequence=submission.replaced_sequence,
                            worker_pending_count=worker_stats.pending_count,
                            worker_busy=worker_stats.busy,
                            application_connected=self.config.joint_policy_enabled,
                        )
            if result is not None:
                self._record_joint_shadow_result(
                    result,
                    policy_input=policy_input,
                    observation=observation,
                    snapshot_sequence=sequence,
                    snapshot_build_ms=build_ms,
                    validation_state_changed=validation_state_changed,
                )
        finally:
            if changed_snapshot_due:
                # A broken shadow observer must not retry unchanged state every tick.
                self._last_policy_snapshot_structural_signature = structural_signature
                self._last_policy_snapshot_physical_signature = physical_signature
                self._last_policy_snapshot_hbm_bucket = (
                    observation.hbm_used_bytes
                    // self.config.reference_policy_hbm_bucket_bytes
                )
                self._last_policy_snapshot_ms = observation.ts_ms

    def _joint_plan_current_state(
        self,
        result: JointShadowResult,
        *,
        source: PolicyInput,
        observation: RuntimeResourceObservation,
        current_runnable: tuple[RunnableInvocation, ...],
        current_control_state: Mapping[str, object],
        strict_global_reasons: tuple[str, ...],
    ) -> JointPlanCurrentState:
        plan = result.plan
        if plan is None:
            raise ValueError("cannot validate a missing joint plan")
        invocation_ids = set(plan.read_set.invocation_fingerprints)
        invocation_ids.update(
            item.invocation_id for item in current_runnable
        )
        invocation_snapshots = {
            invocation_id: self.controller.graph.invocation_snapshot(
                invocation_id
            )
            for invocation_id in sorted(invocation_ids)
        }
        join_snapshots = {
            join_id: self.controller.graph.join_snapshot(join_id)
            for join_id in plan.read_set.join_fingerprints
        }
        raw_transitions = current_control_state.get("transitions", {})
        transitions = (
            raw_transitions if isinstance(raw_transitions, Mapping) else {}
        )
        fairness_accounts = {
            workflow_id: {
                "weight": account.weight,
                "attained_service_ms": account.attained_service_ms,
                "virtual_runtime_ms": account.virtual_runtime,
                "dispatch_count": account.dispatch_count,
            }
            for workflow_id, account in self.controller.fairness.accounts.items()
        }
        source_bundles = {
            item.bundle_id: item for item in source.physical_kv.bundles
        }
        current_bundles = {}
        builder = self.controller.policy_snapshot_builder
        for intent in plan.residency:
            source_bundle = source_bundles.get(intent.bundle_id)
            if source_bundle is None or len(source_bundle.extent_ids) != 1:
                current_bundles[intent.bundle_id] = None
                continue
            try:
                handle = page_handle_from_extent_id(
                    source_bundle.extent_ids[0]
                )
                current_bundles[intent.bundle_id] = (
                    builder.page_bundle_at_safe_point(
                        handle,
                        now_ms=observation.ts_ms,
                    )
                )
            except (KeyError, RuntimeError, ValueError):
                current_bundles[intent.bundle_id] = None
        hbm_available = max(
            0,
            observation.hbm_capacity_bytes
            - observation.hbm_used_bytes
            - self.controller.admission.reserved_bytes,
        )
        return JointPlanCurrentState(
            now_ms=observation.ts_ms,
            runnable_frontier=current_runnable,
            invocation_snapshots=invocation_snapshots,
            join_snapshots=join_snapshots,
            transitions={
                str(workflow_id): (
                    dict(state) if isinstance(state, Mapping) else {}
                )
                for workflow_id, state in transitions.items()
            },
            fairness_revision=self.controller.fairness.revision,
            fairness_accounts=fairness_accounts,
            workflow_memory_charges=(
                self.controller.workflow_memory_charges()
            ),
            transfer_epoch=int(current_control_state.get("transfer_epoch", 0)),
            hbm_capacity_bytes=observation.hbm_capacity_bytes,
            hbm_available_bytes=hbm_available,
            host_free_bytes=observation.host_free_bytes,
            bundle_snapshots=current_bundles,
            strict_global_reasons=strict_global_reasons,
        )

    def _online_joint_admission_decision(
        self,
        *,
        now_ms: float,
    ) -> OnlineJointPlanDecision:
        if not self.config.joint_policy_enabled:
            return OnlineJointPlanDecision(None, "disabled")
        result = getattr(self, "_online_joint_result", None)
        source = getattr(self, "_online_joint_source", None)
        if result is None or source is None or result.plan is None:
            self._online_joint_counts["fallback_no_plan"] += 1
            return OnlineJointPlanDecision(None, "no_validated_plan")
        cached = getattr(self, "_current_online_joint_decision", None)
        reuse = False
        try:
            signature_matches = True
            if (
                self._last_joint_signature_check_ms is None
                or now_ms - self._last_joint_signature_check_ms
                >= _JOINT_SIGNATURE_CHECK_INTERVAL_MS
            ):
                self._last_joint_signature_check_ms = now_ms
                signature_matches = (
                    self._joint_decision_state_signature(now_ms)
                    == self._last_joint_decision_state_signature
                )
            reuse = (
                cached is not None
                and self._last_joint_decision_plan_id == result.plan.plan_id
                and self._last_joint_decision_ms is not None
                and now_ms - self._last_joint_decision_ms
                < _JOINT_DECISION_REUSE_INTERVAL_MS
                and signature_matches
            )
        except Exception:
            reuse = False
        if reuse:
            self._online_joint_counts["safe_point_decision_reused"] += 1
            return cached
        commit_started_ns = time.perf_counter_ns()
        try:
            observation = self._runtime_resource_observation(now_ms=now_ms)
            current_runnable = self._policy_runtime_runnable(now_ms)
            current_control_state = self.controller.policy_control_state(now_ms)
            current_state = self._joint_plan_current_state(
                result,
                source=source,
                observation=observation,
                current_runnable=current_runnable,
                current_control_state=current_control_state,
                strict_global_reasons=(),
            )
            validation = validate_joint_plan_components(
                result.plan,
                source,
                current_state,
            )
            decision = compile_online_joint_view(
                result.plan,
                validation,
                visible_request_ids=(
                    item.request_id for item in current_runnable
                ),
                epoch_sequence=self._online_joint_epoch_sequence + 1,
            )
            if result.plan.semantic_residency and decision.view is not None:
                decision = self._physical_commit_semantic_residency(
                    result.plan,
                    decision,
                    now_ms=now_ms,
                )
        except Exception as error:
            self._online_joint_counts["fallback_validation_error"] += 1
            self.audit.emit(
                "online_joint_plan_fallback",
                now_ms,
                plan_id=result.plan.plan_id,
                reason="current_state_validation_error",
                error=f"{type(error).__name__}: {error}",
            )
            return OnlineJointPlanDecision(None, "current_state_validation_error")
        commit_ms = (time.perf_counter_ns() - commit_started_ns) / 1_000_000.0
        self._joint_shadow_timing_samples["physical_commit_ms"].append(commit_ms)
        if commit_ms > self.config.joint_physical_commit_budget_ms:
            self._online_joint_counts["physical_commit_budget_exceeded"] += 1
            self._record_joint_decision_cache(now_ms=now_ms)
            self.audit.emit(
                "online_joint_physical_commit_budget_exceeded",
                now_ms,
                plan_id=result.plan.plan_id,
                elapsed_ms=commit_ms,
                budget_ms=self.config.joint_physical_commit_budget_ms,
                action="publish_safe_point_seed",
            )
            return OnlineJointPlanDecision(
                None, "physical_commit_budget_exceeded"
            )
        if decision.view is None:
            self._online_joint_counts[f"fallback_{decision.reason}"] += 1
        else:
            self._online_joint_counts["current_state_applicable"] += 1
            self._online_joint_validation = validation
            self._online_joint_epoch_sequence += 1
            self._current_joint_plan_epoch = decision.epoch
            if decision.epoch is not None:
                self._online_joint_counts[
                    f"epoch_{decision.epoch.planner_mode.value}"
                ] += 1
                self._online_joint_counts[
                    f"epoch_{decision.reason}"
                ] += 1
                self.audit.emit(
                    "online_joint_epoch_committed",
                    now_ms,
                    epoch_id=decision.epoch.epoch_id,
                    plan_id=decision.epoch.source_plan_id,
                    planner_mode=decision.epoch.planner_mode.value,
                    commit_reason=decision.reason,
                    source_action_count=decision.epoch.source_action_count,
                    committed_action_count=(
                        decision.epoch.committed_action_count
                    ),
                    actionable_coverage=decision.epoch.actionable_coverage,
                    rejected_slices=[
                        {
                            "slice_id": item.slice_id,
                            "kind": item.kind,
                            "action_key": item.action_key,
                            "reasons": list(item.reasons),
                        }
                for item in decision.epoch.action_slices
                if not item.committed
                ],
            )
        self._record_joint_decision_cache(now_ms=now_ms)
        return decision

    def _joint_decision_state_signature(self, now_ms: float) -> tuple[object, ...]:
        """Cheap state signature for reusing the online joint decision.

        Every field that can change the admission decision or the fallback
        seed epoch is represented here. When the plan id and this signature
        are unchanged, re-validating would produce an identical decision, so
        the cached decision can be reused.
        """

        runnable = self._policy_runtime_runnable(now_ms)
        runnable_signature = self._joint_shadow_runnable_signature(runnable)
        liveness_tracker = getattr(
            self, "_persistent_liveness_revisions", None
        )
        if liveness_tracker is None:
            liveness_tracker = PersistentLivenessRevisionTracker()
            self._persistent_liveness_revisions = liveness_tracker
        lease_index = getattr(self, "_restore_leases", None)
        liveness = liveness_tracker.observe(
            obligations=self._restore_obligation_index().all(),
            leases=(lease_index.all() if lease_index is not None else ()),
            graces=tuple(
                getattr(self, "_restore_service_grace_by_request", {}).values()
            ),
        )
        control_state = self.controller.policy_control_state(now_ms)
        transfer_epoch = int(control_state.get("transfer_epoch", 0))
        return (
            self.controller.graph.graph_version,
            self.controller.data_consumers.version,
            self.controller.runtime_event_sequence,
            self.controller.page_index.revision,
            self.controller.page_index.topology_revision,
            self.controller.fairness.revision,
            transfer_epoch,
            runnable_signature,
            self.controller.action_frontier_observer.revision,
            liveness.obligation_revision,
            liveness.lease_revision,
            liveness.grace_revision,
            tuple(
                (
                    workflow_id,
                    tuple(sorted(state.items())),
                )
                for workflow_id, state in sorted(
                    control_state.get("transitions", {}).items()
                )
                if isinstance(state, Mapping)
            ),
            self.controller.actual_hbm_used_bytes
            // self.config.reference_policy_hbm_bucket_bytes,
            self.controller.signals.host_free_bytes,
        )

    def _record_joint_decision_cache(self, *, now_ms: float) -> None:
        result = getattr(self, "_online_joint_result", None)
        self._last_joint_decision_plan_id = (
            result.plan.plan_id
            if result is not None and result.plan is not None
            else None
        )
        self._last_joint_decision_ms = now_ms
        try:
            self._last_joint_decision_state_signature = (
                self._joint_decision_state_signature(now_ms)
            )
        except Exception:
            self._last_joint_decision_state_signature = None

    def _physical_commit_semantic_residency(
        self,
        plan: Any,
        decision: OnlineJointPlanDecision,
        *,
        now_ms: float,
    ) -> OnlineJointPlanDecision:
        """Resolve at most one stable context target to a live Radix bundle."""

        if decision.view is None or decision.epoch is None:
            return decision
        self._current_semantic_residency_commit = None
        checked_slices: list[ActionSlice] = []
        host_available = self.controller.signals.host_free_bytes
        device_available = max(
            0,
            self.config.hbm_capacity_bytes
            - self.controller.actual_hbm_used_bytes
            - self.controller.admission.reserved_bytes,
        )
        selected: tuple[
            int, SemanticResidencyTarget, PhysicalBundlePreview
        ] | None = None
        for index, target in enumerate(plan.semantic_residency):
            reasons: list[str] = []
            context = self.controller.graph.contexts.get(target.context_id)
            if context is None:
                reasons.append("context_missing")
            elif context.epoch != target.context_epoch:
                reasons.append("context_epoch_changed")
            command_kind = self._online_residency_command_kind(target.action)
            if command_kind is None:
                reasons.append("no_physical_action")
            elif (
                command_kind == CommandKind.SHADOW_CONTEXT
                and not self.config.shadow_enabled
            ):
                reasons.append("shadow_disabled")
            candidates: list[PhysicalBundlePreview] = []
            blockers: set[str] = set()
            if not reasons and command_kind is not None:
                previews = (
                    self.controller.arbiter.bundle_builder.previews_for_context(
                        command_kind,
                        target.context_id,
                        target.context_epoch,
                        now_ms=now_ms,
                        host_available_bytes=host_available,
                        device_available_bytes=device_available,
                    )
                )
                expected_actions = self._online_residency_expected_page_actions(
                    target.action
                )
                for preview in previews:
                    actual_actions = {
                        item.action for item in preview.page_actions
                    }
                    if not actual_actions.intersection(expected_actions):
                        continue
                    if preview.eligible:
                        candidates.append(preview)
                    else:
                        blockers.update(
                            item.code.value for item in preview.blockers
                        )
                if not candidates:
                    reasons.extend(
                        f"physical:{item}" for item in sorted(blockers)
                    )
                    if not blockers:
                        reasons.append("physical_preview_unavailable")
            if reasons:
                checked_slices.append(
                    ActionSlice(
                        slice_id=f"semantic-residency:{index}",
                        kind="semantic_residency",
                        action_key=target.context_id,
                        dependency_keys=(),
                        committed=False,
                        reasons=tuple(reasons),
                    )
                )
                continue
            preview = min(
                candidates,
                key=lambda item: (
                    item.bundle.closure_bytes,
                    item.copy_bytes,
                    item.context_id,
                    item.bundle.bundle_id,
                ),
            )
            checked_slices.append(
                ActionSlice(
                    slice_id=f"semantic-residency:{index}",
                    kind="semantic_residency",
                    action_key=target.context_id,
                    dependency_keys=(),
                    committed=True,
                )
            )
            selected = (index, target, preview)
            break

        view = decision.view
        if selected is not None:
            index, target, preview = selected
            view = replace(view, residency_intent_indices=(index,))
            self._current_semantic_residency_commit = (
                plan.plan_id,
                index,
                target,
                preview,
            )
            self._online_joint_counts["semantic_physical_commit"] += 1
        elif plan.semantic_residency:
            self._online_joint_counts["semantic_physical_no_action"] += 1
        slices = decision.epoch.action_slices + tuple(checked_slices)
        epoch = replace(
            decision.epoch,
            view=view,
            action_slices=slices,
            source_action_count=len(slices),
            committed_action_count=sum(item.committed for item in slices),
        )
        has_action = bool(
            view.ordered_request_ids or view.residency_intent_indices
        )
        partial = any(not item.committed for item in slices)
        return OnlineJointPlanDecision(
            view,
            (
                "no_action"
                if not has_action
                else "partially_applicable"
                if partial
                else "applicable"
            ),
            epoch,
        )

    def _safe_point_seed_decision(
        self, *, now_ms: float
    ) -> OnlineJointPlanDecision:
        runnable = self._policy_runtime_runnable(now_ms)
        visible_ids = frozenset(item.request_id for item in runnable)
        current = getattr(self, "_current_online_joint_decision", None)
        current_epoch = getattr(self, "_current_joint_plan_epoch", None)
        restore_requirements = tuple(
            (
                item.request_id,
                item.required_extent_ids
                or (f"obligation:{item.obligation_id}",),
            )
            for item in self._restore_obligation_index().active()
            if item.request_id in visible_ids
        )
        if (
            current is not None
            and current.view is not None
            and current_epoch is not None
            and current_epoch.planner_mode
            in {JointPlannerMode.BOUNDED_SEED, JointPlannerMode.EMERGENCY}
            and visible_ids
            == frozenset(
                (*current.view.ordered_request_ids, *current.view.deferred_request_ids)
            )
            and current.view.restore_requirements == restore_requirements
        ):
            return current
        workflow_ids = {item.workflow_id for item in runnable}
        fair_order = self.controller.fairness.ordered(
            workflow_ids,
            memory_charges=self.controller.workflow_memory_charges(),
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
        )
        active_workflow_ids = frozenset(
            fair_order[: self.config.joint_workflow_active_window]
        )
        fair_rank = {workflow_id: rank for rank, workflow_id in enumerate(fair_order)}
        frontier_rank = {
            workflow_id: {
                item.invocation_id: rank
                for rank, item in enumerate(
                    self.controller.frontier.candidates(workflow_id)
                )
            }
            for workflow_id in workflow_ids
        }
        ordered = tuple(
            item.request_id
            for item in sorted(
                (
                    item
                    for item in runnable
                    if item.workflow_id in active_workflow_ids
                ),
                key=lambda item: (
                    fair_rank.get(item.workflow_id, 1 << 30),
                    frontier_rank.get(item.workflow_id, {}).get(
                        item.invocation_id, 1 << 30
                    ),
                    item.submitted_ts_ms,
                    item.request_id,
                ),
            )
        )
        self._online_joint_epoch_sequence += 1
        decision = compile_bounded_seed_epoch(
            ordered_request_ids=ordered,
            visible_request_ids=visible_ids,
            epoch_sequence=self._online_joint_epoch_sequence,
            emergency=(
                self.controller.actual_hbm_used_bytes
                / self.config.hbm_capacity_bytes
                >= self.config.joint_emergency_hbm_ratio
            ),
            restore_requirements=restore_requirements,
        )
        self._current_joint_plan_epoch = decision.epoch
        self._online_joint_counts["safe_point_seed_epoch"] += 1
        return decision

    @staticmethod
    def _online_residency_command_kind(
        action: ResidencyAction,
    ) -> CommandKind | None:
        return {
            ResidencyAction.KEEP: None,
            ResidencyAction.PREPARE_HOST: CommandKind.SHADOW_CONTEXT,
            ResidencyAction.COMMIT_CPU: CommandKind.OFFLOAD_CONTEXT,
            ResidencyAction.PREFETCH_GPU: CommandKind.PREFETCH_CONTEXT,
            ResidencyAction.DROP: CommandKind.DROP_CONTEXT,
            ResidencyAction.RECOMPUTE: CommandKind.DROP_CONTEXT,
        }[action]

    @staticmethod
    def _online_residency_expected_page_actions(
        action: ResidencyAction,
    ) -> frozenset[PhysicalPageAction]:
        return {
            ResidencyAction.KEEP: frozenset(),
            ResidencyAction.PREPARE_HOST: frozenset(
                {PhysicalPageAction.START_D2H}
            ),
            ResidencyAction.COMMIT_CPU: frozenset(
                {
                    PhysicalPageAction.START_D2H,
                    PhysicalPageAction.COMMIT_CPU,
                }
            ),
            ResidencyAction.PREFETCH_GPU: frozenset(
                {PhysicalPageAction.START_H2D}
            ),
            ResidencyAction.DROP: frozenset({PhysicalPageAction.DROP}),
            ResidencyAction.RECOMPUTE: frozenset({PhysicalPageAction.DROP}),
        }[action]

    def _online_residency_intent_order(
        self,
        view: OnlineJointPlanView,
    ) -> tuple[int, ...]:
        result = self._online_joint_result
        if result is None or result.plan is None:
            return ()
        plan = result.plan
        dependency_rank: dict[int, int] = {}
        request_rank = {
            request_id: rank
            for rank, request_id in enumerate(view.ordered_request_ids)
        }
        for dependency in plan.dependencies:
            if dependency.before_request_id is None:
                continue
            dependency_rank[dependency.residency_intent_index] = min(
                dependency_rank.get(
                    dependency.residency_intent_index,
                    1 << 30,
                ),
                request_rank.get(dependency.before_request_id, 1 << 30),
            )
        action_rank = {
            ResidencyAction.PREFETCH_GPU: 0,
            ResidencyAction.COMMIT_CPU: 1,
            ResidencyAction.DROP: 2,
            ResidencyAction.RECOMPUTE: 2,
            ResidencyAction.PREPARE_HOST: 3,
            ResidencyAction.KEEP: 4,
        }
        if plan.semantic_residency:
            return tuple(
                sorted(
                    view.residency_intent_indices,
                    key=lambda index: (
                        action_rank[plan.semantic_residency[index].action],
                        plan.semantic_residency[index].deadline_ms,
                        index,
                    ),
                )
            )
        return tuple(
            sorted(
                view.residency_intent_indices,
                key=lambda index: (
                    dependency_rank.get(index, 1 << 30),
                    action_rank[plan.residency[index].action],
                    plan.residency[index].deadline_ms,
                    index,
                ),
            )
        )

    def _online_residency_hysteresis_blocks(
        self,
        *,
        plan_id: str,
        bundle_id: str,
        action: ResidencyAction,
        now_ms: float,
    ) -> bool:
        if action not in {
            ResidencyAction.COMMIT_CPU,
            ResidencyAction.DROP,
            ResidencyAction.RECOMPUTE,
        }:
            return False
        previous = self._online_joint_last_residency_action.get(bundle_id)
        if previous is None or previous[0] != ResidencyAction.PREFETCH_GPU:
            return False
        age_ms = max(0.0, now_ms - previous[1])
        if age_ms >= self.config.residency_hysteresis_ms:
            return False
        hbm_ratio = (
            self.controller.actual_hbm_used_bytes
            / self.config.hbm_capacity_bytes
        )
        if hbm_ratio >= self.config.joint_emergency_hbm_ratio:
            return False
        self._online_joint_counts["residency_hysteresis_blocked"] += 1
        audit_key = (plan_id, bundle_id)
        if audit_key not in self._online_joint_hysteresis_audit:
            self._online_joint_hysteresis_audit.add(audit_key)
            self.audit.emit(
                "online_joint_residency_hysteresis_blocked",
                now_ms,
                plan_id=plan_id,
                bundle_id=bundle_id,
                requested_action=action.value,
                previous_action=previous[0].value,
                age_ms=age_ms,
                hysteresis_ms=self.config.residency_hysteresis_ms,
                hbm_ratio=hbm_ratio,
                emergency_hbm_ratio=(
                    self.config.joint_emergency_hbm_ratio
                ),
            )
        return True

    def _queue_online_joint_residency(
        self,
        view: OnlineJointPlanView,
        *,
        now_ms: float,
    ) -> None:
        if getattr(self, "_pending_online_joint_residency", None) is not None:
            return
        if self.controller.has_pending_transfer_work():
            self._online_joint_counts["residency_wait_existing_transfer"] += 1
            return
        result = self._online_joint_result
        source = self._online_joint_source
        if (
            result is None
            or source is None
            or result.plan is None
            or result.plan.plan_id != view.plan_id
        ):
            return
        source_bundles = {
            item.bundle_id: item for item in source.physical_kv.bundles
        }
        plan = result.plan
        if plan.semantic_residency:
            self._queue_semantic_joint_residency(
                plan,
                view,
                now_ms=now_ms,
            )
            return
        for intent_index in self._online_residency_intent_order(view):
            intent = plan.residency[intent_index]
            if intent.action == ResidencyAction.KEEP:
                continue
            if self._online_residency_hysteresis_blocks(
                plan_id=plan.plan_id,
                bundle_id=intent.bundle_id,
                action=intent.action,
                now_ms=now_ms,
            ):
                continue
            if intent.action == ResidencyAction.RECOMPUTE:
                self._invalidate_online_joint_plan(
                    plan.plan_id,
                    reason="recompute_not_enabled",
                    now_ms=now_ms,
                )
                return
            source_bundle = source_bundles.get(intent.bundle_id)
            if source_bundle is None or len(source_bundle.extent_ids) != 1:
                self._invalidate_online_joint_plan(
                    plan.plan_id,
                    reason="source_extent_missing",
                    now_ms=now_ms,
                    bundle_id=intent.bundle_id,
                )
                return
            try:
                target_handle = page_handle_from_extent_id(
                    source_bundle.extent_ids[0]
                )
            except ValueError:
                self._invalidate_online_joint_plan(
                    plan.plan_id,
                    reason="source_extent_identity_invalid",
                    now_ms=now_ms,
                    bundle_id=intent.bundle_id,
                )
                return
            if not source_bundle.owner_context_ids:
                self._invalidate_online_joint_plan(
                    plan.plan_id,
                    reason="unowned_extent_requires_lifecycle_fallback",
                    now_ms=now_ms,
                    bundle_id=intent.bundle_id,
                )
                return
            command_kind = self._online_residency_command_kind(intent.action)
            if command_kind is None:
                continue
            if (
                command_kind == CommandKind.SHADOW_CONTEXT
                and not self.config.shadow_enabled
            ):
                self._invalidate_online_joint_plan(
                    plan.plan_id,
                    reason="shadow_not_enabled",
                    now_ms=now_ms,
                    bundle_id=intent.bundle_id,
                )
                return
            expected_actions = self._online_residency_expected_page_actions(
                intent.action
            )
            host_available = self.controller.signals.host_free_bytes
            device_available = max(
                0,
                self.config.hbm_capacity_bytes
                - self.controller.actual_hbm_used_bytes
                - self.controller.admission.reserved_bytes,
            )
            candidates: list[PhysicalBundlePreview] = []
            observed_blockers: set[str] = set()
            for context_id in source_bundle.owner_context_ids:
                context = self.controller.graph.contexts.get(context_id)
                if context is None:
                    continue
                previews = self.controller.arbiter.bundle_builder.previews_for_context(
                    command_kind,
                    context_id,
                    context.epoch,
                    now_ms=now_ms,
                    host_available_bytes=host_available,
                    device_available_bytes=device_available,
                )
                for preview in previews:
                    if target_handle not in preview.bundle.handles:
                        continue
                    target_actions = {
                        item.action
                        for item in preview.page_actions
                        if item.handle == target_handle
                    }
                    if not target_actions.intersection(expected_actions):
                        continue
                    if preview.eligible:
                        candidates.append(preview)
                    else:
                        observed_blockers.update(
                            item.code.value for item in preview.blockers
                        )
            if not candidates:
                self._invalidate_online_joint_plan(
                    plan.plan_id,
                    reason="physical_preview_unavailable",
                    now_ms=now_ms,
                    bundle_id=intent.bundle_id,
                    blocker_codes=tuple(sorted(observed_blockers)),
                )
                return
            preview = min(
                candidates,
                key=lambda item: (
                    item.bundle.closure_bytes,
                    item.copy_bytes,
                    item.context_id,
                    item.bundle.bundle_id,
                ),
            )
            self._online_joint_residency_sequence += 1
            transaction_id = (
                f"joint-residency-{self._online_joint_residency_sequence}"
            )
            command_id = f"{transaction_id}-command"
            command = ControlCommand(
                command_id=command_id,
                kind=command_kind,
                created_ts_ms=now_ms,
                context_id=preview.context_id,
                context_epoch=preview.context_epoch,
                target_bytes=preview.bundle.closure_bytes,
                priority=4.0e9,
                deadline_ms=intent.deadline_ms,
                queue_class=(
                    CommandQueueClass.SHADOW
                    if command_kind == CommandKind.SHADOW_CONTEXT
                    else CommandQueueClass.URGENT
                ),
                metadata={
                    "reason": "observed_joint_residency",
                    "joint_plan_id": plan.plan_id,
                    "joint_residency_intent_index": intent_index,
                    "joint_source_bundle_id": intent.bundle_id,
                    "joint_residency_action": intent.action.value,
                    "joint_planned_target_bytes": intent.target_bytes,
                    "joint_physical_closure_bytes": (
                        preview.bundle.closure_bytes
                    ),
                    "physical_bundle_scope": preview.bundle.scope.value,
                    "physical_exclusive_action_bytes": (
                        preview.bundle.exclusive_action_bytes
                    ),
                    "physical_cross_context_action_bytes": (
                        preview.bundle.cross_context_action_bytes
                    ),
                    "physical_foreign_owner_context_ids": list(
                        preview.bundle.foreign_owner_context_ids
                    ),
                },
                physical_bundle=preview.intent(),
            )
            enqueue_outcome = self.controller.enqueue_control_command(command)
            if enqueue_outcome.status != EnqueueStatus.ENQUEUED:
                self._online_joint_counts["residency_queue_conflict"] += 1
                return
            transaction = _OnlineJointResidencyTransaction(
                transaction_id=transaction_id,
                plan_id=plan.plan_id,
                intent_index=intent_index,
                source_bundle_id=intent.bundle_id,
                action=intent.action,
                command_id=command_id,
                command_kind=command_kind,
                context_id=preview.context_id,
                physical_bundle_id=preview.bundle.bundle_id,
                created_ts_ms=now_ms,
            )
            self._pending_online_joint_residency = transaction
            self._online_joint_residency_history.append(transaction)
            self._online_joint_counts["residency_queued"] += 1
            self.audit.emit(
                "online_joint_residency_queued",
                now_ms,
                transaction_id=transaction_id,
                plan_id=plan.plan_id,
                intent_index=intent_index,
                action=intent.action.value,
                command_id=command_id,
                command_kind=command_kind.value,
                source_bundle_id=intent.bundle_id,
                physical_bundle_id=preview.bundle.bundle_id,
                context_id=preview.context_id,
                planned_target_bytes=intent.target_bytes,
                physical_closure_bytes=preview.bundle.closure_bytes,
                copy_bytes=preview.copy_bytes,
            )
            return

    def _queue_semantic_joint_residency(
        self,
        plan: Any,
        view: OnlineJointPlanView,
        *,
        now_ms: float,
    ) -> None:
        committed = getattr(self, "_current_semantic_residency_commit", None)
        if committed is None:
            return
        plan_id, intent_index, target, preview = committed
        if (
            plan_id != plan.plan_id
            or intent_index not in view.residency_intent_indices
        ):
            return
        if self._online_residency_hysteresis_blocks(
            plan_id=plan.plan_id,
            bundle_id=preview.bundle.bundle_id,
            action=target.action,
            now_ms=now_ms,
        ):
            return
        if target.action == ResidencyAction.RECOMPUTE:
            self._online_joint_counts["semantic_recompute_not_enabled"] += 1
            return
        command_kind = self._online_residency_command_kind(target.action)
        if command_kind is None:
            return
        self._online_joint_residency_sequence += 1
        transaction_id = (
            f"joint-residency-{self._online_joint_residency_sequence}"
        )
        command_id = f"{transaction_id}-command"
        command = ControlCommand(
            command_id=command_id,
            kind=command_kind,
            created_ts_ms=now_ms,
            context_id=preview.context_id,
            context_epoch=preview.context_epoch,
            target_bytes=preview.bundle.closure_bytes,
            priority=4.0e9,
            deadline_ms=target.deadline_ms,
            queue_class=(
                CommandQueueClass.SHADOW
                if command_kind == CommandKind.SHADOW_CONTEXT
                else CommandQueueClass.URGENT
            ),
            metadata={
                "reason": "semantic_joint_residency",
                "joint_plan_id": plan.plan_id,
                "joint_residency_intent_index": intent_index,
                "joint_semantic_context_id": target.context_id,
                "joint_semantic_context_epoch": target.context_epoch,
                "joint_residency_action": target.action.value,
                "joint_planned_target_bytes": target.target_bytes_hint,
                "joint_physical_closure_bytes": preview.bundle.closure_bytes,
                "physical_bundle_scope": preview.bundle.scope.value,
                "physical_exclusive_action_bytes": (
                    preview.bundle.exclusive_action_bytes
                ),
                "physical_cross_context_action_bytes": (
                    preview.bundle.cross_context_action_bytes
                ),
                "physical_foreign_owner_context_ids": list(
                    preview.bundle.foreign_owner_context_ids
                ),
            },
            physical_bundle=preview.intent(),
        )
        enqueue_outcome = self.controller.enqueue_control_command(command)
        if enqueue_outcome.status != EnqueueStatus.ENQUEUED:
            self._online_joint_counts["residency_queue_conflict"] += 1
            return
        transaction = _OnlineJointResidencyTransaction(
            transaction_id=transaction_id,
            plan_id=plan.plan_id,
            intent_index=intent_index,
            source_bundle_id=(
                f"semantic:{target.context_id}:{target.context_epoch}"
            ),
            action=target.action,
            command_id=command_id,
            command_kind=command_kind,
            context_id=preview.context_id,
            physical_bundle_id=preview.bundle.bundle_id,
            created_ts_ms=now_ms,
        )
        self._pending_online_joint_residency = transaction
        self._online_joint_residency_history.append(transaction)
        self._online_joint_counts["semantic_residency_queued"] += 1
        self.audit.emit(
            "online_joint_residency_queued",
            now_ms,
            transaction_id=transaction_id,
            plan_id=plan.plan_id,
            intent_index=intent_index,
            intent_kind="semantic_context_target",
            action=target.action.value,
            command_id=command_id,
            command_kind=command_kind.value,
            source_context_id=target.context_id,
            source_context_epoch=target.context_epoch,
            physical_bundle_id=preview.bundle.bundle_id,
            context_id=preview.context_id,
            planned_target_bytes=target.target_bytes_hint,
            physical_closure_bytes=preview.bundle.closure_bytes,
            copy_bytes=preview.copy_bytes,
            source_joint_plan_id=plan.plan_id,
        )

    def _advance_online_joint_residency(
        self,
        acks: tuple[CommandAck, ...] | list[CommandAck],
        *,
        now_ms: float,
    ) -> None:
        transaction = getattr(self, "_pending_online_joint_residency", None)
        if transaction is None:
            return
        ack = next(
            (item for item in acks if item.command_id == transaction.command_id),
            None,
        )
        if ack is None:
            return
        transaction.completed_ts_ms = now_ms
        transaction.actual_bytes = ack.actual_bytes
        if ack.status == CommandStatus.COMPLETED:
            transaction.stage = "completed"
            self._online_joint_counts["residency_completed"] += 1
            self._online_joint_last_residency_action[
                transaction.source_bundle_id
            ] = (transaction.action, now_ms)
        else:
            transaction.stage = "failed"
            transaction.failure_reason = (
                f"{ack.status.value}:{ack.reason or 'unspecified'}"
            )
            self._online_joint_counts[
                f"residency_{ack.status.value}"
            ] += 1
        self.audit.emit(
            "online_joint_residency_terminal",
            now_ms,
            transaction_id=transaction.transaction_id,
            plan_id=transaction.plan_id,
            intent_index=transaction.intent_index,
            action=transaction.action.value,
            command_id=transaction.command_id,
            command_kind=transaction.command_kind.value,
            source_bundle_id=transaction.source_bundle_id,
            physical_bundle_id=transaction.physical_bundle_id,
            context_id=transaction.context_id,
            status=ack.status.value,
            actual_bytes=ack.actual_bytes,
            reason=ack.reason,
            blocker_codes=sorted({item.code.value for item in ack.blockers}),
        )
        self._pending_online_joint_residency = None
        self._invalidate_online_joint_plan(
            transaction.plan_id,
            reason=f"residency_terminal:{ack.status.value}",
            now_ms=now_ms,
            emit=False,
        )

    def _invalidate_online_joint_plan(
        self,
        plan_id: str,
        *,
        reason: str,
        now_ms: float,
        bundle_id: str | None = None,
        blocker_codes: tuple[str, ...] = (),
        emit: bool = True,
    ) -> None:
        result = getattr(self, "_online_joint_result", None)
        if result is None or result.plan is None or result.plan.plan_id != plan_id:
            return
        self._online_joint_result = None
        self._online_joint_source = None
        self._online_joint_validation = None
        self._current_online_joint_view = None
        self._current_online_joint_decision = None
        self._current_joint_plan_epoch = None
        self._current_semantic_residency_commit = None
        self._online_joint_counts[f"invalidated_{reason}"] += 1
        if emit:
            self.audit.emit(
                "online_joint_plan_invalidated",
                now_ms,
                plan_id=plan_id,
                reason=reason,
                bundle_id=bundle_id,
                blocker_codes=list(blocker_codes),
                fallback="bounded_seed_joint_epoch",
            )

    def _publish_online_joint_candidate(
        self,
        result: JointShadowResult,
        *,
        source: PolicyInput,
        validation: JointPlanComponentValidation | None,
        visible_request_ids: tuple[str, ...],
        now_ms: float,
    ) -> None:
        if not self.config.joint_policy_enabled:
            return
        if getattr(self, "_shutdown_state", "running") != "running":
            self._online_joint_counts["publish_rejected_shutdown"] += 1
            return
        if result.plan is None or validation is None:
            self._online_joint_counts["publish_missing_validation"] += 1
            return
        decision = compile_online_joint_view(
            result.plan,
            validation,
            visible_request_ids=visible_request_ids,
        )
        if decision.view is None:
            self._online_joint_counts[f"publish_rejected_{decision.reason}"] += 1
            return
        self._online_joint_result = result
        self._online_joint_source = source
        self._online_joint_validation = validation
        self._current_online_joint_view = None
        self._current_online_joint_decision = None
        self._online_joint_counts["published"] += 1
        self.audit.emit(
            "online_joint_plan_published",
            now_ms,
            plan_id=result.plan.plan_id,
            worker_sequence=result.sequence,
            execution_request_count=len(decision.view.ordered_request_ids),
            immediate_request_count=len(decision.view.immediate_request_ids),
            restore_blocked_request_count=len(
                decision.view.restore_requirements
            ),
            residency_intent_count=len(decision.view.residency_intent_indices),
        )

    def _record_joint_shadow_result(
        self,
        result: JointShadowResult,
        *,
        policy_input: PolicyInput,
        observation: RuntimeResourceObservation,
        snapshot_sequence: int,
        snapshot_build_ms: float,
        validation_state_changed: bool,
        extra_strict_reasons: tuple[str, ...] = (),
        extra_readset_reasons: tuple[str, ...] = (),
        coarse_stamp_reasons: tuple[str, ...] = (),
        current_runnable: tuple[RunnableInvocation, ...] | None = None,
        current_control_state: Mapping[str, object] | None = None,
    ) -> None:
        self._last_joint_shadow_result_sequence = result.sequence
        self._joint_shadow_counts["result_observed"] += 1
        publish_delay_ms = max(
            0.0,
            time.monotonic_ns() / 1_000_000.0 - result.completed_monotonic_ms,
        )
        for name, value in (
            ("plan_queue_wait_ms", result.queue_wait_ms),
            ("plan_compute_ms", result.compute_ms),
            ("plan_publish_to_safe_point_ms", publish_delay_ms),
        ):
            self._joint_shadow_timing_samples[name].append(value)
        worker = getattr(self, "joint_shadow_worker", None)
        worker_stats = worker.stats() if worker is not None else None
        common = {
            "worker_sequence": result.sequence,
            "source_snapshot_id": result.snapshot_id,
            "current_snapshot_id": policy_input.snapshot_id,
            "current_snapshot_sequence": snapshot_sequence,
            "current_snapshot_persisted": bool(snapshot_sequence),
            "validation_state_changed": validation_state_changed,
            "snapshot_build_ms": snapshot_build_ms,
            "snapshot_delta_apply_ms": result.snapshot_delta_apply_ms,
            "snapshot_materialize_ms": result.snapshot_materialize_ms,
            "plan_queue_wait_ms": result.queue_wait_ms,
            "plan_compute_ms": result.compute_ms,
            "trigger_interval_ms": result.trigger_interval_ms,
            "effective_planning_budget_ms": result.planning_budget_ms,
            "plan_publish_to_safe_point_ms": publish_delay_ms,
            "worker_pending_count": (
                worker_stats.pending_count if worker_stats is not None else 0
            ),
            "worker_dropped_pending_count": (
                worker_stats.dropped_pending_count
                if worker_stats is not None
                else 0
            ),
            "worker_coalesced_pending_count": (
                worker_stats.coalesced_pending_count
                if worker_stats is not None
                else 0
            ),
            "application_connected": self.config.joint_policy_enabled,
        }
        if result.error is not None or result.plan is None:
            self._joint_shadow_counts["worker_failed"] += 1
            self.audit.emit(
                "joint_plan_shadow_failed",
                observation.ts_ms,
                error=result.error or "worker returned no plan",
                **common,
            )
            return
        if result.plan.prediction_used:
            self._joint_predictive_counts["plans_using_predictions"] += 1
        for name, value in result.plan.prediction_influence:
            self._joint_predictive_counts[name] += value

        validation_started_ns = time.perf_counter_ns()
        component_validation: JointPlanComponentValidation | None = None
        live_current_state: JointPlanCurrentState | None = None
        try:
            if current_runnable is not None and current_control_state is not None:
                live_current_state = self._joint_plan_current_state(
                    result,
                    source=policy_input,
                    observation=observation,
                    current_runnable=current_runnable,
                    current_control_state=current_control_state,
                    strict_global_reasons=extra_strict_reasons,
                )
                component_validation = validate_joint_plan_components(
                    result.plan,
                    policy_input,
                    live_current_state,
                )
                validation = JointPlanValidation(
                    strict_global_reasons=(
                        component_validation.strict_global_reasons
                    ),
                    readset_conflict_reasons=(
                        component_validation.readset_conflict_reasons
                    ),
                )
            else:
                validation = validate_joint_plan(result.plan, policy_input)
                if extra_strict_reasons or extra_readset_reasons:
                    validation = JointPlanValidation(
                        strict_global_reasons=tuple(
                            sorted(
                                set(validation.strict_global_reasons).union(
                                    extra_strict_reasons
                                )
                            )
                        ),
                        readset_conflict_reasons=tuple(
                            sorted(
                                set(validation.readset_conflict_reasons).union(
                                    extra_readset_reasons
                                )
                            )
                        ),
                    )
        except Exception as error:
            validation_ms = (
                time.perf_counter_ns() - validation_started_ns
            ) / 1_000_000.0
            self._joint_shadow_timing_samples["validation_ms"].append(
                validation_ms
            )
            self._joint_shadow_counts["validation_failed"] += 1
            self.audit.emit(
                "joint_plan_shadow_validation_failed",
                observation.ts_ms,
                plan_id=result.plan.plan_id,
                validation_ms=validation_ms,
                error=f"{type(error).__name__}: {error}",
                **common,
            )
            return
        validation_ms = (
            time.perf_counter_ns() - validation_started_ns
        ) / 1_000_000.0
        plan_age_ms = max(
            0.0,
            observation.ts_ms - result.plan.generated_ts_ms,
        )
        self._joint_shadow_timing_samples["validation_ms"].append(validation_ms)
        self._joint_shadow_timing_samples["plan_age_ms"].append(plan_age_ms)
        self._joint_shadow_counts["validated"] += 1
        self._joint_shadow_strict_stale_reasons.update(
            validation.strict_global_reasons
        )
        self._joint_shadow_readset_stale_reasons.update(
            validation.readset_conflict_reasons
        )
        if not validation.strict_global_fresh:
            self._joint_shadow_counts["strict_global_stale"] += 1
        if not validation.readset_fresh:
            self._joint_shadow_counts["readset_stale"] += 1

        plan = result.plan
        graph_state = policy_input.runtime_graph.state
        fairness_state = graph_state.get("workflow_fairness", {})
        control_state = graph_state.get("control", {})
        if not isinstance(fairness_state, Mapping):
            fairness_state = {}
        if not isinstance(control_state, Mapping):
            control_state = {}
        admission_actions = Counter(item.action.value for item in plan.admissions)
        residency_actions = Counter(item.action.value for item in plan.residency)
        fields = {
            **common,
            "plan_id": plan.plan_id,
            "plan_age_ms": plan_age_ms,
            "planner_reported_ms": plan.planning_ms,
            "planner_phase_ms": dict(plan.planning_phase_ms),
            "planner_search_complete": plan.search_complete,
            "planner_termination_reason": plan.planning_termination_reason,
            "validation_ms": validation_ms,
            "strict_global_fresh": validation.strict_global_fresh,
            "strict_global_reasons": list(validation.strict_global_reasons),
            "readset_fresh": validation.readset_fresh,
            "readset_conflict_reasons": list(
                validation.readset_conflict_reasons
            ),
            "coarse_stamp_reasons": list(coarse_stamp_reasons),
            "source_fairness_revision": plan.read_set.fairness_revision,
            "current_fairness_revision": (
                live_current_state.fairness_revision
                if live_current_state is not None
                else int(fairness_state.get("revision", 0))
            ),
            "source_transfer_epoch": plan.read_set.transfer_epoch,
            "current_transfer_epoch": (
                live_current_state.transfer_epoch
                if live_current_state is not None
                else int(control_state.get("transfer_epoch", 0))
            ),
            "execution_mode": plan.execution.mode,
            "execution_request_count": len(plan.execution.ordered_request_ids),
            "admission_action_counts": dict(sorted(admission_actions.items())),
            "residency_action_counts": dict(sorted(residency_actions.items())),
            "dependency_count": len(plan.dependencies),
            "candidate_count": plan.candidate_count,
            "evaluated_package_count": plan.evaluated_package_count,
            "expected_hbm_peak_bytes": plan.expected_hbm_peak_bytes,
            "expected_unhidden_stall_ms": plan.expected_unhidden_stall_ms,
            "fallback_reason": plan.fallback_reason,
            "transition_open": plan.transition_open,
        }
        if not plan.search_complete:
            self._joint_shadow_counts["search_truncated"] += 1
        if component_validation is not None:
            admission_valid = sum(
                item.valid for item in component_validation.admissions.values()
            )
            residency_valid = sum(
                item.valid for item in component_validation.residency.values()
            )
            dependency_valid = sum(
                item.valid for item in component_validation.dependencies
            )
            fields.update(
                {
                    "component_fully_fresh": (
                        component_validation.fully_fresh
                    ),
                    "component_partially_fresh": (
                        component_validation.partially_fresh
                    ),
                    "execution_valid": component_validation.execution.valid,
                    "execution_validation_reasons": list(
                        component_validation.execution.reasons
                    ),
                    "admission_valid_count": admission_valid,
                    "admission_invalid_count": (
                        len(component_validation.admissions) - admission_valid
                    ),
                    "admission_invalid_validation": {
                        key: value.to_dict()
                        for key, value in component_validation.admissions.items()
                        if not value.valid
                    },
                    "residency_valid_count": residency_valid,
                    "residency_invalid_count": (
                        len(component_validation.residency) - residency_valid
                    ),
                    "residency_invalid_validation": {
                        key: value.to_dict()
                        for key, value in component_validation.residency.items()
                        if not value.valid
                    },
                    "dependency_valid_count": dependency_valid,
                    "dependency_invalid_count": (
                        len(component_validation.dependencies) - dependency_valid
                    ),
                    "dependency_invalid_validation": [
                        item.to_dict()
                        for item in component_validation.dependencies
                        if not item.valid
                    ],
                }
            )
        self._publish_online_joint_candidate(
            result,
            source=policy_input,
            validation=component_validation,
            visible_request_ids=tuple(
                item.request_id
                for item in (
                    current_runnable
                    if current_runnable is not None
                    else policy_input.runnable_frontier
                )
            ),
            now_ms=observation.ts_ms,
        )
        if (
            component_validation is not None
            and component_validation.partially_fresh
        ):
            self._joint_shadow_counts["partial"] += 1
            event = "joint_plan_shadow_partial"
        elif not validation.readset_fresh:
            self._joint_shadow_counts["stale"] += 1
            event = "joint_plan_stale"
        elif plan.execution.mode == "observed_joint_idle":
            self._joint_shadow_counts["idle"] += 1
            event = "joint_plan_shadow_idle"
        elif plan.fallback_reason is not None:
            self._joint_shadow_counts["fallback"] += 1
            event = "joint_plan_shadow_fallback"
        else:
            self._joint_shadow_counts["would_apply"] += 1
            event = "joint_plan_would_apply"
        signature = (
            event,
            plan.fallback_reason,
            plan.planning_termination_reason,
            plan.transition_open,
            tuple(validation.strict_global_reasons),
            tuple(validation.readset_conflict_reasons),
            tuple(sorted(admission_actions.items())),
            tuple(sorted(residency_actions.items())),
            fields.get("component_fully_fresh"),
            fields.get("component_partially_fresh"),
            fields.get("execution_valid"),
            fields.get("admission_invalid_count"),
            fields.get("residency_invalid_count"),
            fields.get("dependency_invalid_count"),
        )
        interval_ms = getattr(
            getattr(self, "config", None),
            "joint_shadow_detailed_audit_interval_ms",
            0.0,
        )
        previous_ms = getattr(self, "_last_joint_detailed_audit_ms", None)
        previous_signature = getattr(
            self, "_last_joint_detailed_signature", None
        )
        detailed = bool(
            interval_ms == 0
            or previous_ms is None
            or signature != previous_signature
            or observation.ts_ms - previous_ms >= interval_ms
        )
        if detailed:
            fields["detail_level"] = "full"
            self.audit.emit(event, observation.ts_ms, **fields)
            self._last_joint_detailed_audit_ms = observation.ts_ms
            self._last_joint_detailed_signature = signature
            self._joint_shadow_counts["detailed_audit_records"] += 1
        else:
            self.audit.emit(
                event,
                observation.ts_ms,
                detail_level="compact",
                plan_id=plan.plan_id,
                source_snapshot_id=policy_input.snapshot_id,
                source_snapshot_persisted=bool(snapshot_sequence),
                planner_reported_ms=plan.planning_ms,
                planner_search_complete=plan.search_complete,
                planner_termination_reason=plan.planning_termination_reason,
                plan_age_ms=plan_age_ms,
                validation_ms=validation_ms,
                strict_global_fresh=validation.strict_global_fresh,
                readset_fresh=validation.readset_fresh,
                execution_request_count=len(
                    plan.execution.ordered_request_ids
                ),
                admission_action_counts=dict(sorted(admission_actions.items())),
                residency_action_counts=dict(sorted(residency_actions.items())),
                fallback_reason=plan.fallback_reason,
                application_connected=self.config.joint_policy_enabled,
            )
            self._joint_shadow_counts["compact_audit_records"] += 1

    def _emit_joint_shadow_summary(
        self,
        worker: LatestWinsJointPlanWorker,
        *,
        worker_closed: bool,
    ) -> None:
        worker_stats = worker.stats()
        counts = getattr(self, "_joint_shadow_counts", Counter())
        timing_samples = getattr(self, "_joint_shadow_timing_samples", {})
        latest_unobserved = int(
            worker_stats.latest_published_sequence
            > getattr(self, "_last_joint_shadow_result_sequence", 0)
        )
        superseded_completed = max(
            0,
            worker_stats.completed_count
            - int(counts.get("result_observed", 0))
            - latest_unobserved,
        )
        timing_summary = {}
        for name, raw_values in sorted(timing_samples.items()):
            values = tuple(raw_values)
            timing_summary[name] = {
                "count": len(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "max": max(values, default=0.0),
            }
        validated = int(counts.get("validated", 0))
        self.audit.emit(
            "joint_plan_shadow_summary",
            self._now_ms(),
            worker_closed=worker_closed,
            worker=worker_stats.to_dict(),
            counts=dict(sorted(counts.items())),
            strict_global_stale_rate=(
                counts.get("strict_global_stale", 0) / validated
                if validated
                else 0.0
            ),
            readset_stale_rate=(
                counts.get("readset_stale", 0) / validated
                if validated
                else 0.0
            ),
            strict_global_stale_reasons=dict(
                sorted(
                    getattr(
                        self,
                        "_joint_shadow_strict_stale_reasons",
                        Counter(),
                    ).items()
                )
            ),
            readset_stale_reasons=dict(
                sorted(
                    getattr(
                        self,
                        "_joint_shadow_readset_stale_reasons",
                        Counter(),
                    ).items()
                )
            ),
            latest_unobserved_result=bool(latest_unobserved),
            superseded_completed_result_count=superseded_completed,
            timing_ms=timing_summary,
            application_connected=self.config.joint_policy_enabled,
        )

    def _policy_runtime_runnable(
        self, now_ms: float
    ) -> tuple[RunnableInvocation, ...]:
        result: dict[str, RunnableInvocation] = {}
        frontier_predictions: dict[str, Any] = {}
        if getattr(self.config, "joint_predictive_enabled", False):
            predictor = getattr(self.controller, "predictor", None)
            if (
                predictor is not None
                and getattr(predictor, "frontier_model", None) is not None
            ):
                active_ids = tuple(
                    invocation.invocation_id
                    for invocation in self.controller.graph.invocations.values()
                    if not invocation.state.terminal
                )[:64]
                try:
                    frontier_predictions = (
                        build_invocation_frontier_predictions(
                            self.controller.graph,
                            predictor,
                            now_ms=now_ms,
                            invocation_ids=active_ids,
                        )
                    )
                except Exception:
                    # A prediction failure must never affect the scheduler
                    # critical path; fall back to observed-only planning.
                    frontier_predictions = {}
        self._last_frontier_predictions = {
            invocation_id: {
                "remaining_decode_tokens_p50": prediction.remaining_decode_tokens.quantile(0.5),
                "remaining_external_wait_ms_p50": prediction.remaining_external_wait.quantile(0.5),
                "next_output_tokens_p50": prediction.next_output_tokens.quantile(0.5),
                "support_level": prediction.support_level,
                "ood_reasons": tuple(prediction.ood_reasons),
            }
            for invocation_id, prediction in frontier_predictions.items()
        }
        raw_waiting_queue = getattr(self.scheduler, "waiting_queue", None)
        waiting_queue = (
            tuple(raw_waiting_queue) if raw_waiting_queue is not None else ()
        )
        running_batch = getattr(self.scheduler, "running_batch", None)
        running_requests = tuple(
            getattr(running_batch, "reqs", ()) or ()
        )

        def add_request(req: Any, queue_state: str) -> None:
            metadata = self._metadata(req)
            if metadata is None or self._metadata_scope_is_terminal(metadata):
                return
            max_new_tokens = int(
                getattr(
                    getattr(req, "sampling_params", None),
                    "max_new_tokens",
                    0,
                )
                or 0
            )
            output_ids = getattr(req, "output_ids", None)
            output_tokens = len(output_ids) if output_ids is not None else 0
            remaining_output_tokens = max(0, max_new_tokens - output_tokens)
            origin_input_ids = getattr(req, "origin_input_ids", None)
            prompt_tokens = (
                len(origin_input_ids) if origin_input_ids is not None else 0
            )
            prefix_indices = getattr(req, "prefix_indices", None)
            prefix_tokens = (
                len(prefix_indices) if prefix_indices is not None else 0
            )
            uncached_prompt_tokens = max(0, prompt_tokens - prefix_tokens)
            invocation = self.controller.graph.invocations.get(
                metadata.invocation_id
            )
            execution_mode = (
                invocation.execution_mode.value
                if invocation is not None
                else metadata.execution_mode
            )
            relation_type = (
                invocation.relation_type.value
                if invocation is not None
                else metadata.relation_type
            )
            prediction = frontier_predictions.get(metadata.invocation_id)
            result[str(req.rid)] = RunnableInvocation(
                request_id=str(req.rid),
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
                context_epoch=metadata.context_epoch,
                submitted_ts_ms=getattr(
                    self, "_request_submitted_ts_by_id", {}
                ).get(str(req.rid), now_ms),
                startup_bytes=(
                    uncached_prompt_tokens + remaining_output_tokens
                )
                * self.config.kv_bytes_per_token,
                causal_class=(
                    f"{queue_state}:{execution_mode}:{relation_type}"
                ),
                program_id=metadata.agent_instance_id,
                predicted_remaining_decode_tokens=(
                    prediction.remaining_decode_tokens.quantile(0.5)
                    if prediction is not None
                    else None
                ),
                predicted_external_wait_ms=(
                    prediction.remaining_external_wait.quantile(0.5)
                    if prediction is not None
                    else None
                ),
                predicted_next_output_tokens=(
                    prediction.next_output_tokens.quantile(0.5)
                    if prediction is not None
                    else None
                ),
                prediction_support_level=(
                    prediction.support_level if prediction is not None else ""
                ),
                prediction_ood_reasons=(
                    tuple(prediction.ood_reasons)
                    if prediction is not None
                    else ()
                ),
            )

        for req in waiting_queue:
            add_request(req, "engine_waiting")
        for req in running_requests:
            add_request(req, "engine_running")
        return tuple(sorted(result.values(), key=lambda item: item.request_id))

    def _policy_capabilities(self) -> CapabilityReport:
        limitations = [
            "HiCache 0.5.2rc1 exposes one in-flight physical operation",
            "running decode reorder requires a drained selective-retraction barrier",
        ]
        if not self.backend.capabilities.operation_merge:
            limitations.append("native HiCache operation merge is unavailable")
        return CapabilityReport(
            runtime_name="sglang-hicache-beliefkv",
            runtime_version=BASE_SGLANG_VERSION,
            supported_residency_actions=frozenset(ResidencyAction),
            execution_order_control=True,
            admission_control=True,
            transfer_dependencies=True,
            native_identity_mapping=True,
            limitations=tuple(limitations),
        )

    def _ensure_causal_identity(self, metadata: BeliefKVRequestMetadata) -> None:
        self._identity_metadata[metadata.invocation_id] = metadata
        now = self._now_ms()
        if metadata.root_workflow_id not in self.controller.graph.workflows:
            self._process_events(
                (RuntimeEvent(
                    event_id=self._next_event_id("workflow"),
                    ts_ms=now,
                    kind=RuntimeEventKind.WORKFLOW_START,
                    workflow_id=metadata.root_workflow_id,
                ),)
            )
        if metadata.invocation_id not in self.controller.graph.invocations:
            parent_known = (
                metadata.parent_invocation_id is not None
                and metadata.parent_invocation_id in self.controller.graph.invocations
            )
            self._process_events(
                (RuntimeEvent(
                    event_id=self._next_event_id("invocation"),
                    ts_ms=now,
                    kind=RuntimeEventKind.INVOCATION_CREATE,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    context_epoch=metadata.context_epoch,
                    agent_definition_id=metadata.agent_definition_id,
                    agent_instance_id=metadata.agent_instance_id,
                    parent_invocation_id=(
                        metadata.parent_invocation_id if parent_known else None
                    ),
                    parent_context_id=(metadata.parent_context_id if parent_known else None),
                    relation_type=RelationType(metadata.relation_type),
                    context_mode=ContextMode(metadata.context_mode),
                    execution_mode=ExecutionMode(metadata.execution_mode),
                    return_target_id=metadata.return_target_id,
                    join_id=metadata.join_id,
                    attributes={"persistent": True, "source": "sglang_metadata"},
                ),)
            )
            self.audit.emit(
                "invocation_created",
                now,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
                context_epoch=metadata.context_epoch,
                parent_invocation_id=metadata.parent_invocation_id,
                relation_type=metadata.relation_type,
            )
        else:
            invocation = self.controller.graph.invocations[metadata.invocation_id]
            if invocation.context_id != metadata.context_id:
                raise SGLangBackendError(
                    "an invocation cannot move across persistent contexts"
                )
            context = self.controller.graph.contexts[metadata.context_id]
            if metadata.context_epoch < context.epoch:
                raise SGLangBackendError(
                    f"stale context epoch for {metadata.context_id}: "
                    f"{metadata.context_epoch} < {context.epoch}"
                )
            if metadata.context_epoch > context.epoch:
                previous_epoch = context.epoch
                self._process_events(
                    (RuntimeEvent(
                        event_id=self._next_event_id("context-advance"),
                        ts_ms=now,
                        kind=RuntimeEventKind.CONTEXT_ADVANCE,
                        workflow_id=metadata.root_workflow_id,
                        invocation_id=metadata.invocation_id,
                        context_id=metadata.context_id,
                        context_epoch=metadata.context_epoch,
                        attributes={"source": "sglang_metadata"},
                    ),)
                )
                self.audit.emit(
                    "context_epoch_advanced",
                    now,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    previous_epoch=previous_epoch,
                    context_epoch=metadata.context_epoch,
                )
        for candidate in tuple(self._identity_metadata.values()):
            self._try_link_identity(candidate)

    def _try_link_identity(self, metadata: BeliefKVRequestMetadata) -> None:
        parent_id = metadata.parent_invocation_id
        if (
            parent_id is None
            or metadata.invocation_id in self._linked_invocations
            or parent_id not in self.controller.graph.invocations
        ):
            return
        relation = RelationType(metadata.relation_type)
        child = self.controller.graph.invocations.get(metadata.invocation_id)
        parent = self.controller.graph.invocations[parent_id]
        if (
            child is not None
            and child.parent_invocation_id == parent_id
            and child.relation_type == relation
            and metadata.invocation_id in parent.child_invocation_ids
        ):
            # Agent-runtime events can establish the relation before the first
            # child LLM request reaches SGLang. Metadata then confirms the same
            # edge instead of emitting a second SPAWN/CALL event.
            self._linked_invocations.add(metadata.invocation_id)
            return
        if relation in {RelationType.CALL, RelationType.SPAWN}:
            kind = (
                RuntimeEventKind.CALL
                if relation == RelationType.CALL
                else RuntimeEventKind.SPAWN
            )
            event = RuntimeEvent(
                event_id=self._next_event_id("relation"),
                ts_ms=self._now_ms(),
                kind=kind,
                workflow_id=metadata.root_workflow_id,
                invocation_id=parent_id,
                target_invocation_id=metadata.invocation_id,
                execution_mode=ExecutionMode(metadata.execution_mode),
                return_target_id=metadata.return_target_id,
            )
        elif relation in {RelationType.MESSAGE, RelationType.HANDOFF}:
            event = RuntimeEvent(
                event_id=self._next_event_id("relation"),
                ts_ms=self._now_ms(),
                kind=(
                    RuntimeEventKind.MESSAGE
                    if relation == RelationType.MESSAGE
                    else RuntimeEventKind.HANDOFF
                ),
                workflow_id=metadata.root_workflow_id,
                invocation_id=parent_id,
                target_invocation_id=metadata.invocation_id,
            )
        else:
            self._linked_invocations.add(metadata.invocation_id)
            return
        self._process_events((event,))
        self._linked_invocations.add(metadata.invocation_id)
        self.audit.emit(
            "causal_relation_linked",
            self._now_ms(),
            workflow_id=metadata.root_workflow_id,
            parent_invocation_id=parent_id,
            invocation_id=metadata.invocation_id,
            relation_type=metadata.relation_type,
        )

    def _emit(
        self,
        metadata: BeliefKVRequestMetadata,
        kind: RuntimeEventKind,
        *,
        attributes: dict[str, Any],
    ) -> None:
        self._process_events(
            (RuntimeEvent(
                event_id=self._next_event_id(kind.value),
                ts_ms=self._now_ms(),
                kind=kind,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
                context_epoch=metadata.context_epoch,
                attributes=attributes,
            ),)
        )

    def _match_terminal_node(self, token_ids: list[int]) -> Any | None:
        if not token_ids:
            return None
        match = self.tree_cache.match_prefix(token_ids)
        return getattr(match, "last_device_node", None)

    @staticmethod
    def _metadata(req: Any) -> BeliefKVRequestMetadata | None:
        raw = getattr(req, "beliefkv_metadata", None)
        if raw is None:
            return None
        return raw if isinstance(raw, BeliefKVRequestMetadata) else BeliefKVRequestMetadata.from_wire(raw)

    def _next_event_id(self, prefix: str) -> str:
        self._event_sequence += 1
        return f"sglang-{prefix}-{self._event_sequence}"
