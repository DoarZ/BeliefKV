from __future__ import annotations

import atexit
import copy
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import json
import time
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
    JointPlanComponentValidation,
    JointPlanCurrentState,
    JointPlanValidation,
    JointPlannerConfig,
    ObservedJointPlanner,
    validate_joint_plan,
    validate_joint_plan_components,
)
from beliefkv.policy.reference import (
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
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandQueueClass,
    CommandStatus,
    ControlCommand,
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
        loaded = self.tree_cache.load_back(
            leaf_node,
            force=True,
            allow_eviction=False,
            beliefkv_source="explicit",
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
                written = self.tree_cache.write_backup(
                    node, beliefkv_source="explicit"
                )
                if written <= 0:
                    raise SGLangBackendError(
                        "HiCache host allocation failed",
                        blocker_code=TransferBlockerCode.HOST_CAPACITY,
                        required_bytes=size_bytes,
                    )
                pending.transfer_handles.add(handle)
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
            loaded = self.tree_cache.load_back(
                node,
                force=True,
                allow_eviction=False,
                beliefkv_source="explicit",
            )
            if loaded is None:
                raise SGLangBackendError(
                    "HiCache device allocation failed",
                    blocker_code=TransferBlockerCode.DEVICE_CAPACITY,
                    required_bytes=size_bytes,
                )
            pending.accepted_handles.add(handle)
            pending.transfer_handles.add(handle)
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
                )
            )


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
        if self.config.joint_policy_enabled:
            raise SGLangBackendError(
                "online JointPlan application is not implemented in P4; "
                "set joint_policy_enabled=false and use shadow mode"
            )
        self.controller = BeliefKVController(self.config)
        self.registry = SGLangNodeRegistry()
        self.backend = HiCacheNodeCommandBackend(
            self.tree_cache,
            self.registry,
            now_ms=self._now_ms,
            h2d_context_is_busy=self._context_has_engine_request,
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
        self.running_retraction_planner = ObservedRetractionPlanner(
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
        self._h2d_context_by_command: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._pending_h2d_contexts: set[str] = set()
        self._active_request_ids: set[str] = set()
        self._request_metadata_by_id: dict[str, BeliefKVRequestMetadata] = {}
        self._request_submitted_ts_by_id: dict[str, float] = {}
        self._request_physical_start_by_id: dict[str, dict[str, Any]] = {}
        self._pending_request_physical_finish_by_id: dict[
            str, dict[str, Any]
        ] = {}
        self._terminal_cancelled_request_ids: set[str] = set()
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
        self._ticket_timing_samples: dict[str, deque[float]] = {
            "compile_ms": deque(maxlen=65_536),
            "validation_ms": deque(maxlen=65_536),
        }
        self._gpu_service_launches: deque[dict[str, Any] | None] = deque()
        self._gpu_service_sequence = 0
        self._gpu_service_sample_count = 0
        self._gpu_service_previous_completion_ms: float | None = None
        self._gpu_service_prefill_chunks_by_episode: dict[str, int] = {}
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
        self.audit = RuntimeAuditLog(self.config.runtime_audit_path)
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
        self._last_persisted_policy_snapshot_ms: float | None = None
        self._shadow_event_sequence = 0
        self._shadow_page_revision = 0
        self._shadow_telemetry_sequence = 0
        self.joint_shadow_worker: LatestWinsJointPlanWorker | None = None
        self._last_joint_shadow_result_sequence = 0
        self._joint_shadow_counts: Counter[str] = Counter()
        self._joint_shadow_strict_stale_reasons: Counter[str] = Counter()
        self._joint_shadow_readset_stale_reasons: Counter[str] = Counter()
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
        elif not self.audit.enabled:
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
            planner = ObservedJointPlanner(
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
                "start": "hicache_operation_enqueue",
                "complete": "scheduler_observed_hicache_ack",
                "first_layer_ready": "unavailable",
                "compute_wait": "unavailable",
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
                "planner": "belief_joint_observed",
                "worker_queue": "latest_wins_capacity_1_lossless_delta_merge",
                "snapshot_builder": "worker_owned_incremental_mirror",
                "validation": "dependency_scoped_optimistic",
                "strict_global_comparator": True,
                "application_connected": False,
                "joint_policy_enabled_requested": self.config.joint_policy_enabled,
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
                "residency_control": "reactive_unchanged",
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
        atexit.register(self.close)

    def close(self) -> None:
        """Release process-local logs and the runtime event socket once."""

        if self._closed:
            return
        self._closed = True
        event_server = self.event_server
        self.event_server = None
        if event_server is not None:
            event_server.close()
        event_log = self.event_log
        self.event_log = None
        if event_log is not None:
            event_log.close()
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
            self.audit.emit(
                "gpu_service_observer_summary",
                self._now_ms(),
                sample_count=self._gpu_service_sample_count,
                pending_launch_count=len(self._gpu_service_launches),
                max_samples=config.queue_service_observer_max_samples,
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
        self.audit.emit("runtime_shutdown", self._now_ms())
        transfer_telemetry_log = getattr(self, "transfer_telemetry_log", None)
        if transfer_telemetry_log is not None:
            transfer_telemetry_log.close()
        self.audit.close()

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
            if selective_retraction or self._now_ms() < cooldown_until:
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
        previous_decision_ms = getattr(self, "_last_retraction_decision_ms", None)
        if (
            previous_decision_ms is not None
            and now_ms - previous_decision_ms
            < self.config.running_batch_retraction_decision_interval_ms
        ):
            return False
        if getattr(self, "_pending_running_retraction_transaction", None) is not None:
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
        if replacement_deficit_bytes == 0 and active_excess_bytes == 0:
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

    def plan_running_batch_retraction(
        self,
        running_batch: Any,
    ) -> RunningRetractionPlan | None:
        """Plan one selective retraction at a scheduler safe point."""

        if not self.config.running_batch_retraction_enabled:
            return None
        now_ms = float(self._now_ms())
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
        page_size = max(
            1,
            int(
                getattr(
                    getattr(self.scheduler, "server_args", None),
                    "page_size",
                    1,
                )
                or 1
            ),
        )
        cooldowns = getattr(self, "_retraction_cooldown_until_by_request", {})
        retraction_counts = getattr(self, "_retraction_counts_by_request", {})
        candidates: list[RunningRetractionCandidate] = []
        for req in requests:
            metadata = self._metadata(req)
            request_id = str(getattr(req, "rid", ""))
            if metadata is None or not request_id:
                continue
            prefix_tokens = _sequence_length(getattr(req, "prefix_indices", None))
            sequence_tokens = max(
                int(getattr(req, "seqlen", 0) or 0),
                _sequence_length(getattr(req, "origin_input_ids", None))
                + _sequence_length(getattr(req, "output_ids", None)),
            )
            last_uncached_pos = (prefix_tokens // page_size) * page_size
            private_bytes = (
                max(0, sequence_tokens - last_uncached_pos)
                * self.config.kv_bytes_per_token
            )
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
        decision = self._running_retraction_planner().decide(snapshot)
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
        )
        self._pending_running_retraction_transaction = transaction
        self._running_retraction_transactions.append(transaction)
        self._running_retraction_counts["planned"] += 1
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
        self._pending_selective_retraction_ids.update(actual_ids)
        self._retracted_engine_request_ids.update(actual_ids)
        cooldown_until = now_ms + self.config.running_batch_retraction_cooldown_ms
        for request_id in actual_ids:
            self._retraction_counts_by_request[request_id] += 1
            self._retraction_cooldown_until_by_request[request_id] = cooldown_until

        self._mark_full_tree_rebuild()
        self.sync_tree(force=True)
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
            if not self.controller.enqueue_control_command(command):
                queue_conflict = True
                continue
            transaction.pending_command_id = command.command_id
            transaction.pending_command_kind = command.kind
            transaction.residency_command_ids.append(command.command_id)
            transaction.stage = "residency_pending"
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

    def _running_retraction_planner(self) -> ObservedRetractionPlanner:
        planner = getattr(self, "running_retraction_planner", None)
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
            self.running_retraction_planner = planner
        return planner

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
        workflow_ids = {item[2].root_workflow_id for item in tagged}
        fair_order = self.controller.fairness.ordered(
            workflow_ids,
            memory_charges=self.controller.workflow_memory_charges(),
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
        )
        fair_rank = {
            workflow_id: index for index, workflow_id in enumerate(fair_order)
        }
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
        # This preserves prepare/commit/drop state-machine ordering.
        acks = self.bridge.drain_acks()
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
        for context_id, bundle_ids, status in retired_h2d:
            if status == CommandStatus.COMPLETED:
                self._release_h2d_waiters(context_id)
            else:
                self._invalidate_h2d_waiters(
                    context_id,
                    bundle_ids=bundle_ids,
                    status=status,
                )
        if hasattr(self, "_last_resource_telemetry_ms"):
            self._emit_resource_snapshot(force=bool(acks or telemetry))
        policy_snapshot_log = getattr(self, "policy_snapshot_log", None)
        if (
            policy_snapshot_log is not None and policy_snapshot_log.enabled
        ) or getattr(self, "joint_shadow_worker", None) is not None:
            self._maybe_record_policy_snapshot(
                self._runtime_resource_observation()
            )
        tick = self.bridge.scheduler_step(self._now_ms(), drain_acks=False)
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
                bundle_ids = (
                    (bundle.bundle_id,)
                    if bundle is not None and bundle.bundle_id
                    else self._context_restore_bundle_ids(context_id)
                )
                if not bundle_ids:
                    bundle_ids = (f"context:{context_id}",)
                h2d_commands[command_id] = (context_id, bundle_ids)
                pending_contexts.add(context_id)
                self._mark_context_wait_restore(
                    context_id,
                    bundle_ids=bundle_ids,
                    reason="h2d_inflight",
                )
            action_counts: dict[str, int] = {}
            for page_action in tick.transfer.page_actions:
                action = page_action.action.value
                action_counts[action] = action_counts.get(action, 0) + 1
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
                policy_reason=tick.transfer.command.metadata.get("reason"),
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
        # Native waiting requests do not own a req-pool slot or a Radix lock.
        # Counting them here would reject every explicit restore after making
        # the waiting queue visible.
        request_ids = set(getattr(self, "_active_request_ids", set()))
        request_ids.difference_update(
            getattr(self, "_retracted_engine_request_ids", set())
        )
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            running_batch = getattr(scheduler, "running_batch", None)
            requests = list(getattr(running_batch, "reqs", ()) or ())
            chunked_req = getattr(scheduler, "chunked_req", None)
            if chunked_req is not None:
                requests.append(chunked_req)
            request_ids.update(
                str(getattr(req, "rid", f"object:{id(req)}")) for req in requests
            )
        for request_id in request_ids:
            metadata = getattr(self, "_request_metadata_by_id", {}).get(request_id)
            if metadata is not None and metadata.context_id == context_id:
                return True
        return False

    def _record_scheduler_timing(
        self, *, total_ms: float, telemetry_ms: float, telemetry_count: int
    ) -> None:
        samples = getattr(self, "_scheduler_timing_samples", None)
        if samples is None:
            samples = deque(maxlen=65_536)
            self._scheduler_timing_samples = samples
        samples.append((total_ms, telemetry_ms, telemetry_count))

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
                self._request_metadata_by_id.pop(request_id, None)
                ledger = getattr(self, "_lock_service_ledger", None)
                if ledger is not None:
                    ledger.forget(str(request_id))
                getattr(self, "_request_submitted_ts_by_id", {}).pop(
                    request_id, None
                )
                getattr(self, "_retracted_engine_request_ids", set()).discard(
                    request_id
                )
                getattr(self, "_pending_selective_retraction_ids", set()).discard(
                    request_id
                )
                getattr(self, "_retraction_cooldown_until_by_request", {}).pop(
                    request_id, None
                )
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
            }
        self._gpu_service_launches.append(descriptor)

    def on_batch_completed(self, batch: Any) -> None:
        mode = getattr(batch, "forward_mode", None)
        if mode is None or mode.is_idle() or mode.is_dummy_first():
            return
        now_ms = self._now_ms()
        self._observe_request_service_completed(
            batch,
            now_ms=now_ms,
            phase=self._gpu_batch_phase(mode),
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
        observed_admission_enabled = (
            self.config.observed_admission_scheduling_enabled
        )
        if observed_admission_enabled:
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
        if observed_window is not None:
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
                "policy_error_reactive_fallback"
                if observed_admission_error is not None
                else "visible_bounded_fallback"
            )
        retraction_priority = tuple(
            request_id
            for request_id in getattr(
                self, "_retraction_priority_request_ids", ()
            )
            if request_id in entries
            and entries[request_id].state == AdmissionSideState.VISIBLE_PENDING
        )
        if retraction_priority:
            priority_set = set(retraction_priority)
            ordered_request_ids = (
                retraction_priority
                + tuple(
                    request_id
                    for request_id in ordered_request_ids
                    if request_id not in priority_set
                )
            )
            ticket_source = f"{ticket_source}+retraction_replacement"
            ticket_reason = "retraction_reclaim_confirmed"
        self._retraction_priority_request_ids = ()
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
            active_budget_binding=(
                observed_window is not None
                and observed_window.active_growth_budget_bytes
                < native_hbm_bytes
            ),
            observed_admission_error=observed_admission_error,
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
        if admitted:
            self._ticket_selected_request_ids.add(request_id)
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
                "reserved_bytes": 0,
            }
        else:
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
                reason=f"retraction_residency_pending:{transaction.transaction_id}",
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
            self.controller.visible_admission.set_wait_restore(
                request_id,
                restore_bundle_ids or (f"context:{metadata.context_id}",),
                reason=(
                    "h2d_inflight"
                    if metadata.context_id in self._pending_h2d_contexts
                    else "cpu_prefix_requires_restore"
                ),
            )
        elif transition_open:
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

        node = getattr(req, "last_node", None) if req is not None else None
        registry = getattr(self, "registry", None)
        tree_cache = getattr(self, "tree_cache", None)
        root = getattr(tree_cache, "root_node", None)
        if node is None or registry is None or root is None:
            return self._context_restore_bundle_ids(context_id)
        page_index = self.controller.page_index
        result: list[str] = []
        seen: set[int] = set()
        while node is not None and node is not root:
            identity = id(node)
            if identity in seen:
                raise SGLangBackendError("Radix parent cycle detected")
            seen.add(identity)
            handle = registry.current_handle(int(node.id))
            if handle is None:
                return self._context_restore_bundle_ids(context_id)
            page = page_index.pages.get(handle)
            if page is None or page.residency == PhysicalResidency.DEAD:
                return self._context_restore_bundle_ids(context_id)
            if page.cpu_resident and not page.gpu_resident:
                result.append(
                    f"page:{handle.page_id}:{handle.allocation_generation}"
                )
            node = getattr(node, "parent", None)
        return tuple(sorted(result))

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
        remaining = self._context_restore_bundle_ids(context_id)
        self._mark_context_wait_restore(
            context_id,
            bundle_ids=remaining or bundle_ids or (f"context:{context_id}",),
            reason=f"h2d_{status.value}",
        )

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
        now_ms = self._now_ms()
        for request_id, finish in tuple(sorted(pending.items())):
            start = starts.get(request_id)
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
        self._request_metadata_by_id.pop(req.rid, None)
        getattr(self, "_request_submitted_ts_by_id", {}).pop(req.rid, None)
        if terminal_cancelled:
            getattr(self, "_request_physical_start_by_id", {}).pop(req.rid, None)
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
        getattr(self, "_pending_request_physical_finish_by_id", {}).clear()
        self._request_partial_commit_count.clear()
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
        )
        self._emit_transfer_telemetry(
            telemetry,
            telemetry_origin="native_hicache_callback",
            backend_operation_id=record.get("backend_operation_id"),
            radix_node_ids=node_ids,
            owner_context_ids=tuple(sorted(owner_context_ids)),
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
        control_state = self.controller.policy_control_state(observation.ts_ms)
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
        )
        # Queue/graph transitions publish immediately. Fairness service and
        # allocator progress change on nearly every decode quantum, so they
        # belong to the interval-gated progress signature below.
        structural_signature: tuple[object, ...] = (
            stamp.graph_version,
            stamp.consumer_version,
            stamp.transfer_epoch,
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
        elapsed = (
            float("inf")
            if self._last_policy_snapshot_ms is None
            else observation.ts_ms - self._last_policy_snapshot_ms
        )
        watchdog_due = bool(additional_runnable) and elapsed >= (
            self.config.reference_policy_snapshot_min_interval_ms
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
                )
                submission = worker.submit_delta(delta)
            except Exception as error:
                self._joint_shadow_counts["submission_failed"] += 1
                self.audit.emit(
                    "joint_plan_shadow_submit_failed",
                    observation.ts_ms,
                    error=f"{type(error).__name__}: {error}",
                    trigger=trigger,
                    application_connected=False,
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
                    application_connected=False,
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
                application_connected=False,
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
                    application_connected=False,
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
                            application_connected=False,
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
                            application_connected=False,
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
            "application_connected": False,
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
                application_connected=False,
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
            application_connected=False,
        )

    def _policy_runtime_runnable(
        self, now_ms: float
    ) -> tuple[RunnableInvocation, ...]:
        result: dict[str, RunnableInvocation] = {}
        raw_waiting_queue = getattr(self.scheduler, "waiting_queue", None)
        waiting_queue = (
            tuple(raw_waiting_queue) if raw_waiting_queue is not None else ()
        )
        for req in waiting_queue:
            metadata = self._metadata(req)
            if metadata is None or self._metadata_scope_is_terminal(metadata):
                continue
            max_new_tokens = int(
                getattr(req.sampling_params, "max_new_tokens", 0) or 0
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
                    f"engine_waiting:{execution_mode}:{relation_type}"
                ),
                program_id=metadata.agent_instance_id,
            )
        return tuple(sorted(result.values(), key=lambda item: item.request_id))

    def _policy_capabilities(self) -> CapabilityReport:
        limitations = [
            "running decode batch cannot yet be reordered by JointPlan",
            "HiCache 0.5.2rc1 exposes one in-flight physical operation",
            "reference outputs are shadow-only in P3",
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
