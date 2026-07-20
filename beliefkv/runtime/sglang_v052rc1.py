from __future__ import annotations

import atexit
from collections import deque
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
from beliefkv.policy.admission import AdmissionRequest
from beliefkv.runtime.audit import RuntimeAuditLog
from beliefkv.runtime.event_channel import (
    JsonlRuntimeEventSink,
    RuntimeEventDatagramServer,
)
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandStatus,
    PageHandle,
    PhysicalPageAction,
    ResolvedCommand,
    ResolvedPageAction,
    TransferBlocker,
    TransferBlockerCode,
    TransferDirection,
    TransferTelemetry,
)
from beliefkv.runtime.sglang_adapter import (
    BackendSubmission,
    BeliefKVRequestMetadata,
    HiCacheCapabilities,
    SGLangSchedulerBridge,
)


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
            prepared.sort(key=lambda pair: (self._node_depth(pair[1]), pair[0].handle))

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
            self._preflight_drop(node, size_bytes)
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
    def _preflight_drop(node: Any, size_bytes: int) -> None:
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
            raise SGLangBackendError(
                "GPU DROP requires a Radix leaf or a complete bundle",
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
                written = self.tree_cache.write_backup(node)
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
        self.controller = BeliefKVController(self.config)
        self.registry = SGLangNodeRegistry()
        self.backend = HiCacheNodeCommandBackend(
            self.tree_cache,
            self.registry,
            now_ms=self._now_ms,
            h2d_context_is_busy=self._context_has_engine_request,
        )
        self.bridge = SGLangSchedulerBridge(self.controller, self.backend)
        self._deferred_requests: dict[str, Any] = {}
        self._admitted_request_ids: set[str] = set()
        self._held_h2d_admissions: dict[str, Any] = {}
        self._h2d_context_by_command: dict[str, str] = {}
        self._pending_h2d_contexts: set[str] = set()
        self._active_request_ids: set[str] = set()
        self._request_metadata_by_id: dict[str, BeliefKVRequestMetadata] = {}
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
        self._tree_dirty = True
        self._closed = False
        self.audit = RuntimeAuditLog(self.config.runtime_audit_path)
        transfer_telemetry_path = self.config.transfer_telemetry_path
        if transfer_telemetry_path is None and self.config.runtime_audit_path is not None:
            transfer_telemetry_path = str(
                Path(self.config.runtime_audit_path).with_name("transfer_telemetry.jsonl")
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
        self._emit_controller_timing_summary()
        controller = getattr(self, "controller", None)
        transfer_guard = getattr(controller, "transfer_guard", None)
        if transfer_guard is not None:
            self.audit.emit(
                "transfer_retry_guard_summary",
                self._now_ms(),
                **transfer_guard.summary(),
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
        if "host_capacity_bytes" not in raw:
            host_pool = getattr(scheduler.tree_cache, "token_to_kv_pool_host", None)
            host_tokens = int(getattr(host_pool, "size", max_tokens * 2))
            raw["host_capacity_bytes"] = max(1, host_tokens * kv_bytes)
        if "reserve_hbm_bytes" not in raw:
            raw["reserve_hbm_bytes"] = min(
                1 << 30, int(raw["hbm_capacity_bytes"]) // 8
            )
        return BeliefKVConfig.from_mapping(raw)

    def defer_request(self, req: Any) -> bool:
        raw_metadata = getattr(req, "beliefkv_metadata", None)
        if raw_metadata is None:
            return False
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
                phase="before_defer",
            )
            return True
        self._ensure_causal_identity(metadata)
        if req.rid in self._deferred_requests or req.rid in self._admitted_request_ids:
            raise SGLangBackendError(f"duplicate deferred request id: {req.rid}")
        max_new_tokens = int(getattr(req.sampling_params, "max_new_tokens", 0) or 0)
        req.init_next_round_input(self.tree_cache)
        estimated_cache_hit_tokens = len(getattr(req, "prefix_indices", ()))
        uncached_prompt_tokens = max(
            0, len(req.origin_input_ids) - estimated_cache_hit_tokens
        )
        self.controller.submit_request(
            AdmissionRequest(
                request_id=req.rid,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
                context_epoch=metadata.context_epoch,
                submitted_ts_ms=self._now_ms(),
                uncached_prompt_tokens=uncached_prompt_tokens,
                expected_output_tokens=max_new_tokens,
                kv_bytes_per_token=self.config.kv_bytes_per_token,
                prompt_tokens=len(req.origin_input_ids),
            )
        )
        self._deferred_requests[req.rid] = req
        self._request_metadata_by_id[req.rid] = metadata
        self.audit.emit(
            "request_deferred",
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
        )
        return True

    def scheduler_step(self) -> None:
        step_started_ns = time.perf_counter_ns()
        telemetry_overhead_ns = 0
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
        completed_h2d_contexts = self._retire_h2d_commands(acks)
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
                self._tree_dirty = True
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
        self._report_allocator_usage()
        for context_id in completed_h2d_contexts:
            self._release_h2d_admissions(context_id)
        if hasattr(self, "_last_resource_telemetry_ms"):
            self._emit_resource_snapshot(force=bool(acks or telemetry))
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
                h2d_commands[command_id] = context_id
                pending_contexts.add(context_id)
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
        if decision is not None and decision.admitted:
            req = self._deferred_requests.get(decision.request_id)
            metadata = self._metadata(req) if req is not None else None
            if (
                req is not None
                and metadata is not None
                and metadata.context_id
                in getattr(self, "_pending_h2d_contexts", set())
            ):
                held = getattr(self, "_held_h2d_admissions", None)
                if held is None:
                    held = {}
                    self._held_h2d_admissions = held
                held[decision.request_id] = decision
                self.audit.emit(
                    "request_admission_waiting_h2d",
                    tick.now_ms,
                    request_id=decision.request_id,
                    context_id=metadata.context_id,
                    reserved_bytes=decision.reserved_bytes,
                )
            else:
                self._apply_admission(decision, now_ms=tick.now_ms)
        self._record_scheduler_timing(
            total_ms=(time.perf_counter_ns() - step_started_ns) / 1_000_000.0,
            telemetry_ms=telemetry_overhead_ns / 1_000_000.0,
            telemetry_count=len(telemetry),
        )

    def _apply_admission(self, decision: Any, *, now_ms: float) -> None:
        req = self._deferred_requests.pop(decision.request_id, None)
        if req is None:
            raise SGLangBackendError(
                f"admitted request is missing: {decision.request_id}"
            )
        getattr(self, "_held_h2d_admissions", {}).pop(
            decision.request_id, None
        )
        self._admitted_request_ids.add(req.rid)
        self.scheduler._add_admitted_beliefkv_request(req)
        metadata = self._metadata(req)
        assert metadata is not None
        self.audit.emit(
            "request_admitted",
            now_ms,
            request_id=req.rid,
            workflow_id=metadata.root_workflow_id,
            invocation_id=metadata.invocation_id,
            reserved_bytes=decision.reserved_bytes,
        )

    def _retire_h2d_commands(
        self, acks: tuple[CommandAck, ...]
    ) -> set[str]:
        command_contexts = getattr(self, "_h2d_context_by_command", None)
        if command_contexts is None:
            return set()
        pending_contexts = getattr(self, "_pending_h2d_contexts", set())
        completed_contexts: set[str] = set()
        for ack in acks:
            context_id = command_contexts.pop(ack.command_id, None)
            if context_id is None:
                continue
            if context_id not in command_contexts.values():
                pending_contexts.discard(context_id)
                completed_contexts.add(context_id)
        return completed_contexts

    def _release_h2d_admissions(self, context_id: str) -> None:
        held = getattr(self, "_held_h2d_admissions", {})
        for request_id, decision in tuple(held.items()):
            req = self._deferred_requests.get(request_id)
            metadata = self._metadata(req) if req is not None else None
            if metadata is None or metadata.context_id != context_id:
                continue
            refresh = getattr(req, "init_next_round_input", None)
            if callable(refresh):
                refresh(self.tree_cache)
            self.audit.emit(
                "request_admission_h2d_dependency_satisfied",
                self._now_ms(),
                request_id=request_id,
                context_id=context_id,
                cache_hit_tokens=len(getattr(req, "prefix_indices", ())),
            )
            self._apply_admission(decision, now_ms=self._now_ms())

    def _context_has_engine_request(self, context_id: str) -> bool:
        request_ids = set(getattr(self, "_admitted_request_ids", set()))
        request_ids.update(getattr(self, "_active_request_ids", set()))
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            running_batch = getattr(scheduler, "running_batch", None)
            requests = list(getattr(running_batch, "reqs", ()) or ())
            requests.extend(getattr(scheduler, "waiting_queue", ()) or ())
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
        if not samples:
            return
        event_samples = [item for item in samples if item[2] > 0]

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
        )

    def on_abort_request(self, abort_request: Any) -> int:
        """Remove requests hidden in BeliefKV's pre-SGLang admission queue."""

        abort_all = bool(getattr(abort_request, "abort_all", False))
        rid_prefix = str(getattr(abort_request, "rid", ""))

        def matches(request_id: str) -> bool:
            return abort_all or request_id.startswith(rid_prefix)

        removed = 0
        for request_id, req in tuple(self._deferred_requests.items()):
            if not matches(request_id):
                continue
            self._deferred_requests.pop(request_id)
            getattr(self, "_held_h2d_admissions", {}).pop(request_id, None)
            self.controller.admission.cancel(request_id)
            self.scheduler.send_to_tokenizer.send_pyobj(
                abort_request.__class__(rid=req.rid)
            )
            removed += 1
        for request_id in tuple(self._admitted_request_ids):
            if matches(request_id):
                self._admitted_request_ids.remove(request_id)
                self.controller.admission.cancel(request_id)
        for request_id in tuple(self._request_metadata_by_id):
            if matches(request_id):
                self._request_metadata_by_id.pop(request_id, None)
        return removed

    def on_batch_selected(self, batch: Any) -> None:
        now_ms = self._now_ms()
        self._charge_previous_batch(now_ms)
        if batch is None or not getattr(batch, "reqs", None):
            self._last_batch_selected_ms = now_ms
            self._last_batch_workflow_counts = {}
            return
        workflow_counts: dict[str, int] = {}
        for req in batch.reqs:
            metadata = self._metadata(req)
            if metadata is None:
                continue
            if self._metadata_scope_is_terminal(metadata):
                self._admitted_request_ids.discard(req.rid)
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
            workflow_counts[metadata.root_workflow_id] = (
                workflow_counts.get(metadata.root_workflow_id, 0) + 1
            )
            if req.rid in self._admitted_request_ids:
                self.controller.acknowledge_admission(req.rid)
                self._admitted_request_ids.remove(req.rid)
            if req.rid not in self._active_request_ids:
                self._active_request_ids.add(req.rid)
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
                self._tree_dirty = True
        self._last_batch_selected_ms = now_ms
        self._last_batch_workflow_counts = workflow_counts

    def _charge_previous_batch(self, now_ms: float) -> None:
        if self._last_batch_selected_ms is None or not self._last_batch_workflow_counts:
            return
        elapsed_ms = max(0.0, now_ms - self._last_batch_selected_ms)
        total_reqs = sum(self._last_batch_workflow_counts.values())
        if elapsed_ms == 0 or total_reqs == 0:
            return
        for workflow_id, count in self._last_batch_workflow_counts.items():
            self.controller.fairness.charge_service(
                workflow_id, elapsed_ms * count / total_reqs
            )

    def reorder_waiting_queue(self, waiting_queue: list[Any]) -> None:
        """Round-robin root workflows while retaining SGLang's local ordering."""

        tagged_positions: list[int] = []
        buckets: dict[str, list[tuple[Any, BeliefKVRequestMetadata]]] = {}
        for index, req in enumerate(waiting_queue):
            metadata = self._metadata(req)
            if metadata is None:
                continue
            tagged_positions.append(index)
            buckets.setdefault(metadata.root_workflow_id, []).append((req, metadata))
        if len(buckets) <= 1:
            return

        for workflow_id, requests in buckets.items():
            frontier_order = {
                item.invocation_id: index
                for index, item in enumerate(
                    self.controller.frontier.candidates(workflow_id)
                )
            }
            requests.sort(
                key=lambda item: frontier_order.get(
                    item[1].invocation_id, 1 << 30
                )
            )
        workflow_order = self.controller.fairness.ordered(
            set(buckets),
            memory_charges=self.controller.workflow_memory_charges(),
            hbm_capacity_bytes=self.config.hbm_capacity_bytes,
        )
        ordered_requests: list[Any] = []
        while any(buckets.values()):
            for workflow_id in workflow_order:
                if buckets[workflow_id]:
                    ordered_requests.append(buckets[workflow_id].pop(0)[0])
        for position, req in zip(tagged_positions, ordered_requests):
            waiting_queue[position] = req

    def on_cache_finished(self, req: Any, token_ids: list[int]) -> None:
        self._ensure_allocator_radix_consistency(reason="cache_finished")
        metadata = self._metadata(req)
        if metadata is None:
            return
        last_node = self._match_terminal_node(token_ids)
        if last_node is not None:
            self._terminal_node_by_context[metadata.context_id] = last_node
        self._tree_dirty = True
        self._active_request_ids.discard(req.rid)
        terminal_cancelled = req.rid in self._terminal_cancelled_request_ids
        self._terminal_cancelled_request_ids.discard(req.rid)
        self._request_metadata_by_id.pop(req.rid, None)
        if terminal_cancelled:
            self.audit.emit(
                "terminal_request_abort_finished",
                self._now_ms(),
                request_id=req.rid,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                context_id=metadata.context_id,
            )
            return
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
        last_node = self._match_terminal_node(token_ids)
        if last_node is not None:
            self._terminal_node_by_context[metadata.context_id] = last_node
        self._tree_dirty = True

    def on_cache_reset(self) -> None:
        for ack in self.backend.abort_all(reason="authoritative_cache_reset"):
            self.controller.acknowledge_command(ack)
        for observation in self.backend.poll_transfer_telemetry():
            self.controller.observe_transfer_telemetry(observation)
            self._emit_transfer_telemetry(observation)
        self.controller.reset_transfer_attempts()
        self._held_h2d_admissions.clear()
        self._h2d_context_by_command.clear()
        self._pending_h2d_contexts.clear()
        self.registry.reset()
        for handle, page in tuple(self.controller.page_index.pages.items()):
            if page.residency.value != "dead" and page.transfer_idle:
                self.controller.page_index.invalidate_page(handle)
        self._terminal_node_by_context.clear()
        self._tree_dirty = True

    def on_radix_mutation(self) -> None:
        self._tree_dirty = True

    def on_lock_change(self, node: Any) -> None:
        root = self.tree_cache.root_node
        while node is not None and node is not root:
            handle = self.registry.current_handle(int(node.id))
            page = self.controller.page_index.pages.get(handle) if handle else None
            if page is None or page.residency.value == "dead":
                self._tree_dirty = True
                return
            previous_lock_ref = page.engine_lock_ref
            page.engine_lock_ref = max(0, int(getattr(node, "lock_ref", 0)))
            if previous_lock_ref != page.engine_lock_ref:
                self.controller.notify_resource_state_changed()
            node = getattr(node, "parent", None)

    def process_runtime_event(self, event: RuntimeEvent) -> None:
        """Entry point used by an instrumented agent-runtime adapter."""

        self._process_events((event,))

    def _process_events(self, events: tuple[RuntimeEvent, ...]) -> None:
        committed_events, adjustments = self._commit_event_times(events)
        self.controller.process_runtime_events(committed_events)
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
            if request_id in self._deferred_requests:
                phase = "deferred"
            elif request_id in self._admitted_request_ids:
                phase = "admitted"
            elif request_id in self._active_request_ids:
                phase = "active"
                self._terminal_cancelled_request_ids.add(request_id)
            else:
                phase = "engine_owned"
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

                page.residency = PhysicalResidency(residency)
                self.controller.page_index.set_parent(handle, parent_handle)
                page.radix_depth = depth
            page.engine_lock_ref = max(0, int(getattr(node, "lock_ref", 0)))

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
        self._tree_dirty = False

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
        engine_request_ids.update(getattr(self, "_admitted_request_ids", set()))
        self._report_native_admission_capacity(engine_request_ids, allocator)
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

    def _emit_transfer_telemetry(self, telemetry: TransferTelemetry) -> None:
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
        }
        observed_ts_ms = max(float(self._now_ms()), telemetry.complete_ts_ms)
        self.audit.emit("transfer_telemetry", observed_ts_ms, **fields)
        transfer_log = getattr(self, "transfer_telemetry_log", None)
        if transfer_log is not None:
            transfer_log.emit("transfer_telemetry", observed_ts_ms, **fields)

    def _emit_resource_snapshot(self, *, force: bool) -> None:
        now_ms = float(self._now_ms())
        last_ts = getattr(self, "_last_resource_telemetry_ms", None)
        if (
            not force
            and last_ts is not None
            and now_ms - last_ts < self.config.resource_telemetry_interval_ms
        ):
            return
        allocator = self.scheduler.token_to_kv_pool_allocator
        hbm_free_tokens = max(0, int(allocator.available_size()))
        hbm_capacity_tokens = max(0, int(self.scheduler.max_total_num_tokens))
        hbm_used_tokens = max(0, hbm_capacity_tokens - hbm_free_tokens)
        allocator_hbm_capacity_bytes = (
            hbm_capacity_tokens * self.config.kv_bytes_per_token
        )

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
        self.audit.emit(
            "resource_snapshot",
            now_ms,
            hbm_capacity_bytes=allocator_hbm_capacity_bytes,
            configured_hbm_capacity_bytes=self.config.hbm_capacity_bytes,
            hbm_used_bytes=min(
                allocator_hbm_capacity_bytes,
                hbm_used_tokens * self.config.kv_bytes_per_token,
            ),
            hbm_free_bytes=min(
                allocator_hbm_capacity_bytes,
                hbm_free_tokens * self.config.kv_bytes_per_token,
            ),
            host_capacity_bytes=host_capacity_tokens * host_bytes_per_token,
            host_used_bytes=host_used_tokens * host_bytes_per_token,
            host_free_bytes=host_free_tokens * host_bytes_per_token,
            page_index_gpu_bytes=self.controller.page_index.gpu_bytes,
            page_index_cpu_bytes=self.controller.page_index.cpu_bytes,
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
        )
        self._last_resource_telemetry_ms = now_ms

    def _report_native_admission_capacity(
        self,
        engine_request_ids: set[str],
        allocator: Any,
    ) -> None:
        """Mirror SGLang's idle PrefillAdder capacity for one liveness target."""

        self.controller.report_native_admission_capacity(None)
        if engine_request_ids:
            self._last_native_admission_audit = None
            return
        pending = self.controller.admission.pending_requests()
        if not pending:
            self._last_native_admission_audit = None
            return
        target = pending[0]
        req = self._deferred_requests.get(target.request_id)
        if req is None:
            self._last_native_admission_audit = None
            return

        req.init_next_round_input(self.tree_cache)
        page_size = max(1, int(getattr(self.scheduler, "page_size", 1)))
        extend_tokens = max(0, int(getattr(req, "extend_input_len", 0)))
        rounded_extend_tokens = (
            (extend_tokens + page_size - 1) // page_size * page_size
        )
        refreshed = self.controller.admission.update_pending_estimate(
            target.request_id,
            uncached_prompt_tokens=rounded_extend_tokens,
        )

        (
            native_capacity_tokens,
            available_tokens,
            evictable_tokens,
            protected_tokens,
        ) = self._native_admission_capacity_tokens(req, allocator)
        native_capacity_bytes = min(
            self.config.hbm_capacity_bytes,
            native_capacity_tokens * self.config.kv_bytes_per_token,
        )
        self.controller.report_native_admission_capacity(
            target.request_id,
            native_capacity_bytes,
        )

        audit_state = (
            target.request_id,
            refreshed.estimated_incremental_bytes,
            native_capacity_bytes,
        )
        if audit_state != self._last_native_admission_audit:
            self.audit.emit(
                "native_admission_capacity",
                self._now_ms(),
                request_id=target.request_id,
                required_bytes=refreshed.estimated_incremental_bytes,
                available_tokens=available_tokens,
                evictable_tokens=evictable_tokens,
                protected_prefix_tokens=protected_tokens,
                native_reclaim_capacity_bytes=native_capacity_bytes,
            )
            self._last_native_admission_audit = audit_state

    def _native_admission_capacity_tokens(
        self,
        req: Any,
        allocator: Any,
    ) -> tuple[int, int, int, int]:
        available_tokens = max(0, int(allocator.available_size()))
        evictable_tokens = max(0, int(self.tree_cache.evictable_size()))
        protected_tokens = self._prefix_tokens_protected_on_admission(req)
        capacity_tokens = max(
            0,
            available_tokens + evictable_tokens - protected_tokens,
        )
        return (
            capacity_tokens,
            available_tokens,
            evictable_tokens,
            protected_tokens,
        )

    def _prefix_tokens_protected_on_admission(self, req: Any) -> int:
        """Return the evictable prefix tokens lost when SGLang locks this request."""

        root = self.tree_cache.root_node
        node = getattr(req, "last_node", None)
        protected_tokens = 0
        seen: set[int] = set()
        while node is not None and node is not root:
            identity = id(node)
            if identity in seen:
                raise SGLangBackendError("Radix parent cycle detected")
            seen.add(identity)
            if (
                int(getattr(node, "lock_ref", 0)) == 0
                and not bool(getattr(node, "evicted", False))
            ):
                value = getattr(node, "value", None)
                if value is not None:
                    protected_tokens += len(value)
            node = getattr(node, "parent", None)
        return protected_tokens

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
