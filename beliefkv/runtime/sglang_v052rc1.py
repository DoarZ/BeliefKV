from __future__ import annotations

import atexit
from dataclasses import dataclass, field, replace
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
)
from beliefkv.runtime.sglang_adapter import (
    BackendSubmission,
    BeliefKVRequestMetadata,
    SGLangSchedulerBridge,
)


class SGLangBackendError(RuntimeError):
    pass


@dataclass
class SGLangNodeRegistry:
    """Generation-checked mapping from BeliefKV handles to Radix TreeNodes."""

    cache_generation: int = 0
    _nodes: dict[PageHandle, Any] = field(default_factory=dict)
    _node_state: dict[int, tuple[tuple[Any, ...], PageHandle]] = field(
        default_factory=dict
    )
    _next_generation: int = 0

    def register(
        self, node: Any, fingerprint: tuple[Any, ...] | None = None
    ) -> PageHandle:
        node_id = int(node.id)
        if fingerprint is None:
            key = getattr(node, "key", None)
            key_length = len(key) if key is not None else 0
            first = key[0] if key_length else None
            last = key[-1] if key_length else None
            fingerprint = (id(node), key_length, first, last, self.cache_generation)
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
            raise SGLangBackendError(f"stale node extent generation: {handle}")
        try:
            node = self._nodes[handle]
        except KeyError as exc:
            raise SGLangBackendError(
                f"unknown or stale Radix node handle: {handle}"
            ) from exc
        if int(node.id) != handle.page_id:
            raise SGLangBackendError("registry/node id divergence")
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
    accepted_handles: set[PageHandle] = field(default_factory=set)
    transfer_handles: set[PageHandle] = field(default_factory=set)
    completed_handles: set[PageHandle] = field(default_factory=set)
    rejected_handles: set[PageHandle] = field(default_factory=set)
    rejection_reasons: dict[PageHandle, str] = field(default_factory=dict)
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
    ) -> None:
        required = (
            "write_backup",
            "check_hicache_events",
            "_evict_backuped",
            "_evict_regular",
            "load_back",
        )
        missing = [name for name in required if not hasattr(tree_cache, name)]
        if missing:
            raise SGLangBackendError(
                f"tree cache lacks required HiCache methods: {missing}"
            )
        self.tree_cache = tree_cache
        self.registry = registry
        self._now_ms = now_ms or (lambda: time.monotonic() * 1000.0)
        self._pending: dict[str, _PendingNodeCommand] = {}
        self._acks: list[CommandAck] = []

    def submit(self, command: ResolvedCommand) -> BackendSubmission:
        command_id = command.command.command_id
        if command_id in self._pending:
            raise SGLangBackendError(f"duplicate command: {command_id}")
        pending = _PendingNodeCommand(command)
        self._pending[command_id] = pending
        for item in command.page_actions:
            try:
                node = self.registry.resolve(item.handle)
                self._submit_page(pending, node, item.handle, item.action)
            except (SGLangBackendError, AssertionError, RuntimeError) as error:
                self._reject(pending, item.handle, error)
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

    def _submit_page(
        self,
        pending: _PendingNodeCommand,
        node: Any,
        handle: PageHandle,
        action: PhysicalPageAction,
    ) -> None:
        if getattr(node, "lock_ref", 0) > 0 or getattr(node, "loading", False):
            raise SGLangBackendError("node is engine-locked or loading")
        if action == PhysicalPageAction.START_D2H:
            if getattr(node, "value", None) is None:
                raise SGLangBackendError("cannot back up an evicted node")
            self._require_gpu_ancestor_closure(node)
            if getattr(node, "backuped", False):
                if node.id in self.tree_cache.ongoing_write_through:
                    pending.transfer_handles.add(handle)
                elif pending.resolved.command.kind == CommandKind.OFFLOAD_CONTEXT:
                    self._require_no_gpu_descendants(node)
                    self.tree_cache._evict_backuped(node)
                    pending.completed_handles.add(handle)
                else:
                    pending.completed_handles.add(handle)
            else:
                written = self.tree_cache.write_backup(node)
                if written <= 0:
                    raise SGLangBackendError("HiCache host allocation failed")
                pending.transfer_handles.add(handle)
            pending.accepted_handles.add(handle)
        elif action == PhysicalPageAction.COMMIT_CPU:
            if (
                getattr(node, "host_value", None) is None
                or getattr(node, "value", None) is None
            ):
                raise SGLangBackendError("COMMIT requires GPU+CPU clean node")
            if node.id in self.tree_cache.ongoing_write_through:
                raise SGLangBackendError("COMMIT cannot race an active D2H copy")
            self._require_no_gpu_descendants(node)
            self.tree_cache._evict_backuped(node)
            pending.accepted_handles.add(handle)
            pending.completed_handles.add(handle)
        elif action == PhysicalPageAction.START_H2D:
            if (
                getattr(node, "value", None) is not None
                or getattr(node, "host_value", None) is None
            ):
                raise SGLangBackendError("H2D requires an evicted backed-up node")
            ancestor = getattr(node, "parent", None)
            while ancestor is not None and ancestor is not self.tree_cache.root_node:
                if getattr(ancestor, "evicted", False):
                    raise SGLangBackendError(
                        "H2D selection violates HiCache ancestor closure"
                    )
                ancestor = getattr(ancestor, "parent", None)
            loaded = self.tree_cache.load_back(node)
            if loaded is None:
                raise SGLangBackendError("HiCache device allocation failed")
            pending.accepted_handles.add(handle)
            pending.transfer_handles.add(handle)
        elif action == PhysicalPageAction.DROP:
            if getattr(node, "value", None) is None:
                if getattr(node, "host_value", None) is None:
                    raise SGLangBackendError("cannot drop an extent with no KV copy")
                if getattr(node, "host_ref_counter", 0) > 0 or node.children:
                    raise SGLangBackendError("host node is protected or not a leaf")
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
                    raise SGLangBackendError("GPU-only DROP requires a Radix leaf")
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
            acks.append(
                CommandAck(
                    command_id=pending.resolved.command.command_id,
                    status=CommandStatus.CANCELLED,
                    completed_ts_ms=float(self._now_ms()),
                    actual_bytes=0,
                    reason=reason,
                )
            )
        self._pending.clear()
        return acks

    def poll_acks(self) -> list[CommandAck]:
        self.tree_cache.check_hicache_events()
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

    def _refresh(self, pending: _PendingNodeCommand) -> None:
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
                            "D2H ended without an authoritative host copy"
                        )
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
                            "H2D ended without an authoritative GPU copy"
                        )
                    pending.completed_handles.add(handle)
            except (SGLangBackendError, AssertionError, RuntimeError) as error:
                self._reject(pending, handle, error)

    @staticmethod
    def _reject(
        pending: _PendingNodeCommand, handle: PageHandle, error: BaseException
    ) -> None:
        pending.rejected_handles.add(handle)
        pending.rejection_reasons[handle] = f"{type(error).__name__}: {error}"

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
                    "D2H target has an evicted Radix ancestor"
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
                raise SGLangBackendError("Radix descendant cycle detected")
            seen.add(identity)
            if getattr(child, "value", None) is not None:
                raise SGLangBackendError(
                    "GPU eviction requires all Radix descendants off device"
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
        self._acks.append(
            CommandAck(
                command_id=command.command_id,
                status=status,
                completed_ts_ms=float(self._now_ms()),
                actual_bytes=actual_bytes,
                page_handles=handles,
                reason=reason,
            )
        )
        self._pending.pop(command.command_id, None)


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
            self.tree_cache, self.registry, now_ms=self._now_ms
        )
        self.bridge = SGLangSchedulerBridge(self.controller, self.backend)
        self._deferred_requests: dict[str, Any] = {}
        self._admitted_request_ids: set[str] = set()
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
        self._stalled_command_audited: set[str] = set()
        self._tree_dirty = True
        self._closed = False
        self.audit = RuntimeAuditLog(self.config.runtime_audit_path)
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
        self.audit.emit("runtime_shutdown", self._now_ms())
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
        for ack in self.bridge.drain_acks():
            self.audit.emit(
                "transfer_acknowledged",
                self._now_ms(),
                command_id=ack.command_id,
                status=ack.status.value,
                actual_bytes=ack.actual_bytes,
                page_count=len(ack.page_handles),
                reason=ack.reason,
            )
            if ack.status in {CommandStatus.PARTIAL, CommandStatus.REJECTED}:
                self._tree_dirty = True
        self.sync_tree()
        self._report_allocator_usage()
        tick = self.bridge.scheduler_step(self._now_ms(), drain_acks=False)
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
                    actual_hbm_used_bytes=self.controller.actual_hbm_used_bytes,
                )
                self._last_admission_audit = audit_state
        else:
            self._last_admission_audit = None
        if tick.transfer is not None:
            self.audit.emit(
                "transfer_dispatched",
                tick.now_ms,
                command_id=tick.transfer.command.command_id,
                kind=tick.transfer.command.kind.value,
                context_id=tick.transfer.command.context_id,
                context_epoch=tick.transfer.command.context_epoch,
                selected_bytes=tick.transfer.resolved_bytes,
                page_count=len(tick.transfer.page_actions),
                policy_reason=tick.transfer.command.metadata.get("reason"),
            )
        for ack in tick.local_acks:
            self.audit.emit(
                "transfer_rejected_local",
                tick.now_ms,
                command_id=ack.command_id,
                status=ack.status.value,
                reason=ack.reason,
            )
        for command_id in getattr(tick, "stalled_command_ids", ()):
            if command_id in self._stalled_command_audited:
                continue
            self._stalled_command_audited.add(command_id)
            self.audit.emit(
                "transfer_watchdog_expired",
                tick.now_ms,
                command_id=command_id,
                action="preserve_physical_state_and_allow_native_reclaim",
            )
        if decision is not None and decision.admitted:
            req = self._deferred_requests.pop(decision.request_id, None)
            if req is None:
                raise SGLangBackendError(
                    f"admitted request is missing: {decision.request_id}"
                )
            self._admitted_request_ids.add(req.rid)
            self.scheduler._add_admitted_beliefkv_request(req)
            metadata = self._metadata(req)
            assert metadata is not None
            self.audit.emit(
                "request_admitted",
                tick.now_ms,
                request_id=req.rid,
                workflow_id=metadata.root_workflow_id,
                invocation_id=metadata.invocation_id,
                reserved_bytes=decision.reserved_bytes,
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
            if previous_lock_ref > 0 and page.engine_lock_ref == 0:
                for context_id in page.owner_contexts:
                    self.controller.unblock_context(context_id)
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
        self._tree_dirty = False

    def _report_allocator_usage(self) -> None:
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
        self.controller.report_hbm_usage(
            used_bytes, workflow_charges=workflow_charges
        )
        self.controller.report_engine_activity(len(engine_request_ids))

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
