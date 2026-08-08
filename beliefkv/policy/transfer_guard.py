from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Mapping

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandKind,
    ControlCommand,
    PageHandle,
    TransferBlocker,
    TransferBlockerCode,
)


_GUARDED_KINDS = {
    CommandKind.OFFLOAD_CONTEXT,
    CommandKind.SHADOW_CONTEXT,
    CommandKind.PREFETCH_CONTEXT,
    CommandKind.DROP_TERMINAL_PRIVATE,
}


@dataclass(frozen=True)
class TransferAttemptKey:
    context_id: str
    context_epoch: int
    command_kind: CommandKind
    bundle_id: str
    closure_fingerprint: str


@dataclass(frozen=True)
class TransferGuardEvent:
    kind: str
    ts_ms: float
    fields: Mapping[str, object]


@dataclass(frozen=True)
class _AttemptSnapshot:
    key: TransferAttemptKey
    target_bytes: int
    device_available_bytes: int
    host_available_bytes: int


@dataclass
class BlockedTransferAttempt:
    key: TransferAttemptKey
    blocker_codes: tuple[TransferBlockerCode, ...]
    required_bytes: int
    failed_device_available_bytes: int
    failed_host_available_bytes: int
    failed_ts_ms: float
    last_command_id: str
    identical_failure_count: int = 1
    next_retry_ts_ms: float | None = None
    suppressed_count: int = 0
    suppression_reported: bool = False
    capacity_snapshot_pending: bool = False


class TransferAttemptGuard:
    """Suppress retries until the physical condition that rejected them changes.

    The guard is deliberately independent of candidate ranking. It remembers a
    failed physical snapshot and prevents scheduler ticks from turning one
    rejection into a command storm. Known blockers are event/state gated;
    only an unclassified backend failure uses bounded time backoff.
    """

    def __init__(
        self,
        graph: RuntimeCausalContextGraph,
        page_index: PageOwnershipIndex,
        *,
        enabled: bool = True,
        max_same_snapshot_attempts: int = 1,
        unknown_base_ms: float = 10.0,
        unknown_max_ms: float = 1000.0,
        unknown_circuit_breaker_failures: int = 8,
    ) -> None:
        if max_same_snapshot_attempts <= 0:
            raise ValueError("max_same_snapshot_attempts must be positive")
        if unknown_base_ms <= 0 or unknown_max_ms < unknown_base_ms:
            raise ValueError("invalid unknown-backend retry interval")
        if unknown_circuit_breaker_failures <= 0:
            raise ValueError("unknown circuit-breaker threshold must be positive")
        self.graph = graph
        self.page_index = page_index
        self.enabled = enabled
        self.max_same_snapshot_attempts = max_same_snapshot_attempts
        self.unknown_base_ms = unknown_base_ms
        self.unknown_max_ms = unknown_max_ms
        self.unknown_circuit_breaker_failures = unknown_circuit_breaker_failures
        self.device_available_bytes = 0
        self.host_available_bytes = 0
        self._inflight: dict[str, _AttemptSnapshot] = {}
        self._blocked: dict[TransferAttemptKey, BlockedTransferAttempt] = {}
        self._unknown_failure_counts: dict[TransferAttemptKey, int] = {}
        self._generations: dict[TransferAttemptKey, int] = {}
        self._events: list[TransferGuardEvent] = []
        self._blocked_count = 0
        self._suppressed_count = 0
        self._released_count = 0

    def update_resources(
        self,
        *,
        device_available_bytes: int,
        host_available_bytes: int,
        now_ms: float,
    ) -> None:
        if device_available_bytes < 0 or host_available_bytes < 0:
            raise ValueError("transfer resource availability must be non-negative")
        self.device_available_bytes = device_available_bytes
        self.host_available_bytes = host_available_bytes
        # A backend can partially complete a bundle before reporting a capacity
        # failure. Anchor retries to the first authoritative allocator snapshot
        # after that ACK, not to the larger free-space value at submission time.
        for blocked in self._blocked.values():
            if not blocked.capacity_snapshot_pending:
                continue
            blocked.failed_device_available_bytes = device_available_bytes
            blocked.failed_host_available_bytes = host_available_bytes
            blocked.capacity_snapshot_pending = False
            self._emit("transfer_retry_capacity_anchored", now_ms, blocked)

    def is_eligible(
        self,
        context_id: str,
        context_epoch: int,
        command_kind: CommandKind,
        *,
        now_ms: float,
    ) -> bool:
        if not self.enabled or command_kind not in _GUARDED_KINDS:
            return True
        key = self.key_for(context_id, context_epoch, command_kind)
        return self._key_is_eligible(key, now_ms=now_ms)

    def _key_is_eligible(
        self, key: TransferAttemptKey, *, now_ms: float
    ) -> bool:
        self._release_changed_snapshots(key, now_ms)
        blocked = self._blocked.get(key)
        if blocked is None:
            return True
        if blocked.identical_failure_count < self.max_same_snapshot_attempts:
            return True
        if self._resource_predicates_satisfied(blocked, now_ms):
            self._release(blocked, now_ms, "resource_predicate_satisfied")
            return True
        blocked.suppressed_count += 1
        self._suppressed_count += 1
        if not blocked.suppression_reported:
            blocked.suppression_reported = True
            self._emit(
                "transfer_retry_suppressed",
                now_ms,
                blocked,
                suppressed_count=blocked.suppressed_count,
            )
        return False

    def command_is_eligible(self, command: ControlCommand, *, now_ms: float) -> bool:
        if command.context_id is None or command.context_epoch is None:
            return True
        if not self.enabled or command.kind not in _GUARDED_KINDS:
            return True
        return self._key_is_eligible(
            self._key_for_command(command), now_ms=now_ms
        )

    def suppression_blockers(
        self, command: ControlCommand
    ) -> tuple[TransferBlocker, ...]:
        """Describe the physical failure suppressing an accepted command."""

        if (
            command.context_id is None
            or command.context_epoch is None
            or not self.enabled
            or command.kind not in _GUARDED_KINDS
        ):
            return ()
        blocked = self._blocked.get(self._key_for_command(command))
        if blocked is None:
            return ()
        detail = f"retry suppressed after {blocked.last_command_id}"
        return tuple(
            TransferBlocker(
                code=code,
                required_bytes=blocked.required_bytes,
                detail=detail,
            )
            for code in blocked.blocker_codes
        )

    def begin_attempt(self, command: ControlCommand, *, now_ms: float) -> str:
        if command.command_id in self._inflight:
            raise ValueError(f"duplicate transfer attempt: {command.command_id}")
        if (
            not self.enabled
            or command.kind not in _GUARDED_KINDS
            or command.context_id is None
            or command.context_epoch is None
        ):
            return ""
        key = self._key_for_command(command)
        self._inflight[command.command_id] = _AttemptSnapshot(
            key=key,
            target_bytes=command.target_bytes,
            device_available_bytes=self.device_available_bytes,
            host_available_bytes=self.host_available_bytes,
        )
        return key.closure_fingerprint

    def record_failure(
        self,
        command_id: str,
        *,
        blockers: tuple[TransferBlocker, ...],
        required_bytes: int,
        now_ms: float,
    ) -> None:
        attempt = self._inflight.pop(command_id, None)
        if not self.enabled or attempt is None:
            return
        normalized = tuple(
            sorted(
                {item.code for item in blockers}
                or {TransferBlockerCode.UNKNOWN_BACKEND},
                key=lambda item: item.value,
            )
        )
        required = max(
            required_bytes if required_bytes > 0 else attempt.target_bytes,
            *(item.required_bytes for item in blockers),
        )
        previous = self._blocked.get(attempt.key)
        if TransferBlockerCode.UNKNOWN_BACKEND in normalized:
            failure_count = self._unknown_failure_counts.get(attempt.key, 0) + 1
            self._unknown_failure_counts[attempt.key] = failure_count
        else:
            failure_count = (
                previous.identical_failure_count + 1 if previous is not None else 1
            )
        next_retry_ts_ms = None
        if (
            TransferBlockerCode.UNKNOWN_BACKEND in normalized
            and failure_count < self.unknown_circuit_breaker_failures
        ):
            exponent = max(0, failure_count - 1)
            delay_ms = min(self.unknown_max_ms, self.unknown_base_ms * (2**exponent))
            next_retry_ts_ms = now_ms + delay_ms
        blocked = BlockedTransferAttempt(
            key=attempt.key,
            blocker_codes=normalized,
            required_bytes=required,
            failed_device_available_bytes=attempt.device_available_bytes,
            failed_host_available_bytes=attempt.host_available_bytes,
            failed_ts_ms=now_ms,
            last_command_id=command_id,
            identical_failure_count=failure_count,
            next_retry_ts_ms=next_retry_ts_ms,
            suppressed_count=previous.suppressed_count if previous is not None else 0,
            suppression_reported=(
                previous.suppression_reported if previous is not None else False
            ),
            capacity_snapshot_pending=any(
                code
                in {
                    TransferBlockerCode.DEVICE_CAPACITY,
                    TransferBlockerCode.HOST_CAPACITY,
                }
                for code in normalized
            ),
        )
        self._blocked[attempt.key] = blocked
        self._generations[attempt.key] = self._generations.get(attempt.key, 0) + 1
        self._blocked_count += 1
        self._emit("transfer_attempt_blocked", now_ms, blocked)

    def record_success(self, command_id: str, *, now_ms: float) -> None:
        attempt = self._inflight.pop(command_id, None)
        if attempt is None:
            return
        for key, blocked in tuple(self._blocked.items()):
            if self._same_identity(key, attempt.key):
                self._release(blocked, now_ms, "transfer_succeeded")
        for key in tuple(self._unknown_failure_counts):
            if self._same_identity(key, attempt.key):
                self._unknown_failure_counts.pop(key, None)

    def cancel_attempt(self, command_id: str) -> None:
        self._inflight.pop(command_id, None)

    def invalidate_context(
        self, context_id: str, *, now_ms: float, keep_epoch: int | None = None
    ) -> None:
        for command_id, attempt in tuple(self._inflight.items()):
            if attempt.key.context_id == context_id and (
                keep_epoch is None or attempt.key.context_epoch != keep_epoch
            ):
                self._inflight.pop(command_id, None)
        for key, blocked in tuple(self._blocked.items()):
            if key.context_id == context_id and (
                keep_epoch is None or key.context_epoch != keep_epoch
            ):
                self._release(blocked, now_ms, "context_epoch_invalidated")
        for key in tuple(self._unknown_failure_counts):
            if key.context_id == context_id and (
                keep_epoch is None or key.context_epoch != keep_epoch
            ):
                self._unknown_failure_counts.pop(key, None)

    def reset(self, *, now_ms: float) -> None:
        self._inflight.clear()
        self._unknown_failure_counts.clear()
        for blocked in tuple(self._blocked.values()):
            self._release(blocked, now_ms, "cache_reset")

    def key_for(
        self, context_id: str, context_epoch: int, command_kind: CommandKind
    ) -> TransferAttemptKey:
        return TransferAttemptKey(
            context_id=context_id,
            context_epoch=context_epoch,
            command_kind=command_kind,
            bundle_id="",
            closure_fingerprint=self._closure_fingerprint(
                context_id, context_epoch, command_kind
            ),
        )

    def _key_for_command(self, command: ControlCommand) -> TransferAttemptKey:
        if command.context_id is None or command.context_epoch is None:
            raise ValueError("a guarded transfer command requires context identity")
        bundle = command.physical_bundle
        if command.target_handles:
            return TransferAttemptKey(
                context_id=command.context_id,
                context_epoch=command.context_epoch,
                command_kind=command.kind,
                bundle_id="terminal-private",
                closure_fingerprint=self._target_handles_fingerprint(
                    command.target_handles
                ),
            )
        if bundle is None:
            return self.key_for(
                command.context_id, command.context_epoch, command.kind
            )
        return TransferAttemptKey(
            context_id=command.context_id,
            context_epoch=command.context_epoch,
            command_kind=command.kind,
            bundle_id=bundle.bundle_id,
            closure_fingerprint=bundle.generation_fingerprint,
        )

    def generation_for(self, command: ControlCommand) -> int:
        """Return the event generation for this exact physical attempt."""

        if (
            command.context_id is None
            or command.context_epoch is None
            or command.kind not in _GUARDED_KINDS
        ):
            return 0
        return self._generations.get(self._key_for_command(command), 0)

    def _target_handles_fingerprint(
        self, handles: tuple[PageHandle, ...]
    ) -> str:
        state: list[tuple[object, ...]] = []
        for handle in sorted(handles):
            page = self.page_index.pages.get(handle)
            if page is None:
                state.append((handle.page_id, handle.allocation_generation, "missing"))
                continue
            state.append(
                (
                    handle.page_id,
                    handle.allocation_generation,
                    page.residency.value,
                    tuple(sorted(page.owner_contexts.items())),
                    page.engine_lock_ref,
                    page.active_reader_count,
                    page.transfer_direction.value
                    if page.transfer_direction is not None
                    else None,
                    tuple(sorted(page.children)),
                )
            )
        return blake2b(repr(tuple(state)).encode("utf-8"), digest_size=16).hexdigest()

    def drain_events(self) -> tuple[TransferGuardEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def summary(self) -> dict[str, int]:
        return {
            "blocked_attempt_count": self._blocked_count,
            "suppressed_retry_count": self._suppressed_count,
            "released_attempt_count": self._released_count,
            "active_blocked_attempt_count": len(self._blocked),
            "inflight_attempt_count": len(self._inflight),
            "unknown_circuit_open_count": sum(
                count >= self.unknown_circuit_breaker_failures
                for count in self._unknown_failure_counts.values()
            ),
        }

    def _closure_fingerprint(
        self, context_id: str, context_epoch: int, command_kind: CommandKind
    ) -> str:
        context_pages = {
            page.handle for page in self.page_index.context_pages(context_id)
        }
        relevant = set(context_pages)
        if command_kind == CommandKind.PREFETCH_CONTEXT:
            stack = list(context_pages)
            while stack:
                page = self.page_index.pages.get(stack.pop())
                if page is None or page.parent is None or page.parent in relevant:
                    continue
                relevant.add(page.parent)
                stack.append(page.parent)
        elif command_kind in {
            CommandKind.OFFLOAD_CONTEXT,
            CommandKind.SHADOW_CONTEXT,
        }:
            stack = list(context_pages)
            while stack:
                page = self.page_index.pages.get(stack.pop())
                if page is None:
                    continue
                for child in page.children:
                    if child not in relevant:
                        relevant.add(child)
                        stack.append(child)

        page_state: list[tuple[object, ...]] = []
        owner_contexts: set[str] = {context_id}
        for handle in sorted(relevant):
            page = self.page_index.pages.get(handle)
            if page is None:
                page_state.append(
                    (handle.page_id, handle.allocation_generation, "missing")
                )
                continue
            owner_contexts.update(page.owner_contexts)
            page_state.append(
                (
                    handle.page_id,
                    handle.allocation_generation,
                    page.size_bytes,
                    page.residency.value,
                    self._handle_tuple(page.parent),
                    tuple(self._handle_tuple(child) for child in sorted(page.children)),
                    tuple(sorted(page.owner_contexts.items())),
                    page.engine_lock_ref,
                    page.active_reader_count,
                    tuple(sorted(page.semantic_pin_contexts)),
                    page.sealed,
                    page.transfer_direction.value
                    if page.transfer_direction is not None
                    else None,
                    handle in context_pages,
                )
            )

        causal_state: list[tuple[object, ...]] = []
        for owner_context_id in sorted(owner_contexts):
            context = self.graph.contexts.get(owner_context_id)
            if context is None:
                causal_state.append((owner_context_id, "missing"))
                continue
            invocations = tuple(
                sorted(
                    (
                        invocation_id,
                        self.graph.invocations[invocation_id].state.value,
                    )
                    for invocation_id in context.invocation_ids
                    if invocation_id in self.graph.invocations
                )
            )
            causal_state.append((owner_context_id, context.epoch, invocations))

        payload = repr(
            (
                context_id,
                context_epoch,
                command_kind.value,
                tuple(page_state),
                tuple(causal_state),
            )
        ).encode("utf-8")
        return blake2b(payload, digest_size=16).hexdigest()

    def _release_changed_snapshots(
        self, current: TransferAttemptKey, now_ms: float
    ) -> None:
        for key, blocked in tuple(self._blocked.items()):
            if self._same_identity(key, current) and key != current:
                remaining = tuple(
                    code
                    for code in blocked.blocker_codes
                    if code
                    in {
                        TransferBlockerCode.DEVICE_CAPACITY,
                        TransferBlockerCode.HOST_CAPACITY,
                    }
                    and not self._capacity_predicate_satisfied(code, blocked)
                )
                if remaining:
                    self._blocked.pop(key, None)
                    previous_fingerprint = key.closure_fingerprint
                    blocked.key = current
                    blocked.blocker_codes = remaining
                    blocked.suppression_reported = False
                    self._blocked[current] = blocked
                    self._emit(
                        "transfer_retry_rekeyed",
                        now_ms,
                        blocked,
                        previous_closure_fingerprint=previous_fingerprint,
                    )
                else:
                    self._release(
                        blocked,
                        now_ms,
                        "physical_fingerprint_changed",
                    )
                self._unknown_failure_counts.pop(key, None)
        for key in tuple(self._unknown_failure_counts):
            if self._same_identity(key, current) and key != current:
                self._unknown_failure_counts.pop(key, None)

    def _resource_predicates_satisfied(
        self, blocked: BlockedTransferAttempt, now_ms: float
    ) -> bool:
        predicates: list[bool] = []
        for code in blocked.blocker_codes:
            if code in {
                TransferBlockerCode.DEVICE_CAPACITY,
                TransferBlockerCode.HOST_CAPACITY,
            }:
                predicates.append(
                    self._capacity_predicate_satisfied(code, blocked)
                )
            elif code == TransferBlockerCode.UNKNOWN_BACKEND:
                predicates.append(
                    blocked.next_retry_ts_ms is not None
                    and now_ms >= blocked.next_retry_ts_ms
                )
            else:
                # Closure, lock, loading, generation and engine predicates are
                # encoded in the closure fingerprint. An unchanged key means
                # the corresponding state-changing event has not occurred.
                predicates.append(False)
        return bool(predicates) and all(predicates)

    def _capacity_predicate_satisfied(
        self,
        code: TransferBlockerCode,
        blocked: BlockedTransferAttempt,
    ) -> bool:
        if blocked.capacity_snapshot_pending:
            return False
        if code == TransferBlockerCode.DEVICE_CAPACITY:
            return (
                self.device_available_bytes
                > blocked.failed_device_available_bytes
                and self.device_available_bytes >= blocked.required_bytes
            )
        if code == TransferBlockerCode.HOST_CAPACITY:
            return (
                self.host_available_bytes > blocked.failed_host_available_bytes
                and self.host_available_bytes >= blocked.required_bytes
            )
        raise ValueError(f"not a capacity blocker: {code.value}")

    def _release(
        self, blocked: BlockedTransferAttempt, now_ms: float, reason: str
    ) -> None:
        if self._blocked.pop(blocked.key, None) is None:
            return
        self._generations[blocked.key] = self._generations.get(blocked.key, 0) + 1
        self._released_count += 1
        self._emit(
            "transfer_retry_released",
            now_ms,
            blocked,
            release_reason=reason,
            suppressed_count=blocked.suppressed_count,
        )

    def _emit(
        self,
        kind: str,
        ts_ms: float,
        blocked: BlockedTransferAttempt,
        **extra: object,
    ) -> None:
        self._events.append(
            TransferGuardEvent(
                kind=kind,
                ts_ms=ts_ms,
                fields={
                    "context_id": blocked.key.context_id,
                    "context_epoch": blocked.key.context_epoch,
                    "command_kind": blocked.key.command_kind.value,
                    "bundle_id": blocked.key.bundle_id,
                    "closure_fingerprint": blocked.key.closure_fingerprint,
                    "blocker_codes": [
                        item.value for item in blocked.blocker_codes
                    ],
                    "required_bytes": blocked.required_bytes,
                    "failed_device_available_bytes": (
                        blocked.failed_device_available_bytes
                    ),
                    "failed_host_available_bytes": blocked.failed_host_available_bytes,
                    "failed_ts_ms": blocked.failed_ts_ms,
                    "next_retry_ts_ms": blocked.next_retry_ts_ms,
                    "identical_failure_count": blocked.identical_failure_count,
                    "last_command_id": blocked.last_command_id,
                    "circuit_open": (
                        TransferBlockerCode.UNKNOWN_BACKEND
                        in blocked.blocker_codes
                        and blocked.next_retry_ts_ms is None
                    ),
                    "capacity_snapshot_pending": (
                        blocked.capacity_snapshot_pending
                    ),
                    **extra,
                },
            )
        )

    @staticmethod
    def _same_identity(
        left: TransferAttemptKey, right: TransferAttemptKey
    ) -> bool:
        return (
            left.context_id == right.context_id
            and left.context_epoch == right.context_epoch
            and left.command_kind == right.command_kind
            and left.bundle_id == right.bundle_id
        )

    @staticmethod
    def _handle_tuple(handle: PageHandle | None) -> tuple[int, int] | None:
        if handle is None:
            return None
        return (handle.page_id, handle.allocation_generation)
