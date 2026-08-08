from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from beliefkv.runtime.protocol import CommandKind


class RestoreObligationState(str, Enum):
    """Lifecycle of one request whose GPU KV must be restored."""

    RETRACTION_PREPARED = "retraction_prepared"
    D2H_INFLIGHT = "d2h_inflight"
    PARKED_WAIT = "parked_wait"
    EVICT_FOR_RESTORE = "evict_for_restore"
    H2D_INFLIGHT = "h2d_inflight"
    RESTORE_ACKED = "restore_acked"
    TICKET_READY = "ticket_ready"
    SATISFIED = "satisfied"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            RestoreObligationState.SATISFIED,
            RestoreObligationState.CANCELLED,
            RestoreObligationState.FAILED,
        }


class RestoreLeaseState(str, Enum):
    """Allocator-backed capacity held until restored KV resumes service."""

    GRANTED = "granted"
    H2D_INFLIGHT = "h2d_inflight"
    RESTORED_RESERVED = "restored_reserved"
    ADMISSION_COMMITTING = "admission_committing"
    ADMITTED = "admitted"
    RELEASED = "released"
    ROLLED_BACK = "rolled_back"

    @property
    def terminal(self) -> bool:
        return self in {
            RestoreLeaseState.RELEASED,
            RestoreLeaseState.ROLLED_BACK,
        }


class RestoreObligationCause(str, Enum):
    """Observed event that created a durable restore debt."""

    RUNNING_RETRACTION = "running_retraction"
    ORDINARY_WAITING_PREFIX = "ordinary_waiting_prefix"


class NativeQueueLocation(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    CHUNKED = "chunked"
    NONE = "none"


class SafePointPhysicalPhase(str, Enum):
    """Ownership snapshot lifecycle inside one scheduler safe point."""

    IDLE = "idle"
    APPLY_EVENTS = "apply_events"
    CAPTURE_AND_PLAN = "capture_and_plan"
    TRANSACTIONAL_COMMIT = "transactional_commit"


@dataclass(frozen=True)
class NativeRequestPhysicalSnapshot:
    """Orthogonal safe-point view reconstructed from native SGLang state."""

    request_id: str
    context_id: str
    queue_location: NativeQueueLocation
    req_pool_slot: int | None
    radix_lock_owned: bool
    native_load_operation_id: str | None
    explicit_transfer_ids: tuple[str, ...]
    request_generation: int
    terminal: bool

    def __post_init__(self) -> None:
        if not self.request_id or not self.context_id:
            raise ValueError("native physical snapshot identity is required")
        if self.req_pool_slot is not None and self.req_pool_slot < 0:
            raise ValueError("request pool slot must be non-negative")
        if self.request_generation < 0:
            raise ValueError("request generation must be non-negative")
        object.__setattr__(
            self,
            "explicit_transfer_ids",
            tuple(sorted(set(self.explicit_transfer_ids))),
        )

    @property
    def engine_owned(self) -> bool:
        return self.req_pool_slot is not None or self.radix_lock_owned


@dataclass(frozen=True)
class SafePointSnapshotBuildTiming:
    """Segmented CPU cost for one lazy native ownership snapshot build."""

    total_ms: float
    queue_collection_ms: float
    metadata_indexing_ms: float
    radix_ownership_lookup_ms: float
    operation_indexing_ms: float
    sorting_allocation_ms: float
    queue_record_count: int = 0
    metadata_record_count: int = 0
    matched_record_count: int = 0
    gc_collections: int = 0
    cold_build: bool = False

    def __post_init__(self) -> None:
        values = (
            self.total_ms,
            self.queue_collection_ms,
            self.metadata_indexing_ms,
            self.radix_ownership_lookup_ms,
            self.operation_indexing_ms,
            self.sorting_allocation_ms,
        )
        if any(value < 0 for value in values):
            raise ValueError("snapshot timing values must be non-negative")
        if min(
            self.queue_record_count,
            self.metadata_record_count,
            self.matched_record_count,
            self.gc_collections,
        ) < 0:
            raise ValueError("snapshot timing counts must be non-negative")


@dataclass(frozen=True)
class SafePointPhysicalSnapshot:
    """Immutable, lazily captured native ownership view for one epoch."""

    epoch: int
    records: tuple[NativeRequestPhysicalSnapshot, ...]
    by_request: Mapping[str, NativeRequestPhysicalSnapshot]
    by_context: Mapping[str, tuple[NativeRequestPhysicalSnapshot, ...]]
    explicit_transfers_by_context: Mapping[str, tuple[str, ...]]
    queue_record_count: int
    metadata_record_count: int
    timing: SafePointSnapshotBuildTiming

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("safe-point physical epoch must be non-negative")
        if self.queue_record_count < 0 or self.metadata_record_count < 0:
            raise ValueError("snapshot record counts must be non-negative")
        if len(self.by_request) != len(self.records):
            raise ValueError("native request IDs must be unique in one snapshot")
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(
            self, "by_request", MappingProxyType(dict(self.by_request))
        )
        object.__setattr__(
            self,
            "by_context",
            MappingProxyType(
                {
                    context_id: tuple(records)
                    for context_id, records in self.by_context.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "explicit_transfers_by_context",
            MappingProxyType(
                {
                    context_id: tuple(sorted(set(command_ids)))
                    for context_id, command_ids in (
                        self.explicit_transfers_by_context.items()
                    )
                }
            ),
        )

    def for_context(
        self, context_id: str
    ) -> tuple[NativeRequestPhysicalSnapshot, ...]:
        return self.by_context.get(context_id, ())

    def context_readset(self, context_id: str) -> tuple[object, ...]:
        """Return the native ownership dimensions a commit must revalidate."""

        return (
            self.for_context(context_id),
            self.explicit_transfers_by_context.get(context_id, ()),
        )


@dataclass(frozen=True)
class ExternalProgressToken:
    """Relevant external state that can make a blocked restore retryable."""

    engine_owner_epoch: tuple[object, ...]
    closure_fingerprint: str
    effective_capacity_threshold_epoch: tuple[object, ...]
    command_ownership_epoch: tuple[object, ...]
    guard_generation: int
    native_load_generation: tuple[object, ...]

    def __post_init__(self) -> None:
        if self.guard_generation < 0:
            raise ValueError("guard generation must be non-negative")


class RestoreTransactionStage(str, Enum):
    WAIT_FEASIBILITY = "wait_feasibility"
    WAIT_FUNDING = "wait_funding"
    WAIT_EVENT = "wait_event"
    PREPARED = "prepared"
    H2D_QUEUED = "h2d_queued"
    H2D_ADOPTED = "h2d_adopted"
    RESTORED_RESERVED = "restored_reserved"
    ADMISSION_COMMITTING = "admission_committing"
    ADMITTED = "admitted"
    SERVICE_GRACE = "service_grace"
    SATISFIED = "satisfied"
    CANCELLED = "cancelled"
    FAILED_UNRECOVERABLE = "failed_unrecoverable"

    @property
    def terminal(self) -> bool:
        return self in {
            RestoreTransactionStage.SATISFIED,
            RestoreTransactionStage.CANCELLED,
            RestoreTransactionStage.FAILED_UNRECOVERABLE,
        }


class RestoreAuthorityMode(str, Enum):
    NORMAL_JOINT = "normal_joint"
    RESTORE_DRAIN_REQUESTED = "restore_drain_requested"
    RESTORE_DRAIN_ACTIVE = "restore_drain_active"


@dataclass(frozen=True)
class RestoreFeasibilityCertificate:
    certificate_generation: int
    context_epoch: int
    attempt_key: tuple[object, ...]
    required_bytes: int
    closure_fingerprint: str

    def __post_init__(self) -> None:
        if min(
            self.certificate_generation,
            self.context_epoch,
            self.required_bytes,
        ) < 0:
            raise ValueError("restore certificate values must be non-negative")


@dataclass
class RestorePhysicalOperation:
    stage: str
    attempt_key: tuple[object, ...]
    certificate_generation: int
    canonical_command_id: str
    adopted: bool
    terminal_status: str | None = None


@dataclass
class RestoreTransaction:
    """Aggregate root for one durable restore debt."""

    transaction_id: str
    obligation: "RestoreObligation"
    stage: RestoreTransactionStage = RestoreTransactionStage.WAIT_FEASIBILITY
    feasibility_certificate: RestoreFeasibilityCertificate | None = None
    capacity_reservation_id: str | None = None
    prefix_pin_token: str | None = None
    physical_operations: list[RestorePhysicalOperation] = field(default_factory=list)
    admission_state: str = "waiting"
    service_grace: RestoreServiceGrace | None = None
    wait_condition: tuple[str, ...] = ()
    external_progress_token: ExternalProgressToken | None = None
    failure_evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise ValueError("restore transaction id must be non-empty")

    def can_submit(
        self,
        *,
        stage: str,
        attempt_key: tuple[object, ...],
        certificate_generation: int,
    ) -> bool:
        return not any(
            item.stage == stage
            and item.attempt_key == attempt_key
            and item.certificate_generation == certificate_generation
            and item.terminal_status is None
            for item in self.physical_operations
        )

    def add_operation(self, operation: RestorePhysicalOperation) -> None:
        if not self.can_submit(
            stage=operation.stage,
            attempt_key=operation.attempt_key,
            certificate_generation=operation.certificate_generation,
        ):
            raise ValueError("duplicate canonical restore operation")
        self.physical_operations.append(operation)


@dataclass
class RestoreServiceGrace:
    """Minimum useful decode service owed after a restore completes."""

    request_id: str
    obligation_id: str
    granted_ts_ms: float
    required_decode_tokens: int
    served_decode_tokens: int = 0
    last_observed_output_tokens: int | None = None
    completed_ts_ms: float | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.obligation_id:
            raise ValueError("restore service grace identity must be non-empty")
        if self.granted_ts_ms < 0:
            raise ValueError("restore service grace time must be non-negative")
        if self.required_decode_tokens <= 0:
            raise ValueError("restore service grace quantum must be positive")
        if (
            self.last_observed_output_tokens is not None
            and self.last_observed_output_tokens < 0
        ):
            raise ValueError("observed output tokens must be non-negative")

    @property
    def active(self) -> bool:
        return self.completed_ts_ms is None

    @property
    def remaining_decode_tokens(self) -> int:
        return max(0, self.required_decode_tokens - self.served_decode_tokens)

    def observe_decode(self, tokens: int, *, now_ms: float) -> bool:
        """Account completed decode service and report a new grace release."""

        if not self.active or tokens <= 0:
            return False
        self.served_decode_tokens += int(tokens)
        if self.served_decode_tokens < self.required_decode_tokens:
            return False
        self.completed_ts_ms = now_ms
        self.terminal_reason = "service_quantum_satisfied"
        return True

    def cancel(self, *, now_ms: float, reason: str) -> None:
        if not self.active:
            return
        self.completed_ts_ms = now_ms
        self.terminal_reason = reason


@dataclass
class RestoreObligation:
    obligation_id: str
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    source_retraction_transaction_id: str
    source_joint_plan_id: str
    created_ts_ms: float
    path_extent_ids: tuple[str, ...]
    cause: RestoreObligationCause = RestoreObligationCause.RUNNING_RETRACTION
    state: RestoreObligationState = RestoreObligationState.RETRACTION_PREPARED
    required_extent_ids: tuple[str, ...] = ()
    source_transaction_terminal: bool = False
    requeued: bool = False
    pending_command_id: str | None = None
    pending_command_kind: CommandKind | None = None
    command_ids: list[str] = field(default_factory=list)
    restore_bytes: int = 0
    required_admission_bytes: int = 0
    funding_reclaim_bytes: int = 0
    funding_reserved_tokens: int = 0
    funding_reserved_bytes: int = 0
    restored_bytes: int = 0
    last_progress_ts_ms: float | None = None
    last_attempt_stamp: tuple[object, ...] | None = None
    last_external_progress_token: ExternalProgressToken | None = None
    wake_conditions: tuple[str, ...] = ()
    blocker_codes: tuple[str, ...] = ()
    blocker_fingerprint: str | None = None
    retry_count: int = 0
    bypass_count: int = 0
    liveness_escalated: bool = False
    native_admission_fallback: bool = False
    first_service_ts_ms: float | None = None
    terminal_ts_ms: float | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        self.cause = RestoreObligationCause(self.cause)
        for value, name in (
            (self.obligation_id, "obligation_id"),
            (self.request_id, "request_id"),
            (self.workflow_id, "workflow_id"),
            (self.invocation_id, "invocation_id"),
            (self.context_id, "context_id"),
            (
                self.source_retraction_transaction_id,
                "source_retraction_transaction_id",
            ),
            (self.source_joint_plan_id, "source_joint_plan_id"),
        ):
            if not value:
                raise ValueError(f"restore obligation {name} must be non-empty")
        if self.context_epoch < 0 or self.created_ts_ms < 0:
            raise ValueError("restore obligation epoch/time must be non-negative")
        self.path_extent_ids = self._extent_ids(self.path_extent_ids)
        self.required_extent_ids = self._extent_ids(self.required_extent_ids)
        if self.last_progress_ts_ms is None:
            self.last_progress_ts_ms = self.created_ts_ms

    @staticmethod
    def _extent_ids(values: tuple[str, ...]) -> tuple[str, ...]:
        result = tuple(sorted(set(values)))
        if any(not item for item in result):
            raise ValueError("restore extent IDs must be non-empty")
        return result

    def set_required_extents(
        self,
        extent_ids: tuple[str, ...],
        *,
        restore_bytes: int,
        now_ms: float,
    ) -> None:
        if self.state.terminal:
            return
        required = self._extent_ids(extent_ids)
        restore_bytes = max(0, int(restore_bytes))
        if (
            required != self.required_extent_ids
            or restore_bytes != self.restore_bytes
        ):
            self.required_extent_ids = required
            self.restore_bytes = restore_bytes
            self.last_progress_ts_ms = now_ms

    def start_command(
        self,
        command_id: str,
        command_kind: CommandKind,
        *,
        now_ms: float,
        attempt_stamp: tuple[object, ...],
    ) -> None:
        if self.state.terminal:
            raise ValueError("terminal restore obligation cannot start a command")
        if self.pending_command_id is not None:
            raise ValueError("restore obligation already has an in-flight command")
        self.pending_command_id = command_id
        self.pending_command_kind = command_kind
        self.command_ids.append(command_id)
        self.last_attempt_stamp = attempt_stamp
        self.blocker_codes = ()
        self.blocker_fingerprint = None
        self.last_progress_ts_ms = now_ms
        self.state = (
            RestoreObligationState.H2D_INFLIGHT
            if command_kind == CommandKind.PREFETCH_CONTEXT
            else RestoreObligationState.EVICT_FOR_RESTORE
        )

    def clear_command(self) -> CommandKind | None:
        kind = self.pending_command_kind
        self.pending_command_id = None
        self.pending_command_kind = None
        return kind

    def block(
        self,
        *,
        blocker_codes: tuple[str, ...],
        blocker_fingerprint: str,
        attempt_stamp: tuple[object, ...] | None = None,
        external_progress_token: ExternalProgressToken | None = None,
        wake_conditions: tuple[str, ...] = (),
        now_ms: float,
    ) -> None:
        if self.state.terminal:
            return
        self.clear_command()
        self.state = RestoreObligationState.PARKED_WAIT
        self.blocker_codes = tuple(sorted(set(blocker_codes)))
        self.blocker_fingerprint = blocker_fingerprint
        self.last_attempt_stamp = attempt_stamp
        self.last_external_progress_token = external_progress_token
        self.wake_conditions = tuple(sorted(set(wake_conditions)))
        self.retry_count += 1
        self.last_progress_ts_ms = now_ms

    def external_progressed(self, token: ExternalProgressToken) -> bool:
        return (
            self.last_external_progress_token is None
            or self.last_external_progress_token != token
        )

    def mark_ticket_ready(self, *, now_ms: float) -> None:
        if self.state.terminal:
            return
        if self.state == RestoreObligationState.TICKET_READY:
            return
        self.clear_command()
        self.required_extent_ids = ()
        self.restore_bytes = 0
        self.blocker_codes = ()
        self.blocker_fingerprint = None
        self.last_external_progress_token = None
        self.wake_conditions = ()
        self.state = RestoreObligationState.TICKET_READY
        self.last_progress_ts_ms = now_ms

    def use_native_admission_fallback(self, *, now_ms: float) -> None:
        """Let SGLang load or recompute the prefix under this debt's lease."""

        if self.state.terminal:
            return
        self.native_admission_fallback = True
        self.mark_ticket_ready(now_ms=now_ms)

    def finish(
        self,
        state: RestoreObligationState,
        *,
        now_ms: float,
        reason: str,
    ) -> None:
        if state not in {
            RestoreObligationState.SATISFIED,
            RestoreObligationState.CANCELLED,
            RestoreObligationState.FAILED,
        }:
            raise ValueError("restore obligation finish requires a terminal state")
        self.clear_command()
        self.state = state
        self.terminal_ts_ms = now_ms
        self.terminal_reason = reason
        self.last_progress_ts_ms = now_ms
        if state == RestoreObligationState.SATISFIED:
            self.first_service_ts_ms = now_ms


@dataclass
class RestoreLease:
    """One admission-capacity lease backed by real allocator tokens."""

    lease_id: str
    obligation_id: str
    request_id: str
    workflow_id: str
    context_id: str
    granted_ts_ms: float
    reserved_tokens: int
    reserved_bytes: int
    h2d_bytes: int
    state: RestoreLeaseState = RestoreLeaseState.GRANTED
    h2d_command_id: str | None = None
    pin_active: bool = False
    admission_attempts: int = 0
    terminal_ts_ms: float | None = None
    terminal_reason: str | None = None

    @property
    def capacity_held(self) -> bool:
        return self.state in {
            RestoreLeaseState.GRANTED,
            RestoreLeaseState.H2D_INFLIGHT,
            RestoreLeaseState.RESTORED_RESERVED,
        }

    def __post_init__(self) -> None:
        for value, name in (
            (self.lease_id, "lease_id"),
            (self.obligation_id, "obligation_id"),
            (self.request_id, "request_id"),
            (self.workflow_id, "workflow_id"),
            (self.context_id, "context_id"),
        ):
            if not value:
                raise ValueError(f"restore lease {name} must be non-empty")
        if self.granted_ts_ms < 0:
            raise ValueError("restore lease grant time must be non-negative")
        if min(self.reserved_tokens, self.reserved_bytes, self.h2d_bytes) < 0:
            raise ValueError("restore lease capacities must be non-negative")
        if self.reserved_tokens == 0 or self.reserved_bytes == 0:
            raise ValueError("restore lease must reserve positive admission capacity")

    def mark_h2d_inflight(self, command_id: str) -> None:
        if self.state.terminal:
            raise ValueError("terminal restore lease cannot start H2D")
        if not command_id:
            raise ValueError("restore lease H2D command id must be non-empty")
        self.h2d_command_id = command_id
        self.state = RestoreLeaseState.H2D_INFLIGHT

    def mark_restored(self) -> None:
        if self.state.terminal:
            return
        self.h2d_command_id = None
        self.state = RestoreLeaseState.RESTORED_RESERVED

    def begin_admission(self) -> None:
        if self.state not in {
            RestoreLeaseState.RESTORED_RESERVED,
            RestoreLeaseState.GRANTED,
        }:
            raise ValueError("restore lease is not ready for admission")
        self.admission_attempts += 1
        self.state = RestoreLeaseState.ADMISSION_COMMITTING

    def admission_rejected(self) -> None:
        if self.state != RestoreLeaseState.ADMISSION_COMMITTING:
            raise ValueError("restore lease has no admission attempt to reject")
        self.state = RestoreLeaseState.RESTORED_RESERVED

    def mark_admitted(self) -> None:
        if self.state != RestoreLeaseState.ADMISSION_COMMITTING:
            raise ValueError("restore lease admission was not started")
        self.state = RestoreLeaseState.ADMITTED

    def finish(
        self,
        state: RestoreLeaseState,
        *,
        now_ms: float,
        reason: str,
    ) -> None:
        if state not in {
            RestoreLeaseState.RELEASED,
            RestoreLeaseState.ROLLED_BACK,
        }:
            raise ValueError("restore lease finish requires a terminal state")
        if now_ms < 0 or not reason:
            raise ValueError("restore lease terminal time/reason is required")
        self.h2d_command_id = None
        self.pin_active = False
        self.state = state
        self.terminal_ts_ms = now_ms
        self.terminal_reason = reason


class RestoreLeaseIndex:
    """Bounded lease table; allocator allocations remain runtime-owned."""

    def __init__(self, *, max_active: int = 1) -> None:
        if max_active <= 0:
            raise ValueError("maximum active restore leases must be positive")
        self.max_active = max_active
        self._by_request: dict[str, RestoreLease] = {}
        self._sequence = 0

    def get(self, request_id: str) -> RestoreLease | None:
        return self._by_request.get(request_id)

    def active(self) -> tuple[RestoreLease, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self._by_request.values()
                    if not item.state.terminal
                ),
                key=lambda item: (item.granted_ts_ms, item.lease_id),
            )
        )

    def all(self) -> tuple[RestoreLease, ...]:
        return tuple(
            sorted(self._by_request.values(), key=lambda item: item.lease_id)
        )

    @property
    def reserved_tokens(self) -> int:
        return sum(
            item.reserved_tokens for item in self.active() if item.capacity_held
        )

    @property
    def reserved_bytes(self) -> int:
        return sum(
            item.reserved_bytes for item in self.active() if item.capacity_held
        )

    def grant(
        self,
        *,
        obligation: RestoreObligation,
        granted_ts_ms: float,
        reserved_tokens: int,
        reserved_bytes: int,
        h2d_bytes: int,
    ) -> RestoreLease:
        current = self.get(obligation.request_id)
        if current is not None and not current.state.terminal:
            raise ValueError(
                f"request already owns a restore lease: {obligation.request_id}"
            )
        if len(self.active()) >= self.max_active:
            raise ValueError("restore lease capacity is exhausted")
        self._sequence += 1
        lease = RestoreLease(
            lease_id=f"restore-lease-{self._sequence}",
            obligation_id=obligation.obligation_id,
            request_id=obligation.request_id,
            workflow_id=obligation.workflow_id,
            context_id=obligation.context_id,
            granted_ts_ms=granted_ts_ms,
            reserved_tokens=reserved_tokens,
            reserved_bytes=reserved_bytes,
            h2d_bytes=h2d_bytes,
        )
        self._by_request[obligation.request_id] = lease
        return lease


class RestoreObligationIndex:
    """Durable request-indexed restore debts that survive JointPlan invalidation."""

    def __init__(self, *, max_active: int) -> None:
        if max_active <= 0:
            raise ValueError("max_active restore obligations must be positive")
        self.max_active = max_active
        self._by_request: dict[str, RestoreObligation] = {}
        self._sequence = 0

    def can_create(self, request_ids: tuple[str, ...]) -> bool:
        unique = set(request_ids)
        if len(unique) != len(request_ids):
            return False
        if any(
            request_id in self._by_request
            and not self._by_request[request_id].state.terminal
            for request_id in unique
        ):
            return False
        return len(self.active()) + len(unique) <= self.max_active

    def create(
        self,
        *,
        request_id: str,
        workflow_id: str,
        invocation_id: str,
        context_id: str,
        context_epoch: int,
        source_retraction_transaction_id: str,
        source_joint_plan_id: str,
        created_ts_ms: float,
        path_extent_ids: tuple[str, ...],
        cause: RestoreObligationCause = (
            RestoreObligationCause.RUNNING_RETRACTION
        ),
    ) -> RestoreObligation:
        if not self.can_create((request_id,)):
            raise ValueError(f"restore obligation capacity conflict: {request_id}")
        self._sequence += 1
        item = RestoreObligation(
            obligation_id=f"restore-{self._sequence}",
            request_id=request_id,
            workflow_id=workflow_id,
            invocation_id=invocation_id,
            context_id=context_id,
            context_epoch=context_epoch,
            source_retraction_transaction_id=source_retraction_transaction_id,
            source_joint_plan_id=source_joint_plan_id,
            created_ts_ms=created_ts_ms,
            path_extent_ids=path_extent_ids,
            cause=cause,
        )
        self._by_request[request_id] = item
        return item

    def get(self, request_id: str) -> RestoreObligation | None:
        return self._by_request.get(request_id)

    def active(self) -> tuple[RestoreObligation, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self._by_request.values()
                    if not item.state.terminal
                ),
                key=lambda item: (item.created_ts_ms, item.obligation_id),
            )
        )

    def all(self) -> tuple[RestoreObligation, ...]:
        return tuple(
            sorted(
                self._by_request.values(),
                key=lambda item: (item.created_ts_ms, item.obligation_id),
            )
        )

    def for_source_transaction(
        self, transaction_id: str
    ) -> tuple[RestoreObligation, ...]:
        return tuple(
            item
            for item in self.active()
            if item.source_retraction_transaction_id == transaction_id
        )


@dataclass(frozen=True)
class PersistentLivenessSnapshot:
    """Versioned P5E state that predictive plans must treat as hard input."""

    obligation_revision: int
    lease_revision: int
    grace_revision: int
    active_obligation_ids: tuple[str, ...]
    active_lease_ids: tuple[str, ...]
    active_grace_request_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if min(
            self.obligation_revision,
            self.lease_revision,
            self.grace_revision,
        ) < 0:
            raise ValueError("persistent liveness revisions must be non-negative")
        for name in (
            "active_obligation_ids",
            "active_lease_ids",
            "active_grace_request_ids",
        ):
            values = tuple(sorted(set(getattr(self, name))))
            if any(not value for value in values):
                raise ValueError(f"persistent liveness {name} is invalid")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, object]:
        return {
            "obligation_revision": self.obligation_revision,
            "lease_revision": self.lease_revision,
            "grace_revision": self.grace_revision,
            "active_obligation_ids": list(self.active_obligation_ids),
            "active_lease_ids": list(self.active_lease_ids),
            "active_grace_request_ids": list(self.active_grace_request_ids),
        }


class PersistentLivenessRevisionTracker:
    """Detect all state mutations, including legacy direct field assignments.

    The P5 runtime still mutates a few obligation fields directly. Computing a
    deterministic fingerprint at scheduler safe points keeps P6 revisions
    complete without making prediction code the owner of those state machines.
    """

    def __init__(self) -> None:
        self._obligation_revision = 0
        self._lease_revision = 0
        self._grace_revision = 0
        self._obligation_fingerprint: tuple[object, ...] | None = None
        self._lease_fingerprint: tuple[object, ...] | None = None
        self._grace_fingerprint: tuple[object, ...] | None = None

    def observe(
        self,
        *,
        obligations: tuple[RestoreObligation, ...],
        leases: tuple[RestoreLease, ...],
        graces: tuple[RestoreServiceGrace, ...],
    ) -> PersistentLivenessSnapshot:
        obligation_fingerprint = tuple(
            (
                item.obligation_id,
                item.request_id,
                item.context_id,
                item.context_epoch,
                item.state.value,
                item.required_extent_ids,
                item.source_transaction_terminal,
                item.requeued,
                item.pending_command_id,
                item.pending_command_kind.value
                if item.pending_command_kind is not None
                else None,
                item.restore_bytes,
                item.required_admission_bytes,
                item.funding_reclaim_bytes,
                item.funding_reserved_tokens,
                item.funding_reserved_bytes,
                item.restored_bytes,
                item.blocker_codes,
                item.retry_count,
                item.bypass_count,
                item.liveness_escalated,
                item.native_admission_fallback,
                item.terminal_reason,
            )
            for item in sorted(obligations, key=lambda value: value.obligation_id)
        )
        lease_fingerprint = tuple(
            (
                item.lease_id,
                item.obligation_id,
                item.request_id,
                item.state.value,
                item.reserved_tokens,
                item.reserved_bytes,
                item.h2d_bytes,
                item.h2d_command_id,
                item.pin_active,
                item.admission_attempts,
                item.terminal_reason,
            )
            for item in sorted(leases, key=lambda value: value.lease_id)
        )
        grace_fingerprint = tuple(
            (
                item.request_id,
                item.obligation_id,
                item.required_decode_tokens,
                item.served_decode_tokens,
                item.last_observed_output_tokens,
                item.completed_ts_ms,
                item.terminal_reason,
            )
            for item in sorted(graces, key=lambda value: value.request_id)
        )
        if obligation_fingerprint != self._obligation_fingerprint:
            self._obligation_revision += 1
            self._obligation_fingerprint = obligation_fingerprint
        if lease_fingerprint != self._lease_fingerprint:
            self._lease_revision += 1
            self._lease_fingerprint = lease_fingerprint
        if grace_fingerprint != self._grace_fingerprint:
            self._grace_revision += 1
            self._grace_fingerprint = grace_fingerprint
        return PersistentLivenessSnapshot(
            obligation_revision=self._obligation_revision,
            lease_revision=self._lease_revision,
            grace_revision=self._grace_revision,
            active_obligation_ids=tuple(
                item.obligation_id for item in obligations if not item.state.terminal
            ),
            active_lease_ids=tuple(
                item.lease_id for item in leases if not item.state.terminal
            ),
            active_grace_request_ids=tuple(
                item.request_id for item in graces if item.active
            ),
        )
