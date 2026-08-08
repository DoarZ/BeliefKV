from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.workflow_fairness import WorkflowFairScheduler
from beliefkv.runtime.page_index import PageOwnershipIndex


@dataclass(frozen=True)
class AdmissionRequest:
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    submitted_ts_ms: float
    uncached_prompt_tokens: int
    expected_output_tokens: int
    kv_bytes_per_token: int
    fixed_overhead_bytes: int = 0
    prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.request_id, "request_id"),
            (self.workflow_id, "workflow_id"),
            (self.invocation_id, "invocation_id"),
            (self.context_id, "context_id"),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.context_epoch < 0 or self.submitted_ts_ms < 0:
            raise ValueError("context epoch and submit time must be non-negative")
        if min(
            self.uncached_prompt_tokens,
            self.expected_output_tokens,
            self.fixed_overhead_bytes,
        ) < 0:
            raise ValueError("admission token/byte estimates must be non-negative")
        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative or null")
        if self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")

    @property
    def estimated_incremental_bytes(self) -> int:
        token_bytes = (
            self.uncached_prompt_tokens + self.expected_output_tokens
        ) * self.kv_bytes_per_token
        return max(0, token_bytes + self.fixed_overhead_bytes)

    @property
    def estimated_working_set_bytes(self) -> int:
        prompt_tokens = (
            self.prompt_tokens
            if self.prompt_tokens is not None
            else self.uncached_prompt_tokens
        )
        return max(
            0,
            (prompt_tokens + self.expected_output_tokens) * self.kv_bytes_per_token
            + self.fixed_overhead_bytes,
        )


@dataclass(frozen=True)
class AdmissionDecision:
    request_id: str
    admitted: bool
    reason: str
    reserved_bytes: int = 0
    required_bytes: int = 0
    native_reclaim_capacity_bytes: int = 0


class AdmissionSideState(str, Enum):
    """Request-local policy state; SGLang remains the request queue owner."""

    VISIBLE_PENDING = "visible_pending"
    WAIT_RESTORE = "wait_restore"
    POLICY_BLOCKED = "policy_blocked"


@dataclass(frozen=True)
class AdmissionLocalVersion:
    """Dependency-scoped version used by one batch-epoch ticket."""

    request_generation: int
    prompt_generation: int
    prefix_generation: int
    transition_generation: int
    bundle_generations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if min(
            self.request_generation,
            self.prompt_generation,
            self.prefix_generation,
            self.transition_generation,
        ) < 0:
            raise ValueError("admission generations must be non-negative")
        bundle_ids = [bundle_id for bundle_id, _ in self.bundle_generations]
        if any(not bundle_id for bundle_id in bundle_ids):
            raise ValueError("bundle IDs must be non-empty")
        if len(bundle_ids) != len(set(bundle_ids)):
            raise ValueError("bundle generations must have unique bundle IDs")
        if any(not generation for _, generation in self.bundle_generations):
            raise ValueError("bundle generations must be non-empty")
        object.__setattr__(
            self,
            "bundle_generations",
            tuple(sorted(self.bundle_generations)),
        )


@dataclass(frozen=True)
class VisibleAdmissionEntry:
    request: AdmissionRequest
    state: AdmissionSideState
    version: AdmissionLocalVersion
    blocker_reason: str | None = None
    restore_bundle_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        restore_ids = tuple(sorted(self.restore_bundle_ids))
        if len(restore_ids) != len(set(restore_ids)):
            raise ValueError("restore bundle IDs must be unique")
        if any(not bundle_id for bundle_id in restore_ids):
            raise ValueError("restore bundle IDs must be non-empty")
        if self.state == AdmissionSideState.VISIBLE_PENDING:
            if self.blocker_reason is not None or restore_ids:
                raise ValueError("visible requests cannot retain blockers")
        elif self.state == AdmissionSideState.WAIT_RESTORE:
            if not restore_ids:
                raise ValueError("WAIT_RESTORE requires at least one bundle")
            if not self.blocker_reason:
                raise ValueError("WAIT_RESTORE requires a blocker reason")
        elif not self.blocker_reason:
            raise ValueError("POLICY_BLOCKED requires a blocker reason")
        object.__setattr__(self, "restore_bundle_ids", restore_ids)


@dataclass(frozen=True)
class AdmissionTicket:
    """Short-lived permission to attempt native admission in one epoch."""

    epoch: int
    rank: int
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    version: AdmissionLocalVersion
    estimated_prefill_tokens: int
    estimated_incremental_bytes: int
    issued_ts_ms: float
    source: str
    reason: str
    reservation_credit_bytes: int = 0

    def __post_init__(self) -> None:
        if min(
            self.epoch,
            self.rank,
            self.context_epoch,
            self.estimated_prefill_tokens,
            self.estimated_incremental_bytes,
            self.reservation_credit_bytes,
        ) < 0:
            raise ValueError("ticket counters must be non-negative")
        if self.issued_ts_ms < 0:
            raise ValueError("ticket issue time must be non-negative")
        for value, name in (
            (self.request_id, "request_id"),
            (self.workflow_id, "workflow_id"),
            (self.invocation_id, "invocation_id"),
            (self.context_id, "context_id"),
            (self.source, "source"),
            (self.reason, "reason"),
        ):
            if not value:
                raise ValueError(f"ticket {name} must be non-empty")


@dataclass(frozen=True)
class AdmissionTicketValidation:
    valid: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionTicketEpoch:
    epoch: int
    tickets: tuple[AdmissionTicket, ...]
    skipped: tuple[tuple[str, str], ...]
    scanned_count: int
    source: str

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.scanned_count < 0:
            raise ValueError("ticket epoch counters must be non-negative")
        request_ids = [ticket.request_id for ticket in self.tickets]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("ticket epoch contains duplicate request IDs")
        if any(ticket.epoch != self.epoch for ticket in self.tickets):
            raise ValueError("ticket belongs to a different epoch")
        if not self.source:
            raise ValueError("ticket epoch source must be non-empty")

    @property
    def by_request_id(self) -> Mapping[str, AdmissionTicket]:
        return MappingProxyType(
            {ticket.request_id: ticket for ticket in self.tickets}
        )


@dataclass(frozen=True)
class AdmissionCompileBudget:
    max_prefill_tokens: int
    max_requests: int
    max_candidates: int
    available_hbm_bytes: int
    reclaimable_hbm_bytes: int = 0
    protected_hbm_bytes: int = 0

    def __post_init__(self) -> None:
        if min(
            self.max_prefill_tokens,
            self.max_requests,
            self.max_candidates,
            self.available_hbm_bytes,
            self.reclaimable_hbm_bytes,
            self.protected_hbm_bytes,
        ) < 0:
            raise ValueError("admission compile budgets must be non-negative")

    @property
    def bounded_hbm_bytes(self) -> int:
        return max(
            0,
            self.available_hbm_bytes
            + self.reclaimable_hbm_bytes
            - self.protected_hbm_bytes,
        )


@dataclass(frozen=True)
class ObservedAdmissionCandidate:
    """One native waiting request ranked only by currently observed facts."""

    request_id: str
    workflow_id: str
    invocation_id: str
    native_index: int
    causal_rank: int
    unblock_depth: int
    frontier_rank: int
    workflow_fair_rank: int
    wait_ms: float
    estimated_incremental_bytes: int
    starvation: bool = False
    policy_eligible: bool = True

    def __post_init__(self) -> None:
        if not self.request_id or not self.workflow_id or not self.invocation_id:
            raise ValueError("observed admission identities must be non-empty")
        if min(
            self.native_index,
            self.causal_rank,
            self.unblock_depth,
            self.frontier_rank,
            self.workflow_fair_rank,
            self.estimated_incremental_bytes,
        ) < 0:
            raise ValueError("observed admission counters must be non-negative")
        if not math.isfinite(self.wait_ms) or self.wait_ms < 0:
            raise ValueError("observed admission wait must be non-negative")

    @property
    def order_key(self) -> tuple[object, ...]:
        if not self.policy_eligible:
            return (
                2,
                self.native_index,
                self.request_id,
            )
        if self.starvation:
            return (
                0,
                -self.wait_ms,
                self.causal_rank,
                -self.unblock_depth,
                self.frontier_rank,
                self.workflow_fair_rank,
                self.estimated_incremental_bytes,
                self.native_index,
                self.request_id,
            )
        return (
            1,
            self.causal_rank,
            -self.unblock_depth,
            self.frontier_rank,
            self.workflow_fair_rank,
            self.estimated_incremental_bytes,
            -self.wait_ms,
            self.native_index,
            self.request_id,
        )


@dataclass(frozen=True)
class ObservedAdmissionSnapshot:
    """Authoritative inputs used to bound one batch-epoch active-set growth."""

    hbm_capacity_bytes: int
    reserve_hbm_bytes: int
    native_available_hbm_bytes: int
    native_max_requests: int
    running_request_count: int
    radix_locked_bytes: int
    running_private_bytes: int

    def __post_init__(self) -> None:
        if self.hbm_capacity_bytes <= 0:
            raise ValueError("observed admission HBM capacity must be positive")
        if not 0 <= self.reserve_hbm_bytes < self.hbm_capacity_bytes:
            raise ValueError("observed admission reserve must be within capacity")
        if min(
            self.native_available_hbm_bytes,
            self.native_max_requests,
            self.running_request_count,
            self.radix_locked_bytes,
            self.running_private_bytes,
        ) < 0:
            raise ValueError("observed admission resources must be non-negative")

    @property
    def active_kv_footprint_bytes(self) -> int:
        return self.radix_locked_bytes + self.running_private_bytes


@dataclass(frozen=True)
class ObservedAdmissionWindow:
    ordered_request_ids: tuple[str, ...]
    max_new_requests: int
    active_growth_budget_bytes: int
    active_kv_budget_bytes: int
    active_kv_footprint_bytes: int
    active_kv_headroom_bytes: int
    radix_locked_bytes: int
    running_private_bytes: int
    running_request_count: int
    mode: str

    def __post_init__(self) -> None:
        if len(self.ordered_request_ids) != len(set(self.ordered_request_ids)):
            raise ValueError("observed admission order contains duplicate requests")
        if min(
            self.max_new_requests,
            self.active_growth_budget_bytes,
            self.active_kv_budget_bytes,
            self.active_kv_footprint_bytes,
            self.active_kv_headroom_bytes,
            self.radix_locked_bytes,
            self.running_private_bytes,
            self.running_request_count,
        ) < 0:
            raise ValueError("observed admission window values must be non-negative")
        if not self.mode:
            raise ValueError("observed admission mode must be non-empty")

    @property
    def active_kv_pressure(self) -> float:
        if self.active_kv_budget_bytes == 0:
            return 1.0 if self.active_kv_footprint_bytes else 0.0
        return self.active_kv_footprint_bytes / self.active_kv_budget_bytes


class ObservedAdmissionScheduler:
    """Bound active request growth without predicting future workflow actions."""

    def __init__(
        self,
        *,
        active_kv_high_watermark_ratio: float = 0.8,
        minimum_active_requests: int = 1,
    ) -> None:
        if (
            not math.isfinite(active_kv_high_watermark_ratio)
            or not 0 < active_kv_high_watermark_ratio <= 1
        ):
            raise ValueError("active KV high watermark must be in (0, 1]")
        if minimum_active_requests < 0:
            raise ValueError("minimum active requests must be non-negative")
        self.active_kv_high_watermark_ratio = active_kv_high_watermark_ratio
        self.minimum_active_requests = minimum_active_requests

    def decide(
        self,
        candidates: Sequence[ObservedAdmissionCandidate],
        snapshot: ObservedAdmissionSnapshot,
    ) -> ObservedAdmissionWindow:
        ordered = tuple(
            item.request_id
            for item in sorted(candidates, key=lambda item: item.order_key)
        )
        allocatable_hbm = max(
            0,
            snapshot.hbm_capacity_bytes - snapshot.reserve_hbm_bytes,
        )
        active_budget = int(
            allocatable_hbm * self.active_kv_high_watermark_ratio
        )
        active_footprint = snapshot.active_kv_footprint_bytes
        active_headroom = max(0, active_budget - active_footprint)
        native_slots = min(snapshot.native_max_requests, len(ordered))
        floor_deficit = max(
            0,
            self.minimum_active_requests - snapshot.running_request_count,
        )

        if native_slots == 0:
            mode = "no_native_slot"
            max_new_requests = 0
            growth_budget = 0
        elif floor_deficit > 0:
            # This is the only high-watermark bypass. It is bounded by a fixed
            # active-request floor and by SGLang's authoritative free capacity.
            mode = "work_conserving_floor"
            max_new_requests = min(native_slots, floor_deficit)
            growth_budget = snapshot.native_available_hbm_bytes
        else:
            mode = (
                "active_kv_pressure_hold"
                if active_headroom == 0
                else "active_kv_bounded"
            )
            max_new_requests = native_slots
            growth_budget = min(
                snapshot.native_available_hbm_bytes,
                active_headroom,
            )

        return ObservedAdmissionWindow(
            ordered_request_ids=ordered,
            max_new_requests=max_new_requests,
            active_growth_budget_bytes=growth_budget,
            active_kv_budget_bytes=active_budget,
            active_kv_footprint_bytes=active_footprint,
            active_kv_headroom_bytes=active_headroom,
            radix_locked_bytes=snapshot.radix_locked_bytes,
            running_private_bytes=snapshot.running_private_bytes,
            running_request_count=snapshot.running_request_count,
            mode=mode,
        )


class VisibleAdmissionIndex:
    """Request-ID side index with no request objects and no HBM reservations."""

    def __init__(self) -> None:
        self._entries: dict[str, VisibleAdmissionEntry] = {}
        self._generation = 0
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def reserved_bytes(self) -> int:
        return 0

    def get(self, request_id: str) -> VisibleAdmissionEntry | None:
        return self._entries.get(request_id)

    def entries(self) -> tuple[VisibleAdmissionEntry, ...]:
        return tuple(self._entries.values())

    def register(
        self,
        request: AdmissionRequest,
        *,
        transition_generation: int = 0,
        bundle_generations: Mapping[str, str] | None = None,
    ) -> VisibleAdmissionEntry:
        if request.request_id in self._entries:
            raise ValueError(f"duplicate visible request id: {request.request_id}")
        generation = self._next_generation()
        entry = VisibleAdmissionEntry(
            request=request,
            state=AdmissionSideState.VISIBLE_PENDING,
            version=AdmissionLocalVersion(
                request_generation=generation,
                prompt_generation=generation,
                prefix_generation=generation,
                transition_generation=transition_generation,
                bundle_generations=self._bundle_tuple(bundle_generations),
            ),
        )
        self._entries[request.request_id] = entry
        self._revision += 1
        return entry

    def cancel(self, request_id: str) -> VisibleAdmissionEntry | None:
        entry = self._entries.pop(request_id, None)
        if entry is not None:
            self._next_generation()
            self._revision += 1
        return entry

    def update_request(
        self,
        request: AdmissionRequest,
        *,
        prompt_changed: bool = False,
    ) -> VisibleAdmissionEntry:
        current = self._require(request.request_id)
        if (
            current.request.workflow_id,
            current.request.invocation_id,
            current.request.context_id,
            current.request.context_epoch,
        ) != (
            request.workflow_id,
            request.invocation_id,
            request.context_id,
            request.context_epoch,
        ):
            raise ValueError("visible request identity cannot change in place")
        if current.request == request and not prompt_changed:
            return current
        generation = self._next_generation()
        version = replace(
            current.version,
            request_generation=generation,
            prompt_generation=(
                generation
                if prompt_changed
                else current.version.prompt_generation
            ),
        )
        return self._replace(current, request=request, version=version)

    def observe_prefix(
        self,
        request_id: str,
        *,
        uncached_prompt_tokens: int,
        bundle_generations: Mapping[str, str] | None = None,
    ) -> VisibleAdmissionEntry:
        if uncached_prompt_tokens < 0:
            raise ValueError("uncached prompt tokens must be non-negative")
        current = self._require(request_id)
        bundles = self._bundle_tuple(bundle_generations)
        request = replace(
            current.request,
            uncached_prompt_tokens=uncached_prompt_tokens,
        )
        if (
            request == current.request
            and bundles == current.version.bundle_generations
        ):
            return current
        generation = self._next_generation()
        version = replace(
            current.version,
            request_generation=generation,
            prefix_generation=generation,
            bundle_generations=bundles,
        )
        return self._replace(current, request=request, version=version)

    def set_transition_generation(
        self,
        request_id: str,
        generation: int,
    ) -> VisibleAdmissionEntry:
        if generation < 0:
            raise ValueError("transition generation must be non-negative")
        current = self._require(request_id)
        if current.version.transition_generation == generation:
            return current
        local_generation = self._next_generation()
        return self._replace(
            current,
            version=replace(
                current.version,
                request_generation=local_generation,
                transition_generation=generation,
            ),
        )

    def set_wait_restore(
        self,
        request_id: str,
        bundle_ids: Sequence[str],
        *,
        reason: str,
    ) -> VisibleAdmissionEntry:
        return self._set_state(
            request_id,
            AdmissionSideState.WAIT_RESTORE,
            blocker_reason=reason,
            restore_bundle_ids=tuple(bundle_ids),
        )

    def set_policy_blocked(
        self,
        request_id: str,
        *,
        reason: str,
    ) -> VisibleAdmissionEntry:
        return self._set_state(
            request_id,
            AdmissionSideState.POLICY_BLOCKED,
            blocker_reason=reason,
        )

    def set_visible(self, request_id: str) -> VisibleAdmissionEntry:
        return self._set_state(request_id, AdmissionSideState.VISIBLE_PENDING)

    def validate_ticket(
        self,
        ticket: AdmissionTicket,
        *,
        epoch: int,
        bundle_generations: Mapping[str, str] | None = None,
    ) -> AdmissionTicketValidation:
        reasons: list[str] = []
        if ticket.epoch != epoch:
            reasons.append("epoch_expired")
        current = self._entries.get(ticket.request_id)
        if current is None:
            reasons.append("request_missing")
            return AdmissionTicketValidation(False, tuple(reasons))
        if current.state != AdmissionSideState.VISIBLE_PENDING:
            reasons.append(current.state.value)
        if ticket.context_epoch != current.request.context_epoch:
            reasons.append("context_epoch")
        for name in (
            "request_generation",
            "prompt_generation",
            "prefix_generation",
            "transition_generation",
        ):
            if getattr(ticket.version, name) != getattr(current.version, name):
                reasons.append(name)
        observed_bundles = dict(
            current.version.bundle_generations
            if bundle_generations is None
            else self._bundle_tuple(bundle_generations)
        )
        for bundle_id, generation in ticket.version.bundle_generations:
            if observed_bundles.get(bundle_id) != generation:
                reasons.append(f"bundle_generation:{bundle_id}")
        return AdmissionTicketValidation(not reasons, tuple(reasons))

    def _set_state(
        self,
        request_id: str,
        state: AdmissionSideState,
        *,
        blocker_reason: str | None = None,
        restore_bundle_ids: tuple[str, ...] = (),
    ) -> VisibleAdmissionEntry:
        current = self._require(request_id)
        candidate = VisibleAdmissionEntry(
            request=current.request,
            state=state,
            version=current.version,
            blocker_reason=blocker_reason,
            restore_bundle_ids=restore_bundle_ids,
        )
        if candidate == current:
            return current
        generation = self._next_generation()
        return self._replace(
            candidate,
            version=replace(
                current.version,
                request_generation=generation,
            ),
        )

    def _replace(
        self,
        current: VisibleAdmissionEntry,
        **changes: object,
    ) -> VisibleAdmissionEntry:
        updated = replace(current, **changes)
        self._entries[current.request.request_id] = updated
        self._revision += 1
        return updated

    def _require(self, request_id: str) -> VisibleAdmissionEntry:
        try:
            return self._entries[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown visible request: {request_id}") from exc

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    @staticmethod
    def _bundle_tuple(
        values: Mapping[str, str] | None,
    ) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(key), str(value)) for key, value in (values or {}).items()))


class AdmissionTicketCompiler:
    """Compile bounded eligibility without mutating queue or allocator state."""

    def compile(
        self,
        *,
        epoch: int,
        now_ms: float,
        ordered_request_ids: Sequence[str],
        entries: Mapping[str, VisibleAdmissionEntry],
        budget: AdmissionCompileBudget,
        source: str,
        reason: str,
        reservation_credits: Mapping[str, int] | None = None,
    ) -> AdmissionTicketEpoch:
        if epoch < 0 or now_ms < 0:
            raise ValueError("ticket epoch and time must be non-negative")
        if not source or not reason:
            raise ValueError("ticket source and reason must be non-empty")
        if len(ordered_request_ids) != len(set(ordered_request_ids)):
            raise ValueError("ordered request IDs must be unique")
        credits = dict(reservation_credits or {})
        if any(not request_id or value < 0 for request_id, value in credits.items()):
            raise ValueError("admission reservation credits must be non-negative")

        tickets: list[AdmissionTicket] = []
        skipped: list[tuple[str, str]] = []
        remaining_tokens = budget.max_prefill_tokens
        remaining_hbm = budget.bounded_hbm_bytes
        scanned = 0
        for request_id in ordered_request_ids:
            if scanned >= budget.max_candidates:
                break
            scanned += 1
            entry = entries.get(request_id)
            if entry is None:
                skipped.append((request_id, "request_missing"))
                continue
            if entry.state != AdmissionSideState.VISIBLE_PENDING:
                skipped.append((request_id, entry.state.value))
                continue
            if len(tickets) >= budget.max_requests:
                skipped.append((request_id, "request_slot_budget"))
                continue
            uncached_prompt_tokens = entry.request.uncached_prompt_tokens
            required_bytes = entry.request.estimated_incremental_bytes
            reservation_credit_bytes = min(
                required_bytes, credits.get(request_id, 0)
            )
            budgeted_bytes = required_bytes - reservation_credit_bytes
            if uncached_prompt_tokens > 0 and remaining_tokens <= 0:
                skipped.append((request_id, "prefill_token_budget"))
                continue
            # The native budget is per scheduler epoch. A request larger than
            # this budget remains admissible because SGLang can prefill its
            # first chunk now and retain the remainder as ``chunked_req``.
            prefill_tokens = min(uncached_prompt_tokens, remaining_tokens)
            if budgeted_bytes > remaining_hbm:
                skipped.append((request_id, "bounded_hbm_budget"))
                continue
            ticket = AdmissionTicket(
                epoch=epoch,
                rank=len(tickets),
                request_id=request_id,
                workflow_id=entry.request.workflow_id,
                invocation_id=entry.request.invocation_id,
                context_id=entry.request.context_id,
                context_epoch=entry.request.context_epoch,
                version=entry.version,
                estimated_prefill_tokens=prefill_tokens,
                estimated_incremental_bytes=required_bytes,
                issued_ts_ms=now_ms,
                source=source,
                reason=reason,
                reservation_credit_bytes=reservation_credit_bytes,
            )
            tickets.append(ticket)
            remaining_tokens -= prefill_tokens
            remaining_hbm -= budgeted_bytes
        return AdmissionTicketEpoch(
            epoch=epoch,
            tickets=tuple(tickets),
            skipped=tuple(skipped),
            scanned_count=scanned,
            source=source,
        )


class AdmissionController:
    """Workflow-fair admission based only on physical bytes and reservations."""

    def __init__(
        self,
        page_index: PageOwnershipIndex,
        fairness: WorkflowFairScheduler,
        frontier: CausalFrontierScheduler,
        *,
        reserve_hbm_bytes: int,
        max_borrow_fraction: float = 0.25,
    ) -> None:
        if reserve_hbm_bytes < 0:
            raise ValueError("reserve_hbm_bytes must be non-negative")
        if not 0 <= max_borrow_fraction <= 1:
            raise ValueError("max_borrow_fraction must be in [0, 1]")
        self.page_index = page_index
        self.fairness = fairness
        self.frontier = frontier
        self.reserve_hbm_bytes = reserve_hbm_bytes
        self.max_borrow_fraction = max_borrow_fraction
        self._pending: dict[str, AdmissionRequest] = {}
        self._reserved: dict[str, AdmissionRequest] = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def enqueue(self, request: AdmissionRequest) -> None:
        if request.request_id in self._pending or request.request_id in self._reserved:
            raise ValueError(f"duplicate request id: {request.request_id}")
        self._pending[request.request_id] = request
        self.fairness.register(request.workflow_id)
        self._revision += 1

    def cancel(self, request_id: str) -> None:
        removed = self._pending.pop(request_id, None)
        reserved = self._reserved.pop(request_id, None)
        if removed is not None or reserved is not None:
            self._revision += 1

    def update_pending_estimate(
        self,
        request_id: str,
        *,
        uncached_prompt_tokens: int,
    ) -> AdmissionRequest:
        """Refresh a deferred request after the physical prefix tree changes."""

        if uncached_prompt_tokens < 0:
            raise ValueError("uncached_prompt_tokens must be non-negative")
        try:
            request = self._pending[request_id]
        except KeyError as exc:
            raise KeyError(f"request is not pending: {request_id}") from exc
        updated = replace(
            request,
            uncached_prompt_tokens=uncached_prompt_tokens,
        )
        self._pending[request_id] = updated
        if updated != request:
            self._revision += 1
        return updated

    @property
    def reserved_bytes(self) -> int:
        return sum(
            request.estimated_incremental_bytes
            for request in self._reserved.values()
        )

    def reserved_requests(self) -> list[AdmissionRequest]:
        return sorted(
            self._reserved.values(),
            key=lambda item: (item.submitted_ts_ms, item.request_id),
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def decide_next(
        self,
        hbm_capacity_bytes: int,
        *,
        actual_hbm_used_bytes: int | None = None,
        external_workflow_charges: dict[str, float] | None = None,
        allow_reserve_borrow: bool = False,
        preferred_request_id: str | None = None,
        native_reclaim_capacity_bytes: int | None = None,
    ) -> AdmissionDecision | None:
        if not self._pending:
            return None
        if (
            native_reclaim_capacity_bytes is not None
            and native_reclaim_capacity_bytes < 0
        ):
            raise ValueError("native reclaim capacity must be non-negative or null")
        allocatable = max(0, hbm_capacity_bytes - self.reserve_hbm_bytes)
        physical_used = max(
            self.page_index.gpu_bytes,
            actual_hbm_used_bytes if actual_hbm_used_bytes is not None else 0,
        )
        free = max(0, allocatable - physical_used - self.reserved_bytes)
        charges = self.page_index.workflow_gpu_charges()
        for workflow_id, charge in (external_workflow_charges or {}).items():
            charges[workflow_id] = charges.get(workflow_id, 0.0) + max(0.0, charge)
        workflows = {item.workflow_id for item in self._pending.values()}
        shares = self.fairness.fair_memory_shares(workflows, allocatable)

        def fitting_requests(
            available_bytes: int,
        ) -> dict[str, list[AdmissionRequest]]:
            result: dict[str, list[AdmissionRequest]] = {}
            for request in self._pending.values():
                if request.estimated_incremental_bytes <= available_bytes:
                    result.setdefault(request.workflow_id, []).append(request)
            return result

        preferred = (
            self._pending.get(preferred_request_id)
            if preferred_request_id is not None
            else None
        )
        if preferred is not None:
            preferred_reason = "admission_liveness_target"
            preferred_free = free
            if (
                preferred.estimated_incremental_bytes > preferred_free
                and allow_reserve_borrow
            ):
                preferred_free = max(
                    0,
                    hbm_capacity_bytes - physical_used - self.reserved_bytes,
                )
                preferred_reason = "admission_liveness_reserve_borrow"
            if preferred.estimated_incremental_bytes > preferred_free:
                if native_reclaim_capacity_bytes is not None:
                    if preferred.estimated_working_set_bytes > hbm_capacity_bytes:
                        return AdmissionDecision(
                            preferred.request_id,
                            False,
                            "request_exceeds_hbm_capacity",
                            required_bytes=preferred.estimated_incremental_bytes,
                            native_reclaim_capacity_bytes=(
                                native_reclaim_capacity_bytes
                            ),
                        )
                    # SGLang's PrefillAdder rejects equality as NO_TOKEN.
                    if (
                        preferred.estimated_incremental_bytes
                        < native_reclaim_capacity_bytes
                    ):
                        self._reserve(preferred)
                        return AdmissionDecision(
                            preferred.request_id,
                            True,
                            "admission_liveness_native_reclaim",
                            preferred.estimated_incremental_bytes,
                            required_bytes=preferred.estimated_incremental_bytes,
                            native_reclaim_capacity_bytes=(
                                native_reclaim_capacity_bytes
                            ),
                        )
                    return AdmissionDecision(
                        preferred.request_id,
                        False,
                        "insufficient_native_reclaim_capacity",
                        required_bytes=preferred.estimated_incremental_bytes,
                        native_reclaim_capacity_bytes=native_reclaim_capacity_bytes,
                    )
                return AdmissionDecision(
                    preferred.request_id,
                    False,
                    "insufficient_actual_hbm",
                    required_bytes=preferred.estimated_incremental_bytes,
                )
            self._reserve(preferred)
            return AdmissionDecision(
                preferred.request_id,
                True,
                preferred_reason,
                preferred.estimated_incremental_bytes,
                required_bytes=preferred.estimated_incremental_bytes,
            )

        fitting_by_workflow = fitting_requests(free)
        borrowed_reserve = False
        if not fitting_by_workflow and allow_reserve_borrow:
            hard_free = max(
                0,
                hbm_capacity_bytes - physical_used - self.reserved_bytes,
            )
            fitting_by_workflow = fitting_requests(hard_free)
            borrowed_reserve = bool(fitting_by_workflow)
        if not fitting_by_workflow:
            request = min(
                self._pending.values(),
                key=lambda item: (item.submitted_ts_ms, item.request_id),
            )
            return AdmissionDecision(
                request.request_id,
                False,
                "insufficient_actual_hbm",
                required_bytes=request.estimated_incremental_bytes,
            )

        eligible_workflows = set(fitting_by_workflow)
        under_soft_share = {
            workflow_id
            for workflow_id, requests in fitting_by_workflow.items()
            if any(
                charges.get(workflow_id, 0.0) + item.estimated_incremental_bytes
                <= shares[workflow_id]
                for item in requests
            )
        }
        if under_soft_share:
            eligible_workflows = under_soft_share
        else:
            borrow_limit = allocatable * self.max_borrow_fraction
            borrowable = {
                workflow_id
                for workflow_id, requests in fitting_by_workflow.items()
                if any(
                    charges.get(workflow_id, 0.0) + item.estimated_incremental_bytes
                    <= shares[workflow_id] + borrow_limit
                    for item in requests
                )
            }
            if borrowable:
                eligible_workflows = borrowable

        selected_workflow = self.fairness.select(
            eligible_workflows,
            memory_charges=charges,
            hbm_capacity_bytes=hbm_capacity_bytes,
        )
        assert selected_workflow is not None
        requests = fitting_by_workflow[selected_workflow]
        frontier_order = {
            item.invocation_id: index
            for index, item in enumerate(self.frontier.candidates(selected_workflow))
        }
        selected = min(
            requests,
            key=lambda item: (
                frontier_order.get(item.invocation_id, 1 << 30),
                item.submitted_ts_ms,
                item.request_id,
            ),
        )
        self._reserve(selected)
        return AdmissionDecision(
            selected.request_id,
            True,
            (
                "engine_idle_reserve_borrow"
                if borrowed_reserve
                else "workflow_fair_causal_frontier"
            ),
            selected.estimated_incremental_bytes,
            required_bytes=selected.estimated_incremental_bytes,
        )

    def acknowledge(self, request_id: str) -> int:
        try:
            request = self._reserved.pop(request_id)
        except KeyError as exc:
            raise KeyError(f"request has no admission reservation: {request_id}") from exc
        self._revision += 1
        return request.estimated_incremental_bytes

    def _reserve(self, request: AdmissionRequest) -> None:
        self._pending.pop(request.request_id)
        self._reserved[request.request_id] = request
        self._revision += 1

    def pending_requests(self) -> list[AdmissionRequest]:
        return sorted(
            self._pending.values(),
            key=lambda item: (item.submitted_ts_ms, item.request_id),
        )
