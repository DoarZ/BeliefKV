from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RunningRetractionCandidate:
    """One tagged request currently owning decode KV and Radix locks."""

    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    private_kv_bytes: int
    service_status: str
    stale_for_ms: float
    causal_rank: int
    unblock_depth: int
    workflow_fair_rank: int
    prior_retraction_count: int = 0
    policy_eligible: bool = True

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.request_id,
                self.workflow_id,
                self.invocation_id,
                self.context_id,
            )
        ):
            raise ValueError("retraction candidate identities must be non-empty")
        if self.service_status not in {"recent", "stale", "warming", "unknown"}:
            raise ValueError("unsupported request service status")
        if min(
            self.private_kv_bytes,
            self.causal_rank,
            self.unblock_depth,
            self.workflow_fair_rank,
            self.prior_retraction_count,
        ) < 0:
            raise ValueError("retraction candidate counters must be non-negative")
        if not math.isfinite(self.stale_for_ms) or self.stale_for_ms < 0:
            raise ValueError("retraction service age must be non-negative")


@dataclass(frozen=True)
class RetractionLockedExtent:
    """A physical Radix extent and the complete observed lock provenance."""

    extent_id: str
    size_bytes: int
    blocker_request_ids: tuple[str, ...]
    fully_attributed: bool

    def __post_init__(self) -> None:
        if not self.extent_id or self.size_bytes <= 0:
            raise ValueError("retraction extent identity and size are required")
        if tuple(sorted(set(self.blocker_request_ids))) != self.blocker_request_ids:
            raise ValueError("extent blockers must be sorted and unique")


@dataclass(frozen=True)
class RetractionReplacement:
    request_id: str
    estimated_incremental_bytes: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("replacement request id must be non-empty")
        if self.estimated_incremental_bytes < 0:
            raise ValueError("replacement bytes must be non-negative")


@dataclass(frozen=True)
class ObservedRetractionSnapshot:
    observed_ts_ms: float
    page_revision: int
    topology_revision: int
    hbm_capacity_bytes: int
    active_kv_budget_bytes: int
    active_kv_footprint_bytes: int
    native_reclaim_capacity_bytes: int
    admission_stall_ms: float
    running_request_count: int
    minimum_active_requests: int
    candidates: tuple[RunningRetractionCandidate, ...]
    locked_extents: tuple[RetractionLockedExtent, ...]
    replacements: tuple[RetractionReplacement, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.observed_ts_ms) or self.observed_ts_ms < 0:
            raise ValueError("retraction observation time must be non-negative")
        if self.hbm_capacity_bytes <= 0:
            raise ValueError("retraction HBM capacity must be positive")
        if min(
            self.page_revision,
            self.topology_revision,
            self.active_kv_budget_bytes,
            self.active_kv_footprint_bytes,
            self.native_reclaim_capacity_bytes,
            self.running_request_count,
            self.minimum_active_requests,
        ) < 0:
            raise ValueError("retraction snapshot counters must be non-negative")
        if not math.isfinite(self.admission_stall_ms) or self.admission_stall_ms < 0:
            raise ValueError("retraction admission stall must be non-negative")
        candidate_ids = [item.request_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("retraction candidates must be unique")
        replacement_ids = [item.request_id for item in self.replacements]
        if len(replacement_ids) != len(set(replacement_ids)):
            raise ValueError("retraction replacements must be unique")


@dataclass(frozen=True)
class RunningRetractionPlan:
    request_ids: tuple[str, ...]
    replacement_request_ids: tuple[str, ...]
    target_reclaim_bytes: int
    expected_private_reclaim_bytes: int
    expected_lock_release_bytes: int
    expected_reclaim_capacity_bytes: int
    native_reclaim_capacity_before: int
    engine_locked_bytes_before: int
    page_revision: int
    topology_revision: int
    observed_ts_ms: float
    reason: str

    def __post_init__(self) -> None:
        if not self.request_ids or len(self.request_ids) != len(set(self.request_ids)):
            raise ValueError("retraction plan requires unique request ids")
        if len(self.replacement_request_ids) != len(
            set(self.replacement_request_ids)
        ):
            raise ValueError("replacement request ids must be unique")
        if min(
            self.target_reclaim_bytes,
            self.expected_private_reclaim_bytes,
            self.expected_lock_release_bytes,
            self.expected_reclaim_capacity_bytes,
            self.native_reclaim_capacity_before,
            self.engine_locked_bytes_before,
            self.page_revision,
            self.topology_revision,
        ) < 0:
            raise ValueError("retraction plan counters must be non-negative")
        if self.expected_reclaim_capacity_bytes < self.target_reclaim_bytes:
            raise ValueError("retraction plan does not cover its reclaim target")
        if not self.reason:
            raise ValueError("retraction plan reason must be non-empty")


@dataclass(frozen=True)
class ObservedRetractionDecision:
    """A plan or an explicit reason why observed-state planning stopped."""

    plan: RunningRetractionPlan | None
    reason: str
    target_reclaim_bytes: int = 0
    candidate_count: int = 0
    eligible_candidate_count: int = 0
    fully_attributed_extent_count: int = 0
    reclaim_capacity_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("retraction decision reason must be non-empty")
        if min(
            self.target_reclaim_bytes,
            self.candidate_count,
            self.eligible_candidate_count,
            self.fully_attributed_extent_count,
            self.reclaim_capacity_bytes,
        ) < 0:
            raise ValueError("retraction decision counters must be non-negative")
        if self.plan is not None and self.reason != "plan_created":
            raise ValueError("a retraction plan requires plan_created reason")


@dataclass(frozen=True)
class ObservedRetractionConfig:
    minimum_admission_stall_ms: float = 100.0
    minimum_reclaim_bytes: int = 64 * 1024 * 1024
    maximum_retractions_per_request: int = 3

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_admission_stall_ms)
            or self.minimum_admission_stall_ms <= 0
        ):
            raise ValueError("minimum admission stall must be positive")
        if self.minimum_reclaim_bytes <= 0:
            raise ValueError("minimum retraction reclaim must be positive")
        if self.maximum_retractions_per_request <= 0:
            raise ValueError("maximum request retractions must be positive")


class ObservedRetractionPlanner:
    """Select a blocker closure without predicting future agent actions."""

    def __init__(self, config: ObservedRetractionConfig | None = None) -> None:
        self.config = config or ObservedRetractionConfig()

    def plan(
        self,
        snapshot: ObservedRetractionSnapshot,
    ) -> RunningRetractionPlan | None:
        return self.decide(snapshot).plan

    def decide(
        self,
        snapshot: ObservedRetractionSnapshot,
    ) -> ObservedRetractionDecision:
        if snapshot.admission_stall_ms < self.config.minimum_admission_stall_ms:
            return ObservedRetractionDecision(
                plan=None,
                reason="admission_stall_below_threshold",
                candidate_count=len(snapshot.candidates),
            )
        if not snapshot.replacements:
            return ObservedRetractionDecision(
                plan=None,
                reason="replacement_absent",
                candidate_count=len(snapshot.candidates),
            )
        if snapshot.running_request_count <= snapshot.minimum_active_requests:
            return ObservedRetractionDecision(
                plan=None,
                reason="active_floor",
                candidate_count=len(snapshot.candidates),
            )

        first_replacement_bytes = snapshot.replacements[0].estimated_incremental_bytes
        replacement_deficit = max(
            0,
            first_replacement_bytes - snapshot.native_reclaim_capacity_bytes,
        )
        active_excess = max(
            0,
            snapshot.active_kv_footprint_bytes - snapshot.active_kv_budget_bytes,
        )
        if replacement_deficit == 0 and active_excess == 0:
            return ObservedRetractionDecision(
                plan=None,
                reason="pressure_absent",
                candidate_count=len(snapshot.candidates),
            )
        target = max(
            self.config.minimum_reclaim_bytes,
            replacement_deficit,
            active_excess,
        )

        eligible = {
            item.request_id: item
            for item in snapshot.candidates
            if item.policy_eligible
            and item.service_status == "stale"
            and item.prior_retraction_count
            < self.config.maximum_retractions_per_request
        }
        max_victims = max(
            0,
            snapshot.running_request_count - snapshot.minimum_active_requests,
        )
        if not eligible or max_victims == 0:
            return ObservedRetractionDecision(
                plan=None,
                reason="no_eligible_stale_candidate",
                target_reclaim_bytes=target,
                candidate_count=len(snapshot.candidates),
                eligible_candidate_count=len(eligible),
            )

        fully_attributed_extents = tuple(
            item
            for item in snapshot.locked_extents
            if item.fully_attributed and item.blocker_request_ids
        )
        packages: set[frozenset[str]] = {
            frozenset((request_id,)) for request_id in eligible
        }
        packages.update(
            frozenset(extent.blocker_request_ids)
            for extent in fully_attributed_extents
            if set(extent.blocker_request_ids).issubset(eligible)
        )

        selected: set[str] = set()
        current_reclaim = 0
        while current_reclaim < target:
            best: tuple[tuple[object, ...], frozenset[str], int] | None = None
            for package in packages:
                additions = set(package) - selected
                if not additions or len(selected | additions) > max_victims:
                    continue
                candidate_selection = selected | additions
                reclaim = self._reclaim_capacity(
                    candidate_selection,
                    eligible,
                    fully_attributed_extents,
                )
                marginal = reclaim - current_reclaim
                if marginal <= 0:
                    continue
                recompute_bytes = sum(
                    eligible[request_id].private_kv_bytes
                    for request_id in additions
                )
                causal_rank = min(
                    eligible[request_id].causal_rank for request_id in additions
                )
                unblock_depth = max(
                    eligible[request_id].unblock_depth for request_id in additions
                )
                fair_rank = min(
                    eligible[request_id].workflow_fair_rank
                    for request_id in additions
                )
                stale_for_ms = min(
                    eligible[request_id].stale_for_ms for request_id in additions
                )
                prior_retractions = max(
                    eligible[request_id].prior_retraction_count
                    for request_id in additions
                )
                amplification = marginal / max(1, recompute_bytes)
                key = (
                    amplification,
                    marginal,
                    causal_rank,
                    -unblock_depth,
                    fair_rank,
                    stale_for_ms,
                    -prior_retractions,
                    -len(additions),
                    tuple(sorted(additions)),
                )
                if best is None or key > best[0]:
                    best = (key, frozenset(additions), reclaim)
            if best is None:
                optimistic_reclaim = self._reclaim_capacity(
                    set(eligible),
                    eligible,
                    fully_attributed_extents,
                )
                return ObservedRetractionDecision(
                    plan=None,
                    reason=(
                        "insufficient_unlock_capacity"
                        if optimistic_reclaim < target
                        else "selection_search_exhausted"
                    ),
                    target_reclaim_bytes=target,
                    candidate_count=len(snapshot.candidates),
                    eligible_candidate_count=len(eligible),
                    fully_attributed_extent_count=len(fully_attributed_extents),
                    reclaim_capacity_bytes=optimistic_reclaim,
                )
            selected.update(best[1])
            current_reclaim = best[2]

        selected_ids = tuple(
            sorted(
                selected,
                key=lambda request_id: (
                    -eligible[request_id].causal_rank,
                    -eligible[request_id].workflow_fair_rank,
                    -eligible[request_id].stale_for_ms,
                    request_id,
                ),
            )
        )
        private_bytes = sum(eligible[item].private_kv_bytes for item in selected)
        lock_release_bytes = sum(
            extent.size_bytes
            for extent in fully_attributed_extents
            if set(extent.blocker_request_ids).issubset(selected)
        )
        replacement_ids: list[str] = []
        remaining_capacity = (
            snapshot.native_reclaim_capacity_bytes + current_reclaim
        )
        for replacement in snapshot.replacements:
            if replacement.estimated_incremental_bytes > remaining_capacity:
                break
            replacement_ids.append(replacement.request_id)
            remaining_capacity -= replacement.estimated_incremental_bytes

        plan = RunningRetractionPlan(
            request_ids=selected_ids,
            replacement_request_ids=tuple(replacement_ids),
            target_reclaim_bytes=target,
            expected_private_reclaim_bytes=private_bytes,
            expected_lock_release_bytes=lock_release_bytes,
            expected_reclaim_capacity_bytes=current_reclaim,
            native_reclaim_capacity_before=snapshot.native_reclaim_capacity_bytes,
            engine_locked_bytes_before=sum(
                extent.size_bytes for extent in snapshot.locked_extents
            ),
            page_revision=snapshot.page_revision,
            topology_revision=snapshot.topology_revision,
            observed_ts_ms=snapshot.observed_ts_ms,
            reason="observed_lock_closure_reclaim",
        )
        return ObservedRetractionDecision(
            plan=plan,
            reason="plan_created",
            target_reclaim_bytes=target,
            candidate_count=len(snapshot.candidates),
            eligible_candidate_count=len(eligible),
            fully_attributed_extent_count=len(fully_attributed_extents),
            reclaim_capacity_bytes=current_reclaim,
        )

    @staticmethod
    def _reclaim_capacity(
        selected: set[str],
        candidates: dict[str, RunningRetractionCandidate],
        extents: tuple[RetractionLockedExtent, ...],
    ) -> int:
        private_bytes = sum(candidates[item].private_kv_bytes for item in selected)
        unlocked_bytes = sum(
            extent.size_bytes
            for extent in extents
            if set(extent.blocker_request_ids).issubset(selected)
        )
        return private_bytes + unlocked_bytes
