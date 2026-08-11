from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Callable, Mapping, Protocol

from beliefkv.predictor.frontier_belief import (
    DemandPhase,
    DemandScenario,
    DependencyMode,
    FrontierBeliefSnapshot,
)
from beliefkv.predictor.hardware_service import (
    GPURequestServiceDemand,
    GPUServiceCurveModel,
    GPUServiceEstimate,
    GPUServiceFeatures,
)


@dataclass(frozen=True)
class PhysicalizedInvocationDemand:
    """Candidate-specific demand after Radix residency has been resolved."""

    invocation_id: str
    uncached_prefill_tokens: int
    remaining_decode_tokens: int
    current_sequence_tokens: int
    cache_hit_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.invocation_id:
            raise ValueError("physicalized invocation identity is required")
        if min(
            self.uncached_prefill_tokens,
            self.remaining_decode_tokens,
            self.current_sequence_tokens,
            self.cache_hit_tokens,
        ) < 0:
            raise ValueError("physicalized token demand must be non-negative")


@dataclass(frozen=True)
class ScheduledRequestQuantum:
    invocation_id: str
    token_delta: int
    sequence_tokens_before: int
    cache_hit_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not self.invocation_id or min(
            self.token_delta, self.sequence_tokens_before
        ) < 0:
            raise ValueError("scheduled request quantum is invalid")
        if not 0.0 <= self.cache_hit_ratio <= 1.0:
            raise ValueError("scheduled cache-hit ratio must be in [0, 1]")


@dataclass(frozen=True)
class ScheduledBatchQuantum:
    batch_id: str
    phase: DemandPhase
    requests: tuple[ScheduledRequestQuantum, ...]
    chunk_position: str = "unknown"
    prefill_decode_mixed: bool = False
    pcie_contention_state: str = "idle"
    hicache_inflight_bytes: int = 0
    earliest_start_offset_ms: float = 0.0
    ready_after_transfer_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.batch_id or not self.requests:
            raise ValueError("scheduled batch identity/composition is required")
        object.__setattr__(self, "phase", DemandPhase(self.phase))
        request_ids = [item.invocation_id for item in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("one batch cannot contain duplicate invocation IDs")
        if self.phase == DemandPhase.EXTERNAL:
            raise ValueError("external demand cannot be submitted as a GPU batch")
        if self.hicache_inflight_bytes < 0:
            raise ValueError("HiCache contention bytes must be non-negative")
        if (
            not math.isfinite(self.earliest_start_offset_ms)
            or self.earliest_start_offset_ms < 0
        ):
            raise ValueError("batch earliest start must be finite and non-negative")
        transfer_ids = tuple(sorted(set(self.ready_after_transfer_ids)))
        if any(not item for item in transfer_ids):
            raise ValueError("batch transfer dependencies must be non-empty IDs")
        object.__setattr__(self, "ready_after_transfer_ids", transfer_ids)


@dataclass(frozen=True)
class ScheduledTransfer:
    transfer_id: str
    start_offset_ms: float
    completion_offset_ms: float
    pcie_busy_ms: float
    hbm_delta_bytes_on_completion: int = 0
    ready_after_invocation_ids: tuple[str, ...] = ()
    ready_after_dependency_release_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.start_offset_ms,
            self.completion_offset_ms,
            self.pcie_busy_ms,
        )
        if not self.transfer_id or any(
            not math.isfinite(item) or item < 0 for item in values
        ):
            raise ValueError("scheduled transfer is invalid")
        if self.completion_offset_ms < self.start_offset_ms:
            raise ValueError("transfer cannot complete before it starts")
        for name in (
            "ready_after_invocation_ids",
            "ready_after_dependency_release_ids",
        ):
            values = tuple(sorted(set(getattr(self, name))))
            if any(not item for item in values):
                raise ValueError("transfer dependency IDs must be non-empty")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class CandidatePhysicalPlan:
    """Result of candidate JointPlan + physical snapshot materialization."""

    package_id: str
    physical_snapshot_id: str
    physical_snapshot_revision: int
    invocation_demands: tuple[PhysicalizedInvocationDemand, ...]
    batches: tuple[ScheduledBatchQuantum, ...]
    transfers: tuple[ScheduledTransfer, ...] = ()
    hbm_capacity_bytes: int = 0
    initial_hbm_used_bytes: int = 0
    initial_hbm_reserved_bytes: int = 0
    modeled_growth_reservation_bytes: int = 0
    kv_bytes_per_token: int = 0
    residual_hbm_time_byte_ms: float = 0.0
    deterministic_feasible: bool = True
    liveness_path_proven: bool = True

    def __post_init__(self) -> None:
        if not self.package_id or not self.physical_snapshot_id:
            raise ValueError("candidate physical plan identity is required")
        if self.physical_snapshot_revision < 0:
            raise ValueError("physical snapshot revision must be non-negative")
        invocation_ids = [item.invocation_id for item in self.invocation_demands]
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ValueError("physicalized invocation demands must be unique")
        batch_ids = [item.batch_id for item in self.batches]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("scheduled batch IDs must be unique")
        transfer_sequence = [item.transfer_id for item in self.transfers]
        if len(transfer_sequence) != len(set(transfer_sequence)):
            raise ValueError("scheduled transfer IDs must be unique")
        if (
            not math.isfinite(self.residual_hbm_time_byte_ms)
            or self.residual_hbm_time_byte_ms < 0
        ):
            raise ValueError("residual HBM-time must be finite and non-negative")
        hbm_values = (
            self.hbm_capacity_bytes,
            self.initial_hbm_used_bytes,
            self.initial_hbm_reserved_bytes,
            self.modeled_growth_reservation_bytes,
            self.kv_bytes_per_token,
        )
        if min(hbm_values) < 0:
            raise ValueError("candidate HBM ledger values must be non-negative")
        if bool(self.hbm_capacity_bytes) != bool(self.kv_bytes_per_token):
            raise ValueError(
                "candidate HBM ledger requires both capacity and KV bytes/token"
            )
        if self.hbm_capacity_bytes and (
            self.initial_hbm_used_bytes + self.initial_hbm_reserved_bytes
            > self.hbm_capacity_bytes
        ):
            raise ValueError("candidate initial HBM commitment exceeds capacity")
        if self.modeled_growth_reservation_bytes > self.initial_hbm_reserved_bytes:
            raise ValueError(
                "modeled growth reservation cannot exceed total HBM reservation"
            )
        transfer_ids = set(transfer_sequence)
        unknown_transfer_dependencies = sorted(
            {
                transfer_id
                for batch in self.batches
                for transfer_id in batch.ready_after_transfer_ids
                if transfer_id not in transfer_ids
            }
        )
        if unknown_transfer_dependencies:
            raise ValueError(
                "batch references unknown transfer dependencies: "
                f"{unknown_transfer_dependencies}"
            )


class CandidateDemandPhysicalizer(Protocol):
    def physicalize(
        self,
        scenario: DemandScenario,
        *,
        package_id: str,
        physical_snapshot: object,
    ) -> CandidatePhysicalPlan: ...


@dataclass(frozen=True)
class TimedServiceQuantum:
    batch_id: str
    start_offset_ms: float
    completion_offset_ms: float
    service_quantile: float
    service_source: str


@dataclass(frozen=True)
class TimedInvocationOutcome:
    invocation_id: str
    completion_offset_ms: float | None
    completion_source: str


@dataclass(frozen=True)
class TimedScenario:
    scenario_id: str
    package_id: str
    physical_snapshot_id: str
    invocation_outcomes: tuple[TimedInvocationOutcome, ...]
    join_reentry_offsets_ms: Mapping[str, float]
    service_quanta: tuple[TimedServiceQuantum, ...]
    transfer_completion_offsets_ms: Mapping[str, float]
    dependency_release_offsets_ms: Mapping[str, float]
    residual_hbm_time_byte_ms: float
    pcie_busy_ms: float
    future_hbm_peak_bytes: int
    future_hbm_overflow_bytes: int
    first_hbm_pressure_offset_ms: float | None
    first_hbm_pressure_deficit_bytes: int
    future_hbm_feasible: bool
    deterministic_feasible: bool
    future_feasible: bool
    liveness_path_proven: bool
    failure_reasons: tuple[str, ...] = ()


class CandidateTimelineEvaluator:
    """Evaluate demand only after candidate-specific physicalization.

    The evaluator serializes GPU batch quanta on the single device, while each
    batch duration still reflects its candidate-selected composition. RCCG JOIN
    and producer dependencies are resolved only after member completion times
    exist.
    """

    def __init__(
        self,
        service_model: GPUServiceCurveModel,
        *,
        service_quantile: float = 0.9,
        service_cache_entries: int = 4_096,
    ) -> None:
        if not 0.5 <= service_quantile <= 0.99:
            raise ValueError("service quantile must be in [0.5, 0.99]")
        self.service_model = service_model
        self.service_quantile = service_quantile
        self.service_cache_entries = max(0, service_cache_entries)
        self._service_cache: OrderedDict[
            GPUServiceFeatures, GPUServiceEstimate
        ] = OrderedDict()
        self.service_cache_hits = 0
        self.service_cache_misses = 0

    def service_cache_stats(self) -> tuple[int, int, int]:
        return (
            self.service_cache_hits,
            self.service_cache_misses,
            len(self._service_cache),
        )

    def _predict_service(
        self,
        features: GPUServiceFeatures,
    ) -> GPUServiceEstimate:
        if self.service_cache_entries:
            cached = self._service_cache.get(features)
            if cached is not None:
                self._service_cache.move_to_end(features)
                self.service_cache_hits += 1
                return cached
        self.service_cache_misses += 1
        estimate = self.service_model.predict(features)
        if self.service_cache_entries:
            self._service_cache[features] = estimate
            self._service_cache.move_to_end(features)
            while len(self._service_cache) > self.service_cache_entries:
                self._service_cache.popitem(last=False)
        return estimate

    def evaluate(
        self,
        scenario: DemandScenario,
        plan: CandidatePhysicalPlan,
    ) -> TimedScenario:
        outcomes = {item.invocation_id: item for item in scenario.outcomes}
        physical = {item.invocation_id: item for item in plan.invocation_demands}
        unknown = set(physical).difference(outcomes)
        if unknown:
            raise ValueError(
                f"physical plan references invocations outside demand scenario: {sorted(unknown)}"
            )
        consumed_prefill = {item: 0 for item in physical}
        consumed_decode = {item: 0 for item in physical}
        completion: dict[str, float | None] = {
            invocation_id: None for invocation_id in outcomes
        }
        dependency_release: dict[str, float | None] = {
            invocation_id: None for invocation_id in outcomes
        }
        completion_source: dict[str, str] = {
            invocation_id: "unresolved" for invocation_id in outcomes
        }
        for invocation_id, outcome in outcomes.items():
            if (
                outcome.external_segments
                and outcome.dependency_mode == DependencyMode.EXTERNAL
            ):
                dependency_release[invocation_id] = sum(
                    item.residual_delay_ms for item in outcome.external_segments
                )
                if invocation_id not in physical:
                    completion[invocation_id] = dependency_release[invocation_id]
                    completion_source[invocation_id] = "external_segment"

        self._settle_dependency_releases(
            outcomes,
            physical,
            completion,
            dependency_release,
            completion_source,
        )

        now_ms = 0.0
        service_quanta: list[TimedServiceQuantum] = []
        failures: list[str] = []
        transfers = {item.transfer_id: item for item in plan.transfers}
        transfer_completion: dict[str, float] = {}
        pcie_cursor_ms = 0.0
        hbm_events: list[tuple[float, int, str]] = []

        def schedule_transfer(transfer_id: str) -> bool:
            nonlocal pcie_cursor_ms
            if transfer_id in transfer_completion:
                return True
            transfer = transfers.get(transfer_id)
            if transfer is None:
                return False
            invocation_releases = [
                completion.get(item)
                for item in transfer.ready_after_invocation_ids
            ]
            dependency_releases = [
                dependency_release.get(item)
                for item in transfer.ready_after_dependency_release_ids
            ]
            if any(item is None for item in (*invocation_releases, *dependency_releases)):
                return False
            duration_ms = transfer.completion_offset_ms - transfer.start_offset_ms
            start_ms = max(
                pcie_cursor_ms,
                transfer.start_offset_ms,
                *(float(item) for item in invocation_releases if item is not None),
                *(float(item) for item in dependency_releases if item is not None),
            )
            transfer_completion[transfer_id] = start_ms + duration_ms
            pcie_cursor_ms = transfer_completion[transfer_id]
            if transfer.hbm_delta_bytes_on_completion:
                hbm_events.append(
                    (
                        transfer_completion[transfer_id],
                        transfer.hbm_delta_bytes_on_completion,
                        f"transfer:{transfer_id}",
                    )
                )
            return True

        # Transfers without causal dependencies are speculative and can start
        # at the beginning of the finite horizon, concurrently with GPU work.
        for transfer in plan.transfers:
            if not (
                transfer.ready_after_invocation_ids
                or transfer.ready_after_dependency_release_ids
            ):
                schedule_transfer(transfer.transfer_id)
        for batch in plan.batches:
            self._settle_dependency_releases(
                outcomes,
                physical,
                completion,
                dependency_release,
                completion_source,
            )
            requests = []
            release_offsets = [batch.earliest_start_offset_ms]
            for request in batch.requests:
                if request.invocation_id not in outcomes:
                    raise ValueError(
                        f"batch {batch.batch_id} references an unknown invocation"
                    )
                outcome = outcomes[request.invocation_id]
                if outcome.dependency_mode != DependencyMode.NONE:
                    release = dependency_release[request.invocation_id]
                    if release is None:
                        failures.append(
                            "dependency_not_released_before_batch:"
                            f"{batch.batch_id}:{request.invocation_id}"
                        )
                        continue
                    release_offsets.append(release)
                requests.append(
                    GPURequestServiceDemand(
                        sequence_tokens=request.sequence_tokens_before,
                        token_delta=request.token_delta,
                        cache_hit_ratio=request.cache_hit_ratio,
                    )
                )
            missing_transfers = [
                item
                for item in batch.ready_after_transfer_ids
                if not schedule_transfer(item)
            ]
            if missing_transfers:
                failures.extend(
                    f"missing_transfer_dependency:{batch.batch_id}:{item}"
                    for item in missing_transfers
                )
                continue
            release_offsets.extend(
                transfer_completion[item] for item in batch.ready_after_transfer_ids
            )
            if len(requests) != len(batch.requests):
                continue
            estimate = self._predict_service(
                GPUServiceFeatures(
                    phase=batch.phase.value,
                    request_demands=tuple(requests),
                    chunk_position=batch.chunk_position,
                    prefill_decode_mixed=batch.prefill_decode_mixed,
                    pcie_contention_state=batch.pcie_contention_state,
                    hicache_inflight_bytes=batch.hicache_inflight_bytes,
                )
            )
            if estimate.source == "unavailable":
                failures.append(f"service_model_unavailable:{batch.batch_id}")
                continue
            elapsed_ms = estimate.quantile(self.service_quantile)
            started_ms = max(now_ms, *release_offsets)
            now_ms = started_ms + elapsed_ms
            service_quanta.append(
                TimedServiceQuantum(
                    batch_id=batch.batch_id,
                    start_offset_ms=started_ms,
                    completion_offset_ms=now_ms,
                    service_quantile=self.service_quantile,
                    service_source=estimate.source,
                )
            )
            if plan.kv_bytes_per_token:
                hbm_events.append(
                    (
                        now_ms,
                        sum(item.token_delta for item in batch.requests)
                        * plan.kv_bytes_per_token,
                        f"batch:{batch.batch_id}",
                    )
                )
            for request in batch.requests:
                target = physical.get(request.invocation_id)
                if target is None:
                    failures.append(
                        f"missing_physicalized_demand:{request.invocation_id}"
                    )
                    continue
                if batch.phase == DemandPhase.PREFILL:
                    consumed_prefill[request.invocation_id] += request.token_delta
                else:
                    consumed_decode[request.invocation_id] += request.token_delta
                if (
                    consumed_prefill[request.invocation_id]
                    >= target.uncached_prefill_tokens
                    and consumed_decode[request.invocation_id]
                    >= target.remaining_decode_tokens
                ):
                    completion[request.invocation_id] = now_ms
                    completion_source[request.invocation_id] = "candidate_gpu_schedule"

        self._settle_dependency_releases(
            outcomes,
            physical,
            completion,
            dependency_release,
            completion_source,
        )
        self._complete_released_nonphysical(
            outcomes,
            physical,
            dependency_release,
            completion,
            completion_source,
        )
        unresolved_required = sorted(
            invocation_id
            for invocation_id, outcome in outcomes.items()
            if completion[invocation_id] is None
            and (
                outcome.dependency_mode != DependencyMode.NONE
                or invocation_id in physical
            )
        )
        failures.extend(f"unresolved:{item}" for item in unresolved_required)
        join_offsets = {
            outcome.join_id: float(dependency_release[outcome.invocation_id])
            for outcome in outcomes.values()
            if outcome.join_id
            and dependency_release[outcome.invocation_id] is not None
        }
        (
            hbm_peak_bytes,
            hbm_overflow_bytes,
            first_hbm_pressure_offset_ms,
            first_hbm_pressure_deficit_bytes,
        ) = self._future_hbm_peak(
            plan,
            hbm_events,
        )
        if hbm_overflow_bytes:
            failures.append(f"future_hbm_overflow:{hbm_overflow_bytes}")
        return TimedScenario(
            scenario_id=scenario.scenario_id,
            package_id=plan.package_id,
            physical_snapshot_id=plan.physical_snapshot_id,
            invocation_outcomes=tuple(
                TimedInvocationOutcome(
                    invocation_id=invocation_id,
                    completion_offset_ms=completion[invocation_id],
                    completion_source=completion_source[invocation_id],
                )
                for invocation_id in sorted(outcomes)
            ),
            join_reentry_offsets_ms=join_offsets,
            service_quanta=tuple(service_quanta),
            transfer_completion_offsets_ms=transfer_completion,
            dependency_release_offsets_ms={
                invocation_id: float(offset)
                for invocation_id, offset in dependency_release.items()
                if offset is not None
            },
            residual_hbm_time_byte_ms=plan.residual_hbm_time_byte_ms,
            pcie_busy_ms=sum(item.pcie_busy_ms for item in plan.transfers),
            future_hbm_peak_bytes=hbm_peak_bytes,
            future_hbm_overflow_bytes=hbm_overflow_bytes,
            first_hbm_pressure_offset_ms=first_hbm_pressure_offset_ms,
            first_hbm_pressure_deficit_bytes=(
                first_hbm_pressure_deficit_bytes
            ),
            future_hbm_feasible=hbm_overflow_bytes == 0,
            deterministic_feasible=plan.deterministic_feasible,
            future_feasible=not failures,
            liveness_path_proven=plan.liveness_path_proven,
            failure_reasons=tuple(sorted(set(failures))),
        )

    @staticmethod
    def _future_hbm_peak(
        plan: CandidatePhysicalPlan,
        events: list[tuple[float, int, str]],
    ) -> tuple[int, int, float | None, int]:
        """Compute a conservative finite-horizon committed-HBM peak."""

        if not plan.hbm_capacity_bytes:
            return 0, 0, None, 0
        committed = plan.initial_hbm_used_bytes + plan.initial_hbm_reserved_bytes
        peak = committed
        cumulative_growth = 0
        transfer_growth = 0
        first_pressure_offset_ms: float | None = None
        first_pressure_deficit_bytes = 0
        for offset, delta_bytes, source in sorted(
            events,
            key=lambda item: (item[0], 0 if item[2].startswith("transfer:") else 1),
        ):
            if source.startswith("batch:"):
                cumulative_growth += delta_bytes
            else:
                transfer_growth += delta_bytes
            current = (
                committed
                + transfer_growth
                + max(
                    0,
                    cumulative_growth - plan.modeled_growth_reservation_bytes,
                )
            )
            peak = max(peak, current)
            if current > plan.hbm_capacity_bytes and first_pressure_offset_ms is None:
                first_pressure_offset_ms = offset
                first_pressure_deficit_bytes = current - plan.hbm_capacity_bytes
        return (
            peak,
            max(0, peak - plan.hbm_capacity_bytes),
            first_pressure_offset_ms,
            first_pressure_deficit_bytes,
        )

    @staticmethod
    def _resolve_dependency_releases(
        outcomes: Mapping[str, object],
        completion: dict[str, float | None],
        dependency_release: dict[str, float | None],
    ) -> None:
        changed = True
        while changed:
            changed = False
            for invocation_id, raw in outcomes.items():
                if dependency_release[invocation_id] is not None:
                    continue
                outcome = raw
                dependency_ids = outcome.dependency_invocation_ids
                values = [
                    completion.get(item)
                    for item in dependency_ids
                    if completion.get(item) is not None
                ]
                resolved: float | None = None
                if outcome.dependency_mode == DependencyMode.JOIN_ALL:
                    if dependency_ids and len(values) == len(dependency_ids):
                        resolved = max(values)
                elif outcome.dependency_mode in {
                    DependencyMode.JOIN_ANY,
                    DependencyMode.PRODUCER,
                }:
                    if values:
                        resolved = min(values)
                if resolved is not None:
                    dependency_release[invocation_id] = resolved
                    changed = True

    @classmethod
    def _settle_dependency_releases(
        cls,
        outcomes: Mapping[str, object],
        physical: Mapping[str, PhysicalizedInvocationDemand],
        completion: dict[str, float | None],
        dependency_release: dict[str, float | None],
        completion_source: dict[str, str],
    ) -> None:
        while True:
            before = (tuple(completion.items()), tuple(dependency_release.items()))
            cls._resolve_dependency_releases(
                outcomes,
                completion,
                dependency_release,
            )
            cls._complete_zero_demand(
                outcomes,
                physical,
                dependency_release,
                completion,
                completion_source,
            )
            after = (tuple(completion.items()), tuple(dependency_release.items()))
            if after == before:
                return

    @staticmethod
    def _complete_zero_demand(
        outcomes: Mapping[str, object],
        physical: Mapping[str, PhysicalizedInvocationDemand],
        dependency_release: Mapping[str, float | None],
        completion: dict[str, float | None],
        completion_source: dict[str, str],
    ) -> None:
        for invocation_id, demand in physical.items():
            if demand.uncached_prefill_tokens or demand.remaining_decode_tokens:
                continue
            release = dependency_release.get(invocation_id)
            if (
                outcomes[invocation_id].dependency_mode != DependencyMode.NONE
                and release is None
            ):
                continue
            completion[invocation_id] = max(0.0, float(release or 0.0))
            completion_source[invocation_id] = "zero_physical_demand"

    @staticmethod
    def _complete_released_nonphysical(
        outcomes: Mapping[str, object],
        physical: Mapping[str, PhysicalizedInvocationDemand],
        dependency_release: Mapping[str, float | None],
        completion: dict[str, float | None],
        completion_source: dict[str, str],
    ) -> None:
        for invocation_id, outcome in outcomes.items():
            if invocation_id in physical or completion[invocation_id] is not None:
                continue
            release = dependency_release[invocation_id]
            if release is None or outcome.dependency_mode == DependencyMode.NONE:
                continue
            completion[invocation_id] = release
            completion_source[invocation_id] = (
                f"dependency:{outcome.dependency_mode.value}"
            )


def evaluate_belief_timelines(
    belief: FrontierBeliefSnapshot,
    *,
    package_id: str,
    physical_snapshot: object,
    physicalizer: CandidateDemandPhysicalizer,
    evaluator: CandidateTimelineEvaluator,
    cancel_check: Callable[[], bool] | None = None,
    conservative: bool = False,
) -> Mapping[str, TimedScenario]:
    """Physicalize and time every scenario for one candidate JointPlan."""

    result = {}
    for source_scenario in belief.scenarios:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("predictive risk evaluation superseded")
        scenario = source_scenario
        if conservative and source_scenario.conservative_outcomes:
            scenario = DemandScenario(
                scenario_id=source_scenario.scenario_id,
                outcomes=source_scenario.conservative_outcomes,
                probability_mass=source_scenario.probability_mass,
                projection=source_scenario.projection,
            )
        plan = physicalizer.physicalize(
            scenario,
            package_id=package_id,
            physical_snapshot=physical_snapshot,
        )
        if plan.package_id != package_id:
            raise ValueError("physicalizer returned a plan for another package")
        result[scenario.scenario_id] = evaluator.evaluate(scenario, plan)
    return result
