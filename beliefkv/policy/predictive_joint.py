from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Mapping

from beliefkv.policy.online_joint import ActionGroup
from beliefkv.policy.predictive_timeline import TimedScenario
from beliefkv.predictor.frontier_belief import FrontierBeliefSnapshot


class PredictiveActionKind(str, Enum):
    OBSERVED_BASELINE = "observed_baseline"
    PREPARE_HOST = "prepare_host"
    PREFETCH_GPU = "prefetch_gpu"
    RECLAIM_AND_PREFETCH = "reclaim_and_prefetch"
    PARTIAL_PREFETCH_GPU = "partial_prefetch_gpu"


@dataclass(frozen=True)
class PredictiveActionPackage:
    package_id: str
    action: PredictiveActionKind
    context_ids: tuple[str, ...] = ()
    source_joint_plan_id: str | None = None
    target_context_id: str | None = None
    victim_context_ids: tuple[str, ...] = ()
    byte_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.package_id:
            raise ValueError("predictive action package ID is required")
        object.__setattr__(self, "action", PredictiveActionKind(self.action))
        contexts = tuple(sorted(set(self.context_ids)))
        if any(not item for item in contexts):
            raise ValueError("predictive package context IDs must be non-empty")
        object.__setattr__(self, "context_ids", contexts)
        if self.action != PredictiveActionKind.OBSERVED_BASELINE and not contexts:
            raise ValueError("predictive transfer package requires a context")
        target = self.target_context_id
        victims = tuple(sorted(set(self.victim_context_ids)))
        if self.action in {
            PredictiveActionKind.PREFETCH_GPU,
            PredictiveActionKind.PARTIAL_PREFETCH_GPU,
        }:
            target = target or contexts[0]
        elif self.action == PredictiveActionKind.PREPARE_HOST:
            victims = victims or contexts
        elif self.action == PredictiveActionKind.RECLAIM_AND_PREFETCH:
            target = target or contexts[0]
            victims = victims or tuple(item for item in contexts if item != target)
            if not victims:
                raise ValueError("joint reclaim/prefetch requires a victim set")
        if target is not None and not target:
            raise ValueError("predictive target context must be non-empty")
        if any(not item for item in victims):
            raise ValueError("predictive victim contexts must be non-empty")
        if target is not None and target in victims:
            raise ValueError("predictive target cannot also be a victim")
        if self.action == PredictiveActionKind.PARTIAL_PREFETCH_GPU:
            if self.byte_budget is None or self.byte_budget <= 0:
                raise ValueError("partial prefetch requires a positive byte budget")
        elif self.byte_budget is not None:
            raise ValueError("byte budget is only valid for partial prefetch")
        object.__setattr__(self, "target_context_id", target)
        object.__setattr__(self, "victim_context_ids", victims)


@dataclass(frozen=True)
class ScenarioCost:
    """One finite-horizon evaluation with non-overlapping accounting terms."""

    action_unlock_delay_ms: float
    workflow_service_lag_ms: float
    residual_hbm_time_byte_ms: float = 0.0
    residual_host_time_byte_ms: float = 0.0
    residual_pcie_time_ms: float = 0.0
    terminal_debt_ms: float = 0.0
    recourse_credit_ms: float = 0.0
    future_hbm_peak_bytes: int = 0
    future_hbm_overflow_bytes: int = 0
    future_hbm_feasible: bool = True
    deterministic_feasible: bool = True
    future_feasible: bool = True
    liveness_path_proven: bool = True

    def __post_init__(self) -> None:
        values = (
            self.action_unlock_delay_ms,
            self.workflow_service_lag_ms,
            self.residual_hbm_time_byte_ms,
            self.residual_host_time_byte_ms,
            self.residual_pcie_time_ms,
            self.terminal_debt_ms,
            self.recourse_credit_ms,
        )
        if any(not math.isfinite(item) or item < 0 for item in values):
            raise ValueError("scenario cost terms must be finite and non-negative")
        if min(self.future_hbm_peak_bytes, self.future_hbm_overflow_bytes) < 0:
            raise ValueError("future HBM metrics must be non-negative")

    def loss(
        self,
        *,
        hbm_shadow_price_ms_per_byte_ms: float,
        host_shadow_price_ms_per_byte_ms: float,
        pcie_shadow_price: float,
    ) -> float:
        return (
            self.action_unlock_delay_ms
            + self.workflow_service_lag_ms
            + self.terminal_debt_ms
            + self.residual_hbm_time_byte_ms
            * hbm_shadow_price_ms_per_byte_ms
            + self.residual_host_time_byte_ms
            * host_shadow_price_ms_per_byte_ms
            + self.residual_pcie_time_ms * pcie_shadow_price
            - self.recourse_credit_ms
        )


@dataclass(frozen=True)
class PrepareRecourseDiagnostic:
    scenario_id: str
    probability_mass: float
    shadow_completion_ms: float | None
    first_pressure_ms: float | None
    pressure_deficit_bytes: int
    parent_reentry_ms: float | None
    exclusive_reclaimable_bytes: int
    full_closure_copy_bytes: int
    cross_context_copy_bytes: int
    baseline_reactive_d2h_ms: float | None
    proactive_interference_ms: float
    transfer_duration_source: str
    transfer_service_epoch: str
    interference_source: str
    interference_service_epoch: str
    interference_to_transfer_ratio: float
    transfer_nearest_bucket_distance: int | None
    transfer_sample_count: int
    transfer_size_coverage_bytes: tuple[int, int] | None
    transfer_extent_count_coverage: tuple[int, int] | None
    transfer_shape_bucket_distance: int | None
    transfer_shape_supported: bool
    predicted_extent_count: int
    shape_fingerprint: str
    byte_only_transfer_ms: float
    shape_aware_transfer_p90_ms: float
    shape_aware_stall_p90_ms: float
    morphology_deadline_ms: float | None
    morphology_slack_ms: float | None
    conservative_morphology_deadline_ms: float | None
    conservative_morphology_slack_ms: float | None
    morphology_debt_ms: float
    morphology_penalty_ms: float
    reactive_victim_model: str
    recourse_credit_ms: float
    recourse_failure_reason: str

    def __post_init__(self) -> None:
        required = (
            self.scenario_id,
            self.recourse_failure_reason,
            self.transfer_duration_source,
            self.transfer_service_epoch,
            self.interference_source,
            self.interference_service_epoch,
            self.reactive_victim_model,
            self.shape_fingerprint,
        )
        if any(not item for item in required):
            raise ValueError("prepare recourse diagnostic identity is required")
        if not 0.0 <= self.probability_mass <= 1.0:
            raise ValueError("recourse probability must be in [0, 1]")
        optional_times = (
            self.shadow_completion_ms,
            self.first_pressure_ms,
            self.parent_reentry_ms,
            self.baseline_reactive_d2h_ms,
        )
        if any(
            item is not None and (not math.isfinite(item) or item < 0)
            for item in optional_times
        ):
            raise ValueError("recourse timestamps must be finite and non-negative")
        values = (
            self.pressure_deficit_bytes,
            self.exclusive_reclaimable_bytes,
            self.full_closure_copy_bytes,
            self.cross_context_copy_bytes,
            self.proactive_interference_ms,
            self.interference_to_transfer_ratio,
            self.transfer_sample_count,
            self.recourse_credit_ms,
            self.predicted_extent_count,
            self.byte_only_transfer_ms,
            self.shape_aware_transfer_p90_ms,
            self.shape_aware_stall_p90_ms,
        )
        if any(not math.isfinite(float(item)) or item < 0 for item in values):
            raise ValueError("recourse values must be finite and non-negative")
        if (
            self.transfer_nearest_bucket_distance is not None
            and self.transfer_nearest_bucket_distance < 0
        ):
            raise ValueError("transfer bucket distance must be non-negative")
        if self.transfer_size_coverage_bytes is not None:
            low, high = self.transfer_size_coverage_bytes
            if low < 0 or high < low:
                raise ValueError("transfer size coverage is invalid")
        if self.transfer_extent_count_coverage is not None:
            low, high = self.transfer_extent_count_coverage
            if low < 0 or high < low:
                raise ValueError("transfer extent-count coverage is invalid")
        if (
            self.transfer_shape_bucket_distance is not None
            and self.transfer_shape_bucket_distance < 0
        ):
            raise ValueError("transfer shape-bucket distance must be non-negative")
        for item in (
            self.morphology_deadline_ms,
            self.morphology_slack_ms,
            self.conservative_morphology_deadline_ms,
            self.conservative_morphology_slack_ms,
        ):
            if item is not None and not math.isfinite(item):
                raise ValueError("morphology timing must be finite")
        if self.morphology_debt_ms < 0 or not math.isfinite(
            self.morphology_debt_ms
        ):
            raise ValueError("morphology debt must be finite and non-negative")
        if not math.isfinite(self.morphology_penalty_ms):
            raise ValueError("morphology penalty must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PackageScenarioEvaluation:
    package: PredictiveActionPackage
    costs_by_scenario: Mapping[str, ScenarioCost]
    other_cost: ScenarioCost
    recourse_diagnostics_by_scenario: Mapping[
        str, PrepareRecourseDiagnostic
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not scenario_id for scenario_id in self.costs_by_scenario):
            raise ValueError("scenario evaluation IDs must be non-empty")
        if not set(self.recourse_diagnostics_by_scenario).issubset(
            self.costs_by_scenario
        ):
            raise ValueError("recourse diagnostics reference unknown scenarios")

    @classmethod
    def from_timed_scenarios(
        cls,
        package: PredictiveActionPackage,
        timelines: Mapping[str, TimedScenario],
        *,
        unlock_invocation_ids: tuple[str, ...],
        service_lag_invocation_ids: tuple[str, ...] = (),
        other_cost: ScenarioCost,
    ) -> "PackageScenarioEvaluation":
        if not unlock_invocation_ids:
            raise ValueError("timed evaluation requires an action-unlock target")
        return cls(
            package=package,
            costs_by_scenario={
                scenario_id: _scenario_cost_from_timeline(
                    timeline,
                    unlock_invocation_ids=unlock_invocation_ids,
                    service_lag_invocation_ids=service_lag_invocation_ids,
                )
                for scenario_id, timeline in timelines.items()
            },
            other_cost=other_cost,
        )


@dataclass(frozen=True)
class ScenarioRiskPlannerConfig:
    benefit_margin_ms: float = 0.0
    cvar_alpha: float = 0.9
    risk_budget_ms: float = 10.0
    minimum_future_feasibility_probability: float = 0.95
    hbm_shadow_price_ms_per_byte_ms: float = 0.0
    host_shadow_price_ms_per_byte_ms: float = 0.0
    pcie_shadow_price: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.benefit_margin_ms)
            or not math.isfinite(self.risk_budget_ms)
            or min(self.benefit_margin_ms, self.risk_budget_ms) < 0
        ):
            raise ValueError("risk planner margins must be finite and non-negative")
        if not 0 < self.cvar_alpha < 1:
            raise ValueError("CVaR alpha must be in (0, 1)")
        if not 0 <= self.minimum_future_feasibility_probability <= 1:
            raise ValueError("future feasibility probability must be in [0, 1]")
        if min(
            self.hbm_shadow_price_ms_per_byte_ms,
            self.host_shadow_price_ms_per_byte_ms,
            self.pcie_shadow_price,
        ) < 0:
            raise ValueError("resource shadow prices must be non-negative")


@dataclass(frozen=True)
class PackageRiskSummary:
    package_id: str
    expected_benefit_ms: float
    expected_recourse_credit_ms: float
    cvar_regret_ms: float
    future_feasibility_probability: float
    future_hbm_feasibility_probability: float
    worst_future_hbm_peak_bytes: int
    worst_future_hbm_overflow_bytes: int
    eligible: bool
    reasons: tuple[str, ...]
    recourse_diagnostics: tuple[PrepareRecourseDiagnostic, ...] = ()


@dataclass(frozen=True)
class ScenarioRiskDecision:
    selected_package_id: str
    baseline_package_id: str
    summaries: tuple[PackageRiskSummary, ...]


@dataclass(frozen=True)
class RiskPlanningTelemetry:
    candidate_generation_ms: float
    scenario_evaluation_ms: tuple[float, ...]
    full_plan_ms: float
    publish_age_ms: float = 0.0
    safe_point_validation_ms: float = 0.0
    rollout_cache_hits: int = 0
    rollout_cache_misses: int = 0

    def __post_init__(self) -> None:
        timings = (
            self.candidate_generation_ms,
            self.full_plan_ms,
            self.publish_age_ms,
            self.safe_point_validation_ms,
            *self.scenario_evaluation_ms,
        )
        if any(not math.isfinite(item) or item < 0 for item in timings):
            raise ValueError("risk planning timings must be finite and non-negative")
        if min(self.rollout_cache_hits, self.rollout_cache_misses) < 0:
            raise ValueError("rollout cache counts must be non-negative")


@dataclass(frozen=True)
class PredictivePlanEnvelope:
    """Atomic P6 publication; actions remain independently grouped."""

    envelope_id: str
    belief: FrontierBeliefSnapshot
    source_joint_plan_id: str
    selected_package_id: str
    action_groups: tuple[ActionGroup, ...]
    generated_ts_ms: float
    telemetry: RiskPlanningTelemetry

    def __post_init__(self) -> None:
        if (
            not self.envelope_id
            or not self.source_joint_plan_id
            or not self.selected_package_id
            or self.generated_ts_ms < 0
        ):
            raise ValueError("predictive plan envelope identity/time is invalid")
        group_ids = [item.group_id for item in self.action_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("predictive action group IDs must be unique")
        expected_evidence = {
            ("belief_id", self.belief.belief_id),
            ("model_version", self.belief.evidence_read_set.model_version),
        }
        for group in self.action_groups:
            if not expected_evidence.issubset(set(group.evidence_read_set)):
                raise ValueError(
                    "predictive action group does not reference envelope belief/model"
                )


class ScenarioRiskPlanner:
    """Select a bounded P6 package without changing the P5 execution path."""

    def __init__(self, config: ScenarioRiskPlannerConfig | None = None) -> None:
        self.config = config or ScenarioRiskPlannerConfig()

    def select(
        self,
        belief: FrontierBeliefSnapshot,
        baseline: PackageScenarioEvaluation,
        candidates: tuple[PackageScenarioEvaluation, ...],
    ) -> ScenarioRiskDecision:
        if baseline.package.action != PredictiveActionKind.OBSERVED_BASELINE:
            raise ValueError("risk planning requires the observed A0 baseline")
        expected_ids = {item.scenario_id for item in belief.scenarios}
        self._validate_scenario_coverage(baseline, expected_ids)
        summaries: list[PackageRiskSummary] = []
        best_package_id = baseline.package.package_id
        best_benefit = 0.0
        for candidate in candidates:
            if candidate.package.action == PredictiveActionKind.OBSERVED_BASELINE:
                raise ValueError("candidate cannot duplicate observed baseline")
            self._validate_scenario_coverage(candidate, expected_ids)
            summary = self._summarize(belief, baseline, candidate)
            summaries.append(summary)
            if summary.eligible and summary.expected_benefit_ms > best_benefit:
                best_benefit = summary.expected_benefit_ms
                best_package_id = candidate.package.package_id
        return ScenarioRiskDecision(
            selected_package_id=best_package_id,
            baseline_package_id=baseline.package.package_id,
            summaries=tuple(summaries),
        )

    @staticmethod
    def _validate_scenario_coverage(
        evaluation: PackageScenarioEvaluation,
        expected_ids: set[str],
    ) -> None:
        if set(evaluation.costs_by_scenario) != expected_ids:
            raise ValueError("package evaluation must cover every global scenario")

    def _summarize(
        self,
        belief: FrontierBeliefSnapshot,
        baseline: PackageScenarioEvaluation,
        candidate: PackageScenarioEvaluation,
    ) -> PackageRiskSummary:
        weighted_regrets: list[tuple[float, float]] = []
        expected_benefit = 0.0
        expected_recourse_credit = 0.0
        feasible_probability = 0.0
        hbm_feasible_probability = 0.0
        worst_hbm_peak = 0
        worst_hbm_overflow = 0
        reasons: list[str] = []
        all_deterministic = True
        all_liveness = True
        for scenario in belief.scenarios:
            base_cost = baseline.costs_by_scenario[scenario.scenario_id]
            candidate_cost = candidate.costs_by_scenario[scenario.scenario_id]
            base_loss = self._loss(base_cost)
            candidate_loss = self._loss(candidate_cost)
            probability = scenario.probability_mass
            expected_benefit += probability * (base_loss - candidate_loss)
            expected_recourse_credit += (
                probability * candidate_cost.recourse_credit_ms
            )
            weighted_regrets.append((max(0.0, candidate_loss - base_loss), probability))
            all_deterministic &= candidate_cost.deterministic_feasible
            all_liveness &= candidate_cost.liveness_path_proven
            if candidate_cost.future_feasible:
                feasible_probability += probability
            if candidate_cost.future_hbm_feasible:
                hbm_feasible_probability += probability
            worst_hbm_peak = max(
                worst_hbm_peak, candidate_cost.future_hbm_peak_bytes
            )
            worst_hbm_overflow = max(
                worst_hbm_overflow, candidate_cost.future_hbm_overflow_bytes
            )

        other_probability = belief.other_probability_mass
        base_other_loss = self._loss(baseline.other_cost)
        candidate_other_loss = self._loss(candidate.other_cost)
        expected_benefit += other_probability * (
            base_other_loss - candidate_other_loss
        )
        expected_recourse_credit += (
            other_probability * candidate.other_cost.recourse_credit_ms
        )
        weighted_regrets.append(
            (max(0.0, candidate_other_loss - base_other_loss), other_probability)
        )
        all_deterministic &= candidate.other_cost.deterministic_feasible
        all_liveness &= candidate.other_cost.liveness_path_proven
        if candidate.other_cost.future_feasible:
            feasible_probability += other_probability
        if candidate.other_cost.future_hbm_feasible:
            hbm_feasible_probability += other_probability
        worst_hbm_peak = max(
            worst_hbm_peak, candidate.other_cost.future_hbm_peak_bytes
        )
        worst_hbm_overflow = max(
            worst_hbm_overflow, candidate.other_cost.future_hbm_overflow_bytes
        )

        if not all_deterministic:
            reasons.append("deterministic_hard_constraint")
        if not all_liveness:
            reasons.append("restore_liveness_path_unproven")
        prepare_diagnostics = tuple(
            candidate.recourse_diagnostics_by_scenario.values()
        )
        if (
            candidate.package.action == PredictiveActionKind.PREPARE_HOST
            and prepare_diagnostics
        ):
            probability_by_scenario = {
                scenario.scenario_id: scenario.probability_mass
                for scenario in belief.scenarios
            }
            expected_stall = sum(
                probability_by_scenario.get(diagnostic.scenario_id, 0.0)
                * diagnostic.shape_aware_stall_p90_ms
                for diagnostic in prepare_diagnostics
            )
            pressure_diagnostics = tuple(
                diagnostic
                for diagnostic in prepare_diagnostics
                if diagnostic.first_pressure_ms is not None
            )
            if not all(
                diagnostic.transfer_shape_supported
                for diagnostic in prepare_diagnostics
            ):
                reasons.append("shape_unsupported")
            if pressure_diagnostics and not any(
                diagnostic.morphology_slack_ms is not None
                and diagnostic.morphology_slack_ms > 0
                for diagnostic in pressure_diagnostics
            ):
                reasons.append("morphology_window_miss")
            if expected_recourse_credit <= expected_stall:
                reasons.append("insufficient_recourse_after_stall")
        if (
            candidate.package.action != PredictiveActionKind.PREPARE_HOST
            and belief.other_probability_mass > 0
            and not belief.other_policy.finite_risk_bound
        ):
            reasons.append("other_has_no_finite_risk_bound")
        if expected_benefit <= self.config.benefit_margin_ms:
            reasons.append("insufficient_expected_benefit")
        cvar = self._weighted_cvar(weighted_regrets, self.config.cvar_alpha)
        if cvar >= self.config.risk_budget_ms:
            reasons.append("cvar_risk_budget")
        if feasible_probability < self.config.minimum_future_feasibility_probability:
            reasons.append("future_chance_constraint")
        # PREPARE_HOST only creates a CPU shadow and retains the GPU copy. A
        # baseline future-HBM overflow is therefore diagnostic for recourse,
        # not a safety failure of the non-destructive prepare itself. Actions
        # that add or rearrange GPU residency must still satisfy this gate.
        if (
            candidate.package.action != PredictiveActionKind.PREPARE_HOST
            and hbm_feasible_probability
            < self.config.minimum_future_feasibility_probability
        ):
            reasons.append("future_hbm_chance_constraint")
        probability_by_scenario = {
            scenario.scenario_id: scenario.probability_mass
            for scenario in belief.scenarios
        }
        recourse_diagnostics = tuple(
            replace(
                diagnostic,
                probability_mass=probability_by_scenario.get(scenario_id, 0.0),
            )
            for scenario_id, diagnostic in sorted(
                candidate.recourse_diagnostics_by_scenario.items()
            )
        )
        return PackageRiskSummary(
            package_id=candidate.package.package_id,
            expected_benefit_ms=expected_benefit,
            expected_recourse_credit_ms=expected_recourse_credit,
            cvar_regret_ms=cvar,
            future_feasibility_probability=feasible_probability,
            future_hbm_feasibility_probability=hbm_feasible_probability,
            worst_future_hbm_peak_bytes=worst_hbm_peak,
            worst_future_hbm_overflow_bytes=worst_hbm_overflow,
            eligible=not reasons,
            reasons=tuple(reasons),
            recourse_diagnostics=recourse_diagnostics,
        )

    def _loss(self, cost: ScenarioCost) -> float:
        return cost.loss(
            hbm_shadow_price_ms_per_byte_ms=(
                self.config.hbm_shadow_price_ms_per_byte_ms
            ),
            host_shadow_price_ms_per_byte_ms=(
                self.config.host_shadow_price_ms_per_byte_ms
            ),
            pcie_shadow_price=self.config.pcie_shadow_price,
        )

    @staticmethod
    def _weighted_cvar(
        weighted_values: list[tuple[float, float]], alpha: float
    ) -> float:
        tail_mass = 1.0 - alpha
        if tail_mass <= 0:
            return max((value for value, _ in weighted_values), default=0.0)
        remaining = tail_mass
        tail_sum = 0.0
        for value, probability in sorted(weighted_values, reverse=True):
            if probability <= 0:
                continue
            consumed = min(remaining, probability)
            tail_sum += value * consumed
            remaining -= consumed
            if remaining <= 1e-12:
                break
        return tail_sum / max(tail_mass - remaining, 1e-12)


def _scenario_cost_from_timeline(
    timeline: TimedScenario,
    *,
    unlock_invocation_ids: tuple[str, ...],
    service_lag_invocation_ids: tuple[str, ...],
) -> ScenarioCost:
    completion = {
        item.invocation_id: item.completion_offset_ms
        for item in timeline.invocation_outcomes
    }
    unlock_values = [completion.get(item) for item in unlock_invocation_ids]
    unlock_resolved = all(item is not None for item in unlock_values)
    lag_values = [
        completion.get(item)
        for item in service_lag_invocation_ids
        if completion.get(item) is not None
    ]
    return ScenarioCost(
        action_unlock_delay_ms=(
            max(float(item) for item in unlock_values if item is not None)
            if unlock_resolved
            else 0.0
        ),
        workflow_service_lag_ms=(
            max(float(item) for item in lag_values) if lag_values else 0.0
        ),
        residual_hbm_time_byte_ms=timeline.residual_hbm_time_byte_ms,
        residual_pcie_time_ms=timeline.pcie_busy_ms,
        future_hbm_peak_bytes=timeline.future_hbm_peak_bytes,
        future_hbm_overflow_bytes=timeline.future_hbm_overflow_bytes,
        future_hbm_feasible=timeline.future_hbm_feasible,
        deterministic_feasible=(
            timeline.deterministic_feasible and unlock_resolved
        ),
        future_feasible=timeline.future_feasible and unlock_resolved,
        liveness_path_proven=timeline.liveness_path_proven,
    )
