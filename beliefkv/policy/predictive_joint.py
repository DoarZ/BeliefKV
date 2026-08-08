from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class PredictiveActionPackage:
    package_id: str
    action: PredictiveActionKind
    context_ids: tuple[str, ...] = ()
    source_joint_plan_id: str | None = None

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


@dataclass(frozen=True)
class ScenarioCost:
    """One finite-horizon evaluation with non-overlapping accounting terms."""

    action_unlock_delay_ms: float
    workflow_service_lag_ms: float
    residual_hbm_time_byte_ms: float = 0.0
    residual_pcie_time_ms: float = 0.0
    terminal_debt_ms: float = 0.0
    deterministic_feasible: bool = True
    future_feasible: bool = True
    liveness_path_proven: bool = True

    def __post_init__(self) -> None:
        values = (
            self.action_unlock_delay_ms,
            self.workflow_service_lag_ms,
            self.residual_hbm_time_byte_ms,
            self.residual_pcie_time_ms,
            self.terminal_debt_ms,
        )
        if any(not math.isfinite(item) or item < 0 for item in values):
            raise ValueError("scenario cost terms must be finite and non-negative")

    def loss(
        self,
        *,
        hbm_shadow_price_ms_per_byte_ms: float,
        pcie_shadow_price: float,
    ) -> float:
        return (
            self.action_unlock_delay_ms
            + self.workflow_service_lag_ms
            + self.terminal_debt_ms
            + self.residual_hbm_time_byte_ms
            * hbm_shadow_price_ms_per_byte_ms
            + self.residual_pcie_time_ms * pcie_shadow_price
        )


@dataclass(frozen=True)
class PackageScenarioEvaluation:
    package: PredictiveActionPackage
    costs_by_scenario: Mapping[str, ScenarioCost]
    other_cost: ScenarioCost

    def __post_init__(self) -> None:
        if any(not scenario_id for scenario_id in self.costs_by_scenario):
            raise ValueError("scenario evaluation IDs must be non-empty")

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
            self.pcie_shadow_price,
        ) < 0:
            raise ValueError("resource shadow prices must be non-negative")


@dataclass(frozen=True)
class PackageRiskSummary:
    package_id: str
    expected_benefit_ms: float
    cvar_regret_ms: float
    future_feasibility_probability: float
    eligible: bool
    reasons: tuple[str, ...]


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
        feasible_probability = 0.0
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
            weighted_regrets.append((max(0.0, candidate_loss - base_loss), probability))
            all_deterministic &= candidate_cost.deterministic_feasible
            all_liveness &= candidate_cost.liveness_path_proven
            if candidate_cost.future_feasible:
                feasible_probability += probability

        other_probability = belief.other_probability_mass
        base_other_loss = self._loss(baseline.other_cost)
        candidate_other_loss = self._loss(candidate.other_cost)
        expected_benefit += other_probability * (
            base_other_loss - candidate_other_loss
        )
        weighted_regrets.append(
            (max(0.0, candidate_other_loss - base_other_loss), other_probability)
        )
        all_deterministic &= candidate.other_cost.deterministic_feasible
        all_liveness &= candidate.other_cost.liveness_path_proven
        if candidate.other_cost.future_feasible:
            feasible_probability += other_probability

        if not all_deterministic:
            reasons.append("deterministic_hard_constraint")
        if not all_liveness:
            reasons.append("restore_liveness_path_unproven")
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
        return PackageRiskSummary(
            package_id=candidate.package.package_id,
            expected_benefit_ms=expected_benefit,
            cvar_regret_ms=cvar,
            future_feasibility_probability=feasible_probability,
            eligible=not reasons,
            reasons=tuple(reasons),
        )

    def _loss(self, cost: ScenarioCost) -> float:
        return cost.loss(
            hbm_shadow_price_ms_per_byte_ms=(
                self.config.hbm_shadow_price_ms_per_byte_ms
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
        deterministic_feasible=(
            timeline.deterministic_feasible and unlock_resolved
        ),
        future_feasible=timeline.future_feasible and unlock_resolved,
        liveness_path_proven=timeline.liveness_path_proven,
    )
