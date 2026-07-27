from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from beliefkv.policy.reference import PolicyInput
from beliefkv.policy.scenario_physicalizer import ScenarioDemand
from beliefkv.policy.whatif_packer import FairnessWindow, ScenarioPlan, WhatIfPacker


class OracleArm(str, Enum):
    O0_SEPARATE = "O0"
    O1_AGENT = "O1"
    O2_KV = "O2"
    O3_JOINT = "O3"


@dataclass(frozen=True)
class OracleCost:
    workflow_jct_ms: float
    causal_blocked_ms: float
    unhidden_stall_ms: float
    action_unlock_ms: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.workflow_jct_ms,
            self.causal_blocked_ms,
            self.unhidden_stall_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("oracle costs must be finite and non-negative")
        if self.action_unlock_ms is not None and (
            not math.isfinite(self.action_unlock_ms) or self.action_unlock_ms < 0
        ):
            raise ValueError("action_unlock_ms must be finite and non-negative")

    @property
    def ordering_key(self) -> tuple[float, float, float, float]:
        return (
            self.workflow_jct_ms,
            self.causal_blocked_ms,
            self.unhidden_stall_ms,
            self.action_unlock_ms if self.action_unlock_ms is not None else math.inf,
        )

    def to_dict(self) -> dict[str, float | None]:
        return {
            "workflow_jct_ms": self.workflow_jct_ms,
            "causal_blocked_ms": self.causal_blocked_ms,
            "unhidden_stall_ms": self.unhidden_stall_ms,
            "action_unlock_ms": self.action_unlock_ms,
        }


@dataclass(frozen=True)
class ResimulationEvidence:
    schedule_recomputed: bool
    queue_service_recomputed: bool
    physical_actions_recomputed: bool
    allocator_recomputed: bool
    service_model_calibrated: bool
    semantic_events_frozen: bool
    token_demand_frozen: bool
    tool_duration_frozen: bool
    transition_hash: str
    service_model_id: str = "unknown"
    physical_model_id: str = "unknown"

    def __post_init__(self) -> None:
        if not self.transition_hash:
            raise ValueError("resimulation transition_hash must be non-empty")
        if not self.service_model_id or not self.physical_model_id:
            raise ValueError("resimulation model IDs must be non-empty")

    @property
    def valid_for_timing(self) -> bool:
        return (
            self.schedule_recomputed
            and self.queue_service_recomputed
            and self.physical_actions_recomputed
            and self.allocator_recomputed
            and self.service_model_calibrated
            and self.token_demand_frozen
            and self.tool_duration_frozen
        )


@dataclass(frozen=True)
class ResimulatedPlanEvaluation:
    cost: OracleCost
    evidence: ResimulationEvidence


class PlanCostEvaluator(Protocol):
    def evaluate(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        *,
        trace_sensitivity: str,
    ) -> ResimulatedPlanEvaluation:
        ...


class ArmAwarePlanCostEvaluator(Protocol):
    def evaluate_arm(
        self,
        arm: OracleArm,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        *,
        trace_sensitivity: str,
    ) -> ResimulatedPlanEvaluation:
        ...


@dataclass(frozen=True)
class OracleCandidate:
    demand: ScenarioDemand
    current_separate_plan: ScenarioPlan
    trace_sensitivity: str

    def __post_init__(self) -> None:
        if self.trace_sensitivity not in {
            "schedule_invariant",
            "timing_sensitive",
            "semantic_race_sensitive",
        }:
            raise ValueError("invalid oracle trace sensitivity")
        if (
            self.current_separate_plan.snapshot_id != self.demand.snapshot_id
            or self.current_separate_plan.scenario_id != self.demand.scenario_id
        ):
            raise ValueError("current plan and scenario demand are misaligned")


@dataclass(frozen=True)
class OracleArmResult:
    arm: OracleArm
    scenario_id: str
    plan: ScenarioPlan
    cost: OracleCost
    evidence: ResimulationEvidence
    bound_kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "scenario_id": self.scenario_id,
            "plan": self.plan.to_dict(),
            "cost": self.cost.to_dict(),
            "evidence": {
                name: getattr(self.evidence, name)
                for name in self.evidence.__dataclass_fields__
            },
            "bound_kind": self.bound_kind,
        }


@dataclass(frozen=True)
class JointOracleResult:
    snapshot_id: str
    current_scenario_id: str
    arms: Mapping[OracleArm, OracleArmResult]
    joint_synergy_gap_ms: float
    jointness_supported: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", MappingProxyType(dict(self.arms)))

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "current_scenario_id": self.current_scenario_id,
            "arms": {
                arm.value: result.to_dict()
                for arm, result in sorted(
                    self.arms.items(), key=lambda item: item[0].value
                )
            },
            "joint_synergy_gap_ms": self.joint_synergy_gap_ms,
            "jointness_supported": self.jointness_supported,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TraceJointOracleResult:
    result: JointOracleResult
    baseline_execution_order: tuple[str, ...]
    candidate_order_count: int
    search_complete: bool
    search_kind: str

    def to_dict(self) -> dict[str, object]:
        payload = self.result.to_dict()
        payload["baseline_execution_order"] = list(self.baseline_execution_order)
        payload["candidate_order_count"] = self.candidate_order_count
        payload["search_complete"] = self.search_complete
        payload["search_kind"] = self.search_kind
        return payload


@dataclass(frozen=True)
class BoundedLagOrderCandidate:
    strategy: str
    execution_order: tuple[str, ...]
    max_pre_dispatch_lag_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "execution_order": list(self.execution_order),
            "max_pre_dispatch_lag_ms": self.max_pre_dispatch_lag_ms,
        }


@dataclass(frozen=True)
class BoundedLagOrderSet:
    candidates: tuple[BoundedLagOrderCandidate, ...]
    lag_budget_ms: float
    max_service_quantum_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "search_kind": "bounded_lag_heuristic_topological",
            "complete": False,
            "lag_budget_ms": self.lag_budget_ms,
            "max_service_quantum_ms": self.max_service_quantum_ms,
            "fairness_semantics": (
                "root-workflow attained standalone GPU service; eligibility is "
                "checked before each non-preemptive request dispatch"
            ),
            "candidates": [item.to_dict() for item in self.candidates],
        }


class JointPlanOracle:
    """Construct O0-O3 without reusing original wall-clock timing."""

    def __init__(self, packer: WhatIfPacker | None = None) -> None:
        self.packer = packer or WhatIfPacker()

    def evaluate(
        self,
        policy_input: PolicyInput,
        candidates: Sequence[OracleCandidate],
        *,
        current_scenario_id: str,
        evaluator: PlanCostEvaluator,
        fairness: FairnessWindow | None = None,
    ) -> JointOracleResult:
        if not candidates:
            raise ValueError("joint oracle requires at least one scenario")
        by_scenario = {item.demand.scenario_id: item for item in candidates}
        if len(by_scenario) != len(candidates):
            raise ValueError("oracle scenario IDs must be unique")
        current = by_scenario.get(current_scenario_id)
        if current is None:
            raise ValueError("current scenario is absent from oracle candidates")
        if any(item.demand.snapshot_id != policy_input.snapshot_id for item in candidates):
            raise ValueError("oracle candidates must use the same physical snapshot")

        baseline_results = [
            self._evaluate_plan(
                OracleArm.O1_AGENT,
                policy_input,
                item,
                item.current_separate_plan,
                evaluator,
            )
            for item in candidates
            if item.current_separate_plan.feasible
        ]
        oracle_plans = {
            item.demand.scenario_id: self.packer.pack(
                policy_input,
                item.demand,
                fairness=fairness,
            )
            for item in candidates
        }
        joint_results = [
            self._evaluate_plan(
                OracleArm.O3_JOINT,
                policy_input,
                item,
                oracle_plans[item.demand.scenario_id],
                evaluator,
            )
            for item in candidates
            if oracle_plans[item.demand.scenario_id].feasible
        ]
        if not baseline_results or not joint_results:
            raise ValueError("oracle has no feasible baseline or joint plan")

        o0 = self._evaluate_plan(
            OracleArm.O0_SEPARATE,
            policy_input,
            current,
            current.current_separate_plan,
            evaluator,
        )
        o1 = self._best(baseline_results, OracleArm.O1_AGENT)
        current_oracle_plan = oracle_plans[current_scenario_id]
        if not current_oracle_plan.feasible:
            raise ValueError("current scenario has no feasible oracle KV plan")
        o2 = self._evaluate_plan(
            OracleArm.O2_KV,
            policy_input,
            current,
            current_oracle_plan,
            evaluator,
        )
        o3 = self._best(joint_results, OracleArm.O3_JOINT)
        gap = min(o1.cost.workflow_jct_ms, o2.cost.workflow_jct_ms) - (
            o3.cost.workflow_jct_ms
        )
        supported = gap > 0
        reason = (
            "O3 improves workflow JCT beyond either oracle axis alone"
            if supported
            else "no positive point-estimate joint synergy gap"
        )
        return JointOracleResult(
            snapshot_id=policy_input.snapshot_id,
            current_scenario_id=current_scenario_id,
            arms={
                OracleArm.O0_SEPARATE: o0,
                OracleArm.O1_AGENT: o1,
                OracleArm.O2_KV: o2,
                OracleArm.O3_JOINT: o3,
            },
            joint_synergy_gap_ms=gap,
            jointness_supported=supported,
            reason=reason,
        )

    def _evaluate_plan(
        self,
        arm: OracleArm,
        policy_input: PolicyInput,
        candidate: OracleCandidate,
        plan: ScenarioPlan,
        evaluator: PlanCostEvaluator,
    ) -> OracleArmResult:
        evaluate_arm = getattr(evaluator, "evaluate_arm", None)
        if callable(evaluate_arm):
            result = evaluate_arm(
                arm,
                policy_input,
                candidate.demand,
                plan,
                trace_sensitivity=candidate.trace_sensitivity,
            )
        else:
            result = evaluator.evaluate(
                policy_input,
                candidate.demand,
                plan,
                trace_sensitivity=candidate.trace_sensitivity,
            )
        if not result.evidence.valid_for_timing:
            raise ValueError(
                "oracle evaluator reused fixed timing or physical actions; JCT is invalid"
            )
        if (
            candidate.trace_sensitivity != "semantic_race_sensitive"
            and not result.evidence.semantic_events_frozen
        ):
            raise ValueError("non-racy oracle must preserve semantic events")
        bound_kind = (
            "optimistic"
            if candidate.trace_sensitivity == "semantic_race_sensitive"
            else "exact_fixed_semantics"
        )
        return OracleArmResult(
            arm=arm,
            scenario_id=candidate.demand.scenario_id,
            plan=plan,
            cost=result.cost,
            evidence=result.evidence,
            bound_kind=bound_kind,
        )

    @staticmethod
    def _best(
        candidates: Sequence[OracleArmResult], arm: OracleArm
    ) -> OracleArmResult:
        selected = min(
            candidates,
            key=lambda item: (item.cost.ordering_key, item.scenario_id),
        )
        return OracleArmResult(
            arm=arm,
            scenario_id=selected.scenario_id,
            plan=selected.plan,
            cost=selected.cost,
            evidence=selected.evidence,
            bound_kind=selected.bound_kind,
        )


class TraceOrderJointOracle:
    """Evaluate O0-O3 over complete or explicitly bounded topological orders."""

    def evaluate(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        base_plan: ScenarioPlan,
        *,
        request_predecessors: Mapping[str, Sequence[str]],
        baseline_execution_order: Sequence[str],
        trace_sensitivity: str,
        evaluator: ArmAwarePlanCostEvaluator,
        max_candidate_orders: int = 4_096,
        candidate_execution_orders: Sequence[Sequence[str]] | None = None,
        candidate_search_kind: str = "bounded_supplied_topological",
    ) -> TraceJointOracleResult:
        if max_candidate_orders <= 0:
            raise ValueError("max_candidate_orders must be positive")
        if base_plan.snapshot_id != policy_input.snapshot_id:
            raise ValueError("trace oracle base plan uses a stale snapshot")
        request_ids = set(request_predecessors)
        baseline = tuple(baseline_execution_order)
        if len(baseline) != len(set(baseline)) or set(baseline) != request_ids:
            raise ValueError("baseline order must contain every request exactly once")
        _validate_topological_order(baseline, request_predecessors)
        if candidate_execution_orders is None:
            orders, complete = enumerate_topological_orders(
                request_predecessors,
                max_orders=max_candidate_orders,
            )
            if baseline not in orders:
                if len(orders) >= max_candidate_orders:
                    orders = (*orders[:-1], baseline)
                    complete = False
                else:
                    orders = (*orders, baseline)
            search_kind = (
                "exhaustive_topological" if complete else "bounded_topological"
            )
        else:
            if not candidate_search_kind:
                raise ValueError("candidate_search_kind must be non-empty")
            deduplicated: list[tuple[str, ...]] = []
            seen: set[tuple[str, ...]] = set()
            for raw_order in candidate_execution_orders:
                order = tuple(raw_order)
                if len(order) != len(set(order)) or set(order) != request_ids:
                    raise ValueError(
                        "candidate order must contain every request exactly once"
                    )
                _validate_topological_order(order, request_predecessors)
                if order not in seen:
                    seen.add(order)
                    deduplicated.append(order)
            if not deduplicated:
                raise ValueError("candidate execution orders must be non-empty")
            orders = tuple(deduplicated[:max_candidate_orders])
            complete = False
            search_kind = candidate_search_kind
        plan_orders = tuple(dict.fromkeys((*orders, baseline)))
        plans = {
            order: replace(
                base_plan,
                execution_order=order,
                admission_actions={request_id: "admit" for request_id in order},
            )
            for order in plan_orders
        }

        o0 = self._evaluate(
            OracleArm.O0_SEPARATE,
            policy_input,
            demand,
            plans[baseline],
            evaluator,
            trace_sensitivity,
        )
        o2 = self._evaluate(
            OracleArm.O2_KV,
            policy_input,
            demand,
            plans[baseline],
            evaluator,
            trace_sensitivity,
        )
        o1_candidates = tuple(
            self._evaluate(
                OracleArm.O1_AGENT,
                policy_input,
                demand,
                plans[order],
                evaluator,
                trace_sensitivity,
            )
            for order in orders
        )
        o3_candidates = tuple(
            self._evaluate(
                OracleArm.O3_JOINT,
                policy_input,
                demand,
                plans[order],
                evaluator,
                trace_sensitivity,
            )
            for order in orders
        )
        o1 = min(
            o1_candidates,
            key=lambda item: (item.cost.ordering_key, item.plan.execution_order),
        )
        o3 = min(
            o3_candidates,
            key=lambda item: (item.cost.ordering_key, item.plan.execution_order),
        )
        # Re-evaluate selected arms so arm-aware evaluators retain their selected
        # timelines rather than the last search candidate.
        o1 = self._evaluate(
            OracleArm.O1_AGENT,
            policy_input,
            demand,
            o1.plan,
            evaluator,
            trace_sensitivity,
        )
        o3 = self._evaluate(
            OracleArm.O3_JOINT,
            policy_input,
            demand,
            o3.plan,
            evaluator,
            trace_sensitivity,
        )
        gap = min(o1.cost.workflow_jct_ms, o2.cost.workflow_jct_ms) - (
            o3.cost.workflow_jct_ms
        )
        supported = gap > 0
        if not complete:
            reason = (
                "positive bounded-search point estimate; exhaustive jointness is unproven"
                if supported
                else "bounded search found no positive joint synergy gap"
            )
        else:
            reason = (
                "O3 improves workflow JCT beyond either exhaustive oracle axis alone"
                if supported
                else "exhaustive topological search found no positive joint synergy gap"
            )
        result = JointOracleResult(
            snapshot_id=policy_input.snapshot_id,
            current_scenario_id=demand.scenario_id,
            arms={
                OracleArm.O0_SEPARATE: o0,
                OracleArm.O1_AGENT: o1,
                OracleArm.O2_KV: o2,
                OracleArm.O3_JOINT: o3,
            },
            joint_synergy_gap_ms=gap,
            jointness_supported=supported and complete,
            reason=reason,
        )
        return TraceJointOracleResult(
            result=result,
            baseline_execution_order=baseline,
            candidate_order_count=len(orders),
            search_complete=complete,
            search_kind=search_kind,
        )

    @staticmethod
    def _evaluate(
        arm: OracleArm,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        evaluator: ArmAwarePlanCostEvaluator,
        trace_sensitivity: str,
    ) -> OracleArmResult:
        evaluation = evaluator.evaluate_arm(
            arm,
            policy_input,
            demand,
            plan,
            trace_sensitivity=trace_sensitivity,
        )
        if not evaluation.evidence.valid_for_timing:
            raise ValueError("trace oracle evaluator did not recompute physical timing")
        return OracleArmResult(
            arm=arm,
            scenario_id=demand.scenario_id,
            plan=plan,
            cost=evaluation.cost,
            evidence=evaluation.evidence,
            bound_kind=(
                "optimistic"
                if trace_sensitivity == "semantic_race_sensitive"
                else "exact_fixed_semantics"
            ),
        )


def enumerate_topological_orders(
    request_predecessors: Mapping[str, Sequence[str]],
    *,
    max_orders: int,
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    if max_orders <= 0:
        raise ValueError("max_orders must be positive")
    request_ids = set(request_predecessors)
    normalized = {
        request_id: tuple(sorted(set(predecessors)))
        for request_id, predecessors in request_predecessors.items()
    }
    for request_id, predecessors in normalized.items():
        unknown = set(predecessors) - request_ids
        if unknown or request_id in predecessors:
            raise ValueError(
                f"invalid predecessors for {request_id}: {sorted(unknown)}"
            )
    successors: dict[str, list[str]] = {request_id: [] for request_id in request_ids}
    indegree = {request_id: len(predecessors) for request_id, predecessors in normalized.items()}
    for request_id, predecessors in normalized.items():
        for predecessor in predecessors:
            successors[predecessor].append(request_id)
    for values in successors.values():
        values.sort()
    results: list[tuple[str, ...]] = []
    truncated = False

    def visit(
        order: tuple[str, ...],
        available: tuple[str, ...],
        current_indegree: Mapping[str, int],
    ) -> None:
        nonlocal truncated
        if len(results) >= max_orders:
            truncated = True
            return
        if len(order) == len(request_ids):
            results.append(order)
            return
        if not available:
            return
        for selected in available:
            if len(results) >= max_orders:
                truncated = True
                return
            updated = dict(current_indegree)
            next_available = set(available)
            next_available.remove(selected)
            for successor in successors[selected]:
                updated[successor] -= 1
                if updated[successor] == 0:
                    next_available.add(successor)
            visit(
                (*order, selected),
                tuple(sorted(next_available)),
                updated,
            )

    initial = tuple(sorted(item for item, degree in indegree.items() if degree == 0))
    visit((), initial, indegree)
    if not results:
        raise ValueError("request dependency graph is cyclic")
    return tuple(results), not truncated


def generate_bounded_lag_topological_orders(
    request_predecessors: Mapping[str, Sequence[str]],
    *,
    workflow_by_request: Mapping[str, str],
    service_ms_by_request: Mapping[str, float],
    baseline_execution_order: Sequence[str],
    kv_tokens_by_request: Mapping[str, int] | None = None,
    lag_budget_ms: float = 50.0,
    max_orders: int = 6,
) -> BoundedLagOrderSet:
    """Build diverse topological orders under a root-workflow service-lag gate."""
    if not math.isfinite(lag_budget_ms) or lag_budget_ms < 0:
        raise ValueError("lag_budget_ms must be finite and non-negative")
    if max_orders <= 0:
        raise ValueError("max_orders must be positive")
    request_ids = set(request_predecessors)
    if set(workflow_by_request) != request_ids:
        raise ValueError("workflow mapping must cover every request exactly")
    if set(service_ms_by_request) != request_ids:
        raise ValueError("service mapping must cover every request exactly")
    service = {key: float(value) for key, value in service_ms_by_request.items()}
    if any(not math.isfinite(value) or value < 0 for value in service.values()):
        raise ValueError("request service must be finite and non-negative")
    if any(not workflow_by_request[item] for item in request_ids):
        raise ValueError("workflow IDs must be non-empty")
    baseline = tuple(baseline_execution_order)
    if len(baseline) != len(set(baseline)) or set(baseline) != request_ids:
        raise ValueError("baseline order must contain every request exactly once")
    _validate_topological_order(baseline, request_predecessors)
    normalized = {
        request_id: tuple(sorted(set(predecessors)))
        for request_id, predecessors in request_predecessors.items()
    }
    successors = _successors(normalized)
    critical = _critical_remaining_service(baseline, successors, service)
    baseline_rank = {request_id: index for index, request_id in enumerate(baseline)}
    kv_tokens = {
        request_id: int((kv_tokens_by_request or {}).get(request_id, 0))
        for request_id in request_ids
    }
    if any(value < 0 for value in kv_tokens.values()):
        raise ValueError("KV token demand must be non-negative")
    strategies: tuple[
        tuple[str, Callable[[str], tuple[float, ...]]], ...
    ] = (
        ("observed", lambda item: (float(baseline_rank[item]),)),
        (
            "shortest_service",
            lambda item: (service[item], float(baseline_rank[item])),
        ),
        (
            "longest_service",
            lambda item: (-service[item], float(baseline_rank[item])),
        ),
        (
            "largest_kv",
            lambda item: (-float(kv_tokens[item]), float(baseline_rank[item])),
        ),
        (
            "critical_path",
            lambda item: (-critical[item], float(baseline_rank[item])),
        ),
        ("reverse_observed", lambda item: (-float(baseline_rank[item]),)),
    )
    candidates: list[BoundedLagOrderCandidate] = []
    seen_orders: set[tuple[str, ...]] = set()
    for strategy, strategy_key in strategies:
        order, max_lag = _bounded_lag_order(
            normalized,
            successors,
            workflow_by_request,
            service,
            lag_budget_ms,
            strategy_key,
        )
        if order in seen_orders:
            continue
        seen_orders.add(order)
        candidates.append(BoundedLagOrderCandidate(strategy, order, max_lag))
        if len(candidates) >= max_orders:
            break
    if not candidates:
        raise ValueError("no bounded-lag topological order was generated")
    return BoundedLagOrderSet(
        candidates=tuple(candidates),
        lag_budget_ms=lag_budget_ms,
        max_service_quantum_ms=max(service.values(), default=0.0),
    )


def _successors(
    predecessors: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    request_ids = set(predecessors)
    result: dict[str, list[str]] = {item: [] for item in request_ids}
    for request_id, required in predecessors.items():
        unknown = set(required) - request_ids
        if unknown or request_id in required:
            raise ValueError(
                f"invalid predecessors for {request_id}: {sorted(unknown)}"
            )
        for predecessor in required:
            result[predecessor].append(request_id)
    return {key: tuple(sorted(value)) for key, value in result.items()}


def _critical_remaining_service(
    topological_order: Sequence[str],
    successors: Mapping[str, Sequence[str]],
    service: Mapping[str, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for request_id in reversed(tuple(topological_order)):
        result[request_id] = service[request_id] + max(
            (result[item] for item in successors[request_id]),
            default=0.0,
        )
    return result


def _bounded_lag_order(
    predecessors: Mapping[str, Sequence[str]],
    successors: Mapping[str, Sequence[str]],
    workflow_by_request: Mapping[str, str],
    service: Mapping[str, float],
    lag_budget_ms: float,
    strategy_key: Callable[[str], tuple[float, ...]],
) -> tuple[tuple[str, ...], float]:
    indegree = {key: len(value) for key, value in predecessors.items()}
    available = {key for key, value in indegree.items() if value == 0}
    attained = {workflow_id: 0.0 for workflow_id in workflow_by_request.values()}
    order: list[str] = []
    max_lag = 0.0
    while available:
        runnable_workflows = {workflow_by_request[item] for item in available}
        floor = min(attained[item] for item in runnable_workflows)
        eligible = [
            item
            for item in available
            if attained[workflow_by_request[item]] <= floor + lag_budget_ms + 1e-12
        ]
        selected = min(eligible, key=lambda item: (strategy_key(item), item))
        workflow_id = workflow_by_request[selected]
        max_lag = max(max_lag, attained[workflow_id] - floor)
        attained[workflow_id] += service[selected]
        available.remove(selected)
        order.append(selected)
        for successor in successors[selected]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                available.add(successor)
    if len(order) != len(predecessors):
        raise ValueError("request dependency graph is cyclic")
    return tuple(order), max_lag


def _validate_topological_order(
    order: Sequence[str],
    request_predecessors: Mapping[str, Sequence[str]],
) -> None:
    completed: set[str] = set()
    for request_id in order:
        missing = set(request_predecessors[request_id]) - completed
        if missing:
            raise ValueError(
                f"execution order violates predecessors for {request_id}: {sorted(missing)}"
            )
        completed.add(request_id)
