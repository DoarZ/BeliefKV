from __future__ import annotations

from dataclasses import replace

import pytest

from beliefkv.policy.joint_oracle import (
    JointPlanOracle,
    OracleCandidate,
    OracleCost,
    ResimulatedPlanEvaluation,
    ResimulationEvidence,
    TraceOrderJointOracle,
    enumerate_topological_orders,
    generate_bounded_lag_topological_orders,
)
from beliefkv.policy.scenario_physicalizer import (
    FrontierScenario,
    ScenarioPhysicalizer,
    ScenarioTransition,
)
from beliefkv.policy.whatif_packer import ScenarioPlan
from tests.test_whatif_packer import _input


class _Evaluator:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid

    def evaluate(
        self,
        policy_input,
        demand,
        plan,
        *,
        trace_sensitivity,
    ) -> ResimulatedPlanEvaluation:
        del policy_input, trace_sensitivity
        baseline = any(value == "baseline_admit" for value in plan.admission_actions.values())
        costs = {
            ("scenario-a", True): 100.0,
            ("scenario-b", True): 80.0,
            ("scenario-a", False): 70.0,
            ("scenario-b", False): 50.0,
        }
        return ResimulatedPlanEvaluation(
            cost=OracleCost(
                workflow_jct_ms=costs[(demand.scenario_id, baseline)],
                causal_blocked_ms=10,
                unhidden_stall_ms=plan.expected_unhidden_stall_ms,
            ),
            evidence=ResimulationEvidence(
                schedule_recomputed=self.valid,
                queue_service_recomputed=True,
                physical_actions_recomputed=True,
                allocator_recomputed=True,
                service_model_calibrated=True,
                semantic_events_frozen=True,
                token_demand_frozen=True,
                tool_duration_frozen=True,
                transition_hash=f"hash-{demand.scenario_id}",
                service_model_id="test-service-model",
                physical_model_id="test-physical-model",
            ),
        )


class _TraceEvaluator:
    def evaluate_arm(
        self,
        arm,
        policy_input,
        demand,
        plan,
        *,
        trace_sensitivity,
    ) -> ResimulatedPlanEvaluation:
        del policy_input, demand, trace_sensitivity
        reversed_order = plan.execution_order == ("b", "a")
        cost = {
            ("O0", False): 100.0,
            ("O1", False): 100.0,
            ("O1", True): 80.0,
            ("O2", False): 70.0,
            ("O3", False): 70.0,
            ("O3", True): 50.0,
        }[(arm.value, reversed_order)]
        return ResimulatedPlanEvaluation(
            cost=OracleCost(cost, 0, 0),
            evidence=ResimulationEvidence(
                schedule_recomputed=True,
                queue_service_recomputed=True,
                physical_actions_recomputed=True,
                allocator_recomputed=True,
                service_model_calibrated=True,
                semantic_events_frozen=True,
                token_demand_frozen=True,
                tool_duration_frozen=True,
                transition_hash="trace-order",
                service_model_id="service",
                physical_model_id="physical",
            ),
        )


def _candidate(policy_input, scenario_id: str, sensitivity: str = "timing_sensitive"):
    scenario = FrontierScenario(
        scenario_id=scenario_id,
        probability=0.5,
        transition=ScenarioTransition.NONBLOCKING,
        candidate_request_ids=("request-target",),
        consumer_context_ids=("ctx-target",),
        earliest_ready_p50_ms=5,
        earliest_ready_p90_ms=10,
    )
    demand = ScenarioPhysicalizer().physicalize(policy_input, scenario)
    oracle_plan = JointPlanOracle().packer.pack(policy_input, demand)
    baseline = replace(
        oracle_plan,
        admission_actions={"request-target": "baseline_admit"},
    )
    return OracleCandidate(demand, baseline, sensitivity)


def test_joint_oracle_constructs_o0_o3_and_positive_synergy_gap() -> None:
    policy_input = _input(capacity=1_500, reserved=0)
    candidates = (
        _candidate(policy_input, "scenario-a"),
        _candidate(policy_input, "scenario-b"),
    )

    result = JointPlanOracle().evaluate(
        policy_input,
        candidates,
        current_scenario_id="scenario-a",
        evaluator=_Evaluator(),
    )

    assert result.arms[next(arm for arm in result.arms if arm.value == "O0")].cost.workflow_jct_ms == 100
    assert result.arms[next(arm for arm in result.arms if arm.value == "O1")].scenario_id == "scenario-b"
    assert result.arms[next(arm for arm in result.arms if arm.value == "O2")].cost.workflow_jct_ms == 70
    assert result.arms[next(arm for arm in result.arms if arm.value == "O3")].cost.workflow_jct_ms == 50
    assert result.joint_synergy_gap_ms == 20
    assert result.jointness_supported


def test_oracle_rejects_evaluator_that_reuses_original_schedule() -> None:
    policy_input = _input(capacity=1_500, reserved=0)
    candidates = (_candidate(policy_input, "scenario-a"),)

    with pytest.raises(ValueError, match="reused fixed timing"):
        JointPlanOracle().evaluate(
            policy_input,
            candidates,
            current_scenario_id="scenario-a",
            evaluator=_Evaluator(valid=False),
        )


def test_semantic_race_result_is_explicitly_optimistic() -> None:
    policy_input = _input(capacity=1_500, reserved=0)
    candidates = (
        _candidate(
            policy_input,
            "scenario-a",
            sensitivity="semantic_race_sensitive",
        ),
    )

    result = JointPlanOracle().evaluate(
        policy_input,
        candidates,
        current_scenario_id="scenario-a",
        evaluator=_Evaluator(),
    )

    assert {item.bound_kind for item in result.arms.values()} == {"optimistic"}


def test_oracle_rejects_misaligned_current_plan() -> None:
    policy_input = _input(capacity=1_500, reserved=0)
    candidate = _candidate(policy_input, "scenario-a")
    bad_plan = replace(candidate.current_separate_plan, scenario_id="other")

    with pytest.raises(ValueError, match="misaligned"):
        OracleCandidate(candidate.demand, bad_plan, "timing_sensitive")


def test_trace_order_oracle_exhaustively_constructs_o0_o3() -> None:
    policy_input = _input(capacity=1_500, reserved=0)
    candidate = _candidate(policy_input, "scenario-a")
    base_plan = replace(
        candidate.current_separate_plan,
        execution_order=("a", "b"),
        admission_actions={"a": "admit", "b": "admit"},
    )

    trace = TraceOrderJointOracle().evaluate(
        policy_input,
        candidate.demand,
        base_plan,
        request_predecessors={"a": (), "b": ()},
        baseline_execution_order=("a", "b"),
        trace_sensitivity="timing_sensitive",
        evaluator=_TraceEvaluator(),
    )

    assert trace.search_complete
    assert trace.candidate_order_count == 2
    assert trace.result.arms[next(arm for arm in trace.result.arms if arm.value == "O1")].plan.execution_order == ("b", "a")
    assert trace.result.arms[next(arm for arm in trace.result.arms if arm.value == "O3")].plan.execution_order == ("b", "a")
    assert trace.result.joint_synergy_gap_ms == 20
    assert trace.result.jointness_supported


def test_trace_order_oracle_marks_truncated_search_as_bounded() -> None:
    policy_input = _input(capacity=1_500, reserved=0)
    candidate = _candidate(policy_input, "scenario-a")
    base_plan = replace(
        candidate.current_separate_plan,
        execution_order=("a", "b"),
        admission_actions={"a": "admit", "b": "admit"},
    )

    trace = TraceOrderJointOracle().evaluate(
        policy_input,
        candidate.demand,
        base_plan,
        request_predecessors={"a": (), "b": ()},
        baseline_execution_order=("a", "b"),
        trace_sensitivity="timing_sensitive",
        evaluator=_TraceEvaluator(),
        max_candidate_orders=1,
    )

    assert not trace.search_complete
    assert trace.search_kind == "bounded_topological"
    assert not trace.result.jointness_supported


def test_trace_order_oracle_uses_supplied_fair_candidates_only_for_o1_o3() -> None:
    policy_input = _input(capacity=1_500, reserved=0)
    candidate = _candidate(policy_input, "scenario-a")
    base_plan = replace(
        candidate.current_separate_plan,
        execution_order=("a", "b"),
        admission_actions={"a": "admit", "b": "admit"},
    )

    trace = TraceOrderJointOracle().evaluate(
        policy_input,
        candidate.demand,
        base_plan,
        request_predecessors={"a": (), "b": ()},
        baseline_execution_order=("a", "b"),
        trace_sensitivity="timing_sensitive",
        evaluator=_TraceEvaluator(),
        candidate_execution_orders=(("b", "a"),),
        candidate_search_kind="bounded_lag_heuristic_topological",
    )

    by_name = {arm.value: result for arm, result in trace.result.arms.items()}
    assert by_name["O0"].plan.execution_order == ("a", "b")
    assert by_name["O2"].plan.execution_order == ("a", "b")
    assert by_name["O1"].plan.execution_order == ("b", "a")
    assert by_name["O3"].plan.execution_order == ("b", "a")
    assert trace.candidate_order_count == 1
    assert trace.search_kind == "bounded_lag_heuristic_topological"
    assert not trace.search_complete


def test_bounded_lag_order_shares_one_root_workflow_budget_across_fanout() -> None:
    result = generate_bounded_lag_topological_orders(
        {
            "a-root": (),
            "a-child": ("a-root",),
            "b-root": (),
            "b-child": ("b-root",),
        },
        workflow_by_request={
            "a-root": "workflow-a",
            "a-child": "workflow-a",
            "b-root": "workflow-b",
            "b-child": "workflow-b",
        },
        service_ms_by_request={
            "a-root": 100,
            "a-child": 100,
            "b-root": 100,
            "b-child": 100,
        },
        baseline_execution_order=("a-root", "a-child", "b-root", "b-child"),
        kv_tokens_by_request={
            "a-root": 10,
            "a-child": 20,
            "b-root": 30,
            "b-child": 40,
        },
        lag_budget_ms=0,
        max_orders=6,
    )

    assert result.candidates
    assert all(
        candidate.execution_order.index("b-root")
        < candidate.execution_order.index("a-child")
        for candidate in result.candidates
        if candidate.execution_order[0] == "a-root"
    )
    assert all(candidate.max_pre_dispatch_lag_ms == 0 for candidate in result.candidates)
    assert result.to_dict()["complete"] is False


def test_topological_order_enumerator_rejects_cycles() -> None:
    with pytest.raises(ValueError, match="cyclic"):
        enumerate_topological_orders(
            {"a": ("b",), "b": ("a",)},
            max_orders=10,
        )
