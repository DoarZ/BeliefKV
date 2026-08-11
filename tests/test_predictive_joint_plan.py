from __future__ import annotations

from dataclasses import replace

from beliefkv.policy.joint_scheduler import (
    AsyncSemanticJointPlanner,
    JointPlannerConfig,
)
from beliefkv.policy.reference import (
    MetadataSource,
    MetadataValue,
    RunnableInvocation,
)
from tests.test_joint_scheduler import (
    _invocation,
    _request,
    _with_runtime_state,
)
from tests.test_whatif_packer import _input


def _predictive_request(
    request_id: str,
    workflow_id: str,
    invocation_id: str,
    context_id: str,
    *,
    submitted_ms: float,
    predicted_decode: float | None,
    predicted_wait: float | None = None,
    support: str = "backoff",
) -> RunnableInvocation:
    return RunnableInvocation(
        request_id=request_id,
        workflow_id=workflow_id,
        invocation_id=invocation_id,
        context_id=context_id,
        context_epoch=0,
        submitted_ts_ms=submitted_ms,
        startup_bytes=100,
        causal_class="pending_admission:foreground:root",
        predicted_remaining_decode_tokens=predicted_decode,
        predicted_external_wait_ms=predicted_wait,
        predicted_next_output_tokens=None,
        prediction_support_level=support,
        prediction_ood_reasons=(
            ("ood_unknown_project",) if support == "backoff" else ()
        ),
    )


def test_runnable_invocation_prediction_fields_round_trip() -> None:
    request = _predictive_request(
        "r1",
        "wf",
        "inv1",
        "ctx1",
        submitted_ms=5.0,
        predicted_decode=128.0,
        predicted_wait=512.0,
    )
    restored = RunnableInvocation.from_dict(request.to_dict())
    assert restored == request
    assert restored.predicted_remaining_decode_tokens == 128.0
    assert restored.prediction_support_level == "backoff"


def test_observed_planner_ignores_predicted_remaining_decode_for_ordering() -> None:
    normal = _predictive_request(
        "request-normal",
        "workflow",
        "normal",
        "ctx-normal",
        submitted_ms=0.0,
        predicted_decode=500.0,
    )
    child = _predictive_request(
        "request-child",
        "workflow",
        "child",
        "ctx-child",
        submitted_ms=100.0,
        predicted_decode=50.0,
    )
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    policy_input = _with_runtime_state(
        policy_input,
        (normal, child),
        {
            "child": _invocation("workflow", "ctx-child"),
            "normal": _invocation("workflow", "ctx-normal"),
        },
    )

    plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)

    assert not plan.prediction_used
    assert plan.execution.ordered_request_ids[:2] == (
        "request-normal",
        "request-child",
    )
    assert not plan.prediction_influence


def test_planner_falls_back_to_observed_order_without_predictions() -> None:
    normal = _request(
        "request-normal",
        "workflow",
        "normal",
        "ctx-normal",
        submitted_ms=0.0,
    )
    child = _request(
        "request-child",
        "workflow",
        "child",
        "ctx-child",
        submitted_ms=100.0,
    )
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    policy_input = _with_runtime_state(
        policy_input,
        (normal, child),
        {
            "child": _invocation("workflow", "ctx-child"),
            "normal": _invocation("workflow", "ctx-normal"),
        },
    )

    plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)

    assert not plan.prediction_used
    assert not plan.prediction_influence
    assert plan.execution.ordered_request_ids[0] == "request-normal"


def test_observed_victim_selection_ignores_predictive_metadata() -> None:
    policy_input = _input(capacity=600, reserved=0, include_cpu_target=False)
    policy_input = _with_runtime_state(
        policy_input,
        (policy_input.runnable_frontier[0],),
        {
            "invocation-target": _invocation(
                "workflow-target", "ctx-target"
            ),
            "invocation-old": _invocation("workflow-target", "ctx-old"),
            "invocation-recent": _invocation(
                "workflow-target", "ctx-recent"
            ),
        },
        joins={},
    )
    state = dict(policy_input.runtime_graph.state)
    rccg = dict(state["rccg"])
    rccg["contexts"] = {
        context_id: {"epoch": 0}
        for context_id in ("ctx-target", "ctx-old", "ctx-recent")
    }
    state["rccg"] = rccg
    policy_input = replace(
        policy_input,
        runtime_graph=replace(policy_input.runtime_graph, state=state),
    )
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                source=MetadataSource.PREDICTED,
                producer="frontier_belief_mvp",
                value={
                    "invocation-old": {
                        "remaining_external_wait_ms_p50": 100.0,
                        "next_output_tokens_p50": 10.0,
                        "remaining_decode_tokens_p50": 10.0,
                        "support_level": "backoff",
                        "ood_reasons": ["ood_unknown_project"],
                    },
                    "invocation-recent": {
                        "remaining_external_wait_ms_p50": 10_000.0,
                        "next_output_tokens_p50": 10.0,
                        "remaining_decode_tokens_p50": 10.0,
                        "support_level": "backoff",
                        "ood_reasons": ["ood_unknown_project"],
                    },
                },
            )
        },
    )

    plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)

    assert plan.semantic_residency
    assert plan.semantic_residency[0].context_id == "ctx-old"
    assert not plan.prediction_used
    assert not plan.prediction_influence


def test_victim_selection_uses_lru_without_predictions() -> None:
    policy_input = _input(capacity=600, reserved=0, include_cpu_target=False)
    policy_input = _with_runtime_state(
        policy_input,
        (policy_input.runnable_frontier[0],),
        {
            "invocation-target": _invocation(
                "workflow-target", "ctx-target"
            ),
        },
        joins={},
    )
    state = dict(policy_input.runtime_graph.state)
    rccg = dict(state["rccg"])
    rccg["contexts"] = {
        context_id: {"epoch": 0}
        for context_id in ("ctx-target", "ctx-old", "ctx-recent")
    }
    state["rccg"] = rccg
    policy_input = replace(
        policy_input,
        runtime_graph=replace(policy_input.runtime_graph, state=state),
    )

    plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)

    assert plan.semantic_residency
    assert plan.semantic_residency[0].context_id == "ctx-old"
    assert not plan.prediction_used
