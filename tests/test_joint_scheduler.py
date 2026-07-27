from __future__ import annotations

from dataclasses import replace

from beliefkv.policy.joint_scheduler import (
    JointPlanCurrentState,
    JointPlannerConfig,
    ObservedJointPlanner,
    validate_joint_plan,
    validate_joint_plan_components,
)
from beliefkv.policy.reference import (
    AdmissionAction,
    ReferencePolicyAdapter,
    ResidencyAction,
    RunnableInvocation,
)
from tests.test_whatif_packer import _input


def _request(
    request_id: str,
    workflow_id: str,
    invocation_id: str,
    context_id: str,
    *,
    submitted_ms: float = 1.0,
    startup_bytes: int = 100,
    causal_class: str = "pending_admission:foreground:root",
) -> RunnableInvocation:
    return RunnableInvocation(
        request_id=request_id,
        workflow_id=workflow_id,
        invocation_id=invocation_id,
        context_id=context_id,
        context_epoch=0,
        submitted_ts_ms=submitted_ms,
        startup_bytes=startup_bytes,
        causal_class=causal_class,
    )


def _invocation(
    workflow_id: str,
    context_id: str,
    *,
    state: str = "ready",
    parent: str | None = None,
    relation: str = "root",
    context_mode: str = "fresh",
    execution_mode: str = "foreground",
    pending_messages: int = 0,
    blocking_children: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "workflow_id": workflow_id,
        "context_id": context_id,
        "state": state,
        "parent_invocation_id": parent,
        "relation_type": relation,
        "context_mode": context_mode,
        "execution_mode": execution_mode,
        "pending_messages": pending_messages,
        "blocking_children": list(blocking_children),
    }


def _with_runtime_state(
    policy_input,
    requests: tuple[RunnableInvocation, ...],
    invocations: dict[str, dict[str, object]],
    *,
    joins: dict[str, dict[str, object]] | None = None,
    virtual_runtime: dict[str, float] | None = None,
    memory_charges: dict[str, float] | None = None,
    transition_open: bool = False,
):
    workflows = {item.workflow_id for item in requests}
    vruntime = virtual_runtime or {}
    state = {
        "rccg": {
            "invocations": invocations,
            "joins": joins or {},
        },
        "workflow_fairness": {
            "accounts": {
                workflow_id: {
                    "weight": 1.0,
                    "attained_service_ms": vruntime.get(workflow_id, 0.0),
                    "virtual_runtime_ms": vruntime.get(workflow_id, 0.0),
                    "dispatch_count": 0,
                }
                for workflow_id in workflows
            },
            "memory_charges_bytes": memory_charges or {},
            "accounting_scope": "root_workflow",
        },
        "transition": {"open": transition_open, "generation": 1},
    }
    return replace(
        policy_input,
        runnable_frontier=requests,
        runtime_graph=replace(policy_input.runtime_graph, state=state),
    )


def _planner(**overrides) -> ObservedJointPlanner:
    defaults = {
        "max_planning_budget_ms": 100.0,
        "max_workflow_candidates": 8,
        "max_total_frontier_candidates": 16,
    }
    defaults.update(overrides)
    return ObservedJointPlanner(JointPlannerConfig(**defaults))


def _current_state(
    source,
    *,
    requests: tuple[RunnableInvocation, ...] | None = None,
    invocations: dict[str, dict[str, object]] | None = None,
    joins: dict[str, dict[str, object]] | None = None,
    bundles=None,
    virtual_runtime: dict[str, float] | None = None,
    fairness_revision: int = 0,
    hbm_available_bytes: int | None = None,
) -> JointPlanCurrentState:
    state = source.runtime_graph.state
    rccg = state.get("rccg", {})
    fairness = state.get("workflow_fairness", {})
    requests = requests or source.runnable_frontier
    invocation_state = invocations or dict(rccg.get("invocations", {}))
    join_state = joins if joins is not None else dict(rccg.get("joins", {}))
    account_state = dict(fairness.get("accounts", {}))
    if virtual_runtime is not None:
        account_state = {
            workflow_id: {
                "virtual_runtime_ms": value,
                "attained_service_ms": value,
                "weight": 1.0,
                "dispatch_count": 0,
            }
            for workflow_id, value in virtual_runtime.items()
        }
    bundle_state = bundles or {
        item.bundle_id: item for item in source.physical_kv.bundles
    }
    return JointPlanCurrentState(
        now_ms=source.resources.ts_ms,
        runnable_frontier=requests,
        invocation_snapshots=invocation_state,
        join_snapshots=join_state,
        transitions={},
        fairness_revision=fairness_revision,
        fairness_accounts=account_state,
        workflow_memory_charges=dict(
            fairness.get("memory_charges_bytes", {})
        ),
        transfer_epoch=0,
        hbm_capacity_bytes=source.resources.hbm_capacity_bytes,
        hbm_available_bytes=(
            source.resources.hbm_available_bytes
            if hbm_available_bytes is None
            else hbm_available_bytes
        ),
        host_free_bytes=source.resources.host_free_bytes,
        bundle_snapshots=bundle_state,
    )


def test_joint_planner_prioritizes_the_observed_join_straggler() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    child = _request("request-child", "workflow", "child", "ctx-child")
    normal = _request(
        "request-normal",
        "workflow",
        "normal",
        "ctx-normal",
        submitted_ms=0,
    )
    policy_input = _with_runtime_state(
        policy_input,
        (normal, child),
        {
            "child": _invocation(
                "workflow", "ctx-child", parent="parent", relation="spawn"
            ),
            "normal": _invocation("workflow", "ctx-normal"),
            "parent": _invocation(
                "workflow",
                "ctx-parent",
                state="wait_join",
                blocking_children=("child",),
            ),
        },
        joins={
            "join": {
                "members": ["child"],
                "completed": [],
                "waiters": ["parent"],
                "satisfied": False,
            }
        },
    )

    plan = _planner().plan(policy_input)

    assert plan.fallback_reason is None
    assert plan.execution.ordered_request_ids[0] == "request-child"
    assert {item.request_id for item in plan.admissions} == {
        "request-child",
        "request-normal",
    }


def test_joint_planner_applies_root_workflow_fairness_before_fanout() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    hot_a = _request("request-hot-a", "hot", "hot-a", "ctx-hot-a")
    hot_b = _request("request-hot-b", "hot", "hot-b", "ctx-hot-b")
    cold = _request("request-cold", "cold", "cold", "ctx-cold")
    policy_input = _with_runtime_state(
        policy_input,
        (hot_a, hot_b, cold),
        {
            "hot-a": _invocation("hot", "ctx-hot-a", relation="spawn"),
            "hot-b": _invocation("hot", "ctx-hot-b", relation="spawn"),
            "cold": _invocation("cold", "ctx-cold"),
        },
        virtual_runtime={"hot": 100.0, "cold": 0.0},
    )

    plan = _planner().plan(policy_input)

    assert plan.execution.ordered_request_ids == ("request-cold",)
    actions = {item.request_id: item.action for item in plan.admissions}
    assert actions["request-cold"] == AdmissionAction.ADMIT
    assert actions["request-hot-a"] == AdmissionAction.DEFER
    assert actions["request-hot-b"] == AdmissionAction.DEFER


def test_anytime_planner_keeps_a_feasible_prefix_when_budget_expires() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    first = _request("request-a", "workflow", "inv-a", "ctx-a")
    second = _request("request-b", "workflow", "inv-b", "ctx-b")
    policy_input = _with_runtime_state(
        policy_input,
        (first, second),
        {
            "inv-a": _invocation("workflow", "ctx-a"),
            "inv-b": _invocation("workflow", "ctx-b"),
        },
    )

    plan = _planner(max_planning_budget_ms=1e-9).plan(policy_input)

    assert plan.fallback_reason is None
    assert plan.execution.ordered_request_ids == ("request-a",)
    assert not plan.search_complete
    assert plan.planning_termination_reason == (
        "planning_budget_exceeded_best_feasible_prefix"
    )
    assert plan.evaluated_package_count == 1
    assert set(dict(plan.planning_phase_ms)) == {
        "candidate_order",
        "materialize",
        "pack",
        "physicalize",
        "prepare",
    }


def test_joint_planner_couples_restore_admission_and_victim_action() -> None:
    policy_input = _input(capacity=800, reserved=0)
    request = policy_input.runnable_frontier[0]
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
    )

    planner = _planner()
    plan = planner.plan(policy_input)
    decision = ReferencePolicyAdapter(planner).evaluate(policy_input)

    assert plan.fallback_reason is None
    assert plan.expected_hbm_peak_bytes <= 800
    actions = {item.bundle_id: item.action for item in plan.residency}
    assert actions["target-cpu"] == ResidencyAction.PREFETCH_GPU
    assert ResidencyAction.COMMIT_CPU in actions.values()
    admission = next(
        item for item in plan.admissions if item.request_id == "request-target"
    )
    assert admission.action == AdmissionAction.RESTORE_THEN_ADMIT
    assert admission.required_bundle_ids == ("target-cpu",)
    dependency = next(
        item
        for item in plan.dependencies
        if item.before_request_id == "request-target"
    )
    assert plan.residency[dependency.residency_intent_index].bundle_id == "target-cpu"
    assert dependency.require_ack
    assert decision.output.decision_id


def test_transition_open_uses_a_no_transfer_settling_barrier() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request = policy_input.runnable_frontier[0]
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
        transition_open=True,
    )

    plan = _planner().plan(policy_input)

    assert plan.transition_open
    assert plan.fallback_reason == "transition_open_settling_barrier"
    assert not plan.execution.ordered_request_ids
    assert not plan.residency
    assert {item.action for item in plan.admissions} == {AdmissionAction.DEFER}


def test_nonblocking_parent_is_not_parked_just_because_it_has_a_child() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request = policy_input.runnable_frontier[0]
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: {
                **_invocation(request.workflow_id, request.context_id),
                "children": ["background-child"],
            },
            "background-child": _invocation(
                request.workflow_id,
                "ctx-child",
                state="wait_tool",
                parent=request.invocation_id,
                relation="spawn",
                execution_mode="background",
            ),
        },
    )

    plan = _planner().plan(policy_input)

    admission = plan.admissions[0]
    assert admission.action == AdmissionAction.ADMIT
    assert plan.execution.ordered_request_ids == (request.request_id,)


def test_joint_plan_is_deterministic_and_ignores_unrelated_global_version() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request = policy_input.runnable_frontier[0]
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
    )
    planner = _planner()

    first = planner.plan(policy_input)
    second = planner.plan(policy_input)
    first_decision = ReferencePolicyAdapter(planner).evaluate(policy_input)
    second_decision = ReferencePolicyAdapter(planner).evaluate(policy_input)

    assert first.plan_id == second.plan_id
    assert first_decision.output.decision_id == second_decision.output.decision_id
    stale = replace(
        policy_input,
        physical_kv=replace(
            policy_input.physical_kv,
            topology_version=policy_input.physical_kv.topology_version + 1,
        ),
    )
    validation = validate_joint_plan(first, stale)
    assert "topology_version" in validation.strict_global_reasons
    assert validation.readset_fresh


def test_readset_validation_rejects_selected_request_and_bundle_conflicts() -> None:
    policy_input = _input(capacity=800, reserved=0)
    request = policy_input.runnable_frontier[0]
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
    )
    plan = _planner().plan(policy_input)
    changed_request = replace(request, context_epoch=1)
    changed = replace(policy_input, runnable_frontier=(changed_request,))

    validation = validate_joint_plan(plan, changed)

    assert not validation.readset_fresh
    assert "request_changed:request-target" in validation.readset_conflict_reasons
    assert "context_epoch:ctx-target" in validation.readset_conflict_reasons


def test_snapshot_lineage_change_alone_is_not_a_readset_conflict() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request = policy_input.runnable_frontier[0]
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
    )
    plan = _planner().plan(policy_input)
    next_snapshot = replace(
        policy_input,
        runtime_graph=replace(
            policy_input.runtime_graph,
            snapshot_id="next-snapshot",
        ),
        physical_kv=replace(
            policy_input.physical_kv,
            snapshot_id="next-snapshot",
        ),
        resources=replace(
            policy_input.resources,
            snapshot_id="next-snapshot",
        ),
    )

    validation = validate_joint_plan(plan, next_snapshot)

    assert "snapshot_id" in validation.strict_global_reasons
    assert validation.readset_fresh


def test_decreasing_runtime_startup_demand_does_not_stale_a_plan() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request = policy_input.runnable_frontier[0]
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
    )
    plan = _planner().plan(policy_input)
    progressed = replace(request, startup_bytes=request.startup_bytes - 1)
    current = replace(policy_input, runnable_frontier=(progressed,))

    validation = validate_joint_plan(plan, current)

    assert validation.readset_fresh


def test_increased_runtime_startup_demand_rechecks_current_capacity() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request = policy_input.runnable_frontier[0]
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
    )
    plan = _planner().plan(policy_input)
    expanded = replace(request, startup_bytes=2_000)
    current = replace(policy_input, runnable_frontier=(expanded,))

    validation = validate_joint_plan(plan, current)

    assert "insufficient_hbm_headroom" in validation.readset_conflict_reasons


def test_fairness_revision_can_advance_without_invalidating_priority() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request_a = _request("request-a", "workflow-a", "inv-a", "ctx-a")
    request_b = _request("request-b", "workflow-b", "inv-b", "ctx-b")
    invocations = {
        "inv-a": _invocation("workflow-a", "ctx-a"),
        "inv-b": _invocation("workflow-b", "ctx-b"),
    }
    source = _with_runtime_state(
        policy_input,
        (request_a, request_b),
        invocations,
        virtual_runtime={"workflow-a": 0, "workflow-b": 100},
    )
    plan = _planner().plan(source)
    current = _with_runtime_state(
        source,
        (request_a, request_b),
        invocations,
        virtual_runtime={"workflow-a": 1, "workflow-b": 100},
    )
    state = current.runtime_graph.to_dict()["state"]
    state["workflow_fairness"]["revision"] = 1
    current = replace(
        current,
        runtime_graph=replace(current.runtime_graph, state=state),
    )

    validation = validate_joint_plan(plan, current)

    assert plan.execution.selected_workflow_id == "workflow-a"
    assert validation.readset_fresh


def test_fairness_priority_change_invalidates_execution_intent() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request_a = _request("request-a", "workflow-a", "inv-a", "ctx-a")
    request_b = _request("request-b", "workflow-b", "inv-b", "ctx-b")
    invocations = {
        "inv-a": _invocation("workflow-a", "ctx-a"),
        "inv-b": _invocation("workflow-b", "ctx-b"),
    }
    source = _with_runtime_state(
        policy_input,
        (request_a, request_b),
        invocations,
        virtual_runtime={"workflow-a": 0, "workflow-b": 10},
    )
    plan = _planner().plan(source)
    current = _with_runtime_state(
        source,
        (request_a, request_b),
        invocations,
        virtual_runtime={"workflow-a": 20, "workflow-b": 10},
    )

    validation = validate_joint_plan(plan, current)

    assert plan.execution.selected_workflow_id == "workflow-a"
    assert "fairness_priority_changed" in validation.readset_conflict_reasons


def test_reserved_request_startup_is_not_counted_twice() -> None:
    policy_input = _input(capacity=700, reserved=100, include_cpu_target=False)
    original = policy_input.runnable_frontier[0]
    request = replace(
        original,
        causal_class="reserved_admission:foreground:root",
    )
    policy_input = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
                state="running_llm",
            )
        },
    )

    plan = _planner().plan(policy_input)

    assert plan.fallback_reason is None
    assert plan.expected_hbm_peak_bytes == 700
    admission = plan.admissions[0]
    assert admission.action == AdmissionAction.ADMIT
    assert admission.reserved_bytes == 0


def test_component_validation_isolates_an_unselected_request_change() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request_a = _request("request-a", "workflow-a", "inv-a", "ctx-a")
    request_b = _request("request-b", "workflow-b", "inv-b", "ctx-b")
    invocations = {
        "inv-a": _invocation("workflow-a", "ctx-a"),
        "inv-b": _invocation("workflow-b", "ctx-b"),
    }
    source = _with_runtime_state(
        policy_input,
        (request_a, request_b),
        invocations,
        virtual_runtime={"workflow-a": 0, "workflow-b": 100},
    )
    plan = _planner().plan(source)
    changed_b = replace(request_b, context_epoch=1)

    validation = validate_joint_plan_components(
        plan,
        source,
        _current_state(
            source,
            requests=(request_a, changed_b),
            invocations=invocations,
        ),
    )

    assert plan.execution.selected_workflow_id == "workflow-a"
    assert validation.admissions["request-a"].valid
    assert not validation.admissions["request-b"].valid
    assert validation.execution.valid
    assert validation.partially_fresh


def test_component_validation_isolates_one_blocked_physical_bundle() -> None:
    policy_input = _input(capacity=800, reserved=0)
    request = policy_input.runnable_frontier[0]
    source = _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
    )
    plan = _planner().plan(source)
    planned_bundle_ids = {item.bundle_id for item in plan.residency}
    victim_bundle_id = next(
        item.bundle_id
        for item in plan.residency
        if item.action == ResidencyAction.COMMIT_CPU
    )
    assert "target-cpu" in planned_bundle_ids
    bundles = {
        item.bundle_id: (
            replace(
                item,
                actionable=False,
                locked_bytes=item.gpu_bytes,
                blocker_codes=("node_locked",),
            )
            if item.bundle_id == victim_bundle_id
            else item
        )
        for item in source.physical_kv.bundles
    }

    validation = validate_joint_plan_components(
        plan,
        source,
        _current_state(
            source,
            bundles=bundles,
            hbm_available_bytes=300,
        ),
    )

    assert not validation.residency[victim_bundle_id].valid
    assert (
        "bundle_blocked:node_locked"
        in validation.residency[victim_bundle_id].reasons
    )
    assert validation.residency["target-cpu"].valid
    assert validation.admissions[request.request_id].valid
    assert validation.dependencies[0].valid
    assert validation.partially_fresh


def test_component_validation_rechecks_fairness_without_revision_staleness() -> None:
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request_a = _request("request-a", "workflow-a", "inv-a", "ctx-a")
    request_b = _request("request-b", "workflow-b", "inv-b", "ctx-b")
    invocations = {
        "inv-a": _invocation("workflow-a", "ctx-a"),
        "inv-b": _invocation("workflow-b", "ctx-b"),
    }
    source = _with_runtime_state(
        policy_input,
        (request_a, request_b),
        invocations,
        virtual_runtime={"workflow-a": 0, "workflow-b": 10},
    )
    plan = _planner().plan(source)

    same_priority = validate_joint_plan_components(
        plan,
        source,
        _current_state(
            source,
            invocations=invocations,
            virtual_runtime={"workflow-a": 1, "workflow-b": 10},
            fairness_revision=1,
        ),
    )
    flipped = validate_joint_plan_components(
        plan,
        source,
        _current_state(
            source,
            invocations=invocations,
            virtual_runtime={"workflow-a": 20, "workflow-b": 10},
            fairness_revision=2,
        ),
    )

    assert same_priority.execution.valid
    assert not flipped.execution.valid
    assert "fairness_priority_changed" in flipped.execution.reasons
    assert all(item.valid for item in flipped.admissions.values())
