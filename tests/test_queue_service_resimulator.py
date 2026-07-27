from __future__ import annotations

from dataclasses import replace

import pytest

from beliefkv.policy.joint_oracle import JointPlanOracle, OracleCandidate
from beliefkv.policy.reference import MetadataSource
from beliefkv.policy.scenario_physicalizer import (
    FrontierScenario,
    ScenarioDemand,
    ScenarioPhysicalizer,
    ScenarioTransition,
)
from beliefkv.policy.whatif_packer import ScenarioPlan, WhatIfPacker
from beliefkv.simulator.queue_service import (
    CounterfactualQueueServiceSimulator,
    CounterfactualSimulationError,
    FrozenCounterfactualWorkload,
    FrozenRequestDemand,
    FrozenTracePlanEvaluator,
    FrozenWorkflowDemand,
    QueueServiceModel,
    RollingFrozenTracePlanEvaluator,
)
from beliefkv.simulator.rolling_physical import ResidencyReplayMode
from beliefkv.simulator.rolling_queue_service import (
    RollingCounterfactualQueueServiceSimulator,
)
from tests.test_whatif_packer import _blocking_scenario, _input


def _workload(*requests: FrozenRequestDemand) -> FrozenCounterfactualWorkload:
    workflow_ids = sorted({item.workflow_id for item in requests})
    predecessor_ids = {
        predecessor
        for request in requests
        for predecessor in request.predecessor_request_ids
    }
    workflows = tuple(
        FrozenWorkflowDemand(
            workflow_id=workflow_id,
            release_ms=0,
            terminal_request_ids=tuple(
                item.request_id
                for item in requests
                if item.workflow_id == workflow_id
                and item.request_id not in predecessor_ids
            ),
        )
        for workflow_id in workflow_ids
    )
    return FrozenCounterfactualWorkload(
        trace_id="frozen-test",
        transition_hash="transition-test",
        trace_sensitivity="timing_sensitive",
        requests=tuple(requests),
        workflows=workflows,
        future_physical_growth_exact=True,
    )


def _request(
    request_id: str,
    *,
    workflow_id: str = "workflow-target",
    predecessors: tuple[str, ...] = (),
    release_delay_ms: float = 0,
    prefill: int = 0,
    decode: int = 10,
    startup_bytes: int = 0,
    growth_bytes: int = 0,
) -> FrozenRequestDemand:
    return FrozenRequestDemand(
        request_id=request_id,
        workflow_id=workflow_id,
        invocation_id=f"invocation-{request_id}",
        context_id=f"context-{request_id}",
        context_epoch=0,
        predecessor_request_ids=predecessors,
        release_delay_ms=release_delay_ms,
        uncached_prompt_tokens=prefill,
        output_tokens=decode,
        startup_bytes=startup_bytes,
        kv_growth_bytes=growth_bytes,
        action_boundary_token_index=decode,
    )


def _service(*, calibrated: bool = True) -> QueueServiceModel:
    return QueueServiceModel(
        model_id="independent-calibration-v1",
        prefill_tokens_per_ms=10,
        decode_tokens_per_ms=10,
        decode_batch_efficiency=(1.0, 2.0),
        max_decode_batch=2,
        prefill_chunk_tokens=10,
        decode_quantum_tokens=8,
        calibrated=calibrated,
        calibration_source="held-out microbenchmark",
    )


def test_resimulator_recomputes_restore_queue_service_and_allocator() -> None:
    policy_input = _input()
    demand = ScenarioPhysicalizer().physicalize(
        policy_input, _blocking_scenario()
    )
    plan = WhatIfPacker().pack(policy_input, demand)
    workload = _workload(
        _request(
            "request-target",
            prefill=10,
            decode=10,
            startup_bytes=100,
            growth_bytes=100,
        )
    )

    result = CounterfactualQueueServiceSimulator(_service()).simulate(
        policy_input, demand, plan, workload
    )

    assert result.d2h_bytes == 0
    assert result.h2d_bytes == 200
    assert result.pcie_busy_ms == pytest.approx(2.1)
    assert result.request_queue_wait_ms["request-target"] == pytest.approx(2.1)
    assert result.workflow_jct_ms["workflow-target"] == pytest.approx(4.1)
    assert result.request_action_unlock_ms["request-target"] == pytest.approx(4.1)
    assert result.hbm_peak_bytes <= policy_input.resources.hbm_capacity_bytes


def test_decode_batch_service_is_recomputed_instead_of_reusing_wall_clock() -> None:
    policy_input = _input(
        capacity=2_000,
        reserved=0,
        include_cpu_target=False,
    )
    scenario = FrontierScenario(
        scenario_id="batch",
        probability=1,
        transition=ScenarioTransition.NONBLOCKING,
        candidate_request_ids=("request-target",),
    )
    demand = ScenarioPhysicalizer().physicalize(policy_input, scenario)
    plan = WhatIfPacker().pack(policy_input, demand)
    workload = _workload(
        _request("request-target"),
        _request("request-peer", workflow_id="workflow-peer"),
    )

    result = CounterfactualQueueServiceSimulator(_service()).simulate(
        policy_input, demand, plan, workload
    )

    assert result.scheduler_steps == 2
    assert result.workflow_jct_ms["workflow-target"] == pytest.approx(1.0)
    assert result.workflow_jct_ms["workflow-peer"] == pytest.approx(1.0)


def test_allocator_deadlock_is_explicit() -> None:
    policy_input = _input(
        capacity=700,
        reserved=0,
        include_cpu_target=False,
    )
    scenario = FrontierScenario(
        scenario_id="no-room",
        probability=1,
        transition=ScenarioTransition.NONBLOCKING,
        candidate_request_ids=("request-target",),
    )
    demand = ScenarioPhysicalizer().physicalize(policy_input, scenario)
    plan = WhatIfPacker().pack(policy_input, demand)
    assert plan.feasible
    workload = _workload(
        _request(
            "request-target",
            startup_bytes=200,
            growth_bytes=200,
        )
    )

    with pytest.raises(CounterfactualSimulationError, match="deadlock"):
        CounterfactualQueueServiceSimulator(_service()).simulate(
            policy_input, demand, plan, workload
        )


def test_joint_oracle_rejects_uncalibrated_or_inexact_trace_evidence() -> None:
    policy_input = _input(capacity=1_500, reserved=0)
    scenario = FrontierScenario(
        scenario_id="candidate",
        probability=1,
        transition=ScenarioTransition.NONBLOCKING,
        candidate_request_ids=("request-target",),
    )
    demand = ScenarioPhysicalizer().physicalize(policy_input, scenario)
    plan = WhatIfPacker().pack(policy_input, demand)
    candidate = OracleCandidate(demand, plan, "timing_sensitive")
    workload = _workload(_request("request-target"))

    with pytest.raises(ValueError, match="reused fixed timing"):
        JointPlanOracle().evaluate(
            policy_input,
            (candidate,),
            current_scenario_id="candidate",
            evaluator=FrozenTracePlanEvaluator(workload, _service(calibrated=False)),
        )

    inexact = replace(workload, future_physical_growth_exact=False)
    with pytest.raises(ValueError, match="reused fixed timing"):
        JointPlanOracle().evaluate(
            policy_input,
            (candidate,),
            current_scenario_id="candidate",
            evaluator=FrozenTracePlanEvaluator(inexact, _service()),
        )


def test_exact_token_radix_replay_can_satisfy_physical_timing_evidence() -> None:
    base = _input(capacity=10_000, reserved=0, include_cpu_target=False)
    policy_input = replace(
        base,
        physical_kv=replace(
            base.physical_kv,
            gpu_bytes=0,
            cpu_bytes=0,
            bundles=(),
        ),
        resources=replace(
            base.resources,
            hbm_capacity_bytes=10_000,
            hbm_used_bytes=0,
            hbm_reserved_bytes=0,
        ),
    )
    scenario = FrontierScenario(
        scenario_id="radix-exact",
        probability=1,
        transition=ScenarioTransition.NONBLOCKING,
        candidate_request_ids=("request-target",),
    )
    demand = ScenarioPhysicalizer().physicalize(policy_input, scenario)
    plan = WhatIfPacker().pack(policy_input, demand)
    prompt = (10, 20, 30, 40)
    commit = (*prompt, 50, 60)
    first = replace(
        _request(
            "request-target",
            prefill=4,
            decode=3,
            startup_bytes=70,
            growth_bytes=70,
        ),
        observed_cache_hit_tokens=0,
        prompt_token_symbols=prompt,
        cache_commit_token_symbols=commit,
    )
    second = replace(
        _request(
            "request-next",
            predecessors=("request-target",),
            prefill=1,
            decode=3,
            startup_bytes=40,
            growth_bytes=40,
        ),
        observed_cache_hit_tokens=3,
        prompt_token_symbols=prompt,
        cache_commit_token_symbols=commit,
    )
    workload = replace(
        _workload(first, second),
        future_physical_growth_exact=False,
        prefix_identity_complete=True,
        initial_radix_state_known=True,
        metadata={"kv_bytes_per_token": 10},
    )

    result = CounterfactualQueueServiceSimulator(_service()).simulate(
        policy_input, demand, plan, workload
    )
    oracle = JointPlanOracle().evaluate(
        policy_input,
        (OracleCandidate(demand, plan, "timing_sensitive"),),
        current_scenario_id="radix-exact",
        evaluator=FrozenTracePlanEvaluator(workload, _service()),
    )

    assert result.radix_demand_recomputed
    assert result.recomputed_cache_hit_tokens == {
        "request-next": 3,
        "request-target": 0,
    }
    assert result.recomputed_unique_growth_bytes == {
        "request-next": 0,
        "request-target": 60,
    }
    assert all(item.evidence.valid_for_timing for item in oracle.arms.values())
    assert all(
        item.evidence.physical_model_id == "token_radix_bundle_allocator_v2"
        for item in oracle.arms.values()
    )


def test_frozen_workload_rejects_dependency_cycle() -> None:
    first = _request("first", predecessors=("second",))
    second = _request("second", predecessors=("first",))

    with pytest.raises(ValueError, match="cycle"):
        FrozenCounterfactualWorkload(
            trace_id="cycle",
            transition_hash="cycle",
            trace_sensitivity="timing_sensitive",
            requests=(first, second),
            workflows=(
                FrozenWorkflowDemand(
                    workflow_id="workflow-target",
                    release_ms=0,
                    terminal_request_ids=("first",),
                ),
            ),
            future_physical_growth_exact=True,
        )


def _rolling_input_and_plan(
    request_ids: tuple[str, ...],
    *,
    capacity_bytes: int,
) -> tuple[object, ScenarioDemand, ScenarioPlan]:
    base = _input(capacity=1_000, reserved=0, include_cpu_target=False)
    policy_input = replace(
        base,
        physical_kv=replace(
            base.physical_kv,
            gpu_bytes=0,
            cpu_bytes=0,
            bundles=(),
        ),
        resources=replace(
            base.resources,
            hbm_capacity_bytes=capacity_bytes,
            hbm_used_bytes=0,
            hbm_reserved_bytes=0,
            host_free_bytes=1_000,
        ),
    )
    demand = ScenarioDemand(
        snapshot_id=policy_input.snapshot_id,
        scenario_id="rolling",
        probability=1,
        transition=ScenarioTransition.NONBLOCKING,
        source=MetadataSource.HINDSIGHT,
        candidate_invocation_ids=(),
        candidate_request_ids=(),
        consumer_context_ids=(),
        required_context_ids=(),
        required_gpu_bundles=(),
        optional_gpu_bundles=(),
        startup_bytes_by_request={},
        projected_growth_bytes={},
        projected_new_bytes=0,
        projected_hbm_peak_bytes=0,
        required_h2d_bytes=0,
        earliest_ready_p50_ms=0,
        earliest_ready_p90_ms=0,
        physical_accounting_exact=True,
    )
    plan = ScenarioPlan(
        snapshot_id=policy_input.snapshot_id,
        scenario_id=demand.scenario_id,
        execution_order=request_ids,
        admission_actions={request_id: "admit" for request_id in request_ids},
        bundle_actions={},
        feasible=True,
        expected_unhidden_stall_ms=0,
        hbm_time_byte_ms=0,
        d2h_bytes=0,
        h2d_bytes=0,
        recompute_tokens=0,
        projected_hbm_peak_bytes=0,
        reclaimed_bytes=0,
        physical_accounting_exact=True,
        blocker_reasons=(),
    )
    return policy_input, demand, plan


def _rolling_request(
    request_id: str,
    prompt: tuple[int, ...],
    commit: tuple[int, ...],
    *,
    predecessor: str | None = None,
    observed_hit: int = 0,
) -> FrozenRequestDemand:
    output_tokens = len(commit) - len(prompt) + 1
    return FrozenRequestDemand(
        request_id=request_id,
        workflow_id="rolling-workflow",
        invocation_id=f"invocation-{request_id}",
        context_id=f"context-{request_id}",
        context_epoch=0,
        predecessor_request_ids=((predecessor,) if predecessor else ()),
        release_delay_ms=0,
        uncached_prompt_tokens=len(prompt) - observed_hit,
        output_tokens=output_tokens,
        startup_bytes=max(
            (len(prompt) - observed_hit + output_tokens - 1) * 10,
            len(commit) * 10,
        ),
        kv_growth_bytes=len(commit) * 10,
        action_boundary_token_index=output_tokens,
        observed_cache_hit_tokens=observed_hit,
        prompt_token_symbols=prompt,
        cache_commit_token_symbols=commit,
    )


def test_rolling_queue_recomputes_tier_actions_and_next_use_jct() -> None:
    requests = (
        _rolling_request("a", (1,), (1, 10)),
        _rolling_request("b", (2,), (2, 20), predecessor="a"),
        _rolling_request("new", (3,), (3, 30), predecessor="b"),
        _rolling_request(
            "reuse-a",
            (1, 10, 99),
            (1, 10, 99, 100),
            predecessor="new",
            observed_hit=2,
        ),
    )
    workload = FrozenCounterfactualWorkload(
        trace_id="rolling-pressure",
        transition_hash="rolling-pressure-transition",
        trace_sensitivity="timing_sensitive",
        requests=requests,
        workflows=(
            FrozenWorkflowDemand(
                workflow_id="rolling-workflow",
                release_ms=0,
                terminal_request_ids=("reuse-a",),
            ),
        ),
        future_physical_growth_exact=False,
        prefix_identity_complete=True,
        initial_radix_state_known=True,
        metadata={"kv_bytes_per_token": 10},
    )
    policy_input, demand, plan = _rolling_input_and_plan(
        tuple(request.request_id for request in requests),
        capacity_bytes=40,
    )
    service = replace(
        _service(),
        max_decode_batch=1,
        decode_quantum_tokens=1,
    )

    reactive = RollingCounterfactualQueueServiceSimulator(
        service,
        residency_mode=ResidencyReplayMode.REACTIVE_LRU,
    ).simulate(policy_input, demand, plan, workload)
    oracle = RollingCounterfactualQueueServiceSimulator(
        service,
        residency_mode=ResidencyReplayMode.HINDSIGHT_NEXT_USE,
    ).simulate(policy_input, demand, plan, workload)

    assert reactive.rolling_physical_replay
    assert oracle.rolling_physical_replay
    assert reactive.recomputed_cache_hit_tokens["reuse-a"] == 2
    assert oracle.recomputed_cache_hit_tokens["reuse-a"] == 2
    assert reactive.h2d_bytes > oracle.h2d_bytes
    assert reactive.d2h_bytes > oracle.d2h_bytes
    assert (
        reactive.workflow_jct_ms["rolling-workflow"]
        > oracle.workflow_jct_ms["rolling-workflow"]
    )
    assert reactive.hbm_peak_bytes <= 40
    assert oracle.hbm_peak_bytes <= 40
    assert any(event["kind"] == "H2D_RESTORE" for event in reactive.physical_timeline)
    assert not any(event["kind"] == "H2D_RESTORE" for event in oracle.physical_timeline)


def test_joint_oracle_uses_reactive_for_o0_o1_and_hindsight_kv_for_o2_o3() -> None:
    requests = (
        _rolling_request("a", (1,), (1, 10)),
        _rolling_request("b", (2,), (2, 20), predecessor="a"),
        _rolling_request("new", (3,), (3, 30), predecessor="b"),
        _rolling_request(
            "reuse-a",
            (1, 10, 99),
            (1, 10, 99, 100),
            predecessor="new",
            observed_hit=2,
        ),
    )
    workload = FrozenCounterfactualWorkload(
        trace_id="rolling-oracle",
        transition_hash="rolling-oracle-transition",
        trace_sensitivity="timing_sensitive",
        requests=requests,
        workflows=(
            FrozenWorkflowDemand(
                workflow_id="rolling-workflow",
                release_ms=0,
                terminal_request_ids=("reuse-a",),
            ),
        ),
        prefix_identity_complete=True,
        initial_radix_state_known=True,
        metadata={"kv_bytes_per_token": 10},
    )
    policy_input, demand, plan = _rolling_input_and_plan(
        tuple(request.request_id for request in requests),
        capacity_bytes=40,
    )
    evaluator = RollingFrozenTracePlanEvaluator(
        workload,
        replace(_service(), max_decode_batch=1, decode_quantum_tokens=1),
    )

    result = JointPlanOracle().evaluate(
        policy_input,
        (OracleCandidate(demand, plan, "timing_sensitive"),),
        current_scenario_id="rolling",
        evaluator=evaluator,
    )

    assert set(evaluator.last_results) == set(result.arms)
    assert evaluator.simulation_count == 2
    assert evaluator.last_results[next(arm for arm in result.arms if arm.value == "O0")].h2d_bytes > 0
    assert evaluator.last_results[next(arm for arm in result.arms if arm.value == "O1")].h2d_bytes > 0
    assert evaluator.last_results[next(arm for arm in result.arms if arm.value == "O2")].h2d_bytes == 0
    assert evaluator.last_results[next(arm for arm in result.arms if arm.value == "O3")].h2d_bytes == 0
    assert {
        item.evidence.physical_model_id for item in result.arms.values()
    } == {"rolling_tiered_token_radix_allocator_v3"}
