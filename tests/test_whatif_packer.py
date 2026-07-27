from __future__ import annotations

from dataclasses import replace

from beliefkv.policy.reference import (
    CapabilityReport,
    PhysicalBundleSnapshot,
    PhysicalKVSnapshot,
    PolicyInput,
    ResidencyAction,
    ResourceSnapshot,
    RunnableInvocation,
    RuntimeGraphSnapshot,
)
from beliefkv.policy.scenario_physicalizer import (
    FrontierScenario,
    ScenarioPhysicalizer,
    ScenarioTransition,
)
from beliefkv.policy.whatif_packer import (
    FairnessWindow,
    WhatIfPacker,
    WhatIfPackerConfig,
)


def _bundle(
    bundle_id: str,
    owners: tuple[str, ...],
    *,
    size: int,
    gpu: int,
    cpu: int,
    extent: str,
    last_access_ms: float,
    locked: int = 0,
) -> PhysicalBundleSnapshot:
    return PhysicalBundleSnapshot(
        bundle_id=bundle_id,
        owner_context_ids=owners,
        scope="exclusive_suffix" if len(owners) == 1 else "shared_subtree",
        physical_unique_bytes=size,
        gpu_bytes=gpu,
        cpu_bytes=cpu,
        marginal_reclaimable_bytes=gpu,
        closure_bytes=size,
        locked_bytes=locked,
        residency=(
            "dual_clean"
            if gpu and cpu
            else "gpu_only"
            if gpu
            else "cpu_only"
        ),
        generation_fingerprint=f"generation-{bundle_id}",
        last_access_ms=last_access_ms,
        extent_ids=(extent,),
    )


def _input(
    *,
    capacity: int = 1_000,
    reserved: int = 100,
    include_cpu_target: bool = True,
    lock_old: bool = False,
    overlap: bool = False,
    host_free: int = 1_000,
) -> PolicyInput:
    bundles = [
        _bundle(
            "target-gpu",
            ("ctx-target",),
            size=200,
            gpu=200,
            cpu=0,
            extent="extent-target",
            last_access_ms=90,
        ),
        _bundle(
            "old",
            ("ctx-old",),
            size=250,
            gpu=250,
            cpu=250,
            extent="extent-target" if overlap else "extent-old",
            last_access_ms=0,
            locked=250 if lock_old else 0,
        ),
        _bundle(
            "recent",
            ("ctx-recent",),
            size=150,
            gpu=150,
            cpu=150,
            extent="extent-recent",
            last_access_ms=95,
        ),
    ]
    if include_cpu_target:
        bundles.append(
            _bundle(
                "target-cpu",
                ("ctx-target",),
                size=200,
                gpu=0,
                cpu=200,
                extent="extent-target-cpu",
                last_access_ms=80,
            )
        )
    snapshot_id = "whatif-snapshot"
    request = RunnableInvocation(
        request_id="request-target",
        workflow_id="workflow-target",
        invocation_id="invocation-target",
        context_id="ctx-target",
        context_epoch=0,
        submitted_ts_ms=1,
        startup_bytes=100,
    )
    return PolicyInput(
        runtime_graph=RuntimeGraphSnapshot(
            snapshot_id=snapshot_id,
            graph_version=3,
            observed_ts_ms=100,
            state={},
        ),
        runnable_frontier=(request,),
        physical_kv=PhysicalKVSnapshot(
            snapshot_id=snapshot_id,
            topology_version=4,
            allocator_version=5,
            gpu_bytes=600,
            cpu_bytes=sum(item.cpu_bytes for item in bundles),
            bundles=tuple(bundles),
        ),
        resources=ResourceSnapshot(
            snapshot_id=snapshot_id,
            ts_ms=100,
            hbm_capacity_bytes=capacity,
            hbm_used_bytes=600,
            hbm_reserved_bytes=reserved,
            host_free_bytes=host_free,
            urgent_d2h_bytes=0,
            urgent_h2d_bytes=0,
            pcie_utilization=0,
            gpu_compute_utilization=0.5,
            recent_kv_growth_bytes_per_ms=1,
            h2d_service_bytes_per_ms=100,
            d2h_service_bytes_per_ms=100,
            transfer_setup_p50_ms=0.1,
            unhidden_stall_per_byte=0.01,
        ),
        capabilities=CapabilityReport(
            runtime_name="whatif-test",
            runtime_version="1",
            supported_residency_actions=frozenset(ResidencyAction),
            execution_order_control=True,
            admission_control=True,
            transfer_dependencies=True,
            native_identity_mapping=True,
        ),
    )


def _blocking_scenario(*, growth: int = 200) -> FrontierScenario:
    return FrontierScenario(
        scenario_id="blocking-child",
        probability=1,
        transition=ScenarioTransition.BLOCKING,
        candidate_request_ids=("request-target",),
        consumer_context_ids=("ctx-target",),
        projected_growth_bytes={"ctx-target": growth},
        earliest_ready_p50_ms=10,
        earliest_ready_p90_ms=20,
    )


def test_blocking_scenario_restores_once_and_packs_atomic_victim() -> None:
    policy_input = _input()
    demand = ScenarioPhysicalizer().physicalize(
        policy_input, _blocking_scenario()
    )
    plan = WhatIfPacker().pack(policy_input, demand)

    assert demand.required_gpu_bundles == ("target-cpu", "target-gpu")
    assert demand.required_h2d_bytes == 200
    assert demand.projected_hbm_peak_bytes == 1_200
    assert plan.feasible
    assert plan.execution_order == ("request-target",)
    assert plan.admission_actions["request-target"] == "restore_then_admit"
    assert plan.bundle_actions["target-cpu"] == ResidencyAction.PREFETCH_GPU
    assert plan.bundle_actions["target-gpu"] == ResidencyAction.KEEP
    assert plan.bundle_actions["old"] == ResidencyAction.COMMIT_CPU
    assert plan.reclaimed_bytes == 250
    assert plan.projected_hbm_peak_bytes == 950


def test_multi_consumer_shared_bundle_is_physically_counted_once() -> None:
    policy_input = _input()
    shared = _bundle(
        "shared-consumer",
        ("ctx-consumer-a", "ctx-consumer-b"),
        size=120,
        gpu=0,
        cpu=120,
        extent="extent-shared-consumer",
        last_access_ms=50,
    )
    policy_input = replace(
        policy_input,
        physical_kv=replace(
            policy_input.physical_kv,
            cpu_bytes=policy_input.physical_kv.cpu_bytes + 120,
            bundles=(*policy_input.physical_kv.bundles, shared),
        ),
    )
    scenario = FrontierScenario(
        scenario_id="broadcast",
        probability=1,
        transition=ScenarioTransition.MULTI_CONSUMER,
        consumer_context_ids=("ctx-consumer-a", "ctx-consumer-b"),
    )

    demand = ScenarioPhysicalizer().physicalize(policy_input, scenario)

    assert demand.required_gpu_bundles == ("shared-consumer",)
    assert demand.required_h2d_bytes == 120


def test_fresh_spawn_reserves_independent_kv_without_parent_reuse() -> None:
    policy_input = _input()
    scenario = FrontierScenario(
        scenario_id="fresh-child",
        probability=0.8,
        transition=ScenarioTransition.FRESH_SPAWN,
        keep_context_ids=("ctx-target",),
        anonymous_fresh_bytes=400,
        earliest_ready_p50_ms=5,
        earliest_ready_p90_ms=15,
    )

    demand = ScenarioPhysicalizer().physicalize(policy_input, scenario)

    assert demand.speculative_only
    assert demand.projected_new_bytes == 400
    assert "target-gpu" in demand.required_gpu_bundles
    assert demand.projected_hbm_peak_bytes == 1_300


def test_cyclic_hysteresis_blocks_recent_bundle_until_real_emergency() -> None:
    normal = _input(
        capacity=650,
        reserved=0,
        include_cpu_target=False,
        lock_old=True,
    )
    scenario = FrontierScenario(
        scenario_id="peer-loop",
        probability=1,
        transition=ScenarioTransition.CYCLIC_REACTIVATION,
        candidate_request_ids=("request-target",),
    )
    demand = ScenarioPhysicalizer().physicalize(normal, scenario)

    stable = WhatIfPacker().pack(normal, demand)
    emergency = WhatIfPacker(
        WhatIfPackerConfig(emergency_hbm_ratio=0.90)
    ).pack(normal, demand)

    assert not stable.feasible
    assert any("hbm_capacity_shortage" in item for item in stable.blocker_reasons)
    assert emergency.feasible
    assert emergency.bundle_actions["recent"] == ResidencyAction.COMMIT_CPU


def test_fairness_can_defer_request_without_charging_its_startup() -> None:
    policy_input = _input(include_cpu_target=False, reserved=0)
    demand = ScenarioPhysicalizer().physicalize(
        policy_input,
        FrontierScenario(
            scenario_id="fairness",
            probability=1,
            transition=ScenarioTransition.NONBLOCKING,
            candidate_request_ids=("request-target",),
        ),
    )
    fairness = FairnessWindow(eligible_workflow_ids=frozenset({"other"}))

    plan = WhatIfPacker().pack(policy_input, demand, fairness=fairness)

    assert not plan.feasible
    assert plan.admission_actions["request-target"] == "defer"
    assert "fairness_ineligible:workflow-target" in plan.blocker_reasons


def test_overlapping_or_missing_extent_identity_fails_closed() -> None:
    policy_input = _input(overlap=True)
    demand = ScenarioPhysicalizer().physicalize(
        policy_input, _blocking_scenario(growth=0)
    )

    plan = WhatIfPacker().pack(policy_input, demand)

    assert not demand.physical_accounting_exact
    assert not plan.feasible
    assert "exact_physical_accounting_unavailable" in plan.blocker_reasons


def test_locked_optional_bytes_are_reported_when_capacity_is_impossible() -> None:
    policy_input = _input(lock_old=True)
    demand = ScenarioPhysicalizer().physicalize(
        policy_input, _blocking_scenario(growth=300)
    )

    plan = WhatIfPacker().pack(policy_input, demand)

    assert not plan.feasible
    assert any(item.startswith("locked_optional_bytes:") for item in plan.blocker_reasons)


def test_physicalization_and_packing_are_deterministic_and_side_effect_free() -> None:
    policy_input = _input()
    before = policy_input.to_dict()
    physicalizer = ScenarioPhysicalizer()
    packer = WhatIfPacker()

    first_demand = physicalizer.physicalize(policy_input, _blocking_scenario())
    second_demand = physicalizer.physicalize(policy_input, _blocking_scenario())
    first_plan = packer.pack(policy_input, first_demand)
    second_plan = packer.pack(policy_input, second_demand)

    assert first_demand == second_demand
    assert first_plan == second_plan
    assert policy_input.to_dict() == before
