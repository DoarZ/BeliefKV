from __future__ import annotations

from dataclasses import replace

from beliefkv.policy.reference import (
    AdmissionAction,
    CapabilityReport,
    IdentityMapping,
    PhysicalBundleSnapshot,
    PhysicalKVSnapshot,
    PolicyInput,
    ReactivePolicy,
    ReferencePolicyAdapter,
    ResidencyAction,
    ResourceSnapshot,
    RunnableInvocation,
    RuntimeGraphSnapshot,
)


def _bundle(
    bundle_id: str,
    context_id: str,
    *,
    physical: int,
    gpu: int,
    cpu: int,
    residency: str,
    last_access_ms: float,
) -> PhysicalBundleSnapshot:
    return PhysicalBundleSnapshot(
        bundle_id=bundle_id,
        owner_context_ids=(context_id,),
        scope="exclusive_suffix",
        physical_unique_bytes=physical,
        gpu_bytes=gpu,
        cpu_bytes=cpu,
        marginal_reclaimable_bytes=gpu,
        closure_bytes=physical,
        locked_bytes=0,
        residency=residency,
        generation_fingerprint=f"generation-{bundle_id}",
        last_access_ms=last_access_ms,
    )


def _input() -> PolicyInput:
    snapshot_id = "reference-snapshot-1"
    frontier = (
        RunnableInvocation(
            request_id="request-a",
            workflow_id="workflow-a",
            invocation_id="invocation-a",
            context_id="ctx-a",
            context_epoch=0,
            submitted_ts_ms=2.0,
            startup_bytes=100,
            program_id="program-a",
        ),
        RunnableInvocation(
            request_id="request-b",
            workflow_id="workflow-b",
            invocation_id="invocation-b",
            context_id="ctx-b",
            context_epoch=0,
            submitted_ts_ms=1.0,
            startup_bytes=100,
            program_id="program-b",
        ),
    )
    bundles = (
        _bundle(
            "gpu-active",
            "ctx-a",
            physical=300,
            gpu=300,
            cpu=0,
            residency="gpu_only",
            last_access_ms=20.0,
        ),
        _bundle(
            "gpu-wait",
            "ctx-c",
            physical=200,
            gpu=200,
            cpu=200,
            residency="dual_clean",
            last_access_ms=5.0,
        ),
        _bundle(
            "cpu-near",
            "ctx-b",
            physical=250,
            gpu=0,
            cpu=250,
            residency="cpu_only",
            last_access_ms=10.0,
        ),
    )
    return PolicyInput(
        runtime_graph=RuntimeGraphSnapshot(
            snapshot_id=snapshot_id,
            graph_version=7,
            observed_ts_ms=30.0,
            state={"ready": [item.invocation_id for item in frontier]},
        ),
        runnable_frontier=frontier,
        physical_kv=PhysicalKVSnapshot(
            snapshot_id=snapshot_id,
            topology_version=3,
            allocator_version=5,
            gpu_bytes=500,
            cpu_bytes=450,
            bundles=bundles,
        ),
        resources=ResourceSnapshot(
            snapshot_id=snapshot_id,
            ts_ms=30.0,
            hbm_capacity_bytes=1_000,
            hbm_used_bytes=500,
            hbm_reserved_bytes=100,
            host_free_bytes=10_000,
            urgent_d2h_bytes=0,
            urgent_h2d_bytes=0,
            pcie_utilization=0.2,
            gpu_compute_utilization=0.7,
            recent_kv_growth_bytes_per_ms=1.0,
            h2d_service_bytes_per_ms=100.0,
            d2h_service_bytes_per_ms=100.0,
            transfer_setup_p50_ms=0.1,
            unhidden_stall_per_byte=0.001,
        ),
        identity_mappings=tuple(
            IdentityMapping(
                request_id=item.request_id,
                workflow_id=item.workflow_id,
                invocation_id=item.invocation_id,
                context_id=item.context_id,
                context_epoch=item.context_epoch,
                program_id=item.program_id,
                native_request_id=f"native-{item.request_id}",
            )
            for item in frontier
        ),
        capabilities=CapabilityReport(
            runtime_name="reference-test",
            runtime_version="1",
            supported_residency_actions=frozenset(ResidencyAction),
            execution_order_control=True,
            admission_control=True,
            transfer_dependencies=True,
            native_identity_mapping=True,
        ),
    )


def test_reactive_reference_restores_fifo_head_and_selects_inactive_lru() -> None:
    record = ReferencePolicyAdapter(ReactivePolicy()).evaluate(_input())

    assert record.output.execution.ordered_request_ids == (
        "request-b",
        "request-a",
    )
    assert [item.action for item in record.output.admissions] == [
        AdmissionAction.RESTORE_THEN_ADMIT,
        AdmissionAction.DEFER,
    ]
    assert {item.bundle_id: item.action for item in record.output.residency} == {
        "cpu-near": ResidencyAction.PREFETCH_GPU,
        "gpu-wait": ResidencyAction.COMMIT_CPU,
    }
    assert len(record.output.dependencies) == 1


def test_shared_cpu_bundle_is_restored_once() -> None:
    policy_input = _input()
    shared = replace(
        policy_input.physical_kv.bundles[-1],
        owner_context_ids=("ctx-a", "ctx-b"),
    )
    shared_input = replace(
        policy_input,
        physical_kv=replace(
            policy_input.physical_kv,
            bundles=(*policy_input.physical_kv.bundles[:-1], shared),
        ),
    )

    output = ReferencePolicyAdapter(ReactivePolicy()).evaluate(shared_input).output
    assert sum(item.bundle_id == "cpu-near" for item in output.residency) == 1
