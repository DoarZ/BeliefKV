from __future__ import annotations

from dataclasses import replace

import pytest

from beliefkv.policy.reference.base import (
    AdmissionAction,
    AdmissionIntent,
    CapabilityReport,
    EvaluationMode,
    ExecutionIntent,
    IdentityMapping,
    MetadataMode,
    MetadataRequirement,
    MetadataSource,
    MetadataValue,
    PhysicalBundleSnapshot,
    PhysicalKVSnapshot,
    PolicyContractError,
    PolicyDecisionRecord,
    PolicyInput,
    PolicyOutput,
    ReferencePolicyAdapter,
    ResidencyAction,
    ResidencyIntent,
    ResourceSnapshot,
    RunnableInvocation,
    RuntimeGraphSnapshot,
    UnsupportedKind,
)
from beliefkv.policy.reference.reactive import ReactivePolicy


def _input(*, metadata_mode: MetadataMode = MetadataMode.ORACLE) -> PolicyInput:
    snapshot_id = "snapshot-7"
    bundles = (
        PhysicalBundleSnapshot(
            bundle_id="bundle-a",
            owner_context_ids=("ctx-a",),
            scope="exclusive_suffix",
            physical_unique_bytes=300,
            gpu_bytes=300,
            cpu_bytes=0,
            marginal_reclaimable_bytes=300,
            closure_bytes=300,
            locked_bytes=0,
            residency="gpu_only",
            generation_fingerprint="generation-a",
            last_access_ms=8.0,
        ),
    )
    frontier = (
        RunnableInvocation(
            request_id="request-b",
            workflow_id="workflow-b",
            invocation_id="invocation-b",
            context_id="ctx-b",
            context_epoch=0,
            submitted_ts_ms=2.0,
            startup_bytes=250,
        ),
        RunnableInvocation(
            request_id="request-a",
            workflow_id="workflow-a",
            invocation_id="invocation-a",
            context_id="ctx-a",
            context_epoch=0,
            submitted_ts_ms=1.0,
            startup_bytes=200,
        ),
    )
    return PolicyInput(
        runtime_graph=RuntimeGraphSnapshot(
            snapshot_id=snapshot_id,
            graph_version=9,
            observed_ts_ms=10.0,
            state={"ready": ["invocation-b", "invocation-a"]},
        ),
        runnable_frontier=frontier,
        physical_kv=PhysicalKVSnapshot(
            snapshot_id=snapshot_id,
            topology_version=4,
            allocator_version=6,
            gpu_bytes=300,
            cpu_bytes=0,
            bundles=bundles,
        ),
        resources=ResourceSnapshot(
            snapshot_id=snapshot_id,
            ts_ms=10.0,
            hbm_capacity_bytes=800,
            hbm_used_bytes=300,
            hbm_reserved_bytes=100,
            host_free_bytes=10_000,
            urgent_d2h_bytes=0,
            urgent_h2d_bytes=0,
            pcie_utilization=0.2,
            gpu_compute_utilization=0.8,
            recent_kv_growth_bytes_per_ms=2.0,
            h2d_service_bytes_per_ms=100.0,
            d2h_service_bytes_per_ms=90.0,
            transfer_setup_p50_ms=0.1,
            unhidden_stall_per_byte=0.001,
        ),
        optional_metadata={
            "oracle_signal": MetadataValue(
                MetadataSource.HINDSIGHT,
                {"invocation-a": 1, "invocation-b": 4},
                producer="trace_replay",
            ),
            "predicted_signal": MetadataValue(
                MetadataSource.PREDICTED,
                {"invocation-a": 2, "invocation-b": 3},
                producer="distance_predictor",
            ),
        },
        identity_mappings=tuple(
            IdentityMapping(
                request_id=item.request_id,
                workflow_id=item.workflow_id,
                invocation_id=item.invocation_id,
                context_id=item.context_id,
                context_epoch=item.context_epoch,
                program_id="program-1",
                native_request_id=f"native-{item.request_id}",
            )
            for item in frontier
        ),
        capabilities=CapabilityReport(
            runtime_name="synthetic",
            runtime_version="1",
            supported_residency_actions=frozenset(ResidencyAction),
            execution_order_control=True,
            admission_control=True,
            transfer_dependencies=True,
            native_identity_mapping=True,
        ),
        metadata_mode=metadata_mode,
    )


def test_reactive_adapter_is_deterministic_shadow_only_and_round_trips() -> None:
    policy_input = _input()
    adapter = ReferencePolicyAdapter(ReactivePolicy())

    first = adapter.evaluate(policy_input)
    second = adapter.evaluate(policy_input)

    assert first == second
    assert first.output.shadow_only
    assert first.output.evaluation_mode == EvaluationMode.SHADOW
    assert first.output.metadata_mode == MetadataMode.ONLINE
    assert first.output.execution.ordered_request_ids == (
        "request-a",
        "request-b",
    )
    assert [item.action for item in first.output.admissions] == [
        AdmissionAction.ADMIT,
        AdmissionAction.DEFER,
    ]
    assert first.output.unsupported == ()
    assert PolicyInput.from_dict(policy_input.to_dict()) == policy_input
    assert PolicyDecisionRecord.from_dict(first.to_dict()) == first


class _MetadataProbePolicy:
    name = "metadata_probe"

    def __init__(self, requirement: MetadataRequirement) -> None:
        self.requirement = requirement
        self.seen_metadata: tuple[str, ...] | None = None

    def metadata_requirements(
        self, mode: MetadataMode
    ) -> tuple[MetadataRequirement, ...]:
        del mode
        return (self.requirement,)

    def decide(self, policy_input: PolicyInput) -> PolicyOutput:
        self.seen_metadata = tuple(policy_input.optional_metadata)
        value = policy_input.optional_metadata[self.requirement.name]
        return PolicyOutput(
            execution=ExecutionIntent(
                ordered_request_ids=(),
                selected_workflow_id=None,
                selected_invocation_id=None,
                mode="probe",
                graph_version=policy_input.runtime_graph.graph_version,
                reason="metadata isolation probe",
            ),
            admissions=(),
            residency=(),
            dependencies=(),
            policy_name=self.name,
            metadata_assumptions=(
                f"{value.source.value}:{self.requirement.name}",
            ),
            input_snapshot_id=policy_input.snapshot_id,
            metadata_mode=policy_input.metadata_mode,
        )


def test_online_adapter_does_not_expose_undeclared_or_hindsight_metadata() -> None:
    policy = _MetadataProbePolicy(
        MetadataRequirement(
            "predicted_signal",
            frozenset({MetadataSource.PREDICTED}),
        )
    )
    record = ReferencePolicyAdapter(policy).evaluate(_input())

    assert policy.seen_metadata == ("predicted_signal",)
    assert record.output.metadata_assumptions == ("predicted:predicted_signal",)
    assert "oracle_signal" not in record.input_fingerprint


def test_online_hindsight_requirement_fails_closed_without_calling_policy() -> None:
    policy = _MetadataProbePolicy(
        MetadataRequirement(
            "oracle_signal",
            frozenset({MetadataSource.HINDSIGHT}),
        )
    )
    record = ReferencePolicyAdapter(policy).evaluate(_input())

    assert policy.seen_metadata is None
    assert record.output.execution.mode == "reference_unsupported"
    assert len(record.output.unsupported) == 1
    assert record.output.unsupported[0].kind == UnsupportedKind.METADATA
    assert "hindsight" in record.output.unsupported[0].reason


def test_oracle_hindsight_is_only_available_in_replay_and_is_declared() -> None:
    policy = _MetadataProbePolicy(
        MetadataRequirement(
            "oracle_signal",
            frozenset({MetadataSource.HINDSIGHT}),
        )
    )
    with pytest.raises(PolicyContractError, match="only available in replay"):
        ReferencePolicyAdapter(policy, metadata_mode=MetadataMode.ORACLE)

    record = ReferencePolicyAdapter(
        policy,
        metadata_mode=MetadataMode.ORACLE,
        evaluation_mode=EvaluationMode.REPLAY,
    ).evaluate(_input())

    assert policy.seen_metadata == ("oracle_signal",)
    assert record.output.metadata_assumptions == (
        "hindsight:oracle_signal",
    )


def test_policy_input_rejects_mixed_physical_accounting() -> None:
    policy_input = _input()
    resources = replace(policy_input.resources, hbm_used_bytes=299)

    with pytest.raises(ValueError, match="must equal resource HBM"):
        replace(policy_input, resources=resources)


class _UnsupportedActionPolicy:
    name = "unsupported_action"

    def metadata_requirements(
        self, mode: MetadataMode
    ) -> tuple[MetadataRequirement, ...]:
        del mode
        return ()

    def decide(self, policy_input: PolicyInput) -> PolicyOutput:
        return PolicyOutput(
            execution=ExecutionIntent(
                ordered_request_ids=(),
                selected_workflow_id=None,
                selected_invocation_id=None,
                mode="probe",
                graph_version=policy_input.runtime_graph.graph_version,
                reason="unsupported action probe",
            ),
            admissions=(),
            residency=(
                ResidencyIntent(
                    bundle_id="bundle-a",
                    action=ResidencyAction.COMMIT_CPU,
                    target_bytes=300,
                    deadline_ms=20.0,
                    scenario_support=frozenset({"observed"}),
                    reason="probe capability reporting",
                ),
            ),
            dependencies=(),
            policy_name=self.name,
            metadata_assumptions=("observed:physical_kv",),
            input_snapshot_id=policy_input.snapshot_id,
        )


def test_unsupported_action_is_preserved_and_explicitly_reported() -> None:
    policy_input = replace(
        _input(),
        capabilities=replace(
            _input().capabilities,
            supported_residency_actions=frozenset({ResidencyAction.KEEP}),
        ),
    )
    record = ReferencePolicyAdapter(_UnsupportedActionPolicy()).evaluate(policy_input)

    assert record.output.residency[0].action == ResidencyAction.COMMIT_CPU
    assert record.output.unsupported[0].kind == UnsupportedKind.ACTION
    assert record.output.unsupported[0].name == "residency:commit_cpu"


def test_audit_record_rebuilds_output_and_detects_tampering() -> None:
    record = ReferencePolicyAdapter(ReactivePolicy()).evaluate(_input())
    audit_record = {
        "event": "reference_policy_decision",
        "ts_ms": 10.0,
        **record.to_audit_fields(),
    }

    assert PolicyDecisionRecord.from_audit_record(audit_record) == record

    raw = record.to_dict()
    raw["output"]["decision_id"] = "tampered"  # type: ignore[index]
    with pytest.raises(PolicyContractError, match="decision_id is invalid"):
        PolicyDecisionRecord.from_dict(raw)


def test_policy_cannot_omit_required_metadata_assumption() -> None:
    class BadPolicy(_MetadataProbePolicy):
        def decide(self, policy_input: PolicyInput) -> PolicyOutput:
            output = super().decide(policy_input)
            return replace(
                output,
                metadata_assumptions=("observed:runtime_graph",),
            )

    policy = BadPolicy(
        MetadataRequirement(
            "predicted_signal",
            frozenset({MetadataSource.PREDICTED}),
        )
    )
    with pytest.raises(PolicyContractError, match="without declaring"):
        ReferencePolicyAdapter(policy).evaluate(_input())
