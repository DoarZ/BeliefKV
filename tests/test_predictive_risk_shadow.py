from dataclasses import replace

from beliefkv.control.causal_graph import (
    ContextRecord,
    InvocationRecord,
    InvocationState,
    RuntimeCausalContextGraph,
    WorkflowRecord,
)
from beliefkv.policy.joint_scheduler import AsyncSemanticJointPlanner, JointPlannerConfig
from beliefkv.policy.reference import (
    MetadataSource,
    MetadataValue,
    PhysicalBundleSnapshot,
)
from beliefkv.policy.risk_shadow import (
    PredictiveIntent,
    PredictiveEligibilityIndex,
    PredictiveRiskShadowConfig,
    PredictiveRiskShadowObserver,
    PredictiveRiskShadowResult,
    _transfer_deadline_and_slack,
    validate_predictive_causal_certificate,
    validate_predictive_certificate,
)
from beliefkv.policy.predictive_joint import PredictiveActionKind
from beliefkv.predictor.frontier_belief import PredictiveEvidenceReadSet
from beliefkv.predictor.hardware_service import GPUServiceCurveModel
from beliefkv.predictor.structured_frontier import (
    EmpiricalDistribution,
    LocalFrontierPrediction,
)
from tests.test_whatif_packer import _input


def _distribution(value: float) -> EmpiricalDistribution:
    return EmpiricalDistribution((value,), (1.0,), 10.0)


def test_transfer_guard_is_recomputed_at_conservative_deadline() -> None:
    deadline, slack = _transfer_deadline_and_slack(
        125.0,
        200.0,
        transfer_ms=100.0,
        guard_ms=25.0,
    )
    assert deadline == 125.0
    assert slack == 0.0

    _, positive = _transfer_deadline_and_slack(
        125.001,
        200.0,
        transfer_ms=100.0,
        guard_ms=25.0,
    )
    assert positive is not None and positive > 0


def _prediction() -> LocalFrontierPrediction:
    return LocalFrontierPrediction(
        invocation_id="invocation-target",
        boundary_distribution={"tool": 1.0},
        current_sequence_tokens=4096,
        remaining_decode_tokens=_distribution(0),
        remaining_external_wait=_distribution(10),
        tool_terminal_distribution={"success": 1.0},
        prompt_growth_tokens=_distribution(32),
        next_output_tokens=_distribution(16),
        support_level="exact",
        calibration_coverage=0.95,
    )


def _service_row(sample_id: str, phase: str, token_delta: int) -> dict[str, object]:
    return {
        "row_type": "gpu_batch_service_interval",
        "sample_id": sample_id,
        "split": "train",
        "phase": phase,
        "batch_size": 1,
        "request_samples": [
            {
                "request_id": sample_id,
                "sequence_tokens_before": 4096,
                "token_delta": token_delta,
                "cache_hit_ratio": 0.0,
            }
        ],
        "chunk_position": "first",
        "prefill_decode_mixed": False,
        "pcie_contention_state": "idle",
        "hicache_inflight_bytes": 0,
        "service_elapsed_ms": 1.0,
        "warmup": False,
        "evidence_role": "controlled_microbenchmark",
    }


def _graph() -> RuntimeCausalContextGraph:
    graph = RuntimeCausalContextGraph()
    graph.workflows["workflow-target"] = WorkflowRecord(
        "workflow-target", 0.0, invocation_ids={"invocation-target"}
    )
    graph.contexts["ctx-target"] = ContextRecord(
        "workflow-target",
        "ctx-target",
        0,
        0.0,
        100.0,
        invocation_ids={"invocation-target"},
    )
    graph.invocations["invocation-target"] = InvocationRecord(
        workflow_id="workflow-target",
        invocation_id="invocation-target",
        context_id="ctx-target",
        agent_definition_id="coder",
        agent_instance_id="coder-0",
        state=InvocationState.WAIT_TOOL,
        created_ts_ms=0.0,
        updated_ts_ms=100.0,
        active_tool_family="shell",
        active_tool_start_ms=90.0,
    )
    graph._graph_version = 3
    return graph


def _attach_graph(policy_input, graph):
    return replace(
        policy_input,
        runtime_graph=replace(
            policy_input.runtime_graph,
            graph_version=graph.graph_version,
            state=graph.snapshot(),
        ),
    )


def _add_waiting_victim(graph: RuntimeCausalContextGraph) -> None:
    graph.workflows["workflow-old"] = WorkflowRecord(
        "workflow-old", 0.0, invocation_ids={"invocation-old"}
    )
    graph.contexts["ctx-old"] = ContextRecord(
        "workflow-old",
        "ctx-old",
        0,
        0.0,
        100.0,
        invocation_ids={"invocation-old"},
    )
    graph.invocations["invocation-old"] = InvocationRecord(
        workflow_id="workflow-old",
        invocation_id="invocation-old",
        context_id="ctx-old",
        agent_definition_id="coder",
        agent_instance_id="coder-old",
        state=InvocationState.WAIT_TOOL,
        created_ts_ms=0.0,
        updated_ts_ms=100.0,
        active_tool_family="shell",
        active_tool_start_ms=90.0,
    )


def test_local_frontier_prediction_round_trip_preserves_distributions() -> None:
    prediction = _prediction()

    restored = LocalFrontierPrediction.from_dict(prediction.to_dict())

    assert restored == prediction


def test_exact_shadow_prefetch_is_evaluated_without_mutating_joint_plan() -> None:
    policy_input = _input(capacity=1_000, reserved=100, include_cpu_target=True)
    prediction = _prediction()
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-b", "prefill", 32),
            _service_row("decode-a", "decode", 16),
            _service_row("decode-b", "decode", 16),
        ]
    )
    observer = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            minimum_calibration_coverage=0.9,
            kv_bytes_per_token=1,
        ),
    )
    evidence = PredictiveEvidenceReadSet(
        graph_version=3,
        page_revision=5,
        topology_revision=4,
        fairness_revision=0,
        admission_revision=0,
        transfer_epoch=0,
        obligation_revision=0,
        lease_revision=0,
        grace_revision=0,
        parser_frontier_revision=0,
        model_version="frontier-test-v1",
    )

    result = observer.evaluate(
        policy_input,
        graph=_graph(),
        source_plan=source_plan,
        evidence_read_set=evidence,
    )

    assert result.status == "evaluated"
    assert result.selected_action == "prefetch_gpu"
    assert result.predictive_intent is not None
    assert result.predictive_intent.action.value == "prefetch_gpu"
    assert result.predictive_intent.context_id == "ctx-target"
    assert not result.predictive_intent.to_dict().get("bundle_evidence")
    assert not source_plan.prediction_used
    assert result.to_dict()["prediction_used"] is False

    repeated = observer.evaluate(
        policy_input,
        graph=_graph(),
        source_plan=source_plan,
        evidence_read_set=evidence,
    )
    assert repeated.belief_cache_hit
    assert repeated.service_cache_hits > 0


def test_full_prefetch_over_canary_cap_is_filtered_before_risk_evaluation() -> None:
    policy_input = _input(capacity=1_000, reserved=100, include_cpu_target=True)
    prediction = _prediction()
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill", "prefill", 32),
            _service_row("decode", "decode", 16),
        ]
    )

    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=8,
            top_k=4,
            max_candidates=4,
            kv_bytes_per_token=1,
            max_full_prefetch_hbm_ratio=0.05,
        ),
    ).evaluate(
        policy_input,
        graph=_graph(),
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    assert result.selected_action == "observed_baseline"
    assert not any(
        item["action"] == "prefetch_gpu" for item in result.candidate_summaries
    )

def test_backoff_shadow_cannot_select_prefetch() -> None:
    policy_input = _input(capacity=1_000, reserved=100, include_cpu_target=True)
    prediction = replace(
        _prediction(),
        support_level="backoff",
        calibration_coverage=0.99,
    )
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-b", "prefill", 32),
            _service_row("decode-a", "decode", 16),
            _service_row("decode-b", "decode", 16),
        ]
    )
    observer = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            minimum_calibration_coverage=0.9,
            kv_bytes_per_token=1,
        ),
    )

    result = observer.evaluate(
        policy_input,
        graph=_graph(),
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    assert result.status == "evaluated"
    assert result.selected_action != "prefetch_gpu"
    assert "prefetch_gpu:reentry_window_unavailable" in result.blocked_reasons


def test_calibrated_backoff_can_supply_prefetch_specific_heads() -> None:
    policy_input = _input(capacity=1_000, reserved=100, include_cpu_target=True)
    prediction = replace(
        _prediction(),
        support_level="backoff",
        calibration_coverage=0.99,
        ood_reasons=("boundary_unavailable",),
        calibrated_intervals={
            "remaining_external_wait_ms": (8.0, 20.0),
            "prompt_growth_tokens": (16.0, 64.0),
        },
    )
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-b", "prefill", 32),
            _service_row("decode-a", "decode", 16),
            _service_row("decode-b", "decode", 16),
        ]
    )

    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            minimum_calibration_coverage=0.9,
            kv_bytes_per_token=1,
        ),
    ).evaluate(
        policy_input,
        graph=_graph(),
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    assert result.selected_action == "prefetch_gpu"
    assert result.predictive_intent is not None
    assert dict(result.predictive_intent.prediction_head_support) == {
        "future_kv_growth": "calibrated_backoff",
        "reentry_window": "calibrated_backoff",
    }


def test_physically_blocked_target_cannot_select_prefetch() -> None:
    policy_input = _input(capacity=1_000, reserved=100, include_cpu_target=True)
    bundles = tuple(
        replace(
            bundle,
            actionable=False,
            blocker_codes=("ancestor_closure",),
        )
        if bundle.bundle_id == "target-cpu"
        else bundle
        for bundle in policy_input.physical_kv.bundles
    )
    prediction = _prediction()
    policy_input = replace(
        policy_input,
        physical_kv=replace(policy_input.physical_kv, bundles=bundles),
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-b", "prefill", 32),
            _service_row("decode-a", "decode", 16),
            _service_row("decode-b", "decode", 16),
        ]
    )

    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            kv_bytes_per_token=1,
        ),
    ).evaluate(
        policy_input,
        graph=_graph(),
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    assert result.selected_action == "observed_baseline"


def test_future_kv_growth_can_reject_statically_feasible_prefetch() -> None:
    policy_input = _input(capacity=930, reserved=100, include_cpu_target=True)
    prediction = _prediction()
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-b", "prefill", 32),
            _service_row("decode-a", "decode", 16),
            _service_row("decode-b", "decode", 16),
        ]
    )

    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            kv_bytes_per_token=1,
        ),
    ).evaluate(
        policy_input,
        graph=_graph(),
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    summary = next(
        item
        for item in result.candidate_summaries
        if ":prefetch:" in str(item["package_id"])
    )
    assert result.selected_action == "observed_baseline"
    assert summary["future_hbm_feasibility_probability"] == 0.0
    assert summary["worst_future_hbm_overflow_bytes"] == 18
    assert "future_hbm_chance_constraint" in summary["reasons"]


def test_prepare_host_receives_recourse_value_only_before_future_pressure() -> None:
    graph = _graph()
    _add_waiting_victim(graph)
    # Keep the first predicted deficit within the victim's exclusive suffix so
    # PREPARE_HOST can actually replace the reactive offload path.
    policy_input = _input(capacity=1_230, reserved=100, include_cpu_target=True)
    policy_input = replace(
        policy_input,
        physical_kv=replace(
            policy_input.physical_kv,
            bundles=tuple(
                replace(bundle, cpu_bytes=0)
                if bundle.bundle_id == "old"
                else bundle
                for bundle in policy_input.physical_kv.bundles
            ),
        ),
    )
    target = replace(
        _prediction(),
        prompt_growth_tokens=_distribution(512),
    )
    victim = replace(
        _prediction(),
        invocation_id="invocation-old",
        remaining_external_wait=_distribution(1000),
    )
    policy_input = _attach_graph(
        replace(
            policy_input,
            optional_metadata={
                "frontier_predictions": MetadataValue(
                    MetadataSource.PREDICTED,
                    {
                        target.invocation_id: target.to_dict(),
                        victim.invocation_id: victim.to_dict(),
                    },
                    "test-frontier",
                ),
                "frontier_prediction_model_version": MetadataValue(
                    MetadataSource.PREDICTED,
                    "frontier-test-v1",
                    "test-frontier",
                ),
                "beliefkv_transfer_interference_policy": MetadataValue(
                    MetadataSource.APPLICATION_PROVIDED,
                    {
                        "mode": "stall_fraction",
                        "stall_fraction": 0.1,
                        "service_epoch": "test-transfer-v1",
                    },
                    "test-interference-policy",
                ),
                "beliefkv_transfer_service_curve_snapshot": MetadataValue(
                    MetadataSource.OBSERVED,
                    {
                        "schema_version": 1,
                        "min_samples": 1,
                        "warm_start_hardware_key": "test-shape-v1",
                        "fallback": {
                            "bandwidth_gbps": 24.0,
                            "overhead_ms": 0.1,
                            "safety_factor": 1.25,
                        },
                        "buckets": [
                            {
                                "direction": "d2h",
                                "size_bucket": 7,
                                "page_count_bucket": 0,
                                "compute_phase": "unknown",
                                "command_kind": "offload_context",
                                "host_copy_state": "missing",
                                "pinned_host": True,
                                "native_traffic_bucket": 0,
                                "sample_count": 1,
                                "usable_count": 1,
                                "outcome_count": 1,
                                "rejection_probability": 0.0,
                                "setup_p90_ms": 0.1,
                                "callback_floor_p90_ms": 0.1,
                                "fixed_overhead_p90_ms": 0.0,
                                "effective_bytes_per_ms_p10": 100.0,
                                "estimated_unhidden_stall_p90_ms": 0.25,
                            }
                        ],
                    },
                    "test-shape-curve",
                ),
            },
        ),
        graph,
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    source_plan = replace(
        source_plan,
        execution=replace(
            source_plan.execution,
            ordered_request_ids=("request-target",),
        ),
    )
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-large", "prefill", 512),
            _service_row("decode-a", "decode", 16),
        ]
    )

    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=6,
            kv_bytes_per_token=1,
            transfer_commit_guard_ms=0.0,
        ),
    ).evaluate(
        policy_input,
        graph=graph,
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    prepare = next(
        item
        for item in result.candidate_summaries
        if ":prepare:ctx-old" in str(item["package_id"])
    )
    assert prepare["scenario_projection"] == "prepare_host"
    assert prepare["expected_benefit_ms"] > 0, prepare
    assert "insufficient_expected_benefit" not in prepare["reasons"]
    assert prepare["prepare_recourse_failure_counts"] == {"eligible": 1}
    diagnostic = prepare["prepare_recourse_scenarios"][0]
    assert diagnostic["shadow_completion_ms"] <= diagnostic["first_pressure_ms"]
    assert diagnostic["first_pressure_ms"] < diagnostic["parent_reentry_ms"]
    assert (
        diagnostic["exclusive_reclaimable_bytes"]
        >= diagnostic["pressure_deficit_bytes"]
    )
    assert diagnostic["baseline_reactive_d2h_ms"] > 0
    assert diagnostic["transfer_duration_source"] == "bucket"
    assert diagnostic["transfer_service_epoch"] == "test-shape-v1"
    assert diagnostic["transfer_shape_supported"] is True
    assert diagnostic["predicted_extent_count"] == 1
    assert diagnostic["morphology_slack_ms"] > 0
    assert diagnostic["conservative_morphology_slack_ms"] > 0
    assert diagnostic["morphology_debt_ms"] == diagnostic[
        "shape_aware_transfer_p90_ms"
    ]
    assert diagnostic["morphology_penalty_ms"] == (
        diagnostic["shape_aware_transfer_p90_ms"]
        - diagnostic["byte_only_transfer_ms"]
    )
    assert diagnostic["interference_source"] == "stall_fraction_sensitivity"
    assert diagnostic["interference_service_epoch"] == "test-transfer-v1"
    assert diagnostic["interference_to_transfer_ratio"] == 0.1
    assert diagnostic["reactive_victim_model"] == "snapshot_consistent_conservative"


def test_join_revision_invalidates_action_specific_causal_certificate() -> None:
    certificate = {
        "context_epochs": [],
        "invocation_evidence": [],
        "join_evidence": [["join-1", "all", False, []]],
        "communication_evidence": [],
        "model_version": "frontier-v1",
    }
    current_graph = {
        "contexts": {},
        "invocations": {},
        "joins": {
            "join-1": {
                "mode": "all",
                "satisfied": True,
                "completed": ["child-1"],
            }
        },
        "communication_edges": [],
    }

    reasons = validate_predictive_causal_certificate(
        certificate,
        current_graph,
        current_model_version="frontier-v1",
    )

    assert reasons == ("join_revision:join-1",)


def test_shared_locked_prefix_does_not_expand_semantic_belief_scope() -> None:
    policy_input = _input(capacity=1_000, reserved=100, include_cpu_target=True)
    graph = _graph()
    predictions = {"invocation-target": _prediction().to_dict()}
    context_ids = {"ctx-target"}
    for index in range(40):
        workflow_id = f"workflow-peer-{index}"
        context_id = f"ctx-peer-{index}"
        invocation_id = f"invocation-peer-{index}"
        graph.workflows[workflow_id] = WorkflowRecord(
            workflow_id, 0.0, invocation_ids={invocation_id}
        )
        graph.contexts[context_id] = ContextRecord(
            workflow_id,
            context_id,
            0,
            0.0,
            100.0,
            invocation_ids={invocation_id},
        )
        graph.invocations[invocation_id] = InvocationRecord(
            workflow_id=workflow_id,
            invocation_id=invocation_id,
            context_id=context_id,
            agent_definition_id="coder",
            agent_instance_id=f"coder-{index + 1}",
            state=InvocationState.WAIT_TOOL,
            created_ts_ms=0.0,
            updated_ts_ms=100.0,
            active_tool_family="shell",
            active_tool_start_ms=90.0,
        )
        predictions[invocation_id] = replace(
            _prediction(), invocation_id=invocation_id
        ).to_dict()
        context_ids.add(context_id)
    shared_prefix = replace(
        policy_input.physical_kv.bundles[0],
        bundle_id="shared-system-prefix",
        owner_context_ids=tuple(sorted(context_ids)),
        scope="shared_subtree",
        locked_bytes=policy_input.physical_kv.bundles[0].gpu_bytes,
        actionable=False,
        blocker_codes=("node_locked",),
    )
    policy_input = replace(
        policy_input,
        physical_kv=replace(
            policy_input.physical_kv,
            bundles=(shared_prefix, *policy_input.physical_kv.bundles[1:]),
        ),
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                predictions,
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-b", "prefill", 32),
            _service_row("decode-a", "decode", 16),
            _service_row("decode-b", "decode", 16),
        ]
    )

    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            kv_bytes_per_token=1,
        ),
    ).evaluate(
        policy_input,
        graph=graph,
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    assert result.status == "evaluated"
    assert not any(
        reason.startswith("belief_compose_failed")
        for reason in result.blocked_reasons
    )


def test_no_candidate_gate_returns_before_belief_composition() -> None:
    graph = _graph()
    graph.invocations["invocation-target"].state = InvocationState.RUNNING_LLM
    policy_input = _attach_graph(
        _input(capacity=1_000, reserved=100, include_cpu_target=False),
        graph,
    )
    observer = PredictiveRiskShadowObserver(
        GPUServiceCurveModel(minimum_support=1),
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            kv_bytes_per_token=1,
        ),
    )
    observer.composer.compose = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("belief compose must not run")
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)

    result = observer.evaluate(
        policy_input,
        graph=graph,
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    assert result.status == "skipped"
    assert result.blocked_reasons == ("no_action_specific_candidate",)
    assert result.planning_ms < 1.0


def test_eligibility_reads_nested_runtime_rccg_snapshot() -> None:
    graph = _graph()
    policy_input = _input(
        capacity=1_000,
        reserved=100,
        include_cpu_target=True,
    )
    policy_input = replace(
        policy_input,
        runtime_graph=replace(
            policy_input.runtime_graph,
            graph_version=graph.graph_version,
            state={"rccg": graph.snapshot()},
        ),
    )

    eligibility = PredictiveEligibilityIndex().probe(policy_input)

    assert eligibility.has_candidate
    assert eligibility.prefetch_targets[0].context_id == "ctx-target"


def _radix_extent(
    extent_id: str,
    owners: tuple[str, ...],
    *,
    size: int,
    children: tuple[str, ...] = (),
    blockers: tuple[str, ...] = (),
) -> PhysicalBundleSnapshot:
    return PhysicalBundleSnapshot(
        bundle_id=f"bundle-{extent_id}",
        owner_context_ids=owners,
        scope="exclusive_suffix" if len(owners) == 1 else "shared_subtree",
        physical_unique_bytes=size,
        gpu_bytes=size,
        cpu_bytes=0,
        marginal_reclaimable_bytes=0 if blockers else size,
        closure_bytes=size,
        locked_bytes=size if "node_locked" in blockers else 0,
        residency="gpu_only",
        generation_fingerprint=f"generation-{extent_id}",
        extent_ids=(extent_id,),
        lease_kind="wait_tool",
        actionable=not blockers,
        blocker_codes=blockers,
        child_extent_ids=children,
    )


def test_prepare_shadow_absorbs_descendant_closure_without_claiming_child_bytes() -> None:
    graph = _graph()
    parent = _radix_extent(
        "parent",
        ("ctx-target",),
        size=300,
        children=("child",),
        blockers=("descendant_closure",),
    )
    child = _radix_extent("child", ("ctx-child",), size=200)
    base = _attach_graph(_input(capacity=1_000, reserved=0), graph)
    policy_input = replace(
        base,
        physical_kv=replace(
            base.physical_kv,
            gpu_bytes=500,
            cpu_bytes=0,
            bundles=(parent, child),
        ),
        resources=replace(base.resources, hbm_used_bytes=500, hbm_reserved_bytes=0),
    )

    eligibility = PredictiveEligibilityIndex().probe(policy_input)

    victim = next(
        item
        for item in eligibility.prepare_host_victims
        if item.context_id == "ctx-target"
    )
    assert victim.shadow_bytes == 500
    assert victim.reclaimable_bytes == 300


def test_prepare_shadow_rejects_running_descendant_owner() -> None:
    graph = _graph()
    parent = _radix_extent(
        "parent",
        ("ctx-target",),
        size=300,
        children=("child",),
        blockers=("descendant_closure",),
    )
    child = _radix_extent(
        "child",
        ("ctx-child",),
        size=200,
        blockers=("owner_running",),
    )
    base = _attach_graph(_input(capacity=1_000, reserved=0), graph)
    policy_input = replace(
        base,
        physical_kv=replace(
            base.physical_kv,
            gpu_bytes=500,
            cpu_bytes=0,
            bundles=(parent, child),
        ),
        resources=replace(base.resources, hbm_used_bytes=500, hbm_reserved_bytes=0),
    )

    eligibility = PredictiveEligibilityIndex().probe(policy_input)

    assert not eligibility.prepare_host_victims


def test_eligibility_trigger_tracks_material_belief_bucket_change() -> None:
    graph = _graph()
    prediction = _prediction()
    policy_input = _attach_graph(
        _input(capacity=1_000, reserved=100, include_cpu_target=True), graph
    )
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    index = PredictiveEligibilityIndex()
    first = index.probe(policy_input)
    changed_prediction = replace(
        prediction,
        next_output_tokens=_distribution(512),
    )
    changed = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {changed_prediction.invocation_id: changed_prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )

    second = index.probe(changed)

    assert first.trigger_signature != second.trigger_signature


def test_current_feasible_and_future_safe_prefetch_passes_hbm_constraint() -> None:
    graph = _graph()
    policy_input = _attach_graph(
        _input(capacity=1_100, reserved=100, include_cpu_target=True), graph
    )
    prediction = _prediction()
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-b", "prefill", 32),
            _service_row("decode-a", "decode", 16),
            _service_row("decode-b", "decode", 16),
        ]
    )

    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            kv_bytes_per_token=1,
        ),
    ).evaluate(
        policy_input,
        graph=graph,
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    summary = next(
        item
        for item in result.candidate_summaries
        if ":prefetch:" in str(item["package_id"])
    )
    assert summary["future_hbm_feasibility_probability"] == 1.0
    assert "deterministic_hard_constraint" not in summary["reasons"]
    assert "future_hbm_chance_constraint" not in summary["reasons"]


def test_reclaim_then_prefetch_is_physically_feasible_under_pressure() -> None:
    graph = _graph()
    _add_waiting_victim(graph)
    policy_input = _input(capacity=800, reserved=100, include_cpu_target=True)
    bundles = tuple(
        replace(bundle, cpu_bytes=0)
        if bundle.bundle_id == "old"
        else bundle
        for bundle in policy_input.physical_kv.bundles
    )
    target_prediction = _prediction()
    old_prediction = replace(
        _prediction(), invocation_id="invocation-old"
    )
    policy_input = _attach_graph(
        replace(
            policy_input,
            physical_kv=replace(policy_input.physical_kv, bundles=bundles),
            optional_metadata={
                "frontier_predictions": MetadataValue(
                    MetadataSource.PREDICTED,
                    {
                        target_prediction.invocation_id: target_prediction.to_dict(),
                        old_prediction.invocation_id: old_prediction.to_dict(),
                    },
                    "test-frontier",
                ),
                "frontier_prediction_model_version": MetadataValue(
                    MetadataSource.PREDICTED,
                    "frontier-test-v1",
                    "test-frontier",
                ),
            },
        ),
        graph,
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("prefill-b", "prefill", 32),
            _service_row("decode-a", "decode", 16),
            _service_row("decode-b", "decode", 16),
        ]
    )

    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=6,
            kv_bytes_per_token=1,
        ),
    ).evaluate(
        policy_input,
        graph=graph,
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=0,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )

    full_prefetch = next(
        item
        for item in result.candidate_summaries
        if ":prefetch:" in str(item["package_id"])
    )
    joint = next(
        item
        for item in result.candidate_summaries
        if ":reclaim-prefetch:" in str(item["package_id"])
    )
    assert "deterministic_hard_constraint" in full_prefetch["reasons"]
    assert "deterministic_hard_constraint" not in joint["reasons"]
    assert "future_hbm_chance_constraint" not in joint["reasons"]


def test_action_certificate_ignores_unrelated_global_revision() -> None:
    graph = _graph()
    policy_input = _attach_graph(
        _input(capacity=1_100, reserved=100, include_cpu_target=True), graph
    )
    prediction = _prediction()
    policy_input = replace(
        policy_input,
        optional_metadata={
            "frontier_predictions": MetadataValue(
                MetadataSource.PREDICTED,
                {prediction.invocation_id: prediction.to_dict()},
                "test-frontier",
            ),
            "frontier_prediction_model_version": MetadataValue(
                MetadataSource.PREDICTED,
                "frontier-test-v1",
                "test-frontier",
            ),
        },
    )
    source_plan = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    ).plan(policy_input)
    service_model = GPUServiceCurveModel(minimum_support=1)
    service_model.fit(
        [
            _service_row("prefill-a", "prefill", 32),
            _service_row("decode-a", "decode", 16),
        ]
    )
    result = PredictiveRiskShadowObserver(
        service_model,
        PredictiveRiskShadowConfig(
            particle_count=16,
            top_k=4,
            max_candidates=4,
            kv_bytes_per_token=1,
        ),
    ).evaluate(
        policy_input,
        graph=graph,
        source_plan=source_plan,
        evidence_read_set=PredictiveEvidenceReadSet(
            graph_version=3,
            page_revision=5,
            topology_revision=4,
            fairness_revision=0,
            admission_revision=0,
            transfer_epoch=7,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
            parser_frontier_revision=0,
            model_version="frontier-test-v1",
        ),
    )
    summary = next(
        item
        for item in result.candidate_summaries
        if ":prefetch:" in str(item["package_id"])
    )
    certificate = summary["action_certificate"]
    unrelated_revision = replace(
        policy_input,
        runtime_graph=replace(
            policy_input.runtime_graph,
            graph_version=policy_input.runtime_graph.graph_version + 1,
        ),
    )

    assert validate_predictive_certificate(
        certificate,
        unrelated_revision,
        current_transfer_epoch=7,
    ) == ()

    changed_bundles = tuple(
        replace(bundle, generation_fingerprint="changed-generation")
        if "ctx-target" in bundle.owner_context_ids
        else bundle
        for bundle in policy_input.physical_kv.bundles
    )
    changed_physical = replace(
        policy_input,
        physical_kv=replace(policy_input.physical_kv, bundles=changed_bundles),
    )
    assert any(
        reason.startswith("bundle_generation:")
        for reason in validate_predictive_certificate(
            certificate,
            changed_physical,
            current_transfer_epoch=7,
        )
    )

    graph.invocations["invocation-target"].updated_ts_ms += 1.0
    changed_causal = _attach_graph(policy_input, graph)
    assert any(
        reason.startswith("invocation_revision:invocation-target")
        for reason in validate_predictive_certificate(
            certificate,
            changed_causal,
            current_transfer_epoch=7,
        )
    )
