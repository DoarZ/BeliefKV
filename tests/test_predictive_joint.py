from beliefkv.policy.predictive_joint import (
    PackageScenarioEvaluation,
    PredictiveActionKind,
    PredictiveActionPackage,
    PredictivePlanEnvelope,
    RiskPlanningTelemetry,
    ScenarioCost,
    ScenarioRiskPlanner,
    ScenarioRiskPlannerConfig,
)
from beliefkv.policy.online_joint import (
    ActionGroup,
    ActionGroupAtomicity,
    ActionGroupResourceCertificate,
    ActionSlice,
)
from beliefkv.predictor.frontier_belief import (
    BeliefScope,
    BoundaryEvent,
    CausalAtom,
    CausalAtomKind,
    DemandPhase,
    DemandScenario,
    DependencyMode,
    ExternalDemandSegment,
    FrontierDemandOutcome,
    FrontierBeliefSnapshot,
    OtherResidualPolicy,
    PredictiveEvidenceReadSet,
)


def _belief(*, finite_other: bool) -> FrontierBeliefSnapshot:
    atom = CausalAtom(
        atom_id="atom",
        kind=CausalAtomKind.SINGLE,
        invocation_ids=("invocation",),
    )
    scope = BeliefScope(
        scope_id="scope",
        graph_version=1,
        active_invocation_ids=("invocation",),
        included_atoms=(atom,),
        other_atoms=(),
        modeled_cost=1,
    )
    scenario = DemandScenario(
        scenario_id="likely",
        outcomes=(
            FrontierDemandOutcome(
                invocation_id="invocation",
                boundary_event=BoundaryEvent.TOOL,
                dependency_mode=DependencyMode.EXTERNAL,
                phase=DemandPhase.EXTERNAL,
                current_sequence_tokens=4096,
                remaining_decode_tokens=5,
                prompt_growth_tokens=128,
                next_output_tokens=16,
                external_segments=(
                    ExternalDemandSegment("tool", "shell", 20.0),
                ),
            ),
        ),
        probability_mass=0.8,
    )
    evidence = PredictiveEvidenceReadSet(
        graph_version=1,
        page_revision=1,
        topology_revision=1,
        fairness_revision=1,
        admission_revision=1,
        transfer_epoch=1,
        obligation_revision=1,
        lease_revision=1,
        grace_revision=1,
        parser_frontier_revision=1,
        model_version="untrained-schema-v1",
    )
    return FrontierBeliefSnapshot(
        belief_id="belief",
        generated_ts_ms=10.0,
        scope=scope,
        scenarios=(scenario,),
        other_probability_mass=0.2,
        calibration_coverage=0.9,
        support_level="exact",
        ood_reasons=(),
        evidence_read_set=evidence,
        other_policy=OtherResidualPolicy(finite_risk_bound=finite_other),
    )


def _evaluation(
    package: PredictiveActionPackage,
    *,
    likely_delay: float,
    other_delay: float,
    liveness: bool = True,
) -> PackageScenarioEvaluation:
    return PackageScenarioEvaluation(
        package=package,
        costs_by_scenario={
            "likely": ScenarioCost(
                action_unlock_delay_ms=likely_delay,
                workflow_service_lag_ms=0.0,
                liveness_path_proven=liveness,
            )
        },
        other_cost=ScenarioCost(
            action_unlock_delay_ms=other_delay,
            workflow_service_lag_ms=0.0,
            liveness_path_proven=liveness,
        ),
    )


def test_unbounded_other_allows_prepare_but_rejects_prefetch() -> None:
    belief = _belief(finite_other=False)
    baseline = _evaluation(
        PredictiveActionPackage("a0", PredictiveActionKind.OBSERVED_BASELINE),
        likely_delay=20.0,
        other_delay=20.0,
    )
    prepare = _evaluation(
        PredictiveActionPackage(
            "prepare", PredictiveActionKind.PREPARE_HOST, ("context",)
        ),
        likely_delay=5.0,
        other_delay=20.0,
    )
    prefetch = _evaluation(
        PredictiveActionPackage(
            "prefetch", PredictiveActionKind.PREFETCH_GPU, ("context",)
        ),
        likely_delay=1.0,
        other_delay=20.0,
    )

    decision = ScenarioRiskPlanner(
        ScenarioRiskPlannerConfig(risk_budget_ms=100.0)
    ).select(belief, baseline, (prepare, prefetch))

    assert decision.selected_package_id == "prepare"
    summaries = {item.package_id: item for item in decision.summaries}
    assert summaries["prepare"].eligible
    assert "other_has_no_finite_risk_bound" in summaries["prefetch"].reasons


def test_prepare_host_reports_future_hbm_without_rejecting_shadow() -> None:
    belief = _belief(finite_other=True)
    baseline = _evaluation(
        PredictiveActionPackage("a0", PredictiveActionKind.OBSERVED_BASELINE),
        likely_delay=20.0,
        other_delay=20.0,
    )
    package = PredictiveActionPackage(
        "prepare", PredictiveActionKind.PREPARE_HOST, ("context",)
    )
    overflow = ScenarioCost(
        action_unlock_delay_ms=1.0,
        workflow_service_lag_ms=0.0,
        future_hbm_feasible=False,
        future_feasible=True,
        future_hbm_peak_bytes=2_000,
        future_hbm_overflow_bytes=1_000,
    )
    prepare = PackageScenarioEvaluation(
        package=package,
        costs_by_scenario={"likely": overflow},
        other_cost=overflow,
    )

    decision = ScenarioRiskPlanner(
        ScenarioRiskPlannerConfig(risk_budget_ms=100.0)
    ).select(belief, baseline, (prepare,))

    assert decision.selected_package_id == "prepare"
    summary = decision.summaries[0]
    assert summary.future_hbm_feasibility_probability == 0.0
    assert summary.worst_future_hbm_overflow_bytes == 1_000
    assert "future_hbm_chance_constraint" not in summary.reasons


def test_unproven_restore_liveness_is_a_deterministic_rejection() -> None:
    belief = _belief(finite_other=True)
    baseline = _evaluation(
        PredictiveActionPackage("a0", PredictiveActionKind.OBSERVED_BASELINE),
        likely_delay=20.0,
        other_delay=20.0,
    )
    candidate = _evaluation(
        PredictiveActionPackage(
            "prefetch", PredictiveActionKind.PREFETCH_GPU, ("context",)
        ),
        likely_delay=1.0,
        other_delay=1.0,
        liveness=False,
    )

    decision = ScenarioRiskPlanner(
        ScenarioRiskPlannerConfig(risk_budget_ms=100.0)
    ).select(belief, baseline, (candidate,))

    assert decision.selected_package_id == "a0"
    assert "restore_liveness_path_unproven" in decision.summaries[0].reasons


def test_predictive_envelope_requires_each_group_to_read_its_belief() -> None:
    belief = _belief(finite_other=True)
    group = ActionGroup(
        group_id="prepare-group",
        atomicity=ActionGroupAtomicity.ALL_OR_NOTHING,
        actions=(ActionSlice("prepare", "residency", "context", (), True),),
        dependency_dag=(),
        resource_certificate=ActionGroupResourceCertificate(),
        compensation=("release speculative host shadow",),
        committed=True,
        evidence_read_set=(
            ("belief_id", belief.belief_id),
            ("model_version", belief.evidence_read_set.model_version),
        ),
    )

    envelope = PredictivePlanEnvelope(
        envelope_id="envelope",
        belief=belief,
        source_joint_plan_id="joint-a0",
        selected_package_id="prepare",
        action_groups=(group,),
        generated_ts_ms=11.0,
        telemetry=RiskPlanningTelemetry(0.1, (0.2,), 0.4),
    )

    assert envelope.action_groups == (group,)
