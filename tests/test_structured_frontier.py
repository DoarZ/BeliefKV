from __future__ import annotations

import json
from pathlib import Path

import pytest

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.predictor.frontier_belief import (
    BeliefScopeBuilder,
    PredictiveEvidenceReadSet,
)
from beliefkv.predictor.structured_frontier import (
    EmpiricalDistribution,
    evaluate_frontier_model,
    _demand_feature_key,
    FrontierBeliefModel,
    FrontierModelHyperparameters,
    FrontierScenarioComposer,
    LocalFrontierFeatures,
    LocalFrontierPrediction,
    load_decision_rows,
    load_evaluation_rows,
    select_frontier_hyperparameters,
    summarize_training_corpus,
    validate_training_corpus_diversity,
)


def _row(decision: str, remaining: int, target: str = "function_call") -> dict:
    return {
        "schema_version": 2,
        "decision_id": decision,
        "episode_group_id": f"episode-{decision}",
        "split": "development",
        "trigger_kind": "llm_submit",
        "invocations": [
            {
                "invocation_id": "child",
                "agent_definition_id": "worker",
                "state": "running_llm",
                "boundary_history": ["tool"],
                "context_tokens": 4096,
                "current_sequence_tokens": 4096,
                "observed_output_tokens": 0,
            }
        ],
        "labels": [
            {
                "invocation_id": "child",
                "next_boundary_kind": target,
                "remaining_output_tokens": remaining,
                "reentry_prompt_delta_tokens": 128,
                "demand_label_semantics": "token demand only",
                "censored": False,
            }
        ],
    }


def _calibration_row(decision: str, remaining: float) -> dict:
    row = _row(decision, remaining, target="final_answer")
    row["split"] = "calibration"
    return row


def _tool_row(decision: str, duration_ms: float, status: str) -> dict:
    return {
        "schema_version": 2,
        "decision_id": decision,
        "episode_group_id": f"episode-{decision}",
        "split": "development",
        "trigger_kind": "tool_start",
        "trigger_attributes": {"tool_family": "shell"},
        "invocations": [
            {
                "invocation_id": "worker",
                "agent_definition_id": "worker",
                "state": "wait_tool",
                "active_tool_family": "shell",
                "active_tool_elapsed_ms": 0,
                "context_tokens": 4096,
                "current_sequence_tokens": 4096,
                "active_tool_count": 1,
                "backend_pressure": "active_family:1",
            }
        ],
        "labels": [
            {
                "invocation_id": "worker",
                "next_boundary_kind": "tool_end",
                "next_boundary_status": status,
                "next_boundary_delay_ms": duration_ms,
                "censored": status == "censored",
            }
        ],
    }


def _ready_row(decision: str, output_tokens: int) -> dict:
    return {
        "schema_version": 2,
        "decision_id": decision,
        "episode_group_id": f"episode-{decision}",
        "split": "development",
        "trigger_kind": "reactivate",
        "invocations": [
            {
                "invocation_id": "worker",
                "agent_definition_id": "worker",
                "state": "ready",
                "context_tokens": 4096,
                "current_sequence_tokens": 4096,
            }
        ],
        "labels": [
            {
                "invocation_id": "worker",
                "next_boundary_kind": "final_answer",
                "next_output_tokens": output_tokens,
                "censored": False,
            }
        ],
    }


def _wait_tool_row_with_next_output(decision: str, output_tokens: int) -> dict:
    row = _tool_row(decision, duration_ms=100.0, status="success")
    row["labels"][0]["next_output_tokens"] = output_tokens
    return row


def _wait_tool_features() -> LocalFrontierFeatures:
    return LocalFrontierFeatures(
        invocation_id="worker",
        state="wait_tool",
        agent_definition_id="worker",
        boundary_history=("tool",),
        tool_family="shell",
        backend_class="unknown",
        generated_tokens=0,
        elapsed_wait_ms=0.0,
        current_sequence_tokens=4096,
        active_tool_count=1,
        backend_pressure="active_family:1",
    )


def _demand_key(features: LocalFrontierFeatures) -> tuple[str, ...]:
    return _demand_feature_key(
        features.agent_definition_id,
        features.state,
        features.tool_family,
        {
            "current_sequence_tokens": features.current_sequence_tokens,
            "generated_tokens": features.generated_tokens,
            "backend_class": features.backend_class,
        },
    )


def _fixed_prediction(invocation_id: str, decode_tokens: float) -> LocalFrontierPrediction:
    point = EmpiricalDistribution((decode_tokens,), (1.0,), 4.0)
    empty = EmpiricalDistribution.empty()
    return LocalFrontierPrediction(
        invocation_id=invocation_id,
        boundary_distribution={"final": 1.0},
        current_sequence_tokens=4096,
        remaining_decode_tokens=point,
        remaining_external_wait=empty,
        tool_terminal_distribution={"success": 1.0},
        prompt_growth_tokens=empty,
        next_output_tokens=empty,
        support_level="exact",
        calibration_coverage=0.0,
    )


def _write_dataset(
    root: Path,
    *,
    run_id: str,
    split: str,
    decision_id: str,
    formal_training_eligible: bool = True,
) -> Path:
    root.mkdir(parents=True)
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "beliefkv_p6_training_evidence",
                "formal_training_eligible": formal_training_eligible,
                "evaluation_role": "frozen_split_training_evidence",
                "source": {
                    "run_id": run_id,
                    "workload_manifest_sha256": f"manifest-{run_id}",
                    "collection_contract": {
                        "plan_id": "p6-agent-semantics-v1",
                        "split": split,
                        "training_eligible": formal_training_eligible,
                        "runtime_source_stable": True,
                        "runtime_policy": "frozen_p5_observed",
                        "predictor_enabled": False,
                        "predictive_actions_enabled": False,
                    },
                },
                "split_contract": {
                    "source": "explicit frozen split manifest",
                    "development_only": False,
                    "manifest_digest": "frozen-split",
                },
            }
        ),
        encoding="utf-8",
    )
    row = _row(decision_id, 10.0)
    row["split"] = split
    (root / "frontier_decision_points.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    return root


def test_formal_loaders_fail_closed_on_ineligible_dataset(tmp_path: Path) -> None:
    train = _write_dataset(
        tmp_path / "train",
        run_id="run-train",
        split="train",
        decision_id="train-decision",
        formal_training_eligible=False,
    )
    calibration = _write_dataset(
        tmp_path / "calibration",
        run_id="run-calibration",
        split="calibration",
        decision_id="calibration-decision",
        formal_training_eligible=False,
    )

    with pytest.raises(ValueError, match="formal training input is ineligible"):
        load_decision_rows((train,), allowed_splits=("train",))
    with pytest.raises(ValueError, match="formal evaluation input is ineligible"):
        load_evaluation_rows((calibration,), split="calibration")


def test_formal_loader_rejects_p5_development_evidence(tmp_path: Path) -> None:
    root = _write_dataset(
        tmp_path / "p5-w4",
        run_id="p5-run",
        split="train",
        decision_id="p5-decision",
    )
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["collection_contract"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="P6 collection plan"):
        load_decision_rows((root,), allowed_splits=("train",))


def test_formal_diversity_gate_counts_workflows_not_decision_rows() -> None:
    rows = []
    for index in range(200):
        row = _row(f"decision-{index}", 10)
        row.update(
            {
                "run_id": "run-w4",
                "workflow_id": f"workflow-{index % 4}",
                "project": f"project-{index % 2}",
                "instance_id": f"task-{index % 4}",
                "base_commit": f"commit-{index % 4}",
                "split": "train",
            }
        )
        rows.append(row)

    summary = summarize_training_corpus(rows)
    assert summary["decision_point_count"] == 200
    assert summary["workflow_count"] == 4
    assert summary["task_count"] == 4
    with pytest.raises(ValueError, match="workflow memorization"):
        validate_training_corpus_diversity(rows)


@pytest.mark.parametrize(
    "contaminated_feature",
    ("batch_size", "elapsed_gpu_service_ms", "observed_gpu_service_ms"),
)
def test_frontier_fit_rejects_load_coupled_semantic_features(
    contaminated_feature: str,
) -> None:
    row = _row("contaminated", 10)
    row["invocations"][0][contaminated_feature] = 4

    with pytest.raises(ValueError, match="scheduler features are forbidden"):
        FrontierBeliefModel().fit([row])


def test_frontier_fit_allows_load_observations_in_diagnostics_only() -> None:
    row = _row("diagnostic", 10)
    row["invocations"][0]["diagnostics"] = {
        "last_observed_batch_size": 4,
        "observed_gpu_service_ms": 12.5,
    }

    summary = FrontierBeliefModel().fit([row])

    assert summary["observation_counts"]["remaining_decode_demand"] == 1


def test_state_semantic_decode_uses_next_output_for_waiting_states() -> None:
    rows = [_wait_tool_row_with_next_output(f"d{i}", 128) for i in range(4)]
    model = FrontierBeliefModel(
        hyperparameters=FrontierModelHyperparameters(
            empirical_minimum_support=1.0
        )
    )
    summary = model.fit(rows)
    assert (
        summary["observation_counts"]["state_conditional_decode_demand"]
        == 4
    )
    features = _wait_tool_features()
    dist, level = model.decode_demand.predict(_demand_key(features))
    assert level in {"exact", "backoff"}
    assert dist.quantile(0.5) > 0


def test_without_state_semantic_labels_waiting_decode_is_unavailable() -> None:
    # _tool_row carries no next_output_tokens: the wait_tool decode key has no
    # observations and must fail closed to unavailable (previous behavior).
    rows = [_tool_row(f"d{i}", 100.0, "success") for i in range(4)]
    model = FrontierBeliefModel(
        hyperparameters=FrontierModelHyperparameters(
            empirical_minimum_support=1.0
        )
    )
    model.fit(rows)
    features = _wait_tool_features()
    _dist, level = model.decode_demand.predict(_demand_key(features))
    assert level == "unavailable"


def test_formal_loaders_reject_duplicate_runs_and_decisions(tmp_path: Path) -> None:
    first = _write_dataset(
        tmp_path / "first",
        run_id="same-run",
        split="train",
        decision_id="decision-a",
    )
    duplicate_run = _write_dataset(
        tmp_path / "duplicate-run",
        run_id="same-run",
        split="train",
        decision_id="decision-b",
    )
    with pytest.raises(ValueError, match="duplicate source run"):
        load_decision_rows(
            (first, duplicate_run),
            allowed_splits=("train",),
        )

    duplicate_decision = _write_dataset(
        tmp_path / "duplicate-decision",
        run_id="different-run",
        split="train",
        decision_id="decision-a",
    )
    with pytest.raises(ValueError, match="duplicate decision point"):
        load_decision_rows(
            (first, duplicate_decision),
            allowed_splits=("train",),
        )


def test_local_model_roundtrip_preserves_distribution(tmp_path) -> None:
    hyperparameters = FrontierModelHyperparameters(
        boundary_max_order=2,
        boundary_minimum_support=2.0,
        empirical_minimum_support=2.0,
        tool_minimum_support=2.0,
    )
    model = FrontierBeliefModel(
        model_version="dev-v1",
        hyperparameters=hyperparameters,
    )
    summary = model.fit(
        [_row("d1", 10), _row("d2", 20), _row("d3", 15), _row("d4", 25)]
    )
    assert summary["episode_count"] == 4
    features = LocalFrontierFeatures(
        invocation_id="child",
        state="running_llm",
        agent_definition_id="worker",
        boundary_history=("tool",),
        current_sequence_tokens=4096,
    )
    before = model.predict(features)
    path = tmp_path / "model.json"
    model.save(path, metadata={"development_only": True})
    loaded = FrontierBeliefModel.load(path)
    after = loaded.predict(features)
    assert before == after
    assert loaded.hyperparameters == hyperparameters
    assert before.boundary_distribution["tool"] > 0.9
    assert before.calibration_coverage == 0.0


def test_lopo_hyperparameter_selection_is_train_only_and_project_macro() -> None:
    rows = []
    for project, base in (("org/a", 10), ("org/b", 20), ("org/c", 30)):
        for offset in range(4):
            row = _row(f"{project}-{offset}", base + offset)
            row["split"] = "train"
            row["project"] = project
            rows.append(row)

    report = select_frontier_hyperparameters(
        rows,
        candidates=(
            FrontierModelHyperparameters(),
            FrontierModelHyperparameters(
                boundary_max_order=2,
                boundary_minimum_support=2.0,
                empirical_minimum_support=2.0,
                tool_minimum_support=2.0,
            ),
        ),
    )
    assert report["projects"] == ["org/a", "org/b", "org/c"]
    assert report["candidate_count"] == 2
    assert len(report["candidates"][0]["folds"]) == 3
    assert report["selected_hyperparameters"] in [
        item["hyperparameters"] for item in report["candidates"]
    ]

    invalid = [dict(rows[0], split="calibration"), *rows[1:]]
    with pytest.raises(ValueError, match="formal train rows"):
        select_frontier_hyperparameters(invalid)


def test_calibration_does_not_refit_training_counts_and_survives_roundtrip(
    tmp_path,
) -> None:
    model = FrontierBeliefModel(model_version="train-v1")
    model.fit(
        [_row("d1", 10), _row("d2", 20), _row("d3", 15), _row("d4", 25)]
    )
    before = model.to_dict()["components"]
    summary = model.calibrate(
        [
            _calibration_row("c1", 100),
            _calibration_row("c2", 120),
            _calibration_row("c3", 80),
        ],
        target_coverage=0.9,
    )
    assert model.to_dict()["components"] == before
    assert summary["training_counts_refit"] is False
    assert summary["interval_slack"]["remaining_decode_tokens"] > 0

    features = LocalFrontierFeatures(
        invocation_id="child",
        state="running_llm",
        agent_definition_id="worker",
        boundary_history=("tool",),
        current_sequence_tokens=4096,
    )
    prediction = model.predict(features)
    assert prediction.calibration_coverage == 0.9
    assert (
        prediction.calibrated_intervals["remaining_decode_tokens"][1]
        >= 120
    )
    path = tmp_path / "calibrated.json"
    model.save(path)
    assert FrontierBeliefModel.load(path).predict(features) == prediction


def test_calibration_rejects_training_or_test_rows() -> None:
    model = FrontierBeliefModel(model_version="train-v1")
    model.fit(
        [_row("d1", 10), _row("d2", 20), _row("d3", 15), _row("d4", 25)]
    )
    with pytest.raises(ValueError, match="calibration split"):
        model.calibrate([_row("train", 30)])


def test_tool_prediction_conditions_competing_risk_on_elapsed_wait(tmp_path) -> None:
    model = FrontierBeliefModel(model_version="tool-survival-v1")
    model.fit(
        [
            _tool_row("short-error", 10, "error"),
            _tool_row("long-a", 100, "success"),
            _tool_row("long-b", 110, "success"),
            _tool_row("long-c", 120, "success"),
        ]
    )
    features = LocalFrontierFeatures(
        invocation_id="worker",
        state="wait_tool",
        agent_definition_id="worker",
        tool_family="shell",
        elapsed_wait_ms=50,
        current_sequence_tokens=4096,
        active_tool_count=1,
        backend_pressure="active_family:1",
    )
    prediction = model.predict(features)
    assert prediction.tool_terminal_distribution["success"] > 0.85
    assert prediction.remaining_external_wait.quantile(0.5) >= 50

    unsupported_tail = model.predict(
        LocalFrontierFeatures(
            invocation_id="worker",
            state="wait_tool",
            agent_definition_id="worker",
            tool_family="shell",
            elapsed_wait_ms=500,
            current_sequence_tokens=4096,
            active_tool_count=1,
            backend_pressure="active_family:1",
        )
    )
    assert unsupported_tail.remaining_external_wait.values == ()
    assert "tool_unavailable" in unsupported_tail.ood_reasons

    path = tmp_path / "tool-survival.json"
    model.save(path)
    assert FrontierBeliefModel.load(path).predict(features) == prediction


def test_reentry_state_learns_next_call_output_demand_without_service_time() -> None:
    model = FrontierBeliefModel(model_version="reentry-v1")
    model.fit(
        [
            _ready_row("r1", 80),
            _ready_row("r2", 96),
            _ready_row("r3", 112),
            _ready_row("r4", 128),
        ]
    )
    prediction = model.predict(
        LocalFrontierFeatures(
            invocation_id="worker",
            state="ready",
            agent_definition_id="worker",
            current_sequence_tokens=4096,
        )
    )
    # State-semantic decode (deepseek): ready invocations now learn the next
    # LLM output demand as their decode target instead of falling to global.
    assert prediction.remaining_decode_tokens.quantile(0.5) > 0
    assert prediction.next_output_tokens.quantile(0.5) > 0


def test_episode_weighted_evaluation_reports_calibration_and_ood() -> None:
    model = FrontierBeliefModel(model_version="train-v1")
    model.fit(
        [_row("d1", 10), _row("d2", 20), _row("d3", 15), _row("d4", 25)]
    )
    calibration = [
        _calibration_row("c1", 30),
        _calibration_row("c2", 35),
        _calibration_row("c3", 40),
    ]
    model.calibrate(calibration)
    metrics = evaluate_frontier_model(model, calibration)
    assert metrics["splits"] == ["calibration"]
    assert metrics["classification"]["boundary"]["episode_weight"] == 3
    assert 0 <= metrics["classification"]["boundary"]["ece_10"] <= 1
    assert (
        0
        <= metrics["scalar"]["remaining_decode_tokens"][
            "calibrated_interval_coverage"
        ]
        <= 1
    )


def test_composer_applies_known_join_all_instead_of_learning_it() -> None:
    graph = RuntimeCausalContextGraph()
    sequence = 0

    def emit(kind: RuntimeEventKind, **kwargs) -> None:
        nonlocal sequence
        sequence += 1
        graph.apply(
            RuntimeEvent(
                event_id=f"e{sequence}",
                ts_ms=float(sequence),
                kind=kind,
                workflow_id="workflow",
                **kwargs,
            )
        )

    emit(RuntimeEventKind.WORKFLOW_START)
    emit(RuntimeEventKind.INVOCATION_CREATE, invocation_id="parent", context_id="p")
    emit(RuntimeEventKind.INVOCATION_CREATE, invocation_id="a", context_id="a")
    emit(RuntimeEventKind.INVOCATION_CREATE, invocation_id="b", context_id="b")
    emit(
        RuntimeEventKind.JOIN_CREATE,
        join_id="join",
        member_invocation_ids=("a", "b"),
    )
    emit(RuntimeEventKind.JOIN_WAIT, invocation_id="parent", join_id="join")
    scope = BeliefScopeBuilder().build(graph, ("parent", "a", "b"))
    model = FrontierBeliefModel(model_version="dev-v1")
    model.fit(
        [_row("d1", 10), _row("d2", 20), _row("d3", 15), _row("d4", 25)]
    )
    predictions = {
        invocation_id: model.predict(
            LocalFrontierFeatures(
                invocation_id=invocation_id,
                state=graph.invocations[invocation_id].state.value,
                agent_definition_id="worker",
                current_sequence_tokens=4096,
            )
        )
        for invocation_id in scope.invocation_ids
    }
    readset = PredictiveEvidenceReadSet(
        graph_version=graph.graph_version,
        page_revision=0,
        topology_revision=0,
        fairness_revision=0,
        admission_revision=0,
        transfer_epoch=0,
        obligation_revision=0,
        lease_revision=0,
        grace_revision=0,
        parser_frontier_revision=0,
        model_version="dev-v1",
    )
    belief = FrontierScenarioComposer(particle_count=16, top_k=4).compose(
        graph=graph,
        scope=scope,
        local_predictions=predictions,
        generated_ts_ms=10,
        evidence_read_set=readset,
    )
    parent = next(
        item
        for item in belief.scenarios[0].outcomes
        if item.invocation_id == "parent"
    )
    assert parent.dependency_mode.value == "join_all"
    assert parent.dependency_invocation_ids == ("a", "b")
    assert parent.join_id == "join"
    assert belief.other_probability_mass + sum(
        item.probability_mass for item in belief.scenarios
    ) == 1.0


def test_composer_applies_blocking_child_and_message_dependencies() -> None:
    graph = RuntimeCausalContextGraph()
    sequence = 0

    def emit(kind: RuntimeEventKind, **kwargs) -> None:
        nonlocal sequence
        sequence += 1
        graph.apply(
            RuntimeEvent(
                event_id=f"dependency-{sequence}",
                ts_ms=float(sequence),
                kind=kind,
                workflow_id="workflow",
                **kwargs,
            )
        )

    emit(RuntimeEventKind.WORKFLOW_START)
    emit(RuntimeEventKind.INVOCATION_CREATE, invocation_id="parent", context_id="p")
    emit(RuntimeEventKind.INVOCATION_CREATE, invocation_id="child", context_id="c")
    emit(
        RuntimeEventKind.CALL,
        invocation_id="parent",
        target_invocation_id="child",
    )
    scope = BeliefScopeBuilder().build(graph, ("parent", "child"))
    readset = PredictiveEvidenceReadSet(
        graph_version=graph.graph_version,
        page_revision=0,
        topology_revision=0,
        fairness_revision=0,
        admission_revision=0,
        transfer_epoch=0,
        obligation_revision=0,
        lease_revision=0,
        grace_revision=0,
        parser_frontier_revision=0,
        model_version="dependency-v1",
    )
    composer = FrontierScenarioComposer(particle_count=4, top_k=2)
    belief = composer.compose(
        graph=graph,
        scope=scope,
        local_predictions={
            "parent": _fixed_prediction("parent", 0),
            "child": _fixed_prediction("child", 25),
        },
        generated_ts_ms=10,
        evidence_read_set=readset,
    )
    parent = next(
        item
        for item in belief.scenarios[0].outcomes
        if item.invocation_id == "parent"
    )
    assert parent.dependency_mode.value == "join_all"
    assert parent.dependency_invocation_ids == ("child",)
    assert parent.remaining_decode_tokens == 0

    message_graph = RuntimeCausalContextGraph()
    graph = message_graph
    sequence = 0
    emit(RuntimeEventKind.WORKFLOW_START)
    emit(RuntimeEventKind.INVOCATION_CREATE, invocation_id="source", context_id="s")
    emit(RuntimeEventKind.INVOCATION_CREATE, invocation_id="producer", context_id="t")
    emit(
        RuntimeEventKind.HANDOFF,
        invocation_id="source",
        target_invocation_id="producer",
    )
    scope = BeliefScopeBuilder().build(graph, ("source", "producer"))
    readset = PredictiveEvidenceReadSet(
        graph_version=graph.graph_version,
        page_revision=0,
        topology_revision=0,
        fairness_revision=0,
        admission_revision=0,
        transfer_epoch=0,
        obligation_revision=0,
        lease_revision=0,
        grace_revision=0,
        parser_frontier_revision=0,
        model_version="dependency-v1",
    )
    belief = composer.compose(
        graph=graph,
        scope=scope,
        local_predictions={
            "source": _fixed_prediction("source", 0),
            "producer": _fixed_prediction("producer", 40),
        },
        generated_ts_ms=10,
        evidence_read_set=readset,
    )
    source = next(
        item
        for item in belief.scenarios[0].outcomes
        if item.invocation_id == "source"
    )
    assert source.dependency_mode.value == "producer"
    assert source.dependency_invocation_ids == ("producer",)
