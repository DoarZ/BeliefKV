from scripts.analyze_predictive_prepare_canary import analyze_canary


def _complete_chain() -> list[dict[str, object]]:
    return [
        {
            "event": "predictive_semantic_intent_committed",
            "action": "prepare_host",
            "intent_id": "intent-1",
            "context_id": "context-1",
            "live_transfer_p90_ms": 12.0,
            "live_morphology_slack_ms": 30.0,
            "predicted_extent_count": 2,
            "live_extent_count": 2,
            "live_shape_fingerprint": "shape-live",
            "live_shape_changed": False,
            "transfer_model_mode": "morphology-aware",
        },
        {
            "event": "online_joint_residency_queued",
            "predictive_intent_id": "intent-1",
            "transaction_id": "transaction-1",
            "command_id": "command-1",
            "copy_bytes": 100,
            "live_extent_count": 2,
            "live_shape_fingerprint": "shape-live",
            "transfer_model_mode": "morphology-aware",
        },
        {
            "event": "transfer_telemetry",
            "command_id": "command-1",
            "status": "completed",
            "actual_bytes": 100,
            "extent_count": 2,
            "submit_ts_ms": 10.0,
            "complete_ts_ms": 20.0,
            "compute_wait_ms": 1.0,
        },
        {
            "event": "transfer_acknowledged",
            "command_id": "command-1",
            "status": "completed",
            "actual_bytes": 100,
        },
        {
            "event": "online_joint_residency_terminal",
            "predictive_intent_id": "intent-1",
            "transaction_id": "transaction-1",
            "command_id": "command-1",
            "status": "completed",
            "actual_bytes": 100,
        },
    ]


def test_zero_actions_do_not_pass_natural_or_attribution_gates() -> None:
    result = analyze_canary(
        (),
        comparison={
            "decision_change_gate": False,
            "decision_relevance_gate": False,
            "shape_action_gate": False,
            "shape_veto_gate": False,
            "selected_action_gate": False,
            "shape_natural_action_available": False,
        },
    )

    assert result["status"] == "gate_not_met"
    assert result["canary_limit_respected"] is True
    assert result["natural_action_available"] is False
    assert result["attribution_chain_complete"] is False
    assert result["transaction_completed"] is False
    assert result["replay_shape_action_gate"] is False
    assert result["replay_shape_veto_gate"] is False


def test_complete_five_stage_chain_is_attributed() -> None:
    result = analyze_canary(
        _complete_chain(),
        comparison={
            "decision_change_gate": True,
            "decision_relevance_gate": True,
            "shape_action_gate": True,
            "shape_veto_gate": False,
            "selected_action_gate": False,
            "shape_natural_action_available": True,
        },
    )

    assert result["status"] == "completed"
    assert result["natural_action_available"] is True
    assert result["attribution_chain_complete"] is True
    assert result["transaction_completed"] is True
    assert result["paired_benefit_available"] is False
    assert result["orphan_predictive_commands"] == []


def test_missing_ack_keeps_chain_incomplete() -> None:
    records = [
        item for item in _complete_chain() if item["event"] != "transfer_acknowledged"
    ]

    result = analyze_canary(
        records,
        comparison={
            "shape_action_gate": True,
            "shape_veto_gate": False,
            "shape_natural_action_available": True,
        },
    )

    assert result["status"] == "incomplete"
    assert result["attribution_chain_complete"] is False
    assert result["transaction_completed"] is False
    assert result["orphan_predictive_commands"] == ["command-1"]


def test_transfer_model_mode_mismatch_keeps_chain_incomplete() -> None:
    records = _complete_chain()
    records[1]["transfer_model_mode"] = "byte-only"

    result = analyze_canary(
        records,
        comparison={
            "shape_action_gate": True,
            "shape_veto_gate": False,
            "shape_natural_action_available": True,
        },
    )

    assert result["status"] == "incomplete"
    assert result["attribution_chain_complete"] is False
    assert result["canaries"][0]["transfer_model_mode_consistent"] is False


def test_veto_does_not_open_shape_prepare_canary() -> None:
    result = analyze_canary(
        (),
        comparison={
            "decision_change_gate": True,
            "decision_relevance_gate": True,
            "shape_action_gate": False,
            "shape_veto_gate": True,
            "selected_action_gate": False,
            "shape_natural_action_available": False,
            "recommended_validation_arm": "byte_only_veto_treatment",
        },
    )

    assert result["status"] == "no_natural_veto"
    assert result["replay_shape_action_gate"] is False
    assert result["replay_shape_veto_gate"] is True
    assert result["replay_online_canary_gate"] is False
    assert result["recommended_validation_arm"] == "byte_only_veto_treatment"


def test_unsupported_paired_veto_is_not_an_executable_morphology_result() -> None:
    result = analyze_canary(
        (
            {
                "event": "predictive_risk_shadow",
                "predictive_intent": {"intent_id": "intent-1"},
                "paired_prepare_veto": {
                    "passed": True,
                    "counterfactual_rejection_reasons": [
                        "shape_unsupported",
                        "cvar_risk_budget",
                    ],
                },
            },
            {
                "event": "predictive_semantic_intent_publish_rejected",
                "reasons": ["invocation_revision:child"],
            },
        ),
        comparison={"shape_action_gate": False, "shape_veto_gate": True},
    )

    assert result["status"] == "no_shape_supported_veto"
    assert result["paired_veto_passed_count"] == 1
    assert result["shape_supported_veto_passed_count"] == 0
    assert result["counterfactual_shape_unsupported_count"] == 1
    assert result["unsupported_context_shape_count"] == 1
    assert result["predictive_intent_publish_rejected_count"] == 1


def test_supported_veto_rejected_at_safe_point_is_not_executable() -> None:
    result = analyze_canary(
        (
            {
                "event": "predictive_risk_shadow",
                "predictive_intent": {"intent_id": "intent-1"},
                "paired_prepare_veto": {
                    "passed": True,
                    "counterfactual_shape_supported": True,
                    "counterfactual_rejection_reasons": ["cvar_risk_budget"],
                },
            },
            {"event": "predictive_semantic_intent_published"},
            {
                "event": "predictive_semantic_intent_rejected",
                "reasons": ["transfer_cannot_finish_before_low_window"],
            },
        ),
        comparison={"shape_action_gate": False, "shape_veto_gate": True},
    )

    assert result["status"] == "no_natural_executable_veto"
    assert result["shape_supported_veto_passed_count"] == 1
    assert result["predictive_intent_published_count"] == 1
    assert result["predictive_intent_safe_point_rejected_count"] == 1


def test_veto_marks_shape_prepare_as_unauthorized() -> None:
    result = analyze_canary(
        _complete_chain(),
        comparison={
            "shape_action_gate": False,
            "shape_veto_gate": True,
            "selected_action_gate": False,
            "shape_natural_action_available": False,
        },
    )

    assert result["status"] == "unauthorized_predictive_prepare"
    assert result["shape_prepare_authorized"] is False
    assert result["attribution_chain_complete"] is True
    assert result["transaction_completed"] is True


def test_veto_treatment_requires_paired_authority_evidence() -> None:
    records = _complete_chain()
    for record in records[:2]:
        record["transfer_model_mode"] = "byte-only"
        record["predictive_prepare_authority_gate"] = "byte-only-veto"
    records[0]["counterfactual_rejection_reasons"] = ["cvar_risk_budget"]
    records[1]["counterfactual_rejection_reasons"] = ["cvar_risk_budget"]

    result = analyze_canary(
        records,
        comparison={
            "shape_action_gate": False,
            "shape_veto_gate": True,
            "selected_action_gate": True,
            "shape_natural_action_available": True,
        },
    )

    assert result["status"] == "completed"
    assert result["veto_treatment_authorized"] is True
    assert result["action_authorized"] is True
    assert result["attribution_chain_complete"] is True
