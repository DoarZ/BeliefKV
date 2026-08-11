from beliefkv.experiments.predictive_joint_canary import (
    analyze_predictive_prepare_canary,
)


def _complete_chain() -> list[dict[str, object]]:
    return [
        {
            "event": "predictive_risk_shadow",
            "predictive_intent": {"intent_id": "intent-1"},
        },
        {
            "event": "predictive_semantic_intent_published",
            "intent_id": "intent-1",
        },
        {
            "event": "predictive_semantic_intent_committed",
            "intent_id": "intent-1",
            "action": "prepare_host",
            "context_id": "context-1",
            "live_extent_count": 3,
        },
        {
            "event": "online_joint_residency_queued",
            "predictive_intent_id": "intent-1",
            "transaction_id": "transaction-1",
            "command_id": "command-1",
            "copy_bytes": 1024,
            "live_extent_count": 3,
        },
        {
            "event": "transfer_telemetry",
            "command_id": "command-1",
            "status": "completed",
            "actual_bytes": 1024,
            "extent_count": 3,
        },
        {
            "event": "transfer_acknowledged",
            "command_id": "command-1",
            "status": "completed",
            "actual_bytes": 1024,
        },
        {
            "event": "online_joint_residency_terminal",
            "predictive_intent_id": "intent-1",
            "transaction_id": "transaction-1",
            "command_id": "command-1",
            "status": "completed",
            "actual_bytes": 1024,
        },
        {
            "event": "predictive_action_outcome",
            "intent_id": "intent-1",
            "command_id": "command-1",
            "state": "useful",
            "actual_bytes": 1024,
            "reason": "reactive_commit_consumed_shadow",
        },
    ]


def test_predictive_prepare_canary_requires_complete_attribution_chain() -> None:
    result = analyze_predictive_prepare_canary(_complete_chain())

    assert result["status"] == "completed"
    assert result["canary_limit_respected"] is True
    assert result["attribution_chain_complete"] is True
    assert result["transaction_completed"] is True
    assert result["outcome_counts"] == {"useful": 1}


def test_predictive_prepare_canary_accepts_natural_no_action() -> None:
    result = analyze_predictive_prepare_canary(
        [{"event": "predictive_risk_shadow", "selected_action": "observed_baseline"}]
    )

    assert result["status"] == "no_positive_action"
    assert result["natural_action_available"] is False
    assert result["canary_limit_respected"] is True


def test_predictive_prepare_canary_rejects_broken_command_chain() -> None:
    records = _complete_chain()
    records = [item for item in records if item["event"] != "transfer_acknowledged"]

    result = analyze_predictive_prepare_canary(records)

    assert result["status"] == "incomplete"
    assert result["attribution_chain_complete"] is False
    assert result["actions"][0]["stage_presence"]["ack"] is False
