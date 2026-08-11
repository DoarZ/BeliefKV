from beliefkv.experiments.frontier_retraction_canary import (
    analyze_frontier_retraction_canary,
)


def _records() -> list[dict[str, object]]:
    return [
        {"event": "frontier_retraction_shadow", "decision_changed": True},
        {"event": "frontier_retraction_canary_committed"},
        {
            "event": "frontier_retraction_canary_bound",
            "transaction_id": "retraction-1",
            "request_ids": ["victim"],
            "replacement_request_ids": ["replacement"],
        },
        {
            "event": "running_retraction_planned",
            "transaction_id": "retraction-1",
            "request_ids": ["victim"],
        },
        {
            "event": "running_retraction_transaction_completed",
            "transaction_id": "retraction-1",
            "ts_ms": 20.0,
        },
        {
            "event": "restore_obligation_terminal",
            "request_id": "victim",
            "state": "satisfied",
            "ts_ms": 30.0,
        },
        {
            "event": "gpu_service_sample",
            "request_ids": ["replacement"],
            "ts_ms": 25.0,
        },
        {
            "event": "gpu_service_sample",
            "request_ids": ["victim"],
            "ts_ms": 35.0,
        },
    ]


def test_frontier_retraction_canary_requires_restore_and_service() -> None:
    result = analyze_frontier_retraction_canary(_records())

    assert result["status"] == "completed"
    assert result["correctness_gate_passed"] is True


def test_frontier_retraction_canary_reports_shadow_only() -> None:
    result = analyze_frontier_retraction_canary(
        [{"event": "frontier_retraction_shadow", "decision_changed": True}]
    )

    assert result["status"] == "shadow_only"
    assert result["natural_action_available"] is False


def test_frontier_retraction_canary_rejects_missing_victim_service() -> None:
    records = [
        item
        for item in _records()
        if not (
            item["event"] == "gpu_service_sample"
            and item["request_ids"] == ["victim"]
        )
    ]

    result = analyze_frontier_retraction_canary(records)

    assert result["status"] == "incomplete"
    assert result["actions"][0]["victim_service_after_restore"] == {
        "victim": False
    }
