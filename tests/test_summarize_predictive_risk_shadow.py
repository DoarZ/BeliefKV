from scripts.summarize_predictive_risk_shadow import (
    _morphology_audit,
    _prepare_morphology_record,
)


def _candidate(
    *,
    generation: str = "generation-a",
    context_id: str = "context-parent",
    invocation_id: str = "invocation-parent",
) -> dict[str, object]:
    return {
        "action": "prepare_host",
        "package_id": "joint-1:prepare:context-parent",
        "action_certificate": {
            "source_snapshot_id": "snapshot-1",
            "target_context_id": context_id,
            "context_epochs": [[context_id, 3]],
            "invocation_evidence": [
                ["invocation-child", "running_llm", 10.0, "join-1"],
                [invocation_id, "wait_join", 5.0, "join-1"],
            ],
            "join_evidence": [["join-1", "all", False, []]],
            "bundle_evidence": [
                [f"extent-{index}", generation, 64 * 1024 * 1024, 0]
                for index in range(22)
            ],
            "transfer_service_evidence": [1024 * 1024, 2 * 1024 * 1024, 1.0],
        },
    }


def test_prepare_morphology_extracts_wait_join_shape() -> None:
    record = _prepare_morphology_record(
        {
            "target_context_id": "context-parent",
            "target_invocation_id": "invocation-parent",
        },
        _candidate(),
    )

    assert record is not None
    assert record["invocation_state"] == "wait_join"
    assert record["join_state"] == "all:pending"
    assert record["extent_count"] == 22
    assert record["copy_bytes"] == 22 * 64 * 1024 * 1024
    assert record["extent_bytes_min"] == 64 * 1024 * 1024
    assert record["extent_bytes_p50"] == 64 * 1024 * 1024
    assert record["extent_bytes_max"] == 64 * 1024 * 1024
    assert record["small_extent_ratio"] == 0.0
    assert record["closure_depth"] is None
    assert float(record["candidate_duration_ms"]) > 1.0


def test_prepare_morphology_does_not_borrow_another_candidate_state() -> None:
    candidate = _candidate()
    certificate = candidate["action_certificate"]
    certificate["target_context_id"] = "deepagents-context:child"
    certificate["context_epochs"] = [["deepagents-context:child", 1]]
    certificate["invocation_evidence"] = [
        ["deepagents-invocation:child", "wait_tool", 10.0, None],
        ["invocation-primary", "wait_join", 5.0, "join-1"],
    ]

    record = _prepare_morphology_record(
        {
            "target_context_id": "context-primary",
            "target_invocation_id": "invocation-primary",
        },
        candidate,
    )

    assert record is not None
    assert record["target_invocation_id"] == "deepagents-invocation:child"
    assert record["invocation_state"] == "wait_tool"
    assert record["join_state"] == "none"


def test_morphology_audit_does_not_treat_generation_churn_as_independence() -> None:
    risk_record = {
        "target_context_id": "context-parent",
        "target_invocation_id": "invocation-parent",
    }
    first = _prepare_morphology_record(risk_record, _candidate(generation="a"))
    repeated = _prepare_morphology_record(risk_record, _candidate(generation="a"))
    changed = _prepare_morphology_record(risk_record, _candidate(generation="b"))
    assert first is not None and repeated is not None and changed is not None

    audit = _morphology_audit([first, repeated, changed])

    assert audit["candidate_epoch_distribution"]["count"] == 3
    assert audit["unique_context_generation_distribution"]["count"] == 2
    assert audit["stable_parked_episode_distribution"]["count"] == 1
    assert audit["gate"]["qualifying_stable_parked_episode_count"] == 1
    assert audit["gate"]["qualifying_context_count"] == 1
    assert audit["gate"]["passed"] is False


def test_morphology_gate_requires_distinct_real_contexts() -> None:
    first = _prepare_morphology_record(
        {
            "target_context_id": "context-a",
            "target_invocation_id": "invocation-a",
        },
        _candidate(context_id="context-a", invocation_id="invocation-a"),
    )
    second = _prepare_morphology_record(
        {
            "target_context_id": "context-b",
            "target_invocation_id": "invocation-b",
        },
        _candidate(context_id="context-b", invocation_id="invocation-b"),
    )
    assert first is not None and second is not None

    audit = _morphology_audit([first, second])

    assert audit["gate"]["qualifying_context_count"] == 2
    assert audit["gate"]["passed"] is True
