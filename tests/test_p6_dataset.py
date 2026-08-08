from __future__ import annotations

import json
from pathlib import Path

import pytest

from beliefkv.experiments.p6_dataset import (
    _apply_runtime_intervention_censors,
    _apply_partial_episode_eligibility,
    _apply_workflow_exclusions,
    _read_workflow_exclusions,
    _invalid_source_markers,
    _join_reentry_row,
    _validate_collection_contract,
    export_p6_training_dataset,
)
from beliefkv.experiments.p6_coverage import P6CoverageError


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def test_workflow_training_exclusions_fail_closed_per_instance(tmp_path: Path) -> None:
    (tmp_path / "TRAINING_EXCLUSIONS.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflows": [
                    {
                        "instance_id": "django__django-11138",
                        "reason": "harness_path_contract_contamination",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    exclusions = _read_workflow_exclusions(tmp_path)
    tables = {
        "request_calls": [
            {
                "instance_id": "django__django-11138",
                "split": "train",
                "training_eligible_remaining_decode_demand": True,
            },
            {
                "instance_id": "django__django-11400",
                "split": "train",
                "training_eligible_remaining_decode_demand": True,
            },
        ]
    }

    _apply_workflow_exclusions(tables, exclusions)

    excluded, retained = tables["request_calls"]
    assert excluded["split"] is None
    assert excluded["training_eligible_remaining_decode_demand"] is False
    assert excluded["training_excluded"] is True
    assert excluded["training_exclusion_reason"] == (
        "harness_path_contract_contamination"
    )
    assert retained["split"] == "train"
    assert retained["training_eligible_remaining_decode_demand"] is True


def test_runtime_intervention_censors_crossing_decision_horizons() -> None:
    rows = [
        {
            "workflow_id": "workflow",
            "timestamp_ms": 90.0,
            "training_eligible": True,
            "censor_reasons": [],
            "labels": [
                {
                    "next_boundary_timestamp_ms": 99.0,
                    "censored": False,
                    "censor_reason": None,
                }
            ],
        },
        {
            "workflow_id": "workflow",
            "timestamp_ms": 95.0,
            "training_eligible": True,
            "censor_reasons": [],
            "labels": [
                {
                    "next_boundary_timestamp_ms": 105.0,
                    "censored": False,
                    "censor_reason": None,
                }
            ],
        },
        {
            "workflow_id": "workflow",
            "timestamp_ms": 110.0,
            "training_eligible": True,
            "censor_reasons": [],
            "labels": [{"next_boundary_timestamp_ms": 120.0}],
        },
    ]

    summary = _apply_runtime_intervention_censors(
        rows,
        {
            "workflow": {
                "ts_ms": 100.0,
                "event_id": "sandbox-audit:8",
                "event": "agent_stuck_detected",
                "reason": "loop_guard_finalization",
                "agent_scope": "autonomous:supervisor",
            }
        },
    )

    assert rows[0]["training_eligible"] is True
    assert rows[0]["clean_episode_eligible"] is False
    assert rows[0]["episode_training_scope"] == "local_pre_intervention_only"
    assert rows[0]["eligible_until_event_id"] == "sandbox-audit:8"
    assert rows[1]["training_eligible"] is False
    assert rows[1]["labels"][0]["censored"] is True
    assert rows[2]["training_eligible"] is False
    assert summary["decision_row_count"] == 2
    assert summary["reason_counts"] == {"loop_guard_finalization": 2}


def test_partial_episode_keeps_only_pre_intervention_local_demand() -> None:
    tables = {
        "request_calls": [
            {
                "workflow_id": "workflow",
                "result_ts_ms": 90.0,
                "training_eligible_remaining_decode_demand": True,
                "training_eligible_unlock_hazard": True,
            },
            {
                "workflow_id": "workflow",
                "result_ts_ms": 110.0,
                "training_eligible_remaining_decode_demand": True,
                "training_eligible_unlock_hazard": True,
            },
        ],
        "external_waits": [
            {"workflow_id": "workflow", "training_eligible_survival": True}
        ],
        "reentries": [
            {"workflow_id": "workflow", "training_eligible": True}
        ],
        "frontier_decision_points": [],
    }
    summary = _apply_partial_episode_eligibility(
        tables,
        {
            "workflow": {
                "ts_ms": 100.0,
                "event_id": "sandbox-audit:4",
                "reason": "loop_guard_finalization",
            }
        },
    )

    before, after = tables["request_calls"]
    assert before["training_eligible_remaining_decode_demand"] is True
    assert before["training_eligible_unlock_hazard"] is False
    assert after["training_eligible_remaining_decode_demand"] is False
    assert tables["external_waits"][0]["training_eligible_survival"] is False
    assert tables["reentries"][0]["training_eligible"] is False
    assert summary["workflow_count"] == 1


def _event(
    sequence: int,
    kind: str,
    *,
    invocation_id: str | None = "root",
    attributes: dict[str, object] | None = None,
    **extra: object,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "event_id": f"event-{sequence}",
        "ts_ms": float(sequence * 10),
        "kind": kind,
        "workflow_id": "workflow",
        "invocation_id": invocation_id,
        "context_id": "context" if invocation_id else None,
        "context_epoch": 0 if invocation_id else None,
        "attributes": attributes or {},
        **extra,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    "marker_name",
    ("PILOT_INVALID.json", "COLLECTION_INVALID.json", "STARTUP_FAILED.json"),
)
def test_all_invalid_collection_markers_fail_closed(
    tmp_path: Path, marker_name: str
) -> None:
    marker = tmp_path / marker_name
    marker.write_text("{}\n", encoding="utf-8")

    assert _invalid_source_markers(tmp_path) == (marker,)
    with pytest.raises(P6CoverageError, match="marked ineligible"):
        export_p6_training_dataset(tmp_path, tmp_path / "dataset")


@pytest.mark.parametrize("workload_layout", ["workloads", "autonomous"])
def test_export_training_tables_preserves_identity_censoring_and_join_closure(
    tmp_path: Path, workload_layout: str,
) -> None:
    run_dir = tmp_path / "run"
    workloads = run_dir / workload_layout
    server = run_dir / "server"
    output = tmp_path / "dataset"
    workloads.mkdir(parents=True)
    (workloads / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "swebench",
                "dataset_revision": "revision",
                "workload_manifest_sha256": "manifest",
            }
        ),
        encoding="utf-8",
    )
    (workloads / "summary.json").write_text(
        json.dumps(
            {
                "workflow_count": 1,
                "system_jct_eligible_workflows": 1,
                "native_agent_jct_eligible_workflows": 1,
                "measurement_valid_workflows": 1,
                "workflows": [
                    {
                        "workflow_id": "workflow",
                        "instance_id": "project__task",
                        "system_jct_eligible": True,
                        "native_agent_jct_eligible": True,
                        "measurement_valid": True,
                        "task_correctness_valid": True,
                        "agent_control": {"guard_intervened_completions": 0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        workloads
        / "workflows"
        / "project__task"
        / "runtime_events.deepagents.jsonl",
        [
            _event(1, "workflow_start", invocation_id=None),
            _event(2, "invocation_create"),
            _event(3, "llm_submit", attributes={"request_id": "request"}),
            _event(
                4,
                "llm_result",
                attributes={
                    "request_id": "request",
                    "parser_status": "valid",
                    "structured_action_kinds": ["function_call"],
                    "structured_action_names": ["search"],
                    "action_boundary_source": "runtime_structured_output",
                    "action_boundary_token_index": None,
                    "output_tokens": 4,
                },
            ),
            _event(
                5,
                "tool_start",
                attributes={
                    "tool_call_id": "tool",
                    "tool_name": "search",
                    "tool_family": "search",
                    "parameter_signature": "signature",
                },
            ),
            _event(
                6,
                "tool_end",
                attributes={
                    "tool_call_id": "tool",
                    "tool_name": "search",
                    "status": "success",
                    "duration_ms": 10.0,
                },
            ),
            _event(7, "invocation_create", invocation_id="child"),
            _event(
                8,
                "join_create",
                invocation_id=None,
                join_id="join",
                member_invocation_ids=["child"],
            ),
            _event(9, "join_wait", join_id="join"),
            _event(10, "return", invocation_id="child"),
            _event(11, "join_satisfied", invocation_id=None, join_id="join"),
        ],
    )
    _write_jsonl(
        server / "runtime_events.sglang.jsonl",
        [
            _event(
                1,
                "llm_submit",
                attributes={
                    "request_id": "request",
                    "prompt_tokens": 100,
                    "cache_hit_tokens": 80,
                    "context_tokens": 100,
                    "expected_output_tokens": 16,
                },
            ),
            _event(
                2,
                "llm_result",
                attributes={"request_id": "request", "output_tokens": 4},
            ),
        ],
    )
    _write_jsonl(
        server / "runtime_audit.jsonl",
        [
            {
                "event": "gpu_service_sample",
                "sample_id": "sample",
                "phase": "decode",
                "batch_size": 1,
                "service_start_ts_ms": 1.0,
                "complete_ts_ms": 2.0,
                "service_elapsed_ms": 1.0,
                "timing_semantics_version": "gpu_service_interval_v1",
                "request_samples": [
                    {
                        "request_id": "request",
                        "workflow_id": "workflow",
                        "invocation_id": "root",
                        "context_id": "context",
                        "context_epoch": 0,
                        "phase": "decode",
                        "token_delta": 1,
                        "token_delta_semantics": "observed_output_ids_delta",
                        "sequence_tokens_before": 100,
                        "output_tokens_before": 0,
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        server / "transfer_telemetry.jsonl",
        [
            {
                "status": "completed",
                "command_id": "transfer",
                "command_kind": "offload_context",
                "telemetry_origin": "backend_telemetry",
                "direction": "d2h",
                "actual_bytes": 4096,
                "closure_bytes": 4096,
                "page_count": 1,
                "source_tier": "gpu",
                "target_tier": "host",
                "host_copy_state": "missing",
                "pinned_host": True,
                "native_concurrent_bytes": 0,
                "allocator_submit_ms": 0.1,
                "callback_overhead_ms": 0.2,
                "submit_ts_ms": 1.0,
                "complete_ts_ms": 2.0,
                "start_timestamp_semantics": "unavailable",
            }
        ],
    )
    (server / "latest_runtime_summary.json").write_text(
        json.dumps({"run_id": "run"}), encoding="utf-8"
    )

    manifest = export_p6_training_dataset(run_dir, output)

    assert manifest["identity_contract"]["ordinal_fallback"] is False
    assert manifest["source"]["collection_status"] == (
        "complete"
        if workload_layout == "workloads"
        else "complete_legacy_autonomous_layout"
    )
    assert manifest["integrity"]["passes"] is True
    assert manifest["tables"]["request_calls"]["row_count"] == 1
    assert manifest["tables"]["gpu_service_intervals"]["row_count"] == 1
    assert manifest["tables"]["gpu_batch_service_intervals"]["row_count"] == 1
    assert manifest["tables"]["external_waits"]["row_count"] == 1
    assert manifest["tables"]["reentries"]["row_count"] == 2
    assert manifest["tables"]["pcie_operations"]["row_count"] == 1
    assert manifest["tables"]["frontier_decision_points"]["row_count"] >= 1
    assert manifest["tables"]["censor_events"]["row_count"] == 0
    assert manifest["training_readiness"] == {
        "remaining_decode_demand_eligible_request_count": 1,
        "unlock_hazard_eligible_request_count": 0,
        "external_survival_eligible_count": 1,
        "join_reentry_eligible_count": 1,
        "pcie_service_eligible_count": 1,
        "runtime_batch_characterization_count": 1,
        "frontier_decision_eligible_count": 4,
        "explicit_censor_event_count": 0,
    }
    request = _read_jsonl(output / "request_calls.jsonl")[0]
    assert request["matching_method"] == "exact_native_request_id"
    assert request["training_eligible_remaining_decode_demand"] is True
    assert request["training_eligible_unlock_hazard"] is False
    join = next(
        row
        for row in _read_jsonl(output / "reentries.jsonl")
        if row["reentry_kind"] == "join"
    )
    assert join["training_eligible"] is True
    assert join["member_outcomes"][0]["return_ts_ms"] == 100.0


def test_join_label_is_not_trainable_when_one_member_has_not_returned() -> None:
    row = _join_reentry_row(
        "run",
        "join",
        {
            "workflow_id": "workflow",
            "waiter_invocation_id": "parent",
            "wait_ts_ms": 10.0,
            "member_invocation_ids": ("child-a", "child-b"),
        },
        terminal_ts_ms=30.0,
        terminal_status="satisfied",
        return_ts={"child-a": 20.0},
        invocation_start_ts={"child-a": 0.0, "child-b": 0.0},
        workflow_metadata={},
    )

    assert row["training_eligible"] is False
    assert row["member_outcomes"][1]["return_ts_ms"] is None


def test_collection_contract_fails_closed_on_predictive_or_invalid_evidence() -> None:
    base = {
        "predictor_enabled": False,
        "predictive_actions_enabled": False,
        "runtime_policy": "frozen_p5_observed",
        "training_eligible": True,
    }
    _validate_collection_contract(base, allow_censored=False)
    with pytest.raises(P6CoverageError, match="predictor enabled"):
        _validate_collection_contract(
            {**base, "predictor_enabled": True}, allow_censored=False
        )
    with pytest.raises(P6CoverageError, match="system eligibility"):
        _validate_collection_contract(
            {**base, "training_eligible": False}, allow_censored=False
        )
    _validate_collection_contract(
        {**base, "training_eligible": False}, allow_censored=True
    )
