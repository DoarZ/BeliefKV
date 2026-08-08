from __future__ import annotations

import json
from pathlib import Path

from beliefkv.experiments.p6_coverage import (
    _service_coverage,
    characterize_p6_coverage,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def _event(
    sequence: int,
    kind: str,
    *,
    attributes: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "event_id": f"agent-{sequence}",
        "ts_ms": float(sequence * 10),
        "kind": kind,
        "workflow_id": "workflow",
        "invocation_id": "invocation",
        "context_id": "context",
        "attributes": attributes or {},
    }


def test_characterization_separates_runtime_boundary_and_service_gap(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    workloads = run_dir / "workloads"
    server = run_dir / "server"
    workloads.mkdir(parents=True)
    (workloads / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "experiment_valid": False,
                "workload_manifest_sha256": "trace",
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        workloads / "workflow" / "runtime_events.agentic.jsonl",
        [
            _event(1, "llm_submit", attributes={"request_id": "request"}),
            _event(
                2,
                "llm_result",
                attributes={
                    "parser_status": "valid",
                    "structured_action_kinds": ["function_call"],
                    "structured_action_names": ["search"],
                    "action_boundary_source": "runtime_structured_output",
                    "action_boundary_token_index": None,
                    "output_tokens": 12,
                    "request_id": "request",
                },
            ),
            _event(3, "tool_start", attributes={"tool_call_id": "tool"}),
            _event(4, "tool_end", attributes={"tool_call_id": "tool"}),
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
                },
            ),
            _event(
                2,
                "llm_result",
                attributes={"request_id": "request", "output_tokens": 12},
            ),
        ],
    )
    _write_jsonl(server / "runtime_audit.jsonl", [{"event": "runtime_initialized"}])
    _write_jsonl(
        server / "transfer_telemetry.jsonl",
        [
            {
                "event": "transfer_telemetry",
                "status": "completed",
                "direction": "d2h",
                "actual_bytes": 4096,
                "closure_bytes": 4096,
                "page_count": 1,
                "command_kind": "offload_context",
                "source_tier": "gpu",
                "target_tier": "host",
                "submit_ts_ms": 1.0,
                "start_ts_ms": 2.0,
                "complete_ts_ms": 3.0,
                "ts_ms": 4.0,
                "host_copy_state": "missing",
                "pinned_host": True,
                "native_concurrent_bytes": 4096,
                "allocator_submit_ms": 0.1,
                "callback_overhead_ms": 0.2,
                "start_timestamp_semantics": "hicache_api_submit_begin",
            }
        ],
    )

    result = characterize_p6_coverage(run_dir)

    assert result["call_matching"]["agentic_to_native_match_coverage"] == 1.0
    assert result["call_matching"][
        "submitted_exact_native_request_id_coverage"
    ] == 1.0
    assert result["action_frontier"]["exact_boundary_call_coverage"] == 0.0
    assert result["action_frontier"]["runtime_only_boundary_call_coverage"] == 1.0
    assert result["action_frontier"]["reentry_cause_coverage"] == 1.0
    assert result["gpu_service"]["status"] == "unavailable"
    assert result["pcie_transfer"]["conditioning_ready"] is True
    assert result["pcie_transfer"]["field_coverage"]["direct_dma_duration"] == 0.0
    assert result["gates"]["llm_result_boundary_fallback_required"] is True
    assert result["gates"]["remaining_decode_demand_training_ready"] is True


def test_deepagents_layout_uses_strict_request_identity_without_ordinal_fallback(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    workloads = run_dir / "workloads"
    server = run_dir / "server"
    workloads.mkdir(parents=True)
    (workloads / "manifest.json").write_text(
        json.dumps({"run_id": "run", "experiment_valid": True}),
        encoding="utf-8",
    )
    _write_jsonl(
        workloads
        / "workflows"
        / "case"
        / "runtime_events.deepagents.jsonl",
        [
            _event(1, "llm_submit", attributes={"request_id": "agent-request"}),
            _event(
                2,
                "llm_result",
                attributes={
                    "request_id": "agent-request",
                    "parser_status": "valid",
                    "structured_action_kinds": ["final_answer"],
                    "output_tokens": 1,
                },
            ),
        ],
    )
    _write_jsonl(
        server / "runtime_events.sglang.jsonl",
        [
            _event(
                1,
                "llm_submit",
                attributes={
                    "request_id": "different-native-request",
                    "prompt_tokens": 1,
                    "cache_hit_tokens": 0,
                },
            ),
            _event(
                2,
                "llm_result",
                attributes={
                    "request_id": "different-native-request",
                    "output_tokens": 1,
                },
            ),
        ],
    )
    _write_jsonl(server / "runtime_audit.jsonl", [])
    _write_jsonl(server / "transfer_telemetry.jsonl", [])

    result = characterize_p6_coverage(run_dir)

    assert result["source"]["agent_trace_layout_counts"] == {"deepagents": 1}
    assert result["call_matching"]["agentic_to_native_match_count"] == 0
    assert result["call_matching"]["matching_failure_counts"] == {
        "native_request_id_not_found": 1
    }
    assert result["call_matching"]["matching_rule"].endswith(
        "ordinal fallback is disabled"
    )


def test_transfer_coverage_excludes_preflight_rejection_from_dma_fields(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    workloads = run_dir / "workloads"
    server = run_dir / "server"
    workloads.mkdir(parents=True)
    (workloads / "manifest.json").write_text(
        json.dumps({"run_id": "run"}), encoding="utf-8"
    )
    _write_jsonl(
        workloads / "workflow" / "runtime_events.agentic.jsonl",
        [
            _event(1, "llm_submit", attributes={"request_id": "request"}),
            _event(
                2,
                "llm_result",
                attributes={"request_id": "request", "output_tokens": 1},
            ),
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
                    "prompt_tokens": 1,
                    "cache_hit_tokens": 0,
                },
            ),
            _event(
                2,
                "llm_result",
                attributes={"request_id": "request", "output_tokens": 1},
            ),
        ],
    )
    _write_jsonl(server / "runtime_audit.jsonl", [])
    _write_jsonl(
        server / "transfer_telemetry.jsonl",
        [
            {
                "status": "rejected",
                "direction": "h2d",
                "actual_bytes": 0,
                "closure_bytes": 1,
                "page_count": 1,
                "command_kind": "prefetch_context",
                "source_tier": "host",
                "target_tier": "gpu",
            },
            {
                "status": "completed",
                "direction": "d2h",
                "actual_bytes": 4096,
                "closure_bytes": 4096,
                "page_count": 1,
                "command_kind": "offload_context",
                "source_tier": "gpu",
                "target_tier": "host",
                "host_copy_state": "missing",
                "pinned_host": True,
                "native_concurrent_bytes": 0,
                "allocator_submit_ms": 0.1,
                "callback_overhead_ms": 0.2,
                "submit_ts_ms": 1.0,
                "start_ts_ms": 1.1,
                "complete_ts_ms": 2.0,
                "ts_ms": 2.2,
                "start_timestamp_semantics": "hicache_api_submit_begin",
            },
        ],
    )

    result = characterize_p6_coverage(run_dir)

    assert result["pcie_transfer"]["attempt_count"] == 2
    assert result["pcie_transfer"]["physical_operation_count"] == 1
    assert result["pcie_transfer"][
        "rejected_before_physical_operation_count"
    ] == 1
    assert result["pcie_transfer"]["conditioning_ready"] is True


def test_service_coverage_expands_request_rows_and_observer_cost(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "runtime_audit.jsonl"
    base = {
        "service_start_ts_ms": 1.0,
        "complete_ts_ms": 2.0,
        "service_elapsed_ms": 1.0,
        "timing_semantics_version": "gpu_service_interval_v1",
        "request_ids": ["request"],
    }
    request = {
        "request_id": "request",
        "workflow_id": "workflow",
        "invocation_id": "invocation",
        "context_id": "context",
        "context_epoch": 0,
        "token_delta": 4,
        "sequence_tokens_before": 100,
    }
    _write_jsonl(
        audit_path,
        [
            {
                "event": "gpu_service_sample",
                "phase": "prefill",
                "request_samples": [
                    {
                        **request,
                        "phase": "prefill",
                        "token_delta_semantics": "prefill_extend_input_len",
                    }
                ],
                **base,
            },
            {
                "event": "gpu_service_sample",
                "phase": "decode",
                "request_samples": [
                    {
                        **request,
                        "phase": "decode",
                        "token_delta_semantics": "observed_output_ids_delta",
                    }
                ],
                **base,
            },
            {
                "event": "controller_timing_summary",
                "scheduler_step_p99_ms": 3.0,
            },
            {
                "event": "gpu_service_observer_summary",
                "sample_cap_count": 0,
                "observer_cpu_ms": {
                    "build_p99": 0.02,
                    "audit_enqueue_p99": 0.01,
                },
            },
            {
                "event": "runtime_audit_writer_summary",
                "pending_count": 0,
                "dropped_debug_count": 2,
            },
        ],
    )
    calls = [
        {
            "runtime_internal": False,
            "native": {"request_id": "request"},
        }
    ]

    result = _service_coverage(calls, audit_path)

    assert result["request_service_label_coverage"] == 1.0
    assert result["request_level_row_field_completeness"] == 1.0
    assert result["request_prefill_service_label_count"] == 1
    assert result["request_decode_service_label_count"] == 1
    assert result["request_both_phase_label_count"] == 1
    assert result["exact_decode_token_delta_coverage"] == 1.0
    assert result["observer_overhead"]["scheduler_step_p99_ms"] == 3.0
    assert result["observer_overhead"]["audit_dropped_debug_count"] == 2


def test_censored_collection_keeps_open_calls_without_faking_manifest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    workloads = run_dir / "workloads.incomplete"
    server = run_dir / "server"
    _write_jsonl(
        workloads / "workflow" / "runtime_events.agentic.jsonl",
        [_event(1, "llm_submit", attributes={"request_id": "request"})],
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
                },
            )
        ],
    )
    _write_jsonl(server / "runtime_audit.jsonl", [])
    _write_jsonl(server / "transfer_telemetry.jsonl", [])
    (server / "latest_runtime_summary.json").write_text(
        json.dumps(
            {
                "run_id": "server-run",
                "audit_writer": {
                    "pending_count": 0,
                    "dropped_debug_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = characterize_p6_coverage(
        run_dir,
        allow_censored=True,
        censor_reason="characterization_complete_liveness_blocked",
    )

    assert result["source"]["collection_status"] == "censored"
    assert result["source"]["experiment_valid_for_performance"] is False
    assert result["source"]["run_id"] == "server-run"
    assert result["trace"]["censored_llm_call_count"] == 1
    assert result["call_matching"][
        "submitted_exact_native_request_id_coverage"
    ] == 1.0
    assert result["call_matching"]["censored_policy_call_count"] == 1
    assert result["gpu_service"]["observer_overhead"][
        "audit_dropped_debug_count"
    ] == 0
