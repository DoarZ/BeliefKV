from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.runtime.action_frontier import (
    ActionFrontierObserver,
    StructuredActionKind,
)


class P6CoverageError(ValueError):
    """Raised when a run lacks the evidence needed for P6.0 characterization."""


def characterize_p6_coverage(
    run_dir: str | Path,
    *,
    allow_censored: bool = False,
    censor_reason: str | None = None,
) -> dict[str, Any]:
    """Measure P6.0 label identifiability without changing the online policy."""

    source = Path(run_dir).resolve()
    completed_workloads_dir = source / "workloads"
    legacy_autonomous_dir = source / "autonomous"
    incomplete_workloads_dir = source / "workloads.incomplete"
    if completed_workloads_dir.is_dir():
        workloads_dir = completed_workloads_dir
        collection_status = "complete"
    elif legacy_autonomous_dir.is_dir():
        workloads_dir = legacy_autonomous_dir
        collection_status = "complete_legacy_autonomous_layout"
    elif allow_censored and incomplete_workloads_dir.is_dir():
        workloads_dir = incomplete_workloads_dir
        collection_status = "censored"
    else:
        workloads_dir = completed_workloads_dir
        collection_status = "missing"
    server_dir = source / "server"
    manifest_path = workloads_dir / "manifest.json"
    server_events_path = server_dir / "runtime_events.sglang.jsonl"
    audit_path = server_dir / "runtime_audit.jsonl"
    transfer_path = server_dir / "transfer_telemetry.jsonl"
    runtime_summary_path = server_dir / "latest_runtime_summary.json"
    workload_summary_path = workloads_dir / "summary.json"
    required = (server_events_path, audit_path, transfer_path)
    if collection_status != "censored":
        required = (manifest_path, *required)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise P6CoverageError(f"run is missing required artifacts: {missing}")
    agent_paths = discover_p6_agent_traces(workloads_dir)
    if not agent_paths:
        raise P6CoverageError(
            "run has no supported per-workflow agent event traces"
        )

    manifest = _read_object(manifest_path) if manifest_path.is_file() else {}
    runtime_summary = (
        _read_object(runtime_summary_path)
        if runtime_summary_path.is_file()
        else {}
    )
    workload_summary = (
        _read_object(workload_summary_path)
        if workload_summary_path.is_file()
        else {}
    )
    agent_records = [_read_jsonl(path) for path in agent_paths]
    server_records = _read_jsonl(server_events_path)
    observer = ActionFrontierObserver()
    calls: list[dict[str, Any]] = []
    event_kind_counts: Counter[str] = Counter()
    workflow_stats: dict[str, Counter[str]] = defaultdict(Counter)
    censored_call_count = 0
    for path, records in zip(agent_paths, agent_records):
        censored_call_count += _replay_agent_trace(
            records,
            observer=observer,
            calls=calls,
            event_kind_counts=event_kind_counts,
            workflow_stats=workflow_stats,
            source_path=path,
        )

    _match_server_calls(calls, server_records)
    action = _action_coverage(calls, observer)
    service = _service_coverage(
        calls,
        audit_path,
        runtime_summary=runtime_summary,
    )
    transfer = _transfer_coverage(transfer_path)
    call_matching = _call_matching_coverage(calls)

    evidence_paths = [
        server_events_path,
        audit_path,
        transfer_path,
        *agent_paths,
    ]
    if manifest_path.is_file():
        evidence_paths.insert(0, manifest_path)
    if runtime_summary_path.is_file():
        evidence_paths.append(runtime_summary_path)
    if workload_summary_path.is_file():
        evidence_paths.append(workload_summary_path)
    workload_quality = _workload_quality(manifest, workload_summary)
    return {
        "schema_version": 2,
        "phase": "p6.0",
        "characterization_only": True,
        "source": {
            "run_dir": str(source),
            "run_id": manifest.get("run_id") or runtime_summary.get("run_id"),
            "workload_manifest_sha256": manifest.get("workload_manifest_sha256"),
            "collection_status": collection_status,
            "censored": collection_status == "censored",
            "censor_reason": (
                censor_reason if collection_status == "censored" else None
            ),
            "experiment_valid_for_performance": bool(
                collection_status == "complete"
                and workload_quality["native_agent_clean_valid"]
                and workload_quality["task_measurement_valid"]
            ),
            "agentic_trace_count": len(agent_paths),
            "agent_trace_layout_counts": dict(
                sorted(
                    Counter(_agent_trace_layout(path) for path in agent_paths).items()
                )
            ),
            "workload_quality": workload_quality,
            "artifact_sha256": {
                str(path.relative_to(source)): _sha256(path)
                for path in evidence_paths
            },
        },
        "trace": {
            "workflow_count": len(agent_paths),
            "agentic_event_count": sum(len(records) for records in agent_records),
            "server_event_count": len(server_records),
            "censored_llm_call_count": censored_call_count,
            "event_kind_counts": dict(sorted(event_kind_counts.items())),
            "workflow_breakdown": {
                workflow_id: dict(sorted(counts.items()))
                for workflow_id, counts in sorted(workflow_stats.items())
            },
        },
        "call_matching": call_matching,
        "action_frontier": action,
        "gpu_service": service,
        "pcie_transfer": transfer,
        "gates": _gates(action, service, call_matching),
    }


def discover_p6_agent_traces(workloads_dir: str | Path) -> tuple[Path, ...]:
    """Return supported per-workflow traces without assuming one runtime layout."""

    root = Path(workloads_dir)
    patterns = (
        "*/runtime_events.agentic.jsonl",
        "workflows/*/runtime_events.deepagents.jsonl",
    )
    paths = {path.resolve() for pattern in patterns for path in root.glob(pattern)}
    return tuple(sorted(paths))


def _agent_trace_layout(path: Path) -> str:
    if path.name == "runtime_events.deepagents.jsonl":
        return "deepagents"
    if path.name == "runtime_events.agentic.jsonl":
        return "agentic"
    return "unknown"


def _workload_quality(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    workflow_count = int(summary.get("workflow_count") or 0)
    system_eligible = int(summary.get("system_jct_eligible_workflows") or 0)
    native_eligible = int(
        summary.get("native_agent_jct_eligible_workflows") or 0
    )
    measurement_valid = int(summary.get("measurement_valid_workflows") or 0)
    if workflow_count <= 0:
        legacy_valid = bool(manifest.get("experiment_valid", False))
        return {
            "workflow_count": 0,
            "system_jct_eligible_workflows": 0,
            "native_agent_jct_eligible_workflows": 0,
            "measurement_valid_workflows": 0,
            "system_gate_valid": legacy_valid,
            "native_agent_clean_valid": legacy_valid,
            "task_measurement_valid": legacy_valid,
            "source": "legacy_manifest",
        }
    return {
        "workflow_count": workflow_count,
        "system_jct_eligible_workflows": system_eligible,
        "native_agent_jct_eligible_workflows": native_eligible,
        "measurement_valid_workflows": measurement_valid,
        "system_gate_valid": system_eligible == workflow_count,
        "native_agent_clean_valid": native_eligible == workflow_count,
        "task_measurement_valid": measurement_valid == workflow_count,
        "source": "workload_summary",
    }


def write_p6_coverage(
    run_dir: str | Path,
    output_path: str | Path,
    *,
    allow_censored: bool = False,
    censor_reason: str | None = None,
) -> dict[str, Any]:
    result = characterize_p6_coverage(
        run_dir,
        allow_censored=allow_censored,
        censor_reason=censor_reason,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return result


def _replay_agent_trace(
    records: list[dict[str, Any]],
    *,
    observer: ActionFrontierObserver,
    calls: list[dict[str, Any]],
    event_kind_counts: Counter[str],
    workflow_stats: dict[str, Counter[str]],
    source_path: Path,
) -> int:
    open_call: dict[str, dict[str, Any]] = {}
    call_by_request: dict[str, dict[str, Any]] = {}
    ordinal_by_invocation: Counter[str] = Counter()
    previous_sequence = 0
    for raw in records:
        event = RuntimeEvent.from_dict(raw)
        sequence = int(raw.get("sequence", 0))
        if sequence != previous_sequence + 1:
            raise P6CoverageError(
                f"non-contiguous sequence in {source_path}: {sequence}"
            )
        previous_sequence = sequence
        event_kind_counts[event.kind.value] += 1
        workflow_stats[event.workflow_id][event.kind.value] += 1
        if event.kind == RuntimeEventKind.LLM_SUBMIT and event.invocation_id:
            invocation_id = event.invocation_id
            if invocation_id in open_call:
                raise P6CoverageError(
                    f"overlapping LLM calls for invocation {invocation_id}"
                )
            ordinal = ordinal_by_invocation[invocation_id]
            ordinal_by_invocation[invocation_id] += 1
            declared_request_id = event.attributes.get("request_id")
            call = {
                "workflow_id": event.workflow_id,
                "invocation_id": invocation_id,
                "context_id": event.context_id,
                "context_epoch": event.context_epoch,
                "ordinal": ordinal,
                "submit_ts_ms": event.ts_ms,
                "runtime_internal": bool(
                    event.attributes.get("runtime_internal", False)
                ),
                "agent_request_id": (
                    str(declared_request_id) if declared_request_id else ""
                ),
                "request_identity_source": (
                    "declared_native_request_id"
                    if declared_request_id
                    else "missing"
                ),
                "source_trace": str(source_path),
                "parser_status": "missing",
                "action_kinds": (),
                "action_boundary_token_index": None,
                "action_boundary_source": "unobserved",
                "output_tokens": None,
            }
            calls.append(call)
            open_call[invocation_id] = call
            if declared_request_id:
                call_by_request[str(declared_request_id)] = call
        elif event.kind == RuntimeEventKind.LLM_RESULT and event.invocation_id:
            call = open_call.pop(event.invocation_id, None)
            if call is None:
                raise P6CoverageError(
                    f"LLM result without submit for {event.invocation_id}"
                )
            result_request_id = event.attributes.get("request_id")
            if (
                result_request_id
                and call["agent_request_id"]
                and str(result_request_id) != call["agent_request_id"]
            ):
                raise P6CoverageError(
                    "LLM submit/result request identity mismatch for "
                    f"{event.invocation_id}: {call['agent_request_id']} != "
                    f"{result_request_id}"
                )
            call.update(
                {
                    "result_ts_ms": event.ts_ms,
                    "parser_status": str(
                        event.attributes.get("parser_status", "unknown")
                    ),
                    "action_kinds": tuple(
                        str(item)
                        for item in _as_sequence(
                            event.attributes.get("structured_action_kinds", ())
                        )
                    ),
                    "action_names": tuple(
                        str(item)
                        for item in _as_sequence(
                            event.attributes.get("structured_action_names", ())
                        )
                    ),
                    "action_boundary_token_index": event.attributes.get(
                        "action_boundary_token_index"
                    ),
                    "action_boundary_source": str(
                        event.attributes.get(
                            "action_boundary_source", "runtime_structured_output"
                        )
                    ),
                    "output_tokens": _optional_nonnegative_int(
                        event.attributes.get("output_tokens")
                    ),
                    "censored": bool(event.attributes.get("censored", False)),
                    "censor_reason": event.attributes.get("censor_reason"),
                }
            )
        elif event.kind == RuntimeEventKind.CALL_CENSORED:
            request_id = str(event.attributes.get("request_id") or "")
            call = call_by_request.get(request_id)
            if call is not None:
                call["censored"] = True
                call["censor_reason"] = event.attributes.get("censor_reason")
        observer.observe_runtime_event(event)
    for call in open_call.values():
        call["censored"] = True
        call["parser_status"] = "censored"
        call.setdefault("censor_reason", "trace_ended_with_open_request")
    return len(open_call)


def _match_server_calls(
    calls: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    open_native: dict[str, dict[str, Any]] = {}
    for raw in records:
        kind = raw.get("kind")
        if kind not in {"llm_submit", "llm_result"}:
            continue
        workflow_id = str(raw.get("workflow_id") or "")
        invocation_id = str(raw.get("invocation_id") or "")
        attributes = raw.get("attributes") or {}
        request_id = str(attributes.get("request_id") or "")
        if not workflow_id or not invocation_id or not request_id:
            continue
        if kind == "llm_submit":
            if request_id in open_native:
                raise P6CoverageError(
                    f"duplicate native request_id in server trace: {request_id}"
                )
            native = {
                "request_id": request_id,
                "workflow_id": workflow_id,
                "invocation_id": invocation_id,
                "context_id": raw.get("context_id"),
                "context_epoch": raw.get("context_epoch"),
                "prompt_tokens": _optional_nonnegative_int(
                    attributes.get("prompt_tokens")
                ),
                "cache_hit_tokens": _optional_nonnegative_int(
                    attributes.get("cache_hit_tokens")
                ),
                "context_tokens": _optional_nonnegative_int(
                    attributes.get("context_tokens")
                ),
                "expected_output_tokens": _optional_nonnegative_int(
                    attributes.get("expected_output_tokens")
                ),
                "model": attributes.get("model"),
                "submit_ts_ms": float(raw["ts_ms"]),
            }
            open_native[request_id] = native
        else:
            native = open_native.get(request_id)
            if native is not None:
                native["result_ts_ms"] = float(raw["ts_ms"])
                native["output_tokens"] = _optional_nonnegative_int(
                    attributes.get("output_tokens")
                )
    for call in calls:
        request_id = call["agent_request_id"]
        if not request_id:
            call["matching_failure"] = "agent_request_id_missing"
            continue
        native = open_native.get(request_id)
        if native is None:
            call["matching_failure"] = "native_request_id_not_found"
            continue
        if (
            native["workflow_id"] != call["workflow_id"]
            or native["invocation_id"] != call["invocation_id"]
        ):
            call["matching_failure"] = "native_identity_scope_conflict"
            continue
        call["native"] = native
        call["matching_method"] = "exact_native_request_id"


def _action_coverage(
    calls: list[dict[str, Any]],
    observer: ActionFrontierObserver,
) -> dict[str, Any]:
    policy_calls = [item for item in calls if not item["runtime_internal"]]
    internal_calls = len(calls) - len(policy_calls)
    parser_status = Counter(str(item["parser_status"]) for item in policy_calls)
    action_kind_counts: Counter[str] = Counter()
    exact = 0
    runtime_only = 0
    demand = 0
    for call in policy_calls:
        kinds = tuple(call["action_kinds"])
        if not kinds:
            action_kind_counts[StructuredActionKind.UNKNOWN.value] += 1
        else:
            action_kind_counts.update(kinds)
        if (
            call["action_boundary_token_index"] is not None
            and call["action_boundary_source"] == "native_incremental_parser"
        ):
            exact += 1
        elif call["parser_status"] == "valid":
            runtime_only += 1
        if (
            call.get("output_tokens") is not None
            and call.get("result_ts_ms", -1) >= call["submit_ts_ms"]
        ):
            demand += 1

    observer_coverage = observer.coverage().to_dict()
    reentry_causes = Counter()
    reentry_missing = Counter()
    for snapshot in observer.snapshots():
        if snapshot.action_kind not in {
            StructuredActionKind.FUNCTION_CALL,
            StructuredActionKind.SPAWN,
            StructuredActionKind.HANDOFF,
        }:
            continue
        if snapshot.reentry_ts_ms is not None:
            reentry_causes[
                "join_satisfied"
                if snapshot.action_kind == StructuredActionKind.SPAWN
                else "tool_or_runtime_event"
            ] += 1
        else:
            reentry_missing[snapshot.action_kind.value] += 1

    action_calls = sum(
        bool(item["action_kinds"]) or item["parser_status"] == "valid"
        for item in policy_calls
    )
    return {
        "raw_llm_call_count": len(calls),
        "runtime_internal_call_count": internal_calls,
        "policy_call_count": len(policy_calls),
        "action_call_count": action_calls,
        "parser_status_counts": dict(sorted(parser_status.items())),
        "action_kind_occurrence_counts": dict(sorted(action_kind_counts.items())),
        "exact_boundary_call_count": exact,
        "exact_boundary_call_coverage": _fraction(exact, action_calls),
        "runtime_only_boundary_call_count": runtime_only,
        "runtime_only_boundary_call_coverage": _fraction(runtime_only, action_calls),
        "unknown_call_count": parser_status["unknown"] + parser_status["missing"],
        "malformed_call_count": parser_status["invalid"],
        "exact_boundary_decode_time_coverage": None,
        "exact_boundary_decode_time_reason": (
            "native per-token parser/service timestamps are absent"
        ),
        "reentry_eligible_call_count": observer_coverage[
            "reentry_eligible_call_count"
        ],
        "reentry_observed_count": observer_coverage["reentry_observed_count"],
        "reentry_cause_coverage": observer_coverage["reentry_cause_coverage"],
        "reentry_cause_counts": dict(sorted(reentry_causes.items())),
        "reentry_missing_by_action_kind": dict(sorted(reentry_missing.items())),
        "demand_label_count": demand,
        "demand_label_completeness": _fraction(demand, len(policy_calls)),
    }


def _call_matching_coverage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    policy_calls = [item for item in calls if not item["runtime_internal"]]
    completed_policy_calls = [
        item for item in policy_calls if not item.get("censored", False)
    ]
    matched = [item for item in policy_calls if "native" in item]
    matched_completed = [
        item for item in matched if not item.get("censored", False)
    ]
    complete = [
        item
        for item in matched_completed
        if all(
            item["native"].get(key) is not None
            for key in (
                "prompt_tokens",
                "cache_hit_tokens",
                "output_tokens",
                "submit_ts_ms",
                "result_ts_ms",
            )
        )
    ]
    output_disagreements = sum(
        item.get("output_tokens") is not None
        and item["native"].get("output_tokens") is not None
        and item["output_tokens"] != item["native"]["output_tokens"]
        for item in matched
    )
    submitted_matching_methods = Counter(
        str(item.get("matching_method", "unknown")) for item in matched
    )
    completed_matching_methods = Counter(
        str(item.get("matching_method", "unknown"))
        for item in matched_completed
    )
    matching_failures = Counter(
        str(item.get("matching_failure", "unmatched"))
        for item in policy_calls
        if "native" not in item
    )
    return {
        "policy_call_count": len(policy_calls),
        "completed_policy_call_count": len(completed_policy_calls),
        "censored_policy_call_count": (
            len(policy_calls) - len(completed_policy_calls)
        ),
        "agentic_to_native_match_count": len(matched),
        "agentic_to_native_match_coverage": _fraction(len(matched), len(policy_calls)),
        "complete_token_demand_label_count": len(complete),
        "complete_token_demand_label_coverage": _fraction(
            len(complete), len(completed_policy_calls)
        ),
        "output_token_disagreement_count": output_disagreements,
        "matching_method_counts": dict(
            sorted(submitted_matching_methods.items())
        ),
        "completed_matching_method_counts": dict(
            sorted(completed_matching_methods.items())
        ),
        "matching_failure_counts": dict(sorted(matching_failures.items())),
        "exact_native_request_id_coverage": _fraction(
            completed_matching_methods["exact_native_request_id"],
            len(completed_policy_calls),
        ),
        "submitted_exact_native_request_id_coverage": _fraction(
            submitted_matching_methods["exact_native_request_id"],
            len(policy_calls),
        ),
        "matching_rule": (
            "exact native request_id with matching workflow_id and invocation_id; "
            "ordinal fallback is disabled"
        ),
    }


def _service_coverage(
    calls: list[dict[str, Any]],
    audit_path: Path,
    *,
    runtime_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy_request_ids = {
        item["native"]["request_id"]
        for item in calls
        if not item["runtime_internal"] and "native" in item
    }
    samples = 0
    failed = 0
    phases: Counter[str] = Counter()
    covered_requests: set[str] = set()
    request_phase_coverage: dict[str, set[str]] = defaultdict(set)
    request_rows = 0
    complete_request_rows = 0
    decode_request_rows = 0
    exact_decode_rows = 0
    conditioned = 0
    observer_summary: Mapping[str, Any] | None = None
    controller_timing: Mapping[str, Any] | None = None
    audit_writer_summary: Mapping[str, Any] | None = None
    launch_observer_cpu_ms: list[float] = []
    for record in _iter_jsonl(audit_path):
        event = record.get("event")
        if event == "gpu_service_observer_summary":
            observer_summary = record
            continue
        if event == "controller_timing_summary":
            controller_timing = record
            continue
        if event == "runtime_audit_writer_summary":
            audit_writer_summary = record
            continue
        if event == "gpu_service_sample_failed":
            failed += 1
            continue
        if event != "gpu_service_sample":
            continue
        samples += 1
        if record.get("launch_observer_cpu_ms") is not None:
            launch_observer_cpu_ms.append(
                float(record["launch_observer_cpu_ms"])
            )
        phases[str(record.get("phase") or "unknown")] += 1
        request_ids = {
            str(item) for item in _as_sequence(record.get("request_ids", ()))
        }
        covered_requests.update(request_ids & policy_request_ids)
        for row in _as_sequence(record.get("request_samples", ())):
            if not isinstance(row, Mapping):
                continue
            request_id = str(row.get("request_id") or "")
            if request_id not in policy_request_ids:
                continue
            request_rows += 1
            row_phase = str(row.get("phase") or "unknown")
            if row_phase == "decode":
                decode_request_rows += 1
            required = (
                "request_id",
                "workflow_id",
                "invocation_id",
                "context_id",
                "context_epoch",
                "phase",
                "token_delta",
                "token_delta_semantics",
                "sequence_tokens_before",
            )
            if all(row.get(key) is not None for key in required):
                complete_request_rows += 1
                request_phase_coverage[request_id].add(row_phase)
            if (
                row_phase == "decode"
                and row.get("token_delta_semantics")
                == "observed_output_ids_delta"
            ):
                exact_decode_rows += 1
        if (
            record.get("service_start_ts_ms") is not None
            and record.get("complete_ts_ms") is not None
            and record.get("service_elapsed_ms") is not None
            and record.get("timing_semantics_version")
            == "gpu_service_interval_v1"
        ):
            conditioned += 1
    latest_audit_writer = (
        runtime_summary.get("audit_writer", {})
        if runtime_summary is not None
        else {}
    )
    observer_cpu = (
        observer_summary.get("observer_cpu_ms")
        if observer_summary is not None
        else {
            "launch_build_p50": _percentile(launch_observer_cpu_ms, 50),
            "launch_build_p95": _percentile(launch_observer_cpu_ms, 95),
            "launch_build_p99": _percentile(launch_observer_cpu_ms, 99),
            "source": "per_sample_fallback",
            "audit_enqueue_p99": None,
        }
    )
    return {
        "gpu_service_sample_count": samples,
        "gpu_service_sample_failed_count": failed,
        "conditioned_sample_count": conditioned,
        "phase_counts": dict(sorted(phases.items())),
        "policy_request_count": len(policy_request_ids),
        "request_service_label_count": len(covered_requests),
        "request_service_label_coverage": _fraction(
            len(covered_requests), len(policy_request_ids)
        ),
        "request_level_row_count": request_rows,
        "request_level_row_field_completeness": _fraction(
            complete_request_rows, request_rows
        ),
        "request_prefill_service_label_count": sum(
            "prefill" in phases for phases in request_phase_coverage.values()
        ),
        "request_decode_service_label_count": sum(
            "decode" in phases for phases in request_phase_coverage.values()
        ),
        "request_both_phase_label_count": sum(
            {"prefill", "decode"}.issubset(phases)
            for phases in request_phase_coverage.values()
        ),
        "exact_decode_token_delta_row_count": exact_decode_rows,
        "exact_decode_token_delta_coverage": _fraction(
            exact_decode_rows, decode_request_rows
        ),
        "observer_overhead": {
            "observer_cpu_ms": observer_cpu,
            "sample_cap_count": (
                observer_summary.get("sample_cap_count")
                if observer_summary is not None
                else None
            ),
            "scheduler_step_p99_ms": (
                controller_timing.get("scheduler_step_p99_ms")
                if controller_timing is not None
                else None
            ),
            "audit_pending_count": (
                audit_writer_summary.get("pending_count")
                if audit_writer_summary is not None
                else latest_audit_writer.get("pending_count")
            ),
            "audit_dropped_debug_count": (
                audit_writer_summary.get("dropped_debug_count")
                if audit_writer_summary is not None
                else latest_audit_writer.get("dropped_debug_count")
            ),
            "shutdown_summary_available": observer_summary is not None,
        },
        "queue_wait_excluded": samples > 0,
        "status": "available" if samples > 0 else "unavailable",
        "unavailable_reason": (
            None
            if samples > 0
            else "fixed trace contains no gpu_service_sample audit events"
        ),
    }


def _transfer_coverage(path: Path) -> dict[str, Any]:
    records = list(_iter_jsonl(path))
    total = len(records)
    completed = [item for item in records if item.get("status") == "completed"]
    rejected = [item for item in records if item.get("status") == "rejected"]
    physical_total = len(completed)
    field_counts = {
        "direction": _count(
            completed, lambda item: item.get("direction") is not None
        ),
        "bytes": _count(
            completed,
            lambda item: item.get("actual_bytes") is not None
            and item.get("closure_bytes") is not None,
        ),
        "extent_or_page_count": _count(
            completed, lambda item: item.get("page_count") is not None
        ),
        "command_kind": _count(
            completed, lambda item: bool(item.get("command_kind"))
        ),
        "tier_state": _count(
            completed,
            lambda item: item.get("source_tier") is not None
            and item.get("target_tier") is not None,
        ),
        "host_copy_state": _count(
            completed,
            lambda item: item.get("host_copy_state") not in {None, "unknown"},
        ),
        "pinned_host_state": _count(
            completed, lambda item: isinstance(item.get("pinned_host"), bool)
        ),
        "native_hicache_concurrency": _count(
            completed,
            lambda item: any(
                item.get(key) is not None
                for key in (
                    "native_hicache_inflight",
                    "native_hicache_concurrency",
                    "concurrent_native_bytes",
                    "native_concurrent_bytes",
                )
            ),
        ),
        "allocator_overhead": _count(
            completed,
            lambda item: any(
                item.get(key) is not None
                for key in (
                    "allocator_wait_ms",
                    "allocator_submit_ms",
                    "allocation_ms",
                    "compute_wait_ms",
                )
            ),
        ),
        "callback_overhead": _count(
            completed,
            lambda item: item.get("callback_overhead_ms") is not None,
        ),
        "callback_overhead_derived": _count(
            completed,
            lambda item: item.get("ts_ms") is not None
            and item.get("complete_ts_ms") is not None,
        ),
        "direct_dma_duration": _count(
            completed,
            lambda item: item.get("start_timestamp_semantics") == "dma_start"
            and item.get("start_ts_ms") is not None
            and item.get("complete_ts_ms") is not None,
        ),
        "hicache_api_submit_to_complete_duration": _count(
            completed,
            lambda item: item.get("start_timestamp_semantics")
            == "hicache_api_submit_begin"
            and item.get("start_ts_ms") is not None
            and item.get("complete_ts_ms") is not None,
        ),
        "submit_to_complete_duration": _count(
            completed,
            lambda item: item.get("submit_ts_ms") is not None
            and item.get("complete_ts_ms") is not None,
        ),
    }
    direction_counts = Counter(str(item.get("direction")) for item in records)
    kind_counts = Counter(str(item.get("command_kind")) for item in records)
    return {
        "attempt_count": total,
        "physical_operation_count": physical_total,
        "rejected_before_physical_operation_count": len(rejected),
        "direction_counts": dict(sorted(direction_counts.items())),
        "command_kind_counts": dict(sorted(kind_counts.items())),
        "field_counts": field_counts,
        "field_coverage": {
            key: _fraction(value, physical_total)
            for key, value in field_counts.items()
        },
        "conditioning_ready": physical_total > 0 and all(
            field_counts[key] == physical_total
            for key in (
                "direction",
                "bytes",
                "extent_or_page_count",
                "command_kind",
                "host_copy_state",
                "pinned_host_state",
                "native_hicache_concurrency",
                "allocator_overhead",
                "callback_overhead",
            )
        ),
    }


def _gates(
    action: Mapping[str, Any],
    service: Mapping[str, Any],
    matching: Mapping[str, Any],
) -> dict[str, Any]:
    exact_ready = float(action["exact_boundary_call_coverage"]) > 0.0
    demand_ready = float(matching["complete_token_demand_label_coverage"]) == 1.0
    identity_ready = float(matching["exact_native_request_id_coverage"]) == 1.0
    service_ready = service["status"] == "available"
    reentry_ready = float(action["reentry_cause_coverage"]) == 1.0
    if not identity_ready:
        decision = "repair exact native request identity before training"
    elif not demand_ready:
        decision = "repair token-demand labels before Frontier training"
    elif not service_ready:
        decision = (
            "remaining decode demand is training-ready; calibrate the independent "
            "hardware service model before timed scenario planning"
        )
    elif not exact_ready:
        decision = (
            "remaining decode demand is training-ready; keep unlock hazard "
            "disabled until exact incremental action boundaries are observed"
        )
    else:
        decision = "label gates pass; proceed to cross-workload calibration"
    return {
        "request_identity_training_ready": identity_ready,
        "unlock_hazard_training_ready": (
            exact_ready and demand_ready and identity_ready and service_ready
        ),
        "llm_result_boundary_fallback_required": not exact_ready,
        "remaining_decode_demand_training_ready": demand_ready and identity_ready,
        "hardware_service_characterization_ready": service_ready,
        "reentry_label_collection_ready": reentry_ready,
        "reentry_model_training_ready": False,
        "reentry_model_training_blocker": (
            "missing explicit reentry or censor cause for some action calls"
            if not reentry_ready
            else "coverage is identifiable, but one fixed trace does not establish "
            "cross-workload support or calibration"
        ),
        "decision": decision,
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P6CoverageError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise P6CoverageError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise P6CoverageError(
                    f"expected JSON object at {path}:{line_number}"
                )
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_sequence(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    result = int(value)
    return result if result >= 0 else None


def _count(records: Iterable[Mapping[str, Any]], predicate: Any) -> int:
    return sum(bool(predicate(item)) for item in records)


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
