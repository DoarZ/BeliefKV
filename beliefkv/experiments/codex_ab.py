from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from beliefkv.metrics.summary import mean, percentile


_TOKEN_USAGE = re.compile(r"token usage: ([0-9.]+)")
_RUNNING_REQUESTS = re.compile(r"#running-req: ([0-9]+)")
_QUEUED_REQUESTS = re.compile(r"#queue-req: ([0-9]+)")
_DECODE_THROUGHPUT = re.compile(r"gen throughput \(token/s\): ([0-9.]+)")
_PREFILL_NEW_TOKENS = re.compile(r"#new-token: ([0-9]+)")
_PREFILL_CACHED_TOKENS = re.compile(r"#cached-token: ([0-9]+)")
_KV_TRANSFER_KINDS = frozenset(
    {"offload_context", "shadow_context", "prefetch_context"}
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "mean": mean(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "max": max(samples, default=0.0),
    }


def summarize_server_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    prefill_lines = [line for line in text.splitlines() if "Prefill batch." in line]
    token_usage = [float(value) for value in _TOKEN_USAGE.findall(text)]
    running = [int(value) for value in _RUNNING_REQUESTS.findall(text)]
    queued = [int(value) for value in _QUEUED_REQUESTS.findall(text)]
    throughput = [float(value) for value in _DECODE_THROUGHPUT.findall(text)]
    return {
        "token_usage": distribution(token_usage),
        "peak_running_requests": max(running, default=0),
        "peak_queued_requests": max(queued, default=0),
        "decode_throughput_tokens_per_second": distribution(throughput),
        "prefill_batch_count": len(prefill_lines),
        "prefill_computed_tokens": sum(
            int(match.group(1))
            for line in prefill_lines
            if (match := _PREFILL_NEW_TOKENS.search(line)) is not None
        ),
        "prefill_cached_token_observations": sum(
            int(match.group(1))
            for line in prefill_lines
            if (match := _PREFILL_CACHED_TOKENS.search(line)) is not None
        ),
        "scheduler_exception": "Scheduler hit an exception" in text,
    }


def summarize_gpu_samples(path: Path) -> dict[str, Any]:
    columns: list[list[float]] = [[], [], [], [], []]
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 6:
                continue
            try:
                values = [float(value.strip()) for value in row[1:]]
            except ValueError:
                continue
            for target, value in zip(columns, values):
                target.append(value)
    names = (
        "memory_used_mib",
        "memory_free_mib",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "power_watts",
    )
    return {name: distribution(values) for name, values in zip(names, columns)}


def summarize_bridge(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    completed = [item for item in records if item.get("event") == "request_completed"]
    return {
        "event_counts": dict(sorted(Counter(item.get("event") for item in records).items())),
        "request_count": len(completed),
        "upstream_request_count": sum(bool(item.get("upstream_called", True)) for item in completed),
        "join_guard_injection_count": sum(
            bool(item.get("join_guard_injected")) for item in completed
        ),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in completed),
        "completion_tokens": sum(
            int(item.get("completion_tokens") or 0) for item in completed
        ),
        "request_duration_ms": distribution(
            float(item.get("duration_ms") or 0.0) for item in completed
        ),
    }


def summarize_runtime_events(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    counts = Counter(str(item.get("kind")) for item in records)
    return {"event_counts": dict(sorted(counts.items())), "event_count": len(records)}


def summarize_run(run_dir: Path, server_log: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    bridge_path = run_dir / "responses_bridge.jsonl"
    runtime_events_path = run_dir / "runtime_events.codex.jsonl"
    gpu_path = run_dir / "gpu_samples.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root_durations = {
        instance_id: float(result["durationMs"])
        for instance_id, result in manifest["root_turn_results"].items()
    }
    return {
        "run_id": manifest["run_id"],
        "manifest": manifest,
        "makespan_seconds": float(manifest["duration_seconds"]),
        "workflow_throughput_per_second": (
            float(manifest["root_workflow_count"])
            / float(manifest["duration_seconds"])
        ),
        "root_duration_ms_by_instance": dict(sorted(root_durations.items())),
        "root_duration_ms": distribution(root_durations.values()),
        "runtime_events": summarize_runtime_events(runtime_events_path),
        "bridge": summarize_bridge(bridge_path),
        "gpu": summarize_gpu_samples(gpu_path),
        "server": summarize_server_log(server_log),
        "source_artifacts": {
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "responses_bridge": {"path": str(bridge_path), "sha256": sha256(bridge_path)},
            "runtime_events": {
                "path": str(runtime_events_path),
                "sha256": sha256(runtime_events_path),
            },
            "gpu_samples": {"path": str(gpu_path), "sha256": sha256(gpu_path)},
            "server_log": {"path": str(server_log), "sha256": sha256(server_log)},
        },
    }


def summarize_reactive_audit(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    counts = Counter(str(item.get("event")) for item in records)
    selected_bytes: dict[str, int] = defaultdict(int)
    acknowledged_bytes: dict[str, int] = defaultdict(int)
    dispatched_by_id: dict[str, dict[str, Any]] = {}
    for item in records:
        if item.get("event") != "transfer_dispatched":
            continue
        command_id = str(item["command_id"])
        kind = str(item.get("kind", "unknown"))
        dispatched_by_id[command_id] = item
        selected_bytes[kind] += int(item.get("selected_bytes") or 0)
    acknowledgements = [
        item for item in records if item.get("event") == "transfer_acknowledged"
    ]
    acknowledgement_statuses = Counter(
        str(item.get("status", "unknown")) for item in acknowledgements
    )
    for item in acknowledgements:
        dispatched = dispatched_by_id.get(str(item.get("command_id")))
        kind = str(dispatched.get("kind", "unknown")) if dispatched else "unknown"
        acknowledged_bytes[kind] += int(item.get("actual_bytes") or 0)
    transfer_callback_ms = [
        float(item["ts_ms"]) - float(dispatched["ts_ms"])
        for item in acknowledgements
        if (dispatched := dispatched_by_id.get(str(item.get("command_id"))))
        and item.get("ts_ms") is not None
        and dispatched.get("ts_ms") is not None
    ]
    deliveries = [
        item for item in records if item.get("event") == "runtime_event_delivery"
    ]
    admissions = [item for item in records if item.get("event") == "admission_decision"]
    deferred_ts: dict[str, float] = {}
    for item in records:
        if item.get("event") != "request_deferred" or item.get("ts_ms") is None:
            continue
        deferred_ts.setdefault(str(item.get("request_id")), float(item["ts_ms"]))
    admission_wait_ms = [
        float(item["ts_ms"]) - deferred_ts[str(item.get("request_id"))]
        for item in records
        if item.get("event") == "request_admitted"
        and item.get("ts_ms") is not None
        and str(item.get("request_id")) in deferred_ts
    ]
    local_rejection_reasons = Counter(
        str(item.get("reason", "unknown"))
        for item in records
        if item.get("event") == "transfer_rejected_local"
    )
    kv_transfer_bytes = sum(
        value
        for kind, value in acknowledged_bytes.items()
        if kind in _KV_TRANSFER_KINDS
    )
    reclamation_bytes = acknowledged_bytes.get("drop_unowned", 0)
    return {
        "event_counts": dict(sorted(counts.items())),
        "runtime_event_deliveries": len(deliveries),
        "rejected_runtime_event_deliveries": sum(
            not bool(item.get("accepted")) for item in deliveries
        ),
        "admission_decisions": len(admissions),
        "admission_deferrals": sum(not bool(item.get("admitted")) for item in admissions),
        "transfer_dispatches": len(dispatched_by_id),
        "selected_transfer_bytes_by_kind": dict(sorted(selected_bytes.items())),
        "transfer_acknowledgements": len(acknowledgements),
        "transfer_callback_latency_ms": distribution(transfer_callback_ms),
        "transfer_acknowledgement_statuses": dict(
            sorted(acknowledgement_statuses.items())
        ),
        "acknowledged_transfer_bytes_by_kind": dict(sorted(acknowledged_bytes.items())),
        "acknowledged_transfer_bytes": sum(acknowledged_bytes.values()),
        "acknowledged_kv_transfer_bytes": kv_transfer_bytes,
        "acknowledged_reclamation_bytes": reclamation_bytes,
        "physical_kv_transfer_observed": kv_transfer_bytes > 0,
        "context_epoch_advances": counts["context_epoch_advanced"],
        "admission_queue_wait_ms": distribution(admission_wait_ms),
        "local_transfer_rejection_reasons": dict(sorted(local_rejection_reasons.items())),
        "transfer_watchdog_expirations": counts["transfer_watchdog_expired"],
        "runtime_initialized": counts["runtime_initialized"] == 1,
        "runtime_shutdown": counts["runtime_shutdown"] == 1,
    }


def compare_runs(
    baseline: dict[str, Any],
    reactive: dict[str, Any],
    reactive_audit: dict[str, Any],
) -> dict[str, Any]:
    baseline_manifest = baseline["manifest"]
    reactive_manifest = reactive["manifest"]
    contract_fields = (
        "runtime",
        "model",
        "codex_version",
        "concurrency",
        "root_workflow_count",
        "subagent_count",
        "max_completion_tokens",
        "workload_manifest_sha256",
        "instance_ids",
        "enforce_child_join_guard",
    )
    mismatches = [
        field
        for field in contract_fields
        if baseline_manifest.get(field) != reactive_manifest.get(field)
    ]
    for name, run in (("baseline", baseline), ("reactive", reactive)):
        if not run["manifest"]["subagent_gate"]["passed"]:
            mismatches.append(f"{name}.subagent_gate")
        if run["server"]["scheduler_exception"]:
            mismatches.append(f"{name}.scheduler_exception")
    if reactive_audit["rejected_runtime_event_deliveries"]:
        mismatches.append("reactive.rejected_runtime_event_deliveries")
    if not reactive_audit["runtime_initialized"]:
        mismatches.append("reactive.runtime_initialized")
    if not reactive_audit["runtime_shutdown"]:
        mismatches.append("reactive.runtime_shutdown")
    if mismatches:
        raise ValueError("runs are not comparable: " + ", ".join(sorted(set(mismatches))))

    warnings: list[str] = []
    for metric in (
        "request_count",
        "upstream_request_count",
        "join_guard_injection_count",
        "prompt_tokens",
        "completion_tokens",
    ):
        baseline_value = float(baseline["bridge"][metric])
        reactive_value = float(reactive["bridge"][metric])
        if baseline_value != reactive_value:
            denominator = max(abs(baseline_value), 1.0)
            warnings.append(
                f"bridge.{metric} differs: baseline={baseline_value:g}, "
                f"reactive={reactive_value:g}, relative_delta="
                f"{(reactive_value - baseline_value) / denominator:.6f}"
            )

    baseline_duration = baseline["makespan_seconds"]
    reactive_duration = reactive["makespan_seconds"]
    baseline_completion_throughput = (
        float(baseline["bridge"]["completion_tokens"]) / baseline_duration
    )
    reactive_completion_throughput = (
        float(reactive["bridge"]["completion_tokens"]) / reactive_duration
    )
    paired = {}
    for instance_id, baseline_ms in baseline["root_duration_ms_by_instance"].items():
        reactive_ms = reactive["root_duration_ms_by_instance"][instance_id]
        paired[instance_id] = {
            "baseline_ms": baseline_ms,
            "reactive_ms": reactive_ms,
            "latency_reduction_fraction": 1.0 - reactive_ms / baseline_ms,
            "speedup": baseline_ms / reactive_ms,
        }
    return {
        "schema_version": 1,
        "comparable": True,
        "replicate_count": 1,
        "statistical_inference": "not_computed_for_single_paired_run",
        "trace_equivalent": not warnings,
        "causal_interpretation_valid": not warnings,
        "interpretation_status": (
            "matched_trace_single_replicate"
            if not warnings
            else "confounded_dynamic_trace"
        ),
        "confounding_warnings": warnings,
        "mechanism_evidence": {
            "physical_kv_transfer_observed": reactive_audit[
                "physical_kv_transfer_observed"
            ],
            "interpretation": (
                "real_hicache_transfer_acknowledged"
                if reactive_audit["physical_kv_transfer_observed"]
                else "no_d2h_or_h2d_context_transfer_completed"
            ),
        },
        "baseline": baseline,
        "reactive": reactive,
        "reactive_audit": reactive_audit,
        "effect": {
            "makespan_reduction_fraction": 1.0 - reactive_duration / baseline_duration,
            "makespan_speedup": baseline_duration / reactive_duration,
            "workflow_throughput_speedup": (
                reactive["workflow_throughput_per_second"]
                / baseline["workflow_throughput_per_second"]
            ),
            "completion_token_throughput_speedup": (
                reactive_completion_throughput / baseline_completion_throughput
                if baseline_completion_throughput > 0
                else 0.0
            ),
            "paired_root_workflows": dict(sorted(paired.items())),
            "mean_paired_root_latency_reduction_fraction": mean(
                [item["latency_reduction_fraction"] for item in paired.values()]
            ),
        },
    }
