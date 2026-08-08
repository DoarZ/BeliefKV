#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transfer_intervals(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    intervals = []
    for record in records:
        if record.get("event") != "transfer_telemetry":
            continue
        complete_ts = record.get("complete_ts_ms")
        start_ts = record.get("start_ts_ms")
        timing_source = "start_ts_ms"
        if start_ts is None:
            start_ts = record.get("submit_ts_ms")
            timing_source = "submit_ts_ms"
        if start_ts is None or complete_ts is None:
            continue
        start_ts = float(start_ts)
        complete_ts = float(complete_ts)
        if complete_ts < start_ts:
            continue
        command_kind = str(record.get("command_kind") or "")
        native = bool(
            record.get("compute_phase") == "native_hicache"
            or command_kind.startswith("native_")
        )
        intervals.append(
            {
                "start_ts_ms": start_ts,
                "complete_ts_ms": complete_ts,
                "timing_source": timing_source,
                "native_hicache": native,
                "concurrent_bytes": max(
                    0,
                    int(
                        record.get("native_concurrent_bytes")
                        or record.get("actual_bytes")
                        or 0
                    ),
                ),
            }
        )
    return tuple(intervals)


def _contention_at_sample(
    sample: dict[str, Any],
    transfer_intervals: tuple[dict[str, Any], ...],
) -> tuple[str, int, str]:
    start_ts = sample.get("service_start_ts_ms")
    if start_ts is None:
        start_ts = sample.get("launch_ts_ms")
    complete_ts = sample.get("complete_ts_ms")
    if complete_ts is None:
        complete_ts = sample.get("ts_ms")
    if start_ts is None or complete_ts is None:
        return "unknown", 0, "service_interval_unavailable"
    start_ts = float(start_ts)
    complete_ts = float(complete_ts)
    overlapping = tuple(
        item
        for item in transfer_intervals
        if start_ts < item["complete_ts_ms"]
        and item["start_ts_ms"] < complete_ts
    )
    if not overlapping:
        return "idle", 0, "observed_no_overlap"
    has_native = any(item["native_hicache"] for item in overlapping)
    has_explicit = any(not item["native_hicache"] for item in overlapping)
    state = (
        "mixed_transfer_observed"
        if has_native and has_explicit
        else "native_hicache_observed"
        if has_native
        else "explicit_transfer_observed"
    )
    native_bytes = max(
        (
            item["concurrent_bytes"]
            for item in overlapping
            if item["native_hicache"]
        ),
        default=0,
    )
    timing_sources = {item["timing_source"] for item in overlapping}
    semantics = (
        "start_to_complete_overlap"
        if timing_sources == {"start_ts_ms"}
        else "start_or_submit_to_complete_observed_upper_bound"
    )
    return state, native_bytes, semantics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join tagged calibration requests with GPU service intervals."
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--runtime-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark_path = args.benchmark_dir / "benchmark_manifest.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = {
        (str(item["split"]), str(item["case_id"])): item
        for item in benchmark.get("results", ())
    }
    request_attributes = {}
    for event in _records(args.runtime_events):
        if event.get("kind") != "llm_submit":
            continue
        attrs = event.get("attributes") or {}
        request_id = str(attrs.get("request_id") or "")
        if request_id:
            request_attributes[request_id] = attrs

    audit_records = tuple(_records(args.runtime_audit))
    transfer_intervals = _transfer_intervals(audit_records)

    rows = []
    phase_sample_index: Counter[tuple[str, str]] = Counter()
    for sample in audit_records:
        if sample.get("event") != "gpu_service_sample":
            continue
        request_rows = []
        sample_cases = []
        for request in sample.get("request_samples", ()):
            workflow_id = str(request.get("workflow_id") or "")
            if not workflow_id.startswith("service-calibration:"):
                if sample.get("request_samples"):
                    raise ValueError(
                        "controlled service batch contains a non-calibration request"
                    )
                continue
            _, split, case_id = workflow_id.split(":", 2)
            case = cases.get((split, case_id))
            if case is None:
                raise ValueError(f"unknown calibration case: {workflow_id}")
            request_id = str(request.get("request_id") or "")
            attrs = request_attributes.get(request_id, {})
            prompt_tokens = int(attrs.get("prompt_tokens") or 0)
            cache_hit_tokens = int(attrs.get("cache_hit_tokens") or 0)
            phase = str(request.get("phase") or sample.get("phase") or "unknown")
            chunk_index = phase_sample_index[(request_id, phase)]
            phase_sample_index[(request_id, phase)] += 1
            request_rows.append(
                {
                    "request_id": request_id,
                    "workflow_id": workflow_id,
                    "phase": phase,
                    "sequence_tokens_before": int(
                        request.get("sequence_tokens_before") or 0
                    ),
                    "token_delta": int(request.get("token_delta") or 0),
                    "chunk_index": chunk_index,
                    "prompt_tokens": prompt_tokens,
                    "cache_hit_tokens": cache_hit_tokens,
                    "cache_hit_ratio": (
                        cache_hit_tokens / prompt_tokens if prompt_tokens else 0.0
                    ),
                }
            )
            sample_cases.append((split, case_id, case))
        if not request_rows:
            continue
        splits = {item[0] for item in sample_cases}
        if len(splits) != 1:
            raise ValueError("one controlled batch cannot mix train and holdout cases")
        phases = sorted({str(item["phase"]) for item in request_rows})
        requested_batch_sizes = {
            int(item[2]["requested_batch_size"]) for item in sample_cases
        }
        contention_state, hicache_inflight_bytes, contention_semantics = (
            _contention_at_sample(sample, transfer_intervals)
        )
        rows.append(
            {
                "row_type": "gpu_batch_service_interval",
                "sample_id": str(sample.get("sample_id") or ""),
                "split": next(iter(splits)),
                "case_ids": sorted({item[1] for item in sample_cases}),
                "profile_ids": sorted(
                    {
                        str(item[2].get("profile_id") or item[1])
                        for item in sample_cases
                    }
                ),
                "phase": phases[0] if len(phases) == 1 else "mixed",
                "batch_size": int(sample.get("batch_size") or len(request_rows)),
                "request_count": len(request_rows),
                "request_samples": request_rows,
                "token_delta_total": sum(
                    int(item["token_delta"]) for item in request_rows
                ),
                "sequence_tokens_mean": sum(
                    int(item["sequence_tokens_before"]) for item in request_rows
                )
                / len(request_rows),
                "sequence_tokens_max": max(
                    int(item["sequence_tokens_before"]) for item in request_rows
                ),
                "requested_batch_sizes": sorted(requested_batch_sizes),
                "prefill_decode_mixed": len(phases) > 1,
                "chunk_position": (
                    "first"
                    if all(int(item["chunk_index"]) == 0 for item in request_rows)
                    else "continuation"
                ),
                "pcie_contention_state": contention_state,
                "hicache_inflight_bytes": hicache_inflight_bytes,
                "pcie_contention_timing_semantics": contention_semantics,
                "service_elapsed_ms": float(
                    sample.get("service_elapsed_ms") or 0.0
                ),
                "warmup": all(bool(item[2].get("warmup")) for item in sample_cases),
                "timing_semantics_version": sample.get("timing_semantics_version"),
                "timing_boundary": (
                    "scheduler/worker interval; controlled case, not CUDA event"
                ),
                "evidence_role": "controlled_microbenchmark",
            }
        )
    if not rows:
        raise ValueError("no tagged GPU service intervals were found")
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    sample_ids = [row["sample_id"] for row in rows]
    if any(not item for item in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("calibration export requires unique non-empty sample IDs")
    rows_path = destination / "gpu_batch_service_intervals.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "dataset_kind": "independent_gpu_service_calibration",
        "evidence_role": "controlled_microbenchmark",
        "row_count": len(rows),
        "batch_sample_count": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "timing_target": (
            "controlled runtime gpu_service_interval_v1; batch-unique and not "
            "claimed as CUDA-event kernel time"
        ),
        "pcie_contention_timing_semantics": (
            "service overlap with transfer start-to-complete when available, "
            "otherwise submit-to-complete observed upper bound"
        ),
        "source_sha256": {
            "benchmark_manifest": _sha256(benchmark_path),
            "runtime_audit": _sha256(args.runtime_audit),
            "runtime_events": _sha256(args.runtime_events),
        },
        "table_sha256": _sha256(rows_path),
    }
    (destination / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
