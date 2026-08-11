#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable

from beliefkv.policy.service_curve import TransferServiceCurve
from beliefkv.policy.transfer_cost import PCIeCostModel
from beliefkv.runtime.protocol import (
    CommandStatus,
    TransferDirection,
    TransferTelemetry,
)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _telemetry(record: dict[str, object]) -> TransferTelemetry:
    return TransferTelemetry(
        command_id=str(record["command_id"]),
        submit_ts_ms=float(record["submit_ts_ms"]),
        start_ts_ms=_optional_float(record.get("start_ts_ms")),
        first_layer_ready_ts_ms=_optional_float(
            record.get("first_layer_ready_ts_ms")
        ),
        complete_ts_ms=float(record["complete_ts_ms"]),
        compute_wait_ms=_optional_float(record.get("compute_wait_ms")),
        actual_bytes=int(record.get("actual_bytes") or 0),
        closure_bytes=int(
            record.get("closure_bytes") or record.get("actual_bytes") or 0
        ),
        merged_operation_count=int(record.get("merged_operation_count") or 0),
        direction=TransferDirection(str(record["direction"])),
        source_tier=str(record.get("source_tier") or "unknown"),
        target_tier=str(record.get("target_tier") or "unknown"),
        status=CommandStatus(str(record["status"])),
        reason=str(record.get("reason") or ""),
        page_count=int(record.get("page_count") or 0),
        context_id=(
            None if record.get("context_id") is None else str(record["context_id"])
        ),
        context_epoch=(
            None
            if record.get("context_epoch") is None
            else int(record["context_epoch"])
        ),
        command_kind=str(record.get("command_kind") or ""),
        compute_phase=str(record.get("compute_phase") or "unknown"),
        host_copy_state=str(record.get("host_copy_state") or "unknown"),
        pinned_host=(
            None
            if record.get("pinned_host") is None
            else bool(record["pinned_host"])
        ),
        native_concurrent_bytes=int(record.get("native_concurrent_bytes") or 0),
        allocator_wait_ms=_optional_float(record.get("allocator_wait_ms")),
        allocator_submit_ms=_optional_float(record.get("allocator_submit_ms")),
        callback_overhead_ms=_optional_float(record.get("callback_overhead_ms")),
        start_timestamp_semantics=str(
            record.get("start_timestamp_semantics") or "unavailable"
        ),
        extent_count=int(record.get("extent_count") or 0),
        extent_bytes_min=int(record.get("extent_bytes_min") or 0),
        extent_bytes_p50=int(record.get("extent_bytes_p50") or 0),
        extent_bytes_max=int(record.get("extent_bytes_max") or 0),
        small_extent_ratio=float(record.get("small_extent_ratio") or 0.0),
        small_extent_threshold_bytes=int(
            record.get("small_extent_threshold_bytes") or 64 * 1024 * 1024
        ),
    )


def _telemetry_records(path: Path) -> Iterable[dict[str, object]]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "transfer_telemetry":
                yield record


def _matrix_samples(
    matrix_path: Path,
) -> tuple[list[TransferTelemetry], dict[str, object]]:
    matrix_path = matrix_path.expanduser().resolve()
    aggregate = json.loads(matrix_path.read_text(encoding="utf-8"))
    samples: list[TransferTelemetry] = []
    source_groups: list[dict[str, object]] = []
    for group in aggregate.get("groups") or ():
        if (
            not isinstance(group, dict)
            or str(group.get("gpu_id")) != "0"
            or not bool(group.get("performance_evidence_eligible", False))
        ):
            continue
        source_groups.append(
            {
                "gpu_id": "0",
                "bytes_class": str(group.get("bytes_class") or "unknown"),
                "fragmentation_class": str(
                    group.get("fragmentation_class") or "unknown"
                ),
                "valid_repetition_count": int(
                    group.get("valid_repetition_count") or 0
                ),
            }
        )
        for run in group.get("runs") or ():
            if not isinstance(run, dict) or not bool(run.get("passed", False)):
                continue
            analysis_path = Path(str(run["analysis_path"])).expanduser().resolve()
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            transfer = analysis.get("transfer") or {}
            interference = analysis.get("unhidden_interference") or {}
            command_id = str(transfer.get("command_id") or "")
            telemetry_path = Path(str(analysis["transfer_telemetry"]))
            if not telemetry_path.is_absolute():
                telemetry_path = analysis_path.parent / telemetry_path
            matching = next(
                (
                    item
                    for item in _telemetry_records(telemetry_path)
                    if str(item.get("command_id") or "") == command_id
                ),
                None,
            )
            if matching is None:
                raise RuntimeError(
                    f"matrix run has no target transfer telemetry: {analysis_path}"
                )
            sample = _telemetry(matching)
            sample = replace(
                sample,
                compute_wait_ms=float(
                    interference.get("stall_ms_p50_reference") or 0.0
                ),
                # The matrix is an idle-link calibration. The target operation
                # itself is not native concurrent traffic.
                native_concurrent_bytes=0,
            )
            samples.append(sample)
    if not samples:
        raise RuntimeError("matrix contains no eligible GPU0 development samples")
    metadata = {
        "evidence_scope": "single_gpu_development",
        "model_scope": "extent_count_aware_v1",
        "conditioned_features": ["direction", "bytes", "extent_count", "contention"],
        "observed_but_not_conditioned_features": [
            "extent_bytes_min",
            "extent_bytes_p50",
            "extent_bytes_max",
            "small_extent_ratio",
        ],
        "unavailable_features": ["closure_depth"],
        "formal_crossover_complete": False,
        "source_matrix": str(matrix_path),
        "source_groups": source_groups,
        "stall_evidence": "paired_run_absolute_stall_ms_p50_reference",
    }
    return samples, metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a persistent TransferServiceCurve warm-start artifact."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--telemetry", type=Path)
    source.add_argument("--matrix-aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-key", required=True)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--min-samples", type=int, default=0)
    parser.add_argument("--fallback-bandwidth-gbps", type=float, default=24.0)
    args = parser.parse_args()

    min_samples = args.min_samples or (3 if args.matrix_aggregate else 8)
    curve = TransferServiceCurve(
        PCIeCostModel(bandwidth_gbps=args.fallback_bandwidth_gbps),
        window=args.window,
        min_samples=min_samples,
    )
    metadata: dict[str, object] = {}
    if args.matrix_aggregate is not None:
        samples, metadata = _matrix_samples(args.matrix_aggregate)
    else:
        assert args.telemetry is not None
        samples = [_telemetry(item) for item in _telemetry_records(args.telemetry)]
    accepted = 0
    for sample in samples:
        curve.observe(sample)
        accepted += 1
    if accepted == 0:
        raise RuntimeError("no transfer telemetry records were found")
    curve.save_artifact(
        args.output.expanduser(),
        hardware_key=args.hardware_key,
        schema_version=2 if args.matrix_aggregate is not None else 1,
        metadata=metadata,
    )
    print(
        json.dumps(
            {
                "accepted_records": accepted,
                "hardware_key": args.hardware_key,
                "schema_version": 2 if args.matrix_aggregate is not None else 1,
                "metadata": metadata,
                "output": str(args.output.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
