#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_mean_ci(
    values: list[float], *, samples: int = 2_000
) -> list[float] | None:
    if len(values) < 2:
        return None
    generator = random.Random(0)
    estimates = [
        mean(generator.choice(values) for _ in values) for _ in range(samples)
    ]
    return [_percentile(estimates, 2.5), _percentile(estimates, 97.5)]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"analysis must be a JSON object: {path}")
    value["_analysis_path"] = str(path.resolve())
    return value


def _correctness_passed(
    measurement: dict[str, Any], analysis_path: Path
) -> bool:
    raw_path = measurement.get("correctness_evidence_path")
    if not raw_path:
        return False
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (analysis_path.parent / path).resolve()
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return isinstance(value, dict) and value.get("passed") is True


def _pair_gate(
    analysis: dict[str, Any],
    *,
    minimum_coverage: float,
    maximum_sequence_delta: int,
) -> tuple[list[str], dict[str, Any]]:
    measurement = analysis.get("measurement")
    reasons: list[str] = []
    if not isinstance(measurement, dict):
        return ["measurement_manifest_missing"], {}
    required = (
        "gpu_id",
        "bytes_class",
        "fragmentation_class",
        "pair_id",
        "repetition",
        "expected_bytes",
        "expected_extent_min",
        "expected_extent_max",
    )
    for key in required:
        if measurement.get(key) is None:
            reasons.append(f"measurement_{key}_missing")

    transfer = analysis.get("transfer") or {}
    actual_bytes = int(transfer.get("actual_bytes") or 0)
    extent_count = int(transfer.get("extent_count") or 0)
    expected_bytes = int(measurement.get("expected_bytes") or 0)
    tolerance = float(measurement.get("bytes_tolerance_fraction", 0.02))
    bytes_error = (
        abs(actual_bytes - expected_bytes) / expected_bytes
        if expected_bytes > 0
        else math.inf
    )
    if bytes_error > tolerance:
        reasons.append("actual_bytes_out_of_range")
    extent_min = int(measurement.get("expected_extent_min") or 0)
    extent_max = int(measurement.get("expected_extent_max") or 0)
    if not extent_min <= extent_count <= extent_max:
        reasons.append("extent_count_out_of_range")
    if int(analysis.get("treatment_concurrent_transfer_count") or 0) and not bool(
        analysis.get("treatment_contamination_excluded")
    ):
        reasons.append("treatment_transfer_contamination")
    if int(analysis.get("control_transfer_overlap_count") or 0):
        reasons.append("control_transfer_contamination")
    if not analysis.get("counterfactual_control_available"):
        reasons.append("independent_control_missing")
    coverage = float(analysis.get("covered_transfer_ratio") or 0.0)
    if coverage < minimum_coverage:
        reasons.append("decode_coverage_below_threshold")
    if int(analysis.get("primary_sample_count") or 0) < 4:
        reasons.append("treatment_decode_samples_insufficient")
    baseline = analysis.get("baseline") or {}
    if int(baseline.get("sample_count") or 0) < 4:
        reasons.append("control_decode_samples_insufficient")
    sequence = analysis.get("sequence_matching") or {}
    sequence_delta = sequence.get("center_delta_tokens")
    if sequence_delta is None or int(sequence_delta) > maximum_sequence_delta:
        reasons.append("sequence_match_failed")
    analysis_path = Path(str(analysis["_analysis_path"]))
    if not _correctness_passed(measurement, analysis_path):
        reasons.append("transfer_correctness_gate_failed_or_missing")
    if not transfer.get("physical_certificate_matches_telemetry"):
        reasons.append("physical_certificate_telemetry_mismatch")

    return reasons, {
        "analysis_path": str(analysis_path),
        "pair_id": measurement.get("pair_id"),
        "repetition": measurement.get("repetition"),
        "actual_bytes": actual_bytes,
        "bytes_error_fraction": bytes_error,
        "extent_count": extent_count,
        "small_extent_ratio": float(transfer.get("small_extent_ratio") or 0.0),
        "extent_bytes_min": int(transfer.get("extent_bytes_min") or 0),
        "extent_bytes_p50": int(transfer.get("extent_bytes_p50") or 0),
        "extent_bytes_max": int(transfer.get("extent_bytes_max") or 0),
        "pinned_host": transfer.get("pinned_host"),
        "command_kind": transfer.get("command_kind"),
        "transfer_completion_ms": float(
            transfer.get("start_to_complete_ms") or 0.0
        ),
        "coverage": coverage,
        "sequence_delta_tokens": sequence_delta,
        "treatment_decode_samples": int(analysis.get("primary_sample_count") or 0),
        "control_decode_samples": int(baseline.get("sample_count") or 0),
        "stall_ratio_p50_reference": float(
            (analysis.get("unhidden_interference") or {}).get(
                "stall_ratio_p50_reference", 0.0
            )
        ),
        "passed": not reasons,
        "failure_reasons": reasons,
    }


def _paths(arguments: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        if argument.is_dir():
            paths.extend(sorted(argument.rglob("d2h_overlap_analysis.json")))
        else:
            paths.append(argument)
    return sorted(set(path.resolve() for path in paths))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate paired D2H overlap measurements at run granularity."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-repetitions", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--maximum-sequence-delta", type=int, default=32)
    parser.add_argument(
        "--expected-group",
        action="append",
        default=[],
        metavar="GPU:BYTES_CLASS:FRAGMENTATION",
    )
    args = parser.parse_args()
    if args.minimum_repetitions <= 0:
        parser.error("minimum repetitions must be positive")
    paths = _paths(args.inputs)
    analyses = [_load(path) for path in paths]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected_without_group: list[dict[str, Any]] = []
    for analysis in analyses:
        reasons, run = _pair_gate(
            analysis,
            minimum_coverage=args.minimum_coverage,
            maximum_sequence_delta=args.maximum_sequence_delta,
        )
        measurement = analysis.get("measurement")
        if not isinstance(measurement, dict) or any(
            measurement.get(key) is None
            for key in ("gpu_id", "bytes_class", "fragmentation_class")
        ):
            rejected_without_group.append(run | {"failure_reasons": reasons})
            continue
        key = (
            str(measurement["gpu_id"]),
            str(measurement["bytes_class"]),
            str(measurement["fragmentation_class"]),
        )
        grouped[key].append(run)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        runs = grouped[key]
        passing = [item for item in runs if item["passed"]]
        ratios = [item["stall_ratio_p50_reference"] for item in passing]
        completion_times = [item["transfer_completion_ms"] for item in passing]
        repetitions = {str(item["repetition"]) for item in passing}
        group_reasons: list[str] = []
        if len(repetitions) < args.minimum_repetitions:
            group_reasons.append("valid_repetitions_below_threshold")
        if len(passing) != len(runs):
            group_reasons.append("one_or_more_pairs_failed")
        groups.append(
            {
                "gpu_id": key[0],
                "bytes_class": key[1],
                "fragmentation_class": key[2],
                "arm_groups": {
                    "treatment": {"valid_run_count": len(passing)},
                    "control": {"valid_run_count": len(passing)},
                },
                "pair_count": len(runs),
                "valid_pair_count": len(passing),
                "valid_repetition_count": len(repetitions),
                "run_level_stall_ratio": {
                    "mean": mean(ratios) if ratios else None,
                    "median": median(ratios) if ratios else None,
                    "bootstrap_mean_ci95": _bootstrap_mean_ci(ratios),
                    "sampling_unit": "paired_run",
                },
                "run_level_transfer_completion_ms": {
                    "mean": mean(completion_times) if completion_times else None,
                    "median": median(completion_times) if completion_times else None,
                    "min": min(completion_times) if completion_times else None,
                    "max": max(completion_times) if completion_times else None,
                    "bootstrap_mean_ci95": _bootstrap_mean_ci(completion_times),
                    "sampling_unit": "paired_run",
                },
                "performance_evidence_eligible": not group_reasons,
                "ineligible_reasons": group_reasons,
                "runs": runs,
            }
        )

    existing_keys = {
        (item["gpu_id"], item["bytes_class"], item["fragmentation_class"])
        for item in groups
    }
    for encoded in args.expected_group:
        parts = tuple(encoded.split(":"))
        if len(parts) != 3:
            parser.error(f"invalid expected group: {encoded}")
        if parts in existing_keys:
            continue
        groups.append(
            {
                "gpu_id": parts[0],
                "bytes_class": parts[1],
                "fragmentation_class": parts[2],
                "arm_groups": {
                    "treatment": {"valid_run_count": 0},
                    "control": {"valid_run_count": 0},
                },
                "pair_count": 0,
                "valid_pair_count": 0,
                "valid_repetition_count": 0,
                "run_level_stall_ratio": None,
                "run_level_transfer_completion_ms": None,
                "performance_evidence_eligible": False,
                "ineligible_reasons": ["expected_group_missing"],
                "runs": [],
            }
        )
    groups.sort(
        key=lambda item: (
            item["gpu_id"],
            item["bytes_class"],
            item["fragmentation_class"],
        )
    )

    payload = {
        "schema_version": 1,
        "method": "paired_run_aggregate",
        "minimum_repetitions": args.minimum_repetitions,
        "minimum_coverage": args.minimum_coverage,
        "maximum_sequence_delta": args.maximum_sequence_delta,
        "analysis_count": len(analyses),
        "groups": groups,
        "ungrouped_rejections": rejected_without_group,
        "all_groups_eligible": bool(groups)
        and all(item["performance_evidence_eligible"] for item in groups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
