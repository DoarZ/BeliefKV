#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping


def _records(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


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


def _anchor_steps(sample: Mapping[str, object], workflow_id: str) -> int:
    steps = [
        int(item.get("token_delta") or 0)
        for item in sample.get("request_samples", ())
        if isinstance(item, Mapping)
        and item.get("workflow_id") == workflow_id
        and item.get("phase") == "decode"
    ]
    return max(steps, default=0)


def _sequence_tokens(sample: Mapping[str, object], workflow_id: str) -> int | None:
    values = [
        (
            int(item["effective_sequence_tokens_before"])
            if item.get("effective_sequence_tokens_before") is not None
            else int(item.get("sequence_tokens_before") or 0)
            + int(item.get("output_tokens_before") or 0)
        )
        for item in sample.get("request_samples", ())
        if isinstance(item, Mapping) and item.get("workflow_id") == workflow_id
    ]
    return max(values) if values else None


def _output_tokens(sample: Mapping[str, object], workflow_id: str) -> int | None:
    values = [
        int(item.get("output_tokens_before") or 0)
        for item in sample.get("request_samples", ())
        if isinstance(item, Mapping) and item.get("workflow_id") == workflow_id
    ]
    return max(values) if values else None


def _signature(sample: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(sample.get("phase") or "unknown"),
        int(sample.get("batch_size") or 0),
        tuple(sorted(str(item) for item in sample.get("workflow_ids", ()))),
    )


def _overlap_ms(
    sample: Mapping[str, object],
    transfer_start_ms: float,
    transfer_complete_ms: float,
) -> float:
    start = float(sample.get("service_start_ts_ms") or 0.0)
    complete = float(sample.get("complete_ts_ms") or sample.get("ts_ms") or 0.0)
    return max(
        0.0,
        min(complete, transfer_complete_ms) - max(start, transfer_start_ms),
    )


def _interval_ms(sample: Mapping[str, object]) -> float:
    start = float(sample.get("service_start_ts_ms") or 0.0)
    complete = float(sample.get("complete_ts_ms") or sample.get("ts_ms") or 0.0)
    return max(0.0, complete - start)


def _transfer_interval(
    transfer: Mapping[str, object], *, observation_end_ms: float
) -> tuple[float, float]:
    start = float(
        transfer.get("start_ts_ms")
        if transfer.get("start_ts_ms") is not None
        else transfer.get("submit_ts_ms")
        or transfer.get("ts_ms")
        or 0.0
    )
    complete_value = transfer.get("complete_ts_ms")
    if complete_value is None and str(transfer.get("status") or "") in {
        "submitted",
        "inflight",
    }:
        complete = observation_end_ms
    else:
        complete = float(complete_value or transfer.get("ts_ms") or start)
    return start, max(start, complete)


def _intervals_intersect(
    left: tuple[float, float], right: tuple[float, float]
) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0])


def _transfer_pollution(
    transfers: Iterable[Mapping[str, object]],
    samples: Iterable[Mapping[str, object]],
    *,
    observation_end_ms: float,
    excluded_command_id: str | None = None,
) -> list[dict[str, object]]:
    sample_intervals = [
        (
            float(sample.get("service_start_ts_ms") or 0.0),
            float(sample.get("complete_ts_ms") or sample.get("ts_ms") or 0.0),
        )
        for sample in samples
    ]
    polluted: dict[str, dict[str, object]] = {}
    for index, transfer in enumerate(transfers):
        if transfer.get("event") != "transfer_telemetry":
            continue
        command_id = str(transfer.get("command_id") or f"record-{index}")
        if excluded_command_id is not None and command_id == excluded_command_id:
            continue
        interval = _transfer_interval(
            transfer, observation_end_ms=observation_end_ms
        )
        if not any(_intervals_intersect(interval, item) for item in sample_intervals):
            continue
        polluted.setdefault(
            command_id,
            {
                "command_id": command_id,
                "direction": transfer.get("direction"),
                "status": transfer.get("status"),
                "command_kind": transfer.get("command_kind"),
                "start_ts_ms": interval[0],
                "complete_ts_ms": interval[1],
                "actual_bytes": int(transfer.get("actual_bytes") or 0),
            },
        )
    return [polluted[key] for key in sorted(polluted)]


def _covered_ms(
    samples: Iterable[Mapping[str, object]],
    transfer_start_ms: float,
    transfer_complete_ms: float,
) -> float:
    intervals = sorted(
        (
            max(
                transfer_start_ms,
                float(sample.get("service_start_ts_ms") or 0.0),
            ),
            min(
                transfer_complete_ms,
                float(sample.get("complete_ts_ms") or sample.get("ts_ms") or 0.0),
            ),
        )
        for sample in samples
        if _overlap_ms(sample, transfer_start_ms, transfer_complete_ms) > 0
    )
    if not intervals:
        return 0.0
    covered = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered += max(0.0, current_end - current_start)
        current_start, current_end = start, end
    return covered + max(0.0, current_end - current_start)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate unhidden D2H interference from matched SGLang GPU service "
            "intervals."
        )
    )
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument(
        "--control-runtime-audit",
        type=Path,
        default=None,
        help=(
            "Independent NO_D2H control audit. Formal interference estimates "
            "require this input."
        ),
    )
    parser.add_argument(
        "--control-transfer-telemetry",
        type=Path,
        default=None,
        help=(
            "Control-run telemetry used to reject baseline intervals with "
            "transfer overlap."
        ),
    )
    parser.add_argument("--transfer-telemetry", type=Path, required=True)
    parser.add_argument("--anchor-workflow-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-window-ms", type=float, default=30_000.0)
    parser.add_argument("--sequence-radius-tokens", type=int, default=128)
    parser.add_argument(
        "--expected-page-count",
        type=int,
        default=None,
        help=(
            "Expected physical extent/page count from the predictive action "
            "certificate."
        ),
    )
    parser.add_argument("--expected-extent-min", type=int, default=None)
    parser.add_argument("--expected-extent-max", type=int, default=None)
    parser.add_argument(
        "--measurement-manifest",
        type=Path,
        default=None,
        help="Paired-run metadata consumed by the cross-run aggregator.",
    )
    parser.add_argument("--command-id", default=None)
    args = parser.parse_args()
    if args.baseline_window_ms <= 0 or args.sequence_radius_tokens < 0:
        parser.error(
            "baseline window must be positive and sequence radius non-negative"
        )

    all_transfers = list(_records(args.transfer_telemetry))
    transfers = [
        item
        for item in all_transfers
        if item.get("event") == "transfer_telemetry"
        and item.get("direction") == "d2h"
        and item.get("status") == "completed"
        and int(item.get("actual_bytes") or 0) > 0
        and (
            args.command_id is None
            or str(item.get("command_id")) == args.command_id
        )
    ]
    if not transfers:
        raise RuntimeError("no completed non-zero D2H transfer was found")
    transfer = max(transfers, key=lambda item: int(item.get("actual_bytes") or 0))
    transfer_start_ms = float(
        transfer.get("start_ts_ms")
        if transfer.get("start_ts_ms") is not None
        else transfer["submit_ts_ms"]
    )
    transfer_submit_ms = float(transfer["submit_ts_ms"])
    transfer_complete_ms = float(transfer["complete_ts_ms"])
    transfer_duration_ms = transfer_complete_ms - transfer_start_ms
    submit_to_complete_ms = transfer_complete_ms - transfer_submit_ms
    if transfer_duration_ms <= 0:
        raise RuntimeError("selected D2H has a non-positive duration")

    runtime_records = list(_records(args.runtime_audit))
    samples = [
        item
        for item in runtime_records
        if item.get("event") == "gpu_service_sample"
        and item.get("phase") == "decode"
        and _anchor_steps(item, args.anchor_workflow_id) > 0
    ]
    overlapping = [
        item
        for item in samples
        if _overlap_ms(item, transfer_start_ms, transfer_complete_ms) > 0
    ]
    fully_overlapping = [
        item
        for item in overlapping
        if float(item.get("service_start_ts_ms") or 0.0) >= transfer_start_ms
        and float(item.get("complete_ts_ms") or item.get("ts_ms") or 0.0)
        <= transfer_complete_ms
    ]
    if not overlapping:
        raise RuntimeError("D2H did not overlap an anchor decode service interval")

    signatures: dict[tuple[object, ...], int] = {}
    for sample in overlapping:
        signature = _signature(sample)
        signatures[signature] = signatures.get(signature, 0) + 1
    matched_signature = max(signatures, key=lambda item: signatures[item])
    primary = [
        item for item in overlapping if _signature(item) == matched_signature
    ]
    treatment_observation_end_ms = max(
        (
            float(item.get("complete_ts_ms") or item.get("ts_ms") or 0.0)
            for item in samples
        ),
        default=transfer_complete_ms,
    )
    treatment_pollution = _transfer_pollution(
        all_transfers,
        primary,
        observation_end_ms=treatment_observation_end_ms,
        excluded_command_id=str(transfer.get("command_id") or ""),
    )
    contaminated_primary = [
        item
        for item in primary
        if any(
            _intervals_intersect(
                (
                    float(item.get("service_start_ts_ms") or 0.0),
                    float(item.get("complete_ts_ms") or item.get("ts_ms") or 0.0),
                ),
                (
                    float(pollution["start_ts_ms"]),
                    float(pollution["complete_ts_ms"]),
                ),
            )
            for pollution in treatment_pollution
        )
    ]
    if contaminated_primary:
        contaminated_ids = {id(item) for item in contaminated_primary}
        primary = [item for item in primary if id(item) not in contaminated_ids]
        fully_overlapping = [
            item for item in fully_overlapping if id(item) not in contaminated_ids
        ]
    if not primary:
        raise RuntimeError(
            "all target-overlapping decode intervals were contaminated by "
            "non-target transfers"
        )
    sequence_values = [
        value
        for item in primary
        if (value := _sequence_tokens(item, args.anchor_workflow_id)) is not None
    ]
    sequence_center = int(median(sequence_values)) if sequence_values else 0

    if args.control_runtime_audit is not None:
        control_samples = [
            item
            for item in _records(args.control_runtime_audit)
            if item.get("event") == "gpu_service_sample"
            and item.get("phase") == "decode"
            and _anchor_steps(item, args.anchor_workflow_id) > 0
        ]
        baseline = [
            item
            for item in control_samples
            if _signature(item) == matched_signature
            and (
                (sequence := _sequence_tokens(item, args.anchor_workflow_id))
                is not None
                and abs(sequence - sequence_center) <= args.sequence_radius_tokens
            )
        ]
        baseline_source = "independent_no_d2h_control"
    else:
        baseline = [
            item
            for item in samples
            if _signature(item) == matched_signature
            and _overlap_ms(item, transfer_start_ms, transfer_complete_ms) == 0
            and transfer_complete_ms
            < float(item.get("service_start_ts_ms") or 0.0)
            <= transfer_complete_ms + args.baseline_window_ms
            and (
                (sequence := _sequence_tokens(item, args.anchor_workflow_id))
                is not None
                and abs(sequence - sequence_center) <= args.sequence_radius_tokens
            )
        ]
        baseline_source = "post_transfer_observational"
        if len(baseline) < 8:
            baseline = [
                item
                for item in samples
                if _signature(item) == matched_signature
                and _overlap_ms(item, transfer_start_ms, transfer_complete_ms) == 0
                and transfer_start_ms - args.baseline_window_ms
                <= float(item.get("complete_ts_ms") or item.get("ts_ms") or 0.0)
                < transfer_start_ms
                and (
                    (sequence := _sequence_tokens(item, args.anchor_workflow_id))
                    is not None
                    and abs(sequence - sequence_center)
                    <= args.sequence_radius_tokens
                )
            ] + baseline
            baseline_source = "pre_and_post_transfer_observational"
    if len(baseline) < 4:
        raise RuntimeError(
            f"only {len(baseline)} matched baseline decode samples were found"
        )

    control_transfer_overlap_count = 0
    control_pollution: list[dict[str, object]] = []
    if args.control_transfer_telemetry is not None:
        control_transfers = list(_records(args.control_transfer_telemetry))
        control_observation_end_ms = max(
            (
                float(item.get("complete_ts_ms") or item.get("ts_ms") or 0.0)
                for item in baseline
            ),
            default=0.0,
        )
        control_pollution = _transfer_pollution(
            control_transfers,
            baseline,
            observation_end_ms=control_observation_end_ms,
        )
        control_transfer_overlap_count = len(control_pollution)

    baseline_per_step = [
        float(item["service_elapsed_ms"])
        / _anchor_steps(item, args.anchor_workflow_id)
        for item in baseline
    ]
    weighted_primary = [
        (
            item,
            _overlap_ms(item, transfer_start_ms, transfer_complete_ms)
            / _interval_ms(item),
        )
        for item in primary
        if _interval_ms(item) > 0
    ]
    primary_elapsed_ms = sum(
        float(item["service_elapsed_ms"]) * overlap_fraction
        for item, overlap_fraction in weighted_primary
    )
    primary_steps = sum(
        _anchor_steps(item, args.anchor_workflow_id) * overlap_fraction
        for item, overlap_fraction in weighted_primary
    )
    covered_transfer_ms = _covered_ms(
        primary, transfer_start_ms, transfer_complete_ms
    )
    covered_transfer_ratio = min(1.0, covered_transfer_ms / transfer_duration_ms)
    observed_per_step_ms = primary_elapsed_ms / primary_steps
    baseline_p10 = _percentile(baseline_per_step, 10)
    baseline_p50 = _percentile(baseline_per_step, 50)
    baseline_p90 = _percentile(baseline_per_step, 90)

    def stall_ratio(reference_per_step_ms: float) -> tuple[float, float]:
        stall_ms = max(
            0.0,
            primary_elapsed_ms - reference_per_step_ms * primary_steps,
        )
        return stall_ms, stall_ms / transfer_duration_ms

    stall_p50_ms, ratio_p50 = stall_ratio(baseline_p50)
    stall_upper_ms, ratio_upper = stall_ratio(baseline_p10)
    stall_lower_ms, ratio_lower = stall_ratio(baseline_p90)
    ineligible_reasons = ["gpu_crossover_and_repetition_not_established"]
    if args.control_runtime_audit is None:
        ineligible_reasons.append("independent_no_d2h_control_missing")
    if args.control_transfer_telemetry is None:
        ineligible_reasons.append("control_transfer_absence_not_verified")
    elif control_transfer_overlap_count:
        ineligible_reasons.append("control_baseline_overlaps_transfer")
    if (
        args.expected_page_count is None
        and args.expected_extent_min is None
        and args.expected_extent_max is None
    ):
        ineligible_reasons.append("expected_physical_layout_missing")
    elif (
        args.expected_page_count is not None
        and int(transfer.get("page_count") or 0) != args.expected_page_count
    ):
        ineligible_reasons.append("physical_layout_page_count_mismatch")
    extent_count = int(
        transfer.get("extent_count")
        if transfer.get("extent_count") is not None
        else transfer.get("page_count")
        or 0
    )
    dispatches = [
        item
        for item in runtime_records
        if item.get("event") == "transfer_dispatched"
        and str(item.get("command_id") or "")
        == str(transfer.get("command_id") or "")
    ]
    physical_certificate = dispatches[-1] if dispatches else None
    certificate_action_count = int(
        ((physical_certificate or {}).get("action_counts") or {}).get(
            "start_d2h", 0
        )
    )
    certificate_matches = bool(physical_certificate) and (
        int(physical_certificate.get("selected_bytes") or 0)
        == int(transfer.get("actual_bytes") or 0)
        and int(physical_certificate.get("page_count") or 0) == extent_count
        and certificate_action_count == extent_count
    )
    if not certificate_matches:
        ineligible_reasons.append("physical_certificate_telemetry_mismatch")
    if (
        args.expected_extent_min is not None
        and extent_count < args.expected_extent_min
    ) or (
        args.expected_extent_max is not None
        and extent_count > args.expected_extent_max
    ):
        ineligible_reasons.append("physical_layout_extent_count_out_of_range")
    measurement = None
    if args.measurement_manifest is not None:
        measurement = json.loads(
            args.measurement_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(measurement, dict):
            raise ValueError("measurement manifest must be a JSON object")
    baseline_sequence_values = [
        value
        for item in baseline
        if (value := _sequence_tokens(item, args.anchor_workflow_id)) is not None
    ]
    baseline_sequence_center = (
        int(median(baseline_sequence_values)) if baseline_sequence_values else None
    )
    payload = {
        "schema_version": 3,
        "method": "effective_sequence_matched_overlap_weighting",
        "counterfactual_control_available": (
            args.control_runtime_audit is not None
        ),
        "control_transfer_overlap_count": control_transfer_overlap_count,
        "control_transfer_pollution": control_pollution,
        "treatment_concurrent_transfer_count": len(treatment_pollution),
        "treatment_concurrent_transfers": treatment_pollution,
        "treatment_contaminated_sample_count": len(contaminated_primary),
        "treatment_contamination_policy": "exclude_decode_interval",
        "treatment_contamination_excluded": True,
        "measurement": measurement,
        "performance_evidence_eligible": False,
        "performance_evidence_ineligible_reasons": ineligible_reasons,
        "runtime_audit": str(args.runtime_audit.resolve()),
        "transfer_telemetry": str(args.transfer_telemetry.resolve()),
        "anchor_workflow_id": args.anchor_workflow_id,
        "transfer": {
            "command_id": transfer.get("command_id"),
            "command_kind": transfer.get("command_kind"),
            "actual_bytes": int(transfer.get("actual_bytes") or 0),
            "page_count": int(transfer.get("page_count") or 0),
            "extent_count": extent_count,
            "extent_bytes_min": int(transfer.get("extent_bytes_min") or 0),
            "extent_bytes_p50": int(transfer.get("extent_bytes_p50") or 0),
            "extent_bytes_max": int(transfer.get("extent_bytes_max") or 0),
            "small_extent_ratio": float(
                transfer.get("small_extent_ratio") or 0.0
            ),
            "small_extent_threshold_bytes": int(
                transfer.get("small_extent_threshold_bytes") or 0
            ),
            "pinned_host": transfer.get("pinned_host"),
            "physical_certificate": (
                {
                    "bundle_id": physical_certificate.get("bundle_id"),
                    "selected_bytes": int(
                        physical_certificate.get("selected_bytes") or 0
                    ),
                    "page_count": int(physical_certificate.get("page_count") or 0),
                    "start_d2h_action_count": certificate_action_count,
                }
                if physical_certificate is not None
                else None
            ),
            "physical_certificate_matches_telemetry": certificate_matches,
            "expected_page_count": args.expected_page_count,
            "physical_layout_page_count_matches": (
                args.expected_page_count is not None
                and int(transfer.get("page_count") or 0)
                == args.expected_page_count
            ),
            "expected_extent_min": args.expected_extent_min,
            "expected_extent_max": args.expected_extent_max,
            "physical_layout_extent_count_matches": (
                (
                    args.expected_extent_min is None
                    or extent_count >= args.expected_extent_min
                )
                and (
                    args.expected_extent_max is None
                    or extent_count <= args.expected_extent_max
                )
            ),
            "submit_ts_ms": transfer_submit_ms,
            "start_ts_ms": transfer_start_ms,
            "complete_ts_ms": transfer_complete_ms,
            "start_timestamp_semantics": transfer.get(
                "start_timestamp_semantics"
            ),
            "start_to_complete_ms": transfer_duration_ms,
            "submit_to_complete_ms": submit_to_complete_ms,
        },
        "matched_signature": list(matched_signature),
        "sequence_center_tokens": sequence_center,
        "primary_uses_boundary_samples": len(primary) > len(fully_overlapping),
        "fully_overlapping_sample_count": len(fully_overlapping),
        "all_overlapping_sample_count": len(overlapping),
        "primary_sample_count": len(primary),
        "primary_decode_steps": primary_steps,
        "primary_elapsed_ms": primary_elapsed_ms,
        "covered_transfer_ms": covered_transfer_ms,
        "covered_transfer_ratio": covered_transfer_ratio,
        "overlap_weighting": "service_interval_fraction",
        "observed_per_step_ms": observed_per_step_ms,
        "baseline": {
            "source": baseline_source,
            "sample_count": len(baseline),
            "per_step_p10_ms": baseline_p10,
            "per_step_p50_ms": baseline_p50,
            "per_step_p90_ms": baseline_p90,
        },
        "sequence_matching": {
            "semantics": (
                "effective_sequence_tokens_before or legacy "
                "sequence_tokens_before + output_tokens_before"
            ),
            "radius_tokens": args.sequence_radius_tokens,
            "baseline_sequence_center_tokens": baseline_sequence_center,
            "center_delta_tokens": (
                abs(baseline_sequence_center - sequence_center)
                if baseline_sequence_center is not None
                else None
            ),
            "primary_output_tokens_min": min(
                (
                    value
                    for item in primary
                    if (
                        value := _output_tokens(
                            item, args.anchor_workflow_id
                        )
                    )
                    is not None
                ),
                default=None,
            ),
            "primary_output_tokens_max": max(
                (
                    value
                    for item in primary
                    if (
                        value := _output_tokens(
                            item, args.anchor_workflow_id
                        )
                    )
                    is not None
                ),
                default=None,
            ),
            "baseline_output_tokens_min": min(
                (
                    value
                    for item in baseline
                    if (
                        value := _output_tokens(
                            item, args.anchor_workflow_id
                        )
                    )
                    is not None
                ),
                default=None,
            ),
            "baseline_output_tokens_max": max(
                (
                    value
                    for item in baseline
                    if (
                        value := _output_tokens(
                            item, args.anchor_workflow_id
                        )
                    )
                    is not None
                ),
                default=None,
            ),
        },
        "unhidden_interference": {
            "stall_ms_p50_reference": stall_p50_ms,
            "stall_ratio_p50_reference": ratio_p50,
            "stall_ms_lower_p90_reference": stall_lower_ms,
            "stall_ratio_lower_p90_reference": ratio_lower,
            "stall_ms_upper_p10_reference": stall_upper_ms,
            "stall_ratio_upper_p10_reference": ratio_upper,
            "positive_benefit_thresholds": [0.102, 0.141],
            "current_cvar_eligibility_thresholds": [0.0052, 0.0103],
        },
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
