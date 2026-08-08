from __future__ import annotations

from collections import Counter
import json
from math import sqrt
from pathlib import Path
from typing import Any, Iterable

from beliefkv.metrics.summary import percentile
from beliefkv.policy.service_curve import TransferServiceCurve
from beliefkv.policy.transfer_cost import PCIeCostModel
from beliefkv.runtime.protocol import (
    CommandStatus,
    TransferDirection,
    TransferTelemetry,
)


def validate_transfer_audit(
    audit_path: Path,
    *,
    fallback: PCIeCostModel | None = None,
    service_curve_window: int = 256,
    service_curve_min_samples: int = 8,
    holdout_fraction: float = 0.2,
    max_underestimation_rate: float = 0.1,
) -> dict[str, Any]:
    """Validate command integrity and a chronological service-curve holdout."""

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if not 0 <= max_underestimation_rate <= 1:
        raise ValueError("max_underestimation_rate must be in [0, 1]")

    records = _read_jsonl(audit_path)
    run_ids = sorted({str(item["run_id"]) for item in records if item.get("run_id")})
    if len(run_ids) > 1:
        raise ValueError(f"audit contains multiple run_ids: {run_ids!r}")

    dispatched: dict[str, dict[str, Any]] = {}
    acknowledged: dict[str, dict[str, Any]] = {}
    telemetry_records: dict[str, dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()
    resources: list[dict[str, Any]] = []
    timing_summaries: list[dict[str, Any]] = []
    for record in records:
        event = record.get("event")
        if event == "resource_snapshot":
            resources.append(record)
            continue
        if event == "controller_timing_summary":
            timing_summaries.append(record)
            continue
        if event not in {
            "transfer_dispatched",
            "transfer_acknowledged",
            "transfer_telemetry",
        }:
            continue
        command_id = str(record.get("command_id", ""))
        target = {
            "transfer_dispatched": dispatched,
            "transfer_acknowledged": acknowledged,
            "transfer_telemetry": telemetry_records,
        }[str(event)]
        if command_id in target:
            duplicate_counts[str(event)] += 1
        target[command_id] = record

    dispatch_ids = set(dispatched)
    ack_ids = set(acknowledged)
    command_telemetry_records = {
        command_id: record
        for command_id, record in telemetry_records.items()
        if record.get("telemetry_origin") != "native_hicache_callback"
    }
    command_telemetry_ids = set(command_telemetry_records)
    telemetry_origin_counts = Counter(
        str(record.get("telemetry_origin", "backend_telemetry"))
        for record in telemetry_records.values()
    )
    expected_dma_ids = {
        command_id
        for command_id, record in dispatched.items()
        if _has_dma_action(record)
    }
    ordering_violations = 0
    timestamp_violations = 0
    byte_bound_violations = 0
    valid_telemetry: list[tuple[int, TransferTelemetry]] = []
    for command_id, record in telemetry_records.items():
        if command_id in command_telemetry_records:
            dispatch = dispatched.get(command_id)
            ack = acknowledged.get(command_id)
            if dispatch is None or ack is None or not (
                int(dispatch.get("sequence", -1))
                < int(ack.get("sequence", -1))
                < int(record.get("sequence", -1))
            ):
                ordering_violations += 1
        submit = float(record.get("submit_ts_ms", -1))
        start = record.get("start_ts_ms")
        complete = float(record.get("complete_ts_ms", -1))
        if submit < 0 or complete < submit or (
            start is not None and not submit <= float(start) <= complete
        ):
            timestamp_violations += 1
            continue
        if int(record.get("actual_bytes", 0)) > int(record.get("closure_bytes", 0)):
            byte_bound_violations += 1
            continue
        try:
            observation = _parse_telemetry(record)
        except (TypeError, ValueError):
            timestamp_violations += 1
            continue
        valid_telemetry.append((int(record.get("sequence", 0)), observation))

    ack_byte_bound_violations = sum(
        int(record.get("actual_bytes", 0))
        > int(dispatched[command_id].get("selected_bytes", 0))
        for command_id, record in acknowledged.items()
        if command_id in dispatched
    )
    status_counts = Counter(
        str(record.get("status", "unknown")) for record in telemetry_records.values()
    )
    direction_counts = Counter(
        str(record.get("direction", "unknown"))
        for record in telemetry_records.values()
    )

    report = {
        "run_id": run_ids[0] if run_ids else "unscoped",
        "source_path": str(audit_path.resolve()),
        "command_integrity": {
            "dispatch_count": len(dispatched),
            "ack_count": len(acknowledged),
            "telemetry_count": len(telemetry_records),
            "command_telemetry_count": len(command_telemetry_records),
            "native_telemetry_count": telemetry_origin_counts.get(
                "native_hicache_callback", 0
            ),
            "telemetry_origin_counts": dict(
                sorted(telemetry_origin_counts.items())
            ),
            "expected_dma_command_count": len(expected_dma_ids),
            "missing_ack_count": len(dispatch_ids - ack_ids),
            "orphan_ack_count": len(ack_ids - dispatch_ids),
            "missing_expected_dma_telemetry_count": len(
                expected_dma_ids - command_telemetry_ids
            ),
            "telemetry_without_dispatch_count": len(
                command_telemetry_ids - dispatch_ids
            ),
            "telemetry_without_ack_count": len(
                command_telemetry_ids - ack_ids
            ),
            "ordering_violation_count": ordering_violations,
            "timestamp_violation_count": timestamp_violations,
            "telemetry_byte_bound_violation_count": byte_bound_violations,
            "ack_byte_bound_violation_count": ack_byte_bound_violations,
            "duplicate_counts": dict(sorted(duplicate_counts.items())),
            "telemetry_status_counts": dict(sorted(status_counts.items())),
            "telemetry_direction_counts": dict(sorted(direction_counts.items())),
            "all_commands_acknowledged": dispatch_ids == ack_ids,
            "ack_precedes_every_command_telemetry": ordering_violations == 0,
            "ack_precedes_every_telemetry": ordering_violations == 0,
        },
        "resource_consistency": _resource_consistency(resources),
        "admission_liveness": _admission_liveness(records),
        "controller_telemetry_overhead": _controller_overhead(timing_summaries),
        "transfer_retry_guard": _retry_guard_metrics(
            records,
            dispatched=dispatched,
            acknowledged=acknowledged,
        ),
        "physical_bundle_characterization": _physical_bundle_metrics(
            records,
            dispatched=dispatched,
            acknowledged=acknowledged,
        ),
        "service_curve_holdout": _validate_service_curve(
            valid_telemetry,
            fallback=fallback or PCIeCostModel(),
            window=service_curve_window,
            min_samples=service_curve_min_samples,
            holdout_fraction=holdout_fraction,
            max_underestimation_rate=max_underestimation_rate,
        ),
    }
    report["command_integrity"]["passes"] = all(
        (
            report["command_integrity"]["all_commands_acknowledged"],
            report["command_integrity"]["ack_precedes_every_telemetry"],
            report["command_integrity"]["missing_expected_dma_telemetry_count"] == 0,
            report["command_integrity"]["timestamp_violation_count"] == 0,
            report["command_integrity"]["telemetry_byte_bound_violation_count"] == 0,
            report["command_integrity"]["ack_byte_bound_violation_count"] == 0,
        )
    )
    return report


def _admission_liveness(records: list[dict[str, Any]]) -> dict[str, Any]:
    deferred: dict[str, float] = {}
    waits: list[float] = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        event = record.get("event")
        request_id = str(record.get("request_id", ""))
        if event == "request_deferred" and request_id:
            deferred[request_id] = float(record.get("ts_ms", 0.0))
        elif event == "request_admitted" and request_id in deferred:
            waits.append(
                max(0.0, float(record.get("ts_ms", 0.0)) - deferred[request_id])
            )
        elif event == "admission_decision":
            reason_counts[str(record.get("reason", "unknown"))] += 1
    if not waits:
        return {
            "available": False,
            "admitted_sample_count": 0,
            "reason_counts": dict(sorted(reason_counts.items())),
        }
    return {
        "available": True,
        "admitted_sample_count": len(waits),
        "wait_p50_ms": percentile(waits, 50),
        "wait_p90_ms": percentile(waits, 90),
        "wait_p95_ms": percentile(waits, 95),
        "wait_p99_ms": percentile(waits, 99),
        "wait_mean_ms": sum(waits) / len(waits),
        "wait_max_ms": max(waits),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _retry_guard_metrics(
    records: list[dict[str, Any]],
    *,
    dispatched: dict[str, dict[str, Any]],
    acknowledged: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    blocked = [
        item for item in records if item.get("event") == "transfer_attempt_blocked"
    ]
    suppressed = [
        item for item in records if item.get("event") == "transfer_retry_suppressed"
    ]
    released = [
        item for item in records if item.get("event") == "transfer_retry_released"
    ]
    rekeyed = [
        item for item in records if item.get("event") == "transfer_retry_rekeyed"
    ]
    summaries = [
        item for item in records if item.get("event") == "transfer_retry_guard_summary"
    ]
    latest_summary = summaries[-1] if summaries else {}
    active_identities: set[tuple[str, str, str, str]] = set()
    retry_without_release_count = 0
    for item in records:
        event = item.get("event")
        if event not in {
            "transfer_attempt_blocked",
            "transfer_retry_released",
        }:
            continue
        identity = (
            str(item.get("context_id", "")),
            str(item.get("context_epoch", "")),
            str(item.get("command_kind", "")),
            str(item.get("bundle_id", "")),
        )
        if event == "transfer_attempt_blocked":
            if identity in active_identities:
                retry_without_release_count += 1
            active_identities.add(identity)
        else:
            active_identities.discard(identity)

    blocker_counts: Counter[str] = Counter()
    blocked_context_counts: Counter[str] = Counter()
    failed_fingerprints: Counter[tuple[str, str, str, str, str]] = Counter()
    unknown_attempts = 0
    for item in blocked:
        codes = {
            str(code) for code in item.get("blocker_codes", ()) if str(code)
        }
        blocker_counts.update(codes)
        if "unknown_backend" in codes:
            unknown_attempts += 1
        blocked_context_counts[str(item.get("context_id", "unknown"))] += 1
        failed_fingerprints[
            (
                str(item.get("context_id", "")),
                str(item.get("context_epoch", "")),
                str(item.get("command_kind", "")),
                str(item.get("bundle_id", "")),
                str(item.get("closure_fingerprint", "")),
            )
        ] += 1

    fingerprint_dispatches: Counter[tuple[str, str, str, str, str]] = Counter()
    zero_byte_rejects: Counter[tuple[str, str, str, str, str]] = Counter()
    for command_id, dispatch in dispatched.items():
        fingerprint = str(dispatch.get("closure_fingerprint") or "")
        if not fingerprint:
            continue
        key = (
            str(dispatch.get("context_id", "")),
            str(dispatch.get("context_epoch", "")),
            str(dispatch.get("kind", "")),
            str(dispatch.get("bundle_id", "")),
            fingerprint,
        )
        fingerprint_dispatches[key] += 1
        ack = acknowledged.get(command_id)
        if (
            ack is not None
            and int(ack.get("actual_bytes", 0)) == 0
            and str(ack.get("status", "")) in {"rejected", "partial", "stale"}
        ):
            zero_byte_rejects[key] += 1

    release_latencies = [
        float(item.get("ts_ms", 0.0)) - float(item.get("failed_ts_ms", 0.0))
        for item in released
        if item.get("failed_ts_ms") is not None
        and float(item.get("ts_ms", 0.0)) >= float(item.get("failed_ts_ms", 0.0))
    ]
    max_dispatches = max(fingerprint_dispatches.values(), default=0)
    max_zero_byte_rejects = max(zero_byte_rejects.values(), default=0)
    max_failed_attempts = max(failed_fingerprints.values(), default=0)
    suppressed_count = int(
        latest_summary.get(
            "suppressed_retry_count",
            sum(int(item.get("suppressed_count", 1)) for item in suppressed),
        )
    )
    return {
        "available": bool(blocked or suppressed or released or summaries),
        "blocked_attempt_count": int(
            latest_summary.get("blocked_attempt_count", len(blocked))
        ),
        "suppressed_retry_count": suppressed_count,
        "released_attempt_count": int(
            latest_summary.get("released_attempt_count", len(released))
        ),
        "rekeyed_attempt_count": len(rekeyed),
        "active_blocked_attempt_count": int(
            latest_summary.get("active_blocked_attempt_count", 0)
        ),
        "unknown_circuit_open_count": int(
            latest_summary.get("unknown_circuit_open_count", 0)
        ),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "unknown_blocker_attempt_count": unknown_attempts,
        "unknown_blocker_ratio": (
            unknown_attempts / len(blocked) if blocked else 0.0
        ),
        "max_blocked_attempts_per_context": max(
            blocked_context_counts.values(), default=0
        ),
        "max_submissions_per_physical_fingerprint": max_dispatches,
        "max_failed_attempts_per_physical_fingerprint": max_failed_attempts,
        "max_zero_byte_rejects_per_physical_fingerprint": max_zero_byte_rejects,
        "identical_zero_byte_retry_count": sum(
            max(0, count - 1) for count in zero_byte_rejects.values()
        ),
        "identical_failed_attempt_retry_count": sum(
            max(0, count - 1) for count in failed_fingerprints.values()
        ),
        "same_snapshot_single_submit": max_dispatches <= 1,
        "same_snapshot_single_failed_attempt": max_failed_attempts <= 1,
        "retry_without_release_count": retry_without_release_count,
        "event_gated_retry_integrity": retry_without_release_count == 0,
        "retry_release_latency_p50_ms": percentile(release_latencies, 50),
        "retry_release_latency_p95_ms": percentile(release_latencies, 95),
        "retry_release_sample_count": len(release_latencies),
    }


def _physical_bundle_metrics(
    records: list[dict[str, Any]],
    *,
    dispatched: dict[str, dict[str, Any]],
    acknowledged: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    previews = [
        item for item in records if item.get("event") == "physical_bundle_preview"
    ]
    context_leases = [
        item for item in records if item.get("event") == "context_lease_issued"
    ]
    bundle_leases = [
        item for item in records if item.get("event") == "bundle_lease_aggregated"
    ]
    bundle_dispatches = {
        command_id: item
        for command_id, item in dispatched.items()
        if str(item.get("bundle_id", ""))
    }
    if not previews and not bundle_dispatches:
        return {
            "available": False,
            "preview_count": 0,
            "bundle_dispatch_count": 0,
            "context_lease_event_count": len(context_leases),
            "bundle_lease_event_count": len(bundle_leases),
            "bundle_scope_snapshot_counts": {},
            "bundle_dispatch_scope_counts": {},
        }

    preview_keys = {
        (
            str(item.get("bundle_id", "")),
            str(item.get("generation_fingerprint", "")),
        )
        for item in previews
    }
    eligible = [item for item in previews if bool(item.get("eligible", False))]
    blocked = [item for item in previews if not bool(item.get("eligible", False))]
    blocker_counts: Counter[str] = Counter()
    lease_counts: Counter[str] = Counter()
    command_kind_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    locked_bytes = 0
    gpu_bytes = 0
    exclusive_action_bytes = 0
    cross_context_action_bytes = 0
    shared_owner_snapshots = 0
    closure_ratios: list[float] = []
    for item in previews:
        compact_blockers = item.get("blocker_histogram")
        if isinstance(compact_blockers, dict):
            blocker_counts.update(
                {
                    str(code): int(count)
                    for code, count in compact_blockers.items()
                }
            )
        else:
            blocker_counts.update(
                str(code) for code in item.get("blocker_codes", ())
            )
        lease_counts[str(item.get("lease_kind", "unknown"))] += 1
        command_kind_counts[str(item.get("command_kind", "unknown"))] += 1
        scope_counts[str(item.get("bundle_scope", "unknown"))] += 1
        locked_bytes += int(item.get("locked_bytes", 0))
        gpu_bytes += int(item.get("gpu_bytes", 0))
        exclusive_action_bytes += int(item.get("exclusive_action_bytes", 0))
        cross_context_action_bytes += int(
            item.get("cross_context_action_bytes", 0)
        )
        owner_context_count = int(
            item.get(
                "owner_context_count",
                len(item.get("owner_context_ids", ())),
            )
        )
        if owner_context_count > 1:
            shared_owner_snapshots += 1
        action_bytes = int(item.get("closure_bytes", 0))
        if action_bytes > 0:
            closure_ratios.append(
                int(item.get("physical_unique_bytes", action_bytes)) / action_bytes
            )

    status_counts: Counter[str] = Counter()
    dispatch_scope_counts: Counter[str] = Counter()
    unmatched_preview_dispatches = 0
    predictable_rejections = 0
    expected_reclaimable = 0
    actual_reclaimed = 0
    absolute_reclaim_error = 0
    predictable_codes = {
        "ancestor_closure",
        "descendant_closure",
        "engine_busy",
        "extent_mutated",
        "inflight",
        "node_loading",
        "node_locked",
        "semantic_pin",
        "stale_generation",
        "unsealed",
    }
    for command_id, dispatch in bundle_dispatches.items():
        dispatch_scope_counts[str(dispatch.get("bundle_scope", "unknown"))] += 1
        key = (
            str(dispatch.get("bundle_id", "")),
            str(dispatch.get("closure_fingerprint", "")),
        )
        if key not in preview_keys:
            unmatched_preview_dispatches += 1
        ack = acknowledged.get(command_id)
        if ack is None:
            status_counts["missing_ack"] += 1
            continue
        status = str(ack.get("status", "unknown"))
        status_counts[status] += 1
        blocker_codes = {
            str(code) for code in ack.get("blocker_codes", ()) if str(code)
        }
        if status in {"partial", "rejected", "stale"} and (
            blocker_codes & predictable_codes
        ):
            predictable_rejections += 1
        if str(dispatch.get("kind", "")) == "offload_context":
            expected = int(dispatch.get("expected_reclaimable_bytes", 0))
            actual = int(ack.get("actual_bytes", 0))
            expected_reclaimable += expected
            actual_reclaimed += actual
            absolute_reclaim_error += abs(expected - actual)

    dispatch_count = len(bundle_dispatches)
    partial_or_rejected = sum(
        status_counts.get(status, 0)
        for status in ("partial", "rejected", "stale")
    )
    return {
        "available": True,
        "preview_count": len(previews),
        "eligible_preview_count": len(eligible),
        "blocked_preview_count": len(blocked),
        "unique_bundle_count": len(
            {str(item.get("bundle_id", "")) for item in previews}
        ),
        "context_lease_event_count": len(context_leases),
        "bundle_lease_event_count": len(bundle_leases),
        "lease_snapshot_counts": dict(sorted(lease_counts.items())),
        "command_kind_snapshot_counts": dict(sorted(command_kind_counts.items())),
        "bundle_scope_snapshot_counts": dict(sorted(scope_counts.items())),
        "exclusive_action_bytes_snapshot": exclusive_action_bytes,
        "cross_context_action_bytes_snapshot": cross_context_action_bytes,
        "preview_blocker_counts": dict(sorted(blocker_counts.items())),
        "shared_owner_preview_count": shared_owner_snapshots,
        "locked_gpu_snapshot_ratio": locked_bytes / gpu_bytes if gpu_bytes else 0.0,
        "physical_to_action_bytes_p50": percentile(closure_ratios, 50),
        "physical_to_action_bytes_p95": percentile(closure_ratios, 95),
        "bundle_dispatch_count": dispatch_count,
        "bundle_dispatch_scope_counts": dict(
            sorted(dispatch_scope_counts.items())
        ),
        "bundle_dispatch_status_counts": dict(sorted(status_counts.items())),
        "bundle_partial_or_reject_rate": (
            partial_or_rejected / dispatch_count if dispatch_count else 0.0
        ),
        "predictable_blocker_reject_count": predictable_rejections,
        "dispatch_without_matching_preview_count": unmatched_preview_dispatches,
        "expected_reclaimable_bytes": expected_reclaimable,
        "actual_reclaimed_bytes": actual_reclaimed,
        "reclaim_realization_ratio": (
            actual_reclaimed / expected_reclaimable
            if expected_reclaimable
            else 0.0
        ),
        "absolute_reclaim_error_bytes": absolute_reclaim_error,
    }


def _validate_service_curve(
    records: list[tuple[int, TransferTelemetry]],
    *,
    fallback: PCIeCostModel,
    window: int,
    min_samples: int,
    holdout_fraction: float,
    max_underestimation_rate: float,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: item[0])
    completed = [
        item
        for item in ordered
        if item[1].status == CommandStatus.COMPLETED
        and item[1].actual_bytes > 0
        and item[1].start_ts_ms is not None
    ]
    if len(completed) < 2:
        return {
            "evaluable": False,
            "completed_operation_count": len(completed),
            "reason": "at least two completed physical operations are required",
            "passes_point_estimate": False,
        }

    split = max(1, min(len(completed) - 1, int(len(completed) * (1 - holdout_fraction))))
    first_holdout_sequence = completed[split][0]
    training = [item for item in ordered if item[0] < first_holdout_sequence]
    holdout = completed[split:]
    curve = TransferServiceCurve(
        fallback,
        window=window,
        min_samples=min_samples,
    )
    for _, observation in training:
        curve.observe(observation)

    predictions: list[dict[str, Any]] = []
    for _, observation in holdout:
        estimate = curve.estimate(
            observation.direction,
            observation.actual_bytes,
            compute_phase=observation.compute_phase,
        )
        actual_ms = observation.complete_ts_ms - observation.submit_ts_ms
        predictions.append(
            {
                "direction": observation.direction.value,
                "actual_ms": actual_ms,
                "estimated_ms": estimate.estimated_callback_ms,
                "underestimated": actual_ms > estimate.estimated_callback_ms,
                "source": estimate.source,
            }
        )

    underestimated = sum(bool(item["underestimated"]) for item in predictions)
    rate = underestimated / len(predictions)
    confidence_low, confidence_high = _wilson_interval(
        underestimated, len(predictions)
    )
    return {
        "evaluable": True,
        "completed_operation_count": len(completed),
        "training_completed_count": split,
        "holdout_count": len(predictions),
        "holdout_fraction": holdout_fraction,
        "underestimated_count": underestimated,
        "underestimation_rate": rate,
        "max_underestimation_rate": max_underestimation_rate,
        "wilson_95_interval": [confidence_low, confidence_high],
        "passes_point_estimate": rate < max_underestimation_rate,
        "confidence_proves_threshold": confidence_high < max_underestimation_rate,
        "estimate_source_counts": dict(
            sorted(Counter(str(item["source"]) for item in predictions).items())
        ),
        "actual_callback_p90_ms": percentile(
            [float(item["actual_ms"]) for item in predictions], 90
        ),
        "estimated_callback_p90_ms": percentile(
            [float(item["estimated_ms"]) for item in predictions], 90
        ),
        "actual_over_estimate_p90": percentile(
            [
                float(item["actual_ms"]) / max(float(item["estimated_ms"]), 1e-12)
                for item in predictions
            ],
            90,
        ),
        "by_direction": {
            direction.value: _prediction_group(
                [item for item in predictions if item["direction"] == direction.value]
            )
            for direction in TransferDirection
        },
        "compute_wait_observed_count": sum(
            item[1].compute_wait_ms is not None for item in ordered
        ),
    }


def _prediction_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "underestimated_count": 0, "underestimation_rate": 0.0}
    underestimated = sum(bool(item["underestimated"]) for item in records)
    return {
        "count": len(records),
        "underestimated_count": underestimated,
        "underestimation_rate": underestimated / len(records),
    }


def _resource_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    allocator_minus_mirror: list[float] = []
    host_mismatches = 0
    host_inflight_mismatches = 0
    host_quiescent_mismatches = 0
    hbm_mirror_exceeds_allocator = 0
    hbm_capacity_mismatches = 0
    host_capacity_mismatches = 0
    for record in records:
        hbm = _optional_int(record.get("hbm_used_bytes"))
        hbm_mirror = _optional_int(record.get("page_index_gpu_bytes"))
        host = _optional_int(record.get("host_used_bytes"))
        host_mirror = _optional_int(record.get("page_index_cpu_bytes"))
        if hbm is not None and hbm_mirror is not None:
            allocator_minus_mirror.append(float(hbm - hbm_mirror))
            if hbm_mirror > hbm:
                hbm_mirror_exceeds_allocator += 1
        if host is not None and host_mirror is not None and host != host_mirror:
            host_mismatches += 1
            inflight = _optional_int(record.get("inflight_command_count"))
            if inflight is not None and inflight > 0:
                host_inflight_mismatches += 1
            else:
                host_quiescent_mismatches += 1
        if all(record.get(key) is not None for key in (
            "hbm_used_bytes", "hbm_free_bytes", "hbm_capacity_bytes"
        )) and int(record["hbm_used_bytes"]) + int(record["hbm_free_bytes"]) != int(
            record["hbm_capacity_bytes"]
        ):
            hbm_capacity_mismatches += 1
        if all(record.get(key) is not None for key in (
            "host_used_bytes", "host_free_bytes", "host_capacity_bytes"
        )) and int(record["host_used_bytes"]) + int(record["host_free_bytes"]) != int(
            record["host_capacity_bytes"]
        ):
            host_capacity_mismatches += 1
    return {
        "sample_count": len(records),
        "peak_hbm_used_bytes": max(
            (int(item.get("hbm_used_bytes") or 0) for item in records), default=None
        ),
        "peak_host_used_bytes": max(
            (int(item.get("host_used_bytes") or 0) for item in records), default=None
        ),
        "host_page_index_mismatch_count": host_mismatches,
        "host_page_index_inflight_mismatch_count": host_inflight_mismatches,
        "host_page_index_quiescent_mismatch_count": host_quiescent_mismatches,
        "hbm_mirror_exceeds_allocator_count": hbm_mirror_exceeds_allocator,
        "allocator_minus_hbm_mirror_p50_bytes": percentile(
            allocator_minus_mirror, 50
        ),
        "allocator_minus_hbm_mirror_p99_bytes": percentile(
            allocator_minus_mirror, 99
        ),
        "allocator_minus_hbm_mirror_max_bytes": max(
            allocator_minus_mirror, default=0.0
        ),
        "hbm_capacity_mismatch_count": hbm_capacity_mismatches,
        "host_capacity_mismatch_count": host_capacity_mismatches,
        "host_residency_matches_page_index": host_quiescent_mismatches == 0,
        "hbm_mirror_is_allocator_subset": hbm_mirror_exceeds_allocator == 0,
    }


def _controller_overhead(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "available": False,
            "passes_event_tick_p99": False,
            "reason": "controller_timing_summary is absent from this run",
        }
    latest = records[-1]
    ratio = float(latest.get("telemetry_event_overhead_ratio_p99", 0.0))
    return {
        "available": True,
        "scheduler_step_sample_count": int(
            latest.get("scheduler_step_sample_count", 0)
        ),
        "telemetry_event_step_count": int(
            latest.get("telemetry_event_step_count", 0)
        ),
        "telemetry_event_count": int(latest.get("telemetry_event_count", 0)),
        "scheduler_step_p99_ms": float(latest.get("scheduler_step_p99_ms", 0.0)),
        "telemetry_event_step_p99_ms": float(
            latest.get("telemetry_event_step_p99_ms", 0.0)
        ),
        "telemetry_event_overhead_p99_ms": float(
            latest.get("telemetry_event_overhead_p99_ms", 0.0)
        ),
        "telemetry_event_overhead_ratio_p99": ratio,
        "max_ratio": 0.05,
        "passes_event_tick_p99": ratio < 0.05,
    }


def _parse_telemetry(record: dict[str, Any]) -> TransferTelemetry:
    return TransferTelemetry(
        command_id=str(record["command_id"]),
        submit_ts_ms=float(record["submit_ts_ms"]),
        start_ts_ms=(
            float(record["start_ts_ms"])
            if record.get("start_ts_ms") is not None
            else None
        ),
        first_layer_ready_ts_ms=(
            float(record["first_layer_ready_ts_ms"])
            if record.get("first_layer_ready_ts_ms") is not None
            else None
        ),
        complete_ts_ms=float(record["complete_ts_ms"]),
        compute_wait_ms=(
            float(record["compute_wait_ms"])
            if record.get("compute_wait_ms") is not None
            else None
        ),
        actual_bytes=int(record.get("actual_bytes", 0)),
        closure_bytes=int(record.get("closure_bytes", 0)),
        merged_operation_count=int(record.get("merged_operation_count", 0)),
        direction=TransferDirection(str(record["direction"])),
        source_tier=str(record["source_tier"]),
        target_tier=str(record["target_tier"]),
        status=CommandStatus(str(record["status"])),
        reason=str(record.get("reason", "")),
        page_count=int(record.get("page_count", 0)),
        context_id=(
            str(record["context_id"])
            if record.get("context_id") is not None
            else None
        ),
        context_epoch=(
            int(record["context_epoch"])
            if record.get("context_epoch") is not None
            else None
        ),
        command_kind=str(record.get("command_kind", "")),
        compute_phase=str(record.get("compute_phase", "unknown")),
        host_copy_state=str(record.get("host_copy_state", "unknown")),
        pinned_host=(
            bool(record["pinned_host"])
            if record.get("pinned_host") is not None
            else None
        ),
        native_concurrent_bytes=max(
            0, int(record.get("native_concurrent_bytes", 0))
        ),
        allocator_wait_ms=(
            float(record["allocator_wait_ms"])
            if record.get("allocator_wait_ms") is not None
            else None
        ),
        allocator_submit_ms=(
            float(record["allocator_submit_ms"])
            if record.get("allocator_submit_ms") is not None
            else None
        ),
        callback_overhead_ms=(
            float(record["callback_overhead_ms"])
            if record.get("callback_overhead_ms") is not None
            else None
        ),
        start_timestamp_semantics=str(
            record.get("start_timestamp_semantics", "unavailable")
        ),
    )


def _has_dma_action(record: dict[str, Any]) -> bool:
    actions = record.get("action_counts")
    if not isinstance(actions, dict):
        return False
    return int(actions.get("start_d2h", 0)) > 0 or int(
        actions.get("start_h2d", 0)
    ) > 0


def _wilson_interval(successes: int, samples: int) -> tuple[float, float]:
    if samples <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    observed = successes / samples
    denominator = 1 + z * z / samples
    center = (observed + z * z / (2 * samples)) / denominator
    margin = (
        z
        * sqrt(
            observed * (1 - observed) / samples
            + z * z / (4 * samples * samples)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
