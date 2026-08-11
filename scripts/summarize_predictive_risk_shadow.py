#!/usr/bin/env python3
"""Summarize P6 risk planning, semantic intents, and control-path overhead."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, TextIO


_VALIDATION_EVENTS = {
    "joint_plan_shadow_partial",
    "joint_plan_stale",
    "joint_plan_shadow_idle",
    "joint_plan_shadow_fallback",
    "joint_plan_would_apply",
}

_PARKED_STATES = {
    "wait_child",
    "wait_join",
    "wait_message",
    "wait_tool",
}

_SMALL_EXTENT_THRESHOLD_BYTES = 64 * 1024 * 1024


def _bundle_generation_fingerprint(
    bundle_evidence: object,
) -> tuple[str, list[tuple[str, str, int, int]]]:
    canonical: list[tuple[str, str, int, int]] = []
    if isinstance(bundle_evidence, (list, tuple)):
        for item in bundle_evidence:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            canonical.append(
                (str(item[0]), str(item[1]), int(item[2]), int(item[3]))
            )
    canonical.sort()
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True)
    fingerprint = hashlib.blake2b(
        payload.encode("utf-8"), digest_size=12, person=b"bkv-morph"
    ).hexdigest()
    return fingerprint, canonical


def _prepare_morphology_record(
    risk_record: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object] | None:
    if str(candidate.get("action") or "") != "prepare_host":
        return None
    certificate = candidate.get("action_certificate")
    if not isinstance(certificate, Mapping):
        return None
    generation_fingerprint, bundles = _bundle_generation_fingerprint(
        certificate.get("bundle_evidence")
    )
    copied = [item for item in bundles if item[2] > item[3]]
    if not copied:
        return None
    copy_bytes = sum(gpu_bytes - cpu_bytes for _, _, gpu_bytes, cpu_bytes in copied)
    extent_count = len(copied)
    extent_sizes = sorted(
        gpu_bytes - cpu_bytes for _, _, gpu_bytes, cpu_bytes in copied
    )
    target_context_id = str(certificate.get("target_context_id") or "")
    outer_context_id = str(risk_record.get("target_context_id") or "")
    target_invocation_id = (
        str(risk_record.get("target_invocation_id") or "")
        if outer_context_id == target_context_id
        else ""
    )
    derived_invocation_id = ""
    if target_context_id.endswith(":context:root"):
        derived_invocation_id = target_context_id.removesuffix(":context:root") + ":root"
    elif target_context_id.startswith("deepagents-context:"):
        derived_invocation_id = target_context_id.replace(
            "deepagents-context:", "deepagents-invocation:", 1
        )

    invocation_evidence: list[tuple[str, str, float, str | None]] = []
    for item in certificate.get("invocation_evidence") or ():
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        invocation_evidence.append(
            (
                str(item[0]),
                str(item[1]),
                float(item[2]),
                None if item[3] is None else str(item[3]),
            )
        )
    selected_invocation = next(
        (
            item
            for item in invocation_evidence
            if item[0] in {target_invocation_id, derived_invocation_id}
        ),
        None,
    )
    if selected_invocation is None:
        parked = [item for item in invocation_evidence if item[1] in _PARKED_STATES]
        selected_invocation = parked[0] if len(parked) == 1 else None
    invocation_state = selected_invocation[1] if selected_invocation else "unknown"
    join_id = selected_invocation[3] if selected_invocation else None

    join_mode = "none"
    join_satisfied: bool | None = None
    join_completed_count = 0
    for item in certificate.get("join_evidence") or ():
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        if join_id is None or str(item[0]) != join_id:
            continue
        join_mode = str(item[1])
        join_satisfied = bool(item[2])
        join_completed_count = len(item[3]) if isinstance(item[3], (list, tuple)) else 0
        break
    join_state = (
        "none"
        if join_satisfied is None
        else f"{join_mode}:{'satisfied' if join_satisfied else 'pending'}"
    )

    context_epoch = None
    for item in certificate.get("context_epochs") or ():
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and str(item[0]) == target_context_id
        ):
            context_epoch = int(item[1])
            break

    duration_ms = None
    service_evidence = certificate.get("transfer_service_evidence")
    if isinstance(service_evidence, (list, tuple)) and len(service_evidence) >= 3:
        d2h_bytes_per_ms = float(service_evidence[0])
        if d2h_bytes_per_ms > 0:
            duration_ms = copy_bytes / d2h_bytes_per_ms + float(service_evidence[2])
    gib = 1024**3
    return {
        "source_snapshot_id": str(
            certificate.get("source_snapshot_id")
            or risk_record.get("source_snapshot_id")
            or ""
        ),
        "package_id": str(candidate.get("package_id") or ""),
        "target_context_id": target_context_id,
        "target_invocation_id": (
            selected_invocation[0] if selected_invocation else target_invocation_id
        ),
        "context_epoch": context_epoch,
        "invocation_state": invocation_state,
        "join_id": join_id,
        "join_state": join_state,
        "join_completed_count": join_completed_count,
        "copy_bytes": copy_bytes,
        "extent_count": extent_count,
        "extents_per_gib": extent_count / (copy_bytes / gib),
        "extent_bytes_min": extent_sizes[0],
        "extent_bytes_p50": int(_quantile([float(item) for item in extent_sizes], 0.50)),
        "extent_bytes_max": extent_sizes[-1],
        "small_extent_ratio": (
            sum(item < _SMALL_EXTENT_THRESHOLD_BYTES for item in extent_sizes)
            / extent_count
        ),
        "small_extent_threshold_bytes": _SMALL_EXTENT_THRESHOLD_BYTES,
        # The current action certificate is a flat closure list and does not
        # retain parent links. Do not infer a fictitious Radix depth from IDs.
        "closure_depth": None,
        "closure_depth_source": "unavailable_in_action_certificate",
        "bundle_generation_fingerprint": generation_fingerprint,
        "candidate_duration_ms": duration_ms,
        "candidate_duration_source": (
            "action_certificate_transfer_service_evidence"
            if duration_ms is not None
            else "unavailable"
        ),
    }


def _morphology_distribution(records: list[dict[str, object]]) -> dict[str, object]:
    states = Counter(str(item["invocation_state"]) for item in records)
    joins = Counter(str(item["join_state"]) for item in records)
    extents = [float(item["extent_count"]) for item in records]
    copy_bytes = [float(item["copy_bytes"]) for item in records]
    extent_bytes_min = [float(item["extent_bytes_min"]) for item in records]
    extent_bytes_p50 = [float(item["extent_bytes_p50"]) for item in records]
    extent_bytes_max = [float(item["extent_bytes_max"]) for item in records]
    small_extent_ratios = [float(item["small_extent_ratio"]) for item in records]
    durations = [
        float(item["candidate_duration_ms"])
        for item in records
        if item.get("candidate_duration_ms") is not None
    ]
    return {
        "count": len(records),
        "context_count": len({str(item["target_context_id"]) for item in records}),
        "by_invocation_state": dict(sorted(states.items())),
        "by_join_state": dict(sorted(joins.items())),
        "at_least_22_extents": sum(value >= 22 for value in extents),
        "at_least_50_extents": sum(value >= 50 for value in extents),
        "extent_count": {
            "p50": _quantile(extents, 0.50),
            "p90": _quantile(extents, 0.90),
            "max": max(extents, default=0.0),
        },
        "copy_bytes": {
            "p50": _quantile(copy_bytes, 0.50),
            "p90": _quantile(copy_bytes, 0.90),
            "max": max(copy_bytes, default=0.0),
        },
        "extent_bytes": {
            "min_observed": min(extent_bytes_min, default=0.0),
            "p50_of_extent_p50": _quantile(extent_bytes_p50, 0.50),
            "max_observed": max(extent_bytes_max, default=0.0),
        },
        "small_extent_ratio": {
            "p50": _quantile(small_extent_ratios, 0.50),
            "p90": _quantile(small_extent_ratios, 0.90),
        },
        "closure_depth_supported_count": sum(
            item.get("closure_depth") is not None for item in records
        ),
        "candidate_duration_ms": {
            "supported_count": len(durations),
            "p50": _quantile(durations, 0.50),
            "p90": _quantile(durations, 0.90),
            "max": max(durations, default=0.0),
        },
    }


def _stable_parked_episode_key(item: Mapping[str, object]) -> tuple[object, ...]:
    """Collapse physical-generation churn within one stable parked episode."""

    return (
        str(item["target_context_id"]),
        item.get("context_epoch"),
        str(item["invocation_state"]),
        str(item.get("join_id") or ""),
        str(item["join_state"]),
        int(item["copy_bytes"]),
        int(item["extent_count"]),
        int(item["extent_bytes_min"]),
        int(item["extent_bytes_p50"]),
        int(item["extent_bytes_max"]),
        round(float(item["small_extent_ratio"]), 6),
    )


def _context_weighted_morphology(
    records: list[dict[str, object]],
) -> dict[str, object]:
    by_context: dict[str, list[dict[str, object]]] = {}
    for item in records:
        by_context.setdefault(str(item["target_context_id"]), []).append(item)
    summaries: list[dict[str, object]] = []
    for context_id, context_records in sorted(by_context.items()):
        high_parked = [
            item
            for item in context_records
            if str(item["invocation_state"]) in {"wait_join", "wait_tool"}
            and int(item["extent_count"]) >= 22
        ]
        summaries.append(
            {
                "target_context_id": context_id,
                "stable_parked_episode_count": len(context_records),
                "high_fragment_parked_episode_count": len(high_parked),
                "has_high_fragment_parked_closure": bool(high_parked),
                "max_extent_count": max(
                    int(item["extent_count"]) for item in context_records
                ),
            }
        )
    high_contexts = sum(
        bool(item["has_high_fragment_parked_closure"]) for item in summaries
    )
    return {
        "sampling_unit": "target_context_id",
        "context_count": len(summaries),
        "high_fragment_parked_context_count": high_contexts,
        "high_fragment_parked_context_ratio": (
            high_contexts / len(summaries) if summaries else 0.0
        ),
        "contexts": summaries,
    }


def _morphology_audit(records: list[dict[str, object]]) -> dict[str, object]:
    unique: dict[tuple[str, str], dict[str, object]] = {}
    for item in records:
        key = (
            str(item["target_context_id"]),
            str(item["bundle_generation_fingerprint"]),
        )
        unique.setdefault(key, item)
    unique_records = list(unique.values())
    stable_episodes: dict[tuple[object, ...], dict[str, object]] = {}
    for item in unique_records:
        stable_episodes.setdefault(_stable_parked_episode_key(item), item)
    stable_episode_records = list(stable_episodes.values())
    qualifying = [
        item
        for item in stable_episode_records
        if str(item["invocation_state"]) in {"wait_join", "wait_tool"}
        and int(item["extent_count"]) >= 22
    ]
    qualifying_contexts = {
        str(item["target_context_id"]) for item in qualifying
    }
    return {
        "physical_generation_deduplication_key": (
            "target_context_id+bundle_generation_fingerprint"
        ),
        "stable_parked_episode_deduplication_key": (
            "target_context_id+context_epoch+parked_state+join+stable_morphology_tuple"
        ),
        "high_fragment_threshold_extents": 22,
        "candidate_epoch_distribution": _morphology_distribution(records),
        # A physical generation is not an independent workload sample. This
        # view is retained only to characterize within-context Radix churn.
        "unique_context_generation_distribution": _morphology_distribution(
            unique_records
        ),
        "unique_context_generations": unique_records,
        "stable_parked_episode_distribution": _morphology_distribution(
            stable_episode_records
        ),
        "stable_parked_episodes": stable_episode_records,
        "context_weighted_distribution": _context_weighted_morphology(
            stable_episode_records
        ),
        "gate": {
            "requirement": (
                "high-fragment parked closures in at least two distinct real "
                "WAIT_JOIN or WAIT_TOOL contexts"
            ),
            "qualifying_stable_parked_episode_count": len(qualifying),
            "qualifying_context_count": len(qualifying_contexts),
            "passed": len(qualifying) >= 2 and len(qualifying_contexts) >= 2,
        },
    }


def _open(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(0, index)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()
    authority_counts: Counter[str] = Counter()
    projection_counts: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    candidate_reasons: Counter[str] = Counter()
    planning_ms: list[float] = []
    planning_ms_by_status: dict[str, list[float]] = {}
    phase_ms: dict[str, list[float]] = {
        name: []
        for name in (
            "eligibility_ms",
            "queue_wait_ms",
            "belief_compose_ms",
            "candidate_generation_ms",
            "deterministic_preflight_ms",
            "scenario_risk_ms",
            "action_certificate_validation_ms",
            "trigger_to_validation_ms",
        )
    }
    evaluated_action_counts: Counter[str] = Counter()
    predictive_plan_ids: set[str] = set()
    validated_plan_ids: set[str] = set()
    stale_plan_ids: set[str] = set()
    chance_evaluated = 0
    positive_benefit_count = 0
    eligible_candidate_count = 0
    expected_recourse_credit_ms = 0.0
    chance_rejected = 0
    hbm_chance_rejected = 0
    max_peak = 0
    max_overflow = 0
    failed_count = 0
    risk_record_count = 0
    certificate_count = 0
    certificate_fresh_count = 0
    certificate_stale_count = 0
    certificate_stale_reasons: Counter[str] = Counter()
    service_cache_hits = 0
    service_cache_misses = 0
    eligibility_counts: Counter[str] = Counter()
    eligibility_suppression_reasons: Counter[str] = Counter()
    eligibility_gate_ms: list[float] = []
    no_candidate_gate_ms: list[float] = []
    enqueue_ms: list[float] = []
    worker_summary: dict[str, object] | None = None
    morphology_records: list[dict[str, object]] = []

    with _open(args.audit_path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event is None and isinstance(record.get("candidate_summaries"), list):
                event = "predictive_risk_shadow"
            if event == "predictive_risk_eligibility":
                eligibility_counts["checked"] += 1
                probe_ms = float(record.get("eligibility_ms") or 0.0)
                eligibility_gate_ms.append(probe_ms)
                enqueue_ms.append(float(record.get("enqueue_ms") or 0.0))
                prefetch_count = int(record.get("prefetch_target_count") or 0)
                prepare_count = int(
                    record.get("prepare_host_victim_count") or 0
                )
                if prefetch_count + prepare_count == 0:
                    eligibility_counts["no_candidate"] += 1
                    no_candidate_gate_ms.append(probe_ms)
                else:
                    eligibility_counts["has_candidate"] += 1
                if bool(record.get("enqueued", False)):
                    eligibility_counts["enqueued"] += 1
                else:
                    eligibility_counts["suppressed"] += 1
                suppression_reason = record.get("suppression_reason")
                if suppression_reason:
                    eligibility_suppression_reasons[str(suppression_reason)] += 1
                continue
            if event == "predictive_risk_worker_summary":
                raw_worker = record.get("worker")
                if isinstance(raw_worker, dict):
                    worker_summary = dict(raw_worker)
                continue
            if event == "predictive_risk_shadow_failed":
                failed_count += 1
                continue
            if event == "predictive_risk_shadow":
                risk_record_count += 1
                plan_id = str(record.get("source_joint_plan_id") or "")
                if plan_id:
                    predictive_plan_ids.add(plan_id)
                status = str(record.get("status") or "unknown")
                action = str(record.get("selected_action") or "unknown")
                status_counts[status] += 1
                action_counts[action] += 1
                if status == "evaluated":
                    evaluated_action_counts[action] += 1
                support_counts[str(record.get("support_level") or "unknown")] += 1
                authority_counts[
                    str(record.get("decision_authority") or "unknown")
                ] += 1
                planning_value = float(record.get("planning_ms") or 0.0)
                planning_ms.append(planning_value)
                planning_ms_by_status.setdefault(status, []).append(planning_value)
                for field_name in ("eligibility_ms", "queue_wait_ms"):
                    if record.get(field_name) is not None:
                        phase_ms[field_name].append(float(record[field_name]))
                if status == "evaluated":
                    for field_name in (
                        "belief_compose_ms",
                        "candidate_generation_ms",
                        "deterministic_preflight_ms",
                        "scenario_risk_ms",
                        "action_certificate_validation_ms",
                        "trigger_to_validation_ms",
                    ):
                        if record.get(field_name) is not None:
                            phase_ms[field_name].append(float(record[field_name]))
                certificate_count += int(
                    record.get("action_certificate_count") or 0
                )
                certificate_fresh_count += int(
                    record.get("action_certificate_fresh_count") or 0
                )
                certificate_stale_count += int(
                    record.get("action_certificate_stale_count") or 0
                )
                service_cache_hits += int(record.get("service_cache_hits") or 0)
                service_cache_misses += int(
                    record.get("service_cache_misses") or 0
                )
                raw_stale_reasons = record.get(
                    "action_certificate_stale_reasons"
                )
                if isinstance(raw_stale_reasons, dict):
                    certificate_stale_reasons.update(
                        {
                            str(reason): int(count)
                            for reason, count in raw_stale_reasons.items()
                        }
                    )
                for reason in record.get("blocked_reasons") or ():
                    blocked_reasons[str(reason)] += 1
                for summary in record.get("candidate_summaries") or ():
                    if not isinstance(summary, dict):
                        continue
                    morphology = _prepare_morphology_record(record, summary)
                    if morphology is not None:
                        morphology_records.append(morphology)
                    chance_evaluated += 1
                    projection_counts[
                        str(summary.get("scenario_projection") or "legacy_full")
                    ] += 1
                    if float(summary.get("expected_benefit_ms") or 0.0) > 0:
                        positive_benefit_count += 1
                    if bool(summary.get("eligible", False)):
                        eligible_candidate_count += 1
                    expected_recourse_credit_ms += float(
                        summary.get("expected_recourse_credit_ms") or 0.0
                    )
                    reasons = {str(item) for item in summary.get("reasons") or ()}
                    candidate_reasons.update(reasons)
                    if "future_chance_constraint" in reasons:
                        chance_rejected += 1
                    if "future_hbm_chance_constraint" in reasons:
                        hbm_chance_rejected += 1
                    max_peak = max(
                        max_peak,
                        int(summary.get("worst_future_hbm_peak_bytes") or 0),
                    )
                    max_overflow = max(
                        max_overflow,
                        int(summary.get("worst_future_hbm_overflow_bytes") or 0),
                    )
                continue
            if event in _VALIDATION_EVENTS:
                plan_id = str(record.get("plan_id") or "")
                if not plan_id:
                    continue
                validated_plan_ids.add(plan_id)
                if not bool(record.get("strict_global_fresh", True)) or not bool(
                    record.get("readset_fresh", True)
                ):
                    stale_plan_ids.add(plan_id)

    matched_validations = predictive_plan_ids.intersection(validated_plan_ids)
    matched_stale = predictive_plan_ids.intersection(stale_plan_ids)
    summary = {
        "development_only": True,
        "decision_authority_counts": dict(sorted(authority_counts.items())),
        "predictive_record_count": risk_record_count,
        "predictive_source_plan_count": len(predictive_plan_ids),
        "predictive_failed_count": failed_count,
        "status_counts": dict(sorted(status_counts.items())),
        "selected_action_counts": dict(sorted(action_counts.items())),
        "evaluated_action_counts": dict(sorted(evaluated_action_counts.items())),
        "support_level_counts": dict(sorted(support_counts.items())),
        "blocked_reason_counts": dict(sorted(blocked_reasons.items())),
        "candidate_reason_counts": dict(sorted(candidate_reasons.items())),
        "candidate_evaluation_count": chance_evaluated,
        "positive_benefit_candidate_count": positive_benefit_count,
        "eligible_candidate_count": eligible_candidate_count,
        "scenario_projection_counts": dict(sorted(projection_counts.items())),
        "summed_expected_recourse_credit_ms": expected_recourse_credit_ms,
        "future_chance_rejected_count": chance_rejected,
        "future_hbm_chance_rejected_count": hbm_chance_rejected,
        "worst_future_hbm_peak_bytes": max_peak,
        "worst_future_hbm_overflow_bytes": max_overflow,
        "agent_morphology_audit": _morphology_audit(morphology_records),
        "planning_ms": {
            "p50": _quantile(planning_ms, 0.50),
            "p95": _quantile(planning_ms, 0.95),
            "p99": _quantile(planning_ms, 0.99),
            "max": max(planning_ms, default=0.0),
        },
        "planning_ms_by_status": {
            status: {
                "count": len(values),
                "p50": _quantile(values, 0.50),
                "p95": _quantile(values, 0.95),
                "p99": _quantile(values, 0.99),
                "max": max(values, default=0.0),
            }
            for status, values in sorted(planning_ms_by_status.items())
        },
        "phase_ms": {
            name: {
                "count": len(values),
                "p50": _quantile(values, 0.50),
                "p95": _quantile(values, 0.95),
                "p99": _quantile(values, 0.99),
                "max": max(values, default=0.0),
            }
            for name, values in phase_ms.items()
        },
        "action_specific_validation": {
            "certificate_count": certificate_count,
            "fresh_count": certificate_fresh_count,
            "stale_count": certificate_stale_count,
            "stale_rate": (
                certificate_stale_count / certificate_count
                if certificate_count
                else 0.0
            ),
            "stale_reason_counts": dict(
                sorted(certificate_stale_reasons.items())
            ),
        },
        "service_estimate_cache": {
            "hits": service_cache_hits,
            "misses": service_cache_misses,
            "hit_rate": (
                service_cache_hits
                / (service_cache_hits + service_cache_misses)
                if service_cache_hits + service_cache_misses
                else 0.0
            ),
        },
        "eligibility_gate": {
            "counts": dict(sorted(eligibility_counts.items())),
            "suppression_reason_counts": dict(
                sorted(eligibility_suppression_reasons.items())
            ),
            "all_ms": {
                "count": len(eligibility_gate_ms),
                "p50": _quantile(eligibility_gate_ms, 0.50),
                "p95": _quantile(eligibility_gate_ms, 0.95),
                "p99": _quantile(eligibility_gate_ms, 0.99),
                "max": max(eligibility_gate_ms, default=0.0),
            },
            "no_candidate_ms": {
                "count": len(no_candidate_gate_ms),
                "p50": _quantile(no_candidate_gate_ms, 0.50),
                "p95": _quantile(no_candidate_gate_ms, 0.95),
                "p99": _quantile(no_candidate_gate_ms, 0.99),
                "max": max(no_candidate_gate_ms, default=0.0),
            },
            "enqueue_ms": {
                "count": len(enqueue_ms),
                "p50": _quantile(enqueue_ms, 0.50),
                "p95": _quantile(enqueue_ms, 0.95),
                "p99": _quantile(enqueue_ms, 0.99),
                "max": max(enqueue_ms, default=0.0),
            },
        },
        "predictive_worker": worker_summary,
        "performance_gates": {
            "eligibility_p95_le_10ms": (
                _quantile(eligibility_gate_ms, 0.95) <= 10.0
                if eligibility_gate_ms
                else None
            ),
            "eligibility_p99_le_20ms": (
                _quantile(eligibility_gate_ms, 0.99) <= 20.0
                if eligibility_gate_ms
                else None
            ),
            "predictive_worker_no_failure": (
                int(worker_summary.get("failed_count") or 0) == 0
                if worker_summary is not None
                else None
            ),
            "predictive_worker_no_pending": (
                int(worker_summary.get("pending_count") or 0) == 0
                if worker_summary is not None
                else None
            ),
        },
        "source_joint_plan_validation_legacy": {
            "matched_count": len(matched_validations),
            "unmatched_count": len(predictive_plan_ids - validated_plan_ids),
            "match_rate": (
                len(matched_validations) / len(predictive_plan_ids)
                if predictive_plan_ids
                else 0.0
            ),
            "stale_count": len(matched_stale),
            "stale_rate": (
                len(matched_stale) / len(matched_validations)
                if matched_validations
                else 0.0
            ),
            "scope": "global_source_plan_diagnostic_not_predictive_action_freshness",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
