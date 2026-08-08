from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def audit_paired_load_invariance(
    cohorts: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Compare identical semantic prompts across independently collected loads."""

    if len(cohorts) < 2:
        raise ValueError("paired load audit requires at least two cohorts")
    grouped: dict[
        tuple[str, ...], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    missing_prompt_digest = 0
    candidate_record_count = 0
    for cohort, raw_root in cohorts.items():
        root = Path(raw_root)
        decisions = _read_jsonl(root / "frontier_decision_points.jsonl")
        calls = {
            str(item.get("request_id") or ""): item
            for item in _read_jsonl(root / "request_calls.jsonl")
            if item.get("request_id")
        }
        for row in decisions:
            if row.get("trigger_kind") != "llm_submit":
                continue
            candidate_record_count += 1
            attrs = row.get("trigger_attributes") or {}
            prompt_digest = str(attrs.get("prompt_semantic_sha256") or "")
            if not prompt_digest:
                missing_prompt_digest += 1
                continue
            request_id = str(row.get("trigger_request_id") or "")
            feature = next(
                (
                    item
                    for item in row.get("invocations", ())
                    if str(item.get("request_id") or "") == request_id
                ),
                None,
            )
            if feature is None:
                continue
            invocation_id = str(feature.get("invocation_id") or "")
            label = next(
                (
                    item
                    for item in row.get("labels", ())
                    if str(item.get("invocation_id") or "") == invocation_id
                ),
                None,
            )
            if label is None:
                continue
            key = (
                str(row.get("project") or "unknown"),
                str(row.get("instance_id") or "unknown"),
                str(row.get("base_commit") or "unknown"),
                prompt_digest,
                str(attrs.get("sampling_seed")),
            )
            call = calls.get(request_id, {})
            grouped[key][cohort].append(
                {
                    "boundary": label.get("next_boundary_kind"),
                    "remaining_decode_tokens": label.get("remaining_output_tokens"),
                    "request_wall_clock_ms": call.get("wall_clock_ms"),
                    "ordinal": call.get("ordinal"),
                    "timestamp_ms": row.get("timestamp_ms"),
                    "sampling_seed": attrs.get("sampling_seed"),
                }
            )

    required = set(cohorts)
    pairs: list[tuple[dict[str, Any], ...]] = []
    multiplicity_mismatch_count = 0
    for values in grouped.values():
        if set(values) != required:
            continue
        ordered = {
            cohort: sorted(
                observations,
                key=lambda item: (
                    item["ordinal"] is None,
                    int(item["ordinal"] or 0),
                    float(item["timestamp_ms"] or 0.0),
                ),
            )
            for cohort, observations in values.items()
        }
        counts = {len(items) for items in ordered.values()}
        multiplicity_mismatch_count += len(counts) > 1
        for index in range(min(counts)):
            pairs.append(tuple(ordered[name][index] for name in sorted(cohorts)))

    controlled_pairs = [
        pair
        for pair in pairs
        if all(item.get("sampling_seed") is not None for item in pair)
    ]
    diagnostic = _summarize_pairs(pairs)
    controlled = _summarize_pairs(controlled_pairs)
    paired_keys = sum(set(values) == required for values in grouped.values())
    seeds = {
        str(item["sampling_seed"])
        for pair in controlled_pairs
        for item in pair
    }
    return {
        "cohorts": sorted(cohorts),
        "candidate_record_count": candidate_record_count,
        "paired_semantic_key_count": paired_keys,
        "paired_occurrence_count": len(pairs),
        "controlled_pair_count": len(controlled_pairs),
        "unpaired_semantic_key_count": len(grouped) - paired_keys,
        "multiplicity_mismatch_count": multiplicity_mismatch_count,
        "missing_prompt_digest_count": missing_prompt_digest,
        "prompt_digest_coverage": (
            1.0 - missing_prompt_digest / candidate_record_count
            if candidate_record_count
            else None
        ),
        "sampling_seed_observed": bool(controlled_pairs),
        "sampling_seeds": sorted(seeds),
        "controlled_audit_eligible": bool(controlled_pairs),
        "controlled_metrics": controlled,
        "diagnostic_metrics": diagnostic,
        "interpretation": (
            "Only controlled_metrics may support demand invariance. They require "
            "the same semantic prompt and sampling seed across loads. Diagnostic "
            "pairs without a seed cannot establish invariance. Demand stability "
            "should coexist with load-sensitive request wall-clock time."
        ),
    }


def _summarize_pairs(
    pairs: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, float | int | None]:
    action_stable = 0
    token_relative_ranges: list[float] = []
    service_relative_ranges: list[float] = []
    for observations in pairs:
        action_stable += len({item["boundary"] for item in observations}) == 1
        token_values = [
            float(item["remaining_decode_tokens"])
            for item in observations
            if item["remaining_decode_tokens"] is not None
        ]
        wall_values = [
            float(item["request_wall_clock_ms"])
            for item in observations
            if item["request_wall_clock_ms"] is not None
        ]
        if len(token_values) == len(observations):
            token_relative_ranges.append(_relative_range(token_values))
        if len(wall_values) == len(observations):
            service_relative_ranges.append(_relative_range(wall_values))
    return {
        "pair_count": len(pairs),
        "action_exact_agreement": action_stable / len(pairs) if pairs else None,
        "remaining_decode_relative_range_p50": _percentile(
            token_relative_ranges, 0.5
        ),
        "remaining_decode_relative_range_p95": _percentile(
            token_relative_ranges, 0.95
        ),
        "request_wall_clock_relative_range_p50": _percentile(
            service_relative_ranges, 0.5
        ),
        "request_wall_clock_relative_range_p95": _percentile(
            service_relative_ranges, 0.95
        ),
    }


def _relative_range(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (max(values) - min(values)) / max(abs(mean), 1.0)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
