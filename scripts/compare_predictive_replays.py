#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping


def _load(path: Path) -> dict[str, Mapping[str, object]]:
    records: dict[str, Mapping[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            snapshot_id = str(record.get("source_snapshot_id") or "")
            if snapshot_id:
                records[snapshot_id] = record
    return records


def _prepare_by_package(
    record: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw in record.get("candidate_summaries", ()):
        if not isinstance(raw, Mapping) or raw.get("action") != "prepare_host":
            continue
        package_id = str(raw.get("package_id") or "")
        if package_id:
            result[package_id] = raw
    return result


def _latest_start(summary: Mapping[str, object]) -> float | None:
    diagnostics = summary.get("prepare_recourse_scenarios", ())
    values = []
    if isinstance(diagnostics, (list, tuple)):
        for item in diagnostics:
            if not isinstance(item, Mapping):
                continue
            deadline = item.get("morphology_deadline_ms")
            transfer = item.get("shape_aware_transfer_p90_ms")
            if deadline is not None and transfer is not None:
                values.append(float(deadline) - float(transfer))
    return min(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare byte-only and morphology-aware P6 risk replays."
    )
    parser.add_argument("--byte-only", type=Path, required=True)
    parser.add_argument("--morphology-aware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    byte_records = _load(args.byte_only)
    shape_records = _load(args.morphology_aware)
    paired_ids = sorted(set(byte_records).intersection(shape_records))
    counts: Counter[str] = Counter()
    duration_differences: list[float] = []
    latest_start_differences: list[float] = []
    candidate_changes: list[dict[str, object]] = []
    selected_action_changes: list[dict[str, object]] = []
    context_shape_keys: set[tuple[str, str]] = set()
    timing_change_context_shape_keys: set[tuple[str, str]] = set()
    reason_change_context_shape_keys: set[tuple[str, str]] = set()
    eligibility_change_context_shape_keys: set[tuple[str, str]] = set()
    promotion_context_shape_keys: set[tuple[str, str]] = set()
    veto_context_shape_keys: set[tuple[str, str]] = set()
    supported_promotion_context_shape_keys: set[tuple[str, str]] = set()
    supported_veto_context_shape_keys: set[tuple[str, str]] = set()
    shape_supported_context_shape_keys: set[tuple[str, str]] = set()
    for snapshot_id in paired_ids:
        byte_record = byte_records[snapshot_id]
        shape_record = shape_records[snapshot_id]
        if byte_record.get("selected_action") != shape_record.get("selected_action"):
            counts["selected_action_changed"] += 1
            selected_action_changes.append(
                {
                    "snapshot_id": snapshot_id,
                    "byte_selected_action": byte_record.get("selected_action"),
                    "shape_selected_action": shape_record.get("selected_action"),
                }
            )
        byte_candidates = _prepare_by_package(byte_record)
        shape_candidates = _prepare_by_package(shape_record)
        for package_id in sorted(set(byte_candidates).intersection(shape_candidates)):
            counts["paired_prepare_candidates"] += 1
            byte = byte_candidates[package_id]
            shape = shape_candidates[package_id]
            context_id = package_id.split(":prepare:", 1)[-1]
            context_shape_key = (
                context_id,
                str(shape.get("morphology_shape_fingerprint") or "unavailable"),
            )
            context_shape_keys.add(context_shape_key)
            for prefix, item in (("byte", byte), ("shape", shape)):
                counts[f"{prefix}_shape_supported"] += bool(
                    item.get("morphology_shape_supported")
                )
                counts[f"{prefix}_positive"] += float(
                    item.get("expected_benefit_ms") or 0.0
                ) > 0
                counts[f"{prefix}_eligible"] += bool(item.get("eligible"))
            if bool(shape.get("morphology_shape_supported")):
                shape_supported_context_shape_keys.add(context_shape_key)
            byte_duration = byte.get("morphology_shape_aware_transfer_p90_ms")
            shape_duration = shape.get("morphology_shape_aware_transfer_p90_ms")
            duration_changed = False
            if (
                byte_duration is not None
                and shape_duration is not None
                and bool(byte.get("morphology_shape_supported"))
                and bool(shape.get("morphology_shape_supported"))
            ):
                duration_difference = float(shape_duration) - float(byte_duration)
                duration_differences.append(duration_difference)
                duration_changed = abs(duration_difference) > 1.0e-6
            byte_latest = _latest_start(byte)
            shape_latest = _latest_start(shape)
            latest_start_changed = False
            if (
                byte_latest is not None
                and shape_latest is not None
                and bool(byte.get("morphology_shape_supported"))
                and bool(shape.get("morphology_shape_supported"))
            ):
                latest_start_difference = shape_latest - byte_latest
                latest_start_differences.append(latest_start_difference)
                latest_start_changed = abs(latest_start_difference) > 1.0e-6
            byte_reasons = set(str(item) for item in byte.get("reasons", ()))
            shape_reasons = set(str(item) for item in shape.get("reasons", ()))
            timing_changed = duration_changed or latest_start_changed
            reasons_changed = byte_reasons != shape_reasons
            byte_eligible = bool(byte.get("eligible"))
            shape_eligible = bool(shape.get("eligible"))
            shape_action_promotion = not byte_eligible and shape_eligible
            shape_action_veto = byte_eligible and not shape_eligible
            shape_supported = bool(shape.get("morphology_shape_supported"))
            supported_shape_action_promotion = (
                shape_action_promotion and shape_supported
            )
            supported_shape_action_veto = shape_action_veto and shape_supported
            eligibility_changed = shape_action_promotion or shape_action_veto
            if timing_changed:
                counts["timing_estimate_changed"] += 1
                timing_change_context_shape_keys.add(context_shape_key)
            if reasons_changed:
                counts["feasibility_reason_changed"] += 1
                reason_change_context_shape_keys.add(context_shape_key)
            if eligibility_changed:
                counts["candidate_eligibility_changed"] += 1
                eligibility_change_context_shape_keys.add(context_shape_key)
            if shape_action_promotion:
                counts["shape_action_promotion"] += 1
                promotion_context_shape_keys.add(context_shape_key)
            if shape_action_veto:
                counts["shape_action_veto"] += 1
                veto_context_shape_keys.add(context_shape_key)
            if supported_shape_action_promotion:
                counts["supported_shape_action_promotion"] += 1
                supported_promotion_context_shape_keys.add(context_shape_key)
            if supported_shape_action_veto:
                counts["supported_shape_action_veto"] += 1
                supported_veto_context_shape_keys.add(context_shape_key)
            if timing_changed or reasons_changed or eligibility_changed:
                candidate_changes.append(
                    {
                        "snapshot_id": snapshot_id,
                        "package_id": package_id,
                        "timing_estimate_changed": timing_changed,
                        "feasibility_reason_changed": reasons_changed,
                        "candidate_eligibility_changed": eligibility_changed,
                        "shape_action_promotion": shape_action_promotion,
                        "shape_action_veto": shape_action_veto,
                        "morphology_shape_supported": shape_supported,
                        "supported_shape_action_promotion": (
                            supported_shape_action_promotion
                        ),
                        "supported_shape_action_veto": supported_shape_action_veto,
                        "byte_eligible": byte_eligible,
                        "shape_eligible": shape_eligible,
                        "byte_reasons": sorted(byte_reasons),
                        "shape_reasons": sorted(shape_reasons),
                        "byte_transfer_ms": byte_duration,
                        "shape_transfer_ms": shape_duration,
                        "latest_start_difference_ms": (
                            shape_latest - byte_latest
                            if byte_latest is not None and shape_latest is not None
                            else None
                        ),
                    }
                )

    shape_action_gate = bool(counts["shape_action_promotion"])
    shape_veto_gate = bool(counts["shape_action_veto"])
    supported_shape_action_gate = bool(
        counts["supported_shape_action_promotion"]
    )
    supported_shape_veto_gate = bool(counts["supported_shape_action_veto"])
    selected_action_gate = bool(counts["selected_action_changed"])
    decision_relevance_gate = bool(
        shape_action_gate or shape_veto_gate or selected_action_gate
    )
    shape_natural_action_available = bool(counts["shape_eligible"])
    recommended_validation_arm = (
        "both_directional_arms"
        if supported_shape_action_gate and supported_shape_veto_gate
        else "shape_aware_prepare_canary"
        if supported_shape_action_gate
        else "byte_only_veto_treatment"
        if supported_shape_veto_gate
        else "shape_support_characterization"
        if shape_action_gate or shape_veto_gate
        else "selected_action_characterization"
        if selected_action_gate
        else "none"
    )
    for metric in (
        "candidate_eligibility_changed",
        "selected_action_changed",
        "shape_action_promotion",
        "shape_action_veto",
        "supported_shape_action_promotion",
        "supported_shape_action_veto",
    ):
        counts.setdefault(metric, 0)
    payload = {
        "byte_only": str(args.byte_only.resolve()),
        "morphology_aware": str(args.morphology_aware.resolve()),
        "paired_snapshot_count": len(paired_ids),
        "context_physical_shape_key_semantics": (
            "target context plus physical shape fingerprint; diagnostic only, "
            "not an independent workload sampling unit"
        ),
        "context_physical_shape_key_count": len(context_shape_keys),
        "shape_supported_context_physical_shape_key_count": len(
            shape_supported_context_shape_keys
        ),
        "timing_changed_context_physical_shape_key_count": len(
            timing_change_context_shape_keys
        ),
        "reason_changed_context_physical_shape_key_count": len(
            reason_change_context_shape_keys
        ),
        "eligibility_changed_context_physical_shape_key_count": len(
            eligibility_change_context_shape_keys
        ),
        "promotion_context_physical_shape_key_count": len(
            promotion_context_shape_keys
        ),
        "veto_context_physical_shape_key_count": len(veto_context_shape_keys),
        "supported_promotion_context_physical_shape_key_count": len(
            supported_promotion_context_shape_keys
        ),
        "supported_veto_context_physical_shape_key_count": len(
            supported_veto_context_shape_keys
        ),
        "counts": dict(sorted(counts.items())),
        "transfer_estimate_difference_ms": {
            "min": min(duration_differences) if duration_differences else None,
            "max": max(duration_differences) if duration_differences else None,
            "mean": (
                sum(duration_differences) / len(duration_differences)
                if duration_differences
                else None
            ),
        },
        "latest_start_difference_ms": {
            "min": min(latest_start_differences) if latest_start_differences else None,
            "max": max(latest_start_differences) if latest_start_differences else None,
        },
        "change_levels": {
            "timing_estimate_changed": bool(
                counts["timing_estimate_changed"]
            ),
            "feasibility_reason_changed": bool(
                counts["feasibility_reason_changed"]
            ),
            "candidate_eligibility_changed": bool(
                counts["candidate_eligibility_changed"]
            ),
            "selected_action_changed": bool(counts["selected_action_changed"]),
        },
        "timing_sensitivity_gate": bool(
            counts["timing_estimate_changed"]
            or counts["feasibility_reason_changed"]
        ),
        "decision_relevance_gate": decision_relevance_gate,
        # Compatibility field. It now has the strict decision-relevance
        # meaning and no longer treats a new rejection reason as a decision.
        "decision_change_gate": decision_relevance_gate,
        "shape_action_gate": shape_action_gate,
        "shape_veto_gate": shape_veto_gate,
        "supported_shape_action_gate": supported_shape_action_gate,
        "supported_shape_veto_gate": supported_shape_veto_gate,
        "selected_action_gate": selected_action_gate,
        "shape_natural_action_available": shape_natural_action_available,
        "recommended_validation_arm": recommended_validation_arm,
        # Compatibility fields. A shape-aware online canary is legal only for
        # a promotion; a veto requires a byte-only treatment arm instead.
        "natural_canary_gate": shape_natural_action_available,
        "online_canary_gate": supported_shape_action_gate,
        "candidate_changes": candidate_changes,
        "selected_action_changes": selected_action_changes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
