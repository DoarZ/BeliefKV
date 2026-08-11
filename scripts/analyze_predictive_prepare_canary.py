#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Mapping


def _records(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def analyze_canary(
    records: Iterable[Mapping[str, object]],
    *,
    comparison: Mapping[str, object] | None = None,
) -> dict[str, object]:
    committed: dict[str, Mapping[str, object]] = {}
    queued: dict[str, Mapping[str, object]] = {}
    telemetry: dict[str, Mapping[str, object]] = {}
    acknowledged: dict[str, Mapping[str, object]] = {}
    terminal: dict[str, Mapping[str, object]] = {}
    commit_counts: Counter[str] = Counter()
    queue_counts: Counter[str] = Counter()
    telemetry_counts: Counter[str] = Counter()
    ack_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    paired_veto_evaluated = 0
    paired_veto_passed = 0
    shape_supported_veto_passed = 0
    counterfactual_shape_unsupported = 0
    unsupported_context_shapes: Counter[
        tuple[str, int | None, int | None, str | None]
    ] = Counter()
    intent_published = 0
    intent_publish_rejected = 0
    intent_safe_point_rejected = 0
    publish_rejection_reasons: Counter[str] = Counter()
    safe_point_rejection_reasons: Counter[str] = Counter()
    for record in records:
        event = record.get("event")
        if event == "predictive_risk_shadow":
            evidence = record.get("paired_prepare_veto")
            if isinstance(evidence, Mapping) and record.get(
                "predictive_intent"
            ) is not None:
                paired_veto_evaluated += 1
                if evidence.get("passed") is True:
                    paired_veto_passed += 1
                rejection_reasons = {
                    str(item)
                    for item in evidence.get(
                        "counterfactual_rejection_reasons", ()
                    )
                }
                explicit_shape_support = evidence.get(
                    "counterfactual_shape_supported"
                )
                shape_supported = (
                    explicit_shape_support
                    if isinstance(explicit_shape_support, bool)
                    else False
                    if "shape_unsupported" in rejection_reasons
                    else True
                    if evidence.get("passed") is True
                    else None
                )
                if evidence.get("passed") is True and shape_supported is True:
                    shape_supported_veto_passed += 1
                if shape_supported is False:
                    counterfactual_shape_unsupported += 1
                    intent = record.get("predictive_intent")
                    intent = intent if isinstance(intent, Mapping) else {}
                    extent_count = evidence.get("counterfactual_extent_count")
                    if extent_count is None:
                        extent_count = intent.get("predicted_extent_count")
                    target_bytes = intent.get("target_bytes_hint")
                    shape_fingerprint = evidence.get(
                        "counterfactual_shape_fingerprint"
                    ) or intent.get("shape_fingerprint")
                    unsupported_context_shapes[
                        (
                            str(intent.get("context_id") or "unknown"),
                            int(target_bytes) if target_bytes is not None else None,
                            int(extent_count) if extent_count is not None else None,
                            (
                                str(shape_fingerprint)
                                if shape_fingerprint is not None
                                else None
                            ),
                        )
                    ] += 1
        elif event == "predictive_semantic_intent_published":
            intent_published += 1
        elif event == "predictive_semantic_intent_publish_rejected":
            intent_publish_rejected += 1
            publish_rejection_reasons.update(
                str(item) for item in record.get("reasons", ())
            )
        elif event == "predictive_semantic_intent_rejected":
            intent_safe_point_rejected += 1
            safe_point_rejection_reasons.update(
                str(item) for item in record.get("reasons", ())
            )
        if event == "predictive_semantic_intent_committed" and record.get(
            "action"
        ) == "prepare_host":
            intent_id = str(record.get("intent_id"))
            committed[intent_id] = record
            commit_counts[intent_id] += 1
        elif event == "online_joint_residency_queued" and record.get(
            "predictive_intent_id"
        ):
            intent_id = str(record["predictive_intent_id"])
            queued[intent_id] = record
            queue_counts[intent_id] += 1
        elif event == "transfer_telemetry" and record.get("command_id"):
            command_id = str(record["command_id"])
            telemetry[command_id] = record
            telemetry_counts[command_id] += 1
        elif event == "transfer_acknowledged" and record.get("command_id"):
            command_id = str(record["command_id"])
            acknowledged[command_id] = record
            ack_counts[command_id] += 1
        elif event == "online_joint_residency_terminal" and record.get(
            "predictive_intent_id"
        ):
            intent_id = str(record["predictive_intent_id"])
            terminal[intent_id] = record
            terminal_counts[intent_id] += 1

    rows = []
    for intent_id, commit in sorted(committed.items()):
        queue = queued.get(intent_id)
        command_id = str(queue.get("command_id") or "") if queue is not None else ""
        transfer = telemetry.get(command_id)
        ack = acknowledged.get(command_id)
        finish = terminal.get(intent_id)
        actual_transfer_ms = None
        actual_stall_ms = None
        if transfer is not None:
            submit = transfer.get("submit_ts_ms")
            complete = transfer.get("complete_ts_ms")
            if submit is not None and complete is not None:
                actual_transfer_ms = max(0.0, float(complete) - float(submit))
            if transfer.get("compute_wait_ms") is not None:
                actual_stall_ms = float(transfer["compute_wait_ms"])
        predicted_transfer = commit.get("live_transfer_p90_ms")
        predicted_slack = commit.get("live_morphology_slack_ms")
        actual_slack = (
            float(predicted_slack)
            + float(predicted_transfer)
            - actual_transfer_ms
            if predicted_slack is not None
            and predicted_transfer is not None
            and actual_transfer_ms is not None
            else None
        )
        stage_presence = {
            "commit": True,
            "queue": queue is not None,
            "transfer_telemetry": transfer is not None,
            "ack": ack is not None,
            "terminal": finish is not None,
        }
        stage_cardinality = {
            "commit": commit_counts[intent_id],
            "queue": queue_counts[intent_id],
            "transfer_telemetry": telemetry_counts[command_id],
            "ack": ack_counts[command_id],
            "terminal": terminal_counts[intent_id],
        }
        stages_unique = all(count == 1 for count in stage_cardinality.values())
        transaction_id = str(queue.get("transaction_id") or "") if queue else ""
        ids_consistent = bool(
            queue is not None
            and command_id
            and str(queue.get("predictive_intent_id") or "") == intent_id
            and transfer is not None
            and str(transfer.get("command_id") or "") == command_id
            and ack is not None
            and str(ack.get("command_id") or "") == command_id
            and finish is not None
            and str(finish.get("predictive_intent_id") or "") == intent_id
            and str(finish.get("command_id") or "") == command_id
            and str(finish.get("transaction_id") or "") == transaction_id
        )
        actual_byte_values = [
            item.get("actual_bytes") if item is not None else None
            for item in (transfer, ack, finish)
        ]
        actual_bytes_consistent = bool(
            all(item is not None and int(item) > 0 for item in actual_byte_values)
            and len({int(item) for item in actual_byte_values}) == 1
        )
        copy_bytes_match_actual = bool(
            actual_bytes_consistent
            and queue is not None
            and queue.get("copy_bytes") is not None
            and int(queue["copy_bytes"]) == int(actual_byte_values[0])
        )
        commit_live_count = commit.get("live_extent_count")
        queue_live_count = queue.get("live_extent_count") if queue else None
        telemetry_count = transfer.get("extent_count") if transfer else None
        extent_count_consistent = bool(
            commit_live_count is not None
            and queue_live_count is not None
            and telemetry_count is not None
            and int(commit_live_count)
            == int(queue_live_count)
            == int(telemetry_count)
        )
        commit_live_shape = str(commit.get("live_shape_fingerprint") or "")
        queue_live_shape = (
            str(queue.get("live_shape_fingerprint") or "") if queue else ""
        )
        shape_fingerprint_consistent = bool(
            commit_live_shape
            and queue_live_shape
            and commit_live_shape == queue_live_shape
        )
        commit_transfer_model_mode = str(
            commit.get("transfer_model_mode") or ""
        )
        queue_transfer_model_mode = (
            str(queue.get("transfer_model_mode") or "") if queue else ""
        )
        transfer_model_mode_consistent = bool(
            commit_transfer_model_mode
            and commit_transfer_model_mode == queue_transfer_model_mode
        )
        commit_authority_gate = str(
            commit.get("predictive_prepare_authority_gate") or "natural"
        )
        queue_authority_gate = (
            str(queue.get("predictive_prepare_authority_gate") or "natural")
            if queue
            else ""
        )
        authority_gate_consistent = bool(
            commit_authority_gate
            and commit_authority_gate == queue_authority_gate
        )
        counterfactual_rejection_reasons = tuple(
            sorted(
                str(item)
                for item in commit.get(
                    "counterfactual_rejection_reasons", ()
                )
            )
        )
        paired_veto_evidence_present = bool(
            commit_authority_gate == "byte-only-veto"
            and commit_transfer_model_mode == "byte-only"
            and counterfactual_rejection_reasons
        )
        queue_counterfactual_rejection_reasons = tuple(
            sorted(
                str(item)
                for item in (
                    queue.get("counterfactual_rejection_reasons", ())
                    if queue
                    else ()
                )
            )
        )
        paired_veto_evidence_consistent = bool(
            commit_authority_gate != "byte-only-veto"
            or (
                paired_veto_evidence_present
                and queue_counterfactual_rejection_reasons
                == counterfactual_rejection_reasons
            )
        )
        attribution_chain_complete = bool(
            all(stage_presence.values())
            and stages_unique
            and ids_consistent
            and actual_bytes_consistent
            and copy_bytes_match_actual
            and extent_count_consistent
            and shape_fingerprint_consistent
            and transfer_model_mode_consistent
            and authority_gate_consistent
            and paired_veto_evidence_consistent
            and (
                commit_authority_gate != "byte-only-veto"
                or paired_veto_evidence_present
            )
        )
        transaction_completed = bool(
            transfer is not None
            and transfer.get("status") == "completed"
            and ack is not None
            and ack.get("status") == "completed"
            and finish is not None
            and finish.get("status") == "completed"
        )
        rows.append(
            {
                "intent_id": intent_id,
                "context_id": commit.get("context_id"),
                "intent_start_ts_ms": commit.get("ts_ms"),
                "command_id": command_id or None,
                "predicted_transfer_p90_ms": predicted_transfer,
                "actual_transfer_ms": actual_transfer_ms,
                "predicted_morphology_slack_ms": predicted_slack,
                "actual_morphology_slack_ms": actual_slack,
                "decode_stall_ms": actual_stall_ms,
                "predicted_extent_count": commit.get("predicted_extent_count"),
                "live_extent_count": commit.get("live_extent_count"),
                "shape_changed_at_safe_point": commit.get("live_shape_changed"),
                "actual_bytes": (
                    transfer.get("actual_bytes") if transfer is not None else None
                ),
                "transaction_status": (
                    finish.get("status") if finish is not None else None
                ),
                "stage_presence": stage_presence,
                "stage_cardinality": stage_cardinality,
                "stages_unique": stages_unique,
                "ids_consistent": ids_consistent,
                "actual_bytes_consistent": actual_bytes_consistent,
                "copy_bytes_match_actual": copy_bytes_match_actual,
                "extent_count_consistent": extent_count_consistent,
                "shape_fingerprint_consistent": shape_fingerprint_consistent,
                "transfer_model_mode": commit_transfer_model_mode or None,
                "transfer_model_mode_consistent": (
                    transfer_model_mode_consistent
                ),
                "predictive_prepare_authority_gate": commit_authority_gate,
                "authority_gate_consistent": authority_gate_consistent,
                "counterfactual_rejection_reasons": list(
                    counterfactual_rejection_reasons
                ),
                "paired_veto_evidence_present": paired_veto_evidence_present,
                "paired_veto_evidence_consistent": (
                    paired_veto_evidence_consistent
                ),
                "attribution_chain_complete": attribution_chain_complete,
                "transaction_completed": transaction_completed,
                "pressure_time_reclaim_bytes": None,
                "admission_wait_delta_ms": None,
                "workflow_jct_delta_ms": None,
            }
        )

    shape_action_gate = (
        bool(comparison.get("shape_action_gate"))
        if isinstance(comparison, Mapping)
        else None
    )
    shape_veto_gate = (
        bool(comparison.get("shape_veto_gate"))
        if isinstance(comparison, Mapping)
        else None
    )
    supported_shape_action_gate = (
        bool(
            comparison.get(
                "supported_shape_action_gate",
                comparison.get("shape_action_gate"),
            )
        )
        if isinstance(comparison, Mapping)
        else None
    )
    supported_shape_veto_gate = (
        bool(
            comparison.get(
                "supported_shape_veto_gate",
                comparison.get("shape_veto_gate"),
            )
        )
        if isinstance(comparison, Mapping)
        else None
    )
    selected_action_gate = (
        bool(comparison.get("selected_action_gate"))
        if isinstance(comparison, Mapping)
        else None
    )
    natural_canary_gate = (
        bool(comparison.get("shape_natural_action_available"))
        if isinstance(comparison, Mapping)
        else None
    )
    committed_ids = set(committed)
    orphan_queued_intents = sorted(set(queued) - committed_ids)
    orphan_terminal_intents = sorted(set(terminal) - committed_ids)
    predictive_command_ids = {
        str(item.get("command_id") or "") for item in queued.values()
    } - {""}
    orphan_predictive_commands = sorted(
        command_id
        for command_id in predictive_command_ids
        if command_id not in telemetry
        or command_id not in acknowledged
        or not any(
            str(item.get("command_id") or "") == command_id
            for item in terminal.values()
        )
    )
    committed_event_count = sum(commit_counts.values())
    canary_limit_respected = committed_event_count <= 1
    natural_action_available = committed_event_count == 1 and len(rows) == 1
    attribution_chain_complete = bool(
        natural_action_available
        and rows[0]["attribution_chain_complete"]
        and not orphan_queued_intents
        and not orphan_terminal_intents
        and not orphan_predictive_commands
    )
    transaction_completed = bool(
        natural_action_available and rows[0]["transaction_completed"]
    )
    veto_treatment_authorized = bool(
        supported_shape_veto_gate is True
        and natural_action_available
        and rows[0]["predictive_prepare_authority_gate"]
        == "byte-only-veto"
        and rows[0]["transfer_model_mode"] == "byte-only"
        and rows[0]["paired_veto_evidence_present"]
    )
    promotion_treatment_authorized = bool(
        supported_shape_action_gate is True
        and natural_action_available
        and rows[0]["transfer_model_mode"] == "morphology-aware"
    )
    action_authorized = bool(
        not rows
        or veto_treatment_authorized
        or promotion_treatment_authorized
    )
    paired_benefit_available = bool(
        natural_action_available
        and all(
            rows[0][field] is not None
            for field in (
                "pressure_time_reclaim_bytes",
                "admission_wait_delta_ms",
                "workflow_jct_delta_ms",
            )
        )
    )
    payload: dict[str, object] = {
        "status": (
            "unauthorized_predictive_prepare"
            if not action_authorized
            else "no_shape_supported_veto"
            if not rows
            and counterfactual_shape_unsupported > 0
            and shape_supported_veto_passed == 0
            else "no_natural_executable_veto"
            if not rows and shape_supported_veto_passed > 0
            else "no_natural_veto"
            if shape_action_gate is False and shape_veto_gate is True and not rows
            else "gate_not_met"
            if shape_action_gate is False and not rows
            else
            "completed"
            if canary_limit_respected
            and attribution_chain_complete
            and transaction_completed
            else "no_natural_prepare"
            if not rows
            else "incomplete"
        ),
        "canary_limit_respected": canary_limit_respected,
        "shape_prepare_authorized": supported_shape_action_gate is True,
        "veto_treatment_authorized": veto_treatment_authorized,
        "promotion_treatment_authorized": promotion_treatment_authorized,
        "action_authorized": action_authorized,
        "natural_action_available": natural_action_available,
        "attribution_chain_complete": attribution_chain_complete,
        "transaction_completed": transaction_completed,
        "paired_benefit_available": paired_benefit_available,
        "natural_prepare_count": len(rows),
        "natural_prepare_commit_event_count": committed_event_count,
        "paired_veto_evaluated_count": paired_veto_evaluated,
        "paired_veto_passed_count": paired_veto_passed,
        "shape_supported_veto_passed_count": shape_supported_veto_passed,
        "counterfactual_shape_unsupported_count": (
            counterfactual_shape_unsupported
        ),
        "unsupported_context_shape_count": len(unsupported_context_shapes),
        "unsupported_context_shapes": [
            {
                "context_id": key[0],
                "target_bytes": key[1],
                "extent_count": key[2],
                "shape_fingerprint": key[3],
                "occurrence_count": count,
            }
            for key, count in sorted(
                unsupported_context_shapes.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1] or -1),
            )
        ],
        "predictive_intent_published_count": intent_published,
        "predictive_intent_publish_rejected_count": intent_publish_rejected,
        "predictive_intent_safe_point_rejected_count": (
            intent_safe_point_rejected
        ),
        "publish_rejection_reasons": dict(
            sorted(publish_rejection_reasons.items())
        ),
        "safe_point_rejection_reasons": dict(
            sorted(safe_point_rejection_reasons.items())
        ),
        "canaries": rows,
        "orphan_queued_intents": orphan_queued_intents,
        "orphan_terminal_intents": orphan_terminal_intents,
        "orphan_predictive_commands": orphan_predictive_commands,
        "replay_decision_change_gate": (
            comparison.get("decision_change_gate")
            if isinstance(comparison, Mapping)
            else None
        ),
        "replay_decision_relevance_gate": (
            comparison.get("decision_relevance_gate")
            if isinstance(comparison, Mapping)
            else None
        ),
        "replay_shape_action_gate": shape_action_gate,
        "replay_shape_veto_gate": shape_veto_gate,
        "replay_supported_shape_action_gate": supported_shape_action_gate,
        "replay_supported_shape_veto_gate": supported_shape_veto_gate,
        "replay_selected_action_gate": selected_action_gate,
        "replay_natural_canary_gate": natural_canary_gate,
        # Compatibility output: only a supported promotion opens PREPARE.
        "replay_online_canary_gate": supported_shape_action_gate,
        "recommended_validation_arm": (
            comparison.get("recommended_validation_arm")
            if isinstance(comparison, Mapping)
            else None
        ),
        "limitations": [
            "pressure-time reclaim, admission wait, and workflow JCT require a paired online control run"
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the single natural morphology-aware PREPARE canary."
    )
    parser.add_argument("--runtime-audit", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-summary", type=Path, default=None)
    args = parser.parse_args()

    comparison = None
    if args.comparison_summary is not None and args.comparison_summary.exists():
        comparison = json.loads(args.comparison_summary.read_text(encoding="utf-8"))
    payload = analyze_canary(
        _records(args.runtime_audit) if args.runtime_audit else (),
        comparison=comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
