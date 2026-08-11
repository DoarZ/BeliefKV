from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


_TERMINAL_OUTCOMES = frozenset(
    {"useful", "wasted", "too_late", "censored", "failed"}
)


def analyze_predictive_prepare_canary(
    records: Iterable[Mapping[str, object]],
    *,
    canary_limit: int = 1,
) -> dict[str, object]:
    """Validate one natural PREPARE_HOST action from belief to attribution.

    The analyzer deliberately has no morphology promotion/veto gate. A run with
    no naturally selected action is a valid negative result; a run with an
    action is valid only when every transaction identifier and byte count is
    conserved through the native transfer path.
    """

    if canary_limit <= 0:
        raise ValueError("canary_limit must be positive")

    risk_evaluations = 0
    positive_risk_intents: set[str] = set()
    published: dict[str, Mapping[str, object]] = {}
    committed: dict[str, Mapping[str, object]] = {}
    queued: dict[str, Mapping[str, object]] = {}
    telemetry: dict[str, list[Mapping[str, object]]] = {}
    acknowledged: dict[str, list[Mapping[str, object]]] = {}
    terminal: dict[str, Mapping[str, object]] = {}
    outcomes: dict[str, list[Mapping[str, object]]] = {}
    rejected: Counter[str] = Counter()

    for record in records:
        event = str(record.get("event") or "")
        if event == "predictive_risk_shadow":
            risk_evaluations += 1
            intent = record.get("predictive_intent")
            if isinstance(intent, Mapping) and intent.get("intent_id"):
                positive_risk_intents.add(str(intent["intent_id"]))
        elif event == "predictive_semantic_intent_published":
            published[str(record.get("intent_id") or "")] = record
        elif event in {
            "predictive_semantic_intent_publish_rejected",
            "predictive_semantic_intent_rejected",
        }:
            reasons = record.get("reasons", ())
            if isinstance(reasons, (list, tuple, set, frozenset)):
                rejected.update(str(item) for item in reasons)
        elif (
            event == "predictive_semantic_intent_committed"
            and record.get("action") == "prepare_host"
        ):
            committed[str(record.get("intent_id") or "")] = record
        elif event == "online_joint_residency_queued" and record.get(
            "predictive_intent_id"
        ):
            queued[str(record["predictive_intent_id"])] = record
        elif event == "transfer_telemetry" and record.get("command_id"):
            telemetry.setdefault(str(record["command_id"]), []).append(record)
        elif event == "transfer_acknowledged" and record.get("command_id"):
            acknowledged.setdefault(str(record["command_id"]), []).append(record)
        elif event == "online_joint_residency_terminal" and record.get(
            "predictive_intent_id"
        ):
            terminal[str(record["predictive_intent_id"])] = record
        elif event == "predictive_action_outcome" and record.get("intent_id"):
            outcomes.setdefault(str(record["intent_id"]), []).append(record)

    rows: list[dict[str, object]] = []
    for intent_id, commit in sorted(committed.items()):
        publish = published.get(intent_id)
        queue = queued.get(intent_id)
        command_id = str(queue.get("command_id") or "") if queue else ""
        command_telemetry = telemetry.get(command_id, ())
        completed_telemetry = tuple(
            item for item in command_telemetry if item.get("status") == "completed"
        )
        command_acks = acknowledged.get(command_id, ())
        completed_acks = tuple(
            item for item in command_acks if item.get("status") == "completed"
        )
        finish = terminal.get(intent_id)
        action_outcomes = outcomes.get(intent_id, ())
        final_outcomes = tuple(
            item
            for item in action_outcomes
            if str(item.get("state") or "") in _TERMINAL_OUTCOMES
        )
        transfer = completed_telemetry[-1] if completed_telemetry else None
        ack = completed_acks[-1] if completed_acks else None
        outcome = final_outcomes[-1] if final_outcomes else None

        transaction_id = str(queue.get("transaction_id") or "") if queue else ""
        stage_presence = {
            "belief": intent_id in positive_risk_intents,
            "publish": publish is not None,
            "joint_commit": True,
            "queue": queue is not None,
            "transfer": transfer is not None,
            "ack": ack is not None,
            "terminal": finish is not None,
            "outcome": outcome is not None,
        }
        ids_consistent = bool(
            intent_id
            and queue is not None
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
            and outcome is not None
            and str(outcome.get("command_id") or "") == command_id
        )
        byte_values = tuple(
            int(item.get("actual_bytes") or 0)
            for item in (transfer, ack, finish, outcome)
            if item is not None
        )
        bytes_consistent = bool(
            len(byte_values) == 4
            and min(byte_values) > 0
            and len(set(byte_values)) == 1
            and queue is not None
            and int(queue.get("copy_bytes") or 0) == byte_values[0]
        )
        extent_consistent = bool(
            transfer is not None
            and commit.get("live_extent_count") is not None
            and queue is not None
            and queue.get("live_extent_count") is not None
            and transfer.get("extent_count") is not None
            and int(commit["live_extent_count"])
            == int(queue["live_extent_count"])
            == int(transfer["extent_count"])
        )
        chain_complete = bool(
            all(stage_presence.values())
            and ids_consistent
            and bytes_consistent
            and extent_consistent
            and len(completed_telemetry) == 1
            and len(completed_acks) == 1
            and len(final_outcomes) == 1
        )
        transaction_completed = bool(
            chain_complete
            and finish is not None
            and finish.get("status") == "completed"
        )
        rows.append(
            {
                "intent_id": intent_id,
                "context_id": commit.get("context_id"),
                "command_id": command_id or None,
                "transaction_id": transaction_id or None,
                "stage_presence": stage_presence,
                "ids_consistent": ids_consistent,
                "bytes_consistent": bytes_consistent,
                "extent_count_consistent": extent_consistent,
                "attribution_chain_complete": chain_complete,
                "transaction_completed": transaction_completed,
                "actual_bytes": byte_values[0] if bytes_consistent else None,
                "outcome": outcome.get("state") if outcome else None,
                "outcome_reason": outcome.get("reason") if outcome else None,
            }
        )

    committed_ids = set(committed)
    orphan_intents = sorted((set(queued) | set(terminal) | set(outcomes)) - committed_ids)
    canary_limit_respected = len(committed) <= canary_limit
    natural_action_available = len(committed) == 1
    chain_complete = bool(
        natural_action_available
        and rows[0]["attribution_chain_complete"]
        and not orphan_intents
    )
    transaction_completed = bool(
        natural_action_available and rows[0]["transaction_completed"]
    )
    if not canary_limit_respected:
        status = "canary_limit_exceeded"
    elif not committed:
        status = "no_positive_action"
    elif chain_complete and transaction_completed:
        status = "completed"
    else:
        status = "incomplete"

    return {
        "schema_version": 1,
        "status": status,
        "risk_evaluation_count": risk_evaluations,
        "positive_risk_intent_count": len(positive_risk_intents),
        "published_intent_count": len(published),
        "natural_prepare_count": len(committed),
        "canary_limit": canary_limit,
        "canary_limit_respected": canary_limit_respected,
        "natural_action_available": natural_action_available,
        "attribution_chain_complete": chain_complete,
        "transaction_completed": transaction_completed,
        "outcome_counts": dict(
            sorted(
                Counter(
                    str(row["outcome"])
                    for row in rows
                    if row["outcome"] is not None
                ).items()
            )
        ),
        "rejection_reasons": dict(sorted(rejected.items())),
        "orphan_intents": orphan_intents,
        "actions": rows,
    }
