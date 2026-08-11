from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


def _request_ids(record: Mapping[str, object]) -> set[str]:
    values = record.get("request_ids", ())
    return {str(item) for item in values} if isinstance(values, (list, tuple)) else set()


def analyze_frontier_retraction_canary(
    records: Iterable[Mapping[str, object]],
    *,
    canary_limit: int = 1,
) -> dict[str, object]:
    if canary_limit <= 0:
        raise ValueError("canary_limit must be positive")

    rows = list(records)
    shadow = [item for item in rows if item.get("event") == "frontier_retraction_shadow"]
    changed = [item for item in shadow if item.get("decision_changed") is True]
    selected = [
        item
        for item in rows
        if item.get("event") == "frontier_retraction_canary_committed"
    ]
    bound = [
        item for item in rows if item.get("event") == "frontier_retraction_canary_bound"
    ]
    planned = {
        str(item.get("transaction_id")): item
        for item in rows
        if item.get("event") == "running_retraction_planned"
        and item.get("transaction_id")
    }
    completed = {
        str(item.get("transaction_id")): item
        for item in rows
        if item.get("event") == "running_retraction_transaction_completed"
        and item.get("transaction_id")
    }
    failed = {
        str(item.get("transaction_id")): item
        for item in rows
        if item.get("event") in {
            "running_retraction_transaction_failed",
            "running_retraction_transaction_terminal",
        }
        and item.get("transaction_id")
        and item.get("status") in {"failed", "aborted"}
    }
    obligations = [
        item for item in rows if item.get("event") == "restore_obligation_terminal"
    ]
    service_samples = [
        item for item in rows if item.get("event") == "gpu_service_sample"
    ]

    action_rows: list[dict[str, object]] = []
    for binding in bound:
        transaction_id = str(binding.get("transaction_id") or "")
        victim_ids = tuple(str(item) for item in binding.get("request_ids", ()))
        replacement_ids = tuple(
            str(item) for item in binding.get("replacement_request_ids", ())
        )
        transaction = completed.get(transaction_id)
        transaction_ts = (
            float(transaction.get("ts_ms") or 0.0) if transaction is not None else 0.0
        )
        victim_obligations = {
            request_id: tuple(
                item
                for item in obligations
                if str(item.get("request_id") or "") == request_id
                and str(item.get("state") or "") == "satisfied"
            )
            for request_id in victim_ids
        }
        victim_service = {
            request_id: any(
                float(sample.get("ts_ms") or 0.0)
                > max(float(item.get("ts_ms") or 0.0) for item in terminal)
                and request_id in _request_ids(sample)
                for sample in service_samples
            )
            if terminal
            else False
            for request_id, terminal in victim_obligations.items()
        }
        replacement_service = {
            request_id: any(
                float(sample.get("ts_ms") or 0.0) > transaction_ts
                and request_id in _request_ids(sample)
                for sample in service_samples
            )
            for request_id in replacement_ids
        }
        chain_complete = bool(
            transaction_id
            and transaction_id in planned
            and transaction is not None
            and transaction_id not in failed
            and victim_ids
            and replacement_ids
            and all(victim_obligations.values())
            and all(victim_service.values())
            and all(replacement_service.values())
        )
        action_rows.append(
            {
                "transaction_id": transaction_id or None,
                "victim_request_ids": list(victim_ids),
                "replacement_request_ids": list(replacement_ids),
                "transaction_completed": transaction is not None,
                "restore_satisfied": {
                    key: bool(value) for key, value in victim_obligations.items()
                },
                "victim_service_after_restore": victim_service,
                "replacement_service_after_transaction": replacement_service,
                "attribution_chain_complete": chain_complete,
            }
        )

    victim_plan_counts: Counter[str] = Counter()
    for item in planned.values():
        victim_plan_counts.update(str(value) for value in item.get("request_ids", ()))
    duplicate_victim_ids = sorted(
        request_id for request_id, count in victim_plan_counts.items() if count > 1
    )
    canary_limit_respected = len(bound) <= canary_limit and len(selected) <= canary_limit
    natural_action_available = len(bound) == 1
    completed_gate = bool(
        natural_action_available
        and canary_limit_respected
        and action_rows[0]["attribution_chain_complete"]
        and not duplicate_victim_ids
    )
    status = (
        "canary_limit_exceeded"
        if not canary_limit_respected
        else "no_frontier_decision_change"
        if not changed
        else "shadow_only"
        if not bound
        else "completed"
        if completed_gate
        else "incomplete"
    )
    return {
        "schema_version": 1,
        "status": status,
        "shadow_evaluation_count": len(shadow),
        "decision_change_count": len(changed),
        "canary_selected_count": len(selected),
        "canary_bound_count": len(bound),
        "canary_limit": canary_limit,
        "canary_limit_respected": canary_limit_respected,
        "natural_action_available": natural_action_available,
        "correctness_gate_passed": completed_gate,
        "duplicate_victim_ids": duplicate_victim_ids,
        "actions": action_rows,
    }
