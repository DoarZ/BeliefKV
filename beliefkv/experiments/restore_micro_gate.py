from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return tuple(records)


def _matching(
    events: Iterable[Mapping[str, Any]],
    event: str,
    **fields: object,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        item
        for item in events
        if item.get("event") == event
        and all(item.get(key) == value for key, value in fields.items())
    )


def analyze_restore_micro_gate(
    events: Iterable[Mapping[str, Any]],
    runtime_summary: Mapping[str, Any],
    *,
    gate_id: str = "p5g-restore-v1",
) -> dict[str, Any]:
    records = tuple(events)
    states = _matching(records, "restore_micro_gate_state", gate_id=gate_id)
    final_state = states[-1] if states else {}
    transaction_id = final_state.get("transaction_id")
    obligation_id = final_state.get("obligation_id")
    victim_request_id = final_state.get("victim_request_id")
    planned = (
        _matching(
            records,
            "running_retraction_planned",
            transaction_id=transaction_id,
        )
        if transaction_id
        else ()
    )
    committed = (
        _matching(
            records,
            "running_retraction_committed",
            transaction_id=transaction_id,
        )
        if transaction_id
        else ()
    )
    reclaim_terminal = (
        _matching(
            records,
            "running_retraction_transaction_completed",
            transaction_id=transaction_id,
        )
        if transaction_id
        else ()
    )
    obligations = (
        _matching(
            records,
            "restore_obligation_created",
            source_retraction_transaction_id=transaction_id,
        )
        if transaction_id
        else ()
    )
    h2d = tuple(
        item
        for item in _matching(
            records,
            "restore_obligation_command_terminal",
            obligation_id=obligation_id,
        )
        if item.get("command_kind") == "prefetch_context"
        and item.get("status") == "completed"
        and int(item.get("actual_bytes", 0) or 0) > 0
    )
    obligation_terminal = tuple(
        item
        for item in _matching(
            records,
            "restore_obligation_terminal",
            obligation_id=obligation_id,
        )
        if item.get("state") == "satisfied"
        and int(item.get("restored_bytes", 0) or 0) > 0
    )
    service_grace = tuple(
        item
        for item in _matching(
            records,
            "restore_service_grace_terminal",
            obligation_id=obligation_id,
        )
        if item.get("status") == "satisfied"
        and int(item.get("served_decode_tokens", 0) or 0)
        >= int(item.get("required_decode_tokens", 1) or 1)
    )
    h2d_complete_ts = max(
        (float(item.get("ts_ms", 0.0) or 0.0) for item in h2d),
        default=float("inf"),
    )
    post_restore_service = tuple(
        sample
        for sample in _matching(records, "gpu_service_sample")
        if float(sample.get("complete_ts_ms", sample.get("ts_ms", 0.0)) or 0.0)
        >= h2d_complete_ts
        and any(
            item.get("request_id") == victim_request_id
            and item.get("phase") == "decode"
            and int(item.get("token_delta", 0) or 0) > 0
            for item in sample.get("request_samples", ())
            if isinstance(item, Mapping)
        )
    )
    explicit_d2h_bytes = max(
        (
            int(item.get("explicit_transfer_bytes", 0) or 0)
            for item in reclaim_terminal
        ),
        default=0,
    )
    shutdown_prepare = _matching(records, "shutdown_prepare")
    shutdown_prepare_ts = min(
        (
            float(item.get("ts_ms", 0.0) or 0.0)
            for item in shutdown_prepare
        ),
        default=float("inf"),
    )
    obligation_satisfied_before_shutdown = bool(
        shutdown_prepare
        and obligation_terminal
        and max(
            float(item.get("ts_ms", 0.0) or 0.0)
            for item in obligation_terminal
        )
        <= shutdown_prepare_ts
    )
    dispatched_command_ids = {
        str(item.get("command_id"))
        for item in _matching(records, "transfer_dispatched")
        if item.get("command_id")
    }
    terminal_command_ids = {
        str(item.get("command_id"))
        for event_name in ("transfer_acknowledged", "transfer_rejected_local")
        for item in _matching(records, event_name)
        if item.get("command_id")
    }
    physical = runtime_summary.get("physical_ownership_snapshot", {})
    correctness = runtime_summary.get("correctness_gates", {})
    checks = {
        "gate_reached_completed_state": final_state.get("stage") == "completed",
        "joint_plan_attributed": bool(
            planned and planned[-1].get("source_joint_plan_id")
        ),
        "selective_retraction_committed": bool(committed),
        "nonzero_d2h_completed": explicit_d2h_bytes > 0,
        "durable_obligation_created": bool(obligations and obligation_id),
        "nonzero_h2d_completed": bool(h2d),
        "obligation_satisfied_before_shutdown": (
            obligation_satisfied_before_shutdown
        ),
        "post_restore_decode_quantum_observed": bool(
            service_grace and post_restore_service
        ),
        "physical_snapshot_exercised": int(physical.get("call_count", 0) or 0)
        > 0,
        "shutdown_acknowledged": runtime_summary.get("shutdown_state")
        == "acknowledged",
        "no_pending_transactions": correctness.get("no_pending_transactions")
        is True,
        "command_ack_conservation": bool(len(dispatched_command_ids) >= 2)
        and dispatched_command_ids.issubset(terminal_command_ids),
        "shutdown_cleanup_did_not_mask_restore": correctness.get(
            "shutdown_cleanup_did_not_mask_unresolved_transactions"
        )
        is True,
        "all_online_actions_have_joint_plan": correctness.get(
            "all_online_actions_have_source_joint_plan_id"
        )
        is True,
    }
    return {
        "schema_version": 1,
        "gate_id": gate_id,
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "victim_request_id": victim_request_id,
            "transaction_id": transaction_id,
            "obligation_id": obligation_id,
            "source_joint_plan_id": (
                planned[-1].get("source_joint_plan_id") if planned else None
            ),
            "explicit_d2h_bytes": explicit_d2h_bytes,
            "restored_h2d_bytes": max(
                (
                    int(item.get("restored_bytes", 0) or 0)
                    for item in obligation_terminal
                ),
                default=0,
            ),
            "post_restore_service_sample_count": len(post_restore_service),
            "physical_snapshot_call_count": int(
                physical.get("call_count", 0) or 0
            ),
        },
    }
