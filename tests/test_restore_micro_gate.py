from beliefkv.experiments.restore_micro_gate import analyze_restore_micro_gate


def _events():
    return (
        {
            "event": "restore_micro_gate_state",
            "gate_id": "p5g-restore-v1",
            "stage": "completed",
            "victim_request_id": "victim",
            "transaction_id": "retraction-1",
            "obligation_id": "obligation-1",
        },
        {
            "event": "running_retraction_planned",
            "transaction_id": "retraction-1",
            "source_joint_plan_id": "joint-1",
        },
        {
            "event": "running_retraction_committed",
            "transaction_id": "retraction-1",
        },
        {
            "event": "running_retraction_transaction_completed",
            "transaction_id": "retraction-1",
            "explicit_transfer_bytes": 1024,
        },
        {
            "event": "restore_obligation_created",
            "source_retraction_transaction_id": "retraction-1",
            "obligation_id": "obligation-1",
        },
        {
            "event": "restore_obligation_command_terminal",
            "obligation_id": "obligation-1",
            "command_kind": "prefetch_context",
            "status": "completed",
            "actual_bytes": 1024,
            "ts_ms": 20.0,
        },
        {
            "event": "restore_obligation_terminal",
            "obligation_id": "obligation-1",
            "state": "satisfied",
            "restored_bytes": 1024,
            "ts_ms": 21.0,
        },
        {
            "event": "restore_service_grace_terminal",
            "obligation_id": "obligation-1",
            "status": "satisfied",
            "served_decode_tokens": 32,
            "required_decode_tokens": 32,
        },
        {
            "event": "gpu_service_sample",
            "complete_ts_ms": 22.0,
            "request_samples": (
                {
                    "request_id": "victim",
                    "phase": "decode",
                    "token_delta": 1,
                },
            ),
        },
        {
            "event": "transfer_dispatched",
            "command_id": "command-1",
        },
        {
            "event": "transfer_acknowledged",
            "command_id": "command-1",
        },
        {
            "event": "transfer_dispatched",
            "command_id": "command-2",
        },
        {
            "event": "transfer_acknowledged",
            "command_id": "command-2",
        },
        {"event": "shutdown_prepare", "ts_ms": 30.0},
    )


def _summary():
    return {
        "shutdown_state": "acknowledged",
        "physical_ownership_snapshot": {"call_count": 2},
        "correctness_gates": {
            "no_pending_transactions": True,
            "shutdown_cleanup_did_not_mask_unresolved_transactions": True,
            "all_online_actions_have_source_joint_plan_id": True,
        },
    }


def test_restore_micro_gate_accepts_complete_physical_cycle():
    result = analyze_restore_micro_gate(_events(), _summary())

    assert result["passed"] is True
    assert result["evidence"]["explicit_d2h_bytes"] == 1024
    assert result["evidence"]["restored_h2d_bytes"] == 1024


def test_restore_micro_gate_rejects_logical_restore_without_h2d():
    events = tuple(
        item
        for item in _events()
        if item["event"] != "restore_obligation_command_terminal"
    )

    result = analyze_restore_micro_gate(events, _summary())

    assert result["passed"] is False
    assert result["checks"]["nonzero_h2d_completed"] is False
    assert result["checks"]["post_restore_decode_quantum_observed"] is False
