from pathlib import Path

from beliefkv.experiments.predictive_joint_ab import (
    PredictiveJointABRun,
    build_ab_run_plan,
    compare_predictive_joint_ab,
)
from scripts.run_p6_predictive_joint_ab import resolve_agent_python


def test_resolve_agent_python_prefers_explicit_path(tmp_path: Path) -> None:
    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    assert resolve_agent_python(executable) == executable.resolve()


def test_resolve_agent_python_uses_environment(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "agent-python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("BELIEFKV_AGENT_PYTHON", str(executable))

    assert resolve_agent_python() == executable.resolve()


def test_build_ab_run_plan_freezes_three_alternating_pairs() -> None:
    payload = build_ab_run_plan(
        {
            "frozen": True,
            "manifest_sha256": "abc",
            "selection_contract": {"paired_order": ["A-B", "B-A", "A-B"]},
            "workload": {"fanout_profile": "parallel_analysis_2to3"},
            "runtime": {},
            "artifacts": {},
            "source_tree": {"sha256": "source"},
        }
    )

    assert [item["arm"] for item in payload["runs"]] == ["A", "B", "B", "A", "A", "B"]
    assert payload["arm_contracts"]["B"]["predictive_prepare_limit"] == 1
    assert payload["arm_contracts"]["B"]["frontier_retraction_canary_limit"] == 1
    assert payload["source_tree"] == {"sha256": "source"}


def _run(pair: str, arm: str, *, duration: float, task_success: int = 1):
    return PredictiveJointABRun(
        pair_id=pair,
        arm=arm,
        summary={
            "duration_seconds": duration,
            "workflow_count": 2,
            "system_jct_eligible_workflows": 2,
            "successful_workflows": task_success,
            "runtime_control_delivery": {"failure_count": 0},
            "server": {
                "runtime_events": {
                    "event_counts": {"workflow_end": 2, "tool_start": 4, "join_satisfied": 2}
                },
                "runtime_audit": {"admission_queue_wait_ms": {"p50": 1, "p95": 2}},
                "server_log": {"peak_running_requests": 4, "prefill_batch_count": 8},
            },
        },
    )


def test_compare_predictive_joint_ab_uses_clean_workflow_throughput() -> None:
    runs = [
        _run("pair-1", "A", duration=100),
        _run("pair-1", "B", duration=80),
        _run("pair-2", "A", duration=100),
        _run("pair-2", "B", duration=90),
        _run("pair-3", "A", duration=100),
        _run("pair-3", "B", duration=110),
    ]

    result = compare_predictive_joint_ab(runs)

    assert result["pairs_with_throughput_improvement"] == 2
    assert result["development_continue_gate"] is True
