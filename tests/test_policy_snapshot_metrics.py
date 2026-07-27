from __future__ import annotations

import json
from pathlib import Path

import pytest

from beliefkv.metrics.policy_snapshot import analyze_policy_snapshot_audit


def test_policy_snapshot_metrics_separate_writer_semantics_and_scale(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    records = [
        {
            "event": "runtime_initialized",
            "reference_policy_snapshot": {
                "writer_mode": "background_single_consumer"
            },
        },
        {
            "event": "policy_snapshot_recorded",
            "build_ms": 0.5,
            "physical_bundle_count": 16,
            "trigger": "graph_or_queue",
        },
        {
            "event": "policy_snapshot_recorded",
            "build_ms": 1.5,
            "physical_bundle_count": 80,
            "trigger": "physical_or_pressure",
        },
        {"event": "controller_timing_summary", "scheduler_step_p99_ms": 10},
        {
            "event": "policy_snapshot_summary",
            "snapshot_count": 2,
            "written_snapshot_count": 2,
        },
    ]
    audit.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    snapshots = tmp_path / "snapshots.gz"
    snapshots.write_bytes(b"123456")

    result = analyze_policy_snapshot_audit(audit, snapshot_path=snapshots)

    assert result["writer_mode"] == "background_single_consumer"
    assert result["recorded_snapshot_count"] == 2
    assert result["written_snapshot_count"] == 2
    assert result["safe_point_snapshot_ms"]["p50"] == 1.0
    assert result["snapshot_to_scheduler_p99_ratio"] == pytest.approx(0.149)
    assert result["compressed_bytes_per_snapshot"] == 3
    assert set(result["safe_point_snapshot_ms_by_extent_scale"]) == {
        "000-032",
        "065-128",
    }
