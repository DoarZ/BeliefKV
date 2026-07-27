from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from beliefkv.experiments.policy_replay import (
    PolicyReplayRunner,
    ReplaySnapshot,
    load_replay_trace,
    write_replay_trace,
)
from beliefkv.policy.reference import (
    CapabilityReport,
    PhysicalKVSnapshot,
    PolicyDecisionRecord,
    PolicyInput,
    ResidencyAction,
    ResourceSnapshot,
    RuntimeGraphSnapshot,
)


def _policy_input(sequence: int) -> PolicyInput:
    snapshot_id = f"snapshot-{sequence}"
    return PolicyInput(
        runtime_graph=RuntimeGraphSnapshot(
            snapshot_id=snapshot_id,
            graph_version=sequence,
            observed_ts_ms=float(sequence),
            state={},
        ),
        runnable_frontier=(),
        physical_kv=PhysicalKVSnapshot(
            snapshot_id=snapshot_id,
            topology_version=sequence,
            allocator_version=sequence,
            gpu_bytes=0,
            cpu_bytes=0,
            bundles=(),
        ),
        resources=ResourceSnapshot(
            snapshot_id=snapshot_id,
            ts_ms=float(sequence),
            hbm_capacity_bytes=1_000,
            hbm_used_bytes=0,
            hbm_reserved_bytes=0,
            host_free_bytes=2_000,
            urgent_d2h_bytes=0,
            urgent_h2d_bytes=0,
            pcie_utilization=0,
            gpu_compute_utilization=0,
            recent_kv_growth_bytes_per_ms=0,
            h2d_service_bytes_per_ms=100,
            d2h_service_bytes_per_ms=100,
            transfer_setup_p50_ms=0.1,
            unhidden_stall_per_byte=0,
        ),
        capabilities=CapabilityReport(
            runtime_name="replay-test",
            runtime_version="1",
            supported_residency_actions=frozenset(ResidencyAction),
            execution_order_control=True,
            admission_control=True,
            transfer_dependencies=True,
            native_identity_mapping=True,
        ),
    )


def _snapshots() -> tuple[ReplaySnapshot, ...]:
    return (
        ReplaySnapshot(1, "trace-a", "timing_sensitive", _policy_input(1)),
        ReplaySnapshot(2, "trace-a", "timing_sensitive", _policy_input(2)),
    )


def test_policy_replay_writes_atomic_auditable_b0_decisions(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "snapshots.jsonl"
    write_replay_trace(trace, _snapshots())
    output = tmp_path / "replay"

    result = PolicyReplayRunner().run(trace, output, run_id="run-1")

    assert result.output_dir == output
    assert not (tmp_path / ".replay.incomplete").exists()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["policies"]) == {"B0"}
    assert summary["policies"]["B0"]["decision_count"] == 2
    assert summary["counterfactual_validity"]["timing_sensitive"] == (
        "decision_only_timing_must_be_resimulated"
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policies"] == [
        {
            "baseline_id": "B0",
            "evaluation_mode": "replay",
            "fidelity": "beliefkv_internal_baseline",
            "metadata_mode": "online",
            "policy_name": "reactive_reference",
        }
    ]
    records = [
        json.loads(line)
        for line in (output / "B0_reactive_reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 2
    assert PolicyDecisionRecord.from_dict(records[0]["decision"])

    with pytest.raises(FileExistsError):
        PolicyReplayRunner().run(trace, output, run_id="run-1")


def test_trace_round_trip_preserves_policy_inputs(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    expected = _snapshots()
    write_replay_trace(path, expected)
    assert load_replay_trace(path) == expected


def test_gzip_trace_round_trip_preserves_policy_inputs(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    expected = _snapshots()
    write_replay_trace(path, expected)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert load_replay_trace(path) == expected


def test_invalid_sequence_fails_before_output_is_created(tmp_path: Path) -> None:
    trace = tmp_path / "bad.jsonl"
    first, second = _snapshots()
    write_replay_trace(trace, (replace(first, sequence=2), second))
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="strictly increasing"):
        PolicyReplayRunner().run(trace, output, run_id="bad")
    assert not output.exists()


def test_mixed_trace_ids_are_rejected(tmp_path: Path) -> None:
    trace = tmp_path / "mixed.jsonl"
    first, second = _snapshots()
    write_replay_trace(trace, (first, replace(second, trace_id="trace-b")))

    with pytest.raises(ValueError, match="mix trace IDs"):
        PolicyReplayRunner().run(trace, tmp_path / "output", run_id="bad")
