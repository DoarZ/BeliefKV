import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def test_analyzer_uses_overlap_weighted_matched_decode_samples() -> None:
    transfer = {
        "event": "transfer_telemetry",
        "direction": "d2h",
        "status": "completed",
        "actual_bytes": 1_000,
        "page_count": 2,
        "command_id": "d2h-1",
        "command_kind": "offload_context",
        "submit_ts_ms": 99.0,
        "start_ts_ms": 100.0,
        "complete_ts_ms": 200.0,
        "start_timestamp_semantics": "hicache_api_submit_begin",
    }

    def sample(index: int, start: float, elapsed: float) -> dict[str, object]:
        return {
            "event": "gpu_service_sample",
            "phase": "decode",
            "batch_size": 1,
            "workflow_ids": ["anchor"],
            "service_start_ts_ms": start,
            "complete_ts_ms": start + elapsed,
            "service_elapsed_ms": elapsed,
            "request_samples": [
                {
                    "workflow_id": "anchor",
                    "phase": "decode",
                    "token_delta": 1,
                    "sequence_tokens_before": 1_000 + index,
                    "output_tokens_before": 0,
                }
            ],
        }

    audit = [sample(index, 110.0 + index * 20.0, 20.0) for index in range(4)]
    audit.extend(sample(index + 4, 210.0 + index * 10.0, 10.0) for index in range(12))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        audit_path = root / "audit.jsonl"
        transfer_path = root / "transfer.jsonl"
        output_path = root / "result.json"
        _write(audit_path, audit)
        _write(transfer_path, [transfer])

        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_d2h_overlap.py",
                "--runtime-audit",
                str(audit_path),
                "--transfer-telemetry",
                str(transfer_path),
                "--anchor-workflow-id",
                "anchor",
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["primary_sample_count"] == 4
    assert result["covered_transfer_ratio"] == 0.8
    assert result["primary_uses_boundary_samples"] is False
    assert result["baseline"]["sample_count"] == 12
    assert result["baseline"]["per_step_p50_ms"] == 10.0
    assert result["unhidden_interference"]["stall_ms_p50_reference"] == 40.0
    assert result["unhidden_interference"]["stall_ratio_p50_reference"] == 0.4
    assert result["performance_evidence_eligible"] is False


def test_analyzer_weights_transfer_boundary_intervals() -> None:
    transfer = {
        "event": "transfer_telemetry",
        "direction": "d2h",
        "status": "completed",
        "actual_bytes": 1_000,
        "page_count": 2,
        "command_id": "d2h-1",
        "command_kind": "offload_context",
        "submit_ts_ms": 99.0,
        "start_ts_ms": 100.0,
        "complete_ts_ms": 200.0,
    }

    def sample(start: float, elapsed: float) -> dict[str, object]:
        return {
            "event": "gpu_service_sample",
            "phase": "decode",
            "batch_size": 1,
            "workflow_ids": ["anchor"],
            "service_start_ts_ms": start,
            "complete_ts_ms": start + elapsed,
            "service_elapsed_ms": elapsed,
            "request_samples": [
                {
                    "workflow_id": "anchor",
                    "phase": "decode",
                    "token_delta": 1,
                    "sequence_tokens_before": 1_000,
                    "output_tokens_before": 10,
                }
            ],
        }

    audit = [sample(90.0, 20.0), sample(110.0, 80.0), sample(190.0, 20.0)]
    audit.extend(sample(210.0 + index * 10.0, 10.0) for index in range(8))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        audit_path = root / "audit.jsonl"
        transfer_path = root / "transfer.jsonl"
        output_path = root / "result.json"
        _write(audit_path, audit)
        _write(transfer_path, [transfer])
        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_d2h_overlap.py",
                "--runtime-audit",
                str(audit_path),
                "--transfer-telemetry",
                str(transfer_path),
                "--anchor-workflow-id",
                "anchor",
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["primary_sample_count"] == 3
    assert result["fully_overlapping_sample_count"] == 1
    assert result["primary_uses_boundary_samples"] is True
    assert result["covered_transfer_ratio"] == 1.0
    assert result["primary_elapsed_ms"] == 100.0
    assert result["unhidden_interference"]["stall_ratio_p50_reference"] == 0.8


def test_analyzer_matches_legacy_effective_sequence_with_output_progress() -> None:
    transfer = {
        "event": "transfer_telemetry",
        "direction": "d2h",
        "status": "completed",
        "actual_bytes": 1_000,
        "page_count": 2,
        "command_id": "d2h-1",
        "command_kind": "offload_context",
        "submit_ts_ms": 99.0,
        "start_ts_ms": 100.0,
        "complete_ts_ms": 140.0,
    }

    def sample(output_tokens: int, start: float, elapsed: float) -> dict[str, object]:
        return {
            "event": "gpu_service_sample",
            "phase": "decode",
            "batch_size": 1,
            "workflow_ids": ["anchor"],
            "service_start_ts_ms": start,
            "complete_ts_ms": start + elapsed,
            "service_elapsed_ms": elapsed,
            "request_samples": [
                {
                    "workflow_id": "anchor",
                    "phase": "decode",
                    "token_delta": 1,
                    "sequence_tokens_before": 1_000,
                    "output_tokens_before": output_tokens,
                }
            ],
        }

    audit = [sample(10, 100.0, 20.0), sample(11, 120.0, 20.0)]
    audit.extend(
        sample(output_tokens, 150.0 + index * 10.0, 10.0)
        for index, output_tokens in enumerate((14, 15, 16, 17, 200))
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        audit_path = root / "audit.jsonl"
        transfer_path = root / "transfer.jsonl"
        output_path = root / "result.json"
        _write(audit_path, audit)
        _write(transfer_path, [transfer])
        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_d2h_overlap.py",
                "--runtime-audit",
                str(audit_path),
                "--transfer-telemetry",
                str(transfer_path),
                "--anchor-workflow-id",
                "anchor",
                "--sequence-radius-tokens",
                "8",
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["sequence_center_tokens"] == 1_010
    assert result["baseline"]["sample_count"] == 4
    assert result["sequence_matching"]["baseline_output_tokens_max"] == 17


def test_analyzer_accepts_independent_no_d2h_control() -> None:
    transfer = {
        "event": "transfer_telemetry",
        "direction": "d2h",
        "status": "completed",
        "actual_bytes": 1_000,
        "page_count": 2,
        "command_id": "d2h-1",
        "command_kind": "offload_context",
        "submit_ts_ms": 99.0,
        "start_ts_ms": 100.0,
        "complete_ts_ms": 140.0,
    }

    def sample(output_tokens: int, start: float, elapsed: float) -> dict[str, object]:
        return {
            "event": "gpu_service_sample",
            "phase": "decode",
            "batch_size": 1,
            "workflow_ids": ["anchor"],
            "service_start_ts_ms": start,
            "complete_ts_ms": start + elapsed,
            "service_elapsed_ms": elapsed,
            "request_samples": [
                {
                    "workflow_id": "anchor",
                    "phase": "decode",
                    "token_delta": 1,
                    "sequence_tokens_before": 1_000,
                    "output_tokens_before": output_tokens,
                    "effective_sequence_tokens_before": 1_000 + output_tokens,
                }
            ],
        }

    treatment = [sample(10, 100.0, 20.0), sample(11, 120.0, 20.0)]
    control = [
        sample(value, 200.0 + index * 10.0, 10.0)
        for index, value in enumerate((8, 9, 10, 11, 12, 13))
    ]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        treatment_path = root / "treatment.jsonl"
        control_path = root / "control.jsonl"
        control_transfer_path = root / "control-transfer.jsonl"
        transfer_path = root / "transfer.jsonl"
        output_path = root / "result.json"
        _write(treatment_path, treatment)
        _write(control_path, control)
        _write(control_transfer_path, [])
        _write(transfer_path, [transfer])
        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_d2h_overlap.py",
                "--runtime-audit",
                str(treatment_path),
                "--control-runtime-audit",
                str(control_path),
                "--control-transfer-telemetry",
                str(control_transfer_path),
                "--transfer-telemetry",
                str(transfer_path),
                "--anchor-workflow-id",
                "anchor",
                "--sequence-radius-tokens",
                "4",
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["counterfactual_control_available"] is True
    assert result["control_transfer_overlap_count"] == 0
    assert result["performance_evidence_eligible"] is False
    assert result["baseline"]["source"] == "independent_no_d2h_control"
    assert result["sequence_center_tokens"] == 1_010


def test_analyzer_rejects_non_target_and_failed_control_transfers() -> None:
    target = {
        "event": "transfer_telemetry",
        "direction": "d2h",
        "status": "completed",
        "actual_bytes": 1_000,
        "page_count": 2,
        "extent_count": 2,
        "command_id": "target",
        "command_kind": "offload_context",
        "submit_ts_ms": 99.0,
        "start_ts_ms": 100.0,
        "complete_ts_ms": 140.0,
    }
    concurrent = {
        "event": "transfer_telemetry",
        "direction": "h2d",
        "status": "inflight",
        "actual_bytes": 0,
        "command_id": "other-h2d",
        "command_kind": "prefetch_context",
        "submit_ts_ms": 110.0,
    }
    failed_control = {
        "event": "transfer_telemetry",
        "direction": "d2h",
        "status": "failed",
        "actual_bytes": 0,
        "command_id": "failed-control",
        "command_kind": "shadow_context",
        "submit_ts_ms": 101.0,
        "complete_ts_ms": 105.0,
    }

    def sample(output: int, start: float) -> dict[str, object]:
        return {
            "event": "gpu_service_sample",
            "phase": "decode",
            "batch_size": 1,
            "workflow_ids": ["anchor"],
            "service_start_ts_ms": start,
            "complete_ts_ms": start + 10.0,
            "service_elapsed_ms": 10.0,
            "request_samples": [
                {
                    "workflow_id": "anchor",
                    "phase": "decode",
                    "token_delta": 1,
                    "sequence_tokens_before": 1_000,
                    "output_tokens_before": output,
                }
            ],
        }

    treatment = [sample(index, 100.0 + index * 10.0) for index in range(4)]
    control = [sample(index, 100.0 + index * 10.0) for index in range(6)]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        treatment_path = root / "treatment.jsonl"
        control_path = root / "control.jsonl"
        transfer_path = root / "transfer.jsonl"
        control_transfer_path = root / "control-transfer.jsonl"
        output_path = root / "result.json"
        _write(treatment_path, treatment)
        _write(control_path, control)
        _write(transfer_path, [target, concurrent])
        _write(control_transfer_path, [failed_control])
        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_d2h_overlap.py",
                "--runtime-audit",
                str(treatment_path),
                "--control-runtime-audit",
                str(control_path),
                "--control-transfer-telemetry",
                str(control_transfer_path),
                "--transfer-telemetry",
                str(transfer_path),
                "--anchor-workflow-id",
                "anchor",
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["treatment_concurrent_transfer_count"] == 1
    assert result["treatment_concurrent_transfers"][0]["status"] == "inflight"
    assert result["treatment_contaminated_sample_count"] == 3
    assert result["treatment_contamination_excluded"] is True
    assert result["control_transfer_overlap_count"] == 1
    assert result["control_transfer_pollution"][0]["status"] == "failed"
    assert "control_baseline_overlaps_transfer" in result[
        "performance_evidence_ineligible_reasons"
    ]
