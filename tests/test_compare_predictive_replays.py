import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def _record(
    *,
    transfer_ms: float,
    eligible: bool,
    reasons: list[str],
    selected_action: str = "observed_baseline",
    shape_supported: bool = True,
) -> dict[str, object]:
    return {
        "source_snapshot_id": "snapshot-1",
        "selected_action": selected_action,
        "candidate_summaries": [
            {
                "action": "prepare_host",
                "package_id": "joint-1:prepare:context-1",
                "eligible": eligible,
                "reasons": reasons,
                "expected_benefit_ms": 0.0,
                "morphology_shape_supported": shape_supported,
                "morphology_shape_fingerprint": "shape-1",
                "morphology_shape_aware_transfer_p90_ms": transfer_ms,
                "prepare_recourse_scenarios": [
                    {
                        "morphology_deadline_ms": 1_000.0,
                        "shape_aware_transfer_p90_ms": transfer_ms,
                    }
                ],
            }
        ],
    }


def _compare(byte: dict[str, object], shape: dict[str, object]) -> dict[str, object]:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        byte_path = root / "byte.jsonl"
        shape_path = root / "shape.jsonl"
        output = root / "comparison.json"
        byte_path.write_text(json.dumps(byte) + "\n", encoding="utf-8")
        shape_path.write_text(json.dumps(shape) + "\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "scripts/compare_predictive_replays.py",
                "--byte-only",
                str(byte_path),
                "--morphology-aware",
                str(shape_path),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(output.read_text(encoding="utf-8"))


def test_timing_and_reason_changes_do_not_pass_decision_gate() -> None:
    result = _compare(
        _record(transfer_ms=100.0, eligible=False, reasons=["hbm"]),
        _record(
            transfer_ms=150.0,
            eligible=False,
            reasons=["hbm", "morphology_window_miss"],
        ),
    )

    assert result["change_levels"] == {
        "timing_estimate_changed": True,
        "feasibility_reason_changed": True,
        "candidate_eligibility_changed": False,
        "selected_action_changed": False,
    }
    assert result["timing_sensitivity_gate"] is True
    assert result["decision_relevance_gate"] is False
    assert result["decision_change_gate"] is False
    assert result["shape_action_gate"] is False
    assert result["shape_veto_gate"] is False
    assert result["selected_action_gate"] is False
    assert result["online_canary_gate"] is False


def test_candidate_eligibility_change_passes_decision_gate() -> None:
    result = _compare(
        _record(transfer_ms=100.0, eligible=False, reasons=["hbm"]),
        _record(transfer_ms=100.0, eligible=True, reasons=[]),
    )

    assert result["change_levels"]["candidate_eligibility_changed"] is True
    assert result["decision_relevance_gate"] is True
    assert result["decision_change_gate"] is True
    assert result["shape_action_gate"] is True
    assert result["shape_veto_gate"] is False
    assert result["selected_action_gate"] is False
    assert result["recommended_validation_arm"] == "shape_aware_prepare_canary"
    assert result["online_canary_gate"] is True


def test_shape_veto_requires_byte_only_treatment() -> None:
    result = _compare(
        _record(transfer_ms=100.0, eligible=True, reasons=[]),
        _record(
            transfer_ms=150.0,
            eligible=False,
            reasons=["morphology_window_miss"],
        ),
    )

    assert result["change_levels"]["candidate_eligibility_changed"] is True
    assert result["decision_relevance_gate"] is True
    assert result["shape_action_gate"] is False
    assert result["shape_veto_gate"] is True
    assert result["supported_shape_veto_gate"] is True
    assert result["selected_action_gate"] is False
    assert result["recommended_validation_arm"] == "byte_only_veto_treatment"
    assert result["online_canary_gate"] is False


def test_unsupported_shape_veto_does_not_open_treatment_canary() -> None:
    result = _compare(
        _record(transfer_ms=100.0, eligible=True, reasons=[]),
        _record(
            transfer_ms=150.0,
            eligible=False,
            reasons=["shape_unsupported"],
            shape_supported=False,
        ),
    )

    assert result["shape_veto_gate"] is True
    assert result["supported_shape_veto_gate"] is False
    assert result["recommended_validation_arm"] == "shape_support_characterization"
    assert result["online_canary_gate"] is False


def test_selected_action_change_is_reported_without_opening_prepare_canary() -> None:
    result = _compare(
        _record(
            transfer_ms=100.0,
            eligible=False,
            reasons=["hbm"],
            selected_action="observed_baseline",
        ),
        _record(
            transfer_ms=100.0,
            eligible=False,
            reasons=["hbm"],
            selected_action="defer:context-1",
        ),
    )

    assert result["shape_action_gate"] is False
    assert result["shape_veto_gate"] is False
    assert result["selected_action_gate"] is True
    assert result["decision_relevance_gate"] is True
    assert result["recommended_validation_arm"] == "selected_action_characterization"
    assert result["online_canary_gate"] is False
