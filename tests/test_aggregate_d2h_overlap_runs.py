import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def test_aggregate_uses_paired_runs_as_sampling_units() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        analyses: list[Path] = []
        for repetition, ratio in enumerate((0.1, 0.2, 0.3)):
            run = root / f"run-{repetition}"
            run.mkdir()
            correctness = run / "correctness.json"
            correctness.write_text('{"passed": true}\n', encoding="utf-8")
            analysis = run / "d2h_overlap_analysis.json"
            analysis.write_text(
                json.dumps(
                    {
                        "counterfactual_control_available": True,
                        "control_transfer_overlap_count": 0,
                        "treatment_concurrent_transfer_count": 0,
                        "covered_transfer_ratio": 0.98,
                        "primary_sample_count": 5,
                        "baseline": {"sample_count": 6},
                        "sequence_matching": {"center_delta_tokens": 4},
                        "transfer": {
                            "actual_bytes": 1_000,
                            "extent_count": 22,
                            "small_extent_ratio": 0.5,
                            "extent_bytes_min": 10,
                            "extent_bytes_p50": 20,
                            "extent_bytes_max": 30,
                            "pinned_host": True,
                            "command_kind": "offload_context",
                            "start_to_complete_ms": 50.0 + repetition,
                            "physical_certificate_matches_telemetry": True,
                        },
                        "unhidden_interference": {
                            "stall_ratio_p50_reference": ratio
                        },
                        "measurement": {
                            "gpu_id": "0",
                            "bytes_class": "small",
                            "fragmentation_class": "high",
                            "pair_id": f"pair-{repetition}",
                            "repetition": repetition,
                            "expected_bytes": 1_000,
                            "expected_extent_min": 20,
                            "expected_extent_max": 24,
                            "correctness_evidence_path": "correctness.json",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            analyses.append(analysis)
        output = root / "aggregate.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/aggregate_d2h_overlap_runs.py",
                *(str(path) for path in analyses),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))

    assert result["all_groups_eligible"] is True
    group = result["groups"][0]
    assert group["valid_pair_count"] == 3
    assert group["arm_groups"]["treatment"]["valid_run_count"] == 3
    assert group["arm_groups"]["control"]["valid_run_count"] == 3
    assert group["run_level_stall_ratio"]["sampling_unit"] == "paired_run"
    assert group["run_level_stall_ratio"]["mean"] == 0.2
    assert group["run_level_stall_ratio"]["bootstrap_mean_ci95"] is not None
    assert group["run_level_transfer_completion_ms"]["mean"] == 51.0
