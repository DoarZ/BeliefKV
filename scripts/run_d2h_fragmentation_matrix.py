#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BYTES = 2_659_221_504


@dataclass(frozen=True)
class Arm:
    gpu: int
    fragmentation: str
    arm: str
    repetition: int

    @property
    def run_id(self) -> str:
        return (
            f"gpu{self.gpu}-{self.fragmentation}-{self.arm}-r{self.repetition}"
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _complete(path: Path) -> bool:
    manifest = path / "arm_manifest.json"
    if not manifest.is_file():
        return False
    value = json.loads(manifest.read_text(encoding="utf-8"))
    return isinstance(value, dict) and value.get("completed") is True


def _analyze_pair(root: Path, gpu: int, fragmentation: str, repetition: int) -> None:
    treatment = root / Arm(gpu, fragmentation, "treatment", repetition).run_id
    control = root / Arm(gpu, fragmentation, "control", repetition).run_id
    if not (_complete(treatment) and _complete(control)):
        return
    extent_min, extent_max = (6, 8) if fragmentation == "low" else (100, 112)
    measurement = {
        "gpu_id": str(gpu),
        "bytes_class": "large",
        "fragmentation_class": fragmentation,
        "pair_id": f"gpu{gpu}-{fragmentation}-r{repetition}",
        "repetition": repetition,
        "expected_bytes": EXPECTED_BYTES,
        "bytes_tolerance_fraction": 0.02,
        "expected_extent_min": extent_min,
        "expected_extent_max": extent_max,
        "treatment_run_id": treatment.name,
        "control_run_id": control.name,
        "correctness_evidence_path": "restore_micro_gate_validation.json",
    }
    measurement_path = treatment / "measurement_manifest.json"
    _write_json(measurement_path, measurement)
    subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "beliefkv",
            "python",
            "scripts/analyze_d2h_overlap.py",
            "--runtime-audit",
            str(treatment / "server/runtime_audit.jsonl"),
            "--control-runtime-audit",
            str(control / "server/runtime_audit.jsonl"),
            "--control-transfer-telemetry",
            str(control / "server/transfer_telemetry.jsonl"),
            "--transfer-telemetry",
            str(treatment / "server/transfer_telemetry.jsonl"),
            "--anchor-workflow-id",
            "restore-micro-gate:anchor",
            "--sequence-radius-tokens",
            "32",
            "--expected-extent-min",
            str(extent_min),
            "--expected-extent-max",
            str(extent_max),
            "--measurement-manifest",
            str(measurement_path),
            "--output",
            str(treatment / "d2h_overlap_analysis.json"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the sequential large-case D2H fragmentation matrix."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--only-gpu",
        type=int,
        action="append",
        default=[],
        help="Execute only these GPU arms while preserving the frozen full order.",
    )
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    root.mkdir(parents=True, exist_ok=True)
    arms = [
        Arm(gpu, fragmentation, arm, repetition)
        for gpu in (0, 1)
        for fragmentation in ("low", "high")
        for arm in ("treatment", "control")
        for repetition in range(args.repetitions)
    ]
    random.Random(args.seed).shuffle(arms)
    manifest_path = root / "matrix_manifest.json"
    if not manifest_path.exists():
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "seed": args.seed,
                "repetitions": args.repetitions,
                "sequential_gpu_execution": True,
                "arms": [asdict(item) | {"run_id": item.run_id} for item in arms],
            },
        )
    else:
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen_ids = [item["run_id"] for item in frozen.get("arms", ())]
        if frozen_ids != [item.run_id for item in arms]:
            raise RuntimeError("existing matrix manifest does not match this plan")

    selected_gpus = set(args.only_gpu)
    for arm in arms:
        if selected_gpus and arm.gpu not in selected_gpus:
            continue
        run_dir = root / arm.run_id
        if not _complete(run_dir):
            if run_dir.exists():
                raise RuntimeError(f"incomplete arm requires manual audit: {run_dir}")
            gate_id = f"p6-fragmentation-{arm.run_id}"
            subprocess.run(
                [
                    "scripts/run_d2h_overlap_arm.sh",
                    str(run_dir),
                    str(arm.gpu),
                    str(18_000 + arm.gpu),
                    arm.fragmentation,
                    arm.arm,
                    str(arm.repetition),
                    gate_id,
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
            )
        _analyze_pair(root, arm.gpu, arm.fragmentation, arm.repetition)

    analyses = sorted(root.glob("gpu*-treatment-r*/d2h_overlap_analysis.json"))
    if not analyses:
        return 0
    subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "beliefkv",
            "python",
            "scripts/aggregate_d2h_overlap_runs.py",
            *(str(path) for path in analyses),
            "--minimum-repetitions",
            str(args.repetitions),
            "--minimum-coverage",
            "0.95",
            "--maximum-sequence-delta",
            "32",
            "--expected-group",
            "0:large:low",
            "--expected-group",
            "0:large:high",
            "--expected-group",
            "1:large:low",
            "--expected-group",
            "1:large:high",
            "--output",
            str(root / "matrix_aggregate.json"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
