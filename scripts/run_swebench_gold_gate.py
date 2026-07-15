#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs/workloads/swebench_verified_gold_gate.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "experiments/raw/swebench_verified_gold_gate"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("SWE-bench gate config must be a JSON object")
    return raw


def _validate_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import swebench
        from swebench.harness.utils import (
            get_predictions_from_file,
            load_swebench_dataset,
        )
    except ImportError as error:
        raise RuntimeError(
            "Run this script in the beliefkv-swe environment"
        ) from error

    harness = config["harness"]
    harness_source = REPOSITORY_ROOT / "third_party/SWE-bench"
    actual_commit = _git_head(harness_source)
    if actual_commit != harness["commit"]:
        raise RuntimeError(
            "SWE-bench source mismatch: "
            f"expected {harness['commit']}, got {actual_commit}"
        )
    if swebench.__version__ != harness["version"]:
        raise RuntimeError(
            "SWE-bench package mismatch: "
            f"expected {harness['version']}, got {swebench.__version__}"
        )

    artifacts = config["local_artifacts"]
    raw_root = _resolve(artifacts["raw_dataset_path"])
    split = config["dataset"]["split"]
    _require_hash(
        raw_root / split / "data-00000-of-00001.arrow",
        artifacts["raw_arrow_sha256"],
        "raw dataset shard",
    )
    _require_hash(
        raw_root / split / "dataset_info.json",
        artifacts["dataset_info_sha256"],
        "raw dataset metadata",
    )

    gate_path = _resolve(artifacts["gold_gate_path"])
    _require_hash(gate_path, artifacts["gold_gate_sha256"], "gold gate")
    instance_ids = list(config["selection"]["instance_ids"])
    dataset = load_swebench_dataset(str(gate_path), split, instance_ids)
    predictions = get_predictions_from_file("gold", str(gate_path), split)
    prediction_ids = {item["instance_id"] for item in predictions}
    if len(dataset) != len(instance_ids) or prediction_ids != set(instance_ids):
        raise RuntimeError("Frozen gate instances do not match the selection")

    return {
        "dataset_revision": config["dataset"]["revision"],
        "harness_commit": actual_commit,
        "harness_version": swebench.__version__,
        "instance_ids": instance_ids,
        "gate_path": str(gate_path),
        "gate_sha256": artifacts["gold_gate_sha256"],
    }


def _check_docker() -> str:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        version = client.version().get("Version", "unknown")
        client.close()
        return str(version)
    except Exception as error:
        raise RuntimeError(
            "Docker daemon is unavailable to this user. Ensure `docker info` "
            "works before running the official SWE-bench evaluator."
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and run the pinned SWE-bench Verified gold gate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--artifact-check-only",
        action="store_true",
        help="Validate frozen inputs without requiring Docker.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_config(config_path)
    report = _validate_artifacts(config)
    if args.artifact_check_only:
        report["docker_checked"] = False
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    report["docker_version"] = _check_docker()
    report["docker_checked"] = True
    print(json.dumps(report, indent=2, sort_keys=True))

    evaluation = config["evaluation"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        report["gate_path"],
        "--predictions_path",
        evaluation["predictions_path"],
        "--max_workers",
        str(evaluation["max_workers"]),
        "--instance_ids",
        *report["instance_ids"],
        "--run_id",
        evaluation["run_id"],
        "--timeout",
        str(evaluation["timeout_seconds"]),
        "--cache_level",
        evaluation["cache_level"],
    ]
    environment = dict(os.environ)
    environment.setdefault("HF_DATASETS_OFFLINE", "1")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    return subprocess.run(command, cwd=output_dir, env=environment).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"SWE-bench gate failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
