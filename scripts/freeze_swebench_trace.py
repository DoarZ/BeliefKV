#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.traces.runtime_validation import (
    relative_event_records,
    validate_runtime_trace,
)


SOURCE_FILES = (
    "beliefkv/core/config.py",
    "beliefkv/core/events.py",
    "beliefkv/runtime/agent_runtime_adapter.py",
    "beliefkv/runtime/event_channel.py",
    "beliefkv/runtime/sglang_adapter.py",
    "beliefkv/runtime/sglang_v052rc1.py",
    "beliefkv/traces/runtime_validation.py",
    "patches/sglang-0.5.2rc1-beliefkv.patch",
    "scripts/freeze_swebench_trace.py",
    "scripts/run_swebench_agent_trace.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            )


def _git_state(path: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": head, "worktree_dirty": bool(status.strip())}


def _verify_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )


def _evaluation_outcome(
    report: dict[str, Any], instance_id: str
) -> str:
    categories = (
        ("resolved", "resolved_ids"),
        ("unresolved", "unresolved_ids"),
        ("empty_patch", "empty_patch_ids"),
        ("evaluator_error", "error_ids"),
        ("incomplete", "incomplete_ids"),
    )
    matches = [
        outcome
        for outcome, field in categories
        if instance_id in report.get(field, [])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"evaluation report does not classify {instance_id} exactly once: "
            f"{matches}"
        )
    return matches[0]


def _model_lock(relative_records: list[dict[str, Any]]) -> dict[str, Any]:
    model_paths = {
        str(record.get("attributes", {}).get("model"))
        for record in relative_records
        if record.get("kind") == "llm_submit"
    }
    model_paths.discard("None")
    if len(model_paths) != 1:
        raise ValueError(f"expected one model in trace, got {sorted(model_paths)}")
    model_path = Path(next(iter(model_paths)))
    lock: dict[str, Any] = {"path": str(model_path)}
    if not model_path.is_dir():
        lock["local_files_available"] = False
        return lock
    lock["local_files_available"] = True
    metadata_files: dict[str, str] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        candidate = model_path / name
        if candidate.is_file():
            metadata_files[name] = _sha256(candidate)
    lock["metadata_sha256"] = metadata_files
    lock["weight_shards"] = [
        {"name": shard.name, "size_bytes": shard.stat().st_size}
        for shard in sorted(model_path.glob("*.safetensors"))
    ]
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and atomically freeze one authoritative SWE-bench trace."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    destination = args.destination.resolve()
    evaluation_report_path = args.evaluation_report.resolve()
    if destination.exists():
        raise FileExistsError(f"frozen trace already exists: {destination}")

    event_path = run_dir / "server/runtime_events.jsonl"
    audit_path = run_dir / "server/runtime_audit.jsonl"
    agent_dir = run_dir / "agent"
    agent_manifest_path = agent_dir / "manifest.json"
    predictions_path = agent_dir / "preds.json"
    trajectory_path = agent_dir / "trajectory.json"
    server_config_path = run_dir / "server_config.json"
    required = (
        event_path,
        audit_path,
        agent_manifest_path,
        predictions_path,
        trajectory_path,
        server_config_path,
        evaluation_report_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"run is incomplete; missing artifacts: {missing}")

    summary = validate_runtime_trace(event_path, audit_path)
    agent_manifest = _read_json(agent_manifest_path)
    predictions = _read_json(predictions_path)
    evaluation_report = _read_json(evaluation_report_path)
    server_config = _read_json(server_config_path)
    instance_id = str(agent_manifest["instance_id"])
    if agent_manifest.get("run_id") != run_dir.name:
        raise ValueError("agent manifest run_id does not match run directory")
    if agent_manifest.get("workflow_id") != summary.workflow_id:
        raise ValueError("agent manifest workflow_id does not match runtime trace")
    if instance_id not in predictions:
        raise ValueError(f"prediction missing instance {instance_id}")
    prediction = predictions[instance_id]
    if prediction.get("instance_id") != instance_id:
        raise ValueError("prediction instance_id mismatch")
    submission = str(prediction.get("model_patch", "") or "")
    if len(submission) != int(agent_manifest.get("submission_chars", -1)):
        raise ValueError("prediction length disagrees with agent manifest")

    _verify_hash(
        predictions_path,
        str(agent_manifest["predictions_sha256"]),
        "predictions",
    )
    _verify_hash(
        trajectory_path,
        str(agent_manifest["trajectory_sha256"]),
        "trajectory",
    )
    dataset_path = Path(str(agent_manifest["dataset_path"]))
    agent_config_path = Path(str(agent_manifest["config_path"]))
    _verify_hash(
        dataset_path,
        str(agent_manifest["dataset_sha256"]),
        "dataset",
    )
    _verify_hash(
        agent_config_path,
        str(agent_manifest["config_sha256"]),
        "agent config",
    )

    outcome = _evaluation_outcome(evaluation_report, instance_id)
    if (outcome == "empty_patch") != (len(submission) == 0):
        raise ValueError("empty-patch classification disagrees with prediction")
    relative_records = relative_event_records(event_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.incomplete-",
            dir=destination.parent,
        )
    )
    try:
        copies = {
            "runtime_events.authoritative.jsonl": event_path,
            "runtime_audit.jsonl": audit_path,
            "agent_manifest.json": agent_manifest_path,
            "predictions.json": predictions_path,
            "trajectory.json": trajectory_path,
            "evaluation_report.json": evaluation_report_path,
            "server_config.json": server_config_path,
            "agent_config.yaml": agent_config_path,
            "dataset_instance.jsonl": dataset_path,
        }
        for name, source in copies.items():
            shutil.copy2(source, staging / name)
        _write_jsonl(staging / "runtime_events.relative.jsonl", relative_records)
        _write_json(staging / "trace_summary.json", summary.to_dict())

        source_hashes_at_freeze = {
            relative: _sha256(REPOSITORY_ROOT / relative)
            for relative in SOURCE_FILES
        }
        artifact_hashes = {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        provenance = {
            "beliefkv": _git_state(REPOSITORY_ROOT),
            "sglang": _git_state(REPOSITORY_ROOT / "third_party/sglang"),
            "mini_swe_agent": _git_state(
                REPOSITORY_ROOT / "third_party/mini-swe-agent"
            ),
            "swebench_harness": _git_state(
                REPOSITORY_ROOT / "third_party/SWE-bench"
            ),
        }
        manifest = {
            "schema_version": 1,
            "trace_id": destination.name,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_id": run_dir.name,
            "instance_id": instance_id,
            "workflow_id": summary.workflow_id,
            "trace_valid": True,
            "trace_purpose": ["mechanism_validation", "deterministic_replay"],
            "task_outcome": outcome,
            "task_resolved": outcome == "resolved",
            "agent_exit_status": agent_manifest.get("exit_status"),
            "submission_chars": len(submission),
            "gold_fields_exposed_to_agent": agent_manifest.get(
                "gold_fields_exposed_to_agent"
            ),
            "policy": {
                "predictor_enabled": server_config.get("predictor_enabled"),
                "shadow_enabled": server_config.get("shadow_enabled"),
            },
            "validation": summary.to_dict(),
            "model": _model_lock(relative_records),
            "provenance": provenance,
            "source_snapshot": {
                "capture_stage": "post_run_freeze",
                "generation_exact": False,
                "limitation": (
                    "The BeliefKV worktree was not snapshotted at server startup; "
                    "these hashes describe the source at freeze time."
                ),
                "file_sha256": source_hashes_at_freeze,
            },
            "artifacts": artifact_hashes,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
