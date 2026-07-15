#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.runtime.agent_runtime_adapter import (
    InstrumentedToolEnvironment,
    InvocationRuntimeEmitter,
)
from beliefkv.runtime.event_channel import UnixDatagramRuntimeEventSink
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


DEFAULT_DATASET = (
    REPOSITORY_ROOT
    / "workloads/frozen/swebench_verified_gold_gate/instances.jsonl"
)
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/workloads/minisweagent_qwen2_5_7b_reactive.yaml"
)
MINI_SOURCE = REPOSITORY_ROOT / "third_party/mini-swe-agent"
EXPECTED_MINI_COMMIT = "388da74aad620a384ab47669b17c52133e30e7c3"
EXPECTED_MINI_VERSION = "2.4.5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_instance(dataset_path: Path, instance_id: str) -> tuple[dict, dict]:
    from swebench.harness.utils import load_swebench_dataset

    matches = load_swebench_dataset(str(dataset_path), "test", [instance_id])
    if len(matches) != 1:
        raise RuntimeError(f"expected one SWE-bench instance, got {len(matches)}")
    source = matches[0]
    agent_view = {
        key: source[key]
        for key in ("instance_id", "problem_statement")
    }
    for optional in ("image_name", "docker_image"):
        if source.get(optional):
            agent_view[optional] = source[optional]
    return source, agent_view


def _load_agent_config(path: Path, metadata: dict[str, object]) -> dict:
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    base = get_config_from_spec(
        str(builtin_config_dir / "benchmarks" / "swebench.yaml")
    )
    override = get_config_from_spec(str(path))
    dynamic = {
        "model": {
            "model_kwargs": {
                "extra_body": {"beliefkv_metadata": metadata},
            }
        }
    }
    return recursive_merge(base, override, dynamic)


def _check_model_server(api_base: str) -> dict:
    response = requests.get(f"{api_base.rstrip('/')}/models", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("data"):
        raise RuntimeError("model server returned no served models")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one pinned SWE-bench instance with authoritative BeliefKV events."
    )
    parser.add_argument("--instance-id", default="sympy__sympy-20590")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--event-socket", type=Path, required=True)
    args = parser.parse_args()

    import minisweagent
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    mini_commit = _git_head(MINI_SOURCE)
    if mini_commit != EXPECTED_MINI_COMMIT:
        raise RuntimeError(
            f"mini-SWE-agent commit mismatch: {mini_commit} != {EXPECTED_MINI_COMMIT}"
        )
    if minisweagent.__version__ != EXPECTED_MINI_VERSION:
        raise RuntimeError(
            "mini-SWE-agent version mismatch: "
            f"{minisweagent.__version__} != {EXPECTED_MINI_VERSION}"
        )

    dataset_path = args.dataset.resolve()
    config_path = args.config.resolve()
    event_socket = args.event_socket.resolve()
    run_dir = args.run_dir.resolve()
    staging_dir = run_dir / ".agent.incomplete"
    final_dir = run_dir / "agent"
    if final_dir.exists() or staging_dir.exists():
        raise FileExistsError("agent output already exists for this run")
    if not event_socket.is_socket():
        raise RuntimeError(f"runtime event socket is not ready: {event_socket}")
    staging_dir.mkdir(parents=True)

    source_instance, agent_instance = _load_instance(
        dataset_path, args.instance_id
    )
    workflow_id = f"swebench:{args.instance_id}:{run_dir.name}"
    metadata = BeliefKVRequestMetadata(
        root_workflow_id=workflow_id,
        invocation_id=f"{workflow_id}:coder",
        context_id=f"{workflow_id}:context",
        context_epoch=0,
        agent_definition_id="mini-swe-agent-coder",
        agent_instance_id=f"coder:{run_dir.name}",
    )
    config = _load_agent_config(config_path, metadata.to_wire())
    api_base = str(config["model"]["model_kwargs"]["api_base"])
    served_models = _check_model_server(api_base)

    model = get_model(config=config["model"])
    environment = get_sb_environment(config, agent_instance)
    start_utc = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()
    exit_status = "RunnerError"
    submission = ""
    runner_error = None
    agent = None

    try:
        with UnixDatagramRuntimeEventSink(event_socket) as sink:
            emitter = InvocationRuntimeEmitter(sink, metadata)
            traced_environment = InstrumentedToolEnvironment(environment, emitter)
            agent_config = dict(config["agent"])
            agent_config["output_path"] = staging_dir / "trajectory.partial.json"
            agent = DefaultAgent(model, traced_environment, **agent_config)
            emitter.start(source="mini-swe-agent")
            try:
                info = agent.run(source_instance["problem_statement"])
                exit_status = str(info.get("exit_status", "Unknown"))
                submission = str(info.get("submission", "") or "")
            except Exception as error:
                runner_error = f"{type(error).__name__}: {error}"
                exit_status = type(error).__name__
                submission = ""
            finally:
                emitter.finish(outcome=exit_status)
    finally:
        environment.cleanup()

    if agent is not None:
        agent.save(
            staging_dir / "trajectory.json",
            {
                "info": {
                    "instance_id": args.instance_id,
                    "exit_status": exit_status,
                    "submission": submission,
                    "runner_error": runner_error,
                }
            },
        )
    predictions = {
        args.instance_id: {
            "instance_id": args.instance_id,
            "model_name_or_path": config["model"]["model_name"],
            "model_patch": submission,
        }
    }
    _write_json(staging_dir / "preds.json", predictions)

    end_utc = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "instance_id": args.instance_id,
        "base_commit": source_instance["base_commit"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "gold_fields_exposed_to_agent": False,
        "agent_input_fields": sorted(agent_instance),
        "mini_swe_agent_commit": mini_commit,
        "mini_swe_agent_version": minisweagent.__version__,
        "model_name": config["model"]["model_name"],
        "served_model_ids": [item["id"] for item in served_models["data"]],
        "workflow_id": workflow_id,
        "invocation_id": metadata.invocation_id,
        "context_id": metadata.context_id,
        "event_socket": str(event_socket),
        "started_at_utc": start_utc.isoformat(),
        "finished_at_utc": end_utc.isoformat(),
        "duration_seconds": time.monotonic() - start_monotonic,
        "exit_status": exit_status,
        "submission_chars": len(submission),
        "runner_error": runner_error,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    if (staging_dir / "trajectory.json").exists():
        manifest["trajectory_sha256"] = _sha256(
            staging_dir / "trajectory.json"
        )
    manifest["predictions_sha256"] = _sha256(staging_dir / "preds.json")
    _write_json(staging_dir / "manifest.json", manifest)
    staging_dir.replace(final_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"SWE-bench agent run failed: {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2) from error
