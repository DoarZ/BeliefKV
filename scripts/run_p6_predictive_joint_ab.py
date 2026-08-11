#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from beliefkv.experiments.predictive_joint_baseline import (
    validate_source_tree_fingerprint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = REPOSITORY_ROOT / "configs/p6/predictive_joint_v9/ab_run_plan.json"


def resolve_agent_python(explicit: Path | None = None) -> Path:
    configured = explicit
    if configured is None:
        value = os.environ.get("BELIEFKV_AGENT_PYTHON")
        configured = Path(value).expanduser() if value else None
    if configured is None:
        control_python = Path(sys.executable).resolve()
        configured = (
            control_python.parent.parent.parent
            / "beliefkv-agents"
            / "bin"
            / "python"
        )
    candidate = configured.expanduser().resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(
            "Deep Agents Python is unavailable; pass --agent-python or set "
            "BELIEFKV_AGENT_PYTHON"
        )
    return candidate


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _gpu_compute_processes(gpu: int) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _wait_for_server(base_url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20 * 60
    root = base_url.removesuffix("/v1")
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"SGLang exited during startup with code {return_code}")
        try:
            with urllib.request.urlopen(f"{root}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(5)
    raise TimeoutError(f"SGLang startup timed out: {last_error}")


def _run_spec(plan: dict[str, object], run_id: str) -> dict[str, object]:
    matches = [item for item in plan["runs"] if item["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate run ID: {run_id}")
    return dict(matches[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute exactly one frozen R5 predictive JointPlan A/B run."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument(
        "--agent-python",
        type=Path,
        help=(
            "Python executable for the beliefkv-agents environment; defaults "
            "to BELIEFKV_AGENT_PYTHON or the sibling Conda environment"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "experiments/ab/p6_predictive_joint_v9",
    )
    args = parser.parse_args()
    agent_python = resolve_agent_python(args.agent_python)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("frozen") is not True:
        raise ValueError("refusing to execute a non-frozen R5 plan")
    source_tree = plan.get("source_tree")
    if not isinstance(source_tree, dict):
        raise ValueError("frozen R5 plan omits source-tree fingerprint")
    validate_source_tree_fingerprint(source_tree, REPOSITORY_ROOT)
    spec = _run_spec(plan, args.run_id)
    output = args.output_root / args.run_id
    result_path = output / "r5_run_result.json"
    if output.exists():
        raise FileExistsError(
            f"run directory already exists; R5 forbids retries: {output}"
        )
    processes = _gpu_compute_processes(args.gpu)
    if processes:
        raise RuntimeError(
            f"GPU {args.gpu} is not idle; refusing to interfere: {processes}"
        )

    output.mkdir(parents=True)
    server_dir = output / "server"
    workload_dir = output / "workloads"
    arm = str(spec["arm"])
    artifacts = plan["artifacts"]
    runtime = plan["runtime"]
    workload = plan["workload"]
    base_url = f"http://127.0.0.1:{args.port}/v1"
    prepare = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/prepare_deepagents_server_config.py"),
        "--server-dir",
        str(server_dir),
        "--queue-service-observer",
        "--request-token-trace",
        "--enable-observed-admission",
        "--enable-online-joint",
        "--enable-running-retraction",
        "--subagent-fanout-profile",
        str(workload["fanout_profile"]),
        "--transfer-service-model",
        str(artifacts["transfer_service"]["path"]),
    ]
    if arm == "B":
        prepare.extend(
            [
                "--predictor-model",
                str(artifacts["predictor"]["path"]),
                "--gpu-service-model",
                str(artifacts["gpu_service"]["path"]),
                "--enable-predictive-risk-shadow",
                "--enable-predictive-joint-overlay",
                "--enable-frontier-retraction-shadow",
                "--predictive-prepare-canary-limit",
                "1",
                "--frontier-retraction-canary-limit",
                "1",
            ]
        )
    subprocess.run(prepare, cwd=REPOSITORY_ROOT, check=True)
    server_config = json.loads(
        (server_dir / "beliefkv_config.json").read_text(encoding="utf-8")
    )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PORT": str(args.port),
        "MAX_TOTAL_TOKENS": str(runtime["kv_pool_tokens"]),
        "CONTEXT_LENGTH": str(runtime["context_length"]),
        "MAX_RUNNING_REQUESTS": str(runtime["max_running_requests"]),
        "HICACHE_SIZE_GB": str(runtime["host_cache_gib"]),
        "SLEEP_ON_IDLE": "1",
    }
    started = datetime.now(timezone.utc).isoformat()
    server = subprocess.Popen(
        [
            str(REPOSITORY_ROOT / "scripts/launch_deepagents_swebench_server.sh"),
            str(server_dir),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        start_new_session=True,
    )
    workload_return_code: int | None = None
    error: str | None = None
    try:
        _wait_for_server(base_url, server)
        workload_command = [
            str(agent_python),
            str(REPOSITORY_ROOT / "scripts/run_deepagents_swebench.py"),
            "--mode",
            "autonomous",
            "--base-url",
            base_url,
            "--model",
            "Qwen3-Coder-30B-A3B-Instruct-FP8",
            "--workload-manifest",
            str(workload["manifest"]["path"]),
            "--control-socket",
            str(server_config["runtime_event_socket_path"]),
            "--server-audit",
            str(server_dir / "runtime_audit.jsonl"),
            "--server-events",
            str(server_dir / "runtime_events.sglang.jsonl"),
            "--server-log",
            str(server_dir / "server.log"),
            "--max-workflows",
            str(workload["concurrency"]),
            "--concurrency",
            str(workload["concurrency"]),
            "--workflow-arrival-interval-ms",
            str(workload["arrival_schedule"]["interval_ms"]),
            "--subagent-fanout-profile",
            str(workload["fanout_profile"]),
            "--gpu",
            str(args.gpu),
            "--pool-tokens",
            str(runtime["kv_pool_tokens"]),
            "--max-completion-tokens",
            "4096",
            "--sampling-seed",
            str(workload["random_seed"]),
            "--context-window-tokens",
            "32768",
            "--context-keep-tokens",
            "8192",
            "--summary-output-tokens",
            "2048",
            "--recursion-limit",
            "512",
            "--request-timeout",
            "7200",
            "--sandbox-preflight-command",
            "",
            "--disable-completion-gate",
            "--completion-repair-attempts",
            "0",
            "--gate",
            "system",
            "--output",
            str(workload_dir),
        ]
        workload_return_code = subprocess.run(
            workload_command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        ).returncode
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        subprocess.run(
            [
                str(REPOSITORY_ROOT / "scripts/stop_deepagents_swebench_server.sh"),
                str(server_dir),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        )

    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "pair_id": spec["pair_id"],
        "arm": arm,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "workload_return_code": workload_return_code,
        "error": error,
        "server_dir": str(server_dir),
        "workload_dir": str(workload_dir),
        "retry_permitted": False,
    }
    _write_json(result_path, payload)
    return 0 if workload_return_code == 0 and error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
