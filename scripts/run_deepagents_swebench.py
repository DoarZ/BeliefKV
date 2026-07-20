#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_PAUSE_FILE = Path("/tmp/beliefkv-experiments.paused")
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.deepagents_swebench import (
    DeepAgentsExperimentConfig,
    SYMPY_SANDBOX_PREFLIGHT,
    run_experiment,
)
from beliefkv.experiments.agent_protocol import LoopGuardPolicy


DEFAULT_WORKLOAD = (
    REPOSITORY_ROOT / "configs/workloads/codex_swebench_sympy_subagents.json"
)
DEFAULT_IMAGE = "swebench/sweb.eval.x86_64.sympy_1776_sympy-20590:latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dynamic Deep Agents workflows on frozen SWE-bench instances and "
            "collect GPU, SGLang, and BeliefKV event telemetry."
        )
    )
    parser.add_argument("--mode", choices=("autonomous", "planned"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument(
        "--model", default="Qwen3-Coder-30B-A3B-Instruct-FP8"
    )
    parser.add_argument("--workload-manifest", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--docker-image", default=DEFAULT_IMAGE)
    parser.add_argument("--control-socket", type=Path)
    parser.add_argument(
        "--server-audit",
        type=Path,
        help="Append-only BeliefKV runtime audit to slice for this experiment",
    )
    parser.add_argument(
        "--server-events",
        type=Path,
        help="Append-only committed SGLang runtime events to slice",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        help="Append-only SGLang server log to slice",
    )
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--max-workflows", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pool-tokens", type=int, default=163_840)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--recursion-limit", type=int, default=200)
    parser.add_argument(
        "--disable-loop-guard",
        action="store_true",
        help="Disable stuck detection while retaining semantic completion",
    )
    parser.add_argument("--stuck-repeated-call-limit", type=int, default=3)
    parser.add_argument("--stuck-alternating-repetitions", type=int, default=3)
    parser.add_argument("--stuck-consecutive-error-limit", type=int, default=3)
    parser.add_argument("--stuck-no-progress-limit", type=int, default=5)
    parser.add_argument("--stuck-diagnostic-probe-limit", type=int, default=8)
    parser.add_argument("--stuck-max-model-calls", type=int, default=32)
    parser.add_argument("--stuck-max-tool-calls", type=int, default=64)
    parser.add_argument("--stuck-recovery-model-calls", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--sandbox-test-env",
        default="/opt/miniconda3/envs/testbed",
        help="Absolute conda environment path inside the SWE-bench image",
    )
    parser.add_argument(
        "--sandbox-preflight-command",
        default=SYMPY_SANDBOX_PREFLIGHT,
        help="Offline command that must pass before any model request is issued",
    )
    parser.add_argument("--completion-repair-attempts", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pause_file = Path(
        os.environ.get(
            "BELIEFKV_EXPERIMENT_PAUSE_FILE", str(DEFAULT_EXPERIMENT_PAUSE_FILE)
        )
    ).expanduser()
    if pause_file.exists():
        print(f"BeliefKV experiments are paused: {pause_file}", file=sys.stderr)
        return 75
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPOSITORY_ROOT
        / "experiments/raw/deepagents_swebench"
        / timestamp
        / args.mode
    )
    config = DeepAgentsExperimentConfig(
        mode=args.mode,
        base_url=args.base_url,
        model=args.model,
        output_dir=output,
        workload_manifest=args.workload_manifest,
        docker_image=args.docker_image,
        control_socket=args.control_socket,
        server_audit_path=args.server_audit,
        server_event_path=args.server_events,
        server_log_path=args.server_log,
        instance_ids=tuple(args.instance),
        max_workflows=args.max_workflows,
        concurrency=args.concurrency,
        gpu_index=args.gpu,
        pool_tokens=args.pool_tokens,
        max_completion_tokens=args.max_completion_tokens,
        recursion_limit=args.recursion_limit,
        request_timeout_s=args.request_timeout,
        sandbox_test_env_path=args.sandbox_test_env,
        sandbox_preflight_command=args.sandbox_preflight_command or None,
        completion_repair_attempts=args.completion_repair_attempts,
        loop_guard=LoopGuardPolicy(
            enabled=not args.disable_loop_guard,
            repeated_call_limit=args.stuck_repeated_call_limit,
            alternating_cycle_repetitions=args.stuck_alternating_repetitions,
            consecutive_error_limit=args.stuck_consecutive_error_limit,
            consecutive_no_progress_limit=args.stuck_no_progress_limit,
            consecutive_diagnostic_probe_limit=args.stuck_diagnostic_probe_limit,
            max_model_calls_without_completion=args.stuck_max_model_calls,
            max_tool_calls_without_completion=args.stuck_max_tool_calls,
            recovery_model_call_limit=args.stuck_recovery_model_calls,
        ),
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["successful_workflows"] == summary["workflow_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
