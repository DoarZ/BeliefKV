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
from beliefkv.runtime.context_lifecycle import ContextLifecyclePolicy
from beliefkv.runtime.langchain_tool_safety import ToolObservationBudgetPolicy


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
    parser.add_argument(
        "--sampling-seed",
        type=int,
        help=(
            "Optional model sampling seed. Set the same value in paired w1/w4/w8 "
            "runs to make load-invariance evidence controlled."
        ),
    )
    parser.add_argument("--context-window-tokens", type=int, default=32_768)
    parser.add_argument("--context-keep-tokens", type=int, default=8_192)
    parser.add_argument("--summary-output-tokens", type=int, default=2_048)
    parser.add_argument("--tool-observation-turn-chars", type=int, default=65_536)
    parser.add_argument("--tool-observation-result-chars", type=int, default=16_384)
    parser.add_argument("--recursion-limit", type=int, default=512)
    parser.add_argument(
        "--disable-loop-guard",
        action="store_true",
        help="Disable stuck detection while retaining semantic completion",
    )
    parser.add_argument("--stuck-repeated-call-limit", type=int, default=3)
    parser.add_argument("--stuck-alternating-repetitions", type=int, default=3)
    parser.add_argument("--stuck-consecutive-error-limit", type=int, default=3)
    parser.add_argument("--stuck-no-progress-limit", type=int, default=5)
    parser.add_argument("--stuck-max-model-calls", type=int, default=32)
    parser.add_argument("--stuck-max-tool-calls", type=int, default=64)
    parser.add_argument("--stuck-recovery-model-calls", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--sandbox-command-timeout", type=int, default=600)
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
    parser.add_argument(
        "--gate",
        choices=("system", "native", "task-correctness"),
        default="system",
        help=(
            "Process exit gate: BeliefKV/runtime validity, native model completion, "
            "or the local SWE-bench correctness proxy"
        ),
    )
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
        sampling_seed=args.sampling_seed,
        recursion_limit=args.recursion_limit,
        request_timeout_s=args.request_timeout,
        sandbox_command_timeout_s=args.sandbox_command_timeout,
        sandbox_test_env_path=args.sandbox_test_env,
        sandbox_preflight_command=args.sandbox_preflight_command or None,
        completion_repair_attempts=args.completion_repair_attempts,
        context_lifecycle=ContextLifecyclePolicy(
            window_tokens=args.context_window_tokens,
            keep_tokens=args.context_keep_tokens,
            intermediate_output_tokens=args.max_completion_tokens,
            summary_output_tokens=args.summary_output_tokens,
        ),
        loop_guard=LoopGuardPolicy(
            enabled=not args.disable_loop_guard,
            repeated_call_limit=args.stuck_repeated_call_limit,
            alternating_cycle_repetitions=args.stuck_alternating_repetitions,
            consecutive_error_limit=args.stuck_consecutive_error_limit,
            consecutive_no_progress_limit=args.stuck_no_progress_limit,
            max_model_calls_without_completion=args.stuck_max_model_calls,
            max_tool_calls_without_completion=args.stuck_max_tool_calls,
            recovery_model_call_limit=args.stuck_recovery_model_calls,
        ),
        tool_observation_budget=ToolObservationBudgetPolicy(
            total_chars_per_turn=args.tool_observation_turn_chars,
            max_chars_per_result=args.tool_observation_result_chars,
        ),
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    passed = {
        "system": summary["system_jct_eligible_workflows"],
        "native": summary["native_agent_jct_eligible_workflows"],
        "task-correctness": summary["successful_workflows"],
    }[args.gate]
    return 0 if passed == summary["workflow_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
