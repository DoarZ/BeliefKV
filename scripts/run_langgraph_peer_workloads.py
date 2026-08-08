#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from beliefkv.experiments.agent_protocol import LoopGuardPolicy
from beliefkv.experiments.arrival_schedule import (
    WorkflowArrival,
    build_workflow_arrivals,
    group_arrivals,
)
from beliefkv.experiments.agentic_peer_backend import (
    AgenticPeerBackendConfig,
    ToolEnabledPeerBackend,
    summarize_agentic_runtime_trace,
)
from beliefkv.experiments.deepagents_swebench import (
    DEFAULT_SANDBOX_TEST_ENV,
    SYMPY_SANDBOX_PREFLIGHT,
    DockerWorkspaceBackend,
    JsonlAudit,
    _task_prompt,
    command_output,
    load_workload_bundle,
    prepare_workspace,
    summarize_agent_control,
    write_json,
)
from beliefkv.experiments.langgraph_peer_workflow import (
    LLMEventSource,
    LangGraphPeerWorkflow,
    OpenAICompatiblePeerBackend,
    TraceSensitivity,
)
from beliefkv.runtime.event_channel import (
    JsonlRuntimeEventSink,
    UnixDatagramRuntimeEventSink,
)
from beliefkv.runtime.context_lifecycle import ContextLifecyclePolicy


DEFAULT_IMAGE = "swebench/sweb.eval.x86_64.sympy_1776_sympy-20590:latest"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-input cyclic/mixed LangGraph peer workloads."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        default=Path("configs/workloads/deepagents_swebench_sympy_12.json"),
    )
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument(
        "--run-namespace",
        help="Stable namespace for runtime identities; defaults to output directory.",
    )
    parser.add_argument("--max-workflows", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--modes", default="mixed,mixed,cyclic")
    parser.add_argument(
        "--spawn-policy",
        choices=("autonomous", "required-range"),
        default="required-range",
    )
    parser.add_argument("--min-initial-subagents", type=int, default=2)
    parser.add_argument("--max-initial-subagents", type=int, default=4)
    parser.add_argument(
        "--arrival-mode",
        choices=("simultaneous", "batched"),
        default="batched",
    )
    parser.add_argument("--arrival-batch-size", type=int, default=2)
    parser.add_argument(
        "--arrival-batch-interval-seconds", type=float, default=20.0
    )
    parser.add_argument(
        "--backend",
        choices=("agentic", "structured-smoke"),
        default="agentic",
        help="Use real persistent tool loops, or the legacy one-request topology smoke.",
    )
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument(
        "--model", default="Qwen3-Coder-30B-A3B-Instruct-FP8"
    )
    parser.add_argument("--max-turns", type=int, default=18)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=32_768,
        help="Dynamic message-history tokens that trigger runtime compaction.",
    )
    parser.add_argument(
        "--context-keep-tokens",
        type=int,
        default=8_192,
        help="Recent dynamic message-history tokens retained after compaction.",
    )
    parser.add_argument(
        "--intermediate-completion-tokens",
        type=int,
        default=1_024,
        help="Output budget for ordinary tool-decision turns.",
    )
    parser.add_argument(
        "--summary-completion-tokens",
        type=int,
        default=2_048,
        help="Output budget for the internal context summarizer.",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--sandbox-command-timeout-seconds",
        type=int,
        default=600,
        help="Per-command timeout inside the offline SWE-bench sandbox.",
    )
    parser.add_argument(
        "--workflow-wall-clock-seconds",
        "--activation-wall-clock-seconds",
        dest="workflow_wall_clock_seconds",
        type=float,
        default=7200.0,
        help=(
            "Hard liveness deadline shared by the workflow and all descendants; "
            "the activation spelling is retained as a deprecated CLI alias."
        ),
    )
    parser.add_argument("--recursion-limit", type=int, default=512)
    parser.add_argument("--docker-image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--sandbox-test-env", default=DEFAULT_SANDBOX_TEST_ENV
    )
    parser.add_argument(
        "--sandbox-preflight-command", default=SYMPY_SANDBOX_PREFLIGHT
    )
    parser.add_argument("--stuck-repeated-call-limit", type=int, default=6)
    parser.add_argument("--stuck-consecutive-error-limit", type=int, default=6)
    parser.add_argument("--stuck-no-progress-limit", type=int, default=8)
    parser.add_argument("--stuck-max-model-calls", type=int, default=48)
    parser.add_argument("--stuck-max-tool-calls", type=int, default=128)
    parser.add_argument(
        "--enforce-stuck-call-budgets",
        action="store_true",
        help=(
            "Treat the model/tool call thresholds as termination conditions. "
            "By default they are telemetry only so progressing agents can run "
            "until model completion or the workflow watchdog."
        ),
    )
    parser.add_argument("--min-workflow-llm-requests", type=int, default=16)
    parser.add_argument("--min-workflow-tool-calls", type=int, default=16)
    parser.add_argument("--min-subagent-llm-requests", type=int, default=4)
    parser.add_argument("--min-subagent-tool-calls", type=int, default=3)
    parser.add_argument("--min-dynamic-subagent-fraction", type=float, default=0.30)
    parser.add_argument("--min-peer-reactivation-fraction", type=float, default=0.20)
    parser.add_argument("--event-ack-timeout", type=float, default=10.0)
    parser.add_argument("--event-retries", type=int, default=6)
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--min-kv-pool-tokens", type=int, default=0)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_events(path: Path, events: tuple[object, ...]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for event in events:
            stream.write(
                json.dumps(event.to_dict(), sort_keys=True, allow_nan=False) + "\n"
            )
    temporary.replace(path)


def _server_root(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def _flush_cache(base_url: str, timeout: float) -> object:
    request = urllib.request.Request(
        f"{_server_root(base_url)}/flush_cache",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        if not payload:
            return {"status": int(response.status), "body": None}
        text = payload.decode("utf-8", errors="replace")
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = text
        return {"status": int(response.status), "body": body}


def _server_info(base_url: str, timeout: float) -> dict[str, object]:
    with urllib.request.urlopen(
        f"{_server_root(base_url)}/get_server_info", timeout=timeout
    ) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("SGLang server info must be a JSON object")
    return payload


def _workflow_id(run_id: str, instance_id: str, mode: str, index: int) -> str:
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:12]
    return f"p3-{run_id}-peer-{mode}-{index:03d}-{digest}"


def _collect_workspace_artifacts(
    workspace: Path,
    destination: Path,
    *,
    command_runner=command_output,
) -> dict[str, object]:
    errors: list[str] = []
    patch = ""
    final_status = ""
    try:
        patch = command_runner(
            ["git", "diff", "--binary", "HEAD"], cwd=workspace
        )
    except Exception as caught:
        errors.append(f"model.patch: {type(caught).__name__}: {caught}")
    try:
        (destination / "model.patch").write_text(
            patch + ("\n" if patch else ""), encoding="utf-8"
        )
    except OSError as caught:
        errors.append(f"model.patch.write: {type(caught).__name__}: {caught}")
    try:
        final_status = command_runner(
            ["git", "status", "--porcelain"], cwd=workspace
        )
    except Exception as caught:
        errors.append(f"git.status: {type(caught).__name__}: {caught}")
    return {
        "patch_chars": len(patch),
        "workspace_modified": bool(final_status),
        "final_status": final_status,
        "artifact_collection_valid": not errors,
        "artifact_collection_errors": errors,
    }


def _unhandled_workflow_result(
    *,
    workload: object,
    arrival: WorkflowArrival,
    mode: str,
    run_id: str,
    caught: BaseException,
) -> dict[str, object]:
    instance_id = str(workload.instance_id)
    return {
        "schema_version": 1,
        "workflow_id": _workflow_id(
            run_id, instance_id, mode, arrival.workflow_index
        ),
        "instance_id": instance_id,
        "mode": mode,
        "completed": False,
        "termination_reason": "worker_unhandled_exception",
        "turn_count": None,
        "transition_hash": None,
        "trace_sensitivity": TraceSensitivity.SEMANTIC_RACE_SENSITIVE.value,
        "orchestration_event_count": 0,
        "duration_seconds": None,
        "arrival": {
            "batch_index": arrival.batch_index,
            "position_in_batch": arrival.position_in_batch,
            "scheduled_start_offset_seconds": arrival.scheduled_offset_seconds,
            "worker_start_offset_seconds": None,
            "runtime_start_offset_seconds": None,
            "runtime_start_lag_seconds": None,
            "spawn_offsets_seconds": [],
        },
        "error": f"{type(caught).__name__}: {caught}",
        "backend": None,
        "agent_control": {
            "event_counts": {},
            "stuck_reasons": {"worker_unhandled_exception": 1},
            "semantic_completions": 0,
            "natural_semantic_completions": 0,
            "forced_semantic_completions": 0,
            "guard_intervened_completions": 0,
            "protocol_repaired_completions": 0,
            "protocol_normalized_completions": 0,
            "protocol_repair_failures": 1,
            "duplicate_tool_calls_suppressed": 0,
        },
        "agent_runtime_trace": {
            "event_count": 0,
            "event_counts": {},
            "dynamic_subagent_count": 0,
            "llm_request_count": 0,
            "tool_call_count": 0,
            "tool_status_counts": {},
            "tool_error_class_counts": {},
            "tool_status_coverage": 1.0,
            "workspace_digest_observation_count": 0,
            "mutating_tool_end_count": 0,
            "workspace_digest_coverage": 1.0,
            "workspace_change_count": 0,
            "multi_turn_subagent_count": 0,
            "all_subagents_returned": False,
            "all_joins_satisfied": False,
            "child_invocations": [],
            "spawn_timestamps_ms": [],
        },
        "patch_chars": 0,
        "workspace_modified": False,
        "final_status": "",
        "artifact_collection_valid": False,
        "artifact_collection_errors": [
            f"worker: {type(caught).__name__}: {caught}"
        ],
        "load_valid": False,
        "guard_valid": False,
        "runtime_protocol_valid": False,
        "native_protocol_valid": False,
        "runtime_valid": False,
        "clean_jct_eligible": False,
        "workflow_intensity_valid": False,
        "subagent_intensity_valid": False,
        "spawn_range_valid": False,
        "subagent_trace_valid": False,
        "peer_reactivation_count": 0,
    }


def _initial_spawn_range_valid(
    *,
    required: bool,
    observed_initial_subagents: object,
    minimum: int,
    maximum: int,
) -> bool:
    if not required:
        return True
    if observed_initial_subagents is None:
        return False
    observed = int(observed_initial_subagents)
    return minimum <= observed <= maximum


def _run_one(
    *,
    workload: object,
    source_repo: Path,
    arrival: WorkflowArrival,
    mode: str,
    run_id: str,
    args: argparse.Namespace,
    output: Path,
    experiment_started_monotonic: float,
) -> dict[str, object]:
    worker_started = time.monotonic()
    index = arrival.workflow_index
    instance_id = str(workload.instance_id)
    workflow_id = _workflow_id(run_id, instance_id, mode, index)
    destination = output / workflow_id
    destination.mkdir(parents=True)
    workspace_backend = None
    audit = None
    runtime_trace_sink = None
    workspace = None
    if args.backend == "agentic":
        workspace = destination / "workspace"
        workspace_metadata = prepare_workspace(source_repo, workload, workspace)
        write_json(destination / "workspace.json", workspace_metadata)
        audit = JsonlAudit(destination / "sandbox_audit.jsonl")
        workspace_backend = DockerWorkspaceBackend(
            workspace,
            image=args.docker_image,
            audit=audit,
            default_timeout_s=args.sandbox_command_timeout_seconds,
            test_env_path=args.sandbox_test_env,
            preflight_command=args.sandbox_preflight_command or None,
        )
        runtime_trace_sink = JsonlRuntimeEventSink(
            destination / "runtime_events.agentic.jsonl"
        )
    started = worker_started
    result = None
    error = None
    workflow = None
    backend = None
    try:
        if workspace_backend is not None:
            workspace_backend.start()
        with UnixDatagramRuntimeEventSink(
            args.control_socket,
            ack_timeout_s=args.event_ack_timeout,
            retries=args.event_retries,
        ) as sink:
            if args.backend == "agentic":
                assert workspace_backend is not None
                assert audit is not None
                assert runtime_trace_sink is not None
                required_range = mode == "mixed" and args.spawn_policy == "required-range"
                backend = ToolEnabledPeerBackend(
                    config=AgenticPeerBackendConfig(
                        model=args.model,
                        base_url=args.base_url,
                        max_completion_tokens=args.max_completion_tokens,
                        context_lifecycle=ContextLifecyclePolicy(
                            window_tokens=args.context_window_tokens,
                            keep_tokens=args.context_keep_tokens,
                            intermediate_output_tokens=(
                                args.intermediate_completion_tokens
                            ),
                            summary_output_tokens=args.summary_completion_tokens,
                        ),
                        request_timeout_s=args.timeout,
                        recursion_limit=args.recursion_limit,
                        enable_subagents=mode == "mixed",
                        required_initial_subagent_min=(
                            args.min_initial_subagents if required_range else 0
                        ),
                        required_initial_subagent_max=(
                            args.max_initial_subagents if required_range else 0
                        ),
                        loop_guard=LoopGuardPolicy(
                            repeated_call_limit=args.stuck_repeated_call_limit,
                            alternating_cycle_repetitions=3,
                            consecutive_error_limit=args.stuck_consecutive_error_limit,
                            consecutive_no_progress_limit=args.stuck_no_progress_limit,
                            max_model_calls_without_completion=(
                                args.stuck_max_model_calls
                            ),
                            max_tool_calls_without_completion=args.stuck_max_tool_calls,
                            enforce_call_budgets=args.enforce_stuck_call_budgets,
                            recovery_model_call_limit=2,
                            activation_wall_clock_s=(
                                args.workflow_wall_clock_seconds
                            ),
                        ),
                    ),
                    workspace_backend=workspace_backend,
                    audit=audit,
                    runtime_trace_sink=runtime_trace_sink,
                    control_sink=sink,
                )
            else:
                required_range = mode == "mixed" and args.spawn_policy == "required-range"
                backend = OpenAICompatiblePeerBackend(
                    model=args.model,
                    base_url=args.base_url,
                    max_completion_tokens=args.max_completion_tokens,
                    timeout_s=args.timeout,
                    min_initial_subagents=(
                        args.min_initial_subagents if required_range else 0
                    ),
                    max_initial_subagents=(
                        args.max_initial_subagents
                        if required_range
                        else (4 if mode == "mixed" else 0)
                    ),
                    max_attempts=2,
                )
            workflow = LangGraphPeerWorkflow(
                backend,
                sink,
                workflow_id=workflow_id,
                max_turns=args.max_turns,
                trace_sensitivity=TraceSensitivity.SEMANTIC_RACE_SENSITIVE,
                llm_event_source=LLMEventSource.MODEL_RUNTIME,
                parallel_subagents=mode == "mixed",
                workflow_timeout_s=args.workflow_wall_clock_seconds,
            )
            result = workflow.run(
                _task_prompt(workload)
                if args.backend == "agentic"
                else str(workload.problem_statement)
            )
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        events = tuple(workflow.emitter.events) if workflow is not None else ()
        if runtime_trace_sink is not None:
            runtime_trace_sink.close()
        if workspace_backend is not None:
            workspace_backend.close()
        if audit is not None:
            audit.close()
    runtime_finished = time.monotonic()
    _write_events(destination / "orchestration_events.jsonl", events)
    artifact_collection = {
        "patch_chars": 0,
        "workspace_modified": False,
        "final_status": "",
        "artifact_collection_valid": True,
        "artifact_collection_errors": [],
    }
    if workspace is not None:
        artifact_collection = _collect_workspace_artifacts(
            workspace, destination
        )
    agent_trace_path = destination / "runtime_events.agentic.jsonl"
    agent_trace = (
        summarize_agentic_runtime_trace(agent_trace_path)
        if agent_trace_path.is_file()
        else {
            "event_count": 0,
            "event_counts": {},
            "dynamic_subagent_count": 0,
            "llm_request_count": 0,
            "tool_call_count": 0,
            "multi_turn_subagent_count": 0,
            "all_subagents_returned": False,
            "all_joins_satisfied": True,
            "child_invocations": [],
            "spawn_timestamps_ms": [],
        }
    )
    agent_control = (
        summarize_agent_control(destination / "sandbox_audit.jsonl")
        if args.backend == "agentic"
        else {
            "event_counts": {},
            "stuck_reasons": {},
            "semantic_completions": 0,
            "natural_semantic_completions": 0,
            "forced_semantic_completions": 0,
            "guard_intervened_completions": 0,
            "protocol_repaired_completions": 0,
            "protocol_normalized_completions": 0,
            "protocol_repair_failures": 0,
            "duplicate_tool_calls_suppressed": 0,
        }
    )
    reactivation_count = sum(
        event.kind.value == "reactivate" for event in events
    )
    runtime_start_offset_seconds = (
        (events[0].ts_ms / 1000.0) - experiment_started_monotonic
        if events
        else None
    )
    spawn_offsets_seconds = [
        (timestamp_ms / 1000.0) - experiment_started_monotonic
        for timestamp_ms in agent_trace.get("spawn_timestamps_ms", ())
    ]
    child_stats = agent_trace["child_invocations"]
    workflow_intensity_valid = (
        agent_trace["llm_request_count"] >= args.min_workflow_llm_requests
        and agent_trace["tool_call_count"] >= args.min_workflow_tool_calls
    )
    subagent_intensity_valid = all(
        int(item["llm_request_count"]) >= args.min_subagent_llm_requests
        and int(item["tool_call_count"]) >= args.min_subagent_tool_calls
        for item in child_stats
    )
    backend_summary = backend.summary() if backend is not None else None
    observed_initial_subagents = (
        backend_summary.get("observed_initial_subagent_count")
        if isinstance(backend_summary, dict)
        else None
    )
    spawn_range_required = mode == "mixed" and args.spawn_policy == "required-range"
    spawn_range_valid = _initial_spawn_range_valid(
        required=spawn_range_required,
        observed_initial_subagents=observed_initial_subagents,
        minimum=args.min_initial_subagents,
        maximum=args.max_initial_subagents,
    )
    guard_valid = (
        not agent_control["stuck_reasons"]
        and int(agent_control["forced_semantic_completions"]) == 0
    )
    runtime_protocol_valid = int(agent_control["protocol_repair_failures"]) == 0
    native_protocol_valid = (
        int(agent_control["protocol_repaired_completions"]) == 0
        and int(agent_control["protocol_normalized_completions"]) == 0
    )
    runtime_observability_valid = (
        float(agent_trace["tool_status_coverage"]) == 1.0
        and float(agent_trace["workspace_digest_coverage"]) == 1.0
    )
    clean_jct_eligible = bool(
        result
        and result.completed
        and result.termination_reason == "semantic_complete"
        and error is None
        and guard_valid
        and runtime_protocol_valid
        and runtime_observability_valid
    )
    runtime_valid = bool(
        result
        and result.completed
        and error is None
        and runtime_protocol_valid
        and runtime_observability_valid
    )
    summary = {
        "schema_version": 1,
        "workflow_id": workflow_id,
        "instance_id": instance_id,
        "mode": mode,
        "completed": bool(result and result.completed),
        "termination_reason": (
            result.termination_reason if result is not None else "runtime_error"
        ),
        "turn_count": result.turn_count if result is not None else None,
        "transition_hash": result.transition_hash if result is not None else None,
        "trace_sensitivity": TraceSensitivity.SEMANTIC_RACE_SENSITIVE.value,
        "orchestration_event_count": len(events),
        "duration_seconds": runtime_finished - started,
        "arrival": {
            "batch_index": arrival.batch_index,
            "position_in_batch": arrival.position_in_batch,
            "scheduled_start_offset_seconds": arrival.scheduled_offset_seconds,
            "worker_start_offset_seconds": (
                worker_started - experiment_started_monotonic
            ),
            "runtime_start_offset_seconds": runtime_start_offset_seconds,
            "runtime_start_lag_seconds": (
                runtime_start_offset_seconds - arrival.scheduled_offset_seconds
                if runtime_start_offset_seconds is not None
                else None
            ),
            "spawn_offsets_seconds": spawn_offsets_seconds,
        },
        "error": error,
        "backend": backend_summary,
        "agent_control": agent_control,
        "agent_runtime_trace": agent_trace,
        **artifact_collection,
        "load_valid": bool(result and result.completed)
        and guard_valid
        and runtime_protocol_valid
        and runtime_observability_valid
        and workflow_intensity_valid
        and subagent_intensity_valid
        and spawn_range_valid,
        "guard_valid": guard_valid,
        "runtime_protocol_valid": runtime_protocol_valid,
        "native_protocol_valid": native_protocol_valid,
        "runtime_valid": runtime_valid,
        "runtime_observability_valid": runtime_observability_valid,
        "clean_jct_eligible": clean_jct_eligible,
        "workflow_intensity_valid": workflow_intensity_valid,
        "subagent_intensity_valid": subagent_intensity_valid,
        "spawn_range_valid": spawn_range_valid,
        "subagent_trace_valid": agent_trace["dynamic_subagent_count"] > 0
        and agent_trace["multi_turn_subagent_count"]
        == agent_trace["dynamic_subagent_count"]
        and agent_trace["all_subagents_returned"]
        and agent_trace["all_joins_satisfied"],
        "peer_reactivation_count": reactivation_count,
    }
    _write_json(destination / "result.json", summary)
    return summary


def main() -> int:
    args = _parse_args()
    if min(
        args.max_workflows,
        args.concurrency,
        args.max_turns,
        args.max_completion_tokens,
        args.context_window_tokens,
        args.context_keep_tokens,
        args.intermediate_completion_tokens,
        args.summary_completion_tokens,
        args.recursion_limit,
        args.event_retries,
        args.arrival_batch_size,
    ) <= 0:
        raise ValueError("workflow, concurrency, turn and token limits must be positive")
    if args.event_ack_timeout <= 0:
        raise ValueError("event ACK timeout must be positive")
    if (
        args.timeout <= 0
        or args.workflow_wall_clock_seconds <= 0
        or args.sandbox_command_timeout_seconds <= 0
    ):
        raise ValueError("request and workflow timeouts must be positive")
    if args.context_keep_tokens >= args.context_window_tokens:
        raise ValueError("context keep tokens must be smaller than the window")
    if (
        args.summary_completion_tokens
        > args.context_window_tokens - args.context_keep_tokens
    ):
        raise ValueError("summary output must fit outside retained context")
    if args.intermediate_completion_tokens > args.max_completion_tokens:
        raise ValueError("intermediate completion budget exceeds the final budget")
    if args.arrival_batch_interval_seconds < 0:
        raise ValueError("arrival batch interval must be non-negative")
    if not (
        0
        <= args.min_initial_subagents
        <= args.max_initial_subagents
        <= 4
    ):
        raise ValueError("initial subagent range must satisfy 0 <= min <= max <= 4")
    if args.spawn_policy == "required-range" and args.min_initial_subagents == 0:
        raise ValueError("required-range needs at least one initial subagent")
    intensity_thresholds = (
        args.min_workflow_llm_requests,
        args.min_workflow_tool_calls,
        args.min_subagent_llm_requests,
        args.min_subagent_tool_calls,
    )
    if min(intensity_thresholds) < 0:
        raise ValueError("workload intensity thresholds must be non-negative")
    coverage_fractions = (
        args.min_dynamic_subagent_fraction,
        args.min_peer_reactivation_fraction,
    )
    if any(value < 0.0 or value > 1.0 for value in coverage_fractions):
        raise ValueError("workload coverage fractions must be in [0, 1]")
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if not modes or set(modes) - {"cyclic", "mixed"}:
        raise ValueError("modes must contain cyclic and/or mixed")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    control_socket = args.control_socket.expanduser().resolve()
    if not control_socket.exists():
        raise FileNotFoundError(control_socket)
    bundle = load_workload_bundle(args.workload_manifest)
    by_id = {item.instance_id: item for item in bundle.workloads}
    if args.instance_id:
        unknown = set(args.instance_id) - set(by_id)
        if unknown:
            raise ValueError(f"unknown SWE-bench instances: {sorted(unknown)}")
        selected = tuple(by_id[item] for item in args.instance_id)
    else:
        selected = bundle.workloads[: args.max_workflows]
    if not selected:
        raise ValueError("no SWE-bench workloads selected")
    if args.concurrency < len(selected):
        raise ValueError(
            "concurrency must cover every selected workflow so client-side worker "
            "queueing cannot distort the configured arrival schedule"
        )
    arrivals = build_workflow_arrivals(
        len(selected),
        mode=args.arrival_mode,
        batch_size=args.arrival_batch_size,
        batch_interval_seconds=args.arrival_batch_interval_seconds,
    )
    if args.min_kv_pool_tokens < 0:
        raise ValueError("minimum KV pool tokens must be non-negative")
    server_info = _server_info(args.base_url, args.timeout)
    actual_pool_tokens = int(server_info.get("max_total_num_tokens", 0))
    if actual_pool_tokens < args.min_kv_pool_tokens:
        raise RuntimeError(
            "SGLang KV pool is below the experiment gate: "
            f"actual={actual_pool_tokens}, required={args.min_kv_pool_tokens}"
        )
    staging.mkdir(parents=True)
    run_namespace = args.run_namespace or str(output)
    run_id = hashlib.sha256(run_namespace.encode("utf-8")).hexdigest()[:10]
    flush_result = _flush_cache(args.base_url, args.timeout) if args.flush_cache else None
    started_at = datetime.now(timezone.utc)
    experiment_started_monotonic = time.monotonic()
    results = []
    try:
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {}
            for batch in group_arrivals(arrivals):
                target = (
                    experiment_started_monotonic
                    + batch[0].scheduled_offset_seconds
                )
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                for arrival in batch:
                    workload = selected[arrival.workflow_index]
                    mode = modes[arrival.workflow_index % len(modes)]
                    future = executor.submit(
                        _run_one,
                        workload=workload,
                        source_repo=bundle.source_repo,
                        arrival=arrival,
                        mode=mode,
                        run_id=run_id,
                        args=args,
                        output=staging,
                        experiment_started_monotonic=(
                            experiment_started_monotonic
                        ),
                    )
                    futures[future] = (workload, arrival, mode)
            for future in as_completed(futures):
                workload, arrival, mode = futures[future]
                try:
                    results.append(future.result())
                except BaseException as caught:
                    failure = _unhandled_workflow_result(
                        workload=workload,
                        arrival=arrival,
                        mode=mode,
                        run_id=run_id,
                        caught=caught,
                    )
                    _write_json(
                        staging / str(failure["workflow_id"]) / "result.json",
                        failure,
                    )
                    results.append(failure)
        results.sort(key=lambda item: str(item["workflow_id"]))
        workflow_results_valid = all(
            bool(item["completed"])
            and bool(item["load_valid"])
            for item in results
        )
        mixed_workflow_count = sum(item["mode"] == "mixed" for item in results)
        required_dynamic_subagent_workflows = min(
            mixed_workflow_count,
            math.ceil(len(results) * args.min_dynamic_subagent_fraction),
        )
        dynamic_subagent_workflow_count = sum(
            bool(item["subagent_trace_valid"]) for item in results
        )
        required_peer_reactivation_workflows = math.ceil(
            len(results) * args.min_peer_reactivation_fraction
        )
        peer_reactivation_workflow_count = sum(
            int(item["peer_reactivation_count"]) > 0 for item in results
        )
        experiment_valid = (
            workflow_results_valid
            and dynamic_subagent_workflow_count
            >= required_dynamic_subagent_workflows
            and peer_reactivation_workflow_count
            >= required_peer_reactivation_workflows
        )
        manifest = {
            "schema_version": 1,
            "run_namespace": run_namespace,
            "run_id": run_id,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": bundle.dataset,
            "dataset_revision": bundle.dataset_revision,
            "workload_manifest": str(bundle.manifest_path),
            "workload_manifest_sha256": bundle.manifest_sha256,
            "instance_ids": [item.instance_id for item in selected],
            "modes": list(modes),
            "concurrency": len(selected),
            "model": args.model,
            "backend": args.backend,
            "max_turns": args.max_turns,
            "max_completion_tokens": args.max_completion_tokens,
            "context_lifecycle": {
                "window_tokens": args.context_window_tokens,
                "keep_tokens": args.context_keep_tokens,
                "intermediate_output_tokens": (
                    args.intermediate_completion_tokens
                ),
                "summary_output_tokens": args.summary_completion_tokens,
            },
            "request_timeout_seconds": args.timeout,
            "workflow_wall_clock_seconds": args.workflow_wall_clock_seconds,
            "activation_wall_clock_seconds": args.workflow_wall_clock_seconds,
            "recursion_limit": args.recursion_limit,
            "spawn_policy": {
                "mode": args.spawn_policy,
                "min_initial_subagents": args.min_initial_subagents,
                "max_initial_subagents": args.max_initial_subagents,
                "scope": "first_coder_activation_of_mixed_workflows",
            },
            "arrival_policy": {
                "mode": args.arrival_mode,
                "batch_size": (
                    len(selected)
                    if args.arrival_mode == "simultaneous"
                    else args.arrival_batch_size
                ),
                "batch_interval_seconds": (
                    0.0
                    if args.arrival_mode == "simultaneous"
                    else args.arrival_batch_interval_seconds
                ),
                "client_worker_count": len(selected),
                "schedule": [
                    {
                        "workflow_index": item.workflow_index,
                        "batch_index": item.batch_index,
                        "position_in_batch": item.position_in_batch,
                        "scheduled_offset_seconds": (
                            item.scheduled_offset_seconds
                        ),
                    }
                    for item in arrivals
                ],
            },
            "loop_guard": {
                "repeated_call_limit": args.stuck_repeated_call_limit,
                "consecutive_error_limit": args.stuck_consecutive_error_limit,
                "consecutive_no_progress_limit": args.stuck_no_progress_limit,
                "max_model_calls_without_completion": (
                    args.stuck_max_model_calls
                ),
                "max_tool_calls_without_completion": (
                    args.stuck_max_tool_calls
                ),
                "enforce_call_budgets": args.enforce_stuck_call_budgets,
                "workflow_wall_clock_seconds": (
                    args.workflow_wall_clock_seconds
                ),
            },
            "intensity_gate": {
                "min_workflow_llm_requests": args.min_workflow_llm_requests,
                "min_workflow_tool_calls": args.min_workflow_tool_calls,
                "min_subagent_llm_requests": args.min_subagent_llm_requests,
                "min_subagent_tool_calls": args.min_subagent_tool_calls,
                "min_dynamic_subagent_fraction": (
                    args.min_dynamic_subagent_fraction
                ),
                "required_dynamic_subagent_workflows": (
                    required_dynamic_subagent_workflows
                ),
                "min_peer_reactivation_fraction": (
                    args.min_peer_reactivation_fraction
                ),
                "required_peer_reactivation_workflows": (
                    required_peer_reactivation_workflows
                ),
            },
            "runtime_event_delivery": {
                "ack_timeout_seconds": args.event_ack_timeout,
                "retries": args.event_retries,
            },
            "server_capacity": {
                "max_total_num_tokens": actual_pool_tokens,
                "context_length": server_info.get("context_length"),
                "mem_fraction_static": server_info.get("mem_fraction_static"),
                "minimum_required_tokens": args.min_kv_pool_tokens,
            },
            "llm_event_source": LLMEventSource.MODEL_RUNTIME.value,
            "parallel_subagents_in_mixed_mode": True,
            "subagent_fanout_policy": {
                "cyclic": {"min": 0, "max": 0},
                "mixed": {
                    "preconfigured_count": None,
                    "selection": (
                        "runtime_enforced_model_choice_within_range"
                        if args.spawn_policy == "required-range"
                        else "model_task_tool_calls_at_runtime"
                    ),
                    "min": (
                        args.min_initial_subagents
                        if args.spawn_policy == "required-range"
                        else 0
                    ),
                    "max": (
                        args.max_initial_subagents
                        if args.spawn_policy == "required-range"
                        else None
                    ),
                },
                "safety": {
                    "semantic_completion_required": True,
                    "loop_guard_enabled": True,
                    "graph_max_turns": args.max_turns,
                },
            },
            "cache_flush_requested": args.flush_cache,
            "cache_flush_result": flush_result,
            "tool_runtime": (
                {
                    "docker_image": args.docker_image,
                    "network": "none",
                    "workspace_isolation": "per_workflow_git_clone",
                    "sandbox_test_env": args.sandbox_test_env,
                    "command_timeout_seconds": (
                        args.sandbox_command_timeout_seconds
                    ),
                }
                if args.backend == "agentic"
                else None
            ),
            "evaluation_scope": (
                "dynamic agent, tool-wait, and KV characterization on real SWE-bench "
                "inputs; SWE-bench correctness requires a separate harness run"
                if args.backend == "agentic"
                else "topology and protocol smoke only"
            ),
            "workload_coverage": {
                "load_valid_workflows": sum(
                    bool(item["load_valid"]) for item in results
                ),
                "workflow_intensity_valid": sum(
                    bool(item["workflow_intensity_valid"]) for item in results
                ),
                "subagent_intensity_valid": sum(
                    bool(item["subagent_intensity_valid"]) for item in results
                ),
                "dynamic_subagent_workflows": sum(
                    bool(item["subagent_trace_valid"]) for item in results
                ),
                "multi_turn_subagent_workflows": sum(
                    bool(item["subagent_trace_valid"]) for item in results
                ),
                "peer_reactivation_workflows": sum(
                    int(item["peer_reactivation_count"]) > 0 for item in results
                ),
                "guard_valid_workflows": sum(
                    bool(item["guard_valid"]) for item in results
                ),
                "runtime_valid_workflows": sum(
                    bool(item["runtime_valid"]) for item in results
                ),
                "native_protocol_valid_workflows": sum(
                    bool(item["native_protocol_valid"]) for item in results
                ),
                "clean_jct_workflows": sum(
                    bool(item["clean_jct_eligible"]) for item in results
                ),
                "forced_semantic_completions": sum(
                    int(
                        item["agent_control"][
                            "forced_semantic_completions"
                        ]
                    )
                    for item in results
                ),
                "guard_intervened_completions": sum(
                    int(
                        item["agent_control"][
                            "guard_intervened_completions"
                        ]
                    )
                    for item in results
                ),
                "protocol_repaired_completions": sum(
                    int(
                        item["agent_control"][
                            "protocol_repaired_completions"
                        ]
                    )
                    for item in results
                ),
                "protocol_normalized_completions": sum(
                    int(
                        item["agent_control"][
                            "protocol_normalized_completions"
                        ]
                    )
                    for item in results
                ),
                "protocol_repair_failures": sum(
                    int(item["agent_control"]["protocol_repair_failures"])
                    for item in results
                ),
                "duplicate_tool_calls_suppressed": sum(
                    int(
                        item["agent_control"][
                            "duplicate_tool_calls_suppressed"
                        ]
                    )
                    for item in results
                ),
            },
            "experiment_valid": experiment_valid,
            "results": results,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.replace(output)
    except BaseException:
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0 if manifest["experiment_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
