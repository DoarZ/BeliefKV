#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.agent_protocol import LoopGuardPolicy
from beliefkv.experiments.deepagents_swebench import (
    DeepAgentsExperimentConfig,
    run_experiment,
    write_json,
)
from beliefkv.experiments.harness_preflight import preflight_command_for_policy
from beliefkv.experiments.p6_collection import load_collection_batch
from beliefkv.runtime.context_lifecycle import ContextLifecyclePolicy
from beliefkv.runtime.langchain_tool_safety import ToolObservationBudgetPolicy


DEFAULT_PAUSE_FILE = Path("/tmp/beliefkv-experiments.paused")
DEFAULT_HARNESS_PROFILES = REPOSITORY_ROOT / "configs/p6/harness_profiles_v1.json"


def _actual_kv_pool_tokens(base_url: str, *, timeout_s: float = 10.0) -> int:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    with urllib.request.urlopen(
        f"{root}/get_server_info",
        timeout=timeout_s,
    ) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("SGLang /get_server_info did not return a JSON object")
    try:
        actual = int(payload["max_total_num_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "SGLang /get_server_info omitted a valid max_total_num_tokens"
        ) from error
    if actual <= 0:
        raise RuntimeError("SGLang reported a non-positive KV pool capacity")
    return actual


def _materialize_runtime_workload_manifest(
    *,
    source_path: Path,
    destination: Path,
    profile_path: Path,
    selected_instance_ids: list[str] | None,
) -> tuple[Path, list[dict[str, object]], int]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    if profiles.get("schema_version") != 1:
        raise ValueError("unsupported P6 harness profile schema")
    profile_by_instance = profiles.get("instances", {})
    selected = set(selected_instance_ids or ())
    workloads = [
        dict(item)
        for item in source.get("workloads", [])
        if not selected or str(item.get("instance_id")) in selected
    ]
    found = {str(item.get("instance_id")) for item in workloads}
    if selected:
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"instances are absent from source manifest: {missing}")
    applied: list[dict[str, object]] = []
    for item in workloads:
        instance_id = str(item.get("instance_id"))
        repo = str(item.get("repo"))
        item.pop("preflight_command", None)
        profile = profile_by_instance.get(instance_id)
        if not isinstance(profile, dict):
            continue
        if str(profile.get("repo")) != repo:
            raise ValueError(f"harness profile repo mismatch for {instance_id}")
        source_image = str(item.get("docker_image"))
        if source_image != str(profile.get("source_image")):
            raise ValueError(f"harness profile source image mismatch for {instance_id}")
        runtime_image = str(profile["runtime_image"])
        item["docker_image"] = runtime_image
        policy = profile.get("preflight_policy")
        preflight = preflight_command_for_policy(
            str(policy) if policy is not None else None
        )
        if preflight is not None:
            item["preflight_command"] = preflight
        applied.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "source_image": source_image,
                "runtime_image": runtime_image,
                "preflight_policy": profile.get("preflight_policy"),
            }
        )
    runtime = {
        **source,
        "workloads": workloads,
        "source_workload_manifest": str(source_path),
        "harness_profile_id": profiles.get("profile_id"),
        "harness_profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    }
    write_json(destination, runtime)
    return destination, applied, len(workloads)


def _runtime_source_fingerprint() -> dict[str, object]:
    roots = (
        REPOSITORY_ROOT / "beliefkv/control",
        REPOSITORY_ROOT / "beliefkv/core",
        REPOSITORY_ROOT / "beliefkv/metrics",
        REPOSITORY_ROOT / "beliefkv/policy",
        REPOSITORY_ROOT / "beliefkv/runtime",
    )
    files = [
        path
        for root in roots
        for path in root.rglob("*.py")
        if path.is_file()
    ]
    files.extend(
        path
        for path in (
            REPOSITORY_ROOT / "beliefkv/experiments/agent_protocol.py",
            REPOSITORY_ROOT / "beliefkv/experiments/deepagents_swebench.py",
            REPOSITORY_ROOT / "beliefkv/experiments/harness_preflight.py",
            REPOSITORY_ROOT / "beliefkv/experiments/langgraph_peer_workflow.py",
            REPOSITORY_ROOT / "scripts/launch_deepagents_swebench_server.sh",
            REPOSITORY_ROOT / "scripts/prepare_deepagents_server_config.py",
            REPOSITORY_ROOT / "scripts/run_deepagents_swebench.py",
            REPOSITORY_ROOT / "scripts/run_p6_collection_batch.py",
            REPOSITORY_ROOT / "patches/sglang-0.5.2rc1-beliefkv.patch",
        )
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content).digest())
    return {
        "algorithm": "sha256(path\\0sha256(content))",
        "digest": digest.hexdigest(),
        "file_count": len(set(files)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen P6 agent-semantics collection batch."
    )
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help=(
            "run only this instance from the frozen batch; repeat for a targeted "
            "harness recovery collection"
        ),
    )
    parser.add_argument("--allow-calibration", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--predictor-shadow-enabled",
        action="store_true",
        help=(
            "Record that the P6 frontier predictor ran in shadow mode on the "
            "serving side. The dataset exporter will mark this run formal "
            "ineligible, which is the correct provenance for a shadow run."
        ),
    )
    parser.add_argument(
        "--predictive-risk-shadow-enabled",
        action="store_true",
        help=(
            "Record that the serving-side P6 scenario-risk observer evaluated "
            "read-only would-actions; no predictive command is dispatched."
        ),
    )
    parser.add_argument(
        "--predictive-joint-enabled",
        action="store_true",
        help=(
            "Deprecated provenance flag for historical runs. The current "
            "serving path never grants this flag predictive action authority."
        ),
    )
    parser.add_argument(
        "--predictive-joint-overlay-enabled",
        action="store_true",
        help=(
            "Record that the serving-side semantic predictive overlay may "
            "dispatch PREPARE_HOST through JointPlan."
        ),
    )
    parser.add_argument(
        "--predictive-prefetch-canary-enabled",
        action="store_true",
        help=(
            "Record that the bounded serving-side PREFETCH_GPU canary is enabled."
        ),
    )
    parser.add_argument(
        "--frontier-retraction-shadow-enabled",
        action="store_true",
        help="Record observed/frontier selective-retraction comparisons.",
    )
    parser.add_argument(
        "--frontier-retraction-canary-limit",
        type=int,
        default=0,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="Qwen3-Coder-30B-A3B-Instruct-FP8")
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--server-audit", type=Path, required=True)
    parser.add_argument("--server-events", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--pool-tokens",
        type=int,
        default=163_840,
        help=(
            "minimum actual SGLang max_total_num_tokens required by this run; "
            "the server-reported value is used for pressure accounting"
        ),
    )
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--context-window-tokens", type=int, default=32_768)
    parser.add_argument("--context-keep-tokens", type=int, default=8_192)
    parser.add_argument("--summary-output-tokens", type=int, default=2_048)
    parser.add_argument("--tool-observation-turn-chars", type=int, default=65_536)
    parser.add_argument("--tool-observation-result-chars", type=int, default=16_384)
    parser.add_argument("--recursion-limit", type=int, default=512)
    parser.add_argument(
        "--subagent-fanout-profile",
        choices=("natural", "parallel_analysis_2to3"),
        default="natural",
    )
    parser.add_argument("--workflow-arrival-interval-ms", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--sandbox-command-timeout", type=int, default=600)
    parser.add_argument("--runtime-event-ack-timeout", type=float, default=10.0)
    parser.add_argument("--runtime-event-ack-retries", type=int, default=3)
    parser.add_argument(
        "--harness-profiles",
        type=Path,
        default=DEFAULT_HARNESS_PROFILES,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pause_file = Path(
        os.environ.get("BELIEFKV_EXPERIMENT_PAUSE_FILE", str(DEFAULT_PAUSE_FILE))
    ).expanduser()
    if pause_file.exists():
        print(f"BeliefKV experiments are paused: {pause_file}", file=sys.stderr)
        return 75
    if (
        args.predictive_prefetch_canary_enabled
        and not args.predictive_joint_overlay_enabled
    ):
        raise ValueError(
            "predictive prefetch canary provenance requires predictive overlay"
        )
    if args.frontier_retraction_canary_limit < 0:
        raise ValueError("frontier retraction canary limit must be non-negative")
    batch = load_collection_batch(
        args.collection_plan,
        args.batch_id,
        allow_calibration=args.allow_calibration,
        allow_test=args.allow_test,
    )
    if args.pool_tokens <= 0:
        raise ValueError("--pool-tokens must be positive")
    actual_pool_tokens = _actual_kv_pool_tokens(args.base_url)
    if actual_pool_tokens < args.pool_tokens:
        raise RuntimeError(
            "SGLang actual KV pool is below the collection requirement: "
            f"actual={actual_pool_tokens}, required={args.pool_tokens}"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPOSITORY_ROOT
        / "experiments/raw/p6_agent_semantics_v1"
        / batch.batch_id
        / timestamp
        / "workloads"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_instance_ids: list[str] | None = None
    if args.instance_id:
        selected_instance_ids = list(dict.fromkeys(args.instance_id))
    workload_manifest, harness_profiles, workflow_count = (
        _materialize_runtime_workload_manifest(
            source_path=batch.workload_manifest,
            destination=output.parent / "runtime_workload_manifest.json",
            profile_path=args.harness_profiles.expanduser().resolve(),
            selected_instance_ids=selected_instance_ids,
        )
    )
    concurrency = min(batch.concurrency, workflow_count)
    source_fingerprint = _runtime_source_fingerprint()
    collection_contract = {
        "schema_version": 1,
        "plan_id": batch.plan_id,
        "batch_id": batch.batch_id,
        "split": batch.split,
        "workload_manifest": str(workload_manifest),
        "source_workload_manifest": str(batch.workload_manifest),
        "harness_profile_manifest": str(args.harness_profiles.resolve()),
        "harness_profile_manifest_sha256": hashlib.sha256(
            args.harness_profiles.read_bytes()
        ).hexdigest(),
        "applied_harness_profiles": harness_profiles,
        "selected_instance_ids": selected_instance_ids,
        "workflow_count": workflow_count,
        "concurrency": concurrency,
        "workflow_arrival_interval_ms": args.workflow_arrival_interval_ms,
        "required_minimum_pool_tokens": args.pool_tokens,
        "actual_pool_tokens": actual_pool_tokens,
        "predictor_enabled": (
            args.predictor_shadow_enabled
            or args.predictive_risk_shadow_enabled
            or args.predictive_joint_enabled
            or args.predictive_joint_overlay_enabled
        ),
        "predictive_actions_enabled": args.predictive_joint_overlay_enabled,
        "predictive_risk_shadow_enabled": args.predictive_risk_shadow_enabled,
        "predictive_joint_overlay_enabled": (
            args.predictive_joint_overlay_enabled
        ),
        "predictive_prefetch_canary_enabled": (
            args.predictive_prefetch_canary_enabled
        ),
        "frontier_retraction_shadow_enabled": (
            args.frontier_retraction_shadow_enabled
        ),
        "frontier_retraction_canary_limit": (
            args.frontier_retraction_canary_limit
        ),
        "predictive_transfer_model": "extent_count_aware",
        "joint_predictive_enabled": args.predictive_joint_enabled,
        "legacy_predictive_flag_requested": args.predictive_joint_enabled,
        "runtime_policy": (
            "p5_observed_plus_p6_predictive_overlay"
            if args.predictive_joint_overlay_enabled
            else "frozen_p5_observed"
        ),
        "subagent_fanout_profile": args.subagent_fanout_profile,
        "request_timeout_s": args.request_timeout,
        "completion_semantics": "model_terminal_no_harness_llm_repair",
        "completion_gate_enabled": False,
        "completion_repair_attempts": 0,
        "runtime_event_ack_timeout_s": args.runtime_event_ack_timeout,
        "runtime_event_ack_retries": args.runtime_event_ack_retries,
        "context_lifecycle": {
            "window_tokens": args.context_window_tokens,
            "keep_tokens": args.context_keep_tokens,
            "intermediate_output_tokens": args.max_completion_tokens,
            "summary_output_tokens": args.summary_output_tokens,
        },
        "graph_step_safety": {
            "semantic_patterns": "telemetry_only",
            "soft_budget_mode": "telemetry_only",
            "soft_budget": LoopGuardPolicy().graph_step_soft_budget,
            "hard_limit": LoopGuardPolicy().graph_step_hard_limit,
            "reserve": LoopGuardPolicy().graph_step_reserve,
            "hard_limit_mode": "safety_finalization",
        },
        "tool_observation_budget": {
            "total_chars_per_turn": args.tool_observation_turn_chars,
            "max_chars_per_result": args.tool_observation_result_chars,
        },
        "runtime_source_fingerprint_start": source_fingerprint,
        "training_eligible": None,
    }
    write_json(
        output.parent / f"{output.name}.p6_collection_contract.json",
        collection_contract,
    )
    config = DeepAgentsExperimentConfig(
        mode="autonomous",
        base_url=args.base_url,
        model=args.model,
        output_dir=output,
        workload_manifest=workload_manifest,
        docker_image="unused:per-workload-image-required",
        control_socket=args.control_socket,
        server_audit_path=args.server_audit,
        server_event_path=args.server_events,
        server_log_path=args.server_log,
        max_workflows=workflow_count,
        concurrency=concurrency,
        workflow_arrival_interval_ms=args.workflow_arrival_interval_ms,
        gpu_index=args.gpu,
        pool_tokens=actual_pool_tokens,
        max_completion_tokens=args.max_completion_tokens,
        subagent_fanout_profile=args.subagent_fanout_profile,
        recursion_limit=args.recursion_limit,
        request_timeout_s=args.request_timeout,
        sandbox_command_timeout_s=args.sandbox_command_timeout,
        sandbox_preflight_command=batch.preflight_command,
        completion_gate_enabled=False,
        completion_repair_attempts=0,
        runtime_event_ack_timeout_s=args.runtime_event_ack_timeout,
        runtime_event_ack_retries=args.runtime_event_ack_retries,
        context_lifecycle=ContextLifecyclePolicy(
            window_tokens=args.context_window_tokens,
            keep_tokens=args.context_keep_tokens,
            intermediate_output_tokens=args.max_completion_tokens,
            summary_output_tokens=args.summary_output_tokens,
        ),
        loop_guard=LoopGuardPolicy(),
        tool_observation_budget=ToolObservationBudgetPolicy(
            total_chars_per_turn=args.tool_observation_turn_chars,
            max_chars_per_result=args.tool_observation_result_chars,
        ),
    )
    summary = run_experiment(config)
    final_fingerprint = _runtime_source_fingerprint()
    source_stable = final_fingerprint == source_fingerprint
    system_eligible = (
        summary["system_jct_eligible_workflows"] == workflow_count
    )
    final_contract = {
        **collection_contract,
        "runtime_source_fingerprint_end": final_fingerprint,
        "runtime_source_stable": source_stable,
        "training_eligible": system_eligible and source_stable,
        "ineligibility_reasons": [
            reason
            for condition, reason in (
                (not system_eligible, "system_jct_gate_failed"),
                (not source_stable, "runtime_source_changed_during_collection"),
            )
            if condition
        ],
    }
    summary["p6_collection"] = final_contract
    write_json(output / "p6_collection_contract.json", final_contract)
    write_json(output / "p6_collection_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if final_contract["training_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
