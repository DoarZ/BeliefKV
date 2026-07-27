#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a no-prediction BeliefKV config for Deep Agents experiments."
    )
    parser.add_argument("--server-dir", type=Path, required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, default=98_304)
    parser.add_argument("--pcie-bandwidth-gbps", type=float, default=24.0)
    parser.add_argument("--runtime-event-max-lateness-ms", type=float, default=60_000.0)
    parser.add_argument("--queue-service-observer", action="store_true")
    parser.add_argument("--request-token-trace", action="store_true")
    parser.add_argument(
        "--enable-observed-admission",
        action="store_true",
        help="Enable current-state active-KV admission windows.",
    )
    parser.add_argument(
        "--enable-running-retraction",
        action="store_true",
        help="Enable P5B observed selective running-batch retraction.",
    )
    parser.add_argument(
        "--allow-running-retraction-recompute-drop",
        action="store_true",
        help="Allow P5 residency transactions to drop GPU-only KV for recompute.",
    )
    parser.add_argument(
        "--observed-admission-active-kv-high-watermark-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--observed-admission-min-active-requests",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--disable-policy-shadow",
        action="store_true",
        help="Disable both reference snapshots and P4 JointPlan shadow work.",
    )
    parser.add_argument(
        "--disable-snapshot-persistence",
        action="store_true",
        help="Run P4 JointPlan shadow without writing full PolicyInput snapshots.",
    )
    args = parser.parse_args()
    if args.enable_running_retraction and not args.enable_observed_admission:
        parser.error(
            "--enable-running-retraction requires --enable-observed-admission"
        )

    server_dir = args.server_dir.expanduser().resolve()
    config_path = server_dir / "beliefkv_config.json"
    if config_path.exists():
        raise FileExistsError(f"server config already exists: {config_path}")
    key = hashlib.sha256(str(server_dir).encode("utf-8")).hexdigest()[:12]
    socket_path = Path(f"/tmp/bkv-da-{key}.sock")
    if socket_path.exists():
        raise FileExistsError(f"runtime event socket already exists: {socket_path}")
    config = {
        "pcie_bandwidth_gbps": args.pcie_bandwidth_gbps,
        "transfer_overhead_ms": 0.08,
        "transfer_watchdog_factor": 20.0,
        "transfer_watchdog_floor_ms": 1000.0,
        "urgent_chunk_bytes": 268_435_456,
        "shadow_chunk_bytes": 67_108_864,
        "shadow_min_parked_ms": 25.0,
        "shadow_slowdown_budget": 0.02,
        "planning_interval_ms": 5.0,
        "admission_liveness_timeout_ms": 1000.0,
        "admission_force_progress_timeout_ms": 5000.0,
        "kv_bytes_per_token": args.kv_bytes_per_token,
        "predictor_enabled": False,
        "predictor_model_path": None,
        "shadow_enabled": False,
        "prefetch_enabled": True,
        "runtime_audit_path": str(server_dir / "runtime_audit.jsonl"),
        "transfer_telemetry_path": str(server_dir / "transfer_telemetry.jsonl"),
        "runtime_event_socket_path": str(socket_path),
        "runtime_event_log_path": str(server_dir / "runtime_events.sglang.jsonl"),
        "runtime_event_max_lateness_ms": args.runtime_event_max_lateness_ms,
        "resource_telemetry_interval_ms": 50.0,
        "service_curve_window": 256,
        "service_curve_min_samples": 8,
        "queue_service_observer_enabled": args.queue_service_observer,
        "queue_service_observer_include_runtime_batches": (
            args.queue_service_observer
        ),
        "queue_service_observer_max_samples": 65_536,
        "request_token_trace_enabled": args.request_token_trace,
        "request_token_trace_path": (
            str(server_dir / "request_tokens.jsonl.gz")
            if args.request_token_trace
            else None
        ),
        "transfer_retry_guard_enabled": True,
        "transfer_retry_max_same_snapshot_attempts": 1,
        "transfer_retry_unknown_base_ms": 10.0,
        "transfer_retry_unknown_max_ms": 1000.0,
        "transfer_retry_unknown_circuit_breaker_failures": 8,
        "reference_policy_shadow_enabled": not (
            args.disable_policy_shadow or args.disable_snapshot_persistence
        ),
        "reference_policy_snapshot_path": (
            None
            if args.disable_policy_shadow or args.disable_snapshot_persistence
            else str(server_dir / "policy_snapshots.jsonl.gz")
        ),
        "reference_policy_snapshot_min_interval_ms": 1000.0,
        "reference_policy_snapshot_persist_interval_ms": 10000.0,
        "reference_policy_snapshot_max_pending": 8,
        "reference_policy_hbm_bucket_bytes": 67_108_864,
        "reference_policy_trace_sensitivity": "timing_sensitive",
        "joint_policy_enabled": False,
        "joint_policy_shadow_mode": not args.disable_policy_shadow,
        "joint_observed_mode_enabled": not args.disable_policy_shadow,
        "observed_admission_scheduling_enabled": (
            args.enable_observed_admission
        ),
        "observed_admission_active_kv_high_watermark_ratio": (
            args.observed_admission_active_kv_high_watermark_ratio
        ),
        "observed_admission_min_active_requests": (
            args.observed_admission_min_active_requests
        ),
        "running_batch_retraction_enabled": args.enable_running_retraction,
        "running_batch_retraction_min_stall_ms": 100.0,
        "running_batch_retraction_min_reclaim_bytes": 67_108_864,
        "running_batch_retraction_cooldown_ms": 1000.0,
        "running_batch_retraction_decision_interval_ms": 50.0,
        "running_batch_retraction_max_per_request": 3,
        "running_batch_retraction_transaction_timeout_ms": 5000.0,
        "running_batch_retraction_allow_recompute_drop": (
            args.allow_running_retraction_recompute_drop
        ),
        "fairness_lag_budget_ms": 50.0,
        "residency_hysteresis_ms": 100.0,
        "max_joint_workflow_candidates": 8,
        "max_frontier_candidates_per_workflow": 4,
        "max_total_frontier_candidates": 16,
        "max_joint_package_evaluations": 8,
        "max_joint_plan_budget_ms": 1.0,
        "max_joint_plan_age_ms": 750.0,
        "joint_transition_settling_timeout_ms": 250.0,
        "joint_shadow_detailed_audit_interval_ms": 1000.0,
    }
    write_json(config_path, config)
    print(
        json.dumps(
            {"config_path": str(config_path), "control_socket": str(socket_path)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
