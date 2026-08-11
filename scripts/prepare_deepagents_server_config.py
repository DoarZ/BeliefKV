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
        "--disable-reactive-transfer",
        action="store_true",
        help=(
            "Disable BeliefKV reactive D2H/H2D planning while retaining runtime "
            "identity and telemetry observers. Intended for hardware calibration."
        ),
    )
    parser.add_argument(
        "--request-queue-timeout-seconds",
        type=float,
        default=1800.0,
        help="Server-side queue/admission timeout, separate from execution timeout.",
    )
    parser.add_argument(
        "--enable-observed-admission",
        action="store_true",
        help="Enable current-state active-KV admission windows.",
    )
    parser.add_argument(
        "--enable-online-joint",
        action="store_true",
        help=(
            "Apply validated observed JointPlan execution, admission, residency, "
            "and transfer dependencies online."
        ),
    )
    parser.add_argument(
        "--enable-joint-predictive",
        action="store_true",
        help=(
            "Deprecated compatibility switch. Predictions are attached as "
            "metadata only and never change the observed JointPlan."
        ),
    )
    parser.add_argument(
        "--enable-predictive-risk-shadow",
        action="store_true",
        help=(
            "Evaluate A0/PREPARE_HOST/PREFETCH_GPU with the P6 scenario-risk "
            "planner without dispatching predictive actions."
        ),
    )
    parser.add_argument(
        "--enable-predictive-joint-overlay",
        action="store_true",
        help=(
            "Allow non-destructive semantic prediction intents to join the "
            "observed JointPlan and be rematerialized at a scheduler safe point."
        ),
    )
    parser.add_argument(
        "--enable-predictive-prefetch-canary",
        action="store_true",
        help=(
            "Enable the bounded PREFETCH_GPU canary (one in flight and at most "
            "5 percent of the configured KV pool)."
        ),
    )
    parser.add_argument(
        "--predictive-prepare-canary-limit",
        type=int,
        default=0,
        help=(
            "Maximum PREPARE_HOST commands admitted during this server run; "
            "zero leaves the normal policy unlimited."
        ),
    )
    parser.add_argument(
        "--gpu-service-model",
        type=Path,
        default=None,
        help="Calibrated GPUServiceCurveModel used by P6 risk shadow.",
    )
    parser.add_argument(
        "--transfer-service-model",
        type=Path,
        default=None,
        help=(
            "Persistent PCIe/HiCache service artifact used to warm-start H2D/D2H "
            "latency before online telemetry has enough samples."
        ),
    )
    parser.add_argument(
        "--transfer-service-hardware-key",
        default=None,
        help="Optional exact hardware/model key required from the transfer artifact.",
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
        "--enable-frontier-retraction-shadow",
        action="store_true",
        help=(
            "Compare observed and FrontierBelief-annotated selective retraction "
            "without changing the online victim."
        ),
    )
    parser.add_argument(
        "--frontier-retraction-canary-limit",
        type=int,
        default=0,
        help=(
            "Maximum retraction transactions whose victim/replacement may be "
            "changed by frontier annotations; zero keeps the path shadow-only."
        ),
    )
    parser.add_argument(
        "--enable-restore-micro-gate",
        action="store_true",
        help=(
            "Enable the one-shot P5 restore correctness probe. This is a "
            "test hook, not a performance policy."
        ),
    )
    parser.add_argument(
        "--restore-micro-gate-id",
        default="p5g-restore-v1",
    )
    parser.add_argument(
        "--restore-micro-gate-victim-workflow-id",
        default="restore-micro-gate:victim",
    )
    parser.add_argument(
        "--restore-micro-gate-replacement-workflow-id",
        default="restore-micro-gate:replacement",
    )
    parser.add_argument(
        "--restore-micro-gate-min-private-mib",
        type=int,
        default=64,
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
        "--joint-workflow-active-window",
        type=int,
        default=12,
        help="Maximum number of fair workflows eligible for tickets per epoch.",
    )
    parser.add_argument(
        "--subagent-fanout-profile",
        choices=("natural", "parallel_analysis_2to3"),
        default="natural",
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
    parser.add_argument(
        "--predictor-model",
        type=Path,
        default=None,
        help=(
            "Load a P6 FrontierBeliefModel artifact into the controller as an "
            "online shadow predictor. Predictions are recorded but never used "
            "for admission, residency, or transfer decisions."
        ),
    )
    args = parser.parse_args()
    if args.enable_running_retraction and not args.enable_observed_admission:
        parser.error(
            "--enable-running-retraction requires --enable-observed-admission"
        )
    if args.frontier_retraction_canary_limit < 0:
        parser.error("--frontier-retraction-canary-limit must be non-negative")
    if (
        args.enable_frontier_retraction_shadow
        or args.frontier_retraction_canary_limit > 0
    ) and not (
        args.enable_running_retraction
        and args.enable_predictive_risk_shadow
        and args.predictor_model is not None
    ):
        parser.error(
            "frontier-aware retraction requires --enable-running-retraction, "
            "--enable-predictive-risk-shadow, and --predictor-model"
        )
    if args.enable_online_joint and args.disable_policy_shadow:
        parser.error("--enable-online-joint cannot be combined with --disable-policy-shadow")
    if args.enable_joint_predictive and not args.enable_online_joint:
        parser.error("--enable-joint-predictive requires --enable-online-joint")
    if args.enable_joint_predictive and args.predictor_model is None:
        parser.error("--enable-joint-predictive requires --predictor-model")
    if args.enable_predictive_risk_shadow and args.predictor_model is None:
        parser.error(
            "--enable-predictive-risk-shadow requires --predictor-model"
        )
    if args.enable_predictive_risk_shadow and args.gpu_service_model is None:
        parser.error(
            "--enable-predictive-risk-shadow requires --gpu-service-model"
        )
    if args.enable_predictive_risk_shadow and args.disable_policy_shadow:
        parser.error(
            "--enable-predictive-risk-shadow cannot use --disable-policy-shadow"
        )
    if args.enable_predictive_joint_overlay and not (
        args.enable_online_joint and args.enable_predictive_risk_shadow
    ):
        parser.error(
            "--enable-predictive-joint-overlay requires --enable-online-joint "
            "and --enable-predictive-risk-shadow"
        )
    if (
        args.enable_predictive_prefetch_canary
        and not args.enable_predictive_joint_overlay
    ):
        parser.error(
            "--enable-predictive-prefetch-canary requires "
            "--enable-predictive-joint-overlay"
        )
    if args.enable_restore_micro_gate and not (
        args.enable_online_joint
        and args.enable_observed_admission
        and args.enable_running_retraction
        and args.queue_service_observer
    ):
        parser.error(
            "--enable-restore-micro-gate requires --enable-online-joint, "
            "--enable-observed-admission, --enable-running-retraction, and "
            "--queue-service-observer"
        )
    if args.restore_micro_gate_min_private_mib <= 0:
        parser.error("--restore-micro-gate-min-private-mib must be positive")
    if args.joint_workflow_active_window <= 0:
        parser.error("--joint-workflow-active-window must be positive")
    if args.request_queue_timeout_seconds <= 0:
        parser.error("--request-queue-timeout-seconds must be positive")

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
        "request_queue_timeout_ms": args.request_queue_timeout_seconds * 1000.0,
        "kv_bytes_per_token": args.kv_bytes_per_token,
        "predictor_enabled": args.predictor_model is not None,
        "predictor_model_path": (
            str(args.predictor_model.expanduser().resolve())
            if args.predictor_model is not None
            else None
        ),
        "shadow_enabled": False,
        "prefetch_enabled": True,
        "reactive_transfer_enabled": not args.disable_reactive_transfer,
        "runtime_audit_path": str(server_dir / "runtime_audit.jsonl"),
        "runtime_audit_queue_capacity": 8192,
        "runtime_audit_debug_sample_rate": 0.05,
        "runtime_audit_max_debug_event_bytes": 16_384,
        "runtime_audit_flush_interval_s": 1.0,
        "runtime_summary_path": str(server_dir / "latest_runtime_summary.json"),
        "runtime_summary_interval_ms": 5_000.0,
        "shutdown_drain_timeout_ms": 5_000.0,
        "runtime_scheduler_pid_path": str(server_dir / "scheduler.pid.json"),
        "runtime_shutdown_ack_path": str(server_dir / "shutdown_ack.json"),
        "transfer_telemetry_path": str(server_dir / "transfer_telemetry.jsonl"),
        "runtime_event_socket_path": str(socket_path),
        "runtime_event_log_path": str(server_dir / "runtime_events.sglang.jsonl"),
        "runtime_event_max_lateness_ms": args.runtime_event_max_lateness_ms,
        "resource_telemetry_interval_ms": 50.0,
        "service_curve_window": 256,
        "service_curve_min_samples": 8,
        "transfer_service_model_path": (
            str(args.transfer_service_model.expanduser().resolve())
            if args.transfer_service_model is not None
            else None
        ),
        "transfer_service_hardware_key": args.transfer_service_hardware_key,
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
        "bundle_preview_audit_max_detailed_per_cycle": 8,
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
        "joint_policy_enabled": args.enable_online_joint,
        "joint_policy_shadow_mode": not args.disable_policy_shadow,
        "joint_observed_mode_enabled": not args.disable_policy_shadow,
        "joint_predictive_enabled": args.enable_joint_predictive,
        "predictive_risk_shadow_enabled": args.enable_predictive_risk_shadow,
        "predictive_joint_overlay_enabled": args.enable_predictive_joint_overlay,
        "predictive_prepare_host_enabled": True,
        "predictive_prepare_host_canary_limit": (
            args.predictive_prepare_canary_limit
        ),
        "predictive_prefetch_canary_enabled": (
            args.enable_predictive_prefetch_canary
        ),
        "predictive_prefetch_canary_max_inflight": 1,
        "predictive_prefetch_canary_max_hbm_ratio": 0.05,
        "predictive_prefetch_min_hbm_feasibility": 0.95,
        "predictive_commit_guard_ms": 25.0,
        "predictive_prefetch_desired_lead_ms": 100.0,
        "predictive_intent_max_age_ms": 60_000.0,
        "gpu_service_model_path": (
            str(args.gpu_service_model.expanduser().resolve())
            if args.gpu_service_model is not None
            else None
        ),
        "predictive_risk_particle_count": 128,
        "predictive_risk_top_k": 8,
        "predictive_risk_max_candidates": 8,
        "predictive_risk_min_calibration_coverage": 0.9,
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
        "frontier_aware_retraction_shadow_enabled": (
            args.enable_frontier_retraction_shadow
        ),
        "frontier_aware_retraction_canary_limit": (
            args.frontier_retraction_canary_limit
        ),
        "restore_obligation_max_active": 8,
        "restore_obligation_escalation_ms": 2000.0,
        "restore_obligation_max_blocked_ms": 30_000.0,
        "restore_lease_enabled": True,
        "restore_lease_max_active": 1,
        "restore_lease_max_bypass_admissions": 1,
        "restore_service_grace_decode_tokens": 32,
        "restore_micro_gate_enabled": args.enable_restore_micro_gate,
        "restore_micro_gate_id": args.restore_micro_gate_id,
        "restore_micro_gate_victim_workflow_id": (
            args.restore_micro_gate_victim_workflow_id
        ),
        "restore_micro_gate_replacement_workflow_id": (
            args.restore_micro_gate_replacement_workflow_id
        ),
        "restore_micro_gate_min_private_bytes": (
            args.restore_micro_gate_min_private_mib * 1024 * 1024
        ),
        "workload_subagent_fanout_profile": args.subagent_fanout_profile,
        "fairness_lag_budget_ms": 50.0,
        "residency_hysteresis_ms": 100.0,
        "joint_emergency_hbm_ratio": 0.98,
        "joint_workflow_active_window": args.joint_workflow_active_window,
        "max_joint_workflow_candidates": 8,
        "max_frontier_candidates_per_workflow": 4,
        "max_total_frontier_candidates": 16,
        "max_joint_package_evaluations": 8,
        "max_joint_plan_budget_ms": 20.0,
        "min_joint_plan_budget_ms": 0.25,
        "joint_trigger_budget_fraction": 0.5,
        "joint_physical_commit_budget_ms": 1.0,
        "max_joint_plan_age_ms": 100.0,
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
