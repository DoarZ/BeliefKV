from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class BeliefKVConfig:
    hbm_capacity_bytes: int = 24 * (1 << 30)
    host_capacity_bytes: int = 96_000_000_000
    reserve_hbm_bytes: int = 1 << 30
    pcie_bandwidth_gbps: float = 24.0
    transfer_overhead_ms: float = 0.08
    transfer_watchdog_factor: float = 20.0
    transfer_watchdog_floor_ms: float = 1000.0
    urgent_chunk_bytes: int = 256 * 1024 * 1024
    shadow_chunk_bytes: int = 64 * 1024 * 1024
    shadow_min_parked_ms: float = 25.0
    shadow_slowdown_budget: float = 0.02
    planning_interval_ms: float = 5.0
    admission_liveness_timeout_ms: float = 1000.0
    admission_force_progress_timeout_ms: float = 5000.0
    request_queue_timeout_ms: float = 1_800_000.0
    kv_bytes_per_token: int = 57344
    predictor_enabled: bool = True
    predictor_model_path: str | None = None
    shadow_enabled: bool = True
    prefetch_enabled: bool = True
    reactive_transfer_enabled: bool = True
    runtime_audit_path: str | None = None
    runtime_audit_queue_capacity: int = 8192
    runtime_audit_debug_sample_rate: float = 0.05
    runtime_audit_max_debug_event_bytes: int = 16 * 1024
    runtime_audit_flush_interval_s: float = 1.0
    runtime_summary_path: str | None = None
    runtime_summary_interval_ms: float = 5000.0
    shutdown_drain_timeout_ms: float = 5000.0
    runtime_scheduler_pid_path: str | None = None
    runtime_shutdown_ack_path: str | None = None
    transfer_telemetry_path: str | None = None
    runtime_event_socket_path: str | None = None
    runtime_event_log_path: str | None = None
    runtime_event_max_lateness_ms: float = 5000.0
    resource_telemetry_interval_ms: float = 50.0
    service_curve_window: int = 256
    service_curve_min_samples: int = 8
    queue_service_observer_enabled: bool = False
    queue_service_observer_include_runtime_batches: bool = False
    queue_service_observer_max_samples: int = 65_536
    request_token_trace_enabled: bool = False
    request_token_trace_path: str | None = None
    transfer_retry_guard_enabled: bool = True
    transfer_retry_max_same_snapshot_attempts: int = 1
    transfer_retry_unknown_base_ms: float = 10.0
    transfer_retry_unknown_max_ms: float = 1000.0
    transfer_retry_unknown_circuit_breaker_failures: int = 8
    bundle_preview_audit_max_detailed_per_cycle: int = 8
    reference_policy_shadow_enabled: bool = True
    reference_policy_snapshot_path: str | None = None
    reference_policy_snapshot_min_interval_ms: float = 1000.0
    reference_policy_snapshot_persist_interval_ms: float = 10_000.0
    reference_policy_snapshot_max_pending: int = 8
    reference_policy_hbm_bucket_bytes: int = 64 * 1024 * 1024
    reference_policy_trace_sensitivity: str = "timing_sensitive"
    joint_policy_enabled: bool = False
    joint_policy_shadow_mode: bool = True
    joint_observed_mode_enabled: bool = True
    joint_predictive_enabled: bool = False
    observed_admission_scheduling_enabled: bool = False
    observed_admission_active_kv_high_watermark_ratio: float = 0.8
    observed_admission_min_active_requests: int = 1
    running_batch_retraction_enabled: bool = False
    running_batch_retraction_min_stall_ms: float = 100.0
    running_batch_retraction_min_reclaim_bytes: int = 64 * 1024 * 1024
    running_batch_retraction_cooldown_ms: float = 1000.0
    running_batch_retraction_decision_interval_ms: float = 50.0
    running_batch_retraction_max_per_request: int = 3
    running_batch_retraction_transaction_timeout_ms: float = 5000.0
    running_batch_retraction_allow_recompute_drop: bool = False
    restore_obligation_max_active: int = 8
    restore_obligation_escalation_ms: float = 2000.0
    restore_obligation_max_blocked_ms: float = 30_000.0
    restore_lease_enabled: bool = True
    restore_lease_max_active: int = 1
    restore_lease_max_bypass_admissions: int = 1
    restore_service_grace_decode_tokens: int = 32
    restore_micro_gate_enabled: bool = False
    restore_micro_gate_id: str = "p5g-restore-v1"
    restore_micro_gate_victim_workflow_id: str = "restore-micro-gate:victim"
    restore_micro_gate_replacement_workflow_id: str = (
        "restore-micro-gate:replacement"
    )
    restore_micro_gate_min_private_bytes: int = 64 * 1024 * 1024
    fairness_lag_budget_ms: float = 50.0
    residency_hysteresis_ms: float = 100.0
    joint_emergency_hbm_ratio: float = 0.98
    joint_workflow_active_window: int = 12
    max_joint_workflow_candidates: int = 8
    max_frontier_candidates_per_workflow: int = 4
    max_total_frontier_candidates: int = 16
    max_joint_package_evaluations: int = 8
    max_joint_plan_budget_ms: float = 20.0
    min_joint_plan_budget_ms: float = 0.25
    joint_trigger_budget_fraction: float = 0.5
    joint_physical_commit_budget_ms: float = 1.0
    max_joint_plan_age_ms: float = 100.0
    joint_transition_settling_timeout_ms: float = 250.0
    joint_shadow_detailed_audit_interval_ms: float = 1000.0

    def __post_init__(self) -> None:
        if self.hbm_capacity_bytes <= 0 or self.host_capacity_bytes <= 0:
            raise ValueError("HBM and host capacities must be positive")
        if not 0 <= self.reserve_hbm_bytes < self.hbm_capacity_bytes:
            raise ValueError("reserve_hbm_bytes must be within HBM capacity")
        if self.pcie_bandwidth_gbps <= 0:
            raise ValueError("pcie_bandwidth_gbps must be positive")
        if self.transfer_overhead_ms < 0:
            raise ValueError("transfer_overhead_ms must be non-negative")
        if (
            not math.isfinite(self.transfer_watchdog_factor)
            or self.transfer_watchdog_factor <= 0
        ):
            raise ValueError("transfer_watchdog_factor must be positive")
        if (
            not math.isfinite(self.transfer_watchdog_floor_ms)
            or self.transfer_watchdog_floor_ms <= 0
        ):
            raise ValueError("transfer_watchdog_floor_ms must be positive")
        if min(self.urgent_chunk_bytes, self.shadow_chunk_bytes) <= 0:
            raise ValueError("transfer chunks must be positive")
        if self.shadow_min_parked_ms < 0:
            raise ValueError("shadow_min_parked_ms must be non-negative")
        if not 0 <= self.shadow_slowdown_budget <= 1:
            raise ValueError("shadow_slowdown_budget must be in [0, 1]")
        if self.planning_interval_ms <= 0:
            raise ValueError("planning_interval_ms must be positive")
        if (
            not math.isfinite(self.admission_liveness_timeout_ms)
            or self.admission_liveness_timeout_ms < 0
        ):
            raise ValueError(
                "admission_liveness_timeout_ms must be finite and non-negative"
            )
        if (
            not math.isfinite(self.admission_force_progress_timeout_ms)
            or self.admission_force_progress_timeout_ms
            < self.admission_liveness_timeout_ms
        ):
            raise ValueError(
                "admission_force_progress_timeout_ms must be finite and no smaller "
                "than admission_liveness_timeout_ms"
            )
        if self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")
        if (
            not math.isfinite(self.runtime_event_max_lateness_ms)
            or self.runtime_event_max_lateness_ms < 0
        ):
            raise ValueError("runtime_event_max_lateness_ms must be finite and non-negative")
        if (
            not math.isfinite(self.resource_telemetry_interval_ms)
            or self.resource_telemetry_interval_ms <= 0
        ):
            raise ValueError("resource_telemetry_interval_ms must be positive")
        if self.service_curve_window <= 0:
            raise ValueError("service_curve_window must be positive")
        if not 1 <= self.service_curve_min_samples <= self.service_curve_window:
            raise ValueError(
                "service_curve_min_samples must be within the service curve window"
            )
        if self.queue_service_observer_max_samples <= 0:
            raise ValueError("queue_service_observer_max_samples must be positive")
        if self.request_token_trace_path is not None and not isinstance(
            self.request_token_trace_path, str
        ):
            raise ValueError("request_token_trace_path must be a string or null")
        if self.transfer_retry_max_same_snapshot_attempts <= 0:
            raise ValueError(
                "transfer_retry_max_same_snapshot_attempts must be positive"
            )
        if (
            not math.isfinite(self.transfer_retry_unknown_base_ms)
            or self.transfer_retry_unknown_base_ms <= 0
        ):
            raise ValueError("transfer_retry_unknown_base_ms must be positive")
        if (
            not math.isfinite(self.transfer_retry_unknown_max_ms)
            or self.transfer_retry_unknown_max_ms
            < self.transfer_retry_unknown_base_ms
        ):
            raise ValueError(
                "transfer_retry_unknown_max_ms must be no smaller than the base"
            )
        if self.transfer_retry_unknown_circuit_breaker_failures <= 0:
            raise ValueError(
                "transfer_retry_unknown_circuit_breaker_failures must be positive"
            )
        if self.bundle_preview_audit_max_detailed_per_cycle <= 0:
            raise ValueError(
                "bundle_preview_audit_max_detailed_per_cycle must be positive"
            )
        if (
            not math.isfinite(self.reference_policy_snapshot_min_interval_ms)
            or self.reference_policy_snapshot_min_interval_ms < 0
        ):
            raise ValueError(
                "reference_policy_snapshot_min_interval_ms must be non-negative"
            )
        if (
            not math.isfinite(self.reference_policy_snapshot_persist_interval_ms)
            or self.reference_policy_snapshot_persist_interval_ms < 0
        ):
            raise ValueError(
                "reference_policy_snapshot_persist_interval_ms must be non-negative"
            )
        if self.reference_policy_snapshot_max_pending <= 0:
            raise ValueError("reference_policy_snapshot_max_pending must be positive")
        if self.reference_policy_hbm_bucket_bytes <= 0:
            raise ValueError("reference_policy_hbm_bucket_bytes must be positive")
        if self.reference_policy_trace_sensitivity not in {
            "schedule_invariant",
            "timing_sensitive",
            "semantic_race_sensitive",
        }:
            raise ValueError("unsupported reference policy trace sensitivity")
        active_kv_high_watermark = float(
            self.observed_admission_active_kv_high_watermark_ratio
        )
        if (
            not math.isfinite(active_kv_high_watermark)
            or not 0 < active_kv_high_watermark <= 1
        ):
            raise ValueError(
                "observed_admission_active_kv_high_watermark_ratio must be "
                "in (0, 1]"
            )
        if self.observed_admission_min_active_requests < 0:
            raise ValueError(
                "observed_admission_min_active_requests must be non-negative"
            )
        if not 0 < self.joint_emergency_hbm_ratio <= 1:
            raise ValueError("joint_emergency_hbm_ratio must be in (0, 1]")
        if self.joint_workflow_active_window <= 0:
            raise ValueError("joint_workflow_active_window must be positive")
        if (
            self.running_batch_retraction_enabled
            and not self.observed_admission_scheduling_enabled
        ):
            raise ValueError(
                "running batch retraction requires observed admission scheduling"
            )
        for field_name in (
            "request_queue_timeout_ms",
            "running_batch_retraction_min_stall_ms",
            "running_batch_retraction_cooldown_ms",
            "running_batch_retraction_decision_interval_ms",
            "running_batch_retraction_transaction_timeout_ms",
            "restore_obligation_escalation_ms",
            "restore_obligation_max_blocked_ms",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.request_queue_timeout_ms == 0:
            raise ValueError("request_queue_timeout_ms must be positive")
        if self.running_batch_retraction_decision_interval_ms == 0:
            raise ValueError("retraction decision interval must be positive")
        if self.running_batch_retraction_min_stall_ms == 0:
            raise ValueError("minimum retraction stall must be positive")
        if self.running_batch_retraction_transaction_timeout_ms == 0:
            raise ValueError("retraction transaction timeout must be positive")
        if self.running_batch_retraction_min_reclaim_bytes <= 0:
            raise ValueError("minimum retraction reclaim bytes must be positive")
        if self.running_batch_retraction_max_per_request <= 0:
            raise ValueError("maximum retractions per request must be positive")
        if self.restore_obligation_max_active <= 0:
            raise ValueError("maximum active restore obligations must be positive")
        if self.restore_lease_max_active <= 0:
            raise ValueError("maximum active restore leases must be positive")
        if self.restore_lease_max_bypass_admissions < 0:
            raise ValueError("restore lease bypass admissions must be non-negative")
        if self.restore_service_grace_decode_tokens < 0:
            raise ValueError(
                "restore service grace decode tokens must be non-negative"
            )
        if self.restore_micro_gate_min_private_bytes <= 0:
            raise ValueError("restore micro-gate private bytes must be positive")
        for field_name in (
            "restore_micro_gate_id",
            "restore_micro_gate_victim_workflow_id",
            "restore_micro_gate_replacement_workflow_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            self.restore_micro_gate_victim_workflow_id
            == self.restore_micro_gate_replacement_workflow_id
        ):
            raise ValueError("restore micro-gate workflows must be distinct")
        if self.restore_micro_gate_enabled and not (
            self.joint_policy_enabled
            and self.observed_admission_scheduling_enabled
            and self.running_batch_retraction_enabled
            and self.queue_service_observer_enabled
            and self.queue_service_observer_include_runtime_batches
        ):
            raise ValueError(
                "restore micro-gate requires online JointPlan, observed admission, "
                "running retraction, and runtime GPU service observation"
            )
        if self.restore_obligation_escalation_ms == 0:
            raise ValueError("restore obligation escalation must be positive")
        if (
            self.restore_obligation_max_blocked_ms
            < self.restore_obligation_escalation_ms
        ):
            raise ValueError(
                "maximum restore blocking time cannot precede escalation"
            )
        for field_name in (
            "fairness_lag_budget_ms",
            "residency_hysteresis_ms",
            "max_joint_plan_budget_ms",
            "min_joint_plan_budget_ms",
            "joint_physical_commit_budget_ms",
            "max_joint_plan_age_ms",
            "joint_transition_settling_timeout_ms",
            "joint_shadow_detailed_audit_interval_ms",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.max_joint_plan_budget_ms == 0 or self.max_joint_plan_age_ms == 0:
            raise ValueError("joint plan budget and age must be positive")
        if self.min_joint_plan_budget_ms > self.max_joint_plan_budget_ms:
            raise ValueError("minimum JointPlan budget cannot exceed maximum")
        if not 0 < self.joint_trigger_budget_fraction <= 1:
            raise ValueError("joint_trigger_budget_fraction must be in (0, 1]")
        for field_name in (
            "max_joint_workflow_candidates",
            "max_frontier_candidates_per_workflow",
            "max_total_frontier_candidates",
            "max_joint_package_evaluations",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.predictor_model_path is not None and not isinstance(
            self.predictor_model_path, str
        ):
            raise ValueError("predictor_model_path must be a string or null")
        if self.runtime_audit_path is not None and not isinstance(
            self.runtime_audit_path, str
        ):
            raise ValueError("runtime_audit_path must be a string or null")
        if self.runtime_audit_queue_capacity <= 0:
            raise ValueError("runtime_audit_queue_capacity must be positive")
        if not 0 < self.runtime_audit_debug_sample_rate <= 1:
            raise ValueError(
                "runtime_audit_debug_sample_rate must be in (0, 1]"
            )
        if self.runtime_audit_max_debug_event_bytes <= 0:
            raise ValueError(
                "runtime_audit_max_debug_event_bytes must be positive"
            )
        if (
            not math.isfinite(self.runtime_audit_flush_interval_s)
            or self.runtime_audit_flush_interval_s <= 0
        ):
            raise ValueError("runtime_audit_flush_interval_s must be positive")
        if self.runtime_summary_path is not None and not isinstance(
            self.runtime_summary_path, str
        ):
            raise ValueError("runtime_summary_path must be a string or null")
        if (
            not math.isfinite(self.runtime_summary_interval_ms)
            or self.runtime_summary_interval_ms <= 0
        ):
            raise ValueError("runtime_summary_interval_ms must be positive")
        if (
            not math.isfinite(self.shutdown_drain_timeout_ms)
            or self.shutdown_drain_timeout_ms < 0
        ):
            raise ValueError("shutdown_drain_timeout_ms must be non-negative")
        for field_name in (
            "runtime_scheduler_pid_path",
            "runtime_shutdown_ack_path",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or null")
        if self.transfer_telemetry_path is not None and not isinstance(
            self.transfer_telemetry_path, str
        ):
            raise ValueError("transfer_telemetry_path must be a string or null")
        if self.runtime_event_socket_path is not None and not isinstance(
            self.runtime_event_socket_path, str
        ):
            raise ValueError("runtime_event_socket_path must be a string or null")
        if self.runtime_event_log_path is not None and not isinstance(
            self.runtime_event_log_path, str
        ):
            raise ValueError("runtime_event_log_path must be a string or null")
        if self.reference_policy_snapshot_path is not None and not isinstance(
            self.reference_policy_snapshot_path, str
        ):
            raise ValueError(
                "reference_policy_snapshot_path must be a string or null"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BeliefKVConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown BeliefKV config fields: {sorted(unknown)}")
        return cls(**dict(raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
