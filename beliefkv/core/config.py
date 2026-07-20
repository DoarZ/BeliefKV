from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class BeliefKVConfig:
    hbm_capacity_bytes: int = 24 * (1 << 30)
    host_capacity_bytes: int = 64 * (1 << 30)
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
    kv_bytes_per_token: int = 57344
    predictor_enabled: bool = True
    predictor_model_path: str | None = None
    shadow_enabled: bool = True
    prefetch_enabled: bool = True
    runtime_audit_path: str | None = None
    transfer_telemetry_path: str | None = None
    runtime_event_socket_path: str | None = None
    runtime_event_log_path: str | None = None
    runtime_event_max_lateness_ms: float = 5000.0
    resource_telemetry_interval_ms: float = 50.0
    service_curve_window: int = 256
    service_curve_min_samples: int = 8
    transfer_retry_guard_enabled: bool = True
    transfer_retry_max_same_snapshot_attempts: int = 1
    transfer_retry_unknown_base_ms: float = 10.0
    transfer_retry_unknown_max_ms: float = 1000.0
    transfer_retry_unknown_circuit_breaker_failures: int = 8

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
        if self.predictor_model_path is not None and not isinstance(
            self.predictor_model_path, str
        ):
            raise ValueError("predictor_model_path must be a string or null")
        if self.runtime_audit_path is not None and not isinstance(
            self.runtime_audit_path, str
        ):
            raise ValueError("runtime_audit_path must be a string or null")
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

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BeliefKVConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown BeliefKV config fields: {sorted(unknown)}")
        return cls(**dict(raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
