from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from beliefkv.metrics.summary import percentile
from beliefkv.policy.reference.base import ResourceSnapshot
from beliefkv.policy.service_curve import TransferServiceCurve
from beliefkv.runtime.protocol import (
    CommandStatus,
    TransferDirection,
    TransferTelemetry,
)


class ResourceSnapshotError(ValueError):
    """Raised when authoritative allocator observations cannot close the ledger."""


@dataclass(frozen=True)
class RuntimeResourceObservation:
    """One scheduler-safe-point observation of physical resource state."""

    ts_ms: float
    hbm_capacity_bytes: int
    hbm_used_bytes: int
    host_capacity_bytes: int
    host_used_bytes: int
    host_free_bytes: int
    urgent_d2h_bytes: int = 0
    urgent_h2d_bytes: int = 0
    pcie_utilization: float | None = None
    gpu_compute_utilization: float | None = None
    source: str = "authoritative_allocator"

    def __post_init__(self) -> None:
        if not math.isfinite(self.ts_ms) or self.ts_ms < 0:
            raise ValueError("resource observation timestamp must be non-negative")
        if self.hbm_capacity_bytes <= 0 or self.host_capacity_bytes < 0:
            raise ValueError("resource capacities are invalid")
        if not 0 <= self.hbm_used_bytes <= self.hbm_capacity_bytes:
            raise ValueError("observed HBM usage exceeds capacity")
        if min(
            self.host_used_bytes,
            self.host_free_bytes,
            self.urgent_d2h_bytes,
            self.urgent_h2d_bytes,
        ) < 0:
            raise ValueError("resource byte observations must be non-negative")
        if self.host_used_bytes + self.host_free_bytes != self.host_capacity_bytes:
            raise ValueError("host used/free bytes do not close host capacity")
        for value, name in (
            (self.pcie_utilization, "pcie_utilization"),
            (self.gpu_compute_utilization, "gpu_compute_utilization"),
        ):
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1] when observed")
        if not self.source:
            raise ValueError("resource observation source must be non-empty")


class ResourceSnapshotBuilder:
    """Build conservative, replayable resource snapshots from physical telemetry."""

    def __init__(
        self,
        *,
        growth_window: int = 32,
        transfer_probe_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if growth_window < 2:
            raise ValueError("growth_window must be at least two")
        if transfer_probe_bytes <= 0:
            raise ValueError("transfer_probe_bytes must be positive")
        self._hbm_history: deque[tuple[float, int]] = deque(maxlen=growth_window)
        self.transfer_probe_bytes = transfer_probe_bytes
        self._last_diagnostics: Mapping[str, object] = MappingProxyType({})

    @property
    def last_diagnostics(self) -> Mapping[str, object]:
        return self._last_diagnostics

    def build(
        self,
        snapshot_id: str,
        observation: RuntimeResourceObservation,
        *,
        hbm_reserved_bytes: int,
        service_curve: TransferServiceCurve,
        transfer_telemetry: Iterable[TransferTelemetry] = (),
    ) -> ResourceSnapshot:
        if not snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        if hbm_reserved_bytes < 0:
            raise ValueError("HBM reservation must be non-negative")
        if (
            observation.hbm_used_bytes + hbm_reserved_bytes
            > observation.hbm_capacity_bytes
        ):
            raise ResourceSnapshotError(
                "authoritative HBM usage plus admission reservations exceed capacity"
            )

        self._observe_hbm(observation.ts_ms, observation.hbm_used_bytes)
        telemetry = tuple(transfer_telemetry)
        probe_bytes = max(
            self.transfer_probe_bytes,
            observation.urgent_d2h_bytes,
            observation.urgent_h2d_bytes,
        )
        h2d = service_curve.estimate(TransferDirection.H2D, probe_bytes)
        d2h = service_curve.estimate(TransferDirection.D2H, probe_bytes)
        setup_samples = [
            item.start_ts_ms - item.submit_ts_ms
            for item in telemetry
            if item.status == CommandStatus.COMPLETED
            and item.start_ts_ms is not None
        ]
        unhidden_samples = [
            item.compute_wait_ms / item.actual_bytes
            for item in telemetry
            if item.status == CommandStatus.COMPLETED
            and item.compute_wait_ms is not None
            and item.actual_bytes > 0
        ]
        setup_ms = (
            percentile(setup_samples, 50)
            if setup_samples
            else max(h2d.setup_p90_ms, d2h.setup_p90_ms)
        )
        if unhidden_samples:
            unhidden_stall_per_byte = percentile(unhidden_samples, 90)
            unhidden_source = "observed_compute_wait_p90"
        else:
            # Missing overlap telemetry must not make D2H appear free.
            unhidden_stall_per_byte = 1.0 / max(
                1.0, d2h.effective_bytes_per_ms_p10
            )
            unhidden_source = "conservative_d2h_service_time"

        self._last_diagnostics = MappingProxyType(
            {
                "observation_source": observation.source,
                "pcie_utilization_observed": observation.pcie_utilization is not None,
                "gpu_compute_utilization_observed": (
                    observation.gpu_compute_utilization is not None
                ),
                "h2d_service_source": h2d.source,
                "d2h_service_source": d2h.source,
                "transfer_setup_source": (
                    "observed_p50" if setup_samples else "service_curve_p90_fallback"
                ),
                "unhidden_stall_source": unhidden_source,
                "transfer_probe_bytes": probe_bytes,
                "completed_transfer_sample_count": sum(
                    item.status == CommandStatus.COMPLETED for item in telemetry
                ),
            }
        )
        return ResourceSnapshot(
            snapshot_id=snapshot_id,
            ts_ms=observation.ts_ms,
            hbm_capacity_bytes=observation.hbm_capacity_bytes,
            hbm_used_bytes=observation.hbm_used_bytes,
            hbm_reserved_bytes=hbm_reserved_bytes,
            host_free_bytes=observation.host_free_bytes,
            urgent_d2h_bytes=observation.urgent_d2h_bytes,
            urgent_h2d_bytes=observation.urgent_h2d_bytes,
            pcie_utilization=(observation.pcie_utilization or 0.0),
            gpu_compute_utilization=(
                observation.gpu_compute_utilization or 0.0
            ),
            recent_kv_growth_bytes_per_ms=self._recent_growth(),
            h2d_service_bytes_per_ms=h2d.effective_bytes_per_ms_p10,
            d2h_service_bytes_per_ms=d2h.effective_bytes_per_ms_p10,
            transfer_setup_p50_ms=setup_ms,
            unhidden_stall_per_byte=unhidden_stall_per_byte,
        )

    def _observe_hbm(self, ts_ms: float, used_bytes: int) -> None:
        if self._hbm_history and ts_ms < self._hbm_history[-1][0]:
            raise ResourceSnapshotError("resource observations must be time-monotonic")
        if self._hbm_history and ts_ms == self._hbm_history[-1][0]:
            self._hbm_history[-1] = (ts_ms, used_bytes)
        else:
            self._hbm_history.append((ts_ms, used_bytes))

    def _recent_growth(self) -> float:
        if len(self._hbm_history) < 2:
            return 0.0
        elapsed = self._hbm_history[-1][0] - self._hbm_history[0][0]
        if elapsed <= 0:
            return 0.0
        samples = tuple(self._hbm_history)
        positive_growth = sum(
            max(0, current[1] - previous[1])
            for previous, current in zip(samples[:-1], samples[1:], strict=True)
        )
        return positive_growth / elapsed
