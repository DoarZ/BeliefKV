from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import log2
from typing import Deque, Iterable

from beliefkv.metrics.summary import percentile
from beliefkv.policy.transfer_cost import PCIeCostModel
from beliefkv.runtime.protocol import (
    CommandStatus,
    TransferDirection,
    TransferTelemetry,
)


@dataclass(frozen=True)
class ServiceCurveKey:
    direction: TransferDirection
    size_bucket: int
    page_count_bucket: int
    compute_phase: str
    command_kind: str
    host_copy_state: str
    pinned_host: bool | None
    native_traffic_bucket: int


@dataclass(frozen=True)
class ServiceCurveEstimate:
    direction: TransferDirection
    size_bytes: int
    estimated_callback_ms: float
    estimated_unhidden_stall_ms: float | None
    setup_p90_ms: float
    callback_floor_p90_ms: float
    fixed_overhead_p90_ms: float
    effective_bytes_per_ms_p10: float
    rejection_probability: float
    sample_count: int
    source: str


@dataclass(frozen=True)
class _TransferSample:
    setup_ms: float | None
    callback_ms: float | None
    effective_bytes_per_ms: float | None
    compute_wait_ms: float | None
    fixed_overhead_ms: float | None


class TransferServiceCurve:
    """Rolling, direction-aware model of observed HiCache transfer service.

    Estimates intentionally use a P90 setup delay and P10 effective bandwidth.
    When there is insufficient matching data, the curve falls back to the
    configured static PCIe model instead of extrapolating from a tiny sample.
    """

    def __init__(
        self,
        fallback: PCIeCostModel,
        *,
        window: int = 256,
        min_samples: int = 8,
        fallback_safety_factor: float = 1.25,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not 1 <= min_samples <= window:
            raise ValueError("min_samples must be within window")
        if fallback_safety_factor < 1:
            raise ValueError("fallback_safety_factor must be at least one")
        self.fallback = fallback
        self.window = window
        self.min_samples = min_samples
        self.fallback_safety_factor = fallback_safety_factor
        self._buckets: dict[ServiceCurveKey, Deque[_TransferSample]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._coarse_buckets: dict[
            tuple[TransferDirection, int, str], Deque[_TransferSample]
        ] = defaultdict(lambda: deque(maxlen=self.window))
        self._direction_samples: dict[
            TransferDirection, Deque[_TransferSample]
        ] = defaultdict(lambda: deque(maxlen=self.window))
        self._bucket_outcomes: dict[ServiceCurveKey, Deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._coarse_outcomes: dict[
            tuple[TransferDirection, int, str], Deque[bool]
        ] = defaultdict(lambda: deque(maxlen=self.window))
        self._direction_outcomes: dict[TransferDirection, Deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    @staticmethod
    def size_bucket(size_bytes: int) -> int:
        if size_bytes <= 0:
            return 0
        return max(0, int(log2(size_bytes)))

    @staticmethod
    def count_bucket(count: int) -> int:
        if count <= 0:
            return 0
        return max(0, int(log2(count)))

    def _key(
        self,
        direction: TransferDirection,
        size_bytes: int,
        *,
        page_count: int,
        compute_phase: str,
        command_kind: str,
        host_copy_state: str,
        pinned_host: bool | None,
        native_concurrent_bytes: int,
    ) -> ServiceCurveKey:
        return ServiceCurveKey(
            direction,
            self.size_bucket(size_bytes),
            self.count_bucket(page_count),
            compute_phase or "unknown",
            command_kind or "",
            host_copy_state or "unknown",
            pinned_host,
            self.size_bucket(native_concurrent_bytes),
        )

    def observe(self, telemetry: TransferTelemetry) -> None:
        start = telemetry.start_ts_ms
        completed = telemetry.status == CommandStatus.COMPLETED
        key = self._key(
            telemetry.direction,
            max(telemetry.actual_bytes, telemetry.closure_bytes),
            page_count=telemetry.page_count,
            compute_phase=telemetry.compute_phase,
            command_kind=telemetry.command_kind,
            host_copy_state=telemetry.host_copy_state,
            pinned_host=telemetry.pinned_host,
            native_concurrent_bytes=telemetry.native_concurrent_bytes,
        )
        rejected = not completed
        coarse_key = (
            key.direction,
            key.size_bucket,
            key.compute_phase,
        )
        self._bucket_outcomes[key].append(rejected)
        self._coarse_outcomes[coarse_key].append(rejected)
        self._direction_outcomes[telemetry.direction].append(rejected)
        if not completed or start is None or telemetry.actual_bytes <= 0:
            return
        setup_ms = (
            max(0.0, start - telemetry.submit_ts_ms)
            if start is not None
            else None
        )
        callback_ms = (
            max(0.0, telemetry.complete_ts_ms - start)
            if start is not None
            else None
        )
        fixed_overhead_ms = sum(
            value
            for value in (
                telemetry.allocator_wait_ms,
                telemetry.callback_overhead_ms,
            )
            if value is not None
        )
        transfer_only_ms = (
            max(0.0, callback_ms - fixed_overhead_ms)
            if callback_ms is not None
            else None
        )
        effective_rate = None
        if transfer_only_ms and transfer_only_ms > 0 and telemetry.actual_bytes > 0:
            effective_rate = telemetry.actual_bytes / transfer_only_ms
        sample = _TransferSample(
            setup_ms=setup_ms,
            callback_ms=callback_ms,
            effective_bytes_per_ms=effective_rate,
            compute_wait_ms=telemetry.compute_wait_ms,
            fixed_overhead_ms=(
                fixed_overhead_ms
                if telemetry.allocator_wait_ms is not None
                or telemetry.callback_overhead_ms is not None
                else None
            ),
        )
        self._buckets[key].append(sample)
        self._coarse_buckets[coarse_key].append(sample)
        self._direction_samples[telemetry.direction].append(sample)

    def estimate(
        self,
        direction: TransferDirection,
        size_bytes: int,
        *,
        compute_phase: str = "unknown",
        page_count: int = 0,
        command_kind: str = "",
        host_copy_state: str = "unknown",
        pinned_host: bool | None = None,
        native_concurrent_bytes: int = 0,
    ) -> ServiceCurveEstimate:
        if min(size_bytes, page_count, native_concurrent_bytes) < 0:
            raise ValueError("transfer demand must be non-negative")
        key = self._key(
            direction,
            size_bytes,
            page_count=page_count,
            compute_phase=compute_phase,
            command_kind=command_kind,
            host_copy_state=host_copy_state,
            pinned_host=pinned_host,
            native_concurrent_bytes=native_concurrent_bytes,
        )
        exact = tuple(self._buckets.get(key, ()))
        coarse_key = (key.direction, key.size_bucket, key.compute_phase)
        coarse = tuple(self._coarse_buckets.get(coarse_key, ()))
        directional = tuple(self._direction_samples.get(direction, ()))
        exact_outcomes = tuple(self._bucket_outcomes.get(key, ()))
        coarse_outcomes = tuple(self._coarse_outcomes.get(coarse_key, ()))
        directional_outcomes = tuple(self._direction_outcomes.get(direction, ()))
        if self._usable_count(exact) >= self.min_samples:
            samples = exact
            outcomes = exact_outcomes
            source = "bucket"
        elif self._usable_count(coarse) >= self.min_samples:
            samples = coarse
            outcomes = coarse_outcomes
            source = "bucket"
        elif self._usable_count(directional) >= self.min_samples:
            samples = directional
            outcomes = directional_outcomes
            source = "direction"
        else:
            return self._fallback_estimate(
                direction, size_bytes, directional, directional_outcomes
            )

        setup_values = [item.setup_ms for item in samples if item.setup_ms is not None]
        rates = [
            item.effective_bytes_per_ms
            for item in samples
            if item.effective_bytes_per_ms is not None
            and item.effective_bytes_per_ms > 0
        ]
        setup_p90_ms = percentile(setup_values, 90) if setup_values else 0.0
        fixed_overheads = [
            item.fixed_overhead_ms
            for item in samples
            if item.fixed_overhead_ms is not None
        ]
        fixed_overhead_p90_ms = (
            percentile(fixed_overheads, 90) if fixed_overheads else 0.0
        )
        rate_p10 = percentile(rates, 10)
        callback_floor_p90_ms = self._callback_floor_p90(directional)
        callback_ms = max(
            callback_floor_p90_ms,
            setup_p90_ms
            + fixed_overhead_p90_ms
            + (size_bytes / rate_p10 if size_bytes else 0.0),
        )
        compute_wait = [
            item.compute_wait_ms
            for item in samples
            if item.compute_wait_ms is not None
        ]
        return ServiceCurveEstimate(
            direction=direction,
            size_bytes=size_bytes,
            estimated_callback_ms=callback_ms,
            estimated_unhidden_stall_ms=(
                percentile(compute_wait, 90) if compute_wait else None
            ),
            setup_p90_ms=setup_p90_ms,
            callback_floor_p90_ms=callback_floor_p90_ms,
            fixed_overhead_p90_ms=fixed_overhead_p90_ms,
            effective_bytes_per_ms_p10=rate_p10,
            rejection_probability=self._rejection_probability(outcomes),
            sample_count=len(samples),
            source=source,
        )

    def _fallback_estimate(
        self,
        direction: TransferDirection,
        size_bytes: int,
        samples: Iterable[_TransferSample],
        outcomes: Iterable[bool],
    ) -> ServiceCurveEstimate:
        observations = tuple(samples)
        callback_floor_p90_ms = self._callback_floor_p90(observations)
        return ServiceCurveEstimate(
            direction=direction,
            size_bytes=size_bytes,
            estimated_callback_ms=(
                max(
                    callback_floor_p90_ms,
                    self.fallback.transfer_ms(size_bytes)
                    * self.fallback_safety_factor,
                )
            ),
            estimated_unhidden_stall_ms=None,
            setup_p90_ms=self.fallback.overhead_ms,
            callback_floor_p90_ms=callback_floor_p90_ms,
            fixed_overhead_p90_ms=0.0,
            effective_bytes_per_ms_p10=self.fallback.bandwidth_gbps * 1_000_000.0,
            rejection_probability=self._rejection_probability(outcomes),
            sample_count=len(observations),
            source="static_fallback",
        )

    @staticmethod
    def _callback_floor_p90(samples: Iterable[_TransferSample]) -> float:
        callbacks = [
            sample.setup_ms + sample.callback_ms
            for sample in samples
            if sample.setup_ms is not None and sample.callback_ms is not None
        ]
        return percentile(callbacks, 90) if callbacks else 0.0

    @staticmethod
    def _usable_count(samples: Iterable[_TransferSample]) -> int:
        return sum(
            1
            for sample in samples
            if sample.effective_bytes_per_ms is not None
            and sample.effective_bytes_per_ms > 0
        )

    @staticmethod
    def _rejection_probability(outcomes: Iterable[bool]) -> float:
        observations = tuple(outcomes)
        if not observations:
            return 0.0
        return sum(observations) / len(observations)

    def snapshot(self) -> dict[str, object]:
        buckets = []
        for key, samples in sorted(
            self._buckets.items(),
            key=lambda item: (
                item[0].direction.value,
                item[0].compute_phase,
                item[0].size_bucket,
            ),
        ):
            outcomes = tuple(self._bucket_outcomes.get(key, ()))
            buckets.append(
                {
                    "direction": key.direction.value,
                    "size_bucket": key.size_bucket,
                    "page_count_bucket": key.page_count_bucket,
                    "compute_phase": key.compute_phase,
                    "command_kind": key.command_kind,
                    "host_copy_state": key.host_copy_state,
                    "pinned_host": key.pinned_host,
                    "native_traffic_bucket": key.native_traffic_bucket,
                    "sample_count": len(samples),
                    "usable_count": self._usable_count(samples),
                    "outcome_count": len(outcomes),
                    "rejection_probability": self._rejection_probability(outcomes),
                }
            )
        return {"window": self.window, "min_samples": self.min_samples, "buckets": buckets}
