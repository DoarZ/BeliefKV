from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import json
from math import log2
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping

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
    nearest_bucket_distance: int | None
    size_coverage_bytes: tuple[int, int] | None
    extent_count_coverage: tuple[int, int] | None
    shape_bucket_distance: int | None
    shape_supported: bool
    estimated_completion_p90_ms: float
    estimated_unhidden_stall_p90_ms: float | None


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
        self._bucket_warm_start_sources: dict[
            ServiceCurveKey, Deque[bool]
        ] = defaultdict(lambda: deque(maxlen=self.window))
        self.warm_start_sample_count = 0
        self.warm_start_min_samples: int | None = None
        self.warm_start_hardware_key: str | None = None
        self.warm_start_metadata: dict[str, object] = {}

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
            page_count=(
                telemetry.extent_count
                if telemetry.extent_count > 0
                else telemetry.page_count
            ),
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
        if not completed or telemetry.actual_bytes <= 0:
            return
        setup_ms = (
            max(0.0, start - telemetry.submit_ts_ms)
            if start is not None
            else None
        )
        callback_ms = max(
            0.0,
            telemetry.complete_ts_ms
            - (start if start is not None else telemetry.submit_ts_ms),
        )
        fixed_overhead_ms = sum(
            value
            for value in (
                telemetry.allocator_wait_ms,
                telemetry.callback_overhead_ms,
            )
            if value is not None
        )
        transfer_only_ms = max(
            0.0,
            callback_ms - fixed_overhead_ms
            if start is not None
            else callback_ms,
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
        self._bucket_warm_start_sources[key].append(False)
        self._coarse_buckets[coarse_key].append(sample)
        self._direction_samples[telemetry.direction].append(sample)

    def save_artifact(
        self,
        path: str | Path,
        *,
        hardware_key: str,
        schema_version: int = 1,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not hardware_key:
            raise ValueError("transfer service hardware key is required")
        if schema_version not in {1, 2}:
            raise ValueError("unsupported transfer service artifact schema")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": schema_version,
            "hardware_key": hardware_key,
            "window": self.window,
            "min_samples": self.min_samples,
            "buckets": [
                {
                    "key": {
                        **asdict(key),
                        "direction": key.direction.value,
                    },
                    "samples": [asdict(sample) for sample in samples],
                    "outcomes": list(self._bucket_outcomes.get(key, ())),
                }
                for key, samples in sorted(
                    self._buckets.items(),
                    key=lambda item: repr(item[0]),
                )
            ],
        }
        if schema_version >= 2:
            payload["metadata"] = dict(metadata or {})
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def warm_start(
        self,
        path: str | Path,
        *,
        expected_hardware_key: str | None = None,
    ) -> int:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema_version = int(payload.get("schema_version", 0))
        if schema_version not in {1, 2}:
            raise ValueError("unsupported transfer service artifact schema")
        hardware_key = str(payload.get("hardware_key") or "")
        if not hardware_key:
            raise ValueError("transfer service artifact has no hardware key")
        if expected_hardware_key is not None and hardware_key != expected_hardware_key:
            raise ValueError(
                "transfer service hardware mismatch: "
                f"expected {expected_hardware_key}, got {hardware_key}"
            )
        artifact_min_samples = int(payload.get("min_samples") or 0)
        artifact_window = int(payload.get("window") or 0)
        if not 1 <= artifact_min_samples <= artifact_window:
            raise ValueError("transfer service artifact has an invalid sample gate")
        if (
            self.warm_start_min_samples is not None
            and self.warm_start_min_samples != artifact_min_samples
        ):
            raise ValueError("transfer service warm-start sample gates differ")
        loaded = 0
        for raw_bucket in payload.get("buckets", ()):
            if not isinstance(raw_bucket, Mapping):
                raise ValueError("invalid transfer service bucket")
            raw_key = raw_bucket.get("key")
            if not isinstance(raw_key, Mapping):
                raise ValueError("transfer service bucket has no key")
            key = ServiceCurveKey(
                direction=TransferDirection(str(raw_key["direction"])),
                size_bucket=int(raw_key["size_bucket"]),
                page_count_bucket=int(raw_key["page_count_bucket"]),
                compute_phase=str(raw_key["compute_phase"]),
                command_kind=str(raw_key["command_kind"]),
                host_copy_state=str(raw_key["host_copy_state"]),
                pinned_host=(
                    None
                    if raw_key.get("pinned_host") is None
                    else bool(raw_key["pinned_host"])
                ),
                native_traffic_bucket=int(raw_key["native_traffic_bucket"]),
            )
            coarse_key = (key.direction, key.size_bucket, key.compute_phase)
            for raw_sample in raw_bucket.get("samples", ()):
                if not isinstance(raw_sample, Mapping):
                    raise ValueError("invalid transfer service sample")
                sample = _TransferSample(
                    setup_ms=_optional_float(raw_sample.get("setup_ms")),
                    callback_ms=_optional_float(raw_sample.get("callback_ms")),
                    effective_bytes_per_ms=_optional_float(
                        raw_sample.get("effective_bytes_per_ms")
                    ),
                    compute_wait_ms=_optional_float(
                        raw_sample.get("compute_wait_ms")
                    ),
                    fixed_overhead_ms=_optional_float(
                        raw_sample.get("fixed_overhead_ms")
                    ),
                )
                self._buckets[key].append(sample)
                self._bucket_warm_start_sources[key].append(True)
                self._coarse_buckets[coarse_key].append(sample)
                self._direction_samples[key.direction].append(sample)
                loaded += 1
            for rejected in raw_bucket.get("outcomes", ()):
                value = bool(rejected)
                self._bucket_outcomes[key].append(value)
                self._coarse_outcomes[coarse_key].append(value)
                self._direction_outcomes[key.direction].append(value)
        self.warm_start_sample_count += loaded
        self.warm_start_min_samples = artifact_min_samples
        self.warm_start_hardware_key = hardware_key
        self.warm_start_metadata = (
            dict(payload.get("metadata") or {}) if schema_version >= 2 else {}
        )
        return loaded

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
        exact_outcomes = tuple(self._bucket_outcomes.get(key, ()))
        exact_usable = self._usable_count(exact)
        exact_warm_usable = self._warm_start_usable_count(key)
        if self._has_qualified_support(exact_usable, exact_warm_usable):
            samples = exact
            outcomes = exact_outcomes
            source = "bucket"
            nearest_bucket_distance = 0
            coverage_buckets = (key.size_bucket, key.size_bucket)
            extent_coverage_buckets = (
                key.page_count_bucket,
                key.page_count_bucket,
            )
        else:
            (
                neighboring,
                neighboring_outcomes,
                nearest_bucket_distance,
                coverage_buckets,
                extent_coverage_buckets,
                neighboring_warm_usable,
            ) = self._neighboring_shape_samples(
                key,
                max_size_bucket_distance=3,
                max_extent_bucket_distance=1,
            )
            if self._has_qualified_support(
                self._usable_count(neighboring),
                neighboring_warm_usable,
            ):
                samples = neighboring
                outcomes = neighboring_outcomes
                source = "bounded_neighboring_shape_extrapolation"
            else:
                return self._fallback_estimate(direction, size_bytes)

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
        callback_floor_p90_ms = self._callback_floor_p90(samples)
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
            nearest_bucket_distance=nearest_bucket_distance,
            size_coverage_bytes=self._bucket_coverage_bytes(coverage_buckets),
            extent_count_coverage=self._bucket_coverage_counts(
                extent_coverage_buckets
            ),
            shape_bucket_distance=nearest_bucket_distance,
            shape_supported=True,
            estimated_completion_p90_ms=callback_ms,
            estimated_unhidden_stall_p90_ms=(
                percentile(compute_wait, 90) if compute_wait else None
            ),
        )

    def _neighboring_shape_samples(
        self,
        key: ServiceCurveKey,
        *,
        max_size_bucket_distance: int,
        max_extent_bucket_distance: int,
    ) -> tuple[
        tuple[_TransferSample, ...],
        tuple[bool, ...],
        int | None,
        tuple[int, int] | None,
        tuple[int, int] | None,
        int,
    ]:
        """Collect nearby size/count buckets without erasing closure shape."""

        samples: list[_TransferSample] = []
        outcomes: list[bool] = []
        used_size_buckets: list[int] = []
        used_count_buckets: list[int] = []
        nearest_distance: int | None = None
        warm_start_usable = 0
        maximum_distance = max_size_bucket_distance + max_extent_bucket_distance
        for distance in range(1, maximum_distance + 1):
            matching_keys = []
            for candidate in self._buckets:
                if (
                    candidate.direction != key.direction
                    or candidate.compute_phase != key.compute_phase
                    or candidate.command_kind != key.command_kind
                    or candidate.host_copy_state != key.host_copy_state
                    or candidate.pinned_host != key.pinned_host
                    or candidate.native_traffic_bucket != key.native_traffic_bucket
                ):
                    continue
                size_delta = abs(candidate.size_bucket - key.size_bucket)
                count_delta = abs(
                    candidate.page_count_bucket - key.page_count_bucket
                )
                if (
                    size_delta > max_size_bucket_distance
                    or count_delta > max_extent_bucket_distance
                    or size_delta + count_delta != distance
                ):
                    continue
                matching_keys.append(candidate)
            for candidate in sorted(matching_keys, key=repr):
                neighboring = self._buckets.get(candidate, ())
                if not neighboring:
                    continue
                nearest_distance = (
                    distance if nearest_distance is None else nearest_distance
                )
                used_size_buckets.append(candidate.size_bucket)
                used_count_buckets.append(candidate.page_count_bucket)
                samples.extend(neighboring)
                warm_start_usable += self._warm_start_usable_count(candidate)
                outcomes.extend(self._bucket_outcomes.get(candidate, ()))
            if self._has_qualified_support(
                self._usable_count(samples), warm_start_usable
            ):
                break
        size_coverage = (
            (min(used_size_buckets), max(used_size_buckets))
            if used_size_buckets
            else None
        )
        count_coverage = (
            (min(used_count_buckets), max(used_count_buckets))
            if used_count_buckets
            else None
        )
        return (
            tuple(samples),
            tuple(outcomes),
            nearest_distance,
            size_coverage,
            count_coverage,
            warm_start_usable,
        )

    def _warm_start_usable_count(self, key: ServiceCurveKey) -> int:
        samples = self._buckets.get(key, ())
        sources = self._bucket_warm_start_sources.get(key, ())
        return sum(
            bool(is_warm_start)
            and sample.effective_bytes_per_ms is not None
            and sample.effective_bytes_per_ms > 0
            for sample, is_warm_start in zip(samples, sources)
        )

    def _has_qualified_support(
        self,
        usable_count: int,
        warm_start_usable_count: int,
    ) -> bool:
        return usable_count >= self.min_samples or bool(
            self.warm_start_min_samples is not None
            and warm_start_usable_count >= self.warm_start_min_samples
        )

    @staticmethod
    def _bucket_coverage_bytes(
        buckets: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if buckets is None:
            return None
        minimum, maximum = buckets
        return 1 << minimum, (1 << (maximum + 1)) - 1

    @staticmethod
    def _bucket_coverage_counts(
        buckets: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if buckets is None:
            return None
        minimum, maximum = buckets
        return 1 << minimum, (1 << (maximum + 1)) - 1

    def _fallback_estimate(
        self,
        direction: TransferDirection,
        size_bytes: int,
    ) -> ServiceCurveEstimate:
        callback_floor_p90_ms = 0.0
        callback_ms = self.fallback.transfer_ms(size_bytes) * self.fallback_safety_factor
        return ServiceCurveEstimate(
            direction=direction,
            size_bytes=size_bytes,
            estimated_callback_ms=callback_ms,
            estimated_unhidden_stall_ms=None,
            setup_p90_ms=self.fallback.overhead_ms,
            callback_floor_p90_ms=callback_floor_p90_ms,
            fixed_overhead_p90_ms=0.0,
            effective_bytes_per_ms_p10=self.fallback.bandwidth_gbps * 1_000_000.0,
            rejection_probability=0.0,
            sample_count=0,
            source="shape_unsupported_static_fallback",
            nearest_bucket_distance=None,
            size_coverage_bytes=None,
            extent_count_coverage=None,
            shape_bucket_distance=None,
            shape_supported=False,
            estimated_completion_p90_ms=callback_ms,
            estimated_unhidden_stall_p90_ms=None,
        )

    @staticmethod
    def _callback_floor_p90(samples: Iterable[_TransferSample]) -> float:
        callbacks = [
            (sample.setup_ms or 0.0) + sample.callback_ms
            for sample in samples
            if sample.callback_ms is not None
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
            setup_values = [
                item.setup_ms for item in samples if item.setup_ms is not None
            ]
            rates = [
                item.effective_bytes_per_ms
                for item in samples
                if item.effective_bytes_per_ms is not None
                and item.effective_bytes_per_ms > 0
            ]
            fixed_overheads = [
                item.fixed_overhead_ms
                for item in samples
                if item.fixed_overhead_ms is not None
            ]
            compute_wait = [
                item.compute_wait_ms
                for item in samples
                if item.compute_wait_ms is not None
            ]
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
                    "warm_start_usable_count": (
                        self._warm_start_usable_count(key)
                    ),
                    "outcome_count": len(outcomes),
                    "rejection_probability": self._rejection_probability(outcomes),
                    "setup_p90_ms": (
                        percentile(setup_values, 90) if setup_values else 0.0
                    ),
                    "callback_floor_p90_ms": self._callback_floor_p90(samples),
                    "fixed_overhead_p90_ms": (
                        percentile(fixed_overheads, 90)
                        if fixed_overheads
                        else 0.0
                    ),
                    "effective_bytes_per_ms_p10": (
                        percentile(rates, 10) if rates else 0.0
                    ),
                    "estimated_unhidden_stall_p90_ms": (
                        percentile(compute_wait, 90) if compute_wait else None
                    ),
                }
            )
        return {
            "schema_version": 1,
            "window": self.window,
            "min_samples": self.min_samples,
            "warm_start_sample_count": self.warm_start_sample_count,
            "warm_start_min_samples": self.warm_start_min_samples,
            "warm_start_hardware_key": self.warm_start_hardware_key,
            "warm_start_metadata": dict(self.warm_start_metadata),
            "fallback": {
                "bandwidth_gbps": self.fallback.bandwidth_gbps,
                "overhead_ms": self.fallback.overhead_ms,
                "safety_factor": self.fallback_safety_factor,
            },
            "buckets": buckets,
        }

    def warm_start_contract(self) -> dict[str, object]:
        """Summarize whether calibrated evidence remains usable after loading.

        Runtime observations and persistent calibration artifacts deliberately use
        independent sample gates.  This preflight catches regressions where a
        stricter online gate silently turns every calibrated bucket into a static
        fallback.
        """

        artifact_keys = tuple(
            key
            for key in self._buckets
            if self._warm_start_usable_count(key) > 0
        )
        supported_representatives = 0
        for key in artifact_keys:
            estimate = self.estimate(
                key.direction,
                max(1, 1 << key.size_bucket),
                page_count=max(1, 1 << key.page_count_bucket),
                compute_phase=key.compute_phase,
                command_kind=key.command_kind,
                host_copy_state=key.host_copy_state,
                pinned_host=key.pinned_host,
                native_concurrent_bytes=(
                    0
                    if key.native_traffic_bucket == 0
                    else 1 << key.native_traffic_bucket
                ),
            )
            supported_representatives += int(estimate.shape_supported)
        return {
            "artifact_loaded": self.warm_start_min_samples is not None,
            "hardware_key": self.warm_start_hardware_key,
            "runtime_min_samples": self.min_samples,
            "artifact_min_samples": self.warm_start_min_samples,
            "loaded_sample_count": self.warm_start_sample_count,
            "artifact_bucket_count": len(artifact_keys),
            "supported_representative_count": supported_representatives,
            "contract_satisfied": bool(
                self.warm_start_min_samples is not None
                and self.warm_start_sample_count > 0
                and supported_representatives > 0
            ),
        }

    def validate_warm_start_contract(self) -> dict[str, object]:
        contract = self.warm_start_contract()
        if not contract["artifact_loaded"]:
            raise ValueError("transfer service warm-start artifact was not loaded")
        if not contract["loaded_sample_count"]:
            raise ValueError("transfer service warm-start artifact has no usable samples")
        if not contract["supported_representative_count"]:
            raise ValueError(
                "transfer service warm-start artifact has no supported query; "
                "check artifact and runtime sample-gate semantics"
            )
        return contract

    @classmethod
    def estimate_snapshot(
        cls,
        snapshot: Mapping[str, object],
        direction: TransferDirection,
        size_bytes: int,
        *,
        page_count: int,
        compute_phase: str = "unknown",
        command_kind: str = "",
        host_copy_state: str = "unknown",
        pinned_host: bool | None = None,
        native_concurrent_bytes: int = 0,
    ) -> ServiceCurveEstimate:
        """Query an immutable service-curve snapshot in a risk worker."""

        if min(size_bytes, page_count, native_concurrent_bytes) < 0:
            raise ValueError("transfer demand must be non-negative")
        size_bucket = cls.size_bucket(size_bytes)
        count_bucket = cls.count_bucket(page_count)
        native_bucket = cls.size_bucket(native_concurrent_bytes)
        buckets = [
            item
            for item in snapshot.get("buckets", ())
            if isinstance(item, Mapping)
            and str(item.get("direction")) == direction.value
            and str(item.get("compute_phase") or "unknown") == compute_phase
            and str(item.get("command_kind") or "") == command_kind
            and str(item.get("host_copy_state") or "unknown") == host_copy_state
            and item.get("pinned_host") == pinned_host
            and int(item.get("native_traffic_bucket") or 0) == native_bucket
        ]
        selected = [
            item
            for item in buckets
            if int(item.get("size_bucket") or 0) == size_bucket
            and int(item.get("page_count_bucket") or 0) == count_bucket
        ]
        source = "bucket"
        distance: int | None = 0
        if not cls._snapshot_has_qualified_support(snapshot, selected):
            selected = []
            distance = None
            for candidate_distance in range(1, 5):
                adjacent = [
                    item
                    for item in buckets
                    if abs(int(item.get("size_bucket") or 0) - size_bucket)
                    + abs(
                        int(item.get("page_count_bucket") or 0) - count_bucket
                    )
                    == candidate_distance
                    and abs(
                        int(item.get("size_bucket") or 0) - size_bucket
                    )
                    <= 3
                    and abs(
                        int(item.get("page_count_bucket") or 0) - count_bucket
                    )
                    <= 1
                ]
                selected.extend(adjacent)
                if cls._snapshot_has_qualified_support(snapshot, selected):
                    distance = candidate_distance
                    break
            source = "bounded_neighboring_shape_extrapolation"
        usable_count = sum(int(item.get("usable_count") or 0) for item in selected)
        if not cls._snapshot_has_qualified_support(snapshot, selected):
            fallback = snapshot.get("fallback")
            fallback = fallback if isinstance(fallback, Mapping) else {}
            bandwidth = float(fallback.get("bandwidth_gbps") or 24.0)
            overhead = float(fallback.get("overhead_ms") or 0.08)
            safety = float(fallback.get("safety_factor") or 1.25)
            completion = (overhead + size_bytes / (bandwidth * 1_000_000.0)) * safety
            return ServiceCurveEstimate(
                direction=direction,
                size_bytes=size_bytes,
                estimated_callback_ms=completion,
                estimated_unhidden_stall_ms=None,
                setup_p90_ms=overhead,
                callback_floor_p90_ms=0.0,
                fixed_overhead_p90_ms=0.0,
                effective_bytes_per_ms_p10=bandwidth * 1_000_000.0,
                rejection_probability=0.0,
                sample_count=0,
                source="shape_unsupported_static_fallback",
                nearest_bucket_distance=None,
                size_coverage_bytes=None,
                extent_count_coverage=None,
                shape_bucket_distance=None,
                shape_supported=False,
                estimated_completion_p90_ms=completion,
                estimated_unhidden_stall_p90_ms=None,
            )
        setup = max(float(item.get("setup_p90_ms") or 0.0) for item in selected)
        fixed = max(
            float(item.get("fixed_overhead_p90_ms") or 0.0)
            for item in selected
        )
        floor = max(
            float(item.get("callback_floor_p90_ms") or 0.0)
            for item in selected
        )
        rate = min(
            float(item.get("effective_bytes_per_ms_p10") or 0.0)
            for item in selected
            if float(item.get("effective_bytes_per_ms_p10") or 0.0) > 0
        )
        stall_values = [
            float(item["estimated_unhidden_stall_p90_ms"])
            for item in selected
            if item.get("estimated_unhidden_stall_p90_ms") is not None
        ]
        completion = max(floor, setup + fixed + size_bytes / rate)
        size_buckets = [int(item.get("size_bucket") or 0) for item in selected]
        count_buckets = [
            int(item.get("page_count_bucket") or 0) for item in selected
        ]
        sample_count = sum(int(item.get("sample_count") or 0) for item in selected)
        rejection = sum(
            float(item.get("rejection_probability") or 0.0)
            * int(item.get("outcome_count") or 0)
            for item in selected
        ) / max(1, sum(int(item.get("outcome_count") or 0) for item in selected))
        return ServiceCurveEstimate(
            direction=direction,
            size_bytes=size_bytes,
            estimated_callback_ms=completion,
            estimated_unhidden_stall_ms=max(stall_values) if stall_values else None,
            setup_p90_ms=setup,
            callback_floor_p90_ms=floor,
            fixed_overhead_p90_ms=fixed,
            effective_bytes_per_ms_p10=rate,
            rejection_probability=rejection,
            sample_count=sample_count,
            source=source,
            nearest_bucket_distance=distance,
            size_coverage_bytes=cls._bucket_coverage_bytes(
                (min(size_buckets), max(size_buckets))
            ),
            extent_count_coverage=cls._bucket_coverage_counts(
                (min(count_buckets), max(count_buckets))
            ),
            shape_bucket_distance=distance,
            shape_supported=True,
            estimated_completion_p90_ms=completion,
            estimated_unhidden_stall_p90_ms=(
                max(stall_values) if stall_values else None
            ),
        )

    @staticmethod
    def _snapshot_has_qualified_support(
        snapshot: Mapping[str, object],
        buckets: Iterable[Mapping[str, object]],
    ) -> bool:
        selected = tuple(buckets)
        usable_count = sum(
            int(item.get("usable_count") or 0) for item in selected
        )
        runtime_min_samples = max(1, int(snapshot.get("min_samples") or 1))
        if usable_count >= runtime_min_samples:
            return True
        warm_start_min_samples = snapshot.get("warm_start_min_samples")
        if warm_start_min_samples is None:
            return False
        warm_start_usable_count = sum(
            int(item.get("warm_start_usable_count") or 0)
            for item in selected
        )
        return warm_start_usable_count >= max(1, int(warm_start_min_samples))

    @classmethod
    def estimate_snapshot_byte_only(
        cls,
        snapshot: Mapping[str, object],
        direction: TransferDirection,
        size_bytes: int,
        *,
        compute_phase: str = "unknown",
        command_kind: str = "",
        host_copy_state: str = "unknown",
        pinned_host: bool | None = None,
        native_concurrent_bytes: int = 0,
    ) -> ServiceCurveEstimate:
        """Query the same evidence after deliberately erasing extent morphology."""

        if min(size_bytes, native_concurrent_bytes) < 0:
            raise ValueError("transfer demand must be non-negative")
        size_bucket = cls.size_bucket(size_bytes)
        native_bucket = cls.size_bucket(native_concurrent_bytes)
        buckets = [
            item
            for item in snapshot.get("buckets", ())
            if isinstance(item, Mapping)
            and str(item.get("direction")) == direction.value
            and str(item.get("compute_phase") or "unknown") == compute_phase
            and str(item.get("command_kind") or "") == command_kind
            and str(item.get("host_copy_state") or "unknown") == host_copy_state
            and item.get("pinned_host") == pinned_host
            and int(item.get("native_traffic_bucket") or 0) == native_bucket
        ]
        selected: list[Mapping[str, object]] = []
        distance: int | None = None
        for candidate_distance in range(0, 4):
            selected.extend(
                item
                for item in buckets
                if abs(int(item.get("size_bucket") or 0) - size_bucket)
                == candidate_distance
            )
            if cls._snapshot_has_qualified_support(snapshot, selected):
                distance = candidate_distance
                break
        usable_count = sum(int(item.get("usable_count") or 0) for item in selected)
        if not cls._snapshot_has_qualified_support(snapshot, selected):
            fallback = snapshot.get("fallback")
            fallback = fallback if isinstance(fallback, Mapping) else {}
            bandwidth = max(1e-9, float(fallback.get("bandwidth_gbps") or 24.0))
            overhead = max(0.0, float(fallback.get("overhead_ms") or 0.05))
            safety = max(1.0, float(fallback.get("safety_factor") or 1.25))
            duration = (overhead + size_bytes / (bandwidth * 1_000_000.0)) * safety
            return ServiceCurveEstimate(
                direction=direction,
                size_bytes=size_bytes,
                estimated_callback_ms=duration,
                estimated_unhidden_stall_ms=None,
                setup_p90_ms=overhead,
                callback_floor_p90_ms=0.0,
                fixed_overhead_p90_ms=0.0,
                effective_bytes_per_ms_p10=bandwidth * 1_000_000.0,
                rejection_probability=0.0,
                sample_count=0,
                source="byte_only_static_fallback",
                nearest_bucket_distance=None,
                size_coverage_bytes=None,
                extent_count_coverage=None,
                shape_bucket_distance=None,
                shape_supported=True,
                estimated_completion_p90_ms=duration,
                estimated_unhidden_stall_p90_ms=None,
            )

        def weighted(field: str, default: float = 0.0) -> float:
            return sum(
                float(item.get(field) or default)
                * int(item.get("usable_count") or 0)
                for item in selected
            ) / usable_count

        setup = weighted("setup_p90_ms")
        fixed = weighted("fixed_overhead_p90_ms")
        floor = weighted("callback_floor_p90_ms")
        rate = max(1.0, weighted("effective_bytes_per_ms_p10"))
        stall_weight = sum(
            int(item.get("usable_count") or 0)
            for item in selected
            if item.get("estimated_unhidden_stall_p90_ms") is not None
        )
        stall = (
            sum(
                float(item["estimated_unhidden_stall_p90_ms"])
                * int(item.get("usable_count") or 0)
                for item in selected
                if item.get("estimated_unhidden_stall_p90_ms") is not None
            )
            / stall_weight
            if stall_weight
            else None
        )
        completion = max(floor, setup + fixed + size_bytes / rate)
        used_sizes = tuple(int(item.get("size_bucket") or 0) for item in selected)
        return ServiceCurveEstimate(
            direction=direction,
            size_bytes=size_bytes,
            estimated_callback_ms=completion,
            estimated_unhidden_stall_ms=stall,
            setup_p90_ms=setup,
            callback_floor_p90_ms=floor,
            fixed_overhead_p90_ms=fixed,
            effective_bytes_per_ms_p10=rate,
            rejection_probability=weighted("rejection_probability"),
            sample_count=usable_count,
            source="byte_only_size_bucket",
            nearest_bucket_distance=distance,
            size_coverage_bytes=cls._bucket_coverage_bytes(
                (min(used_sizes), max(used_sizes))
            ),
            extent_count_coverage=None,
            shape_bucket_distance=None,
            shape_supported=True,
            estimated_completion_p90_ms=completion,
            estimated_unhidden_stall_p90_ms=stall,
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
