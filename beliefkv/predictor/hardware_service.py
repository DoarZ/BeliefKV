from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


HARDWARE_SERVICE_SCHEMA_VERSION = 3
CONTROLLED_SERVICE_EVIDENCE = "controlled_microbenchmark"
RUNTIME_VALIDATION_EVIDENCE = "runtime_validation"


@dataclass(frozen=True)
class GPURequestServiceDemand:
    sequence_tokens: int
    token_delta: int
    cache_hit_ratio: float = 0.0

    def __post_init__(self) -> None:
        if min(self.sequence_tokens, self.token_delta) < 0:
            raise ValueError("GPU request demand must be non-negative")
        if not 0.0 <= self.cache_hit_ratio <= 1.0:
            raise ValueError("cache-hit ratio must be in [0, 1]")


@dataclass(frozen=True)
class GPUServiceFeatures:
    phase: str
    request_demands: tuple[GPURequestServiceDemand, ...]
    chunk_position: str = "unknown"
    prefill_decode_mixed: bool = False
    pcie_contention_state: str = "idle"
    hicache_inflight_bytes: int = 0

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode", "mixed"}:
            raise ValueError("GPU service phase must be prefill, decode, or mixed")
        if not self.request_demands:
            raise ValueError("GPU service features require a complete batch")
        if not self.chunk_position or not self.pcie_contention_state:
            raise ValueError("chunk and contention state must be explicit")
        if self.hicache_inflight_bytes < 0:
            raise ValueError("HiCache in-flight bytes must be non-negative")

    @property
    def batch_size(self) -> int:
        return len(self.request_demands)

    @property
    def token_delta_total(self) -> int:
        return sum(item.token_delta for item in self.request_demands)

    @property
    def sequence_tokens_mean(self) -> float:
        return sum(item.sequence_tokens for item in self.request_demands) / self.batch_size

    @property
    def sequence_tokens_max(self) -> int:
        return max(item.sequence_tokens for item in self.request_demands)

    @property
    def cache_hit_ratio_mean(self) -> float:
        return sum(item.cache_hit_ratio for item in self.request_demands) / self.batch_size


@dataclass(frozen=True)
class GPUServiceEstimate:
    p50_ms: float
    p90_ms: float
    p95_ms: float
    support: float
    source: str
    timing_boundary: str
    calibrated: bool = False
    calibration_source: str = "unavailable"
    neighbor_count: int = 0
    nearest_distance: float | None = None

    def quantile(self, quantile: float) -> float:
        if quantile <= 0.5:
            return self.p50_ms
        if quantile <= 0.9:
            return self.p90_ms
        return self.p95_ms


@dataclass(frozen=True)
class _ServiceObservation:
    features: GPUServiceFeatures
    elapsed_ms: float


@dataclass(frozen=True)
class _CalibrationObservation:
    features: GPUServiceFeatures
    residuals: tuple[float, float, float]


class GPUServiceCurveModel:
    """Conditional batch service distribution fitted from controlled evidence.

    Agent-runtime overlap intervals may be evaluated against this model, but
    are rejected during fitting because their boundaries are not CUDA events.
    """

    def __init__(
        self,
        *,
        minimum_support: float = 4.0,
        neighbor_count: int = 48,
    ) -> None:
        if minimum_support <= 0:
            raise ValueError("minimum_support must be positive")
        if neighbor_count <= 0:
            raise ValueError("neighbor_count must be positive")
        self.minimum_support = minimum_support
        self.neighbor_count = neighbor_count
        self._groups: dict[tuple[str, ...], Counter[float]] = defaultdict(Counter)
        self._observations: list[_ServiceObservation] = []
        self._neighbor_index: dict[
            tuple[str, int, int], list[_ServiceObservation]
        ] = defaultdict(list)
        self._calibration_factors: dict[str, dict[str, float]] = {}
        self._cluster_calibration_floors: dict[str, dict[str, float]] = {}
        self._calibration_observations: list[_CalibrationObservation] = []
        self._calibration_neighbor_index: dict[
            tuple[str, int, int], list[_CalibrationObservation]
        ] = defaultdict(list)
        self.calibration_summary: dict[str, Any] = {}
        self.training_summary: dict[str, Any] = {}

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        values = [dict(row) for row in rows]
        if not values or {str(row.get("split")) for row in values} != {"train"}:
            raise ValueError("GPU service fitting requires only train rows")
        sample_ids = [str(row.get("sample_id") or "") for row in values]
        if any(not item for item in sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("GPU service fitting requires unique batch sample_id values")
        invalid_roles = sorted(
            {
                str(row.get("evidence_role") or "unknown")
                for row in values
                if row.get("evidence_role") != CONTROLLED_SERVICE_EVIDENCE
            }
        )
        if invalid_roles:
            raise ValueError(
                "GPU service fitting accepts controlled microbenchmarks only; "
                f"found {invalid_roles}"
            )
        eligible = [row for row in values if not row.get("warmup")]
        self._groups.clear()
        self._observations.clear()
        self._neighbor_index.clear()
        self._calibration_factors.clear()
        self._cluster_calibration_floors.clear()
        self._calibration_observations.clear()
        self._calibration_neighbor_index.clear()
        self.calibration_summary = {}
        for row in eligible:
            features = _features_from_batch_row(row)
            elapsed_ms = float(row.get("service_elapsed_ms") or 0.0)
            if elapsed_ms <= 0:
                continue
            self._observations.append(_ServiceObservation(features, elapsed_ms))
            bucket = _log_bucket(elapsed_ms)
            for key in _backoff_keys(features):
                self._groups[key][bucket] += 1.0
        self._rebuild_neighbor_index()
        self.training_summary = {
            "batch_sample_count": len(eligible),
            "unique_sample_count": len({str(row["sample_id"]) for row in eligible}),
            "evidence_role": CONTROLLED_SERVICE_EVIDENCE,
            "timing_boundary": (
                "scheduler/worker service interval calibrated against controlled cases; "
                "not claimed as pure CUDA kernel time"
            ),
            "batch_sizes": sorted(
                {int(row.get("batch_size") or 0) for row in eligible}
            ),
            "estimator": "phase-isolated weighted feature-neighborhood",
            "neighbor_count": self.neighbor_count,
            "calibrated": False,
        }
        return dict(self.training_summary)

    def fit_cross_calibrated(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        folds: int = 5,
        minimum_phase_calibration: int = 20,
    ) -> dict[str, Any]:
        """Fit all train rows after profile-grouped out-of-fold calibration."""

        values = [dict(row) for row in rows]
        if folds < 2:
            raise ValueError("cross calibration requires at least two folds")
        if minimum_phase_calibration <= 0:
            raise ValueError("minimum phase calibration must be positive")
        _validate_fit_rows(values)
        eligible = [row for row in values if not row.get("warmup")]
        assignments = _profile_fold_assignments(eligible, folds)
        residuals: dict[str, dict[float, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        profile_residuals: dict[
            str, dict[str, dict[float, list[float]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        calibration_observations: list[_CalibrationObservation] = []
        scored = 0
        unavailable = 0
        for fold in range(folds):
            validation = [
                row
                for row in eligible
                if assignments[_profile_key(row)] == fold
            ]
            training = [
                row
                for row in eligible
                if assignments[_profile_key(row)] != fold
            ]
            if not validation or len(training) < self.minimum_support:
                continue
            temporary = GPUServiceCurveModel(
                minimum_support=self.minimum_support,
                neighbor_count=self.neighbor_count,
            )
            temporary.fit(training)
            for row in validation:
                actual = float(row.get("service_elapsed_ms") or 0.0)
                estimate = temporary.predict(_features_from_batch_row(row))
                if actual <= 0 or estimate.source == "unavailable":
                    unavailable += 1
                    continue
                scored += 1
                phase = str(row.get("phase") or "unknown")
                profile = _profile_key(row)
                row_residuals = []
                for quantile, predicted in (
                    (0.5, estimate.p50_ms),
                    (0.9, estimate.p90_ms),
                    (0.95, estimate.p95_ms),
                ):
                    if predicted <= 0:
                        continue
                    score = math.log(actual / predicted)
                    row_residuals.append(score)
                    residuals["*"][quantile].append(score)
                    residuals[phase][quantile].append(score)
                    profile_residuals["*"][profile][quantile].append(score)
                    profile_residuals[phase][profile][quantile].append(score)
                if len(row_residuals) == 3:
                    calibration_observations.append(
                        _CalibrationObservation(
                            _features_from_batch_row(row), tuple(row_residuals)
                        )
                    )

        self.fit(values)
        factors: dict[str, dict[str, float]] = {}
        for group, by_quantile in residuals.items():
            support = len(by_quantile.get(0.5, ()))
            if group != "*" and support < minimum_phase_calibration:
                continue
            factors[group] = {
                str(quantile): math.exp(_percentile_required(scores, quantile))
                for quantile, scores in sorted(by_quantile.items())
                if scores
            }
            factors[group]["support"] = float(support)
        if "*" not in factors:
            raise ValueError("cross calibration produced no global residual support")
        cluster_floors: dict[str, dict[str, float]] = {}
        for group, by_profile in profile_residuals.items():
            cluster_floors[group] = {}
            for quantile in (0.9, 0.95):
                profile_scores = [
                    _percentile_required(by_quantile[quantile], quantile)
                    for by_quantile in by_profile.values()
                    if by_quantile.get(quantile)
                ]
                if profile_scores:
                    cluster_floors[group][str(quantile)] = math.exp(
                        _percentile_required(profile_scores, quantile)
                    )
        self._calibration_factors = factors
        self._cluster_calibration_floors = cluster_floors
        self._calibration_observations = calibration_observations
        self._rebuild_calibration_neighbor_index()
        self.calibration_summary = {
            "method": (
                "profile-grouped out-of-fold local multiplicative conformal"
            ),
            "folds": folds,
            "profile_count": len(assignments),
            "scored_sample_count": scored,
            "unavailable_sample_count": unavailable,
            "minimum_phase_calibration": minimum_phase_calibration,
            "groups": {
                key: int(value.get("support", 0.0))
                for key, value in sorted(factors.items())
            },
            "holdout_consumed": False,
            "local_residual_count": len(calibration_observations),
            "cluster_floor_groups": {
                group: len(profile_residuals[group])
                for group in sorted(cluster_floors)
            },
        }
        self.training_summary.update(
            {
                "calibrated": True,
                "calibration": dict(self.calibration_summary),
            }
        )
        return dict(self.training_summary)

    def predict(self, features: GPUServiceFeatures) -> GPUServiceEstimate:
        if self._observations:
            return self._predict_neighborhood(features)
        return self._predict_legacy(features)

    def _predict_neighborhood(
        self, features: GPUServiceFeatures
    ) -> GPUServiceEstimate:
        distances = sorted(
            (
                (_feature_distance(features, observation.features), observation)
                for observation in self._neighbor_candidates(features)
            ),
            key=lambda item: item[0],
        )
        if len(distances) < self.minimum_support:
            return GPUServiceEstimate(
                0.0,
                0.0,
                0.0,
                float(len(distances)),
                "unavailable",
                "no phase-local neighborhood support",
            )
        exact = [item for item in distances if item[0] <= 1e-12]
        selected = (
            exact
            if len(exact) >= self.minimum_support
            else distances[: min(self.neighbor_count, len(distances))]
        )
        positive_distances = [distance for distance, _ in selected if distance > 0]
        bandwidth = max(
            0.25,
            positive_distances[min(len(positive_distances) - 1, max(0, len(selected) // 2 - 1))]
            if positive_distances
            else 0.25,
        )
        weighted = [
            (
                observation.elapsed_ms,
                1.0 if distance <= 1e-12 else math.exp(-0.5 * (distance / bandwidth) ** 2),
            )
            for distance, observation in selected
        ]
        raw = {
            quantile: _weighted_sample_quantile(weighted, quantile)
            for quantile in (0.5, 0.9, 0.95)
        }
        calibration, calibration_source = self._calibration_adjustment(features)
        if calibration is not None:
            adjusted = {
                quantile: raw[quantile]
                * float(calibration.get(quantile, 1.0))
                for quantile in raw
            }
            calibrated = True
        else:
            adjusted = raw
            calibrated = False
            calibration_source = "unavailable"
        p50, p90, p95 = _monotone_quantiles(
            adjusted[0.5], adjusted[0.9], adjusted[0.95]
        )
        weights = [weight for _, weight in weighted]
        effective_support = sum(weights) ** 2 / sum(
            weight * weight for weight in weights
        )
        return GPUServiceEstimate(
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            support=effective_support,
            source="exact" if selected is exact else "neighbor_interpolation",
            timing_boundary=str(
                self.training_summary.get("timing_boundary")
                or "controlled service interval"
            ),
            calibrated=calibrated,
            calibration_source=calibration_source,
            neighbor_count=len(selected),
            nearest_distance=selected[0][0],
        )

    def _neighbor_candidates(
        self, features: GPUServiceFeatures
    ) -> tuple[_ServiceObservation, ...]:
        sequence_bucket = _sequence_neighbor_bucket(features.sequence_tokens_mean)
        target = max(int(math.ceil(self.minimum_support)), self.neighbor_count * 4)
        candidates: list[_ServiceObservation] = []
        seen: set[int] = set()
        for radius in range(65):
            buckets = (
                (sequence_bucket,)
                if radius == 0
                else (sequence_bucket - radius, sequence_bucket + radius)
            )
            for bucket in buckets:
                for observation in self._neighbor_index.get(
                    (features.phase, features.batch_size, bucket), ()
                ):
                    identity = id(observation)
                    if identity not in seen:
                        seen.add(identity)
                        candidates.append(observation)
            if len(candidates) >= target:
                break
        if len(candidates) >= self.minimum_support:
            return tuple(candidates)
        return tuple(
            observation
            for observation in self._observations
            if observation.features.phase == features.phase
        )

    def _rebuild_neighbor_index(self) -> None:
        self._neighbor_index.clear()
        for observation in self._observations:
            features = observation.features
            self._neighbor_index[
                (
                    features.phase,
                    features.batch_size,
                    _sequence_neighbor_bucket(features.sequence_tokens_mean),
                )
            ].append(observation)

    def _calibration_adjustment(
        self, features: GPUServiceFeatures
    ) -> tuple[dict[float, float] | None, str]:
        candidates = self._calibration_candidates(features)
        if len(candidates) >= self.minimum_support:
            ranked = sorted(
                (
                    (_feature_distance(features, observation.features), observation)
                    for observation in candidates
                ),
                key=lambda item: item[0],
            )[: min(self.neighbor_count * 2, len(candidates))]
            positive = [distance for distance, _ in ranked if distance > 0]
            bandwidth = max(
                0.25,
                positive[min(len(positive) - 1, max(0, len(ranked) // 2 - 1))]
                if positive
                else 0.25,
            )
            weighted_residuals: dict[float, list[tuple[float, float]]] = {
                0.5: [],
                0.9: [],
                0.95: [],
            }
            for distance, observation in ranked:
                weight = (
                    1.0
                    if distance <= 1e-12
                    else math.exp(-0.5 * (distance / bandwidth) ** 2)
                )
                for index, quantile in enumerate((0.5, 0.9, 0.95)):
                    weighted_residuals[quantile].append(
                        (observation.residuals[index], weight)
                    )
            local = {
                quantile: math.exp(_weighted_sample_quantile(values, quantile))
                for quantile, values in weighted_residuals.items()
            }
            return (
                self._apply_cluster_floor(features.phase, local),
                f"local:{features.phase}",
            )
        phase = features.phase
        calibration_key = phase if phase in self._calibration_factors else "*"
        raw = self._calibration_factors.get(calibration_key)
        if raw is None:
            return None, "unavailable"
        return (
            self._apply_cluster_floor(
                features.phase,
                {
                    quantile: float(raw.get(str(quantile), 1.0))
                    for quantile in (0.5, 0.9, 0.95)
                },
            ),
            f"global:{calibration_key}",
        )

    def _apply_cluster_floor(
        self, phase: str, factors: dict[float, float]
    ) -> dict[float, float]:
        factors = {**factors, 0.5: 1.0}
        floor = self._cluster_calibration_floors.get(
            phase, self._cluster_calibration_floors.get("*", {})
        )
        if not floor:
            return factors
        return {
            quantile: (
                max(value, float(floor.get(str(quantile), value)))
                if quantile > 0.5
                else 1.0
            )
            for quantile, value in factors.items()
        }

    def _calibration_candidates(
        self, features: GPUServiceFeatures
    ) -> tuple[_CalibrationObservation, ...]:
        sequence_bucket = _sequence_neighbor_bucket(features.sequence_tokens_mean)
        target = max(int(math.ceil(self.minimum_support)), self.neighbor_count * 8)
        candidates: list[_CalibrationObservation] = []
        seen: set[int] = set()
        for radius in range(65):
            buckets = (
                (sequence_bucket,)
                if radius == 0
                else (sequence_bucket - radius, sequence_bucket + radius)
            )
            for bucket in buckets:
                for observation in self._calibration_neighbor_index.get(
                    (features.phase, features.batch_size, bucket), ()
                ):
                    identity = id(observation)
                    if identity not in seen:
                        seen.add(identity)
                        candidates.append(observation)
            if len(candidates) >= target:
                break
        return tuple(candidates)

    def _rebuild_calibration_neighbor_index(self) -> None:
        self._calibration_neighbor_index.clear()
        for observation in self._calibration_observations:
            features = observation.features
            self._calibration_neighbor_index[
                (
                    features.phase,
                    features.batch_size,
                    _sequence_neighbor_bucket(features.sequence_tokens_mean),
                )
            ].append(observation)

    def _predict_legacy(self, features: GPUServiceFeatures) -> GPUServiceEstimate:
        candidates = _backoff_keys(features)
        selected = candidates[-1]
        for key in candidates:
            if sum(self._groups.get(key, {}).values()) >= self.minimum_support:
                selected = key
                break
        counts = self._groups.get(selected, Counter())
        support = sum(counts.values())
        if support <= 0:
            return GPUServiceEstimate(
                0.0,
                0.0,
                0.0,
                0.0,
                "unavailable",
                "no calibrated support",
            )
        return GPUServiceEstimate(
            p50_ms=_weighted_quantile(counts, 0.5),
            p90_ms=_weighted_quantile(counts, 0.9),
            p95_ms=_weighted_quantile(counts, 0.95),
            support=support,
            source=(
                "exact"
                if selected == candidates[0]
                else "phase_batch"
                if len(selected) >= 2
                else "global_backoff"
            ),
            timing_boundary=str(
                self.training_summary.get("timing_boundary")
                or "controlled service interval"
            ),
        )

    def evaluate_controlled_rows(
        self, rows: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        values = [dict(row) for row in rows]
        if not values or {str(row.get("split")) for row in values} != {"holdout"}:
            raise ValueError("controlled evaluation requires only holdout rows")
        relative_errors: list[float] = []
        phase_errors: dict[str, list[float]] = defaultdict(list)
        phase_cover90: dict[str, list[bool]] = defaultdict(list)
        phase_cover95: dict[str, list[bool]] = defaultdict(list)
        conditional_cells: dict[
            tuple[str, ...], list[tuple[float, float]]
        ] = defaultdict(list)
        cover90: list[bool] = []
        cover95: list[bool] = []
        sources: Counter[str] = Counter()
        unavailable = 0
        for row in values:
            if row.get("warmup"):
                continue
            if row.get("evidence_role") != CONTROLLED_SERVICE_EVIDENCE:
                raise ValueError("controlled holdout rows require controlled evidence")
            actual = float(row.get("service_elapsed_ms") or 0.0)
            estimate = self.predict(_features_from_batch_row(row))
            sources[estimate.source] += 1
            if actual <= 0 or estimate.source == "unavailable":
                unavailable += 1
                continue
            error = abs(estimate.p50_ms - actual) / actual
            relative_errors.append(error)
            phase_errors[str(row.get("phase") or "unknown")].append(error)
            phase = str(row.get("phase") or "unknown")
            cover90.append(actual <= estimate.p90_ms)
            cover95.append(actual <= estimate.p95_ms)
            phase_cover90[phase].append(actual <= estimate.p90_ms)
            phase_cover95[phase].append(actual <= estimate.p95_ms)
            conditional_cells[_evaluation_cell_key(row)].append(
                (actual, estimate.p50_ms)
            )
        cell_errors: list[float] = []
        phase_cell_errors: dict[str, list[float]] = defaultdict(list)
        for key, values in conditional_cells.items():
            actual_median = _percentile_required(
                [actual for actual, _ in values], 0.5
            )
            predicted_median = _percentile_required(
                [predicted for _, predicted in values], 0.5
            )
            if actual_median <= 0:
                continue
            error = abs(predicted_median - actual_median) / actual_median
            cell_errors.append(error)
            phase_cell_errors[key[0]].append(error)
        return {
            "sample_count": sum(sources.values()),
            "scored_count": len(relative_errors),
            "unavailable_count": unavailable,
            "source_counts": dict(sorted(sources.items())),
            "relative_error_p50": _percentile(relative_errors, 0.5),
            "relative_error_p95": _percentile(relative_errors, 0.95),
            "p90_coverage": _mean_bool(cover90),
            "p95_coverage": _mean_bool(cover95),
            "phase_relative_error_p95": {
                phase: _percentile(errors, 0.95)
                for phase, errors in sorted(phase_errors.items())
            },
            "conditional_cell_count": len(cell_errors),
            "conditional_cell_relative_error_p50": _percentile(
                cell_errors, 0.5
            ),
            "conditional_cell_relative_error_p95": _percentile(
                cell_errors, 0.95
            ),
            "phase_conditional_cell_relative_error_p95": {
                phase: _percentile(errors, 0.95)
                for phase, errors in sorted(phase_cell_errors.items())
            },
            "phase_p90_coverage": {
                phase: _mean_bool(values)
                for phase, values in sorted(phase_cover90.items())
            },
            "phase_p95_coverage": {
                phase: _mean_bool(values)
                for phase, values in sorted(phase_cover95.items())
            },
            "calibrated": bool(self._calibration_factors),
            "holdout_used_for_fit_or_calibration": False,
        }

    def validate_runtime_rows(
        self, rows: Iterable[Mapping[str, Any]], *, quantile: float = 0.9
    ) -> dict[str, Any]:
        """Evaluate runtime overlap intervals without adding them to fit counts."""

        errors: list[float] = []
        unavailable = 0
        sample_ids: set[str] = set()
        for row in rows:
            if row.get("evidence_role") != RUNTIME_VALIDATION_EVIDENCE:
                raise ValueError("runtime validation rows must be explicitly tagged")
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in sample_ids:
                raise ValueError("runtime validation requires unique sample IDs")
            sample_ids.add(sample_id)
            actual = float(row.get("service_elapsed_ms") or 0.0)
            estimate = self.predict(_features_from_batch_row(row))
            if actual <= 0 or estimate.source == "unavailable":
                unavailable += 1
                continue
            errors.append(abs(estimate.quantile(quantile) - actual) / actual)
        return {
            "sample_count": len(sample_ids),
            "scored_count": len(errors),
            "unavailable_count": unavailable,
            "relative_error_p50": _percentile(errors, 0.5),
            "relative_error_p95": _percentile(errors, 0.95),
            "model_updated": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HARDWARE_SERVICE_SCHEMA_VERSION,
            "model_kind": "conditional_batch_gpu_service_distribution",
            "minimum_support": self.minimum_support,
            "neighbor_count": self.neighbor_count,
            "training_summary": self.training_summary,
            "calibration_summary": self.calibration_summary,
            "calibration_factors": self._calibration_factors,
            "cluster_calibration_floors": self._cluster_calibration_floors,
            "calibration_observations": [
                {
                    "features": _features_to_dict(item.features),
                    "residuals": list(item.residuals),
                }
                for item in self._calibration_observations
            ],
            "observations": [
                {
                    "features": _features_to_dict(item.features),
                    "elapsed_ms": item.elapsed_ms,
                }
                for item in self._observations
            ],
            "groups": [
                {
                    "key": list(key),
                    "counts": {
                        str(value): count for value, count in sorted(counts.items())
                    },
                }
                for key, counts in sorted(self._groups.items())
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GPUServiceCurveModel":
        schema_version = int(raw.get("schema_version", -1))
        if schema_version not in {2, HARDWARE_SERVICE_SCHEMA_VERSION}:
            raise ValueError("unsupported GPU service curve schema")
        model = cls(
            minimum_support=float(raw.get("minimum_support", 4.0)),
            neighbor_count=int(raw.get("neighbor_count", 48)),
        )
        for item in raw.get("groups", ()):
            model._groups[tuple(str(value) for value in item["key"])] = Counter(
                {
                    float(value): float(count)
                    for value, count in item["counts"].items()
                }
            )
        model.training_summary = dict(raw.get("training_summary", {}))
        if schema_version >= 3:
            model._observations = [
                _ServiceObservation(
                    _features_from_dict(item["features"]),
                    float(item["elapsed_ms"]),
                )
                for item in raw.get("observations", ())
            ]
            model._calibration_factors = {
                str(group): {
                    str(key): float(value) for key, value in factors.items()
                }
                for group, factors in raw.get("calibration_factors", {}).items()
            }
            model._cluster_calibration_floors = {
                str(group): {
                    str(key): float(value) for key, value in factors.items()
                }
                for group, factors in raw.get(
                    "cluster_calibration_floors", {}
                ).items()
            }
            model.calibration_summary = dict(raw.get("calibration_summary", {}))
            model._rebuild_neighbor_index()
            model._calibration_observations = [
                _CalibrationObservation(
                    _features_from_dict(item["features"]),
                    tuple(float(value) for value in item["residuals"]),
                )
                for item in raw.get("calibration_observations", ())
            ]
            model._rebuild_calibration_neighbor_index()
        return model

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "GPUServiceCurveModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _features_from_batch_row(row: Mapping[str, Any]) -> GPUServiceFeatures:
    if row.get("row_type") not in {None, "gpu_batch_service_interval"}:
        raise ValueError("GPU service model requires one row per complete batch")
    raw_requests = row.get("request_samples") or ()
    if not raw_requests:
        raise ValueError("GPU batch row has no request composition")
    demands = tuple(
        GPURequestServiceDemand(
            sequence_tokens=int(item.get("sequence_tokens_before") or 0),
            token_delta=int(item.get("token_delta") or 0),
            cache_hit_ratio=float(item.get("cache_hit_ratio") or 0.0),
        )
        for item in raw_requests
    )
    declared_batch = int(row.get("batch_size") or len(demands))
    if declared_batch != len(demands):
        raise ValueError("GPU batch row does not contain its complete composition")
    return GPUServiceFeatures(
        phase=str(row.get("phase") or "unknown"),
        request_demands=demands,
        chunk_position=str(row.get("chunk_position") or "unknown"),
        prefill_decode_mixed=bool(row.get("prefill_decode_mixed", False)),
        pcie_contention_state=str(row.get("pcie_contention_state") or "unknown"),
        hicache_inflight_bytes=int(row.get("hicache_inflight_bytes") or 0),
    )


def _features_to_dict(features: GPUServiceFeatures) -> dict[str, Any]:
    return {
        "phase": features.phase,
        "request_demands": [
            {
                "sequence_tokens": item.sequence_tokens,
                "token_delta": item.token_delta,
                "cache_hit_ratio": item.cache_hit_ratio,
            }
            for item in features.request_demands
        ],
        "chunk_position": features.chunk_position,
        "prefill_decode_mixed": features.prefill_decode_mixed,
        "pcie_contention_state": features.pcie_contention_state,
        "hicache_inflight_bytes": features.hicache_inflight_bytes,
    }


def _evaluation_cell_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    features = _features_from_batch_row(row)
    return (
        features.phase,
        str(features.batch_size),
        str(_sequence_neighbor_bucket(features.sequence_tokens_mean)),
        str(_sequence_neighbor_bucket(float(features.sequence_tokens_max))),
        str(_integer_bucket(features.token_delta_total)),
        str(round(features.cache_hit_ratio_mean * 10.0) / 10.0),
        features.chunk_position,
        str(int(features.prefill_decode_mixed)),
        features.pcie_contention_state,
        str(_integer_bucket(features.hicache_inflight_bytes + 1)),
    )


def _features_from_dict(raw: Mapping[str, Any]) -> GPUServiceFeatures:
    return GPUServiceFeatures(
        phase=str(raw["phase"]),
        request_demands=tuple(
            GPURequestServiceDemand(
                sequence_tokens=int(item["sequence_tokens"]),
                token_delta=int(item["token_delta"]),
                cache_hit_ratio=float(item.get("cache_hit_ratio", 0.0)),
            )
            for item in raw["request_demands"]
        ),
        chunk_position=str(raw.get("chunk_position") or "unknown"),
        prefill_decode_mixed=bool(raw.get("prefill_decode_mixed", False)),
        pcie_contention_state=str(raw.get("pcie_contention_state") or "unknown"),
        hicache_inflight_bytes=int(raw.get("hicache_inflight_bytes") or 0),
    )


def _validate_fit_rows(values: list[dict[str, Any]]) -> None:
    if not values or {str(row.get("split")) for row in values} != {"train"}:
        raise ValueError("GPU service fitting requires only train rows")
    sample_ids = [str(row.get("sample_id") or "") for row in values]
    if any(not item for item in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("GPU service fitting requires unique batch sample_id values")
    invalid_roles = sorted(
        {
            str(row.get("evidence_role") or "unknown")
            for row in values
            if row.get("evidence_role") != CONTROLLED_SERVICE_EVIDENCE
        }
    )
    if invalid_roles:
        raise ValueError(
            "GPU service fitting accepts controlled microbenchmarks only; "
            f"found {invalid_roles}"
        )


def _profile_key(row: Mapping[str, Any]) -> str:
    profile_ids = tuple(
        sorted(str(value) for value in row.get("profile_ids", ()) if value)
    )
    if profile_ids:
        return json.dumps(profile_ids, separators=(",", ":"))
    case_ids = tuple(sorted(str(value) for value in row.get("case_ids", ()) if value))
    if case_ids:
        return json.dumps(case_ids, separators=(",", ":"))
    request_profiles = tuple(
        sorted(
            str(item.get("workflow_id") or item.get("request_id") or "")
            for item in row.get("request_samples", ())
        )
    )
    if any(request_profiles):
        return json.dumps(request_profiles, separators=(",", ":"))
    return str(row.get("sample_id") or "")


def _profile_fold_assignments(
    rows: list[dict[str, Any]], folds: int
) -> dict[str, int]:
    profiles = {_profile_key(row) for row in rows}
    ranked = sorted(
        profiles,
        key=lambda profile: hashlib.sha256(profile.encode("utf-8")).hexdigest(),
    )
    return {profile: index % folds for index, profile in enumerate(ranked)}


def _feature_distance(left: GPUServiceFeatures, right: GPUServiceFeatures) -> float:
    if left.phase != right.phase:
        return math.inf
    terms = (
        1.50 * _log_distance(left.batch_size, right.batch_size),
        0.60
        * _log_distance(
            int(left.sequence_tokens_mean) + 1,
            int(right.sequence_tokens_mean) + 1,
        ),
        0.35
        * _log_distance(left.sequence_tokens_max + 1, right.sequence_tokens_max + 1),
        0.90 * _log_distance(left.token_delta_total, right.token_delta_total),
        1.00 * abs(left.cache_hit_ratio_mean - right.cache_hit_ratio_mean),
        0.50 * float(left.chunk_position != right.chunk_position),
        1.50 * float(left.prefill_decode_mixed != right.prefill_decode_mixed),
        0.75 * float(left.pcie_contention_state != right.pcie_contention_state),
        0.20
        * _log_distance(
            left.hicache_inflight_bytes + 1,
            right.hicache_inflight_bytes + 1,
        ),
    )
    return math.sqrt(sum(value * value for value in terms))


def _log_distance(left: int, right: int) -> float:
    return abs(math.log2(max(1, left)) - math.log2(max(1, right)))


def _sequence_neighbor_bucket(value: float) -> int:
    return int(math.floor(math.log2(max(1.0, value + 1.0)) * 4.0))


def _weighted_sample_quantile(
    values: list[tuple[float, float]], quantile: float
) -> float:
    total = sum(weight for _, weight in values)
    if total <= 0:
        return 0.0
    threshold = total * quantile
    cumulative = 0.0
    for value, weight in sorted(values):
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(max(value for value, _ in values))


def _monotone_quantiles(
    p50: float, p90: float, p95: float
) -> tuple[float, float, float]:
    p50 = max(0.0, p50)
    p90 = max(p50, p90)
    p95 = max(p90, p95)
    return p50, p90, p95


def _percentile_required(values: list[float], quantile: float) -> float:
    result = _percentile(values, quantile)
    if result is None:
        raise ValueError("percentile requires observations")
    return result


def _mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(int(value) for value in values) / len(values)


def _backoff_keys(features: GPUServiceFeatures) -> tuple[tuple[str, ...], ...]:
    phase = features.phase
    batch = str(_integer_bucket(features.batch_size))
    sequence_mean = str(_integer_bucket(int(features.sequence_tokens_mean) + 1))
    sequence_max = str(_integer_bucket(features.sequence_tokens_max + 1))
    tokens = str(_integer_bucket(features.token_delta_total))
    cache = str(_ratio_bucket(features.cache_hit_ratio_mean))
    chunk = features.chunk_position
    mixed = str(int(features.prefill_decode_mixed))
    pcie = features.pcie_contention_state
    hicache = str(_integer_bucket(features.hicache_inflight_bytes + 1))
    return (
        (
            phase,
            batch,
            sequence_mean,
            sequence_max,
            tokens,
            cache,
            chunk,
            mixed,
            pcie,
            hicache,
        ),
        (phase, batch, sequence_mean, sequence_max, tokens, cache, chunk, mixed),
        (phase, batch, sequence_mean, sequence_max, tokens),
        (phase, batch),
        (phase,),
        ("*",),
    )


def _integer_bucket(value: int) -> int:
    return max(0, int(math.log2(max(1, value))))


def _ratio_bucket(value: float) -> float:
    bounded = min(1.0, max(0.0, value))
    return round(bounded * 4.0) / 4.0


def _log_bucket(value: float) -> float:
    if value <= 0:
        return 0.0
    exponent = round(math.log2(value) * 2.0) / 2.0
    return round(2.0**exponent, 6)


def _weighted_quantile(counts: Mapping[float, float], quantile: float) -> float:
    total = sum(counts.values())
    threshold = total * quantile
    cumulative = 0.0
    for value in sorted(counts):
        cumulative += counts[value]
        if cumulative >= threshold:
            return float(value)
    return float(max(counts, default=0.0))


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]
