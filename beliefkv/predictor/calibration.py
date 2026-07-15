from __future__ import annotations

from collections import deque
from dataclasses import replace
from math import isfinite

from beliefkv.predictor.types import RemainingTimePrediction


class RollingIntervalCalibrator:
    """Guard prediction use when empirical p95 coverage degrades."""

    def __init__(
        self,
        *,
        window_size: int = 256,
        target_p95_coverage: float = 0.90,
        min_observations: int = 20,
    ) -> None:
        if window_size <= 0 or min_observations <= 0:
            raise ValueError("calibration window and support must be positive")
        if not 0 < target_p95_coverage < 1:
            raise ValueError("target coverage must be in (0, 1)")
        self.window_size = window_size
        self.target_p95_coverage = target_p95_coverage
        self.min_observations = min_observations
        self._coverage: deque[bool] = deque(maxlen=window_size)
        self._ratios: deque[float] = deque(maxlen=window_size)

    def observe(self, prediction: RemainingTimePrediction, actual_ms: float) -> None:
        if actual_ms < 0:
            raise ValueError("actual_ms must be non-negative")
        if isfinite(prediction.p95_ms):
            self._coverage.append(actual_ms <= prediction.p95_ms)
            if prediction.p95_ms > 0:
                self._ratios.append(actual_ms / prediction.p95_ms)

    @property
    def coverage(self) -> float:
        if not self._coverage:
            return 1.0
        return sum(self._coverage) / len(self._coverage)

    @property
    def observation_count(self) -> int:
        return len(self._coverage)

    def adjust(self, prediction: RemainingTimePrediction) -> RemainingTimePrediction:
        if len(self._coverage) < self.min_observations:
            return prediction
        coverage = self.coverage
        if coverage >= self.target_p95_coverage:
            return prediction
        sorted_ratios = sorted(self._ratios)
        index = min(len(sorted_ratios) - 1, int(0.95 * len(sorted_ratios)))
        expansion = max(1.0, sorted_ratios[index]) if sorted_ratios else 1.25
        confidence_scale = max(0.1, coverage / self.target_p95_coverage)
        return replace(
            prediction,
            p50_ms=prediction.p50_ms * expansion,
            p90_ms=prediction.p90_ms * expansion,
            p95_ms=prediction.p95_ms * expansion,
            confidence=prediction.confidence * confidence_scale,
            ood_score=max(prediction.ood_score, 1.0 - confidence_scale),
            backoff_level=f"{prediction.backoff_level}+calibrated",
        )
