from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, Sequence


def _finite_values(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if any(value != value or value in {float("inf"), float("-inf")} for value in result):
        raise ValueError("metric samples must be finite")
    return result


def mean(values: Sequence[float]) -> float:
    samples = _finite_values(values)
    return sum(samples) / len(samples) if samples else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile on the inclusive [0, 100] scale."""

    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(_finite_values(values))
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q / 100.0
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    sample_count: int


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Compute a deterministic non-parametric percentile bootstrap CI."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    samples = _finite_values(values)
    if not samples:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence, 0)
    estimate = mean(samples)
    if len(samples) == 1:
        return ConfidenceInterval(estimate, estimate, estimate, confidence, 1)
    rng = random.Random(seed)
    size = len(samples)
    bootstrapped = [
        sum(samples[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=estimate,
        lower=percentile(bootstrapped, tail * 100.0),
        upper=percentile(bootstrapped, (1.0 - tail) * 100.0),
        confidence=confidence,
        sample_count=size,
    )


def jain_fairness(values: Sequence[float]) -> float:
    """Return Jain's fairness index for non-negative service allocations."""

    samples = _finite_values(values)
    if any(value < 0 for value in samples):
        raise ValueError("fairness samples must be non-negative")
    if not samples:
        return 0.0
    squared_sum = sum(samples) ** 2
    sum_squares = sum(value * value for value in samples)
    return squared_sum / (len(samples) * sum_squares) if sum_squares else 1.0
