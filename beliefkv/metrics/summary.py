from __future__ import annotations


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    index = round((q / 100.0) * (len(ordered) - 1))
    return ordered[index]
