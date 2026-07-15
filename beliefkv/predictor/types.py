from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Mapping


@dataclass(frozen=True)
class RemainingTimePrediction:
    context_id: str
    generated_ts_ms: float
    p50_ms: float = inf
    p90_ms: float = inf
    p95_ms: float = inf
    resume_within_transfer_probability: float = 0.0
    next_event_distribution: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    ood_score: float = 1.0
    backoff_level: str = "reactive"

    @property
    def usable(self) -> bool:
        return self.confidence > 0.0 and self.ood_score < 1.0
