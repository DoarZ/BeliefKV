from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import inf
from typing import Any, Mapping

from beliefkv.predictor.taxonomy import ActionKind
from beliefkv.predictor.tool_survival import KaplanMeierCurve


@dataclass(frozen=True)
class ActionObservation:
    action: ActionKind
    duration_ms: float
    completed: bool = True


@dataclass(frozen=True)
class ActionPrediction:
    next_distribution: dict[ActionKind, float]
    selected_order: int
    support: int
    current_remaining_p50_ms: float
    current_remaining_p95_ms: float
    confidence: float
    ood_score: float


class SemiMarkovContextTree:
    """Variable-order action model with per-state survival durations."""

    def __init__(
        self,
        *,
        max_order: int = 4,
        min_context_count: int = 3,
        smoothing: float = 0.5,
    ) -> None:
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        if min_context_count <= 0:
            raise ValueError("min_context_count must be positive")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.max_order = max_order
        self.min_context_count = min_context_count
        self.smoothing = smoothing
        self._counts: dict[tuple[ActionKind, ...], Counter[ActionKind]] = {}
        self._durations: dict[ActionKind, KaplanMeierCurve] = {}
        self._vocabulary: set[ActionKind] = set()

    def fit(self, trajectories: list[list[ActionObservation]]) -> None:
        counts: dict[tuple[ActionKind, ...], Counter[ActionKind]] = {}
        durations: dict[ActionKind, list[tuple[float, bool]]] = {}
        vocabulary: set[ActionKind] = set()
        for trajectory in trajectories:
            history: list[ActionKind] = []
            for observation in trajectory:
                vocabulary.add(observation.action)
                durations.setdefault(observation.action, []).append(
                    (observation.duration_ms, observation.completed)
                )
                for order in range(0, min(self.max_order, len(history)) + 1):
                    context = tuple(history[-order:]) if order else ()
                    counts.setdefault(context, Counter())[observation.action] += 1
                history.append(observation.action)
        self._counts = counts
        self._durations = {
            action: KaplanMeierCurve().fit(samples)
            for action, samples in durations.items()
        }
        self._vocabulary = vocabulary

    def predict(
        self,
        history: list[ActionKind] | tuple[ActionKind, ...],
        *,
        current_action: ActionKind | None = None,
        elapsed_ms: float = 0.0,
    ) -> ActionPrediction:
        selected_context: tuple[ActionKind, ...] = ()
        selected_counts = self._counts.get((), Counter())
        max_order = min(self.max_order, len(history))
        for order in range(max_order, 0, -1):
            context = tuple(history[-order:])
            counts = self._counts.get(context)
            if counts is not None and sum(counts.values()) >= self.min_context_count:
                selected_context = context
                selected_counts = counts
                break
        support = sum(selected_counts.values())
        distribution = self._smoothed_distribution(selected_counts)
        duration_curve = self._durations.get(current_action) if current_action else None
        if duration_curve is None:
            p50 = p95 = inf
        else:
            p50 = duration_curve.remaining_quantile(elapsed_ms, 0.5)
            p95 = duration_curve.remaining_quantile(elapsed_ms, 0.95)
        confidence = min(1.0, support / max(1, self.min_context_count * 4))
        if selected_context:
            confidence *= 1.0 - 0.08 * (self.max_order - len(selected_context))
        confidence = max(0.0, confidence)
        unknown_actions = sum(action not in self._vocabulary for action in history)
        ood_score = min(
            1.0,
            unknown_actions / max(1, len(history))
            + (0.15 if not selected_context and history else 0.0),
        )
        return ActionPrediction(
            next_distribution=distribution,
            selected_order=len(selected_context),
            support=support,
            current_remaining_p50_ms=p50,
            current_remaining_p95_ms=p95,
            confidence=confidence,
            ood_score=ood_score,
        )

    def _smoothed_distribution(
        self, counts: Counter[ActionKind]
    ) -> dict[ActionKind, float]:
        if not self._vocabulary:
            return {}
        global_counts = self._counts.get((), Counter())
        global_total = sum(global_counts.values())
        local_total = sum(counts.values())
        denominator = local_total + self.smoothing
        result: dict[ActionKind, float] = {}
        for action in sorted(self._vocabulary, key=lambda item: item.value):
            prior = (
                global_counts[action] / global_total
                if global_total
                else 1.0 / len(self._vocabulary)
            )
            result[action] = (counts[action] + self.smoothing * prior) / denominator
        normalizer = sum(result.values())
        return {key: value / normalizer for key, value in result.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_order": self.max_order,
            "min_context_count": self.min_context_count,
            "smoothing": self.smoothing,
            "counts": [
                {
                    "context": [action.value for action in context],
                    "next": {
                        action.value: count
                        for action, count in sorted(
                            counts.items(), key=lambda item: item[0].value
                        )
                    },
                }
                for context, counts in sorted(
                    self._counts.items(),
                    key=lambda item: (len(item[0]), tuple(x.value for x in item[0])),
                )
            ],
            "durations": {
                action.value: curve.to_dict()
                for action, curve in sorted(
                    self._durations.items(), key=lambda item: item[0].value
                )
            },
            "vocabulary": sorted(action.value for action in self._vocabulary),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SemiMarkovContextTree":
        model = cls(
            max_order=int(raw.get("max_order", 4)),
            min_context_count=int(raw.get("min_context_count", 3)),
            smoothing=float(raw.get("smoothing", 0.5)),
        )
        model._counts = {
            tuple(ActionKind(value) for value in item.get("context", [])): Counter(
                {
                    ActionKind(action): int(count)
                    for action, count in dict(item.get("next", {})).items()
                }
            )
            for item in raw.get("counts", [])
        }
        model._durations = {
            ActionKind(action): KaplanMeierCurve.from_dict(curve)
            for action, curve in dict(raw.get("durations", {})).items()
        }
        model._vocabulary = {
            ActionKind(value) for value in raw.get("vocabulary", [])
        }
        if not model._vocabulary:
            model._vocabulary = {
                action for counts in model._counts.values() for action in counts
            }
        return model
