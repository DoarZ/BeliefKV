from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Mapping

from beliefkv.predictor.types import RemainingTimePrediction


@dataclass(frozen=True)
class ToolDurationSample:
    duration_ms: float
    completed: bool
    family: str
    backend_class: str = "unknown"

    def __post_init__(self) -> None:
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        if not self.family:
            raise ValueError("family must be non-empty")


class KaplanMeierCurve:
    """Small dependency-free Kaplan-Meier estimator with censoring support."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.event_count = 0
        self._steps: list[tuple[float, float]] = []

    def fit(self, samples: list[tuple[float, bool]]) -> "KaplanMeierCurve":
        if not samples:
            self.sample_count = 0
            self.event_count = 0
            self._steps = []
            return self
        grouped: dict[float, list[bool]] = {}
        for duration, completed in samples:
            if duration < 0:
                raise ValueError("duration must be non-negative")
            grouped.setdefault(float(duration), []).append(bool(completed))
        at_risk = len(samples)
        survival = 1.0
        steps: list[tuple[float, float]] = []
        event_count = 0
        for duration in sorted(grouped):
            outcomes = grouped[duration]
            events = sum(outcomes)
            censored = len(outcomes) - events
            if events:
                survival *= 1.0 - events / at_risk
                event_count += events
                steps.append((duration, max(0.0, survival)))
            at_risk -= events + censored
        self.sample_count = len(samples)
        self.event_count = event_count
        self._steps = steps
        return self

    def survival(self, time_ms: float) -> float:
        if time_ms < 0:
            return 1.0
        value = 1.0
        for step_time, step_value in self._steps:
            if step_time > time_ms:
                break
            value = step_value
        return value

    def conditional_survival(self, elapsed_ms: float, future_ms: float) -> float:
        if elapsed_ms < 0 or future_ms < 0:
            raise ValueError("elapsed and future times must be non-negative")
        base = self.survival(elapsed_ms)
        if base <= 0:
            return 0.0
        return max(
            0.0,
            min(1.0, self.survival(elapsed_ms + future_ms) / base),
        )

    def remaining_quantile(self, elapsed_ms: float, q: float) -> float:
        if not 0 < q < 1:
            raise ValueError("q must be in (0, 1)")
        base = self.survival(elapsed_ms)
        if base <= 0:
            return 0.0
        threshold = 1.0 - q
        for step_time, step_survival in self._steps:
            if step_time < elapsed_ms:
                continue
            if step_survival / base <= threshold:
                return max(0.0, step_time - elapsed_ms)
        return inf

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "event_count": self.event_count,
            "steps": [[time_ms, survival] for time_ms, survival in self._steps],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "KaplanMeierCurve":
        curve = cls()
        curve.sample_count = int(raw.get("sample_count", 0))
        curve.event_count = int(raw.get("event_count", 0))
        steps = [
            (float(item[0]), float(item[1])) for item in raw.get("steps", [])
        ]
        if curve.sample_count < 0 or not 0 <= curve.event_count <= curve.sample_count:
            raise ValueError("invalid Kaplan-Meier sample/event counts")
        previous_time = -1.0
        previous_survival = 1.0
        for time_ms, survival in steps:
            if time_ms < 0 or time_ms < previous_time:
                raise ValueError("Kaplan-Meier steps must have monotonic times")
            if not 0 <= survival <= previous_survival:
                raise ValueError("Kaplan-Meier survival must be monotonic")
            previous_time = time_ms
            previous_survival = survival
        curve._steps = steps
        return curve


class HierarchicalToolSurvivalModel:
    """Global -> tool family -> exact backend survival backoff."""

    def __init__(
        self,
        *,
        min_family_samples: int = 8,
        min_backend_samples: int = 5,
    ) -> None:
        if min_family_samples <= 0 or min_backend_samples <= 0:
            raise ValueError("survival backoff sample thresholds must be positive")
        self.min_family_samples = min_family_samples
        self.min_backend_samples = min_backend_samples
        self.global_curve = KaplanMeierCurve()
        self.family_curves: dict[str, KaplanMeierCurve] = {}
        self.backend_curves: dict[tuple[str, str], KaplanMeierCurve] = {}
        self.known_families: set[str] = set()

    def fit(self, samples: list[ToolDurationSample]) -> None:
        self.global_curve.fit(
            [(sample.duration_ms, sample.completed) for sample in samples]
        )
        family_groups: dict[str, list[tuple[float, bool]]] = {}
        backend_groups: dict[tuple[str, str], list[tuple[float, bool]]] = {}
        for sample in samples:
            family_groups.setdefault(sample.family, []).append(
                (sample.duration_ms, sample.completed)
            )
            backend_groups.setdefault((sample.family, sample.backend_class), []).append(
                (sample.duration_ms, sample.completed)
            )
        self.family_curves = {
            key: KaplanMeierCurve().fit(values) for key, values in family_groups.items()
        }
        self.backend_curves = {
            key: KaplanMeierCurve().fit(values) for key, values in backend_groups.items()
        }
        self.known_families = set(family_groups)

    def predict(
        self,
        *,
        context_id: str,
        now_ms: float,
        elapsed_ms: float,
        family: str,
        backend_class: str = "unknown",
        transfer_window_ms: float,
    ) -> RemainingTimePrediction:
        curve, level = self._select_curve(family, backend_class)
        if curve.sample_count == 0:
            return RemainingTimePrediction(context_id=context_id, generated_ts_ms=now_ms)
        event_fraction = curve.event_count / curve.sample_count
        sample_scale = {
            "backend": self.min_backend_samples,
            "family": self.min_family_samples,
            "global": max(self.min_family_samples, 1),
        }[level]
        support = min(1.0, curve.sample_count / max(1, sample_scale * 2))
        confidence = support * event_fraction
        unknown_family = family not in self.known_families
        ood_score = 0.85 if unknown_family else {"backend": 0.05, "family": 0.2, "global": 0.55}[level]
        resume_probability = 1.0 - curve.conditional_survival(
            elapsed_ms, transfer_window_ms
        )
        return RemainingTimePrediction(
            context_id=context_id,
            generated_ts_ms=now_ms,
            p50_ms=curve.remaining_quantile(elapsed_ms, 0.5),
            p90_ms=curve.remaining_quantile(elapsed_ms, 0.9),
            p95_ms=curve.remaining_quantile(elapsed_ms, 0.95),
            resume_within_transfer_probability=resume_probability,
            next_event_distribution={"tool_end": 1.0},
            confidence=max(0.0, min(1.0, confidence)),
            ood_score=ood_score,
            backoff_level=level,
        )

    def _select_curve(
        self, family: str, backend_class: str
    ) -> tuple[KaplanMeierCurve, str]:
        backend = self.backend_curves.get((family, backend_class))
        if backend is not None and backend.sample_count >= self.min_backend_samples:
            return backend, "backend"
        family_curve = self.family_curves.get(family)
        if family_curve is not None and family_curve.sample_count >= self.min_family_samples:
            return family_curve, "family"
        return self.global_curve, "global"

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_family_samples": self.min_family_samples,
            "min_backend_samples": self.min_backend_samples,
            "global_curve": self.global_curve.to_dict(),
            "family_curves": {
                key: curve.to_dict() for key, curve in sorted(self.family_curves.items())
            },
            "backend_curves": [
                {
                    "family": family,
                    "backend_class": backend,
                    "curve": curve.to_dict(),
                }
                for (family, backend), curve in sorted(self.backend_curves.items())
            ],
            "known_families": sorted(self.known_families),
        }

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any]
    ) -> "HierarchicalToolSurvivalModel":
        model = cls(
            min_family_samples=int(raw.get("min_family_samples", 8)),
            min_backend_samples=int(raw.get("min_backend_samples", 5)),
        )
        model.global_curve = KaplanMeierCurve.from_dict(raw.get("global_curve", {}))
        model.family_curves = {
            str(key): KaplanMeierCurve.from_dict(value)
            for key, value in dict(raw.get("family_curves", {})).items()
        }
        model.backend_curves = {
            (str(item["family"]), str(item["backend_class"])): KaplanMeierCurve.from_dict(
                item["curve"]
            )
            for item in raw.get("backend_curves", [])
        }
        model.known_families = set(raw.get("known_families", model.family_curves))
        return model
