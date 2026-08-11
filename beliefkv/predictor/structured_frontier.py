from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from beliefkv.control.causal_graph import InvocationState, JoinMode, RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEventKind
from beliefkv.predictor.frontier_belief import (
    BeliefScope,
    BoundaryEvent,
    DemandPhase,
    DemandScenario,
    DependencyMode,
    ExternalDemandSegment,
    FinitePlanningHorizon,
    FrontierDemandOutcome,
    FrontierBeliefSnapshot,
    OtherResidualPolicy,
    PredictiveEvidenceReadSet,
    ScenarioProjection,
)


STRUCTURED_FRONTIER_SCHEMA_VERSION = 2
MINIMUM_DEMAND_DECISION_SCHEMA_VERSION = 2
FORMAL_P6_DATASET_KIND = "beliefkv_p6_training_evidence"
FORMAL_P6_PLAN_ID = "p6-agent-semantics-v1"
FORBIDDEN_LOAD_COUPLED_LABELS = frozenset(
    {"remaining_gpu_service_ms", "next_gpu_service_ms"}
)
FORBIDDEN_LOAD_COUPLED_FEATURES = frozenset(
    {"batch_size", "elapsed_gpu_service_ms", "observed_gpu_service_ms"}
)


@dataclass(frozen=True)
class FrontierModelHyperparameters:
    boundary_max_order: int = 4
    boundary_minimum_support: float = 3.0
    boundary_smoothing: float = 0.5
    empirical_minimum_support: float = 4.0
    tool_minimum_support: float = 4.0
    tool_smoothing: float = 0.5

    def __post_init__(self) -> None:
        if self.boundary_max_order < 0:
            raise ValueError("boundary_max_order must be non-negative")
        if min(
            self.boundary_minimum_support,
            self.boundary_smoothing,
            self.empirical_minimum_support,
            self.tool_minimum_support,
            self.tool_smoothing,
        ) <= 0:
            raise ValueError("frontier hyperparameters must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "boundary_max_order": self.boundary_max_order,
            "boundary_minimum_support": self.boundary_minimum_support,
            "boundary_smoothing": self.boundary_smoothing,
            "empirical_minimum_support": self.empirical_minimum_support,
            "tool_minimum_support": self.tool_minimum_support,
            "tool_smoothing": self.tool_smoothing,
        }

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any] | None
    ) -> "FrontierModelHyperparameters":
        values = raw or {}
        return cls(
            boundary_max_order=int(values.get("boundary_max_order", 4)),
            boundary_minimum_support=float(
                values.get("boundary_minimum_support", 3.0)
            ),
            boundary_smoothing=float(values.get("boundary_smoothing", 0.5)),
            empirical_minimum_support=float(
                values.get("empirical_minimum_support", 4.0)
            ),
            tool_minimum_support=float(values.get("tool_minimum_support", 4.0)),
            tool_smoothing=float(values.get("tool_smoothing", 0.5)),
        )


@dataclass(frozen=True)
class LocalFrontierFeatures:
    invocation_id: str
    state: str
    agent_definition_id: str = "unknown"
    boundary_history: tuple[str, ...] = ()
    tool_family: str = "unknown"
    backend_class: str = "unknown"
    generated_tokens: int = 0
    elapsed_wait_ms: float = 0.0
    current_sequence_tokens: int = 0
    active_tool_count: int = 0
    backend_pressure: str = "unknown"

    def __post_init__(self) -> None:
        if min(
            self.generated_tokens,
            self.current_sequence_tokens,
            self.active_tool_count,
        ) < 0:
            raise ValueError("frontier demand features must be non-negative")


@dataclass(frozen=True)
class EmpiricalDistribution:
    values: tuple[float, ...]
    probability_mass: tuple[float, ...]
    support: float

    def __post_init__(self) -> None:
        if len(self.values) != len(self.probability_mass):
            raise ValueError("empirical values and probabilities must align")
        if self.support < 0 or any(value < 0 for value in self.values):
            raise ValueError("empirical distributions require non-negative values")
        if self.values and not math.isclose(
            sum(self.probability_mass), 1.0, rel_tol=1e-7, abs_tol=1e-7
        ):
            raise ValueError("empirical probability mass must sum to one")

    @classmethod
    def empty(cls) -> "EmpiricalDistribution":
        return cls((), (), 0.0)

    def sample(self, quantile: float) -> float:
        if not self.values:
            return 0.0
        threshold = min(1.0, max(0.0, quantile))
        cumulative = 0.0
        for value, probability in zip(self.values, self.probability_mass):
            cumulative += probability
            if threshold <= cumulative:
                return value
        return self.values[-1]

    def quantile(self, quantile: float) -> float:
        return self.sample(quantile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "probability_mass": list(self.probability_mass),
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EmpiricalDistribution":
        return cls(
            tuple(float(item) for item in raw.get("values", ())),
            tuple(float(item) for item in raw.get("probability_mass", ())),
            float(raw.get("support", 0.0)),
        )


@dataclass(frozen=True)
class LocalFrontierPrediction:
    invocation_id: str
    boundary_distribution: Mapping[str, float]
    current_sequence_tokens: int
    remaining_decode_tokens: EmpiricalDistribution
    remaining_external_wait: EmpiricalDistribution
    tool_terminal_distribution: Mapping[str, float]
    prompt_growth_tokens: EmpiricalDistribution
    next_output_tokens: EmpiricalDistribution
    support_level: str
    calibration_coverage: float
    ood_reasons: tuple[str, ...] = ()
    calibrated_intervals: Mapping[str, tuple[float, float]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "boundary_distribution": dict(self.boundary_distribution),
            "current_sequence_tokens": self.current_sequence_tokens,
            "remaining_decode_tokens": self.remaining_decode_tokens.to_dict(),
            "remaining_external_wait": self.remaining_external_wait.to_dict(),
            "tool_terminal_distribution": dict(self.tool_terminal_distribution),
            "prompt_growth_tokens": self.prompt_growth_tokens.to_dict(),
            "next_output_tokens": self.next_output_tokens.to_dict(),
            "support_level": self.support_level,
            "calibration_coverage": self.calibration_coverage,
            "ood_reasons": list(self.ood_reasons),
            "calibrated_intervals": {
                name: list(interval)
                for name, interval in sorted(self.calibrated_intervals.items())
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LocalFrontierPrediction":
        intervals = raw.get("calibrated_intervals", {})
        return cls(
            invocation_id=str(raw["invocation_id"]),
            boundary_distribution={
                str(name): float(probability)
                for name, probability in raw.get(
                    "boundary_distribution", {}
                ).items()
            },
            current_sequence_tokens=int(raw.get("current_sequence_tokens", 0)),
            remaining_decode_tokens=EmpiricalDistribution.from_dict(
                raw.get("remaining_decode_tokens", {})
            ),
            remaining_external_wait=EmpiricalDistribution.from_dict(
                raw.get("remaining_external_wait", {})
            ),
            tool_terminal_distribution={
                str(name): float(probability)
                for name, probability in raw.get(
                    "tool_terminal_distribution", {}
                ).items()
            },
            prompt_growth_tokens=EmpiricalDistribution.from_dict(
                raw.get("prompt_growth_tokens", {})
            ),
            next_output_tokens=EmpiricalDistribution.from_dict(
                raw.get("next_output_tokens", {})
            ),
            support_level=str(raw.get("support_level", "unavailable")),
            calibration_coverage=float(raw.get("calibration_coverage", 0.0)),
            ood_reasons=tuple(str(item) for item in raw.get("ood_reasons", ())),
            calibrated_intervals={
                str(name): (float(value[0]), float(value[1]))
                for name, value in intervals.items()
            },
        )


class _HierarchicalEmpiricalModel:
    def __init__(self, *, minimum_support: float = 4.0) -> None:
        self.minimum_support = minimum_support
        self._groups: dict[tuple[str, ...], Counter[float]] = defaultdict(Counter)

    def observe(self, key: Sequence[str], value: float, *, weight: float = 1.0) -> None:
        if value < 0 or weight <= 0:
            return
        bucket = _log_bucket(value)
        for candidate in _backoff_keys(tuple(key)):
            self._groups[candidate][bucket] += weight

    def predict(self, key: Sequence[str]) -> tuple[EmpiricalDistribution, str]:
        candidates = _backoff_keys(tuple(key))
        selected = candidates[-1]
        for candidate in candidates:
            support = sum(self._groups.get(candidate, {}).values())
            if support >= self.minimum_support:
                selected = candidate
                break
        counts = self._groups.get(selected, Counter())
        total = sum(counts.values())
        if total <= 0:
            return EmpiricalDistribution.empty(), "unavailable"
        values = tuple(sorted(counts))
        distribution = EmpiricalDistribution(
            values,
            tuple(counts[value] / total for value in values),
            total,
        )
        level = "exact" if selected == candidates[0] else (
            "global" if selected == ("*",) else "backoff"
        )
        return distribution, level

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_support": self.minimum_support,
            "groups": [
                {
                    "key": list(key),
                    "counts": {str(value): count for value, count in sorted(counts.items())},
                }
                for key, counts in sorted(self._groups.items())
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "_HierarchicalEmpiricalModel":
        model = cls(minimum_support=float(raw.get("minimum_support", 4.0)))
        for item in raw.get("groups", ()):
            model._groups[tuple(str(value) for value in item["key"])] = Counter(
                {float(value): float(count) for value, count in item["counts"].items()}
            )
        return model


class _BoundaryContextTree:
    def __init__(
        self, *, max_order: int = 4, minimum_support: float = 3.0, smoothing: float = 0.5
    ) -> None:
        self.max_order = max_order
        self.minimum_support = minimum_support
        self.smoothing = smoothing
        self._counts: dict[tuple[str, str, tuple[str, ...]], Counter[str]] = defaultdict(Counter)

    def observe(
        self,
        *,
        role: str,
        state: str,
        history: Sequence[str],
        target: str,
        weight: float,
    ) -> None:
        history = tuple(history)
        for order in range(min(self.max_order, len(history)) + 1):
            context = history[-order:] if order else ()
            self._counts[(role, state, context)][target] += weight
            self._counts[("*", state, context)][target] += weight

    def predict(
        self, *, role: str, state: str, history: Sequence[str]
    ) -> tuple[dict[str, float], str, float]:
        history = tuple(history)
        selected: tuple[str, str, tuple[str, ...]] | None = None
        for candidate_role in (role, "*"):
            for order in range(min(self.max_order, len(history)), -1, -1):
                candidate = (candidate_role, state, history[-order:] if order else ())
                if sum(self._counts.get(candidate, {}).values()) >= self.minimum_support:
                    selected = candidate
                    break
            if selected is not None:
                break
        global_counts = self._counts.get(("*", state, ()), Counter())
        counts = self._counts.get(selected, Counter()) if selected else global_counts
        vocabulary = tuple(
            item.value
            for item in (
                BoundaryEvent.TOOL,
                BoundaryEvent.SPAWN,
                BoundaryEvent.HANDOFF,
                BoundaryEvent.RETURN,
                BoundaryEvent.FINAL,
            )
        )
        if not global_counts:
            return {BoundaryEvent.UNKNOWN.value: 1.0}, "unavailable", 0.0
        total = sum(counts.values())
        global_total = sum(global_counts.values())
        probabilities = {
            target: (
                counts[target]
                + self.smoothing
                * (
                    global_counts[target] + self.smoothing / len(vocabulary)
                )
                / (global_total + self.smoothing)
            )
            / (total + self.smoothing)
            for target in vocabulary
        }
        normalizer = sum(probabilities.values())
        probabilities = {key: value / normalizer for key, value in probabilities.items()}
        level = (
            "exact"
            if selected and selected[0] == role and selected[2]
            else "role"
            if selected and selected[0] == role
            else "global"
        )
        return probabilities, level, total

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_order": self.max_order,
            "minimum_support": self.minimum_support,
            "smoothing": self.smoothing,
            "counts": [
                {
                    "role": role,
                    "state": state,
                    "history": list(history),
                    "targets": dict(sorted(counts.items())),
                }
                for (role, state, history), counts in sorted(self._counts.items())
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "_BoundaryContextTree":
        model = cls(
            max_order=int(raw.get("max_order", 4)),
            minimum_support=float(raw.get("minimum_support", 3.0)),
            smoothing=float(raw.get("smoothing", 0.5)),
        )
        for item in raw.get("counts", ()):
            key = (
                str(item["role"]),
                str(item["state"]),
                tuple(str(value) for value in item.get("history", ())),
            )
            model._counts[key] = Counter(
                {str(target): float(count) for target, count in item["targets"].items()}
            )
        return model


class _CompetingRiskToolModel:
    def __init__(self, *, minimum_support: float = 4.0, smoothing: float = 0.5) -> None:
        self.minimum_support = minimum_support
        self.smoothing = smoothing
        self._status: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._wait = _HierarchicalEmpiricalModel(minimum_support=minimum_support)
        self._duration_by_status: dict[
            tuple[str, ...], dict[str, Counter[float]]
        ] = defaultdict(lambda: defaultdict(Counter))

    def observe(
        self,
        key: Sequence[str],
        *,
        status: str,
        duration_ms: float,
        weight: float = 1.0,
    ) -> None:
        normalized = _normalize_tool_terminal(status)
        bucket = _log_bucket(duration_ms)
        for candidate in _backoff_keys(tuple(key)):
            self._status[candidate][normalized] += weight
            self._duration_by_status[candidate][normalized][bucket] += weight
        self._wait.observe(key, duration_ms, weight=weight)

    def predict(
        self, key: Sequence[str], *, elapsed_ms: float = 0.0
    ) -> tuple[dict[str, float], EmpiricalDistribution, str]:
        candidates = _backoff_keys(tuple(key))
        selected = candidates[-1]
        for candidate in candidates:
            if sum(self._status.get(candidate, {}).values()) >= self.minimum_support:
                selected = candidate
                break
        duration_by_status = self._duration_by_status.get(selected, {})
        has_joint_observations = bool(duration_by_status)
        survivors: Counter[str] = Counter()
        residuals: Counter[float] = Counter()
        elapsed_ms = max(0.0, float(elapsed_ms))
        for status, durations in duration_by_status.items():
            for duration, weight in durations.items():
                if duration + 1e-9 < elapsed_ms:
                    continue
                survivors[status] += weight
                residuals[_log_bucket(max(0.0, duration - elapsed_ms))] += weight

        # Schema-v1 models have no joint duration/status observations. Preserve
        # their unconditional behavior for development-artifact compatibility.
        counts = (
            survivors
            if has_joint_observations
            else self._status.get(selected, Counter())
        )
        total = sum(counts.values())
        if total <= 0:
            statuses = {"censored": 1.0}
            level = "unavailable"
        else:
            vocabulary = ("success", "error", "censored")
            statuses = {
                item: (counts[item] + self.smoothing / len(vocabulary))
                / (total + self.smoothing)
                for item in vocabulary
            }
            level = "exact" if selected == candidates[0] else (
                "global" if selected == ("*",) else "backoff"
            )
        if residuals:
            residual_total = sum(residuals.values())
            wait = EmpiricalDistribution(
                tuple(sorted(residuals)),
                tuple(
                    residuals[value] / residual_total for value in sorted(residuals)
                ),
                residual_total,
            )
            wait_level = level
        elif not has_joint_observations:
            wait, wait_level = self._wait.predict(key)
        else:
            wait = EmpiricalDistribution.empty()
            wait_level = "unavailable"
        return statuses, wait, level if level != "unavailable" else wait_level

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_support": self.minimum_support,
            "smoothing": self.smoothing,
            "status": [
                {"key": list(key), "counts": dict(sorted(counts.items()))}
                for key, counts in sorted(self._status.items())
            ],
            "duration_by_status": [
                {
                    "key": list(key),
                    "durations": {
                        status: {
                            str(duration): count
                            for duration, count in sorted(counts.items())
                        }
                        for status, counts in sorted(by_status.items())
                    },
                }
                for key, by_status in sorted(self._duration_by_status.items())
            ],
            "wait": self._wait.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "_CompetingRiskToolModel":
        model = cls(
            minimum_support=float(raw.get("minimum_support", 4.0)),
            smoothing=float(raw.get("smoothing", 0.5)),
        )
        for item in raw.get("status", ()):
            model._status[tuple(str(value) for value in item["key"])] = Counter(
                {str(status): float(count) for status, count in item["counts"].items()}
            )
        for item in raw.get("duration_by_status", ()):
            key = tuple(str(value) for value in item["key"])
            model._duration_by_status[key] = defaultdict(
                Counter,
                {
                    str(status): Counter(
                        {
                            float(duration): float(count)
                            for duration, count in counts.items()
                        }
                    )
                    for status, counts in item.get("durations", {}).items()
                },
            )
        model._wait = _HierarchicalEmpiricalModel.from_dict(raw.get("wait", {}))
        return model


class FrontierBeliefModel:
    """One versioned model that publishes local beliefs but never scheduling actions."""

    def __init__(
        self,
        *,
        model_version: str = "frontier-development",
        hyperparameters: FrontierModelHyperparameters | None = None,
    ) -> None:
        self.model_version = model_version
        self.hyperparameters = hyperparameters or FrontierModelHyperparameters()
        self.boundary = _BoundaryContextTree(
            max_order=self.hyperparameters.boundary_max_order,
            minimum_support=self.hyperparameters.boundary_minimum_support,
            smoothing=self.hyperparameters.boundary_smoothing,
        )
        self.decode_demand = _HierarchicalEmpiricalModel(
            minimum_support=self.hyperparameters.empirical_minimum_support
        )
        self.next_output = _HierarchicalEmpiricalModel(
            minimum_support=self.hyperparameters.empirical_minimum_support
        )
        self.prompt_growth = _HierarchicalEmpiricalModel(
            minimum_support=self.hyperparameters.empirical_minimum_support
        )
        self.tool = _CompetingRiskToolModel(
            minimum_support=self.hyperparameters.tool_minimum_support,
            smoothing=self.hyperparameters.tool_smoothing,
        )
        self.join_wait = _HierarchicalEmpiricalModel(
            minimum_support=self.hyperparameters.empirical_minimum_support
        )
        self.training_summary: dict[str, Any] = {}
        self.calibration_summary: dict[str, Any] = {}
        self.boundary_temperature = 1.0
        self.tool_temperature = 1.0
        self.interval_slack: dict[str, float] = {}
        self.calibration_coverage = 0.0

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        values = [dict(row) for row in rows]
        _validate_demand_rows(values)
        episode_counts = Counter(
            str(row.get("episode_group_id") or row.get("decision_id")) for row in values
        )
        local_episode_counts = _local_episode_counts(values)
        workflow_episode_counts = _workflow_local_episode_counts(values)
        observed = Counter()
        split_counts = Counter(str(row.get("split") or "unknown") for row in values)
        for row in values:
            episode = str(row.get("episode_group_id") or row.get("decision_id"))
            labels = {
                str(item.get("invocation_id")): item for item in row.get("labels", ())
            }
            trigger = str(row.get("trigger_kind") or "")
            trigger_attrs = row.get("trigger_attributes") or {}
            for features in row.get("invocations", ()):
                invocation_id = str(features.get("invocation_id") or "")
                label = labels.get(invocation_id)
                if label is None:
                    continue
                weight = 1.0 / max(
                    1, local_episode_counts[(episode, invocation_id)]
                )
                workflow = _workflow_group_id(row)
                weight /= max(1, workflow_episode_counts[workflow])
                role = str(features.get("agent_definition_id") or "unknown")
                state = str(features.get("state") or "unknown")
                family = str(
                    trigger_attrs.get("tool_family")
                    or features.get("active_tool_family")
                    or "unknown"
                )
                key = _demand_feature_key(role, state, family, features)
                boundary = _normalize_boundary(label.get("next_boundary_kind"))
                if state == InvocationState.RUNNING_LLM.value and boundary is not None:
                    self.boundary.observe(
                        role=role,
                        state=state,
                        history=features.get("boundary_history", ()),
                        target=boundary,
                        weight=weight,
                    )
                    observed["boundary"] += 1
                remaining_decode = (
                    label.get("remaining_output_tokens")
                    if state == InvocationState.RUNNING_LLM.value
                    else None
                )
                if remaining_decode is not None:
                    self.decode_demand.observe(
                        key, float(remaining_decode), weight=weight
                    )
                    observed["remaining_decode_demand"] += 1
                elif state != InvocationState.RUNNING_LLM.value:
                    # State-semantic decode target (P6 improvement, deepseek):
                    # for non-running states the scheduler-relevant quantity is
                    # the decode work of the *next* LLM call after the boundary
                    # (tool result / join / child / ready dispatch). Without
                    # this, (role, state, ...) keys have no support and every
                    # non-running invocation falls to the global bucket,
                    # producing constant p50 predictions for the scheduler.
                    next_decode = label.get("next_output_tokens")
                    if next_decode is not None:
                        self.decode_demand.observe(
                            key, float(next_decode), weight=weight
                        )
                        observed["state_conditional_decode_demand"] += 1
                next_output = label.get("next_output_tokens")
                if next_output is not None:
                    self.next_output.observe(key, float(next_output), weight=weight)
                    observed["next_output_demand"] += 1
                prompt_growth = label.get("reentry_prompt_delta_tokens")
                if prompt_growth is not None:
                    self.prompt_growth.observe(key, float(prompt_growth), weight=weight)
                    observed["prompt_growth"] += 1
                if trigger == RuntimeEventKind.TOOL_START.value and state == InvocationState.WAIT_TOOL.value:
                    status = str(label.get("next_boundary_status") or "error")
                    if label.get("censored"):
                        status = "censored"
                    delay = label.get("next_boundary_delay_ms")
                    if delay is not None:
                        self.tool.observe(
                            _tool_feature_key(role, family, features),
                            status=status,
                            duration_ms=float(delay),
                            weight=1.0,
                        )
                        observed["tool"] += 1
                if state == InvocationState.WAIT_JOIN.value:
                    delay = label.get("next_boundary_delay_ms")
                    if delay is not None:
                        self.join_wait.observe(
                            _demand_feature_key(role, state, "join", features),
                            float(delay),
                            weight=1.0,
                        )
                        observed["join_wait"] += 1
        self.training_summary = {
            "decision_point_count": len(values),
            "episode_count": len(episode_counts),
            "local_episode_count": len(local_episode_counts),
            "workflow_count": len(workflow_episode_counts),
            "split_counts": dict(sorted(split_counts.items())),
            "observation_counts": dict(sorted(observed.items())),
            "episode_weighting": (
                "decision points are normalized within each local episode, then "
                "local episodes are normalized within each workflow rollout"
            ),
        }
        return dict(self.training_summary)

    def predict(self, features: LocalFrontierFeatures) -> LocalFrontierPrediction:
        key = _demand_feature_key(
            features.agent_definition_id,
            features.state,
            features.tool_family,
            {
                "current_sequence_tokens": features.current_sequence_tokens,
                "generated_tokens": features.generated_tokens,
                "backend_class": features.backend_class,
            },
        )
        boundary, boundary_level, boundary_support = self.boundary.predict(
            role=features.agent_definition_id,
            state=features.state,
            history=features.boundary_history,
        )
        boundary = _temperature_scale(boundary, self.boundary_temperature)
        decode, decode_level = self.decode_demand.predict(key)
        output, output_level = self.next_output.predict(key)
        prompt, prompt_level = self.prompt_growth.predict(key)
        if features.state == InvocationState.WAIT_JOIN.value:
            wait, join_level = self.join_wait.predict(
                _demand_feature_key(
                    features.agent_definition_id,
                    features.state,
                    "join",
                    {
                        "current_sequence_tokens": features.current_sequence_tokens,
                        "backend_class": features.backend_class,
                    },
                )
            )
            terminal = {"success": 1.0}
            tool_level = join_level
        else:
            terminal, wait, tool_level = self.tool.predict(
                _tool_feature_key(
                    features.agent_definition_id,
                    features.tool_family,
                    {
                        "current_sequence_tokens": features.current_sequence_tokens,
                        "active_tool_count": features.active_tool_count,
                        "backend_pressure": features.backend_pressure,
                    },
                ),
                elapsed_ms=features.elapsed_wait_ms,
            )
        terminal = _temperature_scale(terminal, self.tool_temperature)
        levels = (boundary_level, decode_level, output_level, prompt_level, tool_level)
        unavailable = [name for name, level in zip(
            ("boundary", "decode_demand", "next_output", "prompt_growth", "tool"), levels
        ) if level == "unavailable"]
        support_level = "exact" if all(level == "exact" for level in levels if level != "unavailable") else "backoff"
        if unavailable and len(unavailable) == len(levels):
            support_level = "unavailable"
        del boundary_support
        intervals = {
            name: _calibrated_interval(
                distribution,
                target_coverage=self.calibration_coverage,
                slack=self.interval_slack.get(name, 0.0),
            )
            for name, distribution in (
                ("remaining_decode_tokens", decode),
                ("remaining_external_wait_ms", wait),
                ("prompt_growth_tokens", prompt),
                ("next_output_tokens", output),
            )
            if distribution.values and self.calibration_coverage > 0
        }
        return LocalFrontierPrediction(
            invocation_id=features.invocation_id,
            boundary_distribution=boundary,
            current_sequence_tokens=features.current_sequence_tokens,
            remaining_decode_tokens=decode,
            remaining_external_wait=wait,
            tool_terminal_distribution=terminal,
            prompt_growth_tokens=prompt,
            next_output_tokens=output,
            support_level=support_level,
            calibration_coverage=self.calibration_coverage,
            ood_reasons=tuple(f"{item}_unavailable" for item in unavailable),
            calibrated_intervals=intervals,
        )

    def calibrate(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        target_coverage: float = 0.9,
        allow_development: bool = False,
    ) -> dict[str, Any]:
        """Calibrate probabilities and intervals without refitting train counts."""

        if not 0 < target_coverage < 1:
            raise ValueError("target coverage must be between zero and one")
        if self.calibration_summary:
            raise ValueError("model is already calibrated")
        values = [dict(row) for row in rows]
        if not values:
            raise ValueError("calibration requires decision points")
        _validate_demand_rows(values)
        splits = {str(row.get("split") or "unknown") for row in values}
        if allow_development:
            if not splits.issubset({"calibration", "train", "development"}):
                raise ValueError(
                    "development calibration may only consume "
                    "calibration, train, or development rows"
                )
        elif splits != {"calibration"}:
            raise ValueError("calibration may consume only the calibration split")

        episode_counts = Counter(
            str(row.get("episode_group_id") or row.get("decision_id"))
            for row in values
        )
        local_episode_counts = _local_episode_counts(values)
        workflow_episode_counts = _workflow_local_episode_counts(values)
        boundary_records: list[tuple[Mapping[str, float], str, float]] = []
        tool_records: list[tuple[Mapping[str, float], str, float]] = []
        scores: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        observation_counts: Counter[str] = Counter()
        for row in values:
            episode = str(row.get("episode_group_id") or row.get("decision_id"))
            trigger = str(row.get("trigger_kind") or "")
            labels = {
                str(item.get("invocation_id")): item
                for item in row.get("labels", ())
            }
            for raw_features in row.get("invocations", ()):
                invocation_id = str(raw_features.get("invocation_id") or "")
                label = labels.get(invocation_id)
                if label is None:
                    continue
                local_episode = f"{episode}|{invocation_id}"
                weight = 1.0 / max(
                    1, local_episode_counts[(episode, invocation_id)]
                )
                workflow = _workflow_group_id(row)
                weight /= max(1, workflow_episode_counts[workflow])
                features = _local_features_from_row(row, raw_features)
                prediction = self.predict(features)
                boundary = _normalize_boundary(label.get("next_boundary_kind"))
                if (
                    features.state == InvocationState.RUNNING_LLM.value
                    and boundary is not None
                ):
                    boundary_records.append(
                        (prediction.boundary_distribution, boundary, weight)
                    )
                    observation_counts["boundary"] += 1
                if (
                    trigger == RuntimeEventKind.TOOL_START.value
                    and features.state == InvocationState.WAIT_TOOL.value
                ):
                    status = str(label.get("next_boundary_status") or "error")
                    if label.get("censored"):
                        status = "censored"
                    tool_records.append(
                        (prediction.tool_terminal_distribution, status, weight)
                    )
                    observation_counts["tool_terminal"] += 1
                scalar_targets = (
                    (
                        "remaining_decode_tokens",
                        label.get("remaining_output_tokens")
                        if features.state == InvocationState.RUNNING_LLM.value
                        else None,
                        prediction.remaining_decode_tokens,
                    ),
                    (
                        "remaining_external_wait_ms",
                        label.get("next_boundary_delay_ms")
                        if features.state == InvocationState.WAIT_TOOL.value
                        else None,
                        prediction.remaining_external_wait,
                    ),
                    (
                        "prompt_growth_tokens",
                        label.get("reentry_prompt_delta_tokens"),
                        prediction.prompt_growth_tokens,
                    ),
                    (
                        "next_output_tokens",
                        label.get("next_output_tokens"),
                        prediction.next_output_tokens,
                    ),
                )
                for name, actual, distribution in scalar_targets:
                    if actual is None or not distribution.values:
                        continue
                    lower, upper = _raw_interval(distribution, target_coverage)
                    score = max(lower - float(actual), float(actual) - upper, 0.0)
                    scores[name][local_episode].append(score)
                    observation_counts[name] += 1

        self.boundary_temperature = _fit_temperature(boundary_records)
        self.tool_temperature = _fit_temperature(tool_records)
        self.interval_slack = {
            name: _finite_sample_quantile(
                [max(items) for items in by_episode.values()],
                target_coverage,
            )
            for name, by_episode in scores.items()
            if by_episode
        }
        self.calibration_coverage = target_coverage
        self.calibration_summary = {
            "split": (
                "development_train"
                if allow_development
                else "calibration"
            ),
            "decision_point_count": len(values),
            "episode_count": len(episode_counts),
            "local_episode_count": len(local_episode_counts),
            "target_coverage": target_coverage,
            "boundary_temperature": self.boundary_temperature,
            "tool_temperature": self.tool_temperature,
            "interval_slack": dict(sorted(self.interval_slack.items())),
            "observation_counts": dict(sorted(observation_counts.items())),
            "conformal_unit": "episode_max_nonconformity",
            "training_counts_refit": False,
        }
        return dict(self.calibration_summary)

    def to_dict(self, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURED_FRONTIER_SCHEMA_VERSION,
            "model_kind": "structured_conditional_particle_frontier",
            "model_version": self.model_version,
            "decision_authority": "none; ScenarioRiskPlanner owns actions",
            "join_semantics": "not learned; RCCG composer applies ALL/ANY",
            "hyperparameters": self.hyperparameters.to_dict(),
            "training_summary": self.training_summary,
            "calibration_summary": self.calibration_summary,
            "calibration_coverage": self.calibration_coverage,
            "boundary_temperature": self.boundary_temperature,
            "tool_temperature": self.tool_temperature,
            "interval_slack": dict(sorted(self.interval_slack.items())),
            "metadata": dict(metadata or {}),
            "components": {
                "boundary": self.boundary.to_dict(),
                "decode_demand": self.decode_demand.to_dict(),
                "next_output": self.next_output.to_dict(),
                "prompt_growth": self.prompt_growth.to_dict(),
                "tool": self.tool.to_dict(),
                "join_wait": self.join_wait.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FrontierBeliefModel":
        if int(raw.get("schema_version", -1)) != STRUCTURED_FRONTIER_SCHEMA_VERSION:
            raise ValueError("unsupported structured frontier model schema")
        model = cls(
            model_version=str(raw.get("model_version") or "unknown"),
            hyperparameters=FrontierModelHyperparameters.from_dict(
                raw.get("hyperparameters")
            ),
        )
        components = raw.get("components", {})
        model.boundary = _BoundaryContextTree.from_dict(components.get("boundary", {}))
        model.decode_demand = _HierarchicalEmpiricalModel.from_dict(
            components.get("decode_demand", {})
        )
        model.next_output = _HierarchicalEmpiricalModel.from_dict(
            components.get("next_output", {})
        )
        model.prompt_growth = _HierarchicalEmpiricalModel.from_dict(components.get("prompt_growth", {}))
        model.tool = _CompetingRiskToolModel.from_dict(components.get("tool", {}))
        model.join_wait = _HierarchicalEmpiricalModel.from_dict(
            components.get("join_wait", {})
        )
        model.training_summary = dict(raw.get("training_summary", {}))
        model.calibration_summary = dict(raw.get("calibration_summary", {}))
        model.calibration_coverage = float(raw.get("calibration_coverage", 0.0))
        model.boundary_temperature = float(raw.get("boundary_temperature", 1.0))
        model.tool_temperature = float(raw.get("tool_temperature", 1.0))
        model.interval_slack = {
            str(key): float(value)
            for key, value in raw.get("interval_slack", {}).items()
        }
        return model

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(metadata=metadata)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "FrontierBeliefModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class FrontierScenarioComposer:
    """Construct joint particles from local marginals and deterministic RCCG structure."""

    def __init__(
        self,
        *,
        particle_count: int = 128,
        top_k: int = 8,
        shared_episode_probability: float = 0.35,
    ) -> None:
        if particle_count <= 0 or top_k <= 0:
            raise ValueError("particle count and top-k must be positive")
        if not 0 <= shared_episode_probability <= 1:
            raise ValueError("shared episode probability must be in [0, 1]")
        self.particle_count = particle_count
        self.top_k = top_k
        self.shared_episode_probability = shared_episode_probability

    def compose(
        self,
        *,
        graph: RuntimeCausalContextGraph,
        scope: BeliefScope,
        local_predictions: Mapping[str, LocalFrontierPrediction],
        generated_ts_ms: float,
        evidence_read_set: PredictiveEvidenceReadSet,
        seed: int = 0,
        horizon: FinitePlanningHorizon = FinitePlanningHorizon(),
        projection: ScenarioProjection = ScenarioProjection.FULL,
        target_invocation_id: str | None = None,
    ) -> FrontierBeliefSnapshot:
        particles = self.sample_particles(
            graph=graph,
            scope=scope,
            local_predictions=local_predictions,
            seed=seed,
        )
        return self.reduce_particles(
            particles=particles,
            scope=scope,
            local_predictions=local_predictions,
            generated_ts_ms=generated_ts_ms,
            evidence_read_set=evidence_read_set,
            horizon=horizon,
            projection=projection,
            target_invocation_id=target_invocation_id,
        )

    def sample_particles(
        self,
        *,
        graph: RuntimeCausalContextGraph,
        scope: BeliefScope,
        local_predictions: Mapping[str, LocalFrontierPrediction],
        seed: int = 0,
    ) -> tuple[tuple[FrontierDemandOutcome, ...], ...]:
        missing = set(scope.invocation_ids).difference(local_predictions)
        if missing:
            raise ValueError(f"local predictions missing scoped invocations: {sorted(missing)}")
        rng = random.Random(seed)
        particle_outcomes: list[tuple[FrontierDemandOutcome, ...]] = []
        for _ in range(self.particle_count):
            shared_quantile = rng.random()
            outcomes: dict[str, FrontierDemandOutcome] = {}
            for invocation_id in scope.invocation_ids:
                prediction = local_predictions[invocation_id]
                q = (
                    shared_quantile
                    if rng.random() < self.shared_episode_probability
                    else rng.random()
                )
                outcomes[invocation_id] = self._sample_local(
                    graph, invocation_id, prediction, q, rng.random()
                )
            particle_outcomes.append(tuple(outcomes[key] for key in sorted(outcomes)))
        return tuple(particle_outcomes)

    def reduce_particles(
        self,
        *,
        particles: tuple[tuple[FrontierDemandOutcome, ...], ...],
        scope: BeliefScope,
        local_predictions: Mapping[str, LocalFrontierPrediction],
        generated_ts_ms: float,
        evidence_read_set: PredictiveEvidenceReadSet,
        horizon: FinitePlanningHorizon = FinitePlanningHorizon(),
        projection: ScenarioProjection = ScenarioProjection.FULL,
        target_invocation_id: str | None = None,
    ) -> FrontierBeliefSnapshot:
        projection = ScenarioProjection(projection)
        if not particles:
            raise ValueError("scenario reduction requires sampled particles")
        if projection != ScenarioProjection.FULL and (
            target_invocation_id not in scope.invocation_ids
        ):
            raise ValueError("action-projected reduction requires a scoped target")

        if projection == ScenarioProjection.FULL:
            scenarios, residual_mass = self._reduce_full_particles(particles)
        else:
            scenarios = self._reduce_action_projected_particles(
                particles,
                projection=projection,
                target_invocation_id=str(target_invocation_id),
            )
            residual_mass = 0.0

        selected_mass = sum(item.probability_mass for item in scenarios)
        ood = sorted(
            {
                reason
                for prediction in local_predictions.values()
                for reason in prediction.ood_reasons
            }
        )
        support = (
            "unavailable"
            if all(item.support_level == "unavailable" for item in local_predictions.values())
            else "backoff"
            if any(item.support_level != "exact" for item in local_predictions.values())
            else "exact"
        )
        digest = hashlib.blake2b(
            (
                f"{scope.scope_id}|{generated_ts_ms}|"
                f"{evidence_read_set.model_version}|{projection.value}|"
                f"{target_invocation_id or ''}"
            ).encode(),
            digest_size=16,
            person=b"bkv-frontier",
        ).hexdigest()
        return FrontierBeliefSnapshot(
            belief_id=f"frontier-{digest}",
            generated_ts_ms=generated_ts_ms,
            scope=scope,
            scenarios=scenarios,
            other_probability_mass=max(0.0, 1.0 - selected_mass)
            if projection == ScenarioProjection.FULL
            else residual_mass,
            calibration_coverage=min(
                (item.calibration_coverage for item in local_predictions.values()),
                default=0.0,
            ),
            support_level=support,
            ood_reasons=tuple(ood),
            evidence_read_set=evidence_read_set,
            horizon=horizon,
            other_policy=OtherResidualPolicy(
                finite_risk_bound=(projection != ScenarioProjection.FULL or not ood)
            ),
        )

    def _reduce_full_particles(
        self,
        particles: tuple[tuple[FrontierDemandOutcome, ...], ...],
    ) -> tuple[tuple[DemandScenario, ...], float]:
        counts: Counter[tuple[Any, ...]] = Counter()
        representatives: dict[tuple[Any, ...], tuple[FrontierDemandOutcome, ...]] = {}
        for outcomes in particles:
            key = _scenario_key(outcomes)
            counts[key] += 1
            representatives.setdefault(key, outcomes)
        ranked = sorted(counts, key=lambda key: (-counts[key], key))
        selected = ranked[: self.top_k]
        scenarios = tuple(
            DemandScenario(
                scenario_id=f"scenario-{index:03d}-{hashlib.blake2b(repr(key).encode(), digest_size=8).hexdigest()}",
                outcomes=representatives[key],
                probability_mass=counts[key] / len(particles),
            )
            for index, key in enumerate(selected)
        )
        return scenarios, max(
            0.0, 1.0 - sum(item.probability_mass for item in scenarios)
        )

    def _reduce_action_projected_particles(
        self,
        particles: tuple[tuple[FrontierDemandOutcome, ...], ...],
        *,
        projection: ScenarioProjection,
        target_invocation_id: str,
    ) -> tuple[DemandScenario, ...]:
        vectors = tuple(
            _action_projection_vector(
                outcomes,
                projection=projection,
                target_invocation_id=target_invocation_id,
            )
            for outcomes in particles
        )
        clusters = _deterministic_medoid_clusters(vectors, self.top_k)
        scenarios: list[DemandScenario] = []
        for index, (medoid_index, member_indices) in enumerate(clusters):
            medoid = particles[medoid_index]
            members = tuple(particles[item] for item in member_indices)
            conservative = _conservative_cluster_outcomes(medoid, members)
            identity = (
                projection.value,
                target_invocation_id,
                _scenario_key(medoid),
                tuple(member_indices),
            )
            scenarios.append(
                DemandScenario(
                    scenario_id=(
                        f"{projection.value}-cluster-{index:03d}-"
                        f"{hashlib.blake2b(repr(identity).encode(), digest_size=8).hexdigest()}"
                    ),
                    outcomes=medoid,
                    conservative_outcomes=conservative,
                    probability_mass=len(member_indices) / len(particles),
                    projection=projection,
                )
            )
        return tuple(scenarios)

    @staticmethod
    def _sample_local(
        graph: RuntimeCausalContextGraph,
        invocation_id: str,
        prediction: LocalFrontierPrediction,
        quantile: float,
        categorical_quantile: float,
    ) -> FrontierDemandOutcome:
        invocation = graph.invocations[invocation_id]
        boundary = BoundaryEvent(_sample_categorical(prediction.boundary_distribution, categorical_quantile))
        dependency = DependencyMode.NONE
        dependencies: tuple[str, ...] = ()
        join_id: str | None = None
        external_segments: tuple[ExternalDemandSegment, ...] = ()
        phase = (
            DemandPhase.DECODE
            if invocation.state == InvocationState.RUNNING_LLM
            else DemandPhase.PREFILL
            if invocation.state == InvocationState.READY
            else DemandPhase.EXTERNAL
        )
        if invocation.state == InvocationState.WAIT_TOOL:
            dependency = DependencyMode.EXTERNAL
            boundary = BoundaryEvent.TOOL
            external_segments = (
                ExternalDemandSegment(
                    segment_kind="tool",
                    service_family=invocation.active_tool_family or "unknown",
                    residual_delay_ms=prediction.remaining_external_wait.sample(quantile),
                    terminal_status=_sample_raw_category(
                        prediction.tool_terminal_distribution,
                        categorical_quantile,
                        default="censored",
                    ),
                ),
            )
        elif invocation.state == InvocationState.WAIT_CHILD:
            dependency = DependencyMode.JOIN_ALL
            dependencies = tuple(invocation.blocking_child_ids)
        elif invocation.state == InvocationState.WAIT_JOIN:
            join = graph.joins.get(invocation.join_id or "")
            dependency = (
                DependencyMode.JOIN_ANY
                if join is not None and join.mode == JoinMode.ANY
                else DependencyMode.JOIN_ALL
            )
            dependencies = tuple(join.member_invocation_ids) if join is not None else ()
            join_id = invocation.join_id
            boundary = BoundaryEvent.UNKNOWN
        elif invocation.state == InvocationState.WAIT_MESSAGE:
            dependency = DependencyMode.PRODUCER
            dependencies = tuple(
                sorted(
                    {
                        edge.target_invocation_id
                        for edge in graph.communication_edges.values()
                        if edge.source_invocation_id == invocation_id
                        and edge.target_invocation_id in graph.invocations
                    }
                )
            )
        return FrontierDemandOutcome(
            invocation_id=invocation_id,
            boundary_event=boundary,
            dependency_mode=dependency,
            phase=phase,
            current_sequence_tokens=prediction.current_sequence_tokens,
            remaining_decode_tokens=int(
                round(prediction.remaining_decode_tokens.sample(quantile))
            ),
            prompt_growth_tokens=int(
                round(prediction.prompt_growth_tokens.sample(quantile))
            ),
            next_output_tokens=int(
                round(prediction.next_output_tokens.sample(quantile))
            ),
            external_segments=external_segments,
            dependency_invocation_ids=dependencies,
            join_id=join_id,
        )


def load_decision_rows(
    dataset_dirs: Iterable[str | Path], *, allowed_splits: Iterable[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = frozenset(allowed_splits)
    if allowed.intersection({"calibration", "test_id", "test_ood"}):
        raise ValueError("model fitting cannot consume calibration or test splits")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_decisions: set[str] = set()
    for directory in dataset_dirs:
        root = Path(directory)
        manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
        if "train" in allowed and manifest.get("formal_training_eligible") is not True:
            raise ValueError(f"formal training input is ineligible: {root}")
        if "train" in allowed:
            _validate_formal_p6_manifest(root, manifest, expected_split="train")
        run_id = str(manifest.get("source", {}).get("run_id") or "")
        if not run_id:
            raise ValueError(f"dataset has no source run_id: {root}")
        if run_id in seen_runs:
            raise ValueError(f"duplicate source run in fitting inputs: {run_id}")
        seen_runs.add(run_id)
        manifests.append(manifest)
        for line in (root / "frontier_decision_points.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if str(row.get("split")) in allowed and row.get("training_eligible") is not False:
                decision_id = str(row.get("decision_id") or "")
                if not decision_id:
                    raise ValueError(f"decision point has no identity: {root}")
                if decision_id in seen_decisions:
                    raise ValueError(f"duplicate decision point: {decision_id}")
                seen_decisions.add(decision_id)
                rows.append(row)
    _validate_demand_rows(rows)
    return rows, manifests


def summarize_training_corpus(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize independent sampling units, not scheduler decision rows."""

    values = [dict(row) for row in rows]
    projects = {
        str(row.get("project"))
        for row in values
        if row.get("project") not in {None, "", "unknown"}
    }
    tasks = {
        (
            str(row.get("project") or "unknown"),
            str(row.get("instance_id") or "unknown"),
            str(row.get("base_commit") or "unknown"),
        )
        for row in values
    }
    workflows = {_workflow_group_id(row) for row in values}
    runs = {
        str(row.get("run_id"))
        for row in values
        if row.get("run_id") not in {None, ""}
    }
    return {
        "decision_point_count": len(values),
        "project_count": len(projects),
        "projects": sorted(projects),
        "task_count": len(tasks),
        "workflow_count": len(workflows),
        "run_count": len(runs),
    }


def validate_training_corpus_diversity(
    rows: Iterable[Mapping[str, Any]],
    *,
    minimum_projects: int = 5,
    minimum_tasks: int = 40,
    minimum_workflows: int = 40,
) -> dict[str, Any]:
    """Reject a formally fitted model dominated by a fixed small workflow set."""

    if min(minimum_projects, minimum_tasks, minimum_workflows) <= 0:
        raise ValueError("training diversity thresholds must be positive")
    summary = summarize_training_corpus(rows)
    failures = []
    for field, minimum in (
        ("project_count", minimum_projects),
        ("task_count", minimum_tasks),
        ("workflow_count", minimum_workflows),
    ):
        if int(summary[field]) < minimum:
            failures.append(f"{field}={summary[field]}<{minimum}")
    if failures:
        raise ValueError(
            "formal Frontier corpus is too small and risks workflow memorization: "
            + ", ".join(failures)
        )
    return summary


def load_evaluation_rows(
    dataset_dirs: Iterable[str | Path], *, split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split not in {"calibration", "test_id", "test_ood"}:
        raise ValueError("evaluation rows require calibration, test_id, or test_ood")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_decisions: set[str] = set()
    for directory in dataset_dirs:
        root = Path(directory)
        manifest = json.loads(
            (root / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("formal_training_eligible") is not True:
            raise ValueError(f"formal evaluation input is ineligible: {root}")
        _validate_formal_p6_manifest(root, manifest, expected_split=split)
        run_id = str(manifest.get("source", {}).get("run_id") or "")
        if not run_id:
            raise ValueError(f"dataset has no source run_id: {root}")
        if run_id in seen_runs:
            raise ValueError(f"duplicate source run in evaluation inputs: {run_id}")
        seen_runs.add(run_id)
        manifests.append(manifest)
        for line in (root / "frontier_decision_points.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            row = json.loads(line)
            if str(row.get("split")) == split and row.get("training_eligible") is not False:
                decision_id = str(row.get("decision_id") or "")
                if not decision_id:
                    raise ValueError(f"decision point has no identity: {root}")
                if decision_id in seen_decisions:
                    raise ValueError(f"duplicate decision point: {decision_id}")
                seen_decisions.add(decision_id)
                rows.append(row)
    _validate_demand_rows(rows)
    return rows, manifests


def _validate_formal_p6_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_split: str,
) -> None:
    if manifest.get("dataset_kind") != FORMAL_P6_DATASET_KIND:
        raise ValueError(f"formal input has an unsupported dataset kind: {root}")
    if manifest.get("evaluation_role") != "frozen_split_training_evidence":
        raise ValueError(f"formal input is not frozen-split evidence: {root}")
    split_contract = manifest.get("split_contract") or {}
    if (
        split_contract.get("source") != "explicit frozen split manifest"
        or split_contract.get("development_only") is not False
        or not split_contract.get("manifest_digest")
    ):
        raise ValueError(f"formal input has no frozen project split contract: {root}")
    source = manifest.get("source") or {}
    contract = source.get("collection_contract") or {}
    if contract.get("plan_id") != FORMAL_P6_PLAN_ID:
        raise ValueError(f"formal input did not use the P6 collection plan: {root}")
    if contract.get("split") != expected_split:
        raise ValueError(
            f"collection split {contract.get('split')!r} does not match "
            f"{expected_split!r}: {root}"
        )
    if (
        contract.get("training_eligible") is not True
        or contract.get("runtime_source_stable") is not True
        or contract.get("runtime_policy") != "frozen_p5_observed"
        or bool(contract.get("predictor_enabled"))
        or bool(contract.get("predictive_actions_enabled"))
    ):
        raise ValueError(f"formal input violates the frozen P6 collection contract: {root}")
    if not source.get("workload_manifest_sha256"):
        raise ValueError(f"formal input has no workload manifest identity: {root}")


def select_frontier_hyperparameters(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidates: Iterable[FrontierModelHyperparameters] | None = None,
) -> dict[str, Any]:
    """Select structured-model smoothing/backoff by train-project LOPO.

    Each held-out project is scored as one macro fold. Calibration and test
    rows are rejected so this procedure cannot silently tune on reportable
    evaluation data.
    """

    values = [dict(row) for row in rows]
    _validate_demand_rows(values)
    if not values or {str(row.get("split")) for row in values} != {"train"}:
        raise ValueError("LOPO selection requires only formal train rows")
    projects = sorted({str(row.get("project") or "unknown") for row in values})
    if "unknown" in projects or len(projects) < 2:
        raise ValueError("LOPO selection requires at least two identified projects")
    options = tuple(candidates or _default_hyperparameter_candidates())
    if not options:
        raise ValueError("LOPO selection requires candidate hyperparameters")

    reports = []
    for index, option in enumerate(options):
        folds = []
        for held_out in projects:
            fit_rows = [row for row in values if str(row.get("project")) != held_out]
            validation_rows = [
                {**row, "split": "calibration"}
                for row in values
                if str(row.get("project")) == held_out
            ]
            model = FrontierBeliefModel(
                model_version=f"lopo-candidate-{index}",
                hyperparameters=option,
            )
            model.fit(fit_rows)
            metrics = evaluate_frontier_model(model, validation_rows)
            components = _lopo_loss_components(metrics, validation_rows)
            folds.append(
                {
                    "held_out_project": held_out,
                    "loss": sum(components.values()) / max(len(components), 1),
                    "loss_components": components,
                    "local_episode_count": metrics["local_episode_count"],
                }
            )
        reports.append(
            {
                "candidate_index": index,
                "hyperparameters": option.to_dict(),
                "project_macro_loss": sum(fold["loss"] for fold in folds)
                / len(folds),
                "folds": folds,
            }
        )
    selected = min(
        reports,
        key=lambda item: (item["project_macro_loss"], item["candidate_index"]),
    )
    return {
        "schema_version": 1,
        "selection_method": "leave_one_train_project_out_project_macro",
        "selection_objective": (
            "mean available boundary/tool NLL, per-project scale-normalized "
            "scalar MAE, and 0.25*OOD fallback"
        ),
        "projects": projects,
        "candidate_count": len(reports),
        "selected_candidate_index": selected["candidate_index"],
        "selected_hyperparameters": selected["hyperparameters"],
        "candidates": reports,
    }


def evaluate_frontier_model(
    model: FrontierBeliefModel,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate local beliefs with local-episode weights and no model updates."""

    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("evaluation requires decision points")
    _validate_demand_rows(values)
    splits = {str(row.get("split") or "unknown") for row in values}
    if not splits.issubset({"calibration", "test_id", "test_ood"}):
        raise ValueError("evaluation cannot consume train or development rows")
    local_counts = _local_episode_counts(values)
    workflow_episode_counts = _workflow_local_episode_counts(values)
    classification: dict[str, dict[str, Any]] = {
        "boundary": _classification_accumulator(),
        "tool_terminal": _classification_accumulator(),
    }
    scalar: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "weight": 0.0,
            "absolute_error": 0.0,
            "interval_weight": 0.0,
            "covered_weight": 0.0,
            "interval_width": 0.0,
        }
    )
    prediction_weight = 0.0
    ood_weight = 0.0
    support_weight: Counter[str] = Counter()
    for row in values:
        episode = str(row.get("episode_group_id") or row.get("decision_id"))
        trigger = str(row.get("trigger_kind") or "")
        labels = {
            str(item.get("invocation_id")): item
            for item in row.get("labels", ())
        }
        for raw_features in row.get("invocations", ()):
            invocation_id = str(raw_features.get("invocation_id") or "")
            label = labels.get(invocation_id)
            if label is None:
                continue
            weight = 1.0 / max(1, local_counts[(episode, invocation_id)])
            workflow = _workflow_group_id(row)
            weight /= max(1, workflow_episode_counts[workflow])
            features = _local_features_from_row(row, raw_features)
            prediction = model.predict(features)
            prediction_weight += weight
            ood_weight += weight * bool(prediction.ood_reasons)
            support_weight[prediction.support_level] += weight

            boundary = _normalize_boundary(label.get("next_boundary_kind"))
            if (
                features.state == InvocationState.RUNNING_LLM.value
                and boundary is not None
            ):
                _observe_classification(
                    classification["boundary"],
                    prediction.boundary_distribution,
                    boundary,
                    weight,
                )
            if (
                trigger == RuntimeEventKind.TOOL_START.value
                and features.state == InvocationState.WAIT_TOOL.value
            ):
                status = (
                    "censored"
                    if label.get("censored")
                    else str(label.get("next_boundary_status") or "error")
                )
                if status not in {"success", "error", "censored"}:
                    status = "error"
                _observe_classification(
                    classification["tool_terminal"],
                    prediction.tool_terminal_distribution,
                    status,
                    weight,
                )

            targets = (
                (
                    "remaining_decode_tokens",
                    label.get("remaining_output_tokens")
                    if features.state == InvocationState.RUNNING_LLM.value
                    else None,
                    prediction.remaining_decode_tokens,
                ),
                (
                    "remaining_external_wait_ms",
                    label.get("next_boundary_delay_ms")
                    if features.state == InvocationState.WAIT_TOOL.value
                    else None,
                    prediction.remaining_external_wait,
                ),
                (
                    "prompt_growth_tokens",
                    label.get("reentry_prompt_delta_tokens"),
                    prediction.prompt_growth_tokens,
                ),
                (
                    "next_output_tokens",
                    label.get("next_output_tokens"),
                    prediction.next_output_tokens,
                ),
            )
            for name, actual, distribution in targets:
                if actual is None or not distribution.values:
                    continue
                actual_value = float(actual)
                metrics = scalar[name]
                metrics["weight"] += weight
                metrics["absolute_error"] += weight * abs(
                    actual_value - distribution.quantile(0.5)
                )
                interval = prediction.calibrated_intervals.get(name)
                if interval is not None:
                    lower, upper = interval
                    metrics["interval_weight"] += weight
                    metrics["covered_weight"] += weight * (
                        lower <= actual_value <= upper
                    )
                    metrics["interval_width"] += weight * (upper - lower)

    return {
        "model_version": model.model_version,
        "splits": sorted(splits),
        "decision_point_count": len(values),
        "local_episode_count": len(local_counts),
        "workflow_count": len(workflow_episode_counts),
        "classification": {
            name: _finalize_classification(metrics)
            for name, metrics in classification.items()
        },
        "scalar": {
            name: {
                "episode_weighted_mae": metrics["absolute_error"]
                / max(metrics["weight"], 1e-12),
                "calibrated_interval_coverage": (
                    metrics["covered_weight"] / metrics["interval_weight"]
                    if metrics["interval_weight"]
                    else None
                ),
                "mean_calibrated_interval_width": (
                    metrics["interval_width"] / metrics["interval_weight"]
                    if metrics["interval_weight"]
                    else None
                ),
                "episode_weight": metrics["weight"],
            }
            for name, metrics in sorted(scalar.items())
        },
        "ood_fallback_rate": ood_weight / max(prediction_weight, 1e-12),
        "support_weight": dict(sorted(support_weight.items())),
        "calibration_coverage_target": model.calibration_coverage,
    }


def _classification_accumulator() -> dict[str, Any]:
    return {
        "weight": 0.0,
        "negative_log_likelihood": 0.0,
        "brier": 0.0,
        "correct": 0.0,
        "confidence_records": [],
    }


def _default_hyperparameter_candidates() -> tuple[FrontierModelHyperparameters, ...]:
    return (
        FrontierModelHyperparameters(),
        FrontierModelHyperparameters(
            boundary_max_order=2,
            boundary_minimum_support=2.0,
            empirical_minimum_support=2.0,
            tool_minimum_support=2.0,
        ),
        FrontierModelHyperparameters(
            boundary_max_order=2,
            boundary_minimum_support=4.0,
            boundary_smoothing=1.0,
            empirical_minimum_support=4.0,
            tool_minimum_support=4.0,
            tool_smoothing=1.0,
        ),
        FrontierModelHyperparameters(
            boundary_max_order=4,
            boundary_minimum_support=6.0,
            empirical_minimum_support=8.0,
            tool_minimum_support=8.0,
        ),
    )


def _lopo_loss_components(
    metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    components: dict[str, float] = {}
    for name in ("boundary", "tool_terminal"):
        loss = metrics["classification"][name]["negative_log_likelihood"]
        if loss is not None:
            components[f"{name}_nll"] = float(loss)
    scales = _target_scales(rows)
    for name, item in metrics["scalar"].items():
        if item["episode_weight"] > 0:
            components[f"{name}_normalized_mae"] = (
                float(item["episode_weighted_mae"])
                / max(scales.get(name, 1.0), 1.0)
            )
    components["ood_penalty"] = 0.25 * float(metrics["ood_fallback_rate"])
    return components


def _target_scales(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        labels = {
            str(item.get("invocation_id")): item for item in row.get("labels", ())
        }
        for features in row.get("invocations", ()):
            label = labels.get(str(features.get("invocation_id") or ""))
            if label is None:
                continue
            state = str(features.get("state") or "unknown")
            targets = {
                "remaining_decode_tokens": (
                    label.get("remaining_output_tokens")
                    if state == InvocationState.RUNNING_LLM.value
                    else None
                ),
                "remaining_external_wait_ms": (
                    label.get("next_boundary_delay_ms")
                    if state == InvocationState.WAIT_TOOL.value
                    else None
                ),
                "prompt_growth_tokens": label.get("reentry_prompt_delta_tokens"),
                "next_output_tokens": label.get("next_output_tokens"),
            }
            for name, target in targets.items():
                if target is not None:
                    values[name].append(float(target))
    return {
        name: sorted(items)[len(items) // 2]
        for name, items in values.items()
        if items
    }


def _observe_classification(
    metrics: dict[str, Any],
    distribution: Mapping[str, float],
    target: str,
    weight: float,
) -> None:
    if not distribution:
        return
    probability = max(float(distribution.get(target, 0.0)), 1e-12)
    prediction = max(distribution, key=distribution.get)
    confidence = float(distribution[prediction])
    correct = prediction == target
    vocabulary = set(distribution) | {target}
    brier = sum(
        (float(distribution.get(item, 0.0)) - float(item == target)) ** 2
        for item in vocabulary
    )
    metrics["weight"] += weight
    metrics["negative_log_likelihood"] += weight * -math.log(probability)
    metrics["brier"] += weight * brier
    metrics["correct"] += weight * correct
    metrics["confidence_records"].append((confidence, correct, weight))


def _finalize_classification(metrics: Mapping[str, Any]) -> dict[str, Any]:
    weight = float(metrics["weight"])
    if weight <= 0:
        return {
            "episode_weight": 0.0,
            "negative_log_likelihood": None,
            "brier": None,
            "accuracy": None,
            "ece_10": None,
        }
    bins: list[list[tuple[float, bool, float]]] = [[] for _ in range(10)]
    for confidence, correct, item_weight in metrics["confidence_records"]:
        index = min(9, int(float(confidence) * 10))
        bins[index].append((float(confidence), bool(correct), float(item_weight)))
    ece = 0.0
    for bucket in bins:
        bucket_weight = sum(item[2] for item in bucket)
        if not bucket_weight:
            continue
        mean_confidence = sum(item[0] * item[2] for item in bucket) / bucket_weight
        accuracy = sum(float(item[1]) * item[2] for item in bucket) / bucket_weight
        ece += bucket_weight / weight * abs(mean_confidence - accuracy)
    return {
        "episode_weight": weight,
        "negative_log_likelihood": metrics["negative_log_likelihood"] / weight,
        "brier": metrics["brier"] / weight,
        "accuracy": metrics["correct"] / weight,
        "ece_10": ece,
    }


def _local_features_from_row(
    row: Mapping[str, Any], features: Mapping[str, Any]
) -> LocalFrontierFeatures:
    trigger_attributes = row.get("trigger_attributes") or {}
    return LocalFrontierFeatures(
        invocation_id=str(features.get("invocation_id") or ""),
        state=str(features.get("state") or "unknown"),
        agent_definition_id=str(
            features.get("agent_definition_id") or "unknown"
        ),
        boundary_history=tuple(
            str(item) for item in features.get("boundary_history", ())
        ),
        tool_family=str(
            trigger_attributes.get("tool_family")
            or features.get("active_tool_family")
            or "unknown"
        ),
        backend_class=str(
            trigger_attributes.get("backend_class")
            or features.get("backend_class")
            or "unknown"
        ),
        generated_tokens=int(features.get("observed_output_tokens") or 0),
        elapsed_wait_ms=float(features.get("active_tool_elapsed_ms") or 0.0),
        current_sequence_tokens=int(
            features.get("current_sequence_tokens")
            or features.get("context_tokens")
            or 0
        ),
        active_tool_count=int(features.get("active_tool_count") or 0),
        backend_pressure=str(features.get("backend_pressure") or "unknown"),
    )


def _validate_demand_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        schema_version = int(row.get("schema_version") or 0)
        if schema_version < MINIMUM_DEMAND_DECISION_SCHEMA_VERSION:
            raise ValueError(
                "frontier demand rows must be re-exported with load-independent "
                f"schema v{MINIMUM_DEMAND_DECISION_SCHEMA_VERSION}+"
            )
        for label in row.get("labels", ()):
            contaminated = FORBIDDEN_LOAD_COUPLED_LABELS.intersection(label)
            if contaminated:
                raise ValueError(
                    "load-coupled GPU service labels are forbidden in Frontier fit: "
                    f"{sorted(contaminated)}"
                )
        for features in row.get("invocations", ()):
            contaminated = FORBIDDEN_LOAD_COUPLED_FEATURES.intersection(features)
            if contaminated:
                raise ValueError(
                    "load-coupled scheduler features are forbidden in Frontier fit; "
                    "retain them under diagnostics only: "
                    f"{sorted(contaminated)}"
                )


def _local_episode_counts(
    rows: Sequence[Mapping[str, Any]],
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        episode = str(row.get("episode_group_id") or row.get("decision_id"))
        labels = {
            str(item.get("invocation_id")) for item in row.get("labels", ())
        }
        for features in row.get("invocations", ()):
            invocation_id = str(features.get("invocation_id") or "")
            if invocation_id in labels:
                counts[(episode, invocation_id)] += 1
    return counts


def _workflow_group_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("workflow_id")
        or row.get("workload_group_id")
        or row.get("episode_group_id")
        or row.get("decision_id")
        or "unknown"
    )


def _workflow_local_episode_counts(
    rows: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    local_episodes: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        workflow = _workflow_group_id(row)
        episode = str(row.get("episode_group_id") or row.get("decision_id"))
        labels = {
            str(item.get("invocation_id")) for item in row.get("labels", ())
        }
        for features in row.get("invocations", ()):
            invocation_id = str(features.get("invocation_id") or "")
            if invocation_id in labels:
                local_episodes[workflow].add((episode, invocation_id))
    return Counter(
        {workflow: len(episodes) for workflow, episodes in local_episodes.items()}
    )


def _temperature_scale(
    distribution: Mapping[str, float], temperature: float
) -> dict[str, float]:
    if not distribution:
        return {}
    temperature = max(1e-3, temperature)
    powered = {
        key: max(float(probability), 1e-12) ** (1.0 / temperature)
        for key, probability in distribution.items()
    }
    normalizer = sum(powered.values())
    return {key: value / normalizer for key, value in powered.items()}


def _fit_temperature(
    records: Sequence[tuple[Mapping[str, float], str, float]],
) -> float:
    if not records:
        return 1.0
    candidates = [0.5 + index * 0.05 for index in range(51)]
    return min(
        candidates,
        key=lambda temperature: sum(
            -weight
            * math.log(
                max(
                    _temperature_scale(distribution, temperature).get(target, 0.0),
                    1e-12,
                )
            )
            for distribution, target, weight in records
        ),
    )


def _raw_interval(
    distribution: EmpiricalDistribution, target_coverage: float
) -> tuple[float, float]:
    tail = (1.0 - target_coverage) / 2.0
    return distribution.quantile(tail), distribution.quantile(1.0 - tail)


def _calibrated_interval(
    distribution: EmpiricalDistribution,
    *,
    target_coverage: float,
    slack: float,
) -> tuple[float, float]:
    lower, upper = _raw_interval(distribution, target_coverage)
    return max(0.0, lower - slack), upper + slack


def _finite_sample_quantile(values: Sequence[float], coverage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = math.ceil((len(ordered) + 1) * coverage)
    return ordered[min(len(ordered), max(1, rank)) - 1]


def _demand_feature_key(
    role: str, state: str, family: str, features: Mapping[str, Any]
) -> tuple[str, ...]:
    return (
        role,
        state,
        family,
        f"context:{_power_two_bucket(int(features.get('current_sequence_tokens') or features.get('context_tokens') or 0))}",
        f"generated:{_power_two_bucket(int(features.get('generated_tokens') or features.get('observed_output_tokens') or 0))}",
        f"backend:{str(features.get('backend_class') or 'unknown')}",
    )


def _tool_feature_key(
    role: str, family: str, features: Mapping[str, Any]
) -> tuple[str, ...]:
    return (
        role,
        InvocationState.WAIT_TOOL.value,
        family,
        f"active:{_power_two_bucket(int(features.get('active_tool_count') or 0))}",
        f"context:{_power_two_bucket(int(features.get('current_sequence_tokens') or features.get('context_tokens') or 0))}",
        f"pressure:{str(features.get('backend_pressure') or 'unknown')}",
    )


def _backoff_keys(key: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    role, state, family, condition_a, condition_b, condition_c = key
    return (
        (role, state, family, condition_a, condition_b, condition_c),
        (role, state, family, condition_a, condition_b),
        (role, state, family),
        (role, state),
        ("*", state, family),
        ("*", state),
        ("*",),
    )


def _power_two_bucket(value: int) -> int:
    result = 1
    while result < max(1, value):
        result *= 2
    return result


def _log_bucket(value: float) -> float:
    if value <= 0:
        return 0.0
    exponent = round(math.log2(value) * 4.0) / 4.0
    return round(2.0**exponent, 6)


def _normalize_boundary(value: Any) -> str | None:
    normalized = str(value or "").lower()
    mapping = {
        "function_call": BoundaryEvent.TOOL.value,
        "tool": BoundaryEvent.TOOL.value,
        "spawn": BoundaryEvent.SPAWN.value,
        "handoff": BoundaryEvent.HANDOFF.value,
        "message": BoundaryEvent.MESSAGE.value,
        "return": BoundaryEvent.RETURN.value,
        "final": BoundaryEvent.FINAL.value,
        "final_answer": BoundaryEvent.FINAL.value,
    }
    return mapping.get(normalized)


def _normalize_tool_terminal(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"success", "ok", "completed"}:
        return "success"
    if normalized in {
        "censored",
        "cancelled",
        "canceled",
        "timeout",
        "aborted",
        "duplicate_suppressed",
        "recursion_limit",
    }:
        return "censored"
    return "error"


def _sample_raw_category(
    distribution: Mapping[str, float], quantile: float, *, default: str
) -> str:
    cumulative = 0.0
    for key, probability in sorted(distribution.items()):
        cumulative += probability
        if quantile <= cumulative:
            return str(key)
    return default


def _sample_categorical(distribution: Mapping[str, float], quantile: float) -> str:
    cumulative = 0.0
    for key, probability in sorted(distribution.items()):
        cumulative += probability
        if quantile <= cumulative:
            return _normalize_boundary(key) or BoundaryEvent.UNKNOWN.value
    return BoundaryEvent.UNKNOWN.value


def _action_projection_vector(
    outcomes: tuple[FrontierDemandOutcome, ...],
    *,
    projection: ScenarioProjection,
    target_invocation_id: str,
) -> tuple[float, ...]:
    by_id = {item.invocation_id: item for item in outcomes}
    target = by_id[target_invocation_id]
    target_wait = _external_reentry_proxy_ms(target_invocation_id, by_id, set())
    target_gpu_demand = float(
        target.remaining_decode_tokens
        + target.prompt_growth_tokens
        + target.next_output_tokens
    )
    aggregate_growth = float(
        sum(
            item.remaining_decode_tokens
            + item.prompt_growth_tokens
            + item.next_output_tokens
            for item in outcomes
        )
    )
    pressure_arrival = min(
        (
            _external_reentry_proxy_ms(item.invocation_id, by_id, set())
            for item in outcomes
            if item.invocation_id != target_invocation_id
        ),
        default=target_wait,
    )
    if projection == ScenarioProjection.PREFETCH:
        return (
            target_wait,
            float(target.prompt_growth_tokens),
            aggregate_growth,
            target_gpu_demand,
            float(target.current_sequence_tokens),
        )
    if projection == ScenarioProjection.PREPARE_HOST:
        return (
            target_wait,
            pressure_arrival,
            aggregate_growth,
            float(target.current_sequence_tokens),
        )
    raise ValueError(f"unsupported action projection: {projection.value}")


def _external_reentry_proxy_ms(
    invocation_id: str,
    outcomes: Mapping[str, FrontierDemandOutcome],
    visiting: set[str],
) -> float:
    if invocation_id in visiting:
        return 0.0
    outcome = outcomes[invocation_id]
    if outcome.dependency_mode == DependencyMode.EXTERNAL:
        return sum(item.residual_delay_ms for item in outcome.external_segments)
    dependencies = tuple(
        item
        for item in outcome.dependency_invocation_ids
        if item in outcomes
    )
    if not dependencies:
        return 0.0
    nested_visiting = {*visiting, invocation_id}
    values = tuple(
        _external_reentry_proxy_ms(item, outcomes, nested_visiting)
        for item in dependencies
    )
    if outcome.dependency_mode == DependencyMode.JOIN_ALL:
        return max(values)
    if outcome.dependency_mode in {
        DependencyMode.JOIN_ANY,
        DependencyMode.PRODUCER,
    }:
        return min(values)
    return 0.0


def _deterministic_medoid_clusters(
    vectors: tuple[tuple[float, ...], ...],
    max_clusters: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Small deterministic k-medoids for equal-mass action particles."""

    if not vectors:
        return ()
    dimensions = len(vectors[0])
    if any(len(item) != dimensions for item in vectors):
        raise ValueError("projected scenario vectors must have equal dimensions")
    minima = tuple(min(item[index] for item in vectors) for index in range(dimensions))
    maxima = tuple(max(item[index] for item in vectors) for index in range(dimensions))
    normalized = tuple(
        tuple(
            0.0
            if maxima[index] <= minima[index]
            else (value - minima[index]) / (maxima[index] - minima[index])
            for index, value in enumerate(item)
        )
        for item in vectors
    )
    unique_count = len(set(normalized))
    cluster_count = min(max_clusters, max(1, unique_count), len(vectors))
    ordered = sorted(range(len(vectors)), key=lambda item: (normalized[item], item))
    medoids = [
        ordered[min(len(ordered) - 1, ((2 * index + 1) * len(ordered)) // (2 * cluster_count))]
        for index in range(cluster_count)
    ]
    medoids = list(dict.fromkeys(medoids))

    def distance(left: int, right: int) -> float:
        return sum(
            abs(a - b) for a, b in zip(normalized[left], normalized[right], strict=True)
        )

    assignments: dict[int, list[int]] = {}
    for _ in range(4):
        assignments = {item: [] for item in medoids}
        for particle_index in range(len(vectors)):
            selected = min(
                medoids,
                key=lambda item: (distance(particle_index, item), item),
            )
            assignments[selected].append(particle_index)
        updated = []
        for medoid in medoids:
            members = assignments[medoid]
            updated.append(
                min(
                    members,
                    key=lambda candidate: (
                        sum(distance(candidate, peer) for peer in members),
                        candidate,
                    ),
                )
            )
        updated = list(dict.fromkeys(updated))
        if updated == medoids:
            break
        medoids = updated

    assignments = {item: [] for item in medoids}
    for particle_index in range(len(vectors)):
        selected = min(
            medoids,
            key=lambda item: (distance(particle_index, item), item),
        )
        assignments[selected].append(particle_index)
    return tuple(
        (medoid, tuple(assignments[medoid]))
        for medoid in sorted(medoids, key=lambda item: (normalized[item], item))
    )


def _conservative_cluster_outcomes(
    medoid: tuple[FrontierDemandOutcome, ...],
    members: tuple[tuple[FrontierDemandOutcome, ...], ...],
) -> tuple[FrontierDemandOutcome, ...]:
    by_member = tuple(
        {item.invocation_id: item for item in outcomes}
        for outcomes in members
    )
    conservative = []
    for base in medoid:
        variants = tuple(item[base.invocation_id] for item in by_member)
        segments = tuple(
            replace(
                segment,
                residual_delay_ms=min(
                    variant.external_segments[index].residual_delay_ms
                    for variant in variants
                    if len(variant.external_segments) > index
                ),
            )
            for index, segment in enumerate(base.external_segments)
        )
        conservative.append(
            replace(
                base,
                current_sequence_tokens=max(
                    item.current_sequence_tokens for item in variants
                ),
                remaining_decode_tokens=max(
                    item.remaining_decode_tokens for item in variants
                ),
                prompt_growth_tokens=max(
                    item.prompt_growth_tokens for item in variants
                ),
                next_output_tokens=max(item.next_output_tokens for item in variants),
                external_segments=segments,
            )
        )
    return tuple(conservative)


def _scenario_key(outcomes: Iterable[FrontierDemandOutcome]) -> tuple[Any, ...]:
    return tuple(
        (
            item.invocation_id,
            item.boundary_event.value,
            item.dependency_mode.value,
            item.phase.value,
            item.current_sequence_tokens,
            item.remaining_decode_tokens,
            item.prompt_growth_tokens,
            item.next_output_tokens,
            tuple(
                (
                    segment.segment_kind,
                    segment.service_family,
                    round(segment.residual_delay_ms, 3),
                    segment.terminal_status,
                )
                for segment in item.external_segments
            ),
            item.dependency_invocation_ids,
            item.join_id,
        )
        for item in outcomes
    )
