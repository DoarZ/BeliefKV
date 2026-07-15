from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from math import inf
from pathlib import Path
from typing import Any, Mapping

from beliefkv.control.causal_graph import (
    InvocationRecord,
    InvocationState,
    JoinMode,
    RuntimeCausalContextGraph,
)
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.predictor.action_context_tree import SemiMarkovContextTree
from beliefkv.predictor.calibration import RollingIntervalCalibrator
from beliefkv.predictor.service_cost import LLMServiceCostModel
from beliefkv.predictor.taxonomy import ActionKind
from beliefkv.predictor.tool_survival import HierarchicalToolSurvivalModel
from beliefkv.predictor.types import RemainingTimePrediction


@dataclass
class InvocationPredictionFeatures:
    tool_backend_class: str = "unknown"
    action_history: list[ActionKind] = field(default_factory=list)
    model: str = "unknown"
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    expected_output_tokens: int = 0
    batch_size: int = 1
    context_tokens: int = 0


class RemainingTimePredictor:
    """Compose tool, action and service models over the observed RCCG."""

    ARTIFACT_SCHEMA_VERSION = 1

    def __init__(
        self,
        tool_model: HierarchicalToolSurvivalModel | None = None,
        action_model: SemiMarkovContextTree | None = None,
        service_model: LLMServiceCostModel | None = None,
        calibrator: RollingIntervalCalibrator | None = None,
    ) -> None:
        self.tool_model = tool_model or HierarchicalToolSurvivalModel()
        self.action_model = action_model or SemiMarkovContextTree()
        self.service_model = service_model or LLMServiceCostModel()
        self.calibrator = calibrator or RollingIntervalCalibrator()
        self.features: dict[str, InvocationPredictionFeatures] = {}

    def set_features(
        self, invocation_id: str, features: InvocationPredictionFeatures
    ) -> None:
        self.features[invocation_id] = features

    def observe_event(self, event: RuntimeEvent) -> None:
        """Maintain portable online features without inspecting prompt text."""

        invocation_id = event.invocation_id
        if invocation_id is None:
            return
        features = self.features.setdefault(
            invocation_id, InvocationPredictionFeatures()
        )
        action: ActionKind | None = None
        if event.kind == RuntimeEventKind.LLM_SUBMIT:
            features.model = str(event.attributes.get("model", features.model))
            features.prompt_tokens = int(
                event.attributes.get("prompt_tokens", features.prompt_tokens)
            )
            features.cache_hit_tokens = int(
                event.attributes.get("cache_hit_tokens", features.cache_hit_tokens)
            )
            features.expected_output_tokens = int(
                event.attributes.get(
                    "expected_output_tokens", features.expected_output_tokens
                )
            )
            features.batch_size = max(
                1, int(event.attributes.get("batch_size", features.batch_size))
            )
            features.context_tokens = int(
                event.attributes.get("context_tokens", features.context_tokens)
            )
            action = ActionKind.LLM_TEXT
        elif event.kind == RuntimeEventKind.TOOL_START:
            family = str(event.attributes.get("tool_family", "other"))
            features.tool_backend_class = str(
                event.attributes.get("backend_class", "unknown")
            )
            action = {
                "shell": ActionKind.TOOL_SHELL,
                "search": ActionKind.TOOL_SEARCH,
                "file": ActionKind.TOOL_FILE,
                "browser": ActionKind.TOOL_BROWSER,
            }.get(family, ActionKind.TOOL_OTHER)
        elif event.kind in {RuntimeEventKind.CALL, RuntimeEventKind.SPAWN}:
            action = ActionKind.SPAWN_CHILD
        elif event.kind == RuntimeEventKind.JOIN_WAIT:
            action = ActionKind.WAIT_JOIN
        elif event.kind in {RuntimeEventKind.MESSAGE, RuntimeEventKind.HANDOFF}:
            action = ActionKind.MESSAGE
        elif event.kind == RuntimeEventKind.RETURN:
            action = ActionKind.RETURN
        if action is not None:
            features.action_history.append(action)
            del features.action_history[:-32]

    def to_dict(self, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": self.ARTIFACT_SCHEMA_VERSION,
            "metadata": dict(metadata or {}),
            "models": {
                "tool_survival": self.tool_model.to_dict(),
                "action_context_tree": self.action_model.to_dict(),
                "service_cost": self.service_model.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RemainingTimePredictor":
        version = int(raw.get("schema_version", -1))
        if version != cls.ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported predictor artifact schema {version}; "
                f"expected {cls.ARTIFACT_SCHEMA_VERSION}"
            )
        models = raw.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("predictor artifact has no models object")
        return cls(
            tool_model=HierarchicalToolSurvivalModel.from_dict(
                models.get("tool_survival", {})
            ),
            action_model=SemiMarkovContextTree.from_dict(
                models.get("action_context_tree", {})
            ),
            service_model=LLMServiceCostModel.from_dict(
                models.get("service_cost", {})
            ),
        )

    def save(
        self, path: Path, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                self.to_dict(metadata=metadata),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "RemainingTimePredictor":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("predictor artifact must contain a JSON object")
        return cls.from_dict(raw)

    def predict_all(
        self,
        graph: RuntimeCausalContextGraph,
        *,
        now_ms: float,
        transfer_windows_ms: dict[str, float] | None = None,
    ) -> dict[str, RemainingTimePrediction]:
        windows = transfer_windows_ms or {}
        return {
            context_id: self.predict_context(
                graph,
                context_id=context_id,
                now_ms=now_ms,
                transfer_window_ms=windows.get(context_id, 0.0),
            )
            for context_id in graph.contexts
        }

    def predict_context(
        self,
        graph: RuntimeCausalContextGraph,
        *,
        context_id: str,
        now_ms: float,
        transfer_window_ms: float,
    ) -> RemainingTimePrediction:
        invocations = graph.context_invocations(context_id)
        estimates = [
            self._predict_invocation(
                graph,
                invocation,
                now_ms=now_ms,
                transfer_window_ms=transfer_window_ms,
                visiting=set(),
            )
            for invocation in invocations
            if not invocation.state.terminal or invocation.persistent
        ]
        if not estimates:
            return RemainingTimePrediction(context_id=context_id, generated_ts_ms=now_ms)
        estimate = min(estimates, key=lambda item: item.p50_ms)
        return self.calibrator.adjust(replace(estimate, context_id=context_id))

    def _predict_invocation(
        self,
        graph: RuntimeCausalContextGraph,
        invocation: InvocationRecord,
        *,
        now_ms: float,
        transfer_window_ms: float,
        visiting: set[str],
    ) -> RemainingTimePrediction:
        if invocation.invocation_id in visiting:
            return RemainingTimePrediction(
                context_id=invocation.context_id,
                generated_ts_ms=now_ms,
                backoff_level="cycle_guard",
            )
        visiting = set(visiting)
        visiting.add(invocation.invocation_id)
        features = self.features.get(
            invocation.invocation_id, InvocationPredictionFeatures()
        )
        if invocation.state in {InvocationState.READY, InvocationState.RUNNING_LLM}:
            service = self.service_model.estimate(
                model=features.model,
                prompt_tokens=features.prompt_tokens,
                cache_hit_tokens=features.cache_hit_tokens,
                expected_output_tokens=features.expected_output_tokens,
                batch_size=features.batch_size,
                context_tokens=features.context_tokens,
            )
            resume_probability = 1.0 if service.total_ms <= transfer_window_ms else 0.0
            return RemainingTimePrediction(
                context_id=invocation.context_id,
                generated_ts_ms=now_ms,
                p50_ms=service.total_ms,
                p90_ms=service.total_ms * 1.25,
                p95_ms=service.total_ms * 1.5,
                resume_within_transfer_probability=resume_probability,
                next_event_distribution={"llm_result": 1.0},
                confidence=0.6,
                ood_score=0.2 if features.model != "unknown" else 0.6,
                backoff_level="service_cost",
            )
        if invocation.state == InvocationState.WAIT_TOOL:
            start_ms = invocation.active_tool_start_ms
            elapsed = max(0.0, now_ms - start_ms) if start_ms is not None else 0.0
            return self.tool_model.predict(
                context_id=invocation.context_id,
                now_ms=now_ms,
                elapsed_ms=elapsed,
                family=invocation.active_tool_family or "unknown",
                backend_class=features.tool_backend_class,
                transfer_window_ms=transfer_window_ms,
            )
        if invocation.state == InvocationState.WAIT_CHILD:
            children = [
                graph.invocations[item]
                for item in invocation.blocking_child_ids
                if not graph.invocations[item].state.terminal
            ]
            estimates = [
                self._predict_invocation(
                    graph,
                    child,
                    now_ms=now_ms,
                    transfer_window_ms=transfer_window_ms,
                    visiting=visiting,
                )
                for child in children
            ]
            return self._compose_dependencies(
                invocation.context_id,
                now_ms,
                estimates,
                mode="max",
                level="child_composition",
                transfer_window_ms=transfer_window_ms,
            )
        if invocation.state == InvocationState.WAIT_JOIN and invocation.join_id:
            join = graph.joins[invocation.join_id]
            members = [
                graph.invocations[item]
                for item in join.member_invocation_ids - join.completed_member_ids
            ]
            estimates = [
                self._predict_invocation(
                    graph,
                    member,
                    now_ms=now_ms,
                    transfer_window_ms=transfer_window_ms,
                    visiting=visiting,
                )
                for member in members
            ]
            return self._compose_dependencies(
                invocation.context_id,
                now_ms,
                estimates,
                mode="min" if join.mode == JoinMode.ANY else "max",
                level="join_composition",
                transfer_window_ms=transfer_window_ms,
            )
        action = self.action_model.predict(features.action_history)
        next_distribution = {
            key.value: value for key, value in action.next_distribution.items()
        }
        return RemainingTimePrediction(
            context_id=invocation.context_id,
            generated_ts_ms=now_ms,
            p50_ms=action.current_remaining_p50_ms,
            p90_ms=action.current_remaining_p95_ms,
            p95_ms=action.current_remaining_p95_ms,
            resume_within_transfer_probability=0.0,
            next_event_distribution=next_distribution,
            confidence=action.confidence * 0.7,
            ood_score=max(0.4, action.ood_score),
            backoff_level=f"context_tree_order_{action.selected_order}",
        )

    @staticmethod
    def _compose_dependencies(
        context_id: str,
        now_ms: float,
        estimates: list[RemainingTimePrediction],
        *,
        mode: str,
        level: str,
        transfer_window_ms: float,
    ) -> RemainingTimePrediction:
        if not estimates:
            return RemainingTimePrediction(
                context_id=context_id,
                generated_ts_ms=now_ms,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                resume_within_transfer_probability=1.0,
                confidence=1.0,
                ood_score=0.0,
                backoff_level=level,
            )
        reducer = min if mode == "min" else max
        p50 = reducer(item.p50_ms for item in estimates)
        p90 = reducer(item.p90_ms for item in estimates)
        p95 = reducer(item.p95_ms for item in estimates)
        if mode == "min":
            resume_probability = max(
                item.resume_within_transfer_probability for item in estimates
            )
        else:
            resume_probability = min(
                item.resume_within_transfer_probability for item in estimates
            )
        return RemainingTimePrediction(
            context_id=context_id,
            generated_ts_ms=now_ms,
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            resume_within_transfer_probability=resume_probability,
            confidence=min(item.confidence for item in estimates),
            ood_score=max(item.ood_score for item in estimates),
            backoff_level=level,
        )
