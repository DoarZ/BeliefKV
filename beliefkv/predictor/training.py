from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.predictor.action_context_tree import (
    ActionObservation,
    SemiMarkovContextTree,
)
from beliefkv.predictor.composer import RemainingTimePredictor
from beliefkv.predictor.service_cost import LLMServiceCostModel, LLMServiceSample
from beliefkv.predictor.taxonomy import ActionKind, ToolTaxonomy
from beliefkv.predictor.tool_survival import (
    HierarchicalToolSurvivalModel,
    ToolDurationSample,
)


@dataclass(frozen=True)
class PredictorTrainingSummary:
    event_count: int
    workflow_count: int
    invocation_count: int
    tool_samples: int
    censored_tool_samples: int
    action_trajectories: int
    action_observations: int
    llm_service_samples: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PredictorTrainingCorpus:
    tool_samples: tuple[ToolDurationSample, ...]
    action_trajectories: tuple[tuple[ActionObservation, ...], ...]
    service_samples: tuple[LLMServiceSample, ...]
    summary: PredictorTrainingSummary


def _event_action(event: RuntimeEvent, taxonomy: ToolTaxonomy) -> ActionKind | None:
    if event.kind == RuntimeEventKind.LLM_SUBMIT:
        return ActionKind.LLM_TEXT
    if event.kind == RuntimeEventKind.TOOL_START:
        family = str(event.attributes.get("tool_family", ""))
        if not family:
            family = taxonomy.normalize(
                str(event.attributes.get("tool_name", "unknown"))
            ).family
        return taxonomy.action_for_family(family)
    if event.kind in {RuntimeEventKind.CALL, RuntimeEventKind.SPAWN}:
        return ActionKind.SPAWN_CHILD
    if event.kind == RuntimeEventKind.JOIN_WAIT:
        return ActionKind.WAIT_JOIN
    if event.kind in {RuntimeEventKind.MESSAGE, RuntimeEventKind.HANDOFF}:
        return ActionKind.MESSAGE
    if event.kind == RuntimeEventKind.RETURN:
        return ActionKind.RETURN
    return None


def extract_training_corpus(
    events: Iterable[RuntimeEvent], *, taxonomy: ToolTaxonomy | None = None
) -> PredictorTrainingCorpus:
    taxonomy = taxonomy or ToolTaxonomy()
    indexed_events = list(enumerate(events))
    indexed_events.sort(key=lambda item: (item[1].workflow_id, item[1].ts_ms, item[0]))
    workflows = {event.workflow_id for _, event in indexed_events}
    invocations = {
        (event.workflow_id, event.invocation_id)
        for _, event in indexed_events
        if event.invocation_id is not None
    }
    workflow_end: dict[str, float] = {}
    action_timeline: dict[tuple[str, str], list[tuple[float, int, ActionKind]]] = {}
    active_tools: dict[
        tuple[str, str, str], tuple[float, str, str]
    ] = {}
    active_tool_order: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    tool_samples: list[ToolDurationSample] = []
    active_llm: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
    service_samples: list[LLMServiceSample] = []

    for sequence, event in indexed_events:
        workflow_end[event.workflow_id] = max(
            event.ts_ms, workflow_end.get(event.workflow_id, event.ts_ms)
        )
        if event.invocation_id is not None:
            action = _event_action(event, taxonomy)
            if action is not None:
                action_timeline.setdefault(
                    (event.workflow_id, event.invocation_id), []
                ).append((event.ts_ms, sequence, action))

        if event.kind == RuntimeEventKind.TOOL_START and event.invocation_id:
            family = str(event.attributes.get("tool_family", ""))
            normalized = taxonomy.normalize(
                str(event.attributes.get("tool_name", family or "unknown")),
                str(event.attributes.get("endpoint", "")) or None,
            )
            family = family or normalized.family
            backend = str(
                event.attributes.get("backend_class", normalized.backend_class)
            )
            call_id = str(
                event.attributes.get("tool_call_id", f"event:{event.event_id}")
            )
            key = (event.workflow_id, event.invocation_id, call_id)
            active_tools[key] = (event.ts_ms, family, backend)
            active_tool_order.setdefault(
                (event.workflow_id, event.invocation_id), []
            ).append(key)
        elif event.kind == RuntimeEventKind.TOOL_END and event.invocation_id:
            invocation_key = (event.workflow_id, event.invocation_id)
            call_id = event.attributes.get("tool_call_id")
            key = (
                (event.workflow_id, event.invocation_id, str(call_id))
                if call_id is not None
                else None
            )
            if key not in active_tools:
                candidates = active_tool_order.get(invocation_key, [])
                key = next(
                    (candidate for candidate in reversed(candidates) if candidate in active_tools),
                    None,
                )
            if key is not None and key in active_tools:
                start_ms, family, backend = active_tools.pop(key)
                duration_raw = event.attributes.get("duration_ms")
                duration_ms = (
                    float(duration_raw)
                    if duration_raw is not None
                    else max(0.0, event.ts_ms - start_ms)
                )
                tool_samples.append(
                    ToolDurationSample(max(0.0, duration_ms), True, family, backend)
                )

        if event.kind == RuntimeEventKind.LLM_SUBMIT and event.invocation_id:
            request_id = str(
                event.attributes.get("request_id", f"event:{event.event_id}")
            )
            active_llm[(event.workflow_id, event.invocation_id, request_id)] = (
                event.ts_ms,
                dict(event.attributes),
            )
        elif event.kind == RuntimeEventKind.LLM_RESULT and event.invocation_id:
            request_id = event.attributes.get("request_id")
            key = (
                (event.workflow_id, event.invocation_id, str(request_id))
                if request_id is not None
                else None
            )
            if key not in active_llm:
                key = next(
                    (
                        candidate
                        for candidate in reversed(tuple(active_llm))
                        if candidate[:2] == (event.workflow_id, event.invocation_id)
                    ),
                    None,
                )
            if key is not None and key in active_llm:
                _start_ms, submit = active_llm.pop(key)
                combined = {**submit, **dict(event.attributes)}
                timing = ("queue_ms", "prefill_ms", "decode_ms")
                if all(name in combined for name in timing):
                    service_samples.append(
                        LLMServiceSample(
                            model=str(combined.get("model", "unknown")),
                            prompt_tokens=max(0, int(combined.get("prompt_tokens", 0))),
                            cache_hit_tokens=max(
                                0, int(combined.get("cache_hit_tokens", 0))
                            ),
                            output_tokens=max(
                                0, int(combined.get("output_tokens", 0))
                            ),
                            batch_size=max(1, int(combined.get("batch_size", 1))),
                            context_tokens=max(
                                0, int(combined.get("context_tokens", 0))
                            ),
                            queue_ms=max(0.0, float(combined["queue_ms"])),
                            prefill_ms=max(0.0, float(combined["prefill_ms"])),
                            decode_ms=max(0.0, float(combined["decode_ms"])),
                        )
                    )

    for (workflow_id, _invocation_id, _call_id), (
        start_ms,
        family,
        backend,
    ) in active_tools.items():
        tool_samples.append(
            ToolDurationSample(
                max(0.0, workflow_end.get(workflow_id, start_ms) - start_ms),
                False,
                family,
                backend,
            )
        )

    trajectories: list[tuple[ActionObservation, ...]] = []
    for (workflow_id, _invocation_id), timeline in sorted(action_timeline.items()):
        timeline.sort(key=lambda item: (item[0], item[1]))
        observations: list[ActionObservation] = []
        for index, (ts_ms, _sequence, action) in enumerate(timeline):
            has_next = index + 1 < len(timeline)
            end_ms = (
                timeline[index + 1][0]
                if has_next
                else workflow_end.get(workflow_id, ts_ms)
            )
            observations.append(
                ActionObservation(
                    action=action,
                    duration_ms=max(0.0, end_ms - ts_ms),
                    completed=has_next or action == ActionKind.RETURN,
                )
            )
        if observations:
            trajectories.append(tuple(observations))

    summary = PredictorTrainingSummary(
        event_count=len(indexed_events),
        workflow_count=len(workflows),
        invocation_count=len(invocations),
        tool_samples=len(tool_samples),
        censored_tool_samples=sum(not sample.completed for sample in tool_samples),
        action_trajectories=len(trajectories),
        action_observations=sum(len(item) for item in trajectories),
        llm_service_samples=len(service_samples),
    )
    return PredictorTrainingCorpus(
        tuple(tool_samples),
        tuple(trajectories),
        tuple(service_samples),
        summary,
    )


def train_predictor(
    corpus: PredictorTrainingCorpus,
    *,
    max_context_order: int = 4,
    min_context_count: int = 3,
    min_family_samples: int = 8,
    min_backend_samples: int = 5,
) -> RemainingTimePredictor:
    tool_model = HierarchicalToolSurvivalModel(
        min_family_samples=min_family_samples,
        min_backend_samples=min_backend_samples,
    )
    tool_model.fit(list(corpus.tool_samples))
    action_model = SemiMarkovContextTree(
        max_order=max_context_order,
        min_context_count=min_context_count,
    )
    action_model.fit([list(item) for item in corpus.action_trajectories])
    service_model = LLMServiceCostModel()
    for sample in corpus.service_samples:
        service_model.observe(sample)
    return RemainingTimePredictor(tool_model, action_model, service_model)
