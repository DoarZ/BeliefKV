from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.control.data_consumers import ObservedDataConsumerIndex
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind


@dataclass(frozen=True)
class DynamicTraceCharacterization:
    workflow_count: int
    invocation_count: int
    context_count: int
    event_count: int
    spawn_count: int
    max_spawn_fanout: int
    join_count: int
    cancel_count: int
    handoff_count: int
    message_count: int
    reactivation_count: int
    unique_communication_edges: int
    repeated_communication_transitions: int
    cycle_edge_count: int
    max_consumer_fanout: int
    topology_entropy_bits: float
    llm_result_count: int
    structured_action_valid_count: int
    structured_action_coverage: float
    boundary_token_observed_count: int
    boundary_token_coverage: float
    boundary_token_indices: tuple[int, ...]
    tool_start_gap_ms: tuple[float, ...]
    action_critical_inversion_count: int | None
    trace_sensitivities: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def characterize_dynamic_trace(
    events: Iterable[RuntimeEvent],
) -> DynamicTraceCharacterization:
    ordered = list(enumerate(events))
    ordered.sort(key=lambda item: (item[1].ts_ms, item[0]))
    graph = RuntimeCausalContextGraph(strict_timestamps=False)
    consumers = ObservedDataConsumerIndex(graph)
    transition_counts: Counter[tuple[str, str, str]] = Counter()
    communication_counts: Counter[tuple[str, str]] = Counter()
    spawn_targets: dict[str, set[str]] = defaultdict(set)
    valid_action_ts: dict[str, float] = {}
    tool_gaps: list[float] = []
    boundary_indices: list[int] = []
    llm_results = 0
    valid_action_keys: set[tuple[str, str, int | None]] = set()
    boundary_keys: set[tuple[str, str, int | None]] = set()
    sensitivities: set[str] = set()

    for _, event in ordered:
        graph.apply(event)
        consumers.apply(event)
        sensitivity = event.attributes.get("trace_sensitivity")
        if isinstance(sensitivity, str) and sensitivity:
            sensitivities.add(sensitivity)
        if event.kind == RuntimeEventKind.SPAWN:
            if event.invocation_id and event.target_invocation_id:
                spawn_targets[event.invocation_id].add(event.target_invocation_id)
        if event.kind in {RuntimeEventKind.MESSAGE, RuntimeEventKind.HANDOFF}:
            if event.invocation_id and event.target_invocation_id:
                pair = (event.invocation_id, event.target_invocation_id)
                communication_counts[pair] += 1
                transition_counts[
                    (event.kind.value, event.invocation_id, event.target_invocation_id)
                ] += 1
        elif event.kind == RuntimeEventKind.REACTIVATE and event.invocation_id:
            transition_counts[
                (event.kind.value, event.invocation_id, event.invocation_id)
            ] += 1
        if event.kind == RuntimeEventKind.LLM_RESULT:
            llm_results += 1
            if event.attributes.get("parser_status") == "valid":
                valid_action_keys.add(_action_key(event))
                if event.invocation_id:
                    valid_action_ts[event.invocation_id] = event.ts_ms
            boundary = event.attributes.get("action_boundary_token_index")
            if (
                isinstance(boundary, int)
                and boundary > 0
                and _action_key(event) not in boundary_keys
            ):
                boundary_keys.add(_action_key(event))
                boundary_indices.append(boundary)
        elif event.kind == RuntimeEventKind.STRUCTURED_ACTION:
            if event.attributes.get("parser_status") == "valid":
                valid_action_keys.add(_action_key(event))
                if event.invocation_id:
                    valid_action_ts[event.invocation_id] = event.ts_ms
            boundary = event.attributes.get("action_boundary_token_index")
            if (
                isinstance(boundary, int)
                and boundary > 0
                and _action_key(event) not in boundary_keys
            ):
                boundary_keys.add(_action_key(event))
                boundary_indices.append(boundary)
        elif event.kind == RuntimeEventKind.TOOL_START and event.invocation_id:
            unlocked_at = valid_action_ts.get(event.invocation_id)
            if unlocked_at is not None and event.ts_ms >= unlocked_at:
                tool_gaps.append(event.ts_ms - unlocked_at)

    all_consumer_fanouts = [
        consumers.consumer_fanout(invocation_id)
        for invocation_id in graph.invocations
    ]
    cycle_edges = _cycle_edges(set(communication_counts))
    transition_total = sum(transition_counts.values())
    topology_entropy = 0.0
    if transition_total:
        for count in transition_counts.values():
            probability = count / transition_total
            topology_entropy -= probability * math.log2(probability)

    counts = Counter(event.kind for _, event in ordered)
    return DynamicTraceCharacterization(
        workflow_count=len(graph.workflows),
        invocation_count=len(graph.invocations),
        context_count=len(graph.contexts),
        event_count=len(ordered),
        spawn_count=counts[RuntimeEventKind.SPAWN],
        max_spawn_fanout=max((len(items) for items in spawn_targets.values()), default=0),
        join_count=counts[RuntimeEventKind.JOIN_CREATE],
        cancel_count=counts[RuntimeEventKind.INVOCATION_CANCEL],
        handoff_count=counts[RuntimeEventKind.HANDOFF],
        message_count=counts[RuntimeEventKind.MESSAGE],
        reactivation_count=counts[RuntimeEventKind.REACTIVATE],
        unique_communication_edges=len(communication_counts),
        repeated_communication_transitions=sum(
            max(0, count - 1) for count in communication_counts.values()
        ),
        cycle_edge_count=len(cycle_edges),
        max_consumer_fanout=max(all_consumer_fanouts, default=0),
        topology_entropy_bits=topology_entropy,
        llm_result_count=llm_results,
        structured_action_valid_count=len(valid_action_keys),
        structured_action_coverage=(
            min(len(valid_action_keys), llm_results) / llm_results
            if llm_results
            else 0.0
        ),
        boundary_token_observed_count=len(boundary_indices),
        boundary_token_coverage=(
            len(boundary_indices) / llm_results if llm_results else 0.0
        ),
        boundary_token_indices=tuple(sorted(boundary_indices)),
        tool_start_gap_ms=tuple(sorted(tool_gaps)),
        action_critical_inversion_count=None,
        trace_sensitivities=tuple(sorted(sensitivities)),
    )


def _cycle_edges(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)

    def reaches(start: str, target: str) -> bool:
        stack = [start]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency.get(current, ()))
        return False

    return {
        (source, target)
        for source, target in edges
        if reaches(target, source)
    }


def _action_key(event: RuntimeEvent) -> tuple[str, str, int | None]:
    return (
        event.workflow_id,
        str(event.invocation_id or event.event_id),
        event.context_epoch,
    )
