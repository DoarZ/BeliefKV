from __future__ import annotations

import pytest

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.control.controller import BeliefKVController
from beliefkv.control.data_consumers import (
    ConsumerRelation,
    ObservedDataConsumerIndex,
)
from beliefkv.core.events import (
    EventConfidence,
    ExecutionMode,
    RuntimeEvent,
    RuntimeEventKind,
)


def _event(
    sequence: int,
    kind: RuntimeEventKind,
    *,
    invocation_id: str | None = None,
    target_invocation_id: str | None = None,
    context_id: str | None = None,
    **kwargs: object,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"event-{sequence}",
        ts_ms=float(sequence),
        kind=kind,
        workflow_id="workflow",
        invocation_id=invocation_id,
        target_invocation_id=target_invocation_id,
        context_id=context_id,
        **kwargs,
    )


def _create(
    graph: RuntimeCausalContextGraph,
    sequence: int,
    invocation_id: str,
) -> None:
    graph.apply(
        _event(
            sequence,
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id=invocation_id,
            context_id=f"context-{invocation_id}",
        )
    )


def _runtime() -> tuple[RuntimeCausalContextGraph, ObservedDataConsumerIndex]:
    graph = RuntimeCausalContextGraph()
    graph.apply(_event(0, RuntimeEventKind.WORKFLOW_START))
    _create(graph, 1, "parent")
    _create(graph, 2, "child")
    _create(graph, 3, "reviewer")
    return graph, ObservedDataConsumerIndex(graph)


def test_spawn_is_causal_only_until_child_result_is_consumed() -> None:
    graph, consumers = _runtime()
    spawn = _event(
        4,
        RuntimeEventKind.SPAWN,
        invocation_id="parent",
        target_invocation_id="child",
        execution_mode=ExecutionMode.BACKGROUND,
    )
    graph.apply(spawn)
    consumers.apply(spawn)

    assert graph.invocations["child"].parent_invocation_id == "parent"
    assert consumers.consumers_for("child") == ()

    returned = _event(5, RuntimeEventKind.RETURN, invocation_id="child")
    graph.apply(returned)
    delta = consumers.apply(returned)

    assert len(delta.changed_edges) == 1
    edge = delta.changed_edges[0]
    assert edge.producer_invocation_id == "child"
    assert edge.consumer_invocation_id == "parent"
    assert edge.relation == ConsumerRelation.RETURN


def test_message_handoff_and_broadcast_consumers_are_distinct() -> None:
    graph, consumers = _runtime()
    message = _event(
        4,
        RuntimeEventKind.MESSAGE,
        invocation_id="parent",
        target_invocation_id="reviewer",
        confidence=EventConfidence.OBSERVED_EXACT,
    )
    graph.apply(message)
    consumers.apply(message)
    repeated = _event(
        5,
        RuntimeEventKind.MESSAGE,
        invocation_id="parent",
        target_invocation_id="reviewer",
    )
    graph.apply(repeated)
    consumers.apply(repeated)
    broadcast = _event(
        6,
        RuntimeEventKind.LLM_RESULT,
        invocation_id="parent",
        attributes={
            "consumer_invocation_ids": ["child", "reviewer"],
            "consumer_relation": "broadcast",
        },
    )
    graph.apply(broadcast)
    consumers.apply(broadcast)

    edges = consumers.consumers_for("parent")
    assert {(item.consumer_invocation_id, item.relation) for item in edges} == {
        ("reviewer", ConsumerRelation.MESSAGE),
        ("child", ConsumerRelation.BROADCAST),
        ("reviewer", ConsumerRelation.BROADCAST),
    }
    message_edge = next(item for item in edges if item.relation == ConsumerRelation.MESSAGE)
    assert message_edge.observation_count == 2
    assert message_edge.confidence == 1.0
    assert consumers.consumer_fanout("parent") == 2


def test_handoff_and_reactivation_are_reconstructable() -> None:
    graph, consumers = _runtime()
    handoff = _event(
        4,
        RuntimeEventKind.HANDOFF,
        invocation_id="parent",
        target_invocation_id="reviewer",
    )
    graph.apply(handoff)
    consumers.apply(handoff)
    assert graph.invocations["parent"].state == InvocationState.WAIT_MESSAGE

    reactivate = _event(
        5,
        RuntimeEventKind.REACTIVATE,
        invocation_id="parent",
    )
    delta = graph.apply(reactivate)
    consumer_delta = consumers.apply(reactivate)

    assert graph.invocations["parent"].state == InvocationState.READY
    assert delta.awakened_invocations == frozenset({"parent"})
    assert consumer_delta.reactivated_invocation_ids == frozenset({"parent"})
    assert consumers.consumers_for("parent")[0].relation == ConsumerRelation.HANDOFF


def test_graph_and_consumer_versions_are_monotonic_and_idempotent() -> None:
    graph, consumers = _runtime()
    initial_graph_version = graph.graph_version
    message = _event(
        4,
        RuntimeEventKind.MESSAGE,
        invocation_id="parent",
        target_invocation_id="reviewer",
    )
    first_graph = graph.apply(message)
    first_consumer = consumers.apply(message)
    duplicate_graph = graph.apply(message)
    duplicate_consumer = consumers.apply(message)

    assert first_graph.graph_version == initial_graph_version + 1
    assert duplicate_graph.graph_version == first_graph.graph_version
    assert first_consumer.index_version == 1
    assert duplicate_consumer.index_version == 1
    assert duplicate_consumer.changed_edges == ()
    assert graph.snapshot()["graph_version"] == graph.graph_version


def test_controller_maintains_consumer_index_with_runtime_events() -> None:
    controller = BeliefKVController()
    controller.process_runtime_events(
        (
            _event(0, RuntimeEventKind.WORKFLOW_START),
            _event(
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="parent",
                context_id="context-parent",
            ),
            _event(
                2,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="reviewer",
                context_id="context-reviewer",
            ),
            _event(
                3,
                RuntimeEventKind.HANDOFF,
                invocation_id="parent",
                target_invocation_id="reviewer",
            ),
        )
    )

    edge = controller.data_consumers.consumers_for("parent")[0]
    assert edge.consumer_invocation_id == "reviewer"
    assert edge.relation == ConsumerRelation.HANDOFF


def test_reactivate_rejects_running_or_terminal_invocations() -> None:
    graph, _ = _runtime()
    graph.apply(_event(4, RuntimeEventKind.LLM_SUBMIT, invocation_id="parent"))
    with pytest.raises(Exception, match="already running"):
        graph.apply(_event(5, RuntimeEventKind.REACTIVATE, invocation_id="parent"))


def test_unknown_optional_consumer_does_not_break_runtime_state() -> None:
    controller = BeliefKVController()
    controller.process_runtime_events(
        (
            _event(0, RuntimeEventKind.WORKFLOW_START),
            _event(
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="parent",
                context_id="context-parent",
            ),
            _event(
                2,
                RuntimeEventKind.LLM_RESULT,
                invocation_id="parent",
                attributes={"consumer_invocation_ids": ["not-created-yet"]},
            ),
        )
    )

    assert controller.graph.invocations["parent"].state == InvocationState.READY
    assert controller.data_consumers.consumers_for("parent") == ()
