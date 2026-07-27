from __future__ import annotations

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.prefix_affinity import (
    ContextPrefixAffinityIndex,
    PrefixAffinityKind,
)
from beliefkv.runtime.protocol import PageHandle


def _event(
    event_id: str,
    kind: RuntimeEventKind,
    workflow_id: str,
    *,
    ts_ms: float,
    **kwargs: object,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id,
        kind=kind,
        workflow_id=workflow_id,
        ts_ms=ts_ms,
        **kwargs,
    )


def _create(
    graph: RuntimeCausalContextGraph,
    index: PageOwnershipIndex,
    workflow_id: str,
    invocation_id: str,
    context_id: str,
    *,
    ts_ms: float,
    parent_context_id: str | None = None,
    agent_definition_id: str | None = None,
) -> None:
    graph.apply(
        _event(
            f"create-{invocation_id}",
            RuntimeEventKind.INVOCATION_CREATE,
            workflow_id,
            ts_ms=ts_ms,
            invocation_id=invocation_id,
            context_id=context_id,
            parent_context_id=parent_context_id,
            agent_definition_id=agent_definition_id or invocation_id,
        )
    )
    index.register_context(context_id, workflow_id, 0)


def test_affinity_uses_actual_shared_pages_and_keeps_relation_types_separate() -> None:
    graph = RuntimeCausalContextGraph(strict_timestamps=False)
    index = PageOwnershipIndex()
    graph.apply(_event("start-a", RuntimeEventKind.WORKFLOW_START, "wf-a", ts_ms=0))
    graph.apply(_event("start-b", RuntimeEventKind.WORKFLOW_START, "wf-b", ts_ms=0))
    _create(graph, index, "wf-a", "parent", "ctx-parent", ts_ms=1)
    _create(
        graph,
        index,
        "wf-a",
        "child-a",
        "ctx-child-a",
        ts_ms=2,
        parent_context_id="ctx-parent",
        agent_definition_id="worker",
    )
    _create(
        graph,
        index,
        "wf-a",
        "child-b",
        "ctx-child-b",
        ts_ms=3,
        parent_context_id="ctx-parent",
        agent_definition_id="worker",
    )
    _create(
        graph,
        index,
        "wf-b",
        "external",
        "ctx-external",
        ts_ms=1,
        agent_definition_id="worker",
    )

    shared = PageHandle(1, 0)
    child_private = PageHandle(2, 0)
    index.register_page(shared, size_bytes=100)
    index.register_page(child_private, size_bytes=300)
    for context_id in ("ctx-parent", "ctx-child-a", "ctx-child-b", "ctx-external"):
        index.bind_pages(context_id, 0, (shared,))
    index.bind_pages("ctx-child-a", 0, (child_private,))

    affinities = ContextPrefixAffinityIndex(graph, index).snapshot(now_ms=10)
    by_pair = {(item.context_a, item.context_b): item for item in affinities}

    parent_child = by_pair[("ctx-child-a", "ctx-parent")]
    assert parent_child.kind == PrefixAffinityKind.PARENT_CHILD
    assert parent_child.shared_physical_bytes == 100
    assert parent_child.union_physical_bytes == 400
    assert parent_child.byte_jaccard == 0.25
    assert (
        by_pair[("ctx-child-a", "ctx-child-b")].kind
        == PrefixAffinityKind.SIBLING_TEMPLATE
    )
    assert (
        by_pair[("ctx-child-a", "ctx-external")].kind
        == PrefixAffinityKind.CROSS_WORKFLOW
    )


def test_causal_parent_without_shared_pages_has_no_prefix_affinity() -> None:
    graph = RuntimeCausalContextGraph()
    index = PageOwnershipIndex()
    graph.apply(_event("start", RuntimeEventKind.WORKFLOW_START, "wf", ts_ms=0))
    _create(graph, index, "wf", "parent", "ctx-parent", ts_ms=1)
    _create(
        graph,
        index,
        "wf",
        "fresh-child",
        "ctx-fresh",
        ts_ms=2,
        parent_context_id="ctx-parent",
    )
    parent_page = PageHandle(1, 0)
    child_page = PageHandle(2, 0)
    index.register_page(parent_page, size_bytes=100)
    index.register_page(child_page, size_bytes=100)
    index.bind_pages("ctx-parent", 0, (parent_page,))
    index.bind_pages("ctx-fresh", 0, (child_page,))

    assert ContextPrefixAffinityIndex(graph, index).snapshot(now_ms=3) == ()
