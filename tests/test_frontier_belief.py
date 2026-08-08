from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.predictor.frontier_belief import (
    BeliefScopeBuilder,
    BeliefScopeConfig,
    CausalAtomKind,
)


def _event(sequence: int, kind: RuntimeEventKind, **kwargs) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"belief-{sequence}",
        ts_ms=float(sequence),
        kind=kind,
        workflow_id="workflow",
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


def _join_graph() -> RuntimeCausalContextGraph:
    graph = RuntimeCausalContextGraph()
    graph.apply(_event(0, RuntimeEventKind.WORKFLOW_START))
    _create(graph, 1, "parent")
    _create(graph, 2, "child-a")
    _create(graph, 3, "child-b")
    graph.apply(
        _event(
            4,
            RuntimeEventKind.JOIN_CREATE,
            join_id="join-all",
            member_invocation_ids=("child-a", "child-b"),
        )
    )
    graph.apply(
        _event(
            5,
            RuntimeEventKind.JOIN_WAIT,
            invocation_id="parent",
            join_id="join-all",
        )
    )
    return graph


def test_join_scope_includes_every_unfinished_member_and_waiter() -> None:
    graph = _join_graph()
    scope = BeliefScopeBuilder(
        BeliefScopeConfig(max_atomic_groups=2, max_total_model_cost=8)
    ).build(graph, ("child-a",))

    assert len(scope.included_atoms) == 1
    atom = scope.included_atoms[0]
    assert atom.kind == CausalAtomKind.JOIN
    assert atom.invocation_ids == ("child-a", "child-b", "parent")
    assert atom.join_ids == ("join-all",)
    assert scope.other_atoms == ()


def test_oversized_join_moves_to_other_without_partial_invocations() -> None:
    graph = _join_graph()
    scope = BeliefScopeBuilder(
        BeliefScopeConfig(max_atomic_groups=2, max_total_model_cost=4)
    ).build(graph, ("child-a",))

    assert scope.included_atoms == ()
    assert len(scope.other_atoms) == 1
    assert scope.residual_invocation_ids == ("child-a", "child-b", "parent")


def test_physical_blocker_set_is_an_indivisible_atom() -> None:
    graph = RuntimeCausalContextGraph()
    graph.apply(_event(0, RuntimeEventKind.WORKFLOW_START))
    _create(graph, 1, "owner-a")
    _create(graph, 2, "owner-b")

    scope = BeliefScopeBuilder().build(
        graph,
        ("owner-a",),
        blocker_sets={"extent:1": ("owner-a", "owner-b")},
    )

    assert len(scope.included_atoms) == 1
    assert scope.included_atoms[0].kind == CausalAtomKind.BLOCKER
    assert scope.included_atoms[0].invocation_ids == ("owner-a", "owner-b")
    assert scope.included_atoms[0].blocker_set_ids == ("extent:1",)
