from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.leases import CausalLeaseProjector, LeaseKind


def _event(
    sequence: int,
    kind: RuntimeEventKind,
    *,
    invocation_id: str | None = None,
    context_id: str | None = None,
    **attributes: object,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"event-{sequence}",
        ts_ms=float(sequence),
        kind=kind,
        workflow_id="wf",
        invocation_id=invocation_id,
        context_id=context_id,
        agent_definition_id=invocation_id,
        agent_instance_id=invocation_id,
        attributes=attributes,
    )


def _create(
    graph: RuntimeCausalContextGraph,
    sequence: int,
    invocation_id: str,
    context_id: str,
) -> None:
    graph.apply(
        _event(
            sequence,
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id=invocation_id,
            context_id=context_id,
        )
    )


def test_context_leases_follow_observed_runtime_state() -> None:
    graph = RuntimeCausalContextGraph()
    graph.apply(_event(0, RuntimeEventKind.WORKFLOW_START))
    _create(graph, 1, "ready", "ctx-ready")
    _create(graph, 2, "running", "ctx-running")
    _create(graph, 3, "waiting", "ctx-waiting")
    _create(graph, 4, "done", "ctx-done")
    graph.apply(_event(5, RuntimeEventKind.LLM_SUBMIT, invocation_id="running"))
    graph.apply(
        _event(
            6,
            RuntimeEventKind.TOOL_START,
            invocation_id="waiting",
            tool_family="browser",
        )
    )
    graph.apply(_event(7, RuntimeEventKind.RETURN, invocation_id="done"))
    leases = CausalLeaseProjector(graph)

    assert leases.context("ctx-ready", now_ms=10).kind == LeaseKind.READY
    assert leases.context("ctx-running", now_ms=10).kind == LeaseKind.RUNNING
    waiting = leases.context("ctx-waiting", now_ms=10)
    assert waiting.kind == LeaseKind.CONDITIONAL_RESUME
    assert waiting.condition is not None
    assert waiting.condition.event_kind == "tool_result"
    assert "browser" in waiting.condition.condition_id
    assert leases.context("ctx-done", now_ms=10).kind == LeaseKind.DEAD
    assert leases.context("unknown", now_ms=10).kind == LeaseKind.RUNNING


def test_shared_bundle_inherits_the_strongest_owner_lease() -> None:
    graph = RuntimeCausalContextGraph()
    graph.apply(_event(0, RuntimeEventKind.WORKFLOW_START))
    _create(graph, 1, "parked", "ctx-parked")
    _create(graph, 2, "running", "ctx-running")
    graph.apply(_event(3, RuntimeEventKind.TOOL_START, invocation_id="parked"))
    graph.apply(_event(4, RuntimeEventKind.LLM_SUBMIT, invocation_id="running"))

    lease = CausalLeaseProjector(graph).bundle(
        "shared-prefix",
        ("ctx-parked", "ctx-running"),
        now_ms=5,
    )

    assert lease.strongest_kind == LeaseKind.RUNNING
    assert lease.owner_context_ids == ("ctx-parked", "ctx-running")

