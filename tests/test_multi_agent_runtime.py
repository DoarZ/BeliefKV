from __future__ import annotations

import itertools
import threading

import pytest

pytest.importorskip("langgraph")

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.events import ContextMode, RuntimeEvent, RuntimeEventKind
from beliefkv.experiments.langgraph_peer_workflow import (
    LangGraphPeerWorkflow,
    LLMEventSource,
    OpenAICompatiblePeerBackend,
    PeerRole,
    PeerTurnRequest,
    PeerTurnResult,
    SubagentTask,
    TraceSensitivity,
)
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata
from beliefkv.traces.characterization import characterize_dynamic_trace


class CollectingSink:
    def __init__(self) -> None:
        self.events = []
        self.lock = threading.Lock()

    def emit_batch(self, events) -> None:
        with self.lock:
            self.events.extend(events)


class ScriptedBackend:
    def __init__(self) -> None:
        self.coder_calls = 0

    def invoke(self, request: PeerTurnRequest) -> PeerTurnResult:
        if request.is_subagent:
            return PeerTurnResult(
                summary=f"{request.role} completed",
                next_role=None,
                complete=True,
                output_tokens=7,
            )
        if request.role == PeerRole.CODER.value:
            self.coder_calls += 1
            tasks = (
                (
                    SubagentTask("repository-explorer", "Inspect the parser"),
                    SubagentTask("test-analyst", "Inspect focused tests"),
                )
                if self.coder_calls == 1
                else ()
            )
            return PeerTurnResult(
                summary=f"coder revision {self.coder_calls}",
                next_role=PeerRole.REVIEWER,
                complete=False,
                output_tokens=20,
                subagent_tasks=tasks,
            )
        if request.role == PeerRole.REVIEWER.value:
            if self.coder_calls == 1:
                return PeerTurnResult(
                    summary="review requests tests",
                    next_role=PeerRole.TESTER,
                    complete=False,
                    output_tokens=12,
                )
            return PeerTurnResult(
                summary="review accepted",
                next_role=None,
                complete=True,
                output_tokens=9,
            )
        return PeerTurnResult(
            summary="tests request one revision",
            next_role=PeerRole.CODER,
            complete=False,
            output_tokens=11,
        )


def _run(workflow_id: str = "workflow-mixed"):
    sink = CollectingSink()
    ticks = itertools.count(1)
    workflow = LangGraphPeerWorkflow(
        ScriptedBackend(),
        sink,
        workflow_id=workflow_id,
        max_turns=10,
        trace_sensitivity=TraceSensitivity.SCHEDULE_INVARIANT,
        clock_ms=lambda: float(next(ticks)),
    )
    result = workflow.run("Fix the parser regression and validate it.")
    return result, sink.events


def test_cyclic_mixed_workflow_is_replayable_and_separates_three_edge_types() -> None:
    result, events = _run()

    assert result.completed
    assert result.termination_reason == "semantic_complete"
    assert result.turn_count == 5
    kinds = [event.kind for event in events]
    assert kinds.count(RuntimeEventKind.SPAWN) == 2
    assert kinds.count(RuntimeEventKind.JOIN_WAIT) == 1
    assert kinds.count(RuntimeEventKind.HANDOFF) == 4
    assert RuntimeEventKind.REACTIVATE in kinds

    child_creates = [
        event
        for event in events
        if event.kind == RuntimeEventKind.INVOCATION_CREATE
        and event.context_mode == ContextMode.FRESH
        and event.parent_invocation_id is not None
    ]
    assert len(child_creates) == 2
    assert all(
        event.context_id != event.parent_context_id for event in child_creates
    )

    controller = BeliefKVController()
    controller.process_runtime_events(tuple(events))
    assert all(
        invocation.state.terminal
        for invocation in controller.graph.invocations.values()
    )
    child_ids = {event.invocation_id for event in child_creates}
    for child_id in child_ids:
        edges = controller.data_consumers.consumers_for(str(child_id))
        assert len(edges) == 1
        assert edges[0].relation.value == "return"
    handoff_edges = [
        edge
        for invocation_id in controller.graph.invocations
        for edge in controller.data_consumers.consumers_for(invocation_id)
        if edge.relation.value == "handoff"
    ]
    assert len(handoff_edges) == 3
    assert sum(edge.observation_count for edge in handoff_edges) == 4


def test_fixed_script_produces_a_stable_transition_hash_and_manifest() -> None:
    first, _ = _run("workflow-stable")
    second, _ = _run("workflow-stable")

    assert first.transition_hash == second.transition_hash
    assert first.manifest()["trace_sensitivity"] == "schedule_invariant"
    assert first.manifest()["event_count"] == len(first.events)


def test_dynamic_trace_characterization_reports_cycles_actions_and_fanout() -> None:
    result, _ = _run("workflow-characterization")
    report = characterize_dynamic_trace(result.events)

    assert report.workflow_count == 1
    assert report.spawn_count == 2
    assert report.max_spawn_fanout == 2
    assert report.join_count == 1
    assert report.handoff_count == 4
    assert report.reactivation_count >= 1
    assert report.repeated_communication_transitions == 1
    assert report.cycle_edge_count >= 3
    assert report.max_consumer_fanout >= 1
    assert report.topology_entropy_bits > 0
    assert report.structured_action_coverage == 1.0
    assert report.boundary_token_coverage == 0.0
    assert report.boundary_token_indices == ()
    assert report.action_critical_inversion_count is None
    assert report.trace_sensitivities == ("schedule_invariant",)


def test_max_turn_guard_terminates_a_noncompleting_graph() -> None:
    class LoopingBackend:
        def invoke(self, request: PeerTurnRequest) -> PeerTurnResult:
            target = {
                "coder": PeerRole.REVIEWER,
                "reviewer": PeerRole.TESTER,
                "tester": PeerRole.CODER,
            }[request.role]
            return PeerTurnResult(
                summary="continue",
                next_role=target,
                complete=False,
            )

    sink = CollectingSink()
    ticks = itertools.count(1)
    result = LangGraphPeerWorkflow(
        LoopingBackend(),
        sink,
        workflow_id="workflow-loop-guard",
        max_turns=3,
        trace_sensitivity=TraceSensitivity.TIMING_SENSITIVE,
        clock_ms=lambda: float(next(ticks)),
    ).run("Exercise the loop guard.")

    assert not result.completed
    assert result.termination_reason == "max_turns"
    assert result.turn_count == 3


def test_model_runtime_mode_does_not_duplicate_llm_boundary_events() -> None:
    sink = CollectingSink()
    ticks = itertools.count(1)
    result = LangGraphPeerWorkflow(
        ScriptedBackend(),
        sink,
        workflow_id="workflow-model-runtime-events",
        max_turns=10,
        trace_sensitivity=TraceSensitivity.SCHEDULE_INVARIANT,
        llm_event_source=LLMEventSource.MODEL_RUNTIME,
        clock_ms=lambda: float(next(ticks)),
    ).run("Fix the parser regression and validate it.")

    assert result.completed
    assert not any(
        event.kind in {RuntimeEventKind.LLM_SUBMIT, RuntimeEventKind.LLM_RESULT}
        for event in sink.events
    )
    assert sum(
        event.kind == RuntimeEventKind.STRUCTURED_ACTION for event in sink.events
    ) == 7
    assert any(event.kind == RuntimeEventKind.SPAWN for event in sink.events)
    assert any(event.kind == RuntimeEventKind.HANDOFF for event in sink.events)


def test_internal_agent_loop_advances_outer_context_epoch() -> None:
    class MultiCallBackend:
        def invoke(self, request: PeerTurnRequest) -> PeerTurnResult:
            return PeerTurnResult(
                summary="completed after several internal tool rounds",
                next_role=None,
                complete=True,
                output_tokens=20,
                final_context_epoch=7,
            )

    sink = CollectingSink()
    result = LangGraphPeerWorkflow(
        MultiCallBackend(),
        sink,
        workflow_id="workflow-internal-epochs",
        max_turns=2,
        llm_event_source=LLMEventSource.MODEL_RUNTIME,
    ).run("Exercise a multi-call peer activation.")

    assert result.completed
    action = next(
        event for event in sink.events if event.kind == RuntimeEventKind.STRUCTURED_ACTION
    )
    returned = next(
        event
        for event in sink.events
        if event.kind == RuntimeEventKind.RETURN
        and event.invocation_id == action.invocation_id
    )
    assert action.context_epoch == 7
    assert returned.context_epoch == 7


def test_model_runtime_actions_enrich_unique_server_llm_results() -> None:
    sink = CollectingSink()
    ticks = itertools.count(10, 10)
    result = LangGraphPeerWorkflow(
        ScriptedBackend(),
        sink,
        workflow_id="workflow-model-runtime-merged",
        max_turns=10,
        trace_sensitivity=TraceSensitivity.SCHEDULE_INVARIANT,
        llm_event_source=LLMEventSource.MODEL_RUNTIME,
        clock_ms=lambda: float(next(ticks)),
    ).run("Fix the parser regression and validate it.")
    merged = list(result.events)
    for index, action in enumerate(
        event
        for event in result.events
        if event.kind == RuntimeEventKind.STRUCTURED_ACTION
    ):
        request_id = f"server-request-{index}"
        merged.extend(
            (
                RuntimeEvent(
                    event_id=f"server-submit-{index}",
                    ts_ms=action.ts_ms - 2,
                    kind=RuntimeEventKind.LLM_SUBMIT,
                    workflow_id=action.workflow_id,
                    invocation_id=action.invocation_id,
                    context_id=action.context_id,
                    context_epoch=action.context_epoch,
                    attributes={
                        "request_id": request_id,
                        "prompt_tokens": 10,
                        "cache_hit_tokens": 0,
                    },
                ),
                RuntimeEvent(
                    event_id=f"server-result-{index}",
                    ts_ms=action.ts_ms - 1,
                    kind=RuntimeEventKind.LLM_RESULT,
                    workflow_id=action.workflow_id,
                    invocation_id=action.invocation_id,
                    context_id=action.context_id,
                    context_epoch=action.context_epoch,
                    attributes={"request_id": request_id, "output_tokens": 5},
                ),
            )
        )

    report = characterize_dynamic_trace(merged)

    assert report.llm_result_count == 7
    assert report.structured_action_valid_count == 7
    assert report.structured_action_coverage == 1.0


def test_parallel_subagents_reach_join_after_both_children_complete() -> None:
    class BlockingBackend(ScriptedBackend):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(2)

        def invoke(self, request: PeerTurnRequest) -> PeerTurnResult:
            if request.is_subagent:
                self.barrier.wait(timeout=1)
            return super().invoke(request)

    sink = CollectingSink()
    ticks = itertools.count(1)
    result = LangGraphPeerWorkflow(
        BlockingBackend(),
        sink,
        workflow_id="workflow-parallel-children",
        max_turns=10,
        parallel_subagents=True,
        clock_ms=lambda: float(next(ticks)),
    ).run("Fix the parser regression and validate it.")

    assert result.completed
    join_satisfied = next(
        index
        for index, event in enumerate(sink.events)
        if event.kind == RuntimeEventKind.JOIN_SATISFIED
    )
    child_returns = [
        index
        for index, event in enumerate(sink.events)
        if event.kind == RuntimeEventKind.RETURN
        and event.attributes.get("outcome") == "subagent_complete"
    ]
    assert len(child_returns) == 2
    assert max(child_returns) < join_satisfied


def test_backend_failure_terminates_every_created_invocation() -> None:
    class FailingBackend(ScriptedBackend):
        def invoke(self, request: PeerTurnRequest) -> PeerTurnResult:
            if request.is_subagent:
                raise RuntimeError("injected child failure")
            return super().invoke(request)

    sink = CollectingSink()
    ticks = itertools.count(1)
    workflow = LangGraphPeerWorkflow(
        FailingBackend(),
        sink,
        workflow_id="workflow-error-cleanup",
        max_turns=10,
        clock_ms=lambda: float(next(ticks)),
    )

    with pytest.raises(RuntimeError, match="injected child failure"):
        workflow.run("Exercise error cleanup.")

    created = {
        event.invocation_id
        for event in sink.events
        if event.kind == RuntimeEventKind.INVOCATION_CREATE
    }
    returned = {
        event.invocation_id
        for event in sink.events
        if event.kind == RuntimeEventKind.RETURN
    }
    assert created == returned
    end = [event for event in sink.events if event.kind == RuntimeEventKind.WORKFLOW_END]
    assert len(end) == 1
    assert end[0].attributes["outcome"] == "runtime_error"


class _FakeMessage:
    def __init__(self, text: str, output_tokens: int = 5) -> None:
        self.text = text
        self.usage_metadata = {"output_tokens": output_tokens}


class _FakeStructuredModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.requests = []
        self.response_formats = []

    def bind(self, **kwargs):
        self.response_formats.append(kwargs["response_format"])
        return self

    def invoke(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return _FakeMessage(next(self.responses))


def _backend_with_fake_model(
    responses: list[str], *, minimum: int = 1, maximum: int = 4
) -> OpenAICompatiblePeerBackend:
    backend = OpenAICompatiblePeerBackend.__new__(OpenAICompatiblePeerBackend)
    backend.model = _FakeStructuredModel(responses)
    backend.min_initial_subagents = minimum
    backend.max_initial_subagents = maximum
    backend.max_attempts = 2
    backend._stats_lock = threading.Lock()
    backend._request_count = 0
    backend._retry_count = 0
    backend._model_error_count = 0
    return backend


def _peer_request(
    *,
    is_subagent: bool = False,
    role: str | None = None,
    turn: int = 0,
    must_complete: bool = False,
) -> PeerTurnRequest:
    return PeerTurnRequest(
        task="Investigate a real regression.",
        role=(
            "repository-explorer"
            if is_subagent
            else role or PeerRole.CODER.value
        ),
        turn=turn,
        history=(),
        metadata=BeliefKVRequestMetadata(
            root_workflow_id="workflow",
            invocation_id="invocation",
            context_id="context",
            context_epoch=0,
        ),
        is_subagent=is_subagent,
        must_complete=must_complete,
    )


def test_openai_backend_accepts_model_selected_fanout_within_bounds() -> None:
    backend = _backend_with_fake_model(
        [
            '{"summary":"split investigation","next_role":"reviewer",'
            '"complete":false,"subagent_tasks":['
            '{"agent_definition_id":"api-reader","instruction":"inspect API"},'
            '{"agent_definition_id":"test-reader","instruction":"inspect tests"},'
            '{"agent_definition_id":"history-reader","instruction":"inspect history"}'
            "]}"
        ]
    )

    result = backend.invoke(_peer_request())

    assert len(result.subagent_tasks) == 3
    assert result.next_role == PeerRole.REVIEWER
    assert backend.summary() == {
        "model_request_count": 1,
        "structured_retry_count": 0,
        "model_error_count": 0,
    }
    assert backend.model.requests[0][1]["extra_body"]["beliefkv_metadata"][
        "context_id"
    ] == "context"
    schema = backend.model.response_formats[0]["json_schema"]["schema"]
    assert schema["properties"]["subagent_tasks"]["minItems"] == 1
    assert schema["properties"]["subagent_tasks"]["maxItems"] == 4


def test_openai_backend_places_stable_task_before_dynamic_turn_fields() -> None:
    backend = _backend_with_fake_model(
        [
            '{"summary":"handoff","next_role":"reviewer",'
            '"complete":false,"subagent_tasks":[]}'
        ],
        minimum=0,
        maximum=0,
    )

    backend.invoke(_peer_request(turn=3))

    messages = backend.model.requests[0][0]
    user = messages[1].content
    assert user.startswith("Task:\nInvestigate a real regression.")
    assert user.index("Task:") < user.index("Role:")
    assert user.index("Role:") < user.index("Prior summaries:")
    assert user.index("Prior summaries:") < user.index("Turn: 3")


def test_openai_backend_retries_out_of_policy_fanout_once() -> None:
    backend = _backend_with_fake_model(
        [
            '{"summary":"no split","next_role":"reviewer",'
            '"complete":false,"subagent_tasks":[]}',
            '{"summary":"one useful split","next_role":"tester",'
            '"complete":false,"subagent_tasks":['
            '{"agent_definition_id":"test-reader","instruction":"inspect tests"}'
            "]}",
        ],
        minimum=1,
        maximum=2,
    )

    result = backend.invoke(_peer_request())

    assert len(result.subagent_tasks) == 1
    assert result.next_role == PeerRole.TESTER
    assert backend.summary() == {
        "model_request_count": 2,
        "structured_retry_count": 1,
        "model_error_count": 0,
    }


def test_openai_backend_retries_a_self_handoff_and_excludes_it_from_schema() -> None:
    backend = _backend_with_fake_model(
        [
            '{"summary":"review again","next_role":"reviewer",'
            '"complete":false,"subagent_tasks":[]}',
            '{"summary":"request tests","next_role":"tester",'
            '"complete":false,"subagent_tasks":[]}',
        ],
        minimum=0,
        maximum=0,
    )

    result = backend.invoke(
        _peer_request(role=PeerRole.REVIEWER.value, turn=1)
    )

    assert result.next_role == PeerRole.TESTER
    assert backend.summary()["structured_retry_count"] == 1
    schema = backend.model.response_formats[0]["json_schema"]["schema"]
    role_enums = schema["properties"]["next_role"]["anyOf"][0]["enum"]
    assert PeerRole.REVIEWER.value not in role_enums


def test_openai_backend_final_turn_requires_semantic_completion() -> None:
    backend = _backend_with_fake_model(
        [
            '{"summary":"final evidence","next_role":null,'
            '"complete":true,"subagent_tasks":[]}'
        ],
        minimum=0,
        maximum=0,
    )

    result = backend.invoke(
        _peer_request(
            role=PeerRole.TESTER.value,
            turn=5,
            must_complete=True,
        )
    )

    assert result.complete
    schema = backend.model.response_formats[0]["json_schema"]["schema"]
    assert schema["properties"]["next_role"] == {"type": "null"}
    assert schema["properties"]["complete"]["enum"] == [True]


def test_openai_backend_rejects_nested_subagent_spawn_after_bounded_retries() -> None:
    invalid = (
        '{"summary":"spawn again","next_role":"coder","complete":false,'
        '"subagent_tasks":[{"agent_definition_id":"nested",'
        '"instruction":"recurse"}]}'
    )
    backend = _backend_with_fake_model([invalid, invalid], minimum=0, maximum=4)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        backend.invoke(_peer_request(is_subagent=True))

    assert backend.summary() == {
        "model_request_count": 2,
        "structured_retry_count": 1,
        "model_error_count": 0,
    }
    schema = backend.model.response_formats[0]["json_schema"]["schema"]
    assert schema["properties"]["next_role"] == {"type": "null"}
    assert schema["properties"]["complete"]["enum"] == [True]
    assert schema["properties"]["subagent_tasks"]["maxItems"] == 0
