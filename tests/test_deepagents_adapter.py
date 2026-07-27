from __future__ import annotations

import itertools
import json
import threading
from types import SimpleNamespace
from typing import Any, Sequence
from uuid import uuid4

import pytest

pytest.importorskip("deepagents")

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEventKind
from beliefkv.experiments.agent_protocol import ActivationDeadline
from beliefkv.runtime.deepagents_adapter import (
    BeliefKVChatOpenAI,
    DeepAgentsRuntimeAdapter,
)
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


class CollectingSink:
    def __init__(self) -> None:
        self.events = []
        self._lock = threading.Lock()

    def emit_batch(self, events) -> None:
        with self._lock:
            self.events.extend(events)


class FailingControlSink:
    def emit_batch(self, events) -> None:
        del events
        raise ConnectionError("runtime control socket disappeared")


class QueueToolCallingModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "queue-tool-calling-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        del tools, tool_choice, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        if not self.responses:
            raise AssertionError("fake model response queue is empty")
        return ChatResult(
            generations=[ChatGeneration(message=self.responses.pop(0))]
        )


def test_deepagents_task_callbacks_form_replayable_parent_child_join() -> None:
    trace_sink = CollectingSink()
    control_sink = CollectingSink()
    ticks = itertools.count(1)
    root = BeliefKVRequestMetadata(
        root_workflow_id="wf-deepagents",
        invocation_id="root",
        context_id="ctx-root",
        context_epoch=0,
        agent_definition_id="supervisor",
        agent_instance_id="supervisor-1",
    )
    adapter = DeepAgentsRuntimeAdapter(
        trace_sink,
        root,
        control_sink=control_sink,
        clock_ms=lambda: float(next(ticks)),
    )
    model = QueueToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Inspect the parser and report findings.",
                            "subagent_type": "general-purpose",
                        },
                        "id": "task-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The parser has one relevant branch."),
            AIMessage(content="Integrated the subagent report."),
        ]
    )
    agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt="Delegate repository investigation with task().",
        name="beliefkv-supervisor",
    )

    adapter.start()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Inspect the parser."}]},
        config={"callbacks": [adapter], "recursion_limit": 20},
    )
    adapter.finish(outcome="completed")

    assert result["messages"][-1].text == "Integrated the subagent report."
    kinds = [event.kind for event in trace_sink.events]
    assert kinds.count(RuntimeEventKind.SPAWN) == 1
    assert kinds.count(RuntimeEventKind.JOIN_CREATE) == 1
    assert kinds.count(RuntimeEventKind.JOIN_WAIT) == 1
    assert kinds.count(RuntimeEventKind.JOIN_SATISFIED) == 1
    assert kinds.count(RuntimeEventKind.LLM_SUBMIT) == 3
    assert kinds.count(RuntimeEventKind.LLM_RESULT) == 3

    child_create = next(
        event
        for event in trace_sink.events
        if event.kind == RuntimeEventKind.INVOCATION_CREATE
        and event.invocation_id != "root"
    )
    child_llm = next(
        event
        for event in trace_sink.events
        if event.kind == RuntimeEventKind.LLM_SUBMIT
        and event.invocation_id == child_create.invocation_id
    )
    assert child_llm.context_id == child_create.context_id
    assert child_create.attributes["persistent"] is True
    assert child_create.attributes["description_chars"] > 0
    assert "description" not in child_create.attributes

    graph = RuntimeCausalContextGraph()
    graph.apply_batch(trace_sink.events)
    assert graph.invocations["root"].state == InvocationState.DONE
    assert (
        graph.invocations[child_create.invocation_id].state
        == InvocationState.DONE
    )

    control_kinds = [event.kind for event in control_sink.events]
    assert RuntimeEventKind.WORKFLOW_START not in control_kinds
    assert RuntimeEventKind.LLM_SUBMIT not in control_kinds
    assert RuntimeEventKind.SPAWN in control_kinds
    assert RuntimeEventKind.JOIN_WAIT in control_kinds

def test_parallel_task_declaration_uses_one_all_join() -> None:
    trace_sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(trace_sink, root)
    adapter.start()

    tasks = adapter.declare_runtime_tasks(
        [
            ("explorer", "Inspect package A"),
            ("tester", "Inspect tests B"),
        ],
        group_id="model-run",
    )

    join = next(
        event
        for event in trace_sink.events
        if event.kind == RuntimeEventKind.JOIN_CREATE
    )
    assert len(join.member_invocation_ids) == 2
    assert len(tasks) == 2
    assert join.attributes["mode"] == "all"
    assert sum(event.kind == RuntimeEventKind.JOIN_WAIT for event in trace_sink.events) == 1


def test_code_orchestrator_can_bind_dynamic_child_runs() -> None:
    trace_sink = CollectingSink()
    control_sink = CollectingSink()
    ticks = itertools.count(1)
    root = BeliefKVRequestMetadata(
        root_workflow_id="wf-planned",
        invocation_id="root",
        context_id="ctx-root",
        context_epoch=0,
        agent_definition_id="planner",
        agent_instance_id="planner-1",
    )
    adapter = DeepAgentsRuntimeAdapter(
        trace_sink,
        root,
        control_sink=control_sink,
        clock_ms=lambda: float(next(ticks)),
    )
    adapter.start()
    tasks = adapter.declare_runtime_tasks(
        [
            ("repository-explorer", "Inspect the implementation."),
            ("test-analyst", "Find relevant tests."),
        ],
        group_id="plan-1",
    )

    chain_run_id = uuid4()
    model_run_id = uuid4()
    adapter.on_chain_start(
        {},
        {},
        run_id=chain_run_id,
        metadata=adapter.invocation_scope(tasks[0]),
    )
    adapter.on_chat_model_start(
        {},
        [[HumanMessage(content="inspect")]],
        run_id=model_run_id,
        parent_run_id=chain_run_id,
    )
    metadata = adapter.metadata_for_model_run(model_run_id)
    assert metadata.invocation_id == tasks[0].invocation_id
    assert metadata.context_id == tasks[0].context_id

    adapter.complete_runtime_task(tasks[0])
    adapter.complete_runtime_task(tasks[1])
    adapter.finish(outcome="completed")
    kinds = [event.kind for event in trace_sink.events]
    assert kinds.count(RuntimeEventKind.SPAWN) == 2
    assert kinds.count(RuntimeEventKind.JOIN_SATISFIED) == 1
    assert kinds.count(RuntimeEventKind.RETURN) == 3


def test_cancelled_runtime_task_does_not_satisfy_join() -> None:
    sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(sink, root)
    adapter.start()
    tasks = adapter.declare_runtime_tasks(
        [("explorer", "Inspect"), ("tester", "Test")],
        group_id="cancelled-group",
    )

    adapter.complete_runtime_task(tasks[0])
    adapter.complete_runtime_task(tasks[1], error=TimeoutError("deadline"))

    kinds = [event.kind for event in sink.events]
    assert kinds.count(RuntimeEventKind.RETURN) == 1
    assert kinds.count(RuntimeEventKind.INVOCATION_CANCEL) == 1
    assert RuntimeEventKind.JOIN_SATISFIED not in kinds


def test_chat_client_uses_remaining_deadline_and_aborts_failed_request(
    monkeypatch,
) -> None:
    sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(sink, root)
    adapter.start()
    model_run_id = uuid4()
    adapter.on_chat_model_start(
        {},
        [[HumanMessage(content="inspect")]],
        run_id=model_run_id,
    )
    now = [10.0]
    deadline = ActivationDeadline(clock=lambda: now[0])
    deadline.start(20.0)
    now[0] = 14.0
    client = BeliefKVChatOpenAI(
        beliefkv_adapter=adapter,
        activation_deadline=deadline,
        request_timeout_s=900.0,
        abort_url="http://127.0.0.1:30000/abort_request",
        model="test-model",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        max_retries=0,
    )
    run_manager = SimpleNamespace(run_id=model_run_id)
    payload, rid = client._with_beliefkv_runtime(run_manager, {})
    assert payload["timeout"] == 16.0
    assert payload["extra_body"]["rid"] == rid
    assert payload["extra_body"]["beliefkv_metadata"]["invocation_id"] == "root"

    aborted: list[tuple[str, dict[str, str], float]] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

    def fake_urlopen(request, *, timeout):
        aborted.append(
            (request.full_url, json.loads(request.data.decode("utf-8")), timeout)
        )
        return _Response()

    def fail_generate(self, messages, stop=None, run_manager=None, **kwargs):
        del self, messages, stop, run_manager, kwargs
        raise TimeoutError("request timed out")

    monkeypatch.setattr(
        "beliefkv.runtime.deepagents_adapter.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(ChatOpenAI, "_generate", fail_generate)
    with pytest.raises(TimeoutError, match="request timed out"):
        client._generate([], run_manager=run_manager)

    assert aborted == [
        (
            "http://127.0.0.1:30000/abort_request",
            {"rid": rid},
            1.0,
        )
    ]


def test_ordinary_tool_boundaries_keep_the_model_tool_call_id() -> None:
    sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(sink, root)
    adapter.start()
    run_id = uuid4()
    adapter.on_tool_start(
        {"name": "read_file"},
        "",
        run_id=run_id,
        inputs={"file_path": "/module.py"},
        tool_call_id="call-from-model",
    )
    adapter.on_tool_end("contents", run_id=run_id)
    boundaries = [
        event
        for event in sink.events
        if event.kind in {RuntimeEventKind.TOOL_START, RuntimeEventKind.TOOL_END}
    ]
    assert [item.attributes["tool_call_id"] for item in boundaries] == [
        "call-from-model",
        "call-from-model",
    ]


def test_structured_completion_is_not_counted_as_an_external_tool() -> None:
    sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(
        sink,
        root,
        event_namespace="agenticroot",
        completion_tool_names=frozenset({"WorkflowCompletion"}),
    )
    run_id = uuid4()

    adapter.on_tool_start(
        {"name": "WorkflowCompletion"},
        "",
        run_id=run_id,
        inputs={"status": "blocked"},
        tool_call_id="completion-call",
    )
    adapter.on_tool_end("accepted", run_id=run_id)

    assert not any(
        event.kind in {RuntimeEventKind.TOOL_START, RuntimeEventKind.TOOL_END}
        for event in sink.events
    )


def test_unsupported_subagent_type_does_not_create_a_physical_child() -> None:
    sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(
        sink,
        root,
        allowed_subagent_types=frozenset({"repository-explorer"}),
    )
    model_run_id = uuid4()
    tool_run_id = uuid4()
    adapter.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "task",
                                    "args": {
                                        "description": "Inspect the repository",
                                        "subagent_type": "general-purpose",
                                    },
                                    "id": "invalid-task",
                                }
                            ],
                        )
                    )
                ]
            ]
        ),
        run_id=model_run_id,
    )
    adapter.on_tool_start(
        {"name": "task"},
        "",
        run_id=tool_run_id,
        inputs={
            "description": "Inspect the repository",
            "subagent_type": "general-purpose",
        },
        tool_call_id="invalid-task",
    )
    adapter.on_tool_end("unsupported type", run_id=tool_run_id)

    assert not any(
        event.kind
        in {
            RuntimeEventKind.INVOCATION_CREATE,
            RuntimeEventKind.SPAWN,
            RuntimeEventKind.JOIN_CREATE,
        }
        for event in sink.events
    )
    result = next(event for event in sink.events if event.kind == RuntimeEventKind.LLM_RESULT)
    assert result.attributes["rejected_task_call_count"] == 1


def test_adapter_event_namespace_prevents_cross_peer_id_collisions() -> None:
    first_sink = CollectingSink()
    second_sink = CollectingSink()
    first = DeepAgentsRuntimeAdapter(
        first_sink,
        BeliefKVRequestMetadata("wf", "root-a", "ctx-a", 0),
        event_namespace="peera",
    )
    second = DeepAgentsRuntimeAdapter(
        second_sink,
        BeliefKVRequestMetadata("wf", "root-b", "ctx-b", 0),
        event_namespace="peerb",
    )

    first.start()
    second.start()

    first_ids = {event.event_id for event in first_sink.events}
    second_ids = {event.event_id for event in second_sink.events}
    assert first_ids.isdisjoint(second_ids)


def test_declared_join_ids_are_scoped_to_the_workflow() -> None:
    first_sink = CollectingSink()
    second_sink = CollectingSink()
    first = DeepAgentsRuntimeAdapter(
        first_sink,
        BeliefKVRequestMetadata("wf-a", "root-a", "ctx-a", 0),
    )
    second = DeepAgentsRuntimeAdapter(
        second_sink,
        BeliefKVRequestMetadata("wf-b", "root-b", "ctx-b", 0),
    )
    first.start()
    second.start()
    first_tasks = first.declare_runtime_tasks(
        [("analysis", "Inspect")], group_id="same-logical-group"
    )
    second_tasks = second.declare_runtime_tasks(
        [("analysis", "Inspect")], group_id="same-logical-group"
    )
    assert first_tasks[0].join_id != second_tasks[0].join_id


def test_control_delivery_failure_does_not_change_workflow_trajectory() -> None:
    trace_sink = CollectingSink()
    root = BeliefKVRequestMetadata(
        "wf-control-loss", "root", "ctx-root", 0, "supervisor", "root"
    )
    adapter = DeepAgentsRuntimeAdapter(
        trace_sink,
        root,
        control_sink=FailingControlSink(),
    )

    adapter.start()
    tasks = adapter.declare_runtime_tasks(
        [("explorer", "Inspect the implementation")],
        group_id="control-loss",
    )
    adapter.complete_runtime_task(tasks[0])
    adapter.finish(outcome="completed")

    kinds = [event.kind for event in trace_sink.events]
    assert RuntimeEventKind.SPAWN in kinds
    assert RuntimeEventKind.JOIN_SATISFIED in kinds
    assert RuntimeEventKind.WORKFLOW_END in kinds
    summary = adapter.control_delivery_summary()
    assert summary["degraded"] is True
    assert summary["failure_count"] >= 3
    assert summary["first_failure"]["error_type"] == "ConnectionError"
