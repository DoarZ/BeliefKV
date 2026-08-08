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
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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
from beliefkv.runtime.context_lifecycle import (
    ContextCompactionRecord,
    ContextLifecycleMiddleware,
    ContextLifecyclePolicy,
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


def test_explicit_call_censor_preserves_runtime_identity() -> None:
    sink = CollectingSink()
    root = BeliefKVRequestMetadata(
        root_workflow_id="workflow",
        invocation_id="root",
        context_id="context",
        context_epoch=0,
        agent_definition_id="supervisor",
        agent_instance_id="supervisor",
    )
    adapter = DeepAgentsRuntimeAdapter(sink, root, clock_ms=lambda: 1.0)
    adapter.start()
    adapter.record_call_censor(
        {
            "call_kind": "tool",
            "censor_reason": "duplicate_suppressed",
            "tool_call_id": "tool-1",
            "invocation_id": "root",
        }
    )
    event = next(item for item in sink.events if item.kind == RuntimeEventKind.CALL_CENSORED)
    assert event.invocation_id == "root"
    assert event.context_id == "context"
    assert event.attributes["tool_call_id"] == "tool-1"
    assert event.attributes["invocation_identity_fallback"] is False


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
    model_events = [
        event
        for event in trace_sink.events
        if event.kind in {RuntimeEventKind.LLM_SUBMIT, RuntimeEventKind.LLM_RESULT}
    ]
    assert all(
        str(event.attributes["request_id"]).startswith("beliefkv:")
        for event in model_events
    )

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


def test_model_submit_records_semantic_prompt_contract_and_sampling_seed() -> None:
    sink = CollectingSink()
    adapter = DeepAgentsRuntimeAdapter(
        sink,
        BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root"),
    )
    adapter.start()
    messages = [[HumanMessage(content="inspect")]]
    contract = {
        "seed": 17,
        "model": "test-model",
        "max_tokens": 1024,
        "tools": [{"type": "function", "function": {"name": "search"}}],
    }

    adapter.on_chat_model_start(
        {},
        messages,
        run_id=uuid4(),
        invocation_params=contract,
    )
    first = [
        event for event in sink.events if event.kind == RuntimeEventKind.LLM_SUBMIT
    ][-1]
    adapter.on_chat_model_start(
        {},
        messages,
        run_id=uuid4(),
        invocation_params=dict(contract),
    )
    second = [
        event for event in sink.events if event.kind == RuntimeEventKind.LLM_SUBMIT
    ][-1]

    assert first.attributes["sampling_seed"] == 17
    assert first.attributes["prompt_semantic_sha256"] == second.attributes[
        "prompt_semantic_sha256"
    ]

    adapter.on_chat_model_start(
        {},
        messages,
        run_id=uuid4(),
        invocation_params={**contract, "max_tokens": 2048},
    )
    changed = [
        event for event in sink.events if event.kind == RuntimeEventKind.LLM_SUBMIT
    ][-1]
    assert changed.attributes["prompt_semantic_sha256"] != first.attributes[
        "prompt_semantic_sha256"
    ]


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
    assert kinds.count(RuntimeEventKind.JOIN_TIMEOUT) == 1


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


def test_tool_end_preserves_status_error_class_and_workspace_change() -> None:
    sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    digests = iter(("before", "after"))
    adapter = DeepAgentsRuntimeAdapter(
        sink,
        root,
        workspace_digest_provider=lambda name, payload: next(digests),
    )
    adapter.start()
    run_id = uuid4()
    adapter.on_tool_start(
        {"name": "edit_file"},
        "",
        run_id=run_id,
        inputs={"file_path": "/workspace/module.py", "old_string": "old"},
        tool_call_id="edit-call",
    )
    adapter.on_tool_end(
        ToolMessage(
            content="Error: String not found in file: 'old'",
            name="edit_file",
            tool_call_id="edit-call",
            status="error",
        ),
        run_id=run_id,
    )

    ended = next(event for event in sink.events if event.kind == RuntimeEventKind.TOOL_END)
    assert ended.attributes["status"] == "error"
    assert ended.attributes["tool_error_class"] == "string_not_found"
    assert ended.attributes["workspace_digest_before"] == "before"
    assert ended.attributes["workspace_digest_after"] == "after"
    assert ended.attributes["workspace_changed"] is True


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


def test_context_compaction_advances_epoch_before_next_model_submit() -> None:
    trace_sink = CollectingSink()
    control_sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(
        trace_sink,
        root,
        control_sink=control_sink,
    )
    adapter.start()
    first_run = uuid4()
    adapter.on_chat_model_start(
        {},
        [[HumanMessage(content="first")]],
        run_id=first_run,
    )
    adapter.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="continue"))]]
        ),
        run_id=first_run,
    )

    record = ContextCompactionRecord(
        source_message_count=40,
        retained_message_count=8,
        summary_chars=512,
        summary_sha256="a" * 64,
        trigger_tokens=24_576,
        keep_tokens=8_192,
    )
    second_run = uuid4()
    with adapter.stage_context_compaction(record):
        adapter.on_chat_model_start(
            {},
            [[HumanMessage(content="checkpoint"), HumanMessage(content="recent")]],
            run_id=second_run,
        )

    compact = next(
        event
        for event in trace_sink.events
        if event.kind == RuntimeEventKind.CONTEXT_COMPACT
    )
    second_submit = [
        event
        for event in trace_sink.events
        if event.kind == RuntimeEventKind.LLM_SUBMIT
    ][-1]
    assert trace_sink.events.index(compact) < trace_sink.events.index(second_submit)
    assert compact.context_id == "ctx"
    assert compact.context_epoch == 1
    assert compact.attributes["previous_context_epoch"] == 0
    assert compact.attributes["old_kv_disposition"] == "release_ownership"
    assert compact in control_sink.events

    graph = RuntimeCausalContextGraph()
    graph.apply_batch(trace_sink.events)
    assert graph.contexts["ctx"].epoch == 1


def test_summary_model_call_has_ephemeral_runtime_internal_context() -> None:
    trace_sink = CollectingSink()
    control_sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(
        trace_sink,
        root,
        control_sink=control_sink,
    )
    adapter.start()
    summary_run = uuid4()
    adapter.on_chat_model_start(
        {},
        [[HumanMessage(content="summarize")]],
        run_id=summary_run,
        metadata={"lc_source": "summarization"},
    )
    metadata = adapter.metadata_for_model_run(summary_run)
    adapter.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="checkpoint"))]]
        ),
        run_id=summary_run,
    )

    assert metadata.invocation_id.startswith("root:context-summary:")
    assert metadata.context_id.startswith("ctx:context-summary:")
    assert metadata.context_id != "ctx"
    internal = [
        event
        for event in trace_sink.events
        if bool(event.attributes.get("runtime_internal"))
    ]
    assert {event.kind for event in internal} >= {
        RuntimeEventKind.INVOCATION_CREATE,
        RuntimeEventKind.CALL,
        RuntimeEventKind.LLM_SUBMIT,
        RuntimeEventKind.LLM_RESULT,
        RuntimeEventKind.RETURN,
    }
    assert all(
        event.invocation_id != "root"
        for event in internal
        if event.kind in {RuntimeEventKind.LLM_SUBMIT, RuntimeEventKind.LLM_RESULT}
    )

    graph = RuntimeCausalContextGraph()
    graph.apply_batch(trace_sink.events)
    assert graph.invocations[metadata.invocation_id].state == InvocationState.DONE
    assert graph.invocations["root"].state == InvocationState.READY


def test_context_lifecycle_runs_summary_then_compacts_parent(tmp_path) -> None:
    trace_sink = CollectingSink()
    control_sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(
        trace_sink,
        root,
        control_sink=control_sink,
    )
    main_model = QueueToolCallingModel(
        responses=[AIMessage(content="done"), AIMessage(content="done again")]
    )
    summary_model = QueueToolCallingModel(
        responses=[AIMessage(content="durable checkpoint")],
    )
    policy = ContextLifecyclePolicy(
        window_tokens=1_000,
        keep_tokens=100,
        intermediate_output_tokens=100,
        summary_output_tokens=200,
    )
    context_lifecycle = ContextLifecycleMiddleware(
        summary_model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        policy=policy,
        compaction_sink=adapter,
        summary_callbacks=(adapter,),
        persist_cursor_across_invocations=True,
    )
    agent = create_agent(
        model=main_model,
        tools=[],
        middleware=[context_lifecycle],
    )

    adapter.start()
    agent_config = {"callbacks": [adapter], "recursion_limit": 10}
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "x" * 2_000},
                {"role": "assistant", "content": "y" * 2_000},
                {"role": "user", "content": "z" * 2_000},
            ]
        },
        config=agent_config,
    )

    assert result["messages"][-1].text == "done"
    kinds = [event.kind for event in trace_sink.events]
    assert kinds.count(RuntimeEventKind.CONTEXT_COMPACT) == 1
    internal_summary_count = sum(
        event.kind == RuntimeEventKind.LLM_SUBMIT
        and bool(event.attributes.get("runtime_internal"))
        for event in trace_sink.events
    )
    assert internal_summary_count == 1
    compact_index = kinds.index(RuntimeEventKind.CONTEXT_COMPACT)
    parent_submit_index = max(
        index
        for index, event in enumerate(trace_sink.events)
        if event.kind == RuntimeEventKind.LLM_SUBMIT
        and not bool(event.attributes.get("runtime_internal"))
    )
    assert compact_index < parent_submit_index
    assert any(
        event.kind == RuntimeEventKind.CONTEXT_COMPACT
        for event in control_sink.events
    )

    assert result.get("_summarization_event") is None
    summarization_event = context_lifecycle.latest_summarization_event()
    assert summarization_event is not None
    second_result = agent.invoke(
        {
            "messages": [
                *result["messages"],
                {"role": "user", "content": "small follow-up"},
            ],
        },
        config=agent_config,
    )
    assert second_result["messages"][-1].text == "done again"
    assert sum(
        event.kind == RuntimeEventKind.CONTEXT_COMPACT
        for event in trace_sink.events
    ) == 1


def test_failed_summary_does_not_publish_context_compaction(tmp_path) -> None:
    trace_sink = CollectingSink()
    root = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "supervisor", "root")
    adapter = DeepAgentsRuntimeAdapter(trace_sink, root)
    middleware = ContextLifecycleMiddleware(
        QueueToolCallingModel(responses=[]),
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        policy=ContextLifecyclePolicy(),
        compaction_sink=adapter,
        summary_callbacks=(adapter,),
    )

    adapter.start()
    with pytest.raises(AssertionError, match="response queue is empty"):
        middleware._create_summary([HumanMessage(content="history")])

    assert not any(
        event.kind == RuntimeEventKind.CONTEXT_COMPACT
        for event in trace_sink.events
    )
    assert any(
        event.kind == RuntimeEventKind.INVOCATION_CANCEL
        and bool(event.attributes.get("runtime_internal"))
        for event in trace_sink.events
    )
