from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from beliefkv.core.events import (
    ContextMode,
    EventConfidence,
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.runtime.agent_safety import classify_tool_outcome
from beliefkv.predictor.taxonomy import ToolTaxonomy
from beliefkv.runtime.agent_runtime_adapter import RuntimeEventSink
from beliefkv.runtime.action_frontier import StructuredActionKind
from beliefkv.runtime.context_lifecycle import ContextCompactionRecord
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


def _digest(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _json_stats(value: Any) -> tuple[int, str]:
    try:
        encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except (TypeError, ValueError):
        encoded = repr(value)
    return len(encoded), hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prompt_semantic_digest(
    messages: Sequence[BaseMessage],
    invocation_params: Mapping[str, Any] | None = None,
) -> str:
    message_payload = []
    for message in messages:
        tool_calls = []
        for call in getattr(message, "tool_calls", ()):
            if isinstance(call, Mapping):
                tool_calls.append(
                    {
                        "name": call.get("name"),
                        "args": call.get("args"),
                    }
                )
        message_payload.append(
            {
                "type": getattr(message, "type", type(message).__name__),
                "name": getattr(message, "name", None),
                "content": getattr(message, "content", None),
                "tool_calls": tool_calls,
                "status": getattr(message, "status", None),
            }
        )
    invocation_params = invocation_params or {}
    payload = {
        "messages": message_payload,
        "model_contract": {
            key: invocation_params.get(key)
            for key in (
                "model",
                "model_name",
                "tools",
                "functions",
                "tool_choice",
                "response_format",
                "temperature",
                "top_p",
                "stop",
                "max_tokens",
                "max_completion_tokens",
            )
            if invocation_params.get(key) is not None
        },
    }
    _, digest = _json_stats(payload)
    return digest


def _run_key(run_id: UUID | str | None) -> str | None:
    return str(run_id) if run_id is not None else None


def _native_request_id(run_id: UUID | str) -> str:
    """Return the request identity carried through the OpenAI-compatible API."""

    return f"beliefkv:{run_id}"


def _error_censor_reason(error: BaseException) -> str:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if "recursion" in name or "recursion" in message:
        return "recursion_limit"
    if "timeout" in name or "timed out" in message or "deadline" in message:
        return "timeout"
    if "cancel" in name or "cancel" in message or "abort" in message:
        return "cancelled"
    return "backend_error"


@dataclass(frozen=True)
class _InvocationIdentity:
    metadata: BeliefKVRequestMetadata


@dataclass
class _PendingTask:
    tool_call_id: str
    parent_invocation_id: str
    child_invocation_id: str
    child_context_id: str
    subagent_type: str
    description_chars: int
    description_sha256: str
    join_id: str
    tool_run_id: str | None = None
    terminal: bool = False


@dataclass(frozen=True)
class DeclaredRuntimeTask:
    """Public handle for a child selected by an external code orchestrator."""

    tool_call_id: str
    invocation_id: str
    context_id: str
    subagent_type: str
    join_id: str


@dataclass(frozen=True)
class RuntimeControlDeliveryFailure:
    ts_ms: float
    event_count: int
    first_event_id: str
    error_type: str
    error: str


@dataclass(frozen=True)
class _OrdinaryToolRun:
    invocation_id: str
    tool_name: str
    start_ts_ms: float
    tool_call_id: str
    payload: Mapping[str, Any]
    workspace_digest_before: str | None


@dataclass(frozen=True)
class _InternalSummaryRun:
    parent_invocation_id: str
    invocation_id: str
    context_id: str


class RequestDeadline(Protocol):
    def request_timeout_s(self, cap_s: float) -> float: ...

    def remaining_s(self) -> float | None: ...


class DeepAgentsRuntimeAdapter(BaseCallbackHandler):
    """Translate Deep Agents callbacks into authoritative BeliefKV events.

    The full trace sink receives all framework events. The optional control sink
    receives only events that SGLang cannot reconstruct from request metadata.
    SGLang remains the authority for its own LLM submit/result boundaries.
    """

    raise_error = True
    run_inline = True
    INVOCATION_METADATA_KEY = "beliefkv_invocation_id"

    def __init__(
        self,
        trace_sink: RuntimeEventSink,
        root_metadata: BeliefKVRequestMetadata,
        *,
        control_sink: RuntimeEventSink | None = None,
        clock_ms: Callable[[], float] | None = None,
        event_namespace: str = "deepagents",
        completion_tool_names: frozenset[str] = frozenset(
            {"ChildCompletion", "WorkflowCompletion", "DelegationPlan"}
        ),
        allowed_subagent_types: frozenset[str] | None = None,
        workspace_digest_provider: (
            Callable[[str, Mapping[str, Any]], str | None] | None
        ) = None,
    ) -> None:
        super().__init__()
        if root_metadata.relation_type != RelationType.ROOT.value:
            raise ValueError("Deep Agents root metadata must use relation_type=root")
        self.trace_sink = trace_sink
        self.control_sink = control_sink
        self.root_metadata = root_metadata
        self.clock_ms = clock_ms or (lambda: time.monotonic() * 1000.0)
        if not event_namespace or ":" in event_namespace:
            raise ValueError("event_namespace must be non-empty and contain no colon")
        self.event_namespace = event_namespace
        self.completion_tool_names = completion_tool_names
        self.allowed_subagent_types = allowed_subagent_types
        self.workspace_digest_provider = workspace_digest_provider
        self._lock = threading.RLock()
        self._sequence = 0
        self._last_ts_ms = 0.0
        self._started = False
        self._finished = False
        self._run_parent: dict[str, str | None] = {}
        self._run_invocation: dict[str, str] = {}
        self._model_metadata: dict[str, BeliefKVRequestMetadata] = {}
        self._model_epochs: dict[str, int] = {}
        self._identities: dict[str, _InvocationIdentity] = {
            root_metadata.invocation_id: _InvocationIdentity(root_metadata)
        }
        self._pending_tasks: dict[str, _PendingTask] = {}
        self._pending_by_parent: dict[str, list[str]] = {}
        self._task_run_to_call: dict[str, str] = {}
        self._join_members: dict[str, set[str]] = {}
        self._join_completed: dict[str, set[str]] = {}
        self._join_cancelled: dict[str, set[str]] = {}
        self._ordinary_tools: dict[str, _OrdinaryToolRun] = {}
        self._ignored_tool_runs: set[str] = set()
        self._internal_summary_runs: dict[str, _InternalSummaryRun] = {}
        self._summary_sequence = 0
        self._pending_context_compaction: ContextVar[
            ContextCompactionRecord | None
        ] = ContextVar(
            f"beliefkv_context_compaction_{id(self)}",
            default=None,
        )
        self._taxonomy = ToolTaxonomy()
        self._control_delivery_failure_count = 0
        self._first_control_delivery_failure: RuntimeControlDeliveryFailure | None = None
        self._last_control_delivery_failure: RuntimeControlDeliveryFailure | None = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("Deep Agents runtime adapter already started")
            self._started = True
        ts_ms = self._timestamp()
        events = (
            self._event(
                RuntimeEventKind.WORKFLOW_START,
                ts_ms=ts_ms,
                attributes={"source": "deepagents"},
            ),
            self._event(
                RuntimeEventKind.INVOCATION_CREATE,
                ts_ms=ts_ms,
                invocation_id=self.root_metadata.invocation_id,
                context_id=self.root_metadata.context_id,
                context_epoch=self.root_metadata.context_epoch,
                agent_definition_id=self.root_metadata.agent_definition_id,
                agent_instance_id=self.root_metadata.agent_instance_id,
                relation_type=RelationType.ROOT,
                context_mode=ContextMode.FRESH,
                execution_mode=ExecutionMode.FOREGROUND,
                attributes={"persistent": True, "source": "deepagents"},
            ),
        )
        self._publish(events, control=False)

    def finish(self, *, outcome: str) -> None:
        with self._lock:
            if not self._started:
                raise RuntimeError("Deep Agents runtime adapter was not started")
            if self._finished:
                return
            self._finished = True
        ts_ms = self._timestamp()
        events = (
            self._event(
                RuntimeEventKind.RETURN,
                ts_ms=ts_ms,
                invocation_id=self.root_metadata.invocation_id,
                context_id=self.root_metadata.context_id,
                attributes={"outcome": outcome, "source": "deepagents"},
            ),
            self._event(
                RuntimeEventKind.WORKFLOW_END,
                ts_ms=ts_ms,
                attributes={"outcome": outcome, "source": "deepagents"},
            ),
        )
        self._publish(events, control=True)

    def declare_runtime_tasks(
        self,
        tasks: Sequence[tuple[str, str]],
        *,
        parent_invocation_id: str | None = None,
        group_id: str | None = None,
    ) -> tuple[DeclaredRuntimeTask, ...]:
        """Declare planner-selected children before code dispatches their runs.

        ``tasks`` contains ``(subagent_type, description)`` pairs produced at
        runtime. No task bodies are written to the event stream.
        """

        if not tasks:
            return ()
        parent_id = parent_invocation_id or self.root_metadata.invocation_id
        with self._lock:
            if parent_id not in self._identities:
                raise ValueError(f"unknown parent invocation: {parent_id}")
            sequence = self._sequence + 1
        group_key = group_id or f"code-plan:{parent_id}:{sequence}"
        calls = [
            {
                "name": "task",
                "id": f"{group_key}:task:{index}",
                "type": "tool_call",
                "args": {
                    "subagent_type": subagent_type,
                    "description": description,
                },
            }
            for index, (subagent_type, description) in enumerate(tasks)
        ]
        pending = self._declare_task_group(parent_id, group_key, calls)
        return tuple(
            DeclaredRuntimeTask(
                tool_call_id=item.tool_call_id,
                invocation_id=item.child_invocation_id,
                context_id=item.child_context_id,
                subagent_type=item.subagent_type,
                join_id=item.join_id,
            )
            for item in pending
        )

    def invocation_scope(self, task: DeclaredRuntimeTask) -> dict[str, str]:
        """Return LangChain metadata that binds a top-level run to ``task``."""

        with self._lock:
            pending = self._pending_tasks.get(task.tool_call_id)
            if pending is None or pending.child_invocation_id != task.invocation_id:
                raise ValueError("runtime task handle does not belong to this adapter")
        return {self.INVOCATION_METADATA_KEY: task.invocation_id}

    def complete_runtime_task(
        self,
        task: DeclaredRuntimeTask,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Complete a child dispatched directly by a code orchestrator."""

        with self._lock:
            pending = self._pending_tasks.get(task.tool_call_id)
            if pending is None or pending.child_invocation_id != task.invocation_id:
                raise ValueError("runtime task handle does not belong to this adapter")
        self._complete_task(
            task.tool_call_id,
            cancelled=error is not None,
            error=error,
        )

    def cancel_pending_tasks(self, *, reason: str) -> int:
        """Terminate every declared child that has not produced a RETURN."""

        with self._lock:
            pending_ids = [
                tool_call_id
                for tool_call_id, pending in self._pending_tasks.items()
                if not pending.terminal
            ]
        for tool_call_id in pending_ids:
            self._complete_task(
                tool_call_id,
                cancelled=True,
                error=TimeoutError(reason),
            )
        return len(pending_ids)

    def record_call_censor(self, fields: Mapping[str, Any]) -> None:
        """Publish an identity-bearing censor label for non-executed runtime work."""

        invocation_id = str(
            fields.get("invocation_id") or self.root_metadata.invocation_id
        )
        with self._lock:
            identity = self._identities.get(invocation_id)
            if identity is None:
                invocation_id = self.root_metadata.invocation_id
                identity = self._identities[invocation_id]
            context_id = identity.metadata.context_id
            context_epoch = self._model_epochs.get(
                invocation_id, identity.metadata.context_epoch
            )
        attributes = dict(fields)
        attributes["invocation_identity_fallback"] = (
            str(fields.get("invocation_id") or "") != invocation_id
        )
        self._publish(
            (
                self._event(
                    RuntimeEventKind.CALL_CENSORED,
                    invocation_id=invocation_id,
                    context_id=context_id,
                    context_epoch=context_epoch,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes=attributes,
                ),
            ),
            control=False,
        )

    def metadata_for_model_run(self, run_id: UUID | str) -> BeliefKVRequestMetadata:
        key = _run_key(run_id)
        assert key is not None
        with self._lock:
            try:
                return self._model_metadata[key]
            except KeyError as error:
                raise RuntimeError(
                    f"model run has no BeliefKV identity: {key}"
                ) from error

    def latest_context_epoch(self, invocation_id: str | None = None) -> int:
        """Return the latest model epoch observed for one invocation."""

        target = invocation_id or self.root_metadata.invocation_id
        with self._lock:
            identity = self._identities.get(target)
            if identity is None:
                raise ValueError(f"unknown invocation: {target}")
            return self._model_epochs.get(target, identity.metadata.context_epoch)

    @contextmanager
    def stage_context_compaction(
        self, record: ContextCompactionRecord
    ) -> Iterator[None]:
        """Bind one compaction to the next non-summary model submission."""

        if self._pending_context_compaction.get() is not None:
            raise RuntimeError("nested context compaction is not supported")
        token = self._pending_context_compaction.set(record)
        try:
            yield
        finally:
            self._pending_context_compaction.reset(token)

    def _begin_internal_summary(self, model_run_key: str) -> BeliefKVRequestMetadata:
        parent_invocation_id = self._resolve_invocation(model_run_key)
        with self._lock:
            parent = self._identities[parent_invocation_id].metadata
            self._summary_sequence += 1
            suffix = f"{self._summary_sequence:06d}"
            invocation_id = f"{parent_invocation_id}:context-summary:{suffix}"
            context_id = f"{parent.context_id}:context-summary:{suffix}"
            metadata = BeliefKVRequestMetadata(
                root_workflow_id=parent.root_workflow_id,
                invocation_id=invocation_id,
                context_id=context_id,
                context_epoch=0,
                agent_definition_id="context-summarizer",
                agent_instance_id=invocation_id,
                parent_invocation_id=parent_invocation_id,
                parent_context_id=parent.context_id,
                relation_type=RelationType.CALL.value,
                context_mode=ContextMode.FRESH.value,
                execution_mode=ExecutionMode.FOREGROUND.value,
                return_target_id=parent_invocation_id,
            )
            self._identities[invocation_id] = _InvocationIdentity(metadata)
            self._run_invocation[model_run_key] = invocation_id
            self._model_metadata[model_run_key] = metadata
            self._model_epochs[invocation_id] = 0
            self._internal_summary_runs[model_run_key] = _InternalSummaryRun(
                parent_invocation_id=parent_invocation_id,
                invocation_id=invocation_id,
                context_id=context_id,
            )
        ts_ms = self._timestamp()
        self._publish(
            (
                self._event(
                    RuntimeEventKind.INVOCATION_CREATE,
                    ts_ms=ts_ms,
                    invocation_id=invocation_id,
                    context_id=context_id,
                    context_epoch=0,
                    agent_definition_id="context-summarizer",
                    agent_instance_id=invocation_id,
                    parent_invocation_id=parent_invocation_id,
                    parent_context_id=parent.context_id,
                    relation_type=RelationType.CALL,
                    context_mode=ContextMode.FRESH,
                    execution_mode=ExecutionMode.FOREGROUND,
                    return_target_id=parent_invocation_id,
                    attributes={
                        "persistent": False,
                        "runtime_internal": True,
                        "source": "deepagents_summarization",
                    },
                ),
                self._event(
                    RuntimeEventKind.CALL,
                    ts_ms=ts_ms,
                    invocation_id=parent_invocation_id,
                    target_invocation_id=invocation_id,
                    execution_mode=ExecutionMode.FOREGROUND,
                    return_target_id=parent_invocation_id,
                    attributes={
                        "runtime_internal": True,
                        "source": "deepagents_summarization",
                    },
                ),
            ),
            control=True,
        )
        return metadata

    def _finish_internal_summary(
        self, model_run_key: str, *, error: BaseException | None
    ) -> bool:
        with self._lock:
            active = self._internal_summary_runs.pop(model_run_key, None)
        if active is None:
            return False
        self._publish(
            (
                self._event(
                    (
                        RuntimeEventKind.INVOCATION_CANCEL
                        if error is not None
                        else RuntimeEventKind.RETURN
                    ),
                    invocation_id=active.invocation_id,
                    context_id=active.context_id,
                    attributes={
                        "runtime_internal": True,
                        "source": "deepagents_summarization",
                        "outcome": "error" if error is not None else "completed",
                        "exception_type": type(error).__name__ if error else None,
                    },
                ),
            ),
            control=True,
        )
        return True

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, inputs
        key = self._remember_run(run_id, parent_run_id)
        metadata = kwargs.get("metadata")
        scoped_invocation = (
            metadata.get(self.INVOCATION_METADATA_KEY)
            if isinstance(metadata, dict)
            else None
        )
        if scoped_invocation is None:
            return
        scoped_invocation = str(scoped_invocation)
        with self._lock:
            if scoped_invocation not in self._identities:
                raise RuntimeError(
                    f"LangGraph run references unknown invocation: {scoped_invocation}"
                )
            self._run_invocation[key] = scoped_invocation

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized
        key = self._remember_run(run_id, parent_run_id)
        callback_metadata = kwargs.get("metadata")
        is_summary = (
            isinstance(callback_metadata, Mapping)
            and callback_metadata.get("lc_source") == "summarization"
        )
        if is_summary:
            metadata = self._begin_internal_summary(key)
            invocation_id = metadata.invocation_id
            epoch = metadata.context_epoch
        else:
            invocation_id = self._resolve_invocation(key)
            with self._lock:
                base = self._identities[invocation_id].metadata
                previous_epoch = self._model_epochs.get(invocation_id, -1)
                epoch = previous_epoch + 1
                self._model_epochs[invocation_id] = epoch
                metadata = replace(base, context_epoch=epoch)
                self._model_metadata[key] = metadata
                self._run_invocation[key] = invocation_id
            compaction = self._pending_context_compaction.get()
            if compaction is not None:
                self._publish(
                    (
                        self._event(
                            RuntimeEventKind.CONTEXT_COMPACT,
                            invocation_id=invocation_id,
                            context_id=metadata.context_id,
                            context_epoch=epoch,
                            confidence=EventConfidence.OBSERVED_EXACT,
                            attributes={
                                "source": "deepagents_summarization",
                                "previous_context_epoch": previous_epoch,
                                "source_message_count": (
                                    compaction.source_message_count
                                ),
                                "retained_message_count": (
                                    compaction.retained_message_count
                                ),
                                "summary_chars": compaction.summary_chars,
                                "summary_sha256": compaction.summary_sha256,
                                "trigger_tokens": compaction.trigger_tokens,
                                "keep_tokens": compaction.keep_tokens,
                                "old_kv_disposition": "release_ownership",
                            },
                        ),
                    ),
                    control=True,
                )
                self._pending_context_compaction.set(None)
        prompt_messages = messages[0] if messages else []
        prompt_chars = sum(len(message.text or "") for message in prompt_messages)
        invocation_params = kwargs.get("invocation_params")
        sampling_seed = (
            invocation_params.get("seed")
            if isinstance(invocation_params, Mapping)
            else None
        )
        event = self._event(
            RuntimeEventKind.LLM_SUBMIT,
            invocation_id=invocation_id,
            context_id=metadata.context_id,
            context_epoch=epoch,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "source": "deepagents_callback",
                "runtime_internal": is_summary,
                "request_id": _native_request_id(run_id),
                "message_count": len(prompt_messages),
                "prompt_chars": prompt_chars,
                "prompt_semantic_sha256": _prompt_semantic_digest(
                    prompt_messages,
                    invocation_params
                    if isinstance(invocation_params, Mapping)
                    else None,
                ),
                "sampling_seed": sampling_seed,
            },
        )
        self._publish((event,), control=False)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        key = self._remember_run(run_id, parent_run_id)
        invocation_id = self._resolve_invocation(key)
        with self._lock:
            runtime_internal = key in self._internal_summary_runs
            metadata = self._model_metadata.get(key)
            if metadata is None:
                base = self._identities[invocation_id].metadata
                metadata = replace(
                    base,
                    context_epoch=self._model_epochs.get(invocation_id, 0),
                )
        messages = self._response_messages(response)
        output_chars = sum(len(message.text or "") for message in messages)
        tool_calls = [
            call
            for message in messages
            for call in getattr(message, "tool_calls", ())
            if isinstance(call, dict)
        ]
        action_kinds: list[StructuredActionKind] = []
        action_names: list[str] = []
        for call in tool_calls:
            name = str(call.get("name") or "unknown")
            action_names.append(name)
            if name == "task":
                action_kinds.append(StructuredActionKind.SPAWN)
            elif name == "handoff" or name.startswith("transfer_to_"):
                action_kinds.append(StructuredActionKind.HANDOFF)
            elif name in self.completion_tool_names:
                action_kinds.append(StructuredActionKind.FINAL_ANSWER)
            else:
                action_kinds.append(StructuredActionKind.FUNCTION_CALL)
        if not tool_calls and messages:
            action_kinds.append(StructuredActionKind.FINAL_ANSWER)
        output_tokens = sum(
            int((getattr(message, "usage_metadata", None) or {}).get("output_tokens", 0))
            for message in messages
        )
        task_calls = [call for call in tool_calls if str(call.get("name")) == "task"]
        executable_task_calls = [
            call
            for call in task_calls
            if self.allowed_subagent_types is None
            or str(
                (
                    call.get("args")
                    if isinstance(call.get("args"), dict)
                    else {}
                ).get("subagent_type")
            )
            in self.allowed_subagent_types
        ]
        result = self._event(
            RuntimeEventKind.LLM_RESULT,
            invocation_id=invocation_id,
            context_id=metadata.context_id,
            context_epoch=metadata.context_epoch,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "source": "deepagents_callback",
                "runtime_internal": runtime_internal,
                "request_id": _native_request_id(run_id),
                "output_chars": output_chars,
                "output_tokens": output_tokens or None,
                "tool_call_count": len(tool_calls),
                "rejected_task_call_count": len(task_calls)
                - len(executable_task_calls),
                "parser_status": "valid" if action_kinds else "unknown",
                "structured_action_kinds": [item.value for item in action_kinds],
                "structured_action_names": action_names,
                "action_boundary_token_index": None,
                "action_boundary_source": "runtime_structured_output",
                "parser_reason": (
                    "Deep Agents exposed a complete AIMessage; incremental token boundary "
                    "is unavailable"
                ),
            },
        )
        self._publish((result,), control=False)
        if self._finish_internal_summary(key, error=None):
            return
        if executable_task_calls:
            self._declare_task_group(invocation_id, key, executable_task_calls)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        key = self._remember_run(run_id, parent_run_id)
        invocation_id = self._resolve_invocation(key)
        with self._lock:
            runtime_internal = key in self._internal_summary_runs
            metadata = self._model_metadata.get(key)
            context_id = (
                metadata.context_id
                if metadata is not None
                else self._identities[invocation_id].metadata.context_id
            )
            epoch = metadata.context_epoch if metadata is not None else None
        event = self._event(
            RuntimeEventKind.LLM_RESULT,
            invocation_id=invocation_id,
            context_id=context_id,
            context_epoch=epoch,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "source": "deepagents_callback",
                "runtime_internal": runtime_internal,
                "request_id": _native_request_id(run_id),
                "exception_type": type(error).__name__,
                "censored": True,
                "censor_reason": _error_censor_reason(error),
            },
        )
        self._publish((event,), control=False)
        self.record_call_censor(
            {
                "call_kind": "llm",
                "censor_reason": _error_censor_reason(error),
                "request_id": _native_request_id(run_id),
                "invocation_id": invocation_id,
                "exception_type": type(error).__name__,
            }
        )
        self._finish_internal_summary(key, error=error)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del input_str
        key = self._remember_run(run_id, parent_run_id)
        parent_invocation_id = self._resolve_invocation(key)
        tool_name = str((serialized or {}).get("name") or "unknown")
        payload = inputs or {}
        if tool_name in self.completion_tool_names:
            with self._lock:
                self._ignored_tool_runs.add(key)
            return
        if tool_name == "task":
            requested_type = str(payload.get("subagent_type") or "general-purpose")
            if (
                self.allowed_subagent_types is not None
                and requested_type not in self.allowed_subagent_types
            ):
                with self._lock:
                    self._ignored_tool_runs.add(key)
                return
            pending = self._bind_task_run(
                key,
                parent_invocation_id,
                payload,
                kwargs,
            )
            with self._lock:
                self._run_invocation[key] = pending.child_invocation_id
            return

        ts_ms = self._timestamp()
        normalized = self._taxonomy.normalize(tool_name)
        input_chars, input_sha256 = _json_stats(payload)
        tool_call_id = str(kwargs.get("tool_call_id") or key)
        workspace_digest_before = self._workspace_digest(tool_name, payload)
        with self._lock:
            self._ordinary_tools[key] = _OrdinaryToolRun(
                invocation_id=parent_invocation_id,
                tool_name=tool_name,
                start_ts_ms=ts_ms,
                tool_call_id=tool_call_id,
                payload=dict(payload),
                workspace_digest_before=workspace_digest_before,
            )
        event = self._event(
            RuntimeEventKind.TOOL_START,
            ts_ms=ts_ms,
            invocation_id=parent_invocation_id,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "source": "deepagents_callback",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_family": normalized.family,
                "backend_class": normalized.backend_class,
                "input_chars": input_chars,
                "input_sha256": input_sha256,
                "parameter_signature": input_sha256,
                "workspace_digest_before": workspace_digest_before,
            },
        )
        self._publish((event,), control=True)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        key = self._remember_run(run_id, parent_run_id)
        with self._lock:
            if key in self._ignored_tool_runs:
                self._ignored_tool_runs.remove(key)
                return
            task_call_id = self._task_run_to_call.get(key)
        if task_call_id is not None:
            self._complete_task(task_call_id, cancelled=False, error=None)
            return
        self._finish_ordinary_tool(key, output=output, error=None)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        key = self._remember_run(run_id, parent_run_id)
        with self._lock:
            if key in self._ignored_tool_runs:
                self._ignored_tool_runs.remove(key)
                return
            task_call_id = self._task_run_to_call.get(key)
        if task_call_id is not None:
            self._complete_task(task_call_id, cancelled=True, error=error)
            return
        self._finish_ordinary_tool(key, output=None, error=error)

    def _declare_task_group(
        self,
        parent_invocation_id: str,
        model_run_id: str,
        task_calls: list[dict[str, Any]],
    ) -> tuple[_PendingTask, ...]:
        parent = self._identities[parent_invocation_id].metadata
        join_id = (
            "deepagents-join:"
            f"{_digest(f'{self.root_metadata.root_workflow_id}:{model_run_id}')}"
        )
        pending_items: list[_PendingTask] = []
        events: list[RuntimeEvent] = []
        ts_ms = self._timestamp()
        for index, call in enumerate(task_calls):
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            tool_call_id = str(call.get("id") or f"{model_run_id}:task:{index}")
            subagent_type = str(args.get("subagent_type") or "general-purpose")
            description = str(args.get("description") or "")
            description_chars = len(description)
            description_sha256 = hashlib.sha256(
                description.encode("utf-8")
            ).hexdigest()
            child_suffix = _digest(f"{parent_invocation_id}:{tool_call_id}")
            child_invocation_id = f"deepagents-invocation:{child_suffix}"
            child_context_id = f"deepagents-context:{child_suffix}"
            metadata = BeliefKVRequestMetadata(
                root_workflow_id=self.root_metadata.root_workflow_id,
                invocation_id=child_invocation_id,
                context_id=child_context_id,
                context_epoch=0,
                agent_definition_id=subagent_type,
                agent_instance_id=child_invocation_id,
                parent_invocation_id=parent_invocation_id,
                parent_context_id=parent.context_id,
                relation_type=RelationType.SPAWN.value,
                context_mode=ContextMode.FRESH.value,
                execution_mode=ExecutionMode.BACKGROUND.value,
                return_target_id=parent_invocation_id,
                join_id=join_id,
            )
            pending = _PendingTask(
                tool_call_id=tool_call_id,
                parent_invocation_id=parent_invocation_id,
                child_invocation_id=child_invocation_id,
                child_context_id=child_context_id,
                subagent_type=subagent_type,
                description_chars=description_chars,
                description_sha256=description_sha256,
                join_id=join_id,
            )
            pending_items.append(pending)
            events.extend(
                (
                    self._event(
                        RuntimeEventKind.INVOCATION_CREATE,
                        ts_ms=ts_ms,
                        invocation_id=child_invocation_id,
                        context_id=child_context_id,
                        context_epoch=0,
                        parent_invocation_id=parent_invocation_id,
                        parent_context_id=parent.context_id,
                        agent_definition_id=subagent_type,
                        agent_instance_id=child_invocation_id,
                        relation_type=RelationType.SPAWN,
                        context_mode=ContextMode.FRESH,
                        execution_mode=ExecutionMode.BACKGROUND,
                        return_target_id=parent_invocation_id,
                        join_id=join_id,
                        attributes={
                            "persistent": True,
                            "source": "deepagents_task",
                            "description_chars": description_chars,
                            "description_sha256": description_sha256,
                        },
                    ),
                    self._event(
                        RuntimeEventKind.SPAWN,
                        ts_ms=ts_ms,
                        invocation_id=parent_invocation_id,
                        target_invocation_id=child_invocation_id,
                        execution_mode=ExecutionMode.BACKGROUND,
                        return_target_id=parent_invocation_id,
                        attributes={
                            "source": "deepagents_task",
                            "subagent_type": subagent_type,
                            "tool_call_id": tool_call_id,
                        },
                    ),
                )
            )
        child_ids = tuple(item.child_invocation_id for item in pending_items)
        events.extend(
            (
                self._event(
                    RuntimeEventKind.JOIN_CREATE,
                    ts_ms=ts_ms,
                    join_id=join_id,
                    member_invocation_ids=child_ids,
                    attributes={"mode": "all", "source": "deepagents_task"},
                ),
                self._event(
                    RuntimeEventKind.JOIN_WAIT,
                    ts_ms=ts_ms,
                    invocation_id=parent_invocation_id,
                    join_id=join_id,
                    attributes={"source": "deepagents_task"},
                ),
            )
        )
        with self._lock:
            for pending in pending_items:
                self._pending_tasks[pending.tool_call_id] = pending
                self._identities[pending.child_invocation_id] = _InvocationIdentity(
                    BeliefKVRequestMetadata(
                        root_workflow_id=self.root_metadata.root_workflow_id,
                        invocation_id=pending.child_invocation_id,
                        context_id=pending.child_context_id,
                        context_epoch=0,
                        agent_definition_id=pending.subagent_type,
                        agent_instance_id=pending.child_invocation_id,
                        parent_invocation_id=parent_invocation_id,
                        parent_context_id=parent.context_id,
                        relation_type=RelationType.SPAWN.value,
                        context_mode=ContextMode.FRESH.value,
                        execution_mode=ExecutionMode.BACKGROUND.value,
                        return_target_id=parent_invocation_id,
                        join_id=join_id,
                    )
                )
                self._pending_by_parent.setdefault(parent_invocation_id, []).append(
                    pending.tool_call_id
                )
            self._join_members[join_id] = set(child_ids)
            self._join_completed[join_id] = set()
            self._join_cancelled[join_id] = set()
        self._publish(tuple(events), control=True)
        return tuple(pending_items)

    def _bind_task_run(
        self,
        tool_run_id: str,
        parent_invocation_id: str,
        inputs: dict[str, Any],
        callback_kwargs: dict[str, Any],
    ) -> _PendingTask:
        requested_id = callback_kwargs.get("tool_call_id")
        description = str(inputs.get("description") or "")
        description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()
        subagent_type = str(inputs.get("subagent_type") or "general-purpose")
        with self._lock:
            candidate_ids = list(self._pending_by_parent.get(parent_invocation_id, ()))
            pending = None
            if requested_id is not None:
                pending = self._pending_tasks.get(str(requested_id))
            if pending is None:
                for candidate_id in candidate_ids:
                    candidate = self._pending_tasks[candidate_id]
                    if candidate.tool_run_id is not None:
                        continue
                    if (
                        candidate.subagent_type == subagent_type
                        and candidate.description_sha256 == description_sha256
                    ):
                        pending = candidate
                        break
            if pending is None:
                for candidate_id in candidate_ids:
                    candidate = self._pending_tasks[candidate_id]
                    if candidate.tool_run_id is None:
                        pending = candidate
                        break
            if pending is None:
                raise RuntimeError(
                    "task callback did not match a declared Deep Agents tool call"
                )
            pending.tool_run_id = tool_run_id
            self._task_run_to_call[tool_run_id] = pending.tool_call_id
            return pending

    def _complete_task(
        self,
        tool_call_id: str,
        *,
        cancelled: bool,
        error: BaseException | None,
    ) -> None:
        with self._lock:
            pending = self._pending_tasks[tool_call_id]
            if pending.terminal:
                return
            pending.terminal = True
            completed = self._join_completed[pending.join_id]
            completed.add(pending.child_invocation_id)
            cancelled_members = self._join_cancelled[pending.join_id]
            if cancelled:
                cancelled_members.add(pending.child_invocation_id)
            join_complete = completed >= self._join_members[pending.join_id]
            join_satisfied = join_complete and not cancelled_members
        ts_ms = self._timestamp()
        kind = (
            RuntimeEventKind.INVOCATION_CANCEL
            if cancelled
            else RuntimeEventKind.RETURN
        )
        events = [
            self._event(
                kind,
                ts_ms=ts_ms,
                invocation_id=pending.child_invocation_id,
                context_id=pending.child_context_id,
                attributes={
                    "source": "deepagents_task",
                    "outcome": "error" if cancelled else "completed",
                    "exception_type": type(error).__name__ if error else None,
                },
            )
        ]
        if join_satisfied:
            events.append(
                self._event(
                    RuntimeEventKind.JOIN_SATISFIED,
                    ts_ms=ts_ms,
                    join_id=pending.join_id,
                    attributes={"source": "deepagents_task"},
                )
            )
        elif join_complete and cancelled_members:
            events.append(
                self._event(
                    RuntimeEventKind.JOIN_TIMEOUT,
                    ts_ms=ts_ms,
                    join_id=pending.join_id,
                    member_invocation_ids=tuple(
                        sorted(self._join_members[pending.join_id])
                    ),
                    attributes={
                        "source": "deepagents_task",
                        "cancelled_member_count": len(cancelled_members),
                    },
                )
            )
        self._publish(tuple(events), control=True)

    def _finish_ordinary_tool(
        self,
        tool_run_id: str,
        *,
        output: Any,
        error: BaseException | None,
    ) -> None:
        with self._lock:
            active = self._ordinary_tools.pop(tool_run_id, None)
        if active is None:
            return
        ts_ms = self._timestamp()
        output_chars, output_sha256 = _json_stats(output)
        outcome = classify_tool_outcome(
            output,
            tool_name=active.tool_name,
            error=error,
        )
        workspace_digest_after = self._workspace_digest(
            active.tool_name,
            active.payload,
        )
        event = self._event(
            RuntimeEventKind.TOOL_END,
            ts_ms=ts_ms,
            invocation_id=active.invocation_id,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "source": "deepagents_callback",
                "tool_call_id": active.tool_call_id,
                "tool_name": active.tool_name,
                "duration_ms": max(0.0, ts_ms - active.start_ts_ms),
                "output_chars": output_chars,
                "output_sha256": output_sha256,
                "exception_type": type(error).__name__ if error else None,
                "status": outcome.status,
                "tool_error_class": outcome.error_class,
                "workspace_digest_before": active.workspace_digest_before,
                "workspace_digest_after": workspace_digest_after,
                "workspace_changed": (
                    active.workspace_digest_before != workspace_digest_after
                    if active.workspace_digest_before is not None
                    and workspace_digest_after is not None
                    else None
                ),
            },
        )
        self._publish((event,), control=True)

    def _workspace_digest(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
    ) -> str | None:
        provider = self.workspace_digest_provider
        if provider is None:
            return None
        try:
            return provider(tool_name, payload)
        except Exception:
            # Observability must never alter the tool result seen by the model.
            return None

    def _remember_run(
        self,
        run_id: UUID | str,
        parent_run_id: UUID | str | None,
    ) -> str:
        key = _run_key(run_id)
        assert key is not None
        parent = _run_key(parent_run_id)
        with self._lock:
            self._run_parent[key] = parent
        return key

    def _resolve_invocation(self, run_id: str) -> str:
        with self._lock:
            current: str | None = run_id
            visited: set[str] = set()
            while current is not None:
                if current in visited:
                    raise RuntimeError("LangGraph callback run ancestry contains a cycle")
                visited.add(current)
                invocation_id = self._run_invocation.get(current)
                if invocation_id is not None:
                    return invocation_id
                task_call_id = self._task_run_to_call.get(current)
                if task_call_id is not None:
                    return self._pending_tasks[task_call_id].child_invocation_id
                current = self._run_parent.get(current)
            return self.root_metadata.invocation_id

    @staticmethod
    def _response_messages(response: Any) -> list[AIMessage]:
        messages: list[AIMessage] = []
        for generation_group in getattr(response, "generations", ()):
            for generation in generation_group:
                message = getattr(generation, "message", None)
                if isinstance(message, AIMessage):
                    messages.append(message)
        return messages

    def _timestamp(self) -> float:
        with self._lock:
            value = float(self.clock_ms())
            self._last_ts_ms = max(self._last_ts_ms, value)
            return self._last_ts_ms

    def _event(
        self,
        kind: RuntimeEventKind,
        *,
        ts_ms: float | None = None,
        confidence: EventConfidence = EventConfidence.DECLARED_RUNTIME,
        **kwargs: Any,
    ) -> RuntimeEvent:
        with self._lock:
            self._sequence += 1
            event_id = (
                f"{self.root_metadata.root_workflow_id}:{self.event_namespace}:"
                f"{self._sequence:09d}"
            )
        return RuntimeEvent(
            event_id=event_id,
            ts_ms=self._timestamp() if ts_ms is None else ts_ms,
            kind=kind,
            workflow_id=self.root_metadata.root_workflow_id,
            confidence=confidence,
            **kwargs,
        )

    def _publish(
        self,
        events: tuple[RuntimeEvent, ...],
        *,
        control: bool,
    ) -> None:
        if not events:
            return
        self.trace_sink.emit_batch(events)
        if (
            control
            and self.control_sink is not None
            and self.control_sink is not self.trace_sink
        ):
            try:
                self.control_sink.emit_batch(events)
            except Exception as error:
                # The local trace is authoritative and was durably written
                # above.  Losing the optional live control channel invalidates
                # control-plane measurements, but must not be reported as an
                # agent/tool failure or alter the workflow trajectory.
                failure = RuntimeControlDeliveryFailure(
                    ts_ms=self._timestamp(),
                    event_count=len(events),
                    first_event_id=events[0].event_id,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                with self._lock:
                    self._control_delivery_failure_count += 1
                    if self._first_control_delivery_failure is None:
                        self._first_control_delivery_failure = failure
                    self._last_control_delivery_failure = failure

    def control_delivery_summary(self) -> dict[str, object]:
        with self._lock:
            first = self._first_control_delivery_failure
            last = self._last_control_delivery_failure

            def payload(
                failure: RuntimeControlDeliveryFailure | None,
            ) -> dict[str, object] | None:
                if failure is None:
                    return None
                return {
                    "ts_ms": failure.ts_ms,
                    "event_count": failure.event_count,
                    "first_event_id": failure.first_event_id,
                    "error_type": failure.error_type,
                    "error": failure.error,
                }

            return {
                "degraded": self._control_delivery_failure_count > 0,
                "failure_count": self._control_delivery_failure_count,
                "first_failure": payload(first),
                "last_failure": payload(last),
            }


class BeliefKVChatOpenAI(ChatOpenAI):
    """ChatOpenAI client that tags every request with its runtime identity."""

    _beliefkv_adapter: DeepAgentsRuntimeAdapter = PrivateAttr()
    _activation_deadline: RequestDeadline | None = PrivateAttr(default=None)
    _request_timeout_cap_s: float | None = PrivateAttr(default=None)
    _abort_url: str | None = PrivateAttr(default=None)
    _active_rids: set[str] = PrivateAttr(default_factory=set)
    _active_rids_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(
        self,
        *,
        beliefkv_adapter: DeepAgentsRuntimeAdapter,
        activation_deadline: RequestDeadline | None = None,
        request_timeout_s: float | None = None,
        abort_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        if request_timeout_s is not None and request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        super().__init__(**kwargs)
        self._beliefkv_adapter = beliefkv_adapter
        self._activation_deadline = activation_deadline
        self._request_timeout_cap_s = request_timeout_s
        self._abort_url = abort_url

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        rid: str | None = None
        try:
            kwargs, rid = self._with_beliefkv_runtime(run_manager, kwargs)
            self._track_request(rid)
            return super()._generate(messages, stop, run_manager, **kwargs)
        except BaseException:
            if rid is not None:
                self._abort_request(rid)
            raise
        finally:
            if rid is not None:
                self._untrack_request(rid)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        rid: str | None = None
        try:
            kwargs, rid = self._with_beliefkv_runtime(run_manager, kwargs)
            self._track_request(rid)
            return await super()._agenerate(messages, stop, run_manager, **kwargs)
        except BaseException:
            if rid is not None:
                self._abort_request(rid)
            raise
        finally:
            if rid is not None:
                self._untrack_request(rid)

    def _with_beliefkv_runtime(
        self,
        run_manager: Any | None,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if run_manager is None:
            raise RuntimeError("BeliefKV ChatOpenAI requires a callback run manager")
        metadata = self._beliefkv_adapter.metadata_for_model_run(run_manager.run_id)
        if self._request_timeout_cap_s is not None:
            metadata = replace(
                metadata,
                execution_timeout_s=self._request_timeout_cap_s,
            )
        rid = _native_request_id(run_manager.run_id)
        extra_body = dict(kwargs.get("extra_body") or {})
        existing = extra_body.get("beliefkv_metadata")
        if existing is not None and existing != metadata.to_wire():
            raise RuntimeError("conflicting beliefkv_metadata in ChatOpenAI request")
        existing_rid = extra_body.get("rid")
        if existing_rid is not None and existing_rid != rid:
            raise RuntimeError("conflicting rid in ChatOpenAI request")
        extra_body["beliefkv_metadata"] = metadata.to_wire()
        extra_body["rid"] = rid
        request_kwargs = {**kwargs, "extra_body": extra_body}
        if self._activation_deadline is not None:
            remaining_s = self._activation_deadline.remaining_s()
            if remaining_s is None:
                remaining_s = self._request_timeout_cap_s
            if remaining_s is not None and remaining_s <= 0:
                self._activation_deadline.request_timeout_s(
                    self._request_timeout_cap_s or 1.0
                )
            if remaining_s is not None:
                request_kwargs["timeout"] = remaining_s
        elif self._request_timeout_cap_s is not None:
            request_kwargs["timeout"] = self._request_timeout_cap_s
        return request_kwargs, rid

    def cancel_active_requests(self) -> int:
        with self._active_rids_lock:
            active = tuple(self._active_rids)
        for rid in active:
            self._abort_request(rid)
        return len(active)

    def _track_request(self, rid: str) -> None:
        with self._active_rids_lock:
            self._active_rids.add(rid)

    def _untrack_request(self, rid: str) -> None:
        with self._active_rids_lock:
            self._active_rids.discard(rid)

    def _abort_request(self, rid: str) -> None:
        if self._abort_url is None:
            return
        request = urllib.request.Request(
            self._abort_url,
            data=json.dumps({"rid": rid}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1.0):
                pass
        except Exception:
            # Preserve the original model failure. SGLang abort is idempotent and
            # best-effort when the request failed before reaching the server.
            return
