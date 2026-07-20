from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence
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
from beliefkv.predictor.taxonomy import ToolTaxonomy
from beliefkv.runtime.agent_runtime_adapter import RuntimeEventSink
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


def _digest(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _json_stats(value: Any) -> tuple[int, str]:
    try:
        encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except (TypeError, ValueError):
        encoded = repr(value)
    return len(encoded), hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _run_key(run_id: UUID | str | None) -> str | None:
    return str(run_id) if run_id is not None else None


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
    ) -> None:
        super().__init__()
        if root_metadata.relation_type != RelationType.ROOT.value:
            raise ValueError("Deep Agents root metadata must use relation_type=root")
        self.trace_sink = trace_sink
        self.control_sink = control_sink
        self.root_metadata = root_metadata
        self.clock_ms = clock_ms or (lambda: time.monotonic() * 1000.0)
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
        self._ordinary_tools: dict[str, tuple[str, str, float, str]] = {}
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
        del serialized, kwargs
        key = self._remember_run(run_id, parent_run_id)
        invocation_id = self._resolve_invocation(key)
        with self._lock:
            base = self._identities[invocation_id].metadata
            epoch = self._model_epochs.get(invocation_id, -1) + 1
            self._model_epochs[invocation_id] = epoch
            metadata = replace(base, context_epoch=epoch)
            self._model_metadata[key] = metadata
            self._run_invocation[key] = invocation_id
        prompt_messages = messages[0] if messages else []
        prompt_chars = sum(len(message.text or "") for message in prompt_messages)
        event = self._event(
            RuntimeEventKind.LLM_SUBMIT,
            invocation_id=invocation_id,
            context_id=metadata.context_id,
            context_epoch=epoch,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "source": "deepagents_callback",
                "message_count": len(prompt_messages),
                "prompt_chars": prompt_chars,
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
            metadata = self._model_metadata.get(key)
            if metadata is None:
                base = self._identities[invocation_id].metadata
                metadata = replace(base, context_epoch=self._model_epochs.get(invocation_id, 0))
        messages = self._response_messages(response)
        output_chars = sum(len(message.text or "") for message in messages)
        tool_calls = [
            call
            for message in messages
            for call in getattr(message, "tool_calls", ())
            if isinstance(call, dict)
        ]
        result = self._event(
            RuntimeEventKind.LLM_RESULT,
            invocation_id=invocation_id,
            context_id=metadata.context_id,
            context_epoch=metadata.context_epoch,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "source": "deepagents_callback",
                "output_chars": output_chars,
                "tool_call_count": len(tool_calls),
            },
        )
        self._publish((result,), control=False)
        task_calls = [call for call in tool_calls if str(call.get("name")) == "task"]
        if task_calls:
            self._declare_task_group(invocation_id, key, task_calls)

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
                "exception_type": type(error).__name__,
            },
        )
        self._publish((event,), control=False)

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
        if tool_name == "task":
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
        with self._lock:
            self._ordinary_tools[key] = (
                parent_invocation_id,
                tool_name,
                ts_ms,
                tool_call_id,
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
                            "persistent": False,
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
            join_complete = completed >= self._join_members[pending.join_id]
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
        if join_complete:
            events.append(
                self._event(
                    RuntimeEventKind.JOIN_SATISFIED,
                    ts_ms=ts_ms,
                    join_id=pending.join_id,
                    attributes={"source": "deepagents_task"},
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
        invocation_id, tool_name, start_ts_ms, tool_call_id = active
        ts_ms = self._timestamp()
        output_chars, output_sha256 = _json_stats(output)
        event = self._event(
            RuntimeEventKind.TOOL_END,
            ts_ms=ts_ms,
            invocation_id=invocation_id,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "source": "deepagents_callback",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "duration_ms": max(0.0, ts_ms - start_ts_ms),
                "output_chars": output_chars,
                "output_sha256": output_sha256,
                "exception_type": type(error).__name__ if error else None,
            },
        )
        self._publish((event,), control=True)

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
                f"{self.root_metadata.root_workflow_id}:deepagents:"
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

    def __init__(
        self,
        *,
        beliefkv_adapter: DeepAgentsRuntimeAdapter,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._beliefkv_adapter = beliefkv_adapter

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs = self._with_beliefkv_metadata(run_manager, kwargs)
        return super()._generate(messages, stop, run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs = self._with_beliefkv_metadata(run_manager, kwargs)
        return await super()._agenerate(messages, stop, run_manager, **kwargs)

    def _with_beliefkv_metadata(
        self,
        run_manager: Any | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if run_manager is None:
            raise RuntimeError("BeliefKV ChatOpenAI requires a callback run manager")
        metadata = self._beliefkv_adapter.metadata_for_model_run(run_manager.run_id)
        extra_body = dict(kwargs.get("extra_body") or {})
        existing = extra_body.get("beliefkv_metadata")
        if existing is not None and existing != metadata.to_wire():
            raise RuntimeError("conflicting beliefkv_metadata in ChatOpenAI request")
        extra_body["beliefkv_metadata"] = metadata.to_wire()
        return {**kwargs, "extra_body": extra_body}
