from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from beliefkv.core.events import (
    ContextMode,
    EventConfidence,
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.predictor.taxonomy import ToolTaxonomy
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


class RuntimeEventSink(Protocol):
    def emit_batch(self, events: tuple[RuntimeEvent, ...]) -> None:
        ...


@dataclass(frozen=True)
class CodexThreadIdentity:
    thread_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    parent_thread_id: str | None
    agent_role: str
    agent_nickname: str | None = None


class CodexThreadRegistry:
    """Thread-safe mapping from Codex thread IDs to BeliefKV identities."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._identities: dict[str, CodexThreadIdentity] = {}
        self._request_epochs: dict[str, int] = {}
        self._joined_children: set[str] = set()
        self._active_child_joins: dict[str, set[str]] = {}

    def register(self, identity: CodexThreadIdentity) -> bool:
        with self._condition:
            previous = self._identities.get(identity.thread_id)
            if previous is not None:
                if previous != identity:
                    raise ValueError(
                        f"conflicting identity for Codex thread {identity.thread_id}"
                    )
                return False
            self._identities[identity.thread_id] = identity
            self._request_epochs[identity.thread_id] = 0
            self._condition.notify_all()
            return True

    def get(self, thread_id: str) -> CodexThreadIdentity | None:
        with self._condition:
            return self._identities.get(thread_id)

    def wait(self, thread_id: str, timeout_s: float = 5.0) -> CodexThreadIdentity:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while thread_id not in self._identities:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Codex thread identity was not observed: {thread_id}"
                    )
                self._condition.wait(remaining)
            return self._identities[thread_id]

    def metadata_for_request(
        self, thread_id: str, *, timeout_s: float = 5.0
    ) -> BeliefKVRequestMetadata:
        identity = self.wait(thread_id, timeout_s)
        with self._condition:
            epoch = self._request_epochs[thread_id]
            self._request_epochs[thread_id] = epoch + 1
        parent = (
            self.wait(identity.parent_thread_id, timeout_s)
            if identity.parent_thread_id is not None
            else None
        )
        return BeliefKVRequestMetadata(
            root_workflow_id=identity.workflow_id,
            invocation_id=identity.invocation_id,
            context_id=identity.context_id,
            context_epoch=epoch,
            agent_definition_id=identity.agent_role,
            agent_instance_id=identity.thread_id,
            parent_invocation_id=parent.invocation_id if parent else None,
            parent_context_id=parent.context_id if parent else None,
            relation_type="spawn" if parent else "root",
            context_mode="fork" if parent else ("fresh" if epoch == 0 else "resume"),
            execution_mode="background" if parent else "foreground",
            return_target_id=parent.invocation_id if parent else None,
        )

    def identities(self) -> tuple[CodexThreadIdentity, ...]:
        with self._condition:
            return tuple(self._identities.values())

    def direct_child_ids(self, parent_thread_id: str) -> tuple[str, ...]:
        with self._condition:
            return tuple(
                sorted(
                    identity.thread_id
                    for identity in self._identities.values()
                    if identity.parent_thread_id == parent_thread_id
                )
            )

    def unjoined_child_ids(self, parent_thread_id: str) -> tuple[str, ...]:
        with self._condition:
            return tuple(
                child_id
                for child_id in self.direct_child_ids(parent_thread_id)
                if child_id not in self._joined_children
            )

    def mark_child_joined(self, parent_thread_id: str, child_thread_id: str) -> None:
        with self._condition:
            child = self._identities.get(child_thread_id)
            if child is None or child.parent_thread_id != parent_thread_id:
                raise ValueError(
                    f"{child_thread_id} is not a child of {parent_thread_id}"
                )
            self._joined_children.add(child_thread_id)
            self._condition.notify_all()

    def begin_child_join(
        self, parent_thread_id: str, child_thread_ids: tuple[str, ...]
    ) -> None:
        with self._condition:
            if parent_thread_id in self._active_child_joins:
                raise ValueError(f"child join already active for {parent_thread_id}")
            children = set(child_thread_ids)
            if not children:
                raise ValueError("child join must contain at least one child")
            direct_children = set(self.direct_child_ids(parent_thread_id))
            if not children <= direct_children:
                raise ValueError("child join contains a thread owned by another parent")
            self._active_child_joins[parent_thread_id] = children
            self._condition.notify_all()

    def finish_child_join(self, parent_thread_id: str) -> None:
        with self._condition:
            self._active_child_joins.pop(parent_thread_id, None)
            self._condition.notify_all()

    def child_join_active(self, parent_thread_id: str) -> bool:
        with self._condition:
            return parent_thread_id in self._active_child_joins


class CodexRuntimeAdapter:
    """Translate Codex app-server notifications into authoritative events.

    The adapter consumes only runtime IDs exposed by Codex. It never infers a
    parent/child edge from prompt text or agent names.
    """

    TERMINAL_AGENT_STATES = {"completed", "errored", "shutdown", "interrupted"}

    def __init__(
        self,
        sink: RuntimeEventSink,
        registry: CodexThreadRegistry | None = None,
        *,
        clock_ms: Callable[[], float] | None = None,
    ) -> None:
        self.sink = sink
        self.registry = registry or CodexThreadRegistry()
        self.clock_ms = clock_ms or (lambda: time.monotonic() * 1000.0)
        self._lock = threading.RLock()
        self._sequence = 0
        self._last_ts_ms = 0.0
        self._terminal_threads: set[str] = set()
        self._root_outcomes: dict[str, str] = {}
        self._ended_workflows: set[str] = set()
        self._join_by_item: dict[str, tuple[str, str, tuple[str, ...]]] = {}
        self._tool_start_ms: dict[tuple[str, str], float] = {}

    def register_root(self, thread: dict[str, Any], *, workload_id: str) -> None:
        thread_id = self._required_string(thread, "id")
        if thread.get("parentThreadId") is not None:
            raise ValueError("a root Codex thread cannot have parentThreadId")
        identity = CodexThreadIdentity(
            thread_id=thread_id,
            workflow_id=f"codex:{workload_id}:{thread_id}",
            invocation_id=f"codex-invocation:{thread_id}",
            context_id=f"codex-context:{thread_id}",
            parent_thread_id=None,
            agent_role=str(thread.get("agentRole") or "codex-parent"),
            agent_nickname=thread.get("agentNickname"),
        )
        if not self.registry.register(identity):
            return
        ts_ms = self._timestamp()
        self._emit(
            (
                self._event(
                    RuntimeEventKind.WORKFLOW_START,
                    identity.workflow_id,
                    ts_ms=ts_ms,
                    attributes={
                        "source": "codex_app_server",
                        "codex_thread_id": thread_id,
                        "workload_id": workload_id,
                    },
                ),
                self._invocation_create(identity, ts_ms),
            )
        )

    def handle_notification(self, message: dict[str, Any]) -> None:
        method = str(message.get("method", ""))
        params = message.get("params")
        if not isinstance(params, dict):
            return
        if method == "thread/started":
            thread = params.get("thread")
            if isinstance(thread, dict) and thread.get("parentThreadId"):
                self._register_child(thread)
        elif method in {"item/started", "item/completed"}:
            self._handle_item(method, params)
        elif method == "turn/completed":
            self._handle_turn_completed(params)

    def finish_incomplete_workflows(self, *, outcome: str) -> None:
        """Close roots after app-server failure without fabricating child success."""

        identities = self.registry.identities()
        for identity in sorted(
            identities, key=lambda item: item.parent_thread_id is None
        ):
            if identity.thread_id in self._terminal_threads:
                continue
            self._return_thread(identity.thread_id, outcome=outcome)

    def _register_child(self, thread: dict[str, Any]) -> None:
        thread_id = self._required_string(thread, "id")
        parent_thread_id = self._required_string(thread, "parentThreadId")
        self._register_child_identity(
            thread_id,
            parent_thread_id,
            agent_role=str(thread.get("agentRole") or "codex-subagent"),
            agent_nickname=thread.get("agentNickname"),
            source="codex_thread_started",
        )

    def _register_child_identity(
        self,
        thread_id: str,
        parent_thread_id: str,
        *,
        agent_role: str,
        agent_nickname: str | None,
        source: str,
    ) -> None:
        existing = self.registry.get(thread_id)
        if existing is not None:
            if existing.parent_thread_id != parent_thread_id:
                raise ValueError(f"conflicting parent for Codex thread {thread_id}")
            return
        parent = self.registry.wait(parent_thread_id)
        identity = CodexThreadIdentity(
            thread_id=thread_id,
            workflow_id=parent.workflow_id,
            invocation_id=f"codex-invocation:{thread_id}",
            context_id=f"codex-context:{thread_id}",
            parent_thread_id=parent_thread_id,
            agent_role=agent_role,
            agent_nickname=agent_nickname,
        )
        if not self.registry.register(identity):
            return
        ts_ms = self._timestamp()
        self._emit(
            (
                self._invocation_create(identity, ts_ms),
                self._event(
                    RuntimeEventKind.SPAWN,
                    identity.workflow_id,
                    ts_ms=ts_ms,
                    invocation_id=parent.invocation_id,
                    target_invocation_id=identity.invocation_id,
                    context_id=parent.context_id,
                    target_context_id=identity.context_id,
                    execution_mode=ExecutionMode.BACKGROUND,
                    return_target_id=parent.invocation_id,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "source": source,
                        "child_thread_id": thread_id,
                    },
                ),
            )
        )

    def _handle_item(self, method: str, params: dict[str, Any]) -> None:
        item = params.get("item")
        if not isinstance(item, dict):
            return
        thread_id = str(params.get("threadId", ""))
        identity = self.registry.get(thread_id)
        if identity is None:
            return
        item_type = item.get("type")
        if item_type == "collabAgentToolCall":
            self._handle_collab_item(method, identity, item, params)
        elif item_type == "commandExecution":
            self._handle_command_item(method, identity, item, params)

    def _handle_collab_item(
        self,
        method: str,
        identity: CodexThreadIdentity,
        item: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        tool = str(item.get("tool", ""))
        item_id = self._required_string(item, "id")
        receiver_ids = tuple(str(value) for value in item.get("receiverThreadIds", ()))
        if (
            tool in {"spawnAgent", "spawn_agent"}
            and method == "item/completed"
            and receiver_ids
        ):
            for receiver_id in receiver_ids:
                self._register_child_identity(
                    receiver_id,
                    identity.thread_id,
                    agent_role="codex-subagent",
                    agent_nickname=None,
                    source="codex_collab_spawn_completed",
                )
        elif tool == "wait" and method == "item/started" and receiver_ids:
            members = tuple(
                child.invocation_id
                for receiver_id in receiver_ids
                if (child := self.registry.get(receiver_id)) is not None
            )
            if not members:
                return
            self.registry.begin_child_join(identity.thread_id, receiver_ids)
            join_id = f"codex-join:{item_id}"
            self._join_by_item[item_id] = (identity.thread_id, join_id, receiver_ids)
            ts_ms = self._timestamp()
            self._emit(
                (
                    self._event(
                        RuntimeEventKind.JOIN_CREATE,
                        identity.workflow_id,
                        ts_ms=ts_ms,
                        invocation_id=identity.invocation_id,
                        join_id=join_id,
                        member_invocation_ids=members,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={"mode": "any", "codex_item_id": item_id},
                    ),
                    self._event(
                        RuntimeEventKind.JOIN_WAIT,
                        identity.workflow_id,
                        ts_ms=ts_ms,
                        invocation_id=identity.invocation_id,
                        context_id=identity.context_id,
                        join_id=join_id,
                        member_invocation_ids=members,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "codex_item_id": item_id,
                            "source_started_at_ms": params.get("startedAtMs"),
                        },
                    ),
                )
            )
        elif tool == "wait" and method == "item/completed":
            join = self._join_by_item.pop(item_id, None)
            if join is None:
                return
            parent_thread_id, join_id, joined_threads = join
            self.registry.finish_child_join(parent_thread_id)
            completed_members: list[str] = []
            for child_id in joined_threads:
                state = item.get("agentsStates", {}).get(child_id, {})
                if state.get("status") in self.TERMINAL_AGENT_STATES:
                    child = self.registry.get(child_id)
                    if child is not None:
                        completed_members.append(child.invocation_id)
                        self.registry.mark_child_joined(identity.thread_id, child_id)
                    self._return_thread(
                        child_id,
                        outcome=str(state.get("status")),
                        message_present=bool(state.get("message")),
                    )
            completion_kind = (
                RuntimeEventKind.JOIN_SATISFIED
                if completed_members
                else RuntimeEventKind.JOIN_TIMEOUT
            )
            self._emit(
                (
                    self._event(
                        completion_kind,
                        identity.workflow_id,
                        ts_ms=self._timestamp(),
                        invocation_id=identity.invocation_id,
                        context_id=identity.context_id,
                        join_id=join_id,
                        member_invocation_ids=tuple(completed_members),
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "codex_item_id": item_id,
                            "status": item.get("status"),
                            "completion_reason": (
                                "terminal_member" if completed_members else "timeout"
                            ),
                            "source_completed_at_ms": params.get("completedAtMs"),
                        },
                    ),
                )
            )

    def _handle_command_item(
        self,
        method: str,
        identity: CodexThreadIdentity,
        item: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        item_id = self._required_string(item, "id")
        key = (identity.thread_id, item_id)
        if method == "item/started":
            ts_ms = self._timestamp()
            self._tool_start_ms[key] = ts_ms
            command = str(item.get("command", ""))
            normalized = ToolTaxonomy().normalize("exec_command", command)
            self._emit(
                (
                    self._event(
                        RuntimeEventKind.TOOL_START,
                        identity.workflow_id,
                        ts_ms=ts_ms,
                        invocation_id=identity.invocation_id,
                        context_id=identity.context_id,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "tool_call_id": item_id,
                            "tool_name": "exec_command",
                            "tool_family": normalized.family,
                            "backend_class": normalized.backend_class,
                            "command_chars": len(command),
                            "source_started_at_ms": params.get("startedAtMs"),
                        },
                    ),
                )
            )
        elif method == "item/completed" and key in self._tool_start_ms:
            start_ts_ms = self._tool_start_ms.pop(key)
            ts_ms = self._timestamp()
            output = str(item.get("aggregatedOutput") or "")
            self._emit(
                (
                    self._event(
                        RuntimeEventKind.TOOL_END,
                        identity.workflow_id,
                        ts_ms=ts_ms,
                        invocation_id=identity.invocation_id,
                        context_id=identity.context_id,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "tool_call_id": item_id,
                            "duration_ms": max(0.0, ts_ms - start_ts_ms),
                            "runtime_duration_ms": item.get("durationMs"),
                            "returncode": item.get("exitCode"),
                            "output_chars": len(output),
                            "source_completed_at_ms": params.get("completedAtMs"),
                        },
                    ),
                )
            )

    def _handle_turn_completed(self, params: dict[str, Any]) -> None:
        thread_id = str(params.get("threadId", ""))
        identity = self.registry.get(thread_id)
        if identity is None:
            return
        turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
        outcome = str(turn.get("status", "completed"))
        self._return_thread(thread_id, outcome=outcome)

    def _return_thread(
        self,
        thread_id: str,
        *,
        outcome: str,
        message_present: bool | None = None,
    ) -> None:
        with self._lock:
            if thread_id in self._terminal_threads:
                return
            identity = self.registry.get(thread_id)
            if identity is None:
                return
            self._terminal_threads.add(thread_id)
            if identity.parent_thread_id is None:
                self._root_outcomes[identity.workflow_id] = outcome
            ts_ms = self._timestamp()
            events = [
                self._event(
                    RuntimeEventKind.RETURN,
                    identity.workflow_id,
                    ts_ms=ts_ms,
                    invocation_id=identity.invocation_id,
                    context_id=identity.context_id,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "outcome": outcome,
                        "message_present": message_present,
                        "codex_thread_id": thread_id,
                    },
                )
            ]
            workflow_threads = {
                item.thread_id
                for item in self.registry.identities()
                if item.workflow_id == identity.workflow_id
            }
            if (
                identity.workflow_id in self._root_outcomes
                and identity.workflow_id not in self._ended_workflows
                and workflow_threads <= self._terminal_threads
            ):
                self._ended_workflows.add(identity.workflow_id)
                events.append(
                    self._event(
                        RuntimeEventKind.WORKFLOW_END,
                        identity.workflow_id,
                        ts_ms=ts_ms,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "outcome": self._root_outcomes[identity.workflow_id]
                        },
                    )
                )
            self._emit(tuple(events))

    def _invocation_create(
        self, identity: CodexThreadIdentity, ts_ms: float
    ) -> RuntimeEvent:
        parent = (
            self.registry.get(identity.parent_thread_id)
            if identity.parent_thread_id is not None
            else None
        )
        return self._event(
            RuntimeEventKind.INVOCATION_CREATE,
            identity.workflow_id,
            ts_ms=ts_ms,
            invocation_id=identity.invocation_id,
            context_id=identity.context_id,
            context_epoch=0,
            parent_invocation_id=parent.invocation_id if parent else None,
            parent_context_id=parent.context_id if parent else None,
            agent_definition_id=identity.agent_role,
            agent_instance_id=identity.thread_id,
            relation_type=RelationType.SPAWN if parent else RelationType.ROOT,
            context_mode=ContextMode.FORK if parent else ContextMode.FRESH,
            execution_mode=(
                ExecutionMode.BACKGROUND if parent else ExecutionMode.FOREGROUND
            ),
            return_target_id=parent.invocation_id if parent else None,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={
                "persistent": True,
                "source": "codex_app_server",
                "agent_nickname": identity.agent_nickname,
            },
        )

    def _timestamp(self) -> float:
        with self._lock:
            self._last_ts_ms = max(self._last_ts_ms, float(self.clock_ms()))
            return self._last_ts_ms

    def _event(
        self,
        kind: RuntimeEventKind,
        workflow_id: str,
        *,
        ts_ms: float,
        confidence: EventConfidence = EventConfidence.DECLARED_RUNTIME,
        **kwargs: Any,
    ) -> RuntimeEvent:
        with self._lock:
            self._sequence += 1
            event_id = f"codex-runtime:{self._sequence:09d}"
        return RuntimeEvent(
            event_id=event_id,
            ts_ms=ts_ms,
            kind=kind,
            workflow_id=workflow_id,
            confidence=confidence,
            **kwargs,
        )

    def _emit(self, events: tuple[RuntimeEvent, ...]) -> None:
        self.sink.emit_batch(events)

    @staticmethod
    def _required_string(value: dict[str, Any], field: str) -> str:
        result = value.get(field)
        if not isinstance(result, str) or not result:
            raise ValueError(f"Codex payload is missing {field}")
        return result


class MultiplexRuntimeEventSink:
    def __init__(self, *sinks: RuntimeEventSink) -> None:
        self.sinks = sinks

    def emit_batch(self, events: tuple[RuntimeEvent, ...]) -> None:
        for sink in self.sinks:
            sink.emit_batch(events)
