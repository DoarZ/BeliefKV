from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable

from beliefkv.core.events import (
    ContextMode,
    EventConfidence,
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.core.ids import require_id


class CausalGraphError(RuntimeError):
    pass


class InvocationState(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING_LLM = "running_llm"
    WAIT_TOOL = "wait_tool"
    WAIT_CHILD = "wait_child"
    WAIT_JOIN = "wait_join"
    WAIT_MESSAGE = "wait_message"
    RETURNING = "returning"
    DONE = "done"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {InvocationState.DONE, InvocationState.CANCELLED}


class JoinMode(str, Enum):
    ALL = "all"
    ANY = "any"


@dataclass
class WorkflowRecord:
    workflow_id: str
    start_ts_ms: float
    end_ts_ms: float | None = None
    invocation_ids: set[str] = field(default_factory=set)


@dataclass
class InvocationRecord:
    workflow_id: str
    invocation_id: str
    context_id: str
    agent_definition_id: str
    agent_instance_id: str
    state: InvocationState
    created_ts_ms: float
    updated_ts_ms: float
    parent_invocation_id: str | None = None
    parent_context_id: str | None = None
    relation_type: RelationType = RelationType.ROOT
    context_mode: ContextMode = ContextMode.FRESH
    execution_mode: ExecutionMode = ExecutionMode.FOREGROUND
    return_target_id: str | None = None
    join_id: str | None = None
    persistent: bool = False
    pending_messages: int = 0
    child_invocation_ids: set[str] = field(default_factory=set)
    blocking_child_ids: set[str] = field(default_factory=set)
    confidence: EventConfidence = EventConfidence.DECLARED_RUNTIME
    active_tool_family: str | None = None
    active_tool_start_ms: float | None = None
    llm_round: int = 0


@dataclass
class ContextRecord:
    workflow_id: str
    context_id: str
    epoch: int
    created_ts_ms: float
    updated_ts_ms: float
    parent_context_id: str | None = None
    context_mode: ContextMode = ContextMode.FRESH
    invocation_ids: set[str] = field(default_factory=set)
    persistent: bool = False


@dataclass
class JoinRecord:
    workflow_id: str
    join_id: str
    member_invocation_ids: set[str]
    mode: JoinMode = JoinMode.ALL
    waiter_invocation_ids: set[str] = field(default_factory=set)
    completed_member_ids: set[str] = field(default_factory=set)
    satisfied: bool = False


@dataclass
class CommunicationEdge:
    source_invocation_id: str
    target_invocation_id: str
    count: int
    last_ts_ms: float


@dataclass(frozen=True)
class GraphDelta:
    event_id: str
    awakened_invocations: frozenset[str] = frozenset()
    parked_invocations: frozenset[str] = frozenset()
    completed_invocations: frozenset[str] = frozenset()
    changed_contexts: frozenset[str] = frozenset()
    graph_version: int = 0


class RuntimeCausalContextGraph:
    """Incremental source of truth for observed workflow causality."""

    def __init__(self, *, strict_timestamps: bool = True) -> None:
        self.workflows: dict[str, WorkflowRecord] = {}
        self.invocations: dict[str, InvocationRecord] = {}
        self.contexts: dict[str, ContextRecord] = {}
        self.joins: dict[str, JoinRecord] = {}
        self.communication_edges: dict[tuple[str, str], CommunicationEdge] = {}
        self._processed_event_ids: set[str] = set()
        self._last_ts_by_workflow: dict[str, float] = {}
        self._graph_version = 0
        self.strict_timestamps = strict_timestamps

    @property
    def graph_version(self) -> int:
        return self._graph_version

    def timestamp_watermark(self, workflow_id: str) -> float | None:
        """Return the latest committed event time for one root workflow."""

        return self._last_ts_by_workflow.get(workflow_id)

    def apply(self, event: RuntimeEvent) -> GraphDelta:
        if event.event_id in self._processed_event_ids:
            return GraphDelta(
                event_id=event.event_id,
                graph_version=self._graph_version,
            )

        last_ts = self._last_ts_by_workflow.get(event.workflow_id)
        if self.strict_timestamps and last_ts is not None and event.ts_ms < last_ts:
            raise CausalGraphError(
                f"out-of-order event {event.event_id}: {event.ts_ms} < {last_ts}"
            )

        handler = getattr(self, f"_on_{event.kind.value}")
        delta = handler(event)
        self._processed_event_ids.add(event.event_id)
        self._last_ts_by_workflow[event.workflow_id] = max(
            event.ts_ms, last_ts if last_ts is not None else event.ts_ms
        )
        self._graph_version += 1
        return replace(delta, graph_version=self._graph_version)

    def apply_batch(
        self, events: Iterable[RuntimeEvent], *, atomic: bool = True
    ) -> list[GraphDelta]:
        if not atomic:
            return [self.apply(event) for event in events]

        snapshot = deepcopy(self.__dict__)
        try:
            return [self.apply(event) for event in events]
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            raise

    def _on_workflow_start(self, event: RuntimeEvent) -> GraphDelta:
        if event.workflow_id in self.workflows:
            raise CausalGraphError(f"workflow already exists: {event.workflow_id}")
        self.workflows[event.workflow_id] = WorkflowRecord(
            workflow_id=event.workflow_id, start_ts_ms=event.ts_ms
        )
        return GraphDelta(event.event_id)

    def _on_workflow_end(self, event: RuntimeEvent) -> GraphDelta:
        workflow = self._workflow(event.workflow_id)
        completed: set[str] = set()
        changed: set[str] = set()
        for invocation_id in workflow.invocation_ids:
            invocation = self.invocations[invocation_id]
            if not invocation.state.terminal:
                invocation.state = InvocationState.CANCELLED
                invocation.updated_ts_ms = event.ts_ms
                completed.add(invocation_id)
                changed.add(invocation.context_id)
        workflow.end_ts_ms = event.ts_ms
        return GraphDelta(
            event.event_id,
            completed_invocations=frozenset(completed),
            changed_contexts=frozenset(changed),
        )

    def _on_invocation_create(self, event: RuntimeEvent) -> GraphDelta:
        workflow = self._workflow(event.workflow_id)
        invocation_id = require_id(event.invocation_id, "invocation_id")
        context_id = require_id(event.context_id, "context_id")
        if invocation_id in self.invocations:
            raise CausalGraphError(f"invocation already exists: {invocation_id}")

        parent_id = event.parent_invocation_id
        if parent_id is not None:
            parent = self._invocation(parent_id, event.workflow_id)
            parent_context_id = event.parent_context_id or parent.context_id
        else:
            parent_context_id = event.parent_context_id

        epoch = event.context_epoch if event.context_epoch is not None else 0
        context = self.contexts.get(context_id)
        if context is None:
            context = ContextRecord(
                workflow_id=event.workflow_id,
                context_id=context_id,
                epoch=epoch,
                created_ts_ms=event.ts_ms,
                updated_ts_ms=event.ts_ms,
                parent_context_id=parent_context_id,
                context_mode=event.context_mode or ContextMode.FRESH,
                persistent=bool(event.attributes.get("persistent", False)),
            )
            self.contexts[context_id] = context
        else:
            self._validate_context_epoch(context, event.context_epoch)
            if context.workflow_id != event.workflow_id:
                raise CausalGraphError(
                    f"context {context_id} cannot cross root workflows"
                )
            context.epoch = max(context.epoch, epoch)
            context.updated_ts_ms = event.ts_ms

        invocation = InvocationRecord(
            workflow_id=event.workflow_id,
            invocation_id=invocation_id,
            context_id=context_id,
            agent_definition_id=event.agent_definition_id or "unknown",
            agent_instance_id=event.agent_instance_id or invocation_id,
            state=InvocationState.READY,
            created_ts_ms=event.ts_ms,
            updated_ts_ms=event.ts_ms,
            parent_invocation_id=parent_id,
            parent_context_id=parent_context_id,
            relation_type=event.relation_type or RelationType.ROOT,
            context_mode=event.context_mode or ContextMode.FRESH,
            execution_mode=event.execution_mode or ExecutionMode.FOREGROUND,
            return_target_id=event.return_target_id or parent_id,
            join_id=event.join_id,
            persistent=bool(event.attributes.get("persistent", context.persistent)),
            confidence=event.confidence,
        )
        self.invocations[invocation_id] = invocation
        workflow.invocation_ids.add(invocation_id)
        context.invocation_ids.add(invocation_id)
        context.persistent = context.persistent or invocation.persistent
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset({invocation_id}),
            changed_contexts=frozenset({context_id}),
        )

    def _on_invocation_cancel(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        if invocation.state.terminal:
            return GraphDelta(event.event_id)
        invocation.state = InvocationState.CANCELLED
        invocation.updated_ts_ms = event.ts_ms
        awakened = self._complete_dependencies(invocation, event.ts_ms)
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset(awakened),
            completed_invocations=frozenset({invocation.invocation_id}),
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _on_context_advance(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        self._ensure_not_terminal(invocation)
        if event.context_id not in {None, invocation.context_id}:
            raise CausalGraphError("context advance does not match invocation context")
        self._touch_context(invocation.context_id, event)
        invocation.updated_ts_ms = event.ts_ms
        return GraphDelta(
            event.event_id,
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _on_context_compact(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        self._ensure_not_terminal(invocation)
        if event.context_id not in {None, invocation.context_id}:
            raise CausalGraphError("context compact does not match invocation context")
        context = self.contexts[invocation.context_id]
        previous_epoch = event.attributes.get("previous_context_epoch")
        if previous_epoch is not None and int(previous_epoch) != context.epoch:
            raise CausalGraphError(
                f"context compact previous epoch mismatch for {context.context_id}: "
                f"{previous_epoch} != {context.epoch}"
            )
        if event.context_epoch is None or event.context_epoch <= context.epoch:
            raise CausalGraphError("context compact must advance the context epoch")
        self._touch_context(invocation.context_id, event)
        invocation.updated_ts_ms = event.ts_ms
        return GraphDelta(
            event.event_id,
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _on_structured_action(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        self._validate_context_epoch(
            self.contexts[invocation.context_id], event.context_epoch
        )
        return GraphDelta(event.event_id)

    def _on_call_censored(self, event: RuntimeEvent) -> GraphDelta:
        """Record an observation label without changing executable RCCG state."""

        if event.invocation_id is not None:
            self._event_invocation(event)
        return GraphDelta(event.event_id)

    def _on_call(self, event: RuntimeEvent) -> GraphDelta:
        return self._link_child(event, ExecutionMode.FOREGROUND)

    def _on_spawn(self, event: RuntimeEvent) -> GraphDelta:
        mode = event.execution_mode or ExecutionMode.BACKGROUND
        return self._link_child(event, mode)

    def _link_child(
        self, event: RuntimeEvent, default_mode: ExecutionMode
    ) -> GraphDelta:
        parent = self._event_invocation(event)
        child_id = require_id(event.target_invocation_id, "target_invocation_id")
        child = self._invocation(child_id, event.workflow_id)
        if parent.state.terminal or child.state.terminal:
            raise CausalGraphError("cannot link a terminal invocation")
        if parent.invocation_id == child.invocation_id or self._is_ancestor(
            child.invocation_id, parent.invocation_id
        ):
            raise CausalGraphError("CALL/SPAWN would create an invocation cycle")
        if child.parent_invocation_id not in {None, parent.invocation_id}:
            raise CausalGraphError(f"invocation {child_id} already has another parent")

        mode = event.execution_mode or default_mode
        relation = (
            RelationType.CALL
            if event.kind == RuntimeEventKind.CALL
            else RelationType.SPAWN
        )
        child.parent_invocation_id = parent.invocation_id
        child.parent_context_id = parent.context_id
        child.return_target_id = event.return_target_id or parent.invocation_id
        child.execution_mode = mode
        child.relation_type = relation
        child.updated_ts_ms = event.ts_ms
        child.state = InvocationState.READY
        parent.child_invocation_ids.add(child_id)
        parked: set[str] = set()
        if mode == ExecutionMode.FOREGROUND:
            parent.blocking_child_ids.add(child_id)
            parent.state = InvocationState.WAIT_CHILD
            parent.updated_ts_ms = event.ts_ms
            parked.add(parent.invocation_id)
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset({child_id}),
            parked_invocations=frozenset(parked),
            changed_contexts=frozenset({parent.context_id, child.context_id}),
        )

    def _on_return(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        if invocation.state.terminal:
            return GraphDelta(event.event_id)
        invocation.state = InvocationState.DONE
        invocation.updated_ts_ms = event.ts_ms
        awakened = self._complete_dependencies(invocation, event.ts_ms)
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset(awakened),
            completed_invocations=frozenset({invocation.invocation_id}),
            changed_contexts=frozenset(
                {invocation.context_id}
                | {self.invocations[item].context_id for item in awakened}
            ),
        )

    def _on_message(self, event: RuntimeEvent) -> GraphDelta:
        return self._deliver_message(event, handoff=False)

    def _on_handoff(self, event: RuntimeEvent) -> GraphDelta:
        return self._deliver_message(event, handoff=True)

    def _on_reactivate(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        self._ensure_not_terminal(invocation)
        previous = invocation.state
        if previous == InvocationState.RUNNING_LLM:
            raise CausalGraphError("cannot reactivate an invocation already running")
        invocation.state = InvocationState.READY
        invocation.updated_ts_ms = event.ts_ms
        awakened = (
            frozenset({invocation.invocation_id})
            if previous != InvocationState.READY
            else frozenset()
        )
        return GraphDelta(
            event.event_id,
            awakened_invocations=awakened,
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _deliver_message(self, event: RuntimeEvent, *, handoff: bool) -> GraphDelta:
        source = self._event_invocation(event)
        target_id = require_id(event.target_invocation_id, "target_invocation_id")
        target = self._invocation(target_id, event.workflow_id)
        if target.state.terminal:
            raise CausalGraphError("cannot send a message to a terminal invocation")

        key = (source.invocation_id, target_id)
        edge = self.communication_edges.get(key)
        if edge is None:
            edge = CommunicationEdge(source.invocation_id, target_id, 0, event.ts_ms)
            self.communication_edges[key] = edge
        edge.count += 1
        edge.last_ts_ms = event.ts_ms
        target.pending_messages += 1
        target.state = InvocationState.READY
        target.updated_ts_ms = event.ts_ms
        parked: set[str] = set()
        if handoff and not source.state.terminal:
            source.state = InvocationState.WAIT_MESSAGE
            source.updated_ts_ms = event.ts_ms
            parked.add(source.invocation_id)
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset({target_id}),
            parked_invocations=frozenset(parked),
            changed_contexts=frozenset({source.context_id, target.context_id}),
        )

    def _on_join_create(self, event: RuntimeEvent) -> GraphDelta:
        self._workflow(event.workflow_id)
        join_id = require_id(event.join_id, "join_id")
        if join_id in self.joins:
            raise CausalGraphError(f"join already exists: {join_id}")
        members = set(event.member_invocation_ids)
        if not members:
            raise CausalGraphError("join must contain at least one member")
        for member in members:
            self._invocation(member, event.workflow_id)
        mode = JoinMode(str(event.attributes.get("mode", JoinMode.ALL.value)))
        completed = {
            member for member in members if self.invocations[member].state.terminal
        }
        join = JoinRecord(
            workflow_id=event.workflow_id,
            join_id=join_id,
            member_invocation_ids=members,
            mode=mode,
            completed_member_ids=completed,
        )
        join.satisfied = self._join_is_satisfied(join)
        self.joins[join_id] = join
        return GraphDelta(event.event_id)

    def _on_join_wait(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        join_id = require_id(event.join_id, "join_id")
        join = self._join(join_id, event.workflow_id)
        join.waiter_invocation_ids.add(invocation.invocation_id)
        invocation.join_id = join_id
        invocation.updated_ts_ms = event.ts_ms
        if join.satisfied:
            invocation.state = InvocationState.READY
            return GraphDelta(
                event.event_id,
                awakened_invocations=frozenset({invocation.invocation_id}),
                changed_contexts=frozenset({invocation.context_id}),
            )
        invocation.state = InvocationState.WAIT_JOIN
        return GraphDelta(
            event.event_id,
            parked_invocations=frozenset({invocation.invocation_id}),
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _on_join_satisfied(self, event: RuntimeEvent) -> GraphDelta:
        join_id = require_id(event.join_id, "join_id")
        join = self._join(join_id, event.workflow_id)
        join.satisfied = True
        awakened = self._wake_join_waiters(join, event.ts_ms)
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset(awakened),
            changed_contexts=frozenset(
                self.invocations[item].context_id for item in awakened
            ),
        )

    def _on_join_timeout(self, event: RuntimeEvent) -> GraphDelta:
        join_id = require_id(event.join_id, "join_id")
        join = self._join(join_id, event.workflow_id)
        awakened = self._wake_join_waiters(join, event.ts_ms)
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset(awakened),
            changed_contexts=frozenset(
                self.invocations[item].context_id for item in awakened
            ),
        )

    def _on_tool_start(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        self._ensure_not_terminal(invocation)
        invocation.state = InvocationState.WAIT_TOOL
        invocation.updated_ts_ms = event.ts_ms
        invocation.active_tool_family = str(event.attributes.get("tool_family", "unknown"))
        invocation.active_tool_start_ms = event.ts_ms
        return GraphDelta(
            event.event_id,
            parked_invocations=frozenset({invocation.invocation_id}),
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _on_tool_end(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        self._ensure_not_terminal(invocation)
        invocation.state = InvocationState.READY
        invocation.updated_ts_ms = event.ts_ms
        invocation.active_tool_family = None
        invocation.active_tool_start_ms = None
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset({invocation.invocation_id}),
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _on_llm_submit(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        self._ensure_not_terminal(invocation)
        self._touch_context(invocation.context_id, event)
        invocation.state = InvocationState.RUNNING_LLM
        invocation.pending_messages = max(
            0, invocation.pending_messages - int(event.attributes.get("messages_consumed", 0))
        )
        invocation.updated_ts_ms = event.ts_ms
        invocation.llm_round += 1
        return GraphDelta(
            event.event_id,
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _on_llm_result(self, event: RuntimeEvent) -> GraphDelta:
        invocation = self._event_invocation(event)
        self._ensure_not_terminal(invocation)
        self._touch_context(invocation.context_id, event)
        if bool(event.attributes.get("terminal", False)):
            invocation.state = InvocationState.DONE
            invocation.updated_ts_ms = event.ts_ms
            awakened = self._complete_dependencies(invocation, event.ts_ms)
            return GraphDelta(
                event.event_id,
                awakened_invocations=frozenset(awakened),
                completed_invocations=frozenset({invocation.invocation_id}),
                changed_contexts=frozenset({invocation.context_id}),
            )
        invocation.state = InvocationState.READY
        invocation.updated_ts_ms = event.ts_ms
        return GraphDelta(
            event.event_id,
            awakened_invocations=frozenset({invocation.invocation_id}),
            changed_contexts=frozenset({invocation.context_id}),
        )

    def _complete_dependencies(
        self, invocation: InvocationRecord, ts_ms: float
    ) -> set[str]:
        awakened: set[str] = set()
        target_id = invocation.return_target_id or invocation.parent_invocation_id
        if target_id is not None and target_id in self.invocations:
            parent = self.invocations[target_id]
            parent.blocking_child_ids.discard(invocation.invocation_id)
            if (
                parent.state == InvocationState.WAIT_CHILD
                and not parent.blocking_child_ids
                and not parent.state.terminal
            ):
                parent.state = InvocationState.READY
                parent.updated_ts_ms = ts_ms
                awakened.add(parent.invocation_id)

        for join in self.joins.values():
            if invocation.invocation_id not in join.member_invocation_ids:
                continue
            join.completed_member_ids.add(invocation.invocation_id)
            if self._join_is_satisfied(join):
                join.satisfied = True
                awakened.update(self._wake_join_waiters(join, ts_ms))
        return awakened

    def _join_is_satisfied(self, join: JoinRecord) -> bool:
        if join.mode == JoinMode.ANY:
            return bool(join.completed_member_ids)
        return join.completed_member_ids >= join.member_invocation_ids

    def _wake_join_waiters(self, join: JoinRecord, ts_ms: float) -> set[str]:
        awakened: set[str] = set()
        for waiter_id in join.waiter_invocation_ids:
            waiter = self.invocations[waiter_id]
            if waiter.state == InvocationState.WAIT_JOIN:
                waiter.state = InvocationState.READY
                waiter.updated_ts_ms = ts_ms
                awakened.add(waiter_id)
        return awakened

    def _touch_context(self, context_id: str, event: RuntimeEvent) -> None:
        context = self.contexts[context_id]
        self._validate_context_epoch(context, event.context_epoch)
        if event.context_epoch is not None:
            context.epoch = event.context_epoch
        context.updated_ts_ms = event.ts_ms

    @staticmethod
    def _validate_context_epoch(
        context: ContextRecord, event_epoch: int | None
    ) -> None:
        if event_epoch is not None and event_epoch < context.epoch:
            raise CausalGraphError(
                f"stale context epoch for {context.context_id}: "
                f"{event_epoch} < {context.epoch}"
            )

    def _workflow(self, workflow_id: str) -> WorkflowRecord:
        try:
            return self.workflows[workflow_id]
        except KeyError as exc:
            raise CausalGraphError(f"unknown workflow: {workflow_id}") from exc

    def _event_invocation(self, event: RuntimeEvent) -> InvocationRecord:
        invocation_id = require_id(event.invocation_id, "invocation_id")
        return self._invocation(invocation_id, event.workflow_id)

    def _invocation(self, invocation_id: str, workflow_id: str) -> InvocationRecord:
        try:
            invocation = self.invocations[invocation_id]
        except KeyError as exc:
            raise CausalGraphError(f"unknown invocation: {invocation_id}") from exc
        if invocation.workflow_id != workflow_id:
            raise CausalGraphError(
                f"invocation {invocation_id} belongs to {invocation.workflow_id}, "
                f"not {workflow_id}"
            )
        return invocation

    def _join(self, join_id: str, workflow_id: str) -> JoinRecord:
        try:
            join = self.joins[join_id]
        except KeyError as exc:
            raise CausalGraphError(f"unknown join: {join_id}") from exc
        if join.workflow_id != workflow_id:
            raise CausalGraphError(f"join {join_id} belongs to another workflow")
        return join

    @staticmethod
    def _ensure_not_terminal(invocation: InvocationRecord) -> None:
        if invocation.state.terminal:
            raise CausalGraphError(
                f"invocation {invocation.invocation_id} is terminal"
            )

    def _is_ancestor(self, ancestor_id: str, invocation_id: str) -> bool:
        current = self.invocations.get(invocation_id)
        seen: set[str] = set()
        while current is not None and current.parent_invocation_id is not None:
            if current.parent_invocation_id == ancestor_id:
                return True
            if current.parent_invocation_id in seen:
                raise CausalGraphError("existing invocation graph contains a cycle")
            seen.add(current.parent_invocation_id)
            current = self.invocations.get(current.parent_invocation_id)
        return False

    def ancestors(self, invocation_id: str) -> list[InvocationRecord]:
        result: list[InvocationRecord] = []
        invocation = self.invocations[invocation_id]
        seen: set[str] = set()
        while invocation.parent_invocation_id is not None:
            parent_id = invocation.parent_invocation_id
            if parent_id in seen:
                raise CausalGraphError("invocation graph contains a cycle")
            seen.add(parent_id)
            invocation = self.invocations[parent_id]
            result.append(invocation)
        return result

    def active_descendants(self, invocation_id: str) -> list[InvocationRecord]:
        result: list[InvocationRecord] = []
        stack = list(self.invocations[invocation_id].child_invocation_ids)
        seen: set[str] = set()
        while stack:
            child_id = stack.pop()
            if child_id in seen:
                continue
            seen.add(child_id)
            child = self.invocations[child_id]
            if not child.state.terminal:
                result.append(child)
            stack.extend(child.child_invocation_ids)
        return result

    def ready_invocations(self, workflow_id: str | None = None) -> list[InvocationRecord]:
        return [
            invocation
            for invocation in self.invocations.values()
            if invocation.state == InvocationState.READY
            and (workflow_id is None or invocation.workflow_id == workflow_id)
        ]

    def context_invocations(self, context_id: str) -> list[InvocationRecord]:
        context = self.contexts[context_id]
        return [self.invocations[item] for item in context.invocation_ids]

    def context_is_persistent(self, context_id: str) -> bool:
        context = self.contexts[context_id]
        return context.persistent

    def snapshot(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible state snapshot."""

        return {
            "graph_version": self._graph_version,
            "workflows": {
                key: {
                    "start_ts_ms": value.start_ts_ms,
                    "end_ts_ms": value.end_ts_ms,
                    "invocation_ids": sorted(value.invocation_ids),
                }
                for key, value in sorted(self.workflows.items())
            },
            "invocations": {
                key: self._invocation_snapshot(value)
                for key, value in sorted(self.invocations.items())
            },
            "contexts": {
                key: {
                    "workflow_id": value.workflow_id,
                    "epoch": value.epoch,
                    "created_ts_ms": value.created_ts_ms,
                    "updated_ts_ms": value.updated_ts_ms,
                    "parent_context_id": value.parent_context_id,
                    "context_mode": value.context_mode.value,
                    "invocation_ids": sorted(value.invocation_ids),
                    "persistent": value.persistent,
                }
                for key, value in sorted(self.contexts.items())
            },
            "joins": {
                key: self._join_snapshot(value)
                for key, value in sorted(self.joins.items())
            },
            "communication_edges": [
                {
                    "source_invocation_id": value.source_invocation_id,
                    "target_invocation_id": value.target_invocation_id,
                    "count": value.count,
                    "last_ts_ms": value.last_ts_ms,
                }
                for _, value in sorted(self.communication_edges.items())
            ],
        }

    def invocation_snapshot(self, invocation_id: str) -> dict[str, object]:
        """Serialize one invocation without materializing the complete RCCG."""

        value = self.invocations.get(invocation_id)
        return self._invocation_snapshot(value) if value is not None else {}

    def join_snapshot(self, join_id: str) -> dict[str, object]:
        """Serialize one join without materializing the complete RCCG."""

        value = self.joins.get(join_id)
        return self._join_snapshot(value) if value is not None else {}

    @staticmethod
    def _invocation_snapshot(value: InvocationRecord) -> dict[str, object]:
        return {
            "workflow_id": value.workflow_id,
            "context_id": value.context_id,
            "agent_definition_id": value.agent_definition_id,
            "agent_instance_id": value.agent_instance_id,
            "state": value.state.value,
            "created_ts_ms": value.created_ts_ms,
            "updated_ts_ms": value.updated_ts_ms,
            "parent_invocation_id": value.parent_invocation_id,
            "parent_context_id": value.parent_context_id,
            "relation_type": value.relation_type.value,
            "context_mode": value.context_mode.value,
            "execution_mode": value.execution_mode.value,
            "return_target_id": value.return_target_id,
            "children": sorted(value.child_invocation_ids),
            "blocking_children": sorted(value.blocking_child_ids),
            "pending_messages": value.pending_messages,
            "join_id": value.join_id,
            "persistent": value.persistent,
            "confidence": value.confidence.value,
            "active_tool_family": value.active_tool_family,
            "active_tool_start_ms": value.active_tool_start_ms,
            "llm_round": value.llm_round,
        }

    @staticmethod
    def _join_snapshot(value: JoinRecord) -> dict[str, object]:
        return {
            "members": sorted(value.member_invocation_ids),
            "completed": sorted(value.completed_member_ids),
            "waiters": sorted(value.waiter_invocation_ids),
            "mode": value.mode.value,
            "satisfied": value.satisfied,
        }
