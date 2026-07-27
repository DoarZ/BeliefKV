from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, TypedDict

from beliefkv.core.events import (
    ContextMode,
    EventConfidence,
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.runtime.agent_runtime_adapter import RuntimeEventSink
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


class TraceSensitivity(str, Enum):
    SCHEDULE_INVARIANT = "schedule_invariant"
    TIMING_SENSITIVE = "timing_sensitive"
    SEMANTIC_RACE_SENSITIVE = "semantic_race_sensitive"


class LLMEventSource(str, Enum):
    WORKFLOW = "workflow"
    MODEL_RUNTIME = "model_runtime"


class PeerRole(str, Enum):
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"


@dataclass(frozen=True)
class SubagentTask:
    agent_definition_id: str
    instruction: str

    def __post_init__(self) -> None:
        if not self.agent_definition_id or not self.instruction:
            raise ValueError("subagent task fields must be non-empty")


@dataclass(frozen=True)
class PeerTurnRequest:
    task: str
    role: str
    turn: int
    history: tuple[str, ...]
    metadata: BeliefKVRequestMetadata
    is_subagent: bool = False
    must_complete: bool = False


@dataclass(frozen=True)
class PeerTurnResult:
    summary: str
    next_role: PeerRole | None
    complete: bool
    output_tokens: int = 0
    subagent_tasks: tuple[SubagentTask, ...] = ()
    final_context_epoch: int | None = None

    def __post_init__(self) -> None:
        if not self.summary:
            raise ValueError("peer turn summary must be non-empty")
        if self.complete == (self.next_role is not None):
            raise ValueError("complete turns have no next role; incomplete turns require one")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if self.final_context_epoch is not None and self.final_context_epoch < 0:
            raise ValueError("final_context_epoch must be non-negative")
        if self.complete and self.subagent_tasks:
            raise ValueError("a complete turn cannot create new subagent work")


class PeerAgentBackend(Protocol):
    def invoke(self, request: PeerTurnRequest) -> PeerTurnResult:
        ...


class _PeerState(TypedDict):
    task: str
    current_role: str
    next_role: str
    turn: int
    history: tuple[str, ...]
    done: bool
    termination_reason: str


@dataclass(frozen=True)
class PeerWorkflowResult:
    workflow_id: str
    completed: bool
    turn_count: int
    termination_reason: str
    transition_hash: str
    trace_sensitivity: TraceSensitivity
    events: tuple[RuntimeEvent, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "workflow_id": self.workflow_id,
            "completed": self.completed,
            "turn_count": self.turn_count,
            "termination_reason": self.termination_reason,
            "transition_hash": self.transition_hash,
            "trace_sensitivity": self.trace_sensitivity.value,
            "event_count": len(self.events),
        }


class _RuntimeEmitter:
    def __init__(
        self,
        sink: RuntimeEventSink,
        workflow_id: str,
        *,
        sensitivity: TraceSensitivity,
        clock_ms: Callable[[], float],
    ) -> None:
        self.sink = sink
        self.workflow_id = workflow_id
        self.sensitivity = sensitivity
        self.clock_ms = clock_ms
        self.sequence = 0
        self.last_ts_ms = 0.0
        self.events: list[RuntimeEvent] = []
        self._lock = threading.Lock()

    def emit(self, kind: RuntimeEventKind, **kwargs: Any) -> RuntimeEvent:
        with self._lock:
            self.sequence += 1
            self.last_ts_ms = max(self.last_ts_ms, float(self.clock_ms()))
            attributes = dict(kwargs.pop("attributes", {}))
            attributes.setdefault("source", "langgraph_peer_workflow")
            attributes.setdefault("trace_sensitivity", self.sensitivity.value)
            event = RuntimeEvent(
                event_id=f"{self.workflow_id}:peer:{self.sequence:09d}",
                ts_ms=self.last_ts_ms,
                kind=kind,
                workflow_id=self.workflow_id,
                attributes=attributes,
                **kwargs,
            )
            self.events.append(event)
            self.sink.emit_batch((event,))
            return event


class LangGraphPeerWorkflow:
    """Dynamic cyclic peer workflow with optional nested FRESH subagents."""

    def __init__(
        self,
        backend: PeerAgentBackend,
        event_sink: RuntimeEventSink,
        *,
        workflow_id: str,
        max_turns: int = 18,
        trace_sensitivity: TraceSensitivity = TraceSensitivity.SEMANTIC_RACE_SENSITIVE,
        llm_event_source: LLMEventSource = LLMEventSource.WORKFLOW,
        parallel_subagents: bool = False,
        clock_ms: Callable[[], float] | None = None,
    ) -> None:
        if not workflow_id:
            raise ValueError("workflow_id must be non-empty")
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.backend = backend
        self.workflow_id = workflow_id
        self.max_turns = max_turns
        self.trace_sensitivity = trace_sensitivity
        self.llm_event_source = llm_event_source
        self.parallel_subagents = parallel_subagents
        self.emitter = _RuntimeEmitter(
            event_sink,
            workflow_id,
            sensitivity=trace_sensitivity,
            clock_ms=clock_ms or (lambda: time.monotonic() * 1000.0),
        )
        self._metadata_by_role: dict[str, BeliefKVRequestMetadata] = {}
        self._metadata_by_invocation: dict[str, BeliefKVRequestMetadata] = {}
        self._epoch_by_invocation: dict[str, int] = {}
        self._terminal_invocations: set[str] = set()
        self._task_digest = ""

    def run(self, task: str) -> PeerWorkflowResult:
        if not task:
            raise ValueError("task must be non-empty")
        self._task_digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        self.emitter.emit(
            RuntimeEventKind.WORKFLOW_START,
            attributes={"task_sha256": self._task_digest},
        )
        self._ensure_peer(PeerRole.CODER)
        graph = self._build_graph()
        try:
            state = graph.invoke(
                {
                    "task": task,
                    "current_role": PeerRole.CODER.value,
                    "next_role": PeerRole.CODER.value,
                    "turn": 0,
                    "history": (),
                    "done": False,
                    "termination_reason": "",
                },
                config={"recursion_limit": max(32, self.max_turns * 4)},
            )
        except BaseException as error:
            self._finish_live_invocations("runtime_error")
            self.emitter.emit(
                RuntimeEventKind.WORKFLOW_END,
                attributes={
                    "outcome": "runtime_error",
                    "exception_type": type(error).__name__,
                },
            )
            raise
        self._finish_live_invocations(
            str(state["termination_reason"] or "completed")
        )
        self.emitter.emit(
            RuntimeEventKind.WORKFLOW_END,
            attributes={
                "outcome": state["termination_reason"],
                "turn_count": int(state["turn"]),
            },
        )
        events = tuple(self.emitter.events)
        return PeerWorkflowResult(
            workflow_id=self.workflow_id,
            completed=state["termination_reason"] == "semantic_complete",
            turn_count=int(state["turn"]),
            termination_reason=str(state["termination_reason"]),
            transition_hash=_transition_hash(events),
            trace_sensitivity=self.trace_sensitivity,
            events=events,
        )

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            raise RuntimeError(
                "LangGraph peer workload requires the beliefkv-agents environment"
            ) from error

        builder = StateGraph(_PeerState)
        for role in PeerRole:
            builder.add_node(role.value, self._node(role))
        builder.add_edge(START, PeerRole.CODER.value)
        routes = {role.value: role.value for role in PeerRole}
        routes["end"] = END
        for role in PeerRole:
            builder.add_conditional_edges(role.value, self._route, routes)
        return builder.compile()

    def _node(self, role: PeerRole) -> Callable[[_PeerState], dict[str, object]]:
        def run(state: _PeerState) -> dict[str, object]:
            turn = int(state["turn"])
            if turn >= self.max_turns:
                return {
                    "current_role": role.value,
                    "next_role": "end",
                    "done": True,
                    "termination_reason": "max_turns",
                }
            metadata = self._next_request_metadata(role.value)
            if self.llm_event_source == LLMEventSource.WORKFLOW:
                self.emitter.emit(
                    RuntimeEventKind.LLM_SUBMIT,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    context_epoch=metadata.context_epoch,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "request_id": self._request_id(metadata, turn),
                        "task_sha256": self._task_digest,
                        "role": role.value,
                        "turn": turn,
                    },
                )
            result = self.backend.invoke(
                PeerTurnRequest(
                    task=state["task"],
                    role=role.value,
                    turn=turn,
                    history=tuple(state["history"]),
                    metadata=metadata,
                    must_complete=turn == self.max_turns - 1,
                )
            )
            if result.final_context_epoch is not None:
                self._epoch_by_invocation[metadata.invocation_id] = max(
                    self._epoch_by_invocation[metadata.invocation_id],
                    result.final_context_epoch,
                )
                metadata = BeliefKVRequestMetadata.from_wire(
                    {
                        **metadata.to_wire(),
                        "context_epoch": result.final_context_epoch,
                    }
                )
            action_kinds = (
                ["final_answer"]
                if result.complete
                else [
                    *(["spawn"] if result.subagent_tasks else []),
                    "handoff",
                ]
            )
            if self.llm_event_source == LLMEventSource.WORKFLOW:
                self.emitter.emit(
                    RuntimeEventKind.LLM_RESULT,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    context_epoch=metadata.context_epoch,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "request_id": self._request_id(metadata, turn),
                        "output_tokens": result.output_tokens,
                        "parser_status": "valid",
                        "structured_action_kinds": action_kinds,
                        "structured_action_names": (
                            [
                                item.agent_definition_id
                                for item in result.subagent_tasks
                            ]
                            + ([result.next_role.value] if result.next_role else [])
                        ),
                        "action_boundary_token_index": None,
                        "action_boundary_source": "runtime_structured_output",
                        "parser_reason": (
                            "code orchestrator received a complete structured result; "
                            "incremental token boundary is unavailable"
                        ),
                    },
                )
            else:
                self.emitter.emit(
                    RuntimeEventKind.STRUCTURED_ACTION,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    context_epoch=metadata.context_epoch,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "output_tokens": result.output_tokens,
                        "parser_status": "valid",
                        "structured_action_kinds": action_kinds,
                        "structured_action_names": (
                            [
                                item.agent_definition_id
                                for item in result.subagent_tasks
                            ]
                            + ([result.next_role.value] if result.next_role else [])
                        ),
                        "action_boundary_token_index": None,
                        "action_boundary_source": "runtime_structured_output",
                        "parser_reason": (
                            "code orchestrator received a complete structured result; "
                            "incremental token boundary is unavailable"
                        ),
                    },
                )
            if result.subagent_tasks:
                self._run_subagents(role, metadata, turn, result.subagent_tasks, state)
            history = (*state["history"], f"{role.value}:{result.summary}")
            if result.complete:
                self._return_invocation(metadata, outcome="semantic_complete")
                return {
                    "current_role": role.value,
                    "next_role": "end",
                    "turn": turn + 1,
                    "history": history,
                    "done": True,
                    "termination_reason": "semantic_complete",
                }
            assert result.next_role is not None
            target_existed = result.next_role.value in self._metadata_by_role
            target = self._ensure_peer(result.next_role)
            self.emitter.emit(
                RuntimeEventKind.HANDOFF,
                invocation_id=metadata.invocation_id,
                target_invocation_id=target.invocation_id,
                confidence=EventConfidence.OBSERVED_EXACT,
                attributes={
                    "source_role": role.value,
                    "target_role": result.next_role.value,
                    "turn": turn,
                },
            )
            if target_existed:
                self.emitter.emit(
                    RuntimeEventKind.REACTIVATE,
                    invocation_id=target.invocation_id,
                    context_id=target.context_id,
                    context_epoch=self._epoch_by_invocation[target.invocation_id],
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={"reason": "cyclic_peer_handoff"},
                )
            return {
                "current_role": role.value,
                "next_role": result.next_role.value,
                "turn": turn + 1,
                "history": history,
                "done": False,
                "termination_reason": "",
            }

        return run

    @staticmethod
    def _route(state: _PeerState) -> str:
        return "end" if state["done"] else str(state["next_role"])

    def _ensure_peer(self, role: PeerRole) -> BeliefKVRequestMetadata:
        existing = self._metadata_by_role.get(role.value)
        if existing is not None:
            return existing
        suffix = _short_digest(f"{self.workflow_id}:peer:{role.value}")
        metadata = BeliefKVRequestMetadata(
            root_workflow_id=self.workflow_id,
            invocation_id=f"peer-invocation:{suffix}",
            context_id=f"peer-context:{suffix}",
            context_epoch=0,
            agent_definition_id=role.value,
            agent_instance_id=f"{role.value}:{suffix}",
            relation_type=RelationType.ROOT.value,
            context_mode=ContextMode.FRESH.value,
            execution_mode=ExecutionMode.FOREGROUND.value,
        )
        self._metadata_by_role[role.value] = metadata
        self._metadata_by_invocation[metadata.invocation_id] = metadata
        self._epoch_by_invocation[metadata.invocation_id] = -1
        self.emitter.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            context_epoch=0,
            agent_definition_id=metadata.agent_definition_id,
            agent_instance_id=metadata.agent_instance_id,
            relation_type=RelationType.ROOT,
            context_mode=ContextMode.FRESH,
            execution_mode=ExecutionMode.FOREGROUND,
            attributes={"persistent": True, "role": role.value},
        )
        return metadata

    def _next_request_metadata(self, role: str) -> BeliefKVRequestMetadata:
        base = self._metadata_by_role[role]
        epoch = self._epoch_by_invocation[base.invocation_id] + 1
        self._epoch_by_invocation[base.invocation_id] = epoch
        return BeliefKVRequestMetadata.from_wire(
            {**base.to_wire(), "context_epoch": epoch}
        )

    def _run_subagents(
        self,
        parent_role: PeerRole,
        parent: BeliefKVRequestMetadata,
        turn: int,
        tasks: tuple[SubagentTask, ...],
        state: _PeerState,
    ) -> None:
        join_id = f"peer-join:{_short_digest(f'{self.workflow_id}:{parent.invocation_id}:{turn}') }"
        children: list[tuple[SubagentTask, BeliefKVRequestMetadata]] = []
        for index, task in enumerate(tasks):
            suffix = _short_digest(
                f"{self.workflow_id}:{parent.invocation_id}:{turn}:{index}:{task.agent_definition_id}"
            )
            child = BeliefKVRequestMetadata(
                root_workflow_id=self.workflow_id,
                invocation_id=f"peer-child-invocation:{suffix}",
                context_id=f"peer-child-context:{suffix}",
                context_epoch=0,
                agent_definition_id=task.agent_definition_id,
                agent_instance_id=f"{task.agent_definition_id}:{suffix}",
                parent_invocation_id=parent.invocation_id,
                parent_context_id=parent.context_id,
                relation_type=RelationType.SPAWN.value,
                context_mode=ContextMode.FRESH.value,
                execution_mode=ExecutionMode.BACKGROUND.value,
                return_target_id=parent.invocation_id,
                join_id=join_id,
            )
            children.append((task, child))
            self._metadata_by_invocation[child.invocation_id] = child
            self.emitter.emit(
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id=child.invocation_id,
                context_id=child.context_id,
                context_epoch=0,
                parent_invocation_id=parent.invocation_id,
                parent_context_id=parent.context_id,
                agent_definition_id=child.agent_definition_id,
                agent_instance_id=child.agent_instance_id,
                relation_type=RelationType.SPAWN,
                context_mode=ContextMode.FRESH,
                execution_mode=ExecutionMode.BACKGROUND,
                return_target_id=parent.invocation_id,
                join_id=join_id,
                attributes={
                    "persistent": False,
                    "instruction_sha256": hashlib.sha256(
                        task.instruction.encode("utf-8")
                    ).hexdigest(),
                },
            )
            self.emitter.emit(
                RuntimeEventKind.SPAWN,
                invocation_id=parent.invocation_id,
                target_invocation_id=child.invocation_id,
                execution_mode=ExecutionMode.BACKGROUND,
                return_target_id=parent.invocation_id,
                confidence=EventConfidence.OBSERVED_EXACT,
                attributes={"subagent_type": task.agent_definition_id},
            )
        self.emitter.emit(
            RuntimeEventKind.JOIN_CREATE,
            join_id=join_id,
            member_invocation_ids=tuple(child.invocation_id for _, child in children),
            attributes={"mode": "all", "parent_role": parent_role.value},
        )
        self.emitter.emit(
            RuntimeEventKind.JOIN_WAIT,
            invocation_id=parent.invocation_id,
            join_id=join_id,
            confidence=EventConfidence.OBSERVED_EXACT,
        )
        if self.llm_event_source == LLMEventSource.WORKFLOW:
            for _, child in children:
                request_id = f"{child.invocation_id}:turn:0"
                self.emitter.emit(
                    RuntimeEventKind.LLM_SUBMIT,
                    invocation_id=child.invocation_id,
                    context_id=child.context_id,
                    context_epoch=0,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={"request_id": request_id, "is_subagent": True},
                )
        if self.parallel_subagents and len(children) > 1:
            with ThreadPoolExecutor(max_workers=len(children)) as executor:
                futures = {
                    executor.submit(
                        self._invoke_subagent,
                        task,
                        child,
                        tuple(state["history"]),
                    ): (task, child)
                    for task, child in children
                }
                completed_children = [
                    (*futures[future], future.result())
                    for future in as_completed(futures)
                ]
        else:
            completed_children = [
                (
                    task,
                    child,
                    self._invoke_subagent(
                        task,
                        child,
                        tuple(state["history"]),
                    ),
                )
                for task, child in children
            ]
        for task, child, result in completed_children:
            request_id = f"{child.invocation_id}:turn:0"
            if not result.complete or result.subagent_tasks:
                raise RuntimeError("nested workload children must return one terminal result")
            if self.llm_event_source == LLMEventSource.WORKFLOW:
                self.emitter.emit(
                    RuntimeEventKind.LLM_RESULT,
                    invocation_id=child.invocation_id,
                    context_id=child.context_id,
                    context_epoch=0,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "request_id": request_id,
                        "output_tokens": result.output_tokens,
                        "parser_status": "valid",
                        "structured_action_kinds": ["final_answer"],
                        "structured_action_names": [],
                        "action_boundary_token_index": None,
                        "action_boundary_source": "runtime_structured_output",
                    },
                )
            else:
                self.emitter.emit(
                    RuntimeEventKind.STRUCTURED_ACTION,
                    invocation_id=child.invocation_id,
                    context_id=child.context_id,
                    context_epoch=0,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "output_tokens": result.output_tokens,
                        "parser_status": "valid",
                        "structured_action_kinds": ["final_answer"],
                        "structured_action_names": [],
                        "action_boundary_token_index": None,
                        "action_boundary_source": "runtime_structured_output",
                    },
                )
            self._return_invocation(child, outcome="subagent_complete")
        self.emitter.emit(
            RuntimeEventKind.JOIN_SATISFIED,
            join_id=join_id,
            confidence=EventConfidence.OBSERVED_EXACT,
        )

    def _invoke_subagent(
        self,
        task: SubagentTask,
        child: BeliefKVRequestMetadata,
        history: tuple[str, ...],
    ) -> PeerTurnResult:
        return self.backend.invoke(
            PeerTurnRequest(
                task=task.instruction,
                role=task.agent_definition_id,
                turn=0,
                history=history,
                metadata=child,
                is_subagent=True,
            )
        )

    def _return_invocation(
        self,
        metadata: BeliefKVRequestMetadata,
        *,
        outcome: str,
    ) -> None:
        if metadata.invocation_id in self._terminal_invocations:
            return
        self._terminal_invocations.add(metadata.invocation_id)
        self.emitter.emit(
            RuntimeEventKind.RETURN,
            invocation_id=metadata.invocation_id,
            context_id=metadata.context_id,
            context_epoch=self._epoch_by_invocation.get(
                metadata.invocation_id, metadata.context_epoch
            ),
            return_target_id=metadata.return_target_id,
            confidence=EventConfidence.OBSERVED_EXACT,
            attributes={"outcome": outcome},
        )

    def _finish_live_invocations(self, outcome: str) -> None:
        for metadata in self._metadata_by_invocation.values():
            self._return_invocation(metadata, outcome=outcome)

    @staticmethod
    def _request_id(metadata: BeliefKVRequestMetadata, turn: int) -> str:
        return f"{metadata.invocation_id}:turn:{turn}:epoch:{metadata.context_epoch}"


class OpenAICompatiblePeerBackend:
    """Structured peer backend for an OpenAI-compatible local endpoint."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_completion_tokens: int = 2048,
        timeout_s: float = 300.0,
        min_initial_subagents: int = 0,
        max_initial_subagents: int = 0,
        max_attempts: int = 2,
    ) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenAICompatiblePeerBackend requires the beliefkv-agents environment"
            ) from error
        if not 0 <= min_initial_subagents <= max_initial_subagents <= 4:
            raise ValueError(
                "initial subagent bounds must satisfy 0 <= min <= max <= 4"
            )
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout_s,
            max_retries=0,
            streaming=False,
        )
        self.min_initial_subagents = min_initial_subagents
        self.max_initial_subagents = max_initial_subagents
        self.max_attempts = max_attempts
        self._stats_lock = threading.Lock()
        self._request_count = 0
        self._retry_count = 0
        self._model_error_count = 0

    def invoke(self, request: PeerTurnRequest) -> PeerTurnResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = self._system_prompt(request)
        user = (
            f"Task:\n{request.task}\n\nRole: {request.role}\n"
            f"Prior summaries:\n"
            + "\n".join(request.history[-8:])
            + f"\nTurn: {request.turn}"
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            repair = (
                "\nPrevious structured response was rejected: "
                f"{last_error}. Return a corrected object."
                if last_error is not None
                else ""
            )
            with self._stats_lock:
                self._request_count += 1
                self._retry_count += attempt > 1
            try:
                structured_model = self.model.bind(
                    response_format=self._response_format(request)
                )
                message = structured_model.invoke(
                    [
                        SystemMessage(content=system),
                        HumanMessage(content=user + repair),
                    ],
                    extra_body={"beliefkv_metadata": request.metadata.to_wire()},
                )
            except BaseException:
                with self._stats_lock:
                    self._model_error_count += 1
                raise
            try:
                raw = json.loads(message.text)
                if not isinstance(raw, Mapping):
                    raise TypeError("peer response must be a JSON object")
                result = self._result_from_mapping(request, raw, message)
                return result
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                last_error = error
        raise RuntimeError(
            f"peer structured completion failed after {self.max_attempts} attempts: "
            f"{last_error}"
        )

    def _result_from_mapping(
        self,
        request: PeerTurnRequest,
        raw: Mapping[str, object],
        message: Any,
    ) -> PeerTurnResult:
        if not isinstance(raw.get("summary"), str) or not raw["summary"].strip():
            raise TypeError("summary must be a non-empty string")
        if not isinstance(raw.get("complete"), bool):
            raise TypeError("complete must be a boolean")
        next_raw = raw.get("next_role")
        if next_raw is not None and not isinstance(next_raw, str):
            raise TypeError("next_role must be a string or null")
        tasks_raw = raw.get("subagent_tasks")
        if not isinstance(tasks_raw, list):
            raise TypeError("subagent_tasks must be a list")
        if not all(isinstance(item, Mapping) for item in tasks_raw):
            raise TypeError("each subagent task must be an object")
        tasks = tuple(
            SubagentTask(
                agent_definition_id=self._required_string(
                    item, "agent_definition_id"
                ),
                instruction=self._required_string(item, "instruction"),
            )
            for item in tasks_raw
        )
        if len(tasks) > 4:
            raise ValueError("a peer turn cannot spawn more than four subagents")
        usage = getattr(message, "usage_metadata", None) or {}
        result = PeerTurnResult(
            summary=raw["summary"].strip(),
            next_role=PeerRole(str(next_raw)) if next_raw is not None else None,
            complete=raw["complete"],
            output_tokens=int(usage.get("output_tokens", 0)),
            subagent_tasks=tasks,
        )
        if request.must_complete and not result.complete:
            raise ValueError("the final allowed peer turn must complete")
        if (
            not request.is_subagent
            and not result.complete
            and result.next_role is not None
            and result.next_role.value == request.role
        ):
            raise ValueError("a peer handoff must target a different role")
        if request.is_subagent and (
            not result.complete
            or result.next_role is not None
            or result.subagent_tasks
        ):
            raise ValueError("subagent response must terminate without nested work")
        if (
            not request.is_subagent
            and request.role == PeerRole.CODER.value
            and request.turn == 0
        ):
            if result.complete or result.next_role == PeerRole.CODER:
                raise ValueError(
                    "initial coder turn must hand off to reviewer or tester"
                )
            if not (
                self.min_initial_subagents
                <= len(result.subagent_tasks)
                <= self.max_initial_subagents
            ):
                raise ValueError(
                    "initial coder turn returned a subagent fan-out outside the "
                    "configured bounds"
                )
        if not request.is_subagent and request.turn > 0 and result.subagent_tasks:
            raise ValueError("subagent tasks are only allowed on the initial coder turn")
        return result

    def _system_prompt(self, request: PeerTurnRequest) -> str:
        base = (
            "You are one role in a code-orchestrated software workflow. The server "
            "enforces a JSON schema with summary, next_role, complete, and "
            "subagent_tasks. next_role is coder, reviewer, tester, or null. A "
            "complete response must use null next_role; an incomplete response must "
            "choose a next role. Keep the summary concrete and grounded in the task. "
            "Keep summary under 512 characters and each subagent instruction under "
            "256 characters. Do not include analysis outside the JSON object. "
        )
        if request.is_subagent:
            return base + (
                "You are a leaf investigator. Return complete=true, next_role=null, "
                "and an empty subagent_tasks list."
            )
        if request.must_complete:
            return base + (
                "This is the final allowed workflow turn. Synthesize the available "
                "evidence and return complete=true, next_role=null, and an empty "
                "subagent_tasks list."
            )
        if request.role == PeerRole.CODER.value and request.turn == 0:
            if self.max_initial_subagents == 0:
                spawn_instruction = "Return an empty subagent_tasks list."
            else:
                spawn_instruction = (
                    "Choose the useful fan-out from "
                    f"{self.min_initial_subagents} through "
                    f"{self.max_initial_subagents} independent investigation tasks. "
                    "You decide the count, role labels, and task decomposition from "
                    "the problem; do not create redundant tasks."
                )
            return base + " " + spawn_instruction + (
                " The first coder turn must hand off with complete=false; choose "
                "reviewer or tester as next_role based on what should happen next."
            )
        return base + (
            "Coder hands off to reviewer, reviewer either hands off to tester or "
            "finishes after test evidence, and tester hands back to coder when a "
            "revision is needed. Do not create additional subagents after turn zero."
        )

    def _response_format(self, request: PeerTurnRequest) -> dict[str, object]:
        if request.is_subagent:
            return _peer_response_format(
                name="beliefkv_leaf_turn",
                min_subagents=0,
                max_subagents=0,
                allowed_next_roles=(),
                required_complete=True,
            )
        if request.must_complete:
            return _peer_response_format(
                name="beliefkv_final_peer_turn",
                min_subagents=0,
                max_subagents=0,
                allowed_next_roles=(),
                required_complete=True,
            )
        if request.role == PeerRole.CODER.value and request.turn == 0:
            return _peer_response_format(
                name="beliefkv_initial_peer_turn",
                min_subagents=self.min_initial_subagents,
                max_subagents=self.max_initial_subagents,
                allowed_next_roles=(PeerRole.REVIEWER, PeerRole.TESTER),
                required_complete=False,
            )
        return _peer_response_format(
            name="beliefkv_peer_continuation",
            min_subagents=0,
            max_subagents=0,
            allowed_next_roles=tuple(
                role for role in PeerRole if role.value != request.role
            ),
            required_complete=None,
        )

    def summary(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "model_request_count": self._request_count,
                "structured_retry_count": self._retry_count,
                "model_error_count": self._model_error_count,
            }

    @staticmethod
    def _required_string(raw: Mapping[str, object], key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{key} must be a non-empty string")
        return value.strip()


def _peer_response_format(
    *,
    name: str,
    min_subagents: int,
    max_subagents: int,
    allowed_next_roles: tuple[PeerRole, ...],
    required_complete: bool | None,
) -> dict[str, object]:
    if not 0 <= min_subagents <= max_subagents <= 4:
        raise ValueError("invalid response-format subagent bounds")
    if required_complete is True:
        next_role_schema: dict[str, object] = {"type": "null"}
        complete_schema: dict[str, object] = {"type": "boolean", "enum": [True]}
    elif required_complete is False:
        next_role_schema = {
            "type": "string",
            "enum": [role.value for role in allowed_next_roles],
        }
        complete_schema = {"type": "boolean", "enum": [False]}
    else:
        next_role_schema = {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [role.value for role in allowed_next_roles],
                },
                {"type": "null"},
            ]
        }
        complete_schema = {"type": "boolean"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "next_role": next_role_schema,
                    "complete": complete_schema,
                    "subagent_tasks": {
                        "type": "array",
                        "minItems": min_subagents,
                        "maxItems": max_subagents,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "agent_definition_id": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 64,
                                },
                                "instruction": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 256,
                                },
                            },
                            "required": [
                                "agent_definition_id",
                                "instruction",
                            ],
                        },
                    },
                },
                "required": [
                    "summary",
                    "next_role",
                    "complete",
                    "subagent_tasks",
                ],
            },
        },
    }


def _short_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _transition_hash(events: tuple[RuntimeEvent, ...]) -> str:
    transitions = [
        {
            "kind": event.kind.value,
            "workflow_id": event.workflow_id,
            "invocation_id": event.invocation_id,
            "target_invocation_id": event.target_invocation_id,
            "context_id": event.context_id,
            "join_id": event.join_id,
            "members": list(event.member_invocation_ids),
        }
        for event in events
    ]
    encoded = json.dumps(
        transitions,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
