from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from beliefkv.experiments.agent_protocol import (
    AgentLoopGuardMiddleware,
    ChildCompletion,
    LoopGuardPolicy,
    require_structured_completion,
)
from beliefkv.experiments.deepagents_swebench import (
    CHILD_COMPLETION_INSTRUCTION,
    SANDBOX_PATH_CONTRACT,
    DockerWorkspaceBackend,
    JsonlAudit,
    _filesystem_middleware,
    _invoke_with_partial_state,
    _result_messages,
    _workspace_patch_tool,
)
from beliefkv.experiments.langgraph_peer_workflow import (
    PeerAgentBackend,
    PeerRole,
    PeerTurnRequest,
    PeerTurnResult,
)
from beliefkv.runtime.agent_runtime_adapter import RuntimeEventSink
from beliefkv.runtime.agent_safety import ActivationDeadline
from beliefkv.runtime.context_lifecycle import (
    CONTEXT_LIFECYCLE_PRIVATE_STATE_KEYS,
    CompletionBudgetMiddleware,
    ContextLifecycleMiddleware,
    ContextLifecyclePolicy,
)
from beliefkv.runtime.deepagents_adapter import (
    BeliefKVChatOpenAI,
    DeepAgentsRuntimeAdapter,
)
from beliefkv.runtime.langchain_tool_safety import (
    ToolCircuitBreakerMiddleware,
    ToolOutcomeStatusMiddleware,
)
from beliefkv.runtime.subagent_state import PrivateStateIsolatingSubAgentMiddleware


def summarize_agentic_runtime_trace(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_counts = Counter(str(item.get("kind")) for item in records)
    event_timestamps_ms = sorted(
        float(item["ts_ms"])
        for item in records
        if item.get("ts_ms") is not None
    )
    spawn_timestamps_ms = sorted(
        float(item["ts_ms"])
        for item in records
        if item.get("kind") == "spawn" and item.get("ts_ms") is not None
    )
    children = {
        str(item["invocation_id"])
        for item in records
        if item.get("kind") == "invocation_create"
        and item.get("parent_invocation_id") is not None
        and not bool((item.get("attributes") or {}).get("runtime_internal"))
    }
    llm_by_invocation: Counter[str] = Counter()
    tools_by_invocation: Counter[str] = Counter()
    max_epoch_by_invocation: dict[str, int] = {}
    tool_names: Counter[str] = Counter()
    tool_statuses: Counter[str] = Counter()
    tool_error_classes: Counter[str] = Counter()
    tool_durations_ms: list[float] = []
    workspace_digest_observations = 0
    workspace_change_count = 0
    mutating_tool_end_count = 0
    rejected_task_calls = 0
    for item in records:
        invocation_id = item.get("invocation_id")
        kind = item.get("kind")
        attributes = item.get("attributes") or {}
        if (
            kind == "llm_submit"
            and invocation_id is not None
            and not bool(attributes.get("runtime_internal"))
        ):
            key = str(invocation_id)
            llm_by_invocation[key] += 1
            epoch = item.get("context_epoch")
            if epoch is not None:
                max_epoch_by_invocation[key] = max(
                    max_epoch_by_invocation.get(key, -1), int(epoch)
                )
        elif kind == "tool_start" and invocation_id is not None:
            tools_by_invocation[str(invocation_id)] += 1
            tool_names[str(attributes.get("tool_name", "unknown"))] += 1
        elif kind == "tool_end":
            duration = attributes.get("duration_ms")
            if duration is not None:
                tool_durations_ms.append(float(duration))
            status = attributes.get("status")
            if status is not None:
                tool_statuses[str(status)] += 1
            error_class = attributes.get("tool_error_class")
            if error_class is not None:
                tool_error_classes[str(error_class)] += 1
            if attributes.get("tool_name") in {
                "apply_patch",
                "edit_file",
                "write_file",
            }:
                mutating_tool_end_count += 1
            if (
                attributes.get("workspace_digest_before") is not None
                and attributes.get("workspace_digest_after") is not None
            ):
                workspace_digest_observations += 1
                workspace_change_count += int(
                    bool(attributes.get("workspace_changed"))
                )
        if kind == "llm_result" and not bool(attributes.get("runtime_internal")):
            rejected_task_calls += int(
                attributes.get("rejected_task_call_count", 0) or 0
            )
    child_stats = [
        {
            "invocation_id": invocation_id,
            "llm_request_count": llm_by_invocation[invocation_id],
            "tool_call_count": tools_by_invocation[invocation_id],
            "max_context_epoch": max_epoch_by_invocation.get(invocation_id),
            "returned": any(
                item.get("kind") == "return"
                and item.get("invocation_id") == invocation_id
                for item in records
            ),
        }
        for invocation_id in sorted(children)
    ]
    durations = sorted(tool_durations_ms)

    def percentile(values: list[float], quantile: float) -> float | None:
        if not values:
            return None
        index = round((len(values) - 1) * quantile)
        return values[index]

    return {
        "event_count": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "first_event_ts_ms": min(event_timestamps_ms, default=None),
        "last_event_ts_ms": max(event_timestamps_ms, default=None),
        "spawn_timestamps_ms": spawn_timestamps_ms,
        "dynamic_subagent_count": event_counts["spawn"],
        "llm_request_count": sum(llm_by_invocation.values()),
        "context_compaction_count": event_counts["context_compact"],
        "internal_summary_request_count": sum(
            item.get("kind") == "llm_submit"
            and bool((item.get("attributes") or {}).get("runtime_internal"))
            for item in records
        ),
        "tool_call_count": event_counts["tool_start"],
        "tool_name_counts": dict(sorted(tool_names.items())),
        "tool_status_counts": dict(sorted(tool_statuses.items())),
        "tool_error_class_counts": dict(sorted(tool_error_classes.items())),
        "tool_status_coverage": (
            sum(tool_statuses.values()) / event_counts["tool_end"]
            if event_counts["tool_end"]
            else 1.0
        ),
        "workspace_digest_observation_count": workspace_digest_observations,
        "mutating_tool_end_count": mutating_tool_end_count,
        "workspace_digest_coverage": (
            workspace_digest_observations / mutating_tool_end_count
            if mutating_tool_end_count
            else 1.0
        ),
        "workspace_change_count": workspace_change_count,
        "tool_duration_ms": {
            "count": len(durations),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": max(durations, default=None),
        },
        "rejected_task_call_count": rejected_task_calls,
        "multi_turn_subagent_count": sum(
            item["llm_request_count"] > 1 for item in child_stats
        ),
        "all_subagents_returned": bool(child_stats)
        and all(bool(item["returned"]) for item in child_stats),
        "all_joins_satisfied": event_counts["join_create"]
        == event_counts["join_satisfied"],
        "child_invocations": child_stats,
        "invocation_llm_request_counts": dict(sorted(llm_by_invocation.items())),
        "invocation_tool_call_counts": dict(sorted(tools_by_invocation.items())),
    }


class AgenticPeerDecision(BaseModel):
    summary: str = Field(
        description="Concrete result of this role activation",
        min_length=1,
        max_length=2000,
    )
    next_role: Literal["coder", "reviewer", "tester"] | None = Field(
        description="Peer to activate next, or null when the workflow is terminal"
    )
    complete: bool = Field(
        description="Whether the whole workflow should terminate after this activation"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Repository-relative files, symbols, diffs, or observed behavior",
        max_length=16,
    )
    tests: list[str] = Field(
        default_factory=list,
        description="Focused commands run and their observed outcomes",
        max_length=12,
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description="Concrete work that remains before a correct completion",
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> "AgenticPeerDecision":
        if self.complete == (self.next_role is not None):
            raise ValueError(
                "complete decisions require next_role=null; continuing decisions "
                "require a next_role"
            )
        return self


PEER_COMPLETION_INSTRUCTION = (
    "Return an AgenticPeerDecision. Set complete=true and next_role=null only when "
    "the workflow is terminal. Otherwise choose the role that should run next. The "
    "current role may continue when it has unfinished local work; that is a continuation, "
    "not a peer handoff."
)


TASK_MIDDLEWARE_PROMPT = """## Dynamic subagent delegation

The `task` tool launches a FRESH child context. A child is short-lived at the workflow
level, but it is a real multi-turn agent: it may search files, read code, execute
diagnostics, and run tests before returning one compact report. Decide fan-out from the
current task structure. When the activation prompt specifies a required delegation
range, obey that range before editing or handing off. Otherwise use zero tasks for
direct or dependency-ordered work. Issue independent task calls in one model response
so they can run concurrently. Do not delegate trivial lookups, duplicate an earlier
task, or delegate the final integration decision.
"""


COMMON_PEER_PROMPT = """You are one persistent role in a real software-engineering
workflow running in an isolated repository. Use repository tools before making claims.
The filesystem and execute tools share `/workspace`; the execute tool is offline and
resource-limited. Do not install dependencies or access external networks. Keep tool
queries targeted and use as many distinct calls as the task genuinely needs. Stop only
after obtaining sufficient evidence; do not run cosmetic variants of a command after
its outcome is already known. If a tool call fails, use its error to change the next
action; never submit the identical failed call again.

Each activation must terminate with the required AgenticPeerDecision structured
response. Ordinary prose is not a valid terminal response. A handoff transfers control
to a peer while preserving your own context for a possible later RESUME. Do not hand
off to yourself. A terminal decision must state unresolved work honestly; do not claim
success without a substantive patch and focused passing test evidence.
""" + SANDBOX_PATH_CONTRACT


ROLE_PROMPTS = {
    PeerRole.CODER.value: """You are the coder. Diagnose the issue, edit the smallest
correct implementation surface, and run focused tests. Delegate only independent deep
investigations that would otherwise pollute your context. After a substantive change,
normally hand off to reviewer or tester. You own repository edits and final integration.
""",
    PeerRole.REVIEWER.value: """You are the reviewer. Inspect the current diff and the
surrounding invariants using repository tools. Look for semantic gaps, regressions, and
missing coverage. Hand back to coder for revisions, hand to tester when the patch is
ready for execution, or complete only when evidence is sufficient.
""",
    PeerRole.TESTER.value: """You are the tester. Inspect the current patch and run the
narrowest repository-native tests that validate it. Do not edit source files. Hand back
to coder with exact failures, hand to reviewer for a final audit, or complete when the
patch and focused test evidence satisfy the task.
""",
}


CHILD_SPECS = (
    (
        "repository-explorer",
        "Trace a multi-step implementation or call path and return concrete symbols.",
        "Inspect the assigned implementation question deeply. Use search, file reads, "
        "and targeted diagnostics. Do not edit files. Return concrete repository "
        "evidence and the most likely invariant or symbol to change. You do not own "
        "implementation: once the assigned question is answered with concrete "
        "evidence, return ChildCompletion immediately instead of broadening scope.",
    ),
    (
        "test-analyst",
        "Reproduce a failure and identify focused regression coverage.",
        "Reproduce or characterize the assigned failure. Inspect existing tests and "
        "run focused commands when useful. Do not edit files. Return exact commands, "
        "outcomes, and recommended regression coverage. Once the failure mechanism "
        "and focused coverage are established, return ChildCompletion immediately; "
        "do not pursue unrelated failures or an exhaustive suite.",
    ),
    (
        "dependency-tracer",
        "Audit callers, downstream consumers, and compatibility risks.",
        "Trace callers and downstream consumers for the assigned question. Inspect "
        "compatibility risks and tests. Do not edit files. Once the relevant callers "
        "and risks are supported by concrete evidence, return ChildCompletion "
        "immediately rather than expanding to unrelated modules.",
    ),
    (
        "invariant-auditor",
        "Identify semantic invariants, edge cases, and compatibility constraints.",
        "Inspect the assigned invariant or edge-case question using repository tools. "
        "Do not edit files. Return the exact invariant, supporting symbols or tests, "
        "and concrete compatibility risks. Once supported, return ChildCompletion "
        "without broadening into implementation work.",
    ),
)

SUBAGENT_TYPES = frozenset(item[0] for item in CHILD_SPECS)


def count_accepted_task_calls(messages: list[BaseMessage]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if str(call.get("name", "")) != "task":
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if str(args.get("subagent_type", "")) in SUBAGENT_TYPES:
                count += 1
    return count


def count_initial_accepted_task_calls(messages: list[BaseMessage]) -> int:
    """Count only the first runtime delegation batch, not later dynamic spawns."""

    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        accepted = count_accepted_task_calls([message])
        if accepted:
            return accepted
    return 0


@dataclass(frozen=True)
class AgenticPeerBackendConfig:
    model: str
    base_url: str
    max_completion_tokens: int = 4096
    request_timeout_s: float = 900.0
    recursion_limit: int = 512
    max_decision_repairs: int = 2
    enable_subagents: bool = True
    required_initial_subagent_min: int = 0
    required_initial_subagent_max: int = 0
    context_lifecycle: ContextLifecyclePolicy = field(
        default_factory=ContextLifecyclePolicy
    )
    loop_guard: LoopGuardPolicy = field(
        default_factory=lambda: LoopGuardPolicy(
            repeated_call_limit=6,
            alternating_cycle_repetitions=3,
            consecutive_error_limit=6,
            consecutive_no_progress_limit=8,
            max_model_calls_without_completion=48,
            max_tool_calls_without_completion=128,
            recovery_model_call_limit=3,
        )
    )

    def __post_init__(self) -> None:
        if min(
            self.max_completion_tokens,
            self.recursion_limit,
            self.max_decision_repairs,
        ) <= 0:
            raise ValueError("agentic backend limits must be positive")
        if (
            self.context_lifecycle.intermediate_output_tokens
            > self.max_completion_tokens
        ):
            raise ValueError(
                "intermediate completion budget cannot exceed max_completion_tokens"
            )
        if not (
            0
            <= self.required_initial_subagent_min
            <= self.required_initial_subagent_max
            <= len(CHILD_SPECS)
        ):
            raise ValueError("required initial subagent range is invalid")
        if not self.enable_subagents and self.required_initial_subagent_max:
            raise ValueError("disabled subagents cannot have a required range")


@dataclass
class _PersistentPeerThread:
    role: str
    agent: Any
    model: BeliefKVChatOpenAI
    adapter: DeepAgentsRuntimeAdapter
    context_lifecycle: ContextLifecycleMiddleware
    activation_policy: "_ActivationPolicy"
    activation_deadline: ActivationDeadline
    owns_activation_deadline: bool
    messages: list[BaseMessage] = field(default_factory=list)
    summarization_event: dict[str, Any] | None = None
    activations: int = 0
    initial_subagent_count: int | None = None
    decision_repairs: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _ActivationPolicy:
    must_complete: bool = False
    delegation_allowed: bool = True


class _FinalActivationMiddleware(AgentMiddleware[Any, Any, Any]):
    def __init__(self, policy: _ActivationPolicy) -> None:
        super().__init__()
        self.policy = policy

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Any,
    ) -> ModelResponse[Any]:
        if not self.policy.must_complete and self.policy.delegation_allowed:
            return handler(request)
        retained_tools = [
            item for item in request.tools if getattr(item, "name", "") != "task"
        ]
        base_prompt = request.system_message.text if request.system_message else ""
        if self.policy.must_complete:
            directive = (
                "FINAL ACTIVATION\nDo not spawn subagents. Finish available repository "
                "work with direct tools, then return a terminal AgenticPeerDecision. "
                "List uncertainty in the summary when the patch cannot be fully verified."
            )
        else:
            directive = (
                "DELEGATION WINDOW CLOSED\nThe required initial subagents have already "
                "been created. Do not call task again; use their reports and direct "
                "repository tools."
            )
        final_prompt = f"{base_prompt}\n\n{directive}".strip()
        return handler(
            request.override(
                tools=retained_tools,
                system_message=SystemMessage(content=final_prompt),
            )
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Any,
    ) -> ModelResponse[Any]:
        if not self.policy.must_complete and self.policy.delegation_allowed:
            return await handler(request)
        retained_tools = [
            item for item in request.tools if getattr(item, "name", "") != "task"
        ]
        base_prompt = request.system_message.text if request.system_message else ""
        if self.policy.must_complete:
            directive = (
                "FINAL ACTIVATION\nDo not spawn subagents. Finish available repository "
                "work with direct tools, then return a terminal AgenticPeerDecision. "
                "List uncertainty in the summary when the patch cannot be fully verified."
            )
        else:
            directive = (
                "DELEGATION WINDOW CLOSED\nThe required initial subagents have already "
                "been created. Do not call task again; use their reports and direct "
                "repository tools."
            )
        final_prompt = f"{base_prompt}\n\n{directive}".strip()
        return await handler(
            request.override(
                tools=retained_tools,
                system_message=SystemMessage(content=final_prompt),
            )
        )


class ToolEnabledPeerBackend(PeerAgentBackend):
    """Persistent peer threads with real sandbox tools and dynamic subagents."""

    def __init__(
        self,
        *,
        config: AgenticPeerBackendConfig,
        workspace_backend: DockerWorkspaceBackend,
        audit: JsonlAudit,
        runtime_trace_sink: RuntimeEventSink,
        control_sink: RuntimeEventSink | None,
    ) -> None:
        self.config = config
        self.workspace_backend = workspace_backend
        self.audit = audit
        self.runtime_trace_sink = runtime_trace_sink
        self.control_sink = control_sink
        self._threads: dict[str, _PersistentPeerThread] = {}
        self._threads_lock = threading.Lock()
        self._activation_count = 0
        self._decision_repair_count = 0
        self._model_error_count = 0

    def invoke(self, request: PeerTurnRequest) -> PeerTurnResult:
        if request.is_subagent:
            raise ValueError(
                "ToolEnabledPeerBackend subagents run through the task tool, not "
                "through one-shot workflow leaf invocations"
            )
        thread = self._thread_for(request)
        with thread.lock:
            return self._invoke_thread(thread, request)

    def _thread_for(self, request: PeerTurnRequest) -> _PersistentPeerThread:
        context_id = request.metadata.context_id
        with self._threads_lock:
            existing = self._threads.get(context_id)
            if existing is not None:
                if existing.role != request.role:
                    raise RuntimeError("one context cannot change peer role")
                if (
                    request.workflow_deadline is not None
                    and existing.activation_deadline is not request.workflow_deadline
                ):
                    raise RuntimeError("persistent peer cannot change workflow deadline")
                return existing
            thread = self._build_thread(request)
            self._threads[context_id] = thread
            return thread

    def _build_thread(self, request: PeerTurnRequest) -> _PersistentPeerThread:
        namespace = "agentic" + hashlib.sha256(
            request.metadata.context_id.encode("utf-8")
        ).hexdigest()[:12]
        adapter = DeepAgentsRuntimeAdapter(
            self.runtime_trace_sink,
            request.metadata,
            control_sink=self.control_sink,
            event_namespace=namespace,
            completion_tool_names=frozenset(
                {"AgenticPeerDecision", "ChildCompletion"}
            ),
            allowed_subagent_types=SUBAGENT_TYPES,
            workspace_digest_provider=self.workspace_backend.tool_state_digest,
        )
        owns_activation_deadline = request.workflow_deadline is None
        activation_deadline = request.workflow_deadline or ActivationDeadline()
        server_root = self.config.base_url.rstrip("/")
        if server_root.endswith("/v1"):
            server_root = server_root[:-3]
        model = BeliefKVChatOpenAI(
            beliefkv_adapter=adapter,
            activation_deadline=activation_deadline,
            request_timeout_s=self.config.request_timeout_s,
            abort_url=f"{server_root}/abort_request",
            model=self.config.model,
            base_url=self.config.base_url,
            api_key="EMPTY",
            temperature=0.0,
            max_completion_tokens=self.config.max_completion_tokens,
            timeout=self.config.request_timeout_s,
            max_retries=0,
            streaming=False,
            disable_streaming="tool_calling",
        )
        summary_model = model.model_copy(
            update={"max_tokens": self.config.context_lifecycle.summary_output_tokens}
        )
        required_range = self.config.required_initial_subagent_min > 0
        activation_policy = _ActivationPolicy(
            delegation_allowed=self.config.enable_subagents
            and (not required_range or request.role == PeerRole.CODER.value)
        )
        middleware: list[Any] = [
            ToolCircuitBreakerMiddleware(
                state_epoch=self.workspace_backend.workspace_epoch,
                audit=self.audit,
                scope=f"agentic-peer:{request.role}",
            ),
            ToolOutcomeStatusMiddleware(),
            _filesystem_middleware(
                self.workspace_backend,
                allow_direct_edits=request.role == PeerRole.CODER.value,
            )
        ]
        if self.config.enable_subagents and (
            not required_range or request.role == PeerRole.CODER.value
        ):
            middleware.append(
                PrivateStateIsolatingSubAgentMiddleware(
                    backend=self.workspace_backend,
                    private_state_keys=CONTEXT_LIFECYCLE_PRIVATE_STATE_KEYS,
                    subagents=[
                        {
                            "name": name,
                            "description": description,
                            "system_prompt": child_prompt + SANDBOX_PATH_CONTRACT,
                            "model": model,
                            "tools": [],
                            "middleware": [
                                ToolCircuitBreakerMiddleware(
                                    state_epoch=(
                                        self.workspace_backend.workspace_epoch
                                    ),
                                    audit=self.audit,
                                    scope=(
                                        f"agentic-child:{request.role}:{name}"
                                    ),
                                ),
                                ToolOutcomeStatusMiddleware(),
                                _filesystem_middleware(
                                    self.workspace_backend,
                                    allow_direct_edits=False,
                                ),
                                ContextLifecycleMiddleware(
                                    summary_model,
                                    backend=self.workspace_backend,
                                    policy=self.config.context_lifecycle,
                                    compaction_sink=adapter,
                                    summary_callbacks=(adapter,),
                                ),
                                CompletionBudgetMiddleware(
                                    intermediate_tokens=(
                                        self.config.context_lifecycle.intermediate_output_tokens
                                    ),
                                    final_tokens=self.config.max_completion_tokens,
                                ),
                                AgentLoopGuardMiddleware(
                                    policy=self.config.loop_guard,
                                    completion_schema=ChildCompletion,
                                    completion_instruction=CHILD_COMPLETION_INSTRUCTION,
                                    audit=self.audit,
                                    scope=f"agentic-child:{request.role}:{name}",
                                    activation_deadline=activation_deadline,
                                ),
                            ],
                            "response_format": ToolStrategy(ChildCompletion),
                        }
                        for name, description, child_prompt in CHILD_SPECS
                    ],
                    system_prompt=TASK_MIDDLEWARE_PROMPT,
                )
            )
        context_lifecycle = ContextLifecycleMiddleware(
            summary_model,
            backend=self.workspace_backend,
            policy=self.config.context_lifecycle,
            compaction_sink=adapter,
            summary_callbacks=(adapter,),
            persist_cursor_across_invocations=True,
        )
        middleware.append(context_lifecycle)
        middleware.append(
            CompletionBudgetMiddleware(
                intermediate_tokens=(
                    self.config.context_lifecycle.intermediate_output_tokens
                ),
                final_tokens=self.config.max_completion_tokens,
                final_mode=lambda: activation_policy.must_complete,
            )
        )
        middleware.append(_FinalActivationMiddleware(activation_policy))
        middleware.append(
            AgentLoopGuardMiddleware(
                policy=self.config.loop_guard,
                completion_schema=AgenticPeerDecision,
                completion_instruction=PEER_COMPLETION_INSTRUCTION,
                audit=self.audit,
                scope=f"agentic-peer:{request.role}",
                activation_deadline=activation_deadline,
                finalization_tool_names=(
                    frozenset({"apply_patch", "execute"})
                    if request.role == PeerRole.CODER.value
                    else frozenset({"execute"})
                ),
            )
        )
        tools = (
            [_workspace_patch_tool(self.workspace_backend)]
            if request.role == PeerRole.CODER.value
            else []
        )
        agent = create_agent(
            model=model,
            tools=tools,
            middleware=middleware,
            response_format=ToolStrategy(AgenticPeerDecision),
            system_prompt=COMMON_PEER_PROMPT + "\n\n" + ROLE_PROMPTS[request.role],
            name=f"beliefkv-agentic-{request.role}",
        )
        return _PersistentPeerThread(
            role=request.role,
            agent=agent,
            model=model,
            adapter=adapter,
            context_lifecycle=context_lifecycle,
            activation_policy=activation_policy,
            activation_deadline=activation_deadline,
            owns_activation_deadline=owns_activation_deadline,
        )

    def _invoke_thread(
        self,
        thread: _PersistentPeerThread,
        request: PeerTurnRequest,
    ) -> PeerTurnResult:
        if thread.owns_activation_deadline:
            thread.activation_deadline.start(
                self.config.loop_guard.activation_wall_clock_s
            )
        thread.activation_policy.must_complete = request.must_complete
        try:
            required_initial_delegation = self._requires_initial_delegation(
                thread, request
            )
            result = self._invoke_thread_active(thread, request)
            if required_initial_delegation:
                thread.activation_policy.delegation_allowed = False
            return result
        finally:
            thread.activation_policy.must_complete = False
            if thread.owns_activation_deadline:
                thread.activation_deadline.clear()

    def _requires_initial_delegation(
        self,
        thread: _PersistentPeerThread,
        request: PeerTurnRequest,
    ) -> bool:
        return (
            self.config.required_initial_subagent_min > 0
            and request.role == PeerRole.CODER.value
            and request.turn == 0
            and thread.activations == 0
        )

    def _invoke_thread_active(
        self,
        thread: _PersistentPeerThread,
        request: PeerTurnRequest,
    ) -> PeerTurnResult:
        required_initial_delegation = self._requires_initial_delegation(
            thread, request
        )
        activation_prompt = self._activation_prompt(
            request,
            required_initial_delegation=required_initial_delegation,
        )
        messages = [*thread.messages, HumanMessage(content=activation_prompt)]
        previous_count = len(thread.messages)
        last_error: str | None = None
        result: dict[str, Any] = {}
        for attempt in range(self.config.max_decision_repairs + 1):
            if last_error is not None:
                messages.append(
                    HumanMessage(
                        content=(
                            "Runtime rejected the previous activation result: "
                            f"{last_error}. Satisfy the missing runtime requirement "
                            "using the available tools, without repeating completed "
                            "subagent work, then return a corrected terminal decision. "
                            f"{PEER_COMPLETION_INSTRUCTION}"
                        )
                    )
                )
            try:
                result = _invoke_with_partial_state(
                    thread.agent,
                    {"messages": messages},
                    {
                        "callbacks": [thread.adapter],
                        "recursion_limit": self.config.recursion_limit,
                        "metadata": {
                            "beliefkv_mode": "agentic_peer",
                            "beliefkv_role": request.role,
                            "beliefkv_activation": thread.activations,
                        },
                    },
                )
            except BaseException:
                with self._threads_lock:
                    self._model_error_count += 1
                raise
            thread.messages = _result_messages(result)
            thread.summarization_event = (
                thread.context_lifecycle.latest_summarization_event()
            )
            messages = list(thread.messages)
            decision = require_structured_completion(result, AgenticPeerDecision)
            assert isinstance(decision, AgenticPeerDecision)
            guard_intervened = bool(
                result.get("guard_ever_intervened", False)
                or result.get("guard_forcing_completion", False)
            )
            guard_reason = str(result.get("guard_reason") or "runtime_guard")
            initial_subagent_count = count_initial_accepted_task_calls(
                thread.messages[previous_count:]
            )
            last_error = self._decision_error(
                decision,
                request,
                required_initial_delegation=required_initial_delegation,
                initial_subagent_count=initial_subagent_count,
            )
            if last_error is None:
                if required_initial_delegation:
                    thread.initial_subagent_count = initial_subagent_count
                break
            thread.decision_repairs += 1
            with self._threads_lock:
                self._decision_repair_count += 1
        else:
            raise RuntimeError(
                "agentic peer failed runtime decision validation: " + str(last_error)
            )

        thread.activations += 1
        with self._threads_lock:
            self._activation_count += 1
        new_messages = thread.messages[previous_count:]
        output_tokens = sum(
            int((message.usage_metadata or {}).get("output_tokens", 0))
            for message in new_messages
            if isinstance(message, AIMessage)
        )
        summary = decision.summary.strip()
        if decision.evidence:
            summary += " Evidence: " + "; ".join(decision.evidence)
        if decision.tests:
            summary += " Tests: " + "; ".join(decision.tests)
        if decision.unresolved:
            summary += " Unresolved: " + "; ".join(decision.unresolved)
        if guard_intervened:
            summary += f" Runtime guard: {guard_reason}."
        return PeerTurnResult(
            summary=summary,
            next_role=(
                PeerRole(decision.next_role)
                if decision.next_role is not None
                else None
            ),
            complete=decision.complete,
            output_tokens=output_tokens,
            final_context_epoch=thread.adapter.latest_context_epoch(),
            terminal_outcome=None,
        )

    def cancel(self, *, reason: str) -> None:
        with self._threads_lock:
            threads = tuple(self._threads.values())
        for thread in threads:
            thread.model.cancel_active_requests()
            thread.adapter.cancel_pending_tasks(reason=reason)
        self.workspace_backend.cancel_active_commands(reason=reason)

    def _decision_error(
        self,
        decision: AgenticPeerDecision,
        request: PeerTurnRequest,
        *,
        required_initial_delegation: bool,
        initial_subagent_count: int,
    ) -> str | None:
        if required_initial_delegation:
            minimum = self.config.required_initial_subagent_min
            maximum = self.config.required_initial_subagent_max
            if initial_subagent_count < minimum:
                return (
                    f"initial delegation requires {minimum} through {maximum} accepted "
                    f"task calls; observed {initial_subagent_count}"
                )
            if initial_subagent_count > maximum:
                return (
                    f"initial delegation permits at most {maximum} accepted task calls; "
                    f"observed {initial_subagent_count}"
                )
            if decision.complete:
                return "initial coder activation must hand off after delegated work"
        if request.must_complete and not decision.complete:
            return "the final allowed workflow turn must terminate"
        return None

    def _activation_prompt(
        self,
        request: PeerTurnRequest,
        *,
        required_initial_delegation: bool,
    ) -> str:
        history = "\n".join(request.history) or "(no peer reports yet)"
        final_directive = (
            "This is the final allowed peer activation. Return a terminal decision; "
            "list unresolved work rather than starting another handoff."
            if request.must_complete
            else "Choose the next action from current repository evidence."
        )
        delegation_directive = ""
        if required_initial_delegation:
            delegation_directive = (
                "\n\nMANDATORY INITIAL DELEGATION\n"
                f"Before editing or handing off, create between "
                f"{self.config.required_initial_subagent_min} and "
                f"{self.config.required_initial_subagent_max} independent subagents "
                "with the task tool. Issue their task calls together when independent. "
                "Give each child a distinct, substantive investigation with a concrete "
                "deliverable. Wait for every child, integrate their reports, and then "
                "handoff to reviewer or tester."
            )
        return (
            f"Original task:\n{request.task}\n\n"
            f"Current role: {request.role}\n"
            f"Workflow turn: {request.turn}\n\n"
            f"Reports from other peer activations:\n{history}\n\n"
            f"{final_directive}{delegation_directive}"
        )

    def summary(self) -> dict[str, Any]:
        with self._threads_lock:
            threads = list(self._threads.values())
            return {
                "backend": "persistent_tool_enabled",
                "persistent_peer_threads": len(threads),
                "peer_activation_count": self._activation_count,
                "decision_repair_count": self._decision_repair_count,
                "model_error_count": self._model_error_count,
                "subagents_enabled": self.config.enable_subagents,
                "required_initial_subagent_range": [
                    self.config.required_initial_subagent_min,
                    self.config.required_initial_subagent_max,
                ],
                "observed_initial_subagent_count": next(
                    (
                        thread.initial_subagent_count
                        for thread in threads
                        if thread.role == PeerRole.CODER.value
                    ),
                    None,
                ),
                "thread_activations": {
                    thread.role: thread.activations for thread in threads
                },
                "thread_message_counts": {
                    thread.role: len(thread.messages) for thread in threads
                },
                "thread_context_compacted": {
                    thread.role: thread.summarization_event is not None
                    for thread in threads
                },
                "thread_history_sha256": {
                    thread.role: hashlib.sha256(
                        json.dumps(
                            [message.model_dump(mode="json") for message in thread.messages],
                            sort_keys=True,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    for thread in threads
                },
            }
