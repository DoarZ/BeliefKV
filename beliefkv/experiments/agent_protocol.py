from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Callable, Literal, Protocol, Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from pydantic import BaseModel, Field
from typing_extensions import NotRequired


class AuditSink(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...


class ActivationDeadlineExceeded(TimeoutError):
    """Raised before a model request that has no activation budget left."""


class ActivationDeadline:
    """Thread-safe absolute deadline shared by one activation and its children."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._deadline: float | None = None

    def start(self, budget_s: float) -> None:
        if not math.isfinite(budget_s) or budget_s <= 0:
            raise ValueError("activation deadline budget must be positive")
        now = self._clock()
        with self._lock:
            if self._deadline is not None:
                raise RuntimeError("activation deadline is already active")
            self._started_at = now
            self._deadline = now + budget_s

    def clear(self) -> None:
        with self._lock:
            self._started_at = None
            self._deadline = None

    def elapsed_s(self) -> float | None:
        now = self._clock()
        with self._lock:
            if self._started_at is None:
                return None
            return max(0.0, now - self._started_at)

    def remaining_s(self) -> float | None:
        now = self._clock()
        with self._lock:
            if self._deadline is None:
                return None
            return max(0.0, self._deadline - now)

    def request_timeout_s(self, cap_s: float) -> float:
        if not math.isfinite(cap_s) or cap_s <= 0:
            raise ValueError("single-request timeout cap must be positive")
        remaining = self.remaining_s()
        if remaining is None:
            return cap_s
        if remaining <= 0:
            raise ActivationDeadlineExceeded(
                "activation wall-clock deadline expired before model submission"
            )
        return min(cap_s, remaining)


class ChildCompletion(BaseModel):
    status: Literal["complete", "blocked"] = Field(
        description="Whether the delegated task was completed or blocked"
    )
    summary: str = Field(description="Concise answer to the delegated task")
    evidence: list[str] = Field(
        default_factory=list,
        description="Concrete file, symbol, line, or runtime evidence",
        max_length=16,
    )
    tests: list[str] = Field(
        default_factory=list,
        description="Commands run and their observed outcomes",
        max_length=12,
    )
    files_changed: list[str] = Field(
        default_factory=list,
        description="Files changed by this child, normally empty for analysis children",
        max_length=12,
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description="Remaining uncertainty or blockers",
        max_length=12,
    )
    confidence: Literal["low", "medium", "high"]


class WorkflowCompletion(BaseModel):
    status: Literal[
        "patched_and_tested",
        "patched_unverified",
        "no_patch_needed",
        "blocked",
    ] = Field(description="Terminal status of the complete SWE-bench workflow")
    summary: str = Field(description="Concise implementation summary")
    files_changed: list[str] = Field(
        default_factory=list,
        description="Repository-relative files changed by the workflow",
        max_length=24,
    )
    tests: list[str] = Field(
        default_factory=list,
        description="Commands run and their observed outcomes",
        max_length=16,
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description="Remaining uncertainty, failed checks, or blockers",
        max_length=16,
    )


@dataclass(frozen=True)
class LoopGuardPolicy:
    enabled: bool = True
    repeated_call_limit: int = 3
    alternating_cycle_repetitions: int = 3
    consecutive_error_limit: int = 3
    consecutive_no_progress_limit: int = 5
    max_model_calls_without_completion: int = 48
    max_tool_calls_without_completion: int = 128
    recovery_model_call_limit: int = 3
    activation_wall_clock_s: float = 1800.0

    def __post_init__(self) -> None:
        values = (
            self.repeated_call_limit,
            self.alternating_cycle_repetitions,
            self.consecutive_error_limit,
            self.consecutive_no_progress_limit,
            self.max_model_calls_without_completion,
            self.max_tool_calls_without_completion,
            self.recovery_model_call_limit,
        )
        if min(values) <= 0:
            raise ValueError("loop guard limits must be positive")
        if self.alternating_cycle_repetitions < 2:
            raise ValueError("alternating cycle detection needs at least two repetitions")
        if (
            not math.isfinite(self.activation_wall_clock_s)
            or self.activation_wall_clock_s <= 0
        ):
            raise ValueError("activation wall-clock budget must be positive")


@dataclass(frozen=True)
class LoopGuardSnapshot:
    model_calls: int
    tool_calls: int
    completed_tool_calls: int
    consecutive_errors: int
    consecutive_no_progress: int
    reason: str | None
    recent_tool_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoopGuardState(AgentState[Any]):
    guard_forcing_completion: NotRequired[
        Annotated[bool, UntrackedValue, PrivateStateAttr]
    ]
    guard_reason: NotRequired[Annotated[str, UntrackedValue, PrivateStateAttr]]
    guard_trigger_model_calls: NotRequired[
        Annotated[int, UntrackedValue, PrivateStateAttr]
    ]
    guard_activation_started_monotonic: NotRequired[
        Annotated[float, UntrackedValue, PrivateStateAttr]
    ]


def _canonical_tool_call(tool_call: dict[str, Any]) -> str:
    payload = {
        "name": str(tool_call.get("name", "")),
        "args": tool_call.get("args", {}),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _tool_result_is_error(message: ToolMessage) -> bool:
    if getattr(message, "status", None) == "error":
        return True
    normalized = _message_text(message).strip().lower()
    error_prefixes = (
        "error:",
        "tool error",
        "path_not_found",
        "permission denied",
        "timed out",
        "traceback (most recent call last)",
    )
    if normalized.startswith(error_prefixes):
        return True
    if message.name == "execute":
        execute_markers = (
            "command failed with exit code",
            "command exceeded host timeout",
            "killed by signal",
        )
        return any(marker in normalized for marker in execute_markers)
    return False


def analyze_agent_history(
    messages: Sequence[BaseMessage],
    policy: LoopGuardPolicy,
    *,
    completion_tool_names: frozenset[str] = frozenset(),
) -> LoopGuardSnapshot:
    # A persistent peer thread can complete one activation and later be resumed.
    # Guard budgets apply to the current activation, not the thread's lifetime.
    last_completion = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, ToolMessage)
            and message.name in completion_tool_names
        ),
        default=-1,
    )
    messages = messages[last_completion + 1 :]
    model_calls = sum(isinstance(message, AIMessage) for message in messages)
    calls_by_id: dict[str, tuple[str, str, int]] = {}
    call_signatures: list[str] = []
    call_batch_signatures: list[tuple[str, ...]] = []
    tool_names: list[str] = []

    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        batch_index = len(call_batch_signatures)
        batch_signatures: list[str] = []
        for call in message.tool_calls:
            name = str(call.get("name", ""))
            if name in completion_tool_names:
                continue
            signature = _canonical_tool_call(call)
            call_signatures.append(signature)
            batch_signatures.append(signature)
            tool_names.append(name)
            call_id = str(call.get("id", ""))
            if call_id:
                calls_by_id[call_id] = (signature, name, batch_index)
        if batch_signatures:
            call_batch_signatures.append(tuple(sorted(batch_signatures)))

    seen_signatures: set[str] = set()
    seen_outputs: set[str] = set()
    completed_tool_calls = 0
    consecutive_errors = 0
    consecutive_no_progress = 0
    results_by_batch: dict[int, list[tuple[str, str, bool]]] = {}
    unmatched_batch_index = len(call_batch_signatures)
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name in completion_tool_names:
            continue
        call = calls_by_id.get(str(message.tool_call_id))
        signature = call[0] if call is not None else ""
        batch_index = call[2] if call is not None else unmatched_batch_index
        output_digest = hashlib.sha256(
            _message_text(message).encode("utf-8", errors="replace")
        ).hexdigest()
        is_error = _tool_result_is_error(message)
        completed_tool_calls += 1
        results_by_batch.setdefault(batch_index, []).append(
            (signature, output_digest, is_error)
        )

    for batch_index in sorted(results_by_batch):
        batch_has_progress = False
        batch_all_errors = True
        for signature, output_digest, is_error in results_by_batch[batch_index]:
            novel_action = bool(signature) and signature not in seen_signatures
            novel_observation = output_digest not in seen_outputs
            batch_has_progress |= not is_error and (
                novel_action or novel_observation
            )
            batch_all_errors &= is_error
            if signature:
                seen_signatures.add(signature)
            seen_outputs.add(output_digest)
        consecutive_errors = (
            consecutive_errors + 1 if batch_all_errors else 0
        )
        consecutive_no_progress = (
            0 if batch_has_progress else consecutive_no_progress + 1
        )

    reason: str | None = None
    repeated_limit = policy.repeated_call_limit
    if (
        len(call_batch_signatures) >= repeated_limit
        and len(set(call_batch_signatures[-repeated_limit:])) == 1
    ):
        reason = "repeated_tool_call"

    alternating_span = 2 * policy.alternating_cycle_repetitions
    if reason is None and len(call_batch_signatures) >= alternating_span:
        tail = call_batch_signatures[-alternating_span:]
        if (
            len(set(tail[0::2])) == 1
            and len(set(tail[1::2])) == 1
            and tail[0] != tail[1]
        ):
            reason = "alternating_tool_cycle"

    if reason is None and consecutive_errors >= policy.consecutive_error_limit:
        reason = "consecutive_tool_errors"
    if (
        reason is None
        and consecutive_no_progress >= policy.consecutive_no_progress_limit
    ):
        reason = "no_observable_progress"
    if (
        reason is None
        and model_calls >= policy.max_model_calls_without_completion
    ):
        reason = "completion_budget_exhausted"
    if (
        reason is None
        and len(call_signatures) >= policy.max_tool_calls_without_completion
    ):
        reason = "tool_call_budget_exhausted"

    return LoopGuardSnapshot(
        model_calls=model_calls,
        tool_calls=len(call_signatures),
        completed_tool_calls=completed_tool_calls,
        consecutive_errors=consecutive_errors,
        consecutive_no_progress=consecutive_no_progress,
        reason=reason,
        recent_tool_names=tuple(tool_names[-8:]),
    )


class AgentLoopGuardMiddleware(AgentMiddleware[LoopGuardState, Any, Any]):
    state_schema = LoopGuardState

    def __init__(
        self,
        *,
        policy: LoopGuardPolicy,
        completion_schema: type[BaseModel],
        completion_instruction: str,
        audit: AuditSink | None,
        scope: str,
        finalization_tool_names: frozenset[str] = frozenset(),
        clock: Callable[[], float] = time.monotonic,
        activation_deadline: ActivationDeadline | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.completion_tool_names = frozenset({completion_schema.__name__})
        self.completion_instruction = completion_instruction
        self.audit = audit
        self.scope = scope
        self.finalization_tool_names = finalization_tool_names
        self.clock = clock
        self.activation_deadline = activation_deadline

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit is not None:
            self.audit.emit(event, agent_scope=self.scope, **fields)

    def before_model(self, state: LoopGuardState, runtime: Any) -> dict[str, Any] | None:
        del runtime
        if not self.policy.enabled or state.get("guard_forcing_completion", False):
            return None
        now = self.clock()
        shared_elapsed_s = (
            self.activation_deadline.elapsed_s()
            if self.activation_deadline is not None
            else None
        )
        activation_started = state.get("guard_activation_started_monotonic")
        activation_started_now = shared_elapsed_s is None and activation_started is None
        if activation_started is None:
            activation_started = now
        snapshot = analyze_agent_history(
            state.get("messages", []),
            self.policy,
            completion_tool_names=self.completion_tool_names,
        )
        elapsed_s = (
            shared_elapsed_s
            if shared_elapsed_s is not None
            else max(0.0, now - float(activation_started))
        )
        reason = snapshot.reason
        if reason is None and elapsed_s >= self.policy.activation_wall_clock_s:
            reason = "activation_wall_clock_exhausted"
        if reason is None:
            return (
                {"guard_activation_started_monotonic": now}
                if activation_started_now
                else None
            )
        self._audit(
            "agent_stuck_detected",
            **{
                **snapshot.to_dict(),
                "reason": reason,
                "activation_elapsed_s": elapsed_s,
                "activation_wall_clock_s": self.policy.activation_wall_clock_s,
            },
        )
        return {
            "guard_forcing_completion": True,
            "guard_reason": reason,
            "guard_trigger_model_calls": snapshot.model_calls,
        }

    async def abefore_model(
        self, state: LoopGuardState, runtime: Any
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    def _force_completion_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        reason = str(request.state.get("guard_reason", "stuck"))
        trigger_calls = int(request.state.get("guard_trigger_model_calls", 0))
        current_calls = analyze_agent_history(
            request.messages,
            self.policy,
            completion_tool_names=self.completion_tool_names,
        ).model_calls
        attempt = max(1, current_calls - trigger_calls + 1)
        patch_applied = any(
            isinstance(message, ToolMessage)
            and message.name in {"apply_patch", "edit_file", "write_file"}
            and not _tool_result_is_error(message)
            for message in request.messages
        )
        recovery_stage = "test" if patch_applied else "patch"
        required_tool_name = "execute" if patch_applied else "apply_patch"
        recovery_tools = [
            item
            for item in request.tools
            if str(getattr(item, "name", "")) == required_tool_name
            and required_tool_name in self.finalization_tool_names
        ]
        recovery_active = bool(recovery_tools) and (
            attempt <= self.policy.recovery_model_call_limit
        )
        retained_tools = recovery_tools if recovery_active else []
        self._audit(
            "agent_guard_finalization_attempt",
            reason=reason,
            attempt=attempt,
            recovery_active=recovery_active,
            recovery_stage=recovery_stage if recovery_active else None,
            retained_tools=len(retained_tools),
            regular_tools_removed=len(request.tools) - len(retained_tools),
        )
        base_prompt = request.system_message.text if request.system_message else ""
        if recovery_active:
            if recovery_stage == "patch":
                action = (
                    "Call apply_patch now with the best minimal unified diff supported "
                    "by the evidence. Do not call execute or return completion yet."
                )
            else:
                action = (
                    "The patch was applied. Call execute now with one focused "
                    "repository-native test command. Do not run another diagnostic "
                    "python -c probe and do not modify files from the shell."
                )
            forced_prompt = (
                f"{base_prompt}\n\n"
                "RUNTIME RECOVERY DIRECTIVE\n"
                f"The progress guard stopped further exploration because: {reason}. "
                f"Do not perform more diagnosis, searching, or broad reading. {action} "
                f"After recovery, return the required structured completion. "
                f"{self.completion_instruction}"
            ).strip()
        else:
            forced_prompt = (
                f"{base_prompt}\n\n"
                "RUNTIME COMPLETION DIRECTIVE\n"
                f"The progress guard stopped further tool use because: {reason}. "
                f"{self.completion_instruction} Return the required structured "
                "completion now using only evidence already present in the conversation. "
                "Mark the result blocked and list unresolved items when evidence is "
                "insufficient."
            ).strip()
        return request.override(
            tools=retained_tools,
            tool_choice="required" if recovery_active else None,
            system_message=SystemMessage(content=forced_prompt),
        )

    def wrap_model_call(self, request: ModelRequest[Any], handler: Any) -> ModelResponse[Any]:
        if request.state.get("guard_forcing_completion", False):
            request = self._force_completion_request(request)
        return handler(request)

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Any
    ) -> ModelResponse[Any]:
        if request.state.get("guard_forcing_completion", False):
            request = self._force_completion_request(request)
        return await handler(request)

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: LoopGuardState, runtime: Any) -> dict[str, Any] | None:
        del runtime
        if state.get("structured_response") is not None:
            return None
        last_ai_message = next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, AIMessage)
            ),
            None,
        )
        if last_ai_message is None or last_ai_message.tool_calls:
            return None

        snapshot = analyze_agent_history(
            state.get("messages", []),
            self.policy,
            completion_tool_names=self.completion_tool_names,
        )
        already_forcing = state.get("guard_forcing_completion", False)
        reason = str(state.get("guard_reason", "unstructured_model_stop"))
        trigger_calls = int(
            state.get("guard_trigger_model_calls", snapshot.model_calls)
        )
        if not already_forcing:
            self._audit(
                "agent_unstructured_stop_detected",
                model_calls=snapshot.model_calls,
                tool_calls=snapshot.tool_calls,
            )
        return {
            "guard_forcing_completion": True,
            "guard_reason": reason,
            "guard_trigger_model_calls": trigger_calls,
            "jump_to": "model",
        }

    async def aafter_model(
        self, state: LoopGuardState, runtime: Any
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def after_agent(self, state: LoopGuardState, runtime: Any) -> None:
        del runtime
        response = state.get("structured_response")
        if response is None:
            return None
        status = getattr(response, "status", None)
        self._audit(
            "agent_semantic_completion",
            forced=bool(state.get("guard_forcing_completion", False)),
            reason=state.get("guard_reason"),
            status=status,
        )
        return None

    async def aafter_agent(self, state: LoopGuardState, runtime: Any) -> None:
        return self.after_agent(state, runtime)


def require_structured_completion(
    result: dict[str, Any], expected_type: type[BaseModel]
) -> BaseModel:
    value = result.get("structured_response")
    if isinstance(value, expected_type):
        return value
    if isinstance(value, dict):
        return expected_type.model_validate(value)
    raise RuntimeError(
        f"agent stopped without required {expected_type.__name__} structured completion"
    )
