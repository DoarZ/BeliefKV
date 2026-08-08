from __future__ import annotations

import hashlib
import json
import math
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
from langgraph.config import get_config
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from beliefkv.runtime.agent_safety import (
    ActivationDeadline,
    ActivationDeadlineExceeded,
    classify_tool_outcome,
)


class AuditSink(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...


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


class TerminalProtocolError(RuntimeError):
    """Raised when a model stops twice without the required terminal schema."""


@dataclass(frozen=True)
class LoopGuardPolicy:
    enabled: bool = True
    enforce_semantic_guard: bool = False
    enforce_soft_graph_budget: bool = False
    repeated_call_limit: int = 3
    repeated_failed_call_limit: int = 2
    alternating_cycle_repetitions: int = 3
    consecutive_error_limit: int = 3
    consecutive_no_progress_limit: int = 5
    max_model_calls_without_completion: int = 48
    max_tool_calls_without_completion: int = 128
    enforce_call_budgets: bool = False
    recovery_model_call_limit: int = 3
    suppressed_repeat_intent_limit: int = 3
    graph_step_soft_budget: int = 384
    graph_step_lease_size: int = 256
    graph_step_hard_limit: int = 512
    graph_step_reserve: int = 32
    enforce_graph_step_budget: bool = True
    activation_wall_clock_s: float = 7200.0

    def __post_init__(self) -> None:
        values = (
            self.repeated_call_limit,
            self.repeated_failed_call_limit,
            self.alternating_cycle_repetitions,
            self.consecutive_error_limit,
            self.consecutive_no_progress_limit,
            self.max_model_calls_without_completion,
            self.max_tool_calls_without_completion,
            self.recovery_model_call_limit,
            self.suppressed_repeat_intent_limit,
            self.graph_step_soft_budget,
            self.graph_step_lease_size,
            self.graph_step_hard_limit,
            self.graph_step_reserve,
        )
        if min(values) <= 0:
            raise ValueError("loop guard limits must be positive")
        if self.alternating_cycle_repetitions < 2:
            raise ValueError("alternating cycle detection needs at least two repetitions")
        if self.graph_step_soft_budget >= self.graph_step_hard_limit:
            raise ValueError("graph step soft budget must be below the hard limit")
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
    repeated_failed_calls: int
    physical_failure_count: int
    suppressed_repeat_intent_count: int
    consecutive_no_progress: int
    reason: str | None
    recent_tool_names: tuple[str, ...]
    progress_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoopGuardState(AgentState[Any]):
    guard_phase: NotRequired[
        Annotated[
            Literal["NORMAL", "SUSPECT", "RECOVERY", "FINALIZE"],
            UntrackedValue,
            PrivateStateAttr,
        ]
    ]
    guard_forcing_completion: NotRequired[
        Annotated[bool, UntrackedValue, PrivateStateAttr]
    ]
    guard_ever_intervened: NotRequired[
        Annotated[bool, UntrackedValue, PrivateStateAttr]
    ]
    guard_reason: NotRequired[Annotated[str, UntrackedValue, PrivateStateAttr]]
    guard_trigger_model_calls: NotRequired[
        Annotated[int, UntrackedValue, PrivateStateAttr]
    ]
    guard_recovery_attempt: NotRequired[
        Annotated[int, UntrackedValue, PrivateStateAttr]
    ]
    guard_progress_keys: NotRequired[
        Annotated[tuple[str, ...], UntrackedValue, PrivateStateAttr]
    ]
    guard_recovery_baseline_keys: NotRequired[
        Annotated[tuple[str, ...], UntrackedValue, PrivateStateAttr]
    ]
    guard_graph_progress_keys: NotRequired[
        Annotated[tuple[str, ...], UntrackedValue, PrivateStateAttr]
    ]
    guard_graph_lease_until: NotRequired[
        Annotated[int, UntrackedValue, PrivateStateAttr]
    ]
    guard_observed_patterns: NotRequired[
        Annotated[tuple[str, ...], UntrackedValue, PrivateStateAttr]
    ]
    guard_soft_budget_observed: NotRequired[
        Annotated[bool, UntrackedValue, PrivateStateAttr]
    ]
    guard_activation_started_monotonic: NotRequired[
        Annotated[float, UntrackedValue, PrivateStateAttr]
    ]
    protocol_repair_active: NotRequired[
        Annotated[bool, UntrackedValue, PrivateStateAttr]
    ]
    protocol_repair_failed: NotRequired[
        Annotated[bool, UntrackedValue, PrivateStateAttr]
    ]
    protocol_normalized: NotRequired[
        Annotated[bool, UntrackedValue, PrivateStateAttr]
    ]
    protocol_origin_sha256: NotRequired[
        Annotated[str, UntrackedValue, PrivateStateAttr]
    ]
    protocol_origin_chars: NotRequired[
        Annotated[int, UntrackedValue, PrivateStateAttr]
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
    return classify_tool_outcome(message, tool_name=message.name).status == "error"


_CODE_INSPECTION_TOOLS = frozenset({"read_file", "grep", "glob", "ls"})
_CAUSAL_PROGRESS_TOOLS = frozenset({"task", "handoff", "send_message"})
_NON_EVIDENCE_ERROR_CLASSES = frozenset(
    {"command_failed", "duplicate_suppressed", "exception", "timeout", "tool_error"}
)
_SHELL_EVIDENCE_PREFIXES = (
    "git diff",
    "git status",
    "grep ",
    "rg ",
    "sed ",
    "find ",
    "pytest ",
    "python -m pytest",
)
def _tool_metadata(message: ToolMessage) -> dict[str, Any]:
    return dict(message.additional_kwargs or {})


def _credible_progress_key(
    *,
    message: ToolMessage,
    signature: str,
    tool_args: Any,
    output_digest: str,
    is_error: bool,
) -> str | None:
    metadata = _tool_metadata(message)
    if bool(metadata.get("beliefkv_suppressed_repeat_intent", False)):
        return None
    if bool(metadata.get("beliefkv_workspace_changed", False)):
        before = metadata.get("beliefkv_workspace_epoch_before")
        after = metadata.get("beliefkv_workspace_epoch_after")
        return f"workspace:{before}:{after}"
    tool_name = str(message.name or "")
    if not is_error and tool_name in _CAUSAL_PROGRESS_TOOLS:
        return f"causal:{tool_name}:{signature or output_digest}"
    if not is_error and tool_name in _CODE_INSPECTION_TOOLS:
        return f"evidence:{tool_name}:{signature or output_digest}"
    if not is_error and tool_name == "execute":
        command = ""
        if isinstance(tool_args, dict):
            command = str(tool_args.get("command", "")).strip().lower()
        if command.startswith(_SHELL_EVIDENCE_PREFIXES):
            return f"shell-evidence:{signature or output_digest}"
    if is_error:
        error_class = str(metadata.get("beliefkv_error_class") or "")
        if not error_class:
            outcome = classify_tool_outcome(message, tool_name=tool_name)
            error_class = str(outcome.error_class or "")
        if error_class and error_class not in _NON_EVIDENCE_ERROR_CLASSES:
            return f"error-evidence:{error_class}:{output_digest}"
    return None


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
    calls_by_id: dict[str, tuple[str, str, int, Any]] = {}
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
                calls_by_id[call_id] = (
                    signature,
                    name,
                    batch_index,
                    call.get("args", {}),
                )
        if batch_signatures:
            call_batch_signatures.append(tuple(sorted(batch_signatures)))

    seen_signatures: set[str] = set()
    seen_outputs: set[str] = set()
    seen_progress_keys: set[str] = set()
    completed_tool_calls = 0
    consecutive_errors = 0
    repeated_failed_calls = 0
    physical_failure_count = 0
    suppressed_repeat_intent_count = 0
    consecutive_suppressed_repeat_intents = 0
    previous_failed_signature: tuple[str, ...] | None = None
    consecutive_no_progress = 0
    progress_keys: set[str] = set()
    results_by_batch: dict[
        int, list[tuple[str, str, bool, bool, bool, str | None, str | None]]
    ] = {}
    unmatched_batch_index = len(call_batch_signatures)
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name in completion_tool_names:
            continue
        call = calls_by_id.get(str(message.tool_call_id))
        signature = call[0] if call is not None else ""
        batch_index = call[2] if call is not None else unmatched_batch_index
        tool_args = call[3] if call is not None else {}
        output_digest = hashlib.sha256(
            _message_text(message).encode("utf-8", errors="replace")
        ).hexdigest()
        is_error = _tool_result_is_error(message)
        metadata = _tool_metadata(message)
        suppressed_intent = bool(
            metadata.get("beliefkv_suppressed_repeat_intent", False)
            or metadata.get("beliefkv_error_class") == "duplicate_suppressed"
        )
        physical_execution = bool(
            metadata.get("beliefkv_physical_execution", not suppressed_intent)
        )
        failure_episode_id = metadata.get("beliefkv_failure_episode_id")
        progress_key = _credible_progress_key(
            message=message,
            signature=signature,
            tool_args=tool_args,
            output_digest=output_digest,
            is_error=is_error,
        )
        if progress_key is not None:
            progress_keys.add(progress_key)
        completed_tool_calls += 1
        results_by_batch.setdefault(batch_index, []).append(
            (
                signature,
                output_digest,
                is_error,
                physical_execution,
                suppressed_intent,
                str(failure_episode_id) if failure_episode_id else None,
                progress_key,
            )
        )

    for batch_index in sorted(results_by_batch):
        batch_has_progress = False
        batch_all_errors = True
        batch_physical_error_signatures: list[str] = []
        batch_suppressed_intents = 0
        for (
            signature,
            output_digest,
            is_error,
            physical_execution,
            suppressed_intent,
            _failure_episode_id,
            progress_key,
        ) in results_by_batch[batch_index]:
            batch_has_progress |= bool(
                progress_key and progress_key not in seen_progress_keys
            )
            batch_all_errors &= is_error
            if is_error and physical_execution:
                physical_failure_count += 1
                if signature:
                    batch_physical_error_signatures.append(signature)
            if suppressed_intent:
                suppressed_repeat_intent_count += 1
                batch_suppressed_intents += 1
            if signature:
                seen_signatures.add(signature)
            seen_outputs.add(output_digest)
            if progress_key:
                seen_progress_keys.add(progress_key)
        consecutive_suppressed_repeat_intents = (
            consecutive_suppressed_repeat_intents + batch_suppressed_intents
            if batch_suppressed_intents
            else 0
        )
        consecutive_errors = (
            consecutive_errors + 1 if batch_all_errors else 0
        )
        batch_signature = (
            call_batch_signatures[batch_index]
            if batch_index < len(call_batch_signatures)
            else ()
        )
        physical_error_signature = tuple(sorted(batch_physical_error_signatures))
        if physical_error_signature:
            repeated_failed_calls = (
                repeated_failed_calls + 1
                if physical_error_signature == previous_failed_signature
                else 1
            )
            previous_failed_signature = physical_error_signature
        elif batch_suppressed_intents:
            # A suppressed retry is a repeated model intent, not another physical
            # failure. Preserve the prior physical episode without incrementing it.
            pass
        else:
            repeated_failed_calls = 0
            previous_failed_signature = None
        consecutive_no_progress = (
            0 if batch_has_progress else consecutive_no_progress + 1
        )

    reason: str | None = None
    if repeated_failed_calls >= policy.repeated_failed_call_limit:
        reason = "repeated_failed_tool_call"
    if (
        reason is None
        and consecutive_suppressed_repeat_intents
        >= policy.suppressed_repeat_intent_limit
    ):
        reason = "repeated_suppressed_tool_intent"
    repeated_limit = policy.repeated_call_limit
    if reason is None and (
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
        and policy.enforce_call_budgets
        and model_calls >= policy.max_model_calls_without_completion
    ):
        reason = "completion_budget_exhausted"
    if (
        reason is None
        and policy.enforce_call_budgets
        and len(call_signatures) >= policy.max_tool_calls_without_completion
    ):
        reason = "tool_call_budget_exhausted"

    return LoopGuardSnapshot(
        model_calls=model_calls,
        tool_calls=len(call_signatures),
        completed_tool_calls=completed_tool_calls,
        consecutive_errors=consecutive_errors,
        repeated_failed_calls=repeated_failed_calls,
        physical_failure_count=physical_failure_count,
        suppressed_repeat_intent_count=suppressed_repeat_intent_count,
        consecutive_no_progress=consecutive_no_progress,
        reason=reason,
        recent_tool_names=tuple(tool_names[-8:]),
        progress_keys=tuple(sorted(progress_keys)),
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
        self.completion_schema = completion_schema
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

    @staticmethod
    def _is_safety_finalization(state: LoopGuardState) -> bool:
        return str(state.get("guard_reason", "")) in {
            "graph_step_hard_limit_low",
            "activation_wall_clock_exhausted",
        }

    def _graph_budget(self) -> tuple[int, int, int, int] | None:
        if not self.policy.enforce_graph_step_budget:
            return None
        try:
            config = get_config()
        except RuntimeError:
            return None
        metadata = config.get("metadata", {}) or {}
        try:
            step = int(metadata["langgraph_step"])
            limit = int(config["recursion_limit"])
        except (KeyError, TypeError, ValueError):
            return None
        hard_limit = min(limit, self.policy.graph_step_hard_limit)
        reserve = min(
            self.policy.graph_step_reserve,
            max(4, hard_limit // 8),
        )
        return step, limit, hard_limit, reserve

    @staticmethod
    def _merged_progress_keys(
        state: LoopGuardState,
        snapshot: LoopGuardSnapshot,
    ) -> tuple[str, ...]:
        keys = set(state.get("guard_progress_keys", ()))
        keys.update(snapshot.progress_keys)
        return tuple(sorted(keys))

    def _recovered_update(
        self,
        *,
        state: LoopGuardState,
        progress_keys: tuple[str, ...],
        snapshot: LoopGuardSnapshot,
    ) -> dict[str, Any]:
        prior_phase = str(state.get("guard_phase", "NORMAL"))
        self._audit(
            "agent_guard_recovered",
            prior_phase=prior_phase,
            reason=state.get("guard_reason"),
            recovery_attempt=int(state.get("guard_recovery_attempt", 0)),
            model_calls=snapshot.model_calls,
            tool_calls=snapshot.tool_calls,
            new_progress_keys=len(
                set(progress_keys)
                - set(state.get("guard_recovery_baseline_keys", ()))
            ),
        )
        return {
            "guard_phase": "NORMAL",
            "guard_forcing_completion": False,
            "guard_reason": "",
            "guard_recovery_attempt": 0,
            "guard_progress_keys": progress_keys,
            "guard_recovery_baseline_keys": progress_keys,
        }

    def before_model(self, state: LoopGuardState, runtime: Any) -> dict[str, Any] | None:
        del runtime
        if not self.policy.enabled or state.get("protocol_repair_active", False):
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
        progress_keys = self._merged_progress_keys(state, snapshot)
        elapsed_s = (
            shared_elapsed_s
            if shared_elapsed_s is not None
            else max(0.0, now - float(activation_started))
        )
        graph_budget = self._graph_budget()
        graph_fields = {}
        hard_stop_reason: str | None = None
        if graph_budget is not None:
            graph_step, graph_limit, graph_hard_limit, graph_reserve = graph_budget
            graph_fields = {
                "graph_step": graph_step,
                "graph_recursion_limit": graph_limit,
                "graph_step_hard_limit": graph_hard_limit,
                "graph_step_reserve": graph_reserve,
            }
            if graph_hard_limit - graph_step <= graph_reserve:
                hard_stop_reason = "graph_step_hard_limit_low"
        if elapsed_s >= self.policy.activation_wall_clock_s:
            hard_stop_reason = "activation_wall_clock_exhausted"
        if hard_stop_reason is not None:
            event = (
                "agent_graph_budget_finalization"
                if hard_stop_reason == "graph_step_hard_limit_low"
                else "agent_stuck_detected"
            )
            self._audit(
                event,
                **{
                    **snapshot.to_dict(),
                    "reason": hard_stop_reason,
                    "activation_elapsed_s": elapsed_s,
                    "activation_wall_clock_s": self.policy.activation_wall_clock_s,
                    **graph_fields,
                },
            )
            return {
                "guard_phase": "FINALIZE",
                "guard_forcing_completion": True,
                "guard_ever_intervened": True,
                "guard_reason": hard_stop_reason,
                "guard_trigger_model_calls": snapshot.model_calls,
                "guard_recovery_attempt": self.policy.recovery_model_call_limit + 1,
                "guard_progress_keys": progress_keys,
                "guard_recovery_baseline_keys": progress_keys,
            }

        phase = str(state.get("guard_phase", "NORMAL"))
        if (
            not self.policy.enforce_semantic_guard
            and phase in {"SUSPECT", "RECOVERY", "FINALIZE"}
            and not self._is_safety_finalization(state)
        ):
            self._audit(
                "agent_guard_legacy_state_cleared",
                prior_phase=phase,
                prior_reason=state.get("guard_reason"),
            )
            return {
                "guard_phase": "NORMAL",
                "guard_forcing_completion": False,
                "guard_reason": "",
                "guard_recovery_attempt": 0,
                "guard_progress_keys": progress_keys,
                "guard_recovery_baseline_keys": progress_keys,
                **(
                    {"guard_activation_started_monotonic": now}
                    if activation_started_now
                    else {}
                ),
            }
        if phase == "NORMAL" and state.get("guard_forcing_completion", False):
            phase = "RECOVERY"
        if phase in {"SUSPECT", "RECOVERY"}:
            baseline = set(state.get("guard_recovery_baseline_keys", ()))
            if set(progress_keys) - baseline:
                return self._recovered_update(
                    state=state,
                    progress_keys=progress_keys,
                    snapshot=snapshot,
                )
            if phase == "SUSPECT":
                self._audit(
                    "agent_stuck_detected",
                    **{
                        **snapshot.to_dict(),
                        "reason": state.get("guard_reason"),
                        "activation_elapsed_s": elapsed_s,
                        "activation_wall_clock_s": self.policy.activation_wall_clock_s,
                        **graph_fields,
                    },
                )
                return {
                    "guard_phase": "RECOVERY",
                    "guard_forcing_completion": True,
                    "guard_ever_intervened": True,
                    "guard_recovery_attempt": 1,
                    "guard_progress_keys": progress_keys,
                }
            attempt = int(state.get("guard_recovery_attempt", 0)) + 1
            finalizing = attempt > self.policy.recovery_model_call_limit
            return {
                "guard_phase": "FINALIZE" if finalizing else "RECOVERY",
                "guard_forcing_completion": True,
                "guard_recovery_attempt": attempt,
                "guard_progress_keys": progress_keys,
            }
        if phase == "FINALIZE":
            return None

        update: dict[str, Any] = {}
        if tuple(state.get("guard_progress_keys", ())) != progress_keys:
            update["guard_progress_keys"] = progress_keys
        if activation_started_now:
            update["guard_activation_started_monotonic"] = now

        reason = snapshot.reason if self.policy.enforce_semantic_guard else None
        observed_patterns = set(state.get("guard_observed_patterns", ()))
        if snapshot.reason is not None and snapshot.reason not in observed_patterns:
            self._audit(
                "agent_guard_pattern_observed",
                **{
                    **snapshot.to_dict(),
                    "observed_pattern": snapshot.reason,
                    "enforced": self.policy.enforce_semantic_guard,
                    "activation_elapsed_s": elapsed_s,
                    **graph_fields,
                },
            )
            observed_patterns.add(snapshot.reason)
            update["guard_observed_patterns"] = tuple(sorted(observed_patterns))
        if graph_budget is not None and self.policy.enforce_soft_graph_budget:
            graph_step, _graph_limit, graph_hard_limit, graph_reserve = graph_budget
            lease_until = int(
                state.get("guard_graph_lease_until", self.policy.graph_step_soft_budget)
            )
            graph_progress_baseline = set(
                state.get("guard_graph_progress_keys", ())
            )
            if "guard_graph_progress_keys" not in state:
                update["guard_graph_progress_keys"] = progress_keys
            update["guard_graph_lease_until"] = lease_until
            if graph_step >= lease_until:
                has_progress = bool(set(progress_keys) - graph_progress_baseline)
                maximum_lease = graph_hard_limit - graph_reserve
                if has_progress and lease_until < maximum_lease:
                    extended_until = min(
                        maximum_lease,
                        max(lease_until, graph_step) + self.policy.graph_step_lease_size,
                    )
                    update["guard_graph_lease_until"] = extended_until
                    update["guard_graph_progress_keys"] = progress_keys
                    self._audit(
                        "agent_graph_step_lease_extended",
                        graph_step=graph_step,
                        previous_lease_until=lease_until,
                        lease_until=extended_until,
                        new_progress_keys=len(
                            set(progress_keys) - graph_progress_baseline
                        ),
                    )
                elif reason is None:
                    reason = "graph_soft_budget_without_progress"
        elif (
            graph_budget is not None
            and graph_budget[0] >= self.policy.graph_step_soft_budget
            and not state.get("guard_soft_budget_observed", False)
        ):
            self._audit(
                "agent_graph_step_soft_budget_observed",
                graph_step=graph_budget[0],
                graph_recursion_limit=graph_budget[1],
                graph_step_soft_budget=self.policy.graph_step_soft_budget,
                enforced=False,
            )
            update["guard_soft_budget_observed"] = True

        if reason is None:
            return update or None

        graph_fields = (
            {
                "graph_step": graph_budget[0],
                "graph_recursion_limit": graph_budget[1],
                "graph_step_hard_limit": graph_budget[2],
                "graph_step_reserve": graph_budget[3],
            }
            if graph_budget is not None
            else {}
        )
        self._audit(
            "agent_guard_suspected",
            **{
                **snapshot.to_dict(),
                "reason": reason,
                "activation_elapsed_s": elapsed_s,
                "activation_wall_clock_s": self.policy.activation_wall_clock_s,
                **graph_fields,
            },
        )
        update.update({
            "guard_phase": "SUSPECT",
            "guard_forcing_completion": False,
            "guard_reason": reason,
            "guard_trigger_model_calls": snapshot.model_calls,
            "guard_recovery_attempt": 0,
            "guard_recovery_baseline_keys": progress_keys,
        })
        return update

    async def abefore_model(
        self, state: LoopGuardState, runtime: Any
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    def _guard_recovery_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        reason = str(request.state.get("guard_reason", "stuck"))
        attempt = max(1, int(request.state.get("guard_recovery_attempt", 1)))
        phase = str(request.state.get("guard_phase", "RECOVERY"))
        if "guard_phase" not in request.state and reason in {
            "graph_step_budget_low",
            "graph_step_hard_limit_low",
            "activation_wall_clock_exhausted",
        }:
            phase = "FINALIZE"
        recovery_active = (
            phase == "RECOVERY" and attempt <= self.policy.recovery_model_call_limit
        )
        retained_tools = (
            list(request.tools)
            if recovery_active
            else [
                item
                for item in request.tools
                if str(getattr(item, "name", "")) in self.completion_tool_names
            ]
        )
        self._audit(
            "agent_guard_finalization_attempt",
            reason=reason,
            attempt=attempt,
            recovery_active=recovery_active,
            failed_recovery=False,
            recovery_stage="alternate_action" if recovery_active else None,
            retained_tools=len(retained_tools),
            regular_tools_removed=len(request.tools) - len(retained_tools),
        )
        base_prompt = request.system_message.text if request.system_message else ""
        if recovery_active:
            forced_prompt = (
                f"{base_prompt}\n\n"
                "RUNTIME RECOVERY DIRECTIVE\n"
                f"The progress guard detected a non-progressing action pattern: {reason}. "
                "Do not repeat the same tool call with the same arguments and unchanged "
                "workspace. Choose a materially different action that can change the "
                "workspace or add new evidence, hand off when appropriate, or return an "
                "honest terminal result using the evidence already available. "
                f"{self.completion_instruction}"
            ).strip()
        else:
            forced_prompt = (
                f"{base_prompt}\n\n"
                "RUNTIME COMPLETION DIRECTIVE\n"
                f"The bounded recovery opportunity ended because: {reason}. Further "
                "tool use is disabled. Return the required structured completion using "
                "only evidence already present. Select the terminal status honestly; do "
                "not claim success when requirements remain unresolved. "
                f"{self.completion_instruction}"
            ).strip()
        return request.override(
            tools=retained_tools,
            tool_choice=None,
            system_message=SystemMessage(content=forced_prompt),
        )

    def _format_repair_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        base_prompt = request.system_message.text if request.system_message else ""
        repair_prompt = (
            f"{base_prompt}\n\n"
            "RUNTIME FORMAT-ONLY REPAIR\n"
            "Your previous response stopped without the required structured terminal "
            "object. Restate that response in the required schema without adding new "
            "claims, changing its semantic outcome, or calling tools. Preserve uncertainty "
            "and failed checks exactly. "
            f"{self.completion_instruction}"
        ).strip()
        self._audit(
            "agent_protocol_repair_attempt",
            origin_sha256=request.state.get("protocol_origin_sha256"),
            origin_chars=request.state.get("protocol_origin_chars"),
        )
        return request.override(
            tools=[],
            tool_choice=None,
            system_message=SystemMessage(content=repair_prompt),
        )

    def _normalize_json_terminal(self, message: AIMessage) -> BaseModel | None:
        text = _message_text(message).strip()
        candidates = [text]
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                candidates.append("\n".join(lines[1:-1]).strip())
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            try:
                return self.completion_schema.model_validate(payload)
            except (TypeError, ValueError):
                continue
        return None

    def wrap_model_call(self, request: ModelRequest[Any], handler: Any) -> ModelResponse[Any]:
        if request.state.get("protocol_repair_active", False):
            request = self._format_repair_request(request)
        elif request.state.get("guard_forcing_completion", False) and (
            self.policy.enforce_semantic_guard
            or self._is_safety_finalization(request.state)
        ):
            request = self._guard_recovery_request(request)
        return handler(request)

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Any
    ) -> ModelResponse[Any]:
        if request.state.get("protocol_repair_active", False):
            request = self._format_repair_request(request)
        elif request.state.get("guard_forcing_completion", False) and (
            self.policy.enforce_semantic_guard
            or self._is_safety_finalization(request.state)
        ):
            request = self._guard_recovery_request(request)
        return await handler(request)

    @hook_config(can_jump_to=["model", "end"])
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

        normalized = self._normalize_json_terminal(last_ai_message)
        if normalized is not None:
            text = _message_text(last_ai_message)
            self._audit(
                "agent_protocol_normalized",
                origin_sha256=hashlib.sha256(
                    text.encode("utf-8", errors="replace")
                ).hexdigest(),
                origin_chars=len(text),
            )
            return {
                "structured_response": normalized,
                "protocol_normalized": True,
                "jump_to": "end",
            }

        snapshot = analyze_agent_history(
            state.get("messages", []),
            self.policy,
            completion_tool_names=self.completion_tool_names,
        )
        if state.get("protocol_repair_active", False):
            failed_text = _message_text(last_ai_message)
            self._audit(
                "agent_protocol_repair_failed",
                origin_sha256=state.get("protocol_origin_sha256"),
                origin_chars=state.get("protocol_origin_chars"),
                repair_output_sha256=hashlib.sha256(
                    failed_text.encode("utf-8", errors="replace")
                ).hexdigest(),
                repair_output_chars=len(failed_text),
                repair_output_preview=failed_text[:4096],
                repair_output_truncated=len(failed_text) > 4096,
                model_calls=snapshot.model_calls,
                tool_calls=snapshot.tool_calls,
            )
            return {
                "protocol_repair_failed": True,
                "jump_to": "end",
            }

        text = _message_text(last_ai_message)
        origin_sha256 = hashlib.sha256(
            text.encode("utf-8", errors="replace")
        ).hexdigest()
        self._audit(
            "agent_unstructured_stop_detected",
            model_calls=snapshot.model_calls,
            tool_calls=snapshot.tool_calls,
            origin_sha256=origin_sha256,
            origin_chars=len(text),
            origin_preview=text[:4096],
            origin_truncated=len(text) > 4096,
            guard_intervened=bool(
                state.get("guard_ever_intervened", False)
                or state.get("guard_forcing_completion", False)
            ),
        )
        return {
            "protocol_repair_active": True,
            "protocol_origin_sha256": origin_sha256,
            "protocol_origin_chars": len(text),
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
        protocol_repaired = bool(state.get("protocol_repair_active", False))
        protocol_normalized = bool(state.get("protocol_normalized", False))
        guard_intervened = bool(
            state.get("guard_ever_intervened", False)
            or state.get("guard_forcing_completion", False)
        )
        if protocol_repaired:
            if isinstance(response, BaseModel):
                repaired_payload = response.model_dump(mode="json")
            else:
                repaired_payload = response
            repaired_json = json.dumps(
                repaired_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            )
            self._audit(
                "agent_protocol_repaired",
                origin_sha256=state.get("protocol_origin_sha256"),
                origin_chars=state.get("protocol_origin_chars"),
                repaired_sha256=hashlib.sha256(
                    repaired_json.encode("utf-8")
                ).hexdigest(),
            )
        self._audit(
            "agent_semantic_completion",
            forced=False,
            guard_intervened=guard_intervened,
            protocol_repaired=protocol_repaired,
            protocol_normalized=protocol_normalized,
            reason=state.get("guard_reason"),
            status=getattr(response, "status", None),
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
    if result.get("protocol_repair_failed", False):
        raise TerminalProtocolError(
            f"agent stopped twice without required {expected_type.__name__} schema"
        )
    raise RuntimeError(
        f"agent stopped without required {expected_type.__name__} structured completion"
    )
