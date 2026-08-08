from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Mapping

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from beliefkv.runtime.agent_safety import classify_tool_outcome


@dataclass
class _CircuitRecord:
    execution_count: int
    outcome_status: str
    error_class: str | None
    failure_episode_id: str | None


@dataclass(frozen=True)
class ToolObservationBudgetPolicy:
    """Bound model-visible tool observations without suppressing tool execution."""

    total_chars_per_turn: int = 65_536
    max_chars_per_result: int = 16_384

    def __post_init__(self) -> None:
        if self.total_chars_per_turn <= 0 or self.max_chars_per_result <= 0:
            raise ValueError("tool observation budgets must be positive")
        if self.max_chars_per_result > self.total_chars_per_turn:
            raise ValueError(
                "per-result tool observation budget exceeds the per-turn budget"
            )


class ToolObservationBudgetMiddleware(AgentMiddleware[Any, Any, Any]):
    """Deterministically divide one AI turn's observation budget across its tools."""

    def __init__(
        self,
        *,
        policy: ToolObservationBudgetPolicy,
        audit: Any | None,
        scope: str,
    ) -> None:
        super().__init__()
        self._policy = policy
        self._audit = audit
        self._scope = scope

    def _emit(self, event: str, **fields: Any) -> None:
        if self._audit is not None:
            self._audit.emit(event, agent_scope=self._scope, **fields)

    @staticmethod
    def _turn_shape(request: ToolCallRequest) -> tuple[str | None, int]:
        state = request.state if isinstance(request.state, dict) else {}
        call_id = str(request.tool_call.get("id", ""))
        for message in reversed(state.get("messages", [])):
            if not isinstance(message, AIMessage):
                continue
            calls = list(message.tool_calls)
            if call_id and not any(
                str(item.get("id", "")) == call_id for item in calls
            ):
                continue
            return message.id, max(1, len(calls))
        return None, 1

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )

    @staticmethod
    def _bounded_text(text: str, *, budget: int, digest: str) -> str:
        if len(text) <= budget:
            return text
        marker = (
            "\n... [BeliefKV observation truncated; "
            f"original_chars={len(text)} sha256={digest}] ...\n"
        )
        if len(marker) >= budget:
            return marker[:budget]
        payload_budget = budget - len(marker)
        head_chars = (payload_budget * 2) // 3
        tail_chars = payload_budget - head_chars
        tail = text[-tail_chars:] if tail_chars else ""
        return text[:head_chars] + marker + tail

    def _bound(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        if not isinstance(result, ToolMessage):
            return result
        turn_id, fanout = self._turn_shape(request)
        result_budget = min(
            self._policy.max_chars_per_result,
            max(1, self._policy.total_chars_per_turn // fanout),
        )
        text = self._content_text(result.content)
        if len(text) <= result_budget:
            return result
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        bounded = self._bounded_text(text, budget=result_budget, digest=digest)
        tool_name = result.name or str(request.tool_call.get("name", ""))
        self._emit(
            "agent_tool_observation_truncated",
            tool_name=tool_name,
            tool_call_id=str(request.tool_call.get("id", "")),
            turn_message_id=turn_id,
            turn_tool_fanout=fanout,
            per_result_budget_chars=result_budget,
            per_turn_budget_chars=self._policy.total_chars_per_turn,
            original_chars=len(text),
            visible_chars=len(bounded),
            original_sha256=digest,
        )
        return result.model_copy(
            update={
                "content": bounded,
                "additional_kwargs": {
                    **result.additional_kwargs,
                    "beliefkv_observation_truncated": True,
                    "beliefkv_original_chars": len(text),
                    "beliefkv_original_sha256": digest,
                    "beliefkv_turn_tool_fanout": fanout,
                },
            }
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._bound(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        return self._bound(request, await handler(request))


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


class ToolCircuitBreakerMiddleware(AgentMiddleware[Any, Any, Any]):
    """Suppress repeated failed or observably inert tool work."""

    _COMMAND_SUCCESS_ONLY = re.compile(
        r"^\s*\[Command succeeded with exit code 0\]\s*$"
    )

    def __init__(
        self,
        *,
        state_epoch: Callable[[], int],
        audit: Any | None,
        scope: str,
        transient_retry_limit: int = 1,
        no_effect_execution_limit: int = 2,
        max_records: int = 2048,
        excluded_tools: frozenset[str] = frozenset({"task"}),
        censor_observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()
        if transient_retry_limit < 0:
            raise ValueError("transient retry limit cannot be negative")
        if no_effect_execution_limit <= 0:
            raise ValueError("no-effect execution limit must be positive")
        if max_records <= 0:
            raise ValueError("circuit record capacity must be positive")
        self._state_epoch = state_epoch
        self._audit = audit
        self._scope = scope
        self._transient_retry_limit = transient_retry_limit
        self._no_effect_execution_limit = no_effect_execution_limit
        self._max_records = max_records
        self._excluded_tools = excluded_tools
        self._censor_observer = censor_observer
        self._records: OrderedDict[tuple[str, str, int], _CircuitRecord] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def _emit(self, event: str, **fields: Any) -> None:
        if self._audit is not None:
            self._audit.emit(event, agent_scope=self._scope, **fields)

    @staticmethod
    def _suppressed_result(
        request: ToolCallRequest,
        record: _CircuitRecord,
        *,
        signature: str,
        epoch: int,
    ) -> ToolMessage:
        tool_name = str(request.tool_call.get("name", ""))
        no_effect = record.outcome_status == "success_no_effect"
        previous = (
            "completed successfully but produced no output or workspace change"
            if no_effect
            else f"failed ({record.error_class or 'tool_error'})"
        )
        detail = (
            "this identical tool call was already executed in workspace epoch "
            f"{epoch} and {previous}"
        )
        return ToolMessage(
            content=(
                f"Runtime duplicate circuit breaker: {detail}. The "
                "previous observation remains in the conversation. Change the arguments "
                "or choose a different action; do not retry unchanged. "
                f"[duplicate_suppressed signature={signature[:16]}]"
            ),
            tool_call_id=str(request.tool_call.get("id", "")),
            name=tool_name,
            status="error",
            additional_kwargs={
                "beliefkv_error_class": "duplicate_suppressed",
                "beliefkv_workspace_epoch": epoch,
                "beliefkv_tool_signature": signature,
                "beliefkv_physical_execution": False,
                "beliefkv_suppressed_repeat_intent": True,
                "beliefkv_failure_episode_id": record.failure_episode_id,
                "beliefkv_duplicate_reason": (
                    "successful_no_effect" if no_effect else "failed_call"
                ),
            },
        )

    def _should_suppress(self, record: _CircuitRecord) -> bool:
        if record.outcome_status == "inflight":
            return False
        if record.outcome_status == "success":
            return False
        if record.outcome_status == "success_no_effect":
            return record.execution_count >= self._no_effect_execution_limit
        if record.error_class in {"timeout", "exception"}:
            return record.execution_count > self._transient_retry_limit
        return True

    @classmethod
    def _success_has_no_observable_effect(
        cls,
        result: ToolMessage,
        *,
        workspace_changed: bool,
    ) -> bool:
        if workspace_changed:
            return False
        artifact = getattr(result, "artifact", None)
        if artifact not in (None, "", (), [], {}):
            return False
        content = result.content
        if content is None:
            return True
        if isinstance(content, str):
            return not content.strip() or bool(
                cls._COMMAND_SUCCESS_ONLY.fullmatch(content)
            )
        if isinstance(content, (list, tuple, dict)):
            return len(content) == 0
        return False

    @staticmethod
    def _conversation_key(request: ToolCallRequest) -> str:
        config = getattr(request.runtime, "config", {}) or {}
        configurable = config.get("configurable", {}) or {}
        metadata = config.get("metadata", {}) or {}
        components: list[str] = []
        for key in (
            "thread_id",
            "beliefkv_invocation_id",
        ):
            value = configurable.get(key) or metadata.get(key)
            if value:
                components.append(f"config:{key}:{value}")
        state = request.state if isinstance(request.state, dict) else {}
        messages = state.get("messages", [])
        if messages:
            first = messages[0]
            message_id = getattr(first, "id", None)
            if message_id:
                components.append(f"message:{message_id}")
            else:
                content = getattr(first, "content", first)
                encoded = json.dumps(
                    content,
                    sort_keys=True,
                    ensure_ascii=True,
                    default=str,
                ).encode("utf-8")
                components.append(
                    "content:" + hashlib.sha256(encoded).hexdigest()
                )
        return "|".join(components) or "scope-local"

    def _before_call(
        self, request: ToolCallRequest
    ) -> tuple[
        str,
        tuple[str, str, int] | None,
        _CircuitRecord | None,
        ToolMessage | None,
    ]:
        tool_name = str(request.tool_call.get("name", ""))
        signature = _canonical_tool_call(request.tool_call)
        epoch = int(self._state_epoch())
        conversation_key = self._conversation_key(request)
        if tool_name in self._excluded_tools:
            return signature, None, None, None
        key = (conversation_key, signature, epoch)
        with self._lock:
            record = self._records.get(key)
            if record is not None:
                self._records.move_to_end(key)
            if record is None or not self._should_suppress(record):
                reservation = _CircuitRecord(
                    execution_count=(record.execution_count + 1 if record else 1),
                    outcome_status="inflight",
                    error_class=None,
                    failure_episode_id=None,
                )
                self._records[key] = reservation
                self._records.move_to_end(key)
                while len(self._records) > self._max_records:
                    self._records.popitem(last=False)
                return signature, key, reservation, None
        assert record is not None
        self._emit(
            "agent_tool_duplicate_suppressed",
            tool_name=tool_name,
            signature=signature,
            workspace_epoch=epoch,
            previous_status=record.outcome_status,
            previous_error_class=record.error_class,
            previous_execution_count=record.execution_count,
            duplicate_reason=(
                "successful_no_effect"
                if record.outcome_status == "success_no_effect"
                else "failed_call"
            ),
        )
        if self._censor_observer is not None:
            config = getattr(request.runtime, "config", {}) or {}
            configurable = config.get("configurable", {}) or {}
            metadata = config.get("metadata", {}) or {}
            self._censor_observer(
                {
                    "call_kind": "tool",
                    "censor_reason": "duplicate_suppressed",
                    "tool_call_id": str(request.tool_call.get("id", "")),
                    "tool_name": tool_name,
                    "parameter_signature": signature,
                    "invocation_id": (
                        configurable.get("beliefkv_invocation_id")
                        or metadata.get("beliefkv_invocation_id")
                    ),
                    "workspace_epoch": epoch,
                    "previous_status": record.outcome_status,
                    "previous_error_class": record.error_class,
                    "duplicate_reason": (
                        "successful_no_effect"
                        if record.outcome_status == "success_no_effect"
                        else "failed_call"
                    ),
                    "physical_execution": False,
                    "suppressed_repeat_intent": True,
                    "failure_episode_id": record.failure_episode_id,
                }
            )
        return (
            signature,
            None,
            None,
            self._suppressed_result(
                request,
                record,
                signature=signature,
                epoch=epoch,
            ),
        )

    def _after_call(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
        *,
        key: tuple[str, str, int] | None,
        reservation: _CircuitRecord | None,
    ) -> ToolMessage | Command[Any]:
        if key is None or reservation is None:
            return result
        if not isinstance(result, ToolMessage):
            with self._lock:
                if self._records.get(key) is reservation:
                    self._records.pop(key, None)
            return result
        tool_name = result.name or str(request.tool_call.get("name", ""))
        if tool_name in self._excluded_tools:
            return result
        outcome = classify_tool_outcome(result, tool_name=tool_name)
        workspace_epoch_after = int(self._state_epoch())
        workspace_changed = workspace_epoch_after > key[2]
        no_observable_effect = (
            outcome.status == "success"
            and self._success_has_no_observable_effect(
                result,
                workspace_changed=workspace_changed,
            )
        )
        failure_episode_id = None
        if outcome.status == "error":
            failure_material = "\0".join(
                (
                    key[0],
                    key[1],
                    str(key[2]),
                    str(reservation.execution_count),
                )
            )
            failure_episode_id = hashlib.sha256(
                failure_material.encode("utf-8", errors="replace")
            ).hexdigest()
        with self._lock:
            if self._records.get(key) is reservation:
                if outcome.status == "success":
                    if no_observable_effect:
                        self._records[key] = _CircuitRecord(
                            execution_count=reservation.execution_count,
                            outcome_status="success_no_effect",
                            error_class="no_observable_effect",
                            failure_episode_id=None,
                        )
                        self._records.move_to_end(key)
                    else:
                        self._records.pop(key, None)
                else:
                    self._records[key] = _CircuitRecord(
                        execution_count=reservation.execution_count,
                        outcome_status=outcome.status,
                        error_class=outcome.error_class,
                        failure_episode_id=failure_episode_id,
                    )
                    self._records.move_to_end(key)
        return result.model_copy(
            update={
                "additional_kwargs": {
                    **(result.additional_kwargs or {}),
                    "beliefkv_error_class": outcome.error_class,
                    "beliefkv_tool_signature": key[1],
                    "beliefkv_physical_execution": True,
                    "beliefkv_suppressed_repeat_intent": False,
                    "beliefkv_failure_episode_id": failure_episode_id,
                    "beliefkv_workspace_epoch_before": key[2],
                    "beliefkv_workspace_epoch_after": workspace_epoch_after,
                    "beliefkv_workspace_changed": workspace_changed,
                    "beliefkv_no_observable_effect": no_observable_effect,
                }
            }
        )

    def _after_exception(
        self,
        *,
        key: tuple[str, str, int] | None,
        reservation: _CircuitRecord | None,
    ) -> None:
        if key is None or reservation is None:
            return
        with self._lock:
            if self._records.get(key) is reservation:
                self._records[key] = _CircuitRecord(
                    execution_count=reservation.execution_count,
                    outcome_status="error",
                    error_class="exception",
                    failure_episode_id=hashlib.sha256(
                        "\0".join(
                            (
                                key[0],
                                key[1],
                                str(key[2]),
                                str(reservation.execution_count),
                            )
                        ).encode("utf-8", errors="replace")
                    ).hexdigest(),
                )
                self._records.move_to_end(key)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        _signature, key, reservation, suppressed = self._before_call(request)
        if suppressed is not None:
            return suppressed
        try:
            result = handler(request)
        except BaseException:
            self._after_exception(key=key, reservation=reservation)
            raise
        return self._after_call(
            request,
            result,
            key=key,
            reservation=reservation,
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        _signature, key, reservation, suppressed = self._before_call(request)
        if suppressed is not None:
            return suppressed
        try:
            result = await handler(request)
        except BaseException:
            self._after_exception(key=key, reservation=reservation)
            raise
        return self._after_call(
            request,
            result,
            key=key,
            reservation=reservation,
        )


class ToolOutcomeStatusMiddleware(AgentMiddleware[Any, Any, Any]):
    """Preserve tool observations while normalizing semantic failures to error."""

    @staticmethod
    def _normalize(
        result: ToolMessage | Command[Any],
        request: ToolCallRequest,
    ) -> ToolMessage | Command[Any]:
        if not isinstance(result, ToolMessage) or result.status == "error":
            return result
        tool_name = result.name or str(request.tool_call.get("name") or "")
        outcome = classify_tool_outcome(result, tool_name=tool_name)
        if outcome.status != "error":
            return result
        return result.model_copy(update={"status": "error"})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._normalize(handler(request), request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        return self._normalize(await handler(request), request)
