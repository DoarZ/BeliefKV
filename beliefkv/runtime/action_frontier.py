from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from beliefkv.core.events import RuntimeEvent, RuntimeEventKind


class StructuredActionKind(str, Enum):
    FUNCTION_CALL = "function_call"
    SPAWN = "spawn"
    HANDOFF = "handoff"
    FINAL_ANSWER = "final_answer"
    UNKNOWN = "unknown"


class ParserStatus(str, Enum):
    EMPTY = "empty"
    INCOMPLETE = "incomplete"
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParserUpdate:
    status: ParserStatus
    generated_tokens: int
    action_kind: StructuredActionKind = StructuredActionKind.UNKNOWN
    action_name: str | None = None
    boundary_token_index: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative")
        if self.boundary_token_index is not None:
            if self.status != ParserStatus.VALID:
                raise ValueError("only a valid parser state can carry a boundary token")
            if not 0 < self.boundary_token_index <= self.generated_tokens:
                raise ValueError("boundary token must lie within generated output")
        if self.status == ParserStatus.VALID and self.action_kind == StructuredActionKind.UNKNOWN:
            raise ValueError("a valid parser state requires a known action kind")


class JsonActionParser:
    """Incremental parser for a normalized JSON action envelope.

    Model-specific parsers should emit the same ``ParserUpdate`` contract. This
    parser is deliberately narrow; arbitrary free text is UNKNOWN rather than
    being guessed as a final answer.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._valid: ParserUpdate | None = None

    def feed(self, fragment: str, *, generated_tokens: int) -> ParserUpdate:
        if generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative")
        if self._valid is not None:
            return replace(self._valid, generated_tokens=generated_tokens)
        self._buffer += fragment
        stripped = self._buffer.strip()
        if not stripped:
            return ParserUpdate(ParserStatus.EMPTY, generated_tokens)
        if not stripped.startswith("{"):
            return ParserUpdate(
                ParserStatus.UNKNOWN,
                generated_tokens,
                reason="output does not use the normalized structured-action envelope",
            )
        try:
            decoded, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError as error:
            status = (
                ParserStatus.INVALID
                if stripped.endswith("}")
                else ParserStatus.INCOMPLETE
            )
            return ParserUpdate(status, generated_tokens, reason=error.msg)
        if stripped[end:].strip():
            return ParserUpdate(
                ParserStatus.INVALID,
                generated_tokens,
                reason="non-whitespace content follows the action envelope",
            )
        if not isinstance(decoded, Mapping):
            return ParserUpdate(
                ParserStatus.INVALID,
                generated_tokens,
                reason="action envelope must be a JSON object",
            )
        action_raw = str(decoded.get("action", ""))
        kind = {
            "tool": StructuredActionKind.FUNCTION_CALL,
            "function_call": StructuredActionKind.FUNCTION_CALL,
            "spawn": StructuredActionKind.SPAWN,
            "task": StructuredActionKind.SPAWN,
            "handoff": StructuredActionKind.HANDOFF,
            "final": StructuredActionKind.FINAL_ANSWER,
            "final_answer": StructuredActionKind.FINAL_ANSWER,
        }.get(action_raw, StructuredActionKind.UNKNOWN)
        if kind == StructuredActionKind.UNKNOWN:
            return ParserUpdate(
                ParserStatus.INVALID,
                generated_tokens,
                reason=f"unsupported structured action {action_raw!r}",
            )
        name = decoded.get("name") or decoded.get("target") or decoded.get("role")
        if kind in {
            StructuredActionKind.FUNCTION_CALL,
            StructuredActionKind.SPAWN,
            StructuredActionKind.HANDOFF,
        } and not name:
            return ParserUpdate(
                ParserStatus.INVALID,
                generated_tokens,
                reason="structured action is missing name/target/role",
            )
        self._valid = ParserUpdate(
            ParserStatus.VALID,
            generated_tokens,
            action_kind=kind,
            action_name=str(name) if name is not None else None,
            boundary_token_index=generated_tokens,
            reason="normalized JSON action became valid",
        )
        return self._valid


@dataclass(frozen=True)
class ActionFrontierSnapshot:
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    started_ts_ms: float
    updated_ts_ms: float
    parser_status: ParserStatus
    action_kind: StructuredActionKind
    action_name: str | None
    generated_tokens: int
    boundary_token_index: int | None
    valid_action_ts_ms: float | None
    boundary_source: str
    action_event_kind: str | None
    action_event_ts_ms: float | None
    tool_start_gap_ms: float | None
    runnable_frontier_before: tuple[str, ...]
    runnable_frontier_after: tuple[str, ...]
    frontier_added: tuple[str, ...]
    frontier_removed: tuple[str, ...]
    active_kv_bytes_before: int | None
    waiting_kv_bytes_after: int | None
    reentry_ts_ms: float | None
    reentry_delay_ms: float | None


@dataclass(frozen=True)
class ActionFrontierAuditEvent:
    kind: str
    ts_ms: float
    fields: Mapping[str, object]


class ActionFrontierObserver:
    """Correlate parser validity with subsequent causal/runtime transitions."""

    _ACTION_EVENTS = {
        RuntimeEventKind.TOOL_START,
        RuntimeEventKind.SPAWN,
        RuntimeEventKind.HANDOFF,
        RuntimeEventKind.RETURN,
    }
    _REENTRY_EVENTS = {
        RuntimeEventKind.TOOL_END,
        RuntimeEventKind.REACTIVATE,
        RuntimeEventKind.MESSAGE,
    }

    def __init__(self) -> None:
        self._states: dict[str, ActionFrontierSnapshot] = {}
        self._latest_request_by_invocation: dict[str, str] = {}
        self._events: list[ActionFrontierAuditEvent] = []

    def begin(
        self,
        *,
        request_id: str,
        workflow_id: str,
        invocation_id: str,
        context_id: str,
        ts_ms: float,
        runnable_frontier: tuple[str, ...] = (),
        active_kv_bytes: int | None = None,
    ) -> ActionFrontierSnapshot:
        if not all((request_id, workflow_id, invocation_id, context_id)):
            raise ValueError("action frontier identities must be non-empty")
        if ts_ms < 0 or (active_kv_bytes is not None and active_kv_bytes < 0):
            raise ValueError("action frontier time/bytes must be non-negative")
        state = ActionFrontierSnapshot(
            request_id=request_id,
            workflow_id=workflow_id,
            invocation_id=invocation_id,
            context_id=context_id,
            started_ts_ms=ts_ms,
            updated_ts_ms=ts_ms,
            parser_status=ParserStatus.EMPTY,
            action_kind=StructuredActionKind.UNKNOWN,
            action_name=None,
            generated_tokens=0,
            boundary_token_index=None,
            valid_action_ts_ms=None,
            boundary_source="unobserved",
            action_event_kind=None,
            action_event_ts_ms=None,
            tool_start_gap_ms=None,
            runnable_frontier_before=tuple(sorted(set(runnable_frontier))),
            runnable_frontier_after=(),
            frontier_added=(),
            frontier_removed=(),
            active_kv_bytes_before=active_kv_bytes,
            waiting_kv_bytes_after=None,
            reentry_ts_ms=None,
            reentry_delay_ms=None,
        )
        self._states[request_id] = state
        self._latest_request_by_invocation[invocation_id] = request_id
        return state

    def observe_parser_update(
        self,
        request_id: str,
        update: ParserUpdate,
        *,
        ts_ms: float,
        source: str = "native_incremental_parser",
    ) -> ActionFrontierSnapshot:
        state = self._require(request_id)
        if ts_ms < state.updated_ts_ms:
            raise ValueError("action parser updates must be timestamp-monotonic")
        if update.generated_tokens < state.generated_tokens:
            raise ValueError("generated token count cannot move backwards")
        valid_ts = state.valid_action_ts_ms
        boundary = state.boundary_token_index
        boundary_source = state.boundary_source
        if state.parser_status != ParserStatus.VALID and update.status == ParserStatus.VALID:
            valid_ts = ts_ms
            boundary = update.boundary_token_index
            boundary_source = source
            self._events.append(
                ActionFrontierAuditEvent(
                    kind="valid_action_unlocked",
                    ts_ms=ts_ms,
                    fields={
                        "request_id": request_id,
                        "workflow_id": state.workflow_id,
                        "invocation_id": state.invocation_id,
                        "action_kind": update.action_kind.value,
                        "action_name": update.action_name,
                        "boundary_token_index": boundary,
                        "boundary_source": source,
                    },
                )
            )
        next_state = replace(
            state,
            updated_ts_ms=ts_ms,
            parser_status=update.status,
            action_kind=update.action_kind,
            action_name=update.action_name,
            generated_tokens=update.generated_tokens,
            boundary_token_index=boundary,
            valid_action_ts_ms=valid_ts,
            boundary_source=boundary_source,
        )
        self._states[request_id] = next_state
        self._events.append(
            ActionFrontierAuditEvent(
                kind="action_frontier_updated",
                ts_ms=ts_ms,
                fields={
                    "request_id": request_id,
                    "parser_status": update.status.value,
                    "generated_tokens": update.generated_tokens,
                    "boundary_token_index": boundary,
                    "reason": update.reason,
                },
            )
        )
        return next_state

    def observe_runtime_event(
        self,
        event: RuntimeEvent,
        *,
        runnable_frontier_before: tuple[str, ...] = (),
        runnable_frontier_after: tuple[str, ...] = (),
        context_gpu_bytes: int | None = None,
    ) -> ActionFrontierSnapshot | None:
        invocation_id = (
            event.target_invocation_id
            if event.kind == RuntimeEventKind.MESSAGE
            else event.invocation_id
        )
        if event.kind == RuntimeEventKind.LLM_SUBMIT and event.invocation_id:
            request_id = str(event.attributes.get("request_id") or event.event_id)
            return self.begin(
                request_id=request_id,
                workflow_id=event.workflow_id,
                invocation_id=event.invocation_id,
                context_id=str(event.context_id or event.invocation_id),
                ts_ms=event.ts_ms,
                runnable_frontier=runnable_frontier_before,
                active_kv_bytes=context_gpu_bytes,
            )
        if invocation_id is None:
            return None
        request_id = self._latest_request_by_invocation.get(invocation_id)
        if request_id is None:
            return None
        state = self._states[request_id]

        if event.kind in {
            RuntimeEventKind.LLM_RESULT,
            RuntimeEventKind.STRUCTURED_ACTION,
        }:
            update = _parser_update_from_runtime_result(event, state.generated_tokens)
            return self.observe_parser_update(
                request_id,
                update,
                ts_ms=event.ts_ms,
                source=str(
                    event.attributes.get(
                        "action_boundary_source", "runtime_structured_output"
                    )
                ),
            )
        if event.kind in self._ACTION_EVENTS:
            before = set(state.runnable_frontier_before)
            after = set(runnable_frontier_after)
            gap = (
                event.ts_ms - state.valid_action_ts_ms
                if event.kind == RuntimeEventKind.TOOL_START
                and state.valid_action_ts_ms is not None
                else state.tool_start_gap_ms
            )
            next_state = replace(
                state,
                updated_ts_ms=max(state.updated_ts_ms, event.ts_ms),
                action_event_kind=event.kind.value,
                action_event_ts_ms=event.ts_ms,
                tool_start_gap_ms=gap,
                runnable_frontier_after=tuple(sorted(after)),
                frontier_added=tuple(sorted(after - before)),
                frontier_removed=tuple(sorted(before - after)),
                waiting_kv_bytes_after=context_gpu_bytes,
            )
            self._states[request_id] = next_state
            return next_state
        if event.kind in self._REENTRY_EVENTS:
            base_ts = state.action_event_ts_ms or state.valid_action_ts_ms
            next_state = replace(
                state,
                updated_ts_ms=max(state.updated_ts_ms, event.ts_ms),
                reentry_ts_ms=event.ts_ms,
                reentry_delay_ms=(event.ts_ms - base_ts if base_ts is not None else None),
            )
            self._states[request_id] = next_state
            return next_state
        return state

    def snapshot(self, request_id: str) -> ActionFrontierSnapshot:
        return self._require(request_id)

    def snapshots(self) -> tuple[ActionFrontierSnapshot, ...]:
        return tuple(self._states[key] for key in sorted(self._states))

    def drain_audit_events(self) -> tuple[ActionFrontierAuditEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def _require(self, request_id: str) -> ActionFrontierSnapshot:
        try:
            return self._states[request_id]
        except KeyError as error:
            raise KeyError(f"unknown action-frontier request: {request_id}") from error


def _parser_update_from_runtime_result(
    event: RuntimeEvent,
    previous_generated_tokens: int,
) -> ParserUpdate:
    attributes = event.attributes
    generated_tokens = max(
        previous_generated_tokens,
        int(attributes.get("output_tokens") or previous_generated_tokens),
    )
    try:
        status = ParserStatus(str(attributes.get("parser_status", "unknown")))
    except ValueError:
        status = ParserStatus.UNKNOWN
    raw_kinds = attributes.get("structured_action_kinds", ())
    if isinstance(raw_kinds, str):
        raw_kinds = (raw_kinds,)
    kind = StructuredActionKind.UNKNOWN
    if isinstance(raw_kinds, (list, tuple)) and raw_kinds:
        try:
            kind = StructuredActionKind(str(raw_kinds[0]))
        except ValueError:
            kind = StructuredActionKind.UNKNOWN
    names = attributes.get("structured_action_names", ())
    if isinstance(names, str):
        names = (names,)
    name = str(names[0]) if isinstance(names, (list, tuple)) and names else None
    boundary_raw = attributes.get("action_boundary_token_index")
    boundary = int(boundary_raw) if boundary_raw is not None else None
    if status == ParserStatus.VALID and kind == StructuredActionKind.UNKNOWN:
        status = ParserStatus.UNKNOWN
    return ParserUpdate(
        status=status,
        generated_tokens=generated_tokens,
        action_kind=kind,
        action_name=name,
        boundary_token_index=boundary,
        reason=str(attributes.get("parser_reason", "runtime result observation")),
    )
