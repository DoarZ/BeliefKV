from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from beliefkv.core.events import (
    ContextMode,
    EventConfidence,
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.predictor.taxonomy import ToolTaxonomy


@dataclass
class NormalizationReport:
    input_records: int = 0
    output_events: int = 0
    inferred_events: int = 0
    unresolved_parent_events: int = 0
    ignored_records: int = 0
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizedTrace:
    events: tuple[RuntimeEvent, ...]
    report: NormalizationReport


class ClawTraceNormalizer:
    """Normalize ClawTrace ingest envelopes into BeliefKV runtime events."""

    def __init__(self, taxonomy: ToolTaxonomy | None = None) -> None:
        self.taxonomy = taxonomy or ToolTaxonomy()
        self._reset()

    def _reset(self) -> None:
        self._workflow_started: set[str] = set()
        self._workflow_last_ts: dict[str, float] = {}
        self._session_to_invocation: dict[str, str] = {}
        self._session_to_context: dict[str, str] = {}
        self._invocation_workflow: dict[str, str] = {}
        self._created_invocations: set[str] = set()
        self._span_to_invocation: dict[str, str] = {}
        self._active_tool_by_span: dict[str, str] = {}
        self._context_epoch: dict[str, int] = {}
        self._event_sequence = 0
        self.report = NormalizationReport()

    def normalize(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        close_workflows: bool = True,
    ) -> NormalizedTrace:
        self._reset()
        output: list[RuntimeEvent] = []
        for record in records:
            self.report.input_records += 1
            output.extend(self._normalize_one(record))
        if close_workflows:
            for workflow_id in sorted(self._workflow_started):
                output.append(
                    self._event(
                        workflow_id=workflow_id,
                        ts_ms=self._workflow_last_ts[workflow_id],
                        kind=RuntimeEventKind.WORKFLOW_END,
                        source_event_id=f"synthetic-end-{workflow_id}",
                        confidence=EventConfidence.INFERRED,
                        attributes={"normalizer_synthetic": True},
                    )
                )
                self.report.inferred_events += 1
        self.report.output_events = len(output)
        return NormalizedTrace(tuple(output), self.report)

    def _normalize_one(self, envelope: Mapping[str, Any]) -> list[RuntimeEvent]:
        raw_event = envelope.get("event", envelope)
        if not isinstance(raw_event, Mapping):
            self.report.ignored_records += 1
            self.report.diagnostics.append("record has no object-valued event")
            return []
        event_type = raw_event.get("eventType", raw_event.get("event_type"))
        if event_type is None:
            self.report.ignored_records += 1
            self.report.diagnostics.append("record has no eventType")
            return []
        payload = raw_event.get("payload", {})
        if not isinstance(payload, Mapping):
            payload = {}
        workflow_id = str(raw_event.get("traceId", raw_event.get("trace_id", "")))
        if not workflow_id:
            self.report.ignored_records += 1
            self.report.diagnostics.append(f"{event_type}: missing traceId")
            return []
        ts_ms = float(raw_event.get("tsMs", raw_event.get("ts_ms", 0.0)))
        source_id = str(
            raw_event.get("eventId", raw_event.get("event_id", f"source-{self._event_sequence}"))
        )
        span_id = str(raw_event.get("spanId", raw_event.get("span_id", "")))
        parent_span_id = raw_event.get("parentSpanId", raw_event.get("parent_span_id"))
        self._workflow_last_ts[workflow_id] = max(
            ts_ms, self._workflow_last_ts.get(workflow_id, ts_ms)
        )
        events: list[RuntimeEvent] = []
        if workflow_id not in self._workflow_started:
            self._workflow_started.add(workflow_id)
            events.append(
                self._event(
                    workflow_id=workflow_id,
                    ts_ms=ts_ms,
                    kind=RuntimeEventKind.WORKFLOW_START,
                    source_event_id=f"{source_id}:workflow",
                    confidence=EventConfidence.OBSERVED_EXACT,
                )
            )

        if event_type == "session_start":
            events.extend(
                self._session_start(
                    workflow_id, ts_ms, source_id, span_id, parent_span_id, payload, envelope
                )
            )
        elif event_type == "session_end":
            invocation_id = self._lookup_invocation(payload)
            if invocation_id is not None:
                events.append(
                    self._event(
                        workflow_id=workflow_id,
                        ts_ms=ts_ms,
                        kind=RuntimeEventKind.RETURN,
                        source_event_id=source_id,
                        invocation_id=invocation_id,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={"source": "clawtrace_session_end"},
                    )
                )
            else:
                self._unresolved(event_type, source_id)
        elif event_type == "llm_before_call":
            invocation_id = self._lookup_invocation(payload, span_id, parent_span_id)
            if invocation_id is not None:
                self._span_to_invocation[span_id] = invocation_id
                events.append(
                    self._event(
                        workflow_id=workflow_id,
                        ts_ms=ts_ms,
                        kind=RuntimeEventKind.LLM_SUBMIT,
                        source_event_id=source_id,
                        invocation_id=invocation_id,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "model": payload.get("model", "unknown"),
                            "provider": payload.get("provider", "unknown"),
                            "history_messages": payload.get("historyMessagesCount", 0),
                        },
                    )
                )
            else:
                self._unresolved(event_type, source_id)
        elif event_type == "llm_after_call":
            invocation_id = self._lookup_invocation(payload, span_id, parent_span_id)
            if invocation_id is not None:
                events.append(
                    self._event(
                        workflow_id=workflow_id,
                        ts_ms=ts_ms,
                        kind=RuntimeEventKind.LLM_RESULT,
                        source_event_id=source_id,
                        invocation_id=invocation_id,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={"usage": payload.get("usage", {})},
                    )
                )
            else:
                self._unresolved(event_type, source_id)
        elif event_type == "tool_before_call":
            invocation_id = self._lookup_invocation(payload, span_id, parent_span_id)
            if invocation_id is not None:
                tool_name = str(payload.get("toolName", "unknown"))
                normalized = self.taxonomy.normalize(tool_name)
                self._span_to_invocation[span_id] = invocation_id
                tool_call_id = str(payload.get("toolCallId", span_id))
                self._active_tool_by_span[span_id] = tool_call_id
                events.append(
                    self._event(
                        workflow_id=workflow_id,
                        ts_ms=ts_ms,
                        kind=RuntimeEventKind.TOOL_START,
                        source_event_id=source_id,
                        invocation_id=invocation_id,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "tool_name": tool_name,
                            "tool_family": normalized.family,
                            "backend_class": normalized.backend_class,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )
            else:
                self._unresolved(event_type, source_id)
        elif event_type == "tool_after_call":
            invocation_id = self._lookup_invocation(payload, span_id, parent_span_id)
            if invocation_id is not None:
                events.append(
                    self._event(
                        workflow_id=workflow_id,
                        ts_ms=ts_ms,
                        kind=RuntimeEventKind.TOOL_END,
                        source_event_id=source_id,
                        invocation_id=invocation_id,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "tool_call_id": payload.get("toolCallId", span_id),
                            "duration_ms": payload.get("durationMs"),
                            "error": payload.get("error"),
                        },
                    )
                )
            else:
                self._unresolved(event_type, source_id)
        elif event_type == "subagent_spawn":
            events.extend(
                self._subagent_spawn(
                    workflow_id, ts_ms, source_id, span_id, payload, envelope
                )
            )
        elif event_type == "subagent_join":
            session_key = str(payload.get("targetSessionKey", ""))
            invocation_id = self._session_to_invocation.get(session_key)
            if invocation_id is not None:
                events.append(
                    self._event(
                        workflow_id=workflow_id,
                        ts_ms=ts_ms,
                        kind=RuntimeEventKind.RETURN,
                        source_event_id=source_id,
                        invocation_id=invocation_id,
                        confidence=EventConfidence.OBSERVED_EXACT,
                        attributes={
                            "outcome": payload.get("outcome"),
                            "reason": payload.get("reason"),
                            "error": payload.get("error"),
                        },
                    )
                )
            else:
                self._unresolved(event_type, source_id)
        else:
            self.report.ignored_records += 1
        return events

    def _session_start(
        self,
        workflow_id: str,
        ts_ms: float,
        source_id: str,
        span_id: str,
        parent_span_id: Any,
        payload: Mapping[str, Any],
        envelope: Mapping[str, Any],
    ) -> list[RuntimeEvent]:
        session_key = str(payload.get("sessionKey", payload.get("sessionId", span_id)))
        run_id = str(payload.get("runId", span_id or session_key))
        invocation_id = self._session_to_invocation.get(session_key, run_id)
        context_id = self._session_to_context.get(session_key, session_key)
        self._session_to_invocation[session_key] = invocation_id
        self._session_to_context[session_key] = context_id
        self._invocation_workflow[invocation_id] = workflow_id
        self._span_to_invocation[span_id] = invocation_id
        if invocation_id in self._created_invocations:
            return []
        self._created_invocations.add(invocation_id)
        epoch = self._context_epoch.setdefault(context_id, 0)
        parent_id = (
            self._span_to_invocation.get(str(parent_span_id))
            if parent_span_id is not None
            else None
        )
        confidence = (
            EventConfidence.OBSERVED_EXACT
            if parent_id is not None or not payload.get("isSubAgent")
            else EventConfidence.INFERRED
        )
        if confidence == EventConfidence.INFERRED:
            self.report.inferred_events += 1
        return [
            self._event(
                workflow_id=workflow_id,
                ts_ms=ts_ms,
                kind=RuntimeEventKind.INVOCATION_CREATE,
                source_event_id=source_id,
                invocation_id=invocation_id,
                context_id=context_id,
                parent_invocation_id=parent_id,
                agent_definition_id=str(
                    payload.get("agentName", payload.get("agentId", envelope.get("agentId", "unknown")))
                ),
                agent_instance_id=session_key,
                context_epoch=epoch,
                context_mode=ContextMode.FRESH,
                relation_type=(RelationType.SPAWN if parent_id else RelationType.ROOT),
                confidence=confidence,
                attributes={"persistent": True, "source": "clawtrace_session"},
            )
        ]

    def _subagent_spawn(
        self,
        workflow_id: str,
        ts_ms: float,
        source_id: str,
        span_id: str,
        payload: Mapping[str, Any],
        envelope: Mapping[str, Any],
    ) -> list[RuntimeEvent]:
        parent_session = str(payload.get("requesterSessionKey", ""))
        child_session = str(payload.get("childSessionKey", ""))
        parent_id = self._session_to_invocation.get(parent_session)
        if not child_session:
            self._unresolved("subagent_spawn_child", source_id)
            return []
        child_id = self._session_to_invocation.get(
            child_session, str(payload.get("runId", child_session))
        )
        child_context = self._session_to_context.get(child_session, child_session)
        self._session_to_invocation[child_session] = child_id
        self._session_to_context[child_session] = child_context
        self._invocation_workflow[child_id] = workflow_id
        output: list[RuntimeEvent] = []
        if child_id not in self._created_invocations:
            self._created_invocations.add(child_id)
            epoch = self._context_epoch.setdefault(child_context, 0)
            output.append(
                self._event(
                    workflow_id=workflow_id,
                    ts_ms=ts_ms,
                    kind=RuntimeEventKind.INVOCATION_CREATE,
                    source_event_id=f"{source_id}:create",
                    invocation_id=child_id,
                    context_id=child_context,
                    parent_invocation_id=parent_id,
                    agent_definition_id=str(payload.get("label", payload.get("subagentId", "subagent"))),
                    agent_instance_id=child_session,
                    context_epoch=epoch,
                    context_mode=ContextMode.FRESH,
                    relation_type=RelationType.SPAWN,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={"persistent": False, "source": "clawtrace_subagent_spawn"},
                )
            )
        self._span_to_invocation[span_id] = child_id
        if parent_id is None:
            self._unresolved("subagent_spawn_parent", source_id)
            return output
        mode_value = str(payload.get("mode", "foreground")).lower()
        background = mode_value in {"background", "async", "session", "parallel"}
        output.append(
            self._event(
                workflow_id=workflow_id,
                ts_ms=ts_ms,
                kind=(RuntimeEventKind.SPAWN if background else RuntimeEventKind.CALL),
                source_event_id=source_id,
                invocation_id=parent_id,
                target_invocation_id=child_id,
                execution_mode=(
                    ExecutionMode.BACKGROUND if background else ExecutionMode.FOREGROUND
                ),
                confidence=EventConfidence.OBSERVED_EXACT,
                attributes={"clawtrace_mode": mode_value},
            )
        )
        return output

    def _lookup_invocation(
        self,
        payload: Mapping[str, Any],
        span_id: str | None = None,
        parent_span_id: Any = None,
    ) -> str | None:
        session_key = payload.get("sessionKey")
        if session_key is not None and str(session_key) in self._session_to_invocation:
            return self._session_to_invocation[str(session_key)]
        run_id = payload.get("runId")
        if run_id is not None and str(run_id) in self._invocation_workflow:
            return str(run_id)
        for candidate in (span_id, parent_span_id):
            if candidate is not None and str(candidate) in self._span_to_invocation:
                return self._span_to_invocation[str(candidate)]
        return None

    def _event(
        self,
        *,
        workflow_id: str,
        ts_ms: float,
        kind: RuntimeEventKind,
        source_event_id: str,
        confidence: EventConfidence,
        **kwargs: Any,
    ) -> RuntimeEvent:
        self._event_sequence += 1
        return RuntimeEvent(
            event_id=f"{source_event_id}:{kind.value}:{self._event_sequence}",
            ts_ms=ts_ms,
            kind=kind,
            workflow_id=workflow_id,
            confidence=confidence,
            **kwargs,
        )

    def _unresolved(self, event_type: str, source_id: str) -> None:
        self.report.unresolved_parent_events += 1
        self.report.diagnostics.append(f"{source_id}: unresolved {event_type} invocation")


def load_jsonl_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    import json

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object")
        records.append(value)
    return records
