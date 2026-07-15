from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from beliefkv.core.events import RuntimeEvent, RuntimeEventKind


class RuntimeTraceValidationError(ValueError):
    """Raised when an authoritative runtime trace is not replay-safe."""


@dataclass(frozen=True)
class RuntimeTraceSummary:
    workflow_id: str
    event_count: int
    invocation_count: int
    llm_call_count: int
    tool_call_count: int
    span_ms: float
    total_tool_duration_ms: float
    total_prompt_tokens: int
    total_cache_hit_tokens: int
    total_uncached_prompt_tokens: int
    total_output_tokens: int
    event_kind_counts: Mapping[str, int]
    controller_replay_valid: bool = True
    audit_record_count: int = 0
    runtime_delivery_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeTraceValidationError(
                    f"{source}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeTraceValidationError(
                    f"{source}:{line_number}: record must be an object"
                )
            records.append(record)
    if not records:
        raise RuntimeTraceValidationError(f"{source}: trace is empty")
    return records


def validate_runtime_trace(
    event_path: str | Path,
    audit_path: str | Path | None = None,
) -> RuntimeTraceSummary:
    raw_events = read_jsonl(event_path)
    events: list[RuntimeEvent] = []
    event_ids: set[str] = set()
    previous_ts = -1.0

    for sequence, raw in enumerate(raw_events, start=1):
        if raw.get("schema_version") != 1:
            raise RuntimeTraceValidationError(
                f"event {sequence}: unsupported schema_version"
            )
        if raw.get("sequence") != sequence:
            raise RuntimeTraceValidationError(
                f"event sequence is not contiguous at record {sequence}"
            )
        try:
            event = RuntimeEvent.from_dict(raw)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeTraceValidationError(
                f"event {sequence}: invalid runtime event: {error}"
            ) from error
        if not math.isfinite(event.ts_ms):
            raise RuntimeTraceValidationError(
                f"event {sequence}: timestamp must be finite"
            )
        if event.ts_ms < previous_ts:
            raise RuntimeTraceValidationError(
                f"event {sequence}: timestamps are not monotonic"
            )
        if event.event_id in event_ids:
            raise RuntimeTraceValidationError(
                f"event {sequence}: duplicate event_id {event.event_id}"
            )
        previous_ts = event.ts_ms
        event_ids.add(event.event_id)
        events.append(event)

    workflows = {event.workflow_id for event in events}
    if len(workflows) != 1:
        raise RuntimeTraceValidationError(
            f"a frozen trace must contain exactly one workflow, got {workflows}"
        )
    workflow_id = next(iter(workflows))
    if events[0].kind != RuntimeEventKind.WORKFLOW_START:
        raise RuntimeTraceValidationError("first event must be workflow_start")
    if events[-1].kind != RuntimeEventKind.WORKFLOW_END:
        raise RuntimeTraceValidationError("last event must be workflow_end")

    kind_counts = Counter(event.kind.value for event in events)
    if kind_counts[RuntimeEventKind.WORKFLOW_START.value] != 1:
        raise RuntimeTraceValidationError("trace must have one workflow_start")
    if kind_counts[RuntimeEventKind.WORKFLOW_END.value] != 1:
        raise RuntimeTraceValidationError("trace must have one workflow_end")

    created_invocations: set[str] = set()
    terminal_invocations: set[str] = set()
    tool_starts: dict[str, RuntimeEvent] = {}
    tool_ends: set[str] = set()
    llm_submits: dict[str, RuntimeEvent] = {}
    llm_results: set[str] = set()
    total_tool_duration_ms = 0.0
    total_prompt_tokens = 0
    total_cache_hit_tokens = 0
    total_output_tokens = 0

    for event in events:
        if event.kind == RuntimeEventKind.INVOCATION_CREATE:
            invocation_id = _required_id(event.invocation_id, "invocation_id", event)
            if invocation_id in created_invocations:
                raise RuntimeTraceValidationError(
                    f"invocation created twice: {invocation_id}"
                )
            created_invocations.add(invocation_id)
            continue

        if event.invocation_id is not None and event.invocation_id not in created_invocations:
            raise RuntimeTraceValidationError(
                f"event {event.event_id} references an invocation before create: "
                f"{event.invocation_id}"
            )

        if event.target_invocation_id is not None:
            if event.target_invocation_id not in created_invocations:
                raise RuntimeTraceValidationError(
                    f"event {event.event_id} references an unknown target invocation: "
                    f"{event.target_invocation_id}"
                )

        if event.kind in {
            RuntimeEventKind.RETURN,
            RuntimeEventKind.INVOCATION_CANCEL,
        }:
            invocation_id = _required_id(event.invocation_id, "invocation_id", event)
            if invocation_id in terminal_invocations:
                raise RuntimeTraceValidationError(
                    f"invocation has multiple terminal events: {invocation_id}"
                )
            terminal_invocations.add(invocation_id)
        elif event.kind == RuntimeEventKind.TOOL_START:
            tool_call_id = _attribute_id(event, "tool_call_id")
            if tool_call_id in tool_starts:
                raise RuntimeTraceValidationError(
                    f"tool call started twice: {tool_call_id}"
                )
            tool_starts[tool_call_id] = event
        elif event.kind == RuntimeEventKind.TOOL_END:
            tool_call_id = _attribute_id(event, "tool_call_id")
            start = tool_starts.get(tool_call_id)
            if start is None:
                raise RuntimeTraceValidationError(
                    f"tool call ended before start: {tool_call_id}"
                )
            if tool_call_id in tool_ends:
                raise RuntimeTraceValidationError(
                    f"tool call ended twice: {tool_call_id}"
                )
            _require_same_owner(start, event, "tool", tool_call_id)
            duration_ms = _nonnegative_number(event, "duration_ms")
            observed_duration_ms = event.ts_ms - start.ts_ms
            if abs(duration_ms - observed_duration_ms) > 1e-3:
                raise RuntimeTraceValidationError(
                    f"tool duration disagrees with timestamps: {tool_call_id}"
                )
            total_tool_duration_ms += duration_ms
            tool_ends.add(tool_call_id)
        elif event.kind == RuntimeEventKind.LLM_SUBMIT:
            request_id = _attribute_id(event, "request_id")
            if request_id in llm_submits:
                raise RuntimeTraceValidationError(
                    f"LLM request submitted twice: {request_id}"
                )
            prompt_tokens = _nonnegative_integer(event, "prompt_tokens")
            cache_hit_tokens = _nonnegative_integer(event, "cache_hit_tokens")
            if cache_hit_tokens > prompt_tokens:
                raise RuntimeTraceValidationError(
                    f"cache hit exceeds prompt length: {request_id}"
                )
            total_prompt_tokens += prompt_tokens
            total_cache_hit_tokens += cache_hit_tokens
            llm_submits[request_id] = event
        elif event.kind == RuntimeEventKind.LLM_RESULT:
            request_id = _attribute_id(event, "request_id")
            start = llm_submits.get(request_id)
            if start is None:
                raise RuntimeTraceValidationError(
                    f"LLM result precedes submit: {request_id}"
                )
            if request_id in llm_results:
                raise RuntimeTraceValidationError(
                    f"LLM request has multiple results: {request_id}"
                )
            _require_same_owner(start, event, "LLM request", request_id)
            total_output_tokens += _nonnegative_integer(event, "output_tokens")
            llm_results.add(request_id)

    if created_invocations != terminal_invocations:
        raise RuntimeTraceValidationError(
            "invocation lifecycle mismatch: "
            f"created={sorted(created_invocations)}, "
            f"terminal={sorted(terminal_invocations)}"
        )
    if set(tool_starts) != tool_ends:
        raise RuntimeTraceValidationError(
            f"unmatched tool calls: {sorted(set(tool_starts) ^ tool_ends)}"
        )
    if set(llm_submits) != llm_results:
        raise RuntimeTraceValidationError(
            f"unmatched LLM requests: {sorted(set(llm_submits) ^ llm_results)}"
        )

    audit_record_count = 0
    runtime_delivery_count = 0
    if audit_path is not None:
        audit_record_count, runtime_delivery_count = _validate_audit(
            audit_path,
            expected_request_ids=set(llm_submits),
        )
    _validate_controller_replay(events, workflow_id, len(created_invocations))

    return RuntimeTraceSummary(
        workflow_id=workflow_id,
        event_count=len(events),
        invocation_count=len(created_invocations),
        llm_call_count=len(llm_submits),
        tool_call_count=len(tool_starts),
        span_ms=events[-1].ts_ms - events[0].ts_ms,
        total_tool_duration_ms=total_tool_duration_ms,
        total_prompt_tokens=total_prompt_tokens,
        total_cache_hit_tokens=total_cache_hit_tokens,
        total_uncached_prompt_tokens=(
            total_prompt_tokens - total_cache_hit_tokens
        ),
        total_output_tokens=total_output_tokens,
        event_kind_counts=dict(sorted(kind_counts.items())),
        controller_replay_valid=True,
        audit_record_count=audit_record_count,
        runtime_delivery_count=runtime_delivery_count,
    )


def relative_event_records(event_path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(event_path)
    base_ts = float(records[0]["ts_ms"])
    normalized: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["ts_ms"] = float(item["ts_ms"]) - base_ts
        normalized.append(item)
    return normalized


def _validate_controller_replay(
    events: list[RuntimeEvent],
    workflow_id: str,
    expected_invocation_count: int,
) -> None:
    from beliefkv.control.causal_graph import InvocationState
    from beliefkv.control.controller import BeliefKVController
    from beliefkv.core.config import BeliefKVConfig

    controller = BeliefKVController(
        BeliefKVConfig(predictor_enabled=False, shadow_enabled=False)
    )
    try:
        controller.process_runtime_events(tuple(events))
    except Exception as error:
        raise RuntimeTraceValidationError(
            f"control-plane replay failed: {type(error).__name__}: {error}"
        ) from error
    workflow = controller.graph.workflows.get(workflow_id)
    if workflow is None or workflow.end_ts_ms is None:
        raise RuntimeTraceValidationError(
            "control-plane replay did not terminate the workflow"
        )
    invocations = [
        invocation
        for invocation in controller.graph.invocations.values()
        if invocation.workflow_id == workflow_id
    ]
    if len(invocations) != expected_invocation_count:
        raise RuntimeTraceValidationError(
            "control-plane replay invocation count mismatch"
        )
    terminal_states = {InvocationState.DONE, InvocationState.CANCELLED}
    if any(invocation.state not in terminal_states for invocation in invocations):
        raise RuntimeTraceValidationError(
            "control-plane replay left a non-terminal invocation"
        )


def _validate_audit(
    audit_path: str | Path,
    *,
    expected_request_ids: set[str],
) -> tuple[int, int]:
    records = read_jsonl(audit_path)
    run_ids: set[str] = set()
    previous_ts = -1.0
    started: set[str] = set()
    finished: set[str] = set()
    delivery_count = 0
    initialized = 0

    for sequence, record in enumerate(records, start=1):
        if record.get("schema_version") != 1:
            raise RuntimeTraceValidationError(
                f"audit record {sequence}: unsupported schema_version"
            )
        if record.get("sequence") != sequence:
            raise RuntimeTraceValidationError(
                f"audit sequence is not contiguous at record {sequence}"
            )
        run_id = str(record.get("run_id", ""))
        if not run_id:
            raise RuntimeTraceValidationError(
                f"audit record {sequence}: missing run_id"
            )
        run_ids.add(run_id)
        ts_ms = float(record.get("ts_ms", -1.0))
        if not math.isfinite(ts_ms) or ts_ms < previous_ts:
            raise RuntimeTraceValidationError(
                f"audit record {sequence}: timestamp is invalid or non-monotonic"
            )
        previous_ts = ts_ms
        event = str(record.get("event", ""))
        if event == "runtime_initialized":
            initialized += 1
        elif event == "runtime_event_delivery":
            delivery_count += 1
            if record.get("accepted") is not True or record.get("error"):
                raise RuntimeTraceValidationError(
                    f"runtime event delivery failed at audit record {sequence}"
                )
        elif event == "runtime_event_log_error":
            raise RuntimeTraceValidationError(
                f"authoritative event log failed at audit record {sequence}"
            )
        elif event == "request_started":
            request_id = str(record.get("request_id", ""))
            if not request_id or request_id in started:
                raise RuntimeTraceValidationError(
                    f"invalid duplicate request_started at audit record {sequence}"
                )
            started.add(request_id)
        elif event == "request_finished":
            request_id = str(record.get("request_id", ""))
            if request_id not in started or request_id in finished:
                raise RuntimeTraceValidationError(
                    f"invalid request_finished at audit record {sequence}"
                )
            finished.add(request_id)

    if len(run_ids) != 1:
        raise RuntimeTraceValidationError(
            f"audit contains multiple run IDs: {sorted(run_ids)}"
        )
    if initialized != 1:
        raise RuntimeTraceValidationError(
            f"audit must contain one runtime_initialized record, got {initialized}"
        )
    if started != finished or started != expected_request_ids:
        raise RuntimeTraceValidationError(
            "audit/trace LLM request mismatch: "
            f"trace={sorted(expected_request_ids)}, started={sorted(started)}, "
            f"finished={sorted(finished)}"
        )
    if delivery_count == 0:
        raise RuntimeTraceValidationError(
            "audit contains no external runtime event deliveries"
        )
    return len(records), delivery_count


def _required_id(value: str | None, field: str, event: RuntimeEvent) -> str:
    if value is None or not str(value):
        raise RuntimeTraceValidationError(
            f"event {event.event_id}: missing {field}"
        )
    return str(value)


def _attribute_id(event: RuntimeEvent, field: str) -> str:
    value = event.attributes.get(field)
    if value is None or not str(value):
        raise RuntimeTraceValidationError(
            f"event {event.event_id}: missing attributes.{field}"
        )
    return str(value)


def _nonnegative_number(event: RuntimeEvent, field: str) -> float:
    value = float(event.attributes.get(field, -1.0))
    if not math.isfinite(value) or value < 0:
        raise RuntimeTraceValidationError(
            f"event {event.event_id}: attributes.{field} must be non-negative"
        )
    return value


def _nonnegative_integer(event: RuntimeEvent, field: str) -> int:
    raw = event.attributes.get(field)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise RuntimeTraceValidationError(
            f"event {event.event_id}: attributes.{field} must be a non-negative integer"
        )
    return raw


def _require_same_owner(
    start: RuntimeEvent,
    end: RuntimeEvent,
    kind: str,
    identity: str,
) -> None:
    if (
        start.workflow_id != end.workflow_id
        or start.invocation_id != end.invocation_id
        or start.context_id != end.context_id
    ):
        raise RuntimeTraceValidationError(
            f"{kind} ownership changed between start and end: {identity}"
        )
