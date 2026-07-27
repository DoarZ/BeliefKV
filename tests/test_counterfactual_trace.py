from __future__ import annotations

import json
from pathlib import Path

import pytest

from beliefkv.core.events import (
    ContextMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.experiments.counterfactual_trace import (
    CounterfactualTraceBuilder,
    CounterfactualTraceError,
)
from beliefkv.runtime.audit import RequestTokenTraceLog
from beliefkv.simulator.queue_service import FrozenCounterfactualWorkload


def _event(
    sequence: int,
    ts_ms: float,
    kind: RuntimeEventKind,
    *,
    invocation_id: str | None = None,
    context_id: str | None = None,
    context_epoch: int | None = None,
    workflow_id: str = "workflow",
    **kwargs,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"event-{sequence}",
        ts_ms=ts_ms,
        kind=kind,
        workflow_id=workflow_id,
        invocation_id=invocation_id,
        context_id=context_id,
        context_epoch=context_epoch,
        **kwargs,
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def _simple_trace(tmp_path: Path) -> tuple[Path, Path]:
    events = [
        _event(1, 0, RuntimeEventKind.WORKFLOW_START),
        _event(
            2,
            0,
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="root",
            context_id="context-root",
            context_epoch=0,
            relation_type=RelationType.ROOT,
            context_mode=ContextMode.FRESH,
        ),
        _event(
            3,
            1,
            RuntimeEventKind.LLM_SUBMIT,
            invocation_id="root",
            context_id="context-root",
            context_epoch=0,
            attributes={
                "request_id": "request-1",
                "prompt_tokens": 10,
                "cache_hit_tokens": 0,
            },
        ),
        _event(
            4,
            3,
            RuntimeEventKind.LLM_RESULT,
            invocation_id="root",
            context_id="context-root",
            context_epoch=0,
            attributes={
                "request_id": "request-1",
                "output_tokens": 2,
                "structured_action_kinds": ["function_call"],
                "action_boundary_token_index": None,
            },
        ),
        _event(
            5,
            3.2,
            RuntimeEventKind.TOOL_START,
            invocation_id="root",
            attributes={
                "tool_call_id": "tool-1",
                "tool_name": "read",
                "beliefkv_event_time_adjusted": True,
                "beliefkv_source_ts_ms": 3.1,
            },
        ),
        _event(
            6,
            8.1,
            RuntimeEventKind.TOOL_END,
            invocation_id="root",
            attributes={
                "tool_call_id": "tool-1",
                "tool_name": "read",
                "duration_ms": 5.0,
            },
        ),
        _event(
            7,
            10,
            RuntimeEventKind.LLM_SUBMIT,
            invocation_id="root",
            context_id="context-root",
            context_epoch=1,
            attributes={
                "request_id": "request-2",
                "prompt_tokens": 15,
                "cache_hit_tokens": 12,
            },
        ),
        _event(
            8,
            12,
            RuntimeEventKind.LLM_RESULT,
            invocation_id="root",
            context_id="context-root",
            context_epoch=1,
            attributes={
                "request_id": "request-2",
                "output_tokens": 1,
                "structured_action_kinds": ["final_answer"],
                "action_boundary_token_index": None,
            },
        ),
        _event(
            9,
            12.1,
            RuntimeEventKind.RETURN,
            invocation_id="root",
            context_id="context-root",
        ),
        _event(10, 13, RuntimeEventKind.WORKFLOW_END),
    ]
    runtime = tmp_path / "runtime.jsonl"
    _write_jsonl(runtime, [item.to_dict() for item in events])
    audit = tmp_path / "audit.jsonl"
    _write_jsonl(
        audit,
        [
            {"event": "runtime_initialized", "kv_bytes_per_token": 100, "ts_ms": 0},
            {"event": "request_deferred", "request_id": "request-1", "ts_ms": 1},
            {"event": "request_started", "request_id": "request-1", "ts_ms": 1.2},
            {"event": "request_finished", "request_id": "request-1", "ts_ms": 3},
            {"event": "request_deferred", "request_id": "request-2", "ts_ms": 9},
            {"event": "request_started", "request_id": "request-2", "ts_ms": 10},
            {"event": "request_finished", "request_id": "request-2", "ts_ms": 12},
        ],
    )
    return runtime, audit


def test_builder_freezes_causal_delay_but_excludes_original_queue_wait(
    tmp_path: Path,
) -> None:
    runtime, audit = _simple_trace(tmp_path)

    result = CounterfactualTraceBuilder().build(runtime, audit)
    requests = {item.request_id: item for item in result.workload.requests}

    assert result.request_count == 2
    assert result.tool_interval_count == 1
    assert requests["request-1"].release_delay_ms == 1
    assert requests["request-2"].predecessor_request_ids == ("request-1",)
    assert requests["request-2"].release_delay_ms == 6
    assert requests["request-2"].tool_duration_ms == 5
    assert requests["request-2"].uncached_prompt_tokens == 3
    assert requests["request-2"].startup_bytes == 400
    assert result.arrival_source_counts == {"request_deferred": 2}
    assert result.workload.metadata["observed_request_order"] == [
        "request-1",
        "request-2",
    ]
    assert not result.workload.future_physical_growth_exact
    assert FrozenCounterfactualWorkload.from_dict(
        result.workload.to_dict()
    ) == result.workload


def test_builder_adds_spawn_and_join_all_dependencies(tmp_path: Path) -> None:
    events = [
        _event(1, 0, RuntimeEventKind.WORKFLOW_START),
        _event(
            2,
            0,
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="parent",
            context_id="parent-context",
            context_epoch=0,
            relation_type=RelationType.ROOT,
            context_mode=ContextMode.FRESH,
        ),
        _event(
            3,
            0.5,
            RuntimeEventKind.LLM_SUBMIT,
            invocation_id="parent",
            context_id="parent-context",
            context_epoch=0,
            attributes={
                "request_id": "parent-1",
                "prompt_tokens": 2,
                "cache_hit_tokens": 0,
            },
        ),
        _event(
            4,
            2,
            RuntimeEventKind.LLM_RESULT,
            invocation_id="parent",
            context_id="parent-context",
            context_epoch=0,
            attributes={"request_id": "parent-1", "output_tokens": 1},
        ),
        _event(
            5,
            2.1,
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="child",
            context_id="child-context",
            context_epoch=0,
            parent_invocation_id="parent",
            parent_context_id="parent-context",
            relation_type=RelationType.SPAWN,
            context_mode=ContextMode.FRESH,
        ),
        _event(
            6,
            2.1,
            RuntimeEventKind.JOIN_CREATE,
            join_id="join",
            member_invocation_ids=("child",),
        ),
        _event(
            7,
            2.1,
            RuntimeEventKind.JOIN_WAIT,
            invocation_id="parent",
            join_id="join",
        ),
        _event(
            8,
            3,
            RuntimeEventKind.LLM_SUBMIT,
            invocation_id="child",
            context_id="child-context",
            context_epoch=0,
            attributes={
                "request_id": "child-1",
                "prompt_tokens": 2,
                "cache_hit_tokens": 0,
            },
        ),
        _event(
            9,
            4,
            RuntimeEventKind.LLM_RESULT,
            invocation_id="child",
            context_id="child-context",
            context_epoch=0,
            attributes={"request_id": "child-1", "output_tokens": 1},
        ),
        _event(
            10,
            4.1,
            RuntimeEventKind.RETURN,
            invocation_id="child",
            context_id="child-context",
        ),
        _event(11, 4.1, RuntimeEventKind.JOIN_SATISFIED, join_id="join"),
        _event(
            12,
            5,
            RuntimeEventKind.LLM_SUBMIT,
            invocation_id="parent",
            context_id="parent-context",
            context_epoch=1,
            attributes={
                "request_id": "parent-2",
                "prompt_tokens": 4,
                "cache_hit_tokens": 3,
            },
        ),
        _event(
            13,
            6,
            RuntimeEventKind.LLM_RESULT,
            invocation_id="parent",
            context_id="parent-context",
            context_epoch=1,
            attributes={"request_id": "parent-2", "output_tokens": 1},
        ),
        _event(
            14,
            6.1,
            RuntimeEventKind.RETURN,
            invocation_id="parent",
            context_id="parent-context",
        ),
        _event(15, 7, RuntimeEventKind.WORKFLOW_END),
    ]
    runtime = tmp_path / "runtime.jsonl"
    _write_jsonl(runtime, [item.to_dict() for item in events])
    audit = tmp_path / "audit.jsonl"
    records: list[dict[str, object]] = [
        {"event": "runtime_initialized", "kv_bytes_per_token": 100, "ts_ms": 0}
    ]
    for request_id, arrival, finish in (
        ("parent-1", 0.5, 2),
        ("child-1", 3, 4),
        ("parent-2", 5, 6),
    ):
        records.extend(
            (
                {"event": "request_deferred", "request_id": request_id, "ts_ms": arrival},
                {"event": "request_finished", "request_id": request_id, "ts_ms": finish},
            )
        )
    _write_jsonl(audit, records)

    result = CounterfactualTraceBuilder().build(
        runtime,
        audit,
        exact_kv_growth_bytes_by_request={
            "parent-1": 300,
            "child-1": 300,
            "parent-2": 200,
        },
    )
    requests = {item.request_id: item for item in result.workload.requests}

    assert requests["child-1"].predecessor_request_ids == ("parent-1",)
    assert requests["parent-2"].predecessor_request_ids == (
        "child-1",
        "parent-1",
    )
    assert result.semantic_edge_count == 2
    assert result.workload.future_physical_growth_exact


def test_builder_rejects_incomplete_exact_growth_mapping(tmp_path: Path) -> None:
    runtime, audit = _simple_trace(tmp_path)

    with pytest.raises(CounterfactualTraceError, match="cover"):
        CounterfactualTraceBuilder().build(
            runtime,
            audit,
            exact_kv_growth_bytes_by_request={"request-1": 1_200},
        )


def test_builder_attaches_exact_anonymized_prefix_identity(tmp_path: Path) -> None:
    runtime, audit = _simple_trace(tmp_path)
    token_trace = tmp_path / "request_tokens.jsonl.gz"
    first_prompt = list(range(10))
    first_commit = [*first_prompt, 10]
    second_prompt = [*first_commit, 11, 12, 13, 14]
    with RequestTokenTraceLog(token_trace, run_id="trace-run") as trace:
        trace.emit("cache_reset", 0, ())
        trace.emit(
            "request_prompt",
            1,
            first_prompt,
            request_id="request-1",
        )
        trace.emit(
            "cache_final_commit",
            3,
            first_commit,
            request_id="request-1",
        )
        trace.emit(
            "request_prompt",
            9,
            second_prompt,
            request_id="request-2",
        )
        trace.emit(
            "cache_final_commit",
            12,
            second_prompt,
            request_id="request-2",
        )

    result = CounterfactualTraceBuilder().build(
        runtime,
        audit,
        request_token_trace_path=token_trace,
    )
    requests = {item.request_id: item for item in result.workload.requests}

    assert result.request_prefix_identity_coverage == 1
    assert result.workload.prefix_identity_complete
    assert result.workload.initial_radix_state_known
    assert not result.workload.future_physical_growth_exact
    assert (
        requests["request-1"].cache_commit_token_symbols
        == requests["request-2"].prompt_token_symbols[:11]
    )
    restored = FrozenCounterfactualWorkload.from_dict(result.workload.to_dict())
    assert restored == result.workload


def test_builder_slices_one_workflow_and_its_latest_token_segment(
    tmp_path: Path,
) -> None:
    runtime, audit = _simple_trace(tmp_path)
    with runtime.open("a", encoding="utf-8") as stream:
        for event in (
            _event(20, 20, RuntimeEventKind.WORKFLOW_START, workflow_id="other"),
            _event(
                21,
                21,
                RuntimeEventKind.LLM_SUBMIT,
                workflow_id="other",
                invocation_id="other-invocation",
                context_id="other-context",
                context_epoch=0,
                attributes={
                    "request_id": "other-request",
                    "prompt_tokens": 1,
                    "cache_hit_tokens": 0,
                },
            ),
            _event(
                22,
                22,
                RuntimeEventKind.LLM_RESULT,
                workflow_id="other",
                invocation_id="other-invocation",
                context_id="other-context",
                context_epoch=0,
                attributes={"request_id": "other-request", "output_tokens": 1},
            ),
            _event(23, 23, RuntimeEventKind.WORKFLOW_END, workflow_id="other"),
        ):
            stream.write(json.dumps(event.to_dict()) + "\n")
    with audit.open("a", encoding="utf-8") as stream:
        for record in (
            {"event": "request_deferred", "request_id": "other-request", "ts_ms": 21},
            {"event": "request_finished", "request_id": "other-request", "ts_ms": 22},
        ):
            stream.write(json.dumps(record) + "\n")
    token_trace = tmp_path / "segmented_tokens.jsonl.gz"
    first_prompt = list(range(10))
    first_commit = [*first_prompt, 10]
    second_prompt = [*first_commit, 11, 12, 13, 14]
    with RequestTokenTraceLog(token_trace, run_id="trace-run") as trace:
        trace.emit("request_prompt", 0, [99], request_id="old-request")
        trace.emit("cache_final_commit", 1, [99], request_id="old-request")
        trace.emit("cache_reset", 2, ())
        trace.emit("request_prompt", 3, first_prompt, request_id="request-1")
        trace.emit(
            "cache_partial_commit",
            3.5,
            first_prompt,
            request_id="request-1",
            chunk_index=0,
        )
        trace.emit("cache_final_commit", 4, first_commit, request_id="request-1")
        trace.emit("request_prompt", 5, second_prompt, request_id="request-2")
        trace.emit("cache_final_commit", 6, second_prompt, request_id="request-2")
        trace.emit("request_prompt", 7, [100], request_id="other-request")
        trace.emit("cache_final_commit", 8, [100], request_id="other-request")

    result = CounterfactualTraceBuilder().build(
        runtime,
        audit,
        workflow_ids=("workflow",),
        request_token_trace_path=token_trace,
    )

    assert result.workflow_count == 1
    assert result.request_count == 2
    assert result.workload.initial_radix_state_known
    assert result.workload.metadata["selected_workflow_ids"] == ["workflow"]
    requests = {item.request_id: item for item in result.workload.requests}
    assert requests["request-1"].partial_cache_commit_token_symbols


def test_builder_rejects_unselected_token_interference_after_reset(
    tmp_path: Path,
) -> None:
    runtime, audit = _simple_trace(tmp_path)
    token_trace = tmp_path / "interfering_tokens.jsonl.gz"
    first_prompt = list(range(10))
    second_prompt = [*first_prompt, 10, 11, 12, 13, 14]
    with RequestTokenTraceLog(token_trace, run_id="trace-run") as trace:
        trace.emit("cache_reset", 0, ())
        trace.emit("request_prompt", 1, first_prompt, request_id="request-1")
        trace.emit(
            "cache_final_commit",
            3,
            [*first_prompt, 10],
            request_id="request-1",
        )
        trace.emit("request_prompt", 4, [99], request_id="interference")
        trace.emit("cache_final_commit", 5, [99], request_id="interference")
        trace.emit("request_prompt", 9, second_prompt, request_id="request-2")
        trace.emit("cache_final_commit", 12, second_prompt, request_id="request-2")

    with pytest.raises(CounterfactualTraceError, match="unselected requests"):
        CounterfactualTraceBuilder().build(
            runtime,
            audit,
            workflow_ids=("workflow",),
            request_token_trace_path=token_trace,
        )
