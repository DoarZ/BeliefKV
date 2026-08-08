from __future__ import annotations

from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.runtime.action_frontier import (
    ActionFrontierObserver,
    JsonActionParser,
    ParserStatus,
    StructuredActionKind,
    characterize_action_frontier_coverage,
)


def _event(
    sequence: int,
    kind: RuntimeEventKind,
    *,
    invocation_id: str = "coder",
    target_invocation_id: str | None = None,
    attributes: dict[str, object] | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"event-{sequence}",
        ts_ms=float(sequence * 10),
        kind=kind,
        workflow_id="workflow",
        invocation_id=invocation_id,
        target_invocation_id=target_invocation_id,
        context_id="context-coder",
        attributes=attributes or {},
    )


def test_incremental_json_parser_reports_first_real_boundary_token() -> None:
    parser = JsonActionParser()

    partial = parser.feed('{"action":"tool",', generated_tokens=4)
    valid = parser.feed('"name":"search","arguments":{}}', generated_tokens=11)
    later = parser.feed("   ", generated_tokens=12)

    assert partial.status == ParserStatus.INCOMPLETE
    assert partial.boundary_token_index is None
    assert valid.status == ParserStatus.VALID
    assert valid.action_kind == StructuredActionKind.FUNCTION_CALL
    assert valid.boundary_token_index == 11
    assert later.boundary_token_index == 11


def test_free_text_is_unknown_and_never_gets_a_fabricated_boundary() -> None:
    update = JsonActionParser().feed("I think the answer is...", generated_tokens=6)

    assert update.status == ParserStatus.UNKNOWN
    assert update.action_kind == StructuredActionKind.UNKNOWN
    assert update.boundary_token_index is None


def test_runtime_terminal_valid_action_keeps_boundary_unknown() -> None:
    observer = ActionFrontierObserver()
    observer.observe_runtime_event(
        _event(
            1,
            RuntimeEventKind.LLM_SUBMIT,
            attributes={"request_id": "request", "output_tokens": 0},
        ),
        runnable_frontier_before=("coder", "reviewer"),
        context_gpu_bytes=1024,
    )
    state = observer.observe_runtime_event(
        _event(
            2,
            RuntimeEventKind.LLM_RESULT,
            attributes={
                "request_id": "request",
                "output_tokens": 20,
                "parser_status": "valid",
                "structured_action_kinds": ["function_call"],
                "structured_action_names": ["search"],
                "action_boundary_token_index": None,
                "action_boundary_source": "runtime_structured_output",
            },
        )
    )

    assert state is not None
    assert state.parser_status == ParserStatus.VALID
    assert state.valid_action_ts_ms == 20.0
    assert state.boundary_token_index is None
    assert state.boundary_source == "runtime_structured_output"


def test_observer_records_tool_gap_frontier_delta_kv_transition_and_reentry() -> None:
    observer = ActionFrontierObserver()
    observer.observe_runtime_event(
        _event(
            1,
            RuntimeEventKind.LLM_SUBMIT,
            attributes={"request_id": "request"},
        ),
        runnable_frontier_before=("coder", "reviewer"),
        context_gpu_bytes=4096,
    )
    observer.observe_runtime_event(
        _event(
            2,
            RuntimeEventKind.LLM_RESULT,
            attributes={
                "output_tokens": 8,
                "parser_status": "valid",
                "structured_action_kinds": ["function_call"],
                "structured_action_names": ["run_tests"],
            },
        )
    )
    action = observer.observe_runtime_event(
        _event(3, RuntimeEventKind.TOOL_START),
        runnable_frontier_after=("reviewer",),
        context_gpu_bytes=3072,
    )
    reentered = observer.observe_runtime_event(
        _event(8, RuntimeEventKind.TOOL_END),
        runnable_frontier_after=("coder", "reviewer"),
        context_gpu_bytes=3072,
    )

    assert action is not None
    assert action.tool_start_gap_ms == 10.0
    assert action.frontier_added == ()
    assert action.frontier_removed == ("coder",)
    assert action.active_kv_bytes_before == 4096
    assert action.waiting_kv_bytes_after == 3072
    assert reentered is not None
    assert reentered.reentry_delay_ms == 50.0
    assert {item.kind for item in observer.drain_audit_events()} == {
        "action_frontier_updated",
        "valid_action_unlocked",
    }


def test_p6_coverage_separates_exact_and_runtime_only_boundaries() -> None:
    observer = ActionFrontierObserver()
    observer.begin(
        request_id="exact",
        workflow_id="workflow",
        invocation_id="exact-invocation",
        context_id="exact-context",
        ts_ms=0.0,
    )
    observer.observe_parser_update(
        "exact",
        JsonActionParser().feed(
            '{"action":"tool","name":"search"}', generated_tokens=8
        ),
        ts_ms=1.0,
    )
    observer.begin(
        request_id="runtime-only",
        workflow_id="workflow",
        invocation_id="runtime-invocation",
        context_id="runtime-context",
        ts_ms=2.0,
    )
    observer.observe_runtime_event(
        RuntimeEvent(
            event_id="runtime-result",
            ts_ms=3.0,
            kind=RuntimeEventKind.LLM_RESULT,
            workflow_id="workflow",
            invocation_id="runtime-invocation",
            context_id="runtime-context",
            attributes={
                "request_id": "runtime-only",
                "output_tokens": 12,
                "parser_status": "valid",
                "structured_action_kinds": ["function_call"],
                "structured_action_names": ["read_file"],
            },
        )
    )

    coverage = characterize_action_frontier_coverage(observer.snapshots())

    assert observer.revision == 4
    assert coverage.action_call_count == 2
    assert coverage.exact_boundary_call_count == 1
    assert coverage.runtime_only_boundary_call_count == 1
    assert coverage.exact_boundary_call_coverage == 0.5


def test_internal_summary_is_not_an_agent_action() -> None:
    observer = ActionFrontierObserver()

    assert observer.observe_runtime_event(
        _event(
            1,
            RuntimeEventKind.LLM_SUBMIT,
            attributes={"request_id": "summary", "runtime_internal": True},
        )
    ) is None
    assert observer.observe_runtime_event(
        _event(
            2,
            RuntimeEventKind.LLM_RESULT,
            attributes={
                "runtime_internal": True,
                "parser_status": "valid",
                "structured_action_kinds": ["final_answer"],
                "output_tokens": 16,
            },
        )
    ) is None

    assert observer.snapshots() == ()
    assert observer.coverage().call_count == 0


def test_join_satisfied_is_attributed_to_the_waiting_parent() -> None:
    observer = ActionFrontierObserver()
    observer.observe_runtime_event(
        _event(
            1,
            RuntimeEventKind.LLM_SUBMIT,
            invocation_id="parent",
            attributes={"request_id": "spawn-call"},
        )
    )
    observer.observe_runtime_event(
        _event(
            2,
            RuntimeEventKind.LLM_RESULT,
            invocation_id="parent",
            attributes={
                "parser_status": "valid",
                "structured_action_kinds": ["spawn"],
                "structured_action_names": ["task"],
                "output_tokens": 8,
            },
        )
    )
    observer.observe_runtime_event(
        RuntimeEvent(
            event_id="join-wait",
            ts_ms=30.0,
            kind=RuntimeEventKind.JOIN_WAIT,
            workflow_id="workflow",
            invocation_id="parent",
            join_id="join",
        )
    )
    state = observer.observe_runtime_event(
        RuntimeEvent(
            event_id="join-satisfied",
            ts_ms=80.0,
            kind=RuntimeEventKind.JOIN_SATISFIED,
            workflow_id="workflow",
            join_id="join",
        )
    )

    assert state is not None
    assert state.reentry_ts_ms == 80.0
    coverage = observer.coverage()
    assert coverage.reentry_eligible_call_count == 1
    assert coverage.reentry_observed_count == 1
    assert coverage.reentry_cause_coverage == 1.0
