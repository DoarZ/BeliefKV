from __future__ import annotations

import json

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import AIMessage

from beliefkv.experiments.agentic_peer_backend import (
    AgenticPeerBackendConfig,
    AgenticPeerDecision,
    count_accepted_task_calls,
    summarize_agentic_runtime_trace,
)


def test_agentic_decision_enforces_handoff_terminal_shape() -> None:
    with pytest.raises(ValueError):
        AgenticPeerDecision(
            summary="invalid",
            next_role="reviewer",
            complete=True,
        )
    with pytest.raises(ValueError):
        AgenticPeerDecision(
            summary="invalid",
            next_role=None,
            complete=False,
        )


def test_required_initial_subagent_range_is_validated() -> None:
    config = AgenticPeerBackendConfig(
        model="model",
        base_url="http://localhost:18000/v1",
        required_initial_subagent_min=2,
        required_initial_subagent_max=4,
    )
    assert config.required_initial_subagent_min == 2
    assert config.required_initial_subagent_max == 4

    with pytest.raises(ValueError, match="required initial subagent range"):
        AgenticPeerBackendConfig(
            model="model",
            base_url="http://localhost:18000/v1",
            required_initial_subagent_min=3,
            required_initial_subagent_max=2,
        )
    with pytest.raises(ValueError, match="disabled subagents"):
        AgenticPeerBackendConfig(
            model="model",
            base_url="http://localhost:18000/v1",
            enable_subagents=False,
            required_initial_subagent_min=2,
            required_initial_subagent_max=4,
        )


def test_count_accepted_task_calls_filters_unknown_types() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"subagent_type": "repository-explorer"},
                    "id": "accepted-1",
                },
                {
                    "name": "task",
                    "args": {"subagent_type": "invariant-auditor"},
                    "id": "accepted-2",
                },
                {
                    "name": "task",
                    "args": {"subagent_type": "general-purpose"},
                    "id": "rejected",
                },
                {"name": "read_file", "args": {"path": "/a"}, "id": "ordinary"},
            ],
        )
    ]

    assert count_accepted_task_calls(messages) == 2


def test_agentic_trace_summary_requires_multiturn_returned_children(tmp_path) -> None:
    path = tmp_path / "runtime.jsonl"
    records = [
        {
            "kind": "invocation_create",
            "invocation_id": "child",
            "parent_invocation_id": "parent",
            "attributes": {"persistent": True},
        },
        {
            "kind": "spawn",
            "invocation_id": "parent",
            "target_invocation_id": "child",
            "ts_ms": 15.0,
            "attributes": {},
        },
        {"kind": "join_create", "join_id": "join", "attributes": {}},
        {
            "kind": "llm_submit",
            "invocation_id": "child",
            "context_epoch": 0,
            "attributes": {},
        },
        {
            "kind": "llm_result",
            "invocation_id": "child",
            "attributes": {"rejected_task_call_count": 1},
        },
        {
            "kind": "tool_start",
            "invocation_id": "child",
            "attributes": {"tool_name": "read_file"},
        },
        {
            "kind": "tool_end",
            "invocation_id": "child",
            "attributes": {"tool_name": "read_file", "duration_ms": 4.0},
        },
        {
            "kind": "llm_submit",
            "invocation_id": "child",
            "context_epoch": 1,
            "attributes": {},
        },
        {"kind": "return", "invocation_id": "child", "attributes": {}},
        {"kind": "join_satisfied", "join_id": "join", "attributes": {}},
    ]
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )

    summary = summarize_agentic_runtime_trace(path)

    assert summary["dynamic_subagent_count"] == 1
    assert summary["multi_turn_subagent_count"] == 1
    assert summary["all_subagents_returned"] is True
    assert summary["all_joins_satisfied"] is True
    assert summary["rejected_task_call_count"] == 1
    assert summary["tool_name_counts"] == {"read_file": 1}
    assert summary["spawn_timestamps_ms"] == [15.0]
    assert summary["child_invocations"] == [
        {
            "invocation_id": "child",
            "llm_request_count": 2,
            "tool_call_count": 1,
            "max_context_epoch": 1,
            "returned": True,
        }
    ]
