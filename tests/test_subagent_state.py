from __future__ import annotations

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from beliefkv.runtime.subagent_state import (
    PrivateStateIsolatingSubAgentMiddleware,
    strip_private_state_update,
)


def test_child_private_state_is_removed_from_parent_command() -> None:
    message = ToolMessage(content="done", tool_call_id="call-1")
    command = Command(
        graph="parent",
        goto="next",
        update={
            "messages": [message],
            "shared_result": "kept",
            "_summarization_event": {"cutoff_index": 7},
        },
    )

    isolated = strip_private_state_update(
        command, frozenset({"_summarization_event"})
    )

    assert isolated is not command
    assert isolated.graph == "parent"
    assert isolated.goto == "next"
    assert isolated.update == {
        "messages": [message],
        "shared_result": "kept",
    }


def test_non_command_tool_results_are_unchanged() -> None:
    result = "child failed before producing graph state"

    assert strip_private_state_update(
        result, frozenset({"_summarization_event"})
    ) is result


def test_private_key_update_rebuild_keeps_task_output_isolation() -> None:
    middleware = PrivateStateIsolatingSubAgentMiddleware(
        backend=object(),
        private_state_keys=frozenset({"initial_private"}),
        subagents=[
            {
                "name": "child",
                "description": "test child",
                "runnable": RunnableLambda(lambda state: state),
            }
        ],
    )

    initial_func = middleware.tools[0].func
    middleware.private_state_keys = frozenset({"_summarization_event"})

    assert middleware.private_state_keys == frozenset(
        {"_summarization_event"}
    )
    assert middleware.tools[0].func is not initial_func
    assert middleware.tools[0].func is not None
    assert middleware.tools[0].coroutine is not None
