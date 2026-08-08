from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult

from beliefkv.runtime.context_lifecycle import (
    CONTEXT_LIFECYCLE_PRIVATE_STATE_KEYS,
    CompletionBudgetMiddleware,
    ContextLifecycleMiddleware,
    ContextLifecyclePolicy,
    ContextLifecycleState,
)


class _FakeModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "context-lifecycle-test"

    def _generate(self, messages, **kwargs: Any) -> ChatResult:
        del messages, kwargs
        raise AssertionError("the token accounting test must not invoke the model")


def test_context_lifecycle_defaults_to_32k_with_8k_retention() -> None:
    policy = ContextLifecyclePolicy()

    assert policy.window_tokens == 32_768
    assert policy.keep_tokens == 8_192
    assert policy.intermediate_output_tokens == 1_024
    assert policy.summary_output_tokens == 2_048


def test_context_lifecycle_rejects_invalid_retention() -> None:
    with pytest.raises(ValueError, match="keep_tokens"):
        ContextLifecyclePolicy(window_tokens=8_192, keep_tokens=8_192)


def test_32k_budget_excludes_static_system_prompt_and_tool_schema() -> None:
    model = _FakeModel()
    middleware = ContextLifecycleMiddleware(
        model,
        backend=SimpleNamespace(),
        policy=ContextLifecyclePolicy(),
        compaction_sink=SimpleNamespace(),
    )
    dynamic = [HumanMessage(content="x" * 4_000)]
    static = SystemMessage(content="y" * 80_000)
    tools = [
        {
            "type": "function",
            "function": {"name": "large", "description": "z" * 80_000},
        }
    ]

    dynamic_only = middleware._count_tokens(dynamic, None, None)
    with_static_schema = middleware._count_tokens(dynamic, static, tools)

    assert with_static_schema == dynamic_only
    assert with_static_schema < 32_768


def test_summarization_cursor_is_declared_as_private_graph_state() -> None:
    from deepagents.middleware._state import private_state_field_names

    assert "_summarization_event" in private_state_field_names(
        ContextLifecycleState
    )


def test_completion_budget_separates_tool_turns_from_finalization() -> None:
    final_mode = [False]
    middleware = CompletionBudgetMiddleware(
        intermediate_tokens=1_024,
        final_tokens=4_096,
        final_mode=lambda: final_mode[0],
    )
    request = ModelRequest(
        model=_FakeModel(),
        messages=[HumanMessage(content="continue")],
    )
    observed: list[int] = []

    def handler(current: ModelRequest) -> ModelResponse:
        observed.append(int(current.model_settings["max_tokens"]))
        return ModelResponse(result=[HumanMessage(content="ok")])

    middleware.wrap_model_call(request, handler)
    middleware.wrap_model_call(
        request.override(state={"guard_forcing_completion": True}), handler
    )
    final_mode[0] = True
    middleware.wrap_model_call(request, handler)

    assert observed == [1_024, 4_096, 4_096]


def test_context_lifecycle_rejects_foreign_out_of_bounds_summary_event() -> None:
    request = ModelRequest(
        model=_FakeModel(),
        messages=[HumanMessage(content="short child context")],
        state={
            "messages": [HumanMessage(content="short child context")],
            "_summarization_event": {
                "cutoff_index": 11,
                "summary_message": HumanMessage(content="parent summary"),
                "file_path": None,
            },
        },
    )

    sanitized = ContextLifecycleMiddleware._sanitize_summarization_event(request)

    assert request.state["_summarization_event"] is not None
    assert sanitized.state["_summarization_event"] is None
    assert CONTEXT_LIFECYCLE_PRIVATE_STATE_KEYS == frozenset(
        {"_summarization_event"}
    )
