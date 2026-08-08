from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from typing import Any

from deepagents.middleware.subagents import SubAgentMiddleware
from langchain_core.tools import BaseTool
from langgraph.types import Command


def strip_private_state_update(
    result: Any,
    private_state_keys: frozenset[str],
) -> Any:
    """Prevent child-private graph state from being merged into its parent."""

    if not isinstance(result, Command) or not isinstance(result.update, Mapping):
        return result
    update = {
        key: value
        for key, value in result.update.items()
        if key not in private_state_keys
    }
    if len(update) == len(result.update):
        return result
    return replace(result, update=update)


def _isolate_tool_output(
    tool: BaseTool,
    private_state_keys: frozenset[str],
) -> None:
    func = getattr(tool, "func", None)
    if func is not None:

        @wraps(func)
        def isolated_func(*args: Any, **kwargs: Any) -> Any:
            return strip_private_state_update(
                func(*args, **kwargs), private_state_keys
            )

        tool.func = isolated_func

    coroutine = getattr(tool, "coroutine", None)
    if coroutine is not None:

        @wraps(coroutine)
        async def isolated_coroutine(*args: Any, **kwargs: Any) -> Any:
            return strip_private_state_update(
                await coroutine(*args, **kwargs), private_state_keys
            )

        tool.coroutine = isolated_coroutine


class PrivateStateIsolatingSubAgentMiddleware(SubAgentMiddleware):
    """Deep Agents task middleware with symmetric private-state isolation.

    Deep Agents 0.6.12 strips ``private_state_keys`` from child input, but its
    task return path can still copy child-private fields into ``Command.update``.
    That makes concurrent children write the same parent state channel. This
    wrapper also strips private fields at the child-to-parent boundary.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._isolate_current_tool()

    @property
    def private_state_keys(self) -> frozenset[str]:
        getter = SubAgentMiddleware.private_state_keys.fget
        assert getter is not None
        return getter(self)

    @private_state_keys.setter
    def private_state_keys(self, value: frozenset[str]) -> None:
        setter = SubAgentMiddleware.private_state_keys.fset
        assert setter is not None
        setter(self, value)
        self._isolate_current_tool()

    def _isolate_current_tool(self) -> None:
        for tool in self.tools:
            _isolate_tool_output(tool, self.private_state_keys)
