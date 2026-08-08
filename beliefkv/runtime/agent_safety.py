from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal


class ActivationDeadlineExceeded(TimeoutError):
    """Raised before runtime work that has no workflow budget left."""


class ActivationDeadline:
    """Thread-safe absolute deadline shared by a workflow and its descendants."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._deadline: float | None = None

    def start(self, budget_s: float) -> None:
        if not math.isfinite(budget_s) or budget_s <= 0:
            raise ValueError("workflow deadline budget must be positive")
        now = self._clock()
        with self._lock:
            if self._deadline is not None:
                raise RuntimeError("workflow deadline is already active")
            self._started_at = now
            self._deadline = now + budget_s

    def clear(self) -> None:
        with self._lock:
            self._started_at = None
            self._deadline = None

    def elapsed_s(self) -> float | None:
        now = self._clock()
        with self._lock:
            if self._started_at is None:
                return None
            return max(0.0, now - self._started_at)

    def remaining_s(self) -> float | None:
        now = self._clock()
        with self._lock:
            if self._deadline is None:
                return None
            return max(0.0, self._deadline - now)

    def expired(self) -> bool:
        remaining = self.remaining_s()
        return remaining is not None and remaining <= 0

    def request_timeout_s(self, cap_s: float) -> float:
        if not math.isfinite(cap_s) or cap_s <= 0:
            raise ValueError("single-request timeout cap must be positive")
        remaining = self.remaining_s()
        if remaining is None:
            return cap_s
        if remaining <= 0:
            raise ActivationDeadlineExceeded(
                "workflow wall-clock deadline expired before model submission"
            )
        return min(cap_s, remaining)


@dataclass(frozen=True)
class ToolOutcome:
    status: Literal["success", "error"]
    error_class: str | None = None


def _content_fragments(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        fragments: list[str] = []
        for key, nested in value.items():
            if str(key).lower() in {"error", "message", "text", "content", "output"}:
                fragments.extend(_content_fragments(nested))
        return tuple(fragments) or (str(value),)
    if isinstance(value, (list, tuple)):
        return tuple(
            fragment for nested in value for fragment in _content_fragments(nested)
        )
    return (str(value),)


def classify_tool_outcome(
    output: Any,
    *,
    tool_name: str | None = None,
    error: BaseException | None = None,
) -> ToolOutcome:
    """Classify a tool observation without changing what the model receives."""

    if error is not None:
        error_name = type(error).__name__.lower()
        error_class = "timeout" if "timeout" in error_name else "exception"
        return ToolOutcome(status="error", error_class=error_class)

    status = getattr(output, "status", None)
    exit_code = (
        output.get("exit_code")
        if isinstance(output, dict)
        else getattr(output, "exit_code", None)
    )
    content = getattr(output, "content", output)
    normalized = tuple(
        fragment.strip().lower() for fragment in _content_fragments(content)
    )
    joined = "\n".join(normalized)
    classifications = (
        ("duplicate_suppressed", ("duplicate_suppressed",)),
        ("permission_denied", ("permission denied",)),
        ("path_not_found", ("path_not_found", "file not found", "no such file")),
        ("string_not_found", ("string not found", "old_string", "not found in file")),
        ("multiple_matches", ("appears multiple times", "multiple occurrences")),
        ("timeout", ("timed out", "timeout", "exceeded host timeout")),
        ("command_failed", ("command failed with exit code", "killed by signal")),
        ("validation_error", ("validation error", "invalid arguments")),
    )
    explicit_error = status == "error" or (
        exit_code is not None and exit_code != 0
    ) or any(
        item.startswith(
            (
                "error:",
                "tool error",
                "runtime duplicate circuit breaker:",
                "path_not_found",
                "permission denied",
                "traceback (most recent call last)",
            )
        )
        for item in normalized
    )
    if tool_name == "execute":
        explicit_error |= any(
            marker in joined
            for marker in (
                "command failed with exit code",
                "command exceeded host timeout",
                "killed by signal",
            )
        )
    if not explicit_error:
        return ToolOutcome(status="success")
    if exit_code is not None and exit_code != 0:
        return ToolOutcome(status="error", error_class="command_failed")
    for error_class, markers in classifications:
        if any(marker in joined for marker in markers):
            return ToolOutcome(status="error", error_class=error_class)
    return ToolOutcome(status="error", error_class="tool_error")
