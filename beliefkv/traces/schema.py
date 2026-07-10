from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class TraceEventKind(str, Enum):
    WORKFLOW_START = "workflow_start"
    AGENT_START = "agent_start"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    WORKFLOW_END = "workflow_end"


@dataclass(frozen=True)
class TraceEvent:
    ts_ms: float
    kind: TraceEventKind
    workflow_id: str
    agent_id: Optional[str] = None
    branch_id: Optional[str] = None
    token_count: int = 0
    payload: dict[str, Any] | None = None
