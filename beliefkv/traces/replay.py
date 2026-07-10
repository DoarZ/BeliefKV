from __future__ import annotations

from beliefkv.core.types import ContinuationBelief
from beliefkv.traces.schema import TraceEvent, TraceEventKind


def frontier_from_recent_events(events: list[TraceEvent]) -> list[ContinuationBelief]:
    """Build a conservative frontier from trace events.

    This is a bootstrap predictor for replay experiments. It should be replaced
    by richer template or learned predictors once traces are normalized.
    """

    frontier: list[ContinuationBelief] = []
    for event in events:
        if event.kind in {TraceEventKind.AGENT_START, TraceEventKind.LLM_START} and event.agent_id:
            frontier.append(
                ContinuationBelief(
                    workflow_id=event.workflow_id,
                    agent_id=event.agent_id,
                    branch_id=event.branch_id,
                    probability=1.0,
                    ready_time_p50_ms=max(0.0, event.ts_ms),
                    ready_time_p95_ms=max(0.0, event.ts_ms),
                    confidence=0.5,
                )
            )
    return frontier
