from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.runtime.page_index import PageOwnershipIndex, PhysicalPageRecord


class ResidencyClass(str, Enum):
    DEAD_UNOWNED = "dead_unowned"
    PARKED = "parked"
    IMMINENT = "imminent"
    PINNED = "pinned"


_CLASS_RANK = {
    ResidencyClass.DEAD_UNOWNED: 0,
    ResidencyClass.PARKED: 1,
    ResidencyClass.IMMINENT: 2,
    ResidencyClass.PINNED: 3,
}


@dataclass(frozen=True)
class ResidencyAssessment:
    residency_class: ResidencyClass
    reason: str
    since_ms: float


class ResidencyClassifier:
    def __init__(
        self, graph: RuntimeCausalContextGraph, page_index: PageOwnershipIndex
    ) -> None:
        self.graph = graph
        self.page_index = page_index

    def context(self, context_id: str, now_ms: float) -> ResidencyAssessment:
        context = self.graph.contexts.get(context_id)
        if context is None:
            return ResidencyAssessment(
                ResidencyClass.DEAD_UNOWNED, "missing_context", now_ms
            )
        workflow = self.graph.workflows.get(context.workflow_id)
        if workflow is not None and workflow.end_ts_ms is not None:
            return ResidencyAssessment(
                ResidencyClass.DEAD_UNOWNED,
                "workflow_ended",
                workflow.end_ts_ms,
            )
        invocations = self.graph.context_invocations(context_id)
        if any(item.state == InvocationState.RUNNING_LLM for item in invocations):
            since = min(
                item.updated_ts_ms
                for item in invocations
                if item.state == InvocationState.RUNNING_LLM
            )
            return ResidencyAssessment(ResidencyClass.PINNED, "active_llm", since)
        if any(
            item.state == InvocationState.READY or item.pending_messages > 0
            for item in invocations
        ):
            since = min(
                item.updated_ts_ms
                for item in invocations
                if item.state == InvocationState.READY or item.pending_messages > 0
            )
            return ResidencyAssessment(ResidencyClass.IMMINENT, "causal_frontier", since)
        live = [item for item in invocations if not item.state.terminal]
        if live:
            since = min(item.updated_ts_ms for item in live)
            reasons = sorted({item.state.value for item in live})
            return ResidencyAssessment(
                ResidencyClass.PARKED, "+".join(reasons), since
            )
        if context.persistent:
            return ResidencyAssessment(
                ResidencyClass.PARKED, "persistent_inactive", context.updated_ts_ms
            )
        return ResidencyAssessment(
            ResidencyClass.DEAD_UNOWNED, "all_invocations_terminal", context.updated_ts_ms
        )

    def page(self, page: PhysicalPageRecord, now_ms: float) -> ResidencyAssessment:
        if page.engine_lock_ref > 0 or page.active_reader_count > 0:
            return ResidencyAssessment(ResidencyClass.PINNED, "engine_active", now_ms)
        if page.semantic_pin_contexts:
            return ResidencyAssessment(ResidencyClass.PINNED, "semantic_pin", now_ms)
        if not page.owner_contexts:
            return ResidencyAssessment(
                ResidencyClass.DEAD_UNOWNED, "no_semantic_owner", page.last_access_ms
            )
        assessments = [self.context(context_id, now_ms) for context_id in page.owner_contexts]
        strongest = max(assessments, key=lambda item: _CLASS_RANK[item.residency_class])
        return ResidencyAssessment(
            strongest.residency_class,
            f"strongest_owner:{strongest.reason}",
            strongest.since_ms,
        )

    def release_terminal_owners(self) -> set[str]:
        """Drop semantic ownership only for non-persistent dead contexts."""

        released: set[str] = set()
        for context_id in tuple(self.graph.contexts):
            assessment = self.context(context_id, now_ms=0.0)
            if assessment.residency_class == ResidencyClass.DEAD_UNOWNED:
                self.page_index.unbind_context(context_id)
                released.add(context_id)
        return released
