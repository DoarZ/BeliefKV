from __future__ import annotations

from beliefkv.core.types import ContinuationBelief, WorkflowState


class WorkflowStateStore:
    def __init__(self):
        self._states: dict[str, WorkflowState] = {}
        self._frontiers: dict[str, list[ContinuationBelief]] = {}

    def get_or_create(self, workflow_id: str) -> WorkflowState:
        if workflow_id not in self._states:
            self._states[workflow_id] = WorkflowState(workflow_id=workflow_id)
        return self._states[workflow_id]

    def update_frontier(
        self, workflow_id: str, continuations: list[ContinuationBelief]
    ) -> None:
        self.get_or_create(workflow_id)
        self._frontiers[workflow_id] = continuations

    def all_continuations(self) -> list[ContinuationBelief]:
        result: list[ContinuationBelief] = []
        for continuations in self._frontiers.values():
            result.extend(continuations)
        return result
