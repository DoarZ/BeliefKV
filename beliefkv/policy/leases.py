from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from beliefkv.control.causal_graph import (
    InvocationRecord,
    InvocationState,
    RuntimeCausalContextGraph,
)


class LeaseKind(str, Enum):
    DEAD = "dead"
    SPECULATIVE = "speculative"
    CONDITIONAL_RESUME = "conditional_resume"
    READY = "ready"
    RUNNING = "running"


_LEASE_STRENGTH = {
    LeaseKind.DEAD: 0,
    LeaseKind.SPECULATIVE: 1,
    LeaseKind.CONDITIONAL_RESUME: 2,
    LeaseKind.READY: 3,
    LeaseKind.RUNNING: 4,
}


@dataclass(frozen=True, order=True)
class LeaseCondition:
    event_kind: str
    subject_id: str
    condition_id: str


@dataclass(frozen=True)
class ContextLease:
    context_id: str
    context_epoch: int
    workflow_id: str
    kind: LeaseKind
    condition: LeaseCondition | None
    issued_ts_ms: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class BundleLease:
    bundle_id: str
    owner_context_ids: tuple[str, ...]
    strongest_kind: LeaseKind
    conditions: tuple[LeaseCondition, ...]
    scenario_support: frozenset[str] = frozenset()


class CausalLeaseProjector:
    """Project observed RCCG state into finite resource commitments."""

    def __init__(self, graph: RuntimeCausalContextGraph) -> None:
        self.graph = graph

    def context(self, context_id: str, *, now_ms: float) -> ContextLease:
        if now_ms < 0:
            raise ValueError("lease timestamp must be non-negative")
        context = self.graph.contexts.get(context_id)
        if context is None:
            # Unknown ownership is a safety condition, not dead cache.
            return ContextLease(
                context_id=context_id,
                context_epoch=0,
                workflow_id="unknown",
                kind=LeaseKind.RUNNING,
                condition=None,
                issued_ts_ms=now_ms,
                confidence=1.0,
                reason="missing_context_safety_pin",
            )
        workflow = self.graph.workflows.get(context.workflow_id)
        if workflow is not None and workflow.end_ts_ms is not None:
            return self._lease(
                context_id,
                LeaseKind.DEAD,
                now_ms,
                reason="workflow_ended",
            )

        invocations = self.graph.context_invocations(context_id)
        running = [
            item for item in invocations if item.state == InvocationState.RUNNING_LLM
        ]
        if running:
            return self._lease(
                context_id,
                LeaseKind.RUNNING,
                now_ms,
                reason="running_llm",
            )
        ready = [
            item
            for item in invocations
            if item.state
            in {
                InvocationState.CREATED,
                InvocationState.READY,
                InvocationState.RETURNING,
            }
            or item.pending_messages > 0
        ]
        if ready:
            return self._lease(
                context_id,
                LeaseKind.READY,
                now_ms,
                reason="ready_or_pending_message",
            )
        waiting = [item for item in invocations if not item.state.terminal]
        if waiting:
            invocation = min(
                waiting,
                key=lambda item: (item.updated_ts_ms, item.invocation_id),
            )
            return self._lease(
                context_id,
                LeaseKind.CONDITIONAL_RESUME,
                now_ms,
                condition=self._condition(invocation),
                reason="+".join(sorted({item.state.value for item in waiting})),
            )
        if context.persistent:
            return self._lease(
                context_id,
                LeaseKind.CONDITIONAL_RESUME,
                now_ms,
                condition=LeaseCondition(
                    event_kind="context_resume",
                    subject_id=context_id,
                    condition_id=f"persistent:{context_id}:{context.epoch}",
                ),
                reason="persistent_inactive",
            )
        return self._lease(
            context_id,
            LeaseKind.DEAD,
            now_ms,
            reason="all_invocations_terminal",
        )

    def speculative(
        self,
        context_id: str,
        *,
        workflow_id: str,
        context_epoch: int,
        now_ms: float,
        reason: str,
        confidence: float,
    ) -> ContextLease:
        if not 0 <= confidence <= 1:
            raise ValueError("lease confidence must be in [0, 1]")
        return ContextLease(
            context_id=context_id,
            context_epoch=context_epoch,
            workflow_id=workflow_id,
            kind=LeaseKind.SPECULATIVE,
            condition=None,
            issued_ts_ms=now_ms,
            confidence=confidence,
            reason=reason,
        )

    def bundle(
        self,
        bundle_id: str,
        owner_context_ids: tuple[str, ...] | list[str] | set[str],
        *,
        now_ms: float,
    ) -> BundleLease:
        owners = tuple(sorted(set(owner_context_ids)))
        leases = [self.context(context_id, now_ms=now_ms) for context_id in owners]
        strongest = max(
            (item.kind for item in leases),
            key=lambda item: _LEASE_STRENGTH[item],
            default=LeaseKind.DEAD,
        )
        conditions = tuple(
            sorted(
                {
                    item.condition
                    for item in leases
                    if item.condition is not None
                }
            )
        )
        return BundleLease(
            bundle_id=bundle_id,
            owner_context_ids=owners,
            strongest_kind=strongest,
            conditions=conditions,
        )

    def _lease(
        self,
        context_id: str,
        kind: LeaseKind,
        now_ms: float,
        *,
        reason: str,
        condition: LeaseCondition | None = None,
    ) -> ContextLease:
        context = self.graph.contexts[context_id]
        return ContextLease(
            context_id=context_id,
            context_epoch=context.epoch,
            workflow_id=context.workflow_id,
            kind=kind,
            condition=condition,
            issued_ts_ms=now_ms,
            confidence=1.0,
            reason=reason,
        )

    @staticmethod
    def _condition(invocation: InvocationRecord) -> LeaseCondition:
        state = invocation.state
        if state == InvocationState.WAIT_TOOL:
            event_kind = "tool_result"
            condition_id = (
                f"tool:{invocation.invocation_id}:"
                f"{invocation.active_tool_family or 'unknown'}:{invocation.llm_round}"
            )
        elif state == InvocationState.WAIT_CHILD:
            event_kind = "child_return"
            condition_id = (
                f"children:{invocation.invocation_id}:"
                f"{','.join(sorted(invocation.blocking_child_ids))}"
            )
        elif state == InvocationState.WAIT_JOIN:
            event_kind = "join_satisfied"
            condition_id = f"join:{invocation.join_id or 'unknown'}"
        elif state == InvocationState.WAIT_MESSAGE:
            event_kind = "message"
            condition_id = f"message:{invocation.invocation_id}"
        else:
            event_kind = "invocation_ready"
            condition_id = f"state:{invocation.invocation_id}:{state.value}"
        return LeaseCondition(
            event_kind=event_kind,
            subject_id=invocation.invocation_id,
            condition_id=condition_id,
        )
