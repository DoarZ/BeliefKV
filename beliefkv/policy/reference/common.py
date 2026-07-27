from __future__ import annotations

from collections.abc import Sequence

from beliefkv.policy.reference.base import (
    ExecutionIntent,
    PhysicalBundleSnapshot,
    PolicyInput,
    RunnableInvocation,
)


def execution_intent(
    policy_input: PolicyInput,
    ordered: Sequence[RunnableInvocation],
    *,
    mode: str,
    reason: str,
) -> ExecutionIntent:
    first = ordered[0] if ordered else None
    return ExecutionIntent(
        ordered_request_ids=tuple(item.request_id for item in ordered),
        selected_workflow_id=first.workflow_id if first is not None else None,
        selected_invocation_id=first.invocation_id if first is not None else None,
        mode=mode,
        graph_version=policy_input.runtime_graph.graph_version,
        reason=reason,
    )


def fifo_frontier(policy_input: PolicyInput) -> tuple[RunnableInvocation, ...]:
    return tuple(
        sorted(
            policy_input.runnable_frontier,
            key=lambda item: (
                item.submitted_ts_ms,
                item.workflow_id,
                item.invocation_id,
                item.request_id,
            ),
        )
    )


def bundles_for_context(
    policy_input: PolicyInput,
    context_id: str,
) -> tuple[PhysicalBundleSnapshot, ...]:
    return tuple(
        bundle
        for bundle in policy_input.physical_kv.bundles
        if context_id in bundle.owner_context_ids
    )


def required_restore_bundles(
    policy_input: PolicyInput,
    context_id: str,
) -> tuple[PhysicalBundleSnapshot, ...]:
    return tuple(
        bundle
        for bundle in bundles_for_context(policy_input, context_id)
        if (
            bundle.residency == "prefetching"
            or (
                bundle.gpu_bytes < bundle.physical_unique_bytes
                and (
                    bundle.cpu_bytes > 0
                    or bundle.residency in {"cpu_only", "host_only"}
                )
            )
        )
    )


def blocked_restore_bundles(
    bundles: Sequence[PhysicalBundleSnapshot],
) -> tuple[PhysicalBundleSnapshot, ...]:
    return tuple(
        bundle
        for bundle in bundles
        if bundle.residency == "prefetching" or not bundle.actionable
    )
