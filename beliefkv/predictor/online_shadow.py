"""Bounded online P6 frontier shadow publisher.

The frontier model publishes local beliefs over the live RCCG at scheduler
safe points. Records are audit-only: they never alter admission, residency, or
transfer decisions, and ``prediction_used`` remains false in runtime summaries.

Online features are best-effort approximations of the offline decision-point
schema and are tagged ``feature_source="online_approx"`` so that downstream
analysis never mistakes them for exported training evidence.
"""

from __future__ import annotations

import hashlib
import collections
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.predictor.composer import RemainingTimePredictor
from beliefkv.predictor.structured_frontier import (
    LocalFrontierFeatures,
    LocalFrontierPrediction,
)


@dataclass(frozen=True)
class FrontierShadowRecord:
    """One bounded audit record for a single scoped invocation belief."""

    ts_ms: float
    context_id: str
    invocation_id: str
    workflow_id: str
    state: str
    agent_definition_id: str
    support_level: str
    ood_reasons: tuple[str, ...]
    boundary_top: str
    boundary_distribution: Mapping[str, float]
    remaining_decode_tokens_p50: float
    remaining_external_wait_ms_p50: float
    prompt_growth_tokens_p50: float
    next_output_tokens_p50: float
    feature_source: str = "online_approx"
    signature: str = ""


def _boundary_top(distribution: Mapping[str, float]) -> str:
    if not distribution:
        return ""
    return max(distribution.items(), key=lambda item: item[1])[0]


def _record_signature(prediction: LocalFrontierPrediction) -> str:
    payload = (
        prediction.support_level,
        tuple(sorted(prediction.ood_reasons)),
        _boundary_top(prediction.boundary_distribution),
        round(prediction.remaining_decode_tokens.quantile(0.5), 1),
        round(prediction.remaining_external_wait.quantile(0.5), 1),
        round(prediction.prompt_growth_tokens.quantile(0.5), 1),
        round(prediction.next_output_tokens.quantile(0.5), 1),
    )
    digest = hashlib.blake2b(
        repr(payload).encode(), digest_size=8, person=b"bkv-fr-shadow"
    ).hexdigest()
    return digest


def _features_for_invocation(
    graph: RuntimeCausalContextGraph,
    invocation_id: str,
    predictor: RemainingTimePredictor,
    *,
    now_ms: float,
    active_tool_count: int,
    family_counts: Any,
) -> LocalFrontierFeatures:
    invocation = graph.invocations[invocation_id]
    online = predictor.features.get(invocation_id)
    history: tuple[str, ...] = ()
    context_tokens = 0
    generated_tokens = 0
    tool_backend_class = "unknown"
    if online is not None:
        history = tuple(online.boundary_history)[-8:]
        context_tokens = max(0, int(online.context_tokens or 0))
        generated_tokens = max(0, int(online.generated_tokens or 0))
        tool_backend_class = str(online.tool_backend_class or "unknown")
    elapsed_wait_ms = 0.0
    if (
        invocation.state == InvocationState.WAIT_TOOL
        and invocation.active_tool_start_ms is not None
    ):
        elapsed_wait_ms = max(0.0, now_ms - invocation.active_tool_start_ms)
    tool_family = invocation.active_tool_family or "unknown"
    backend_pressure = (
        f"active_family:{int(family_counts.get(tool_family, 0))}"
        if tool_family != "unknown"
        else "unknown"
    )
    return LocalFrontierFeatures(
        invocation_id=invocation.invocation_id,
        state=invocation.state.value,
        agent_definition_id=invocation.agent_definition_id,
        boundary_history=history,
        tool_family=tool_family,
        backend_class=tool_backend_class,
        generated_tokens=generated_tokens,
        elapsed_wait_ms=elapsed_wait_ms,
        current_sequence_tokens=context_tokens,
        active_tool_count=active_tool_count,
        backend_pressure=backend_pressure,
    )


def build_invocation_frontier_predictions(
    graph: RuntimeCausalContextGraph,
    predictor: RemainingTimePredictor,
    *,
    now_ms: float,
    invocation_ids: Sequence[str],
) -> dict[str, LocalFrontierPrediction]:
    """Build bounded frontier predictions for a set of invocations.

    Used both by the shadow publisher and by the predictive joint planner.
    Prediction failures are never raised to the caller; the caller treats a
    missing invocation as observed-only fallback.
    """

    frontier = predictor.frontier_model
    if frontier is None:
        return {}
    active_tool_count = sum(
        1
        for invocation in graph.invocations.values()
        if invocation.state == InvocationState.WAIT_TOOL
    )
    family_counts = collections.Counter(
        invocation.active_tool_family
        for invocation in graph.invocations.values()
        if invocation.state == InvocationState.WAIT_TOOL
        and invocation.active_tool_family is not None
    )
    result: dict[str, LocalFrontierPrediction] = {}
    for invocation_id in invocation_ids:
        if invocation_id not in graph.invocations:
            continue
        features = _features_for_invocation(
            graph,
            invocation_id,
            predictor,
            now_ms=now_ms,
            active_tool_count=active_tool_count,
            family_counts=family_counts,
        )
        try:
            result[invocation_id] = frontier.predict(features)
        except Exception:
            # A prediction failure must never affect the scheduler critical path.
            continue
    return result


def build_frontier_shadow_records(
    graph: RuntimeCausalContextGraph,
    predictor: RemainingTimePredictor,
    *,
    now_ms: float,
    signals: Any | None = None,
    last_signatures: Mapping[str, tuple[str, float]] | None = None,
    min_interval_ms: float = 1000.0,
    max_invocations: int = 64,
) -> tuple[tuple[FrontierShadowRecord, ...], dict[str, tuple[str, float]]]:
    """Build changed, bounded frontier shadow records for one safe point.

    Returns ``(records, updated_signatures)``. A record is emitted for an
    invocation only when its prediction signature changed since the previous
    emission and at least ``min_interval_ms`` elapsed for that invocation.
    """

    frontier = predictor.frontier_model
    if frontier is None:
        return (), dict(last_signatures or {})
    signatures = dict(last_signatures or {})
    active = sorted(
        (
            invocation
            for invocation in graph.invocations.values()
            if not invocation.state.terminal
        ),
        key=lambda item: item.updated_ts_ms,
        reverse=True,
    )[:max_invocations]
    if not active:
        return (), signatures
    predictions = build_invocation_frontier_predictions(
        graph,
        predictor,
        now_ms=now_ms,
        invocation_ids=tuple(
            invocation.invocation_id for invocation in active
        ),
    )
    records: list[FrontierShadowRecord] = []
    for invocation in active:
        prediction = predictions.get(invocation.invocation_id)
        if prediction is None:
            continue
        signature = _record_signature(prediction)
        previous_signature, previous_ts = signatures.get(
            invocation.invocation_id, ("", float("-inf"))
        )
        if signature == previous_signature:
            continue
        if now_ms - previous_ts >= min_interval_ms:
            records.append(
                FrontierShadowRecord(
                    ts_ms=now_ms,
                    context_id=invocation.context_id,
                    invocation_id=invocation.invocation_id,
                    workflow_id=invocation.workflow_id,
                    state=invocation.state.value,
                    agent_definition_id=invocation.agent_definition_id,
                    support_level=prediction.support_level,
                    ood_reasons=tuple(prediction.ood_reasons),
                    boundary_top=_boundary_top(
                        prediction.boundary_distribution
                    ),
                    boundary_distribution=dict(
                        prediction.boundary_distribution
                    ),
                    remaining_decode_tokens_p50=(
                        prediction.remaining_decode_tokens.quantile(0.5)
                    ),
                    remaining_external_wait_ms_p50=(
                        prediction.remaining_external_wait.quantile(0.5)
                    ),
                    prompt_growth_tokens_p50=(
                        prediction.prompt_growth_tokens.quantile(0.5)
                    ),
                    next_output_tokens_p50=(
                        prediction.next_output_tokens.quantile(0.5)
                    ),
                    signature=signature,
                )
            )
            signatures[invocation.invocation_id] = (signature, now_ms)
        # A changed signature inside the interval is not stored yet; it is
        # re-evaluated on the next safe point against the last emission time.
    return tuple(records), signatures
