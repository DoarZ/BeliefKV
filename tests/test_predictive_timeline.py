from __future__ import annotations

from beliefkv.policy.predictive_joint import (
    PackageScenarioEvaluation,
    PredictiveActionKind,
    PredictiveActionPackage,
    ScenarioCost,
)
from beliefkv.policy.predictive_timeline import (
    CandidatePhysicalPlan,
    CandidateTimelineEvaluator,
    PhysicalizedInvocationDemand,
    ScheduledBatchQuantum,
    ScheduledRequestQuantum,
    ScheduledTransfer,
)
from beliefkv.predictor.frontier_belief import (
    BoundaryEvent,
    DemandPhase,
    DemandScenario,
    DependencyMode,
    FrontierDemandOutcome,
)
from beliefkv.predictor.hardware_service import GPUServiceCurveModel


def _batch_row(
    sample_id: str,
    requests: tuple[tuple[int, int], ...],
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        "row_type": "gpu_batch_service_interval",
        "sample_id": sample_id,
        "split": "train",
        "phase": "decode",
        "batch_size": len(requests),
        "request_samples": [
            {
                "request_id": f"{sample_id}-{index}",
                "sequence_tokens_before": sequence,
                "token_delta": tokens,
                "cache_hit_ratio": 0.0,
            }
            for index, (sequence, tokens) in enumerate(requests)
        ],
        "chunk_position": "first",
        "prefill_decode_mixed": False,
        "pcie_contention_state": "idle",
        "hicache_inflight_bytes": 0,
        "service_elapsed_ms": elapsed_ms,
        "warmup": False,
        "evidence_role": "controlled_microbenchmark",
    }


def _service_model() -> GPUServiceCurveModel:
    model = GPUServiceCurveModel(minimum_support=1)
    model.fit(
        [
            _batch_row("single-a", ((4096, 100),), 10.0),
            _batch_row("single-b", ((4096, 100),), 10.0),
            _batch_row("batched-a", ((4096, 100), (4096, 100)), 12.0),
            _batch_row("batched-b", ((4096, 100), (4096, 100)), 12.0),
        ]
    )
    return model


def _scenario() -> DemandScenario:
    children = tuple(
        FrontierDemandOutcome(
            invocation_id=child,
            boundary_event=BoundaryEvent.RETURN,
            dependency_mode=DependencyMode.NONE,
            phase=DemandPhase.DECODE,
            current_sequence_tokens=4096,
            remaining_decode_tokens=100,
            prompt_growth_tokens=0,
            next_output_tokens=0,
        )
        for child in ("child-a", "child-b")
    )
    parent = FrontierDemandOutcome(
        invocation_id="parent",
        boundary_event=BoundaryEvent.UNKNOWN,
        dependency_mode=DependencyMode.JOIN_ALL,
        phase=DemandPhase.EXTERNAL,
        current_sequence_tokens=8192,
        remaining_decode_tokens=0,
        prompt_growth_tokens=128,
        next_output_tokens=16,
        dependency_invocation_ids=("child-a", "child-b"),
        join_id="join",
    )
    return DemandScenario("scenario", (*children, parent), 1.0)


def _physical_demands() -> tuple[PhysicalizedInvocationDemand, ...]:
    return tuple(
        PhysicalizedInvocationDemand(child, 0, 100, 4096)
        for child in ("child-a", "child-b")
    )


def test_join_is_resolved_after_candidate_batch_schedule() -> None:
    evaluator = CandidateTimelineEvaluator(_service_model(), service_quantile=0.9)
    serial = CandidatePhysicalPlan(
        package_id="serial",
        physical_snapshot_id="snapshot",
        physical_snapshot_revision=1,
        invocation_demands=_physical_demands(),
        batches=tuple(
            ScheduledBatchQuantum(
                f"serial-{index}",
                DemandPhase.DECODE,
                (ScheduledRequestQuantum(child, 100, 4096),),
                chunk_position="first",
            )
            for index, child in enumerate(("child-a", "child-b"))
        ),
    )
    batched = CandidatePhysicalPlan(
        package_id="batched",
        physical_snapshot_id="snapshot",
        physical_snapshot_revision=1,
        invocation_demands=_physical_demands(),
        batches=(
            ScheduledBatchQuantum(
                "batched",
                DemandPhase.DECODE,
                (
                    ScheduledRequestQuantum("child-a", 100, 4096),
                    ScheduledRequestQuantum("child-b", 100, 4096),
                ),
                chunk_position="first",
            ),
        ),
    )

    serial_timeline = evaluator.evaluate(_scenario(), serial)
    batched_timeline = evaluator.evaluate(_scenario(), batched)

    assert (
        serial_timeline.join_reentry_offsets_ms["join"]
        > batched_timeline.join_reentry_offsets_ms["join"]
    )
    assert serial_timeline.join_reentry_offsets_ms["join"] != max(100, 100)


def test_timed_scenario_is_the_only_input_to_risk_cost() -> None:
    evaluator = CandidateTimelineEvaluator(_service_model())
    plan = CandidatePhysicalPlan(
        package_id="batched",
        physical_snapshot_id="snapshot",
        physical_snapshot_revision=1,
        invocation_demands=_physical_demands(),
        batches=(
            ScheduledBatchQuantum(
                "batched",
                DemandPhase.DECODE,
                (
                    ScheduledRequestQuantum("child-a", 100, 4096),
                    ScheduledRequestQuantum("child-b", 100, 4096),
                ),
                chunk_position="first",
            ),
        ),
    )
    timeline = evaluator.evaluate(_scenario(), plan)
    package = PredictiveActionPackage(
        "batched", PredictiveActionKind.PREFETCH_GPU, ("context",)
    )
    evaluation = PackageScenarioEvaluation.from_timed_scenarios(
        package,
        {"scenario": timeline},
        unlock_invocation_ids=("parent",),
        other_cost=ScenarioCost(100, 0),
    )

    assert evaluation.costs_by_scenario["scenario"].action_unlock_delay_ms == (
        timeline.join_reentry_offsets_ms["join"]
    )
    assert evaluation.costs_by_scenario["scenario"].deterministic_feasible


def test_join_release_and_parent_completion_are_distinct() -> None:
    evaluator = CandidateTimelineEvaluator(_service_model())
    plan = CandidatePhysicalPlan(
        package_id="parent-resume",
        physical_snapshot_id="snapshot",
        physical_snapshot_revision=1,
        invocation_demands=(
            *_physical_demands(),
            PhysicalizedInvocationDemand("parent", 0, 16, 8192),
        ),
        batches=(
            ScheduledBatchQuantum(
                "children",
                DemandPhase.DECODE,
                (
                    ScheduledRequestQuantum("child-a", 100, 4096),
                    ScheduledRequestQuantum("child-b", 100, 4096),
                ),
                chunk_position="first",
            ),
            ScheduledBatchQuantum(
                "parent",
                DemandPhase.DECODE,
                (ScheduledRequestQuantum("parent", 16, 8192),),
                chunk_position="first",
                ready_after_transfer_ids=("restore-parent",),
            ),
        ),
        transfers=(ScheduledTransfer("restore-parent", 0.0, 25.0, 25.0),),
    )

    timeline = evaluator.evaluate(_scenario(), plan)
    parent = next(
        item for item in timeline.invocation_outcomes if item.invocation_id == "parent"
    )

    assert timeline.join_reentry_offsets_ms["join"] == (
        timeline.service_quanta[0].completion_offset_ms
    )
    assert parent.completion_offset_ms == (
        timeline.service_quanta[-1].completion_offset_ms
    )
    assert parent.completion_offset_ms > timeline.join_reentry_offsets_ms["join"]
    assert timeline.service_quanta[-1].start_offset_ms == 25.0
