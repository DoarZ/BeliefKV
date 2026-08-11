#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
from typing import Iterable, Mapping

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.policy.joint_scheduler import AsyncSemanticJointPlanner, JointPlannerConfig
from beliefkv.policy.reference import MetadataSource, MetadataValue, PolicyInput
from beliefkv.policy.risk_shadow import (
    PredictiveRiskShadowConfig,
    PredictiveRiskShadowObserver,
)
from beliefkv.policy.service_curve import TransferServiceCurve
from beliefkv.policy.transfer_cost import PCIeCostModel
from beliefkv.predictor.frontier_belief import PredictiveEvidenceReadSet
from beliefkv.predictor.hardware_service import GPUServiceCurveModel
from beliefkv.runtime.protocol import TransferDirection


def _records(path: Path) -> Iterable[Mapping[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _transfer_estimates(
    policy_input: PolicyInput,
    curve: TransferServiceCurve,
) -> Mapping[str, object]:
    by_context: dict[str, list[object]] = {}
    for bundle in policy_input.physical_kv.bundles:
        for context_id in bundle.owner_context_ids:
            by_context.setdefault(context_id, []).append(bundle)
    contexts: dict[str, object] = {}
    for context_id, bundles in sorted(by_context.items()):
        values: dict[str, object] = {}
        page_count = sum(max(1, len(item.extent_ids)) for item in bundles)
        for direction, size_bytes, command_kind, host_state in (
            (
                TransferDirection.H2D,
                sum(max(0, item.cpu_bytes - item.gpu_bytes) for item in bundles),
                "prefetch_context",
                "present",
            ),
            (
                TransferDirection.D2H,
                sum(max(0, item.gpu_bytes - item.cpu_bytes) for item in bundles),
                "offload_context",
                "missing",
            ),
        ):
            if size_bytes <= 0:
                continue
            estimate = curve.estimate(
                direction,
                size_bytes,
                page_count=page_count,
                command_kind=command_kind,
                host_copy_state=host_state,
                pinned_host=True,
            )
            values[direction.value] = {
                "size_bytes": size_bytes,
                "estimated_callback_ms": estimate.estimated_callback_ms,
                "setup_p90_ms": estimate.setup_p90_ms,
                "callback_floor_p90_ms": estimate.callback_floor_p90_ms,
                "fixed_overhead_p90_ms": estimate.fixed_overhead_p90_ms,
                "effective_bytes_per_ms_p10": estimate.effective_bytes_per_ms_p10,
                "sample_count": estimate.sample_count,
                "source": estimate.source,
                "nearest_bucket_distance": estimate.nearest_bucket_distance,
                "size_coverage_bytes": estimate.size_coverage_bytes,
                "service_epoch": curve.warm_start_hardware_key,
            }
        if values:
            contexts[context_id] = values
    return {
        "hardware_key": curve.warm_start_hardware_key,
        "warm_start_sample_count": curve.warm_start_sample_count,
        "contexts": contexts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay P6 action-projected risk planning on frozen PolicyInput snapshots."
    )
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--gpu-service-model", type=Path, required=True)
    parser.add_argument("--transfer-service-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument(
        "--snapshot-id",
        action="append",
        default=[],
        help="Replay only these snapshot IDs; repeat for multiple snapshots.",
    )
    parser.add_argument(
        "--transfer-model",
        choices=("byte-only", "morphology-aware"),
        default="morphology-aware",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--stall-fraction",
        type=float,
        default=None,
        help=(
            "Override proactive D2H interference as this fraction of the "
            "same transfer-duration estimate used by recourse."
        ),
    )
    args = parser.parse_args()
    if args.stall_fraction is not None and not 0.0 <= args.stall_fraction <= 1.0:
        parser.error("--stall-fraction must be in [0, 1]")

    transfer_artifact = json.loads(
        args.transfer_service_model.read_text(encoding="utf-8")
    )
    transfer_curve = TransferServiceCurve(
        PCIeCostModel(),
        window=max(1, int(transfer_artifact.get("window") or 1024)),
        min_samples=max(1, int(transfer_artifact.get("min_samples") or 1)),
    )
    transfer_curve.warm_start(args.transfer_service_model)
    observer = PredictiveRiskShadowObserver(
        GPUServiceCurveModel.load(args.gpu_service_model),
        PredictiveRiskShadowConfig(
            particle_count=128,
            top_k=8,
            max_candidates=8,
            max_full_prefetch_hbm_ratio=0.05,
        ),
    )
    planner = AsyncSemanticJointPlanner(
        JointPlannerConfig(max_planning_budget_ms=100.0)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    counts: Counter[str] = Counter()
    prepare_failure_counts: Counter[str] = Counter()
    positive = 0
    eligible = 0
    prepare_candidates = 0
    positive_prepare_candidates = 0
    shape_supported_prepare_candidates = 0
    prepare_reason_counts: Counter[str] = Counter()
    latest_start_ms: list[float] = []
    pressure_candidates = 0
    recourse_scenarios = 0
    max_prepare_overflow_bytes = 0
    exclusive_full_ratios: list[float] = []
    requested_snapshot_ids = set(args.snapshot_id)
    with temporary.open("w", encoding="utf-8") as output:
        for raw in _records(args.snapshots):
            policy_raw = raw.get("policy_input")
            if not isinstance(policy_raw, Mapping):
                counts["invalid"] += 1
                continue
            graph_raw = policy_raw.get("runtime_graph")
            snapshot_id = str(
                graph_raw.get("snapshot_id")
                if isinstance(graph_raw, Mapping)
                else ""
            )
            if requested_snapshot_ids and snapshot_id not in requested_snapshot_ids:
                continue
            if args.max_snapshots and counts["seen"] >= args.max_snapshots:
                break
            counts["seen"] += 1
            policy_input = PolicyInput.from_dict(policy_raw)
            if "frontier_predictions" not in policy_input.optional_metadata:
                counts["no_predictions"] += 1
                continue
            metadata = dict(policy_input.optional_metadata)
            metadata["beliefkv_transfer_service_estimates"] = MetadataValue(
                MetadataSource.OBSERVED,
                _transfer_estimates(policy_input, transfer_curve),
                "offline_transfer_service_replay",
            )
            metadata["beliefkv_transfer_service_curve_snapshot"] = MetadataValue(
                MetadataSource.OBSERVED,
                transfer_curve.snapshot(),
                "offline_transfer_service_snapshot",
            )
            metadata["beliefkv_transfer_model_mode"] = MetadataValue(
                MetadataSource.APPLICATION_PROVIDED,
                args.transfer_model,
                "offline_transfer_model_ablation",
            )
            if args.stall_fraction is not None:
                metadata["beliefkv_transfer_interference_policy"] = MetadataValue(
                    MetadataSource.APPLICATION_PROVIDED,
                    {
                        "mode": "stall_fraction",
                        "stall_fraction": args.stall_fraction,
                        "service_epoch": (
                            transfer_curve.warm_start_hardware_key
                            or "unavailable"
                        ),
                    },
                    "offline_interference_sensitivity",
                )
            policy_input = replace(policy_input, optional_metadata=metadata)
            graph = RuntimeCausalContextGraph.from_snapshot(
                policy_input.runtime_graph.state
            )
            source_plan = planner.plan(policy_input)
            result = observer.evaluate(
                policy_input,
                graph=graph,
                source_plan=source_plan,
                evidence_read_set=PredictiveEvidenceReadSet(
                    graph_version=graph.graph_version,
                    page_revision=policy_input.physical_kv.allocator_version,
                    topology_revision=policy_input.physical_kv.topology_version,
                    fairness_revision=0,
                    admission_revision=0,
                    transfer_epoch=0,
                    obligation_revision=0,
                    lease_revision=0,
                    grace_revision=0,
                    parser_frontier_revision=0,
                    model_version=str(
                        policy_input.optional_metadata[
                            "frontier_prediction_model_version"
                        ].value
                    ),
                ),
            )
            payload = result.to_dict()
            payload["transfer_model"] = args.transfer_model
            output.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            counts[f"status:{result.status}"] += 1
            counts[f"selected:{result.selected_action}"] += 1
            for summary in result.candidate_summaries:
                if float(summary.get("expected_benefit_ms") or 0.0) > 0:
                    positive += 1
                if bool(summary.get("eligible", False)):
                    eligible += 1
                if summary.get("action") != "prepare_host":
                    continue
                prepare_candidates += 1
                positive_prepare_candidates += (
                    float(summary.get("expected_benefit_ms") or 0.0) > 0
                )
                shape_supported_prepare_candidates += bool(
                    summary.get("morphology_shape_supported")
                )
                prepare_reason_counts.update(
                    str(item) for item in summary.get("reasons", ())
                )
                overflow = int(summary.get("worst_future_hbm_overflow_bytes") or 0)
                max_prepare_overflow_bytes = max(
                    max_prepare_overflow_bytes,
                    overflow,
                )
                if overflow > 0:
                    pressure_candidates += 1
                diagnostics = summary.get("prepare_recourse_scenarios", ())
                if not isinstance(diagnostics, (list, tuple)):
                    continue
                for diagnostic in diagnostics:
                    if not isinstance(diagnostic, Mapping):
                        continue
                    reason = str(
                        diagnostic.get("recourse_failure_reason") or "unknown"
                    )
                    prepare_failure_counts[reason] += 1
                    if reason == "eligible":
                        recourse_scenarios += 1
                    closure_bytes = int(
                        diagnostic.get("full_closure_copy_bytes") or 0
                    )
                    if closure_bytes > 0:
                        exclusive_full_ratios.append(
                            int(
                                diagnostic.get("exclusive_reclaimable_bytes")
                                or 0
                            )
                            / closure_bytes
                        )
                    deadline = diagnostic.get("morphology_deadline_ms")
                    duration = diagnostic.get("shape_aware_transfer_p90_ms")
                    if deadline is not None and duration is not None:
                        latest_start_ms.append(float(deadline) - float(duration))
    temporary.replace(args.output)
    summary = {
        "transfer_model": args.transfer_model,
        "requested_snapshot_ids": sorted(requested_snapshot_ids),
        "counts": dict(sorted(counts.items())),
        "positive_benefit_candidates": positive,
        "eligible_candidates": eligible,
        "stall_fraction": args.stall_fraction,
        "prepare_host": {
            "candidate_count": prepare_candidates,
            "positive_benefit_candidate_count": positive_prepare_candidates,
            "shape_supported_candidate_count": (
                shape_supported_prepare_candidates
            ),
            "candidate_reason_counts": dict(sorted(prepare_reason_counts.items())),
            "pressure_candidate_count": pressure_candidates,
            "max_overflow_bytes": max_prepare_overflow_bytes,
            "recourse_scenario_count": recourse_scenarios,
            "scenario_failure_counts": dict(
                sorted(prepare_failure_counts.items())
            ),
            "exclusive_full_ratio_p50": (
                sorted(exclusive_full_ratios)[len(exclusive_full_ratios) // 2]
                if exclusive_full_ratios
                else None
            ),
            "latest_start_ms_min": (
                min(latest_start_ms) if latest_start_ms else None
            ),
            "latest_start_ms_max": (
                max(latest_start_ms) if latest_start_ms else None
            ),
        },
        "output": str(args.output.resolve()),
    }
    summary_output = args.summary_output or args.output.with_suffix(
        args.output.suffix + ".summary.json"
    )
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_output"] = str(summary_output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
