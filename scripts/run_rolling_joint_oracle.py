#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from beliefkv.metrics.transfer_timeline import (
    TimelineResourcePoint,
    TimelineTransfer,
    TransferTimeline,
    render_transfer_timeline,
    summarize_transfer_timeline,
)
from beliefkv.policy.joint_oracle import (
    OracleArm,
    TraceOrderJointOracle,
    generate_bounded_lag_topological_orders,
)
from beliefkv.policy.reference import (
    CapabilityReport,
    MetadataSource,
    PhysicalKVSnapshot,
    PolicyInput,
    ResidencyAction,
    ResourceSnapshot,
    RuntimeGraphSnapshot,
)
from beliefkv.policy.scenario_physicalizer import ScenarioDemand, ScenarioTransition
from beliefkv.policy.whatif_packer import ScenarioPlan
from beliefkv.simulator.queue_service import (
    FrozenCounterfactualWorkload,
    FrozenRequestDemand,
    QueueServiceModel,
    RollingFrozenTracePlanEvaluator,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact rolling token-Radix O0-O3 replay on a frozen trace."
    )
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--service-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hbm-capacity-tokens", type=int, required=True)
    parser.add_argument("--hbm-reserved-tokens", type=int, default=0)
    parser.add_argument("--host-capacity-tokens", type=int, required=True)
    parser.add_argument("--pcie-bandwidth-gbps", type=float, required=True)
    parser.add_argument("--transfer-setup-ms", type=float, required=True)
    parser.add_argument("--max-candidate-orders", type=int, default=4096)
    parser.add_argument("--fair-bounded-search", action="store_true")
    parser.add_argument("--fairness-lag-budget-ms", type=float, default=50.0)
    parser.add_argument("--fair-order-candidates", type=int, default=4)
    parser.add_argument("--allow-topological-baseline-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workload_path = args.workload.expanduser().resolve()
    service_path = args.service_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workload = FrozenCounterfactualWorkload.from_dict(
        _read_object(workload_path)
    )
    service_raw = _read_object(service_path)
    model_raw = service_raw.get("model", service_raw)
    if not isinstance(model_raw, Mapping):
        raise ValueError("service model payload is not an object")
    service_model = QueueServiceModel.from_dict(model_raw)
    if not service_model.calibrated:
        raise ValueError("rolling oracle requires a calibrated queue/service model")
    bytes_per_token = int(workload.metadata.get("kv_bytes_per_token", 0))
    if bytes_per_token <= 0:
        raise ValueError("workload lacks kv_bytes_per_token metadata")
    if args.hbm_capacity_tokens <= 0 or args.host_capacity_tokens <= 0:
        raise ValueError("HBM and Host token capacities must be positive")
    if not 0 <= args.hbm_reserved_tokens < args.hbm_capacity_tokens:
        raise ValueError("reserved HBM tokens must be smaller than capacity")
    if args.pcie_bandwidth_gbps <= 0 or args.transfer_setup_ms < 0:
        raise ValueError("PCIe bandwidth/setup parameters are invalid")

    baseline, baseline_source = _baseline_order(
        workload,
        allow_fallback=args.allow_topological_baseline_fallback,
    )
    policy_input, demand, plan = _empty_replay_input(
        workload,
        baseline,
        hbm_capacity_tokens=args.hbm_capacity_tokens,
        hbm_reserved_tokens=args.hbm_reserved_tokens,
        host_capacity_tokens=args.host_capacity_tokens,
        pcie_bandwidth_gbps=args.pcie_bandwidth_gbps,
        transfer_setup_ms=args.transfer_setup_ms,
    )
    evaluator = RollingFrozenTracePlanEvaluator(workload, service_model)
    request_predecessors = {
        request.request_id: request.predecessor_request_ids
        for request in workload.requests
    }
    fair_orders = None
    if args.fair_bounded_search:
        fair_orders = generate_bounded_lag_topological_orders(
            request_predecessors,
            workflow_by_request={
                request.request_id: request.workflow_id
                for request in workload.requests
            },
            service_ms_by_request={
                request.request_id: _standalone_request_service_ms(
                    request,
                    service_model,
                )
                for request in workload.requests
            },
            baseline_execution_order=baseline,
            kv_tokens_by_request={
                request.request_id: len(request.cache_commit_token_symbols)
                for request in workload.requests
            },
            lag_budget_ms=args.fairness_lag_budget_ms,
            max_orders=args.fair_order_candidates,
        )
    oracle = TraceOrderJointOracle().evaluate(
        policy_input,
        demand,
        plan,
        request_predecessors=request_predecessors,
        baseline_execution_order=baseline,
        trace_sensitivity=workload.trace_sensitivity,
        evaluator=evaluator,
        max_candidate_orders=args.max_candidate_orders,
        candidate_execution_orders=(
            tuple(item.execution_order for item in fair_orders.candidates)
            if fair_orders is not None
            else None
        ),
        candidate_search_kind=(
            "bounded_lag_heuristic_topological"
            if fair_orders is not None
            else "bounded_supplied_topological"
        ),
    )
    manifest = {
        "schema_version": 1,
        "workload_path": str(workload_path),
        "workload_sha256": _sha256(workload_path),
        "trace_id": workload.trace_id,
        "transition_hash": workload.transition_hash,
        "trace_sensitivity": workload.trace_sensitivity,
        "service_model_path": str(service_path),
        "service_model_sha256": _sha256(service_path),
        "service_model_id": service_model.model_id,
        "physical_model_id": "rolling_tiered_token_radix_allocator_v3",
        "baseline_order_source": baseline_source,
        "hbm_capacity_tokens": args.hbm_capacity_tokens,
        "hbm_reserved_tokens": args.hbm_reserved_tokens,
        "host_capacity_tokens": args.host_capacity_tokens,
        "kv_bytes_per_token": bytes_per_token,
        "pcie_bandwidth_gbps": args.pcie_bandwidth_gbps,
        "transfer_setup_ms": args.transfer_setup_ms,
        "max_candidate_orders": args.max_candidate_orders,
        "simulation_count": evaluator.simulation_count,
        "search_complete": oracle.search_complete,
        "candidate_order_count": oracle.candidate_order_count,
        "fairness_constraint": (
            fair_orders.to_dict()
            if fair_orders is not None
            else "not_modeled_trace_order_upper_bound"
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    if fair_orders is not None:
        _write_json(output_dir / "fair_order_candidates.json", fair_orders.to_dict())
    _write_json(output_dir / "joint_oracle.json", oracle.to_dict())
    for arm in OracleArm:
        result = evaluator.last_results[arm]
        _write_json(
            output_dir / f"{arm.value}_selected_result.json",
            {
                "arm": arm.value,
                "execution_order": list(oracle.result.arms[arm].plan.execution_order),
                "cost": oracle.result.arms[arm].cost.to_dict(),
                "simulation": {
                    "workflow_jct_ms": dict(result.workflow_jct_ms),
                    "request_finish_ms": dict(result.request_finish_ms),
                    "request_queue_wait_ms": dict(result.request_queue_wait_ms),
                    "request_action_unlock_ms": dict(result.request_action_unlock_ms),
                    "hbm_peak_bytes": result.hbm_peak_bytes,
                    "hbm_final_bytes": result.hbm_final_bytes,
                    "host_consumed_bytes": result.host_consumed_bytes,
                    "d2h_bytes": result.d2h_bytes,
                    "h2d_bytes": result.h2d_bytes,
                    "pcie_busy_ms": result.pcie_busy_ms,
                    "scheduler_steps": result.scheduler_steps,
                    "recomputed_cache_hit_tokens": dict(
                        result.recomputed_cache_hit_tokens
                    ),
                    "recomputed_unique_growth_bytes": dict(
                        result.recomputed_unique_growth_bytes
                    ),
                    "physical_timeline": list(result.physical_timeline),
                },
            },
        )
        timeline = _timeline(
            arm,
            result.physical_timeline,
            final_ts_ms=result.final_ts_ms,
            initial_hbm_bytes=policy_input.resources.hbm_reserved_bytes,
            hbm_capacity_bytes=policy_input.resources.hbm_capacity_bytes,
            host_capacity_bytes=args.host_capacity_tokens * bytes_per_token,
        )
        render_transfer_timeline(
            timeline,
            output_dir / f"{arm.value}_kv_timeline.html",
            title=f"BeliefKV {arm.value} Rolling KV Timeline",
        )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_order_count": oracle.candidate_order_count,
                "search_complete": oracle.search_complete,
                "simulation_count": evaluator.simulation_count,
                "joint_synergy_gap_ms": oracle.result.joint_synergy_gap_ms,
                "jointness_supported": oracle.result.jointness_supported,
                "arm_jct_ms": {
                    arm.value: oracle.result.arms[arm].cost.workflow_jct_ms
                    for arm in OracleArm
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _baseline_order(
    workload: FrozenCounterfactualWorkload,
    *,
    allow_fallback: bool,
) -> tuple[tuple[str, ...], str]:
    raw = workload.metadata.get("observed_request_order")
    if isinstance(raw, (list, tuple)):
        order = tuple(str(item) for item in raw)
        if set(order) == {item.request_id for item in workload.requests}:
            return order, "observed_causal_arrival"
    if not allow_fallback:
        raise ValueError(
            "workload lacks a complete observed_request_order; regenerate the trace "
            "or explicitly allow a non-observed topological fallback"
        )
    predecessors = {
        item.request_id: set(item.predecessor_request_ids)
        for item in workload.requests
    }
    completed: set[str] = set()
    order = []
    while len(order) < len(predecessors):
        ready = sorted(
            request_id
            for request_id, required in predecessors.items()
            if request_id not in completed and required.issubset(completed)
        )
        if not ready:
            raise ValueError("workload dependency graph is cyclic")
        selected = ready[0]
        order.append(selected)
        completed.add(selected)
    return tuple(order), "lexicographic_topological_fallback_not_observed"


def _standalone_request_service_ms(
    request: FrozenRequestDemand,
    service_model: QueueServiceModel,
) -> float:
    remaining = len(request.prompt_token_symbols)
    chunk_index = 0
    elapsed = 0.0
    while remaining:
        tokens = min(remaining, service_model.prefill_chunk_tokens)
        elapsed += service_model.prefill_elapsed_ms(
            tokens,
            chunk_index=chunk_index,
        )
        remaining -= tokens
        chunk_index += 1
    if request.output_tokens:
        quanta = math.ceil(
            request.output_tokens / service_model.decode_quantum_tokens
        )
        elapsed += quanta * service_model.decode_launch_ms
        elapsed += request.output_tokens / service_model.decode_rate(1)
    return elapsed


def _empty_replay_input(
    workload: FrozenCounterfactualWorkload,
    execution_order: tuple[str, ...],
    *,
    hbm_capacity_tokens: int,
    hbm_reserved_tokens: int,
    host_capacity_tokens: int,
    pcie_bandwidth_gbps: float,
    transfer_setup_ms: float,
) -> tuple[PolicyInput, ScenarioDemand, ScenarioPlan]:
    bytes_per_token = int(workload.metadata["kv_bytes_per_token"])
    snapshot_id = f"rolling-{workload.trace_id}"
    resources = ResourceSnapshot(
        snapshot_id=snapshot_id,
        ts_ms=0,
        hbm_capacity_bytes=hbm_capacity_tokens * bytes_per_token,
        hbm_used_bytes=0,
        hbm_reserved_bytes=hbm_reserved_tokens * bytes_per_token,
        host_free_bytes=host_capacity_tokens * bytes_per_token,
        urgent_d2h_bytes=0,
        urgent_h2d_bytes=0,
        pcie_utilization=0,
        gpu_compute_utilization=0,
        recent_kv_growth_bytes_per_ms=0,
        h2d_service_bytes_per_ms=pcie_bandwidth_gbps * 1_000_000,
        d2h_service_bytes_per_ms=pcie_bandwidth_gbps * 1_000_000,
        transfer_setup_p50_ms=transfer_setup_ms,
        unhidden_stall_per_byte=1 / (pcie_bandwidth_gbps * 1_000_000),
    )
    policy_input = PolicyInput(
        runtime_graph=RuntimeGraphSnapshot(
            snapshot_id=snapshot_id,
            graph_version=0,
            observed_ts_ms=0,
            state={},
        ),
        runnable_frontier=(),
        physical_kv=PhysicalKVSnapshot(
            snapshot_id=snapshot_id,
            topology_version=0,
            allocator_version=0,
            gpu_bytes=0,
            cpu_bytes=0,
            bundles=(),
        ),
        resources=resources,
        capabilities=CapabilityReport(
            runtime_name="rolling-counterfactual",
            runtime_version="v1",
            supported_residency_actions=frozenset(ResidencyAction),
            execution_order_control=True,
            admission_control=True,
            transfer_dependencies=True,
            native_identity_mapping=True,
        ),
    )
    demand = ScenarioDemand(
        snapshot_id=snapshot_id,
        scenario_id="frozen-trace",
        probability=1,
        transition=ScenarioTransition.NONBLOCKING,
        source=MetadataSource.HINDSIGHT,
        candidate_invocation_ids=(),
        candidate_request_ids=(),
        consumer_context_ids=(),
        required_context_ids=(),
        required_gpu_bundles=(),
        optional_gpu_bundles=(),
        startup_bytes_by_request={},
        projected_growth_bytes={},
        projected_new_bytes=0,
        projected_hbm_peak_bytes=0,
        required_h2d_bytes=0,
        earliest_ready_p50_ms=0,
        earliest_ready_p90_ms=0,
        physical_accounting_exact=True,
    )
    plan = ScenarioPlan(
        snapshot_id=snapshot_id,
        scenario_id=demand.scenario_id,
        execution_order=execution_order,
        admission_actions={request_id: "admit" for request_id in execution_order},
        bundle_actions={},
        feasible=True,
        expected_unhidden_stall_ms=0,
        hbm_time_byte_ms=0,
        d2h_bytes=0,
        h2d_bytes=0,
        recompute_tokens=0,
        projected_hbm_peak_bytes=0,
        reclaimed_bytes=0,
        physical_accounting_exact=True,
        blocker_reasons=(),
    )
    return policy_input, demand, plan


def _timeline(
    arm: OracleArm,
    raw_events: tuple[Mapping[str, object], ...],
    *,
    final_ts_ms: float,
    initial_hbm_bytes: int,
    hbm_capacity_bytes: int,
    host_capacity_bytes: int,
) -> TransferTimeline:
    transfers = []
    resources = [
        TimelineResourcePoint(
            0,
            initial_hbm_bytes,
            hbm_capacity_bytes,
            0,
            host_capacity_bytes,
            "runtime_resource_snapshot",
        )
    ]
    for event in raw_events:
        kind = str(event["kind"])
        start_ms = float(event["start_ms"])
        end_ms = float(event["end_ms"])
        transfer_bytes = int(event["transfer_bytes"])
        resources.append(
            TimelineResourcePoint(
                start_ms,
                int(event["hbm_bytes_before"]),
                hbm_capacity_bytes,
                int(event["host_bytes_before"]),
                host_capacity_bytes,
                "runtime_resource_snapshot",
            )
        )
        resources.append(
            TimelineResourcePoint(
                end_ms,
                int(event["hbm_bytes_after"]),
                hbm_capacity_bytes,
                int(event["host_bytes_after"]),
                host_capacity_bytes,
                "runtime_resource_snapshot",
            )
        )
        if transfer_bytes:
            transfers.append(
                TimelineTransfer(
                    command_id=f"{arm.value}-{int(event['sequence']):06d}",
                    direction="h2d" if kind.startswith("H2D") else "d2h",
                    submit_ts_ms=start_ms,
                    start_ts_ms=start_ms,
                    complete_ts_ms=end_ms,
                    actual_bytes=transfer_bytes,
                    closure_bytes=transfer_bytes,
                    status="completed",
                    context_id=(
                        str(event["request_id"])
                        if event.get("request_id") is not None
                        else None
                    ),
                    command_kind=kind,
                    reason=str(event["reason"]),
                    measurement="rolling_counterfactual",
                )
            )
    resources.sort(key=lambda item: item.ts_ms)
    summary = summarize_transfer_timeline(
        transfers,
        resources,
        start_ts_ms=0,
        end_ts_ms=final_ts_ms,
    )
    summary["counterfactual"] = True
    return TransferTimeline(
        run_id=f"rolling-{arm.value}",
        source_path="counterfactual",
        start_ts_ms=0,
        end_ts_ms=final_ts_ms,
        transfers=tuple(transfers),
        resources=tuple(resources),
        summary=summary,
    )


def _read_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
