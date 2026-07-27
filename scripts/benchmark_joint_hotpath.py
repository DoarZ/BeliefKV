#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.metrics.summary import percentile
from beliefkv.policy.joint_scheduler import JointPlannerConfig, ObservedJointPlanner
from beliefkv.policy.reference import PolicyInput, RunnableInvocation, RuntimeGraphSnapshot
from beliefkv.policy.resource_snapshot import RuntimeResourceObservation
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import PageHandle, PhysicalResidency, TransferDirection


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "count": len(samples),
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "max_ms": max(samples),
    }


def _count_summary(samples: list[int]) -> dict[str, float | int]:
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _observation(
    *,
    ts_ms: float,
    capacity_bytes: int,
    hbm_used_bytes: int,
    host_used_bytes: int = 0,
) -> RuntimeResourceObservation:
    return RuntimeResourceObservation(
        ts_ms=ts_ms,
        hbm_capacity_bytes=capacity_bytes,
        hbm_used_bytes=hbm_used_bytes,
        host_capacity_bytes=capacity_bytes,
        host_used_bytes=host_used_bytes,
        host_free_bytes=capacity_bytes - host_used_bytes,
        source="joint_hotpath_cpu_benchmark",
    )


def _planner_input(
    policy_input: PolicyInput,
    *,
    runnable_count: int,
) -> PolicyInput:
    requests = tuple(
        RunnableInvocation(
            request_id=f"request-{index:04d}",
            workflow_id=f"workflow-{index:04d}",
            invocation_id=f"invocation-{index:04d}",
            context_id=f"context-{index:04d}",
            context_epoch=0,
            submitted_ts_ms=float(index),
            startup_bytes=98_304,
            causal_class="engine_waiting:foreground:root",
        )
        for index in range(runnable_count)
    )
    invocations = {
        request.invocation_id: {
            "workflow_id": request.workflow_id,
            "context_id": request.context_id,
            "state": "ready",
            "pending_messages": 0,
            "children": [],
            "blocking_children": [],
            "execution_mode": "foreground",
            "relation_type": "root",
            "context_mode": "fresh",
        }
        for request in requests
    }
    accounts = {
        request.workflow_id: {
            "weight": 1.0,
            "attained_service_ms": 0.0,
            "virtual_runtime_ms": float(index % 3),
            "dispatch_count": 0,
        }
        for index, request in enumerate(requests)
    }
    graph = RuntimeGraphSnapshot(
        snapshot_id=policy_input.snapshot_id,
        graph_version=1,
        observed_ts_ms=policy_input.resources.ts_ms,
        state={
            "rccg": {"invocations": invocations, "joins": {}},
            "workflow_fairness": {
                "accounts": accounts,
                "memory_charges_bytes": {},
                "revision": 1,
            },
            "control": {"transitions": {}},
        },
    )
    return replace(
        policy_input,
        runtime_graph=graph,
        runnable_frontier=requests,
    )


def _delta_replication_benchmark(
    *,
    page_count: int,
    changed_page_count: int,
    iterations: int,
    page_bytes: int,
) -> dict[str, object]:
    source = PageOwnershipIndex()
    context_count = min(64, page_count)
    for context_index in range(context_count):
        source.register_context(
            f"context-{context_index:04d}",
            f"workflow-{context_index:04d}",
            0,
        )
    handles = tuple(PageHandle(page_id, 0) for page_id in range(1, page_count + 1))
    for handle in handles:
        source.register_page(handle, size_bytes=page_bytes)
    for context_index in range(context_count):
        owned = handles[context_index::context_count]
        source.bind_pages(f"context-{context_index:04d}", 0, owned)

    mirror = PageOwnershipIndex()
    initial = source.replica_delta_since(0)
    mirror.apply_replica_delta(initial)
    revision = initial.to_revision
    targets = handles[-min(changed_page_count, page_count) :]
    capture_samples: list[float] = []
    apply_samples: list[float] = []
    copied_context_counts: list[int] = []
    for iteration in range(iterations):
        lock_value = 1 if iteration % 2 == 0 else 0
        for handle in targets:
            source.set_engine_lock(handle, lock_value)
        started_ns = time.perf_counter_ns()
        delta = source.replica_delta_since(revision)
        capture_samples.append(_elapsed_ms(started_ns))
        copied_context_counts.append(len(delta.contexts))
        started_ns = time.perf_counter_ns()
        mirror.apply_replica_delta(delta, full_validation=False)
        apply_samples.append(_elapsed_ms(started_ns))
        revision = delta.to_revision

    mirror.assert_consistent()
    return {
        "changed_page_count": len(targets),
        "context_count": context_count,
        "capture": _summary(capture_samples),
        "mirror_apply": _summary(apply_samples),
        "copied_context_count": _count_summary(copied_context_counts),
    }


def run(
    *,
    page_count: int,
    iterations: int,
    page_bytes: int,
    runnable_count: int,
    changed_page_count: int,
) -> dict[str, object]:
    if (
        page_count <= 0
        or iterations <= 0
        or page_bytes <= 0
        or changed_page_count <= 0
    ):
        raise ValueError("benchmark dimensions must be positive")
    capacity = max(page_count * page_bytes * 2, page_bytes * 4)
    controller = BeliefKVController(
        BeliefKVConfig(
            hbm_capacity_bytes=capacity,
            host_capacity_bytes=capacity,
            reserve_hbm_bytes=0,
            kv_bytes_per_token=page_bytes,
            predictor_enabled=False,
            shadow_enabled=False,
            prefetch_enabled=False,
        )
    )
    for page_id in range(1, page_count + 1):
        controller.page_index.register_page(
            PageHandle(page_id, 0),
            size_bytes=page_bytes,
            residency=PhysicalResidency.GPU_ONLY,
            radix_depth=1,
            last_access_ms=float(page_id),
        )
    hbm_used = page_count * page_bytes

    started_ns = time.perf_counter_ns()
    controller.build_policy_input(
        _observation(
            ts_ms=1.0,
            capacity_bytes=capacity,
            hbm_used_bytes=hbm_used,
        )
    )
    cold_ms = _elapsed_ms(started_ns)

    unchanged: list[float] = []
    lock_delta: list[float] = []
    target = PageHandle(page_count, 0)
    ts_ms = 2.0
    for iteration in range(iterations):
        started_ns = time.perf_counter_ns()
        controller.build_policy_input(
            _observation(
                ts_ms=ts_ms,
                capacity_bytes=capacity,
                hbm_used_bytes=hbm_used,
            )
        )
        unchanged.append(_elapsed_ms(started_ns))
        ts_ms += 1.0

        controller.page_index.set_engine_lock(
            target,
            1 if iteration % 2 == 0 else 0,
        )
        started_ns = time.perf_counter_ns()
        controller.build_policy_input(
            _observation(
                ts_ms=ts_ms,
                capacity_bytes=capacity,
                hbm_used_bytes=hbm_used,
            )
        )
        lock_delta.append(_elapsed_ms(started_ns))
        ts_ms += 1.0

    planner_source = controller.build_policy_input(
        _observation(
            ts_ms=ts_ms,
            capacity_bytes=capacity,
            hbm_used_bytes=hbm_used,
        )
    )
    planner_input = _planner_input(
        planner_source,
        runnable_count=runnable_count,
    )
    planner_samples: list[float] = []
    planner_reported: list[float] = []
    planner_phases: dict[str, list[float]] = {}
    for _ in range(iterations):
        planner = ObservedJointPlanner(
            JointPlannerConfig(
                max_workflow_candidates=max(1, runnable_count),
                max_total_frontier_candidates=max(1, runnable_count),
                max_package_evaluations=8,
                max_planning_budget_ms=1_000.0,
            )
        )
        started_ns = time.perf_counter_ns()
        plan = planner.plan(planner_input)
        planner_samples.append(_elapsed_ms(started_ns))
        planner_reported.append(plan.planning_ms)
        for name, elapsed_ms in plan.planning_phase_ms:
            planner_phases.setdefault(name, []).append(elapsed_ms)

    controller.page_index.set_engine_lock(target, 0)
    controller.page_index.begin_transfer(target, TransferDirection.D2H)
    controller.page_index.complete_transfer(
        target,
        TransferDirection.D2H,
        keep_gpu=False,
    )
    started_ns = time.perf_counter_ns()
    controller.build_policy_input(
        _observation(
            ts_ms=ts_ms,
            capacity_bytes=capacity,
            hbm_used_bytes=hbm_used - page_bytes,
            host_used_bytes=page_bytes,
        )
    )
    residency_delta_ms = _elapsed_ms(started_ns)

    return {
        "schema_version": 1,
        "page_count": page_count,
        "page_bytes": page_bytes,
        "iterations": iterations,
        "runnable_count": runnable_count,
        "cold_build_ms": cold_ms,
        "unchanged_build": _summary(unchanged),
        "single_lock_delta_build": _summary(lock_delta),
        "single_residency_delta_build_ms": residency_delta_ms,
        "delta_replication": _delta_replication_benchmark(
            page_count=page_count,
            changed_page_count=changed_page_count,
            iterations=iterations,
            page_bytes=page_bytes,
        ),
        "observed_joint_plan_wall": _summary(planner_samples),
        "observed_joint_plan_reported": _summary(planner_reported),
        "observed_joint_plan_phases": {
            name: _summary(samples)
            for name, samples in sorted(planner_phases.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark worker-side PolicyInput materialization."
    )
    parser.add_argument("--pages", type=int, default=4_096)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--page-bytes", type=int, default=98_304)
    parser.add_argument("--runnable", type=int, default=16)
    parser.add_argument("--changed-pages", type=int, default=384)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        page_count=args.pages,
        iterations=args.iterations,
        page_bytes=args.page_bytes,
        runnable_count=args.runnable,
        changed_page_count=args.changed_pages,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
