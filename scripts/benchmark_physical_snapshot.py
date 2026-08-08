#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from statistics import mean
from types import SimpleNamespace

from beliefkv.metrics.summary import percentile
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata
from beliefkv.runtime.sglang_v052rc1 import EmbeddedSGLangRuntime


PHASE_FIELDS = (
    "queue_collection_ms",
    "metadata_indexing_ms",
    "radix_ownership_lookup_ms",
    "operation_indexing_ms",
    "sorting_allocation_ms",
)


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": mean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }


def _runtime(request_count: int) -> EmbeddedSGLangRuntime:
    runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
    requests = []
    metadata = {}
    submitted = {}
    for index in range(request_count):
        request_id = f"request-{index:04d}"
        node = SimpleNamespace(lock_ref=int(index % 4 == 0))
        request = SimpleNamespace(
            rid=request_id,
            req_pool_idx=index if index % 3 == 1 else None,
            last_node=node,
            load_operation_id=(
                f"native-load-{index:04d}" if index % 11 == 0 else None
            ),
        )
        requests.append(request)
        metadata[request_id] = BeliefKVRequestMetadata(
            f"workflow-{index:04d}",
            f"invocation-{index:04d}",
            f"context-{index:04d}",
            index % 3,
        )
        submitted[request_id] = float(index + 1)
    waiting_end = request_count // 3
    running_end = 2 * request_count // 3
    runtime.scheduler = SimpleNamespace(
        waiting_queue=requests[:waiting_end],
        running_batch=SimpleNamespace(reqs=requests[waiting_end:running_end]),
        chunked_req=(requests[running_end] if running_end < request_count else None),
    )
    runtime._request_metadata_by_id = metadata
    runtime._request_submitted_ts_by_id = submitted
    runtime._h2d_context_by_command = {
        f"explicit-h2d-{index:04d}": (
            f"context-{index:04d}",
            (f"page:{index}:0",),
        )
        for index in range(0, request_count, 8)
    }
    runtime._terminal_cancelled_request_ids = set()
    runtime._native_physical_snapshot_counts = Counter()
    return runtime


def _one_size(
    request_count: int,
    *,
    iterations: int,
    scheduler_steps_per_build: int,
) -> dict[str, object]:
    runtime = _runtime(request_count)
    runtime._begin_physical_safe_point_apply_events()
    runtime._begin_physical_safe_point_capture_and_plan()
    cold = runtime._lazy_safe_point_physical_snapshot()
    if len(cold.records) != request_count:
        raise RuntimeError("cold snapshot lost native request ownership")

    warm = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(iterations):
            runtime._finish_physical_safe_point()
            runtime._begin_physical_safe_point_apply_events()
            runtime._begin_physical_safe_point_capture_and_plan()
            snapshot = runtime._lazy_safe_point_physical_snapshot()
            reused = runtime._lazy_safe_point_physical_snapshot()
            if reused is not snapshot or len(snapshot.by_request) != request_count:
                raise RuntimeError("epoch-local snapshot reuse is inconsistent")
            warm.append(snapshot.timing)
    finally:
        if gc_was_enabled:
            gc.enable()

    totals = [item.total_ms for item in warm]
    return {
        "request_count": request_count,
        "iterations": iterations,
        "cold_build_ms": cold.timing.total_ms,
        "warm_build": _distribution(totals),
        "warm_mean_us_per_record": mean(totals) * 1000.0 / request_count,
        "amortized_cpu_ms_per_scheduler_step": (
            mean(totals) / scheduler_steps_per_build
        ),
        "phase_ms": {
            field: _distribution([getattr(item, field) for item in warm])
            for field in PHASE_FIELDS
        },
        "cache_hit_count": runtime._native_physical_snapshot_counts[
            "cache_hit_count"
        ],
        "gc_collections_during_warm_build": sum(
            item.gc_collections for item in warm
        ),
    }


def _linear_slope(points: list[tuple[int, float]]) -> float:
    x_mean = mean(item[0] for item in points)
    y_mean = mean(item[1] for item in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def run(
    cardinalities: tuple[int, ...],
    *,
    iterations: int,
    scheduler_steps_per_build: int,
) -> dict[str, object]:
    if not cardinalities or min(cardinalities) <= 0:
        raise ValueError("snapshot cardinalities must be positive")
    if iterations <= 0 or scheduler_steps_per_build <= 0:
        raise ValueError("benchmark iteration counts must be positive")
    results = [
        _one_size(
            request_count,
            iterations=iterations,
            scheduler_steps_per_build=scheduler_steps_per_build,
        )
        for request_count in cardinalities
    ]
    slope = _linear_slope(
        [
            (int(item["request_count"]), float(item["warm_build"]["mean_ms"]))
            for item in results
        ]
    )
    return {
        "schema_version": 1,
        "semantics": "lazy_safe_point_physical_snapshot_cpu_microbenchmark",
        "cardinalities": list(cardinalities),
        "iterations_per_cardinality": iterations,
        "scheduler_steps_per_build": scheduler_steps_per_build,
        "warm_gc_disabled": True,
        "mean_build_slope_us_per_request": slope * 1000.0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark lazy SafePointPhysicalSnapshot construction."
    )
    parser.add_argument("--cardinalities", default="8,16,32,64,128")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--scheduler-steps-per-build", type=int, default=1729)
    parser.add_argument("--output")
    args = parser.parse_args()
    cardinalities = tuple(
        int(item.strip())
        for item in args.cardinalities.split(",")
        if item.strip()
    )
    result = run(
        cardinalities,
        iterations=args.iterations,
        scheduler_steps_per_build=args.scheduler_steps_per_build,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
