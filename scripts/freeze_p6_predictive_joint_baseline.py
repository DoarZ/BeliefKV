#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beliefkv.experiments.predictive_joint_baseline import (
    build_baseline_manifest,
    write_baseline_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the R0 P6 baseline")
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--gpu-service", type=Path, required=True)
    parser.add_argument("--transfer-service", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-name", default="NVIDIA RTX 6000 Ada Generation")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--kv-pool-tokens", type=int, default=163_840)
    parser.add_argument("--host-cache-gib", type=int, default=96)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=262_144)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20260811)
    parser.add_argument("--arrival-interval-ms", type=int, default=1500)
    args = parser.parse_args()
    payload = build_baseline_manifest(
        repository_root=ROOT,
        workload_manifest=args.workload_manifest,
        predictor_artifact=args.predictor,
        gpu_service_artifact=args.gpu_service,
        transfer_service_artifact=args.transfer_service,
        model_path=args.model_path,
        gpu_name=args.gpu_name,
        gpu_index=args.gpu_index,
        kv_pool_tokens=args.kv_pool_tokens,
        host_cache_gib=args.host_cache_gib,
        max_running_requests=args.max_running_requests,
        context_length=args.context_length,
        concurrency=args.concurrency,
        random_seed=args.random_seed,
        arrival_interval_ms=args.arrival_interval_ms,
    )
    write_baseline_manifest(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
