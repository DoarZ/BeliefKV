#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.decision_characterization import (
    build_manifest,
    validate_manifest,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or verify a P6 decision-relevance characterization."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--study-id", required=True)
    freeze.add_argument("--collection-plan", type=Path, required=True)
    freeze.add_argument("--batch-id", required=True)
    freeze.add_argument("--predictor-model", type=Path, required=True)
    freeze.add_argument("--gpu-service-model", type=Path, required=True)
    freeze.add_argument("--transfer-service-model", type=Path, required=True)
    freeze.add_argument("--gpu-index", type=int, default=0)
    freeze.add_argument("--port", type=int, default=18000)
    freeze.add_argument("--pool-tokens", type=int, default=163_840)
    freeze.add_argument("--host-cache-gib", type=int, default=96)
    freeze.add_argument("--max-running-requests", type=int, default=16)
    freeze.add_argument("--context-length", type=int, default=262_144)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--server-config", type=Path)
    args = parser.parse_args()

    if args.command == "freeze":
        payload = build_manifest(
            repository_root=REPOSITORY_ROOT,
            study_id=args.study_id,
            collection_plan=args.collection_plan,
            batch_id=args.batch_id,
            predictor_model=args.predictor_model,
            gpu_service_model=args.gpu_service_model,
            transfer_service_model=args.transfer_service_model,
            gpu_index=args.gpu_index,
            port=args.port,
            pool_tokens=args.pool_tokens,
            host_cache_gib=args.host_cache_gib,
            max_running_requests=args.max_running_requests,
            context_length=args.context_length,
        )
        write_manifest(args.output, payload)
    else:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        server_config = (
            json.loads(args.server_config.read_text(encoding="utf-8"))
            if args.server_config is not None
            else None
        )
        validate_manifest(
            payload,
            repository_root=REPOSITORY_ROOT,
            server_config=server_config,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
