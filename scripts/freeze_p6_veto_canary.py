#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.predictive_canary import (
    build_veto_canary_manifest,
    validate_veto_canary_manifest,
    write_veto_canary_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the paired P6 morphology-veto canary."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--study-id", required=True)
    freeze.add_argument("--characterization-manifest", type=Path, required=True)
    freeze.add_argument("--replay-comparison", type=Path, required=True)
    freeze.add_argument("--harness-profiles", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument(
        "--arm",
        choices=("p5_observed_control", "byte_only_veto_treatment"),
    )
    verify.add_argument("--server-config", type=Path)
    args = parser.parse_args()

    if args.command == "freeze":
        payload = build_veto_canary_manifest(
            repository_root=REPOSITORY_ROOT,
            study_id=args.study_id,
            characterization_manifest=args.characterization_manifest,
            replay_comparison=args.replay_comparison,
            harness_profiles=args.harness_profiles,
        )
        write_veto_canary_manifest(args.output, payload)
    else:
        if (args.arm is None) != (args.server_config is None):
            parser.error("--arm and --server-config must be supplied together")
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        server_config = (
            json.loads(args.server_config.read_text(encoding="utf-8"))
            if args.server_config is not None
            else None
        )
        validate_veto_canary_manifest(
            payload,
            repository_root=REPOSITORY_ROOT,
            arm=args.arm,
            server_config=server_config,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
