#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.p6_coverage import write_p6_coverage
from beliefkv.experiments.p6_dataset import export_p6_training_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Characterize P6.0 label coverage on one fixed BeliefKV run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="also export versioned P6 training-evidence tables",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="explicit frozen project/task split; omit only for development-only data",
    )
    parser.add_argument(
        "--allow-censored",
        action="store_true",
        help="analyze workloads.incomplete without treating it as a valid run",
    )
    parser.add_argument(
        "--allow-development-only",
        action="store_true",
        help=(
            "export an explicitly development-only dataset even when the run "
            "was collected with the predictor enabled (MVP shadow evidence; "
            "never treated as formal training evidence)"
        ),
    )
    parser.add_argument(
        "--censor-reason",
        help="record why a deliberately stopped characterization was censored",
    )
    args = parser.parse_args()

    result = write_p6_coverage(
        args.run_dir,
        args.output,
        allow_censored=args.allow_censored,
        censor_reason=args.censor_reason,
    )
    dataset = None
    if args.dataset_dir is not None:
        dataset = export_p6_training_dataset(
            args.run_dir,
            args.dataset_dir,
            allow_censored=args.allow_censored,
            allow_development_only=args.allow_development_only,
            split_manifest=args.split_manifest,
        )
    summary = {
        "output": str(args.output.resolve()),
        "dataset_manifest": (
            str((args.dataset_dir / "dataset_manifest.json").resolve())
            if args.dataset_dir is not None
            else None
        ),
        "dataset_training_readiness": (
            dataset["training_readiness"] if dataset is not None else None
        ),
        "action_frontier": result["action_frontier"],
        "gpu_service": result["gpu_service"],
        "gates": result["gates"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
