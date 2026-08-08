#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.p6_invariance import audit_paired_load_invariance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit semantic-demand invariance across GPU load cohorts."
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        help="LABEL=DATASET_DIR; provide at least two load cohorts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = {}
    for raw in args.cohort:
        label, separator, directory = raw.partition("=")
        if not separator or not label or not directory or label in cohorts:
            raise SystemExit(f"invalid or duplicate cohort: {raw}")
        cohorts[label] = Path(directory)
    report = audit_paired_load_invariance(cohorts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
