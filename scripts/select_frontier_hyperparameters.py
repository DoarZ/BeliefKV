#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.predictor.structured_frontier import (
    load_decision_rows,
    select_frontier_hyperparameters,
    validate_training_corpus_diversity,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select FrontierBeliefModel parameters within train projects."
    )
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument("--minimum-projects", type=int, default=5)
    parser.add_argument("--minimum-tasks", type=int, default=40)
    parser.add_argument("--minimum-workflows", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, _ = load_decision_rows(args.dataset_dir, allowed_splits=("train",))
    diversity = validate_training_corpus_diversity(
        rows,
        minimum_projects=args.minimum_projects,
        minimum_tasks=args.minimum_tasks,
        minimum_workflows=args.minimum_workflows,
    )
    report = select_frontier_hyperparameters(rows)
    report["formal_diversity_gate"] = diversity
    report["dataset_dirs"] = [str(path.resolve()) for path in args.dataset_dir]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
