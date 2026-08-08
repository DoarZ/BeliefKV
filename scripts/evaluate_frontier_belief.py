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
    FrontierBeliefModel,
    evaluate_frontier_model,
    load_evaluation_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local FrontierBelief distributions on a sealed split."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--split", choices=("calibration", "test_id", "test_ood"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_model = json.loads(args.model.read_text(encoding="utf-8"))
    model = FrontierBeliefModel.from_dict(raw_model)
    rows, _ = load_evaluation_rows(args.dataset_dir, split=args.split)
    if not rows:
        raise SystemExit(f"no {args.split} decision points were found")
    evaluation_projects = {
        str(row.get("project") or "unknown") for row in rows
    }
    metadata = raw_model.get("metadata", {})
    forbidden_projects = set(metadata.get("fit_projects", ()))
    if args.split != "calibration":
        forbidden_projects.update(metadata.get("calibration_projects", ()))
    overlap = sorted(forbidden_projects.intersection(evaluation_projects))
    if overlap:
        raise SystemExit(
            f"{args.split} projects overlap model-selection projects: {overlap}"
        )
    metrics = evaluate_frontier_model(model, rows)
    metrics["evaluation_projects"] = sorted(evaluation_projects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
