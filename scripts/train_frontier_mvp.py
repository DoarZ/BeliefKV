#!/usr/bin/env python3
"""Quick MVP training of FrontierBeliefModel without manifest validation.

The formal pipeline (``train_frontier_belief.py``) enforces frozen-split
manifest eligibility and the 5-project / 40-task / 40-workflow diversity gate.
This script intentionally skips manifest validation so MVP iterations can fit
on development-labelled corpora; outputs are marked ``development_only``.
"""
from __future__ import annotations

import argparse
import json, sys, os, glob
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from beliefkv.predictor.structured_frontier import (
    FrontierBeliefModel,
    FrontierModelHyperparameters,
    evaluate_frontier_model,
    summarize_training_corpus,
    validate_training_corpus_diversity,
)


def load_rows(data_dirs: list[str], split: str) -> list[dict]:
    rows = []
    seen = set()
    for data_dir in data_dirs:
        for manifest_path in sorted(
            glob.glob(
                f"{data_dir}/p6-0*/**/dataset/dataset_manifest.json",
                recursive=True,
            )
        ):
            try:
                manifest = json.load(open(manifest_path))
            except Exception:
                continue
            if manifest.get("formal_training_eligible") is not True:
                continue
            data_path = os.path.join(
                os.path.dirname(manifest_path), "frontier_decision_points.jsonl"
            )
            if not os.path.isfile(data_path):
                continue
            with open(data_path) as fh:
                for line in fh:
                    d = json.loads(line)
                    if d.get("split") == split and d.get("training_eligible") is not False:
                        did = d.get("decision_id", "")
                        if did in seen:
                            continue
                        seen.add(did)
                        rows.append(d)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        action="append",
        required=True,
        help="processed dataset root (repeat for multiple roots)",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-version", default="p6-mvp-v1")
    parser.add_argument(
        "--output",
        default=(
            "/home/longhao/experiment/BeliefKV/experiments/models/"
            "frontier_belief_mvp_v1.json"
        ),
    )
    parser.add_argument("--minimum-projects", type=int, default=3)
    parser.add_argument("--minimum-tasks", type=int, default=30)
    parser.add_argument("--minimum-workflows", type=int, default=30)
    args = parser.parse_args()

    data_dirs = args.data_dir
    split = args.split
    model_version = args.model_version
    model_path = args.output

    train_rows = load_rows(data_dirs, "train")
    print(f"Train rows: {len(train_rows)}")

    diversity = validate_training_corpus_diversity(
        train_rows,
        minimum_projects=args.minimum_projects,
        minimum_tasks=args.minimum_tasks,
        minimum_workflows=args.minimum_workflows,
    )
    print(f"Diversity: {json.dumps(diversity, indent=2)}")

    model = FrontierBeliefModel(
        model_version=model_version,
        hyperparameters=FrontierModelHyperparameters(),
    )
    summary = model.fit(train_rows)
    print(f"Training summary: {json.dumps(summary, indent=2)}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path, metadata={
        "fit_split": "train",
        "development_only": True,
        "fit_projects": sorted(set(r.get("project", "unknown") for r in train_rows)),
        "fit_task_count": diversity.get("task_count"),
        "fit_workflow_count": diversity.get("workflow_count"),
        "fit_decision_point_count": len(train_rows),
        "formal_diversity_gate": diversity,
    })
    print(f"Model saved to {model_path}")

    cal_rows = load_rows(data_dirs, "calibration")
    print(f"\nCalibration rows: {len(cal_rows)}")
    if cal_rows:
        eval_result = evaluate_frontier_model(model, cal_rows)
        print(f"Calibration:\n{json.dumps(eval_result, indent=2)}")

    test_rows = load_rows(data_dirs, "test_id")
    print(f"\nTest-ID rows: {len(test_rows)}")
    if test_rows:
        eval_result = evaluate_frontier_model(model, test_rows)
        print(f"Test:\n{json.dumps(eval_result, indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
