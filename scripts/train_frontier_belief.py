#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.predictor.structured_frontier import (
    FrontierBeliefModel,
    FrontierModelHyperparameters,
    load_decision_rows,
    validate_training_corpus_diversity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the local P6 FrontierBeliefModel")
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--split", choices=("train", "development"), default="train"
    )
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--minimum-projects", type=int, default=5)
    parser.add_argument("--minimum-tasks", type=int, default=40)
    parser.add_argument("--minimum-workflows", type=int, default=40)
    parser.add_argument(
        "--hyperparameter-selection",
        type=Path,
        help="LOPO selection manifest generated from the same formal train projects.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, manifests = load_decision_rows(
        args.dataset_dir, allowed_splits=(args.split,)
    )
    if not rows:
        raise SystemExit(f"no {args.split} decision points were found")
    projects = sorted({str(row.get("project") or "unknown") for row in rows})
    diversity = None
    if args.split == "train":
        diversity = validate_training_corpus_diversity(
            rows,
            minimum_projects=args.minimum_projects,
            minimum_tasks=args.minimum_tasks,
            minimum_workflows=args.minimum_workflows,
        )
    hyperparameters = FrontierModelHyperparameters()
    selection_digest = None
    if args.hyperparameter_selection is not None:
        selection_raw = json.loads(
            args.hyperparameter_selection.read_text(encoding="utf-8")
        )
        if selection_raw.get("selection_method") != (
            "leave_one_train_project_out_project_macro"
        ):
            raise SystemExit("unsupported hyperparameter selection method")
        if sorted(selection_raw.get("projects", ())) != projects:
            raise SystemExit(
                "hyperparameter selection projects do not match fit projects"
            )
        hyperparameters = FrontierModelHyperparameters.from_dict(
            selection_raw.get("selected_hyperparameters")
        )
        selection_digest = hashlib.sha256(
            args.hyperparameter_selection.read_bytes()
        ).hexdigest()
    model = FrontierBeliefModel(
        model_version=args.model_version,
        hyperparameters=hyperparameters,
    )
    summary = model.fit(rows)
    tasks = sorted(
        {
            (
                str(row.get("project") or "unknown"),
                str(row.get("instance_id") or "unknown"),
                str(row.get("base_commit") or "unknown"),
            )
            for row in rows
        }
    )
    manifest_digests = [
        hashlib.sha256(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for item in manifests
    ]
    model.save(
        args.output,
        metadata={
            "fit_split": args.split,
            "development_only": args.split == "development",
            "dataset_manifest_digests": manifest_digests,
            "dataset_dirs": [str(item.resolve()) for item in args.dataset_dir],
            "fit_projects": projects,
            "fit_task_count": len(tasks),
            "formal_diversity_gate": diversity,
            "hyperparameter_selection_path": (
                str(args.hyperparameter_selection.resolve())
                if args.hyperparameter_selection is not None
                else None
            ),
            "hyperparameter_selection_sha256": selection_digest,
        },
    )
    print(
        json.dumps(
            {"output": str(args.output), "summary": summary, "diversity": diversity},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
