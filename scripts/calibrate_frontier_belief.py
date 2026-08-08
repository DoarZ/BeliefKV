#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.predictor.structured_frontier import (
    FrontierBeliefModel,
    load_evaluation_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate a fitted FrontierBeliefModel on held-out projects."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument("--target-coverage", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--development-on-train",
        action="store_true",
        help=(
            "MVP-only mode: calibrate on train/development rows and mark the "
            "artifact development_only. Never use this for formal evidence."
        ),
    )
    args = parser.parse_args()
    if args.output.resolve() == args.model.resolve():
        raise SystemExit("calibration output must not overwrite the fitted model")

    raw_model = json.loads(args.model.read_text(encoding="utf-8"))
    model = FrontierBeliefModel.from_dict(raw_model)
    if args.development_on_train:
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for directory in args.dataset_dir:
            for path in sorted(
                glob.glob(
                    f"{directory}/p6-0*/gpu0/dataset/"
                    "frontier_decision_points.jsonl"
                )
            ):
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row.get("split") not in {"train", "development"}:
                            continue
                        if row.get("training_eligible") is False:
                            continue
                        decision_id = str(row.get("decision_id") or "")
                        if not decision_id or decision_id in seen:
                            continue
                        seen.add(decision_id)
                        rows.append(row)
        if not rows:
            raise SystemExit("no train/development decision points were found")
        manifests: list[dict[str, object]] = []
        summary = model.calibrate(
            rows, target_coverage=args.target_coverage, allow_development=True
        )
    else:
        rows, manifests = load_evaluation_rows(
            args.dataset_dir, split="calibration"
        )
        if not rows:
            raise SystemExit("no calibration decision points were found")
        fit_projects = set(raw_model.get("metadata", {}).get("fit_projects", ()))
        calibration_projects = sorted(
            {str(row.get("project") or "unknown") for row in rows}
        )
        overlap = sorted(fit_projects.intersection(calibration_projects))
        if overlap:
            raise SystemExit(
                f"calibration projects overlap model-fitting projects: {overlap}"
            )
        summary = model.calibrate(
            rows, target_coverage=args.target_coverage
        )
    calibration_projects = sorted(
        {str(row.get("project") or "unknown") for row in rows}
    )
    metadata = dict(raw_model.get("metadata", {}))
    metadata.update(
        {
            "calibration_split": (
                "development_train" if args.development_on_train else "calibration"
            ),
            "calibration_dataset_dirs": [
                str(item.resolve()) for item in args.dataset_dir
            ],
            "calibration_projects": calibration_projects,
        }
    )
    if manifests:
        metadata["calibration_dataset_manifest_digests"] = [
            hashlib.sha256(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for item in manifests
        ]
    if args.development_on_train:
        metadata["development_only"] = True
        metadata["calibration_source"] = "train_development_mvp"
    model.save(args.output, metadata=metadata)
    print(
        json.dumps(
            {"output": str(args.output), "summary": summary},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
