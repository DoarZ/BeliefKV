#!/usr/bin/env python3
"""MVP-only evaluation of a FrontierBeliefModel on exported decision points.

Unlike the formal evaluator, this script does not enforce formal training
eligibility or frozen-split provenance, because MVP shadow runs carry
``predictor_enabled=true`` and are ineligible by design. Outputs are labeled
``development_only`` and must never be presented as paper evidence.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.predictor.structured_frontier import (
    FrontierBeliefModel,
    evaluate_frontier_model,
)


def load_rows(dataset_dirs: list[str], split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for directory in dataset_dirs:
        candidates = [f"{directory}/frontier_decision_points.jsonl"]
        candidates += sorted(
            glob.glob(f"{directory}/p6-0*/gpu0/dataset/frontier_decision_points.jsonl")
        )
        for path in candidates:
            if not Path(path).is_file():
                continue
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("split") != split:
                        continue
                    if row.get("training_eligible") is False:
                        continue
                    decision_id = str(row.get("decision_id") or "")
                    if not decision_id or decision_id in seen:
                        continue
                    seen.add(decision_id)
                    rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--split", choices=("calibration", "test_id", "test_ood"), default="test_id"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_model = json.loads(args.model.read_text(encoding="utf-8"))
    model = FrontierBeliefModel.from_dict(raw_model)
    rows = load_rows([str(item) for item in args.dataset_dir], args.split)
    if not rows:
        raise SystemExit(f"no {args.split} decision points were found")
    metrics = evaluate_frontier_model(model, rows)
    metrics["development_only"] = True
    metrics["evaluation_projects"] = sorted(
        {str(row.get("project") or "unknown") for row in rows}
    )
    metrics["dataset_dirs"] = [str(item.resolve()) for item in args.dataset_dir]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
