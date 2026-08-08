#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.predictor.hardware_service import GPUServiceCurveModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the independent GPU service curve.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        required=True,
        help="Independent dataset directory; repeat to merge multiple runs.",
    )
    parser.add_argument("--minimum-support", type=float, default=4.0)
    parser.add_argument("--neighbor-count", type=int, default=48)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--minimum-phase-calibration", type=int, default=20)
    parser.add_argument(
        "--max-relative-error-p95", type=float, default=0.25
    )
    parser.add_argument("--coverage-tolerance", type=float, default=0.03)
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    sources = []
    for dataset_dir in args.dataset_dir:
        manifest_path = dataset_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("dataset_kind") != "independent_gpu_service_calibration"
            or manifest.get("evidence_role") != "controlled_microbenchmark"
        ):
            raise SystemExit(
                "GPU service fitting requires independent controlled microbenchmarks"
            )
        namespace = dataset_dir.resolve().name
        source_rows = [
            json.loads(line)
            for line in (dataset_dir / "gpu_batch_service_intervals.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for row in source_rows:
            rows.append(
                {
                    **row,
                    "sample_id": f"{namespace}:{row['sample_id']}",
                    "source_dataset": str(dataset_dir.resolve()),
                }
            )
        sources.append(
            {
                "dataset_dir": str(dataset_dir.resolve()),
                "row_count": len(source_rows),
                "table_sha256": manifest.get("table_sha256"),
            }
        )
    model = GPUServiceCurveModel(
        minimum_support=args.minimum_support,
        neighbor_count=args.neighbor_count,
    )
    summary = model.fit_cross_calibrated(
        (row for row in rows if row.get("split") == "train"),
        folds=args.calibration_folds,
        minimum_phase_calibration=args.minimum_phase_calibration,
    )
    model.save(args.output)
    evaluation = model.evaluate_controlled_rows(
        row for row in rows if row.get("split") == "holdout"
    )
    online_eligible = bool(
        evaluation["calibrated"]
        and evaluation["unavailable_count"] == 0
        and evaluation["relative_error_p95"] is not None
        and evaluation["relative_error_p95"] <= args.max_relative_error_p95
        and evaluation["p90_coverage"] is not None
        and abs(evaluation["p90_coverage"] - 0.90) <= args.coverage_tolerance
        and evaluation["p95_coverage"] is not None
        and abs(evaluation["p95_coverage"] - 0.95) <= args.coverage_tolerance
        and all(
            value is not None and value <= args.max_relative_error_p95
            for value in evaluation[
                "phase_conditional_cell_relative_error_p95"
            ].values()
        )
        and all(
            abs(value - 0.90) <= args.coverage_tolerance
            for value in evaluation["phase_p90_coverage"].values()
        )
        and all(
            abs(value - 0.95) <= args.coverage_tolerance
            for value in evaluation["phase_p95_coverage"].values()
        )
    )
    result = {
        "output": str(args.output),
        "source_datasets": sources,
        "summary": summary,
        "holdout_evaluation": evaluation,
        "acceptance_gate": {
            "online_eligible": online_eligible,
            "max_relative_error_p95": args.max_relative_error_p95,
            "coverage_tolerance": args.coverage_tolerance,
            "holdout_used_for_fit_or_calibration": False,
        },
    }
    if args.evaluation_output is not None:
        args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.evaluation_output.with_suffix(
            args.evaluation_output.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.evaluation_output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
