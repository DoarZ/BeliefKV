#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from beliefkv.experiments.predictive_joint_ab import (
    PredictiveJointABRun,
    compare_predictive_joint_ab,
)


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _gpu_utilization(path: Path) -> tuple[float, ...]:
    if not path.exists():
        return ()
    values = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 6:
                continue
            try:
                values.append(float(row[3].strip()))
            except ValueError:
                continue
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the frozen R5 A/B runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="PAIR:ARM=/absolute/run/directory (three complete pairs required)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for value in args.run:
        identity, separator, raw_path = value.partition("=")
        pair_id, arm_separator, arm = identity.partition(":")
        if not separator or not arm_separator:
            raise ValueError(f"invalid --run value: {value}")
        path = Path(raw_path).resolve()
        summary_path = path / "p6_collection_summary.json"
        if not summary_path.exists():
            summary_path = path / "summary.json"
        runs.append(
            PredictiveJointABRun(
                pair_id=pair_id,
                arm=arm,
                summary=json.loads(
                    summary_path.read_text(encoding="utf-8")
                ),
                audit_records=_jsonl(path / "server/runtime_audit.jsonl"),
                gpu_utilization=_gpu_utilization(path / "gpu_samples.csv"),
            )
        )
    payload = compare_predictive_joint_ab(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
