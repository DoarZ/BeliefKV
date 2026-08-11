#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from beliefkv.experiments.predictive_joint_canary import (
    analyze_predictive_prepare_canary,
)


def _records(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one natural predictive JointPlan PREPARE_HOST action."
    )
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary-limit", type=int, default=1)
    args = parser.parse_args()

    payload = analyze_predictive_prepare_canary(
        _records(args.runtime_audit),
        canary_limit=args.canary_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
