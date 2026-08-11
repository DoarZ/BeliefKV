#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beliefkv.experiments.predictive_joint_ab import write_immutable_ab_run_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the R5 paired A/B run plan.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_immutable_ab_run_plan(args.baseline, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
