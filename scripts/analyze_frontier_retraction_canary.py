#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beliefkv.experiments.frontier_retraction_canary import (
    analyze_frontier_retraction_canary,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the single FrontierBelief retraction canary."
    )
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary-limit", type=int, default=1)
    args = parser.parse_args()
    with args.runtime_audit.open("r", encoding="utf-8") as handle:
        payload = analyze_frontier_retraction_canary(
            (json.loads(line) for line in handle if line.strip()),
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
