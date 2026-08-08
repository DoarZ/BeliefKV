#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beliefkv.experiments.restore_micro_gate import (
    analyze_restore_micro_gate,
    read_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the deterministic P5 restore micro-gate artifacts."
    )
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--runtime-summary", type=Path, required=True)
    parser.add_argument("--gate-id", default="p5g-restore-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.runtime_summary.read_text(encoding="utf-8"))
    result = analyze_restore_micro_gate(
        read_jsonl(args.runtime_audit),
        summary,
        gate_id=args.gate_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
