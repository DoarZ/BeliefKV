#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.codex_ab import (
    compare_runs,
    summarize_reactive_audit,
    summarize_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a guarded Codex subagent A/B pair.")
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--baseline-server-log", type=Path, required=True)
    parser.add_argument("--reactive-run", type=Path, required=True)
    parser.add_argument("--reactive-server-log", type=Path, required=True)
    parser.add_argument("--reactive-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare_runs(
        summarize_run(args.baseline_run.resolve(), args.baseline_server_log.resolve()),
        summarize_run(args.reactive_run.resolve(), args.reactive_server_log.resolve()),
        summarize_reactive_audit(args.reactive_audit.resolve()),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result["effect"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
