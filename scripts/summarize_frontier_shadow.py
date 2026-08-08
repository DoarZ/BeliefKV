#!/usr/bin/env python3
"""Summarize online ``frontier_shadow`` audit records from a BeliefKV run.

These records prove that the P6 frontier model loaded and published beliefs in
the serving path without altering decisions. The summary is development-only
evidence; model quality against observed labels is measured separately with
``evaluate_frontier_mvp.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_state: Counter[str] = Counter()
    by_support: Counter[str] = Counter()
    by_boundary_top: Counter[str] = Counter()
    ood_reasons: Counter[str] = Counter()
    workflows: set[str] = set()
    invocations: set[str] = set()
    decode_values: list[float] = []
    wait_values: list[float] = []
    growth_values: list[float] = []
    output_values: list[float] = []
    total = 0
    with args.audit_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "frontier_shadow":
                continue
            total += 1
            by_state[str(record.get("state") or "unknown")] += 1
            by_support[str(record.get("support_level") or "unknown")] += 1
            by_boundary_top[str(record.get("boundary_top") or "")] += 1
            for reason in record.get("ood_reasons") or ():
                ood_reasons[str(reason)] += 1
            workflows.add(str(record.get("workflow_id") or ""))
            invocations.add(str(record.get("invocation_id") or ""))
            decode_values.append(float(record.get("remaining_decode_tokens_p50") or 0.0))
            wait_values.append(float(record.get("remaining_external_wait_ms_p50") or 0.0))
            growth_values.append(float(record.get("prompt_growth_tokens_p50") or 0.0))
            output_values.append(float(record.get("next_output_tokens_p50") or 0.0))

    summary = {
        "development_only": True,
        "record_count": total,
        "workflow_count": len(workflows),
        "invocation_count": len(invocations),
        "by_state": dict(sorted(by_state.items())),
        "by_support_level": dict(sorted(by_support.items())),
        "by_boundary_top": dict(sorted(by_boundary_top.items())),
        "ood_reason_counts": dict(sorted(ood_reasons.items())),
        "p50_of_p50": {
            "remaining_decode_tokens": _p50(decode_values),
            "remaining_external_wait_ms": _p50(wait_values),
            "prompt_growth_tokens": _p50(growth_values),
            "next_output_tokens": _p50(output_values),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
