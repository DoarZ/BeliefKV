from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from beliefkv.metrics.summary import percentile


def analyze_policy_snapshot_audit(
    audit_path: Path,
    *,
    snapshot_path: Path | None = None,
) -> dict[str, object]:
    build_ms: list[float] = []
    bundle_counts: list[int] = []
    trigger_counts: Counter[str] = Counter()
    by_scale: dict[str, list[float]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    runtime_snapshot_config: Mapping[str, object] = {}
    scheduler_p99_ms: float | None = None
    summary_count: int | None = None
    written_count: int | None = None
    for raw in _read_jsonl(audit_path):
        event = raw.get("event")
        if event == "runtime_initialized":
            value = raw.get("reference_policy_snapshot", {})
            if isinstance(value, Mapping):
                runtime_snapshot_config = value
        elif event == "policy_snapshot_recorded":
            elapsed = float(raw["build_ms"])
            count = int(raw["physical_bundle_count"])
            build_ms.append(elapsed)
            bundle_counts.append(count)
            trigger_counts[str(raw.get("trigger", "unknown"))] += 1
            by_scale[_scale_bucket(count)].append(elapsed)
        elif event == "policy_snapshot_failed":
            failure_counts[str(raw.get("error", "unknown"))] += 1
        elif event == "controller_timing_summary":
            scheduler_p99_ms = float(raw["scheduler_step_p99_ms"])
        elif event == "policy_snapshot_summary":
            summary_count = int(raw.get("snapshot_count", 0))
            if raw.get("written_snapshot_count") is not None:
                written_count = int(raw["written_snapshot_count"])

    snapshot_bytes = (
        snapshot_path.expanduser().resolve().stat().st_size
        if snapshot_path is not None
        else None
    )
    writer_mode = str(runtime_snapshot_config.get("writer_mode", "legacy_synchronous"))
    result = {
        "schema_version": 1,
        "audit_path": str(audit_path.expanduser().resolve()),
        "snapshot_path": (
            str(snapshot_path.expanduser().resolve())
            if snapshot_path is not None
            else None
        ),
        "writer_mode": writer_mode,
        "recorded_snapshot_count": len(build_ms),
        "summary_snapshot_count": summary_count,
        "written_snapshot_count": written_count,
        "snapshot_failure_count": sum(failure_counts.values()),
        "snapshot_failure_counts": dict(sorted(failure_counts.items())),
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "physical_bundle_count": _summary(bundle_counts),
        "safe_point_snapshot_ms": _summary(build_ms),
        "safe_point_snapshot_ms_by_extent_scale": {
            name: _summary(values) for name, values in sorted(by_scale.items())
        },
        "scheduler_step_p99_ms": scheduler_p99_ms,
        "snapshot_to_scheduler_p99_ratio": (
            percentile(build_ms, 99) / scheduler_p99_ms
            if build_ms and scheduler_p99_ms and scheduler_p99_ms > 0
            else None
        ),
        "compressed_snapshot_bytes": snapshot_bytes,
        "compressed_bytes_per_snapshot": (
            snapshot_bytes / len(build_ms)
            if snapshot_bytes is not None and build_ms
            else None
        ),
        "timing_caveat": (
            "legacy build_ms includes PolicyInput serialization, gzip and flush"
            if writer_mode == "legacy_synchronous"
            else "build_ms ends after immutable PolicyInput enqueue; gzip runs off-thread"
        ),
    }
    return result


def _summary(values: Iterable[int | float]) -> dict[str, float | int | None]:
    samples = [float(item) for item in values]
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "max": max(samples),
    }


def _scale_bucket(bundle_count: int) -> str:
    if bundle_count <= 32:
        return "000-032"
    if bundle_count <= 64:
        return "033-064"
    if bundle_count <= 128:
        return "065-128"
    return "129-plus"


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"audit line {line_number} must be an object")
            yield raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze BeliefKV common-policy snapshot overhead."
    )
    parser.add_argument("audit", type=Path)
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = analyze_policy_snapshot_audit(
        args.audit,
        snapshot_path=args.snapshots,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
