#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.traces.runtime_validation import (
    read_jsonl,
    validate_runtime_trace,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify hashes, event invariants, and replay of a frozen trace."
    )
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()

    trace_dir = args.trace_dir.resolve()
    manifest = _read_json(trace_dir / "manifest.json")
    if manifest.get("trace_valid") is not True:
        raise ValueError("manifest does not mark the trace as valid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest has no artifact lock")
    for name, expected in artifacts.items():
        path = trace_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact is missing: {path}")
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"artifact size mismatch: {name}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")

    summary = validate_runtime_trace(
        trace_dir / "runtime_events.authoritative.jsonl",
        trace_dir / "runtime_audit.jsonl",
    )
    if summary.to_dict() != manifest.get("validation"):
        raise ValueError("current validation summary disagrees with manifest")
    if summary.to_dict() != _read_json(trace_dir / "trace_summary.json"):
        raise ValueError("trace_summary.json disagrees with validated trace")

    authoritative = read_jsonl(
        trace_dir / "runtime_events.authoritative.jsonl"
    )
    relative = read_jsonl(trace_dir / "runtime_events.relative.jsonl")
    if len(authoritative) != len(relative):
        raise ValueError("relative and authoritative event counts differ")
    base_ts = float(authoritative[0]["ts_ms"])
    for raw, normalized in zip(authoritative, relative):
        raw_without_ts = {key: value for key, value in raw.items() if key != "ts_ms"}
        normalized_without_ts = {
            key: value for key, value in normalized.items() if key != "ts_ms"
        }
        if raw_without_ts != normalized_without_ts:
            raise ValueError("relative trace modified a non-timestamp field")
        expected_ts = float(raw["ts_ms"]) - base_ts
        if not math.isclose(
            float(normalized["ts_ms"]), expected_ts, abs_tol=1e-9
        ):
            raise ValueError("relative trace timestamp mismatch")

    result = {
        "trace_dir": str(trace_dir),
        "trace_id": manifest.get("trace_id"),
        "task_outcome": manifest.get("task_outcome"),
        "artifact_count": len(artifacts),
        "event_count": summary.event_count,
        "controller_replay_valid": summary.controller_replay_valid,
        "verified": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
