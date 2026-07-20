#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a no-prediction BeliefKV config for Deep Agents experiments."
    )
    parser.add_argument("--server-dir", type=Path, required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, default=98_304)
    parser.add_argument("--pcie-bandwidth-gbps", type=float, default=24.0)
    parser.add_argument("--runtime-event-max-lateness-ms", type=float, default=60_000.0)
    args = parser.parse_args()

    server_dir = args.server_dir.expanduser().resolve()
    config_path = server_dir / "beliefkv_config.json"
    if config_path.exists():
        raise FileExistsError(f"server config already exists: {config_path}")
    key = hashlib.sha256(str(server_dir).encode("utf-8")).hexdigest()[:12]
    socket_path = Path(f"/tmp/bkv-da-{key}.sock")
    if socket_path.exists():
        raise FileExistsError(f"runtime event socket already exists: {socket_path}")
    config = {
        "pcie_bandwidth_gbps": args.pcie_bandwidth_gbps,
        "transfer_overhead_ms": 0.08,
        "transfer_watchdog_factor": 20.0,
        "transfer_watchdog_floor_ms": 1000.0,
        "urgent_chunk_bytes": 268_435_456,
        "shadow_chunk_bytes": 67_108_864,
        "shadow_min_parked_ms": 25.0,
        "shadow_slowdown_budget": 0.02,
        "planning_interval_ms": 5.0,
        "admission_liveness_timeout_ms": 1000.0,
        "admission_force_progress_timeout_ms": 5000.0,
        "kv_bytes_per_token": args.kv_bytes_per_token,
        "predictor_enabled": False,
        "predictor_model_path": None,
        "shadow_enabled": False,
        "prefetch_enabled": True,
        "runtime_audit_path": str(server_dir / "runtime_audit.jsonl"),
        "transfer_telemetry_path": str(server_dir / "transfer_telemetry.jsonl"),
        "runtime_event_socket_path": str(socket_path),
        "runtime_event_log_path": str(server_dir / "runtime_events.sglang.jsonl"),
        "runtime_event_max_lateness_ms": args.runtime_event_max_lateness_ms,
        "resource_telemetry_interval_ms": 50.0,
        "service_curve_window": 256,
        "service_curve_min_samples": 8,
        "transfer_retry_guard_enabled": True,
        "transfer_retry_max_same_snapshot_attempts": 1,
        "transfer_retry_unknown_base_ms": 10.0,
        "transfer_retry_unknown_max_ms": 1000.0,
        "transfer_retry_unknown_circuit_breaker_failures": 8,
    }
    write_json(config_path, config)
    print(
        json.dumps(
            {"config_path": str(config_path), "control_socket": str(socket_path)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
