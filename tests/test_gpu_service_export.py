from __future__ import annotations

from scripts.export_gpu_service_calibration import (
    _contention_at_sample,
    _transfer_intervals,
)


def test_contention_uses_submit_as_observed_native_upper_bound() -> None:
    intervals = _transfer_intervals(
        (
            {
                "event": "transfer_telemetry",
                "submit_ts_ms": 5.0,
                "start_ts_ms": None,
                "complete_ts_ms": 15.0,
                "command_kind": "native_write_back",
                "compute_phase": "native_hicache",
                "native_concurrent_bytes": 4096,
            },
        )
    )

    state, inflight_bytes, semantics = _contention_at_sample(
        {"service_start_ts_ms": 10.0, "complete_ts_ms": 20.0},
        intervals,
    )

    assert state == "native_hicache_observed"
    assert inflight_bytes == 4096
    assert semantics == "start_or_submit_to_complete_observed_upper_bound"


def test_contention_distinguishes_idle_and_mixed_transfer_windows() -> None:
    intervals = _transfer_intervals(
        (
            {
                "event": "transfer_telemetry",
                "start_ts_ms": 12.0,
                "complete_ts_ms": 18.0,
                "command_kind": "native_write_back",
                "actual_bytes": 1024,
            },
            {
                "event": "transfer_telemetry",
                "start_ts_ms": 14.0,
                "complete_ts_ms": 16.0,
                "command_kind": "offload_context",
                "actual_bytes": 2048,
            },
        )
    )

    assert _contention_at_sample(
        {"service_start_ts_ms": 0.0, "complete_ts_ms": 10.0}, intervals
    ) == ("idle", 0, "observed_no_overlap")
    assert _contention_at_sample(
        {"service_start_ts_ms": 10.0, "complete_ts_ms": 20.0}, intervals
    ) == ("mixed_transfer_observed", 1024, "start_to_complete_overlap")
