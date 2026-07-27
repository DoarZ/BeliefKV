import json
import tempfile
import unittest
from pathlib import Path

from beliefkv.metrics.transfer_timeline import (
    load_transfer_timeline,
    render_transfer_timeline,
)


class TransferTimelineTest(unittest.TestCase):
    def test_renders_measured_hbm_host_and_transfer_timeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "runtime_audit.jsonl"
            records = [
                {
                    "event": "runtime_initialized",
                    "run_id": "run-a",
                    "ts_ms": 100.0,
                },
                {
                    "event": "resource_snapshot",
                    "run_id": "run-a",
                    "ts_ms": 100.0,
                    "hbm_used_bytes": 800,
                    "hbm_capacity_bytes": 1000,
                    "host_used_bytes": 100,
                    "host_capacity_bytes": 2000,
                },
                {
                    "event": "transfer_telemetry",
                    "run_id": "run-a",
                    "ts_ms": 130.0,
                    "command_id": "d2h-1",
                    "submit_ts_ms": 110.0,
                    "start_ts_ms": 112.0,
                    "first_layer_ready_ts_ms": None,
                    "complete_ts_ms": 130.0,
                    "compute_wait_ms": None,
                    "actual_bytes": 200,
                    "closure_bytes": 220,
                    "merged_operation_count": 0,
                    "direction": "d2h",
                    "source_tier": "gpu",
                    "target_tier": "host",
                    "status": "completed",
                    "reason": "",
                    "page_count": 1,
                    "context_id": "context-a",
                    "context_epoch": 0,
                    "command_kind": "offload_context",
                },
                {
                    "event": "resource_snapshot",
                    "run_id": "run-a",
                    "ts_ms": 131.0,
                    "hbm_used_bytes": 600,
                    "hbm_capacity_bytes": 1000,
                    "host_used_bytes": 300,
                    "host_capacity_bytes": 2000,
                },
                {
                    "event": "transfer_telemetry",
                    "run_id": "run-a",
                    "ts_ms": 135.0,
                    "command_id": "h2d-rejected",
                    "submit_ts_ms": 134.0,
                    "start_ts_ms": None,
                    "first_layer_ready_ts_ms": None,
                    "complete_ts_ms": 135.0,
                    "compute_wait_ms": None,
                    "actual_bytes": 0,
                    "closure_bytes": 900,
                    "merged_operation_count": 0,
                    "direction": "h2d",
                    "source_tier": "host",
                    "target_tier": "gpu",
                    "status": "rejected",
                    "reason": "allocator unavailable",
                    "page_count": 0,
                    "context_id": "context-b",
                    "context_epoch": 0,
                    "command_kind": "prefetch_context",
                },
            ]
            audit.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            timeline = load_transfer_timeline(audit)
            html_path, data_path = render_transfer_timeline(
                timeline, root / "timeline.html"
            )

            self.assertEqual(timeline.run_id, "run-a")
            self.assertEqual(timeline.summary["transfer_count"], 1)
            self.assertEqual(timeline.summary["physical_transfer_count"], 1)
            self.assertEqual(timeline.summary["telemetry_record_count"], 2)
            self.assertEqual(timeline.summary["no_dma_record_count"], 1)
            self.assertEqual(timeline.summary["no_dma_rejected_count"], 1)
            self.assertEqual(
                timeline.summary["directions"]["h2d"]["physical_count"], 0
            )
            self.assertEqual(
                timeline.summary["directions"]["h2d"]["attempted_closure_bytes"],
                900,
            )
            self.assertEqual(timeline.summary["peak_hbm_used_bytes"], 800)
            self.assertIsNone(
                timeline.summary["peak_untracked_allocator_delta_bytes"]
            )
            self.assertEqual(timeline.summary["peak_host_used_bytes"], 300)
            self.assertEqual(timeline.transfers[0].measurement, "backend_telemetry")
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("HBM occupancy", html)
            self.assertIn("Host KV occupancy", html)
            self.assertIn("d2h-1", html)
            self.assertIn("Backend telemetry", html)
            self.assertIn("No-DMA rejects", html)
            self.assertIn('class="physical-transfer"', html)
            self.assertIn('class="no-dma-attempt"', html)
            self.assertIn("h2d-rejected", html)
            self.assertIn("backend_telemetry", html)
            self.assertIn("runtime_resource_snapshot", html)
            self.assertIn('class="allocator-hbm-series"', html)
            self.assertNotIn('class="untracked-allocator-series"', html)
            self.assertEqual(json.loads(data_path.read_text())["run_id"], "run-a")

    def test_legacy_dispatch_ack_is_marked_as_aggregate_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            records = [
                {
                    "event": "transfer_dispatched",
                    "run_id": "legacy",
                    "ts_ms": 10.0,
                    "command_id": "old-1",
                    "kind": "offload_context",
                    "selected_bytes": 400,
                    "action_counts": {"start_d2h": 1},
                },
                {
                    "event": "transfer_acknowledged",
                    "run_id": "legacy",
                    "ts_ms": 20.0,
                    "command_id": "old-1",
                    "actual_bytes": 300,
                    "status": "partial",
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            timeline = load_transfer_timeline(path)

            self.assertEqual(timeline.transfers[0].direction, "d2h")
            self.assertEqual(
                timeline.transfers[0].measurement,
                "legacy_dispatch_ack_aggregate",
            )
            self.assertFalse(timeline.summary["host_telemetry_available"])
            html_path, _ = render_transfer_timeline(
                timeline, Path(temporary) / "legacy.html"
            )
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("not exact DMA intervals", html)
            self.assertIn("legacy_dispatch_ack_aggregate", html)

    def test_terminal_host_drop_is_visible_but_not_counted_as_pcie_dma(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            records = [
                {
                    "event": "transfer_dispatched",
                    "run_id": "terminal-cleanup",
                    "ts_ms": 10.0,
                    "command_id": "drop-1",
                    "kind": "drop_terminal_private",
                    "context_id": "child-context",
                    "selected_bytes": 400,
                    "action_counts": {"drop_host": 1},
                },
                {
                    "event": "transfer_acknowledged",
                    "run_id": "terminal-cleanup",
                    "ts_ms": 11.0,
                    "command_id": "drop-1",
                    "actual_bytes": 400,
                    "status": "completed",
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            timeline = load_transfer_timeline(path)
            html_path, _ = render_transfer_timeline(
                timeline, Path(temporary) / "terminal-cleanup.html"
            )

            self.assertEqual(timeline.transfers[0].direction, "reclaim")
            self.assertEqual(timeline.summary["physical_transfer_count"], 0)
            self.assertEqual(timeline.summary["no_dma_record_count"], 1)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("drop_terminal_private", html)
            self.assertIn("drop-1", html)

    def test_native_hicache_callback_measurement_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "event": "transfer_telemetry",
                        "run_id": "native",
                        "ts_ms": 20.0,
                        "command_id": "native-hicache-h2d-1",
                        "submit_ts_ms": 10.0,
                        "start_ts_ms": None,
                        "complete_ts_ms": 20.0,
                        "actual_bytes": 400,
                        "closure_bytes": 400,
                        "direction": "h2d",
                        "status": "completed",
                        "context_id": "ctx",
                        "command_kind": "native_demand_load",
                        "telemetry_origin": "native_hicache_callback",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            timeline = load_transfer_timeline(path)

            self.assertEqual(timeline.summary["physical_transfer_count"], 1)
            self.assertEqual(
                timeline.transfers[0].measurement,
                "native_hicache_callback",
            )

    def test_legacy_metrics_warn_that_sampling_can_miss_allocator_peaks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit.jsonl"
            metrics = root / "metrics.jsonl"
            audit.write_text(
                json.dumps(
                    {
                        "event": "transfer_dispatched",
                        "run_id": "legacy",
                        "ts_ms": 10.0,
                        "command_id": "old-1",
                        "action_counts": {"start_d2h": 1},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "event": "transfer_acknowledged",
                        "run_id": "legacy",
                        "ts_ms": 20.0,
                        "command_id": "old-1",
                        "actual_bytes": 100,
                        "status": "completed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metrics.write_text(
                json.dumps(
                    {
                        "monotonic_ts_ms": 15.0,
                        "num_used_tokens": 50,
                        "resident_pressure": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            timeline = load_transfer_timeline(
                audit, metrics_path=metrics, kv_bytes_per_token=16
            )
            html_path, _ = render_transfer_timeline(
                timeline, root / "legacy-metrics.html"
            )

            self.assertEqual(
                timeline.summary["resource_sources"],
                ["sglang_metrics_derived_hbm"],
            )
            self.assertEqual(timeline.summary["peak_hbm_used_bytes"], 800)
            self.assertIsNone(
                timeline.summary["peak_untracked_allocator_delta_bytes"]
            )
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("sampled metrics can miss short-lived changes", html)
            self.assertNotIn('class="untracked-allocator-series"', html)
            self.assertIn('class="allocator-hbm-series"', html)

    def test_prefers_runtime_allocator_series_when_sampled_metrics_are_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit.jsonl"
            metrics = root / "metrics.jsonl"
            audit.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in (
                        {
                            "event": "resource_snapshot",
                            "run_id": "combined",
                            "ts_ms": 100.0,
                            "hbm_used_bytes": 900,
                            "hbm_capacity_bytes": 1000,
                            "host_used_bytes": 100,
                            "host_capacity_bytes": 2000,
                        },
                        {
                            "event": "resource_snapshot",
                            "run_id": "combined",
                            "ts_ms": 200.0,
                            "hbm_used_bytes": 800,
                            "hbm_capacity_bytes": 1000,
                            "host_used_bytes": 200,
                            "host_capacity_bytes": 2000,
                        },
                    )
                ),
                encoding="utf-8",
            )
            metrics.write_text(
                json.dumps(
                    {
                        "monotonic_ts_ms": 150.0,
                        "num_used_tokens": 25,
                        "resident_pressure": 0.1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            timeline = load_transfer_timeline(
                audit,
                metrics_path=metrics,
                kv_bytes_per_token=4,
                hbm_capacity_bytes=1000,
            )
            html_path, _ = render_transfer_timeline(
                timeline, root / "combined.html"
            )

            self.assertEqual(timeline.summary["peak_hbm_used_bytes"], 900)
            self.assertIsNone(
                timeline.summary["peak_untracked_allocator_delta_bytes"]
            )
            html = html_path.read_text(encoding="utf-8")
            self.assertIn('class="allocator-hbm-series"', html)
            self.assertIn("Allocator HBM occupancy", html)
            self.assertNotIn("Protected/non-evictable KV", html)

    def test_derives_untracked_delta_and_renders_physical_state_series(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit.jsonl"
            audit.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in (
                        {
                            "event": "resource_snapshot",
                            "run_id": "runtime-protected",
                            "ts_ms": 100.0,
                            "hbm_used_bytes": 900,
                            "page_index_gpu_bytes": 700,
                            "engine_locked_gpu_bytes": 300,
                            "engine_lock_ref_gpu_bytes": 250,
                            "engine_lock_full_attribution_coverage": 0.8,
                            "locked_but_not_served_gpu_bytes_100ms": 200,
                            "locked_but_not_served_gpu_bytes_500ms": 100,
                            "closure_blocked_gpu_bytes": 200,
                            "migratable_gpu_bytes": 150,
                            "dual_resident_gpu_bytes": 100,
                            "hbm_capacity_bytes": 1000,
                            "host_used_bytes": 100,
                            "host_capacity_bytes": 2000,
                        },
                        {
                            "event": "resource_snapshot",
                            "run_id": "runtime-protected",
                            "ts_ms": 200.0,
                            "hbm_used_bytes": 800,
                            "page_index_gpu_bytes": 750,
                            "engine_locked_gpu_bytes": 100,
                            "engine_lock_ref_gpu_bytes": 100,
                            "engine_lock_full_attribution_coverage": 1.0,
                            "locked_but_not_served_gpu_bytes_100ms": 50,
                            "locked_but_not_served_gpu_bytes_500ms": 0,
                            "closure_blocked_gpu_bytes": 50,
                            "migratable_gpu_bytes": 500,
                            "dual_resident_gpu_bytes": 200,
                            "hbm_capacity_bytes": 1000,
                            "host_used_bytes": 200,
                            "host_capacity_bytes": 2000,
                        },
                    )
                ),
                encoding="utf-8",
            )

            timeline = load_transfer_timeline(audit)
            html_path, _ = render_transfer_timeline(
                timeline, root / "runtime-protected.html"
            )

            assert timeline.summary["peak_hbm_used_bytes"] == 900
            assert timeline.summary["peak_untracked_allocator_delta_bytes"] == 200
            assert timeline.summary["peak_engine_locked_gpu_bytes"] == 300
            assert timeline.summary["peak_engine_lock_ref_gpu_bytes"] == 250
            assert (
                timeline.summary[
                    "peak_locked_but_not_served_gpu_100ms_bytes"
                ]
                == 200
            )
            assert (
                timeline.summary[
                    "peak_locked_but_not_served_gpu_500ms_bytes"
                ]
                == 100
            )
            assert timeline.summary["lock_service_sample_count"] == 2
            assert (
                timeline.summary[
                    "locked_but_not_served_lower_bound_ratio_100ms_p50"
                ]
                == 0.65
            )
            assert timeline.summary["peak_closure_blocked_gpu_bytes"] == 200
            assert timeline.summary["peak_migratable_gpu_bytes"] == 500
            assert timeline.summary["peak_dual_resident_gpu_bytes"] == 200
            assert [
                point.untracked_allocator_delta_bytes
                for point in timeline.resources
            ] == [200, 50]
            html = html_path.read_text(encoding="utf-8")
            assert 'class="untracked-allocator-series"' in html
            assert 'class="engine-locked-series"' in html
            assert 'class="locked-not-served-100ms-series"' in html
            assert 'class="locked-not-served-500ms-series"' in html
            assert 'class="closure-blocked-series"' in html
            assert 'class="migratable-series"' in html
            assert 'class="dual-resident-series"' in html
            assert "max(allocator HBM - indexed GPU KV, 0)" in html
            assert "this is not classified as protected KV" in html
            assert "a conservative physical-byte lower bound" in html


if __name__ == "__main__":
    unittest.main()
