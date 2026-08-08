import json
import tempfile
import unittest
from pathlib import Path

from beliefkv.metrics.transfer_validation import validate_transfer_audit


class TransferValidationTest(unittest.TestCase):
    def test_host_mismatch_is_transient_only_while_dma_is_inflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            records = [
                {
                    "event": "resource_snapshot",
                    "run_id": "run-inflight",
                    "sequence": 1,
                    "ts_ms": 1.0,
                    "host_used_bytes": 200,
                    "page_index_cpu_bytes": 100,
                    "inflight_command_count": 1,
                },
                {
                    "event": "resource_snapshot",
                    "run_id": "run-inflight",
                    "sequence": 2,
                    "ts_ms": 2.0,
                    "host_used_bytes": 200,
                    "page_index_cpu_bytes": 200,
                    "inflight_command_count": 0,
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = validate_transfer_audit(path)
            resources = report["resource_consistency"]

            self.assertEqual(resources["host_page_index_mismatch_count"], 1)
            self.assertEqual(
                resources["host_page_index_inflight_mismatch_count"], 1
            )
            self.assertEqual(
                resources["host_page_index_quiescent_mismatch_count"], 0
            )
            self.assertTrue(resources["host_residency_matches_page_index"])

    def test_host_mismatch_fails_when_no_dma_is_inflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "event": "resource_snapshot",
                        "run_id": "run-quiescent",
                        "sequence": 1,
                        "ts_ms": 1.0,
                        "host_used_bytes": 200,
                        "page_index_cpu_bytes": 100,
                        "inflight_command_count": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            resources = validate_transfer_audit(path)["resource_consistency"]

            self.assertEqual(
                resources["host_page_index_quiescent_mismatch_count"], 1
            )
            self.assertFalse(resources["host_residency_matches_page_index"])

    def test_native_hicache_telemetry_does_not_require_command_ack(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            records = [
                {
                    "event": "runtime_initialized",
                    "run_id": "run-native",
                    "sequence": 1,
                    "ts_ms": 1.0,
                },
                {
                    "event": "transfer_telemetry",
                    "run_id": "run-native",
                    "sequence": 2,
                    "ts_ms": 20.0,
                    "command_id": "native-hicache-1",
                    "telemetry_origin": "native_hicache_callback",
                    "submit_ts_ms": 10.0,
                    "start_ts_ms": None,
                    "complete_ts_ms": 20.0,
                    "actual_bytes": 1000,
                    "closure_bytes": 1000,
                    "direction": "d2h",
                    "source_tier": "gpu",
                    "target_tier": "host",
                    "status": "completed",
                    "page_count": 1,
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = validate_transfer_audit(
                path,
                service_curve_min_samples=1,
                holdout_fraction=0.5,
            )

            integrity = report["command_integrity"]
            self.assertTrue(integrity["passes"])
            self.assertEqual(integrity["telemetry_count"], 1)
            self.assertEqual(integrity["command_telemetry_count"], 0)
            self.assertEqual(integrity["native_telemetry_count"], 1)
            self.assertEqual(integrity["telemetry_without_dispatch_count"], 0)
            self.assertEqual(integrity["telemetry_without_ack_count"], 0)

    def test_validates_ack_order_resources_and_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            records = [
                {
                    "event": "runtime_initialized",
                    "run_id": "run-a",
                    "sequence": 1,
                    "ts_ms": 1.0,
                }
            ]
            sequence = 2
            for index, duration in enumerate((10.0, 11.0, 10.5, 10.2), start=1):
                command_id = f"command-{index}"
                records.extend(
                    [
                        {
                            "event": "transfer_dispatched",
                            "run_id": "run-a",
                            "sequence": sequence,
                            "ts_ms": index * 100.0,
                            "command_id": command_id,
                            "selected_bytes": 1000,
                            "action_counts": {"start_d2h": 1},
                        },
                        {
                            "event": "transfer_acknowledged",
                            "run_id": "run-a",
                            "sequence": sequence + 1,
                            "ts_ms": index * 100.0 + duration,
                            "command_id": command_id,
                            "actual_bytes": 1000,
                            "status": "completed",
                        },
                        {
                            "event": "transfer_telemetry",
                            "run_id": "run-a",
                            "sequence": sequence + 2,
                            "ts_ms": index * 100.0 + duration,
                            "command_id": command_id,
                            "submit_ts_ms": index * 100.0,
                            "start_ts_ms": index * 100.0 + 1.0,
                            "complete_ts_ms": index * 100.0 + duration,
                            "actual_bytes": 1000,
                            "closure_bytes": 1000,
                            "direction": "d2h",
                            "source_tier": "gpu",
                            "target_tier": "host",
                            "status": "completed",
                            "page_count": 1,
                        },
                    ]
                )
                sequence += 3
            records.append(
                {
                    "event": "resource_snapshot",
                    "run_id": "run-a",
                    "sequence": sequence,
                    "ts_ms": 600.0,
                    "hbm_used_bytes": 800,
                    "hbm_free_bytes": 200,
                    "hbm_capacity_bytes": 1000,
                    "page_index_gpu_bytes": 700,
                    "host_used_bytes": 300,
                    "host_free_bytes": 700,
                    "host_capacity_bytes": 1000,
                    "page_index_cpu_bytes": 300,
                }
            )
            records.append(
                {
                    "event": "controller_timing_summary",
                    "run_id": "run-a",
                    "sequence": sequence + 1,
                    "ts_ms": 601.0,
                    "scheduler_step_sample_count": 100,
                    "scheduler_step_p99_ms": 10.0,
                    "telemetry_event_step_count": 4,
                    "telemetry_event_count": 4,
                    "telemetry_event_step_p99_ms": 12.0,
                    "telemetry_event_overhead_p99_ms": 0.2,
                    "telemetry_event_overhead_ratio_p99": 0.02,
                }
            )
            records.extend(
                [
                    {
                        "event": "request_deferred",
                        "run_id": "run-a",
                        "sequence": sequence + 2,
                        "ts_ms": 700.0,
                        "request_id": "request-a",
                    },
                    {
                        "event": "admission_decision",
                        "run_id": "run-a",
                        "sequence": sequence + 3,
                        "ts_ms": 725.0,
                        "request_id": "request-a",
                        "reason": "insufficient_actual_hbm",
                    },
                    {
                        "event": "request_admitted",
                        "run_id": "run-a",
                        "sequence": sequence + 4,
                        "ts_ms": 750.0,
                        "request_id": "request-a",
                    },
                ]
            )
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = validate_transfer_audit(
                path,
                service_curve_min_samples=1,
                holdout_fraction=0.25,
            )

            self.assertTrue(report["command_integrity"]["passes"])
            self.assertEqual(
                report["command_integrity"]["expected_dma_command_count"], 4
            )
            self.assertTrue(
                report["resource_consistency"]["host_residency_matches_page_index"]
            )
            self.assertEqual(
                report["resource_consistency"][
                    "allocator_minus_hbm_mirror_max_bytes"
                ],
                100.0,
            )
            self.assertEqual(
                report["service_curve_holdout"]["holdout_count"], 1
            )
            self.assertTrue(
                report["controller_telemetry_overhead"]["passes_event_tick_p99"]
            )
            self.assertEqual(
                report["admission_liveness"],
                {
                    "available": True,
                    "admitted_sample_count": 1,
                    "wait_p50_ms": 50.0,
                    "wait_p90_ms": 50.0,
                    "wait_p95_ms": 50.0,
                    "wait_p99_ms": 50.0,
                    "wait_mean_ms": 50.0,
                    "wait_max_ms": 50.0,
                    "reason_counts": {"insufficient_actual_hbm": 1},
                },
            )

    def test_reports_retry_guard_concentration_and_release_latency(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            records = [
                {
                    "event": "transfer_dispatched",
                    "run_id": "run-guard",
                    "sequence": 1,
                    "ts_ms": 10.0,
                    "command_id": "reactive-287",
                    "kind": "prefetch_context",
                    "context_id": "ctx",
                    "context_epoch": 0,
                    "closure_fingerprint": "fingerprint-a",
                    "selected_bytes": 400,
                    "action_counts": {},
                },
                {
                    "event": "transfer_acknowledged",
                    "run_id": "run-guard",
                    "sequence": 2,
                    "ts_ms": 10.1,
                    "command_id": "reactive-287",
                    "status": "rejected",
                    "actual_bytes": 0,
                },
                {
                    "event": "transfer_attempt_blocked",
                    "run_id": "run-guard",
                    "sequence": 3,
                    "ts_ms": 10.1,
                    "context_id": "ctx",
                    "blocker_codes": ["ancestor_closure", "device_capacity"],
                    "failed_ts_ms": 10.1,
                },
                {
                    "event": "transfer_retry_suppressed",
                    "run_id": "run-guard",
                    "sequence": 4,
                    "ts_ms": 11.0,
                    "context_id": "ctx",
                    "suppressed_count": 1,
                },
                {
                    "event": "transfer_retry_released",
                    "run_id": "run-guard",
                    "sequence": 5,
                    "ts_ms": 20.1,
                    "context_id": "ctx",
                    "failed_ts_ms": 10.1,
                    "suppressed_count": 267,
                },
                {
                    "event": "transfer_retry_guard_summary",
                    "run_id": "run-guard",
                    "sequence": 6,
                    "ts_ms": 30.0,
                    "blocked_attempt_count": 1,
                    "suppressed_retry_count": 267,
                    "released_attempt_count": 1,
                    "active_blocked_attempt_count": 0,
                    "unknown_circuit_open_count": 0,
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            metrics = validate_transfer_audit(path)["transfer_retry_guard"]

            self.assertTrue(metrics["available"])
            self.assertEqual(metrics["suppressed_retry_count"], 267)
            self.assertEqual(
                metrics["blocker_counts"],
                {"ancestor_closure": 1, "device_capacity": 1},
            )
            self.assertEqual(metrics["max_submissions_per_physical_fingerprint"], 1)
            self.assertEqual(
                metrics["max_failed_attempts_per_physical_fingerprint"], 1
            )
            self.assertTrue(metrics["same_snapshot_single_submit"])
            self.assertTrue(metrics["same_snapshot_single_failed_attempt"])
            self.assertEqual(metrics["retry_without_release_count"], 0)
            self.assertTrue(metrics["event_gated_retry_integrity"])
            self.assertAlmostEqual(metrics["retry_release_latency_p95_ms"], 10.0)

    def test_reports_physical_bundle_preview_to_ack_characterization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            records = [
                {
                    "event": "context_lease_issued",
                    "run_id": "run-bundle",
                    "sequence": 1,
                    "ts_ms": 1.0,
                    "context_id": "ctx-a",
                    "context_epoch": 0,
                    "lease_kind": "conditional_resume",
                },
                {
                    "event": "bundle_lease_aggregated",
                    "run_id": "run-bundle",
                    "sequence": 2,
                    "ts_ms": 1.0,
                    "bundle_id": "bundle-a",
                    "owner_context_ids": ["ctx-a", "ctx-b"],
                    "strongest_lease_kind": "conditional_resume",
                },
                {
                    "event": "physical_bundle_preview",
                    "run_id": "run-bundle",
                    "sequence": 3,
                    "ts_ms": 1.0,
                    "context_id": "ctx-a",
                    "context_epoch": 0,
                    "command_kind": "offload_context",
                    "bundle_id": "bundle-a",
                    "bundle_scope": "shared_subtree",
                    "generation_fingerprint": "generation-a",
                    "owner_context_ids": ["ctx-a", "ctx-b"],
                    "exclusive_action_bytes": 20,
                    "cross_context_action_bytes": 80,
                    "physical_unique_bytes": 120,
                    "gpu_bytes": 100,
                    "cpu_bytes": 20,
                    "closure_bytes": 100,
                    "locked_bytes": 20,
                    "lease_kind": "conditional_resume",
                    "eligible": True,
                    "blocker_codes": [],
                },
                {
                    "event": "transfer_dispatched",
                    "run_id": "run-bundle",
                    "sequence": 4,
                    "ts_ms": 2.0,
                    "command_id": "offload-a",
                    "kind": "offload_context",
                    "context_id": "ctx-a",
                    "context_epoch": 0,
                    "bundle_id": "bundle-a",
                    "bundle_scope": "shared_subtree",
                    "closure_fingerprint": "generation-a",
                    "selected_bytes": 100,
                    "expected_reclaimable_bytes": 100,
                    "action_counts": {"commit_cpu": 1},
                },
                {
                    "event": "transfer_acknowledged",
                    "run_id": "run-bundle",
                    "sequence": 5,
                    "ts_ms": 2.1,
                    "command_id": "offload-a",
                    "status": "partial",
                    "actual_bytes": 80,
                    "blocker_codes": ["node_locked"],
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            metrics = validate_transfer_audit(path)[
                "physical_bundle_characterization"
            ]

            self.assertTrue(metrics["available"])
            self.assertEqual(metrics["preview_count"], 1)
            self.assertEqual(metrics["shared_owner_preview_count"], 1)
            self.assertEqual(
                metrics["bundle_scope_snapshot_counts"], {"shared_subtree": 1}
            )
            self.assertEqual(metrics["exclusive_action_bytes_snapshot"], 20)
            self.assertEqual(metrics["cross_context_action_bytes_snapshot"], 80)
            self.assertEqual(
                metrics["bundle_dispatch_scope_counts"], {"shared_subtree": 1}
            )
            self.assertAlmostEqual(metrics["locked_gpu_snapshot_ratio"], 0.2)
            self.assertEqual(metrics["predictable_blocker_reject_count"], 1)
            self.assertEqual(metrics["expected_reclaimable_bytes"], 100)
            self.assertEqual(metrics["actual_reclaimed_bytes"], 80)
            self.assertAlmostEqual(metrics["reclaim_realization_ratio"], 0.8)
            self.assertEqual(metrics["absolute_reclaim_error_bytes"], 20)
            self.assertEqual(
                metrics["dispatch_without_matching_preview_count"], 0
            )


if __name__ == "__main__":
    unittest.main()
