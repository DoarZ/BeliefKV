import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from beliefkv.policy.service_curve import TransferServiceCurve
from beliefkv.policy.transfer_cost import PCIeCostModel
from beliefkv.runtime.protocol import (
    CommandStatus,
    TransferDirection,
    TransferTelemetry,
)


def telemetry(
    sequence: int,
    *,
    duration_ms: float,
    size_bytes: int = 1000,
    status: CommandStatus = CommandStatus.COMPLETED,
    compute_wait_ms: float | None = None,
    actual_bytes: int | None = None,
    page_count: int = 1,
) -> TransferTelemetry:
    return TransferTelemetry(
        command_id=f"command-{sequence}",
        submit_ts_ms=sequence * 100.0,
        start_ts_ms=sequence * 100.0 + 1.0,
        first_layer_ready_ts_ms=None,
        complete_ts_ms=sequence * 100.0 + 1.0 + duration_ms,
        compute_wait_ms=compute_wait_ms,
        actual_bytes=(
            actual_bytes
            if actual_bytes is not None
            else size_bytes if status == CommandStatus.COMPLETED else 0
        ),
        closure_bytes=size_bytes,
        merged_operation_count=0,
        direction=TransferDirection.D2H,
        source_tier="gpu",
        target_tier="host",
        status=status,
        page_count=page_count if status == CommandStatus.COMPLETED else 0,
    )


class TransferServiceCurveTest(unittest.TestCase):
    def test_byte_only_snapshot_view_erases_extent_bucket_difference(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=2)
        for sequence, page_count, duration in (
            (1, 7, 100.0),
            (2, 7, 110.0),
            (3, 106, 700.0),
            (4, 106, 720.0),
        ):
            curve.observe(
                replace(
                    telemetry(
                    sequence,
                    duration_ms=duration,
                    size_bytes=1 << 30,
                    page_count=page_count,
                    ),
                command_kind="offload_context",
                host_copy_state="missing",
                pinned_host=True,
                ),
            )

        snapshot = curve.snapshot()
        low = TransferServiceCurve.estimate_snapshot_byte_only(
            snapshot,
            TransferDirection.D2H,
            1 << 30,
            command_kind="offload_context",
            host_copy_state="missing",
            pinned_host=True,
        )
        high = TransferServiceCurve.estimate_snapshot_byte_only(
            snapshot,
            TransferDirection.D2H,
            1 << 30,
            command_kind="offload_context",
            host_copy_state="missing",
            pinned_host=True,
        )

        self.assertEqual(low.source, "byte_only_size_bucket")
        self.assertEqual(low.estimated_completion_p90_ms, high.estimated_completion_p90_ms)
        self.assertIsNone(low.extent_count_coverage)

    def test_insufficient_data_uses_conservative_static_fallback(self):
        static = PCIeCostModel(bandwidth_gbps=10.0, overhead_ms=0.1)
        curve = TransferServiceCurve(static, min_samples=3)

        estimate = curve.estimate(TransferDirection.H2D, 10_000_000)

        self.assertEqual(estimate.source, "shape_unsupported_static_fallback")
        self.assertGreater(estimate.estimated_callback_ms, static.transfer_ms(10_000_000))
        self.assertIsNone(estimate.estimated_unhidden_stall_ms)
        self.assertEqual(estimate.callback_floor_p90_ms, 0.0)

    def test_bucket_curve_uses_slow_tail_and_observed_compute_wait(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=8)
        observations = [
            telemetry(index, duration_ms=duration, compute_wait_ms=duration / 10)
            for index, duration in enumerate(range(10, 20), start=1)
        ]
        for observation in observations:
            curve.observe(observation)

        estimate = curve.estimate(TransferDirection.D2H, 1000)

        self.assertEqual(estimate.source, "bucket")
        self.assertEqual(estimate.sample_count, 10)
        self.assertGreaterEqual(estimate.estimated_callback_ms, 18.0)
        self.assertIsNotNone(estimate.estimated_unhidden_stall_ms)
        actual = [
            item.complete_ts_ms - item.submit_ts_ms for item in observations
        ]
        underestimation_rate = sum(
            value > estimate.estimated_callback_ms for value in actual
        ) / len(actual)
        self.assertLessEqual(underestimation_rate, 0.1)

    def test_rejections_are_counted_but_not_used_as_bandwidth_samples(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=2)
        curve.observe(telemetry(1, duration_ms=10))
        curve.observe(telemetry(2, duration_ms=11))
        curve.observe(
            telemetry(3, duration_ms=0, status=CommandStatus.REJECTED)
        )

        estimate = curve.estimate(TransferDirection.D2H, 1000)

        self.assertEqual(estimate.source, "bucket")
        self.assertAlmostEqual(estimate.rejection_probability, 1 / 3)

    def test_partial_physical_copy_does_not_pollute_service_samples(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=2)
        curve.observe(telemetry(1, duration_ms=10))
        curve.observe(telemetry(2, duration_ms=11))
        curve.observe(
            telemetry(
                3,
                duration_ms=1000,
                status=CommandStatus.PARTIAL,
                actual_bytes=900,
            )
        )

        estimate = curve.estimate(TransferDirection.D2H, 1000)

        self.assertEqual(estimate.source, "bucket")
        self.assertEqual(estimate.sample_count, 2)
        self.assertLess(estimate.estimated_callback_ms, 20)
        self.assertAlmostEqual(estimate.rejection_probability, 1 / 3)

    def test_reject_storm_does_not_evict_completed_service_samples(self):
        curve = TransferServiceCurve(PCIeCostModel(), window=8, min_samples=2)
        curve.observe(telemetry(1, duration_ms=10))
        curve.observe(telemetry(2, duration_ms=11))
        for sequence in range(3, 23):
            curve.observe(
                telemetry(
                    sequence,
                    duration_ms=1,
                    status=CommandStatus.REJECTED,
                    actual_bytes=0,
                )
            )

        estimate = curve.estimate(TransferDirection.D2H, 1000)

        self.assertEqual(estimate.source, "bucket")
        self.assertEqual(estimate.sample_count, 2)
        self.assertEqual(estimate.rejection_probability, 1.0)

    def test_shape_unsupported_fallback_does_not_mix_direction_samples(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=3)
        for sequence, duration in enumerate((30, 32, 34, 36), start=1):
            curve.observe(
                telemetry(
                    sequence,
                    duration_ms=duration,
                    size_bytes=1 << 20,
                )
            )

        estimate = curve.estimate(TransferDirection.D2H, 1000)

        self.assertEqual(estimate.source, "shape_unsupported_static_fallback")
        self.assertFalse(estimate.shape_supported)
        self.assertEqual(estimate.callback_floor_p90_ms, 0.0)

    def test_neighbor_bucket_does_not_extrapolate_tiny_transfers_to_gib(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=4)
        for sequence in range(1, 5):
            curve.observe(
                telemetry(
                    sequence,
                    duration_ms=180 + sequence,
                    size_bytes=1 << 28,
                )
            )

        estimate = curve.estimate(TransferDirection.D2H, 1 << 29)

        self.assertEqual(
            estimate.source,
            "bounded_neighboring_shape_extrapolation",
        )
        self.assertEqual(estimate.nearest_bucket_distance, 1)
        self.assertEqual(
            estimate.size_coverage_bytes,
            (1 << 28, (1 << 29) - 1),
        )
        self.assertGreater(estimate.estimated_callback_ms, 300)
        self.assertLess(estimate.estimated_callback_ms, 1_000)

    def test_same_bytes_preserve_low_and_high_fragment_shapes(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=3)
        size = 2_659_221_504
        for sequence, duration in enumerate((180.0, 190.0, 200.0), start=1):
            curve.observe(
                telemetry(
                    sequence,
                    duration_ms=duration,
                    size_bytes=size,
                    page_count=7,
                    compute_wait_ms=duration * 0.2,
                )
            )
        for sequence, duration in enumerate((740.0, 770.0, 800.0), start=10):
            curve.observe(
                telemetry(
                    sequence,
                    duration_ms=duration,
                    size_bytes=size,
                    page_count=106,
                    compute_wait_ms=duration * 0.7,
                )
            )

        low = curve.estimate(TransferDirection.D2H, size, page_count=7)
        high = curve.estimate(TransferDirection.D2H, size, page_count=106)
        middle = curve.estimate(TransferDirection.D2H, size, page_count=50)
        unsupported = curve.estimate(TransferDirection.D2H, size, page_count=1)

        self.assertTrue(low.shape_supported)
        self.assertTrue(high.shape_supported)
        self.assertGreater(high.estimated_completion_p90_ms, low.estimated_completion_p90_ms)
        self.assertEqual(middle.source, "bounded_neighboring_shape_extrapolation")
        self.assertEqual(middle.extent_count_coverage, (64, 127))
        self.assertGreater(
            middle.estimated_completion_p90_ms,
            low.estimated_completion_p90_ms,
        )
        self.assertFalse(unsupported.shape_supported)
        self.assertEqual(
            unsupported.source,
            "shape_unsupported_static_fallback",
        )

    def test_insufficient_exact_shape_does_not_mix_distant_extent_buckets(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=3)
        size = 2_659_221_504
        for sequence in (1, 2):
            curve.observe(
                telemetry(
                    sequence,
                    duration_ms=180.0,
                    size_bytes=size,
                    page_count=7,
                )
            )
        for sequence in (3, 4):
            curve.observe(
                telemetry(
                    sequence,
                    duration_ms=780.0,
                    size_bytes=size,
                    page_count=106,
                )
            )

        estimate = curve.estimate(TransferDirection.D2H, size, page_count=7)

        self.assertFalse(estimate.shape_supported)
        self.assertEqual(estimate.sample_count, 0)

    def test_detailed_curve_conditions_on_page_count_and_fixed_overhead(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=2)
        for sequence in (1, 2):
            observation = telemetry(sequence, duration_ms=10)
            observation = TransferTelemetry(
                **{
                    **observation.__dict__,
                    "page_count": 16,
                    "command_kind": "offload_context",
                    "host_copy_state": "missing",
                    "pinned_host": True,
                    "native_concurrent_bytes": 1 << 20,
                    "allocator_wait_ms": 2.0,
                    "callback_overhead_ms": 1.0,
                }
            )
            curve.observe(observation)

        estimate = curve.estimate(
            TransferDirection.D2H,
            1000,
            page_count=16,
            command_kind="offload_context",
            host_copy_state="missing",
            pinned_host=True,
            native_concurrent_bytes=1 << 20,
        )

        self.assertEqual(estimate.source, "bucket")
        self.assertEqual(estimate.fixed_overhead_p90_ms, 3.0)

    def test_missing_native_start_uses_submit_to_complete_service(self):
        curve = TransferServiceCurve(PCIeCostModel(), min_samples=2)
        for sequence in (1, 2):
            observation = telemetry(sequence, duration_ms=250, size_bytes=310_000_000)
            curve.observe(
                TransferTelemetry(
                    **{
                        **observation.__dict__,
                        "start_ts_ms": None,
                        "complete_ts_ms": observation.submit_ts_ms + 250,
                        "start_timestamp_semantics": "unavailable",
                    }
                )
            )

        estimate = curve.estimate(TransferDirection.D2H, 310_000_000)

        self.assertEqual(estimate.source, "bucket")
        self.assertGreaterEqual(estimate.estimated_callback_ms, 250)
        self.assertLess(estimate.effective_bytes_per_ms_p10, 2_000_000)

    def test_persistent_artifact_warm_starts_without_static_bandwidth(self):
        source = TransferServiceCurve(PCIeCostModel(), min_samples=2)
        for sequence in (1, 2, 3):
            source.observe(
                telemetry(
                    sequence,
                    duration_ms=300 + sequence,
                    size_bytes=310_000_000,
                )
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transfer-service.json"
            source.save_artifact(
                path,
                hardware_key="gpu:model:runtime",
                schema_version=2,
                metadata={"evidence_scope": "single_gpu_development"},
            )
            restored = TransferServiceCurve(PCIeCostModel(), min_samples=2)

            loaded = restored.warm_start(
                path,
                expected_hardware_key="gpu:model:runtime",
            )
            estimate = restored.estimate(
                TransferDirection.D2H,
                310_000_000,
            )

        self.assertEqual(loaded, 3)
        self.assertEqual(estimate.source, "bucket")
        self.assertGreater(estimate.estimated_callback_ms, 250)
        self.assertEqual(
            restored.warm_start_metadata["evidence_scope"],
            "single_gpu_development",
        )

    def test_calibrated_artifact_keeps_its_sample_gate_online(self):
        source = TransferServiceCurve(PCIeCostModel(), min_samples=3)
        for sequence in (1, 2, 3):
            source.observe(
                telemetry(
                    sequence,
                    duration_ms=300 + sequence,
                    size_bytes=310_000_000,
                    page_count=22,
                )
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transfer-service.json"
            source.save_artifact(path, hardware_key="gpu:model:runtime")
            restored = TransferServiceCurve(PCIeCostModel(), min_samples=8)
            restored.warm_start(path)

            live = restored.estimate(
                TransferDirection.D2H,
                310_000_000,
                page_count=22,
            )
            snapshot = restored.snapshot()
            immutable = TransferServiceCurve.estimate_snapshot(
                snapshot,
                TransferDirection.D2H,
                310_000_000,
                page_count=22,
            )
            byte_only = TransferServiceCurve.estimate_snapshot_byte_only(
                snapshot,
                TransferDirection.D2H,
                310_000_000,
            )
            contract = restored.validate_warm_start_contract()

        self.assertEqual(restored.min_samples, 8)
        self.assertEqual(restored.warm_start_min_samples, 3)
        self.assertEqual(snapshot["warm_start_min_samples"], 3)
        self.assertTrue(contract["contract_satisfied"])
        self.assertEqual(contract["runtime_min_samples"], 8)
        self.assertEqual(contract["artifact_min_samples"], 3)
        self.assertGreater(contract["supported_representative_count"], 0)
        self.assertTrue(live.shape_supported)
        self.assertTrue(immutable.shape_supported)
        self.assertEqual(byte_only.source, "byte_only_size_bucket")

        online_only = TransferServiceCurve(PCIeCostModel(), min_samples=8)
        for sequence in (1, 2, 3):
            online_only.observe(
                telemetry(
                    sequence,
                    duration_ms=300 + sequence,
                    size_bytes=310_000_000,
                    page_count=22,
                )
            )
        self.assertFalse(
            online_only.estimate(
                TransferDirection.D2H,
                310_000_000,
                page_count=22,
            ).shape_supported
        )

    def test_warm_start_contract_rejects_unusable_artifact(self):
        source = TransferServiceCurve(PCIeCostModel(), min_samples=3)
        for sequence in (1, 2):
            source.observe(
                telemetry(
                    sequence,
                    duration_ms=300 + sequence,
                    size_bytes=310_000_000,
                    page_count=22,
                )
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transfer-service.json"
            source.save_artifact(path, hardware_key="gpu:model:runtime")
            restored = TransferServiceCurve(PCIeCostModel(), min_samples=8)
            restored.warm_start(path)

            with self.assertRaisesRegex(ValueError, "no supported query"):
                restored.validate_warm_start_contract()


if __name__ == "__main__":
    unittest.main()
