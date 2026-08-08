import unittest

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
        page_count=1 if status == CommandStatus.COMPLETED else 0,
    )


class TransferServiceCurveTest(unittest.TestCase):
    def test_insufficient_data_uses_conservative_static_fallback(self):
        static = PCIeCostModel(bandwidth_gbps=10.0, overhead_ms=0.1)
        curve = TransferServiceCurve(static, min_samples=3)

        estimate = curve.estimate(TransferDirection.H2D, 10_000_000)

        self.assertEqual(estimate.source, "static_fallback")
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

    def test_direction_callback_floor_covers_sparse_small_transfer_bucket(self):
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

        self.assertEqual(estimate.source, "direction")
        self.assertGreaterEqual(estimate.callback_floor_p90_ms, 34)
        self.assertGreaterEqual(
            estimate.estimated_callback_ms, estimate.callback_floor_p90_ms
        )

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


if __name__ == "__main__":
    unittest.main()
