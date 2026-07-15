import unittest

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.admission import AdmissionRequest
from beliefkv.predictor.types import RemainingTimePrediction
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandStatus,
    PageHandle,
    PhysicalResidency,
)


class ControllerHarness:
    def __init__(
        self,
        *,
        reserve_hbm_bytes: int = 100,
        predictor_enabled: bool = False,
        **config_overrides,
    ) -> None:
        self.controller = BeliefKVController(
            BeliefKVConfig(
                hbm_capacity_bytes=1000,
                host_capacity_bytes=10000,
                reserve_hbm_bytes=reserve_hbm_bytes,
                urgent_chunk_bytes=1000,
                shadow_chunk_bytes=500,
                shadow_min_parked_ms=0,
                predictor_enabled=predictor_enabled,
                **config_overrides,
            )
        )
        self.sequence = 0

    def emit(self, kind: RuntimeEventKind, ts_ms: float | None = None, **kwargs):
        self.sequence += 1
        return self.controller.process_runtime_event(
            RuntimeEvent(
                event_id=f"e{self.sequence}",
                ts_ms=float(self.sequence if ts_ms is None else ts_ms),
                kind=kind,
                workflow_id="wf",
                **kwargs,
            )
        )

    def create_parked(self) -> None:
        self.emit(RuntimeEventKind.WORKFLOW_START)
        self.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="parent",
            context_id="ctx",
            context_epoch=0,
        )
        self.emit(
            RuntimeEventKind.TOOL_START,
            invocation_id="parent",
            attributes={"tool_family": "search"},
        )

    def add_page(self, size_bytes: int) -> PageHandle:
        handle = PageHandle(1, 0)
        self.controller.page_index.register_page(handle, size_bytes=size_bytes)
        self.controller.page_index.bind_pages("ctx", 0, [handle])
        return handle


class ControllerTest(unittest.TestCase):
    def test_shadow_prepare_keeps_gpu_until_reactive_commit(self):
        h = ControllerHarness()
        h.create_parked()
        handle = h.add_page(400)
        tick = h.controller.tick(10)
        self.assertEqual(tick.transfer.command.kind, CommandKind.SHADOW_CONTEXT)
        command_id = tick.transfer.command.command_id
        h.controller.mark_command_started(command_id, [handle])
        self.assertEqual(
            h.controller.page_index.pages[handle].residency,
            PhysicalResidency.MIRRORING,
        )
        h.controller.acknowledge_command(
            CommandAck(
                command_id=command_id,
                status=CommandStatus.COMPLETED,
                completed_ts_ms=12,
                actual_bytes=400,
                page_handles=(handle,),
            )
        )
        self.assertEqual(
            h.controller.page_index.pages[handle].residency,
            PhysicalResidency.DUAL_CLEAN,
        )
        self.assertEqual(h.controller.page_index.gpu_bytes, 400)

    def test_wakeup_cancels_inflight_shadow_without_resume_stall(self):
        h = ControllerHarness()
        h.create_parked()
        handle = h.add_page(400)
        tick = h.controller.tick(10)
        command_id = tick.transfer.command.command_id
        h.controller.mark_command_started(command_id, [handle])
        h.emit(RuntimeEventKind.TOOL_END, ts_ms=11, invocation_id="parent")
        cancellation = h.controller.tick(11)
        self.assertIn(command_id, cancellation.cancel_command_ids)
        h.controller.acknowledge_command(
            CommandAck(
                command_id=command_id,
                status=CommandStatus.CANCELLED,
                completed_ts_ms=11.5,
                actual_bytes=0,
                reason="wakeup",
            )
        )
        self.assertEqual(
            h.controller.page_index.pages[handle].residency,
            PhysicalResidency.GPU_ONLY,
        )

    def test_admission_waits_for_offload_ack(self):
        h = ControllerHarness(reserve_hbm_bytes=300)
        h.create_parked()
        handle = h.add_page(800)
        h.controller.submit_request(
            AdmissionRequest(
                request_id="request",
                workflow_id="wf",
                invocation_id="parent",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=4,
                uncached_prompt_tokens=3,
                expected_output_tokens=1,
                kv_bytes_per_token=100,
            )
        )
        first = h.controller.tick(10)
        self.assertFalse(first.admission.admitted)
        self.assertEqual(first.transfer.command.kind, CommandKind.OFFLOAD_CONTEXT)
        command_id = first.transfer.command.command_id
        h.controller.mark_command_started(command_id, [handle])
        before_ack = h.controller.tick(11)
        self.assertFalse(before_ack.admission.admitted)
        h.controller.acknowledge_command(
            CommandAck(
                command_id=command_id,
                status=CommandStatus.COMPLETED,
                completed_ts_ms=12,
                actual_bytes=800,
                page_handles=(handle,),
            )
        )
        after_ack = h.controller.tick(12)
        self.assertTrue(after_ack.admission.admitted)
        self.assertEqual(after_ack.admission.reserved_bytes, 400)
        self.assertEqual(
            h.controller.page_index.pages[handle].residency,
            PhysicalResidency.CPU_ONLY,
        )

    def test_stalled_transfer_allows_native_reclaim_without_fake_ack(self):
        h = ControllerHarness(
            reserve_hbm_bytes=300,
            admission_liveness_timeout_ms=1,
            admission_force_progress_timeout_ms=5,
            transfer_watchdog_factor=1,
            transfer_watchdog_floor_ms=1,
        )
        h.create_parked()
        handle = h.add_page(800)
        h.controller.report_engine_activity(0)
        h.controller.submit_request(
            AdmissionRequest(
                request_id="request",
                workflow_id="wf",
                invocation_id="parent",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=4,
                uncached_prompt_tokens=3,
                expected_output_tokens=1,
                kv_bytes_per_token=100,
            )
        )

        first = h.controller.tick(4)
        self.assertFalse(first.admission.admitted)
        command_id = first.transfer.command.command_id
        h.controller.mark_command_started(command_id, [handle])

        recovered = h.controller.tick(10)
        self.assertEqual(recovered.stalled_command_ids, (command_id,))
        self.assertTrue(recovered.admission.admitted)
        self.assertEqual(
            recovered.admission.reason, "admission_liveness_native_reclaim"
        )
        self.assertEqual(h.controller.inflight_command_ids, (command_id,))
        self.assertEqual(
            h.controller.page_index.pages[handle].residency,
            PhysicalResidency.MIRRORING,
        )

    def test_ack_cannot_complete_unstarted_transfer(self):
        h = ControllerHarness()
        h.create_parked()
        handle = h.add_page(400)
        tick = h.controller.tick(10)
        with self.assertRaises(ValueError):
            h.controller.acknowledge_command(
                CommandAck(
                    command_id=tick.transfer.command.command_id,
                    status=CommandStatus.COMPLETED,
                    completed_ts_ms=11,
                    actual_bytes=400,
                    page_handles=(handle,),
                )
            )

    def test_reported_allocator_usage_controls_admission_and_pressure(self):
        h = ControllerHarness(reserve_hbm_bytes=100)
        h.create_parked()
        h.add_page(400)
        h.controller.report_hbm_usage(950, workflow_charges={"wf": 950.0})
        h.controller.submit_request(
            AdmissionRequest(
                request_id="request",
                workflow_id="wf",
                invocation_id="parent",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=4,
                uncached_prompt_tokens=1,
                expected_output_tokens=0,
                kv_bytes_per_token=100,
            )
        )
        tick = h.controller.tick(10)
        self.assertFalse(tick.admission.admitted)
        self.assertEqual(tick.transfer.command.kind, CommandKind.OFFLOAD_CONTEXT)
        self.assertEqual(h.controller.actual_hbm_used_bytes, 950)

    def test_idle_engine_progress_borrows_reserve_for_one_request(self):
        h = ControllerHarness(reserve_hbm_bytes=300)
        h.create_parked()
        h.controller.report_hbm_usage(750)
        h.controller.submit_request(
            AdmissionRequest(
                request_id="request",
                workflow_id="wf",
                invocation_id="parent",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=4,
                uncached_prompt_tokens=1,
                expected_output_tokens=0,
                kv_bytes_per_token=100,
            )
        )

        h.controller.report_engine_activity(1)
        blocked = h.controller.tick(10)
        self.assertFalse(blocked.admission.admitted)

        h.controller.report_engine_activity(0)
        admitted = h.controller.tick(11)
        self.assertTrue(admitted.admission.admitted)
        self.assertEqual(admitted.admission.reason, "engine_idle_reserve_borrow")
        self.assertEqual(admitted.admission.reserved_bytes, 100)

    def test_engine_activity_count_rejects_negative_values(self):
        h = ControllerHarness()
        with self.assertRaises(ValueError):
            h.controller.report_engine_activity(-1)

    def test_wakeup_outcome_updates_online_interval_calibration(self):
        h = ControllerHarness(predictor_enabled=True)
        h.create_parked()
        h.controller._last_predictions["ctx"] = RemainingTimePrediction(
            context_id="ctx",
            generated_ts_ms=5,
            p50_ms=10,
            p90_ms=20,
            p95_ms=30,
            confidence=0.8,
            ood_score=0.1,
        )
        h.emit(RuntimeEventKind.TOOL_END, ts_ms=15, invocation_id="parent")
        self.assertEqual(h.controller.predictor.calibrator.observation_count, 1)


if __name__ == "__main__":
    unittest.main()
