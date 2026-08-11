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
    ControlCommand,
    EnqueueStatus,
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
    def test_predictive_overlay_requires_explicit_online_dependencies(self):
        with self.assertRaisesRegex(
            ValueError, "requires online observed JointPlan"
        ):
            BeliefKVConfig(predictive_joint_overlay_enabled=True)
        with self.assertRaisesRegex(ValueError, "requires predictive JointPlan"):
            BeliefKVConfig(predictive_prefetch_canary_enabled=True)

        config = BeliefKVConfig(
            joint_policy_enabled=True,
            predictor_model_path="frontier.json",
            gpu_service_model_path="service.json",
            predictive_risk_shadow_enabled=True,
            predictive_joint_overlay_enabled=True,
            predictive_prefetch_canary_enabled=True,
        )
        self.assertTrue(config.predictive_prepare_host_enabled)
        self.assertEqual(config.predictive_prefetch_canary_max_inflight, 1)
        self.assertEqual(config.predictive_prefetch_canary_max_hbm_ratio, 0.05)

        unified = BeliefKVConfig(
            joint_policy_enabled=True,
            predictor_model_path="frontier.json",
            gpu_service_model_path="service.json",
            predictive_risk_shadow_enabled=True,
            predictive_joint_overlay_enabled=True,
            predictive_prepare_host_canary_limit=1,
        )
        self.assertEqual(unified.predictive_prepare_host_canary_limit, 1)

    def test_frontier_retraction_requires_predictor_and_p5_retraction(self):
        with self.assertRaisesRegex(ValueError, "frontier-aware retraction"):
            BeliefKVConfig(frontier_aware_retraction_shadow_enabled=True)

        config = BeliefKVConfig(
            joint_policy_enabled=True,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            predictor_model_path="frontier.json",
            gpu_service_model_path="service.json",
            predictive_risk_shadow_enabled=True,
            frontier_aware_retraction_shadow_enabled=True,
            frontier_aware_retraction_canary_limit=1,
        )
        self.assertTrue(config.frontier_aware_retraction_shadow_enabled)
        self.assertEqual(config.frontier_aware_retraction_canary_limit, 1)

    def test_typed_enqueue_adopts_only_equivalent_canonical_command(self):
        h = ControllerHarness()
        first = ControlCommand(
            command_id="restore-1",
            kind=CommandKind.PREFETCH_CONTEXT,
            created_ts_ms=1.0,
            context_id="ctx",
            context_epoch=0,
            target_bytes=100,
        )
        adopted = ControlCommand(
            command_id="restore-2",
            kind=CommandKind.PREFETCH_CONTEXT,
            created_ts_ms=2.0,
            context_id="ctx",
            context_epoch=0,
            target_bytes=100,
        )
        conflict = ControlCommand(
            command_id="offload-1",
            kind=CommandKind.OFFLOAD_CONTEXT,
            created_ts_ms=3.0,
            context_id="ctx",
            context_epoch=0,
            target_bytes=100,
        )

        first_outcome = h.controller.enqueue_control_command(first)
        adopted_outcome = h.controller.enqueue_control_command(adopted)
        conflict_outcome = h.controller.enqueue_control_command(conflict)

        self.assertEqual(first_outcome.status, EnqueueStatus.ENQUEUED)
        self.assertEqual(adopted_outcome.status, EnqueueStatus.ADOPT_EXISTING)
        self.assertEqual(adopted_outcome.canonical_command_id, "restore-1")
        self.assertEqual(conflict_outcome.status, EnqueueStatus.CONTEXT_CONFLICT)
        self.assertEqual(len(h.controller.command_queue), 1)

    def test_context_epoch_advance_cannot_adopt_stale_command(self):
        h = ControllerHarness()
        old = ControlCommand(
            command_id="restore-old",
            kind=CommandKind.PREFETCH_CONTEXT,
            created_ts_ms=1.0,
            context_id="ctx",
            context_epoch=0,
            target_bytes=100,
        )
        new = ControlCommand(
            command_id="restore-new",
            kind=CommandKind.PREFETCH_CONTEXT,
            created_ts_ms=2.0,
            context_id="ctx",
            context_epoch=1,
            target_bytes=100,
        )

        self.assertEqual(
            h.controller.enqueue_control_command(old).status,
            EnqueueStatus.ENQUEUED,
        )
        outcome = h.controller.enqueue_control_command(new)

        self.assertEqual(outcome.status, EnqueueStatus.CONTEXT_CONFLICT)
        self.assertEqual(outcome.canonical_command_id, "restore-old")

    def test_online_writer_can_disable_reactive_victim_selection(self):
        h = ControllerHarness()
        h.create_parked()
        h.add_page(400)

        tick = h.controller.tick(10, allow_reactive_transfer=False)

        self.assertIsNone(tick.transfer)
        self.assertEqual(h.controller.command_history, [])

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

    def test_terminal_child_drops_only_its_private_host_extent(self):
        h = ControllerHarness(shadow_enabled=False, prefetch_enabled=False)
        h.emit(RuntimeEventKind.WORKFLOW_START)
        h.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="parent",
            context_id="ctx-parent",
            context_epoch=0,
        )
        h.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="child",
            context_id="ctx-child",
            context_epoch=0,
            parent_invocation_id="parent",
        )
        prefix = PageHandle(1, 0)
        private = PageHandle(2, 0)
        h.controller.page_index.register_page(
            prefix, size_bytes=100, residency=PhysicalResidency.DUAL_CLEAN
        )
        h.controller.page_index.register_page(
            private,
            size_bytes=200,
            residency=PhysicalResidency.CPU_ONLY,
            parent=prefix,
        )
        h.controller.page_index.bind_pages("ctx-parent", 0, (prefix,))
        h.controller.page_index.bind_pages("ctx-child", 0, (prefix, private))

        h.emit(RuntimeEventKind.RETURN, invocation_id="child")
        tick = h.controller.tick(10)

        self.assertEqual(
            tick.transfer.command.kind, CommandKind.DROP_TERMINAL_PRIVATE
        )
        self.assertEqual(
            tuple(item.handle for item in tick.transfer.page_actions), (private,)
        )
        self.assertEqual(
            h.controller.page_index.pages[prefix].owner_contexts,
            {"ctx-parent": 0},
        )
        command_id = tick.transfer.command.command_id
        h.controller.mark_command_started(command_id, (private,))
        h.controller.acknowledge_command(
            CommandAck(
                command_id=command_id,
                status=CommandStatus.COMPLETED,
                completed_ts_ms=11,
                actual_bytes=200,
                page_handles=(private,),
            )
        )

        self.assertEqual(
            h.controller.page_index.pages[private].residency,
            PhysicalResidency.DEAD,
        )
        self.assertEqual(
            h.controller.page_index.pages[prefix].residency,
            PhysicalResidency.DUAL_CLEAN,
        )

    def test_terminal_dual_clean_private_extent_releases_host_copy_only(self):
        h = ControllerHarness(shadow_enabled=False, prefetch_enabled=False)
        h.emit(RuntimeEventKind.WORKFLOW_START)
        h.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="child",
            context_id="ctx-child",
            context_epoch=0,
        )
        handle = PageHandle(1, 0)
        h.controller.page_index.register_page(
            handle, size_bytes=200, residency=PhysicalResidency.DUAL_CLEAN
        )
        h.controller.page_index.bind_pages("ctx-child", 0, (handle,))

        h.emit(RuntimeEventKind.RETURN, invocation_id="child")
        tick = h.controller.tick(10)
        command_id = tick.transfer.command.command_id
        h.controller.mark_command_started(command_id, (handle,))
        h.controller.acknowledge_command(
            CommandAck(
                command_id=command_id,
                status=CommandStatus.COMPLETED,
                completed_ts_ms=11,
                actual_bytes=200,
                page_handles=(handle,),
            )
        )

        self.assertEqual(
            h.controller.page_index.pages[handle].residency,
            PhysicalResidency.GPU_ONLY,
        )

    def test_terminal_cleanup_waits_for_lock_state_change_without_retry_storm(self):
        h = ControllerHarness(shadow_enabled=False, prefetch_enabled=False)
        h.emit(RuntimeEventKind.WORKFLOW_START)
        h.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="child",
            context_id="ctx-child",
            context_epoch=0,
        )
        handle = PageHandle(1, 0)
        h.controller.page_index.register_page(
            handle, size_bytes=200, residency=PhysicalResidency.CPU_ONLY
        )
        h.controller.page_index.bind_pages("ctx-child", 0, (handle,))
        h.controller.page_index.set_engine_lock(handle, 1)
        h.emit(RuntimeEventKind.RETURN, invocation_id="child")

        rejected = h.controller.tick(10)
        self.assertIsNone(rejected.transfer)
        self.assertEqual(len(rejected.local_acks), 1)
        suppressed = h.controller.tick(11)
        self.assertIsNone(suppressed.transfer)
        self.assertEqual(suppressed.local_acks, ())

        h.controller.page_index.set_engine_lock(handle, 0)
        resumed = h.controller.tick(12)
        self.assertEqual(
            resumed.transfer.command.kind, CommandKind.DROP_TERMINAL_PRIVATE
        )

    def test_terminal_cleanup_captures_late_cache_finish_rebind(self):
        h = ControllerHarness(shadow_enabled=False, prefetch_enabled=False)
        h.emit(RuntimeEventKind.WORKFLOW_START)
        h.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="child",
            context_id="ctx-child",
            context_epoch=0,
        )
        h.emit(RuntimeEventKind.RETURN, invocation_id="child")

        handle = PageHandle(1, 0)
        h.controller.page_index.register_page(
            handle, size_bytes=200, residency=PhysicalResidency.CPU_ONLY
        )
        h.controller.page_index.bind_pages("ctx-child", 0, (handle,))

        tick = h.controller.tick(10)

        self.assertEqual(
            tick.transfer.command.kind, CommandKind.DROP_TERMINAL_PRIVATE
        )
        self.assertEqual(tick.transfer.command.target_handles, (handle,))

    def test_context_compaction_releases_old_epoch_ownership(self):
        h = ControllerHarness(shadow_enabled=False, prefetch_enabled=False)
        h.emit(RuntimeEventKind.WORKFLOW_START)
        h.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="parent",
            context_id="ctx",
            context_epoch=0,
        )
        handle = PageHandle(1, 0)
        h.controller.page_index.register_page(
            handle,
            size_bytes=200,
            residency=PhysicalResidency.DUAL_CLEAN,
        )
        h.controller.page_index.bind_pages("ctx", 0, (handle,))

        h.emit(
            RuntimeEventKind.CONTEXT_COMPACT,
            invocation_id="parent",
            context_id="ctx",
            context_epoch=1,
            attributes={"previous_context_epoch": 0},
        )

        self.assertEqual(h.controller.page_index.context_epoch("ctx"), 1)
        self.assertEqual(h.controller.page_index.context_pages("ctx"), [])
        self.assertEqual(h.controller.page_index.pages[handle].owner_contexts, {})
        cleanup = h.controller.tick(10)
        self.assertEqual(
            cleanup.transfer.command.kind,
            CommandKind.DROP_TERMINAL_PRIVATE,
        )
        self.assertEqual(cleanup.transfer.command.target_handles, (handle,))

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

    def test_stalled_transfer_cannot_authorize_native_reclaim(self):
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
        h.controller.report_native_admission_capacity("request", 500)

        blocked = h.controller.tick(10)
        self.assertEqual(blocked.stalled_command_ids, (command_id,))
        self.assertFalse(blocked.admission.admitted)
        self.assertEqual(blocked.admission.reason, "insufficient_actual_hbm")
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
        with self.assertRaises(ValueError):
            h.controller.report_engine_activity(1, running_request_count=2)

    def test_admitted_waiter_drives_frontier_spill_until_engine_can_run_it(self):
        h = ControllerHarness(
            reserve_hbm_bytes=100,
            admission_liveness_timeout_ms=1,
            admission_force_progress_timeout_ms=5,
        )
        for workflow_id in ("target-wf", "victim-wf"):
            h.controller.process_runtime_event(
                RuntimeEvent(
                    event_id=f"start-{workflow_id}",
                    ts_ms=1,
                    kind=RuntimeEventKind.WORKFLOW_START,
                    workflow_id=workflow_id,
                )
            )
            h.controller.process_runtime_event(
                RuntimeEvent(
                    event_id=f"inv-{workflow_id}",
                    ts_ms=2,
                    kind=RuntimeEventKind.INVOCATION_CREATE,
                    workflow_id=workflow_id,
                    invocation_id=f"inv-{workflow_id}",
                    context_id=f"ctx-{workflow_id}",
                    context_epoch=0,
                )
            )
        for page_id, context_id in (
            (1, "ctx-target-wf"),
            (2, "ctx-victim-wf"),
        ):
            handle = PageHandle(page_id, 0)
            h.controller.page_index.register_page(handle, size_bytes=400)
            h.controller.page_index.bind_pages(context_id, 0, [handle])
        h.controller.report_hbm_usage(950)
        h.controller.report_engine_activity(0, running_request_count=0)
        h.controller.submit_request(
            AdmissionRequest(
                request_id="target-request",
                workflow_id="target-wf",
                invocation_id="inv-target-wf",
                context_id="ctx-target-wf",
                context_epoch=0,
                submitted_ts_ms=3,
                uncached_prompt_tokens=2,
                expected_output_tokens=1,
                kv_bytes_per_token=100,
            )
        )
        h.controller.report_native_admission_capacity("target-request", 301)

        admitted = h.controller.tick(10)
        self.assertTrue(admitted.admission.admitted)
        self.assertEqual(
            admitted.admission.reason, "admission_liveness_native_reclaim"
        )
        self.assertIsNone(admitted.transfer)
        h.controller.report_engine_activity(1, running_request_count=0)
        reclaim = h.controller.tick(11)

        self.assertEqual(reclaim.transfer.command.kind, CommandKind.OFFLOAD_CONTEXT)
        self.assertEqual(reclaim.transfer.command.context_id, "ctx-victim-wf")
        self.assertEqual(
            reclaim.transfer.command.metadata["reason"],
            "admission_liveness_frontier_spill",
        )

    def test_unexecutable_drop_unowned_is_not_retried_until_state_changes(self):
        h = ControllerHarness(reserve_hbm_bytes=300)
        h.create_parked()
        h.add_page(800)
        locked = PageHandle(2, 0)
        h.controller.page_index.register_page(locked, size_bytes=200)
        h.controller.page_index.set_engine_lock(locked, 1)

        rejected = h.controller.tick(10)
        self.assertEqual(rejected.local_acks[0].reason, "no_unowned_pages")
        self.assertEqual(len(h.controller.command_history), 1)

        fallback = h.controller.tick(11)
        self.assertEqual(fallback.transfer.command.kind, CommandKind.OFFLOAD_CONTEXT)
        self.assertEqual(len(h.controller.command_history), 2)

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
