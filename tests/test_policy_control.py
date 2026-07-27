import unittest

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.admission import AdmissionController, AdmissionRequest
from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.residency import ResidencyClass, ResidencyClassifier
from beliefkv.policy.shadow_controller import ShadowConfig, ShadowController, ShadowSignals
from beliefkv.policy.transfer_planner import ReactiveTransferPlanner, TransferPlannerConfig
from beliefkv.policy.workflow_fairness import WorkflowFairScheduler
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import CommandKind, PageHandle, PhysicalResidency


class Harness:
    def __init__(self) -> None:
        self.sequence = 0
        self.graph = RuntimeCausalContextGraph()
        self.index = PageOwnershipIndex()

    def emit(self, kind: RuntimeEventKind, workflow_id: str, **kwargs):
        self.sequence += 1
        result = self.graph.apply(
            RuntimeEvent(
                event_id=f"e{self.sequence}",
                ts_ms=float(self.sequence),
                kind=kind,
                workflow_id=workflow_id,
                **kwargs,
            )
        )
        return result

    def workflow(self, workflow_id: str) -> None:
        self.emit(RuntimeEventKind.WORKFLOW_START, workflow_id)

    def invocation(self, workflow_id: str, invocation_id: str, context_id: str) -> None:
        self.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            workflow_id,
            invocation_id=invocation_id,
            context_id=context_id,
            context_epoch=0,
        )
        self.index.register_context(context_id, workflow_id, 0)


class CausalPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness()
        self.h.workflow("wf")
        self.frontier = CausalFrontierScheduler(self.h.graph)
        self.classifier = ResidencyClassifier(self.h.graph, self.h.index)

    def test_blocking_child_precedes_unrelated_ready_agent(self):
        self.h.invocation("wf", "parent", "ctx-parent")
        self.h.invocation("wf", "child", "ctx-child")
        self.h.invocation("wf", "peer", "ctx-peer")
        self.h.emit(
            RuntimeEventKind.CALL,
            "wf",
            invocation_id="parent",
            target_invocation_id="child",
        )
        selected = self.frontier.select("wf")
        self.assertEqual(selected.invocation_id, "child")
        self.assertEqual(selected.causal_class, "blocking_chain")

    def test_last_join_member_has_straggler_priority(self):
        self.h.invocation("wf", "parent", "ctx-parent")
        self.h.invocation("wf", "a", "ctx-a")
        self.h.invocation("wf", "b", "ctx-b")
        self.h.invocation("wf", "peer", "ctx-peer")
        self.h.emit(
            RuntimeEventKind.JOIN_CREATE,
            "wf",
            join_id="j",
            member_invocation_ids=("a", "b"),
        )
        self.h.emit(
            RuntimeEventKind.JOIN_WAIT,
            "wf",
            invocation_id="parent",
            join_id="j",
        )
        self.h.emit(RuntimeEventKind.RETURN, "wf", invocation_id="a")
        selected = self.frontier.select("wf")
        self.assertEqual(selected.invocation_id, "b")
        self.assertEqual(selected.causal_class, "join_straggler")

    def test_shared_page_uses_strongest_owner_residency(self):
        self.h.invocation("wf", "parked", "ctx-parked")
        self.h.invocation("wf", "active", "ctx-active")
        self.h.emit(RuntimeEventKind.TOOL_START, "wf", invocation_id="parked")
        self.h.emit(RuntimeEventKind.LLM_SUBMIT, "wf", invocation_id="active")
        handle = PageHandle(1, 0)
        page = self.h.index.register_page(handle, size_bytes=100)
        self.h.index.bind_pages("ctx-parked", 0, [handle])
        self.h.index.bind_pages("ctx-active", 0, [handle])
        result = self.classifier.page(page, 10)
        self.assertEqual(result.residency_class, ResidencyClass.PINNED)

    def test_terminal_nonpersistent_context_releases_owner(self):
        self.h.invocation("wf", "once", "ctx")
        handle = PageHandle(1, 0)
        self.h.index.register_page(handle, size_bytes=100)
        self.h.index.bind_pages("ctx", 0, [handle])
        self.h.emit(RuntimeEventKind.RETURN, "wf", invocation_id="once")
        released = self.classifier.release_terminal_owners()
        self.assertEqual(released, {"ctx"})
        self.assertFalse(self.h.index.pages[handle].owner_contexts)

    def test_workflow_end_releases_persistent_context_owner(self):
        self.h.emit(
            RuntimeEventKind.INVOCATION_CREATE,
            "wf",
            invocation_id="persistent",
            context_id="ctx-persistent",
            context_epoch=0,
            attributes={"persistent": True},
        )
        self.h.index.register_context("ctx-persistent", "wf", 0)
        handle = PageHandle(1, 0)
        self.h.index.register_page(handle, size_bytes=100)
        self.h.index.bind_pages("ctx-persistent", 0, [handle])
        self.h.emit(RuntimeEventKind.RETURN, "wf", invocation_id="persistent")

        before_end = self.classifier.context("ctx-persistent", 10)
        self.assertEqual(before_end.residency_class, ResidencyClass.PARKED)
        self.assertEqual(before_end.reason, "persistent_inactive")

        self.h.emit(RuntimeEventKind.WORKFLOW_END, "wf")
        released = self.classifier.release_terminal_owners()

        self.assertEqual(released, {"ctx-persistent"})
        self.assertFalse(self.h.index.pages[handle].owner_contexts)
        after_end = self.classifier.context("ctx-persistent", 10)
        self.assertEqual(after_end.residency_class, ResidencyClass.DEAD_UNOWNED)
        self.assertEqual(after_end.reason, "workflow_ended")


class FairAdmissionTest(unittest.TestCase):
    def test_agent_fanout_does_not_change_root_workflow_fairness(self):
        scheduler = WorkflowFairScheduler()
        scheduler.register("fanout")
        scheduler.register("single")
        scheduler.charge_service("fanout", 10)
        selected = scheduler.select({"fanout", "single"})
        self.assertEqual(selected, "single")

    def test_admission_reserves_actual_capacity_before_next_request(self):
        h = Harness()
        h.workflow("a")
        h.workflow("b")
        h.invocation("a", "ia", "ca")
        h.invocation("b", "ib", "cb")
        fairness = WorkflowFairScheduler()
        controller = AdmissionController(
            h.index,
            fairness,
            CausalFrontierScheduler(h.graph),
            reserve_hbm_bytes=100,
        )
        for request_id, workflow_id, invocation_id, context_id in (
            ("ra", "a", "ia", "ca"),
            ("rb", "b", "ib", "cb"),
        ):
            controller.enqueue(
                AdmissionRequest(
                    request_id=request_id,
                    workflow_id=workflow_id,
                    invocation_id=invocation_id,
                    context_id=context_id,
                    context_epoch=0,
                    submitted_ts_ms=0,
                    uncached_prompt_tokens=4,
                    expected_output_tokens=1,
                    kv_bytes_per_token=100,
                )
            )
        first = controller.decide_next(1000)
        self.assertTrue(first.admitted)
        self.assertEqual(first.reserved_bytes, 500)
        second = controller.decide_next(1000)
        self.assertFalse(second.admitted)
        self.assertEqual(second.reason, "insufficient_actual_hbm")

    def test_idle_engine_can_borrow_reserve_but_not_hard_capacity(self):
        h = Harness()
        h.workflow("wf")
        h.invocation("wf", "inv", "ctx")
        fairness = WorkflowFairScheduler()
        controller = AdmissionController(
            h.index,
            fairness,
            CausalFrontierScheduler(h.graph),
            reserve_hbm_bytes=200,
        )
        controller.enqueue(
            AdmissionRequest(
                request_id="fits-hard-capacity",
                workflow_id="wf",
                invocation_id="inv",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=0,
                uncached_prompt_tokens=1,
                expected_output_tokens=0,
                kv_bytes_per_token=150,
            )
        )

        normal = controller.decide_next(1000, actual_hbm_used_bytes=850)
        self.assertFalse(normal.admitted)
        borrowed = controller.decide_next(
            1000,
            actual_hbm_used_bytes=850,
            allow_reserve_borrow=True,
        )
        self.assertTrue(borrowed.admitted)
        self.assertEqual(borrowed.reason, "engine_idle_reserve_borrow")
        self.assertEqual(borrowed.reserved_bytes, 150)

        blocked = AdmissionController(
            h.index,
            WorkflowFairScheduler(),
            CausalFrontierScheduler(h.graph),
            reserve_hbm_bytes=200,
        )
        blocked.enqueue(
            AdmissionRequest(
                request_id="exceeds-hard-capacity",
                workflow_id="wf",
                invocation_id="inv",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=0,
                uncached_prompt_tokens=1,
                expected_output_tokens=0,
                kv_bytes_per_token=151,
            )
        )
        decision = blocked.decide_next(
            1000,
            actual_hbm_used_bytes=850,
            allow_reserve_borrow=True,
        )
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "insufficient_actual_hbm")

    def test_liveness_target_is_not_bypassed_by_a_smaller_request(self):
        h = Harness()
        h.workflow("a")
        h.workflow("b")
        h.invocation("a", "ia", "ca")
        h.invocation("b", "ib", "cb")
        controller = AdmissionController(
            h.index,
            WorkflowFairScheduler(),
            CausalFrontierScheduler(h.graph),
            reserve_hbm_bytes=100,
        )
        for request in (
            AdmissionRequest("target", "a", "ia", "ca", 0, 0, 4, 0, 100),
            AdmissionRequest("small", "b", "ib", "cb", 0, 1, 1, 0, 100),
        ):
            controller.enqueue(request)

        blocked = controller.decide_next(
            1000,
            actual_hbm_used_bytes=700,
            preferred_request_id="target",
        )
        self.assertFalse(blocked.admitted)
        self.assertEqual(blocked.request_id, "target")

        admitted = controller.decide_next(
            1000,
            actual_hbm_used_bytes=500,
            preferred_request_id="target",
        )
        self.assertTrue(admitted.admitted)
        self.assertEqual(admitted.request_id, "target")
        self.assertEqual(admitted.reason, "admission_liveness_target")

    def test_idle_liveness_requires_proven_native_reclaim_capacity(self):
        h = Harness()
        h.workflow("wf")
        h.invocation("wf", "inv", "ctx")
        controller = AdmissionController(
            h.index,
            WorkflowFairScheduler(),
            CausalFrontierScheduler(h.graph),
            reserve_hbm_bytes=100,
        )
        controller.enqueue(
            AdmissionRequest(
                request_id="target",
                workflow_id="wf",
                invocation_id="inv",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=0,
                uncached_prompt_tokens=2,
                expected_output_tokens=1,
                kv_bytes_per_token=100,
                prompt_tokens=6,
            )
        )

        unproven = controller.decide_next(
            1000,
            actual_hbm_used_bytes=950,
            allow_reserve_borrow=True,
            preferred_request_id="target",
        )
        self.assertFalse(unproven.admitted)
        self.assertEqual(unproven.reason, "insufficient_actual_hbm")

        decision = controller.decide_next(
            1000,
            actual_hbm_used_bytes=950,
            allow_reserve_borrow=True,
            preferred_request_id="target",
            native_reclaim_capacity_bytes=301,
        )

        self.assertTrue(decision.admitted)
        self.assertEqual(decision.reason, "admission_liveness_native_reclaim")
        self.assertEqual(decision.reserved_bytes, 300)
        self.assertEqual(decision.native_reclaim_capacity_bytes, 301)

    def test_native_reclaim_rejects_equality_as_sglang_no_token(self):
        h = Harness()
        h.workflow("wf")
        h.invocation("wf", "inv", "ctx")
        controller = AdmissionController(
            h.index,
            WorkflowFairScheduler(),
            CausalFrontierScheduler(h.graph),
            reserve_hbm_bytes=100,
        )
        controller.enqueue(
            AdmissionRequest("target", "wf", "inv", "ctx", 0, 0, 2, 1, 100)
        )

        decision = controller.decide_next(
            1000,
            actual_hbm_used_bytes=950,
            preferred_request_id="target",
            native_reclaim_capacity_bytes=300,
        )

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "insufficient_native_reclaim_capacity")
        self.assertEqual(decision.required_bytes, 300)

    def test_native_reclaim_rejects_a_request_larger_than_the_kv_pool(self):
        h = Harness()
        h.workflow("wf")
        h.invocation("wf", "inv", "ctx")
        controller = AdmissionController(
            h.index,
            WorkflowFairScheduler(),
            CausalFrontierScheduler(h.graph),
            reserve_hbm_bytes=100,
        )
        controller.enqueue(
            AdmissionRequest(
                request_id="oversized",
                workflow_id="wf",
                invocation_id="inv",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=0,
                uncached_prompt_tokens=2,
                expected_output_tokens=1,
                kv_bytes_per_token=100,
                prompt_tokens=10,
            )
        )

        decision = controller.decide_next(
            1000,
            actual_hbm_used_bytes=950,
            allow_reserve_borrow=True,
            preferred_request_id="oversized",
            native_reclaim_capacity_bytes=1001,
        )

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "request_exceeds_hbm_capacity")


class TransferPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness()
        self.h.workflow("wf")
        self.h.invocation("wf", "parent", "ctx-parent")
        self.h.invocation("wf", "child", "ctx-child")
        self.h.emit(
            RuntimeEventKind.CALL,
            "wf",
            invocation_id="parent",
            target_invocation_id="child",
        )
        self.classifier = ResidencyClassifier(self.h.graph, self.h.index)
        self.frontier = CausalFrontierScheduler(self.h.graph)
        self.shadow = ShadowController(
            self.h.graph,
            self.h.index,
            self.classifier,
            self.frontier,
            ShadowConfig(
                min_parked_ms=0,
                chunk_bytes=100,
                min_chunk_bytes=1,
                host_reserve_bytes=0,
            ),
        )
        self.signals = ShadowSignals(
            urgent_queue_depth=0,
            pcie_utilization=0,
            gpu_compute_utilization=0,
            measured_inference_slowdown=0,
            hbm_pressure=0.5,
            host_free_bytes=1000,
        )

    def test_pressure_uses_reactive_offload(self):
        handle = PageHandle(1, 0)
        self.h.index.register_page(handle, size_bytes=800)
        self.h.index.bind_pages("ctx-parent", 0, [handle])
        planner = ReactiveTransferPlanner(
            self.h.graph,
            self.h.index,
            self.classifier,
            self.frontier,
            self.shadow,
            TransferPlannerConfig(reserve_hbm_bytes=500, urgent_chunk_bytes=1000),
        )
        command = planner.plan_next(
            now_ms=10,
            hbm_capacity_bytes=1000,
            signals=self.signals,
        )
        self.assertEqual(command.kind, CommandKind.OFFLOAD_CONTEXT)
        self.assertEqual(command.context_id, "ctx-parent")

    def test_idle_liveness_spills_another_workflow_frontier(self):
        h = Harness()
        for workflow_id in ("a", "b"):
            h.workflow(workflow_id)
            h.invocation(workflow_id, f"inv-{workflow_id}", f"ctx-{workflow_id}")
        for page_id, context_id in ((1, "ctx-a"), (2, "ctx-b")):
            handle = PageHandle(page_id, 0)
            h.index.register_page(handle, size_bytes=400)
            h.index.bind_pages(context_id, 0, [handle])
        classifier = ResidencyClassifier(h.graph, h.index)
        frontier = CausalFrontierScheduler(h.graph)
        shadow = ShadowController(
            h.graph,
            h.index,
            classifier,
            frontier,
            ShadowConfig(min_parked_ms=0, host_reserve_bytes=0),
        )
        planner = ReactiveTransferPlanner(
            h.graph,
            h.index,
            classifier,
            frontier,
            shadow,
            TransferPlannerConfig(reserve_hbm_bytes=100, urgent_chunk_bytes=1000),
        )

        command = planner.plan_next(
            now_ms=2000,
            hbm_capacity_bytes=1000,
            actual_hbm_used_bytes=950,
            admission_required_bytes=300,
            protected_context_id="ctx-a",
            allow_frontier_spill=True,
            signals=self.signals,
        )

        self.assertEqual(command.kind, CommandKind.OFFLOAD_CONTEXT)
        self.assertEqual(command.context_id, "ctx-b")
        self.assertTrue(command.metadata["allow_ready_owners"])
        self.assertEqual(command.metadata["protected_context_id"], "ctx-a")

    def test_idle_liveness_does_not_spill_cross_context_bundle(self):
        h = Harness()
        for workflow_id in ("target", "first", "second"):
            h.workflow(workflow_id)
            h.invocation(
                workflow_id,
                f"inv-{workflow_id}",
                f"ctx-{workflow_id}",
            )
        shared = PageHandle(1, 0)
        h.index.register_page(shared, size_bytes=400)
        h.index.bind_pages("ctx-first", 0, [shared])
        h.index.bind_pages("ctx-second", 0, [shared])
        classifier = ResidencyClassifier(h.graph, h.index)
        frontier = CausalFrontierScheduler(h.graph)
        shadow = ShadowController(
            h.graph,
            h.index,
            classifier,
            frontier,
            ShadowConfig(min_parked_ms=0, host_reserve_bytes=0),
        )
        planner = ReactiveTransferPlanner(
            h.graph,
            h.index,
            classifier,
            frontier,
            shadow,
            TransferPlannerConfig(
                reserve_hbm_bytes=100,
                urgent_chunk_bytes=1000,
                prefetch_enabled=False,
            ),
        )

        command = planner.plan_next(
            now_ms=2000,
            hbm_capacity_bytes=1000,
            actual_hbm_used_bytes=950,
            admission_required_bytes=300,
            protected_context_id="ctx-target",
            allow_frontier_spill=True,
            drop_unowned_enabled=False,
            signals=self.signals,
        )

        self.assertIsNone(command)

    def test_prefetch_can_be_disabled_without_changing_cpu_residency(self):
        handle = PageHandle(1, 0)
        self.h.index.register_page(
            handle,
            size_bytes=100,
            residency=PhysicalResidency.CPU_ONLY,
        )
        self.h.index.bind_pages("ctx-child", 0, [handle])
        planner = ReactiveTransferPlanner(
            self.h.graph,
            self.h.index,
            self.classifier,
            self.frontier,
            self.shadow,
            TransferPlannerConfig(
                reserve_hbm_bytes=100,
                prefetch_enabled=False,
            ),
        )

        command = planner.plan_next(
            now_ms=10,
            hbm_capacity_bytes=1000,
            signals=self.signals,
        )

        self.assertIsNone(command)
        self.assertEqual(
            self.h.index.pages[handle].residency,
            PhysicalResidency.CPU_ONLY,
        )

    def test_prefetch_capacity_excludes_outstanding_admission_reservations(self):
        handle = PageHandle(1, 0)
        self.h.index.register_page(
            handle,
            size_bytes=100,
            residency=PhysicalResidency.CPU_ONLY,
        )
        self.h.index.bind_pages("ctx-child", 0, [handle])
        planner = ReactiveTransferPlanner(
            self.h.graph,
            self.h.index,
            self.classifier,
            self.frontier,
            self.shadow,
            TransferPlannerConfig(
                reserve_hbm_bytes=100,
                prefetch_enabled=True,
                prefetch_chunk_bytes=1000,
            ),
        )

        blocked = planner.plan_next(
            now_ms=10,
            hbm_capacity_bytes=1000,
            actual_hbm_used_bytes=0,
            reserved_hbm_bytes=850,
            signals=self.signals,
        )
        eligible = planner.plan_next(
            now_ms=11,
            hbm_capacity_bytes=1000,
            actual_hbm_used_bytes=0,
            reserved_hbm_bytes=700,
            signals=self.signals,
        )

        self.assertIsNone(blocked)
        self.assertIsNotNone(eligible)
        self.assertEqual(eligible.kind, CommandKind.PREFETCH_CONTEXT)

    def test_prefetch_restores_preferred_context_as_largest_bundle(self):
        parent = PageHandle(1, 0)
        child = PageHandle(2, 0)
        self.h.index.register_page(
            parent,
            size_bytes=100,
            residency=PhysicalResidency.CPU_ONLY,
        )
        self.h.index.register_page(
            child,
            size_bytes=400,
            residency=PhysicalResidency.CPU_ONLY,
            parent=parent,
        )
        self.h.index.bind_pages("ctx-child", 0, (parent, child))
        self.h.invocation("wf", "other", "ctx-other")
        other = PageHandle(3, 0)
        self.h.index.register_page(
            other,
            size_bytes=50,
            residency=PhysicalResidency.CPU_ONLY,
        )
        self.h.index.bind_pages("ctx-other", 0, (other,))
        planner = ReactiveTransferPlanner(
            self.h.graph,
            self.h.index,
            self.classifier,
            self.frontier,
            self.shadow,
            TransferPlannerConfig(
                reserve_hbm_bytes=0,
                prefetch_enabled=True,
                prefetch_chunk_bytes=1000,
            ),
        )

        command = planner.plan_next(
            now_ms=10,
            hbm_capacity_bytes=2000,
            actual_hbm_used_bytes=0,
            signals=self.signals,
            preferred_restore_context_ids=("ctx-child",),
        )

        self.assertIsNotNone(command)
        self.assertEqual(command.context_id, "ctx-child")
        self.assertEqual(command.target_bytes, 500)
        self.assertEqual(
            {item.handle for item in command.physical_bundle.page_actions},
            {parent, child},
        )

    def test_idle_window_prepares_non_destructive_shadow(self):
        handle = PageHandle(1, 0)
        self.h.index.register_page(handle, size_bytes=100)
        self.h.index.bind_pages("ctx-parent", 0, [handle])
        command = self.shadow.plan(now_ms=10, signals=self.signals)
        self.assertEqual(command.kind, CommandKind.SHADOW_CONTEXT)
        self.assertTrue(command.metadata["non_destructive"])

    def test_shadow_is_disabled_when_urgent_queue_is_nonempty(self):
        handle = PageHandle(1, 0)
        self.h.index.register_page(handle, size_bytes=100)
        self.h.index.bind_pages("ctx-parent", 0, [handle])
        busy = ShadowSignals(
            urgent_queue_depth=1,
            pcie_utilization=0,
            gpu_compute_utilization=0,
            measured_inference_slowdown=0,
            hbm_pressure=0.5,
            host_free_bytes=1000,
        )
        self.assertIsNone(self.shadow.plan(now_ms=10, signals=busy))

    def test_interference_feedback_reduces_chunk_size(self):
        initial = self.shadow.config.chunk_bytes
        self.shadow.observe_interference(0.5)
        self.assertLess(self.shadow.config.chunk_bytes, initial)


if __name__ == "__main__":
    unittest.main()
