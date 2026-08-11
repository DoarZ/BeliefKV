import ast
import unittest
from collections import Counter
from pathlib import Path
from types import MethodType
from types import SimpleNamespace

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.retraction import (
    ObservedRetractionConfig,
    ObservedRetractionDecision,
    ObservedRetractionPlanner,
    ObservedRetractionSnapshot,
    RetractionLockedExtent,
    RetractionReplacement,
    RunningRetractionCandidate,
    RunningRetractionPlan,
)
from beliefkv.policy.joint_scheduler import JointPlannerMode
from beliefkv.policy.online_joint import JointPlanEpoch, OnlineJointPlanView
from beliefkv.policy.reference import AdmissionAction, AdmissionIntent
from beliefkv.runtime.lock_service import (
    LockedExtentAttribution,
    RequestServiceLedger,
    TentativeUnlockPreview,
)
from beliefkv.runtime.protocol import CommandAck, CommandKind, CommandStatus, PageHandle
from beliefkv.runtime.sglang_v052rc1 import EmbeddedSGLangRuntime


def candidate(
    request_id: str,
    *,
    private_bytes: int = 100,
    service_status: str = "stale",
    causal_rank: int = 3,
    frontier_class: str = "unknown",
    prediction_support: str = "unavailable",
    service_to_boundary_tokens: float | None = None,
    join_criticality: float = 0.0,
) -> RunningRetractionCandidate:
    return RunningRetractionCandidate(
        request_id=request_id,
        workflow_id=f"wf-{request_id}",
        invocation_id=f"inv-{request_id}",
        context_id=f"ctx-{request_id}",
        private_kv_bytes=private_bytes,
        service_status=service_status,
        stale_for_ms=500.0,
        causal_rank=causal_rank,
        unblock_depth=0,
        workflow_fair_rank=1,
        frontier_class=frontier_class,
        prediction_support=prediction_support,
        service_to_boundary_tokens=service_to_boundary_tokens,
        join_criticality=join_criticality,
    )


def snapshot(
    *,
    candidates,
    extents=(),
    active_footprint=1500,
    active_budget=1000,
    native_capacity=0,
    running_count=3,
) -> ObservedRetractionSnapshot:
    return ObservedRetractionSnapshot(
        observed_ts_ms=1000.0,
        page_revision=7,
        topology_revision=3,
        hbm_capacity_bytes=2000,
        active_kv_budget_bytes=active_budget,
        active_kv_footprint_bytes=active_footprint,
        native_reclaim_capacity_bytes=native_capacity,
        admission_stall_ms=500.0,
        running_request_count=running_count,
        minimum_active_requests=1,
        candidates=tuple(candidates),
        locked_extents=tuple(extents),
        replacements=(RetractionReplacement("replacement", 400),),
    )


class ObservedRetractionPlannerTest(unittest.TestCase):
    def setUp(self):
        self.planner = ObservedRetractionPlanner(
            ObservedRetractionConfig(
                minimum_admission_stall_ms=100.0,
                minimum_reclaim_bytes=1,
            )
        )

    def test_selects_complete_blocker_set_for_physical_unlock(self):
        plan = self.planner.plan(
            snapshot(
                candidates=(candidate("a"), candidate("b")),
                extents=(
                    RetractionLockedExtent(
                        "extent-shared",
                        600,
                        ("a", "b"),
                        True,
                    ),
                ),
            )
        )

        self.assertIsNotNone(plan)
        self.assertEqual(set(plan.request_ids), {"a", "b"})
        self.assertEqual(plan.expected_private_reclaim_bytes, 200)
        self.assertEqual(plan.expected_lock_release_bytes, 600)
        self.assertEqual(plan.expected_reclaim_capacity_bytes, 800)
        self.assertEqual(plan.replacement_request_ids, ("replacement",))

    def test_disabled_frontier_annotations_preserve_observed_result(self):
        baseline = self.planner.plan(
            snapshot(
                candidates=(candidate("a", private_bytes=600), candidate("b", private_bytes=600)),
                active_footprint=1400,
            )
        )
        annotated = self.planner.plan(
            snapshot(
                candidates=(
                    candidate(
                        "a",
                        private_bytes=600,
                        frontier_class="expand",
                        prediction_support="exact",
                        service_to_boundary_tokens=8,
                    ),
                    candidate(
                        "b",
                        private_bytes=600,
                        frontier_class="hold",
                        prediction_support="exact",
                        service_to_boundary_tokens=512,
                    ),
                ),
                active_footprint=1400,
            )
        )

        self.assertEqual(annotated, baseline)

    def test_frontier_aware_planner_protects_expand_and_prefers_hold(self):
        planner = ObservedRetractionPlanner(
            ObservedRetractionConfig(
                minimum_admission_stall_ms=100.0,
                minimum_reclaim_bytes=1,
                frontier_aware_enabled=True,
            )
        )
        plan = planner.plan(
            snapshot(
                candidates=(
                    candidate(
                        "expand",
                        private_bytes=600,
                        frontier_class="expand",
                        prediction_support="exact",
                        service_to_boundary_tokens=8,
                    ),
                    candidate(
                        "hold",
                        private_bytes=600,
                        frontier_class="hold",
                        prediction_support="exact",
                        service_to_boundary_tokens=512,
                    ),
                ),
                active_footprint=1400,
            )
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.request_ids, ("hold",))
        self.assertEqual(plan.reason, "frontier_aware_lock_closure_reclaim")

    def test_frontier_aware_planner_prioritizes_expand_replacement(self):
        planner = ObservedRetractionPlanner(
            ObservedRetractionConfig(
                minimum_admission_stall_ms=100.0,
                minimum_reclaim_bytes=1,
                frontier_aware_enabled=True,
            )
        )
        source = snapshot(
            candidates=(
                candidate(
                    "hold",
                    private_bytes=1000,
                    frontier_class="hold",
                    prediction_support="exact",
                ),
            ),
            active_footprint=1400,
            running_count=2,
        )
        source = source.__class__(
            **{
                **source.__dict__,
                "replacements": (
                    RetractionReplacement("observed-first", 100),
                    RetractionReplacement(
                        "expand-first",
                        100,
                        frontier_class="expand",
                        prediction_support="exact",
                        service_to_boundary_tokens=16,
                    ),
                ),
            }
        )

        plan = planner.plan(source)

        self.assertIsNotNone(plan)
        self.assertEqual(
            plan.replacement_request_ids,
            ("expand-first", "observed-first"),
        )
    def test_does_not_count_partially_attributed_extent(self):
        decision = self.planner.decide(
            snapshot(
                candidates=(candidate("a"), candidate("b")),
                extents=(
                    RetractionLockedExtent(
                        "extent-unknown",
                        600,
                        ("a", "b"),
                        False,
                    ),
                ),
            )
        )

        self.assertIsNone(decision.plan)
        self.assertEqual(decision.reason, "insufficient_unlock_capacity")
        self.assertEqual(decision.target_reclaim_bytes, 500)
        self.assertEqual(decision.eligible_candidate_count, 2)
        self.assertEqual(decision.fully_attributed_extent_count, 0)
        self.assertEqual(decision.reclaim_capacity_bytes, 200)

    def test_recent_service_and_active_floor_fail_closed(self):
        recent = snapshot(
            candidates=(candidate("a", service_status="recent"), candidate("b")),
            running_count=2,
        )
        self.assertIsNone(self.planner.plan(recent))

        floor = snapshot(
            candidates=(candidate("a", private_bytes=1000),),
            running_count=1,
        )
        self.assertIsNone(self.planner.plan(floor))

    def test_no_pressure_does_not_retract(self):
        no_pressure = snapshot(
            candidates=(candidate("a", private_bytes=1000), candidate("b")),
            active_footprint=800,
            active_budget=1000,
            native_capacity=500,
        )
        self.assertIsNone(self.planner.plan(no_pressure))
        self.assertEqual(
            self.planner.decide(no_pressure).reason,
            "pressure_absent",
        )

    def test_decision_reports_missing_stale_candidate(self):
        decision = self.planner.decide(
            snapshot(
                candidates=(
                    candidate("a", service_status="recent"),
                    candidate("b", service_status="unknown"),
                ),
            )
        )

        self.assertIsNone(decision.plan)
        self.assertEqual(decision.reason, "no_eligible_stale_candidate")
        self.assertEqual(decision.candidate_count, 2)
        self.assertEqual(decision.eligible_candidate_count, 0)

    def test_config_requires_observed_admission(self):
        with self.assertRaisesRegex(ValueError, "observed admission"):
            BeliefKVConfig(running_batch_retraction_enabled=True)


def test_runtime_projects_frontier_prediction_into_retraction_annotation():
    runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
    runtime.controller = SimpleNamespace(graph=SimpleNamespace(joins={}))
    runtime._last_frontier_predictions = {
        "inv": {
            "boundary_distribution": {"tool": 0.8, "final": 0.2},
            "remaining_decode_tokens": {
                "values": [16, 64],
                "probability_mass": [0.6, 0.4],
            },
            "support_level": "exact",
        }
    }

    annotation = runtime._frontier_retraction_annotation("inv")

    assert annotation == {
        "frontier_class": "expand",
        "prediction_support": "exact",
        "service_to_boundary_tokens": 16.0,
        "join_criticality": 0.0,
    }


class _Audit:
    def __init__(self):
        self.events = []

    def emit(self, event, ts_ms, **fields):
        self.events.append((event, ts_ms, fields))


class RestoreMicroGateTest(unittest.TestCase):
    @staticmethod
    def _runtime() -> EmbeddedSGLangRuntime:
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            queue_service_observer_enabled=True,
            queue_service_observer_include_runtime_batches=True,
            joint_policy_enabled=True,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_reclaim_bytes=1,
            restore_micro_gate_enabled=True,
            restore_micro_gate_min_private_bytes=1,
        )
        runtime.audit = _Audit()
        runtime._restore_micro_gate_state = {
            "enabled": True,
            "gate_id": runtime.config.restore_micro_gate_id,
            "stage": "armed",
        }
        runtime._restore_micro_gate_last_audit_signature = None
        runtime._online_joint_epoch_sequence = 0
        runtime._lock_service_ledger = RequestServiceLedger()
        runtime._lock_service_ledger.observe_selected(
            request_id="victim",
            workflow_id="restore-micro-gate:victim",
            invocation_id="victim-invocation",
            context_id="victim-context",
            ts_ms=900.0,
        )
        runtime._lock_service_ledger.observe_completed(
            "victim",
            ts_ms=901.0,
            phase="decode",
        )
        runtime._joint_retraction_solver = ObservedRetractionPlanner(
            ObservedRetractionConfig(
                minimum_admission_stall_ms=100.0,
                minimum_reclaim_bytes=1,
            )
        )
        runtime._request_metadata_by_id = {
            "replacement": SimpleNamespace(
                root_workflow_id=(
                    runtime.config.restore_micro_gate_replacement_workflow_id
                )
            )
        }
        return runtime

    @staticmethod
    def _snapshot() -> ObservedRetractionSnapshot:
        return ObservedRetractionSnapshot(
            observed_ts_ms=1000.0,
            page_revision=7,
            topology_revision=3,
            hbm_capacity_bytes=2000,
            active_kv_budget_bytes=1000,
            active_kv_footprint_bytes=500,
            native_reclaim_capacity_bytes=1000,
            admission_stall_ms=0.0,
            running_request_count=2,
            minimum_active_requests=1,
            candidates=(
                RunningRetractionCandidate(
                    request_id="victim",
                    workflow_id="restore-micro-gate:victim",
                    invocation_id="victim-invocation",
                    context_id="victim-context",
                    private_kv_bytes=200,
                    service_status="recent",
                    stale_for_ms=0.0,
                    causal_rank=1,
                    unblock_depth=0,
                    workflow_fair_rank=0,
                ),
            ),
            locked_extents=(
                RetractionLockedExtent(
                    "victim-extent", 600, ("victim",), True
                ),
            ),
            replacements=(RetractionReplacement("replacement", 400),),
        )

    def test_explicit_pair_forces_pressure_but_preserves_physical_eligibility(self):
        runtime = self._runtime()

        forced, gate_id = runtime._restore_micro_gate_snapshot(self._snapshot())

        self.assertEqual(gate_id, "p5g-restore-v1")
        self.assertEqual(forced.candidates[0].service_status, "stale")
        self.assertEqual(forced.native_reclaim_capacity_bytes, 1000)
        self.assertGreater(
            forced.active_kv_footprint_bytes - forced.active_kv_budget_bytes,
            forced.candidates[0].private_kv_bytes,
        )

    def test_forced_retraction_is_one_atomic_joint_action_group(self):
        runtime = self._runtime()
        forced, gate_id = runtime._restore_micro_gate_snapshot(self._snapshot())

        decision, plan_id = runtime._running_retraction_decision(
            forced,
            restore_micro_gate_id=gate_id,
        )

        self.assertIsNotNone(decision.plan)
        self.assertEqual(decision.plan.request_ids, ("victim",))
        self.assertEqual(decision.plan.reason, "restore_micro_gate_forced")
        self.assertEqual(plan_id, runtime._current_joint_plan_epoch.source_plan_id)
        groups = runtime._current_joint_plan_epoch.action_groups
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            {item.slice_id for item in groups[0].actions},
            {"request:replacement", "retraction:victim"},
        )

    def test_explicit_pair_drains_overlap_without_natural_pressure(self):
        runtime = self._runtime()
        runtime._now_ms = lambda: 1000.0
        runtime._last_retraction_decision_ms = None
        runtime._pending_running_retraction_transaction = None
        runtime._pending_online_joint_residency = None
        runtime._pending_running_retraction_barrier = None
        runtime._retraction_admission_stall_since_ms = 0.0
        runtime._running_retraction_counts = Counter()
        runtime.controller = SimpleNamespace(has_pending_transfer_work=lambda: False)
        runtime._running_retraction_replacements = lambda now_ms: (
            RetractionReplacement("replacement", 400),
        )
        runtime._native_reclaim_capacity_bytes = lambda: 500
        runtime._observed_admission_snapshot = (
            lambda **_kwargs: SimpleNamespace(active_kv_footprint_bytes=500)
        )
        runtime._metadata = lambda req: req.metadata
        runtime._running_retraction_barrier_state = lambda *args, **kwargs: {}
        runtime._preview_running_retraction_barrier_unlock = (
            lambda **_kwargs: (None, "unavailable", 1.0)
        )
        victim = SimpleNamespace(
            rid="victim",
            seqlen=1,
            metadata=SimpleNamespace(
                root_workflow_id=runtime.config.restore_micro_gate_victim_workflow_id
            ),
        )

        self.assertTrue(
            runtime.running_batch_retraction_barrier_required(
                SimpleNamespace(reqs=[victim, object()])
            )
        )
        requested = next(
            fields
            for event, _ts_ms, fields in runtime.audit.events
            if event == "running_retraction_overlap_barrier_requested"
        )
        self.assertTrue(requested["restore_micro_gate_forced"])
        self.assertEqual(requested["replacement_deficit_bytes"], 0)
        self.assertEqual(requested["active_excess_bytes"], 0)

    def test_explicit_pair_does_not_drain_before_private_kv_threshold(self):
        runtime = self._runtime()
        runtime._metadata = lambda req: req.metadata
        victim = SimpleNamespace(
            rid="victim",
            seqlen=0,
            metadata=SimpleNamespace(
                root_workflow_id=runtime.config.restore_micro_gate_victim_workflow_id
            ),
        )

        self.assertFalse(
            runtime._restore_micro_gate_barrier_pair_visible(
                (victim,),
                (RetractionReplacement("replacement", 400),),
            )
        )


class RunningRetractionCommitTest(unittest.TestCase):
    def test_online_joint_plan_authorizes_retraction_victims_and_replacement(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_stall_ms=100.0,
            running_batch_retraction_min_reclaim_bytes=1,
            joint_policy_enabled=True,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime._current_online_joint_view = OnlineJointPlanView(
            plan_id="joint-1",
            ordered_request_ids=("replacement",),
            immediate_request_ids=("replacement",),
            restore_requirements=(),
            deferred_request_ids=("a", "b"),
            residency_intent_indices=(),
        )
        runtime._current_joint_plan_epoch = JointPlanEpoch(
            epoch_id="joint-1:epoch:1",
            source_plan_id="joint-1",
            planner_mode=JointPlannerMode.OPTIMIZED,
            view=runtime._current_online_joint_view,
            action_slices=(),
            source_action_count=0,
            committed_action_count=0,
        )
        runtime.audit = _Audit()
        runtime._online_joint_result = SimpleNamespace(
            plan=SimpleNamespace(
                plan_id="joint-1",
                retractions=(SimpleNamespace(request_id="a"),),
                admissions=(
                    AdmissionIntent(
                        "a", AdmissionAction.DEFER, 0, (), "pause a"
                    ),
                    AdmissionIntent(
                        "b", AdmissionAction.DEFER, 0, (), "pause b"
                    ),
                    AdmissionIntent(
                        "replacement",
                        AdmissionAction.ADMIT,
                        400,
                        (),
                        "replace",
                    ),
                ),
            )
        )

        decision, source_plan_id = runtime._running_retraction_decision(
            snapshot(
                candidates=(
                    candidate("a", private_bytes=600, service_status="recent"),
                    candidate("b", private_bytes=100, service_status="recent"),
                )
            )
        )

        self.assertEqual(source_plan_id, "joint-1")
        self.assertIsNotNone(decision.plan)
        assert decision.plan is not None
        self.assertEqual(decision.plan.request_ids, ("a",))
        self.assertEqual(decision.plan.replacement_request_ids, ("replacement",))
        self.assertEqual(
            decision.plan.reason,
            "observed_joint_pause_authorized_lock_reclaim",
        )

    def test_predrain_tentative_preview_is_an_observed_stale_upper_bound(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_stall_ms=100.0,
            running_batch_retraction_min_reclaim_bytes=1,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = BeliefKVController(config)
        parent = PageHandle(1, 0)
        child = PageHandle(2, 0)
        runtime.controller.page_index.register_page(
            parent, size_bytes=100, radix_depth=1
        )
        runtime.controller.page_index.register_page(
            child,
            size_bytes=200,
            radix_depth=2,
            parent=parent,
        )
        runtime.controller.page_index.set_engine_lock(child, 1)
        runtime._lock_provenance_extents = lambda: (
            (
                LockedExtentAttribution(
                    handle=child,
                    size_bytes=200,
                    engine_lock_ref=1,
                    blocker_request_ids=("victim",),
                ),
            ),
            0,
        )
        ledger = RequestServiceLedger()
        ledger.observe_selected(
            request_id="victim",
            workflow_id="wf",
            invocation_id="inv",
            context_id="ctx",
            ts_ms=0.0,
        )
        ledger.observe_completed("victim", ts_ms=10.0, phase="decode")
        runtime._lock_service_ledger = ledger

        preview, status, compute_us = (
            runtime._preview_running_retraction_barrier_unlock(now_ms=1000.0)
        )

        self.assertEqual(status, "previewed")
        self.assertIsNotNone(preview)
        self.assertEqual(preview.request_ids, ("victim",))
        self.assertEqual(preview.newly_migratable_bytes, 300)
        self.assertGreaterEqual(compute_us, 0.0)
        self.assertEqual(
            runtime.controller.page_index.pages[child].engine_lock_ref,
            1,
        )

    def test_overlap_barrier_waits_for_pressure_and_stall(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_stall_ms=100.0,
            running_batch_retraction_min_reclaim_bytes=1,
        )
        now_ms = [1000.0]
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.audit = _Audit()
        runtime._now_ms = lambda: now_ms[0]
        runtime._last_retraction_decision_ms = None
        runtime._pending_running_retraction_transaction = None
        runtime._retraction_admission_stall_since_ms = None
        runtime._running_retraction_counts = Counter()
        runtime._running_retraction_replacements = lambda now_ms: (
            RetractionReplacement("replacement", 400),
        )
        runtime._native_reclaim_capacity_bytes = lambda: 0
        runtime._observed_admission_snapshot = (
            lambda **_kwargs: SimpleNamespace(active_kv_footprint_bytes=1900)
        )
        batch = SimpleNamespace(reqs=[object(), object()])

        self.assertFalse(
            runtime.running_batch_retraction_barrier_required(batch)
        )
        now_ms[0] = 1101.0
        self.assertTrue(
            runtime.running_batch_retraction_barrier_required(batch)
        )
        self.assertEqual(
            runtime._running_retraction_counts["overlap_barrier_requested"],
            1,
        )
        self.assertEqual(
            runtime.audit.events[-1][0],
            "running_retraction_overlap_barrier_requested",
        )
        self.assertEqual(
            runtime.audit.events[-1][2]["barrier_intent_id"],
            "retraction-barrier-1",
        )

    def test_overlap_barrier_attributes_pressure_resolved_by_drain(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_stall_ms=100.0,
            running_batch_retraction_min_reclaim_bytes=1,
        )
        now_ms = [1000.0]
        native_reclaim_bytes = [0]
        active_footprint_bytes = [1900]
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.audit = _Audit()
        runtime._now_ms = lambda: now_ms[0]
        runtime._last_retraction_decision_ms = None
        runtime._pending_running_retraction_transaction = None
        runtime._pending_running_retraction_barrier = None
        runtime._retraction_admission_stall_since_ms = None
        runtime._running_retraction_counts = Counter()
        runtime._running_retraction_barrier_outcomes = Counter()
        runtime._running_retraction_replacements = lambda now_ms: (
            RetractionReplacement("replacement", 400),
        )
        runtime._native_reclaim_capacity_bytes = lambda: native_reclaim_bytes[0]
        runtime._observed_admission_snapshot = (
            lambda **_kwargs: SimpleNamespace(
                active_kv_footprint_bytes=active_footprint_bytes[0]
            )
        )
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]
        )

        self.assertFalse(runtime.running_batch_retraction_barrier_required(batch))
        now_ms[0] = 1101.0
        self.assertTrue(runtime.running_batch_retraction_barrier_required(batch))
        native_reclaim_bytes[0] = 500
        active_footprint_bytes[0] = 500
        now_ms[0] = 1110.0
        runtime.on_running_batch_retraction_barrier_drained(batch)
        intent_id = runtime._attribute_running_retraction_barrier(
            batch,
            now_ms=1111.0,
            planning_reason="pressure_absent",
            decision=ObservedRetractionDecision(
                plan=None,
                reason="pressure_absent",
                candidate_count=2,
            ),
        )

        self.assertEqual(intent_id, "retraction-barrier-1")
        self.assertIsNone(runtime._pending_running_retraction_barrier)
        self.assertEqual(
            runtime._running_retraction_barrier_outcomes,
            Counter({"pressure_resolved_by_drain": 1}),
        )
        events = {event: fields for event, _ts_ms, fields in runtime.audit.events}
        self.assertEqual(
            events["running_retraction_overlap_barrier_requested"][
                "barrier_intent_id"
            ],
            events["running_retraction_overlap_barrier_drained"][
                "barrier_intent_id"
            ],
        )
        outcome = events["running_retraction_overlap_barrier_outcome"]
        self.assertEqual(outcome["barrier_intent_id"], "retraction-barrier-1")
        self.assertEqual(outcome["outcome"], "pressure_resolved_by_drain")
        self.assertEqual(
            outcome["requested_to_drained_delta"][
                "native_reclaim_capacity_bytes"
            ],
            500,
        )

    def test_overlap_barrier_is_not_inserted_without_reclaim_pressure(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_stall_ms=100.0,
            running_batch_retraction_min_reclaim_bytes=1,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.audit = _Audit()
        runtime._now_ms = lambda: 1000.0
        runtime._last_retraction_decision_ms = None
        runtime._pending_running_retraction_transaction = None
        runtime._retraction_admission_stall_since_ms = 0.0
        runtime._running_retraction_counts = Counter()
        runtime._running_retraction_replacements = lambda now_ms: (
            RetractionReplacement("replacement", 400),
        )
        runtime._native_reclaim_capacity_bytes = lambda: 500
        runtime._observed_admission_snapshot = (
            lambda **_kwargs: SimpleNamespace(active_kv_footprint_bytes=500)
        )

        self.assertFalse(
            runtime.running_batch_retraction_barrier_required(
                SimpleNamespace(reqs=[object(), object()])
            )
        )
        self.assertEqual(
            runtime._running_retraction_counts["barrier_no_pressure"], 1
        )

    def test_overlap_barrier_is_suppressed_before_overdue_restore_rejection(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
        )
        runtime._now_ms = lambda: 1000.0
        runtime._running_retraction_counts = Counter()
        runtime._overdue_restore_obligation = lambda **_kwargs: SimpleNamespace(
            obligation_id="restore-1"
        )

        self.assertFalse(
            runtime.running_batch_retraction_barrier_required(
                SimpleNamespace(reqs=[object(), object()])
            )
        )
        self.assertEqual(
            runtime._running_retraction_counts[
                "barrier_restore_debt_suppressed"
            ],
            1,
        )

    def test_safe_point_starts_stall_timer_before_prefill_epoch_exists(self):
        config = BeliefKVConfig(
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime._now_ms = lambda: 1000.0
        runtime._last_retraction_decision_ms = None
        runtime._pending_running_retraction_transaction = None
        runtime._retraction_admission_stall_since_ms = None
        runtime._running_retraction_counts = Counter()
        runtime._running_retraction_replacements = lambda now_ms: (
            RetractionReplacement("replacement", 100),
        )

        plan = runtime.plan_running_batch_retraction(
            SimpleNamespace(reqs=[object(), object()])
        )

        self.assertIsNone(plan)
        self.assertEqual(runtime._retraction_admission_stall_since_ms, 1000.0)
        self.assertEqual(runtime._running_retraction_counts["stall_warming"], 1)

    def test_replacement_priority_requires_confirmed_native_capacity(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_reclaim_bytes=1,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = BeliefKVController(config)
        runtime.audit = _Audit()
        runtime._now_ms = lambda: 1000.0
        runtime._pending_selective_retraction_ids = set()
        runtime._retracted_engine_request_ids = set()
        runtime._retraction_counts_by_request = Counter()
        runtime._retraction_cooldown_until_by_request = {}
        runtime._retraction_priority_request_ids = ()
        runtime._running_retraction_counts = Counter()
        runtime._running_retraction_actual_reclaim_bytes = 0
        runtime._running_retraction_actual_lock_release_bytes = 0
        runtime._mark_full_tree_rebuild = lambda: None
        runtime.sync_tree = lambda force=False: None
        plan = RunningRetractionPlan(
            request_ids=("victim",),
            replacement_request_ids=("replacement",),
            target_reclaim_bytes=500,
            expected_private_reclaim_bytes=200,
            expected_lock_release_bytes=600,
            expected_reclaim_capacity_bytes=800,
            native_reclaim_capacity_before=100,
            engine_locked_bytes_before=600,
            page_revision=0,
            topology_revision=0,
            observed_ts_ms=900.0,
            reason="test",
        )
        transaction = SimpleNamespace(
            transaction_id="retraction-1",
            plan=plan,
            barrier_intent_id="retraction-barrier-1",
            tentative_unlock_preview=TentativeUnlockPreview(
                request_ids=("victim",),
                page_revision=0,
                topology_revision=0,
                exact=True,
                reason="projected_unlock",
                path_error_count=0,
                provenance_extent_count=1,
                selected_blocker_extent_count=1,
                attribution_gap_bytes=0,
                baseline_engine_locked_bytes=600,
                projected_engine_locked_bytes=0,
                projected_engine_lock_release_bytes=600,
                baseline_migratable_bytes=0,
                projected_migratable_bytes=600,
                lock_ref_zeroed_bytes=600,
                newly_migratable_bytes=600,
                lock_ref_zeroed_handles=(),
                newly_migratable_handles=(),
            ),
            stage="planned",
            actual_request_ids=(),
            native_reclaim_capacity_after=0,
            actual_reclaim_capacity_bytes=0,
            actual_engine_lock_release_bytes=0,
            failure_reason=None,
        )
        runtime._pending_running_retraction_transaction = transaction

        runtime.on_running_batch_retracted(
            plan,
            (SimpleNamespace(rid="victim"),),
            native_reclaim_capacity_before_tokens=10,
            native_reclaim_capacity_after_tokens=90,
        )

        self.assertEqual(transaction.stage, "reclaim_confirmed")
        self.assertEqual(runtime._retraction_priority_request_ids, ("replacement",))
        self.assertIn("victim", runtime._pending_selective_retraction_ids)
        self.assertIn("victim", runtime._retracted_engine_request_ids)
        self.assertGreater(
            runtime._retraction_cooldown_until_by_request["victim"], 1000.0
        )
        realized = next(
            fields
            for event, _ts_ms, fields in runtime.audit.events
            if event == "running_retraction_tentative_unlock_realized"
        )
        self.assertEqual(realized["barrier_intent_id"], "retraction-barrier-1")
        self.assertEqual(realized["preview_newly_migratable_bytes"], 600)
        self.assertEqual(realized["realized_migratable_delta_bytes"], 0)
        self.assertTrue(realized["request_set_matches"])

    def test_evictable_growth_does_not_release_replacement_without_free_hbm(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_reclaim_bytes=1,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = BeliefKVController(config)
        runtime.audit = _Audit()
        runtime._now_ms = lambda: 1000.0
        runtime._pending_selective_retraction_ids = set()
        runtime._retracted_engine_request_ids = set()
        runtime._retraction_counts_by_request = Counter()
        runtime._retraction_cooldown_until_by_request = {}
        runtime._retraction_priority_request_ids = ()
        runtime._running_retraction_counts = Counter()
        runtime._running_retraction_actual_reclaim_bytes = 0
        runtime._running_retraction_actual_lock_release_bytes = 0
        runtime._request_metadata_by_id = {}
        runtime._mark_full_tree_rebuild = lambda: None
        runtime.sync_tree = lambda force=False: None
        runtime._activate_retraction_admission_barrier = lambda _transaction: None
        queued = []
        runtime._queue_next_retraction_residency_command = (
            lambda transaction, now_ms: queued.append((transaction, now_ms))
        )
        plan = RunningRetractionPlan(
            request_ids=("victim",),
            replacement_request_ids=("replacement",),
            target_reclaim_bytes=500,
            expected_private_reclaim_bytes=200,
            expected_lock_release_bytes=600,
            expected_reclaim_capacity_bytes=800,
            native_reclaim_capacity_before=100,
            engine_locked_bytes_before=600,
            page_revision=0,
            topology_revision=0,
            observed_ts_ms=900.0,
            reason="test",
        )
        transaction = SimpleNamespace(
            transaction_id="retraction-2",
            plan=plan,
            stage="planned",
            actual_request_ids=(),
            native_reclaim_capacity_after=0,
            actual_reclaim_capacity_bytes=0,
            actual_engine_lock_release_bytes=0,
            failure_reason=None,
        )
        runtime._pending_running_retraction_transaction = transaction

        runtime.on_running_batch_retracted(
            plan,
            (SimpleNamespace(rid="victim"),),
            native_reclaim_capacity_before_tokens=10,
            native_reclaim_capacity_after_tokens=90,
            native_available_before_tokens=10,
            native_available_after_tokens=20,
        )

        self.assertEqual(transaction.stage, "residency_pending")
        self.assertEqual(transaction.actual_reclaim_capacity_bytes, 100)
        self.assertEqual(runtime._retraction_priority_request_ids, ())
        self.assertEqual(len(queued), 1)

    def test_physical_ack_releases_replacement_only_after_allocator_delta(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            host_capacity_bytes=4000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            shadow_enabled=False,
            prefetch_enabled=False,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
            running_batch_retraction_min_reclaim_bytes=1,
        )
        controller = BeliefKVController(config)
        controller.process_runtime_events(
            (
                RuntimeEvent("start", 1.0, RuntimeEventKind.WORKFLOW_START, "wf"),
                RuntimeEvent(
                    "create",
                    2.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id="victim",
                    context_id="ctx-victim",
                    context_epoch=0,
                ),
                RuntimeEvent(
                    "submit",
                    3.0,
                    RuntimeEventKind.LLM_SUBMIT,
                    "wf",
                    invocation_id="victim",
                ),
            )
        )
        handle = PageHandle(1, 0)
        controller.page_index.register_page(handle, size_bytes=500)
        controller.page_index.bind_pages("ctx-victim", 0, (handle,))

        plan = RunningRetractionPlan(
            request_ids=("victim",),
            replacement_request_ids=("replacement",),
            target_reclaim_bytes=500,
            expected_private_reclaim_bytes=100,
            expected_lock_release_bytes=500,
            expected_reclaim_capacity_bytes=600,
            native_reclaim_capacity_before=100,
            engine_locked_bytes_before=500,
            page_revision=controller.page_index.revision,
            topology_revision=controller.page_index.topology_revision,
            observed_ts_ms=900.0,
            reason="test",
        )
        transaction = SimpleNamespace(
            transaction_id="retraction-physical",
            plan=plan,
            created_ts_ms=900.0,
            stage="residency_pending",
            actual_reclaim_capacity_bytes=100,
            allocator_available_before_bytes=100,
            allocator_available_after_bytes=200,
            required_allocator_available_bytes=400,
            victim_context_ids=("ctx-victim",),
            pending_command_id=None,
            pending_command_kind=None,
            residency_command_ids=[],
            explicit_reclaim_bytes=0,
            explicit_transfer_bytes=0,
            private_reclaim_bytes=100,
            command_attempt_count=0,
            failure_reason=None,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.scheduler = SimpleNamespace(waiting_queue=())
        runtime.audit = _Audit()
        runtime._running_retraction_counts = Counter()
        runtime._retraction_priority_request_ids = ()
        runtime._pending_running_retraction_transaction = transaction
        runtime._runtime_resource_observation = lambda now_ms: SimpleNamespace(
            host_free_bytes=4000
        )

        runtime._queue_next_retraction_residency_command(
            transaction, now_ms=1000.0
        )
        tick = controller.tick(1000.0)

        self.assertIsNotNone(tick.transfer)
        self.assertEqual(tick.transfer.command.kind, CommandKind.OFFLOAD_CONTEXT)
        self.assertEqual(runtime._retraction_priority_request_ids, ())
        self.assertEqual(transaction.pending_command_id, tick.transfer.command.command_id)

        runtime._allocator_available_bytes = lambda: 700
        runtime._advance_retraction_transaction(
            (
                CommandAck(
                    transaction.pending_command_id,
                    CommandStatus.COMPLETED,
                    1010.0,
                    actual_bytes=500,
                    page_handles=(handle,),
                ),
            ),
            now_ms=1010.0,
        )

        self.assertEqual(transaction.stage, "reclaim_confirmed")
        self.assertEqual(runtime._retraction_priority_request_ids, ("replacement",))
        self.assertEqual(transaction.explicit_transfer_bytes, 500)
        self.assertEqual(transaction.actual_reclaim_capacity_bytes, 600)


class SGLangRetractionPatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scheduler_path = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "sglang"
            / "python"
            / "sglang"
            / "srt"
            / "managers"
            / "scheduler.py"
        )
        source = cls.scheduler_path.read_text(encoding="utf-8")
        cls.source = source
        tree = ast.parse(source, filename=str(cls.scheduler_path))
        scheduler = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
        )
        cls.methods = {
            node.name: node
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef)
        }

    @classmethod
    def _load_method(cls, name):
        node = cls.methods[name]
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(cls.scheduler_path), "exec"), namespace)
        return namespace[name]

    def test_overlap_safe_point_requires_an_empty_result_pipeline(self):
        safe_point = self._load_method(
            "_beliefkv_running_retraction_safe_point"
        )
        scheduler = SimpleNamespace(
            enable_overlap=True,
            last_batch=object(),
            result_queue=[object()],
        )

        self.assertFalse(safe_point(scheduler))
        scheduler.last_batch = None
        scheduler.result_queue.clear()
        self.assertTrue(safe_point(scheduler))

    def test_retraction_commit_does_not_prepare_decode_tensors(self):
        apply_retraction = self._load_method(
            "_apply_beliefkv_running_retraction"
        )

        class Batch:
            is_hybrid = False
            batch_is_full = True

            def __init__(self):
                self.prepare_calls = 0

            def retract_selected(self, request_ids, server_args):
                self.request_ids = tuple(request_ids)
                return [SimpleNamespace(rid="victim")]

            def prepare_for_decode(self):
                self.prepare_calls += 1
                raise AssertionError("retraction commit prepared decode twice")

        batch = Batch()
        callbacks = []
        requeued = []
        scheduler = SimpleNamespace(
            running_batch=batch,
            server_args=object(),
            beliefkv_runtime=SimpleNamespace(
                on_running_batch_retracted=lambda *args, **kwargs: callbacks.append(
                    (args, kwargs)
                )
            ),
            total_retracted_reqs=0,
            _extend_requests_to_queue=lambda reqs, is_retracted: requeued.append(
                (reqs, is_retracted)
            ),
        )
        scheduler._beliefkv_native_retraction_capacity = MethodType(
            lambda _self: (10, 20), scheduler
        )

        apply_retraction(
            scheduler,
            SimpleNamespace(request_ids=("victim",)),
        )

        self.assertEqual(batch.prepare_calls, 0)
        self.assertFalse(batch.batch_is_full)
        self.assertEqual(scheduler.total_retracted_reqs, 1)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(len(requeued), 1)

    def test_overlap_barrier_fails_closed_during_chunked_prefill_and_pp(self):
        method_source = ast.get_source_segment(
            self.source,
            self.methods["get_next_batch_to_run"],
        )

        self.assertIn("self.chunked_req is None", method_source)
        self.assertIn("self.pp_size == 1", method_source)


if __name__ == "__main__":
    unittest.main()
