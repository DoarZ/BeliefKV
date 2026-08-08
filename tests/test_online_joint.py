from __future__ import annotations

from types import SimpleNamespace
import unittest

from beliefkv.policy.joint_scheduler import (
    IntentValidation,
    JointPlanComponentValidation,
    JointPlannerMode,
)
from beliefkv.policy.online_joint import (
    ActionGroup,
    ActionGroupAtomicity,
    ActionGroupResourceCertificate,
    ActionGroupValidationState,
    ActionSlice,
    append_committed_action_slice,
    compile_bounded_seed_epoch,
    compile_online_joint_view,
    validate_action_group_resource_certificate,
)
from beliefkv.policy.reference import (
    AdmissionAction,
    AdmissionIntent,
    ExecutionIntent,
    ResidencyAction,
    ResidencyIntent,
    TransferDependency,
)


def _validation(
    *,
    requests: tuple[str, ...],
    bundle_ids: tuple[str, ...] = (),
    dependency_count: int = 0,
    dependency_reason: str | None = None,
) -> JointPlanComponentValidation:
    return JointPlanComponentValidation(
        strict_global_reasons=(),
        global_reasons=(),
        execution=IntentValidation(),
        admissions={request_id: IntentValidation() for request_id in requests},
        residency={bundle_id: IntentValidation() for bundle_id in bundle_ids},
        dependencies=tuple(
            IntentValidation(
                (dependency_reason,) if dependency_reason is not None else ()
            )
            for _ in range(dependency_count)
        ),
    )


def _plan(
    *,
    admissions: tuple[AdmissionIntent, ...],
    residency: tuple[ResidencyIntent, ...] = (),
    dependencies: tuple[TransferDependency, ...] = (),
):
    request_ids = tuple(item.request_id for item in admissions)
    return SimpleNamespace(
        plan_id="plan-1",
        fallback_reason=None,
        transition_open=False,
        execution=ExecutionIntent(
            ordered_request_ids=request_ids,
            selected_workflow_id=None,
            selected_invocation_id=None,
            mode="observed_joint",
            graph_version=1,
            reason="test",
        ),
        admissions=admissions,
        residency=residency,
        dependencies=dependencies,
    )


class OnlineJointPlanCompilerTest(unittest.TestCase):
    def test_appended_action_rebuilds_dependency_closed_atomic_group(self):
        decision = compile_bounded_seed_epoch(
            ordered_request_ids=("replacement",),
            visible_request_ids=("replacement",),
            epoch_sequence=1,
            emergency=True,
        )
        epoch = append_committed_action_slice(
            decision.epoch,
            ActionSlice(
                "retraction:victim",
                "retraction",
                "victim",
                ("request:replacement",),
                True,
            ),
        )

        self.assertEqual(len(epoch.action_groups), 1)
        self.assertEqual(
            {item.slice_id for item in epoch.action_groups[0].actions},
            {"request:replacement", "retraction:victim"},
        )
        self.assertEqual(
            epoch.action_groups[0].atomicity,
            ActionGroupAtomicity.ALL_OR_NOTHING,
        )

    def test_bounded_seed_preserves_restore_requirement(self):
        decision = compile_bounded_seed_epoch(
            ordered_request_ids=("restore", "ready"),
            visible_request_ids=("restore", "ready"),
            restore_requirements=(("restore", ("page:1:0",)),),
            epoch_sequence=1,
            emergency=True,
        )

        assert decision.view is not None
        self.assertEqual(decision.view.immediate_request_ids, ("ready",))
        self.assertEqual(
            decision.view.restore_requirements,
            (("restore", ("page:1:0",)),),
        )
        self.assertEqual(
            decision.epoch.planner_mode,
            JointPlannerMode.EMERGENCY,
        )

    def test_compiles_immediate_requests_and_defers_unselected_visible_work(self):
        plan = _plan(
            admissions=(
                AdmissionIntent(
                    request_id="request-a",
                    action=AdmissionAction.ADMIT,
                    reserved_bytes=10,
                    required_bundle_ids=(),
                    reason="selected",
                ),
            )
        )

        decision = compile_online_joint_view(
            plan,
            _validation(requests=("request-a",)),
            visible_request_ids=("request-a", "request-b"),
        )

        self.assertEqual(decision.reason, "applicable")
        assert decision.view is not None
        self.assertEqual(decision.view.immediate_request_ids, ("request-a",))
        self.assertEqual(decision.view.deferred_request_ids, ("request-b",))

    def test_restore_dependency_blocks_ticket_and_selects_residency_intent(self):
        residency = ResidencyIntent(
            bundle_id="bundle-a",
            action=ResidencyAction.PREFETCH_GPU,
            target_bytes=1024,
            deadline_ms=10.0,
            scenario_support=frozenset({"observed"}),
            reason="restore",
        )
        plan = _plan(
            admissions=(
                AdmissionIntent(
                    request_id="request-a",
                    action=AdmissionAction.RESTORE_THEN_ADMIT,
                    reserved_bytes=10,
                    required_bundle_ids=("bundle-a",),
                    reason="restore first",
                ),
            ),
            residency=(residency,),
            dependencies=(
                TransferDependency(
                    before_request_id="request-a",
                    residency_intent_index=0,
                    require_ack=True,
                ),
            ),
        )

        decision = compile_online_joint_view(
            plan,
            _validation(
                requests=("request-a",),
                bundle_ids=("bundle-a",),
                dependency_count=1,
            ),
            visible_request_ids=("request-a",),
        )

        self.assertEqual(decision.reason, "applicable")
        assert decision.view is not None
        self.assertEqual(decision.view.immediate_request_ids, ())
        self.assertEqual(
            decision.view.restore_requirements,
            (("request-a", ("bundle-a",)),),
        )
        self.assertEqual(decision.view.residency_intent_indices, (0,))
        assert decision.epoch is not None
        self.assertEqual(len(decision.epoch.action_groups), 1)
        self.assertTrue(decision.epoch.action_groups[0].committed)

    def test_invalid_restore_dependency_removes_only_its_action_slice(self):
        residency = ResidencyIntent(
            bundle_id="bundle-a",
            action=ResidencyAction.PREFETCH_GPU,
            target_bytes=1024,
            deadline_ms=10.0,
            scenario_support=frozenset(),
            reason="restore",
        )
        plan = _plan(
            admissions=(
                AdmissionIntent(
                    request_id="request-a",
                    action=AdmissionAction.RESTORE_THEN_ADMIT,
                    reserved_bytes=10,
                    required_bundle_ids=("bundle-a",),
                    reason="restore first",
                ),
            ),
            residency=(residency,),
            dependencies=(
                TransferDependency(
                    before_request_id="request-a",
                    residency_intent_index=0,
                    require_ack=True,
                ),
            ),
        )

        decision = compile_online_joint_view(
            plan,
            _validation(
                requests=("request-a",),
                bundle_ids=("bundle-a",),
                dependency_count=1,
                dependency_reason="bundle_changed",
            ),
            visible_request_ids=("request-a",),
        )

        self.assertIsNotNone(decision.view)
        self.assertEqual(decision.reason, "no_action")
        assert decision.epoch is not None
        rejected = [
            item for item in decision.epoch.action_slices if not item.committed
        ]
        self.assertTrue(
            any(
                "dependency:bundle_changed" in item.reasons
                for item in rejected
            )
        )

    def test_missing_selected_request_produces_an_explicit_no_action_epoch(self):
        plan = _plan(
            admissions=(
                AdmissionIntent(
                    request_id="request-a",
                    action=AdmissionAction.ADMIT,
                    reserved_bytes=0,
                    required_bundle_ids=(),
                    reason="selected",
                ),
            )
        )

        decision = compile_online_joint_view(
            plan,
            _validation(requests=("request-a",)),
            visible_request_ids=("request-b",),
        )

        self.assertIsNotNone(decision.view)
        self.assertEqual(decision.reason, "no_action")
        assert decision.view is not None
        self.assertEqual(decision.view.ordered_request_ids, ())
        self.assertEqual(decision.view.deferred_request_ids, ("request-b",))

    def test_missing_request_does_not_invalidate_unrelated_request(self):
        plan = _plan(
            admissions=tuple(
                AdmissionIntent(
                    request_id=request_id,
                    action=AdmissionAction.ADMIT,
                    reserved_bytes=0,
                    required_bundle_ids=(),
                    reason="selected",
                )
                for request_id in ("request-a", "request-b")
            )
        )

        decision = compile_online_joint_view(
            plan,
            _validation(requests=("request-a", "request-b")),
            visible_request_ids=("request-b", "request-c"),
            epoch_sequence=7,
        )

        self.assertEqual(decision.reason, "partially_applicable")
        assert decision.view is not None
        assert decision.epoch is not None
        self.assertEqual(decision.view.ordered_request_ids, ("request-b",))
        self.assertEqual(decision.view.deferred_request_ids, ("request-c",))
        self.assertTrue(decision.epoch.epoch_id.endswith(":epoch:7"))
        self.assertGreater(decision.epoch.actionable_coverage, 0.0)
        self.assertLess(decision.epoch.actionable_coverage, 1.0)

    def test_action_group_certificate_revalidates_all_liveness_revisions(self):
        action = ActionSlice("prepare", "residency", "context", (), True)
        group = ActionGroup(
            group_id="group",
            atomicity=ActionGroupAtomicity.ALL_OR_NOTHING,
            actions=(action,),
            dependency_dag=(),
            resource_certificate=ActionGroupResourceCertificate(
                required_host_bytes=1024,
                planned_pcie_bytes=1024,
                topology_revision=4,
                allocator_revision=5,
                obligation_revision=6,
                lease_revision=7,
                grace_revision=8,
            ),
            compensation=("release speculative host shadow",),
            committed=True,
        )
        current = ActionGroupValidationState(
            hbm_available_bytes=0,
            host_free_bytes=2048,
            topology_revision=4,
            allocator_revision=5,
            obligation_revision=6,
            lease_revision=9,
            grace_revision=8,
        )

        self.assertEqual(
            validate_action_group_resource_certificate(group, current),
            ("lease_revision",),
        )

    def test_unrelated_allocator_progress_does_not_stale_a_group(self):
        group = ActionGroup(
            group_id="group",
            atomicity=ActionGroupAtomicity.ALL_OR_NOTHING,
            actions=(ActionSlice("admit", "request", "request", (), True),),
            dependency_dag=(),
            resource_certificate=ActionGroupResourceCertificate(
                required_hbm_bytes=100,
                topology_revision=4,
                allocator_revision=5,
            ),
            compensation=(),
            committed=True,
        )
        current = ActionGroupValidationState(
            hbm_available_bytes=100,
            host_free_bytes=0,
            topology_revision=4,
            allocator_revision=9,
            obligation_revision=0,
            lease_revision=0,
            grace_revision=0,
        )

        self.assertEqual(validate_action_group_resource_certificate(group, current), ())


if __name__ == "__main__":
    unittest.main()
