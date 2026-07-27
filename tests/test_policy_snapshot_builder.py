from __future__ import annotations

from dataclasses import replace

import pytest

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import (
    ContextMode,
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.policy.admission import AdmissionRequest
from beliefkv.policy.reference import PolicySnapshotError
from beliefkv.policy.resource_snapshot import RuntimeResourceObservation
from beliefkv.runtime.protocol import (
    PageHandle,
    PhysicalResidency,
    TransferDirection,
)


def _controller() -> BeliefKVController:
    controller = BeliefKVController(
        BeliefKVConfig(
            hbm_capacity_bytes=1_000,
            host_capacity_bytes=1_000,
            reserve_hbm_bytes=0,
            kv_bytes_per_token=10,
            predictor_enabled=False,
            shadow_enabled=False,
            prefetch_enabled=False,
        )
    )
    controller.process_runtime_events(
        [
            RuntimeEvent(
                event_id="workflow-start",
                ts_ms=0,
                kind=RuntimeEventKind.WORKFLOW_START,
                workflow_id="workflow",
            ),
            RuntimeEvent(
                event_id="root-create",
                ts_ms=1,
                kind=RuntimeEventKind.INVOCATION_CREATE,
                workflow_id="workflow",
                invocation_id="root",
                context_id="ctx-root",
                context_epoch=0,
                agent_definition_id="supervisor",
                agent_instance_id="supervisor-0",
                relation_type=RelationType.ROOT,
                context_mode=ContextMode.FRESH,
                execution_mode=ExecutionMode.FOREGROUND,
            ),
        ]
    )
    return controller


def _observation(
    *,
    ts_ms: float = 10,
    hbm_used: int = 0,
    host_used: int = 0,
) -> RuntimeResourceObservation:
    return RuntimeResourceObservation(
        ts_ms=ts_ms,
        hbm_capacity_bytes=1_000,
        hbm_used_bytes=hbm_used,
        host_capacity_bytes=1_000,
        host_used_bytes=host_used,
        host_free_bytes=1_000 - host_used,
    )


def _bind_two_level_tree(controller: BeliefKVController) -> None:
    root = PageHandle(1, 0)
    leaf = PageHandle(2, 0)
    controller.page_index.register_page(
        root,
        size_bytes=100,
        residency=PhysicalResidency.GPU_ONLY,
        radix_depth=1,
        last_access_ms=2,
    )
    controller.page_index.register_page(
        leaf,
        size_bytes=200,
        residency=PhysicalResidency.GPU_ONLY,
        radix_depth=2,
        parent=root,
        last_access_ms=3,
    )
    controller.page_index.bind_pages("ctx-root", 0, {root, leaf})


def test_snapshot_closes_authoritative_allocator_with_protected_untracked_bytes() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)
    controller.submit_request(
        AdmissionRequest(
            request_id="request-root",
            workflow_id="workflow",
            invocation_id="root",
            context_id="ctx-root",
            context_epoch=0,
            submitted_ts_ms=2,
            uncached_prompt_tokens=2,
            expected_output_tokens=3,
            kv_bytes_per_token=10,
        )
    )

    snapshot = controller.build_policy_input(
        _observation(hbm_used=450, host_used=50)
    )

    assert snapshot.physical_kv.gpu_bytes == 450
    assert snapshot.physical_kv.cpu_bytes == 50
    assert sum(item.gpu_bytes for item in snapshot.physical_kv.bundles) == 450
    assert sum(item.cpu_bytes for item in snapshot.physical_kv.bundles) == 50
    protected = {
        item.bundle_id: item
        for item in snapshot.physical_kv.bundles
        if item.scope == "protected_untracked"
    }
    assert protected["protected-untracked-hbm"].gpu_bytes == 150
    assert protected["protected-untracked-host"].cpu_bytes == 50
    assert all(not item.actionable for item in protected.values())
    assert snapshot.runnable_frontier[0].startup_bytes == 50
    assert snapshot.identity_mappings[0].native_request_id == "request-root"
    accounting = snapshot.runtime_graph.state["physical_accounting"]
    assert accounting["tracked_hbm_bytes"] == 300
    assert accounting["untracked_hbm_bytes"] == 150


def test_disjoint_extent_snapshot_only_counts_current_leaf_as_reclaimable() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)
    controller.process_runtime_event(
        RuntimeEvent(
            event_id="tool-start",
            ts_ms=4,
            kind=RuntimeEventKind.TOOL_START,
            workflow_id="workflow",
            invocation_id="root",
            context_id="ctx-root",
            attributes={"tool_family": "shell"},
        )
    )

    snapshot = controller.build_policy_input(_observation(hbm_used=300))
    extents = {
        item.extent_ids[0]: item
        for item in snapshot.physical_kv.bundles
    }
    root = extents["page:1:generation:0"]
    leaf = extents["page:2:generation:0"]

    assert root.parent_extent_id is None
    assert root.child_extent_ids == ("page:2:generation:0",)
    assert root.marginal_reclaimable_bytes == 0
    assert not root.actionable
    assert "descendant_closure" in root.blocker_codes
    assert leaf.parent_extent_id == "page:1:generation:0"
    assert leaf.lease_kind == "conditional_resume"
    assert leaf.actionable
    assert leaf.marginal_reclaimable_bytes == 200


def test_targeted_bundle_validation_matches_full_snapshot_and_live_lock() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)
    controller.process_runtime_event(
        RuntimeEvent(
            event_id="tool-start-targeted",
            ts_ms=4,
            kind=RuntimeEventKind.TOOL_START,
            workflow_id="workflow",
            invocation_id="root",
            context_id="ctx-root",
            attributes={"tool_family": "shell"},
        )
    )
    snapshot = controller.build_policy_input(_observation(hbm_used=300))
    source = next(
        item
        for item in snapshot.physical_kv.bundles
        if item.extent_ids == ("page:2:generation:0",)
    )

    current = controller.policy_snapshot_builder.page_bundle_at_safe_point(
        PageHandle(2, 0),
        now_ms=10,
    )
    assert current.to_dict() == source.to_dict()

    controller.page_index.set_engine_lock(PageHandle(2, 0), 1)
    locked = controller.policy_snapshot_builder.page_bundle_at_safe_point(
        PageHandle(2, 0),
        now_ms=11,
    )
    assert locked.generation_fingerprint == source.generation_fingerprint
    assert not locked.actionable
    assert locked.blocker_codes == ("node_locked",)


def test_snapshot_versions_change_only_with_corresponding_physical_state() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)
    first = controller.build_policy_input(_observation(ts_ms=10, hbm_used=300))
    second = controller.build_policy_input(_observation(ts_ms=11, hbm_used=300))

    assert second.physical_kv.topology_version == first.physical_kv.topology_version
    assert second.physical_kv.allocator_version == first.physical_kv.allocator_version
    assert second.snapshot_id != first.snapshot_id

    controller.page_index.set_engine_lock(PageHandle(2, 0), 1)
    third = controller.build_policy_input(_observation(ts_ms=12, hbm_used=300))
    assert third.physical_kv.topology_version == first.physical_kv.topology_version
    assert third.physical_kv.allocator_version == first.physical_kv.allocator_version + 1

    controller.page_index.set_engine_lock(PageHandle(2, 0), 0)
    controller.page_index.register_page(
        PageHandle(3, 0),
        size_bytes=50,
        residency=PhysicalResidency.CPU_ONLY,
        radix_depth=3,
        parent=PageHandle(2, 0),
    )
    fourth = controller.build_policy_input(
        _observation(ts_ms=13, hbm_used=300, host_used=50)
    )
    assert fourth.physical_kv.topology_version == first.physical_kv.topology_version + 1
    assert fourth.physical_kv.allocator_version > third.physical_kv.allocator_version


def test_residency_delta_rebuilds_only_touched_radix_ancestors() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)
    independent = PageHandle(3, 0)
    controller.page_index.register_page(
        independent,
        size_bytes=50,
        residency=PhysicalResidency.GPU_ONLY,
        radix_depth=1,
        last_access_ms=3,
    )
    controller.process_runtime_event(
        RuntimeEvent(
            event_id="incremental-tool-start",
            ts_ms=4,
            kind=RuntimeEventKind.TOOL_START,
            workflow_id="workflow",
            invocation_id="root",
            context_id="ctx-root",
            attributes={"tool_family": "shell"},
        )
    )
    first = controller.build_policy_input(_observation(hbm_used=350))
    first_by_extent = {
        item.extent_ids[0]: item for item in first.physical_kv.bundles
    }

    leaf = PageHandle(2, 0)
    controller.page_index.begin_transfer(leaf, TransferDirection.D2H)
    controller.page_index.complete_transfer(
        leaf,
        TransferDirection.D2H,
        keep_gpu=False,
    )
    second = controller.build_policy_input(
        _observation(ts_ms=11, hbm_used=150, host_used=200)
    )
    second_by_extent = {
        item.extent_ids[0]: item for item in second.physical_kv.bundles
    }

    assert not controller.policy_snapshot_builder._page_physical_state_cache.full_rebuild
    assert second_by_extent["page:1:generation:0"].actionable
    assert second_by_extent["page:2:generation:0"].gpu_bytes == 0
    assert second_by_extent["page:2:generation:0"].cpu_bytes == 200
    assert (
        second_by_extent["page:3:generation:0"]
        is first_by_extent["page:3:generation:0"]
    )


def test_lease_delta_reuses_unrelated_physical_bundles() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)
    controller.page_index.register_page(
        PageHandle(3, 0),
        size_bytes=50,
        residency=PhysicalResidency.GPU_ONLY,
        radix_depth=1,
        last_access_ms=3,
    )
    first = controller.build_policy_input(_observation(hbm_used=350))
    first_by_extent = {
        item.extent_ids[0]: item for item in first.physical_kv.bundles
    }

    controller.process_runtime_event(
        RuntimeEvent(
            event_id="lease-tool-start",
            ts_ms=11,
            kind=RuntimeEventKind.TOOL_START,
            workflow_id="workflow",
            invocation_id="root",
            context_id="ctx-root",
            attributes={"tool_family": "shell"},
        )
    )
    second = controller.build_policy_input(
        _observation(ts_ms=12, hbm_used=350)
    )
    second_by_extent = {
        item.extent_ids[0]: item for item in second.physical_kv.bundles
    }

    assert second_by_extent["page:2:generation:0"].lease_kind == "conditional_resume"
    assert second_by_extent["page:2:generation:0"].actionable
    assert (
        second_by_extent["page:3:generation:0"]
        is first_by_extent["page:3:generation:0"]
    )


def test_snapshot_rejects_page_mirror_larger_than_authoritative_allocator() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)

    with pytest.raises(PolicySnapshotError, match="exceed authoritative"):
        controller.build_policy_input(_observation(hbm_used=299))


def test_fresh_child_has_no_parent_physical_owner_or_restore_bundle() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)
    controller.process_runtime_events(
        [
            RuntimeEvent(
                event_id="child-create",
                ts_ms=5,
                kind=RuntimeEventKind.INVOCATION_CREATE,
                workflow_id="workflow",
                invocation_id="child",
                context_id="ctx-child",
                context_epoch=0,
                parent_invocation_id="root",
                parent_context_id="ctx-root",
                agent_definition_id="browser",
                agent_instance_id="browser-0",
                relation_type=RelationType.SPAWN,
                context_mode=ContextMode.FRESH,
                execution_mode=ExecutionMode.BACKGROUND,
            ),
            RuntimeEvent(
                event_id="child-spawn",
                ts_ms=6,
                kind=RuntimeEventKind.SPAWN,
                workflow_id="workflow",
                invocation_id="root",
                target_invocation_id="child",
                execution_mode=ExecutionMode.BACKGROUND,
            ),
        ]
    )
    controller.submit_request(
        AdmissionRequest(
            request_id="request-child",
            workflow_id="workflow",
            invocation_id="child",
            context_id="ctx-child",
            context_epoch=0,
            submitted_ts_ms=7,
            uncached_prompt_tokens=5,
            expected_output_tokens=5,
            kv_bytes_per_token=10,
        )
    )

    snapshot = controller.build_policy_input(_observation(hbm_used=300))

    assert snapshot.runnable_frontier[0].context_id == "ctx-child"
    assert all(
        "ctx-child" not in bundle.owner_context_ids
        for bundle in snapshot.physical_kv.bundles
    )
    child_context = snapshot.runtime_graph.state["rccg"]["contexts"]["ctx-child"]
    assert child_context["parent_context_id"] == "ctx-root"


def test_resource_observations_are_monotonic_and_growth_is_positive_only() -> None:
    controller = _controller()
    first = controller.build_policy_input(_observation(ts_ms=10, hbm_used=100))
    second = controller.build_policy_input(_observation(ts_ms=20, hbm_used=160))
    third = controller.build_policy_input(_observation(ts_ms=30, hbm_used=120))

    assert first.resources.recent_kv_growth_bytes_per_ms == 0
    assert second.resources.recent_kv_growth_bytes_per_ms == 6
    assert third.resources.recent_kv_growth_bytes_per_ms == 3
    with pytest.raises(ValueError, match="time-monotonic"):
        controller.build_policy_input(
            replace(_observation(ts_ms=29, hbm_used=120), source="test")
        )


def test_snapshot_freezes_root_workflow_service_and_memory_charges() -> None:
    controller = _controller()
    _bind_two_level_tree(controller)
    controller.fairness.charge_service("workflow", 12.5)
    controller.report_hbm_usage(
        300,
        workflow_charges={"workflow": 25.0},
    )

    snapshot = controller.build_policy_input(_observation(hbm_used=300))
    fairness = snapshot.runtime_graph.state["workflow_fairness"]
    account = fairness["accounts"]["workflow"]

    assert account["attained_service_ms"] == 12.5
    assert account["virtual_runtime_ms"] == 12.5
    assert account["dispatch_count"] == 1
    assert fairness["memory_charges_bytes"]["workflow"] == 325.0
    assert fairness["accounting_scope"] == "root_workflow"
    assert fairness["revision"] == controller.fairness.revision
    assert snapshot.runtime_graph.state["request_queue"]["admission_revision"] == 0
    assert snapshot.runtime_graph.state["control"]["transfer_epoch"] == 0
