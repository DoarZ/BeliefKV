import pytest

from beliefkv.control.controller import BeliefKVController
from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.leases import LeaseKind
from beliefkv.runtime.bundles import BundleScope, PhysicalBundleBuilder
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandKind,
    ControlCommand,
    PageHandle,
    PhysicalPageAction,
    PhysicalResidency,
    TransferBlockerCode,
)
from beliefkv.runtime.radix_arbiter import RadixArbiter


def _event(
    sequence: int,
    kind: RuntimeEventKind,
    *,
    invocation_id: str | None = None,
    context_id: str | None = None,
    workflow_id: str = "wf",
    **attributes: object,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"event-{sequence}-{workflow_id}",
        ts_ms=float(sequence),
        kind=kind,
        workflow_id=workflow_id,
        invocation_id=invocation_id,
        context_id=context_id,
        agent_definition_id=invocation_id,
        agent_instance_id=invocation_id,
        attributes=attributes,
    )


def _runtime(*contexts: tuple[str, str, str]) -> tuple[
    RuntimeCausalContextGraph, PageOwnershipIndex
]:
    graph = RuntimeCausalContextGraph()
    index = PageOwnershipIndex()
    workflows = sorted({workflow_id for _, _, workflow_id in contexts})
    sequence = 0
    for workflow_id in workflows:
        graph.apply(
            _event(sequence, RuntimeEventKind.WORKFLOW_START, workflow_id=workflow_id)
        )
        sequence += 1
    for invocation_id, context_id, workflow_id in contexts:
        graph.apply(
            _event(
                sequence,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id=invocation_id,
                context_id=context_id,
                workflow_id=workflow_id,
            )
        )
        index.register_context(context_id, workflow_id, 0)
        sequence += 1
    return graph, index


def test_shared_extent_uses_strongest_owner_and_blocks_parked_victim() -> None:
    graph, index = _runtime(
        ("parked", "ctx-parked", "wf"),
        ("running", "ctx-running", "wf"),
    )
    graph.apply(_event(3, RuntimeEventKind.TOOL_START, invocation_id="parked"))
    graph.apply(_event(4, RuntimeEventKind.LLM_SUBMIT, invocation_id="running"))
    shared = PageHandle(1, 0)
    index.register_page(shared, size_bytes=1024)
    index.bind_pages("ctx-parked", 0, (shared,))
    index.bind_pages("ctx-running", 0, (shared,))

    preview = PhysicalBundleBuilder(graph, index).previews_for_context(
        CommandKind.OFFLOAD_CONTEXT,
        "ctx-parked",
        0,
        now_ms=5,
    )[0]

    assert not preview.eligible
    assert preview.bundle.lease.strongest_kind == LeaseKind.RUNNING
    assert preview.bundle.owner_context_ids == ("ctx-parked", "ctx-running")
    assert {item.code for item in preview.blockers} == {
        TransferBlockerCode.ENGINE_BUSY
    }


def test_locked_large_extent_does_not_hide_independent_reclaimable_bundle() -> None:
    graph, index = _runtime(("parked", "ctx", "wf"))
    graph.apply(_event(2, RuntimeEventKind.TOOL_START, invocation_id="parked"))
    locked = PageHandle(1, 0)
    free = PageHandle(2, 0)
    index.register_page(locked, size_bytes=1000)
    index.register_page(free, size_bytes=500)
    index.bind_pages("ctx", 0, (locked, free))
    index.pages[locked].engine_lock_ref = 1

    previews = PhysicalBundleBuilder(graph, index).previews_for_context(
        CommandKind.OFFLOAD_CONTEXT,
        "ctx",
        0,
        now_ms=3,
    )
    by_handles = {item.bundle.handles: item for item in previews}

    assert not by_handles[(locked,)].eligible
    assert by_handles[(locked,)].bundle.locked_bytes == 1000
    assert by_handles[(locked,)].bundle.marginal_reclaimable_bytes == 0
    assert by_handles[(free,)].eligible
    assert by_handles[(free,)].bundle.marginal_reclaimable_bytes == 500


def test_preview_reports_host_capacity_without_erasing_required_bytes() -> None:
    graph, index = _runtime(("parked", "ctx", "wf"))
    graph.apply(_event(2, RuntimeEventKind.TOOL_START, invocation_id="parked"))
    handle = PageHandle(1, 0)
    index.register_page(handle, size_bytes=500)
    index.bind_pages("ctx", 0, (handle,))

    preview = PhysicalBundleBuilder(graph, index).previews_for_context(
        CommandKind.OFFLOAD_CONTEXT,
        "ctx",
        0,
        now_ms=3,
        host_available_bytes=100,
    )[0]

    assert not preview.eligible
    assert preview.bundle.closure_bytes == 500
    assert preview.copy_bytes == 500
    assert preview.bundle.locked_bytes == 0
    assert {item.code for item in preview.blockers} == {
        TransferBlockerCode.HOST_CAPACITY
    }
    with pytest.raises(ValueError, match="blocked physical bundle"):
        preview.intent()


def test_shared_physical_bytes_are_counted_once_in_a_bundle() -> None:
    graph, index = _runtime(
        ("first", "ctx-first", "wf"),
        ("second", "ctx-second", "wf"),
    )
    graph.apply(_event(3, RuntimeEventKind.TOOL_START, invocation_id="first"))
    graph.apply(_event(4, RuntimeEventKind.TOOL_START, invocation_id="second"))
    shared = PageHandle(1, 0)
    index.register_page(shared, size_bytes=4096)
    index.bind_pages("ctx-first", 0, (shared,))
    index.bind_pages("ctx-second", 0, (shared,))

    preview = PhysicalBundleBuilder(graph, index).previews_for_context(
        CommandKind.OFFLOAD_CONTEXT,
        "ctx-first",
        0,
        now_ms=5,
    )[0]

    assert preview.eligible
    assert preview.bundle.closure_bytes == 4096
    assert preview.bundle.marginal_reclaimable_bytes == 4096
    assert preview.copy_bytes == 4096
    assert preview.bundle.scope == BundleScope.SHARED_SUBTREE
    assert preview.bundle.exclusive_action_bytes == 0
    assert preview.bundle.cross_context_action_bytes == 4096
    assert preview.bundle.foreign_owner_context_ids == ("ctx-second",)


def test_parent_private_suffix_and_shared_root_have_distinct_scopes() -> None:
    graph, index = _runtime(
        ("parent", "ctx-parent", "wf"),
        ("child", "ctx-child", "wf"),
    )
    graph.apply(_event(3, RuntimeEventKind.TOOL_START, invocation_id="parent"))
    graph.apply(_event(4, RuntimeEventKind.TOOL_START, invocation_id="child"))
    shared_root = PageHandle(1, 0)
    parent_suffix = PageHandle(2, 0)
    child_suffix = PageHandle(3, 0)
    index.register_page(shared_root, size_bytes=100, radix_depth=1)
    index.register_page(
        parent_suffix,
        size_bytes=300,
        radix_depth=2,
        parent=shared_root,
    )
    index.register_page(
        child_suffix,
        size_bytes=400,
        radix_depth=2,
        parent=shared_root,
    )
    index.bind_pages("ctx-parent", 0, (shared_root, parent_suffix))
    index.bind_pages("ctx-child", 0, (shared_root, child_suffix))

    previews = PhysicalBundleBuilder(graph, index).previews_for_context(
        CommandKind.OFFLOAD_CONTEXT,
        "ctx-parent",
        0,
        now_ms=5,
    )
    by_handles = {item.bundle.handles: item for item in previews}

    private = by_handles[(parent_suffix,)].bundle
    assert private.scope == BundleScope.EXCLUSIVE_SUFFIX
    assert private.exclusive_action_bytes == 300
    assert private.cross_context_action_bytes == 0
    assert private.foreign_owner_context_ids == ()

    shared = by_handles[(shared_root, parent_suffix, child_suffix)].bundle
    assert shared.scope == BundleScope.SHARED_SUBTREE
    assert shared.exclusive_action_bytes == 300
    assert shared.cross_context_action_bytes == 500
    assert shared.foreign_owner_context_ids == ("ctx-child",)


def test_h2d_bundle_contains_cpu_ancestor_and_orders_actions_top_down() -> None:
    graph, index = _runtime(("agent", "ctx", "wf"))
    parent = PageHandle(1, 0)
    child = PageHandle(2, 0)
    index.register_page(
        parent,
        size_bytes=300,
        residency=PhysicalResidency.CPU_ONLY,
        radix_depth=1,
    )
    index.register_page(
        child,
        size_bytes=500,
        residency=PhysicalResidency.CPU_ONLY,
        radix_depth=2,
        parent=parent,
    )
    index.bind_pages("ctx", 0, (child,))

    preview = PhysicalBundleBuilder(graph, index).previews_for_context(
        CommandKind.PREFETCH_CONTEXT,
        "ctx",
        0,
        now_ms=3,
    )[0]

    assert preview.eligible
    assert preview.bundle.handles == (parent, child)
    assert [item.handle for item in preview.page_actions] == [parent, child]
    assert all(
        item.action == PhysicalPageAction.START_H2D
        for item in preview.page_actions
    )
    assert preview.bundle.closure_bytes == 800


def test_h2d_gpu_anchor_lock_is_not_a_transfer_blocker() -> None:
    graph, index = _runtime(("agent", "ctx", "wf"))
    anchor = PageHandle(1, 0)
    target = PageHandle(2, 0)
    index.register_page(anchor, size_bytes=300, radix_depth=1)
    index.register_page(
        target,
        size_bytes=500,
        residency=PhysicalResidency.CPU_ONLY,
        radix_depth=2,
        parent=anchor,
    )
    index.bind_pages("ctx", 0, (target,))
    index.pages[anchor].engine_lock_ref = 1

    preview = PhysicalBundleBuilder(graph, index).previews_for_context(
        CommandKind.PREFETCH_CONTEXT,
        "ctx",
        0,
        now_ms=3,
    )[0]

    assert preview.eligible
    assert preview.bundle.handles == (anchor, target)
    assert [item.handle for item in preview.page_actions] == [target]
    assert preview.bundle.scope == BundleScope.EXCLUSIVE_SUFFIX
    assert preview.bundle.exclusive_action_bytes == 500
    assert preview.bundle.cross_context_action_bytes == 0


def test_arbiter_rejects_bundle_when_authoritative_lock_changes() -> None:
    graph, index = _runtime(("agent", "ctx", "wf"))
    handle = PageHandle(1, 0)
    index.register_page(
        handle,
        size_bytes=500,
        residency=PhysicalResidency.CPU_ONLY,
    )
    index.bind_pages("ctx", 0, (handle,))
    builder = PhysicalBundleBuilder(graph, index)
    preview = builder.previews_for_context(
        CommandKind.PREFETCH_CONTEXT,
        "ctx",
        0,
        now_ms=3,
    )[0]
    command = ControlCommand(
        command_id="stale-bundle",
        kind=CommandKind.PREFETCH_CONTEXT,
        created_ts_ms=3,
        context_id="ctx",
        context_epoch=0,
        target_bytes=500,
        physical_bundle=preview.intent(),
    )
    index.pages[handle].engine_lock_ref = 1

    resolved = RadixArbiter(graph, index, bundle_builder=builder).resolve(command)

    assert not resolved.page_actions
    assert resolved.reason == "physical_bundle_fingerprint_changed"
    assert {item.code for item in resolved.blockers} == {
        TransferBlockerCode.NODE_LOCKED
    }


def test_arbiter_rejects_exclusive_intent_when_foreign_owner_appears() -> None:
    graph, index = _runtime(
        ("parent", "ctx-parent", "wf"),
        ("child", "ctx-child", "wf"),
    )
    graph.apply(_event(3, RuntimeEventKind.TOOL_START, invocation_id="parent"))
    graph.apply(_event(4, RuntimeEventKind.TOOL_START, invocation_id="child"))
    handle = PageHandle(1, 0)
    index.register_page(handle, size_bytes=500)
    index.bind_pages("ctx-parent", 0, (handle,))
    builder = PhysicalBundleBuilder(graph, index)
    preview = builder.previews_for_context(
        CommandKind.OFFLOAD_CONTEXT,
        "ctx-parent",
        0,
        now_ms=5,
    )[0]
    assert preview.bundle.scope == BundleScope.EXCLUSIVE_SUFFIX
    command = ControlCommand(
        command_id="stale-exclusive-scope",
        kind=CommandKind.OFFLOAD_CONTEXT,
        created_ts_ms=5,
        context_id="ctx-parent",
        context_epoch=0,
        target_bytes=500,
        physical_bundle=preview.intent(),
    )

    index.bind_pages("ctx-child", 0, (handle,))
    resolved = RadixArbiter(graph, index, bundle_builder=builder).resolve(command)

    assert not resolved.page_actions
    assert resolved.reason == "physical_bundle_fingerprint_changed"


def test_pressure_planner_selects_reclaimable_extent_not_locked_context_total() -> None:
    controller = BeliefKVController(
        BeliefKVConfig(
            hbm_capacity_bytes=2000,
            host_capacity_bytes=10_000,
            reserve_hbm_bytes=1000,
            urgent_chunk_bytes=2000,
            predictor_enabled=False,
            shadow_enabled=False,
        )
    )
    controller.process_runtime_events(
        (
            _event(0, RuntimeEventKind.WORKFLOW_START),
            _event(
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="parked",
                context_id="ctx",
            ),
            _event(2, RuntimeEventKind.TOOL_START, invocation_id="parked"),
        )
    )
    locked = PageHandle(1, 0)
    free = PageHandle(2, 0)
    controller.page_index.register_page(locked, size_bytes=1000)
    controller.page_index.register_page(free, size_bytes=500)
    controller.page_index.bind_pages("ctx", 0, (locked, free))
    controller.page_index.pages[locked].engine_lock_ref = 1

    tick = controller.tick(3)

    assert tick.transfer is not None
    assert tick.transfer.command.physical_bundle is not None
    assert [item.handle for item in tick.transfer.page_actions] == [free]
    assert tick.transfer.command.physical_bundle.expected_reclaimable_bytes == 500
    assert {item.kind for item in tick.bundle_preview_events} >= {
        "context_lease_issued",
        "bundle_lease_aggregated",
        "physical_bundle_preview",
    }


def test_pressure_prefers_private_suffix_over_larger_shared_subtree() -> None:
    controller = BeliefKVController(
        BeliefKVConfig(
            hbm_capacity_bytes=1200,
            host_capacity_bytes=10_000,
            reserve_hbm_bytes=800,
            urgent_chunk_bytes=2000,
            predictor_enabled=False,
            shadow_enabled=False,
        )
    )
    controller.process_runtime_events(
        (
            _event(0, RuntimeEventKind.WORKFLOW_START),
            _event(
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="parent",
                context_id="ctx-parent",
            ),
            _event(
                2,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="child",
                context_id="ctx-child",
            ),
            _event(3, RuntimeEventKind.TOOL_START, invocation_id="parent"),
            _event(4, RuntimeEventKind.TOOL_START, invocation_id="child"),
        )
    )
    shared_root = PageHandle(1, 0)
    parent_suffix = PageHandle(2, 0)
    child_suffix = PageHandle(3, 0)
    controller.page_index.register_page(shared_root, size_bytes=100, radix_depth=1)
    controller.page_index.register_page(
        parent_suffix,
        size_bytes=300,
        radix_depth=2,
        parent=shared_root,
    )
    controller.page_index.register_page(
        child_suffix,
        size_bytes=600,
        radix_depth=2,
        parent=shared_root,
    )
    controller.page_index.bind_pages(
        "ctx-parent", 0, (shared_root, parent_suffix)
    )
    controller.page_index.bind_pages(
        "ctx-child", 0, (shared_root, child_suffix)
    )

    tick = controller.tick(5)

    assert tick.transfer is not None
    assert [item.handle for item in tick.transfer.page_actions] == [child_suffix]
    assert (
        tick.transfer.command.metadata["physical_bundle_scope"]
        == BundleScope.EXCLUSIVE_SUFFIX.value
    )
    assert (
        tick.transfer.command.metadata["reason"]
        == "context_exclusive_suffix_offload"
    )
    assert tick.transfer.command.metadata["physical_cross_context_action_bytes"] == 0


def test_pressure_uses_shared_bundle_only_as_global_fallback() -> None:
    controller = BeliefKVController(
        BeliefKVConfig(
            hbm_capacity_bytes=1000,
            host_capacity_bytes=10_000,
            reserve_hbm_bytes=800,
            urgent_chunk_bytes=1000,
            predictor_enabled=False,
            shadow_enabled=False,
        )
    )
    controller.process_runtime_events(
        (
            _event(0, RuntimeEventKind.WORKFLOW_START),
            _event(
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="first",
                context_id="ctx-first",
            ),
            _event(
                2,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="second",
                context_id="ctx-second",
            ),
            _event(3, RuntimeEventKind.TOOL_START, invocation_id="first"),
            _event(4, RuntimeEventKind.TOOL_START, invocation_id="second"),
        )
    )
    shared = PageHandle(1, 0)
    controller.page_index.register_page(shared, size_bytes=500)
    controller.page_index.bind_pages("ctx-first", 0, (shared,))
    controller.page_index.bind_pages("ctx-second", 0, (shared,))

    tick = controller.tick(5)

    assert tick.transfer is not None
    assert [item.handle for item in tick.transfer.page_actions] == [shared]
    assert (
        tick.transfer.command.metadata["physical_bundle_scope"]
        == BundleScope.SHARED_SUBTREE.value
    )
    assert tick.transfer.command.metadata["reason"] == "global_shared_bundle_reclaim"


def test_predictive_shadow_only_materializes_private_suffix() -> None:
    controller = BeliefKVController(
        BeliefKVConfig(
            hbm_capacity_bytes=5000,
            host_capacity_bytes=10_000,
            reserve_hbm_bytes=0,
            urgent_chunk_bytes=1000,
            shadow_chunk_bytes=1000,
            shadow_min_parked_ms=0,
            predictor_enabled=False,
            prefetch_enabled=False,
            shadow_enabled=True,
        )
    )
    controller.process_runtime_events(
        (
            _event(0, RuntimeEventKind.WORKFLOW_START),
            _event(
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="parent",
                context_id="ctx-parent",
            ),
            _event(
                2,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="child",
                context_id="ctx-child",
            ),
            _event(3, RuntimeEventKind.TOOL_START, invocation_id="parent"),
            _event(4, RuntimeEventKind.TOOL_START, invocation_id="child"),
        )
    )
    shared_root = PageHandle(1, 0)
    parent_suffix = PageHandle(2, 0)
    controller.page_index.register_page(shared_root, size_bytes=100, radix_depth=1)
    controller.page_index.register_page(
        parent_suffix,
        size_bytes=300,
        radix_depth=2,
        parent=shared_root,
    )
    controller.page_index.bind_pages(
        "ctx-parent", 0, (shared_root, parent_suffix)
    )
    controller.page_index.bind_pages("ctx-child", 0, (shared_root,))

    tick = controller.tick(5)

    assert tick.transfer is not None
    assert tick.transfer.command.kind == CommandKind.SHADOW_CONTEXT
    assert [item.handle for item in tick.transfer.page_actions] == [parent_suffix]
    assert (
        tick.transfer.command.metadata["physical_bundle_scope"]
        == BundleScope.EXCLUSIVE_SUFFIX.value
    )
    assert tick.transfer.command.metadata["physical_cross_context_action_bytes"] == 0
