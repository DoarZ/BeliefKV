import unittest

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.runtime.command_queue import TransferCommandQueue
from beliefkv.runtime.page_index import PageIndexError, PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandKind,
    CommandQueueClass,
    ControlCommand,
    PageHandle,
    PhysicalPageAction,
    PhysicalResidency,
    TransferBlockerCode,
    TransferDirection,
)
from beliefkv.runtime.radix_arbiter import RadixArbiter


def runtime_event(sequence: int, kind: RuntimeEventKind, **kwargs) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"e{sequence}",
        ts_ms=float(sequence),
        kind=kind,
        workflow_id=kwargs.pop("workflow_id", "wf"),
        **kwargs,
    )


class PageIndexTest(unittest.TestCase):
    def test_revision_changes_only_when_mirrored_state_changes(self):
        index = PageOwnershipIndex()
        index.register_context("ctx", "wf", 0)
        handle = PageHandle(1, 0)
        index.register_page(handle, size_bytes=100)
        index.bind_pages("ctx", 0, [handle])
        initial = index.revision

        index.set_engine_lock(handle, 0)
        index.update_runtime_state(
            handle,
            residency=PhysicalResidency.GPU_ONLY,
            radix_depth=0,
            last_access_ms=0,
        )
        self.assertEqual(index.revision, initial)

        index.set_engine_lock(handle, 1)
        self.assertEqual(index.revision, initial + 1)
        index.update_runtime_state(handle, last_access_ms=2)
        self.assertEqual(index.revision, initial + 2)

    def test_page_generation_prevents_stale_reuse(self):
        index = PageOwnershipIndex()
        first = PageHandle(1, 0)
        index.register_page(first, size_bytes=4096)
        with self.assertRaises(PageIndexError):
            index.register_page(PageHandle(1, 1), size_bytes=4096)
        index.free_page(first)
        second = PageHandle(1, 1)
        index.register_page(second, size_bytes=4096)
        with self.assertRaises(PageIndexError):
            index.require_page(first)

    def test_shared_page_is_charged_once_and_split_across_workflows(self):
        index = PageOwnershipIndex()
        index.register_context("a", "wf-a", 0)
        index.register_context("b", "wf-b", 0)
        handle = PageHandle(1, 0)
        index.register_page(handle, size_bytes=100)
        index.bind_pages("a", 0, [handle])
        index.bind_pages("b", 0, [handle])
        self.assertEqual(index.gpu_bytes, 100)
        self.assertEqual(index.workflow_gpu_charges(), {"wf-a": 50.0, "wf-b": 50.0})
        index.assert_consistent()

    def test_prepare_commit_updates_residency_only_at_completion(self):
        index = PageOwnershipIndex()
        handle = PageHandle(1, 0)
        index.register_page(handle, size_bytes=4096)
        index.begin_transfer(handle, TransferDirection.D2H)
        self.assertEqual(index.pages[handle].residency, PhysicalResidency.MIRRORING)
        index.complete_transfer(handle, TransferDirection.D2H, keep_gpu=True)
        self.assertEqual(index.pages[handle].residency, PhysicalResidency.DUAL_CLEAN)
        index.commit_cpu(handle)
        self.assertEqual(index.pages[handle].residency, PhysicalResidency.CPU_ONLY)
        index.commit_cpu(handle)
        self.assertEqual(index.pages[handle].residency, PhysicalResidency.CPU_ONLY)

    def test_physical_breakdown_is_closure_aware_and_revision_cached(self):
        index = PageOwnershipIndex()
        parent = PageHandle(1, 0)
        child = PageHandle(2, 0)
        index.register_page(parent, size_bytes=100, radix_depth=1)
        index.register_page(
            child,
            size_bytes=200,
            radix_depth=2,
            parent=parent,
            residency=PhysicalResidency.DUAL_CLEAN,
        )
        index.set_engine_lock(child, 1)

        blocked = index.physical_kv_state_breakdown()

        self.assertEqual(blocked.gpu_bytes, 300)
        self.assertEqual(blocked.cpu_bytes, 200)
        self.assertEqual(blocked.engine_locked_bytes, 200)
        self.assertEqual(blocked.closure_blocked_bytes, 100)
        self.assertEqual(blocked.migratable_bytes, 0)
        self.assertEqual(blocked.dual_resident_bytes, 200)
        locked_pages = index.engine_locked_gpu_pages()
        self.assertEqual(
            tuple(page.handle for page in locked_pages),
            (child,),
        )

        index.update_runtime_state(parent, last_access_ms=2)
        self.assertIs(index.physical_kv_state_breakdown(), blocked)
        self.assertIs(index.engine_locked_gpu_pages(), locked_pages)

        index.set_engine_lock(child, 0)
        unblocked = index.physical_kv_state_breakdown()
        self.assertIsNot(unblocked, blocked)
        self.assertEqual(unblocked.engine_locked_bytes, 0)
        self.assertEqual(unblocked.closure_blocked_bytes, 0)
        self.assertEqual(unblocked.migratable_bytes, 300)
        self.assertEqual(index.engine_locked_gpu_pages(), ())

        index.set_active_readers(child, 1)
        reader_locked = index.physical_kv_state_breakdown()
        self.assertEqual(reader_locked.engine_locked_bytes, 200)
        self.assertEqual(index.engine_locked_gpu_pages(), ())

    def test_tentative_unlock_preview_is_read_only_and_closure_aware(self):
        index = PageOwnershipIndex()
        parent = PageHandle(1, 0)
        child = PageHandle(2, 0)
        index.register_page(parent, size_bytes=100, radix_depth=1)
        index.register_page(
            child,
            size_bytes=200,
            radix_depth=2,
            parent=parent,
        )
        index.set_engine_lock(child, 1)
        baseline = index.physical_kv_state_breakdown()
        revision = index.revision
        topology_revision = index.topology_revision

        preview = index.preview_engine_lock_release({child: 0})

        self.assertEqual(preview.baseline.migratable_bytes, 0)
        self.assertEqual(preview.projected.migratable_bytes, 300)
        self.assertEqual(preview.lock_ref_zeroed_handles, (child,))
        self.assertEqual(preview.lock_ref_zeroed_bytes, 200)
        self.assertEqual(
            preview.newly_migratable_handles,
            (parent, child),
        )
        self.assertEqual(preview.newly_migratable_bytes, 300)
        self.assertEqual(index.pages[child].engine_lock_ref, 1)
        self.assertEqual(index.revision, revision)
        self.assertEqual(index.topology_revision, topology_revision)
        self.assertIs(index.physical_kv_state_breakdown(), baseline)

    def test_tentative_unlock_respects_non_lock_physical_blockers(self):
        index = PageOwnershipIndex()
        parent = PageHandle(1, 0)
        child = PageHandle(2, 0)
        index.register_page(parent, size_bytes=100, radix_depth=1)
        index.register_page(
            child,
            size_bytes=200,
            radix_depth=2,
            parent=parent,
        )
        index.set_engine_lock(child, 1)
        index.set_active_readers(child, 1)

        preview = index.preview_engine_lock_release({child: 0})

        self.assertEqual(preview.lock_ref_zeroed_bytes, 200)
        self.assertEqual(preview.newly_migratable_bytes, 0)
        self.assertEqual(preview.newly_migratable_handles, ())
        self.assertEqual(preview.projected.engine_locked_bytes, 200)

    def test_unbinding_context_invalidates_semantic_pin_breakdown(self):
        index = PageOwnershipIndex()
        index.register_context("ctx", "wf", 0)
        handle = PageHandle(1, 0)
        index.register_page(handle, size_bytes=100)
        index.bind_pages("ctx", 0, [handle])
        index.pin_context("ctx")
        self.assertEqual(
            index.physical_kv_state_breakdown().migratable_bytes,
            0,
        )

        index.unbind_context("ctx")

        self.assertEqual(
            index.physical_kv_state_breakdown().migratable_bytes,
            100,
        )

    def test_reparent_updates_both_sides_of_radix_edge(self):
        index = PageOwnershipIndex()
        old_parent = PageHandle(1, 0)
        new_parent = PageHandle(2, 0)
        child = PageHandle(3, 0)
        index.register_page(old_parent, size_bytes=100)
        index.register_page(new_parent, size_bytes=100)
        index.register_page(child, size_bytes=100, parent=old_parent)

        index.set_parent(child, new_parent)

        self.assertNotIn(child, index.pages[old_parent].children)
        self.assertIn(child, index.pages[new_parent].children)
        self.assertEqual(index.pages[child].parent, new_parent)
        index.assert_consistent()

    def test_reparent_rejects_radix_cycle(self):
        index = PageOwnershipIndex()
        parent = PageHandle(1, 0)
        child = PageHandle(2, 0)
        index.register_page(parent, size_bytes=100)
        index.register_page(child, size_bytes=100, parent=parent)

        with self.assertRaisesRegex(PageIndexError, "cycle"):
            index.set_parent(parent, child)

    def test_urgent_command_preempts_shadow(self):
        queue = TransferCommandQueue()
        shadow = ControlCommand(
            command_id="shadow",
            kind=CommandKind.SHADOW_CONTEXT,
            created_ts_ms=0,
            queue_class=CommandQueueClass.SHADOW,
        )
        urgent = ControlCommand(
            command_id="urgent",
            kind=CommandKind.OFFLOAD_CONTEXT,
            created_ts_ms=1,
            queue_class=CommandQueueClass.URGENT,
        )
        queue.put(shadow)
        queue.put(urgent)
        self.assertEqual(queue.pop().command_id, "urgent")
        self.assertEqual(queue.pop().command_id, "shadow")


class RadixArbiterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = RuntimeCausalContextGraph()
        self.graph.apply(runtime_event(0, RuntimeEventKind.WORKFLOW_START))
        self.index = PageOwnershipIndex()

    def create_invocation(
        self, sequence: int, invocation_id: str, context_id: str, workflow_id: str = "wf"
    ) -> None:
        if workflow_id not in self.graph.workflows:
            self.graph.apply(
                runtime_event(
                    sequence - 0.5,
                    RuntimeEventKind.WORKFLOW_START,
                    workflow_id=workflow_id,
                )
            )
        self.graph.apply(
            runtime_event(
                sequence,
                RuntimeEventKind.INVOCATION_CREATE,
                workflow_id=workflow_id,
                invocation_id=invocation_id,
                context_id=context_id,
                context_epoch=0,
            )
        )
        self.index.register_context(context_id, workflow_id, 0)

    def park_with_tool(self, sequence: int, invocation_id: str) -> None:
        self.graph.apply(
            runtime_event(
                sequence,
                RuntimeEventKind.TOOL_START,
                workflow_id=self.graph.invocations[invocation_id].workflow_id,
                invocation_id=invocation_id,
            )
        )

    def command(self, kind: CommandKind, context_id: str, target_bytes: int = 1 << 30):
        return ControlCommand(
            command_id=f"{kind.value}-{context_id}",
            kind=kind,
            created_ts_ms=10,
            context_id=context_id,
            context_epoch=0,
            target_bytes=target_bytes,
        )

    def test_active_shared_owner_protects_page(self):
        self.create_invocation(1, "parent", "ctx-parent")
        self.create_invocation(2, "child", "ctx-child")
        self.park_with_tool(3, "parent")
        handle = PageHandle(1, 0)
        self.index.register_page(handle, size_bytes=4096)
        self.index.bind_pages("ctx-parent", 0, [handle])
        self.index.bind_pages("ctx-child", 0, [handle])

        resolved = RadixArbiter(self.graph, self.index).resolve(
            self.command(CommandKind.OFFLOAD_CONTEXT, "ctx-parent")
        )
        self.assertFalse(resolved.page_actions)
        self.assertEqual(resolved.reason, "no_migratable_marginal_pages")
        self.assertIn(
            TransferBlockerCode.ENGINE_BUSY,
            {item.code for item in resolved.blockers},
        )

    def test_private_suffix_is_selected_but_active_shared_prefix_is_not(self):
        self.create_invocation(1, "parent", "ctx-parent")
        self.create_invocation(2, "child", "ctx-child")
        self.park_with_tool(3, "parent")
        prefix = PageHandle(1, 0)
        suffix = PageHandle(2, 0)
        self.index.register_page(prefix, size_bytes=4096, radix_depth=1)
        self.index.register_page(suffix, size_bytes=8192, radix_depth=2, parent=prefix)
        self.index.bind_pages("ctx-parent", 0, [prefix, suffix])
        self.index.bind_pages("ctx-child", 0, [prefix])

        resolved = RadixArbiter(self.graph, self.index).resolve(
            self.command(CommandKind.OFFLOAD_CONTEXT, "ctx-parent")
        )
        self.assertEqual(len(resolved.page_actions), 1)
        self.assertEqual(resolved.page_actions[0].handle, suffix)
        self.assertEqual(resolved.resolved_bytes, 8192)

    def test_liveness_spill_moves_only_ready_private_suffix(self):
        self.create_invocation(1, "target", "ctx-target", "target-wf")
        self.create_invocation(2, "victim", "ctx-victim", "victim-wf")
        prefix = PageHandle(1, 0)
        suffix = PageHandle(2, 0)
        self.index.register_page(prefix, size_bytes=4096, radix_depth=1)
        self.index.register_page(
            suffix, size_bytes=8192, radix_depth=2, parent=prefix
        )
        self.index.bind_pages("ctx-target", 0, [prefix])
        self.index.bind_pages("ctx-victim", 0, [prefix, suffix])
        command = ControlCommand(
            command_id="liveness-spill",
            kind=CommandKind.OFFLOAD_CONTEXT,
            created_ts_ms=10,
            context_id="ctx-victim",
            context_epoch=0,
            target_bytes=1 << 30,
            metadata={
                "allow_ready_owners": True,
                "protected_context_id": "ctx-target",
            },
        )

        resolved = RadixArbiter(self.graph, self.index).resolve(command)

        self.assertEqual([item.handle for item in resolved.page_actions], [suffix])

    def test_offload_checks_transitive_gpu_descendant_closure(self):
        self.create_invocation(1, "parent", "ctx-parent")
        self.park_with_tool(2, "parent")
        prefix = PageHandle(1, 0)
        middle = PageHandle(2, 0)
        suffix = PageHandle(3, 0)
        self.index.register_page(prefix, size_bytes=100, radix_depth=1)
        self.index.register_page(
            middle,
            size_bytes=100,
            residency=PhysicalResidency.CPU_ONLY,
            radix_depth=2,
            parent=prefix,
        )
        self.index.register_page(
            suffix, size_bytes=100, radix_depth=3, parent=middle
        )
        self.index.bind_pages("ctx-parent", 0, [prefix])

        resolved = RadixArbiter(self.graph, self.index).resolve(
            self.command(CommandKind.OFFLOAD_CONTEXT, "ctx-parent")
        )

        self.assertFalse(resolved.page_actions)
        self.assertEqual(resolved.reason, "no_migratable_marginal_pages")

    def test_shadow_uses_transfer_and_reactive_uses_clean_commit(self):
        self.create_invocation(1, "parent", "ctx-parent")
        self.park_with_tool(2, "parent")
        handle = PageHandle(1, 0)
        self.index.register_page(handle, size_bytes=4096)
        self.index.bind_pages("ctx-parent", 0, [handle])
        arbiter = RadixArbiter(self.graph, self.index)
        shadow = arbiter.resolve(
            self.command(CommandKind.SHADOW_CONTEXT, "ctx-parent")
        )
        self.assertEqual(shadow.page_actions[0].action, PhysicalPageAction.START_D2H)
        self.index.begin_transfer(handle, TransferDirection.D2H)
        self.index.complete_transfer(handle, TransferDirection.D2H, keep_gpu=True)
        commit = arbiter.resolve(
            self.command(CommandKind.OFFLOAD_CONTEXT, "ctx-parent")
        )
        self.assertEqual(commit.page_actions[0].action, PhysicalPageAction.COMMIT_CPU)

    def test_stale_context_command_is_rejected(self):
        self.create_invocation(1, "parent", "ctx-parent")
        command = self.command(CommandKind.OFFLOAD_CONTEXT, "ctx-parent")
        self.index.update_context_epoch("ctx-parent", 1)
        resolved = RadixArbiter(self.graph, self.index).resolve(command)
        self.assertEqual(resolved.reason, "stale_context_epoch")

    def test_prefetch_requires_cpu_ancestor_closure(self):
        self.create_invocation(1, "parent", "ctx-parent")
        prefix = PageHandle(1, 0)
        suffix = PageHandle(2, 0)
        self.index.register_page(
            prefix,
            size_bytes=100,
            residency=PhysicalResidency.CPU_ONLY,
            radix_depth=1,
        )
        self.index.register_page(
            suffix,
            size_bytes=100,
            residency=PhysicalResidency.CPU_ONLY,
            radix_depth=2,
            parent=prefix,
        )
        self.index.bind_pages("ctx-parent", 0, [suffix])
        arbiter = RadixArbiter(self.graph, self.index)
        rejected = arbiter.resolve(
            self.command(CommandKind.PREFETCH_CONTEXT, "ctx-parent", 200)
        )
        self.assertEqual(rejected.reason, "no_cpu_pages")
        self.assertEqual(
            {item.code for item in rejected.blockers},
            {TransferBlockerCode.ANCESTOR_CLOSURE},
        )

        self.index.bind_pages("ctx-parent", 0, [prefix])
        resolved = arbiter.resolve(
            self.command(CommandKind.PREFETCH_CONTEXT, "ctx-parent", 200)
        )
        self.assertEqual(
            [item.handle for item in resolved.page_actions], [prefix, suffix]
        )

    def test_revision_journal_reports_scoped_components_without_draining(self):
        self.index.register_context("ctx", "wf", 0)
        context_revision = self.index.revision
        parent = PageHandle(1, 0)
        child = PageHandle(2, 0)
        self.index.register_page(parent, size_bytes=100)
        self.index.register_page(child, size_bytes=100, parent=parent)
        self.index.bind_pages("ctx", 0, [parent, child])
        before_lock = self.index.revision
        self.index.set_engine_lock(child, 1)

        lock_delta = self.index.changes_since(before_lock)
        self.assertEqual(lock_delta.handles, {child})
        self.assertEqual(lock_delta.components, {"lock"})
        self.assertFalse(lock_delta.full_rebuild_required)

        full_delta = self.index.changes_since(context_revision)
        self.assertIn(parent, full_delta.handles)
        self.assertIn(child, full_delta.handles)
        self.assertIn("ctx", full_delta.context_ids)
        self.assertIn("topology", full_delta.components)
        self.assertIn("owner", full_delta.components)
        self.assertIn("lock", full_delta.components)

        repeated = self.index.changes_since(before_lock)
        self.assertEqual(repeated, lock_delta)

    def test_replica_delta_is_detached_and_applies_incrementally(self):
        source = PageOwnershipIndex()
        source.register_context("ctx", "wf", 0)
        root = PageHandle(1, 0)
        source.register_page(root, size_bytes=100)
        source.bind_pages("ctx", 0, (root,))

        initial = source.replica_delta_since(0)
        replica = PageOwnershipIndex()
        replica.apply_replica_delta(initial)
        self.assertEqual(replica.revision, source.revision)
        self.assertEqual(replica.require_page(root).owner_contexts, {"ctx": 0})
        mirrored_page = replica.require_page(root)

        source.set_engine_lock(root, 2)
        incremental = source.replica_delta_since(initial.to_revision)
        self.assertEqual(incremental.pages, ())
        self.assertEqual(len(incremental.page_states), 1)
        self.assertEqual(incremental.contexts, ())
        source.set_engine_lock(root, 3)

        replica.apply_replica_delta(incremental)
        self.assertIs(replica.require_page(root), mirrored_page)
        self.assertEqual(replica.require_page(root).engine_lock_ref, 2)
        self.assertEqual(source.require_page(root).engine_lock_ref, 3)
        changes = replica.changes_since(initial.to_revision)
        self.assertEqual(changes.handles, {root})
        self.assertEqual(changes.context_ids, set())
        self.assertEqual(changes.components, {"lock"})

        residency_revision = source.revision
        source.begin_transfer(root, TransferDirection.D2H)
        source.complete_transfer(root, TransferDirection.D2H, keep_gpu=False)
        residency = source.replica_delta_since(residency_revision)
        self.assertEqual(residency.contexts, ())
        replica.apply_replica_delta(
            source.replica_delta_since(incremental.to_revision),
            full_validation=False,
        )
        self.assertEqual(replica.gpu_bytes, 0)
        self.assertEqual(replica.cpu_bytes, 100)

    def test_replica_mirror_preserves_coalesced_revision_coverage(self):
        source = PageOwnershipIndex()
        root = PageHandle(1, 0)
        source.register_page(root, size_bytes=100)
        initial = source.replica_delta_since(0)
        replica = PageOwnershipIndex()
        replica.apply_replica_delta(initial)
        before = source.revision

        for value in range(1, 101):
            source.set_engine_lock(root, value % 2)
        incremental = source.replica_delta_since(before)
        self.assertGreater(incremental.to_revision - incremental.from_revision, 1)
        replica.apply_replica_delta(incremental, full_validation=False)

        mirrored_changes = replica.changes_since(before)
        self.assertFalse(mirrored_changes.full_rebuild_required)
        self.assertEqual(mirrored_changes.to_revision, source.revision)
        self.assertEqual(mirrored_changes.handles, {root})
        self.assertEqual(mirrored_changes.components, {"lock"})


if __name__ == "__main__":
    unittest.main()
