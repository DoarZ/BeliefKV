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

        self.index.bind_pages("ctx-parent", 0, [prefix])
        resolved = arbiter.resolve(
            self.command(CommandKind.PREFETCH_CONTEXT, "ctx-parent", 200)
        )
        self.assertEqual(
            [item.handle for item in resolved.page_actions], [prefix, suffix]
        )


if __name__ == "__main__":
    unittest.main()
