import unittest

from beliefkv.control.causal_graph import (
    CausalGraphError,
    InvocationState,
    RuntimeCausalContextGraph,
)
from beliefkv.core.events import ExecutionMode, RuntimeEvent, RuntimeEventKind


def event(
    sequence: int,
    kind: RuntimeEventKind,
    *,
    workflow_id: str = "wf",
    invocation_id: str | None = None,
    target_invocation_id: str | None = None,
    context_id: str | None = None,
    **kwargs,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"e{sequence}",
        ts_ms=float(sequence),
        kind=kind,
        workflow_id=workflow_id,
        invocation_id=invocation_id,
        target_invocation_id=target_invocation_id,
        context_id=context_id,
        **kwargs,
    )


class CausalGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = RuntimeCausalContextGraph()
        self.graph.apply(event(0, RuntimeEventKind.WORKFLOW_START))

    def create(self, sequence: int, invocation_id: str, context_id: str) -> None:
        self.graph.apply(
            event(
                sequence,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id=invocation_id,
                context_id=context_id,
                agent_definition_id=invocation_id,
                agent_instance_id=invocation_id,
            )
        )

    def test_nested_foreground_return_wakes_stack_in_order(self):
        self.create(1, "parent", "ctx-parent")
        self.create(2, "searcher", "ctx-searcher")
        self.create(3, "browser", "ctx-browser")
        self.graph.apply(
            event(
                4,
                RuntimeEventKind.CALL,
                invocation_id="parent",
                target_invocation_id="searcher",
            )
        )
        self.graph.apply(
            event(
                5,
                RuntimeEventKind.CALL,
                invocation_id="searcher",
                target_invocation_id="browser",
            )
        )

        self.assertEqual(
            self.graph.invocations["parent"].state, InvocationState.WAIT_CHILD
        )
        self.assertEqual(
            self.graph.invocations["searcher"].state, InvocationState.WAIT_CHILD
        )
        first = self.graph.apply(
            event(6, RuntimeEventKind.RETURN, invocation_id="browser")
        )
        self.assertEqual(first.awakened_invocations, frozenset({"searcher"}))
        self.assertEqual(
            self.graph.invocations["parent"].state, InvocationState.WAIT_CHILD
        )

        second = self.graph.apply(
            event(7, RuntimeEventKind.RETURN, invocation_id="searcher")
        )
        self.assertEqual(second.awakened_invocations, frozenset({"parent"}))
        self.assertEqual(self.graph.invocations["parent"].state, InvocationState.READY)

    def test_background_spawn_does_not_park_parent(self):
        self.create(1, "parent", "ctx-parent")
        self.create(2, "child", "ctx-child")
        delta = self.graph.apply(
            event(
                3,
                RuntimeEventKind.SPAWN,
                invocation_id="parent",
                target_invocation_id="child",
                execution_mode=ExecutionMode.BACKGROUND,
            )
        )
        self.assertFalse(delta.parked_invocations)
        self.assertEqual(self.graph.invocations["parent"].state, InvocationState.READY)

    def test_join_wakes_waiter_only_after_all_members_return(self):
        self.create(1, "parent", "ctx-parent")
        self.create(2, "a", "ctx-a")
        self.create(3, "b", "ctx-b")
        self.graph.apply(
            event(
                4,
                RuntimeEventKind.JOIN_CREATE,
                join_id="join",
                member_invocation_ids=("a", "b"),
            )
        )
        self.graph.apply(
            event(
                5,
                RuntimeEventKind.JOIN_WAIT,
                invocation_id="parent",
                join_id="join",
            )
        )
        self.graph.apply(event(6, RuntimeEventKind.RETURN, invocation_id="a"))
        self.assertEqual(
            self.graph.invocations["parent"].state, InvocationState.WAIT_JOIN
        )
        delta = self.graph.apply(
            event(7, RuntimeEventKind.RETURN, invocation_id="b")
        )
        self.assertEqual(delta.awakened_invocations, frozenset({"parent"}))

    def test_handoff_parks_sender_and_wakes_peer(self):
        self.create(1, "coder", "ctx-coder")
        self.create(2, "reviewer", "ctx-reviewer")
        delta = self.graph.apply(
            event(
                3,
                RuntimeEventKind.HANDOFF,
                invocation_id="coder",
                target_invocation_id="reviewer",
            )
        )
        self.assertEqual(delta.parked_invocations, frozenset({"coder"}))
        self.assertEqual(
            self.graph.invocations["coder"].state, InvocationState.WAIT_MESSAGE
        )
        self.assertEqual(self.graph.invocations["reviewer"].pending_messages, 1)

    def test_duplicate_event_is_idempotent(self):
        self.create(1, "root", "ctx")
        message = event(2, RuntimeEventKind.TOOL_START, invocation_id="root")
        self.graph.apply(message)
        duplicate = self.graph.apply(message)
        self.assertFalse(duplicate.changed_contexts)

    def test_atomic_batch_rolls_back_on_invalid_event(self):
        valid = event(
            1,
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="root",
            context_id="ctx",
        )
        invalid = event(2, RuntimeEventKind.CALL, invocation_id="missing")
        with self.assertRaises((CausalGraphError, ValueError)):
            self.graph.apply_batch([valid, invalid])
        self.assertNotIn("root", self.graph.invocations)

    def test_stale_context_epoch_is_rejected(self):
        self.graph.apply(
            event(
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="root",
                context_id="ctx",
                context_epoch=3,
            )
        )
        with self.assertRaises(CausalGraphError):
            self.graph.apply(
                event(
                    2,
                    RuntimeEventKind.LLM_RESULT,
                    invocation_id="root",
                    context_epoch=2,
                )
            )

    def test_context_advance_is_monotonic_without_changing_invocation_state(self):
        self.graph.apply(
            event(
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="root",
                context_id="ctx",
                context_epoch=0,
            )
        )
        delta = self.graph.apply(
            event(
                2,
                RuntimeEventKind.CONTEXT_ADVANCE,
                invocation_id="root",
                context_id="ctx",
                context_epoch=3,
            )
        )
        self.assertEqual(self.graph.contexts["ctx"].epoch, 3)
        self.assertEqual(self.graph.invocations["root"].state, InvocationState.READY)
        self.assertEqual(delta.changed_contexts, frozenset({"ctx"}))

        with self.assertRaises(CausalGraphError):
            self.graph.apply(
                event(
                    3,
                    RuntimeEventKind.CONTEXT_ADVANCE,
                    invocation_id="root",
                    context_id="ctx",
                    context_epoch=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
