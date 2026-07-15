import unittest

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.runtime.agent_runtime_adapter import (
    InstrumentedToolEnvironment,
    InvocationRuntimeEmitter,
)
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


def event(event_id, kind, *, ts_ms, **kwargs):
    return RuntimeEvent(
        event_id=event_id,
        ts_ms=ts_ms,
        kind=kind,
        workflow_id="wf",
        **kwargs,
    )


class RuntimeEventAdapterTest(unittest.TestCase):
    def test_invalid_batch_rolls_back_graph_and_controller_side_effects(self):
        controller = BeliefKVController()
        events = (
            event("start", RuntimeEventKind.WORKFLOW_START, ts_ms=0),
            event(
                "invalid-tool",
                RuntimeEventKind.TOOL_START,
                ts_ms=1,
                invocation_id="missing",
            ),
        )
        with self.assertRaises(Exception):
            controller.process_runtime_events(events)
        self.assertNotIn("wf", controller.graph.workflows)
        self.assertEqual(controller.fairness.accounts, {})

    def test_instrumented_environment_emits_exact_tool_boundaries(self):
        batches = []
        clock = iter([1.0, 2.0, 7.5, 8.0])
        metadata = BeliefKVRequestMetadata("wf", "root", "ctx", 0, "coder", "coder-1")
        class CollectingSink:
            def emit_batch(self, events):
                batches.append(events)

        emitter = InvocationRuntimeEmitter(
            CollectingSink(), metadata, clock_ms=lambda: next(clock)
        )

        class FakeEnvironment:
            def execute(self, action):
                return {"output": action["command"], "returncode": 0}

            def get_template_vars(self, **kwargs):
                return kwargs

            def serialize(self):
                return {"fake": True}

        emitter.start(source="test")
        wrapped = InstrumentedToolEnvironment(FakeEnvironment(), emitter)
        self.assertEqual(wrapped.execute({"command": "pwd"})["returncode"], 0)
        emitter.finish(outcome="submitted")

        events = tuple(item for batch in batches for item in batch)
        self.assertEqual(
            [item.kind for item in events],
            [
                RuntimeEventKind.WORKFLOW_START,
                RuntimeEventKind.INVOCATION_CREATE,
                RuntimeEventKind.TOOL_START,
                RuntimeEventKind.TOOL_END,
                RuntimeEventKind.RETURN,
                RuntimeEventKind.WORKFLOW_END,
            ],
        )
        self.assertEqual(events[3].attributes["duration_ms"], 5.5)
        self.assertEqual(events[3].attributes["returncode"], 0)
        self.assertNotIn("command", events[2].attributes)


if __name__ == "__main__":
    unittest.main()
