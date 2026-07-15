import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.runtime.event_channel import (
    JsonlRuntimeEventSink,
    RuntimeEventDatagramServer,
    UnixDatagramRuntimeEventSink,
)


def event(event_id, kind, *, ts_ms, **kwargs):
    return RuntimeEvent(
        event_id=event_id,
        ts_ms=ts_ms,
        kind=kind,
        workflow_id="wf",
        **kwargs,
    )


class RuntimeEventChannelTest(unittest.TestCase):
    def test_event_wire_roundtrip(self):
        original = event(
            "create",
            RuntimeEventKind.INVOCATION_CREATE,
            ts_ms=1.0,
            invocation_id="root",
            context_id="ctx",
            context_epoch=2,
            attributes={"persistent": True},
        )
        self.assertEqual(RuntimeEvent.from_dict(original.to_dict()), original)

    def test_acknowledged_batch_reaches_controller(self):
        controller = BeliefKVController()
        events = (
            event("start", RuntimeEventKind.WORKFLOW_START, ts_ms=0.0),
            event(
                "create",
                RuntimeEventKind.INVOCATION_CREATE,
                ts_ms=1.0,
                invocation_id="root",
                context_id="ctx",
                context_epoch=0,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self._server_or_skip(
                Path(temporary) / "server.sock", controller
            ) as server:
                worker = threading.Thread(target=self._drain_one, args=(server,))
                worker.start()
                with UnixDatagramRuntimeEventSink(server.path) as sink:
                    sink.emit_batch(events)
                worker.join(timeout=2.0)
                self.assertFalse(worker.is_alive())
        self.assertIn("wf", controller.graph.workflows)
        self.assertIn("root", controller.graph.invocations)

    def test_rejected_batch_is_reported_to_client(self):
        controller = BeliefKVController()
        invalid = event(
            "tool",
            RuntimeEventKind.TOOL_START,
            ts_ms=1.0,
            invocation_id="missing",
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self._server_or_skip(
                Path(temporary) / "server.sock", controller
            ) as server:
                worker = threading.Thread(target=self._drain_one, args=(server,))
                worker.start()
                with UnixDatagramRuntimeEventSink(server.path) as sink:
                    with self.assertRaisesRegex(RuntimeError, "rejected"):
                        sink.emit_batch((invalid,))
                worker.join(timeout=2.0)
                self.assertFalse(worker.is_alive())

    def test_jsonl_sink_records_ordered_roundtrippable_events(self):
        events = (
            event("start", RuntimeEventKind.WORKFLOW_START, ts_ms=0.0),
            event("end", RuntimeEventKind.WORKFLOW_END, ts_ms=2.0),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            with JsonlRuntimeEventSink(path) as sink:
                sink.emit_batch(events)
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([item["sequence"] for item in records], [1, 2])
        self.assertEqual(
            tuple(RuntimeEvent.from_dict(item) for item in records),
            events,
        )

    @staticmethod
    def _drain_one(server: RuntimeEventDatagramServer) -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if server.drain():
                return
            time.sleep(0.001)
        raise TimeoutError("server did not receive a runtime event batch")

    def _server_or_skip(
        self, path: Path, controller: BeliefKVController
    ) -> RuntimeEventDatagramServer:
        try:
            return RuntimeEventDatagramServer(
                path,
                controller.process_runtime_events,
            )
        except PermissionError as error:
            self.skipTest(f"Unix sockets are blocked by the test sandbox: {error}")


if __name__ == "__main__":
    unittest.main()
