import json
from pathlib import Path
import tempfile
import unittest

from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.traces.runtime_validation import (
    RuntimeTraceValidationError,
    relative_event_records,
    validate_runtime_trace,
)


def _event(sequence, kind, ts_ms, **kwargs):
    event = RuntimeEvent(
        event_id=f"event-{sequence}",
        ts_ms=ts_ms,
        kind=kind,
        workflow_id="workflow-1",
        **kwargs,
    )
    return {"schema_version": 1, "sequence": sequence, **event.to_dict()}


def _write_jsonl(path: Path, records):
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class RuntimeTraceValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.events_path = root / "events.jsonl"
        self.audit_path = root / "audit.jsonl"
        self.events = [
            _event(1, RuntimeEventKind.WORKFLOW_START, 100.0),
            _event(
                2,
                RuntimeEventKind.INVOCATION_CREATE,
                101.0,
                invocation_id="root",
                context_id="context-1",
                context_epoch=0,
            ),
            _event(
                3,
                RuntimeEventKind.LLM_SUBMIT,
                102.0,
                invocation_id="root",
                context_id="context-1",
                attributes={
                    "request_id": "request-1",
                    "prompt_tokens": 100,
                    "cache_hit_tokens": 60,
                },
            ),
            _event(
                4,
                RuntimeEventKind.LLM_RESULT,
                105.0,
                invocation_id="root",
                context_id="context-1",
                attributes={"request_id": "request-1", "output_tokens": 10},
            ),
            _event(
                5,
                RuntimeEventKind.TOOL_START,
                106.0,
                invocation_id="root",
                context_id="context-1",
                attributes={"tool_call_id": "tool-1"},
            ),
            _event(
                6,
                RuntimeEventKind.TOOL_END,
                110.0,
                invocation_id="root",
                context_id="context-1",
                attributes={"tool_call_id": "tool-1", "duration_ms": 4.0},
            ),
            _event(
                7,
                RuntimeEventKind.RETURN,
                111.0,
                invocation_id="root",
                context_id="context-1",
            ),
            _event(8, RuntimeEventKind.WORKFLOW_END, 112.0),
        ]
        self.audit = [
            {
                "schema_version": 1,
                "sequence": 1,
                "run_id": "run-1",
                "event": "runtime_initialized",
                "ts_ms": 1.0,
            },
            {
                "schema_version": 1,
                "sequence": 2,
                "run_id": "run-1",
                "event": "runtime_event_delivery",
                "ts_ms": 2.0,
                "accepted": True,
                "error": "",
            },
            {
                "schema_version": 1,
                "sequence": 3,
                "run_id": "run-1",
                "event": "request_started",
                "ts_ms": 3.0,
                "request_id": "request-1",
            },
            {
                "schema_version": 1,
                "sequence": 4,
                "run_id": "run-1",
                "event": "request_finished",
                "ts_ms": 4.0,
                "request_id": "request-1",
            },
        ]

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_valid_trace_produces_replay_summary(self):
        _write_jsonl(self.events_path, self.events)
        _write_jsonl(self.audit_path, self.audit)

        summary = validate_runtime_trace(self.events_path, self.audit_path)

        self.assertEqual(summary.workflow_id, "workflow-1")
        self.assertEqual(summary.event_count, 8)
        self.assertEqual(summary.llm_call_count, 1)
        self.assertEqual(summary.tool_call_count, 1)
        self.assertEqual(summary.total_uncached_prompt_tokens, 40)
        self.assertEqual(summary.total_output_tokens, 10)
        self.assertEqual(summary.runtime_delivery_count, 1)
        self.assertTrue(summary.controller_replay_valid)
        self.assertEqual(
            [record["ts_ms"] for record in relative_event_records(self.events_path)],
            [0.0, 1.0, 2.0, 5.0, 6.0, 10.0, 11.0, 12.0],
        )

    def test_schema_two_runtime_audit_is_backward_compatible(self):
        for record in self.audit:
            record["schema_version"] = 2
        _write_jsonl(self.events_path, self.events)
        _write_jsonl(self.audit_path, self.audit)

        summary = validate_runtime_trace(self.events_path, self.audit_path)

        self.assertTrue(summary.controller_replay_valid)

    def test_unmatched_tool_identity_is_rejected(self):
        self.events[5]["attributes"]["tool_call_id"] = "tool-2"
        _write_jsonl(self.events_path, self.events)

        with self.assertRaisesRegex(
            RuntimeTraceValidationError, "ended before start"
        ):
            validate_runtime_trace(self.events_path)

    def test_rejected_runtime_delivery_is_rejected(self):
        self.audit[1]["accepted"] = False
        self.audit[1]["error"] = "invalid event"
        _write_jsonl(self.events_path, self.events)
        _write_jsonl(self.audit_path, self.audit)

        with self.assertRaisesRegex(
            RuntimeTraceValidationError, "runtime event delivery failed"
        ):
            validate_runtime_trace(self.events_path, self.audit_path)


if __name__ == "__main__":
    unittest.main()
