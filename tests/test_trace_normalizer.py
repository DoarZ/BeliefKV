import unittest

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEventKind
from beliefkv.traces.normalizer import ClawTraceNormalizer, load_jsonl_records


def envelope(event_id, event_type, ts_ms, span_id, payload, parent_span_id=None):
    return {
        "schemaVersion": 1,
        "agentId": "main",
        "event": {
            "eventId": event_id,
            "eventType": event_type,
            "traceId": "trace",
            "spanId": span_id,
            "parentSpanId": parent_span_id,
            "tsMs": ts_ms,
            "payload": payload,
        },
    }


class ClawTraceNormalizerTest(unittest.TestCase):
    def test_explicit_subagent_lifecycle_replays_into_rccg(self):
        records = [
            envelope(
                "s0",
                "session_start",
                0,
                "root-span",
                {"sessionKey": "agent:main:main", "runId": "root", "agentName": "main"},
            ),
            envelope(
                "l0",
                "llm_before_call",
                1,
                "llm-span",
                {"sessionKey": "agent:main:main", "runId": "root", "model": "qwen"},
                "root-span",
            ),
            envelope(
                "t0",
                "tool_before_call",
                2,
                "tool-span",
                {
                    "sessionKey": "agent:main:main",
                    "runId": "root",
                    "toolName": "web_search",
                    "toolCallId": "tool-1",
                },
                "llm-span",
            ),
            envelope(
                "t1",
                "tool_after_call",
                3,
                "tool-span",
                {
                    "sessionKey": "agent:main:main",
                    "runId": "root",
                    "toolName": "web_search",
                    "toolCallId": "tool-1",
                    "durationMs": 1,
                },
                "llm-span",
            ),
            envelope(
                "spawn",
                "subagent_spawn",
                4,
                "spawn-span",
                {
                    "runId": "child-run",
                    "requesterSessionKey": "agent:main:main",
                    "childSessionKey": "agent:browser:subagent:1",
                    "subagentId": "browser",
                    "mode": "foreground",
                },
                "root-span",
            ),
            envelope(
                "child-session",
                "session_start",
                5,
                "child-root",
                {
                    "sessionKey": "agent:browser:subagent:1",
                    "runId": "child-run",
                    "agentName": "browser",
                    "isSubAgent": True,
                },
                "spawn-span",
            ),
            envelope(
                "join",
                "subagent_join",
                10,
                "spawn-span",
                {"targetSessionKey": "agent:browser:subagent:1", "outcome": "ok"},
                "root-span",
            ),
            envelope(
                "end",
                "session_end",
                11,
                "root-span",
                {"sessionKey": "agent:main:main"},
            ),
        ]
        normalized = ClawTraceNormalizer().normalize(records)
        kinds = [event.kind for event in normalized.events]
        self.assertIn(RuntimeEventKind.CALL, kinds)
        self.assertIn(RuntimeEventKind.TOOL_START, kinds)
        self.assertIn(RuntimeEventKind.TOOL_END, kinds)
        graph = RuntimeCausalContextGraph()
        graph.apply_batch(normalized.events)
        self.assertEqual(graph.invocations["child-run"].parent_invocation_id, "root")
        self.assertEqual(graph.invocations["child-run"].state, InvocationState.DONE)
        self.assertIsNotNone(graph.workflows["trace"].end_ts_ms)
        self.assertEqual(normalized.report.unresolved_parent_events, 0)

    def test_child_session_race_is_repaired_when_spawn_arrives(self):
        records = [
            envelope(
                "parent",
                "session_start",
                0,
                "root-span",
                {"sessionKey": "parent-session", "runId": "parent-run"},
            ),
            envelope(
                "child",
                "session_start",
                1,
                "child-root",
                {
                    "sessionKey": "child-session",
                    "runId": "child-run",
                    "isSubAgent": True,
                },
                "unknown-spawn-span",
            ),
            envelope(
                "spawn",
                "subagent_spawn",
                2,
                "unknown-spawn-span",
                {
                    "runId": "child-run",
                    "requesterSessionKey": "parent-session",
                    "childSessionKey": "child-session",
                    "mode": "background",
                },
                "root-span",
            ),
        ]
        normalized = ClawTraceNormalizer().normalize(records)
        graph = RuntimeCausalContextGraph()
        graph.apply_batch(normalized.events)
        self.assertEqual(graph.invocations["child-run"].parent_invocation_id, "parent-run")
        self.assertEqual(graph.invocations["parent-run"].state, InvocationState.CANCELLED)

    def test_jsonl_loader_reports_line_number(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            load_jsonl_records(['{"ok": 1}\n', "not-json\n"])

    def test_normalizer_can_be_reused_without_cross_trace_state(self):
        records = [
            envelope(
                "start",
                "session_start",
                0,
                "span",
                {"sessionKey": "session", "runId": "run"},
            )
        ]
        normalizer = ClawTraceNormalizer()
        first = normalizer.normalize(records)
        second = normalizer.normalize(records)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.report.input_records, second.report.input_records)


if __name__ == "__main__":
    unittest.main()
