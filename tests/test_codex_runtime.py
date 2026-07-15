import itertools
import os
import tempfile
import unittest
from unittest.mock import patch

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEventKind
from beliefkv.runtime.codex_adapter import CodexRuntimeAdapter, CodexThreadRegistry
from beliefkv.runtime.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    sanitized_codex_environment,
)


class CollectingSink:
    def __init__(self):
        self.events = []

    def emit_batch(self, events):
        self.events.extend(events)


class CodexRuntimeAdapterTest(unittest.TestCase):
    def setUp(self):
        self.sink = CollectingSink()
        ticks = itertools.count(1)
        self.registry = CodexThreadRegistry()
        self.adapter = CodexRuntimeAdapter(
            self.sink,
            self.registry,
            clock_ms=lambda: float(next(ticks)),
        )
        self.adapter.register_root(
            {"id": "root-thread", "parentThreadId": None},
            workload_id="sympy__sympy-20590",
        )

    def test_subagent_join_events_are_authoritative_and_replayable(self):
        self.adapter.handle_notification(
            {
                "method": "thread/started",
                "params": {
                    "thread": {
                        "id": "child-thread",
                        "parentThreadId": "root-thread",
                        "agentRole": "explorer",
                        "agentNickname": "child-a",
                    }
                },
            }
        )
        self.adapter.handle_notification(
            {
                "method": "item/started",
                "params": {
                    "threadId": "root-thread",
                    "startedAtMs": 100,
                    "item": {
                        "id": "wait-call",
                        "type": "collabAgentToolCall",
                        "tool": "wait",
                        "status": "inProgress",
                        "receiverThreadIds": ["child-thread"],
                        "senderThreadId": "root-thread",
                        "agentsStates": {},
                    },
                },
            }
        )
        self.assertTrue(self.registry.child_join_active("root-thread"))
        self.adapter.handle_notification(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "child-thread",
                    "turn": {"id": "child-turn", "status": "completed", "items": []},
                },
            }
        )
        self.adapter.handle_notification(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "root-thread",
                    "completedAtMs": 200,
                    "item": {
                        "id": "wait-call",
                        "type": "collabAgentToolCall",
                        "tool": "wait",
                        "status": "completed",
                        "receiverThreadIds": ["child-thread"],
                        "senderThreadId": "root-thread",
                        "agentsStates": {
                            "child-thread": {
                                "status": "completed",
                                "message": "done",
                            }
                        },
                    },
                },
            }
        )
        self.assertFalse(self.registry.child_join_active("root-thread"))
        self.adapter.handle_notification(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "root-thread",
                    "turn": {"id": "root-turn", "status": "completed", "items": []},
                },
            }
        )

        kinds = [event.kind for event in self.sink.events]
        self.assertEqual(kinds.count(RuntimeEventKind.SPAWN), 1)
        self.assertEqual(kinds.count(RuntimeEventKind.JOIN_CREATE), 1)
        self.assertEqual(kinds.count(RuntimeEventKind.JOIN_WAIT), 1)
        self.assertEqual(kinds.count(RuntimeEventKind.JOIN_SATISFIED), 1)
        self.assertEqual(kinds.count(RuntimeEventKind.RETURN), 2)
        join_create = next(
            event for event in self.sink.events
            if event.kind == RuntimeEventKind.JOIN_CREATE
        )
        self.assertEqual(join_create.attributes["mode"], "any")

        graph = RuntimeCausalContextGraph()
        graph.apply_batch(self.sink.events)
        self.assertEqual(
            graph.invocations["codex-invocation:child-thread"].state,
            InvocationState.DONE,
        )
        self.assertEqual(
            graph.invocations["codex-invocation:root-thread"].state,
            InvocationState.DONE,
        )

    def test_request_metadata_advances_epoch_and_preserves_parent(self):
        self.adapter.handle_notification(
            {
                "method": "thread/started",
                "params": {
                    "thread": {
                        "id": "child-thread",
                        "parentThreadId": "root-thread",
                        "agentRole": "explorer",
                    }
                },
            }
        )
        first = self.registry.metadata_for_request("child-thread")
        second = self.registry.metadata_for_request("child-thread")
        self.assertEqual(first.context_epoch, 0)
        self.assertEqual(second.context_epoch, 1)
        self.assertEqual(first.relation_type, "spawn")
        self.assertEqual(first.context_mode, "fork")
        self.assertEqual(first.parent_invocation_id, "codex-invocation:root-thread")

    def test_registry_tracks_joined_children(self):
        self.adapter.handle_notification(
            {
                "method": "thread/started",
                "params": {
                    "thread": {
                        "id": "child-thread",
                        "parentThreadId": "root-thread",
                    }
                },
            }
        )
        self.assertEqual(
            self.registry.unjoined_child_ids("root-thread"), ("child-thread",)
        )
        self.registry.mark_child_joined("root-thread", "child-thread")
        self.assertEqual(self.registry.unjoined_child_ids("root-thread"), ())

    def test_registry_tracks_an_active_child_join(self):
        self.adapter.handle_notification(
            {
                "method": "thread/started",
                "params": {
                    "thread": {
                        "id": "child-thread",
                        "parentThreadId": "root-thread",
                    }
                },
            }
        )
        self.assertFalse(self.registry.child_join_active("root-thread"))

        self.registry.begin_child_join("root-thread", ("child-thread",))

        self.assertTrue(self.registry.child_join_active("root-thread"))
        with self.assertRaisesRegex(ValueError, "already active"):
            self.registry.begin_child_join("root-thread", ("child-thread",))
        self.registry.finish_child_join("root-thread")
        self.assertFalse(self.registry.child_join_active("root-thread"))

    def test_workflow_end_waits_for_background_child(self):
        self.adapter.handle_notification(
            {
                "method": "thread/started",
                "params": {
                    "thread": {
                        "id": "child-thread",
                        "parentThreadId": "root-thread",
                        "agentRole": "explorer",
                    }
                },
            }
        )
        self.adapter.handle_notification(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "root-thread",
                    "turn": {"id": "root-turn", "status": "completed"},
                },
            }
        )
        self.assertNotIn(
            RuntimeEventKind.WORKFLOW_END,
            [event.kind for event in self.sink.events],
        )

        self.adapter.handle_notification(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "child-thread",
                    "turn": {"id": "child-turn", "status": "completed"},
                },
            }
        )
        self.assertEqual(
            [event.kind for event in self.sink.events].count(
                RuntimeEventKind.WORKFLOW_END
            ),
            1,
        )

        graph = RuntimeCausalContextGraph()
        graph.apply_batch(self.sink.events)
        self.assertEqual(
            graph.invocations["codex-invocation:child-thread"].state,
            InvocationState.DONE,
        )

    def test_wait_timeout_wakes_parent_without_satisfying_join(self):
        self.adapter.handle_notification(
            {
                "method": "thread/started",
                "params": {
                    "thread": {
                        "id": "child-thread",
                        "parentThreadId": "root-thread",
                    }
                },
            }
        )
        started = {
            "method": "item/started",
            "params": {
                "threadId": "root-thread",
                "item": {
                    "id": "wait-timeout",
                    "type": "collabAgentToolCall",
                    "tool": "wait",
                    "receiverThreadIds": ["child-thread"],
                },
            },
        }
        completed = {
            "method": "item/completed",
            "params": {
                "threadId": "root-thread",
                "item": {
                    "id": "wait-timeout",
                    "type": "collabAgentToolCall",
                    "tool": "wait",
                    "status": "completed",
                    "receiverThreadIds": [],
                    "agentsStates": {},
                },
            },
        }
        self.adapter.handle_notification(started)
        self.adapter.handle_notification(completed)

        kinds = [event.kind for event in self.sink.events]
        self.assertIn(RuntimeEventKind.JOIN_TIMEOUT, kinds)
        self.assertNotIn(RuntimeEventKind.JOIN_SATISFIED, kinds)
        graph = RuntimeCausalContextGraph()
        graph.apply_batch(self.sink.events)
        self.assertEqual(
            graph.invocations["codex-invocation:root-thread"].state,
            InvocationState.READY,
        )
        self.assertFalse(graph.joins["codex-join:wait-timeout"].satisfied)

    def test_spawn_completion_registers_child_without_thread_started(self):
        self.adapter.handle_notification(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "root-thread",
                    "item": {
                        "id": "spawn-call",
                        "type": "collabAgentToolCall",
                        "tool": "spawnAgent",
                        "status": "completed",
                        "senderThreadId": "root-thread",
                        "receiverThreadIds": ["child-from-spawn"],
                        "agentsStates": {
                            "child-from-spawn": {"status": "pendingInit"}
                        },
                    },
                },
            }
        )

        child = self.registry.get("child-from-spawn")
        self.assertIsNotNone(child)
        self.assertEqual(child.parent_thread_id, "root-thread")
        spawn_events = [
            event for event in self.sink.events if event.kind == RuntimeEventKind.SPAWN
        ]
        self.assertEqual(len(spawn_events), 1)
        self.assertEqual(
            spawn_events[0].attributes["source"],
            "codex_collab_spawn_completed",
        )

    def test_command_notifications_emit_tool_boundaries_without_output_body(self):
        started = {
            "method": "item/started",
            "params": {
                "threadId": "root-thread",
                "startedAtMs": 100,
                "item": {
                    "id": "command-1",
                    "type": "commandExecution",
                    "command": "rg -n TODO .",
                    "status": "inProgress",
                },
            },
        }
        completed = {
            "method": "item/completed",
            "params": {
                "threadId": "root-thread",
                "completedAtMs": 120,
                "item": {
                    "id": "command-1",
                    "type": "commandExecution",
                    "command": "rg -n TODO .",
                    "status": "completed",
                    "durationMs": 20,
                    "exitCode": 0,
                    "aggregatedOutput": "large output is not persisted",
                },
            },
        }
        self.adapter.handle_notification(started)
        self.adapter.handle_notification(completed)
        tool_events = [
            event
            for event in self.sink.events
            if event.kind in {RuntimeEventKind.TOOL_START, RuntimeEventKind.TOOL_END}
        ]
        self.assertEqual(len(tool_events), 2)
        self.assertEqual(tool_events[1].attributes["output_chars"], 29)
        self.assertNotIn("output", tool_events[1].attributes)


class CodexAppServerEnvironmentTest(unittest.TestCase):
    def test_credentials_and_proxies_are_not_inherited(self):
        inherited = {
            "OPENAI_API_KEY": "do-not-copy",
            "GITHUB_TOKEN": "do-not-copy",
            "HTTPS_PROXY": "http://credentialed-proxy.invalid",
            "PATH": "/usr/bin",
        }
        with patch.dict(os.environ, inherited, clear=True):
            environment = sanitized_codex_environment(
                {
                    "CUSTOM_AUTH_TOKEN": "also-do-not-copy",
                    "NO_PROXY": "127.0.0.1,localhost",
                }
            )

        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["NO_PROXY"], "127.0.0.1,localhost")
        for name in (
            "OPENAI_API_KEY",
            "GITHUB_TOKEN",
            "HTTPS_PROXY",
            "CUSTOM_AUTH_TOKEN",
        ):
            self.assertNotIn(name, environment)

    def test_reader_failure_is_available_to_health_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CodexAppServerClient(["codex"], codex_home=directory)
            failure = RuntimeError("notification callback failed")
            client._reader_error = failure

            with self.assertRaises(CodexAppServerError) as caught:
                client.raise_if_reader_failed()

        self.assertIs(caught.exception.__cause__, failure)


if __name__ == "__main__":
    unittest.main()
