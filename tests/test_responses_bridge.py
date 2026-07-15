import json
import unittest

from beliefkv.runtime.responses_bridge import (
    chat_response_to_responses_events,
    encode_responses_sse,
    responses_to_chat_request,
)
from beliefkv.runtime.codex_adapter import CodexThreadIdentity, CodexThreadRegistry
from beliefkv.runtime.responses_server import (
    ResponsesChatBridgeServer,
    sensitive_request_header_names,
)
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


class ResponsesBridgeTest(unittest.TestCase):
    def setUp(self):
        self.metadata = BeliefKVRequestMetadata(
            root_workflow_id="wf",
            invocation_id="parent",
            context_id="ctx",
            context_epoch=2,
        )

    def test_codex_history_and_tool_schema_convert_to_chat(self):
        request = {
            "model": "Qwen2.5-14B-Instruct",
            "instructions": "system",
            "prompt_cache_key": "thread-1",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "inspect"}],
                },
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                    "call_id": "call-1",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "chunk",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                    "strict": False,
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        converted = responses_to_chat_request(request, self.metadata)
        chat = converted.chat_request
        self.assertEqual(converted.thread_id, "thread-1")
        self.assertEqual([item["role"] for item in chat["messages"]], [
            "system", "user", "assistant", "tool"
        ])
        self.assertEqual(
            chat["messages"][2]["tool_calls"][0]["function"]["name"],
            "exec_command",
        )
        self.assertEqual(chat["beliefkv_metadata"]["context_epoch"], 2)
        self.assertTrue(chat["parallel_tool_calls"])
        self.assertFalse(converted.unsupported_tool_types)

    def test_chat_tool_call_becomes_responses_function_item(self):
        request = {
            "model": "Qwen2.5-14B-Instruct",
            "instructions": "system",
            "prompt_cache_key": "thread-1",
            "input": [{"type": "message", "role": "user", "content": "spawn"}],
            "tools": [],
            "stream": True,
        }
        chat_response = {
            "id": "chat-1",
            "model": request["model"],
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-spawn",
                                "type": "function",
                                "function": {
                                    "name": "spawn_agent",
                                    "arguments": '{"message":"inspect"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }
        events = chat_response_to_responses_events(chat_response, request)
        completed = events[-1]["response"]
        function = completed["output"][0]
        self.assertEqual(function["type"], "function_call")
        self.assertEqual(function["name"], "spawn_agent")
        self.assertEqual(function["call_id"], "call-spawn")
        self.assertEqual(completed["usage"]["input_tokens"], 100)
        encoded = encode_responses_sse(events).decode()
        self.assertIn("event: response.output_item.done", encoded)
        self.assertIn("event: response.completed", encoded)
        self.assertTrue(encoded.endswith("data: [DONE]\n\n"))

    def test_web_search_is_reported_for_explicit_server_side_filtering(self):
        request = {
            "model": "Qwen2.5-14B-Instruct",
            "prompt_cache_key": "thread-1",
            "input": [{"type": "message", "role": "user", "content": "inspect"}],
            "tools": [{"type": "web_search"}],
        }
        converted = responses_to_chat_request(request, self.metadata)
        self.assertEqual(converted.unsupported_tool_types, ("web_search",))
        self.assertNotIn("tools", converted.chat_request)

    def test_namespace_tool_uses_reversible_short_alias(self):
        request = {
            "model": "Qwen2.5-14B-Instruct",
            "prompt_cache_key": "thread-1",
            "input": [
                {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "arguments": '{"message":"inspect"}',
                    "call_id": "old-spawn",
                },
                {
                    "type": "function_call_output",
                    "call_id": "old-spawn",
                    "output": "child-1",
                },
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "multi_agent_v1",
                    "tools": [
                        {
                            "name": "spawn_agent",
                            "parameters": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "wait",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    ],
                }
            ],
        }
        converted = responses_to_chat_request(request, self.metadata)
        aliases = dict(converted.tool_name_aliases)
        namespaces = dict(converted.tool_namespaces)
        self.assertEqual(aliases["spawn_agent"], "spawn_agent")
        self.assertEqual(aliases["wait"], "wait")
        self.assertEqual(namespaces["spawn_agent"], "multi_agent_v1")
        self.assertEqual(
            converted.chat_request["messages"][0]["tool_calls"][0]["function"]["name"],
            "spawn_agent",
        )
        response = {
            "id": "chat-aliased-tool",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "new-spawn",
                                "type": "function",
                                "function": {
                                    "name": "spawn_agent",
                                    "arguments": '{"message":"inspect tests"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {},
        }
        completed = chat_response_to_responses_events(
            response,
            request,
            tool_name_aliases=aliases,
            tool_namespaces=namespaces,
        )[-1]["response"]
        self.assertEqual(completed["output"][0]["name"], "spawn_agent")
        self.assertEqual(completed["output"][0]["namespace"], "multi_agent_v1")

    def test_null_prompt_token_details_are_valid(self):
        request = {
            "model": "Qwen2.5-14B-Instruct",
            "prompt_cache_key": "thread-1",
            "input": [{"type": "message", "role": "user", "content": "done"}],
        }
        response = {
            "id": "chat-null-details",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "prompt_tokens_details": None,
            },
        }
        completed = chat_response_to_responses_events(response, request)[-1]["response"]
        self.assertEqual(
            completed["usage"]["input_tokens_details"]["cached_tokens"], 0
        )

    def test_sensitive_header_detection_never_returns_values(self):
        headers = {
            "Authorization": "Bearer secret",
            "X-API-Key": "secret",
            "Content-Type": "application/json",
        }
        self.assertEqual(
            sensitive_request_header_names(headers),
            ("authorization", "x-api-key"),
        )

    def test_fastapi_injects_request_instead_of_query_parameter(self):
        registry = CodexThreadRegistry()
        registry.register(
            CodexThreadIdentity(
                thread_id="thread-1",
                workflow_id="workflow-1",
                invocation_id="invocation-1",
                context_id="context-1",
                parent_thread_id=None,
                agent_role="parent",
            )
        )
        server = ResponsesChatBridgeServer(
            registry,
            upstream_base_url="http://127.0.0.1:1/v1",
        )
        route = next(
            item for item in server.create_app().routes if item.path == "/v1/responses"
        )
        self.assertEqual(route.dependant.request_param_name, "raw_request")
        self.assertFalse(route.dependant.query_params)

    def test_join_guard_targets_each_unjoined_child(self):
        registry = CodexThreadRegistry()
        registry.register(
            CodexThreadIdentity(
                thread_id="root",
                workflow_id="workflow",
                invocation_id="root-invocation",
                context_id="root-context",
                parent_thread_id=None,
                agent_role="parent",
            )
        )
        for child_id in ("child-b", "child-a"):
            registry.register(
                CodexThreadIdentity(
                    thread_id=child_id,
                    workflow_id="workflow",
                    invocation_id=f"invocation-{child_id}",
                    context_id=f"context-{child_id}",
                    parent_thread_id="root",
                    agent_role="subagent",
                )
            )
        server = ResponsesChatBridgeServer(
            registry,
            upstream_base_url="http://127.0.0.1:1/v1",
            enforce_child_join_guard=True,
        )
        self.assertEqual(server._join_guard_target("root"), "child-a")
        registry.begin_child_join("root", ("child-a",))
        self.assertIsNone(server._join_guard_target("root"))
        registry.finish_child_join("root")
        response = server._join_guard_chat_response(
            {
                "rid": "request-1",
                "model": "Qwen2.5-14B-Instruct",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "wait_agent", "parameters": {}},
                    }
                ],
            },
            target_thread_id="child-a",
            context_epoch=2,
        )
        function = response["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "wait_agent")
        self.assertEqual(
            json.loads(function["arguments"]),
            {"targets": ["child-a"], "timeout_ms": 120000},
        )

        registry.mark_child_joined("root", "child-a")
        self.assertEqual(server._join_guard_target("root"), "child-b")
        registry.mark_child_joined("root", "child-b")
        self.assertIsNone(server._join_guard_target("root"))


if __name__ == "__main__":
    unittest.main()
