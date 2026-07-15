from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


class ResponsesBridgeError(ValueError):
    pass


@dataclass(frozen=True)
class ConvertedResponsesRequest:
    thread_id: str
    chat_request: dict[str, Any]
    unsupported_tool_types: tuple[str, ...] = ()
    tool_name_aliases: tuple[tuple[str, str], ...] = ()
    tool_namespaces: tuple[tuple[str, str], ...] = ()


def extract_codex_thread_id(request: dict[str, Any]) -> str:
    cache_key = request.get("prompt_cache_key")
    if isinstance(cache_key, str) and cache_key:
        return cache_key

    client_metadata = request.get("client_metadata")
    if isinstance(client_metadata, dict):
        encoded = client_metadata.get("x-codex-turn-metadata")
        if isinstance(encoded, str):
            try:
                turn_metadata = json.loads(encoded)
            except json.JSONDecodeError:
                turn_metadata = None
            if isinstance(turn_metadata, dict):
                thread_id = turn_metadata.get("thread_id")
                if isinstance(thread_id, str) and thread_id:
                    return thread_id
    raise ResponsesBridgeError(
        "Codex request has neither prompt_cache_key nor thread_id metadata"
    )


def responses_to_chat_request(
    request: dict[str, Any],
    metadata: BeliefKVRequestMetadata,
    *,
    max_completion_tokens: int = 768,
    temperature: float = 0.0,
) -> ConvertedResponsesRequest:
    if max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")
    thread_id = extract_codex_thread_id(request)
    tools, unsupported, aliases, namespaces = _convert_tools(
        request.get("tools") or []
    )
    response_to_chat = {response_name: chat_name for chat_name, response_name in aliases}
    messages = _convert_input(
        request.get("instructions"),
        request.get("input", []),
        response_to_chat=response_to_chat,
    )
    chat_request: dict[str, Any] = {
        "model": request.get("model"),
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_completion_tokens,
        "rid": (
            f"codex-{thread_id}-{metadata.context_epoch}-"
            f"{uuid.uuid4().hex[:12]}"
        ),
        "beliefkv_metadata": metadata.to_wire(),
    }
    if tools:
        chat_request["tools"] = tools
        chat_request["tool_choice"] = request.get("tool_choice", "auto")
        chat_request["parallel_tool_calls"] = bool(
            request.get("parallel_tool_calls", True)
        )
    return ConvertedResponsesRequest(
        thread_id=thread_id,
        chat_request=chat_request,
        unsupported_tool_types=tuple(sorted(unsupported)),
        tool_name_aliases=tuple(aliases),
        tool_namespaces=tuple(namespaces),
    )


def chat_response_to_responses_events(
    chat_response: dict[str, Any],
    request: dict[str, Any],
    *,
    tool_name_aliases: Mapping[str, str] | None = None,
    tool_namespaces: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    tool_name_aliases = tool_name_aliases or {}
    tool_namespaces = tool_namespaces or {}
    choices = chat_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ResponsesBridgeError("SGLang must return exactly one chat choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ResponsesBridgeError("SGLang chat choice has no message")

    response_id = f"resp_{uuid.uuid4().hex}"
    output: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )

    for index, tool_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        chat_name = function.get("name")
        if not isinstance(chat_name, str) or not chat_name:
            raise ResponsesBridgeError("SGLang returned a tool call without a name")
        name = tool_name_aliases.get(chat_name, chat_name)
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"))
        call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex}")
        function_item = {
            "id": f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "status": "completed",
            "arguments": arguments,
            "call_id": call_id,
            "name": name,
        }
        namespace = tool_namespaces.get(chat_name)
        if namespace:
            function_item["namespace"] = namespace
        output.append(function_item)

    usage = _responses_usage(chat_response.get("usage"))
    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": request.get("instructions"),
        "max_output_tokens": request.get("max_output_tokens"),
        "model": request.get("model") or chat_response.get("model") or "unknown",
        "output": output,
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
        "previous_response_id": None,
        "reasoning": request.get("reasoning"),
        "store": False,
        "temperature": request.get("temperature"),
        "text": request.get("text"),
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools") or [],
        "top_p": request.get("top_p"),
        "truncation": request.get("truncation", "disabled"),
        "usage": usage,
        "metadata": request.get("metadata") or {},
    }

    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        }
    ]
    sequence = 1
    for output_index, item in enumerate(output):
        in_progress = {**item, "status": "in_progress"}
        if item["type"] == "message":
            in_progress["content"] = []
        elif item["type"] == "function_call":
            in_progress["arguments"] = ""
        events.append(
            {
                "type": "response.output_item.added",
                "sequence_number": sequence,
                "output_index": output_index,
                "item": in_progress,
            }
        )
        sequence += 1
        if item["type"] == "message":
            text = item["content"][0]["text"]
            events.append(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                }
            )
            sequence += 1
        else:
            events.append(
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": sequence,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": item["arguments"],
                }
            )
            sequence += 1
        events.append(
            {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": output_index,
                "item": item,
            }
        )
        sequence += 1
    events.append(
        {
            "type": "response.completed",
            "sequence_number": sequence,
            "response": response,
        }
    )
    return tuple(events)


def encode_responses_sse(events: Iterable[dict[str, Any]]) -> bytes:
    chunks = []
    for event in events:
        event_type = str(event["type"])
        chunks.append(
            f"event: {event_type}\n"
            f"data: {json.dumps(event, separators=(',', ':'), allow_nan=False)}\n\n"
        )
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


def _convert_input(
    instructions: Any,
    items: Any,
    *,
    response_to_chat: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    response_to_chat = response_to_chat or {}
    messages: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    if not isinstance(items, list):
        raise ResponsesBridgeError("Responses input must be a list")

    pending_calls: list[dict[str, Any]] = []

    def flush_calls() -> None:
        if pending_calls:
            messages.append(
                {"role": "assistant", "content": "", "tool_calls": pending_calls.copy()}
            )
            pending_calls.clear()

    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            arguments = item.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, separators=(",", ":"))
            response_name = str(item.get("name", ""))
            pending_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or "call_unknown"),
                    "type": "function",
                    "function": {
                        "name": response_to_chat.get(response_name, response_name),
                        "arguments": arguments,
                    },
                }
            )
            continue

        flush_calls()
        if item_type == "message":
            role = str(item.get("role", "user"))
            if role == "developer":
                role = "system"
            content = _content_text(item.get("content"))
            messages.append({"role": role, "content": content})
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id", "")),
                    "content": _tool_output_text(item.get("output")),
                }
            )
        elif item_type == "custom_tool_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id", "")),
                    "content": _tool_output_text(item.get("output")),
                }
            )
    flush_calls()
    if not messages:
        raise ResponsesBridgeError("Responses request produced no chat messages")
    return messages


def _convert_tools(
    tools: list[Any],
) -> tuple[
    list[dict[str, Any]],
    set[str],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    candidates: list[tuple[dict[str, Any], str, str, str | None]] = []
    unsupported: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type", "unknown"))
        if tool_type == "function":
            name = str(tool.get("name", ""))
            candidates.append((dict(tool), name, name, None))
        elif tool_type == "namespace":
            namespace = str(tool.get("name", ""))
            for child in tool.get("tools") or []:
                if not isinstance(child, dict):
                    continue
                flat = dict(child)
                flat["type"] = "function"
                child_name = str(child.get("name", ""))
                # Responses namespaces group schemas; Codex routes calls by the
                # child function name and does not accept a namespace prefix.
                response_name = child_name
                candidates.append((flat, response_name, child_name, namespace))
        else:
            unsupported.add(tool_type)

    preferred_counts = Counter(preferred for _, _, preferred, _ in candidates)
    duplicate_names = sorted(
        name for name, count in preferred_counts.items() if name and count > 1
    )
    if duplicate_names:
        raise ResponsesBridgeError(
            "duplicate function names across Codex namespaces: "
            + ", ".join(duplicate_names)
        )
    converted: list[dict[str, Any]] = []
    aliases: list[tuple[str, str]] = []
    namespaces: list[tuple[str, str]] = []
    for flat, response_name, preferred_name, namespace in candidates:
        chat_name = preferred_name or response_name
        flat["name"] = chat_name
        converted.append(_function_tool(flat))
        aliases.append((chat_name, response_name))
        if namespace:
            namespaces.append((chat_name, namespace))
    return converted, unsupported, aliases, namespaces


def _function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise ResponsesBridgeError("function tool has no name")
    function = {
        "name": name,
        "description": str(tool.get("description") or ""),
        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
    }
    if "strict" in tool:
        function["strict"] = bool(tool["strict"])
    return {"type": "function", "function": function}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts)


def _tool_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _content_text(output)
    return json.dumps(output, separators=(",", ":"), default=str)


def _responses_usage(usage: Any) -> dict[str, Any]:
    source = usage if isinstance(usage, dict) else {}
    input_tokens = int(source.get("prompt_tokens", source.get("input_tokens", 0)) or 0)
    output_tokens = int(
        source.get("completion_tokens", source.get("output_tokens", 0)) or 0
    )
    prompt_details = source.get("prompt_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }
