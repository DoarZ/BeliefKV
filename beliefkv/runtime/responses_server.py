import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from beliefkv.runtime.codex_adapter import CodexThreadRegistry
from beliefkv.runtime.responses_bridge import (
    ResponsesBridgeError,
    chat_response_to_responses_events,
    encode_responses_sse,
    extract_codex_thread_id,
    responses_to_chat_request,
)


_SENSITIVE_REQUEST_HEADERS = frozenset(
    {
        "api-key",
        "authorization",
        "cookie",
        "openai-api-key",
        "proxy-authorization",
        "x-api-key",
        "x-auth-token",
    }
)


def sensitive_request_header_names(headers: Mapping[str, str]) -> tuple[str, ...]:
    """Return credential-bearing header names without reading their values."""

    return tuple(
        sorted(
            name.lower()
            for name in headers
            if name.lower() in _SENSITIVE_REQUEST_HEADERS
        )
    )


class ResponsesChatBridgeServer:
    """Local Codex Responses endpoint backed by SGLang Chat Completions."""

    def __init__(
        self,
        registry: CodexThreadRegistry,
        *,
        upstream_base_url: str,
        host: str = "127.0.0.1",
        port: int = 18080,
        max_completion_tokens: int = 768,
        request_timeout_s: float = 300.0,
        audit_path: str | Path | None = None,
        strict_tool_types: bool = True,
        ignored_tool_types: tuple[str, ...] = ("web_search",),
        enforce_child_join_guard: bool = False,
        join_guard_min_children: int = 2,
        join_guard_timeout_ms: int = 120_000,
    ) -> None:
        self.registry = registry
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.host = host
        self.port = port
        self.max_completion_tokens = max_completion_tokens
        self.request_timeout_s = request_timeout_s
        self.strict_tool_types = strict_tool_types
        self.ignored_tool_types = frozenset(ignored_tool_types)
        if join_guard_min_children <= 0:
            raise ValueError("join_guard_min_children must be positive")
        if join_guard_timeout_ms <= 0:
            raise ValueError("join_guard_timeout_ms must be positive")
        self.enforce_child_join_guard = enforce_child_join_guard
        self.join_guard_min_children = join_guard_min_children
        self.join_guard_timeout_ms = join_guard_timeout_ms
        self.audit_path = Path(audit_path).resolve() if audit_path else None
        self._audit_stream = None
        self._audit_lock = threading.Lock()
        self._audit_sequence = 0
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def create_app(self):
        try:
            import httpx
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse, Response
        except ImportError as error:
            raise RuntimeError(
                "Responses bridge requires fastapi, httpx, and uvicorn"
            ) from error

        app = FastAPI(title="BeliefKV Codex Responses Bridge")

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {"status": "ok", "upstream": self.upstream_base_url}

        @app.get("/v1/models")
        async def models():
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.upstream_base_url}/models")
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
            )

        async def responses_handler(raw_request: Request):
            started = time.monotonic()
            try:
                sensitive_headers = sensitive_request_header_names(raw_request.headers)
                if sensitive_headers:
                    self._audit(
                        "request_rejected",
                        reason="sensitive_request_headers",
                        header_names=sensitive_headers,
                        duration_ms=(time.monotonic() - started) * 1000.0,
                    )
                    raise ResponsesBridgeError(
                        "credential-bearing request headers are forbidden"
                    )
                body = await raw_request.json()
                if not isinstance(body, dict):
                    raise ResponsesBridgeError("Responses body must be an object")
                thread_id = extract_codex_thread_id(body)
                metadata = self.registry.metadata_for_request(thread_id)
                converted = responses_to_chat_request(
                    body,
                    metadata,
                    max_completion_tokens=self.max_completion_tokens,
                )
                collaboration_tool_contracts = self._collaboration_tool_contracts(
                    converted.chat_request.get("tools") or []
                )
                unexpected_tool_types = tuple(
                    tool_type
                    for tool_type in converted.unsupported_tool_types
                    if tool_type not in self.ignored_tool_types
                )
                if self.strict_tool_types and unexpected_tool_types:
                    raise ResponsesBridgeError(
                        "unsupported Codex tool types: "
                        + ", ".join(unexpected_tool_types)
                    )
                guard_target = self._join_guard_target(thread_id)
                upstream_called = guard_target is None
                if guard_target is None:
                    async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                        upstream = await client.post(
                            f"{self.upstream_base_url}/chat/completions",
                            json=converted.chat_request,
                        )
                    if upstream.status_code != 200:
                        self._audit(
                            "upstream_error",
                            thread_id=thread_id,
                            status_code=upstream.status_code,
                            duration_ms=(time.monotonic() - started) * 1000.0,
                            response_excerpt=upstream.text[:1000],
                        )
                        return Response(
                            content=upstream.content,
                            status_code=upstream.status_code,
                            media_type=upstream.headers.get(
                                "content-type", "application/json"
                            ),
                        )
                    chat_response = upstream.json()
                else:
                    chat_response = self._join_guard_chat_response(
                        converted.chat_request,
                        target_thread_id=guard_target,
                        context_epoch=metadata.context_epoch,
                    )
                tool_name_aliases = dict(converted.tool_name_aliases)
                tool_namespaces = dict(converted.tool_namespaces)
                events = chat_response_to_responses_events(
                    chat_response,
                    body,
                    tool_name_aliases=tool_name_aliases,
                    tool_namespaces=tool_namespaces,
                )
                usage = chat_response.get("usage") or {}
                choice = (chat_response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                tool_names = []
                for item in message.get("tool_calls") or []:
                    if not isinstance(item, dict):
                        continue
                    function = item.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = function.get("name")
                    if isinstance(name, str) and name:
                        tool_names.append(tool_name_aliases.get(name, name))
                self._audit(
                    "request_completed",
                    thread_id=thread_id,
                    workflow_id=metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    context_epoch=metadata.context_epoch,
                    parent_invocation_id=metadata.parent_invocation_id,
                    request_id=converted.chat_request["rid"],
                    input_items=len(body.get("input") or []),
                    chat_messages=len(converted.chat_request["messages"]),
                    tool_schema_count=len(converted.chat_request.get("tools") or []),
                    tool_name_alias_count=sum(
                        chat_name != response_name
                        for chat_name, response_name in converted.tool_name_aliases
                    ),
                    tool_namespace_count=len(converted.tool_namespaces),
                    collaboration_tool_contracts=collaboration_tool_contracts,
                    ignored_tool_types=[
                        tool_type
                        for tool_type in converted.unsupported_tool_types
                        if tool_type in self.ignored_tool_types
                    ],
                    tool_names=tool_names,
                    upstream_called=upstream_called,
                    join_guard_injected=guard_target is not None,
                    join_guard_target_thread_id=guard_target,
                    finish_reason=choice.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    duration_ms=(time.monotonic() - started) * 1000.0,
                    body_sha256=hashlib.sha256(
                        json.dumps(body, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest(),
                )
                return Response(
                    content=encode_responses_sse(events),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "close",
                        "X-Accel-Buffering": "no",
                    },
                )
            except Exception as error:
                self._audit(
                    "bridge_error",
                    duration_ms=(time.monotonic() - started) * 1000.0,
                    error=f"{type(error).__name__}: {error}",
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": str(error),
                            "type": "invalid_request_error",
                        }
                    },
                )

        app.post("/v1/responses")(responses_handler)
        app.post("/responses")(responses_handler)
        return app

    def start(self, timeout_s: float = 30.0) -> None:
        if self._thread is not None:
            raise RuntimeError("Responses bridge is already running")
        try:
            import uvicorn
        except ImportError as error:
            raise RuntimeError("Responses bridge requires uvicorn") from error
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._audit_stream = self.audit_path.open("x", encoding="utf-8", buffering=1)
        config = uvicorn.Config(
            self.create_app(),
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="beliefkv-responses-bridge",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("Responses bridge exited during startup")
            if time.monotonic() >= deadline:
                raise TimeoutError("Responses bridge did not start")
            time.sleep(0.01)

    def close(self) -> None:
        server = self._server
        thread = self._thread
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=30.0)
            if thread.is_alive() and server is not None:
                server.force_exit = True
                thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError("Responses bridge did not stop")
        self._thread = None
        self._server = None
        if self._audit_stream is not None:
            self._audit_stream.close()
            self._audit_stream = None

    def __enter__(self) -> "ResponsesChatBridgeServer":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _audit(self, event: str, **fields: Any) -> None:
        if self._audit_stream is None:
            return
        with self._audit_lock:
            self._audit_sequence += 1
            payload = {
                "schema_version": 1,
                "sequence": self._audit_sequence,
                "event": event,
                "ts_ms": time.monotonic() * 1000.0,
                **fields,
            }
            self._audit_stream.write(
                json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
            )

    @staticmethod
    def _collaboration_tool_contracts(
        tools: list[dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        contracts: dict[str, dict[str, dict[str, Any]]] = {}
        collaboration_names = {
            "close_agent",
            "resume_agent",
            "send_input",
            "spawn_agent",
            "wait_agent",
        }
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if name not in collaboration_names:
                continue
            parameters = function.get("parameters")
            properties = (
                parameters.get("properties", {})
                if isinstance(parameters, dict)
                else {}
            )
            contracts[str(name)] = {
                str(key): {
                    contract_key: schema[contract_key]
                    for contract_key in ("type", "minimum", "maximum", "default")
                    if contract_key in schema
                }
                for key, schema in properties.items()
                if isinstance(schema, dict)
            }
        return contracts

    def _join_guard_target(self, thread_id: str) -> str | None:
        if not self.enforce_child_join_guard:
            return None
        if self.registry.child_join_active(thread_id):
            return None
        children = self.registry.direct_child_ids(thread_id)
        if len(children) < self.join_guard_min_children:
            return None
        pending = self.registry.unjoined_child_ids(thread_id)
        return pending[0] if pending else None

    def _join_guard_chat_response(
        self,
        chat_request: dict[str, Any],
        *,
        target_thread_id: str,
        context_epoch: int,
    ) -> dict[str, Any]:
        tool_names = {
            str(tool.get("function", {}).get("name"))
            for tool in chat_request.get("tools") or []
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        wait_name = next(
            (name for name in ("wait_agent", "wait") if name in tool_names),
            None,
        )
        if wait_name is None:
            raise ResponsesBridgeError(
                "child join guard requires a wait_agent tool schema"
            )
        call_suffix = hashlib.sha256(
            f"{chat_request['rid']}:{target_thread_id}".encode("utf-8")
        ).hexdigest()[:24]
        return {
            "id": f"chatcmpl-beliefkv-join-{call_suffix}",
            "model": chat_request.get("model"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_beliefkv_join_{call_suffix}",
                                "type": "function",
                                "function": {
                                    "name": wait_name,
                                    "arguments": json.dumps(
                                        {
                                            "targets": [target_thread_id],
                                            "timeout_ms": self.join_guard_timeout_ms,
                                        },
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "beliefkv_join_guard": {
                "context_epoch": context_epoch,
                "target_thread_id": target_thread_id,
            },
        }
