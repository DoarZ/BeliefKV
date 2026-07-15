from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Callable, Protocol

from beliefkv.core.events import (
    ContextMode,
    EventConfidence,
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.predictor.taxonomy import ToolTaxonomy
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata


class RuntimeEventSink(Protocol):
    def emit_batch(self, events: tuple[RuntimeEvent, ...]) -> None:
        ...


class InvocationRuntimeEmitter:
    """Synchronous exact-boundary emitter for one runtime invocation."""

    def __init__(
        self,
        sink: RuntimeEventSink,
        metadata: BeliefKVRequestMetadata,
        *,
        clock_ms: Callable[[], float] | None = None,
    ) -> None:
        self.sink = sink
        self.metadata = metadata
        self.clock_ms = clock_ms or (lambda: time.monotonic() * 1000.0)
        self._sequence = 0
        self._last_ts_ms = 0.0
        self._lock = threading.Lock()

    def start(self, *, source: str) -> None:
        ts_ms = self._timestamp()
        self.sink.emit_batch(
            (
                self._event(
                    RuntimeEventKind.WORKFLOW_START,
                    ts_ms=ts_ms,
                    attributes={"source": source},
                ),
                self._event(
                    RuntimeEventKind.INVOCATION_CREATE,
                    ts_ms=ts_ms,
                    invocation_id=self.metadata.invocation_id,
                    context_id=self.metadata.context_id,
                    context_epoch=self.metadata.context_epoch,
                    agent_definition_id=self.metadata.agent_definition_id,
                    agent_instance_id=self.metadata.agent_instance_id,
                    parent_invocation_id=self.metadata.parent_invocation_id,
                    parent_context_id=self.metadata.parent_context_id,
                    relation_type=RelationType(self.metadata.relation_type),
                    context_mode=ContextMode(self.metadata.context_mode),
                    execution_mode=ExecutionMode(self.metadata.execution_mode),
                    return_target_id=self.metadata.return_target_id,
                    join_id=self.metadata.join_id,
                    attributes={"persistent": True, "source": source},
                ),
            )
        )

    def tool_start(self, action: dict[str, Any]) -> tuple[str, float]:
        ts_ms = self._timestamp()
        tool_call_id = f"{self.metadata.invocation_id}:tool:{self._sequence + 1}"
        command = str(action.get("command", ""))
        normalized = ToolTaxonomy().normalize("bash")
        self.sink.emit_batch(
            (
                self._event(
                    RuntimeEventKind.TOOL_START,
                    ts_ms=ts_ms,
                    invocation_id=self.metadata.invocation_id,
                    context_id=self.metadata.context_id,
                    context_epoch=self.metadata.context_epoch,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "tool_call_id": tool_call_id,
                        "tool_name": "bash",
                        "tool_family": normalized.family,
                        "backend_class": normalized.backend_class,
                        "command_chars": len(command),
                        "command_sha256": hashlib.sha256(
                            command.encode("utf-8")
                        ).hexdigest(),
                    },
                ),
            )
        )
        return tool_call_id, ts_ms

    def tool_end(
        self,
        tool_call_id: str,
        start_ts_ms: float,
        *,
        output: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        ts_ms = self._timestamp()
        output_text = str((output or {}).get("output", ""))
        self.sink.emit_batch(
            (
                self._event(
                    RuntimeEventKind.TOOL_END,
                    ts_ms=ts_ms,
                    invocation_id=self.metadata.invocation_id,
                    context_id=self.metadata.context_id,
                    context_epoch=self.metadata.context_epoch,
                    confidence=EventConfidence.OBSERVED_EXACT,
                    attributes={
                        "tool_call_id": tool_call_id,
                        "duration_ms": max(0.0, ts_ms - start_ts_ms),
                        "returncode": (
                            int(output["returncode"])
                            if output is not None and "returncode" in output
                            else None
                        ),
                        "output_chars": len(output_text),
                        "exception_type": (
                            type(error).__name__ if error is not None else None
                        ),
                    },
                ),
            )
        )

    def finish(self, *, outcome: str) -> None:
        ts_ms = self._timestamp()
        self.sink.emit_batch(
            (
                self._event(
                    RuntimeEventKind.RETURN,
                    ts_ms=ts_ms,
                    invocation_id=self.metadata.invocation_id,
                    context_id=self.metadata.context_id,
                    context_epoch=self.metadata.context_epoch,
                    attributes={"outcome": outcome},
                ),
                self._event(
                    RuntimeEventKind.WORKFLOW_END,
                    ts_ms=ts_ms,
                    attributes={"outcome": outcome},
                ),
            )
        )

    def request_metadata(self) -> dict[str, object]:
        return self.metadata.to_wire()

    def _timestamp(self) -> float:
        with self._lock:
            self._last_ts_ms = max(self._last_ts_ms, float(self.clock_ms()))
            return self._last_ts_ms

    def _event(
        self,
        kind: RuntimeEventKind,
        *,
        ts_ms: float,
        confidence: EventConfidence = EventConfidence.DECLARED_RUNTIME,
        **kwargs: Any,
    ) -> RuntimeEvent:
        with self._lock:
            self._sequence += 1
            event_id = (
                f"{self.metadata.root_workflow_id}:runtime:{self._sequence:06d}"
            )
        return RuntimeEvent(
            event_id=event_id,
            ts_ms=ts_ms,
            kind=kind,
            workflow_id=self.metadata.root_workflow_id,
            confidence=confidence,
            **kwargs,
        )


class InstrumentedToolEnvironment:
    """Transparent mini-agent environment wrapper with exact tool hooks."""

    def __init__(self, environment: Any, emitter: InvocationRuntimeEmitter) -> None:
        self.environment = environment
        self.emitter = emitter

    def execute(self, action: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        tool_call_id, start_ts_ms = self.emitter.tool_start(action)
        output = None
        error = None
        try:
            output = self.environment.execute(action, *args, **kwargs)
            return output
        except BaseException as caught:
            error = caught
            raise
        finally:
            self.emitter.tool_end(
                tool_call_id,
                start_ts_ms,
                output=output,
                error=error,
            )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self.environment.get_template_vars(**kwargs)

    def serialize(self) -> dict[str, Any]:
        return self.environment.serialize()

    def cleanup(self) -> None:
        cleanup = getattr(self.environment, "cleanup", None)
        if cleanup is not None:
            cleanup()
