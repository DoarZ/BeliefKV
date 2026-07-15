from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from beliefkv.core.ids import require_id


class RuntimeEventKind(str, Enum):
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    INVOCATION_CREATE = "invocation_create"
    INVOCATION_CANCEL = "invocation_cancel"
    CONTEXT_ADVANCE = "context_advance"
    CALL = "call"
    SPAWN = "spawn"
    RETURN = "return"
    MESSAGE = "message"
    HANDOFF = "handoff"
    JOIN_CREATE = "join_create"
    JOIN_WAIT = "join_wait"
    JOIN_SATISFIED = "join_satisfied"
    JOIN_TIMEOUT = "join_timeout"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    LLM_SUBMIT = "llm_submit"
    LLM_RESULT = "llm_result"


class RelationType(str, Enum):
    ROOT = "root"
    CALL = "call"
    SPAWN = "spawn"
    MESSAGE = "message"
    HANDOFF = "handoff"


class ContextMode(str, Enum):
    FRESH = "fresh"
    FORK = "fork"
    RESUME = "resume"


class ExecutionMode(str, Enum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"


class EventConfidence(str, Enum):
    DECLARED_RUNTIME = "declared_runtime"
    OBSERVED_EXACT = "observed_exact"
    INFERRED = "inferred"


@dataclass(frozen=True)
class RuntimeEvent:
    """Framework-neutral event consumed by the BeliefKV control plane.

    The common identity fields remain explicit so adapters cannot silently hide
    causal identity in an unvalidated payload. Event-specific data belongs in
    ``attributes``.
    """

    event_id: str
    ts_ms: float
    kind: RuntimeEventKind
    workflow_id: str
    invocation_id: str | None = None
    target_invocation_id: str | None = None
    context_id: str | None = None
    target_context_id: str | None = None
    parent_invocation_id: str | None = None
    parent_context_id: str | None = None
    agent_definition_id: str | None = None
    agent_instance_id: str | None = None
    relation_type: RelationType | None = None
    context_mode: ContextMode | None = None
    execution_mode: ExecutionMode | None = None
    return_target_id: str | None = None
    join_id: str | None = None
    member_invocation_ids: tuple[str, ...] = ()
    context_epoch: int | None = None
    confidence: EventConfidence = EventConfidence.DECLARED_RUNTIME
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.event_id, "event_id")
        require_id(self.workflow_id, "workflow_id")
        if self.ts_ms < 0:
            raise ValueError("ts_ms must be non-negative")
        if self.context_epoch is not None and self.context_epoch < 0:
            raise ValueError("context_epoch must be non-negative")
        if len(set(self.member_invocation_ids)) != len(self.member_invocation_ids):
            raise ValueError("member_invocation_ids must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ts_ms": self.ts_ms,
            "kind": self.kind.value,
            "workflow_id": self.workflow_id,
            "invocation_id": self.invocation_id,
            "target_invocation_id": self.target_invocation_id,
            "context_id": self.context_id,
            "target_context_id": self.target_context_id,
            "parent_invocation_id": self.parent_invocation_id,
            "parent_context_id": self.parent_context_id,
            "agent_definition_id": self.agent_definition_id,
            "agent_instance_id": self.agent_instance_id,
            "relation_type": (
                self.relation_type.value if self.relation_type is not None else None
            ),
            "context_mode": (
                self.context_mode.value if self.context_mode is not None else None
            ),
            "execution_mode": (
                self.execution_mode.value if self.execution_mode is not None else None
            ),
            "return_target_id": self.return_target_id,
            "join_id": self.join_id,
            "member_invocation_ids": list(self.member_invocation_ids),
            "context_epoch": self.context_epoch,
            "confidence": self.confidence.value,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimeEvent":
        known = {
            "event_id",
            "ts_ms",
            "kind",
            "workflow_id",
            "invocation_id",
            "target_invocation_id",
            "context_id",
            "target_context_id",
            "parent_invocation_id",
            "parent_context_id",
            "agent_definition_id",
            "agent_instance_id",
            "relation_type",
            "context_mode",
            "execution_mode",
            "return_target_id",
            "join_id",
            "member_invocation_ids",
            "context_epoch",
            "confidence",
            "attributes",
            "schema_version",
            "sequence",
        }
        attributes = dict(raw.get("attributes", {}))
        attributes.update({key: value for key, value in raw.items() if key not in known})
        return cls(
            event_id=str(raw["event_id"]),
            ts_ms=float(raw["ts_ms"]),
            kind=RuntimeEventKind(raw["kind"]),
            workflow_id=str(raw["workflow_id"]),
            invocation_id=raw.get("invocation_id"),
            target_invocation_id=raw.get("target_invocation_id"),
            context_id=raw.get("context_id"),
            target_context_id=raw.get("target_context_id"),
            parent_invocation_id=raw.get("parent_invocation_id"),
            parent_context_id=raw.get("parent_context_id"),
            agent_definition_id=raw.get("agent_definition_id"),
            agent_instance_id=raw.get("agent_instance_id"),
            relation_type=(
                RelationType(raw["relation_type"])
                if raw.get("relation_type") is not None
                else None
            ),
            context_mode=(
                ContextMode(raw["context_mode"])
                if raw.get("context_mode") is not None
                else None
            ),
            execution_mode=(
                ExecutionMode(raw["execution_mode"])
                if raw.get("execution_mode") is not None
                else None
            ),
            return_target_id=raw.get("return_target_id"),
            join_id=raw.get("join_id"),
            member_invocation_ids=tuple(raw.get("member_invocation_ids", ())),
            context_epoch=(
                int(raw["context_epoch"])
                if raw.get("context_epoch") is not None
                else None
            ),
            confidence=EventConfidence(
                raw.get("confidence", EventConfidence.DECLARED_RUNTIME.value)
            ),
            attributes=attributes,
        )
