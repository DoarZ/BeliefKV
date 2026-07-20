from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import inf
from typing import Any, Mapping


@dataclass(frozen=True, order=True)
class PageHandle:
    page_id: int
    allocation_generation: int

    def __post_init__(self) -> None:
        if self.page_id < 0:
            raise ValueError("page_id must be non-negative")
        if self.allocation_generation < 0:
            raise ValueError("allocation_generation must be non-negative")


class PhysicalResidency(str, Enum):
    GPU_ONLY = "gpu_only"
    MIRRORING = "mirroring"
    DUAL_CLEAN = "dual_clean"
    CPU_ONLY = "cpu_only"
    PREFETCHING = "prefetching"
    DEAD = "dead"


class TransferDirection(str, Enum):
    D2H = "d2h"
    H2D = "h2d"


class CommandKind(str, Enum):
    ADMIT_REQUEST = "admit_request"
    DEFER_REQUEST = "defer_request"
    OFFLOAD_CONTEXT = "offload_context"
    SHADOW_CONTEXT = "shadow_context"
    PREFETCH_CONTEXT = "prefetch_context"
    DROP_UNOWNED = "drop_unowned"
    PIN_CONTEXT = "pin_context"
    UNPIN_CONTEXT = "unpin_context"
    SET_WORKFLOW_BUDGET = "set_workflow_budget"


class CommandQueueClass(str, Enum):
    URGENT = "urgent"
    SHADOW = "shadow"


class CommandStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    REJECTED = "rejected"
    STALE = "stale"
    CANCELLED = "cancelled"


class TransferBlockerCode(str, Enum):
    """Machine-readable reason why a physical transfer could not proceed."""

    ANCESTOR_CLOSURE = "ancestor_closure"
    DESCENDANT_CLOSURE = "descendant_closure"
    DEVICE_CAPACITY = "device_capacity"
    HOST_CAPACITY = "host_capacity"
    ENGINE_BUSY = "engine_busy"
    NODE_LOCKED = "node_locked"
    NODE_LOADING = "node_loading"
    INFLIGHT = "inflight"
    SEMANTIC_PIN = "semantic_pin"
    UNSEALED = "unsealed"
    STALE_GENERATION = "stale_generation"
    EXTENT_MUTATED = "extent_mutated"
    UNKNOWN_BACKEND = "unknown_backend"


class PhysicalPageAction(str, Enum):
    START_D2H = "start_d2h"
    COMMIT_CPU = "commit_cpu"
    START_H2D = "start_h2d"
    DROP = "drop"
    PIN = "pin"
    UNPIN = "unpin"


@dataclass(frozen=True)
class ResolvedPageAction:
    handle: PageHandle
    action: PhysicalPageAction
    size_bytes: int

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError("resolved page action size must be positive")


@dataclass(frozen=True)
class PhysicalBundleIntent:
    """Versioned physical closure selected before command dispatch."""

    bundle_id: str
    closure_handles: tuple[PageHandle, ...]
    page_actions: tuple[ResolvedPageAction, ...]
    generation_fingerprint: str
    closure_bytes: int
    expected_reclaimable_bytes: int = 0
    locked_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.bundle_id or not self.generation_fingerprint:
            raise ValueError("physical bundle identity and fingerprint are required")
        if self.closure_bytes < 0 or self.expected_reclaimable_bytes < 0:
            raise ValueError("physical bundle byte counts must be non-negative")
        if self.locked_bytes < 0:
            raise ValueError("physical bundle locked bytes must be non-negative")
        if len(set(self.closure_handles)) != len(self.closure_handles):
            raise ValueError("physical bundle closure handles must be unique")
        action_handles = [item.handle for item in self.page_actions]
        if len(set(action_handles)) != len(action_handles):
            raise ValueError("physical bundle action handles must be unique")
        if not set(action_handles).issubset(self.closure_handles):
            raise ValueError("physical bundle actions must belong to its closure")
        if sum(item.size_bytes for item in self.page_actions) != self.closure_bytes:
            raise ValueError("physical bundle closure bytes must equal action bytes")


@dataclass(frozen=True)
class ControlCommand:
    command_id: str
    kind: CommandKind
    created_ts_ms: float
    context_id: str | None = None
    context_epoch: int | None = None
    target_bytes: int = 0
    priority: float = 0.0
    deadline_ms: float = inf
    queue_class: CommandQueueClass = CommandQueueClass.URGENT
    metadata: Mapping[str, Any] = field(default_factory=dict)
    physical_bundle: PhysicalBundleIntent | None = None

    def __post_init__(self) -> None:
        if not self.command_id:
            raise ValueError("command_id must be non-empty")
        if self.created_ts_ms < 0:
            raise ValueError("created_ts_ms must be non-negative")
        if self.target_bytes < 0:
            raise ValueError("target_bytes must be non-negative")
        if self.context_epoch is not None and self.context_epoch < 0:
            raise ValueError("context_epoch must be non-negative")

@dataclass(frozen=True)
class TransferBlocker:
    code: TransferBlockerCode
    page_handle: PageHandle | None = None
    required_bytes: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        if self.required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")


@dataclass(frozen=True)
class ResolvedCommand:
    command: ControlCommand
    page_actions: tuple[ResolvedPageAction, ...]
    resolved_bytes: int
    reason: str
    blockers: tuple[TransferBlocker, ...] = ()
    closure_fingerprint: str = ""


@dataclass(frozen=True)
class CommandAck:
    command_id: str
    status: CommandStatus
    completed_ts_ms: float
    actual_bytes: int
    page_handles: tuple[PageHandle, ...] = ()
    reason: str = ""
    blockers: tuple[TransferBlocker, ...] = ()

    def __post_init__(self) -> None:
        if self.actual_bytes < 0:
            raise ValueError("actual_bytes must be non-negative")


@dataclass(frozen=True)
class TransferTelemetry:
    """Performance observation for one physical KV transfer command.

    This record is deliberately separate from :class:`CommandAck`. ACKs are
    the correctness boundary for residency changes; telemetry may be missing
    optional performance fields without changing command semantics.
    """

    command_id: str
    submit_ts_ms: float
    start_ts_ms: float | None
    first_layer_ready_ts_ms: float | None
    complete_ts_ms: float
    compute_wait_ms: float | None
    actual_bytes: int
    closure_bytes: int
    merged_operation_count: int
    direction: TransferDirection
    source_tier: str
    target_tier: str
    status: CommandStatus
    reason: str = ""
    page_count: int = 0
    context_id: str | None = None
    context_epoch: int | None = None
    command_kind: str = ""
    compute_phase: str = "unknown"

    def __post_init__(self) -> None:
        if not self.command_id:
            raise ValueError("command_id must be non-empty")
        if self.submit_ts_ms < 0 or self.complete_ts_ms < 0:
            raise ValueError("transfer timestamps must be non-negative")
        if self.complete_ts_ms < self.submit_ts_ms:
            raise ValueError("complete_ts_ms cannot precede submit_ts_ms")
        if self.start_ts_ms is not None:
            if not self.submit_ts_ms <= self.start_ts_ms <= self.complete_ts_ms:
                raise ValueError("start_ts_ms must be within the transfer interval")
        if self.first_layer_ready_ts_ms is not None:
            lower = self.start_ts_ms or self.submit_ts_ms
            if not lower <= self.first_layer_ready_ts_ms <= self.complete_ts_ms:
                raise ValueError(
                    "first_layer_ready_ts_ms must be within the transfer interval"
                )
        if self.compute_wait_ms is not None and self.compute_wait_ms < 0:
            raise ValueError("compute_wait_ms must be non-negative when observed")
        if self.actual_bytes < 0 or self.closure_bytes < 0:
            raise ValueError("transfer byte counts must be non-negative")
        if self.actual_bytes > self.closure_bytes:
            raise ValueError("actual_bytes cannot exceed selected closure_bytes")
        if self.merged_operation_count < 0 or self.page_count < 0:
            raise ValueError("operation and page counts must be non-negative")
        if not self.source_tier or not self.target_tier:
            raise ValueError("source_tier and target_tier must be non-empty")
        if self.context_epoch is not None and self.context_epoch < 0:
            raise ValueError("context_epoch must be non-negative")
