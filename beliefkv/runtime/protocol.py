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


class PhysicalPageAction(str, Enum):
    START_D2H = "start_d2h"
    COMMIT_CPU = "commit_cpu"
    START_H2D = "start_h2d"
    DROP = "drop"
    PIN = "pin"
    UNPIN = "unpin"


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
class ResolvedPageAction:
    handle: PageHandle
    action: PhysicalPageAction
    size_bytes: int


@dataclass(frozen=True)
class ResolvedCommand:
    command: ControlCommand
    page_actions: tuple[ResolvedPageAction, ...]
    resolved_bytes: int
    reason: str


@dataclass(frozen=True)
class CommandAck:
    command_id: str
    status: CommandStatus
    completed_ts_ms: float
    actual_bytes: int
    page_handles: tuple[PageHandle, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.actual_bytes < 0:
            raise ValueError("actual_bytes must be non-negative")
