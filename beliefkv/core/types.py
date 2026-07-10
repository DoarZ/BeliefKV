from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import inf
from typing import Optional


class DeviceState(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    LOADING = "loading"
    OFFLOADING = "offloading"
    RECOMPUTE_ONLY = "recompute_only"
    RAW_TEXT = "raw_text"


class KVAction(str, Enum):
    KEEP_GPU = "keep_gpu"
    OFFLOAD_CPU = "offload_cpu"
    PREFETCH_GPU = "prefetch_gpu"
    DROP_GPU = "drop_gpu"
    MATERIALIZE = "materialize"
    RECOMPUTE_LATER = "recompute_later"
    NOOP = "noop"


@dataclass(frozen=True)
class ContinuationBelief:
    workflow_id: str
    agent_id: str
    branch_id: Optional[str]
    probability: float
    ready_time_p50_ms: float
    ready_time_p95_ms: float
    expected_prompt_delta_tokens: int = 0
    expected_output_tokens: int = 0
    branch_survival_prob: float = 1.0
    confidence: float = 1.0

    @property
    def effective_probability(self) -> float:
        return max(0.0, min(1.0, self.probability * self.branch_survival_prob))


@dataclass
class WorkflowState:
    workflow_id: str
    active_agents: set[str] = field(default_factory=set)
    active_branches: set[str] = field(default_factory=set)
    slo_deadline_ms: Optional[float] = None
    attained_service_ms: float = 0.0


@dataclass(frozen=True)
class KVObjectMeta:
    object_id: str
    workflow_ids: frozenset[str]
    agent_ids: frozenset[str]
    branch_ids: frozenset[str]
    token_count: int
    size_bytes: int
    device_state: DeviceState
    is_shared_prefix: bool = False
    is_branch_delta: bool = False
    is_active_decode: bool = False
    last_access_ms: float = 0.0
    recompute_cost_ms: Optional[float] = None
    d2h_cost_ms: Optional[float] = None
    h2d_cost_ms: Optional[float] = None


@dataclass(frozen=True)
class RuntimeSnapshot:
    now_ms: float
    hbm_capacity_bytes: int
    hbm_used_bytes: int
    active_decode_workflows: frozenset[str] = frozenset()

    @property
    def hbm_pressure(self) -> float:
        if self.hbm_capacity_bytes <= 0:
            return 1.0
        return max(0.0, min(1.0, self.hbm_used_bytes / self.hbm_capacity_bytes))

    @property
    def hbm_free_bytes(self) -> int:
        return max(0, self.hbm_capacity_bytes - self.hbm_used_bytes)


@dataclass(frozen=True)
class PlannerConfig:
    reserve_hbm_bytes: int = 1 << 30
    prefill_tokens_per_ms: float = 80.0
    pcie_bandwidth_gbps: float = 24.0
    transfer_overhead_ms: float = 0.08
    decode_protection_ms: float = 80.0
    prefetch_slack_ms: float = 25.0
    offload_min_benefit_ms: float = 2.0
    default_reuse_probability: float = 0.05
    min_branch_probability: float = 0.03
    high_hbm_pressure: float = 0.82


@dataclass(frozen=True)
class KVDecision:
    object_id: str
    action: KVAction
    reason: str
    priority: float = 0.0
    next_use_p50_ms: float = inf
    reuse_probability: float = 0.0
    expected_benefit_ms: float = 0.0
