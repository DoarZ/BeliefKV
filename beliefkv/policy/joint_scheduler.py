from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from beliefkv.policy.reference import (
    AdmissionAction,
    AdmissionIntent,
    ExecutionIntent,
    MetadataMode,
    MetadataRequirement,
    PhysicalBundleSnapshot,
    PolicyInput,
    PolicyOutput,
    ResidencyAction,
    ResidencyIntent,
    RunnableInvocation,
    TransferDependency,
)
from beliefkv.policy.scenario_physicalizer import (
    FrontierScenario,
    PreparedPolicyInput,
    ScenarioDemand,
    ScenarioPhysicalizer,
    ScenarioTransition,
)
from beliefkv.policy.whatif_packer import (
    FairnessWindow,
    ScenarioPlan,
    WhatIfPacker,
    WhatIfPackerConfig,
)


OBSERVED_SCENARIO_ID = "observed-runtime-frontier"


class JointPlannerMode(str, Enum):
    BOUNDED_SEED = "bounded_seed"
    OPTIMIZED = "optimized"
    EMERGENCY = "emergency"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class SemanticResidencyTarget:
    """Stable residency goal resolved to Radix extents at a scheduler safe point."""

    context_id: str
    context_epoch: int
    action: ResidencyAction
    target_bytes_hint: int
    deadline_ms: float
    reason: str

    def __post_init__(self) -> None:
        if not self.context_id or not self.reason:
            raise ValueError("semantic residency target identity must be non-empty")
        if self.context_epoch < 0 or self.target_bytes_hint < 0:
            raise ValueError("semantic residency target values must be non-negative")
        if not math.isfinite(self.deadline_ms) or self.deadline_ms < 0:
            raise ValueError("semantic residency deadline must be non-negative")
        object.__setattr__(self, "action", ResidencyAction(self.action))
        if self.action == ResidencyAction.KEEP:
            raise ValueError("semantic residency targets must request a state change")

    def to_dict(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "context_epoch": self.context_epoch,
            "action": self.action.value,
            "target_bytes_hint": self.target_bytes_hint,
            "deadline_ms": self.deadline_ms,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RetractionIntent:
    request_id: str
    context_id: str
    context_epoch: int
    reason: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.context_id or not self.reason:
            raise ValueError("retraction intent identity must be non-empty")
        if self.context_epoch < 0:
            raise ValueError("retraction intent context epoch must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "context_id": self.context_id,
            "context_epoch": self.context_epoch,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JointPlannerConfig:
    fairness_lag_budget_ms: float = 50.0
    max_workflow_candidates: int = 8
    max_frontier_candidates_per_workflow: int = 4
    max_total_frontier_candidates: int = 16
    max_package_evaluations: int = 8
    max_planning_budget_ms: float = 1.0
    min_planning_budget_ms: float = 0.25
    trigger_interval_budget_fraction: float = 0.5
    max_plan_age_ms: float = 100.0
    residency_hysteresis_ms: float = 100.0
    emergency_hbm_ratio: float = 0.98
    memory_penalty_ms: float = 5.0
    allow_recompute: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_workflow_candidates",
            "max_frontier_candidates_per_workflow",
            "max_total_frontier_candidates",
            "max_package_evaluations",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "fairness_lag_budget_ms",
            "max_planning_budget_ms",
            "min_planning_budget_ms",
            "max_plan_age_ms",
            "residency_hysteresis_ms",
            "memory_penalty_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_planning_budget_ms == 0 or self.max_plan_age_ms == 0:
            raise ValueError("planning budget and plan age must be positive")
        if self.min_planning_budget_ms > self.max_planning_budget_ms:
            raise ValueError("minimum planning budget cannot exceed maximum")
        if not 0 < self.trigger_interval_budget_fraction <= 1:
            raise ValueError("trigger interval budget fraction must be in (0, 1]")
        if not 0 < self.emergency_hbm_ratio <= 1:
            raise ValueError("emergency_hbm_ratio must be in (0, 1]")


@dataclass(frozen=True)
class ResourceFeasibilityCertificate:
    required_headroom_bytes: int
    planned_reclaim_bytes: int
    required_host_bytes: int
    planned_h2d_bytes: int

    def __post_init__(self) -> None:
        if min(
            self.required_headroom_bytes,
            self.planned_reclaim_bytes,
            self.required_host_bytes,
            self.planned_h2d_bytes,
        ) < 0:
            raise ValueError("resource certificate bytes must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PlanReadSet:
    lineage_snapshot_id: str
    request_fingerprints: Mapping[str, str]
    request_startup_bytes: Mapping[str, int]
    workflow_frontier_fingerprints: Mapping[str, str]
    context_epochs: Mapping[str, int]
    invocation_fingerprints: Mapping[str, str]
    join_fingerprints: Mapping[str, str]
    transition_generations: Mapping[str, int]
    touched_bundle_generations: Mapping[str, str]
    touched_bundle_actions: Mapping[str, str]
    fairness_revision: int
    fairness_order: tuple[str, ...]
    fairness_lag_budget_ms: float
    fairness_memory_penalty_ms: float
    fairness_max_workflow_candidates: int
    transfer_epoch: int
    expires_at_ms: float
    resource_certificate: ResourceFeasibilityCertificate

    def __post_init__(self) -> None:
        if not self.lineage_snapshot_id:
            raise ValueError("read-set lineage snapshot must be non-empty")
        if min(self.fairness_revision, self.transfer_epoch) < 0:
            raise ValueError("read-set revisions must be non-negative")
        if not math.isfinite(self.expires_at_ms) or self.expires_at_ms < 0:
            raise ValueError("read-set expiry must be finite and non-negative")
        for field_name in (
            "request_fingerprints",
            "request_startup_bytes",
            "workflow_frontier_fingerprints",
            "context_epochs",
            "invocation_fingerprints",
            "join_fingerprints",
            "transition_generations",
            "touched_bundle_generations",
            "touched_bundle_actions",
        ):
            values = dict(sorted(getattr(self, field_name).items()))
            if any(not str(key) for key in values):
                raise ValueError(f"{field_name} keys must be non-empty")
            object.__setattr__(self, field_name, MappingProxyType(values))
        object.__setattr__(self, "fairness_order", tuple(self.fairness_order))
        if any(value < 0 for value in self.request_startup_bytes.values()):
            raise ValueError("request startup bytes must be non-negative")
        if len(self.fairness_order) != len(set(self.fairness_order)):
            raise ValueError("fairness order must contain unique workflows")
        if any(not workflow_id for workflow_id in self.fairness_order):
            raise ValueError("fairness workflow IDs must be non-empty")
        if (
            not math.isfinite(self.fairness_lag_budget_ms)
            or self.fairness_lag_budget_ms < 0
            or not math.isfinite(self.fairness_memory_penalty_ms)
            or self.fairness_memory_penalty_ms < 0
        ):
            raise ValueError("fairness validation parameters are invalid")
        if self.fairness_max_workflow_candidates <= 0:
            raise ValueError("fairness candidate limit must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "lineage_snapshot_id": self.lineage_snapshot_id,
            "request_fingerprints": dict(self.request_fingerprints),
            "request_startup_bytes": dict(self.request_startup_bytes),
            "workflow_frontier_fingerprints": dict(
                self.workflow_frontier_fingerprints
            ),
            "context_epochs": dict(self.context_epochs),
            "invocation_fingerprints": dict(self.invocation_fingerprints),
            "join_fingerprints": dict(self.join_fingerprints),
            "transition_generations": dict(self.transition_generations),
            "touched_bundle_generations": dict(
                self.touched_bundle_generations
            ),
            "touched_bundle_actions": dict(self.touched_bundle_actions),
            "fairness_revision": self.fairness_revision,
            "fairness_order": list(self.fairness_order),
            "fairness_lag_budget_ms": self.fairness_lag_budget_ms,
            "fairness_memory_penalty_ms": self.fairness_memory_penalty_ms,
            "fairness_max_workflow_candidates": (
                self.fairness_max_workflow_candidates
            ),
            "transfer_epoch": self.transfer_epoch,
            "expires_at_ms": self.expires_at_ms,
            "resource_certificate": self.resource_certificate.to_dict(),
        }


@dataclass(frozen=True)
class JointPlanValidation:
    strict_global_reasons: tuple[str, ...]
    readset_conflict_reasons: tuple[str, ...]

    @property
    def strict_global_fresh(self) -> bool:
        return not self.strict_global_reasons

    @property
    def readset_fresh(self) -> bool:
        return not self.readset_conflict_reasons

    def to_dict(self) -> dict[str, object]:
        return {
            "strict_global_fresh": self.strict_global_fresh,
            "readset_fresh": self.readset_fresh,
            "strict_global_reasons": list(self.strict_global_reasons),
            "readset_conflict_reasons": list(self.readset_conflict_reasons),
        }


@dataclass(frozen=True)
class IntentValidation:
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))

    @property
    def valid(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class JointPlanCurrentState:
    """Bounded live state read for one plan at a scheduler safe point."""

    now_ms: float
    runnable_frontier: tuple[RunnableInvocation, ...]
    invocation_snapshots: Mapping[str, Mapping[str, object]]
    join_snapshots: Mapping[str, Mapping[str, object]]
    transitions: Mapping[str, Mapping[str, object]]
    fairness_revision: int
    fairness_accounts: Mapping[str, Mapping[str, object]]
    workflow_memory_charges: Mapping[str, float]
    transfer_epoch: int
    hbm_capacity_bytes: int
    hbm_available_bytes: int
    host_free_bytes: int
    bundle_snapshots: Mapping[str, PhysicalBundleSnapshot | None]
    strict_global_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.now_ms) or self.now_ms < 0:
            raise ValueError("current-state timestamp must be non-negative")
        if min(
            self.fairness_revision,
            self.transfer_epoch,
            self.hbm_capacity_bytes,
            self.hbm_available_bytes,
            self.host_free_bytes,
        ) < 0:
            raise ValueError("current-state revisions and bytes must be non-negative")
        if self.hbm_capacity_bytes <= 0:
            raise ValueError("current-state HBM capacity must be positive")
        object.__setattr__(
            self,
            "runnable_frontier",
            tuple(sorted(self.runnable_frontier, key=lambda item: item.request_id)),
        )
        for name in (
            "invocation_snapshots",
            "join_snapshots",
            "transitions",
            "fairness_accounts",
            "workflow_memory_charges",
            "bundle_snapshots",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(sorted(getattr(self, name).items()))),
            )
        object.__setattr__(
            self,
            "strict_global_reasons",
            tuple(sorted(set(self.strict_global_reasons))),
        )


@dataclass(frozen=True)
class JointPlanComponentValidation:
    strict_global_reasons: tuple[str, ...]
    global_reasons: tuple[str, ...]
    execution: IntentValidation
    admissions: Mapping[str, IntentValidation]
    residency: Mapping[str, IntentValidation]
    dependencies: tuple[IntentValidation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strict_global_reasons",
            tuple(sorted(set(self.strict_global_reasons))),
        )
        object.__setattr__(
            self, "global_reasons", tuple(sorted(set(self.global_reasons)))
        )
        object.__setattr__(
            self,
            "admissions",
            MappingProxyType(dict(sorted(self.admissions.items()))),
        )
        object.__setattr__(
            self,
            "residency",
            MappingProxyType(dict(sorted(self.residency.items()))),
        )
        object.__setattr__(self, "dependencies", tuple(self.dependencies))

    @property
    def fully_fresh(self) -> bool:
        return (
            not self.global_reasons
            and self.execution.valid
            and all(item.valid for item in self.admissions.values())
            and all(item.valid for item in self.residency.values())
            and all(item.valid for item in self.dependencies)
        )

    @property
    def readset_conflict_reasons(self) -> tuple[str, ...]:
        reasons = list(self.global_reasons)
        reasons.extend(f"execution:{item}" for item in self.execution.reasons)
        for request_id, validation in self.admissions.items():
            reasons.extend(
                f"admission:{request_id}:{item}" for item in validation.reasons
            )
        for bundle_id, validation in self.residency.items():
            reasons.extend(
                f"residency:{bundle_id}:{item}" for item in validation.reasons
            )
        for index, validation in enumerate(self.dependencies):
            reasons.extend(
                f"dependency:{index}:{item}" for item in validation.reasons
            )
        return tuple(sorted(set(reasons)))

    @property
    def has_fresh_component(self) -> bool:
        if self.global_reasons:
            return False
        return (
            self.execution.valid
            or any(item.valid for item in self.admissions.values())
            or any(item.valid for item in self.residency.values())
        )

    @property
    def partially_fresh(self) -> bool:
        return self.has_fresh_component and not self.fully_fresh

    def to_dict(self) -> dict[str, object]:
        return {
            "strict_global_fresh": not self.strict_global_reasons,
            "strict_global_reasons": list(self.strict_global_reasons),
            "global_reasons": list(self.global_reasons),
            "fully_fresh": self.fully_fresh,
            "partially_fresh": self.partially_fresh,
            "readset_conflict_reasons": list(self.readset_conflict_reasons),
            "execution": self.execution.to_dict(),
            "admissions": {
                key: value.to_dict() for key, value in self.admissions.items()
            },
            "residency": {
                key: value.to_dict() for key, value in self.residency.items()
            },
            "dependencies": [item.to_dict() for item in self.dependencies],
        }


@dataclass(frozen=True)
class JointPlan:
    plan_id: str
    input_snapshot_id: str
    generated_ts_ms: float
    execution: ExecutionIntent
    admissions: tuple[AdmissionIntent, ...]
    residency: tuple[ResidencyIntent, ...]
    dependencies: tuple[TransferDependency, ...]
    expected_hbm_peak_bytes: int
    expected_unhidden_stall_ms: float
    topology_version: int
    allocator_version: int
    read_set: PlanReadSet
    semantic_residency: tuple[SemanticResidencyTarget, ...] = ()
    retractions: tuple[RetractionIntent, ...] = ()
    planner_mode: JointPlannerMode = JointPlannerMode.OPTIMIZED
    fallback_reason: str | None = None
    candidate_count: int = 0
    evaluated_package_count: int = 0
    planning_ms: float = 0.0
    transition_open: bool = False
    search_complete: bool = True
    planning_termination_reason: str = "complete"
    planning_phase_ms: tuple[tuple[str, float], ...] = ()
    prediction_used: bool = False
    prediction_influence: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id or not self.input_snapshot_id:
            raise ValueError("joint plan IDs must be non-empty")
        if not math.isfinite(self.generated_ts_ms) or self.generated_ts_ms < 0:
            raise ValueError("joint plan timestamp must be finite and non-negative")
        if min(
            self.expected_hbm_peak_bytes,
            self.topology_version,
            self.allocator_version,
            self.candidate_count,
            self.evaluated_package_count,
        ) < 0:
            raise ValueError("joint plan counters must be non-negative")
        object.__setattr__(self, "planner_mode", JointPlannerMode(self.planner_mode))
        if (
            not math.isfinite(self.expected_unhidden_stall_ms)
            or self.expected_unhidden_stall_ms < 0
            or not math.isfinite(self.planning_ms)
            or self.planning_ms < 0
        ):
            raise ValueError("joint plan timing values must be finite and non-negative")
        if self.read_set.lineage_snapshot_id != self.input_snapshot_id:
            raise ValueError("joint plan and read-set lineage differ")
        if not self.planning_termination_reason:
            raise ValueError("planning termination reason must be non-empty")
        phase_names = [name for name, _ in self.planning_phase_ms]
        if (
            len(phase_names) != len(set(phase_names))
            or any(not name for name in phase_names)
            or any(not math.isfinite(value) or value < 0 for _, value in self.planning_phase_ms)
        ):
            raise ValueError("planning phase timings must be unique and non-negative")
        admission_ids = [item.request_id for item in self.admissions]
        residency_ids = [item.bundle_id for item in self.residency]
        semantic_ids = [
            (item.context_id, item.context_epoch, item.action)
            for item in self.semantic_residency
        ]
        retraction_ids = [item.request_id for item in self.retractions]
        if len(admission_ids) != len(set(admission_ids)):
            raise ValueError("joint plan admissions must be unique")
        if len(residency_ids) != len(set(residency_ids)):
            raise ValueError("joint plan residency intents must be unique")
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("joint plan semantic residency targets must be unique")
        if len(retraction_ids) != len(set(retraction_ids)):
            raise ValueError("joint plan retraction intents must be unique")
        object.__setattr__(
            self,
            "prediction_influence",
            tuple(sorted(self.prediction_influence)),
        )
        influence_names = [name for name, _ in self.prediction_influence]
        if len(influence_names) != len(set(influence_names)):
            raise ValueError("joint plan prediction influence names must be unique")
        if any(not name or value < 0 for name, value in self.prediction_influence):
            raise ValueError(
                "joint plan prediction influence must use non-empty names "
                "and non-negative counts"
            )
        if self.prediction_used and not self.prediction_influence:
            raise ValueError(
                "joint plan using predictions must carry influence telemetry"
            )
        if self.residency and self.semantic_residency:
            raise ValueError(
                "a JointPlan cannot mix semantic and physical residency intents"
            )
        for dependency in self.dependencies:
            if dependency.residency_intent_index >= len(self.residency):
                raise ValueError("joint plan dependency references a missing intent")

    def to_policy_output(self, *, policy_name: str) -> PolicyOutput:
        return PolicyOutput(
            execution=self.execution,
            admissions=self.admissions,
            residency=self.residency,
            dependencies=self.dependencies,
            policy_name=policy_name,
            metadata_assumptions=(
                "observed:physical_kv",
                "observed:resources",
                "observed:runnable_frontier",
                "observed:runtime_graph",
            ),
            policy_state_updates={
                "joint_plan": {
                    "plan_id": self.plan_id,
                    "topology_version": self.topology_version,
                    "allocator_version": self.allocator_version,
                    "read_set": self.read_set.to_dict(),
                    "expected_hbm_peak_bytes": self.expected_hbm_peak_bytes,
                    "expected_unhidden_stall_ms": self.expected_unhidden_stall_ms,
                    "fallback_reason": self.fallback_reason,
                    "planner_mode": self.planner_mode.value,
                    "candidate_count": self.candidate_count,
                    "transition_open": self.transition_open,
                    "search_complete": self.search_complete,
                    "planning_termination_reason": (
                        self.planning_termination_reason
                    ),
                }
            },
            input_snapshot_id=self.input_snapshot_id,
            metadata_mode=MetadataMode.ONLINE,
            shadow_only=True,
        )

    @property
    def bundle_generations(self) -> Mapping[str, str]:
        return self.read_set.touched_bundle_generations

    def strict_global_stale_reasons(
        self, policy_input: PolicyInput
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.input_snapshot_id != policy_input.snapshot_id:
            reasons.append("snapshot_id")
        if self.execution.graph_version != policy_input.runtime_graph.graph_version:
            reasons.append("graph_version")
        if self.topology_version != policy_input.physical_kv.topology_version:
            reasons.append("topology_version")
        if self.allocator_version != policy_input.physical_kv.allocator_version:
            reasons.append("allocator_version")
        return tuple(reasons)

    def stale_reasons(self, policy_input: PolicyInput) -> tuple[str, ...]:
        return validate_joint_plan(self, policy_input).readset_conflict_reasons

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "input_snapshot_id": self.input_snapshot_id,
            "generated_ts_ms": self.generated_ts_ms,
            "execution": self.execution.to_dict(),
            "admissions": [item.to_dict() for item in self.admissions],
            "residency": [item.to_dict() for item in self.residency],
            "semantic_residency": [
                item.to_dict() for item in self.semantic_residency
            ],
            "retractions": [item.to_dict() for item in self.retractions],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "expected_hbm_peak_bytes": self.expected_hbm_peak_bytes,
            "expected_unhidden_stall_ms": self.expected_unhidden_stall_ms,
            "topology_version": self.topology_version,
            "allocator_version": self.allocator_version,
            "read_set": self.read_set.to_dict(),
            "planner_mode": self.planner_mode.value,
            "fallback_reason": self.fallback_reason,
            "candidate_count": self.candidate_count,
            "evaluated_package_count": self.evaluated_package_count,
            "planning_ms": self.planning_ms,
            "transition_open": self.transition_open,
            "search_complete": self.search_complete,
            "planning_termination_reason": self.planning_termination_reason,
            "planning_phase_ms": dict(self.planning_phase_ms),
            "prediction_used": self.prediction_used,
            "prediction_influence": dict(self.prediction_influence),
        }


@dataclass(frozen=True)
class _Candidate:
    request: RunnableInvocation
    causal_rank: int
    unblock_depth: int
    pending_messages: int
    invocation_state: str
    relation_type: str
    context_mode: str
    execution_mode: str
    predicted_remaining_decode_tokens: float | None = None
    predicted_external_wait_ms: float | None = None
    predicted_next_output_tokens: float | None = None
    prediction_support_level: str = ""

    @property
    def prediction_order_key(self) -> tuple[int, float]:
        tokens = self.predicted_remaining_decode_tokens
        if tokens is not None and self.prediction_support_level != "unavailable":
            return (0, float(tokens))
        return (1, self.request.submitted_ts_ms)

    @property
    def observed_order_key(self) -> tuple[object, ...]:
        return (
            self.causal_rank,
            -self.unblock_depth,
            -self.pending_messages,
            (1, self.request.submitted_ts_ms),
            self.request.invocation_id,
            self.request.request_id,
        )

    @property
    def order_key(self) -> tuple[object, ...]:
        return (
            self.causal_rank,
            -self.unblock_depth,
            -self.pending_messages,
            self.prediction_order_key,
            self.request.submitted_ts_ms,
            self.request.invocation_id,
            self.request.request_id,
        )


class ObservedJointPlanner:
    """Bounded P4 planner using only observed graph and physical state.

    The planner is intentionally side-effect free. It emits a complete shadow
    plan and never mutates admission, waiting queues, or physical residency.
    """

    name = "belief_joint_observed"

    def __init__(
        self,
        config: JointPlannerConfig | None = None,
        *,
        physicalizer: ScenarioPhysicalizer | None = None,
        packer: WhatIfPacker | None = None,
    ) -> None:
        self.config = config or JointPlannerConfig()
        self.physicalizer = physicalizer or ScenarioPhysicalizer()
        self.packer = packer or WhatIfPacker(
            WhatIfPackerConfig(
                max_frontier_per_workflow=(
                    self.config.max_frontier_candidates_per_workflow
                ),
                handoff_hysteresis_ms=self.config.residency_hysteresis_ms,
                emergency_hbm_ratio=self.config.emergency_hbm_ratio,
                allow_recompute=self.config.allow_recompute,
            )
        )

    def metadata_requirements(
        self, mode: MetadataMode
    ) -> tuple[MetadataRequirement, ...]:
        del mode
        return ()

    def decide(self, policy_input: PolicyInput) -> PolicyOutput:
        return self.plan(policy_input).to_policy_output(policy_name=self.name)

    def plan(
        self,
        policy_input: PolicyInput,
        *,
        planning_budget_ms: float | None = None,
        cancel_check: object | None = None,
    ) -> JointPlan:
        budget_ms = (
            self.config.max_planning_budget_ms
            if planning_budget_ms is None
            else float(planning_budget_ms)
        )
        if not math.isfinite(budget_ms) or budget_ms <= 0:
            raise ValueError("planning budget must be finite and positive")
        budget_ms = min(self.config.max_planning_budget_ms, budget_ms)
        cancelled = cancel_check if callable(cancel_check) else lambda: False
        started_ns = time.perf_counter_ns()
        transition_open = _transition_open(policy_input)
        candidate_started_ns = time.perf_counter_ns()
        candidates, fairness, prediction_influence = self._ordered_candidates(
            policy_input
        )
        prediction_used = bool(prediction_influence)
        phase_ms = {
            "candidate_order": self._elapsed_ms(candidate_started_ns),
            "prepare": 0.0,
            "physicalize": 0.0,
            "pack": 0.0,
            "materialize": 0.0,
        }
        if transition_open:
            plan = self._fallback(
                policy_input,
                candidates,
                reason="transition_open_settling_barrier",
                started_ns=started_ns,
                transition_open=True,
                admit_fitting=False,
                prediction_used=prediction_used,
                prediction_influence=tuple(prediction_influence.items()),
            )
            return replace(plan, planning_phase_ms=_phase_timings(phase_ms))
        if not candidates:
            reason = (
                None
                if not policy_input.runnable_frontier
                else "no_factual_runnable_request"
            )
            plan = self._fallback(
                policy_input,
                candidates,
                reason=reason,
                started_ns=started_ns,
                transition_open=False,
                admit_fitting=False,
                prediction_used=prediction_used,
                prediction_influence=tuple(prediction_influence.items()),
            )
            return replace(plan, planning_phase_ms=_phase_timings(phase_ms))

        seed = self._bounded_seed(
            policy_input,
            candidates,
            started_ns=started_ns,
            prediction_used=prediction_used,
            prediction_influence=tuple(prediction_influence.items()),
        )

        prepare_started_ns = time.perf_counter_ns()
        try:
            prepared = self.physicalizer.prepare(policy_input)
        except (KeyError, TypeError, ValueError) as error:
            phase_ms["prepare"] = self._elapsed_ms(prepare_started_ns)
            return replace(
                seed,
                planning_ms=self._elapsed_ms(started_ns),
                search_complete=False,
                planning_termination_reason=(
                    f"optimization_prepare_error:{type(error).__name__}"
                ),
                planning_phase_ms=_phase_timings(phase_ms),
            )
        phase_ms["prepare"] = self._elapsed_ms(prepare_started_ns)
        if self._elapsed_ms(started_ns) >= budget_ms or cancelled():
            return replace(
                seed,
                planning_ms=self._elapsed_ms(started_ns),
                search_complete=False,
                planning_termination_reason=(
                    "superseded_seed_published"
                    if cancelled()
                    else "planning_budget_exceeded_seed_published"
                ),
                planning_phase_ms=_phase_timings(phase_ms),
            )
        search_started_ns = time.perf_counter_ns()
        evaluated = 0
        selected: tuple[ScenarioDemand, ScenarioPlan] | None = None
        low = 1
        high = len(candidates)
        search_complete = False
        budget_exhausted = False
        probed_maximum = False
        while low <= high and evaluated < self.config.max_package_evaluations:
            if (
                evaluated
                and self._elapsed_ms(search_started_ns)
                >= budget_ms
            ):
                budget_exhausted = True
                break
            if cancelled():
                budget_exhausted = True
                break
            # The first package establishes a publishable answer. Later probes
            # maximize the feasible prefix without sacrificing that answer when
            # the bounded search budget expires.
            if evaluated == 0:
                length = 1
            elif not probed_maximum:
                length = high
                probed_maximum = True
            else:
                length = (low + high) // 2
            request_ids = tuple(
                item.request.request_id for item in candidates[:length]
            )
            scenario = FrontierScenario(
                scenario_id=OBSERVED_SCENARIO_ID,
                probability=1.0,
                transition=self._transition(candidates[:length]),
                candidate_request_ids=request_ids,
            )
            try:
                phase_started_ns = time.perf_counter_ns()
                demand = self.physicalizer.physicalize(
                    policy_input,
                    scenario,
                    prepared=prepared,
                )
                phase_ms["physicalize"] += self._elapsed_ms(phase_started_ns)
                phase_started_ns = time.perf_counter_ns()
                scenario_plan = self.packer.pack(
                    policy_input,
                    demand,
                    fairness=fairness,
                    prepared=prepared,
                )
                phase_ms["pack"] += self._elapsed_ms(phase_started_ns)
            except (KeyError, TypeError, ValueError) as error:
                return replace(
                    seed,
                    planning_ms=self._elapsed_ms(started_ns),
                    search_complete=False,
                    evaluated_package_count=evaluated,
                    planning_termination_reason=(
                        f"optimization_error_seed_published:{type(error).__name__}"
                    ),
                    planning_phase_ms=_phase_timings(phase_ms),
                )
            evaluated += 1
            if scenario_plan.feasible:
                selected = demand, scenario_plan
                low = length + 1
                if length == len(candidates):
                    search_complete = True
                    break
            else:
                high = length - 1
                if length == 1:
                    search_complete = True
                    break

        if low > high:
            search_complete = True
        if (
            not search_complete
            and self._elapsed_ms(search_started_ns)
            >= budget_ms
        ):
            budget_exhausted = True
        if selected is None:
            reason = (
                "planning_budget_exceeded_without_feasible_prefix"
                if budget_exhausted
                else "no_physically_feasible_joint_package"
            )
            return replace(
                seed,
                planning_ms=self._elapsed_ms(started_ns),
                evaluated_package_count=evaluated,
                search_complete=search_complete,
                planning_termination_reason=f"{reason}_seed_published",
                planning_phase_ms=_phase_timings(phase_ms),
            )
        demand, scenario_plan = selected
        materialize_started_ns = time.perf_counter_ns()
        plan = self._from_scenario_plan(
            policy_input,
            demand,
            scenario_plan,
            prepared=prepared,
            candidate_count=len(candidates),
            evaluated_package_count=evaluated,
            prediction_used=prediction_used,
            prediction_influence=tuple(prediction_influence.items()),
        )
        phase_ms["materialize"] = self._elapsed_ms(materialize_started_ns)
        termination_reason = (
            "complete"
            if search_complete
            else "planning_budget_exceeded_best_feasible_prefix"
            if budget_exhausted
            else "package_evaluation_limit_best_feasible_prefix"
        )
        return replace(
            plan,
            planning_ms=self._elapsed_ms(started_ns),
            planner_mode=(
                JointPlannerMode.BOUNDED_SEED
                if not search_complete
                else JointPlannerMode.OPTIMIZED
            ),
            search_complete=search_complete,
            planning_termination_reason=termination_reason,
            planning_phase_ms=_phase_timings(phase_ms),
        )

    def trigger_budget_ms(self, trigger_interval_ms: float | None) -> float:
        if trigger_interval_ms is None or not math.isfinite(trigger_interval_ms):
            return self.config.min_planning_budget_ms
        return min(
            self.config.max_planning_budget_ms,
            max(
                self.config.min_planning_budget_ms,
                trigger_interval_ms
                * self.config.trigger_interval_budget_fraction,
            ),
        )

    def _ordered_candidates(
        self, policy_input: PolicyInput
    ) -> tuple[tuple[_Candidate, ...], FairnessWindow]:
        state = _mapping(policy_input.runtime_graph.state)
        rccg = _mapping(state.get("rccg"))
        invocations = _mapping(rccg.get("invocations"))
        joins = _mapping(rccg.get("joins"))
        join_stragglers = _join_stragglers(joins)
        candidates_by_workflow: dict[str, list[_Candidate]] = defaultdict(list)
        for request in policy_input.runnable_frontier:
            invocation = _mapping(invocations.get(request.invocation_id))
            invocation_state = str(invocation.get("state", "unknown"))
            pending_messages = _nonnegative_int(
                invocation.get("pending_messages", 0)
            )
            if not _is_factual_runnable(
                invocation_state,
                pending_messages=pending_messages,
                causal_class=request.causal_class,
            ):
                continue
            unblock_depth = _unblock_depth(
                request.invocation_id,
                invocations,
                joins,
            )
            execution_mode = str(
                invocation.get(
                    "execution_mode",
                    _causal_class_part(request.causal_class, 1, "foreground"),
                )
            )
            relation_type = str(
                invocation.get(
                    "relation_type",
                    _causal_class_part(request.causal_class, 2, "root"),
                )
            )
            context_mode = str(invocation.get("context_mode", "fresh"))
            if request.invocation_id in join_stragglers:
                causal_rank = 0
            elif unblock_depth > 0:
                causal_rank = 1
            elif pending_messages > 0 or relation_type in {"handoff", "message"}:
                causal_rank = 2
            elif execution_mode == "background":
                causal_rank = 4
            else:
                causal_rank = 3
            candidates_by_workflow[request.workflow_id].append(
                _Candidate(
                    request=request,
                    causal_rank=causal_rank,
                    unblock_depth=unblock_depth,
                    pending_messages=pending_messages,
                    invocation_state=invocation_state,
                    relation_type=relation_type,
                    context_mode=context_mode,
                    execution_mode=execution_mode,
                    predicted_remaining_decode_tokens=(
                        request.predicted_remaining_decode_tokens
                    ),
                    predicted_external_wait_ms=(
                        request.predicted_external_wait_ms
                    ),
                    predicted_next_output_tokens=(
                        request.predicted_next_output_tokens
                    ),
                    prediction_support_level=(
                        request.prediction_support_level
                    ),
                )
            )

        workflow_order, fairness = self._fair_workflow_order(
            policy_input,
            tuple(candidates_by_workflow),
        )
        influence: Counter[str] = Counter()
        for items in candidates_by_workflow.values():
            items.sort(key=lambda item: item.order_key)
            observed = sorted(items, key=lambda item: item.observed_order_key)
            predictive_index = {
                item.request.invocation_id: index
                for index, item in enumerate(items)
            }
            for index, item in enumerate(observed):
                if predictive_index.get(item.request.invocation_id) != index:
                    influence["ordering_changed"] += 1
        for items in candidates_by_workflow.values():
            for item in items:
                if item.prediction_support_level:
                    influence["prediction_available"] += 1
                    influence[f"support_{item.prediction_support_level}"] += 1
        ordered: list[_Candidate] = []
        for index in range(self.config.max_frontier_candidates_per_workflow):
            for workflow_id in workflow_order:
                items = candidates_by_workflow[workflow_id]
                if index < len(items):
                    ordered.append(items[index])
                    if len(ordered) >= self.config.max_total_frontier_candidates:
                        return tuple(ordered), fairness, influence
        return tuple(ordered), fairness, influence

    def _bounded_seed(
        self,
        policy_input: PolicyInput,
        candidates: Sequence[_Candidate],
        *,
        started_ns: int,
        prediction_used: bool = False,
        prediction_influence: tuple[tuple[str, int], ...] = (),
    ) -> JointPlan:
        """Build an O(frontier + bundles) plan before physical optimization."""

        cpu_resident_contexts = {
            context_id
            for bundle in policy_input.physical_kv.bundles
            if bundle.gpu_bytes < bundle.physical_unique_bytes
            for context_id in bundle.owner_context_ids
        }
        available = policy_input.resources.hbm_available_bytes
        admitted: list[RunnableInvocation] = []
        for candidate in candidates:
            request = candidate.request
            needed = _unreserved_startup_bytes(request)
            if request.context_id in cpu_resident_contexts or needed > available:
                continue
            admitted.append(request)
            available -= needed
        admitted_ids = {item.request_id for item in admitted}
        admissions = tuple(
            AdmissionIntent(
                request_id=request.request_id,
                action=(
                    AdmissionAction.ADMIT
                    if request.request_id in admitted_ids
                    else AdmissionAction.DEFER
                ),
                reserved_bytes=(
                    _unreserved_startup_bytes(request)
                    if request.request_id in admitted_ids
                    else 0
                ),
                required_bundle_ids=(),
                reason=(
                    "bounded seed selected current GPU-resident work"
                    if request.request_id in admitted_ids
                    else "bounded seed deferred work requiring optimization"
                ),
            )
            for request in policy_input.runnable_frontier
        )
        first = admitted[0] if admitted else None
        execution = ExecutionIntent(
            ordered_request_ids=tuple(item.request_id for item in admitted),
            selected_workflow_id=first.workflow_id if first else None,
            selected_invocation_id=first.invocation_id if first else None,
            mode="observed_joint_seed",
            graph_version=policy_input.runtime_graph.graph_version,
            reason="bounded semantic seed before physical package search",
        )
        projected = (
            policy_input.resources.hbm_used_bytes
            + policy_input.resources.hbm_reserved_bytes
            + sum(_unreserved_startup_bytes(item) for item in admitted)
        )
        plan = _make_plan(
            policy_input,
            execution=execution,
            admissions=admissions,
            residency=(),
            dependencies=(),
            expected_hbm_peak_bytes=projected,
            expected_unhidden_stall_ms=0.0,
            fallback_reason=None,
            candidate_count=len(candidates),
            evaluated_package_count=0,
            transition_open=False,
            max_plan_age_ms=self.config.max_plan_age_ms,
            fairness_lag_budget_ms=self.config.fairness_lag_budget_ms,
            fairness_memory_penalty_ms=self.config.memory_penalty_ms,
            fairness_max_workflow_candidates=self.config.max_workflow_candidates,
            prediction_used=prediction_used,
            prediction_influence=prediction_influence,
        )
        used_ratio = (
            policy_input.resources.hbm_used_bytes
            / policy_input.resources.hbm_capacity_bytes
        )
        return replace(
            plan,
            planner_mode=(
                JointPlannerMode.EMERGENCY
                if used_ratio >= self.config.emergency_hbm_ratio
                else JointPlannerMode.BOUNDED_SEED
            ),
            planning_ms=self._elapsed_ms(started_ns),
            search_complete=False,
            planning_termination_reason="bounded_seed",
        )

    def _fair_workflow_order(
        self,
        policy_input: PolicyInput,
        workflow_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], FairnessWindow]:
        state = _mapping(policy_input.runtime_graph.state)
        fairness_state = _mapping(state.get("workflow_fairness"))
        accounts = _mapping(fairness_state.get("accounts"))
        memory_charges = _mapping(
            fairness_state.get("memory_charges_bytes")
        )
        vruntime = {
            workflow_id: _nonnegative_float(
                _mapping(accounts.get(workflow_id)).get(
                    "virtual_runtime_ms", 0.0
                )
            )
            for workflow_id in workflow_ids
        }
        minimum = min(vruntime.values(), default=0.0)
        lag = {
            workflow_id: max(0.0, value - minimum)
            for workflow_id, value in vruntime.items()
        }
        within_lag = {
            workflow_id
            for workflow_id, value in lag.items()
            if value <= self.config.fairness_lag_budget_ms
        }

        def key(workflow_id: str) -> tuple[float, float, str]:
            memory = _nonnegative_float(memory_charges.get(workflow_id, 0.0))
            share = memory / policy_input.resources.hbm_capacity_bytes
            effective = (
                vruntime[workflow_id]
                + share * self.config.memory_penalty_ms
            )
            return effective, share, workflow_id

        ordered = tuple(
            sorted(within_lag, key=key)[: self.config.max_workflow_candidates]
        )
        eligible = frozenset(ordered)
        return ordered, FairnessWindow(
            eligible_workflow_ids=eligible,
            lag_ms_by_workflow={item: lag[item] for item in ordered},
            lag_budget_ms=self.config.fairness_lag_budget_ms,
        )

    @staticmethod
    def _transition(candidates: Sequence[_Candidate]) -> ScenarioTransition:
        first = candidates[0]
        if first.causal_rank <= 1:
            return ScenarioTransition.BLOCKING
        if first.relation_type == "handoff":
            return ScenarioTransition.HANDOFF
        if first.pending_messages > 0 or first.relation_type == "message":
            return ScenarioTransition.MULTI_CONSUMER
        if first.context_mode == "resume":
            return ScenarioTransition.CYCLIC_REACTIVATION
        return ScenarioTransition.NONBLOCKING

    def _from_scenario_plan(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        scenario_plan: ScenarioPlan,
        *,
        prepared: PreparedPolicyInput,
        candidate_count: int,
        evaluated_package_count: int,
        prediction_used: bool = False,
        prediction_influence: tuple[tuple[str, int], ...] = (),
    ) -> JointPlan:
        request_by_id = prepared.request_by_id
        residency = tuple(
            ResidencyIntent(
                bundle_id=bundle_id,
                action=action,
                target_bytes=_residency_target_bytes(prepared.bundle_by_id[bundle_id], action),
                deadline_ms=policy_input.resources.ts_ms,
                scenario_support=frozenset({OBSERVED_SCENARIO_ID}),
                reason="observed joint physical package",
            )
            for bundle_id, action in sorted(scenario_plan.bundle_actions.items())
        )
        residency_index = {
            item.bundle_id: index for index, item in enumerate(residency)
        }
        dependencies: list[TransferDependency] = []
        admissions: list[AdmissionIntent] = []
        restore_charged: set[str] = set()
        execution_set = set(scenario_plan.execution_order)
        for request in policy_input.runnable_frontier:
            if request.request_id not in execution_set:
                admissions.append(
                    AdmissionIntent(
                        request_id=request.request_id,
                        action=AdmissionAction.DEFER,
                        reserved_bytes=0,
                        required_bundle_ids=(),
                        reason="outside the bounded observed joint package",
                    )
                )
                continue
            required = tuple(
                bundle
                for bundle_id in sorted(
                    prepared.bundle_ids_by_context.get(
                        request.context_id,
                        frozenset(),
                    )
                )
                for bundle in (prepared.bundle_by_id[bundle_id],)
                if scenario_plan.bundle_actions.get(bundle.bundle_id)
                == ResidencyAction.PREFETCH_GPU
            )
            restore_bytes = 0
            for bundle in required:
                index = residency_index[bundle.bundle_id]
                dependencies.append(
                    TransferDependency(
                        before_request_id=request.request_id,
                        residency_intent_index=index,
                        require_ack=True,
                    )
                )
                if bundle.bundle_id not in restore_charged:
                    restore_bytes += max(
                        0, bundle.physical_unique_bytes - bundle.gpu_bytes
                    )
                    restore_charged.add(bundle.bundle_id)
            admissions.append(
                AdmissionIntent(
                    request_id=request.request_id,
                    action=(
                        AdmissionAction.RESTORE_THEN_ADMIT
                        if required
                        else AdmissionAction.ADMIT
                    ),
                    reserved_bytes=(
                        _unreserved_startup_bytes(request) + restore_bytes
                    ),
                    required_bundle_ids=tuple(
                        item.bundle_id for item in required
                    ),
                    reason="selected by the observed joint physical package",
                )
            )
        first = (
            request_by_id[scenario_plan.execution_order[0]]
            if scenario_plan.execution_order
            else None
        )
        execution = ExecutionIntent(
            ordered_request_ids=scenario_plan.execution_order,
            selected_workflow_id=first.workflow_id if first else None,
            selected_invocation_id=first.invocation_id if first else None,
            mode="observed_joint",
            graph_version=policy_input.runtime_graph.graph_version,
            reason=(
                "bounded root-workflow fairness and causal priority jointly packed "
                "with exact physical extents"
            ),
        )
        return _make_plan(
            policy_input,
            execution=execution,
            admissions=tuple(sorted(admissions, key=lambda item: item.request_id)),
            residency=residency,
            dependencies=tuple(dependencies),
            expected_hbm_peak_bytes=scenario_plan.projected_hbm_peak_bytes,
            expected_unhidden_stall_ms=scenario_plan.expected_unhidden_stall_ms,
            fallback_reason=None,
            candidate_count=candidate_count,
            evaluated_package_count=evaluated_package_count,
            transition_open=False,
            max_plan_age_ms=self.config.max_plan_age_ms,
            fairness_lag_budget_ms=self.config.fairness_lag_budget_ms,
            fairness_memory_penalty_ms=self.config.memory_penalty_ms,
            fairness_max_workflow_candidates=(
                self.config.max_workflow_candidates
            ),
            prepared=prepared,
            prediction_used=prediction_used,
            prediction_influence=prediction_influence,
        )

    def _fallback(
        self,
        policy_input: PolicyInput,
        candidates: Sequence[_Candidate],
        *,
        reason: str | None,
        started_ns: int,
        transition_open: bool,
        admit_fitting: bool,
        evaluated_package_count: int = 0,
        prediction_used: bool = False,
        prediction_influence: tuple[tuple[str, int], ...] = (),
    ) -> JointPlan:
        available = policy_input.resources.hbm_available_bytes
        admitted: list[RunnableInvocation] = []
        prepared = self.physicalizer.prepare(policy_input) if admit_fitting else None
        if admit_fitting:
            assert prepared is not None
            for candidate in candidates:
                request = candidate.request
                startup_bytes = _unreserved_startup_bytes(request)
                missing_restore = any(
                    bundle.gpu_bytes < bundle.physical_unique_bytes
                    for bundle_id in prepared.bundle_ids_by_context.get(
                        request.context_id,
                        frozenset(),
                    )
                    for bundle in (prepared.bundle_by_id[bundle_id],)
                )
                if missing_restore or startup_bytes > available:
                    continue
                admitted.append(request)
                available -= startup_bytes
        admitted_ids = {item.request_id for item in admitted}
        admissions = tuple(
            AdmissionIntent(
                request_id=request.request_id,
                action=(
                    AdmissionAction.ADMIT
                    if request.request_id in admitted_ids
                    else AdmissionAction.DEFER
                ),
                reserved_bytes=(
                    _unreserved_startup_bytes(request)
                    if request.request_id in admitted_ids
                    else 0
                ),
                required_bundle_ids=(),
                reason=(
                    "conservative fallback admits only GPU-resident fitting work"
                    if request.request_id in admitted_ids
                    else "conservative fallback defers this request"
                ),
            )
            for request in policy_input.runnable_frontier
        )
        mode = (
            "observed_joint_idle"
            if not policy_input.runnable_frontier and reason is None
            else "observed_joint_settling"
            if transition_open
            else "observed_joint_fallback"
        )
        first = admitted[0] if admitted else None
        execution = ExecutionIntent(
            ordered_request_ids=tuple(item.request_id for item in admitted),
            selected_workflow_id=first.workflow_id if first else None,
            selected_invocation_id=first.invocation_id if first else None,
            mode=mode,
            graph_version=policy_input.runtime_graph.graph_version,
            reason=reason or "no observed runnable work",
        )
        projected = (
            policy_input.resources.hbm_used_bytes
            + policy_input.resources.hbm_reserved_bytes
            + sum(_unreserved_startup_bytes(item) for item in admitted)
        )
        plan = _make_plan(
            policy_input,
            execution=execution,
            admissions=admissions,
            residency=(),
            dependencies=(),
            expected_hbm_peak_bytes=projected,
            expected_unhidden_stall_ms=0.0,
            fallback_reason=reason,
            candidate_count=len(candidates),
            evaluated_package_count=evaluated_package_count,
            transition_open=transition_open,
            max_plan_age_ms=self.config.max_plan_age_ms,
            fairness_lag_budget_ms=self.config.fairness_lag_budget_ms,
            fairness_memory_penalty_ms=self.config.memory_penalty_ms,
            fairness_max_workflow_candidates=(
                self.config.max_workflow_candidates
            ),
            prepared=prepared,
            prediction_used=prediction_used,
            prediction_influence=prediction_influence,
        )
        return replace(
            plan,
            planning_ms=self._elapsed_ms(started_ns),
            planner_mode=(
                JointPlannerMode.NO_ACTION
                if not admitted
                else JointPlannerMode.BOUNDED_SEED
            ),
        )

    @staticmethod
    def _elapsed_ms(started_ns: int) -> float:
        return (time.perf_counter_ns() - started_ns) / 1_000_000.0


class AsyncSemanticJointPlanner(ObservedJointPlanner):
    """Latest-wins planner that never binds plans to Radix extents.

    The worker chooses a causal/fair execution order and context-level tier
    goals. Exact closure construction, allocator checks, and page generations
    are intentionally deferred to the scheduler safe point.
    """

    name = "belief_joint_semantic"

    def plan(
        self,
        policy_input: PolicyInput,
        *,
        planning_budget_ms: float | None = None,
        cancel_check: object | None = None,
    ) -> JointPlan:
        budget_ms = (
            self.config.max_planning_budget_ms
            if planning_budget_ms is None
            else min(
                self.config.max_planning_budget_ms,
                float(planning_budget_ms),
            )
        )
        if not math.isfinite(budget_ms) or budget_ms <= 0:
            raise ValueError("planning budget must be finite and positive")
        cancelled = cancel_check if callable(cancel_check) else lambda: False
        started_ns = time.perf_counter_ns()
        candidates, _fairness, prediction_influence = self._ordered_candidates(
            policy_input
        )
        prediction_used = bool(prediction_influence)
        if _transition_open(policy_input):
            return self._fallback(
                policy_input,
                candidates,
                reason="transition_open_settling_barrier",
                started_ns=started_ns,
                transition_open=True,
                admit_fitting=False,
                prediction_used=prediction_used,
                prediction_influence=tuple(prediction_influence.items()),
            )
        if not candidates:
            return self._fallback(
                policy_input,
                candidates,
                reason=(
                    None
                    if not policy_input.runnable_frontier
                    else "no_factual_runnable_request"
                ),
                started_ns=started_ns,
                transition_open=False,
                admit_fitting=False,
                prediction_used=prediction_used,
                prediction_influence=tuple(prediction_influence.items()),
            )

        seed = self._bounded_seed(
            policy_input,
            candidates,
            started_ns=started_ns,
            prediction_used=prediction_used,
            prediction_influence=tuple(prediction_influence.items()),
        )
        if cancelled() or self._elapsed_ms(started_ns) >= budget_ms:
            return replace(
                seed,
                planning_ms=self._elapsed_ms(started_ns),
                planning_termination_reason=(
                    "superseded_seed_published"
                    if cancelled()
                    else "planning_budget_exceeded_seed_published"
                ),
            )

        targets, victim_prediction_count = self._semantic_residency_targets(
            policy_input,
            candidates,
            seed,
        )
        if victim_prediction_count:
            prediction_influence["victim_prediction_selected"] = (
                victim_prediction_count
            )
            prediction_used = True
        execution_ids = frozenset(seed.execution.ordered_request_ids)
        locked_by_context: dict[str, int] = defaultdict(int)
        gpu_by_context: dict[str, int] = defaultdict(int)
        for bundle in policy_input.physical_kv.bundles:
            for context_id in bundle.owner_context_ids:
                locked_by_context[context_id] += bundle.locked_bytes
                gpu_by_context[context_id] += bundle.gpu_bytes
        retractions = tuple(
            RetractionIntent(
                request_id=request.request_id,
                context_id=request.context_id,
                context_epoch=request.context_epoch,
                reason=(
                    "running request is outside the semantic execution set"
                ),
            )
            for request in sorted(
                policy_input.runnable_frontier,
                key=lambda item: (
                    -locked_by_context[item.context_id],
                    -gpu_by_context[item.context_id],
                    item.submitted_ts_ms,
                    item.request_id,
                ),
            )
            if request.causal_class.startswith("engine_running:")
            and request.request_id not in execution_ids
        )
        plan = _make_plan(
            policy_input,
            execution=seed.execution,
            admissions=seed.admissions,
            residency=(),
            dependencies=(),
            expected_hbm_peak_bytes=seed.expected_hbm_peak_bytes,
            expected_unhidden_stall_ms=seed.expected_unhidden_stall_ms,
            fallback_reason=None,
            candidate_count=len(candidates),
            evaluated_package_count=0,
            transition_open=False,
            max_plan_age_ms=self.config.max_plan_age_ms,
            fairness_lag_budget_ms=self.config.fairness_lag_budget_ms,
            fairness_memory_penalty_ms=self.config.memory_penalty_ms,
            fairness_max_workflow_candidates=(
                self.config.max_workflow_candidates
            ),
            semantic_residency=targets,
            retractions=retractions,
            prediction_used=prediction_used,
            prediction_influence=tuple(prediction_influence.items()),
        )
        return replace(
            plan,
            planner_mode=(
                JointPlannerMode.OPTIMIZED
                if plan.execution.ordered_request_ids or targets or retractions
                else JointPlannerMode.NO_ACTION
            ),
            planning_ms=self._elapsed_ms(started_ns),
            search_complete=True,
            planning_termination_reason="semantic_plan_complete",
            planning_phase_ms=(
                ("semantic_total", self._elapsed_ms(started_ns)),
            ),
        )

    def _semantic_residency_targets(
        self,
        policy_input: PolicyInput,
        candidates: Sequence[_Candidate],
        seed: JointPlan,
    ) -> tuple[tuple[SemanticResidencyTarget, ...], int]:
        state = _mapping(policy_input.runtime_graph.state)
        rccg = _mapping(state.get("rccg"))
        context_snapshots = _mapping(rccg.get("contexts"))
        context_epochs = {
            str(context_id): _nonnegative_int(_mapping(raw).get("epoch", 0))
            for context_id, raw in context_snapshots.items()
        }
        context_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "missing_gpu_bytes": 0.0,
                "reclaimable_bytes": 0.0,
                "last_access_ms": 0.0,
            }
        )
        for bundle in policy_input.physical_kv.bundles:
            for context_id in bundle.owner_context_ids:
                stats = context_stats[context_id]
                stats["missing_gpu_bytes"] += max(
                    0, bundle.physical_unique_bytes - bundle.gpu_bytes
                )
                if bundle.actionable:
                    stats["reclaimable_bytes"] += (
                        bundle.marginal_reclaimable_bytes
                    )
                stats["last_access_ms"] = max(
                    stats["last_access_ms"], bundle.last_access_ms
                )

        admitted_ids = set(seed.execution.ordered_request_ids)
        runnable_contexts = {
            item.context_id for item in policy_input.runnable_frontier
        }
        selected_contexts = {
            item.context_id
            for item in policy_input.runnable_frontier
            if item.request_id in admitted_ids
        }
        deferred = [
            item.request
            for item in candidates
            if item.request.request_id not in admitted_ids
        ]
        available = policy_input.resources.hbm_available_bytes
        reclaim_goal = 0
        if deferred:
            first = deferred[0]
            missing = int(
                context_stats[first.context_id]["missing_gpu_bytes"]
            )
            reclaim_goal = max(
                0,
                _unreserved_startup_bytes(first) + missing - available,
            )
        used_ratio = (
            policy_input.resources.hbm_used_bytes
            / policy_input.resources.hbm_capacity_bytes
        )
        if used_ratio >= self.config.emergency_hbm_ratio:
            target_ratio = max(0.0, self.config.emergency_hbm_ratio - 0.05)
            reclaim_goal = max(
                reclaim_goal,
                int(
                    policy_input.resources.hbm_used_bytes
                    - policy_input.resources.hbm_capacity_bytes * target_ratio
                ),
            )

        predicted_idle_ms: dict[str, float] = {}
        prediction_metadata = policy_input.optional_metadata.get(
            "frontier_predictions"
        )
        if (
            prediction_metadata is not None
            and isinstance(prediction_metadata.value, Mapping)
        ):
            invocations = _mapping(rccg.get("invocations"))
            invocation_contexts = {
                str(invocation_id): str(_mapping(raw).get("context_id", ""))
                for invocation_id, raw in invocations.items()
            }
            for invocation_id, raw in prediction_metadata.value.items():
                if not isinstance(raw, Mapping):
                    continue
                idle: float | None = None
                for key in (
                    "remaining_external_wait_ms_p50",
                    "next_output_tokens_p50",
                    "remaining_decode_tokens_p50",
                ):
                    candidate = raw.get(key)
                    if isinstance(candidate, (int, float)) and candidate >= 0:
                        idle = float(candidate)
                        break
                if idle is None:
                    continue
                context_id = invocation_contexts.get(str(invocation_id), "")
                if not context_id:
                    continue
                predicted_idle_ms[context_id] = min(
                    predicted_idle_ms.get(context_id, idle),
                    idle,
                )

        targets: list[SemanticResidencyTarget] = []
        reclaimed = 0
        victims = sorted(
            (
                (
                    context_id,
                    int(stats["reclaimable_bytes"]),
                    float(stats["last_access_ms"]),
                )
                for context_id, stats in context_stats.items()
                if context_id not in runnable_contexts
                and context_id not in selected_contexts
                and int(stats["reclaimable_bytes"]) > 0
                and context_id in context_epochs
            ),
            key=lambda item: (
                (
                    -predicted_idle_ms[item[0]]
                    if item[0] in predicted_idle_ms
                    else item[2]
                ),
                -item[1],
                item[0],
            ),
        )
        victim_prediction_count = sum(
            1 for item in victims if item[0] in predicted_idle_ms
        )
        for context_id, reclaimable, _last_access in victims:
            if reclaimed >= reclaim_goal:
                break
            targets.append(
                SemanticResidencyTarget(
                    context_id=context_id,
                    context_epoch=context_epochs[context_id],
                    action=ResidencyAction.COMMIT_CPU,
                    target_bytes_hint=reclaimable,
                    deadline_ms=policy_input.resources.ts_ms,
                    reason="semantic headroom target for a parked context",
                )
            )
            reclaimed += reclaimable

        seen_prefetch_contexts: set[str] = set()
        for request in deferred:
            context_id = request.context_id
            missing = int(context_stats[context_id]["missing_gpu_bytes"])
            if missing <= 0 or context_id in seen_prefetch_contexts:
                continue
            targets.append(
                SemanticResidencyTarget(
                    context_id=context_id,
                    context_epoch=request.context_epoch,
                    action=ResidencyAction.PREFETCH_GPU,
                    target_bytes_hint=missing,
                    deadline_ms=policy_input.resources.ts_ms,
                    reason="semantic restore target for deferred causal work",
                )
            )
            seen_prefetch_contexts.add(context_id)
            if len(seen_prefetch_contexts) >= self.config.max_workflow_candidates:
                break
        return tuple(targets), victim_prediction_count


def _phase_timings(values: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(
        (name, max(0.0, float(value)))
        for name, value in sorted(values.items())
    )


def _make_plan(
    policy_input: PolicyInput,
    *,
    execution: ExecutionIntent,
    admissions: tuple[AdmissionIntent, ...],
    residency: tuple[ResidencyIntent, ...],
    dependencies: tuple[TransferDependency, ...],
    expected_hbm_peak_bytes: int,
    expected_unhidden_stall_ms: float,
    fallback_reason: str | None,
    candidate_count: int,
    evaluated_package_count: int,
    transition_open: bool,
    max_plan_age_ms: float,
    fairness_lag_budget_ms: float,
    fairness_memory_penalty_ms: float,
    fairness_max_workflow_candidates: int,
    prepared: PreparedPolicyInput | None = None,
    semantic_residency: tuple[SemanticResidencyTarget, ...] = (),
    retractions: tuple[RetractionIntent, ...] = (),
    prediction_used: bool = False,
    prediction_influence: tuple[tuple[str, int], ...] = (),
) -> JointPlan:
    read_set = _build_read_set(
        policy_input,
        execution=execution,
        admissions=admissions,
        residency=residency,
        max_plan_age_ms=max_plan_age_ms,
        fairness_lag_budget_ms=fairness_lag_budget_ms,
        fairness_memory_penalty_ms=fairness_memory_penalty_ms,
        fairness_max_workflow_candidates=fairness_max_workflow_candidates,
        prepared=prepared,
        semantic_residency=semantic_residency,
    )
    provisional = JointPlan(
        plan_id="pending",
        input_snapshot_id=policy_input.snapshot_id,
        generated_ts_ms=policy_input.resources.ts_ms,
        execution=execution,
        admissions=admissions,
        residency=residency,
        dependencies=dependencies,
        expected_hbm_peak_bytes=expected_hbm_peak_bytes,
        expected_unhidden_stall_ms=expected_unhidden_stall_ms,
        topology_version=policy_input.physical_kv.topology_version,
        allocator_version=policy_input.physical_kv.allocator_version,
        read_set=read_set,
        semantic_residency=semantic_residency,
        retractions=retractions,
        fallback_reason=fallback_reason,
        candidate_count=candidate_count,
        evaluated_package_count=evaluated_package_count,
        transition_open=transition_open,
        prediction_used=prediction_used,
        prediction_influence=prediction_influence,
    )
    semantic = provisional.to_dict()
    semantic.pop("plan_id")
    semantic.pop("planning_ms")
    digest = hashlib.blake2b(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
        digest_size=16,
        person=b"bk-joint-plan",
    ).hexdigest()
    return replace(provisional, plan_id=f"joint-{digest}")


def _build_read_set(
    policy_input: PolicyInput,
    *,
    execution: ExecutionIntent,
    admissions: Sequence[AdmissionIntent],
    residency: Sequence[ResidencyIntent],
    max_plan_age_ms: float,
    fairness_lag_budget_ms: float,
    fairness_memory_penalty_ms: float,
    fairness_max_workflow_candidates: int,
    prepared: PreparedPolicyInput | None = None,
    semantic_residency: Sequence[SemanticResidencyTarget] = (),
) -> PlanReadSet:
    request_by_id = (
        prepared.request_by_id
        if prepared is not None
        else {item.request_id: item for item in policy_input.runnable_frontier}
    )
    read_request_ids = {
        item.request_id for item in admissions
    }.union(execution.ordered_request_ids)
    requests = {
        request_id: request_by_id[request_id]
        for request_id in sorted(read_request_ids)
        if request_id in request_by_id
    }
    state = _mapping(policy_input.runtime_graph.state)
    rccg = _mapping(state.get("rccg"))
    invocations = _mapping(rccg.get("invocations"))
    joins = _mapping(rccg.get("joins"))
    invocation_ids = {item.invocation_id for item in requests.values()}
    relevant_joins = {
        join_id: raw
        for join_id, raw in joins.items()
        if invocation_ids.intersection(
            {str(item) for item in _sequence(_mapping(raw).get("members"))}
            | {str(item) for item in _sequence(_mapping(raw).get("waiters"))}
        )
    }
    by_workflow: dict[str, list[dict[str, object]]] = defaultdict(list)
    for request in policy_input.runnable_frontier:
        by_workflow[request.workflow_id].append(request.to_dict())
    control = _mapping(state.get("control"))
    transitions = _mapping(control.get("transitions"))
    workflow_ids = {item.workflow_id for item in requests.values()}
    touched_bundle_ids = {item.bundle_id for item in residency}
    bundle_by_id = (
        prepared.bundle_by_id
        if prepared is not None
        else {
            item.bundle_id: item
            for item in policy_input.physical_kv.bundles
            if item.bundle_id in touched_bundle_ids
        }
    )
    touched = {
        intent.bundle_id: bundle_by_id[intent.bundle_id]
        for intent in residency
        if intent.bundle_id in bundle_by_id
    }
    required_headroom = sum(
        item.reserved_bytes
        for item in admissions
        if item.action
        in {AdmissionAction.ADMIT, AdmissionAction.RESTORE_THEN_ADMIT}
    )
    planned_reclaim = sum(
        item.target_bytes
        for item in residency
        if item.action in {ResidencyAction.COMMIT_CPU, ResidencyAction.DROP}
    )
    required_host = sum(
        max(0, touched[item.bundle_id].gpu_bytes - touched[item.bundle_id].cpu_bytes)
        for item in residency
        if item.action == ResidencyAction.COMMIT_CPU
        and item.bundle_id in touched
    )
    planned_h2d = sum(
        item.target_bytes
        for item in residency
        if item.action == ResidencyAction.PREFETCH_GPU
    )
    fairness = _mapping(state.get("workflow_fairness"))
    fairness_order = _current_fairness_order(
        policy_input,
        lag_budget_ms=fairness_lag_budget_ms,
        memory_penalty_ms=fairness_memory_penalty_ms,
        max_workflow_candidates=fairness_max_workflow_candidates,
    )
    return PlanReadSet(
        lineage_snapshot_id=policy_input.snapshot_id,
        request_fingerprints={
            request_id: _fingerprint_json(_request_dependency_payload(request))
            for request_id, request in requests.items()
        },
        request_startup_bytes={
            request_id: _unreserved_startup_bytes(request)
            for request_id, request in requests.items()
        },
        workflow_frontier_fingerprints={
            workflow_id: _fingerprint_json(
                sorted(
                    (
                        _request_dependency_payload_dict(item)
                        for item in items
                    ),
                    key=lambda item: str(item["request_id"]),
                )
            )
            for workflow_id, items in sorted(by_workflow.items())
        },
        context_epochs={
            **{
                request.context_id: request.context_epoch
                for request in requests.values()
            },
            **{
                item.context_id: item.context_epoch
                for item in semantic_residency
            },
        },
        invocation_fingerprints={
            invocation_id: _fingerprint_json(_mapping(invocations.get(invocation_id)))
            for invocation_id in sorted(invocation_ids)
        },
        join_fingerprints={
            str(join_id): _fingerprint_json(_mapping(raw))
            for join_id, raw in sorted(relevant_joins.items())
        },
        transition_generations={
            workflow_id: _transition_generation(transitions, workflow_id)
            for workflow_id in sorted(workflow_ids)
        },
        touched_bundle_generations={
            bundle_id: bundle.generation_fingerprint
            for bundle_id, bundle in sorted(touched.items())
        },
        touched_bundle_actions={
            item.bundle_id: item.action.value for item in residency
        },
        fairness_revision=_nonnegative_int(fairness.get("revision", 0)),
        fairness_order=fairness_order,
        fairness_lag_budget_ms=fairness_lag_budget_ms,
        fairness_memory_penalty_ms=fairness_memory_penalty_ms,
        fairness_max_workflow_candidates=fairness_max_workflow_candidates,
        transfer_epoch=_nonnegative_int(control.get("transfer_epoch", 0)),
        expires_at_ms=policy_input.resources.ts_ms + max_plan_age_ms,
        resource_certificate=ResourceFeasibilityCertificate(
            required_headroom_bytes=required_headroom,
            planned_reclaim_bytes=planned_reclaim,
            required_host_bytes=required_host,
            planned_h2d_bytes=planned_h2d,
        ),
    )


def validate_joint_plan(
    plan: JointPlan,
    policy_input: PolicyInput,
) -> JointPlanValidation:
    strict = plan.strict_global_stale_reasons(policy_input)
    conflicts: list[str] = []
    read_set = plan.read_set
    if policy_input.resources.ts_ms > read_set.expires_at_ms:
        conflicts.append("plan_expired")

    state = _mapping(policy_input.runtime_graph.state)
    rccg = _mapping(state.get("rccg"))
    invocations = _mapping(rccg.get("invocations"))
    joins = _mapping(rccg.get("joins"))
    fairness = _mapping(state.get("workflow_fairness"))
    control = _mapping(state.get("control"))
    transitions = _mapping(control.get("transitions"))
    current_requests = {
        item.request_id: item for item in policy_input.runnable_frontier
    }
    for request_id, expected in read_set.request_fingerprints.items():
        request = current_requests.get(request_id)
        if request is None:
            conflicts.append(f"request_missing:{request_id}")
        elif _fingerprint_json(_request_dependency_payload(request)) != expected:
            conflicts.append(f"request_changed:{request_id}")

    current_by_workflow: dict[str, list[dict[str, object]]] = defaultdict(list)
    for request in policy_input.runnable_frontier:
        current_by_workflow[request.workflow_id].append(
            _request_dependency_payload(request)
        )
    if set(current_by_workflow) != set(read_set.workflow_frontier_fingerprints):
        conflicts.append("workflow_frontier_membership")
    for workflow_id, expected in read_set.workflow_frontier_fingerprints.items():
        current = sorted(
            current_by_workflow.get(workflow_id, ()),
            key=lambda item: str(item["request_id"]),
        )
        if _fingerprint_json(current) != expected:
            conflicts.append(f"workflow_frontier_changed:{workflow_id}")

    current_context_epochs = {
        item.context_id: item.context_epoch
        for item in policy_input.runnable_frontier
    }
    for context_id, expected in read_set.context_epochs.items():
        if current_context_epochs.get(context_id) != expected:
            conflicts.append(f"context_epoch:{context_id}")
    for invocation_id, expected in read_set.invocation_fingerprints.items():
        if _fingerprint_json(_mapping(invocations.get(invocation_id))) != expected:
            conflicts.append(f"invocation_changed:{invocation_id}")
    for join_id, expected in read_set.join_fingerprints.items():
        if _fingerprint_json(_mapping(joins.get(join_id))) != expected:
            conflicts.append(f"join_changed:{join_id}")

    current_fairness_revision = _nonnegative_int(fairness.get("revision", 0))
    if current_fairness_revision < read_set.fairness_revision:
        conflicts.append("fairness_revision_regressed")
    if plan.execution.selected_workflow_id is not None:
        current_fairness_order = _current_fairness_order(
            policy_input,
            lag_budget_ms=read_set.fairness_lag_budget_ms,
            memory_penalty_ms=read_set.fairness_memory_penalty_ms,
            max_workflow_candidates=(
                read_set.fairness_max_workflow_candidates
            ),
        )
        if (
            not current_fairness_order
            or current_fairness_order[0]
            != plan.execution.selected_workflow_id
        ):
            conflicts.append("fairness_priority_changed")
    current_transfer_epoch = _nonnegative_int(control.get("transfer_epoch", 0))
    if current_transfer_epoch < read_set.transfer_epoch:
        conflicts.append("transfer_epoch_regressed")
    elif current_transfer_epoch != read_set.transfer_epoch and plan.residency:
        conflicts.append("transfer_epoch")
    for workflow_id, expected in read_set.transition_generations.items():
        transition = _mapping(transitions.get(workflow_id))
        if _nonnegative_int(transition.get("generation", 0)) != expected:
            conflicts.append(f"transition_generation:{workflow_id}")
        if bool(transition.get("open", False)):
            conflicts.append(f"transition_open:{workflow_id}")
        if bool(transition.get("degraded", False)):
            conflicts.append(f"transition_degraded:{workflow_id}")

    bundle_by_id = {
        item.bundle_id: item for item in policy_input.physical_kv.bundles
    }
    actual_reclaim = 0
    for bundle_id, expected in read_set.touched_bundle_generations.items():
        bundle = bundle_by_id.get(bundle_id)
        action = read_set.touched_bundle_actions[bundle_id]
        if bundle is None:
            conflicts.append(f"bundle_missing:{bundle_id}")
            continue
        if bundle.generation_fingerprint != expected:
            conflicts.append(f"bundle_generation:{bundle_id}")
            continue
        if action != ResidencyAction.KEEP.value and not bundle.actionable:
            conflicts.append(f"bundle_blocked:{bundle_id}")
        if action == ResidencyAction.PREFETCH_GPU.value:
            if bundle.gpu_bytes >= bundle.physical_unique_bytes:
                conflicts.append(f"prefetch_obsolete:{bundle_id}")
        elif action in {
            ResidencyAction.COMMIT_CPU.value,
            ResidencyAction.DROP.value,
        }:
            if bundle.gpu_bytes <= 0:
                conflicts.append(f"reclaim_obsolete:{bundle_id}")
            actual_reclaim += bundle.marginal_reclaimable_bytes

    certificate = read_set.resource_certificate
    required_headroom = certificate.required_headroom_bytes
    for admission in plan.admissions:
        if admission.action not in {
            AdmissionAction.ADMIT,
            AdmissionAction.RESTORE_THEN_ADMIT,
        }:
            continue
        request = current_requests.get(admission.request_id)
        source_startup = read_set.request_startup_bytes.get(
            admission.request_id
        )
        if request is None or source_startup is None:
            continue
        required_headroom += (
            _unreserved_startup_bytes(request) - source_startup
        )
    required_headroom = max(0, required_headroom)
    if (
        policy_input.resources.hbm_available_bytes + actual_reclaim
        < required_headroom
    ):
        conflicts.append("insufficient_hbm_headroom")
    if policy_input.resources.host_free_bytes < certificate.required_host_bytes:
        conflicts.append("insufficient_host_headroom")
    return JointPlanValidation(
        strict_global_reasons=tuple(strict),
        readset_conflict_reasons=tuple(sorted(set(conflicts))),
    )


def validate_joint_plan_components(
    plan: JointPlan,
    source: PolicyInput,
    current: JointPlanCurrentState,
) -> JointPlanComponentValidation:
    """Validate independent plan actions against bounded current state.

    Version changes remain visible through ``strict_global_reasons`` but do not
    invalidate unrelated actions. Every executable action is instead checked
    against the exact requests, RCCG records, transitions, and physical bundles
    that it consumed at planning time.
    """

    read_set = plan.read_set
    global_reasons: list[str] = []
    if current.now_ms > read_set.expires_at_ms:
        global_reasons.append("plan_expired")
    if current.transfer_epoch < read_set.transfer_epoch:
        global_reasons.append("transfer_epoch_regressed")

    current_requests = {
        item.request_id: item for item in current.runnable_frontier
    }
    source_requests = {
        item.request_id: item for item in source.runnable_frontier
    }
    source_state = _mapping(source.runtime_graph.state)
    source_rccg = _mapping(source_state.get("rccg"))
    source_joins = _mapping(source_rccg.get("joins"))

    admission_reasons: dict[str, list[str]] = {
        item.request_id: [] for item in plan.admissions
    }
    for admission in plan.admissions:
        request_id = admission.request_id
        reasons = admission_reasons[request_id]
        request = current_requests.get(request_id)
        source_request = source_requests.get(request_id)
        expected = read_set.request_fingerprints.get(request_id)
        if request is None:
            reasons.append("request_missing")
            continue
        if source_request is None or expected is None:
            reasons.append("request_not_in_readset")
            continue
        if _fingerprint_json(_request_dependency_payload(request)) != expected:
            reasons.append("request_changed")
        expected_epoch = read_set.context_epochs.get(request.context_id)
        if expected_epoch is None or request.context_epoch != expected_epoch:
            reasons.append("context_epoch")
        expected_invocation = read_set.invocation_fingerprints.get(
            request.invocation_id
        )
        current_invocation = _mapping(
            current.invocation_snapshots.get(request.invocation_id)
        )
        if (
            expected_invocation is None
            or _fingerprint_json(current_invocation) != expected_invocation
        ):
            reasons.append("invocation_changed")
        transition = _mapping(current.transitions.get(request.workflow_id))
        expected_generation = read_set.transition_generations.get(
            request.workflow_id
        )
        if expected_generation is None or _transition_generation(
            current.transitions, request.workflow_id
        ) != expected_generation:
            reasons.append("transition_generation")
        if bool(transition.get("open", False)):
            reasons.append("transition_open")
        if bool(transition.get("degraded", False)):
            reasons.append("transition_degraded")

        for join_id, expected_join in read_set.join_fingerprints.items():
            source_join = _mapping(source_joins.get(join_id))
            affected = {
                str(item)
                for item in (
                    *_sequence(source_join.get("members")),
                    *_sequence(source_join.get("waiters")),
                )
            }
            if request.invocation_id not in affected:
                continue
            if _fingerprint_json(
                _mapping(current.join_snapshots.get(join_id))
            ) != expected_join:
                reasons.append(f"join_changed:{join_id}")

    residency_reasons: dict[str, list[str]] = {
        item.bundle_id: [] for item in plan.residency
    }
    current_bundle_by_id = dict(current.bundle_snapshots)
    for intent in plan.residency:
        reasons = residency_reasons[intent.bundle_id]
        bundle = current_bundle_by_id.get(intent.bundle_id)
        expected = read_set.touched_bundle_generations.get(intent.bundle_id)
        if bundle is None:
            reasons.append("bundle_missing")
            continue
        if expected is None or bundle.generation_fingerprint != expected:
            reasons.append("bundle_generation")
            continue
        if intent.action != ResidencyAction.KEEP and not bundle.actionable:
            reasons.extend(
                f"bundle_blocked:{item}" for item in bundle.blocker_codes
            )
        if intent.action == ResidencyAction.PREFETCH_GPU:
            if bundle.gpu_bytes >= bundle.physical_unique_bytes:
                reasons.append("prefetch_obsolete")
        elif intent.action in {
            ResidencyAction.COMMIT_CPU,
            ResidencyAction.DROP,
        }:
            if bundle.gpu_bytes <= 0:
                reasons.append("reclaim_obsolete")
        if (
            intent.action != ResidencyAction.KEEP
            and _residency_target_bytes(bundle, intent.action)
            != intent.target_bytes
        ):
            reasons.append("bundle_target_changed")

    host_remaining = current.host_free_bytes
    for intent in plan.residency:
        if (
            intent.action != ResidencyAction.COMMIT_CPU
            or residency_reasons[intent.bundle_id]
        ):
            continue
        bundle = current_bundle_by_id[intent.bundle_id]
        assert bundle is not None
        host_needed = max(0, bundle.gpu_bytes - bundle.cpu_bytes)
        if host_needed > host_remaining:
            residency_reasons[intent.bundle_id].append(
                "insufficient_host_headroom"
            )
        else:
            host_remaining -= host_needed

    residency_validation = {
        bundle_id: IntentValidation(tuple(reasons))
        for bundle_id, reasons in residency_reasons.items()
    }
    for admission in plan.admissions:
        reasons = admission_reasons[admission.request_id]
        for bundle_id in admission.required_bundle_ids:
            bundle_validation = residency_validation.get(bundle_id)
            if bundle_validation is None:
                reasons.append(f"required_bundle_missing:{bundle_id}")
            elif not bundle_validation.valid:
                reasons.append(f"required_bundle_invalid:{bundle_id}")

    reclaimable = 0
    for intent in plan.residency:
        validation = residency_validation[intent.bundle_id]
        bundle = current_bundle_by_id.get(intent.bundle_id)
        if (
            validation.valid
            and bundle is not None
            and intent.action
            in {ResidencyAction.COMMIT_CPU, ResidencyAction.DROP}
        ):
            reclaimable += bundle.marginal_reclaimable_bytes
    available = current.hbm_available_bytes + reclaimable
    admission_by_id = {item.request_id: item for item in plan.admissions}
    admission_order = tuple(plan.execution.ordered_request_ids) + tuple(
        item.request_id
        for item in plan.admissions
        if item.request_id not in plan.execution.ordered_request_ids
    )
    for request_id in admission_order:
        admission = admission_by_id.get(request_id)
        if admission is None or admission.action not in {
            AdmissionAction.ADMIT,
            AdmissionAction.RESTORE_THEN_ADMIT,
        }:
            continue
        reasons = admission_reasons[request_id]
        request = current_requests.get(request_id)
        source_startup = read_set.request_startup_bytes.get(request_id)
        if reasons or request is None or source_startup is None:
            continue
        needed = max(
            0,
            admission.reserved_bytes
            + _unreserved_startup_bytes(request)
            - source_startup,
        )
        if needed > available:
            reasons.append("insufficient_hbm_headroom")
        else:
            available -= needed

    admission_validation = {
        request_id: IntentValidation(tuple(reasons))
        for request_id, reasons in admission_reasons.items()
    }

    execution_reasons: list[str] = []
    for request_id in plan.execution.ordered_request_ids:
        validation = admission_validation.get(request_id)
        if validation is None:
            execution_reasons.append(f"admission_missing:{request_id}")
        elif not validation.valid:
            execution_reasons.append(f"admission_invalid:{request_id}")
    selected_workflow = plan.execution.selected_workflow_id
    if selected_workflow is not None:
        current_frontier = sorted(
            (
                _request_dependency_payload(item)
                for item in current.runnable_frontier
                if item.workflow_id == selected_workflow
            ),
            key=lambda item: str(item["request_id"]),
        )
        expected_frontier = read_set.workflow_frontier_fingerprints.get(
            selected_workflow
        )
        if (
            expected_frontier is None
            or _fingerprint_json(current_frontier) != expected_frontier
        ):
            execution_reasons.append("selected_workflow_frontier_changed")
        fairness_order = _current_fairness_order_from_parts(
            current.runnable_frontier,
            current.invocation_snapshots,
            current.fairness_accounts,
            current.workflow_memory_charges,
            hbm_capacity_bytes=current.hbm_capacity_bytes,
            lag_budget_ms=read_set.fairness_lag_budget_ms,
            memory_penalty_ms=read_set.fairness_memory_penalty_ms,
            max_workflow_candidates=read_set.fairness_max_workflow_candidates,
        )
        if not fairness_order or fairness_order[0] != selected_workflow:
            execution_reasons.append("fairness_priority_changed")
    if current.fairness_revision < read_set.fairness_revision:
        execution_reasons.append("fairness_revision_regressed")

    dependency_validation: list[IntentValidation] = []
    for dependency in plan.dependencies:
        reasons: list[str] = []
        if dependency.residency_intent_index >= len(plan.residency):
            reasons.append("residency_index")
        else:
            bundle_id = plan.residency[
                dependency.residency_intent_index
            ].bundle_id
            if not residency_validation[bundle_id].valid:
                reasons.append(f"residency_invalid:{bundle_id}")
        if dependency.before_request_id is not None:
            validation = admission_validation.get(dependency.before_request_id)
            if validation is None:
                reasons.append(
                    f"admission_missing:{dependency.before_request_id}"
                )
            elif not validation.valid:
                reasons.append(
                    f"admission_invalid:{dependency.before_request_id}"
                )
        dependency_validation.append(IntentValidation(tuple(reasons)))

    return JointPlanComponentValidation(
        strict_global_reasons=current.strict_global_reasons,
        global_reasons=tuple(global_reasons),
        execution=IntentValidation(tuple(execution_reasons)),
        admissions=admission_validation,
        residency=residency_validation,
        dependencies=tuple(dependency_validation),
    )


def _residency_target_bytes(bundle: object, action: ResidencyAction) -> int:
    physical_unique_bytes = int(getattr(bundle, "physical_unique_bytes"))
    gpu_bytes = int(getattr(bundle, "gpu_bytes"))
    marginal_reclaimable_bytes = int(
        getattr(bundle, "marginal_reclaimable_bytes")
    )
    if action == ResidencyAction.PREFETCH_GPU:
        return max(0, physical_unique_bytes - gpu_bytes)
    if action in {ResidencyAction.COMMIT_CPU, ResidencyAction.DROP}:
        return marginal_reclaimable_bytes
    return gpu_bytes


def _unreserved_startup_bytes(request: RunnableInvocation) -> int:
    if request.causal_class.startswith(
        ("reserved_admission:", "engine_waiting:", "engine_running:")
    ):
        return 0
    return request.startup_bytes


def _transition_open(policy_input: PolicyInput) -> bool:
    state = _mapping(policy_input.runtime_graph.state)
    transition = _mapping(state.get("transition"))
    if bool(transition.get("open", False)):
        return True
    control = _mapping(state.get("control"))
    transitions = _mapping(control.get("transitions"))
    workflow_ids = {item.workflow_id for item in policy_input.runnable_frontier}
    return any(
        bool(_mapping(transitions.get(workflow_id)).get("open", False))
        or bool(_mapping(transitions.get(workflow_id)).get("degraded", False))
        for workflow_id in workflow_ids
    )


def _transition_generation(
    transitions: Mapping[str, object], workflow_id: str
) -> int:
    return _nonnegative_int(
        _mapping(transitions.get(workflow_id)).get("generation", 0)
    )


def _request_dependency_payload(
    request: RunnableInvocation,
) -> dict[str, object]:
    return _request_dependency_payload_dict(request.to_dict())


def _request_dependency_payload_dict(
    request: Mapping[str, object],
) -> dict[str, object]:
    return {
        name: request.get(name)
        for name in (
            "request_id",
            "workflow_id",
            "invocation_id",
            "context_id",
            "context_epoch",
            "submitted_ts_ms",
            "causal_class",
            "program_id",
        )
    }


def _current_fairness_order(
    policy_input: PolicyInput,
    *,
    lag_budget_ms: float,
    memory_penalty_ms: float,
    max_workflow_candidates: int,
) -> tuple[str, ...]:
    state = _mapping(policy_input.runtime_graph.state)
    rccg = _mapping(state.get("rccg"))
    invocations = _mapping(rccg.get("invocations"))
    workflow_ids = {
        request.workflow_id
        for request in policy_input.runnable_frontier
        if _is_factual_runnable(
            str(
                _mapping(invocations.get(request.invocation_id)).get(
                    "state", "unknown"
                )
            ),
            pending_messages=_nonnegative_int(
                _mapping(invocations.get(request.invocation_id)).get(
                    "pending_messages", 0
                )
            ),
            causal_class=request.causal_class,
        )
    }
    if not workflow_ids:
        return ()
    fairness = _mapping(state.get("workflow_fairness"))
    return _current_fairness_order_from_parts(
        policy_input.runnable_frontier,
        invocations,
        _mapping(fairness.get("accounts")),
        _mapping(fairness.get("memory_charges_bytes")),
        hbm_capacity_bytes=policy_input.resources.hbm_capacity_bytes,
        lag_budget_ms=lag_budget_ms,
        memory_penalty_ms=memory_penalty_ms,
        max_workflow_candidates=max_workflow_candidates,
    )


def _current_fairness_order_from_parts(
    runnable_frontier: Sequence[RunnableInvocation],
    invocations: Mapping[str, object],
    accounts: Mapping[str, object],
    memory_charges: Mapping[str, object],
    *,
    hbm_capacity_bytes: int,
    lag_budget_ms: float,
    memory_penalty_ms: float,
    max_workflow_candidates: int,
) -> tuple[str, ...]:
    virtual_runtime = {
        workflow_id: _nonnegative_float(
            _mapping(accounts.get(workflow_id)).get("virtual_runtime_ms", 0.0)
        )
        for workflow_id in {
            request.workflow_id
            for request in runnable_frontier
            if _is_factual_runnable(
                str(
                    _mapping(invocations.get(request.invocation_id)).get(
                        "state", "unknown"
                    )
                ),
                pending_messages=_nonnegative_int(
                    _mapping(invocations.get(request.invocation_id)).get(
                        "pending_messages", 0
                    )
                ),
                causal_class=request.causal_class,
            )
        }
    }
    if not virtual_runtime:
        return ()
    minimum = min(virtual_runtime.values(), default=0.0)
    within_lag = {
        workflow_id
        for workflow_id, value in virtual_runtime.items()
        if value - minimum <= lag_budget_ms
    }

    def key(workflow_id: str) -> tuple[float, float, str]:
        memory = _nonnegative_float(memory_charges.get(workflow_id, 0.0))
        share = memory / hbm_capacity_bytes
        return (
            virtual_runtime[workflow_id] + share * memory_penalty_ms,
            share,
            workflow_id,
        )

    return tuple(
        sorted(within_lag, key=key)[:max_workflow_candidates]
    )


def _fingerprint_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_thaw_json(item) for item in value]
    return value


def _join_stragglers(joins: Mapping[str, object]) -> frozenset[str]:
    result: set[str] = set()
    for raw in joins.values():
        join = _mapping(raw)
        if bool(join.get("satisfied", False)):
            continue
        members = {str(item) for item in _sequence(join.get("members"))}
        completed = {str(item) for item in _sequence(join.get("completed"))}
        waiters = {str(item) for item in _sequence(join.get("waiters"))}
        remaining = members - completed
        if len(remaining) == 1 and waiters:
            result.update(remaining)
    return frozenset(result)


def _unblock_depth(
    invocation_id: str,
    invocations: Mapping[str, object],
    joins: Mapping[str, object],
) -> int:
    depth = 0
    current_id = invocation_id
    seen: set[str] = set()
    while current_id not in seen:
        seen.add(current_id)
        current = _mapping(invocations.get(current_id))
        parent_id = current.get("parent_invocation_id")
        if not isinstance(parent_id, str) or not parent_id:
            break
        parent = _mapping(invocations.get(parent_id))
        blocking = {str(item) for item in _sequence(parent.get("blocking_children"))}
        if current_id not in blocking or str(parent.get("state")) not in {
            "wait_child",
            "wait_join",
        }:
            break
        depth += 1
        current_id = parent_id
    if depth:
        return depth
    for raw in joins.values():
        join = _mapping(raw)
        members = {str(item) for item in _sequence(join.get("members"))}
        completed = {str(item) for item in _sequence(join.get("completed"))}
        waiters = {str(item) for item in _sequence(join.get("waiters"))}
        if invocation_id in members - completed and waiters:
            return 1
    return 0


def _is_factual_runnable(
    state: str,
    *,
    pending_messages: int,
    causal_class: str,
) -> bool:
    if state in {"ready", "created"} or pending_messages > 0:
        return True
    if state == "running_llm":
        return causal_class.startswith(
            (
                "engine_running:",
                "engine_waiting:",
                "pending_admission:",
                "reserved_admission:",
            )
        )
    return state == "unknown"


def _causal_class_part(value: str, index: int, default: str) -> str:
    parts = value.split(":")
    return parts[index] if len(parts) > index and parts[index] else default


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _nonnegative_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result >= 0 else 0.0


def _nonnegative_int(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)
