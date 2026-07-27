from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from beliefkv.policy.reference import (
    PhysicalBundleSnapshot,
    PolicyInput,
    ResidencyAction,
    RunnableInvocation,
)
from beliefkv.policy.scenario_physicalizer import (
    PreparedPolicyInput,
    ScenarioDemand,
    ScenarioTransition,
)


@dataclass(frozen=True)
class FairnessWindow:
    eligible_workflow_ids: frozenset[str]
    lag_ms_by_workflow: Mapping[str, float] = field(default_factory=dict)
    lag_budget_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.lag_budget_ms < 0:
            raise ValueError("lag_budget_ms must be non-negative")
        lag = {str(key): float(value) for key, value in self.lag_ms_by_workflow.items()}
        if any(not math.isfinite(value) or value < 0 for value in lag.values()):
            raise ValueError("fairness lag must be finite and non-negative")
        object.__setattr__(self, "lag_ms_by_workflow", MappingProxyType(lag))


@dataclass(frozen=True)
class WhatIfPackerConfig:
    max_victim_candidates: int = 8
    max_frontier_per_workflow: int = 4
    handoff_hysteresis_ms: float = 100.0
    emergency_hbm_ratio: float = 0.98
    kv_bytes_per_token: int = 98_304
    recompute_ms_per_token: float = 0.05
    allow_recompute: bool = True
    require_exact_extent_identity: bool = True

    def __post_init__(self) -> None:
        if self.max_victim_candidates <= 0 or self.max_frontier_per_workflow <= 0:
            raise ValueError("what-if candidate limits must be positive")
        if self.handoff_hysteresis_ms < 0:
            raise ValueError("handoff_hysteresis_ms must be non-negative")
        if not 0 < self.emergency_hbm_ratio <= 1:
            raise ValueError("emergency_hbm_ratio must be in (0, 1]")
        if self.kv_bytes_per_token <= 0 or self.recompute_ms_per_token < 0:
            raise ValueError("recompute parameters are invalid")


@dataclass(frozen=True)
class ScenarioPlan:
    snapshot_id: str
    scenario_id: str
    execution_order: tuple[str, ...]
    admission_actions: Mapping[str, str]
    bundle_actions: Mapping[str, ResidencyAction]
    feasible: bool
    expected_unhidden_stall_ms: float
    hbm_time_byte_ms: float
    d2h_bytes: int
    h2d_bytes: int
    recompute_tokens: int
    projected_hbm_peak_bytes: int
    reclaimed_bytes: int
    physical_accounting_exact: bool
    blocker_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "admission_actions",
            MappingProxyType(dict(sorted(self.admission_actions.items()))),
        )
        object.__setattr__(
            self,
            "bundle_actions",
            MappingProxyType(dict(sorted(self.bundle_actions.items()))),
        )
        object.__setattr__(
            self, "blocker_reasons", tuple(sorted(set(self.blocker_reasons)))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "scenario_id": self.scenario_id,
            "execution_order": list(self.execution_order),
            "admission_actions": dict(self.admission_actions),
            "bundle_actions": {
                key: value.value for key, value in self.bundle_actions.items()
            },
            "feasible": self.feasible,
            "expected_unhidden_stall_ms": self.expected_unhidden_stall_ms,
            "hbm_time_byte_ms": self.hbm_time_byte_ms,
            "d2h_bytes": self.d2h_bytes,
            "h2d_bytes": self.h2d_bytes,
            "recompute_tokens": self.recompute_tokens,
            "projected_hbm_peak_bytes": self.projected_hbm_peak_bytes,
            "reclaimed_bytes": self.reclaimed_bytes,
            "physical_accounting_exact": self.physical_accounting_exact,
            "blocker_reasons": list(self.blocker_reasons),
        }


@dataclass(frozen=True)
class _VictimChoice:
    actions: tuple[tuple[PhysicalBundleSnapshot, ResidencyAction], ...]
    reclaimed_bytes: int
    d2h_bytes: int
    recompute_tokens: int


class WhatIfPacker:
    """Side-effect-free, closure/capacity/fairness-aware scenario packer."""

    def __init__(self, config: WhatIfPackerConfig | None = None) -> None:
        self.config = config or WhatIfPackerConfig()
        self._prepared_input: PolicyInput | None = None
        self._request_by_id: Mapping[str, RunnableInvocation] = {}
        self._bundle_by_id: Mapping[str, PhysicalBundleSnapshot] = {}
        self._potential_victims: tuple[PhysicalBundleSnapshot, ...] = ()

    def _prepare(
        self,
        policy_input: PolicyInput,
        prepared: PreparedPolicyInput | None = None,
    ) -> None:
        if self._prepared_input is policy_input:
            return
        if prepared is not None and prepared.policy_input is not policy_input:
            raise ValueError("prepared physical index uses a stale PolicyInput")
        self._prepared_input = policy_input
        self._request_by_id = (
            prepared.request_by_id
            if prepared is not None
            else {
                item.request_id: item for item in policy_input.runnable_frontier
            }
        )
        self._bundle_by_id = (
            prepared.bundle_by_id
            if prepared is not None
            else {
                item.bundle_id: item for item in policy_input.physical_kv.bundles
            }
        )
        self._potential_victims = (
            prepared.potential_victims
            if prepared is not None
            else tuple(
                item
                for item in policy_input.physical_kv.bundles
                if item.gpu_bytes > 0
                and item.marginal_reclaimable_bytes > 0
                and item.locked_bytes == 0
                and item.actionable
                and item.lease_kind not in {"ready", "running"}
            )
        )

    def pack(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        *,
        fairness: FairnessWindow | None = None,
        prepared: PreparedPolicyInput | None = None,
    ) -> ScenarioPlan:
        self._prepare(policy_input, prepared)
        if demand.snapshot_id != policy_input.snapshot_id:
            raise ValueError("scenario demand uses a stale physical snapshot")
        blockers = list(demand.blocker_reasons)
        if self.config.require_exact_extent_identity and not demand.physical_accounting_exact:
            blockers.append("exact_physical_accounting_unavailable")
            return self._infeasible(policy_input, demand, blockers)

        execution = []
        per_workflow: dict[str, int] = {}
        admission: dict[str, str] = {}
        for request_id in demand.candidate_request_ids:
            invocation = self._request_by_id.get(request_id)
            if invocation is None:
                blockers.append(f"unknown_request:{request_id}")
                admission[request_id] = "defer"
                continue
            if (
                fairness is not None
                and invocation.workflow_id not in fairness.eligible_workflow_ids
            ):
                blockers.append(f"fairness_ineligible:{invocation.workflow_id}")
                admission[request_id] = "defer"
                continue
            count = per_workflow.get(invocation.workflow_id, 0)
            if count >= self.config.max_frontier_per_workflow:
                blockers.append(f"workflow_frontier_limit:{invocation.workflow_id}")
                admission[request_id] = "defer"
                continue
            per_workflow[invocation.workflow_id] = count + 1
            execution.append(request_id)

        if not execution and not demand.speculative_only:
            blockers.append("no_fair_runnable_request")
            return self._infeasible(policy_input, demand, blockers, admission)

        required = [
            self._bundle_by_id[item]
            for item in demand.required_gpu_bundles
            if item in self._bundle_by_id
        ]
        missing_required = set(demand.required_gpu_bundles) - set(
            self._bundle_by_id
        )
        blockers.extend(f"missing_required_bundle:{item}" for item in missing_required)
        if missing_required:
            return self._infeasible(policy_input, demand, blockers, admission)
        blocked_required = [
            bundle
            for bundle in required
            if (
                bundle.residency == "prefetching"
                or (
                    bundle.gpu_bytes < bundle.physical_unique_bytes
                    and not bundle.actionable
                )
            )
        ]
        if blocked_required:
            blockers.extend(
                f"required_bundle_not_actionable:{bundle.bundle_id}:"
                + ",".join(bundle.blocker_codes)
                for bundle in blocked_required
            )
            return self._infeasible(policy_input, demand, blockers, admission)

        actions: dict[str, ResidencyAction] = {}
        h2d_bytes = 0
        for bundle in required:
            missing = max(0, bundle.physical_unique_bytes - bundle.gpu_bytes)
            if missing:
                actions[bundle.bundle_id] = ResidencyAction.PREFETCH_GPU
                h2d_bytes += missing
            else:
                actions[bundle.bundle_id] = ResidencyAction.KEEP

        deferred_startup = sum(
            bytes_
            for request_id, bytes_ in demand.startup_bytes_by_request.items()
            if request_id not in execution
        )
        projected_peak = max(
            policy_input.resources.hbm_used_bytes
            + policy_input.resources.hbm_reserved_bytes,
            demand.projected_hbm_peak_bytes - deferred_startup,
        )
        capacity = policy_input.resources.hbm_capacity_bytes
        shortage = max(0, projected_peak - capacity)
        choice = _VictimChoice((), 0, 0, 0)
        if shortage:
            candidates = self._victim_candidates(
                policy_input,
                demand,
                shortage=shortage,
            )
            choice = self._choose_victims(
                policy_input,
                candidates,
                shortage=shortage,
            )
            if choice.reclaimed_bytes < shortage:
                blockers.append(
                    f"hbm_capacity_shortage:{shortage - choice.reclaimed_bytes}"
                )
                locked = sum(
                    bundle.locked_bytes
                    for bundle in policy_input.physical_kv.bundles
                    if bundle.bundle_id in demand.optional_gpu_bundles
                )
                if locked:
                    blockers.append(f"locked_optional_bytes:{locked}")
                return self._infeasible(
                    policy_input,
                    demand,
                    blockers,
                    admission,
                    h2d_bytes=h2d_bytes,
                )
            actions.update(
                {bundle.bundle_id: action for bundle, action in choice.actions}
            )

        for request_id in execution:
            admission[request_id] = (
                "restore_then_admit" if h2d_bytes else "admit"
            )
        final_peak = projected_peak - choice.reclaimed_bytes
        horizon_ms = max(1.0, demand.earliest_ready_p90_ms)
        d2h_stall = (
            choice.d2h_bytes * policy_input.resources.unhidden_stall_per_byte
        )
        h2d_stall = self._transfer_time_ms(
            h2d_bytes,
            policy_input.resources.h2d_service_bytes_per_ms,
            policy_input.resources.transfer_setup_p50_ms,
        )
        urgent_delay = self._urgent_transfer_delay(policy_input)
        recompute_ms = choice.recompute_tokens * self.config.recompute_ms_per_token
        expected_stall = d2h_stall + h2d_stall + urgent_delay + recompute_ms
        return ScenarioPlan(
            snapshot_id=policy_input.snapshot_id,
            scenario_id=demand.scenario_id,
            execution_order=tuple(execution),
            admission_actions=admission,
            bundle_actions=actions,
            feasible=True,
            expected_unhidden_stall_ms=expected_stall,
            hbm_time_byte_ms=max(0, final_peak) * horizon_ms,
            d2h_bytes=choice.d2h_bytes,
            h2d_bytes=h2d_bytes,
            recompute_tokens=choice.recompute_tokens,
            projected_hbm_peak_bytes=final_peak,
            reclaimed_bytes=choice.reclaimed_bytes,
            physical_accounting_exact=demand.physical_accounting_exact,
            blocker_reasons=tuple(blockers),
        )

    def _victim_candidates(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        *,
        shortage: int,
    ) -> tuple[PhysicalBundleSnapshot, ...]:
        emergency = (
            policy_input.resources.hbm_used_bytes
            + policy_input.resources.hbm_reserved_bytes
        ) / policy_input.resources.hbm_capacity_bytes >= self.config.emergency_hbm_ratio
        result = []
        optional_ids = set(demand.optional_gpu_bundles)
        for bundle in self._potential_victims:
            if bundle.bundle_id not in optional_ids:
                continue
            age_ms = max(0.0, policy_input.resources.ts_ms - bundle.last_access_ms)
            if (
                demand.transition
                in {ScenarioTransition.HANDOFF, ScenarioTransition.CYCLIC_REACTIVATION}
                and age_ms < self.config.handoff_hysteresis_ms
                and not emergency
            ):
                continue
            result.append(bundle)
        result.sort(
            key=lambda item: (
                item.scope != "exclusive_suffix",
                item.last_access_ms,
                -min(shortage, item.marginal_reclaimable_bytes),
                item.bundle_id,
            )
        )
        return tuple(result[: self.config.max_victim_candidates])

    def _choose_victims(
        self,
        policy_input: PolicyInput,
        candidates: Sequence[PhysicalBundleSnapshot],
        *,
        shortage: int,
    ) -> _VictimChoice:
        best: tuple[tuple[object, ...], _VictimChoice] | None = None

        def visit(
            index: int,
            actions: list[tuple[PhysicalBundleSnapshot, ResidencyAction]],
            reclaimed: int,
            d2h_bytes: int,
            host_bytes: int,
            recompute_tokens: int,
        ) -> None:
            nonlocal best
            if reclaimed >= shortage or index == len(candidates):
                if reclaimed < shortage:
                    return
                stall = (
                    d2h_bytes * policy_input.resources.unhidden_stall_per_byte
                    + recompute_tokens * self.config.recompute_ms_per_token
                )
                key: tuple[object, ...] = (
                    stall,
                    recompute_tokens,
                    d2h_bytes,
                    reclaimed - shortage,
                    tuple((item.bundle_id, action.value) for item, action in actions),
                )
                choice = _VictimChoice(
                    tuple(actions), reclaimed, d2h_bytes, recompute_tokens
                )
                if best is None or key < best[0]:
                    best = (key, choice)
                return

            bundle = candidates[index]
            visit(
                index + 1,
                actions,
                reclaimed,
                d2h_bytes,
                host_bytes,
                recompute_tokens,
            )
            host_needed = max(0, bundle.gpu_bytes - bundle.cpu_bytes)
            if host_bytes + host_needed <= policy_input.resources.host_free_bytes:
                actions.append((bundle, ResidencyAction.COMMIT_CPU))
                visit(
                    index + 1,
                    actions,
                    reclaimed + bundle.marginal_reclaimable_bytes,
                    d2h_bytes + host_needed,
                    host_bytes + host_needed,
                    recompute_tokens,
                )
                actions.pop()
            if self.config.allow_recompute:
                actions.append((bundle, ResidencyAction.DROP))
                tokens = math.ceil(
                    bundle.physical_unique_bytes / self.config.kv_bytes_per_token
                )
                visit(
                    index + 1,
                    actions,
                    reclaimed + bundle.marginal_reclaimable_bytes,
                    d2h_bytes,
                    host_bytes,
                    recompute_tokens + tokens,
                )
                actions.pop()

        visit(0, [], 0, 0, 0, 0)
        return best[1] if best is not None else _VictimChoice((), 0, 0, 0)

    def _infeasible(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        blockers: Sequence[str],
        admission: Mapping[str, str] | None = None,
        *,
        h2d_bytes: int = 0,
    ) -> ScenarioPlan:
        actions = {
            request_id: "defer" for request_id in demand.candidate_request_ids
        }
        actions.update(admission or {})
        return ScenarioPlan(
            snapshot_id=policy_input.snapshot_id,
            scenario_id=demand.scenario_id,
            execution_order=(),
            admission_actions=actions,
            bundle_actions={},
            feasible=False,
            expected_unhidden_stall_ms=0.0,
            hbm_time_byte_ms=0.0,
            d2h_bytes=0,
            h2d_bytes=h2d_bytes,
            recompute_tokens=0,
            projected_hbm_peak_bytes=demand.projected_hbm_peak_bytes,
            reclaimed_bytes=0,
            physical_accounting_exact=demand.physical_accounting_exact,
            blocker_reasons=tuple(blockers),
        )

    @staticmethod
    def _transfer_time_ms(bytes_: int, rate: float, setup_ms: float) -> float:
        if bytes_ <= 0:
            return 0.0
        if rate <= 0:
            return math.inf
        return setup_ms + bytes_ / rate

    def _urgent_transfer_delay(self, policy_input: PolicyInput) -> float:
        resources = policy_input.resources
        return self._transfer_time_ms(
            resources.urgent_h2d_bytes,
            resources.h2d_service_bytes_per_ms,
            resources.transfer_setup_p50_ms,
        ) + self._transfer_time_ms(
            resources.urgent_d2h_bytes,
            resources.d2h_service_bytes_per_ms,
            resources.transfer_setup_p50_ms,
        )
