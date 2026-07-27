from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from beliefkv.policy.reference import (
    MetadataSource,
    PhysicalBundleSnapshot,
    PolicyInput,
    RunnableInvocation,
)


class ScenarioTransition(str, Enum):
    BLOCKING = "blocking"
    NONBLOCKING = "nonblocking"
    HANDOFF = "handoff"
    MULTI_CONSUMER = "multi_consumer"
    CYCLIC_REACTIVATION = "cyclic_reactivation"
    FRESH_SPAWN = "fresh_spawn"
    TOOL_WAIT = "tool_wait"


@dataclass(frozen=True)
class FrontierScenario:
    scenario_id: str
    probability: float
    transition: ScenarioTransition
    candidate_request_ids: tuple[str, ...] = ()
    consumer_context_ids: tuple[str, ...] = ()
    keep_context_ids: tuple[str, ...] = ()
    projected_growth_bytes: Mapping[str, int] = field(default_factory=dict)
    anonymous_fresh_bytes: int = 0
    earliest_ready_p50_ms: float = 0.0
    earliest_ready_p90_ms: float = 0.0
    source: MetadataSource = MetadataSource.OBSERVED

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("scenario probability must be in [0, 1]")
        if self.anonymous_fresh_bytes < 0:
            raise ValueError("anonymous_fresh_bytes must be non-negative")
        if (
            not math.isfinite(self.earliest_ready_p50_ms)
            or not math.isfinite(self.earliest_ready_p90_ms)
            or self.earliest_ready_p50_ms < 0
            or self.earliest_ready_p90_ms < self.earliest_ready_p50_ms
        ):
            raise ValueError("scenario ready quantiles are invalid")
        for name, values in (
            ("candidate_request_ids", self.candidate_request_ids),
            ("consumer_context_ids", self.consumer_context_ids),
            ("keep_context_ids", self.keep_context_ids),
        ):
            if len(set(values)) != len(values) or any(not item for item in values):
                raise ValueError(f"{name} must contain unique non-empty IDs")
        growth: dict[str, int] = {}
        for context_id, bytes_ in sorted(self.projected_growth_bytes.items()):
            if not context_id or bytes_ < 0:
                raise ValueError("projected growth must use valid contexts and bytes")
            growth[context_id] = int(bytes_)
        object.__setattr__(self, "projected_growth_bytes", MappingProxyType(growth))


@dataclass(frozen=True)
class ScenarioDemand:
    snapshot_id: str
    scenario_id: str
    probability: float
    transition: ScenarioTransition
    source: MetadataSource
    candidate_invocation_ids: tuple[str, ...]
    candidate_request_ids: tuple[str, ...]
    consumer_context_ids: tuple[str, ...]
    required_context_ids: tuple[str, ...]
    required_gpu_bundles: tuple[str, ...]
    optional_gpu_bundles: tuple[str, ...]
    startup_bytes_by_request: Mapping[str, int]
    projected_growth_bytes: Mapping[str, int]
    projected_new_bytes: int
    projected_hbm_peak_bytes: int
    required_h2d_bytes: int
    earliest_ready_p50_ms: float
    earliest_ready_p90_ms: float
    physical_accounting_exact: bool
    blocker_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.scenario_id:
            raise ValueError("scenario demand IDs must be non-empty")
        growth = {
            str(key): int(value)
            for key, value in sorted(self.projected_growth_bytes.items())
        }
        object.__setattr__(self, "projected_growth_bytes", MappingProxyType(growth))
        startup = {
            str(key): int(value)
            for key, value in sorted(self.startup_bytes_by_request.items())
        }
        object.__setattr__(
            self, "startup_bytes_by_request", MappingProxyType(startup)
        )
        object.__setattr__(
            self, "blocker_reasons", tuple(sorted(set(self.blocker_reasons)))
        )

    @property
    def speculative_only(self) -> bool:
        return not self.candidate_request_ids

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "scenario_id": self.scenario_id,
            "probability": self.probability,
            "transition": self.transition.value,
            "source": self.source.value,
            "candidate_invocation_ids": list(self.candidate_invocation_ids),
            "candidate_request_ids": list(self.candidate_request_ids),
            "consumer_context_ids": list(self.consumer_context_ids),
            "required_context_ids": list(self.required_context_ids),
            "required_gpu_bundles": list(self.required_gpu_bundles),
            "optional_gpu_bundles": list(self.optional_gpu_bundles),
            "startup_bytes_by_request": dict(self.startup_bytes_by_request),
            "projected_growth_bytes": dict(self.projected_growth_bytes),
            "projected_new_bytes": self.projected_new_bytes,
            "projected_hbm_peak_bytes": self.projected_hbm_peak_bytes,
            "required_h2d_bytes": self.required_h2d_bytes,
            "earliest_ready_p50_ms": self.earliest_ready_p50_ms,
            "earliest_ready_p90_ms": self.earliest_ready_p90_ms,
            "physical_accounting_exact": self.physical_accounting_exact,
            "blocker_reasons": list(self.blocker_reasons),
        }


@dataclass(frozen=True)
class PreparedPolicyInput:
    policy_input: PolicyInput
    request_by_id: Mapping[str, RunnableInvocation]
    bundle_by_id: Mapping[str, PhysicalBundleSnapshot]
    bundle_ids: tuple[str, ...]
    bundle_ids_by_context: Mapping[str, frozenset[str]]
    potential_victims: tuple[PhysicalBundleSnapshot, ...]
    physical_accounting_exact: bool
    physical_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("request_by_id", "bundle_by_id", "bundle_ids_by_context"):
            if not isinstance(getattr(self, name), MappingProxyType):
                raise TypeError(f"{name} must be an immutable mapping")


class ScenarioPhysicalizer:
    """Map causal frontier scenarios to unique physical KV demand."""

    def __init__(self) -> None:
        self._prepared: PreparedPolicyInput | None = None

    def prepare(self, policy_input: PolicyInput) -> PreparedPolicyInput:
        if self._prepared is not None and self._prepared.policy_input is policy_input:
            return self._prepared
        request_by_id = {
            item.request_id: item for item in policy_input.runnable_frontier
        }
        bundle_by_id = {
            item.bundle_id: item for item in policy_input.physical_kv.bundles
        }
        bundle_ids_by_context: dict[str, set[str]] = {}
        extent_owner: dict[str, str] = {}
        blockers: list[str] = []
        potential_victims: list[PhysicalBundleSnapshot] = []
        exact = True
        for bundle in policy_input.physical_kv.bundles:
            if not bundle.extent_ids:
                exact = False
                blockers.append(f"missing_extent_identity:{bundle.bundle_id}")
            for extent_id in bundle.extent_ids:
                previous = extent_owner.setdefault(extent_id, bundle.bundle_id)
                if previous != bundle.bundle_id:
                    exact = False
                    blockers.append(
                        f"overlapping_extent:{extent_id}:{previous}:{bundle.bundle_id}"
                    )
            for context_id in bundle.owner_context_ids:
                bundle_ids_by_context.setdefault(context_id, set()).add(
                    bundle.bundle_id
                )
            if (
                bundle.gpu_bytes > 0
                and bundle.marginal_reclaimable_bytes > 0
                and bundle.locked_bytes == 0
                and bundle.actionable
                and bundle.lease_kind not in {"ready", "running"}
            ):
                potential_victims.append(bundle)
        self._prepared = PreparedPolicyInput(
            policy_input=policy_input,
            request_by_id=MappingProxyType(request_by_id),
            bundle_by_id=MappingProxyType(bundle_by_id),
            bundle_ids=tuple(bundle_by_id),
            bundle_ids_by_context=MappingProxyType(
                {
                    context_id: frozenset(bundle_ids)
                    for context_id, bundle_ids in bundle_ids_by_context.items()
                }
            ),
            potential_victims=tuple(potential_victims),
            physical_accounting_exact=exact,
            physical_blockers=tuple(blockers),
        )
        return self._prepared

    def bundles_for_context(
        self,
        policy_input: PolicyInput,
        context_id: str,
    ) -> tuple[PhysicalBundleSnapshot, ...]:
        prepared = self.prepare(policy_input)
        return tuple(
            prepared.bundle_by_id[bundle_id]
            for bundle_id in sorted(
                prepared.bundle_ids_by_context.get(context_id, frozenset())
            )
        )

    def bundle_by_id(
        self,
        policy_input: PolicyInput,
        bundle_id: str,
    ) -> PhysicalBundleSnapshot:
        return self.prepare(policy_input).bundle_by_id[bundle_id]

    def physicalize(
        self,
        policy_input: PolicyInput,
        scenario: FrontierScenario,
        *,
        prepared: PreparedPolicyInput | None = None,
    ) -> ScenarioDemand:
        prepared = prepared or self.prepare(policy_input)
        if prepared.policy_input is not policy_input:
            raise ValueError("prepared physical index uses a stale PolicyInput")
        unknown_requests = set(scenario.candidate_request_ids) - set(
            prepared.request_by_id
        )
        if unknown_requests:
            raise ValueError(
                "predicted or unknown requests cannot enter an execution scenario: "
                f"{sorted(unknown_requests)}"
            )
        candidates = tuple(
            prepared.request_by_id[item] for item in scenario.candidate_request_ids
        )
        candidate_contexts = {item.context_id for item in candidates}
        required_contexts = set(candidate_contexts)
        required_contexts.update(scenario.consumer_context_ids)
        required_contexts.update(scenario.keep_context_ids)

        required_ids: set[str] = set()
        for context_id in required_contexts:
            required_ids.update(
                prepared.bundle_ids_by_context.get(context_id, frozenset())
            )
        required = tuple(sorted(required_ids))
        optional = (
            prepared.bundle_ids
            if not required_ids
            else tuple(
                bundle_id
                for bundle_id in prepared.bundle_ids
                if bundle_id not in required_ids
            )
        )
        h2d_bytes = sum(
            max(
                0,
                prepared.bundle_by_id[bundle_id].physical_unique_bytes
                - prepared.bundle_by_id[bundle_id].gpu_bytes,
            )
            for bundle_id in required
        )
        blockers = list(prepared.physical_blockers)

        growth = dict(scenario.projected_growth_bytes)
        projected_new_bytes = sum(growth.values()) + scenario.anonymous_fresh_bytes
        startup_by_request = {
            item.request_id: _unreserved_startup_bytes(item)
            for item in candidates
        }
        candidate_startup = sum(startup_by_request.values())
        projected_new_bytes += candidate_startup
        projected_peak = (
            policy_input.resources.hbm_used_bytes
            + policy_input.resources.hbm_reserved_bytes
            + h2d_bytes
            + projected_new_bytes
        )
        if scenario.transition == ScenarioTransition.FRESH_SPAWN:
            parent_contexts = {
                item.context_id for item in candidates
            }.intersection(scenario.keep_context_ids)
            if parent_contexts and scenario.anonymous_fresh_bytes == 0:
                blockers.append("fresh_spawn_missing_independent_kv_estimate")

        return ScenarioDemand(
            snapshot_id=policy_input.snapshot_id,
            scenario_id=scenario.scenario_id,
            probability=scenario.probability,
            transition=scenario.transition,
            source=scenario.source,
            candidate_invocation_ids=tuple(item.invocation_id for item in candidates),
            candidate_request_ids=tuple(item.request_id for item in candidates),
            consumer_context_ids=tuple(sorted(scenario.consumer_context_ids)),
            required_context_ids=tuple(sorted(required_contexts)),
            required_gpu_bundles=required,
            optional_gpu_bundles=optional,
            startup_bytes_by_request=startup_by_request,
            projected_growth_bytes=growth,
            projected_new_bytes=projected_new_bytes,
            projected_hbm_peak_bytes=projected_peak,
            required_h2d_bytes=h2d_bytes,
            earliest_ready_p50_ms=scenario.earliest_ready_p50_ms,
            earliest_ready_p90_ms=scenario.earliest_ready_p90_ms,
            physical_accounting_exact=prepared.physical_accounting_exact,
            blocker_reasons=tuple(blockers),
        )


def _unreserved_startup_bytes(invocation: object) -> int:
    causal_class = str(getattr(invocation, "causal_class"))
    if causal_class.startswith(("reserved_admission:", "engine_waiting:")):
        return 0
    return int(getattr(invocation, "startup_bytes"))
