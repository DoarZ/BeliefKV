from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Iterable, Mapping

from beliefkv.control.causal_graph import (
    RuntimeCausalContextGraph,
)


class CausalAtomKind(str, Enum):
    JOIN = "join"
    BLOCKER = "blocker"
    BLOCKING_CHAIN = "blocking_chain"
    MESSAGE = "message"
    SINGLE = "single"


@dataclass(frozen=True)
class CausalAtom:
    """An indivisible causal closure used by one belief snapshot."""

    atom_id: str
    kind: CausalAtomKind
    invocation_ids: tuple[str, ...]
    join_ids: tuple[str, ...] = ()
    blocker_set_ids: tuple[str, ...] = ()
    estimated_model_cost: int = 1

    def __post_init__(self) -> None:
        if not self.atom_id or not self.invocation_ids:
            raise ValueError("causal atom identity and invocations are required")
        object.__setattr__(self, "kind", CausalAtomKind(self.kind))
        for name in ("invocation_ids", "join_ids", "blocker_set_ids"):
            values = tuple(sorted(set(getattr(self, name))))
            if any(not value for value in values):
                raise ValueError(f"causal atom {name} must be non-empty")
            object.__setattr__(self, name, values)
        if self.estimated_model_cost <= 0:
            raise ValueError("causal atom model cost must be positive")


@dataclass(frozen=True)
class BeliefScopeConfig:
    max_atomic_groups: int = 8
    max_total_model_cost: int = 32
    invocation_cost: int = 1
    join_cost: int = 2
    blocker_set_cost: int = 1
    message_edge_cost: int = 1

    def __post_init__(self) -> None:
        if min(
            self.max_atomic_groups,
            self.max_total_model_cost,
            self.invocation_cost,
            self.join_cost,
            self.blocker_set_cost,
            self.message_edge_cost,
        ) <= 0:
            raise ValueError("belief scope limits and costs must be positive")


@dataclass(frozen=True)
class BeliefScope:
    scope_id: str
    graph_version: int
    active_invocation_ids: tuple[str, ...]
    included_atoms: tuple[CausalAtom, ...]
    other_atoms: tuple[CausalAtom, ...]
    modeled_cost: int

    def __post_init__(self) -> None:
        if not self.scope_id or self.graph_version < 0 or self.modeled_cost < 0:
            raise ValueError("belief scope identity/version/cost is invalid")
        active = tuple(dict.fromkeys(self.active_invocation_ids))
        if any(not value for value in active):
            raise ValueError("active invocation IDs must be non-empty")
        object.__setattr__(self, "active_invocation_ids", active)
        included_ids = {
            invocation_id
            for atom in self.included_atoms
            for invocation_id in atom.invocation_ids
        }
        other_ids = {
            invocation_id
            for atom in self.other_atoms
            for invocation_id in atom.invocation_ids
        }
        if included_ids.intersection(other_ids):
            raise ValueError("an invocation cannot be partially modeled and OTHER")
        if not set(active).issubset(included_ids | other_ids):
            raise ValueError("every active invocation must belong to one complete atom")
        if self.modeled_cost != sum(
            atom.estimated_model_cost for atom in self.included_atoms
        ):
            raise ValueError("belief scope modeled cost does not match its atoms")

    @property
    def invocation_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    invocation_id
                    for atom in self.included_atoms
                    for invocation_id in atom.invocation_ids
                }
            )
        )

    @property
    def residual_invocation_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    invocation_id
                    for atom in self.other_atoms
                    for invocation_id in atom.invocation_ids
                }
            )
        )


class BeliefScopeBuilder:
    """Close active-window seeds over joins, blockers, and message producers."""

    def __init__(self, config: BeliefScopeConfig | None = None) -> None:
        self.config = config or BeliefScopeConfig()

    def build(
        self,
        graph: RuntimeCausalContextGraph,
        active_invocation_ids: Iterable[str],
        *,
        blocker_sets: Mapping[str, Iterable[str]] | None = None,
    ) -> BeliefScope:
        active = tuple(dict.fromkeys(str(item) for item in active_invocation_ids))
        unknown = sorted(set(active).difference(graph.invocations))
        if unknown:
            raise KeyError(f"unknown belief-scope invocations: {unknown}")
        blocker_sets = blocker_sets or {}
        relations: list[tuple[CausalAtomKind, str, set[str]]] = []

        for join_id, join in sorted(graph.joins.items()):
            if join.satisfied:
                continue
            unfinished = set(join.member_invocation_ids).difference(
                join.completed_member_ids
            )
            members = unfinished | set(join.waiter_invocation_ids)
            if members:
                relations.append((CausalAtomKind.JOIN, join_id, members))

        for invocation in graph.invocations.values():
            blocking = {
                child_id
                for child_id in invocation.blocking_child_ids
                if child_id in graph.invocations
            }
            if blocking:
                relations.append(
                    (
                        CausalAtomKind.BLOCKING_CHAIN,
                        invocation.invocation_id,
                        {invocation.invocation_id, *blocking},
                    )
                )

        for blocker_set_id, raw_ids in sorted(blocker_sets.items()):
            members = {
                str(item) for item in raw_ids if str(item) in graph.invocations
            }
            if blocker_set_id in graph.invocations:
                members.add(blocker_set_id)
            if members:
                relations.append(
                    (CausalAtomKind.BLOCKER, str(blocker_set_id), members)
                )

        for edge in graph.communication_edges.values():
            members = {
                edge.source_invocation_id,
                edge.target_invocation_id,
            }
            if members.issubset(graph.invocations):
                edge_id = f"{edge.source_invocation_id}->{edge.target_invocation_id}"
                relations.append((CausalAtomKind.MESSAGE, edge_id, members))

        components = self._seeded_components(active, relations)
        atoms = tuple(
            self._make_atom(component, relations, graph)
            for component in components
        )
        active_rank = {invocation_id: index for index, invocation_id in enumerate(active)}
        ordered_atoms = sorted(
            atoms,
            key=lambda atom: (
                min(active_rank.get(item, 1 << 30) for item in atom.invocation_ids),
                atom.estimated_model_cost,
                atom.atom_id,
            ),
        )
        included: list[CausalAtom] = []
        other: list[CausalAtom] = []
        modeled_cost = 0
        for atom in ordered_atoms:
            fits = (
                len(included) < self.config.max_atomic_groups
                and modeled_cost + atom.estimated_model_cost
                <= self.config.max_total_model_cost
            )
            if fits:
                included.append(atom)
                modeled_cost += atom.estimated_model_cost
            else:
                other.append(atom)

        digest_payload = (
            f"{graph.graph_version}|"
            + ",".join(active)
            + "|"
            + ",".join(atom.atom_id for atom in included)
            + "|other:"
            + ",".join(atom.atom_id for atom in other)
        )
        digest = hashlib.blake2b(
            digest_payload.encode(), digest_size=16, person=b"bkv-scope"
        ).hexdigest()
        return BeliefScope(
            scope_id=f"belief-scope-{digest}",
            graph_version=graph.graph_version,
            active_invocation_ids=active,
            included_atoms=tuple(included),
            other_atoms=tuple(other),
            modeled_cost=modeled_cost,
        )

    @staticmethod
    def _seeded_components(
        active: tuple[str, ...],
        relations: list[tuple[CausalAtomKind, str, set[str]]],
    ) -> tuple[frozenset[str], ...]:
        remaining = set(active)
        components: list[frozenset[str]] = []
        while remaining:
            seed = next(item for item in active if item in remaining)
            component = {seed}
            changed = True
            while changed:
                changed = False
                for _kind, _relation_id, members in relations:
                    if component.intersection(members) and not members.issubset(component):
                        component.update(members)
                        changed = True
            remaining.difference_update(component)
            components.append(frozenset(component))
        return tuple(components)

    def _make_atom(
        self,
        component: frozenset[str],
        relations: list[tuple[CausalAtomKind, str, set[str]]],
        graph: RuntimeCausalContextGraph,
    ) -> CausalAtom:
        relevant = [
            (kind, relation_id, members)
            for kind, relation_id, members in relations
            if members.intersection(component)
        ]
        priority = {
            CausalAtomKind.JOIN: 0,
            CausalAtomKind.BLOCKER: 1,
            CausalAtomKind.BLOCKING_CHAIN: 2,
            CausalAtomKind.MESSAGE: 3,
            CausalAtomKind.SINGLE: 4,
        }
        kind = min(
            (item[0] for item in relevant),
            key=lambda item: priority[item],
            default=CausalAtomKind.SINGLE,
        )
        join_ids = tuple(
            sorted(
                relation_id
                for relation_kind, relation_id, _members in relevant
                if relation_kind == CausalAtomKind.JOIN
            )
        )
        blocker_ids = tuple(
            sorted(
                relation_id
                for relation_kind, relation_id, _members in relevant
                if relation_kind == CausalAtomKind.BLOCKER
            )
        )
        message_edges = sum(
            relation_kind == CausalAtomKind.MESSAGE
            for relation_kind, _relation_id, _members in relevant
        )
        cost = (
            len(component) * self.config.invocation_cost
            + len(join_ids) * self.config.join_cost
            + len(blocker_ids) * self.config.blocker_set_cost
            + message_edges * self.config.message_edge_cost
        )
        invocation_ids = tuple(sorted(component))
        digest = hashlib.blake2b(
            (kind.value + "|" + ",".join(invocation_ids)).encode(),
            digest_size=12,
            person=b"bkv-atom",
        ).hexdigest()
        if any(item not in graph.invocations for item in invocation_ids):
            raise ValueError("causal closure references an unknown invocation")
        return CausalAtom(
            atom_id=f"causal-atom-{digest}",
            kind=kind,
            invocation_ids=invocation_ids,
            join_ids=join_ids,
            blocker_set_ids=blocker_ids,
            estimated_model_cost=max(1, cost),
        )


class BoundaryEvent(str, Enum):
    TOOL = "tool"
    SPAWN = "spawn"
    HANDOFF = "handoff"
    MESSAGE = "message"
    RETURN = "return"
    FINAL = "final"
    UNKNOWN = "unknown"


class DependencyMode(str, Enum):
    NONE = "none"
    EXTERNAL = "external"
    JOIN_ALL = "join_all"
    JOIN_ANY = "join_any"
    PRODUCER = "producer"


class DemandPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ExternalDemandSegment:
    segment_kind: str
    service_family: str
    residual_delay_ms: float
    terminal_status: str = "success"

    def __post_init__(self) -> None:
        if not self.segment_kind or not self.service_family or not self.terminal_status:
            raise ValueError("external demand segment identity/status is required")
        if (
            not math.isfinite(self.residual_delay_ms)
            or self.residual_delay_ms < 0
        ):
            raise ValueError("external residual delay must be finite and non-negative")


@dataclass(frozen=True)
class FrontierDemandOutcome:
    invocation_id: str
    boundary_event: BoundaryEvent
    dependency_mode: DependencyMode
    phase: DemandPhase
    current_sequence_tokens: int
    remaining_decode_tokens: int
    prompt_growth_tokens: int
    next_output_tokens: int
    external_segments: tuple[ExternalDemandSegment, ...] = ()
    dependency_invocation_ids: tuple[str, ...] = ()
    join_id: str | None = None
    release_on_boundary: bool = True
    existing_target_invocation_id: str | None = None
    anonymous_role_class: str | None = None

    def __post_init__(self) -> None:
        if not self.invocation_id:
            raise ValueError("belief outcome invocation is required")
        object.__setattr__(self, "boundary_event", BoundaryEvent(self.boundary_event))
        object.__setattr__(self, "dependency_mode", DependencyMode(self.dependency_mode))
        object.__setattr__(self, "phase", DemandPhase(self.phase))
        token_demands = (
            self.current_sequence_tokens,
            self.remaining_decode_tokens,
            self.prompt_growth_tokens,
            self.next_output_tokens,
        )
        if min(token_demands) < 0:
            raise ValueError("belief outcome token demand must be non-negative")
        dependency_ids = tuple(sorted(set(self.dependency_invocation_ids)))
        if any(not item or item == self.invocation_id for item in dependency_ids):
            raise ValueError("dependency invocation IDs must be non-empty peers")
        object.__setattr__(self, "dependency_invocation_ids", dependency_ids)
        object.__setattr__(self, "external_segments", tuple(self.external_segments))
        if self.dependency_mode in {DependencyMode.JOIN_ALL, DependencyMode.JOIN_ANY}:
            if not dependency_ids:
                raise ValueError("JOIN demand must retain every dependency member")
            if not self.join_id and self.dependency_mode != DependencyMode.JOIN_ALL:
                raise ValueError("JOIN_ANY demand requires a join identity")
        elif self.dependency_mode == DependencyMode.PRODUCER and not dependency_ids:
            raise ValueError("producer demand requires producer invocation IDs")
        elif self.dependency_mode == DependencyMode.EXTERNAL and not self.external_segments:
            raise ValueError("external demand requires an external segment")
        if self.existing_target_invocation_id and self.anonymous_role_class:
            raise ValueError("belief outcome target must be existing or anonymous")


@dataclass(frozen=True)
class DemandScenario:
    scenario_id: str
    outcomes: tuple[FrontierDemandOutcome, ...]
    probability_mass: float

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.outcomes:
            raise ValueError("belief scenario identity and outcomes are required")
        if not 0 < self.probability_mass <= 1:
            raise ValueError("belief scenario probability must be in (0, 1]")
        invocation_ids = [item.invocation_id for item in self.outcomes]
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ValueError("a scenario must assign each invocation once")


@dataclass(frozen=True)
class FinitePlanningHorizon:
    max_gpu_service_ms: float = 100.0
    max_transitions_per_invocation: int = 2
    stop_at_next_observable_boundary: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_gpu_service_ms) or self.max_gpu_service_ms <= 0:
            raise ValueError("planning service horizon must be finite and positive")
        if self.max_transitions_per_invocation <= 0:
            raise ValueError("planning transition horizon must be positive")


@dataclass(frozen=True)
class OtherResidualPolicy:
    no_unlock_credit: bool = True
    use_earliest_legal_reentry: bool = True
    prepare_host_allowed: bool = True
    commit_cpu_allowed: bool = False
    retraction_allowed: bool = False
    finite_risk_bound: bool = False


@dataclass(frozen=True)
class PredictiveEvidenceReadSet:
    graph_version: int
    page_revision: int
    topology_revision: int
    fairness_revision: int
    admission_revision: int
    transfer_epoch: int
    obligation_revision: int
    lease_revision: int
    grace_revision: int
    parser_frontier_revision: int
    model_version: str

    def __post_init__(self) -> None:
        revisions = (
            self.graph_version,
            self.page_revision,
            self.topology_revision,
            self.fairness_revision,
            self.admission_revision,
            self.transfer_epoch,
            self.obligation_revision,
            self.lease_revision,
            self.grace_revision,
            self.parser_frontier_revision,
        )
        if min(revisions) < 0 or not self.model_version:
            raise ValueError("predictive evidence revisions/model are invalid")


@dataclass(frozen=True)
class FrontierBeliefSnapshot:
    belief_id: str
    generated_ts_ms: float
    scope: BeliefScope
    scenarios: tuple[DemandScenario, ...]
    other_probability_mass: float
    calibration_coverage: float
    support_level: str
    ood_reasons: tuple[str, ...]
    evidence_read_set: PredictiveEvidenceReadSet
    horizon: FinitePlanningHorizon = FinitePlanningHorizon()
    other_policy: OtherResidualPolicy = OtherResidualPolicy()

    def __post_init__(self) -> None:
        if not self.belief_id or self.generated_ts_ms < 0 or not self.support_level:
            raise ValueError("frontier belief identity/time/support are required")
        if not 0 <= self.other_probability_mass <= 1:
            raise ValueError("OTHER probability must be in [0, 1]")
        if not 0 <= self.calibration_coverage <= 1:
            raise ValueError("calibration coverage must be in [0, 1]")
        total = self.other_probability_mass + sum(
            item.probability_mass for item in self.scenarios
        )
        if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("scenario and OTHER probability must sum to one")
        scenario_ids = [item.scenario_id for item in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("belief scenario IDs must be unique")
        scoped = set(self.scope.invocation_ids)
        if any(
            set(outcome.invocation_id for outcome in scenario.outcomes) != scoped
            for scenario in self.scenarios
        ):
            raise ValueError("each global scenario must assign the complete modeled scope")
        object.__setattr__(self, "ood_reasons", tuple(sorted(set(self.ood_reasons))))
