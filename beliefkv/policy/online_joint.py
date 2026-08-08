from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Iterable

from beliefkv.policy.joint_scheduler import (
    JointPlan,
    JointPlanComponentValidation,
    JointPlannerMode,
)
from beliefkv.policy.reference import AdmissionAction
from beliefkv.policy.reference import ResidencyAction


@dataclass(frozen=True)
class OnlineJointPlanView:
    """Validated execution/admission slice consumable by one scheduler epoch."""

    plan_id: str
    ordered_request_ids: tuple[str, ...]
    immediate_request_ids: tuple[str, ...]
    restore_requirements: tuple[tuple[str, tuple[str, ...]], ...]
    deferred_request_ids: tuple[str, ...]
    residency_intent_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("online JointPlan view requires a plan id")
        for values, name in (
            (self.ordered_request_ids, "ordered requests"),
            (self.immediate_request_ids, "immediate requests"),
            (self.deferred_request_ids, "deferred requests"),
            (self.residency_intent_indices, "residency intents"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"online JointPlan {name} must be unique")
        ordered = set(self.ordered_request_ids)
        if not set(self.immediate_request_ids).issubset(ordered):
            raise ValueError("immediate requests must belong to execution order")
        restore_ids = [request_id for request_id, _ in self.restore_requirements]
        if len(restore_ids) != len(set(restore_ids)):
            raise ValueError("restore requirements must have unique requests")
        if not set(restore_ids).issubset(ordered):
            raise ValueError("restore requests must belong to execution order")
        if set(restore_ids).intersection(self.immediate_request_ids):
            raise ValueError("a request cannot be immediate and restore-blocked")
        for _request_id, bundle_ids in self.restore_requirements:
            if not bundle_ids or len(bundle_ids) != len(set(bundle_ids)):
                raise ValueError("restore requirements need unique bundle ids")

    @property
    def restore_by_request(self) -> dict[str, tuple[str, ...]]:
        return dict(self.restore_requirements)


@dataclass(frozen=True)
class ActionSlice:
    slice_id: str
    kind: str
    action_key: str
    dependency_keys: tuple[str, ...]
    committed: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.slice_id or not self.kind or not self.action_key:
            raise ValueError("JointPlan action slice identity must be non-empty")
        object.__setattr__(
            self, "dependency_keys", tuple(sorted(set(self.dependency_keys)))
        )
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))
        if self.committed and self.reasons:
            raise ValueError("a committed action slice cannot carry rejection reasons")


class ActionGroupAtomicity(str, Enum):
    ALL_OR_NOTHING = "all_or_nothing"
    PREFIX_COMMITTABLE = "prefix_committable"


@dataclass(frozen=True)
class ActionGroupResourceCertificate:
    """Safe-point assumptions that must be rechecked for a complete group."""

    required_hbm_bytes: int = 0
    planned_reclaim_bytes: int = 0
    required_host_bytes: int = 0
    planned_pcie_bytes: int = 0
    topology_revision: int = 0
    allocator_revision: int = 0
    obligation_revision: int = 0
    lease_revision: int = 0
    grace_revision: int = 0
    physical_generation_fingerprints: tuple[tuple[str, str], ...] = ()
    restore_path_proven: bool = True
    finite_future_risk_bound: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.required_hbm_bytes,
            self.planned_reclaim_bytes,
            self.required_host_bytes,
            self.planned_pcie_bytes,
            self.topology_revision,
            self.allocator_revision,
            self.obligation_revision,
            self.lease_revision,
            self.grace_revision,
        )
        if min(numeric) < 0:
            raise ValueError("action-group resource certificate is invalid")
        fingerprints = tuple(sorted(set(self.physical_generation_fingerprints)))
        if any(not key or not value for key, value in fingerprints):
            raise ValueError("physical generation fingerprints must be non-empty")
        object.__setattr__(
            self, "physical_generation_fingerprints", fingerprints
        )


@dataclass(frozen=True)
class ActionGroupValidationState:
    hbm_available_bytes: int
    host_free_bytes: int
    topology_revision: int
    allocator_revision: int
    obligation_revision: int
    lease_revision: int
    grace_revision: int
    physical_generation_fingerprints: tuple[tuple[str, str], ...] = ()
    pcie_dispatch_available: bool = True
    recomputed_reclaim_bytes: int | None = None
    recomputed_required_hbm_bytes: int | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.hbm_available_bytes,
            self.host_free_bytes,
            self.topology_revision,
            self.allocator_revision,
            self.obligation_revision,
            self.lease_revision,
            self.grace_revision,
        )
        if min(numeric) < 0:
            raise ValueError("action-group validation state is invalid")
        if any(
            value is not None and value < 0
            for value in (
                self.recomputed_reclaim_bytes,
                self.recomputed_required_hbm_bytes,
            )
        ):
            raise ValueError("recomputed action-group bytes must be non-negative")


@dataclass(frozen=True)
class ActionGroup:
    group_id: str
    atomicity: ActionGroupAtomicity
    actions: tuple[ActionSlice, ...]
    dependency_dag: tuple[tuple[str, str], ...]
    resource_certificate: ActionGroupResourceCertificate
    compensation: tuple[str, ...]
    committed: bool
    reasons: tuple[str, ...] = ()
    evidence_read_set: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.group_id or not self.actions:
            raise ValueError("action group identity and actions are required")
        object.__setattr__(self, "atomicity", ActionGroupAtomicity(self.atomicity))
        action_ids = [item.slice_id for item in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action group slices must be unique")
        action_id_set = set(action_ids)
        edges = tuple(sorted(set(self.dependency_dag)))
        if any(
            not before
            or not after
            or before == after
            or before not in action_id_set
            or after not in action_id_set
            for before, after in edges
        ):
            raise ValueError("action group dependency DAG references invalid slices")
        self._validate_acyclic(action_id_set, edges)
        object.__setattr__(self, "dependency_dag", edges)
        compensation = tuple(dict.fromkeys(self.compensation))
        if any(not item for item in compensation):
            raise ValueError("action group compensation steps must be non-empty")
        object.__setattr__(self, "compensation", compensation)
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))
        evidence = tuple(sorted(set(self.evidence_read_set)))
        if any(not key or not value for key, value in evidence):
            raise ValueError("action group evidence read set is invalid")
        object.__setattr__(self, "evidence_read_set", evidence)
        if self.committed and (
            self.reasons or any(not item.committed for item in self.actions)
        ):
            raise ValueError("committed action group must contain only valid actions")
        if self.atomicity == ActionGroupAtomicity.PREFIX_COMMITTABLE:
            committed_ids = {
                item.slice_id for item in self.actions if item.committed
            }
            if any(
                after in committed_ids and before not in committed_ids
                for before, after in edges
            ):
                raise ValueError("committed prefix is not dependency-closed")

    @staticmethod
    def _validate_acyclic(
        action_ids: set[str], edges: tuple[tuple[str, str], ...]
    ) -> None:
        incoming = {action_id: 0 for action_id in action_ids}
        outgoing = {action_id: set() for action_id in action_ids}
        for before, after in edges:
            outgoing[before].add(after)
            incoming[after] += 1
        ready = [item for item, count in incoming.items() if count == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for target in outgoing[current]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
        if visited != len(action_ids):
            raise ValueError("action group dependency graph must be acyclic")


def validate_action_group_resource_certificate(
    group: ActionGroup,
    current: ActionGroupValidationState,
) -> tuple[str, ...]:
    certificate = group.resource_certificate
    reasons: list[str] = []
    if current.topology_revision < certificate.topology_revision:
        reasons.append("topology_revision_regressed")
    if current.allocator_revision < certificate.allocator_revision:
        reasons.append("allocator_revision_regressed")
    for name in ("obligation_revision", "lease_revision", "grace_revision"):
        if getattr(current, name) != getattr(certificate, name):
            reasons.append(name)
    expected = dict(certificate.physical_generation_fingerprints)
    observed = dict(current.physical_generation_fingerprints)
    physical_changed = any(
        observed.get(key) != value for key, value in expected.items()
    )
    if physical_changed:
        reasons.append("physical_generation")
    topology_changed = current.topology_revision != certificate.topology_revision
    if topology_changed and not expected:
        reasons.append("topology_change_unscoped")
    if topology_changed or physical_changed:
        if current.recomputed_reclaim_bytes is None:
            reasons.append("recomputed_reclaim_missing")
        if current.recomputed_required_hbm_bytes is None:
            reasons.append("recomputed_startup_missing")
    reclaim_bytes = (
        current.recomputed_reclaim_bytes
        if current.recomputed_reclaim_bytes is not None
        else certificate.planned_reclaim_bytes
    )
    required_hbm_bytes = (
        current.recomputed_required_hbm_bytes
        if current.recomputed_required_hbm_bytes is not None
        else certificate.required_hbm_bytes
    )
    if current.hbm_available_bytes + reclaim_bytes < required_hbm_bytes:
        reasons.append("hbm_capacity")
    if current.host_free_bytes < certificate.required_host_bytes:
        reasons.append("host_capacity")
    if certificate.planned_pcie_bytes and not current.pcie_dispatch_available:
        reasons.append("pcie_inflight")
    if not certificate.restore_path_proven:
        reasons.append("restore_path_unproven")
    if not certificate.finite_future_risk_bound:
        reasons.append("future_risk_unbounded")
    return tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class JointPlanEpoch:
    epoch_id: str
    source_plan_id: str
    planner_mode: JointPlannerMode
    view: OnlineJointPlanView
    action_slices: tuple[ActionSlice, ...]
    source_action_count: int
    committed_action_count: int
    action_groups: tuple[ActionGroup, ...] = ()

    def __post_init__(self) -> None:
        if not self.epoch_id or not self.source_plan_id:
            raise ValueError("JointPlan epoch identity must be non-empty")
        object.__setattr__(self, "planner_mode", JointPlannerMode(self.planner_mode))
        if min(self.source_action_count, self.committed_action_count) < 0:
            raise ValueError("JointPlan epoch action counts must be non-negative")
        if self.committed_action_count > self.source_action_count:
            raise ValueError("committed actions cannot exceed source actions")
        if self.view.plan_id != self.source_plan_id:
            raise ValueError("JointPlan epoch and view source differ")
        grouped_slice_ids = [
            action.slice_id
            for group in self.action_groups
            for action in group.actions
        ]
        if len(grouped_slice_ids) != len(set(grouped_slice_ids)):
            raise ValueError("an action slice cannot belong to multiple groups")

    @property
    def actionable_coverage(self) -> float:
        if self.source_action_count == 0:
            return 1.0
        return self.committed_action_count / self.source_action_count

    @property
    def committed_group_count(self) -> int:
        return sum(group.committed for group in self.action_groups)


def _action_groups_from_slices(
    slices: tuple[ActionSlice, ...],
) -> tuple[ActionGroup, ...]:
    """Build dependency-connected groups for legacy observed JointPlans.

    P6 plans will publish explicit groups. This derivation preserves P5 local
    validation while ensuring shared restore dependencies have one transaction
    boundary in diagnostics.
    """

    by_id = {item.slice_id: item for item in slices}
    adjacency = {slice_id: set() for slice_id in by_id}
    for item in slices:
        for dependency in item.dependency_keys:
            if dependency in by_id:
                adjacency[item.slice_id].add(dependency)
                adjacency[dependency].add(item.slice_id)
    groups: list[ActionGroup] = []
    remaining = set(by_id)
    sequence = 0
    while remaining:
        sequence += 1
        root = min(remaining)
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in component:
                    component.add(neighbor)
                    stack.append(neighbor)
        remaining.difference_update(component)
        actions = tuple(by_id[item] for item in sorted(component))
        reasons = tuple(
            sorted({reason for action in actions for reason in action.reasons})
        )
        groups.append(
            ActionGroup(
                group_id=f"derived-group-{sequence}",
                atomicity=ActionGroupAtomicity.ALL_OR_NOTHING,
                actions=actions,
                dependency_dag=tuple(
                    sorted(
                        (dependency, action.slice_id)
                        for action in actions
                        for dependency in action.dependency_keys
                        if dependency in component
                    )
                ),
                resource_certificate=ActionGroupResourceCertificate(),
                compensation=(),
                committed=all(action.committed for action in actions),
                reasons=() if all(action.committed for action in actions) else reasons,
            )
        )
    return tuple(groups)


def append_committed_action_slice(
    epoch: JointPlanEpoch,
    action: ActionSlice,
) -> JointPlanEpoch:
    """Append one committed action and rebuild dependency-closed groups."""

    if not action.committed:
        raise ValueError("only committed action slices can extend an online epoch")
    if any(item.slice_id == action.slice_id for item in epoch.action_slices):
        return epoch
    slices = epoch.action_slices + (action,)
    return JointPlanEpoch(
        epoch_id=epoch.epoch_id,
        source_plan_id=epoch.source_plan_id,
        planner_mode=epoch.planner_mode,
        view=epoch.view,
        action_slices=slices,
        source_action_count=len(slices),
        committed_action_count=sum(item.committed for item in slices),
        action_groups=_action_groups_from_slices(slices),
    )


@dataclass(frozen=True)
class OnlineJointPlanDecision:
    view: OnlineJointPlanView | None
    reason: str
    epoch: JointPlanEpoch | None = None

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("online JointPlan decision requires a reason")
        if self.view is not None and self.reason not in {
            "applicable",
            "partially_applicable",
            "no_action",
        }:
            raise ValueError("an online JointPlan view needs a commit reason")
        if self.epoch is not None and self.epoch.view != self.view:
            raise ValueError("JointPlan decision epoch and view differ")


def compile_bounded_seed_epoch(
    *,
    ordered_request_ids: Iterable[str],
    visible_request_ids: Iterable[str],
    epoch_sequence: int,
    emergency: bool = False,
    restore_requirements: Iterable[tuple[str, tuple[str, ...]]] = (),
) -> OnlineJointPlanDecision:
    """Create a safe-point seed using the same online epoch contract.

    The caller supplies the current fairness/causal order. Capacity remains
    enforced by the admission ticket compiler, so this O(K) path never binds
    stale Radix extents or claims speculative reclaim capacity.
    """

    if epoch_sequence < 0:
        raise ValueError("JointPlan epoch sequence must be non-negative")
    visible = tuple(dict.fromkeys(str(item) for item in visible_request_ids))
    visible_set = frozenset(visible)
    ordered = tuple(
        request_id
        for request_id in dict.fromkeys(str(item) for item in ordered_request_ids)
        if request_id in visible_set
    )
    ordered_set = frozenset(ordered)
    restore = tuple(
        sorted(
            (
                request_id,
                tuple(sorted(set(bundle_ids))),
            )
            for request_id, bundle_ids in restore_requirements
            if request_id in ordered_set and bundle_ids
        )
    )
    restore_ids = frozenset(request_id for request_id, _ in restore)
    immediate = tuple(
        request_id for request_id in ordered if request_id not in restore_ids
    )
    digest = hashlib.blake2b(
        (
            "|".join(ordered)
            + "\0"
            + "|".join(sorted(visible_set))
            + "\0"
            + "|".join(
                f"{request_id}:{','.join(bundle_ids)}"
                for request_id, bundle_ids in restore
            )
        ).encode(),
        digest_size=16,
        person=b"bkv-bounded-seed",
    ).hexdigest()
    plan_id = f"joint-seed-{digest}"
    view = OnlineJointPlanView(
        plan_id=plan_id,
        ordered_request_ids=ordered,
        immediate_request_ids=immediate,
        restore_requirements=restore,
        deferred_request_ids=tuple(sorted(visible_set.difference(ordered))),
        residency_intent_indices=(),
    )
    slices = tuple(
        ActionSlice(
            slice_id=f"request:{request_id}",
            kind="request",
            action_key=request_id,
            dependency_keys=(
                (f"restore:{request_id}",)
                if request_id in restore_ids
                else ()
            ),
            committed=True,
        )
        for request_id in ordered
    )
    mode = JointPlannerMode.EMERGENCY if emergency else JointPlannerMode.BOUNDED_SEED
    epoch = JointPlanEpoch(
        epoch_id=f"{plan_id}:epoch:{epoch_sequence}",
        source_plan_id=plan_id,
        planner_mode=mode if ordered else JointPlannerMode.NO_ACTION,
        view=view,
        action_slices=slices,
        source_action_count=len(slices),
        committed_action_count=len(slices),
        action_groups=_action_groups_from_slices(slices),
    )
    return OnlineJointPlanDecision(
        view,
        "applicable" if ordered else "no_action",
        epoch,
    )


def compile_online_joint_view(
    plan: JointPlan,
    validation: JointPlanComponentValidation,
    *,
    visible_request_ids: Iterable[str],
    epoch_sequence: int = 0,
) -> OnlineJointPlanDecision:
    """Compile only actions whose complete dependency slice is current.

    Physical residency remains a separate transaction. Requests that require a
    restore are surfaced as blockers and cannot receive an admission ticket
    until their acknowledged residency dependency is complete.
    """

    if epoch_sequence < 0:
        raise ValueError("JointPlan epoch sequence must be non-negative")
    if validation.global_reasons:
        return OnlineJointPlanDecision(None, "global_validation_failed")

    nonlocal_execution_reasons = tuple(
        reason
        for reason in validation.execution.reasons
        if not reason.startswith(("admission_invalid:", "admission_missing:"))
    )
    if nonlocal_execution_reasons:
        return OnlineJointPlanDecision(None, "execution_order_validation_failed")

    visible = frozenset(str(item) for item in visible_request_ids)
    admission_by_request = {item.request_id: item for item in plan.admissions}
    dependency_indices_by_request: dict[str, list[int]] = {}
    for dependency_index, dependency in enumerate(plan.dependencies):
        if dependency.before_request_id is None:
            continue
        dependency_indices_by_request.setdefault(
            dependency.before_request_id, []
        ).append(dependency_index)

    ordered: list[str] = []
    immediate: list[str] = []
    restore: list[tuple[str, tuple[str, ...]]] = []
    residency_indices: set[int] = set()
    slices: list[ActionSlice] = []
    dropped_request_count = 0
    for request_id in plan.execution.ordered_request_ids:
        request_reasons: list[str] = []
        if request_id not in visible:
            request_reasons.append("request_not_visible")
        admission = admission_by_request.get(request_id)
        admission_validation = validation.admissions.get(request_id)
        if admission is None or admission_validation is None:
            request_reasons.append("admission_missing")
        elif not admission_validation.valid:
            request_reasons.extend(admission_validation.reasons)
        elif admission.action not in {
            AdmissionAction.ADMIT,
            AdmissionAction.RESTORE_THEN_ADMIT,
        }:
            request_reasons.append("admission_not_runnable")

        dependency_indices = dependency_indices_by_request.get(request_id, ())
        dependent_bundle_ids: list[str] = []
        for dependency_index in dependency_indices:
            if dependency_index >= len(validation.dependencies):
                request_reasons.append("dependency_validation_missing")
                continue
            dependency_validation = validation.dependencies[dependency_index]
            if not dependency_validation.valid:
                request_reasons.extend(
                    f"dependency:{item}" for item in dependency_validation.reasons
                )
                continue
            residency_index = plan.dependencies[
                dependency_index
            ].residency_intent_index
            if residency_index >= len(plan.residency):
                request_reasons.append("residency_intent_missing")
                continue
            residency = plan.residency[residency_index]
            if not plan.dependencies[dependency_index].require_ack:
                request_reasons.append("restore_ack_dependency_missing")
            if residency.action != ResidencyAction.PREFETCH_GPU:
                request_reasons.append("restore_action_not_prefetch")
            residency_validation = validation.residency.get(residency.bundle_id)
            if residency_validation is None or not residency_validation.valid:
                request_reasons.append("residency_validation_failed")
            dependent_bundle_ids.append(residency.bundle_id)

        if admission is not None and admission.action == AdmissionAction.RESTORE_THEN_ADMIT:
            if not admission.required_bundle_ids or not dependency_indices:
                request_reasons.append("restore_dependency_missing")
            if set(dependent_bundle_ids) != set(admission.required_bundle_ids):
                request_reasons.append("restore_dependency_mismatch")
        elif admission is not None:
            if admission.required_bundle_ids or dependency_indices:
                request_reasons.append("unexpected_admit_dependency")

        committed = not request_reasons
        slices.append(
            ActionSlice(
                slice_id=f"request:{request_id}",
                kind="request",
                action_key=request_id,
                dependency_keys=tuple(
                    f"dependency:{index}" for index in dependency_indices
                ),
                committed=committed,
                reasons=tuple(request_reasons),
            )
        )
        if not committed:
            dropped_request_count += 1
            continue
        assert admission is not None
        ordered.append(request_id)
        if admission.action == AdmissionAction.RESTORE_THEN_ADMIT:
            required_bundle_ids = tuple(sorted(admission.required_bundle_ids))
            restore.append((request_id, required_bundle_ids))
            residency_indices.update(
                plan.dependencies[index].residency_intent_index
                for index in dependency_indices
            )
        else:
            immediate.append(request_id)

    for residency_index, residency in enumerate(plan.residency):
        if residency.action == ResidencyAction.KEEP:
            continue
        validation_entry = validation.residency.get(residency.bundle_id)
        reasons = (
            ("residency_validation_missing",)
            if validation_entry is None
            else validation_entry.reasons
        )
        dependency_required = residency_index in residency_indices
        commit = not reasons and (
            dependency_required
            or (
                bool(ordered)
                and residency.action
                in {ResidencyAction.COMMIT_CPU, ResidencyAction.DROP, ResidencyAction.RECOMPUTE}
            )
        )
        if not commit and not reasons:
            reasons = ("no_committed_consumer",)
        slices.append(
            ActionSlice(
                slice_id=f"residency:{residency_index}:{residency.bundle_id}",
                kind="residency",
                action_key=residency.bundle_id,
                dependency_keys=tuple(
                    f"request:{dependency.before_request_id}"
                    for dependency in plan.dependencies
                    if dependency.residency_intent_index == residency_index
                    and dependency.before_request_id is not None
                ),
                committed=commit,
                reasons=() if commit else tuple(reasons),
            )
        )
        if commit:
            residency_indices.add(residency_index)

    deferred = tuple(sorted(visible.difference(ordered)))
    view = OnlineJointPlanView(
        plan_id=plan.plan_id,
        ordered_request_ids=tuple(ordered),
        immediate_request_ids=tuple(immediate),
        restore_requirements=tuple(restore),
        deferred_request_ids=deferred,
        residency_intent_indices=tuple(sorted(residency_indices)),
    )
    source_action_count = len(slices)
    committed_action_count = sum(item.committed for item in slices)
    action_slices = tuple(slices)
    epoch = JointPlanEpoch(
        epoch_id=f"{plan.plan_id}:epoch:{epoch_sequence}",
        source_plan_id=plan.plan_id,
        planner_mode=JointPlannerMode(
            getattr(plan, "planner_mode", JointPlannerMode.OPTIMIZED)
        ),
        view=view,
        action_slices=action_slices,
        source_action_count=source_action_count,
        committed_action_count=committed_action_count,
        action_groups=_action_groups_from_slices(action_slices),
    )
    if not ordered and not residency_indices:
        reason = "no_action"
    elif dropped_request_count or committed_action_count < source_action_count:
        reason = "partially_applicable"
    else:
        reason = "applicable"
    return OnlineJointPlanDecision(view, reason, epoch)
