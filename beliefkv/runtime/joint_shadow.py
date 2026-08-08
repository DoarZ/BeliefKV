from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Protocol

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.control.data_consumers import ObservedDataConsumerIndex
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent
from beliefkv.policy.admission import AdmissionController
from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.joint_scheduler import JointPlan, ObservedJointPlanner
from beliefkv.policy.leases import CausalLeaseProjector
from beliefkv.policy.reference import (
    CapabilityReport,
    MetadataSource,
    MetadataValue,
    PolicyInput,
    RunnableInvocation,
)
from beliefkv.policy.reference.snapshot_builder import (
    PolicyInputSnapshotBuilder,
    SnapshotBuildStats,
)
from beliefkv.policy.resource_snapshot import RuntimeResourceObservation
from beliefkv.policy.service_curve import TransferServiceCurve
from beliefkv.policy.transfer_cost import PCIeCostModel
from beliefkv.policy.workflow_fairness import WorkflowFairScheduler
from beliefkv.runtime.page_index import (
    ContextPageReplica,
    PageIndexReplicaDelta,
    PageOwnershipIndex,
    PhysicalPageReplica,
    PhysicalPageStateReplica,
)
from beliefkv.runtime.protocol import PageHandle, TransferTelemetry


class JointPlanProducer(Protocol):
    def plan(self, policy_input: PolicyInput) -> JointPlan:
        ...


@dataclass(frozen=True)
class WorkflowFairnessReplica:
    workflow_id: str
    weight: float
    attained_service_ms: float
    virtual_runtime_ms: float
    dispatch_count: int


@dataclass(frozen=True)
class JointShadowStateStamp:
    graph_version: int
    consumer_version: int
    event_sequence: int
    page_revision: int
    topology_revision: int
    fairness_revision: int
    transfer_epoch: int
    runnable_signature: tuple[tuple[object, ...], ...]
    hbm_used_bytes: int
    host_free_bytes: int
    obligation_revision: int = 0
    lease_revision: int = 0
    grace_revision: int = 0
    parser_frontier_revision: int = 0


@dataclass(frozen=True)
class JointShadowDelta:
    """Immutable safe-point publication consumed only by the shadow worker."""

    event_from_sequence: int
    event_to_sequence: int
    runtime_events: tuple[RuntimeEvent, ...]
    page_delta: PageIndexReplicaDelta
    observation: RuntimeResourceObservation
    runnable_frontier: tuple[RunnableInvocation, ...]
    fairness_accounts: tuple[WorkflowFairnessReplica, ...]
    external_workflow_charges: tuple[tuple[str, float], ...]
    control_state: Mapping[str, object]
    transfer_telemetry: tuple[TransferTelemetry, ...]
    capabilities: CapabilityReport
    stamp: JointShadowStateStamp
    trigger: str
    captured_monotonic_ms: float
    frontier_predictions: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.event_to_sequence < self.event_from_sequence:
            raise ValueError("shadow event sequence cannot move backwards")
        if not self.trigger:
            raise ValueError("shadow delta trigger must be non-empty")
        if self.captured_monotonic_ms < 0:
            raise ValueError("shadow capture time must be non-negative")
        object.__setattr__(
            self,
            "frontier_predictions",
            MappingProxyType(
                {
                    str(invocation_id): dict(prediction)
                    for invocation_id, prediction in sorted(
                        self.frontier_predictions.items()
                    )
                }
            ),
        )


def coalesce_joint_shadow_deltas(
    deltas: tuple[JointShadowDelta, ...],
) -> JointShadowDelta:
    """Collapse contiguous publications to their final physical replica state.

    RCCG events and transfer telemetry remain lossless. Page/context replacement
    records use last-write-wins semantics, so the worker does not replay every
    intermediate allocator quantum before building one latest-wins snapshot.
    """

    if not deltas:
        raise ValueError("at least one shadow delta is required")
    if len(deltas) == 1:
        return deltas[0]

    first = deltas[0]
    event_cursor = first.event_from_sequence
    page_cursor = first.page_delta.from_revision
    events: list[RuntimeEvent] = []
    telemetry: list[TransferTelemetry] = []
    pages: dict[PageHandle, PhysicalPageReplica] = {}
    page_states: dict[PageHandle, PhysicalPageStateReplica] = {}
    contexts: dict[str, ContextPageReplica] = {}
    changed_handles: set[PageHandle] = set()
    changed_context_ids: set[str] = set()
    components: set[str] = set()
    full_rebuild = False

    for delta in deltas:
        if delta.event_from_sequence != event_cursor:
            raise RuntimeError("cannot coalesce shadow deltas with an RCCG gap")
        if delta.page_delta.from_revision != page_cursor:
            raise RuntimeError("cannot coalesce shadow deltas with a page gap")
        event_cursor = delta.event_to_sequence
        page_cursor = delta.page_delta.to_revision
        events.extend(delta.runtime_events)
        telemetry.extend(delta.transfer_telemetry)

        page_delta = delta.page_delta
        if page_delta.full_rebuild_required:
            full_rebuild = True
            pages = {item.handle: item for item in page_delta.pages}
            page_states = {
                item.handle: item for item in page_delta.page_states
            }
            contexts = {item.context_id: item for item in page_delta.contexts}
            changed_handles = set(page_delta.changed_handles)
            changed_context_ids = set(page_delta.changed_context_ids)
        else:
            for item in page_delta.pages:
                pages[item.handle] = item
                page_states.pop(item.handle, None)
            for item in page_delta.page_states:
                page_states[item.handle] = item
            for context_id in page_delta.changed_context_ids:
                contexts.pop(context_id, None)
            for item in page_delta.contexts:
                contexts[item.context_id] = item
            changed_handles.update(page_delta.changed_handles)
            changed_context_ids.update(page_delta.changed_context_ids)
        components.update(page_delta.components)

    last = deltas[-1]
    page_delta = PageIndexReplicaDelta(
        from_revision=first.page_delta.from_revision,
        to_revision=last.page_delta.to_revision,
        topology_revision=last.page_delta.topology_revision,
        pages=tuple(pages[handle] for handle in sorted(pages)),
        page_states=tuple(
            page_states[handle] for handle in sorted(page_states)
        ),
        contexts=tuple(contexts[key] for key in sorted(contexts)),
        changed_handles=frozenset(changed_handles),
        changed_context_ids=frozenset(changed_context_ids),
        components=frozenset(components),
        full_rebuild_required=full_rebuild,
    )
    return JointShadowDelta(
        event_from_sequence=first.event_from_sequence,
        event_to_sequence=last.event_to_sequence,
        runtime_events=tuple(events),
        page_delta=page_delta,
        observation=last.observation,
        runnable_frontier=last.runnable_frontier,
        fairness_accounts=last.fairness_accounts,
        external_workflow_charges=last.external_workflow_charges,
        control_state=last.control_state,
        transfer_telemetry=tuple(telemetry),
        capabilities=last.capabilities,
        stamp=last.stamp,
        trigger=last.trigger,
        captured_monotonic_ms=last.captured_monotonic_ms,
        frontier_predictions=last.frontier_predictions,
    )


@dataclass(frozen=True)
class JointShadowSubmission:
    sequence: int
    snapshot_id: str
    submitted_monotonic_ms: float
    enqueue_ms: float
    replaced_sequence: int | None


@dataclass(frozen=True)
class JointShadowResult:
    sequence: int
    snapshot_id: str
    submitted_monotonic_ms: float
    started_monotonic_ms: float
    completed_monotonic_ms: float
    plan: JointPlan | None
    error: str | None
    policy_input: PolicyInput | None = None
    snapshot_build_ms: float = 0.0
    snapshot_delta_apply_ms: float = 0.0
    snapshot_materialize_ms: float = 0.0
    state_stamp: JointShadowStateStamp | None = None
    trigger: str = "legacy_policy_input"
    trigger_interval_ms: float | None = None
    planning_budget_ms: float | None = None

    @property
    def queue_wait_ms(self) -> float:
        return max(0.0, self.started_monotonic_ms - self.submitted_monotonic_ms)

    @property
    def compute_ms(self) -> float:
        return max(0.0, self.completed_monotonic_ms - self.started_monotonic_ms)


@dataclass(frozen=True)
class JointShadowWorkerStats:
    submitted_count: int
    started_count: int
    completed_count: int
    failed_count: int
    dropped_pending_count: int
    coalesced_pending_count: int
    superseded_result_count: int
    pending_count: int
    busy: bool
    latest_published_sequence: int

    def to_dict(self) -> dict[str, int | bool]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class _WorkItem:
    sequence: int
    submitted_monotonic_ms: float
    policy_input: PolicyInput | None = None
    deltas: tuple[JointShadowDelta, ...] = ()


class IncrementalPolicyInputAssembler:
    """Worker-owned RCCG/page mirrors and PolicyInput builder."""

    def __init__(self, config: BeliefKVConfig) -> None:
        self.config = config
        self.graph = RuntimeCausalContextGraph()
        self.data_consumers = ObservedDataConsumerIndex(self.graph)
        self.page_index = PageOwnershipIndex()
        fairness = WorkflowFairScheduler()
        frontier = CausalFrontierScheduler(self.graph)
        self.admission = AdmissionController(
            self.page_index,
            fairness,
            frontier,
            reserve_hbm_bytes=0,
        )
        self.leases = CausalLeaseProjector(self.graph)
        self.service_curve = TransferServiceCurve(
            PCIeCostModel(
                bandwidth_gbps=config.pcie_bandwidth_gbps,
                overhead_ms=config.transfer_overhead_ms,
            ),
            window=config.service_curve_window,
            min_samples=config.service_curve_min_samples,
        )
        self.builder = PolicyInputSnapshotBuilder(
            self.graph,
            self.data_consumers,
            self.page_index,
            self.admission,
            self.leases,
            self.service_curve,
        )
        self._event_sequence = 0
        self._latest: JointShadowDelta | None = None
        self._telemetry: deque[TransferTelemetry] = deque(
            maxlen=max(256, config.service_curve_window)
        )

    @property
    def last_stats(self) -> SnapshotBuildStats | None:
        return self.builder.last_stats

    def apply(self, delta: JointShadowDelta) -> None:
        if delta.event_from_sequence != self._event_sequence:
            raise RuntimeError(
                "shadow RCCG event gap: "
                f"{delta.event_from_sequence} != {self._event_sequence}"
            )
        if delta.page_delta.full_rebuild_required and self.page_index.revision != 0:
            raise RuntimeError("shadow page journal gap requires fail-closed restart")
        if delta.runtime_events:
            self.graph.apply_batch(delta.runtime_events, atomic=True)
            self.data_consumers.apply_batch(delta.runtime_events, atomic=True)
        self._event_sequence = delta.event_to_sequence
        self.page_index.apply_replica_delta(
            delta.page_delta,
            full_validation=False,
        )
        for telemetry in delta.transfer_telemetry:
            self.service_curve.observe(telemetry)
            self._telemetry.append(telemetry)
        if self.graph.graph_version != delta.stamp.graph_version:
            raise RuntimeError(
                "shadow graph version diverged from safe-point publication"
            )
        if self.data_consumers.version != delta.stamp.consumer_version:
            raise RuntimeError(
                "shadow consumer version diverged from safe-point publication"
            )
        self._latest = delta

    def build(self) -> PolicyInput:
        delta = self._latest
        if delta is None:
            raise RuntimeError("shadow assembler has no safe-point state")
        charges = self.page_index.workflow_gpu_charges()
        for workflow_id, amount in delta.external_workflow_charges:
            charges[workflow_id] = charges.get(workflow_id, 0.0) + amount
        fairness_state = {
            "accounts": {
                item.workflow_id: {
                    "weight": item.weight,
                    "attained_service_ms": item.attained_service_ms,
                    "virtual_runtime_ms": item.virtual_runtime_ms,
                    "dispatch_count": item.dispatch_count,
                }
                for item in delta.fairness_accounts
            },
            "revision": delta.stamp.fairness_revision,
        }
        policy_input = self.builder.build(
            delta.observation,
            additional_runnable=delta.runnable_frontier,
            workflow_memory_charges=charges,
            workflow_fairness_state=fairness_state,
            control_state=delta.control_state,
            transfer_telemetry=tuple(self._telemetry),
            capabilities=delta.capabilities,
        )
        if delta.frontier_predictions:
            metadata = dict(policy_input.optional_metadata)
            metadata["frontier_predictions"] = MetadataValue(
                source=MetadataSource.PREDICTED,
                value=dict(delta.frontier_predictions),
                producer="frontier_belief_mvp",
            )
            policy_input = replace(
                policy_input,
                optional_metadata=metadata,
            )
        return policy_input


class LatestWinsJointPlanWorker:
    """Capacity-one asynchronous planner with failure isolation.

    A pending snapshot is replaced by a newer snapshot. The item currently
    being evaluated is never cancelled, and the worker never reaches into live
    scheduler state.
    """

    def __init__(
        self,
        planner: JointPlanProducer | None = None,
        *,
        assembler: IncrementalPolicyInputAssembler | None = None,
        thread_name: str = "beliefkv-joint-shadow",
    ) -> None:
        self.planner = planner or ObservedJointPlanner()
        self.assembler = assembler
        self._condition = threading.Condition()
        self._pending: _WorkItem | None = None
        self._latest: JointShadowResult | None = None
        self._closed = False
        self._busy = False
        self._next_sequence = 0
        self._submitted_count = 0
        self._started_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._dropped_pending_count = 0
        self._coalesced_pending_count = 0
        self._superseded_result_count = 0
        self._last_trigger_capture_ms: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    def submit(self, policy_input: PolicyInput) -> JointShadowSubmission:
        enqueue_started_ns = time.perf_counter_ns()
        submitted_ms = _monotonic_ms()
        with self._condition:
            if self._closed:
                raise RuntimeError("joint shadow worker is closed")
            self._next_sequence += 1
            sequence = self._next_sequence
            replaced = self._pending.sequence if self._pending is not None else None
            if replaced is not None:
                self._dropped_pending_count += 1
            self._pending = _WorkItem(
                sequence=sequence,
                submitted_monotonic_ms=submitted_ms,
                policy_input=policy_input,
            )
            self._submitted_count += 1
            self._condition.notify()
        return JointShadowSubmission(
            sequence=sequence,
            snapshot_id=policy_input.snapshot_id,
            submitted_monotonic_ms=submitted_ms,
            enqueue_ms=(time.perf_counter_ns() - enqueue_started_ns) / 1_000_000.0,
            replaced_sequence=replaced,
        )

    @property
    def supports_incremental_delta(self) -> bool:
        return self.assembler is not None

    def submit_delta(self, delta: JointShadowDelta) -> JointShadowSubmission:
        if self.assembler is None:
            raise RuntimeError("joint shadow worker has no incremental assembler")
        enqueue_started_ns = time.perf_counter_ns()
        submitted_ms = _monotonic_ms()
        with self._condition:
            if self._closed:
                raise RuntimeError("joint shadow worker is closed")
            self._next_sequence += 1
            sequence = self._next_sequence
            replaced = self._pending.sequence if self._pending is not None else None
            pending_deltas: tuple[JointShadowDelta, ...] = ()
            if self._pending is not None:
                if self._pending.policy_input is not None:
                    raise RuntimeError(
                        "cannot mix legacy snapshots and incremental deltas"
                    )
                pending_deltas = self._pending.deltas
                self._coalesced_pending_count += 1
            self._pending = _WorkItem(
                sequence=sequence,
                submitted_monotonic_ms=submitted_ms,
                deltas=pending_deltas + (delta,),
            )
            self._submitted_count += 1
            self._condition.notify()
        return JointShadowSubmission(
            sequence=sequence,
            snapshot_id=f"pending-shadow-delta-{sequence:08d}",
            submitted_monotonic_ms=submitted_ms,
            enqueue_ms=(time.perf_counter_ns() - enqueue_started_ns) / 1_000_000.0,
            replaced_sequence=replaced,
        )

    def latest(self, *, after_sequence: int = 0) -> JointShadowResult | None:
        with self._condition:
            if self._latest is None or self._latest.sequence <= after_sequence:
                return None
            return self._latest

    def stats(self) -> JointShadowWorkerStats:
        with self._condition:
            return JointShadowWorkerStats(
                submitted_count=self._submitted_count,
                started_count=self._started_count,
                completed_count=self._completed_count,
                failed_count=self._failed_count,
                dropped_pending_count=self._dropped_pending_count,
                coalesced_pending_count=self._coalesced_pending_count,
                superseded_result_count=self._superseded_result_count,
                pending_count=int(self._pending is not None),
                busy=self._busy,
                latest_published_sequence=(
                    self._latest.sequence if self._latest is not None else 0
                ),
            )

    def close(self, *, timeout_s: float = 5.0) -> bool:
        if timeout_s < 0:
            raise ValueError("worker close timeout must be non-negative")
        with self._condition:
            if not self._closed:
                self._closed = True
                if self._pending is not None:
                    self._pending = None
                    self._dropped_pending_count += 1
                self._condition.notify_all()
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                item = self._pending
                self._pending = None
                self._busy = True
                self._started_count += 1
            assert item is not None
            started_ms = _monotonic_ms()
            plan = None
            error = None
            policy_input = item.policy_input
            snapshot_build_ms = 0.0
            snapshot_delta_apply_ms = 0.0
            snapshot_materialize_ms = 0.0
            state_stamp = None
            trigger = "legacy_policy_input"
            trigger_interval_ms = None
            planning_budget_ms = None
            try:
                if item.deltas:
                    assert self.assembler is not None
                    apply_started_ns = time.perf_counter_ns()
                    delta = coalesce_joint_shadow_deltas(item.deltas)
                    self.assembler.apply(delta)
                    snapshot_delta_apply_ms = (
                        time.perf_counter_ns() - apply_started_ns
                    ) / 1_000_000.0
                    materialize_started_ns = time.perf_counter_ns()
                    policy_input = self.assembler.build()
                    snapshot_materialize_ms = (
                        time.perf_counter_ns() - materialize_started_ns
                    ) / 1_000_000.0
                    snapshot_build_ms = (
                        snapshot_delta_apply_ms + snapshot_materialize_ms
                    )
                    state_stamp = delta.stamp
                    trigger = delta.trigger
                    if self._last_trigger_capture_ms is not None:
                        trigger_interval_ms = max(
                            0.0,
                            delta.captured_monotonic_ms
                            - self._last_trigger_capture_ms,
                        )
                    self._last_trigger_capture_ms = delta.captured_monotonic_ms
                if policy_input is None:
                    raise RuntimeError("joint shadow work item has no policy input")
                budget_for_trigger = getattr(
                    self.planner, "trigger_budget_ms", None
                )
                if callable(budget_for_trigger):
                    planning_budget_ms = budget_for_trigger(trigger_interval_ms)
                    plan = self.planner.plan(
                        policy_input,
                        planning_budget_ms=planning_budget_ms,
                        cancel_check=lambda: self._has_newer_pending(item.sequence),
                    )
                else:
                    plan = self.planner.plan(policy_input)
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            completed_ms = _monotonic_ms()
            result = JointShadowResult(
                sequence=item.sequence,
                snapshot_id=(
                    policy_input.snapshot_id
                    if policy_input is not None
                    else f"failed-shadow-work-{item.sequence:08d}"
                ),
                submitted_monotonic_ms=item.submitted_monotonic_ms,
                started_monotonic_ms=started_ms,
                completed_monotonic_ms=completed_ms,
                plan=plan,
                error=error,
                policy_input=policy_input,
                snapshot_build_ms=snapshot_build_ms,
                snapshot_delta_apply_ms=snapshot_delta_apply_ms,
                snapshot_materialize_ms=snapshot_materialize_ms,
                state_stamp=state_stamp,
                trigger=trigger,
                trigger_interval_ms=trigger_interval_ms,
                planning_budget_ms=planning_budget_ms,
            )
            with self._condition:
                self._busy = False
                self._completed_count += 1
                if error is not None:
                    self._failed_count += 1
                superseded = (
                    self._pending is not None
                    and self._pending.sequence > result.sequence
                )
                if superseded:
                    self._superseded_result_count += 1
                elif self._latest is None or result.sequence > self._latest.sequence:
                    self._latest = result
                self._condition.notify_all()

    def _has_newer_pending(self, sequence: int) -> bool:
        with self._condition:
            return self._pending is not None and self._pending.sequence > sequence

    def __enter__(self) -> "LatestWinsJointPlanWorker":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _monotonic_ms() -> float:
    return time.monotonic_ns() / 1_000_000.0
