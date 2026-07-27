from __future__ import annotations

import threading

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.joint_scheduler import (
    JointPlannerConfig,
    ObservedJointPlanner,
)
from beliefkv.policy.resource_snapshot import RuntimeResourceObservation
from beliefkv.runtime.joint_shadow import (
    IncrementalPolicyInputAssembler,
    JointShadowDelta,
    JointShadowStateStamp,
    LatestWinsJointPlanWorker,
    WorkflowFairnessReplica,
    coalesce_joint_shadow_deltas,
)
from beliefkv.runtime.protocol import PageHandle
from tests.test_joint_scheduler import _invocation, _with_runtime_state
from tests.test_whatif_packer import _input


def _policy_input():
    policy_input = _input(capacity=1_500, reserved=0, include_cpu_target=False)
    request = policy_input.runnable_frontier[0]
    return _with_runtime_state(
        policy_input,
        (request,),
        {
            request.invocation_id: _invocation(
                request.workflow_id,
                request.context_id,
            )
        },
    )


def _event(sequence: int, kind: RuntimeEventKind, **kwargs) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"shadow-{sequence}",
        ts_ms=float(sequence),
        kind=kind,
        workflow_id="wf",
        **kwargs,
    )


def _delta(
    controller: BeliefKVController,
    *,
    event_sequence: int,
    page_revision: int,
    ts_ms: float,
) -> JointShadowDelta:
    events = controller.runtime_events_since(event_sequence)
    pages = controller.page_index.replica_delta_since(page_revision)
    account = controller.fairness.accounts["wf"]
    control_state = controller.policy_control_state(ts_ms)
    observation = RuntimeResourceObservation(
        ts_ms=ts_ms,
        hbm_capacity_bytes=1_000,
        hbm_used_bytes=controller.page_index.gpu_bytes,
        host_capacity_bytes=1_000,
        host_used_bytes=controller.page_index.cpu_bytes,
        host_free_bytes=1_000 - controller.page_index.cpu_bytes,
    )
    return JointShadowDelta(
        event_from_sequence=events.from_sequence,
        event_to_sequence=events.to_sequence,
        runtime_events=events.events,
        page_delta=pages,
        observation=observation,
        runnable_frontier=(),
        fairness_accounts=(
            WorkflowFairnessReplica(
                workflow_id="wf",
                weight=account.weight,
                attained_service_ms=account.attained_service_ms,
                virtual_runtime_ms=account.virtual_runtime,
                dispatch_count=account.dispatch_count,
            ),
        ),
        external_workflow_charges=(),
        control_state=control_state,
        transfer_telemetry=(),
        capabilities=_policy_input().capabilities,
        stamp=JointShadowStateStamp(
            graph_version=controller.graph.graph_version,
            consumer_version=controller.data_consumers.version,
            event_sequence=events.to_sequence,
            page_revision=pages.to_revision,
            topology_revision=pages.topology_revision,
            fairness_revision=controller.fairness.revision,
            transfer_epoch=int(control_state["transfer_epoch"]),
            runnable_signature=(),
            hbm_used_bytes=observation.hbm_used_bytes,
            host_free_bytes=observation.host_free_bytes,
        ),
        trigger="test",
        captured_monotonic_ms=0,
    )


class _BlockingPlanner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.sequences: list[str] = []
        self.delegate = ObservedJointPlanner(
            JointPlannerConfig(max_planning_budget_ms=100.0)
        )

    def plan(self, policy_input):
        self.sequences.append(policy_input.snapshot_id)
        if len(self.sequences) == 1:
            self.started.set()
            assert self.release.wait(timeout=2)
        return self.delegate.plan(policy_input)


class _FailingPlanner:
    def plan(self, policy_input):
        del policy_input
        raise ValueError("expected failure")


def test_coalesced_delta_keeps_events_and_latest_page_state() -> None:
    config = BeliefKVConfig(
        hbm_capacity_bytes=1_000,
        host_capacity_bytes=1_000,
        reserve_hbm_bytes=0,
        predictor_enabled=False,
        shadow_enabled=False,
    )
    controller = BeliefKVController(config)
    controller.process_runtime_events(
        (
            _event(1, RuntimeEventKind.WORKFLOW_START),
            _event(
                2,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="root",
                context_id="ctx",
                context_epoch=0,
            ),
        )
    )
    handle = PageHandle(1, 0)
    controller.page_index.register_page(handle, size_bytes=100)
    controller.page_index.bind_pages("ctx", 0, (handle,))
    initial = _delta(controller, event_sequence=0, page_revision=0, ts_ms=2)

    controller.page_index.set_engine_lock(handle, 1)
    controller.process_runtime_event(
        _event(
            3,
            RuntimeEventKind.CONTEXT_ADVANCE,
            invocation_id="root",
            context_id="ctx",
            context_epoch=0,
        )
    )
    second = _delta(
        controller,
        event_sequence=initial.event_to_sequence,
        page_revision=initial.page_delta.to_revision,
        ts_ms=3,
    )
    controller.page_index.set_engine_lock(handle, 2)
    controller.process_runtime_event(
        _event(
            4,
            RuntimeEventKind.CONTEXT_ADVANCE,
            invocation_id="root",
            context_id="ctx",
            context_epoch=0,
        )
    )
    third = _delta(
        controller,
        event_sequence=second.event_to_sequence,
        page_revision=second.page_delta.to_revision,
        ts_ms=4,
    )

    merged = coalesce_joint_shadow_deltas((second, third))
    assert tuple(event.event_id for event in merged.runtime_events) == (
        "shadow-3",
        "shadow-4",
    )
    assert merged.page_delta.pages == ()
    assert len(merged.page_delta.page_states) == 1
    assert merged.page_delta.page_states[0].engine_lock_ref == 2
    assert merged.page_delta.contexts == ()

    assembler = IncrementalPolicyInputAssembler(config)
    assembler.apply(initial)
    assembler.apply(merged)
    assert assembler.page_index.require_page(handle).engine_lock_ref == 2
    assert assembler.graph.graph_version == controller.graph.graph_version


def test_worker_replaces_only_the_pending_snapshot() -> None:
    planner = _BlockingPlanner()
    worker = LatestWinsJointPlanWorker(planner)
    policy_input = _policy_input()
    first = worker.submit(policy_input)
    assert planner.started.wait(timeout=2)
    second = worker.submit(policy_input)
    third = worker.submit(policy_input)
    planner.release.set()

    result = None
    for _ in range(100):
        result = worker.latest(after_sequence=first.sequence)
        if result is not None and result.sequence == third.sequence:
            break
        threading.Event().wait(0.01)

    assert result is not None
    assert result.sequence == third.sequence
    assert result.plan is not None
    assert result.error is None
    stats = worker.stats()
    assert second.sequence == third.sequence - 1
    assert stats.submitted_count == 3
    assert stats.started_count == 2
    assert stats.completed_count == 2
    assert stats.dropped_pending_count == 1
    assert worker.close()


def test_worker_contains_planner_failure_and_remains_closeable() -> None:
    worker = LatestWinsJointPlanWorker(_FailingPlanner())
    submission = worker.submit(_policy_input())

    result = None
    for _ in range(100):
        result = worker.latest(after_sequence=submission.sequence - 1)
        if result is not None:
            break
        threading.Event().wait(0.01)

    assert result is not None
    assert result.plan is None
    assert result.error == "ValueError: expected failure"
    assert worker.stats().failed_count == 1
    assert worker.close()


def test_incremental_worker_merges_pending_deltas_without_losing_events() -> None:
    config = BeliefKVConfig(
        hbm_capacity_bytes=1_000,
        host_capacity_bytes=1_000,
        reserve_hbm_bytes=0,
        predictor_enabled=False,
        shadow_enabled=False,
    )
    controller = BeliefKVController(config)
    controller.process_runtime_events(
        (
            _event(1, RuntimeEventKind.WORKFLOW_START),
            _event(
                2,
                RuntimeEventKind.INVOCATION_CREATE,
                invocation_id="root",
                context_id="ctx",
                context_epoch=0,
            ),
        )
    )
    handle = PageHandle(1, 0)
    controller.page_index.register_page(handle, size_bytes=100)
    controller.page_index.bind_pages("ctx", 0, (handle,))
    first_delta = _delta(
        controller,
        event_sequence=0,
        page_revision=0,
        ts_ms=2,
    )

    planner = _BlockingPlanner()
    worker = LatestWinsJointPlanWorker(
        planner,
        assembler=IncrementalPolicyInputAssembler(config),
    )
    first = worker.submit_delta(first_delta)
    assert planner.started.wait(timeout=2)

    controller.process_runtime_event(
        _event(
            3,
            RuntimeEventKind.CONTEXT_ADVANCE,
            invocation_id="root",
            context_id="ctx",
            context_epoch=0,
        )
    )
    second_delta = _delta(
        controller,
        event_sequence=first_delta.event_to_sequence,
        page_revision=first_delta.page_delta.to_revision,
        ts_ms=3,
    )
    second = worker.submit_delta(second_delta)

    controller.process_runtime_event(
        _event(
            4,
            RuntimeEventKind.CONTEXT_ADVANCE,
            invocation_id="root",
            context_id="ctx",
            context_epoch=0,
        )
    )
    third_delta = _delta(
        controller,
        event_sequence=second_delta.event_to_sequence,
        page_revision=second_delta.page_delta.to_revision,
        ts_ms=4,
    )
    third = worker.submit_delta(third_delta)
    planner.release.set()

    result = None
    for _ in range(100):
        result = worker.latest(after_sequence=first.sequence)
        if result is not None and result.sequence == third.sequence:
            break
        threading.Event().wait(0.01)

    assert result is not None
    assert result.error is None
    assert result.policy_input is not None
    assert result.state_stamp is not None
    assert result.snapshot_build_ms == (
        result.snapshot_delta_apply_ms + result.snapshot_materialize_ms
    )
    assert result.snapshot_delta_apply_ms >= 0
    assert result.snapshot_materialize_ms >= 0
    assert result.state_stamp.event_sequence == controller.runtime_event_sequence
    assert (
        result.policy_input.runtime_graph.graph_version
        == controller.graph.graph_version
    )
    assert second.sequence == third.sequence - 1
    assert worker.stats().dropped_pending_count == 0
    assert worker.stats().coalesced_pending_count == 1
    assert worker.close()
