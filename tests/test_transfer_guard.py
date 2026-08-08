import json
from dataclasses import replace
from pathlib import Path

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandStatus,
    ControlCommand,
    PageHandle,
    PhysicalResidency,
    TransferBlocker,
    TransferBlockerCode,
)


def _controller(
    *, unknown_base_ms: float = 10.0, unknown_circuit_failures: int = 8
) -> BeliefKVController:
    controller = BeliefKVController(
        BeliefKVConfig(
            hbm_capacity_bytes=1000,
            host_capacity_bytes=10_000,
            reserve_hbm_bytes=100,
            urgent_chunk_bytes=1000,
            shadow_chunk_bytes=500,
            predictor_enabled=False,
            shadow_enabled=False,
            transfer_retry_unknown_base_ms=unknown_base_ms,
            transfer_retry_unknown_max_ms=1000,
            transfer_retry_unknown_circuit_breaker_failures=(
                unknown_circuit_failures
            ),
        )
    )
    controller.process_runtime_events(
        (
            RuntimeEvent("start", 0, RuntimeEventKind.WORKFLOW_START, "wf"),
            RuntimeEvent(
                "create",
                1,
                RuntimeEventKind.INVOCATION_CREATE,
                "wf",
                invocation_id="inv",
                context_id="ctx",
                context_epoch=0,
            ),
        )
    )
    handle = PageHandle(1, 0)
    controller.page_index.register_page(
        handle,
        size_bytes=400,
        residency=PhysicalResidency.CPU_ONLY,
    )
    controller.page_index.bind_pages("ctx", 0, (handle,))
    return controller


def _reject_prefetch(
    controller: BeliefKVController,
    *,
    now_ms: float,
    blocker_codes: tuple[TransferBlockerCode, ...],
) -> str:
    tick = controller.tick(now_ms)
    assert tick.transfer is not None
    assert tick.transfer.command.kind == CommandKind.PREFETCH_CONTEXT
    command_id = tick.transfer.command.command_id
    _ack_rejection(
        controller,
        command_id=command_id,
        now_ms=now_ms + 0.1,
        blocker_codes=blocker_codes,
    )
    return command_id


def _ack_rejection(
    controller: BeliefKVController,
    *,
    command_id: str,
    now_ms: float,
    blocker_codes: tuple[TransferBlockerCode, ...],
) -> None:
    controller.acknowledge_command(
        CommandAck(
            command_id=command_id,
            status=CommandStatus.REJECTED,
            completed_ts_ms=now_ms,
            actual_bytes=0,
            reason="fixture backend rejection",
            blockers=tuple(
                TransferBlocker(code, PageHandle(1, 0), 400)
                for code in blocker_codes
            ),
        )
    )


def test_frozen_p1_retry_storm_submits_one_command_per_physical_snapshot() -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "h2d_retry_storm_p1_short.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    controller = _controller()
    _reject_prefetch(
        controller,
        now_ms=10,
        blocker_codes=tuple(
            TransferBlockerCode(item) for item in fixture["blocker_codes"]
        ),
    )

    guard_events = []
    for offset in range(1, fixture["zero_byte_h2d_reject_count"]):
        tick = controller.tick(10 + offset)
        assert tick.transfer is None
        guard_events.extend(tick.transfer_guard_events)

    assert len(controller.command_history) == 1
    assert len(controller.ack_history) == 1
    assert controller.transfer_guard.summary()["suppressed_retry_count"] == 267
    assert sum(
        event.kind == "transfer_retry_suppressed" for event in guard_events
    ) == 1


def test_external_command_fails_fast_when_retry_guard_blocks_snapshot() -> None:
    controller = _controller()
    first = controller.tick(10)
    assert first.transfer is not None
    command = first.transfer.command
    _ack_rejection(
        controller,
        command_id=command.command_id,
        now_ms=10.1,
        blocker_codes=(TransferBlockerCode.ENGINE_BUSY,),
    )

    accepted = controller.enqueue_control_command(
        replace(command, command_id="restore-retry")
    )

    assert not accepted
    assert len(controller.command_queue) == 0
    assert controller.transfer_guard.summary()["suppressed_retry_count"] == 1


def test_accepted_command_gets_local_ack_if_guard_blocks_before_dispatch() -> None:
    controller = _controller()
    command = ControlCommand(
        command_id="accepted-restore",
        kind=CommandKind.PREFETCH_CONTEXT,
        created_ts_ms=10,
        context_id="ctx",
        context_epoch=0,
        target_bytes=400,
    )
    assert controller.enqueue_control_command(command)

    competing = replace(command, command_id="competing-restore")
    controller.transfer_guard.begin_attempt(competing, now_ms=10.1)
    controller.transfer_guard.record_failure(
        competing.command_id,
        blockers=(
            TransferBlocker(
                TransferBlockerCode.ENGINE_BUSY,
                PageHandle(1, 0),
                400,
            ),
        ),
        required_bytes=400,
        now_ms=10.2,
    )

    tick = controller.tick(11)

    assert tick.transfer is None
    assert len(tick.local_acks) == 1
    ack = tick.local_acks[0]
    assert ack.command_id == command.command_id
    assert ack.status == CommandStatus.REJECTED
    assert ack.reason == "transfer_retry_guard_suppressed"
    assert tuple(item.code for item in ack.blockers) == (
        TransferBlockerCode.ENGINE_BUSY,
    )
    assert len(controller.command_queue) == 0


def test_prefetch_materializes_cpu_ancestor_as_one_physical_bundle() -> None:
    controller = _controller()
    child = controller.page_index.pages[PageHandle(1, 0)]
    parent = PageHandle(2, 0)
    controller.page_index.register_page(
        parent,
        size_bytes=100,
        residency=PhysicalResidency.CPU_ONLY,
    )
    controller.page_index.set_parent(child.handle, parent)

    tick = controller.tick(10)

    assert tick.transfer is not None
    assert tick.transfer.command.physical_bundle is not None
    assert {item.handle for item in tick.transfer.page_actions} == {
        child.handle,
        parent,
    }
    assert all(
        item.action.value == "start_h2d" for item in tick.transfer.page_actions
    )
    assert tick.transfer.resolved_bytes == 500


def test_same_bundle_fingerprint_change_releases_closure_failure() -> None:
    controller = _controller()
    first = controller.tick(10)
    assert first.transfer is not None
    command = first.transfer.command
    _ack_rejection(
        controller,
        command_id=command.command_id,
        now_ms=10.1,
        blocker_codes=(TransferBlockerCode.ANCESTOR_CLOSURE,),
    )
    assert controller.tick(11).transfer is None

    assert command.physical_bundle is not None
    changed = replace(
        command,
        command_id="same-bundle-new-fingerprint",
        physical_bundle=replace(
            command.physical_bundle,
            generation_fingerprint="authoritative-state-changed",
        ),
    )
    assert controller.transfer_guard.command_is_eligible(changed, now_ms=12)
    events = controller.transfer_guard.drain_events()
    assert any(
        event.kind == "transfer_retry_released"
        and event.fields["release_reason"] == "physical_fingerprint_changed"
        for event in events
    )


def test_device_capacity_retry_requires_more_free_space_than_failed_snapshot() -> None:
    controller = _controller()
    controller.report_hbm_usage(500)
    _reject_prefetch(
        controller,
        now_ms=10,
        blocker_codes=(TransferBlockerCode.DEVICE_CAPACITY,),
    )

    assert controller.tick(11).transfer is None
    controller.report_hbm_usage(400)
    released = controller.tick(12)

    assert released.transfer is not None
    assert any(
        event.kind == "transfer_retry_released"
        and event.fields["release_reason"] == "resource_predicate_satisfied"
        for event in released.transfer_guard_events
    )


def test_same_bundle_fingerprint_change_preserves_unresolved_capacity_debt() -> None:
    controller = _controller()
    controller.report_hbm_usage(500)
    first = controller.tick(10)
    assert first.transfer is not None
    command = first.transfer.command
    _ack_rejection(
        controller,
        command_id=command.command_id,
        now_ms=10.1,
        blocker_codes=(
            TransferBlockerCode.ANCESTOR_CLOSURE,
            TransferBlockerCode.DEVICE_CAPACITY,
        ),
    )
    controller.transfer_guard.update_resources(
        device_available_bytes=400,
        host_available_bytes=10_000,
        now_ms=10.2,
    )
    assert command.physical_bundle is not None
    changed = replace(
        command,
        command_id="same-bundle-capacity-change",
        physical_bundle=replace(
            command.physical_bundle,
            generation_fingerprint="authoritative-state-changed",
        ),
    )
    assert not controller.transfer_guard.command_is_eligible(changed, now_ms=11)
    events = controller.transfer_guard.drain_events()
    assert any(
        event.kind == "transfer_retry_rekeyed"
        and event.fields["blocker_codes"] == ["device_capacity"]
        for event in events
    )

    controller.transfer_guard.update_resources(
        device_available_bytes=500,
        host_available_bytes=10_000,
        now_ms=12,
    )
    assert controller.transfer_guard.command_is_eligible(changed, now_ms=12)


def test_blocked_context_does_not_head_of_line_block_another_prefetch() -> None:
    controller = _controller()
    _reject_prefetch(
        controller,
        now_ms=10,
        blocker_codes=(TransferBlockerCode.ANCESTOR_CLOSURE,),
    )
    controller.process_runtime_event(
        RuntimeEvent(
            "create-other",
            11,
            RuntimeEventKind.INVOCATION_CREATE,
            "wf",
            invocation_id="inv-other",
            context_id="ctx-other",
            context_epoch=0,
        )
    )
    other = PageHandle(2, 0)
    controller.page_index.register_page(
        other,
        size_bytes=400,
        residency=PhysicalResidency.CPU_ONLY,
    )
    controller.page_index.bind_pages("ctx-other", 0, (other,))

    tick = controller.tick(12)

    assert tick.transfer is not None
    assert tick.transfer.command.context_id == "ctx-other"


def test_cache_reset_clears_blocked_attempt_without_permanent_suppression() -> None:
    controller = _controller()
    _reject_prefetch(
        controller,
        now_ms=10,
        blocker_codes=(TransferBlockerCode.ANCESTOR_CLOSURE,),
    )
    assert controller.tick(11).transfer is None

    controller.reset_transfer_attempts()
    retried = controller.tick(12)

    assert retried.transfer is not None
    assert controller.transfer_guard.summary()["active_blocked_attempt_count"] == 0


def test_unknown_backend_uses_exponential_backoff_then_opens_circuit() -> None:
    controller = _controller(unknown_base_ms=10, unknown_circuit_failures=3)
    _reject_prefetch(
        controller,
        now_ms=10,
        blocker_codes=(TransferBlockerCode.UNKNOWN_BACKEND,),
    )

    assert controller.tick(19.9).transfer is None
    second = controller.tick(20.1)
    assert second.transfer is not None
    _ack_rejection(
        controller,
        command_id=second.transfer.command.command_id,
        now_ms=20.2,
        blocker_codes=(TransferBlockerCode.UNKNOWN_BACKEND,),
    )

    assert controller.tick(40.1).transfer is None
    third = controller.tick(40.2)
    assert third.transfer is not None
    _ack_rejection(
        controller,
        command_id=third.transfer.command.command_id,
        now_ms=40.3,
        blocker_codes=(TransferBlockerCode.UNKNOWN_BACKEND,),
    )

    assert controller.tick(2000).transfer is None
    assert controller.transfer_guard.summary()["unknown_circuit_open_count"] == 1


def test_partial_ack_tracks_only_failed_page_bytes_as_capacity_debt() -> None:
    controller = _controller()
    second = PageHandle(2, 0)
    controller.page_index.register_page(
        second,
        size_bytes=300,
        residency=PhysicalResidency.CPU_ONLY,
    )
    controller.page_index.set_parent(PageHandle(1, 0), second)
    tick = controller.tick(10)
    assert tick.transfer is not None
    first = PageHandle(1, 0)
    assert {item.handle for item in tick.transfer.page_actions} == {first, second}
    controller.mark_command_started(tick.transfer.command.command_id, (first,))

    controller.acknowledge_command(
        CommandAck(
            command_id=tick.transfer.command.command_id,
            status=CommandStatus.PARTIAL,
            completed_ts_ms=10.1,
            actual_bytes=400,
            page_handles=(first,),
            blockers=(
                TransferBlocker(
                    TransferBlockerCode.DEVICE_CAPACITY,
                    second,
                    300,
                ),
            ),
        )
    )

    blocked = [
        event
        for event in controller.transfer_guard.drain_events()
        if event.kind == "transfer_attempt_blocked"
    ]
    assert blocked[-1].fields["required_bytes"] == 300
    assert blocked[-1].fields["capacity_snapshot_pending"] is True

    # The first post-ACK tick anchors the blocker to the allocator state after
    # the 400-byte successful prefix was restored. It must not retry yet.
    anchored_tick = controller.tick(10.2)
    assert anchored_tick.transfer is None
    anchored = [
        event
        for event in anchored_tick.transfer_guard_events
        if event.kind == "transfer_retry_capacity_anchored"
    ]
    assert anchored[-1].fields["failed_device_available_bytes"] == 600
    assert anchored[-1].fields["capacity_snapshot_pending"] is False

    # Only a later increase over the post-ACK free-space baseline releases it.
    controller.page_index.commit_cpu(first)
    released_tick = controller.tick(10.3)
    assert released_tick.transfer is not None
    assert released_tick.transfer.command.kind == CommandKind.PREFETCH_CONTEXT
