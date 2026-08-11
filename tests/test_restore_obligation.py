from beliefkv.runtime.protocol import CommandKind
from beliefkv.runtime.restore_obligation import (
    ExternalProgressToken,
    NativeQueueLocation,
    NativeRequestPhysicalSnapshot,
    SafePointPhysicalSnapshot,
    SafePointSnapshotBuildTiming,
    RestoreLeaseIndex,
    RestoreLeaseState,
    RestoreObligationCause,
    RestoreObligationIndex,
    RestoreObligationState,
    PersistentLivenessRevisionTracker,
    RestoreServiceGrace,
    RestorePhysicalOperation,
    RestoreTransaction,
)


def _create(index: RestoreObligationIndex, request_id: str = "request"):
    return index.create(
        request_id=request_id,
        workflow_id="workflow",
        invocation_id="invocation",
        context_id="context",
        context_epoch=0,
        source_retraction_transaction_id="retraction-1",
        source_joint_plan_id="joint-plan-1",
        created_ts_ms=10.0,
        path_extent_ids=("page:1:0", "page:2:0"),
    )


def test_restore_obligation_tracks_funding_restore_and_service():
    index = RestoreObligationIndex(max_active=2)
    obligation = _create(index)

    obligation.requeued = True
    obligation.source_transaction_terminal = True
    obligation.set_required_extents(
        ("page:2:0",), restore_bytes=128, now_ms=20.0
    )
    stamp = (1, 2, 64)
    obligation.start_command(
        "fund", CommandKind.OFFLOAD_CONTEXT, now_ms=21.0, attempt_stamp=stamp
    )
    assert obligation.state == RestoreObligationState.EVICT_FOR_RESTORE
    assert obligation.clear_command() == CommandKind.OFFLOAD_CONTEXT
    obligation.funding_reclaim_bytes += 128

    obligation.start_command(
        "restore", CommandKind.PREFETCH_CONTEXT, now_ms=22.0, attempt_stamp=stamp
    )
    assert obligation.state == RestoreObligationState.H2D_INFLIGHT
    assert obligation.clear_command() == CommandKind.PREFETCH_CONTEXT
    obligation.restored_bytes += 128
    obligation.mark_ticket_ready(now_ms=23.0)
    obligation.finish(
        RestoreObligationState.SATISFIED,
        now_ms=24.0,
        reason="gpu_service_resumed",
    )

    assert obligation.state.terminal
    assert obligation.first_service_ts_ms == 24.0
    assert index.active() == ()


def test_restore_obligation_records_ordinary_waiting_cause():
    index = RestoreObligationIndex(max_active=1)
    obligation = index.create(
        request_id="waiting",
        workflow_id="workflow",
        invocation_id="invocation",
        context_id="context",
        context_epoch=0,
        source_retraction_transaction_id="ordinary-waiting:waiting",
        source_joint_plan_id="joint-liveness:ordinary-waiting-restore",
        created_ts_ms=10.0,
        path_extent_ids=("page:1:0",),
        cause=RestoreObligationCause.ORDINARY_WAITING_PREFIX,
    )

    assert obligation.cause == RestoreObligationCause.ORDINARY_WAITING_PREFIX


def test_restore_service_grace_requires_completed_decode_quantum():
    grace = RestoreServiceGrace(
        request_id="request",
        obligation_id="restore-1",
        granted_ts_ms=10.0,
        required_decode_tokens=4,
    )

    assert not grace.observe_decode(3, now_ms=11.0)
    assert grace.active
    assert grace.remaining_decode_tokens == 1
    assert grace.observe_decode(1, now_ms=12.0)
    assert not grace.active
    assert grace.terminal_reason == "service_quantum_satisfied"


def test_restore_obligation_index_fails_closed_at_capacity_or_duplicate():
    index = RestoreObligationIndex(max_active=1)
    _create(index)

    assert not index.can_create(("other",))
    assert not index.can_create(("request",))


def test_blocked_restore_retries_only_after_state_stamp_changes():
    index = RestoreObligationIndex(max_active=1)
    obligation = _create(index)
    stamp = (3, 4, 0)

    obligation.block(
        blocker_codes=("device_capacity",),
        blocker_fingerprint="capacity:128",
        attempt_stamp=stamp,
        now_ms=30.0,
    )

    assert obligation.state == RestoreObligationState.PARKED_WAIT
    assert obligation.last_attempt_stamp == stamp
    assert obligation.retry_count == 1


def test_restore_retry_depends_only_on_external_progress_token():
    obligation = _create(RestoreObligationIndex(max_active=1))
    token = ExternalProgressToken(
        engine_owner_epoch=(("request", 1, "waiting", None, False, False),),
        closure_fingerprint="closure-1",
        effective_capacity_threshold_epoch=(128, False),
        command_ownership_epoch=("context", "none"),
        guard_generation=2,
        native_load_generation=(),
    )
    obligation.block(
        blocker_codes=("engine_busy",),
        blocker_fingerprint="engine_busy",
        external_progress_token=token,
        wake_conditions=("engine_owner_changed",),
        now_ms=20.0,
    )

    assert not obligation.external_progressed(token)
    assert obligation.external_progressed(
        ExternalProgressToken(
            engine_owner_epoch=(),
            closure_fingerprint="closure-1",
            effective_capacity_threshold_epoch=(128, False),
            command_ownership_epoch=("context", "none"),
            guard_generation=2,
            native_load_generation=(),
        )
    )

    lease_token = ExternalProgressToken(
        engine_owner_epoch=token.engine_owner_epoch,
        closure_fingerprint=token.closure_fingerprint,
        effective_capacity_threshold_epoch=token.effective_capacity_threshold_epoch,
        command_ownership_epoch=token.command_ownership_epoch,
        guard_generation=token.guard_generation,
        native_load_generation=token.native_load_generation,
        restore_lease_epoch=(("restore-lease-1", "request", "granted", 8, 16),),
    )
    assert obligation.external_progressed(lease_token)


def test_restore_transaction_deduplicates_stage_attempt_certificate():
    obligation = _create(RestoreObligationIndex(max_active=1))
    transaction = RestoreTransaction("tx-1", obligation)
    operation = RestorePhysicalOperation(
        stage="prefetch_context",
        attempt_key=("ctx", 0, "prefetch"),
        certificate_generation=3,
        canonical_command_id="command-1",
        adopted=False,
    )
    transaction.add_operation(operation)

    assert not transaction.can_submit(
        stage=operation.stage,
        attempt_key=operation.attempt_key,
        certificate_generation=operation.certificate_generation,
    )


def test_native_request_snapshot_keeps_state_dimensions_orthogonal():
    snapshot = NativeRequestPhysicalSnapshot(
        request_id="request",
        context_id="context",
        queue_location=NativeQueueLocation.WAITING,
        req_pool_slot=3,
        radix_lock_owned=False,
        native_load_operation_id="native-1",
        explicit_transfer_ids=("explicit-1",),
        request_generation=7,
        terminal=False,
    )

    assert snapshot.engine_owned
    assert snapshot.native_load_operation_id == "native-1"
    assert snapshot.explicit_transfer_ids == ("explicit-1",)


def test_safe_point_physical_snapshot_indexes_are_immutable():
    record = NativeRequestPhysicalSnapshot(
        request_id="request",
        context_id="context",
        queue_location=NativeQueueLocation.WAITING,
        req_pool_slot=None,
        radix_lock_owned=False,
        native_load_operation_id=None,
        explicit_transfer_ids=("transfer-1",),
        request_generation=1,
        terminal=False,
    )
    snapshot = SafePointPhysicalSnapshot(
        epoch=3,
        records=(record,),
        by_request={"request": record},
        by_context={"context": (record,)},
        explicit_transfers_by_context={"context": ("transfer-1",)},
        queue_record_count=1,
        metadata_record_count=1,
        timing=SafePointSnapshotBuildTiming(
            total_ms=0.1,
            queue_collection_ms=0.01,
            metadata_indexing_ms=0.01,
            radix_ownership_lookup_ms=0.05,
            operation_indexing_ms=0.01,
            sorting_allocation_ms=0.02,
            queue_record_count=1,
            metadata_record_count=1,
            matched_record_count=1,
        ),
    )

    assert snapshot.for_context("context") == (record,)
    assert snapshot.context_readset("context") == (
        (record,),
        ("transfer-1",),
    )
    try:
        snapshot.by_request["other"] = record
    except TypeError:
        pass
    else:
        raise AssertionError("snapshot request index must be immutable")


def test_restore_lease_holds_capacity_until_admission_commits():
    obligation_index = RestoreObligationIndex(max_active=1)
    obligation = _create(obligation_index)
    leases = RestoreLeaseIndex(max_active=1)

    lease = leases.grant(
        obligation=obligation,
        granted_ts_ms=20.0,
        reserved_tokens=16,
        reserved_bytes=160,
        h2d_bytes=320,
    )
    lease.mark_h2d_inflight("h2d-1")
    lease.mark_restored()

    assert leases.reserved_tokens == 16
    assert lease.state == RestoreLeaseState.RESTORED_RESERVED

    lease.begin_admission()
    assert leases.reserved_tokens == 0
    lease.admission_rejected()
    assert lease.state == RestoreLeaseState.RESTORED_RESERVED

    lease.begin_admission()
    lease.mark_admitted()
    assert leases.reserved_tokens == 0
    lease.finish(
        RestoreLeaseState.RELEASED,
        now_ms=30.0,
        reason="gpu_service_resumed",
    )
    assert leases.active() == ()


def test_restore_lease_index_bounds_concurrent_capacity_debt():
    obligation_index = RestoreObligationIndex(max_active=2)
    first = _create(obligation_index, "first")
    second = _create(obligation_index, "second")
    leases = RestoreLeaseIndex(max_active=1)

    leases.grant(
        obligation=first,
        granted_ts_ms=20.0,
        reserved_tokens=1,
        reserved_bytes=10,
        h2d_bytes=20,
    )

    try:
        leases.grant(
            obligation=second,
            granted_ts_ms=21.0,
            reserved_tokens=1,
            reserved_bytes=10,
            h2d_bytes=20,
        )
    except ValueError as error:
        assert "capacity" in str(error)
    else:
        raise AssertionError("second concurrent restore lease must fail closed")


def test_persistent_liveness_revisions_detect_direct_legacy_mutations():
    obligation_index = RestoreObligationIndex(max_active=1)
    obligation = _create(obligation_index)
    leases = RestoreLeaseIndex(max_active=1)
    tracker = PersistentLivenessRevisionTracker()

    initial = tracker.observe(
        obligations=obligation_index.all(),
        leases=leases.all(),
        graces=(),
    )
    obligation.requeued = True
    changed = tracker.observe(
        obligations=obligation_index.all(),
        leases=leases.all(),
        graces=(),
    )
    stable = tracker.observe(
        obligations=obligation_index.all(),
        leases=leases.all(),
        graces=(),
    )

    assert changed.obligation_revision == initial.obligation_revision + 1
    assert stable.obligation_revision == changed.obligation_revision
    assert changed.lease_revision == initial.lease_revision
    assert changed.grace_revision == initial.grace_revision
