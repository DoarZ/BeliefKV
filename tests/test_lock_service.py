from __future__ import annotations

from beliefkv.runtime.lock_service import (
    LockedExtentAttribution,
    RequestServiceLedger,
    TentativeUnlockPreviewer,
)
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import PageHandle


def _extent(
    page_id: int,
    size_bytes: int,
    lock_ref: int,
    *blockers: str,
) -> LockedExtentAttribution:
    return LockedExtentAttribution(
        handle=PageHandle(page_id, 0),
        size_bytes=size_bytes,
        engine_lock_ref=lock_ref,
        blocker_request_ids=tuple(sorted(blockers)),
    )


def _select(ledger: RequestServiceLedger, request_id: str) -> None:
    ledger.observe_selected(
        request_id=request_id,
        workflow_id=f"workflow-{request_id}",
        invocation_id=f"invocation-{request_id}",
        context_id=f"context-{request_id}",
        ts_ms=0.0,
    )


def test_service_windows_distinguish_warming_recent_and_stale_locks() -> None:
    ledger = RequestServiceLedger()
    _select(ledger, "r1")
    _select(ledger, "r2")
    extents = (
        _extent(1, 100, 1, "r1"),
        _extent(2, 200, 2, "r1", "r2"),
        _extent(3, 50, 1),
    )

    warming = ledger.summarize(extents, now_ms=50.0)
    assert warming.engine_locked_unique_bytes == 350
    assert warming.attributed_unique_bytes == 300
    assert warming.unattributed_unique_bytes == 50
    assert warming.windows[0].warming_unique_bytes == 300
    assert warming.windows[0].locked_but_not_served_unique_bytes == 0

    ledger.observe_completed("r1", ts_ms=60.0, phase="decode")
    mixed = ledger.summarize(extents, now_ms=120.0)
    window_100 = mixed.windows[0]
    assert window_100.recently_served_request_count == 1
    assert window_100.not_served_request_count == 1
    assert window_100.recently_served_unique_bytes == 300
    assert window_100.locked_but_not_served_unique_bytes == 0
    assert window_100.not_served_logical_request_bytes == 200
    assert mixed.windows[1].warming_request_count == 1

    stale = ledger.summarize(extents, now_ms=700.0)
    assert stale.windows[0].locked_but_not_served_unique_bytes == 300
    assert stale.windows[1].locked_but_not_served_unique_bytes == 300


def test_partial_lock_provenance_never_becomes_physical_stale_bytes() -> None:
    ledger = RequestServiceLedger()
    _select(ledger, "known")
    diagnostics = ledger.summarize(
        (_extent(1, 100, 2, "known"),),
        now_ms=700.0,
    )

    assert diagnostics.partially_attributed_unique_bytes == 100
    assert diagnostics.fully_attributed_unique_bytes == 0
    assert diagnostics.lock_ref_mismatch_extent_count == 1
    assert diagnostics.windows[0].locked_but_not_served_unique_bytes == 0
    assert diagnostics.windows[0].unknown_unique_bytes == 100


def test_audit_fields_use_explicit_100_and_500_ms_names() -> None:
    ledger = RequestServiceLedger()
    _select(ledger, "r1")
    fields = ledger.summarize(
        (_extent(1, 100, 1, "r1"),),
        now_ms=700.0,
    ).to_audit_fields()

    assert fields["locked_but_not_served_gpu_bytes_100ms"] == 100
    assert fields["locked_but_not_served_gpu_bytes_500ms"] == 100
    assert fields["engine_lock_service_evidence"] == "completed_gpu_batch"


def test_forget_removes_terminal_request_record() -> None:
    ledger = RequestServiceLedger()
    _select(ledger, "terminal")
    assert ledger.tracks("terminal")

    ledger.observe_completed("terminal", ts_ms=10.0, phase="decode")
    ledger.forget("terminal")

    assert not ledger.tracks("terminal")


def test_tentative_unlock_projects_radix_closure_without_mutation() -> None:
    index = PageOwnershipIndex()
    parent = PageHandle(1, 0)
    child = PageHandle(2, 0)
    index.register_page(parent, size_bytes=100, radix_depth=1)
    index.register_page(child, size_bytes=200, radix_depth=2, parent=parent)
    index.set_engine_lock(child, 1)
    revision = index.revision

    preview = TentativeUnlockPreviewer(index).preview(
        (_extent(2, 200, 1, "victim"),),
        ("victim",),
    )

    assert preview.exact
    assert preview.reason == "projected_unlock"
    assert preview.projected_engine_lock_release_bytes == 200
    assert preview.lock_ref_zeroed_bytes == 200
    assert preview.newly_migratable_bytes == 300
    assert preview.closure_amplification == 1.5
    assert preview.newly_migratable_handles == (parent, child)
    assert index.pages[child].engine_lock_ref == 1
    assert index.revision == revision


def test_tentative_unlock_requires_complete_blocker_set() -> None:
    index = PageOwnershipIndex()
    handle = PageHandle(1, 0)
    index.register_page(handle, size_bytes=200)
    index.set_engine_lock(handle, 2)
    extents = (_extent(1, 200, 2, "a", "b"),)
    previewer = TentativeUnlockPreviewer(index)

    partial = previewer.preview(extents, ("a",))
    complete = previewer.preview(extents, ("a", "b"))

    assert partial.exact
    assert partial.reason == "lock_release_without_migratable_closure"
    assert partial.projected_engine_locked_bytes == 200
    assert partial.newly_migratable_bytes == 0
    assert complete.exact
    assert complete.reason == "projected_unlock"
    assert complete.projected_engine_locked_bytes == 0
    assert complete.newly_migratable_bytes == 200


def test_tentative_unlock_fails_closed_on_partial_provenance() -> None:
    index = PageOwnershipIndex()
    handle = PageHandle(1, 0)
    index.register_page(handle, size_bytes=200)
    index.set_engine_lock(handle, 2)

    preview = TentativeUnlockPreviewer(index).preview(
        (_extent(1, 200, 2, "known"),),
        ("known",),
        path_error_count=1,
    )

    assert not preview.exact
    assert preview.reason == "provenance_incomplete"
    assert preview.attribution_gap_bytes == 200
    assert preview.projected_engine_lock_release_bytes == 0
    assert preview.newly_migratable_bytes == 0
