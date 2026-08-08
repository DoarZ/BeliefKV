from __future__ import annotations

import math
from dataclasses import dataclass, replace

from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import PageHandle


DEFAULT_SERVICE_WINDOWS_MS = (100.0, 500.0)


@dataclass(frozen=True)
class RequestServiceRecord:
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    first_selected_ts_ms: float
    last_selected_ts_ms: float
    last_completed_service_ts_ms: float | None = None
    last_service_phase: str | None = None
    completed_service_count: int = 0


@dataclass(frozen=True)
class LockedExtentAttribution:
    handle: PageHandle
    size_bytes: int
    engine_lock_ref: int
    blocker_request_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError("locked extent size must be positive")
        if self.engine_lock_ref <= 0:
            raise ValueError("locked extent requires a positive engine lock ref")
        if tuple(sorted(set(self.blocker_request_ids))) != self.blocker_request_ids:
            raise ValueError("blocker request ids must be sorted and unique")


@dataclass(frozen=True)
class TentativeUnlockPreview:
    """Conservative physical closure unlocked by hypothetical request removal."""

    request_ids: tuple[str, ...]
    page_revision: int
    topology_revision: int
    exact: bool
    reason: str
    path_error_count: int
    provenance_extent_count: int
    selected_blocker_extent_count: int
    attribution_gap_bytes: int
    baseline_engine_locked_bytes: int
    projected_engine_locked_bytes: int
    projected_engine_lock_release_bytes: int
    baseline_migratable_bytes: int
    projected_migratable_bytes: int
    lock_ref_zeroed_bytes: int
    newly_migratable_bytes: int
    lock_ref_zeroed_handles: tuple[PageHandle, ...]
    newly_migratable_handles: tuple[PageHandle, ...]

    @property
    def closure_amplification(self) -> float:
        if self.lock_ref_zeroed_bytes <= 0:
            return 0.0
        return self.newly_migratable_bytes / self.lock_ref_zeroed_bytes

    def to_audit_fields(self) -> dict[str, object]:
        return {
            "preview_request_ids": list(self.request_ids),
            "preview_page_revision": self.page_revision,
            "preview_topology_revision": self.topology_revision,
            "preview_exact": self.exact,
            "preview_reason": self.reason,
            "preview_path_error_count": self.path_error_count,
            "preview_provenance_extent_count": self.provenance_extent_count,
            "preview_selected_blocker_extent_count": (
                self.selected_blocker_extent_count
            ),
            "preview_attribution_gap_bytes": self.attribution_gap_bytes,
            "preview_baseline_engine_locked_bytes": (
                self.baseline_engine_locked_bytes
            ),
            "preview_projected_engine_locked_bytes": (
                self.projected_engine_locked_bytes
            ),
            "preview_projected_engine_lock_release_bytes": (
                self.projected_engine_lock_release_bytes
            ),
            "preview_baseline_migratable_bytes": self.baseline_migratable_bytes,
            "preview_projected_migratable_bytes": (
                self.projected_migratable_bytes
            ),
            "preview_lock_ref_zeroed_bytes": self.lock_ref_zeroed_bytes,
            "preview_newly_migratable_bytes": self.newly_migratable_bytes,
            "preview_closure_amplification": self.closure_amplification,
            "preview_lock_ref_zeroed_handles": [
                _handle_label(handle) for handle in self.lock_ref_zeroed_handles
            ],
            "preview_newly_migratable_handles": [
                _handle_label(handle) for handle in self.newly_migratable_handles
            ],
        }


class TentativeUnlockPreviewer:
    """Translate request lock provenance into a read-only physical projection."""

    def __init__(self, page_index: PageOwnershipIndex) -> None:
        self.page_index = page_index

    def preview(
        self,
        extents: tuple[LockedExtentAttribution, ...],
        request_ids: tuple[str, ...],
        *,
        path_error_count: int = 0,
    ) -> TentativeUnlockPreview:
        selected_ids = tuple(sorted(set(request_ids)))
        if not selected_ids or len(selected_ids) != len(request_ids):
            raise ValueError("tentative unlock requires unique request ids")
        if path_error_count < 0:
            raise ValueError("path error count must be non-negative")

        selected = set(selected_ids)
        extent_by_handle: dict[PageHandle, LockedExtentAttribution] = {}
        duplicate_handles: set[PageHandle] = set()
        for extent in extents:
            if extent.handle in extent_by_handle:
                duplicate_handles.add(extent.handle)
                continue
            extent_by_handle[extent.handle] = extent

        locked_pages = {
            page.handle: page
            for page in self.page_index.engine_locked_gpu_pages()
        }
        gap_bytes_by_handle: dict[PageHandle, int] = {
            handle: locked_pages[handle].size_bytes
            for handle in duplicate_handles
            if handle in locked_pages
        }
        overrides: dict[PageHandle, int] = {}
        selected_extent_count = 0
        for handle, page in locked_pages.items():
            extent = extent_by_handle.get(handle)
            if extent is None:
                gap_bytes_by_handle[handle] = page.size_bytes
                continue
            matches_page = (
                extent.size_bytes == page.size_bytes
                and extent.engine_lock_ref == page.engine_lock_ref
            )
            fully_attributed = (
                extent.blocker_request_ids
                and extent.engine_lock_ref == len(extent.blocker_request_ids)
            )
            selected_blockers = selected.intersection(extent.blocker_request_ids)
            if selected_blockers:
                selected_extent_count += 1
            if not matches_page or not fully_attributed:
                gap_bytes_by_handle[handle] = page.size_bytes
                continue
            if selected_blockers:
                overrides[handle] = (
                    page.engine_lock_ref - len(selected_blockers)
                )

        stale_extent_bytes = sum(
            extent.size_bytes
            for handle, extent in extent_by_handle.items()
            if handle not in locked_pages
        )
        projection = self.page_index.preview_engine_lock_release(overrides)
        attribution_gap_bytes = sum(gap_bytes_by_handle.values()) + stale_extent_bytes
        exact = (
            path_error_count == 0
            and attribution_gap_bytes == 0
            and not duplicate_handles
        )
        if not exact:
            reason = "provenance_incomplete"
        elif selected_extent_count == 0:
            reason = "no_selected_blocker"
        elif projection.newly_migratable_bytes == 0:
            reason = "lock_release_without_migratable_closure"
        else:
            reason = "projected_unlock"
        return TentativeUnlockPreview(
            request_ids=selected_ids,
            page_revision=projection.page_revision,
            topology_revision=projection.topology_revision,
            exact=exact,
            reason=reason,
            path_error_count=path_error_count,
            provenance_extent_count=len(extents),
            selected_blocker_extent_count=selected_extent_count,
            attribution_gap_bytes=attribution_gap_bytes,
            baseline_engine_locked_bytes=(
                projection.baseline.engine_locked_bytes
            ),
            projected_engine_locked_bytes=(
                projection.projected.engine_locked_bytes
            ),
            projected_engine_lock_release_bytes=max(
                0,
                projection.baseline.engine_locked_bytes
                - projection.projected.engine_locked_bytes,
            ),
            baseline_migratable_bytes=projection.baseline.migratable_bytes,
            projected_migratable_bytes=projection.projected.migratable_bytes,
            lock_ref_zeroed_bytes=projection.lock_ref_zeroed_bytes,
            newly_migratable_bytes=projection.newly_migratable_bytes,
            lock_ref_zeroed_handles=projection.lock_ref_zeroed_handles,
            newly_migratable_handles=projection.newly_migratable_handles,
        )


@dataclass(frozen=True)
class LockServiceWindowDiagnostics:
    window_ms: float
    recently_served_request_count: int
    not_served_request_count: int
    warming_request_count: int
    unknown_request_count: int
    recently_served_unique_bytes: int
    locked_but_not_served_unique_bytes: int
    warming_unique_bytes: int
    unknown_unique_bytes: int
    not_served_logical_request_bytes: int


@dataclass(frozen=True)
class LockServiceDiagnostics:
    observed_ts_ms: float
    engine_locked_unique_bytes: int
    attributed_unique_bytes: int
    fully_attributed_unique_bytes: int
    partially_attributed_unique_bytes: int
    unattributed_unique_bytes: int
    locked_extent_count: int
    attributed_extent_count: int
    blocker_request_count: int
    request_path_error_count: int
    lock_ref_mismatch_extent_count: int
    logical_request_lock_bytes: int
    windows: tuple[LockServiceWindowDiagnostics, ...]

    @property
    def attribution_coverage(self) -> float:
        if self.engine_locked_unique_bytes == 0:
            return 1.0
        return self.attributed_unique_bytes / self.engine_locked_unique_bytes

    def to_audit_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "engine_lock_ref_gpu_bytes": self.engine_locked_unique_bytes,
            "engine_lock_attributed_gpu_bytes": self.attributed_unique_bytes,
            "engine_lock_fully_attributed_gpu_bytes": (
                self.fully_attributed_unique_bytes
            ),
            "engine_lock_partially_attributed_gpu_bytes": (
                self.partially_attributed_unique_bytes
            ),
            "engine_lock_unattributed_gpu_bytes": self.unattributed_unique_bytes,
            "engine_lock_attribution_coverage": self.attribution_coverage,
            "engine_lock_full_attribution_coverage": (
                1.0
                if self.engine_locked_unique_bytes == 0
                else self.fully_attributed_unique_bytes
                / self.engine_locked_unique_bytes
            ),
            "engine_lock_extent_count": self.locked_extent_count,
            "engine_lock_attributed_extent_count": self.attributed_extent_count,
            "engine_lock_blocker_request_count": self.blocker_request_count,
            "engine_lock_request_path_error_count": self.request_path_error_count,
            "engine_lock_ref_mismatch_extent_count": (
                self.lock_ref_mismatch_extent_count
            ),
            "engine_lock_logical_request_bytes": self.logical_request_lock_bytes,
            "engine_lock_provenance_scope": (
                "running_request_last_node_to_radix_root"
            ),
            "engine_lock_service_evidence": "completed_gpu_batch",
        }
        for item in self.windows:
            suffix = _window_suffix(item.window_ms)
            fields.update(
                {
                    f"lock_recently_served_request_count_{suffix}": (
                        item.recently_served_request_count
                    ),
                    f"lock_not_served_request_count_{suffix}": (
                        item.not_served_request_count
                    ),
                    f"lock_warming_request_count_{suffix}": (
                        item.warming_request_count
                    ),
                    f"lock_unknown_request_count_{suffix}": (
                        item.unknown_request_count
                    ),
                    f"lock_recently_served_gpu_bytes_{suffix}": (
                        item.recently_served_unique_bytes
                    ),
                    f"locked_but_not_served_gpu_bytes_{suffix}": (
                        item.locked_but_not_served_unique_bytes
                    ),
                    f"lock_warming_gpu_bytes_{suffix}": item.warming_unique_bytes,
                    f"lock_unknown_gpu_bytes_{suffix}": item.unknown_unique_bytes,
                    f"lock_not_served_logical_request_bytes_{suffix}": (
                        item.not_served_logical_request_bytes
                    ),
                }
            )
        return fields


class RequestServiceLedger:
    """Track completed GPU service independently from queue/running residency."""

    def __init__(self) -> None:
        self._records: dict[str, RequestServiceRecord] = {}

    def observe_selected(
        self,
        *,
        request_id: str,
        workflow_id: str,
        invocation_id: str,
        context_id: str,
        ts_ms: float,
    ) -> None:
        _validate_identity(request_id, workflow_id, invocation_id, context_id)
        _validate_ts(ts_ms)
        previous = self._records.get(request_id)
        if previous is None:
            self._records[request_id] = RequestServiceRecord(
                request_id=request_id,
                workflow_id=workflow_id,
                invocation_id=invocation_id,
                context_id=context_id,
                first_selected_ts_ms=ts_ms,
                last_selected_ts_ms=ts_ms,
            )
            return
        if (
            previous.workflow_id != workflow_id
            or previous.invocation_id != invocation_id
            or previous.context_id != context_id
        ):
            raise ValueError("request service identity changed after selection")
        if ts_ms < previous.last_selected_ts_ms:
            raise ValueError("request selection time moved backwards")
        self._records[request_id] = replace(
            previous,
            last_selected_ts_ms=ts_ms,
        )

    def observe_completed(
        self,
        request_id: str,
        *,
        ts_ms: float,
        phase: str,
    ) -> None:
        _validate_ts(ts_ms)
        if not phase:
            raise ValueError("GPU service phase must be non-empty")
        previous = self._records.get(request_id)
        if previous is None:
            raise KeyError(f"GPU service completed before selection: {request_id}")
        if ts_ms < previous.last_selected_ts_ms or (
            previous.last_completed_service_ts_ms is not None
            and ts_ms < previous.last_completed_service_ts_ms
        ):
            raise ValueError("GPU service completion time moved backwards")
        self._records[request_id] = replace(
            previous,
            last_completed_service_ts_ms=ts_ms,
            last_service_phase=phase,
            completed_service_count=previous.completed_service_count + 1,
        )

    def forget(self, request_id: str) -> None:
        self._records.pop(request_id, None)

    def tracks(self, request_id: str) -> bool:
        return request_id in self._records

    def progress_record(self, request_id: str) -> RequestServiceRecord | None:
        """Return the latest immutable GPU-service evidence for a request."""

        return self._records.get(request_id)

    def service_status(
        self,
        request_id: str,
        *,
        now_ms: float,
        window_ms: float,
    ) -> str:
        _validate_ts(now_ms)
        if not math.isfinite(window_ms) or window_ms <= 0:
            raise ValueError("service status window must be positive")
        return self._status(request_id, now_ms=now_ms, window_ms=window_ms)

    def stale_for_ms(self, request_id: str, *, now_ms: float) -> float:
        _validate_ts(now_ms)
        record = self._records.get(request_id)
        if record is None:
            return 0.0
        evidence_ts = (
            record.last_completed_service_ts_ms
            if record.last_completed_service_ts_ms is not None
            else record.first_selected_ts_ms
        )
        return max(0.0, now_ms - evidence_ts)

    def clear(self) -> None:
        self._records.clear()

    def summarize(
        self,
        extents: tuple[LockedExtentAttribution, ...],
        *,
        now_ms: float,
        windows_ms: tuple[float, ...] = DEFAULT_SERVICE_WINDOWS_MS,
        request_path_error_count: int = 0,
    ) -> LockServiceDiagnostics:
        _validate_ts(now_ms)
        if request_path_error_count < 0:
            raise ValueError("request path error count must be non-negative")
        if not windows_ms or any(window <= 0 for window in windows_ms):
            raise ValueError("service windows must be positive")

        blocker_ids = {
            request_id
            for extent in extents
            for request_id in extent.blocker_request_ids
        }
        total_bytes = sum(extent.size_bytes for extent in extents)
        attributed_bytes = sum(
            extent.size_bytes for extent in extents if extent.blocker_request_ids
        )
        fully_attributed_bytes = sum(
            extent.size_bytes
            for extent in extents
            if extent.blocker_request_ids
            and extent.engine_lock_ref == len(extent.blocker_request_ids)
        )
        partially_attributed_bytes = sum(
            extent.size_bytes
            for extent in extents
            if extent.blocker_request_ids
            and extent.engine_lock_ref != len(extent.blocker_request_ids)
        )
        logical_bytes = sum(
            extent.size_bytes * len(extent.blocker_request_ids)
            for extent in extents
        )
        windows = tuple(
            self._window_diagnostics(
                extents,
                blocker_ids=blocker_ids,
                now_ms=now_ms,
                window_ms=float(window_ms),
            )
            for window_ms in windows_ms
        )
        return LockServiceDiagnostics(
            observed_ts_ms=now_ms,
            engine_locked_unique_bytes=total_bytes,
            attributed_unique_bytes=attributed_bytes,
            fully_attributed_unique_bytes=fully_attributed_bytes,
            partially_attributed_unique_bytes=partially_attributed_bytes,
            unattributed_unique_bytes=total_bytes - attributed_bytes,
            locked_extent_count=len(extents),
            attributed_extent_count=sum(
                bool(extent.blocker_request_ids) for extent in extents
            ),
            blocker_request_count=len(blocker_ids),
            request_path_error_count=request_path_error_count,
            lock_ref_mismatch_extent_count=sum(
                extent.engine_lock_ref != len(extent.blocker_request_ids)
                for extent in extents
            ),
            logical_request_lock_bytes=logical_bytes,
            windows=windows,
        )

    def _window_diagnostics(
        self,
        extents: tuple[LockedExtentAttribution, ...],
        *,
        blocker_ids: set[str],
        now_ms: float,
        window_ms: float,
    ) -> LockServiceWindowDiagnostics:
        status_by_request = {
            request_id: self._status(request_id, now_ms=now_ms, window_ms=window_ms)
            for request_id in blocker_ids
        }
        physical_bytes = {name: 0 for name in ("recent", "stale", "warming", "unknown")}
        stale_logical_bytes = 0
        for extent in extents:
            if not extent.blocker_request_ids:
                continue
            statuses = tuple(
                status_by_request[request_id]
                for request_id in extent.blocker_request_ids
            )
            if "recent" in statuses:
                extent_status = "recent"
            elif extent.engine_lock_ref != len(extent.blocker_request_ids):
                extent_status = "unknown"
            elif "warming" in statuses:
                extent_status = "warming"
            elif "unknown" in statuses:
                extent_status = "unknown"
            else:
                extent_status = "stale"
            physical_bytes[extent_status] += extent.size_bytes
            stale_logical_bytes += extent.size_bytes * sum(
                status == "stale" for status in statuses
            )

        return LockServiceWindowDiagnostics(
            window_ms=window_ms,
            recently_served_request_count=sum(
                status == "recent" for status in status_by_request.values()
            ),
            not_served_request_count=sum(
                status == "stale" for status in status_by_request.values()
            ),
            warming_request_count=sum(
                status == "warming" for status in status_by_request.values()
            ),
            unknown_request_count=sum(
                status == "unknown" for status in status_by_request.values()
            ),
            recently_served_unique_bytes=physical_bytes["recent"],
            locked_but_not_served_unique_bytes=physical_bytes["stale"],
            warming_unique_bytes=physical_bytes["warming"],
            unknown_unique_bytes=physical_bytes["unknown"],
            not_served_logical_request_bytes=stale_logical_bytes,
        )

    def _status(self, request_id: str, *, now_ms: float, window_ms: float) -> str:
        record = self._records.get(request_id)
        if record is None:
            return "unknown"
        completed_ts = record.last_completed_service_ts_ms
        if completed_ts is not None and now_ms - completed_ts <= window_ms:
            return "recent"
        if completed_ts is None and now_ms - record.first_selected_ts_ms <= window_ms:
            return "warming"
        return "stale"


def _window_suffix(window_ms: float) -> str:
    return f"{int(window_ms)}ms" if window_ms.is_integer() else f"{window_ms:g}ms"


def _handle_label(handle: PageHandle) -> str:
    return f"{handle.page_id}:{handle.allocation_generation}"


def _validate_identity(*values: str) -> None:
    if any(not value for value in values):
        raise ValueError("request service identities must be non-empty")


def _validate_ts(ts_ms: float) -> None:
    if not math.isfinite(ts_ms) or ts_ms < 0:
        raise ValueError("request service timestamp must be finite and non-negative")
