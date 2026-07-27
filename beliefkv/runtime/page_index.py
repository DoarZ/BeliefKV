from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from beliefkv.runtime.protocol import (
    PageHandle,
    PhysicalResidency,
    TransferDirection,
)


class PageIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class PageIndexChangeSet:
    from_revision: int
    to_revision: int
    topology_revision: int
    handles: frozenset[PageHandle]
    context_ids: frozenset[str]
    components: frozenset[str]
    full_rebuild_required: bool = False


@dataclass(frozen=True)
class PhysicalPageReplica:
    """Immutable page metadata copied at a scheduler safe point."""

    handle: PageHandle
    size_bytes: int
    residency: PhysicalResidency
    radix_depth: int
    parent: PageHandle | None
    children: tuple[PageHandle, ...]
    owner_contexts: tuple[tuple[str, int], ...]
    engine_lock_ref: int
    semantic_pin_contexts: tuple[str, ...]
    active_reader_count: int
    sealed: bool
    transfer_direction: TransferDirection | None
    last_access_ms: float


@dataclass(frozen=True)
class PhysicalPageStateReplica:
    """Small replacement record for non-topology physical state changes."""

    handle: PageHandle
    residency: PhysicalResidency
    engine_lock_ref: int
    semantic_pin_contexts: tuple[str, ...]
    active_reader_count: int
    transfer_direction: TransferDirection | None
    last_access_ms: float


@dataclass(frozen=True)
class ContextPageReplica:
    context_id: str
    workflow_id: str
    epoch: int
    handles: tuple[PageHandle, ...]


@dataclass(frozen=True)
class PageIndexReplicaDelta:
    """Self-contained replacement records for one PageOwnershipIndex delta."""

    from_revision: int
    to_revision: int
    topology_revision: int
    pages: tuple[PhysicalPageReplica, ...]
    page_states: tuple[PhysicalPageStateReplica, ...]
    contexts: tuple[ContextPageReplica, ...]
    changed_handles: frozenset[PageHandle]
    changed_context_ids: frozenset[str]
    components: frozenset[str]
    full_rebuild_required: bool = False


@dataclass(frozen=True)
class _PageIndexMutation:
    from_revision: int
    revision: int
    topology_revision: int
    handles: frozenset[PageHandle]
    context_ids: frozenset[str]
    components: frozenset[str]


@dataclass
class PhysicalPageRecord:
    handle: PageHandle
    size_bytes: int
    residency: PhysicalResidency
    radix_depth: int = 0
    parent: PageHandle | None = None
    children: set[PageHandle] = field(default_factory=set)
    owner_contexts: dict[str, int] = field(default_factory=dict)
    engine_lock_ref: int = 0
    semantic_pin_contexts: set[str] = field(default_factory=set)
    active_reader_count: int = 0
    sealed: bool = True
    transfer_direction: TransferDirection | None = None
    last_access_ms: float = 0.0

    @property
    def gpu_resident(self) -> bool:
        return self.residency in {
            PhysicalResidency.GPU_ONLY,
            PhysicalResidency.MIRRORING,
            PhysicalResidency.DUAL_CLEAN,
            PhysicalResidency.PREFETCHING,
        }

    @property
    def cpu_resident(self) -> bool:
        return self.residency in {
            PhysicalResidency.MIRRORING,
            PhysicalResidency.DUAL_CLEAN,
            PhysicalResidency.CPU_ONLY,
            PhysicalResidency.PREFETCHING,
        }

    @property
    def transfer_idle(self) -> bool:
        return self.transfer_direction is None


@dataclass(frozen=True)
class PhysicalKvStateBreakdown:
    """Physical GPU-residency facts at one PageOwnershipIndex revision.

    The categories are intentionally not a partition. ``dual_resident_bytes``
    describes replica placement, while the other fields describe current D2H
    feasibility. A dual-resident extent can therefore also be migratable.
    """

    gpu_bytes: int = 0
    cpu_bytes: int = 0
    engine_locked_bytes: int = 0
    closure_blocked_bytes: int = 0
    migratable_bytes: int = 0
    dual_resident_bytes: int = 0


@dataclass(frozen=True)
class PhysicalKvUnlockProjection:
    """Read-only physical result of hypothetical engine-lock ref changes."""

    page_revision: int
    topology_revision: int
    baseline: PhysicalKvStateBreakdown
    projected: PhysicalKvStateBreakdown
    overridden_handles: tuple[PageHandle, ...]
    lock_ref_zeroed_handles: tuple[PageHandle, ...]
    lock_ref_zeroed_bytes: int
    newly_migratable_handles: tuple[PageHandle, ...]
    newly_migratable_bytes: int


class PageOwnershipIndex:
    """CPU-side mirror of SGLang's page ownership and location metadata.

    SGLang remains the allocator source of truth. This index only changes
    physical residency after an explicit transfer completion call, mirroring
    the ACK contract used by the real adapter.
    """

    def __init__(self) -> None:
        self.pages: dict[PageHandle, PhysicalPageRecord] = {}
        self._latest_generation: dict[int, int] = {}
        self._context_pages: dict[str, set[PageHandle]] = {}
        self._context_epoch: dict[str, int] = {}
        self._context_workflow: dict[str, str] = {}
        self._revision = 0
        self._topology_revision = 0
        self._physical_state_revision = 0
        self._physical_breakdown_revision = -1
        self._physical_breakdown = PhysicalKvStateBreakdown()
        self._engine_locked_pages: tuple[PhysicalPageRecord, ...] = ()
        self._migratable_gpu_handles: frozenset[PageHandle] = frozenset()
        self._accounting_revision = 0
        self._workflow_charge_cache_revision = -1
        self._workflow_charge_cache: dict[str, float] = {}
        self._mutation_journal: deque[_PageIndexMutation] = deque(maxlen=65_536)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def topology_revision(self) -> int:
        return self._topology_revision

    def _touch(
        self,
        *,
        topology: bool = False,
        handles: set[PageHandle] | tuple[PageHandle, ...] = (),
        context_ids: set[str] | tuple[str, ...] = (),
        components: set[str] | tuple[str, ...] = (),
    ) -> None:
        previous_revision = self._revision
        self._revision += 1
        if topology:
            self._topology_revision += 1
        if set(components).intersection(
            {"topology", "residency", "lock", "reader", "semantic_pin", "transfer"}
        ):
            self._physical_state_revision += 1
        if set(components).intersection({"residency", "owner", "context"}):
            self._accounting_revision += 1
        self._mutation_journal.append(
            _PageIndexMutation(
                from_revision=previous_revision,
                revision=self._revision,
                topology_revision=self._topology_revision,
                handles=frozenset(handles),
                context_ids=frozenset(context_ids),
                components=frozenset(components),
            )
        )

    def _mutations_since(
        self, revision: int
    ) -> tuple[_PageIndexMutation, ...]:
        reverse_mutations: list[_PageIndexMutation] = []
        for item in reversed(self._mutation_journal):
            if item.revision <= revision:
                break
            reverse_mutations.append(item)
        return tuple(reversed(reverse_mutations))

    def changes_since(self, revision: int) -> PageIndexChangeSet:
        if revision < 0 or revision > self._revision:
            raise ValueError("page-index revision is outside the valid range")
        if revision == self._revision:
            return PageIndexChangeSet(
                revision,
                revision,
                self._topology_revision,
                frozenset(),
                frozenset(),
                frozenset(),
            )
        mutations = self._mutations_since(revision)
        covered_revision = revision
        full_rebuild = not mutations
        for item in mutations:
            if item.from_revision > covered_revision:
                full_rebuild = True
                break
            covered_revision = max(covered_revision, item.revision)
        if covered_revision != self._revision:
            full_rebuild = True
        return PageIndexChangeSet(
            revision,
            self._revision,
            self._topology_revision,
            frozenset(
                handle for item in mutations for handle in item.handles
            ),
            frozenset(
                context_id
                for item in mutations
                for context_id in item.context_ids
            ),
            frozenset(
                component
                for item in mutations
                for component in item.components
            ),
            full_rebuild_required=full_rebuild,
        )

    def replica_delta_since(self, revision: int) -> PageIndexReplicaDelta:
        """Copy changed records without exposing mutable live page objects."""

        changes = self.changes_since(revision)
        full_rebuild = revision == 0 or changes.full_rebuild_required
        handles = (
            frozenset(self.pages)
            if full_rebuild
            else changes.handles
        )
        full_page_handles = set(handles) if full_rebuild else set()
        if not full_rebuild:
            for mutation in self._mutations_since(revision):
                if mutation.components.intersection({"topology", "owner"}):
                    full_page_handles.update(mutation.handles)
        state_page_handles = set(handles) - full_page_handles
        context_ids = (
            frozenset(self._context_epoch)
            if full_rebuild
            else frozenset(
                context_id
                for item in self._mutations_since(revision)
                if item.components.intersection({"owner", "context"})
                for context_id in item.context_ids
            )
        )
        return PageIndexReplicaDelta(
            from_revision=revision,
            to_revision=self._revision,
            topology_revision=self._topology_revision,
            pages=tuple(
                self._page_replica(self.pages[handle])
                for handle in sorted(full_page_handles)
                if handle in self.pages
            ),
            page_states=tuple(
                self._page_state_replica(self.pages[handle])
                for handle in sorted(state_page_handles)
                if handle in self.pages
            ),
            contexts=tuple(
                ContextPageReplica(
                    context_id=context_id,
                    workflow_id=self._context_workflow[context_id],
                    epoch=self._context_epoch[context_id],
                    handles=tuple(sorted(self._context_pages.get(context_id, ()))),
                )
                for context_id in sorted(context_ids)
                if context_id in self._context_epoch
            ),
            changed_handles=handles,
            changed_context_ids=context_ids,
            components=changes.components,
            full_rebuild_required=full_rebuild,
        )

    def apply_replica_delta(
        self,
        delta: PageIndexReplicaDelta,
        *,
        full_validation: bool = True,
    ) -> None:
        """Apply a safe-point delta to a worker-owned mirror.

        This method is intentionally unavailable to the runtime data plane. It
        replaces metadata only and never initiates allocation or transfer.
        """

        if delta.from_revision != self._revision:
            raise PageIndexError(
                "page replica revision gap: "
                f"{delta.from_revision} != {self._revision}"
            )
        if delta.to_revision < delta.from_revision:
            raise PageIndexError("page replica revision cannot move backwards")
        if delta.full_rebuild_required:
            self.pages.clear()
            self._latest_generation.clear()
            self._context_pages.clear()
            self._context_epoch.clear()
            self._context_workflow.clear()
            self._mutation_journal.clear()
            self._workflow_charge_cache_revision = -1
            self._workflow_charge_cache.clear()

        for replica in delta.pages:
            page = self.pages.get(replica.handle)
            if page is None:
                page = PhysicalPageRecord(
                    handle=replica.handle,
                    size_bytes=replica.size_bytes,
                    residency=replica.residency,
                )
                self.pages[replica.handle] = page
            page.size_bytes = replica.size_bytes
            page.residency = replica.residency
            page.radix_depth = replica.radix_depth
            page.parent = replica.parent
            page.children = set(replica.children)
            page.owner_contexts = dict(replica.owner_contexts)
            page.engine_lock_ref = replica.engine_lock_ref
            page.semantic_pin_contexts = set(replica.semantic_pin_contexts)
            page.active_reader_count = replica.active_reader_count
            page.sealed = replica.sealed
            page.transfer_direction = replica.transfer_direction
            page.last_access_ms = replica.last_access_ms
            self._latest_generation[replica.handle.page_id] = max(
                replica.handle.allocation_generation,
                self._latest_generation.get(replica.handle.page_id, -1),
            )
        for replica in delta.page_states:
            page = self.pages.get(replica.handle)
            if page is None:
                raise PageIndexError(
                    f"physical-state patch references unknown page: {replica.handle}"
                )
            page.residency = replica.residency
            page.engine_lock_ref = replica.engine_lock_ref
            page.semantic_pin_contexts = set(replica.semantic_pin_contexts)
            page.active_reader_count = replica.active_reader_count
            page.transfer_direction = replica.transfer_direction
            page.last_access_ms = replica.last_access_ms
        for context in delta.contexts:
            self._context_epoch[context.context_id] = context.epoch
            self._context_workflow[context.context_id] = context.workflow_id
            self._context_pages[context.context_id] = set(context.handles)

        previous_revision = self._revision
        self._revision = delta.to_revision
        self._topology_revision = delta.topology_revision
        if delta.full_rebuild_required or delta.components.intersection(
            {"topology", "residency", "lock", "reader", "semantic_pin", "transfer"}
        ):
            self._physical_state_revision += 1
        if delta.full_rebuild_required or delta.components.intersection(
            {"residency", "owner", "context"}
        ):
            self._accounting_revision += 1
        if delta.to_revision > previous_revision:
            # One worker publication may cover many source revisions. Recording
            # the covered interval avoids manufacturing thousands of empty
            # journal entries while retaining gap detection for later readers.
            self._mutation_journal.append(
                _PageIndexMutation(
                    from_revision=previous_revision,
                    revision=delta.to_revision,
                    topology_revision=delta.topology_revision,
                    handles=delta.changed_handles,
                    context_ids=delta.changed_context_ids,
                    components=delta.components,
                )
            )
        if full_validation or delta.full_rebuild_required:
            self.assert_consistent()
        else:
            self._assert_delta_consistent(delta)

    @staticmethod
    def _page_replica(page: PhysicalPageRecord) -> PhysicalPageReplica:
        return PhysicalPageReplica(
            handle=page.handle,
            size_bytes=page.size_bytes,
            residency=page.residency,
            radix_depth=page.radix_depth,
            parent=page.parent,
            children=tuple(sorted(page.children)),
            owner_contexts=tuple(sorted(page.owner_contexts.items())),
            engine_lock_ref=page.engine_lock_ref,
            semantic_pin_contexts=tuple(sorted(page.semantic_pin_contexts)),
            active_reader_count=page.active_reader_count,
            sealed=page.sealed,
            transfer_direction=page.transfer_direction,
            last_access_ms=page.last_access_ms,
        )

    @staticmethod
    def _page_state_replica(page: PhysicalPageRecord) -> PhysicalPageStateReplica:
        return PhysicalPageStateReplica(
            handle=page.handle,
            residency=page.residency,
            engine_lock_ref=page.engine_lock_ref,
            semantic_pin_contexts=tuple(sorted(page.semantic_pin_contexts)),
            active_reader_count=page.active_reader_count,
            transfer_direction=page.transfer_direction,
            last_access_ms=page.last_access_ms,
        )

    def register_context(self, context_id: str, workflow_id: str, epoch: int) -> None:
        if not context_id or not workflow_id:
            raise ValueError("context_id and workflow_id must be non-empty")
        old_epoch = self._context_epoch.get(context_id)
        if old_epoch is not None and epoch < old_epoch:
            raise PageIndexError(
                f"stale context epoch for {context_id}: {epoch} < {old_epoch}"
            )
        old_workflow = self._context_workflow.get(context_id)
        if old_workflow is not None and old_workflow != workflow_id:
            raise PageIndexError("a context cannot move across root workflows")
        self._context_epoch[context_id] = epoch
        self._context_workflow[context_id] = workflow_id
        self._context_pages.setdefault(context_id, set())
        if old_epoch != epoch or old_workflow is None:
            self._touch(
                context_ids={context_id}, components={"context"}
            )

    def register_page(
        self,
        handle: PageHandle,
        *,
        size_bytes: int,
        residency: PhysicalResidency = PhysicalResidency.GPU_ONLY,
        radix_depth: int = 0,
        parent: PageHandle | None = None,
        sealed: bool = True,
        last_access_ms: float = 0.0,
    ) -> PhysicalPageRecord:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        latest = self._latest_generation.get(handle.page_id)
        if latest is not None and handle.allocation_generation < latest:
            raise PageIndexError(f"stale page handle: {handle}")
        if latest is not None and handle.allocation_generation > latest:
            old = PageHandle(handle.page_id, latest)
            old_record = self.pages.get(old)
            if old_record is not None and old_record.residency != PhysicalResidency.DEAD:
                raise PageIndexError(
                    f"page id {handle.page_id} reused before generation {latest} was freed"
                )
        if handle in self.pages:
            raise PageIndexError(f"page already registered: {handle}")
        if parent is not None:
            self.require_page(parent)
        page = PhysicalPageRecord(
            handle=handle,
            size_bytes=size_bytes,
            residency=residency,
            radix_depth=radix_depth,
            parent=parent,
            sealed=sealed,
            last_access_ms=last_access_ms,
        )
        self.pages[handle] = page
        self._latest_generation[handle.page_id] = handle.allocation_generation
        if parent is not None:
            self.pages[parent].children.add(handle)
        dirty_handles = {handle}
        if parent is not None:
            dirty_handles.add(parent)
        self._touch(
            topology=True,
            handles=dirty_handles,
            components={"topology", "residency"},
        )
        return page

    def bind_pages(
        self,
        context_id: str,
        context_epoch: int,
        handles: set[PageHandle] | list[PageHandle] | tuple[PageHandle, ...],
        *,
        replace: bool = False,
    ) -> None:
        self.validate_context_epoch(context_id, context_epoch)
        resolved = {self.require_page(handle).handle for handle in handles}
        before = set(self._context_pages.get(context_id, set()))
        if replace:
            self.unbind_context(context_id)
        for handle in resolved:
            self.pages[handle].owner_contexts[context_id] = context_epoch
        self._context_pages.setdefault(context_id, set()).update(resolved)
        if not replace and not resolved.issubset(before):
            self._touch(
                handles=resolved | before,
                context_ids={context_id},
                components={"owner"},
            )
        elif replace and resolved:
            self._touch(
                handles=resolved,
                context_ids={context_id},
                components={"owner"},
            )

    def unbind_context(self, context_id: str) -> None:
        handles = tuple(self._context_pages.get(context_id, set()))
        removed_semantic_pin = False
        for handle in handles:
            page = self.pages.get(handle)
            if page is not None:
                page.owner_contexts.pop(context_id, None)
                removed_semantic_pin = (
                    context_id in page.semantic_pin_contexts
                    or removed_semantic_pin
                )
                page.semantic_pin_contexts.discard(context_id)
        self._context_pages[context_id] = set()
        if handles:
            self._touch(
                handles=set(handles),
                context_ids={context_id},
                components=(
                    {"owner", "semantic_pin"}
                    if removed_semantic_pin
                    else {"owner"}
                ),
            )

    def free_page(self, handle: PageHandle) -> None:
        page = self.require_page(handle)
        if page.engine_lock_ref or page.active_reader_count or not page.transfer_idle:
            raise PageIndexError(f"cannot free active page {handle}")
        affected_contexts = set(page.owner_contexts)
        affected_handles = {handle, *page.children}
        if page.parent is not None:
            affected_handles.add(page.parent)
        for context_id in tuple(page.owner_contexts):
            self._context_pages.get(context_id, set()).discard(handle)
        page.owner_contexts.clear()
        page.semantic_pin_contexts.clear()
        page.residency = PhysicalResidency.DEAD
        if page.parent is not None and page.parent in self.pages:
            self.pages[page.parent].children.discard(handle)
        for child_handle in page.children:
            child = self.pages.get(child_handle)
            if child is not None:
                child.parent = None
        page.children.clear()
        self._touch(
            topology=True,
            handles=affected_handles,
            context_ids=affected_contexts,
            components={"topology", "owner", "residency"},
        )

    def invalidate_page(self, handle: PageHandle) -> None:
        """Invalidate an extent after an authoritative Radix split/delete event.

        Engine locks may legitimately be copied during a Radix split, so unlike
        policy-driven ``free_page`` this operation does not reject lock refs.
        An in-flight transfer is still forbidden because changing its extent
        would make the transfer ACK ambiguous.
        """

        page = self.require_page(handle)
        if not page.transfer_idle:
            raise PageIndexError(f"cannot mutate in-flight page extent {handle}")
        affected_contexts = set(page.owner_contexts)
        affected_handles = {handle, *page.children}
        if page.parent is not None:
            affected_handles.add(page.parent)
        for context_id in tuple(page.owner_contexts):
            self._context_pages.get(context_id, set()).discard(handle)
        page.owner_contexts.clear()
        page.semantic_pin_contexts.clear()
        page.residency = PhysicalResidency.DEAD
        if page.parent is not None and page.parent in self.pages:
            self.pages[page.parent].children.discard(handle)
        for child_handle in tuple(page.children):
            child = self.pages.get(child_handle)
            if child is not None and child.residency != PhysicalResidency.DEAD:
                child.parent = None
        page.children.clear()
        self._touch(
            topology=True,
            handles=affected_handles,
            context_ids=affected_contexts,
            components={"topology", "owner", "residency"},
        )

    def latest_handle(self, page_id: int) -> PageHandle | None:
        generation = self._latest_generation.get(page_id)
        if generation is None:
            return None
        return PageHandle(page_id, generation)

    def require_page(self, handle: PageHandle) -> PhysicalPageRecord:
        latest = self._latest_generation.get(handle.page_id)
        if latest is not None and latest != handle.allocation_generation:
            raise PageIndexError(
                f"stale page generation {handle}; latest is {latest}"
            )
        try:
            page = self.pages[handle]
        except KeyError as exc:
            raise PageIndexError(f"unknown page: {handle}") from exc
        if page.residency == PhysicalResidency.DEAD:
            raise PageIndexError(f"page is dead: {handle}")
        return page

    def set_parent(
        self, handle: PageHandle, parent: PageHandle | None
    ) -> None:
        """Update a Radix edge while preserving both sides of the mirror."""

        page = self.require_page(handle)
        if parent == handle:
            raise PageIndexError("a page cannot be its own parent")
        if parent is not None:
            self.require_page(parent)
            ancestor = parent
            seen: set[PageHandle] = set()
            while ancestor is not None:
                if ancestor == handle:
                    raise PageIndexError("Radix parent update would create a cycle")
                if ancestor in seen:
                    raise PageIndexError("existing Radix parent cycle detected")
                seen.add(ancestor)
                ancestor_page = self.require_page(ancestor)
                ancestor = ancestor_page.parent
        if page.parent == parent:
            if parent is not None and handle not in self.pages[parent].children:
                self.pages[parent].children.add(handle)
                self._touch(
                    topology=True,
                    handles={handle, parent},
                    components={"topology"},
                )
            return
        previous_parent = page.parent
        if page.parent is not None and page.parent in self.pages:
            self.pages[page.parent].children.discard(handle)
        page.parent = parent
        if parent is not None:
            self.pages[parent].children.add(handle)
        dirty_handles = {handle}
        if previous_parent is not None:
            dirty_handles.add(previous_parent)
        if parent is not None:
            dirty_handles.add(parent)
        self._touch(
            topology=True,
            handles=dirty_handles,
            components={"topology"},
        )

    def context_pages(self, context_id: str) -> list[PhysicalPageRecord]:
        return [
            self.pages[handle]
            for handle in sorted(self._context_pages.get(context_id, set()))
            if self.pages[handle].residency != PhysicalResidency.DEAD
        ]

    def context_epoch(self, context_id: str) -> int:
        try:
            return self._context_epoch[context_id]
        except KeyError as exc:
            raise PageIndexError(f"unknown context: {context_id}") from exc

    def has_context(self, context_id: str) -> bool:
        return context_id in self._context_epoch

    def context_workflow(self, context_id: str) -> str:
        try:
            return self._context_workflow[context_id]
        except KeyError as exc:
            raise PageIndexError(f"unknown context: {context_id}") from exc

    def validate_context_epoch(self, context_id: str, epoch: int) -> None:
        current = self.context_epoch(context_id)
        if current != epoch:
            raise PageIndexError(
                f"context epoch mismatch for {context_id}: {epoch} != {current}"
            )

    def update_context_epoch(self, context_id: str, epoch: int) -> None:
        current = self.context_epoch(context_id)
        if epoch < current:
            raise PageIndexError("context epoch cannot move backwards")
        self._context_epoch[context_id] = epoch
        for handle in self._context_pages.get(context_id, set()):
            self.pages[handle].owner_contexts[context_id] = epoch
        if epoch != current:
            self._touch(
                handles=set(self._context_pages.get(context_id, set())),
                context_ids={context_id},
                components={"context", "owner"},
            )

    def set_engine_lock(self, handle: PageHandle, value: int) -> None:
        if value < 0:
            raise ValueError("engine lock must be non-negative")
        page = self.require_page(handle)
        if page.engine_lock_ref != value:
            page.engine_lock_ref = value
            self._touch(handles={handle}, components={"lock"})

    def set_active_readers(self, handle: PageHandle, value: int) -> None:
        if value < 0:
            raise ValueError("active readers must be non-negative")
        page = self.require_page(handle)
        if page.active_reader_count != value:
            page.active_reader_count = value
            self._touch(handles={handle}, components={"reader"})

    def pin_context(self, context_id: str) -> None:
        changed = False
        for page in self.context_pages(context_id):
            if context_id not in page.semantic_pin_contexts:
                page.semantic_pin_contexts.add(context_id)
                changed = True
        if changed:
            self._touch(
                handles={page.handle for page in self.context_pages(context_id)},
                context_ids={context_id},
                components={"semantic_pin"},
            )

    def unpin_context(self, context_id: str) -> None:
        changed = False
        for page in self.context_pages(context_id):
            if context_id in page.semantic_pin_contexts:
                page.semantic_pin_contexts.discard(context_id)
                changed = True
        if changed:
            self._touch(
                handles={page.handle for page in self.context_pages(context_id)},
                context_ids={context_id},
                components={"semantic_pin"},
            )

    def update_runtime_state(
        self,
        handle: PageHandle,
        *,
        residency: PhysicalResidency | None = None,
        radix_depth: int | None = None,
        engine_lock_ref: int | None = None,
        last_access_ms: float | None = None,
    ) -> None:
        """Mirror authoritative engine fields with one revision update."""

        page = self.require_page(handle)
        if radix_depth is not None and radix_depth < 0:
            raise ValueError("radix_depth must be non-negative")
        if engine_lock_ref is not None and engine_lock_ref < 0:
            raise ValueError("engine_lock_ref must be non-negative")
        if last_access_ms is not None and last_access_ms < 0:
            raise ValueError("last_access_ms must be non-negative")
        changed = False
        topology_changed = False
        for field_name, value in (
            ("residency", residency),
            ("radix_depth", radix_depth),
            ("engine_lock_ref", engine_lock_ref),
            ("last_access_ms", last_access_ms),
        ):
            if value is not None and getattr(page, field_name) != value:
                setattr(page, field_name, value)
                changed = True
                topology_changed = topology_changed or field_name == "radix_depth"
        if changed:
            changed_components = {
                field_name
                for field_name, value in (
                    ("residency", residency),
                    ("topology", radix_depth),
                    ("lock", engine_lock_ref),
                    ("access", last_access_ms),
                )
                if value is not None
            }
            self._touch(
                topology=topology_changed,
                handles={handle},
                context_ids=set(page.owner_contexts),
                components=changed_components,
            )

    def begin_transfer(
        self, handle: PageHandle, direction: TransferDirection
    ) -> None:
        page = self.require_page(handle)
        if not page.transfer_idle:
            raise PageIndexError(f"page already in transfer: {handle}")
        if direction == TransferDirection.D2H:
            if page.residency != PhysicalResidency.GPU_ONLY:
                raise PageIndexError(f"D2H requires GPU_ONLY, got {page.residency}")
            page.residency = PhysicalResidency.MIRRORING
        else:
            if page.residency != PhysicalResidency.CPU_ONLY:
                raise PageIndexError(f"H2D requires CPU_ONLY, got {page.residency}")
            page.residency = PhysicalResidency.PREFETCHING
        page.transfer_direction = direction
        self._touch(
            handles={handle},
            context_ids=set(page.owner_contexts),
            components={"residency", "transfer"},
        )

    def complete_transfer(
        self,
        handle: PageHandle,
        direction: TransferDirection,
        *,
        keep_gpu: bool = True,
    ) -> None:
        page = self.require_page(handle)
        if page.transfer_direction != direction:
            raise PageIndexError(f"unexpected transfer ACK for {handle}")
        if direction == TransferDirection.D2H:
            page.residency = (
                PhysicalResidency.DUAL_CLEAN
                if keep_gpu
                else PhysicalResidency.CPU_ONLY
            )
        else:
            page.residency = PhysicalResidency.DUAL_CLEAN
        page.transfer_direction = None
        self._touch(
            handles={handle},
            context_ids=set(page.owner_contexts),
            components={"residency", "transfer"},
        )

    def abort_transfer(self, handle: PageHandle) -> None:
        page = self.require_page(handle)
        if page.transfer_direction == TransferDirection.D2H:
            page.residency = PhysicalResidency.GPU_ONLY
        elif page.transfer_direction == TransferDirection.H2D:
            page.residency = PhysicalResidency.CPU_ONLY
        else:
            raise PageIndexError(f"page is not in transfer: {handle}")
        page.transfer_direction = None
        self._touch(
            handles={handle},
            context_ids=set(page.owner_contexts),
            components={"residency", "transfer"},
        )

    def commit_cpu(self, handle: PageHandle) -> None:
        page = self.require_page(handle)
        if page.residency == PhysicalResidency.CPU_ONLY:
            return
        if page.residency != PhysicalResidency.DUAL_CLEAN:
            raise PageIndexError(
                "commit requires DUAL_CLEAN or the CPU_ONLY postcondition, "
                f"got {page.residency} for {handle}"
            )
        page.residency = PhysicalResidency.CPU_ONLY
        self._touch(
            handles={handle},
            context_ids=set(page.owner_contexts),
            components={"residency"},
        )

    def drop_page(self, handle: PageHandle) -> None:
        page = self.require_page(handle)
        if not page.transfer_idle:
            raise PageIndexError("cannot drop an in-flight page")
        if page.residency == PhysicalResidency.DUAL_CLEAN:
            page.residency = PhysicalResidency.CPU_ONLY
            self._touch(
                handles={handle},
                context_ids=set(page.owner_contexts),
                components={"residency"},
            )
        elif page.residency == PhysicalResidency.GPU_ONLY:
            self.free_page(handle)
        elif page.residency == PhysicalResidency.CPU_ONLY:
            self.free_page(handle)
        else:
            raise PageIndexError(f"cannot drop page in state {page.residency}")

    def drop_host_copy(self, handle: PageHandle) -> None:
        """Mirror an authoritative Host-only replica release."""

        page = self.require_page(handle)
        if not page.transfer_idle:
            raise PageIndexError("cannot drop an in-flight host copy")
        if page.residency == PhysicalResidency.DUAL_CLEAN:
            page.residency = PhysicalResidency.GPU_ONLY
            self._touch(
                handles={handle},
                context_ids=set(page.owner_contexts),
                components={"residency"},
            )
        elif page.residency == PhysicalResidency.CPU_ONLY:
            self.free_page(handle)
        else:
            raise PageIndexError(
                f"host-copy drop requires DUAL_CLEAN or CPU_ONLY, got {page.residency}"
            )

    @property
    def gpu_bytes(self) -> int:
        return self.physical_kv_state_breakdown().gpu_bytes

    @property
    def cpu_bytes(self) -> int:
        return self.physical_kv_state_breakdown().cpu_bytes

    def physical_kv_state_breakdown(self) -> PhysicalKvStateBreakdown:
        """Return cached closure-aware physical KV diagnostics.

        A GPU extent is ``migratable`` when its local transfer gates and every
        GPU-resident descendant's gates are open. A locally eligible extent is
        ``closure_blocked`` when at least one GPU descendant prevents that
        closed subtree from being reclaimed. Agent urgency and owner state are
        deliberately excluded; these counters report physical facts only.
        """

        if self._physical_breakdown_revision == self._physical_state_revision:
            return self._physical_breakdown

        (
            breakdown,
            engine_locked_pages,
            migratable_handles,
        ) = self._calculate_physical_kv_state()
        self._physical_breakdown = breakdown
        self._engine_locked_pages = engine_locked_pages
        self._migratable_gpu_handles = migratable_handles
        self._physical_breakdown_revision = self._physical_state_revision
        return self._physical_breakdown

    def preview_engine_lock_release(
        self,
        engine_lock_ref_overrides: Mapping[PageHandle, int],
    ) -> PhysicalKvUnlockProjection:
        """Project closure feasibility without mutating live page records."""

        overrides = dict(engine_lock_ref_overrides)
        for handle, projected_ref in overrides.items():
            page = self.require_page(handle)
            if projected_ref < 0 or projected_ref > page.engine_lock_ref:
                raise ValueError(
                    "projected engine lock ref must be within the current ref"
                )
        baseline = self.physical_kv_state_breakdown()
        projected, _, projected_migratable = self._calculate_physical_kv_state(
            engine_lock_ref_overrides=overrides
        )
        zeroed = tuple(
            sorted(
                handle
                for handle, projected_ref in overrides.items()
                if self.pages[handle].engine_lock_ref > 0 and projected_ref == 0
            )
        )
        newly_migratable = tuple(
            sorted(projected_migratable - self._migratable_gpu_handles)
        )
        return PhysicalKvUnlockProjection(
            page_revision=self._revision,
            topology_revision=self._topology_revision,
            baseline=baseline,
            projected=projected,
            overridden_handles=tuple(sorted(overrides)),
            lock_ref_zeroed_handles=zeroed,
            lock_ref_zeroed_bytes=sum(
                self.pages[handle].size_bytes for handle in zeroed
            ),
            newly_migratable_handles=newly_migratable,
            newly_migratable_bytes=(
                projected.migratable_bytes - baseline.migratable_bytes
            ),
        )

    def _calculate_physical_kv_state(
        self,
        *,
        engine_lock_ref_overrides: Mapping[PageHandle, int] | None = None,
    ) -> tuple[
        PhysicalKvStateBreakdown,
        tuple[PhysicalPageRecord, ...],
        frozenset[PageHandle],
    ]:
        overrides = engine_lock_ref_overrides or {}
        live_pages = {
            handle: page
            for handle, page in self.pages.items()
            if page.residency != PhysicalResidency.DEAD
        }
        subtree_gpu_eligible: dict[PageHandle, bool] = {}
        gpu_bytes = 0
        cpu_bytes = 0
        engine_locked_bytes = 0
        closure_blocked_bytes = 0
        migratable_bytes = 0
        dual_resident_bytes = 0
        engine_locked_pages: list[PhysicalPageRecord] = []
        migratable_handles: set[PageHandle] = set()

        for page in sorted(
            live_pages.values(),
            key=lambda item: (-item.radix_depth, item.handle),
        ):
            child_subtrees_eligible = all(
                subtree_gpu_eligible.get(child_handle, False)
                for child_handle in page.children
                if child_handle in live_pages
            )
            engine_lock_ref = overrides.get(page.handle, page.engine_lock_ref)
            locally_eligible = (
                page.sealed
                and engine_lock_ref == 0
                and page.active_reader_count == 0
                and not page.semantic_pin_contexts
                and page.transfer_idle
            )
            subtree_gpu_eligible[page.handle] = child_subtrees_eligible and (
                locally_eligible if page.gpu_resident else True
            )

            if page.cpu_resident:
                cpu_bytes += page.size_bytes
            if not page.gpu_resident:
                continue
            gpu_bytes += page.size_bytes
            if engine_lock_ref > 0 or page.active_reader_count > 0:
                engine_locked_bytes += page.size_bytes
            if engine_lock_ref > 0:
                engine_locked_pages.append(page)
            if page.residency == PhysicalResidency.DUAL_CLEAN:
                dual_resident_bytes += page.size_bytes
            if subtree_gpu_eligible[page.handle]:
                migratable_bytes += page.size_bytes
                migratable_handles.add(page.handle)
            elif locally_eligible:
                closure_blocked_bytes += page.size_bytes

        return (
            PhysicalKvStateBreakdown(
            gpu_bytes=gpu_bytes,
            cpu_bytes=cpu_bytes,
            engine_locked_bytes=engine_locked_bytes,
            closure_blocked_bytes=closure_blocked_bytes,
            migratable_bytes=migratable_bytes,
            dual_resident_bytes=dual_resident_bytes,
            ),
            tuple(engine_locked_pages),
            frozenset(migratable_handles),
        )

    def engine_locked_gpu_pages(self) -> tuple[PhysicalPageRecord, ...]:
        """Return revision-cached GPU pages protected by engine lock refs."""

        self.physical_kv_state_breakdown()
        return self._engine_locked_pages

    def workflow_gpu_charges(self) -> dict[str, float]:
        if self._workflow_charge_cache_revision == self._accounting_revision:
            return dict(self._workflow_charge_cache)
        charges: dict[str, float] = {}
        for page in self.pages.values():
            if not page.gpu_resident:
                continue
            workflows = {
                self._context_workflow[context_id]
                for context_id in page.owner_contexts
                if context_id in self._context_workflow
            }
            if not workflows:
                continue
            share = page.size_bytes / len(workflows)
            for workflow_id in workflows:
                charges[workflow_id] = charges.get(workflow_id, 0.0) + share
        self._workflow_charge_cache = charges
        self._workflow_charge_cache_revision = self._accounting_revision
        return dict(charges)

    def _assert_delta_consistent(self, delta: PageIndexReplicaDelta) -> None:
        """Validate only records whose replacement can change an invariant."""

        context_ids = set(delta.changed_context_ids)
        handles = set(delta.changed_handles)
        for handle in tuple(handles):
            page = self.pages.get(handle)
            if page is None:
                continue
            context_ids.update(page.owner_contexts)
            if page.parent is not None:
                handles.add(page.parent)
            handles.update(page.children)

        for context_id in context_ids:
            if context_id not in self._context_epoch:
                raise AssertionError("page owner references an unknown context")
            for handle in self._context_pages.get(context_id, set()):
                page = self.pages.get(handle)
                if page is None:
                    raise AssertionError("context references a missing page")
                if context_id not in page.owner_contexts:
                    raise AssertionError("context->page mapping lacks reverse owner")
                if page.owner_contexts[context_id] != self._context_epoch[context_id]:
                    raise AssertionError("page owner epoch differs from context epoch")

        for handle in handles:
            page = self.pages.get(handle)
            if page is None:
                continue
            for context_id in page.owner_contexts:
                if page.handle not in self._context_pages.get(context_id, set()):
                    raise AssertionError("page owner lacks reverse context mapping")
            if page.engine_lock_ref < 0 or page.active_reader_count < 0:
                raise AssertionError("negative page reference count")
            if page.residency == PhysicalResidency.DEAD:
                if page.children:
                    raise AssertionError("dead page retains Radix children")
                continue
            if page.parent is not None:
                parent = self.pages.get(page.parent)
                if parent is None or parent.residency == PhysicalResidency.DEAD:
                    raise AssertionError("live page has a missing or dead Radix parent")
                if page.handle not in parent.children:
                    raise AssertionError("Radix child lacks reverse parent edge")
            for child_handle in page.children:
                child = self.pages.get(child_handle)
                if child is None or child.residency == PhysicalResidency.DEAD:
                    raise AssertionError("Radix parent retains a missing or dead child")
                if child.parent != page.handle:
                    raise AssertionError("Radix parent/child edge diverged")

    def assert_consistent(self) -> None:
        for context_id, handles in self._context_pages.items():
            for handle in handles:
                page = self.pages[handle]
                if context_id not in page.owner_contexts:
                    raise AssertionError("context->page mapping lacks reverse owner")
                if page.owner_contexts[context_id] != self._context_epoch[context_id]:
                    raise AssertionError("page owner epoch differs from context epoch")
        for page in self.pages.values():
            for context_id in page.owner_contexts:
                if page.handle not in self._context_pages.get(context_id, set()):
                    raise AssertionError("page owner lacks reverse context mapping")
            if page.engine_lock_ref < 0 or page.active_reader_count < 0:
                raise AssertionError("negative page reference count")
            if page.residency == PhysicalResidency.DEAD:
                if page.children:
                    raise AssertionError("dead page retains Radix children")
                continue
            if page.parent is not None:
                parent = self.pages.get(page.parent)
                if parent is None or parent.residency == PhysicalResidency.DEAD:
                    raise AssertionError("live page has a missing or dead Radix parent")
                if page.handle not in parent.children:
                    raise AssertionError("Radix child lacks reverse parent edge")
            for child_handle in page.children:
                child = self.pages.get(child_handle)
                if child is None or child.residency == PhysicalResidency.DEAD:
                    raise AssertionError("Radix parent retains a missing or dead child")
                if child.parent != page.handle:
                    raise AssertionError("Radix parent/child edge diverged")
