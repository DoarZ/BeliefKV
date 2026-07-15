from __future__ import annotations

from dataclasses import dataclass, field

from beliefkv.runtime.protocol import (
    PageHandle,
    PhysicalResidency,
    TransferDirection,
)


class PageIndexError(RuntimeError):
    pass


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
        if replace:
            self.unbind_context(context_id)
        for handle in resolved:
            self.pages[handle].owner_contexts[context_id] = context_epoch
        self._context_pages.setdefault(context_id, set()).update(resolved)

    def unbind_context(self, context_id: str) -> None:
        for handle in tuple(self._context_pages.get(context_id, set())):
            page = self.pages.get(handle)
            if page is not None:
                page.owner_contexts.pop(context_id, None)
                page.semantic_pin_contexts.discard(context_id)
        self._context_pages[context_id] = set()

    def free_page(self, handle: PageHandle) -> None:
        page = self.require_page(handle)
        if page.engine_lock_ref or page.active_reader_count or not page.transfer_idle:
            raise PageIndexError(f"cannot free active page {handle}")
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
            if parent is not None:
                self.pages[parent].children.add(handle)
            return
        if page.parent is not None and page.parent in self.pages:
            self.pages[page.parent].children.discard(handle)
        page.parent = parent
        if parent is not None:
            self.pages[parent].children.add(handle)

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

    def set_engine_lock(self, handle: PageHandle, value: int) -> None:
        if value < 0:
            raise ValueError("engine lock must be non-negative")
        self.require_page(handle).engine_lock_ref = value

    def set_active_readers(self, handle: PageHandle, value: int) -> None:
        if value < 0:
            raise ValueError("active readers must be non-negative")
        self.require_page(handle).active_reader_count = value

    def pin_context(self, context_id: str) -> None:
        for page in self.context_pages(context_id):
            page.semantic_pin_contexts.add(context_id)

    def unpin_context(self, context_id: str) -> None:
        for page in self.context_pages(context_id):
            page.semantic_pin_contexts.discard(context_id)

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

    def abort_transfer(self, handle: PageHandle) -> None:
        page = self.require_page(handle)
        if page.transfer_direction == TransferDirection.D2H:
            page.residency = PhysicalResidency.GPU_ONLY
        elif page.transfer_direction == TransferDirection.H2D:
            page.residency = PhysicalResidency.CPU_ONLY
        else:
            raise PageIndexError(f"page is not in transfer: {handle}")
        page.transfer_direction = None

    def commit_cpu(self, handle: PageHandle) -> None:
        page = self.require_page(handle)
        if page.residency != PhysicalResidency.DUAL_CLEAN:
            raise PageIndexError("commit requires a clean CPU shadow")
        page.residency = PhysicalResidency.CPU_ONLY

    def drop_page(self, handle: PageHandle) -> None:
        page = self.require_page(handle)
        if not page.transfer_idle:
            raise PageIndexError("cannot drop an in-flight page")
        if page.residency == PhysicalResidency.DUAL_CLEAN:
            page.residency = PhysicalResidency.CPU_ONLY
        elif page.residency == PhysicalResidency.GPU_ONLY:
            self.free_page(handle)
        elif page.residency == PhysicalResidency.CPU_ONLY:
            self.free_page(handle)
        else:
            raise PageIndexError(f"cannot drop page in state {page.residency}")

    @property
    def gpu_bytes(self) -> int:
        return sum(page.size_bytes for page in self.pages.values() if page.gpu_resident)

    @property
    def cpu_bytes(self) -> int:
        return sum(page.size_bytes for page in self.pages.values() if page.cpu_resident)

    def workflow_gpu_charges(self) -> dict[str, float]:
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
        return charges

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
