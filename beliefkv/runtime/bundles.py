from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import blake2b

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.policy.leases import BundleLease, CausalLeaseProjector, LeaseKind
from beliefkv.runtime.page_index import PageOwnershipIndex, PhysicalPageRecord
from beliefkv.runtime.protocol import (
    CommandKind,
    PageHandle,
    PhysicalBundleIntent,
    PhysicalPageAction,
    PhysicalResidency,
    ResolvedPageAction,
    TransferBlocker,
    TransferBlockerCode,
)


class BundleScope(str, Enum):
    """Cross-context impact of the extents changed by one bundle action."""

    EXCLUSIVE_SUFFIX = "exclusive_suffix"
    SHARED_SUBTREE = "shared_subtree"


@dataclass(frozen=True)
class PhysicalBundle:
    bundle_id: str
    handles: tuple[PageHandle, ...]
    owner_context_ids: tuple[str, ...]
    scope: BundleScope
    exclusive_action_bytes: int
    cross_context_action_bytes: int
    foreign_owner_context_ids: tuple[str, ...]
    physical_unique_bytes: int
    gpu_bytes: int
    cpu_bytes: int
    marginal_reclaimable_bytes: int
    closure_bytes: int
    locked_bytes: int
    residency: str
    generation_fingerprint: str
    lease: BundleLease


@dataclass(frozen=True)
class PhysicalBundlePreview:
    command_kind: CommandKind
    context_id: str
    context_epoch: int
    bundle: PhysicalBundle
    page_actions: tuple[ResolvedPageAction, ...]
    blockers: tuple[TransferBlocker, ...]
    copy_bytes: int

    @property
    def eligible(self) -> bool:
        return bool(self.page_actions) and not self.blockers

    def intent(self) -> PhysicalBundleIntent:
        if not self.eligible:
            raise ValueError("a blocked physical bundle cannot become an intent")
        return PhysicalBundleIntent(
            bundle_id=self.bundle.bundle_id,
            closure_handles=self.bundle.handles,
            page_actions=self.page_actions,
            generation_fingerprint=self.bundle.generation_fingerprint,
            closure_bytes=self.bundle.closure_bytes,
            expected_reclaimable_bytes=self.bundle.marginal_reclaimable_bytes,
            locked_bytes=self.bundle.locked_bytes,
        )


@dataclass(frozen=True)
class BundlePreviewEvent:
    kind: str
    ts_ms: float
    fields: dict[str, object]


class PhysicalBundleBuilder:
    """Build immutable, closure-complete transfer candidates from physical facts."""

    def __init__(
        self,
        graph: RuntimeCausalContextGraph,
        page_index: PageOwnershipIndex,
        leases: CausalLeaseProjector | None = None,
    ) -> None:
        self.graph = graph
        self.page_index = page_index
        self.leases = leases or CausalLeaseProjector(graph)

    def previews_for_context(
        self,
        command_kind: CommandKind,
        context_id: str,
        context_epoch: int,
        *,
        now_ms: float,
        allow_ready_owners: bool = False,
        protected_context_id: str | None = None,
        host_available_bytes: int | None = None,
        device_available_bytes: int | None = None,
    ) -> tuple[PhysicalBundlePreview, ...]:
        context = self.graph.contexts.get(context_id)
        if (
            context is None
            or context.epoch != context_epoch
            or not self.page_index.has_context(context_id)
            or self.page_index.context_epoch(context_id) != context_epoch
        ):
            return ()
        if command_kind in {
            CommandKind.OFFLOAD_CONTEXT,
            CommandKind.SHADOW_CONTEXT,
        }:
            previews = self._offload_previews(
                command_kind,
                context_id,
                context_epoch,
                now_ms=now_ms,
                allow_ready_owners=allow_ready_owners,
                protected_context_id=protected_context_id,
                host_available_bytes=host_available_bytes,
            )
        elif command_kind == CommandKind.PREFETCH_CONTEXT:
            previews = self._prefetch_previews(
                context_id,
                context_epoch,
                now_ms=now_ms,
                device_available_bytes=device_available_bytes,
            )
        else:
            return ()
        return tuple(
            sorted(
                previews,
                key=lambda item: (
                    not item.eligible,
                    -item.bundle.marginal_reclaimable_bytes,
                    item.bundle.closure_bytes,
                    item.bundle.bundle_id,
                ),
            )
        )

    def find_intent_preview(
        self,
        command_kind: CommandKind,
        context_id: str,
        context_epoch: int,
        bundle_id: str,
        *,
        now_ms: float,
        allow_ready_owners: bool = False,
        protected_context_id: str | None = None,
        host_available_bytes: int | None = None,
        device_available_bytes: int | None = None,
    ) -> PhysicalBundlePreview | None:
        return next(
            (
                item
                for item in self.previews_for_context(
                    command_kind,
                    context_id,
                    context_epoch,
                    now_ms=now_ms,
                    allow_ready_owners=allow_ready_owners,
                    protected_context_id=protected_context_id,
                    host_available_bytes=host_available_bytes,
                    device_available_bytes=device_available_bytes,
                )
                if item.bundle.bundle_id == bundle_id
            ),
            None,
        )

    def _offload_previews(
        self,
        command_kind: CommandKind,
        context_id: str,
        context_epoch: int,
        *,
        now_ms: float,
        allow_ready_owners: bool,
        protected_context_id: str | None,
        host_available_bytes: int | None,
    ) -> list[PhysicalBundlePreview]:
        roots = [
            page
            for page in self.page_index.context_pages(context_id)
            if page.gpu_resident
        ]
        previews: list[PhysicalBundlePreview] = []
        seen_closures: set[tuple[PageHandle, ...]] = set()
        for root in sorted(roots, key=lambda page: (-page.radix_depth, page.handle)):
            closure, closure_blockers = self._gpu_descendant_closure(root)
            handles = tuple(sorted(closure))
            if not handles or handles in seen_closures:
                continue
            seen_closures.add(handles)
            blockers = list(closure_blockers)
            actions: list[ResolvedPageAction] = []
            blocked_handles: set[PageHandle] = {
                item.page_handle
                for item in blockers
                if item.page_handle is not None
            }
            for page in sorted(
                closure.values(),
                key=lambda item: (-item.radix_depth, item.handle),
            ):
                if page.residency == PhysicalResidency.DUAL_CLEAN:
                    if command_kind == CommandKind.OFFLOAD_CONTEXT:
                        actions.append(
                            ResolvedPageAction(
                                page.handle,
                                PhysicalPageAction.COMMIT_CPU,
                                page.size_bytes,
                            )
                        )
                elif page.residency == PhysicalResidency.GPU_ONLY:
                    actions.append(
                        ResolvedPageAction(
                            page.handle,
                            PhysicalPageAction.START_D2H,
                            page.size_bytes,
                        )
                    )
                page_blockers = self._page_blockers(page)
                page_blockers += self._owner_blockers(
                    page,
                    now_ms=now_ms,
                    allow_ready_owners=allow_ready_owners,
                    protected_context_id=protected_context_id,
                )
                blockers.extend(page_blockers)
                if page_blockers:
                    blocked_handles.add(page.handle)
                    continue
                ancestor = page.parent
                while ancestor is not None:
                    parent = self.page_index.pages.get(ancestor)
                    if parent is None or parent.residency == PhysicalResidency.DEAD:
                        blocker = TransferBlocker(
                            TransferBlockerCode.ANCESTOR_CLOSURE,
                            page.handle,
                            page.size_bytes,
                            "D2H closure has a missing ancestor",
                        )
                        blockers.append(blocker)
                        blocked_handles.add(page.handle)
                        break
                    if not parent.gpu_resident:
                        blocker = TransferBlocker(
                            TransferBlockerCode.ANCESTOR_CLOSURE,
                            page.handle,
                            page.size_bytes,
                            "D2H target has a non-resident ancestor",
                        )
                        blockers.append(blocker)
                        blocked_handles.add(page.handle)
                        break
                    ancestor = parent.parent
                if page.handle in blocked_handles:
                    continue
            copy_bytes = sum(
                item.size_bytes
                for item in actions
                if item.action == PhysicalPageAction.START_D2H
            )
            if (
                host_available_bytes is not None
                and copy_bytes > host_available_bytes
            ):
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.HOST_CAPACITY,
                        root.handle,
                        copy_bytes,
                        "D2H bundle exceeds current host availability",
                    )
                )
            blockers_tuple = self._deduplicate_blockers(blockers)
            previews.append(
                self._preview(
                    command_kind,
                    context_id,
                    context_epoch,
                    closure,
                    tuple(actions),
                    blockers_tuple,
                    blocked_handles,
                    now_ms=now_ms,
                )
            )
        return previews

    def _prefetch_previews(
        self,
        context_id: str,
        context_epoch: int,
        *,
        now_ms: float,
        device_available_bytes: int | None,
    ) -> list[PhysicalBundlePreview]:
        targets = [
            page
            for page in self.page_index.context_pages(context_id)
            if page.residency == PhysicalResidency.CPU_ONLY
        ]
        previews: list[PhysicalBundlePreview] = []
        seen_closures: set[tuple[PageHandle, ...]] = set()
        for target in sorted(targets, key=lambda page: (-page.radix_depth, page.handle)):
            closure: dict[PageHandle, PhysicalPageRecord] = {}
            blockers: list[TransferBlocker] = []
            node: PhysicalPageRecord | None = target
            seen: set[PageHandle] = set()
            while node is not None and not node.gpu_resident:
                if node.handle in seen:
                    blockers.append(
                        TransferBlocker(
                            TransferBlockerCode.EXTENT_MUTATED,
                            node.handle,
                            node.size_bytes,
                            "Radix ancestor cycle",
                        )
                    )
                    break
                seen.add(node.handle)
                closure[node.handle] = node
                if node.parent is None:
                    break
                parent = self.page_index.pages.get(node.parent)
                if parent is None or parent.residency == PhysicalResidency.DEAD:
                    blockers.append(
                        TransferBlocker(
                            TransferBlockerCode.ANCESTOR_CLOSURE,
                            node.handle,
                            node.size_bytes,
                            "H2D closure has a missing ancestor",
                        )
                    )
                    break
                node = parent
            if node is not None and node.gpu_resident:
                closure[node.handle] = node
            handles = tuple(sorted(closure))
            if not handles or handles in seen_closures:
                continue
            seen_closures.add(handles)
            actions: list[ResolvedPageAction] = []
            blocked_handles: set[PageHandle] = set()
            for page in sorted(
                closure.values(), key=lambda item: (item.radix_depth, item.handle)
            ):
                if page.gpu_resident:
                    continue
                if page.residency == PhysicalResidency.CPU_ONLY:
                    actions.append(
                        ResolvedPageAction(
                            page.handle,
                            PhysicalPageAction.START_H2D,
                            page.size_bytes,
                        )
                    )
                page_blockers = self._page_blockers(page)
                blockers.extend(page_blockers)
                if page_blockers:
                    blocked_handles.add(page.handle)
                    continue
                if page.residency != PhysicalResidency.CPU_ONLY:
                    blocker = TransferBlocker(
                        TransferBlockerCode.NODE_LOADING,
                        page.handle,
                        page.size_bytes,
                        "H2D closure extent is not CPU_ONLY",
                    )
                    blockers.append(blocker)
                    blocked_handles.add(page.handle)
                    continue
            h2d_bytes = sum(item.size_bytes for item in actions)
            if (
                device_available_bytes is not None
                and h2d_bytes > device_available_bytes
            ):
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.DEVICE_CAPACITY,
                        target.handle,
                        h2d_bytes,
                        "H2D bundle exceeds current device availability",
                    )
                )
            blockers_tuple = self._deduplicate_blockers(blockers)
            previews.append(
                self._preview(
                    CommandKind.PREFETCH_CONTEXT,
                    context_id,
                    context_epoch,
                    closure,
                    tuple(actions),
                    blockers_tuple,
                    blocked_handles,
                    now_ms=now_ms,
                )
            )
        return previews

    def _preview(
        self,
        command_kind: CommandKind,
        context_id: str,
        context_epoch: int,
        closure: dict[PageHandle, PhysicalPageRecord],
        actions: tuple[ResolvedPageAction, ...],
        blockers: tuple[TransferBlocker, ...],
        blocked_handles: set[PageHandle],
        *,
        now_ms: float,
    ) -> PhysicalBundlePreview:
        handles = tuple(sorted(closure))
        bundle_id = self._bundle_id(command_kind, handles)
        owner_context_ids = tuple(
            sorted(
                {
                    owner
                    for page in closure.values()
                    for owner in page.owner_contexts
                }
            )
        )
        (
            scope,
            exclusive_action_bytes,
            cross_context_action_bytes,
            foreign_owner_context_ids,
        ) = self._action_scope(context_id, closure, actions)
        lease = self.leases.bundle(bundle_id, owner_context_ids, now_ms=now_ms)
        closure_bytes = sum(item.size_bytes for item in actions)
        reclaimable = (
            closure_bytes
            if command_kind == CommandKind.OFFLOAD_CONTEXT and not blockers
            else 0
        )
        residencies = sorted({page.residency.value for page in closure.values()})
        bundle = PhysicalBundle(
            bundle_id=bundle_id,
            handles=handles,
            owner_context_ids=owner_context_ids,
            scope=scope,
            exclusive_action_bytes=exclusive_action_bytes,
            cross_context_action_bytes=cross_context_action_bytes,
            foreign_owner_context_ids=foreign_owner_context_ids,
            physical_unique_bytes=sum(
                page.size_bytes for page in closure.values()
            ),
            gpu_bytes=sum(page.size_bytes for page in closure.values() if page.gpu_resident),
            cpu_bytes=sum(page.size_bytes for page in closure.values() if page.cpu_resident),
            marginal_reclaimable_bytes=reclaimable,
            closure_bytes=closure_bytes,
            locked_bytes=sum(
                closure[handle].size_bytes
                for handle in blocked_handles
                if handle in closure
            ),
            residency=residencies[0] if len(residencies) == 1 else "mixed",
            generation_fingerprint=self._fingerprint(
                command_kind,
                closure,
                lease,
                blocker_scope=(
                    set(closure)
                    if command_kind
                    in {
                        CommandKind.OFFLOAD_CONTEXT,
                        CommandKind.SHADOW_CONTEXT,
                    }
                    else {
                        handle
                        for handle, page in closure.items()
                        if not page.gpu_resident
                    }
                ),
            ),
            lease=lease,
        )
        return PhysicalBundlePreview(
            command_kind=command_kind,
            context_id=context_id,
            context_epoch=context_epoch,
            bundle=bundle,
            page_actions=actions,
            blockers=blockers,
            copy_bytes=sum(
                item.size_bytes
                for item in actions
                if item.action
                in {PhysicalPageAction.START_D2H, PhysicalPageAction.START_H2D}
            ),
        )

    @staticmethod
    def _action_scope(
        context_id: str,
        closure: dict[PageHandle, PhysicalPageRecord],
        actions: tuple[ResolvedPageAction, ...],
    ) -> tuple[BundleScope, int, int, tuple[str, ...]]:
        exclusive_bytes = 0
        cross_context_bytes = 0
        foreign_owners: set[str] = set()
        for action in actions:
            page = closure[action.handle]
            page_foreign_owners = set(page.owner_contexts) - {context_id}
            if page_foreign_owners:
                cross_context_bytes += action.size_bytes
                foreign_owners.update(page_foreign_owners)
            else:
                exclusive_bytes += action.size_bytes
        scope = (
            BundleScope.SHARED_SUBTREE
            if foreign_owners
            else BundleScope.EXCLUSIVE_SUFFIX
        )
        return (
            scope,
            exclusive_bytes,
            cross_context_bytes,
            tuple(sorted(foreign_owners)),
        )

    def _gpu_descendant_closure(
        self, root: PhysicalPageRecord
    ) -> tuple[dict[PageHandle, PhysicalPageRecord], list[TransferBlocker]]:
        closure: dict[PageHandle, PhysicalPageRecord] = {}
        blockers: list[TransferBlocker] = []
        stack = [root.handle]
        while stack:
            handle = stack.pop()
            if handle in closure:
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.EXTENT_MUTATED,
                        handle,
                        0,
                        "Radix descendant cycle",
                    )
                )
                continue
            page = self.page_index.pages.get(handle)
            if page is None or page.residency == PhysicalResidency.DEAD:
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.DESCENDANT_CLOSURE,
                        handle,
                        0,
                        "Radix descendant is missing",
                    )
                )
                continue
            if not page.gpu_resident:
                continue
            closure[handle] = page
            stack.extend(sorted(page.children, reverse=True))
        return closure, blockers

    def _owner_blockers(
        self,
        page: PhysicalPageRecord,
        *,
        now_ms: float,
        allow_ready_owners: bool,
        protected_context_id: str | None,
    ) -> list[TransferBlocker]:
        blockers: list[TransferBlocker] = []
        for context_id in sorted(page.owner_contexts):
            lease = self.leases.context(context_id, now_ms=now_ms)
            blocked = (
                context_id == protected_context_id
                or lease.kind == LeaseKind.RUNNING
                or (lease.kind == LeaseKind.READY and not allow_ready_owners)
            )
            if blocked:
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.ENGINE_BUSY,
                        page.handle,
                        page.size_bytes,
                        f"owner {context_id} has {lease.kind.value} lease",
                    )
                )
        return blockers

    @staticmethod
    def _page_blockers(page: PhysicalPageRecord) -> list[TransferBlocker]:
        blockers: list[TransferBlocker] = []
        if not page.sealed:
            blockers.append(
                TransferBlocker(
                    TransferBlockerCode.UNSEALED,
                    page.handle,
                    page.size_bytes,
                    "page extent is not sealed",
                )
            )
        if page.engine_lock_ref > 0 or page.active_reader_count > 0:
            blockers.append(
                TransferBlocker(
                    TransferBlockerCode.NODE_LOCKED,
                    page.handle,
                    page.size_bytes,
                    "page has an engine lock or active reader",
                )
            )
        if page.semantic_pin_contexts:
            blockers.append(
                TransferBlocker(
                    TransferBlockerCode.SEMANTIC_PIN,
                    page.handle,
                    page.size_bytes,
                    "page is semantically pinned",
                )
            )
        if not page.transfer_idle:
            blockers.append(
                TransferBlocker(
                    TransferBlockerCode.NODE_LOADING
                    if page.residency == PhysicalResidency.PREFETCHING
                    else TransferBlockerCode.INFLIGHT,
                    page.handle,
                    page.size_bytes,
                    "page has an in-flight transfer",
                )
            )
        return blockers

    @staticmethod
    def _bundle_id(
        command_kind: CommandKind, handles: tuple[PageHandle, ...]
    ) -> str:
        payload = repr(
            (
                command_kind.value,
                tuple(
                    (item.page_id, item.allocation_generation) for item in handles
                ),
            )
        ).encode("utf-8")
        return f"bundle-{blake2b(payload, digest_size=10).hexdigest()}"

    @staticmethod
    def _fingerprint(
        command_kind: CommandKind,
        closure: dict[PageHandle, PhysicalPageRecord],
        lease: BundleLease,
        blocker_scope: set[PageHandle],
    ) -> str:
        state = []
        for handle in sorted(closure):
            page = closure[handle]
            state.append(
                (
                    handle.page_id,
                    handle.allocation_generation,
                    page.size_bytes,
                    page.residency.value,
                    (
                        (page.parent.page_id, page.parent.allocation_generation)
                        if page.parent is not None
                        else None
                    ),
                    tuple(
                        (child.page_id, child.allocation_generation)
                        for child in sorted(page.children)
                    ),
                    tuple(sorted(page.owner_contexts.items())),
                    page.engine_lock_ref if handle in blocker_scope else None,
                    page.active_reader_count if handle in blocker_scope else None,
                    (
                        tuple(sorted(page.semantic_pin_contexts))
                        if handle in blocker_scope
                        else None
                    ),
                    page.sealed if handle in blocker_scope else None,
                    (
                        page.transfer_direction.value
                        if handle in blocker_scope
                        and page.transfer_direction is not None
                        else None
                    ),
                )
            )
        payload = repr(
            (
                command_kind.value,
                tuple(state),
                lease.strongest_kind.value,
                lease.owner_context_ids,
                tuple(lease.conditions),
            )
        ).encode("utf-8")
        return blake2b(payload, digest_size=16).hexdigest()

    @staticmethod
    def _deduplicate_blockers(
        blockers: list[TransferBlocker],
    ) -> tuple[TransferBlocker, ...]:
        unique = {
            (item.code, item.page_handle, item.required_bytes, item.detail): item
            for item in blockers
        }
        return tuple(
            unique[key]
            for key in sorted(
                unique,
                key=lambda item: (
                    item[0].value,
                    item[1] or PageHandle(0, 0),
                    item[2],
                    item[3],
                ),
            )
        )
