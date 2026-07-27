from __future__ import annotations

from dataclasses import dataclass

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.runtime.bundles import PhysicalBundleBuilder
from beliefkv.runtime.page_index import PageIndexError, PageOwnershipIndex, PhysicalPageRecord
from beliefkv.runtime.protocol import (
    CommandKind,
    ControlCommand,
    PageHandle,
    PhysicalPageAction,
    PhysicalResidency,
    ResolvedCommand,
    ResolvedPageAction,
    TransferBlocker,
    TransferBlockerCode,
)


@dataclass(frozen=True)
class ArbitrationConfig:
    shadow_chunk_bytes: int = 64 * 1024 * 1024
    urgent_chunk_bytes: int = 256 * 1024 * 1024


class RadixArbiter:
    """Resolve logical context intents against current physical page facts."""

    def __init__(
        self,
        graph: RuntimeCausalContextGraph,
        page_index: PageOwnershipIndex,
        config: ArbitrationConfig | None = None,
        bundle_builder: PhysicalBundleBuilder | None = None,
    ) -> None:
        self.graph = graph
        self.page_index = page_index
        self.config = config or ArbitrationConfig()
        self.bundle_builder = bundle_builder or PhysicalBundleBuilder(graph, page_index)

    def resolve(self, command: ControlCommand) -> ResolvedCommand:
        if command.kind == CommandKind.DROP_UNOWNED:
            return self._resolve_drop_unowned(command)
        if command.context_id is None or command.context_epoch is None:
            return ResolvedCommand(
                command,
                (),
                0,
                "missing_context_identity",
                (TransferBlocker(TransferBlockerCode.STALE_GENERATION),),
            )
        context = self.graph.contexts.get(command.context_id)
        if context is None:
            return ResolvedCommand(
                command,
                (),
                0,
                "unknown_context",
                (TransferBlocker(TransferBlockerCode.STALE_GENERATION),),
            )
        try:
            self.page_index.validate_context_epoch(
                command.context_id, command.context_epoch
            )
        except PageIndexError:
            return ResolvedCommand(
                command,
                (),
                0,
                "stale_context_epoch",
                (TransferBlocker(TransferBlockerCode.STALE_GENERATION),),
            )
        if context.epoch != command.context_epoch:
            return ResolvedCommand(
                command,
                (),
                0,
                "graph_page_epoch_divergence",
                (TransferBlocker(TransferBlockerCode.STALE_GENERATION),),
            )

        if command.physical_bundle is not None:
            return self._resolve_physical_bundle(command)

        if command.kind == CommandKind.OFFLOAD_CONTEXT:
            return self._resolve_offload(command, shadow=False)
        if command.kind == CommandKind.SHADOW_CONTEXT:
            return self._resolve_offload(command, shadow=True)
        if command.kind == CommandKind.PREFETCH_CONTEXT:
            return self._resolve_prefetch(command)
        if command.kind == CommandKind.DROP_TERMINAL_PRIVATE:
            return self._resolve_terminal_private_drop(command)
        if command.kind in {CommandKind.PIN_CONTEXT, CommandKind.UNPIN_CONTEXT}:
            action = (
                PhysicalPageAction.PIN
                if command.kind == CommandKind.PIN_CONTEXT
                else PhysicalPageAction.UNPIN
            )
            pages = tuple(
                ResolvedPageAction(page.handle, action, page.size_bytes)
                for page in self.page_index.context_pages(command.context_id)
            )
            return ResolvedCommand(
                command, pages, sum(item.size_bytes for item in pages), "resolved"
            )
        return ResolvedCommand(command, (), 0, "command_not_page_resident")

    def _resolve_physical_bundle(self, command: ControlCommand) -> ResolvedCommand:
        assert command.context_id is not None
        assert command.context_epoch is not None
        intent = command.physical_bundle
        assert intent is not None
        preview = self.bundle_builder.find_intent_preview(
            command.kind,
            command.context_id,
            command.context_epoch,
            intent.bundle_id,
            now_ms=command.created_ts_ms,
            allow_ready_owners=bool(command.metadata.get("allow_ready_owners", False)),
            protected_context_id=command.metadata.get("protected_context_id"),
            bypass_owner_context_ids=frozenset(
                str(item)
                for item in command.metadata.get("bypass_owner_context_ids", ())
            ),
        )
        if preview is None:
            return ResolvedCommand(
                command,
                (),
                0,
                "physical_bundle_no_longer_exists",
                (
                    TransferBlocker(
                        TransferBlockerCode.EXTENT_MUTATED,
                        detail="bundle closure is absent from the authoritative mirror",
                    ),
                ),
                closure_fingerprint=intent.generation_fingerprint,
            )
        if preview.bundle.generation_fingerprint != intent.generation_fingerprint:
            blockers = preview.blockers or (
                TransferBlocker(
                    TransferBlockerCode.EXTENT_MUTATED,
                    detail="physical bundle fingerprint changed before execution",
                ),
            )
            return ResolvedCommand(
                command,
                (),
                0,
                "physical_bundle_fingerprint_changed",
                blockers,
                closure_fingerprint=preview.bundle.generation_fingerprint,
            )
        if preview.blockers:
            return ResolvedCommand(
                command,
                (),
                0,
                "physical_bundle_blocked",
                preview.blockers,
                closure_fingerprint=preview.bundle.generation_fingerprint,
            )
        if (
            preview.bundle.handles != intent.closure_handles
            or preview.page_actions != intent.page_actions
            or preview.bundle.closure_bytes != intent.closure_bytes
        ):
            return ResolvedCommand(
                command,
                (),
                0,
                "physical_bundle_intent_diverged",
                (
                    TransferBlocker(
                        TransferBlockerCode.EXTENT_MUTATED,
                        detail="resolved physical actions differ from the selected intent",
                    ),
                ),
                closure_fingerprint=preview.bundle.generation_fingerprint,
            )
        return ResolvedCommand(
            command,
            preview.page_actions,
            preview.bundle.closure_bytes,
            "physical_bundle_resolved",
            (),
            closure_fingerprint=preview.bundle.generation_fingerprint,
        )

    def _resolve_offload(
        self, command: ControlCommand, *, shadow: bool
    ) -> ResolvedCommand:
        assert command.context_id is not None
        limit = command.target_bytes or (
            self.config.shadow_chunk_bytes if shadow else self.config.urgent_chunk_bytes
        )
        if shadow:
            limit = min(limit, self.config.shadow_chunk_bytes)
        pages = self.page_index.context_pages(command.context_id)
        allow_ready_owners = bool(command.metadata.get("allow_ready_owners", False))
        protected_context_id = command.metadata.get("protected_context_id")
        candidates: list[tuple[PhysicalPageRecord, PhysicalPageAction]] = []
        blockers: list[TransferBlocker] = []
        for page in pages:
            page_blockers = self._page_transfer_blockers(page)
            if page_blockers:
                blockers.extend(page_blockers)
                continue
            if self._page_has_active_owner(
                page,
                allow_ready_owners=allow_ready_owners,
                protected_context_id=protected_context_id,
            ):
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.ENGINE_BUSY,
                        page.handle,
                        page.size_bytes,
                        "active context owner",
                    )
                )
                continue
            if page.residency == PhysicalResidency.DUAL_CLEAN and not shadow:
                candidates.append((page, PhysicalPageAction.COMMIT_CPU))
            elif page.residency == PhysicalResidency.GPU_ONLY:
                candidates.append((page, PhysicalPageAction.START_D2H))

        candidates.sort(
            key=lambda item: (
                -item[0].radix_depth,
                len(item[0].owner_contexts),
                item[0].last_access_ms,
                item[0].handle,
            )
        )
        selected: list[ResolvedPageAction] = []
        selected_handles: set = set()
        resolved_bytes = 0
        for page, action in candidates:
            if resolved_bytes >= limit:
                break
            if not self._leaf_closure_allows(page, selected_handles):
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.DESCENDANT_CLOSURE,
                        page.handle,
                        page.size_bytes,
                        "GPU-resident descendant is outside the selected bundle",
                    )
                )
                continue
            selected.append(ResolvedPageAction(page.handle, action, page.size_bytes))
            selected_handles.add(page.handle)
            resolved_bytes += page.size_bytes
        reason = "resolved" if selected else "no_migratable_marginal_pages"
        return ResolvedCommand(
            command,
            tuple(selected),
            resolved_bytes,
            reason,
            self._deduplicate_blockers(blockers) if not selected else (),
        )

    def _resolve_prefetch(self, command: ControlCommand) -> ResolvedCommand:
        assert command.context_id is not None
        limit = command.target_bytes or self.config.urgent_chunk_bytes
        context_pages = self.page_index.context_pages(command.context_id)
        blockers: list[TransferBlocker] = []
        pages: list[PhysicalPageRecord] = []
        for page in context_pages:
            if page.residency != PhysicalResidency.CPU_ONLY:
                if page.residency == PhysicalResidency.PREFETCHING:
                    blockers.append(
                        TransferBlocker(
                            TransferBlockerCode.NODE_LOADING,
                            page.handle,
                            page.size_bytes,
                            "page is already prefetching",
                        )
                    )
                continue
            if not page.transfer_idle:
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.INFLIGHT,
                        page.handle,
                        page.size_bytes,
                        "page has an in-flight transfer",
                    )
                )
                continue
            if not page.sealed:
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.UNSEALED,
                        page.handle,
                        page.size_bytes,
                        "page extent is not sealed",
                    )
                )
                continue
            pages.append(page)
        pages.sort(key=lambda page: (page.radix_depth, page.handle))
        selected: list[ResolvedPageAction] = []
        selected_handles: set[PageHandle] = set()
        resolved_bytes = 0
        for page in pages:
            if resolved_bytes >= limit:
                break
            if not self._prefetch_closure_allows(page, selected_handles):
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.ANCESTOR_CLOSURE,
                        page.handle,
                        page.size_bytes,
                        "GPU ancestor closure is incomplete",
                    )
                )
                continue
            selected.append(
                ResolvedPageAction(
                    page.handle, PhysicalPageAction.START_H2D, page.size_bytes
                )
            )
            selected_handles.add(page.handle)
            resolved_bytes += page.size_bytes
        return ResolvedCommand(
            command,
            tuple(selected),
            resolved_bytes,
            "resolved" if selected else "no_cpu_pages",
            self._deduplicate_blockers(blockers) if not selected else (),
        )

    def _resolve_drop_unowned(self, command: ControlCommand) -> ResolvedCommand:
        limit = command.target_bytes or self.config.urgent_chunk_bytes
        pages = [
            page
            for page in self.page_index.pages.values()
            if page.residency != PhysicalResidency.DEAD
            and not page.owner_contexts
            and self._transfer_eligible(page)
        ]
        pages.sort(key=lambda page: (-page.radix_depth, page.last_access_ms, page.handle))
        selected: list[ResolvedPageAction] = []
        selected_handles: set = set()
        resolved_bytes = 0
        for page in pages:
            if resolved_bytes >= limit:
                break
            if not self._leaf_closure_allows(page, selected_handles):
                continue
            selected.append(
                ResolvedPageAction(page.handle, PhysicalPageAction.DROP, page.size_bytes)
            )
            selected_handles.add(page.handle)
            resolved_bytes += page.size_bytes
        return ResolvedCommand(
            command,
            tuple(selected),
            resolved_bytes,
            "resolved" if selected else "no_unowned_pages",
        )

    def _resolve_terminal_private_drop(
        self, command: ControlCommand
    ) -> ResolvedCommand:
        """Release only Host copies proven private when a context terminated."""

        limit = command.target_bytes or self.config.urgent_chunk_bytes
        pages: list[PhysicalPageRecord] = []
        blockers: list[TransferBlocker] = []
        for handle in command.target_handles:
            page = self.page_index.pages.get(handle)
            if page is None or page.residency == PhysicalResidency.DEAD:
                continue
            if page.owner_contexts:
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.EXTENT_MUTATED,
                        handle,
                        page.size_bytes,
                        "terminal-private extent acquired another owner",
                    )
                )
                continue
            if page.residency not in {
                PhysicalResidency.DUAL_CLEAN,
                PhysicalResidency.CPU_ONLY,
            }:
                continue
            page_blockers = self._page_transfer_blockers(page)
            if page_blockers:
                blockers.extend(page_blockers)
                continue
            if page.residency == PhysicalResidency.CPU_ONLY and any(
                child in self.page_index.pages
                and self.page_index.pages[child].residency != PhysicalResidency.DEAD
                for child in page.children
            ):
                blockers.append(
                    TransferBlocker(
                        TransferBlockerCode.DESCENDANT_CLOSURE,
                        handle,
                        page.size_bytes,
                        "CPU-only terminal extent is not yet a Radix leaf",
                    )
                )
                continue
            pages.append(page)

        pages.sort(key=lambda page: (-page.radix_depth, page.last_access_ms, page.handle))
        selected: list[ResolvedPageAction] = []
        resolved_bytes = 0
        for page in pages:
            if resolved_bytes >= limit:
                break
            selected.append(
                ResolvedPageAction(
                    page.handle,
                    PhysicalPageAction.DROP_HOST,
                    page.size_bytes,
                )
            )
            resolved_bytes += page.size_bytes
        return ResolvedCommand(
            command,
            tuple(selected),
            resolved_bytes,
            "resolved" if selected else "no_terminal_private_host_pages",
            self._deduplicate_blockers(blockers) if not selected else (),
        )

    @staticmethod
    def _transfer_eligible(page: PhysicalPageRecord) -> bool:
        return (
            page.sealed
            and page.engine_lock_ref == 0
            and not page.semantic_pin_contexts
            and page.active_reader_count == 0
            and page.transfer_idle
        )

    @staticmethod
    def _page_transfer_blockers(
        page: PhysicalPageRecord,
    ) -> tuple[TransferBlocker, ...]:
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
            code = (
                TransferBlockerCode.NODE_LOADING
                if page.residency == PhysicalResidency.PREFETCHING
                else TransferBlockerCode.INFLIGHT
            )
            blockers.append(
                TransferBlocker(
                    code,
                    page.handle,
                    page.size_bytes,
                    "page has an in-flight transfer",
                )
            )
        return tuple(blockers)

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

    def _page_has_active_owner(
        self,
        page: PhysicalPageRecord,
        *,
        allow_ready_owners: bool = False,
        protected_context_id: str | None = None,
    ) -> bool:
        for context_id in page.owner_contexts:
            if context_id == protected_context_id:
                return True
            context = self.graph.contexts.get(context_id)
            if context is None:
                return True
            for invocation_id in context.invocation_ids:
                invocation = self.graph.invocations[invocation_id]
                if invocation.state == InvocationState.RUNNING_LLM:
                    return True
                if (
                    invocation.state == InvocationState.READY
                    and not allow_ready_owners
                ):
                    return True
        return False

    def _leaf_closure_allows(self, page: PhysicalPageRecord, selected: set) -> bool:
        stack = list(page.children)
        seen: set[PageHandle] = set()
        while stack:
            child_handle = stack.pop()
            if child_handle in seen:
                return False
            seen.add(child_handle)
            child = self.page_index.pages.get(child_handle)
            if child is None or child.residency == PhysicalResidency.DEAD:
                continue
            if child.gpu_resident and child_handle not in selected:
                return False
            stack.extend(child.children)
        return True

    def _prefetch_closure_allows(
        self, page: PhysicalPageRecord, selected: set[PageHandle]
    ) -> bool:
        """Do not let HiCache implicitly load unaccounted evicted ancestors."""

        parent_handle = page.parent
        while parent_handle is not None:
            parent = self.page_index.pages.get(parent_handle)
            if parent is None or parent.residency == PhysicalResidency.DEAD:
                return False
            if not parent.gpu_resident and parent_handle not in selected:
                return False
            parent_handle = parent.parent
        return True
