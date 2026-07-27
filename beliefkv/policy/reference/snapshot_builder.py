from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import blake2b
from typing import Mapping, Sequence

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.control.data_consumers import ObservedDataConsumerIndex
from beliefkv.policy.admission import AdmissionController, AdmissionRequest
from beliefkv.policy.leases import CausalLeaseProjector, LeaseKind
from beliefkv.policy.reference.base import (
    CapabilityReport,
    IdentityMapping,
    MetadataMode,
    MetadataSource,
    MetadataValue,
    PhysicalBundleSnapshot,
    PhysicalKVSnapshot,
    PolicyInput,
    ResidencyAction,
    RunnableInvocation,
    RuntimeGraphSnapshot,
)
from beliefkv.policy.resource_snapshot import (
    ResourceSnapshotBuilder,
    RuntimeResourceObservation,
)
from beliefkv.policy.service_curve import TransferServiceCurve
from beliefkv.runtime.page_index import PageOwnershipIndex, PhysicalPageRecord
from beliefkv.runtime.protocol import PageHandle, PhysicalResidency, TransferTelemetry


class PolicySnapshotError(ValueError):
    """Raised when logical and physical sources cannot form one atomic snapshot."""


@dataclass(frozen=True)
class SnapshotBuildStats:
    snapshot_id: str
    tracked_hbm_bytes: int
    untracked_hbm_bytes: int
    tracked_host_bytes: int
    untracked_host_bytes: int
    physical_extent_count: int
    runnable_request_count: int
    topology_version: int
    allocator_version: int


@dataclass(frozen=True)
class _TrackedPhysicalSnapshot:
    cache_key: tuple[object, ...]
    bundles: tuple[PhysicalBundleSnapshot, ...]
    hbm_bytes: int
    host_bytes: int
    topology_fingerprint: str
    allocator_fingerprint: str


@dataclass(frozen=True)
class _PagePhysicalState:
    revision: int
    topology_revision: int
    pages: tuple[PhysicalPageRecord, ...]
    gpu_descendants: frozenset[PageHandle]
    owner_context_ids: tuple[str, ...]
    hbm_bytes: int
    host_bytes: int
    topology_fingerprint: str
    allocator_fingerprint: str
    dirty_handles: frozenset[PageHandle] = frozenset()
    dirty_components: frozenset[str] = frozenset()
    full_rebuild: bool = True


class PolicyInputSnapshotBuilder:
    """Create a common policy snapshot at a scheduler-owned safe point.

    Every real PageHandle is represented exactly once. Overlapping action
    closures are intentionally not materialized as accounting bundles: an
    extent contributes reclaimable bytes only when it can be reclaimed by
    itself in the current Radix state. This keeps replay accounting exact and
    leaves multi-extent closure composition to the what-if resimulator.
    """

    def __init__(
        self,
        graph: RuntimeCausalContextGraph,
        data_consumers: ObservedDataConsumerIndex,
        page_index: PageOwnershipIndex,
        admission: AdmissionController,
        leases: CausalLeaseProjector,
        service_curve: TransferServiceCurve,
        *,
        runtime_name: str = "beliefkv-control-plane",
        runtime_version: str = "development",
        capabilities: CapabilityReport | None = None,
    ) -> None:
        self.graph = graph
        self.data_consumers = data_consumers
        self.page_index = page_index
        self.admission = admission
        self.leases = leases
        self.service_curve = service_curve
        self.resource_builder = ResourceSnapshotBuilder()
        self.capabilities = capabilities or CapabilityReport(
            runtime_name=runtime_name,
            runtime_version=runtime_version,
            supported_residency_actions=frozenset(ResidencyAction),
            execution_order_control=True,
            admission_control=True,
            transfer_dependencies=True,
            native_identity_mapping=False,
            limitations=(
                "running decode batch ordering is runtime-specific",
                "cross-extent closure actions require physical recompilation",
            ),
        )
        self._sequence = 0
        self._topology_fingerprint: str | None = None
        self._allocator_fingerprint: str | None = None
        self._topology_version = 0
        self._allocator_version = 0
        self._last_stats: SnapshotBuildStats | None = None
        self._bundle_cache: dict[
            tuple[PageHandle, tuple[object, ...]], PhysicalBundleSnapshot
        ] = {}
        self._bundle_cache_key_by_handle: dict[
            PageHandle, tuple[PageHandle, tuple[object, ...]]
        ] = {}
        self._tracked_physical_cache: _TrackedPhysicalSnapshot | None = None
        self._page_physical_state_cache: _PagePhysicalState | None = None
        self._bundle_by_handle: dict[PageHandle, PhysicalBundleSnapshot] = {}
        self._lease_kind_by_context: dict[str, LeaseKind] = {}

    @property
    def last_stats(self) -> SnapshotBuildStats | None:
        return self._last_stats

    def build(
        self,
        observation: RuntimeResourceObservation,
        *,
        additional_runnable: Sequence[RunnableInvocation] = (),
        workflow_memory_charges: Mapping[str, float] | None = None,
        workflow_fairness_state: Mapping[str, object] | None = None,
        control_state: Mapping[str, object] | None = None,
        identity_mappings: Sequence[IdentityMapping] = (),
        optional_metadata: Mapping[str, MetadataValue] | None = None,
        transfer_telemetry: Sequence[TransferTelemetry] = (),
        capabilities: CapabilityReport | None = None,
        metadata_mode: MetadataMode = MetadataMode.ONLINE,
    ) -> PolicyInput:
        tracked = self._tracked_physical(now_ms=observation.ts_ms)
        tracked_hbm = tracked.hbm_bytes
        tracked_host = tracked.host_bytes
        if tracked_hbm > observation.hbm_used_bytes:
            raise PolicySnapshotError(
                "PageOwnershipIndex GPU bytes exceed authoritative allocator usage"
            )
        if tracked_host > observation.host_used_bytes:
            raise PolicySnapshotError(
                "PageOwnershipIndex CPU bytes exceed authoritative host usage"
            )

        topology_fingerprint = tracked.topology_fingerprint
        allocator_fingerprint = _fingerprint(
            {
                "tracked_allocator_fingerprint": tracked.allocator_fingerprint,
                "hbm_capacity_bytes": observation.hbm_capacity_bytes,
                "hbm_used_bytes": observation.hbm_used_bytes,
                "host_capacity_bytes": observation.host_capacity_bytes,
                "host_used_bytes": observation.host_used_bytes,
                "host_free_bytes": observation.host_free_bytes,
                "hbm_reserved_bytes": self.admission.reserved_bytes,
            },
            person=b"bk-allocator",
        )
        self._topology_version = self._advance_version(
            topology_fingerprint,
            previous=self._topology_fingerprint,
            current=self._topology_version,
        )
        self._allocator_version = self._advance_version(
            allocator_fingerprint,
            previous=self._allocator_fingerprint,
            current=self._allocator_version,
        )
        self._topology_fingerprint = topology_fingerprint
        self._allocator_fingerprint = allocator_fingerprint

        frontier, queue_state = self._runnable_frontier(additional_runnable)
        fairness_state = self._workflow_fairness_state(
            workflow_memory_charges,
            supplied_state=workflow_fairness_state,
        )
        graph_state = {
            "rccg": self.graph.snapshot(),
            "data_consumers": self.data_consumers.snapshot(),
            "request_queue": queue_state,
            "workflow_fairness": fairness_state,
            "control": dict(control_state or {}),
            "physical_accounting": {
                "tracked_hbm_bytes": tracked_hbm,
                "untracked_hbm_bytes": observation.hbm_used_bytes - tracked_hbm,
                "tracked_host_bytes": tracked_host,
                "untracked_host_bytes": observation.host_used_bytes - tracked_host,
                "accounting_unit": "disjoint_page_handle_extent",
                "closure_policy": "single_extent_marginal_only",
            },
        }
        self._sequence += 1
        snapshot_payload = {
            "sequence": self._sequence,
            "ts_ms": observation.ts_ms,
            "graph_version": self.graph.graph_version,
            "consumer_version": self.data_consumers.version,
            "topology_fingerprint": topology_fingerprint,
            "allocator_fingerprint": allocator_fingerprint,
            "runnable": [item.to_dict() for item in frontier],
            "workflow_fairness": fairness_state,
            "control": dict(control_state or {}),
        }
        snapshot_id = (
            f"policy-{self._sequence:08d}-"
            f"{_fingerprint(snapshot_payload, person=b'bk-policy-in')}"
        )
        bundles = self._physical_bundles_with_untracked(
            tracked,
            observation,
            now_ms=observation.ts_ms,
        )
        physical = PhysicalKVSnapshot(
            snapshot_id=snapshot_id,
            topology_version=self._topology_version,
            allocator_version=self._allocator_version,
            gpu_bytes=observation.hbm_used_bytes,
            cpu_bytes=observation.host_used_bytes,
            bundles=bundles,
        )
        resources = self.resource_builder.build(
            snapshot_id,
            observation,
            hbm_reserved_bytes=self.admission.reserved_bytes,
            service_curve=self.service_curve,
            transfer_telemetry=transfer_telemetry,
        )
        metadata = dict(optional_metadata or {})
        reserved_metadata_name = "beliefkv_resource_observation"
        if reserved_metadata_name in metadata:
            raise PolicySnapshotError(
                f"optional metadata cannot override {reserved_metadata_name}"
            )
        metadata[reserved_metadata_name] = MetadataValue(
            source=MetadataSource.OBSERVED,
            value=dict(self.resource_builder.last_diagnostics),
            producer="resource_snapshot_builder",
        )
        mappings = self._identity_mappings(frontier, identity_mappings)
        result = PolicyInput(
            runtime_graph=RuntimeGraphSnapshot(
                snapshot_id=snapshot_id,
                graph_version=self.graph.graph_version,
                observed_ts_ms=observation.ts_ms,
                state=graph_state,
            ),
            runnable_frontier=frontier,
            physical_kv=physical,
            resources=resources,
            optional_metadata=metadata,
            identity_mappings=mappings,
            capabilities=capabilities or self.capabilities,
            metadata_mode=metadata_mode,
        )
        self._last_stats = SnapshotBuildStats(
            snapshot_id=snapshot_id,
            tracked_hbm_bytes=tracked_hbm,
            untracked_hbm_bytes=observation.hbm_used_bytes - tracked_hbm,
            tracked_host_bytes=tracked_host,
            untracked_host_bytes=observation.host_used_bytes - tracked_host,
            physical_extent_count=len(bundles),
            runnable_request_count=len(frontier),
            topology_version=self._topology_version,
            allocator_version=self._allocator_version,
        )
        return result

    def _workflow_fairness_state(
        self,
        supplied_memory_charges: Mapping[str, float] | None,
        *,
        supplied_state: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        charges = (
            dict(supplied_memory_charges)
            if supplied_memory_charges is not None
            else self.page_index.workflow_gpu_charges()
        )
        if any(
            not isinstance(workflow_id, str)
            or not workflow_id
            or not math.isfinite(float(value))
            or float(value) < 0
            for workflow_id, value in charges.items()
        ):
            raise PolicySnapshotError(
                "workflow memory charges must be finite and non-negative"
            )
        if supplied_state is None:
            accounts = {
                workflow_id: {
                    "weight": account.weight,
                    "attained_service_ms": account.attained_service_ms,
                    "virtual_runtime_ms": account.virtual_runtime,
                    "dispatch_count": account.dispatch_count,
                }
                for workflow_id, account in sorted(
                    self.admission.fairness.accounts.items()
                )
            }
            revision = self.admission.fairness.revision
        else:
            raw_accounts = supplied_state.get("accounts", {})
            if not isinstance(raw_accounts, Mapping):
                raise PolicySnapshotError("workflow fairness accounts must be a mapping")
            accounts = {
                str(workflow_id): dict(raw)
                for workflow_id, raw in sorted(raw_accounts.items())
                if isinstance(raw, Mapping)
            }
            if len(accounts) != len(raw_accounts):
                raise PolicySnapshotError("workflow fairness account must be a mapping")
            revision = int(supplied_state.get("revision", 0))
            if revision < 0:
                raise PolicySnapshotError("workflow fairness revision must be non-negative")
        return {
            "accounts": accounts,
            "memory_charges_bytes": {
                workflow_id: float(value)
                for workflow_id, value in sorted(charges.items())
            },
            "accounting_scope": "root_workflow",
            "revision": revision,
        }

    def _runnable_frontier(
        self,
        additional: Sequence[RunnableInvocation],
    ) -> tuple[tuple[RunnableInvocation, ...], dict[str, object]]:
        by_request: dict[str, RunnableInvocation] = {}
        states: dict[str, str] = {}
        for request in self.admission.pending_requests():
            by_request[request.request_id] = self._from_admission(
                request, "pending_admission"
            )
            states[request.request_id] = "pending_admission"
        for request in self.admission.reserved_requests():
            by_request[request.request_id] = self._from_admission(
                request, "reserved_admission"
            )
            states[request.request_id] = "reserved_admission"
        for item in additional:
            previous = by_request.get(item.request_id)
            if previous is not None and (
                previous.workflow_id,
                previous.invocation_id,
                previous.context_id,
                previous.context_epoch,
            ) != (
                item.workflow_id,
                item.invocation_id,
                item.context_id,
                item.context_epoch,
            ):
                raise PolicySnapshotError(
                    f"runnable identity changed for request {item.request_id}"
                )
            by_request[item.request_id] = item
            states[item.request_id] = item.causal_class
        frontier = tuple(sorted(by_request.values(), key=lambda item: item.request_id))
        return frontier, {
            "states": dict(sorted(states.items())),
            "pending_count": len(self.admission.pending_requests()),
            "reserved_count": len(self.admission.reserved_requests()),
            "additional_runtime_count": len(additional),
            "admission_revision": self.admission.revision,
        }

    def _from_admission(
        self, request: AdmissionRequest, queue_state: str
    ) -> RunnableInvocation:
        invocation = self.graph.invocations.get(request.invocation_id)
        if invocation is None:
            raise PolicySnapshotError(
                f"admission request refers to unknown invocation {request.invocation_id}"
            )
        context = self.graph.contexts.get(request.context_id)
        if context is None or context.epoch != request.context_epoch:
            raise PolicySnapshotError(
                f"admission request has stale context {request.context_id}"
            )
        return RunnableInvocation(
            request_id=request.request_id,
            workflow_id=request.workflow_id,
            invocation_id=request.invocation_id,
            context_id=request.context_id,
            context_epoch=request.context_epoch,
            submitted_ts_ms=request.submitted_ts_ms,
            startup_bytes=request.estimated_incremental_bytes,
            causal_class=(
                f"{queue_state}:{invocation.execution_mode.value}:"
                f"{invocation.relation_type.value}"
            ),
            program_id=invocation.agent_instance_id,
        )

    def _identity_mappings(
        self,
        frontier: Sequence[RunnableInvocation],
        supplied: Sequence[IdentityMapping],
    ) -> tuple[IdentityMapping, ...]:
        mappings = {item.request_id: item for item in supplied}
        if len(mappings) != len(supplied):
            raise PolicySnapshotError("supplied identity mappings contain duplicates")
        frontier_ids = {item.request_id for item in frontier}
        unknown = set(mappings) - frontier_ids
        if unknown:
            raise PolicySnapshotError(
                f"identity mappings refer to non-runnable requests: {sorted(unknown)}"
            )
        for item in frontier:
            mapping = mappings.get(item.request_id)
            if mapping is not None:
                if (
                    mapping.workflow_id,
                    mapping.invocation_id,
                    mapping.context_id,
                    mapping.context_epoch,
                ) != (
                    item.workflow_id,
                    item.invocation_id,
                    item.context_id,
                    item.context_epoch,
                ):
                    raise PolicySnapshotError(
                        f"identity mapping disagrees for request {item.request_id}"
                    )
                continue
            mappings[item.request_id] = IdentityMapping(
                request_id=item.request_id,
                workflow_id=item.workflow_id,
                invocation_id=item.invocation_id,
                context_id=item.context_id,
                context_epoch=item.context_epoch,
                program_id=item.program_id,
                native_request_id=item.request_id,
                native_program_id=item.program_id,
                native_context_id=item.context_id,
            )
        return tuple(sorted(mappings.values(), key=lambda item: item.request_id))

    def _tracked_physical(
        self,
        *,
        now_ms: float,
    ) -> _TrackedPhysicalSnapshot:
        page_state = self._page_physical_state()
        context_lease_kinds = {
            context_id: self.leases.context(context_id, now_ms=now_ms).kind
            for context_id in page_state.owner_context_ids
        }
        lease_signature = tuple(
            (context_id, kind.value)
            for context_id, kind in sorted(context_lease_kinds.items())
        )
        cache_key = (page_state.revision, lease_signature)
        cached = self._tracked_physical_cache
        if cached is not None and cached.cache_key == cache_key:
            return cached

        incremental_handles = set(page_state.dirty_handles)
        changed_lease_contexts = {
            context_id
            for context_id in (
                set(self._lease_kind_by_context) | set(context_lease_kinds)
            )
            if self._lease_kind_by_context.get(context_id)
            != context_lease_kinds.get(context_id)
        }
        for context_id in changed_lease_contexts:
            if self.page_index.has_context(context_id):
                incremental_handles.update(
                    page.handle
                    for page in self.page_index.context_pages(context_id)
                )
        incremental = bool(
            cached is not None
            and not page_state.full_rebuild
            and self._bundle_by_handle
        )
        if not incremental:
            live_handles = {page.handle for page in page_state.pages}
            incremental_handles = live_handles
            for handle in tuple(self._bundle_by_handle):
                if handle not in live_handles:
                    self._bundle_by_handle.pop(handle, None)
                    previous_key = self._bundle_cache_key_by_handle.pop(
                        handle, None
                    )
                    if previous_key is not None:
                        self._bundle_cache.pop(previous_key, None)
            pages_to_refresh = page_state.pages
        else:
            pages_to_refresh = tuple(
                self.page_index.require_page(handle)
                for handle in sorted(incremental_handles)
            )
        for page in pages_to_refresh:
            lease_kind = max(
                (
                    context_lease_kinds[context_id]
                    for context_id in page.owner_contexts
                ),
                key=_lease_strength,
                default=LeaseKind.DEAD,
            )
            state = self._page_bundle_state(
                page,
                has_gpu_descendant=page.handle in page_state.gpu_descendants,
                lease_kind=lease_kind,
            )
            bundle_cache_key = (page.handle, state)
            previous_key = self._bundle_cache_key_by_handle.get(page.handle)
            if previous_key is not None and previous_key != bundle_cache_key:
                self._bundle_cache.pop(previous_key, None)
            bundle = self._bundle_cache.get(bundle_cache_key)
            if bundle is None:
                bundle = self._page_bundle(
                    page,
                    has_gpu_descendant=page.handle in page_state.gpu_descendants,
                    lease_kind=lease_kind,
                )
                self._bundle_cache[bundle_cache_key] = bundle
            self._bundle_cache_key_by_handle[page.handle] = bundle_cache_key
            self._bundle_by_handle[page.handle] = bundle
        result = _TrackedPhysicalSnapshot(
            cache_key=cache_key,
            bundles=tuple(self._bundle_by_handle.values()),
            hbm_bytes=page_state.hbm_bytes,
            host_bytes=page_state.host_bytes,
            topology_fingerprint=page_state.topology_fingerprint,
            allocator_fingerprint=page_state.allocator_fingerprint,
        )
        self._lease_kind_by_context = context_lease_kinds
        self._tracked_physical_cache = result
        return result

    def _page_physical_state(self) -> _PagePhysicalState:
        cached = self._page_physical_state_cache
        if cached is not None and cached.revision == self.page_index.revision:
            return cached
        delta = (
            self.page_index.changes_since(cached.revision)
            if cached is not None
            else None
        )
        topology_changed = bool(
            cached is None
            or cached.topology_revision != self.page_index.topology_revision
            or (delta is not None and delta.full_rebuild_required)
        )
        residency_changed = bool(delta is not None and "residency" in delta.components)
        full_rebuild = topology_changed
        if (
            residency_changed
            and delta is not None
            and any(handle not in self._bundle_by_handle for handle in delta.handles)
        ):
            full_rebuild = True
        if topology_changed:
            self.page_index.assert_consistent()
            topology_fingerprint = (
                f"page-index-topology:{self.page_index.topology_revision}"
            )
        else:
            topology_fingerprint = cached.topology_fingerprint
        if full_rebuild:
            pages = tuple(
                page
                for _, page in sorted(self.page_index.pages.items())
                if page.residency != PhysicalResidency.DEAD
            )
            gpu_descendants = frozenset(self._gpu_descendant_flags(pages))
            owner_context_ids = tuple(
                sorted(
                    {
                        context_id
                        for page in pages
                        for context_id in page.owner_contexts
                    }
                )
            )
            hbm_bytes = sum(
                page.size_bytes for page in pages if page.gpu_resident
            )
            host_bytes = sum(
                page.size_bytes for page in pages if page.cpu_resident
            )
        else:
            assert cached is not None
            pages = cached.pages
            dirty_handles = set(delta.handles) if delta is not None else set()
            gpu_descendants = set(cached.gpu_descendants)
            if residency_changed and delta is not None:
                affected_ancestors: set[PageHandle] = set()
                for handle in delta.handles:
                    page = self.page_index.pages.get(handle)
                    ancestor = page.parent if page is not None else None
                    seen: set[PageHandle] = set()
                    while ancestor is not None:
                        if ancestor in seen:
                            raise PolicySnapshotError(
                                "Radix ancestor cycle detected during incremental snapshot"
                            )
                        seen.add(ancestor)
                        affected_ancestors.add(ancestor)
                        ancestor = self.page_index.require_page(ancestor).parent
                for handle in sorted(
                    affected_ancestors,
                    key=lambda item: self.page_index.require_page(item).radix_depth,
                    reverse=True,
                ):
                    page = self.page_index.require_page(handle)
                    has_gpu_descendant = any(
                        self.page_index.require_page(child).gpu_resident
                        or child in gpu_descendants
                        for child in page.children
                    )
                    if has_gpu_descendant:
                        gpu_descendants.add(handle)
                    else:
                        gpu_descendants.discard(handle)
                dirty_handles.update(affected_ancestors)
            owner_context_ids = (
                tuple(
                    sorted(
                        {
                            context_id
                            for page in pages
                            for context_id in page.owner_contexts
                        }
                    )
                )
                if delta is not None and "owner" in delta.components
                else cached.owner_context_ids
            )
            hbm_bytes = cached.hbm_bytes
            host_bytes = cached.host_bytes
            if residency_changed and delta is not None:
                for handle in delta.handles:
                    page = self.page_index.require_page(handle)
                    previous = self._bundle_by_handle[handle]
                    hbm_bytes += (
                        page.size_bytes if page.gpu_resident else 0
                    ) - previous.gpu_bytes
                    host_bytes += (
                        page.size_bytes if page.cpu_resident else 0
                    ) - previous.cpu_bytes
        result = _PagePhysicalState(
            revision=self.page_index.revision,
            topology_revision=self.page_index.topology_revision,
            pages=pages,
            gpu_descendants=frozenset(gpu_descendants),
            owner_context_ids=owner_context_ids,
            hbm_bytes=hbm_bytes,
            host_bytes=host_bytes,
            topology_fingerprint=topology_fingerprint,
            allocator_fingerprint=f"page-index-state:{self.page_index.revision}",
            dirty_handles=(
                frozenset(dirty_handles)
                if not full_rebuild
                else (delta.handles if delta is not None else frozenset())
            ),
            dirty_components=(
                delta.components if delta is not None else frozenset()
            ),
            full_rebuild=full_rebuild,
        )
        self._page_physical_state_cache = result
        return result

    def _physical_bundles_with_untracked(
        self,
        tracked: _TrackedPhysicalSnapshot,
        observation: RuntimeResourceObservation,
        *,
        now_ms: float,
    ) -> tuple[PhysicalBundleSnapshot, ...]:
        bundles = list(tracked.bundles)
        untracked_hbm = observation.hbm_used_bytes - tracked.hbm_bytes
        if untracked_hbm:
            bundles.append(
                self._protected_untracked_bundle(
                    tier="hbm",
                    bytes_=untracked_hbm,
                    now_ms=now_ms,
                )
            )
        untracked_host = observation.host_used_bytes - tracked.host_bytes
        if untracked_host:
            bundles.append(
                self._protected_untracked_bundle(
                    tier="host",
                    bytes_=untracked_host,
                    now_ms=now_ms,
                )
            )
        return tuple(sorted(bundles, key=lambda item: item.bundle_id))

    def _page_bundle(
        self,
        page: PhysicalPageRecord,
        *,
        has_gpu_descendant: bool,
        lease_kind: LeaseKind,
    ) -> PhysicalBundleSnapshot:
        extent_id = _extent_id(page.handle)
        blocker_codes: set[str] = set()
        physical_blocked = False
        if not page.sealed:
            blocker_codes.add("unsealed")
            physical_blocked = True
        if page.engine_lock_ref > 0 or page.active_reader_count > 0:
            blocker_codes.add("node_locked")
            physical_blocked = True
        if page.semantic_pin_contexts:
            blocker_codes.add("semantic_pin")
            physical_blocked = True
        if not page.transfer_idle:
            blocker_codes.add("inflight")
            physical_blocked = True

        if page.gpu_resident:
            if has_gpu_descendant:
                blocker_codes.add("descendant_closure")
            if lease_kind in {LeaseKind.RUNNING, LeaseKind.READY}:
                blocker_codes.add(f"owner_{lease_kind.value}")
        actionable = not blocker_codes
        reclaimable = page.size_bytes if page.gpu_resident and actionable else 0
        scope = (
            "unowned_extent"
            if not page.owner_contexts
            else "exclusive_suffix"
            if len(page.owner_contexts) == 1
            else "shared_subtree"
        )
        generation_payload = {
            "extent_id": extent_id,
            "size_bytes": page.size_bytes,
            "parent": _optional_extent_id(page.parent),
            "children": [_extent_id(item) for item in sorted(page.children)],
            "owners": sorted(page.owner_contexts.items()),
            "sealed": page.sealed,
        }
        return PhysicalBundleSnapshot(
            bundle_id=f"extent-bundle-{_short_hash(extent_id)}",
            owner_context_ids=tuple(page.owner_contexts),
            scope=scope,
            physical_unique_bytes=page.size_bytes,
            gpu_bytes=page.size_bytes if page.gpu_resident else 0,
            cpu_bytes=page.size_bytes if page.cpu_resident else 0,
            marginal_reclaimable_bytes=reclaimable,
            closure_bytes=page.size_bytes,
            locked_bytes=page.size_bytes if physical_blocked and page.gpu_resident else 0,
            residency=page.residency.value,
            generation_fingerprint=_fingerprint(
                generation_payload, person=b"bk-generation"
            ),
            last_access_ms=page.last_access_ms,
            extent_ids=(extent_id,),
            lease_kind=lease_kind.value,
            actionable=actionable,
            blocker_codes=tuple(blocker_codes),
            parent_extent_id=_optional_extent_id(page.parent),
            child_extent_ids=tuple(_extent_id(item) for item in page.children),
        )

    def page_bundle_at_safe_point(
        self,
        handle: PageHandle,
        *,
        now_ms: float,
    ) -> PhysicalBundleSnapshot:
        """Build one exact live bundle for bounded plan validation."""

        page = self.page_index.require_page(handle)
        lease_kind = max(
            (
                self.leases.context(context_id, now_ms=now_ms).kind
                for context_id in page.owner_contexts
            ),
            key=_lease_strength,
            default=LeaseKind.DEAD,
        )
        return self._page_bundle(
            page,
            has_gpu_descendant=self._has_gpu_descendant(handle),
            lease_kind=lease_kind,
        )

    def _has_gpu_descendant(self, handle: PageHandle) -> bool:
        page = self.page_index.require_page(handle)
        pending = list(page.children)
        seen: set[PageHandle] = set()
        while pending:
            child_handle = pending.pop()
            if child_handle in seen:
                raise PolicySnapshotError("Radix descendant cycle detected")
            seen.add(child_handle)
            child = self.page_index.require_page(child_handle)
            if child.gpu_resident:
                return True
            pending.extend(child.children)
        return False

    @staticmethod
    def _page_bundle_state(
        page: PhysicalPageRecord,
        *,
        has_gpu_descendant: bool,
        lease_kind: LeaseKind,
    ) -> tuple[object, ...]:
        return (
            page.handle,
            page.size_bytes,
            page.residency,
            page.radix_depth,
            page.parent,
            tuple(sorted(page.children)),
            tuple(sorted(page.owner_contexts.items())),
            page.engine_lock_ref,
            tuple(sorted(page.semantic_pin_contexts)),
            page.active_reader_count,
            page.sealed,
            page.transfer_direction,
            page.last_access_ms,
            has_gpu_descendant,
            lease_kind,
        )

    def _protected_untracked_bundle(
        self,
        *,
        tier: str,
        bytes_: int,
        now_ms: float,
    ) -> PhysicalBundleSnapshot:
        extent_id = f"allocator-untracked:{tier}"
        return PhysicalBundleSnapshot(
            bundle_id=f"protected-untracked-{tier}",
            owner_context_ids=(),
            scope="protected_untracked",
            physical_unique_bytes=bytes_,
            gpu_bytes=bytes_ if tier == "hbm" else 0,
            cpu_bytes=bytes_ if tier == "host" else 0,
            marginal_reclaimable_bytes=0,
            closure_bytes=bytes_,
            locked_bytes=bytes_ if tier == "hbm" else 0,
            residency="gpu_private" if tier == "hbm" else "host_untracked",
            generation_fingerprint=_fingerprint(
                {"tier": tier, "bytes": bytes_}, person=b"bk-untracked"
            ),
            last_access_ms=now_ms,
            extent_ids=(extent_id,),
            lease_kind=LeaseKind.RUNNING.value,
            actionable=False,
            blocker_codes=("allocator_untracked",),
        )

    def _gpu_descendant_flags(
        self,
        pages: Sequence[PhysicalPageRecord],
    ) -> set[PageHandle]:
        """Return nodes with any GPU-resident descendant in one bottom-up pass."""

        result: set[PageHandle] = set()
        by_handle = {page.handle: page for page in pages}
        for page in sorted(
            pages,
            key=lambda item: (-item.radix_depth, item.handle),
        ):
            if page.parent is None:
                continue
            if page.gpu_resident or page.handle in result:
                if page.parent not in by_handle:
                    raise PolicySnapshotError(
                        "live Radix extent has a missing parent during snapshot"
                    )
                result.add(page.parent)
        return result

    def _topology_payload(self) -> dict[str, object]:
        topology = []
        for handle, page in sorted(self.page_index.pages.items()):
            if page.residency == PhysicalResidency.DEAD:
                continue
            topology.append(
                {
                    "extent": _extent_id(handle),
                    "size": page.size_bytes,
                    "parent": _optional_extent_id(page.parent),
                    "children": [_extent_id(item) for item in sorted(page.children)],
                    "sealed": page.sealed,
                }
            )
        return {"extents": topology}

    def _allocator_payload(self) -> dict[str, object]:
        allocator = []
        for handle, page in sorted(self.page_index.pages.items()):
            if page.residency == PhysicalResidency.DEAD:
                continue
            allocator.append(
                {
                    "extent": _extent_id(handle),
                    "residency": page.residency.value,
                    "owners": sorted(page.owner_contexts.items()),
                    "engine_lock_ref": page.engine_lock_ref,
                    "active_reader_count": page.active_reader_count,
                    "semantic_pins": sorted(page.semantic_pin_contexts),
                    "transfer_direction": (
                        page.transfer_direction.value
                        if page.transfer_direction is not None
                        else None
                    ),
                    "last_access_ms": page.last_access_ms,
                }
            )
        return {"extents": allocator}

    @staticmethod
    def _advance_version(
        fingerprint: str,
        *,
        previous: str | None,
        current: int,
    ) -> int:
        return current + 1 if fingerprint != previous else current


def _extent_id(handle: PageHandle) -> str:
    return f"page:{handle.page_id}:generation:{handle.allocation_generation}"


def page_handle_from_extent_id(extent_id: str) -> PageHandle:
    parts = extent_id.split(":")
    if len(parts) != 4 or parts[0] != "page" or parts[2] != "generation":
        raise ValueError(f"not a page extent ID: {extent_id}")
    try:
        return PageHandle(int(parts[1]), int(parts[3]))
    except ValueError as error:
        raise ValueError(f"invalid page extent ID: {extent_id}") from error


def _optional_extent_id(handle: PageHandle | None) -> str | None:
    return _extent_id(handle) if handle is not None else None


def _short_hash(value: str) -> str:
    return blake2b(value.encode("utf-8"), digest_size=10).hexdigest()


def _lease_strength(kind: LeaseKind) -> int:
    return {
        LeaseKind.DEAD: 0,
        LeaseKind.SPECULATIVE: 1,
        LeaseKind.CONDITIONAL_RESUME: 2,
        LeaseKind.READY: 3,
        LeaseKind.RUNNING: 4,
    }[kind]


def _fingerprint(value: Mapping[str, object], *, person: bytes) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return blake2b(payload, digest_size=16, person=person).hexdigest()
