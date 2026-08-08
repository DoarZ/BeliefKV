from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.residency import ResidencyClass, ResidencyClassifier
from beliefkv.policy.shadow_controller import ShadowController, ShadowSignals
from beliefkv.policy.transfer_guard import TransferAttemptGuard
from beliefkv.predictor.types import RemainingTimePrediction
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.bundles import (
    BundlePreviewEvent,
    BundleScope,
    PhysicalBundleBuilder,
    PhysicalBundlePreview,
)
from beliefkv.runtime.protocol import (
    CommandKind,
    CommandQueueClass,
    ControlCommand,
    PhysicalBundleIntent,
    PageHandle,
)


@dataclass(frozen=True)
class TransferPlannerConfig:
    reserve_hbm_bytes: int = 1 << 30
    urgent_chunk_bytes: int = 256 * 1024 * 1024
    prefetch_chunk_bytes: int = 256 * 1024 * 1024
    prefetch_horizon_ms: float = 50.0
    prefetch_enabled: bool = True
    bundle_preview_audit_max_detailed_per_cycle: int = 8

    def __post_init__(self) -> None:
        if self.bundle_preview_audit_max_detailed_per_cycle <= 0:
            raise ValueError("bundle preview audit limit must be positive")


class ReactiveTransferPlanner:
    """Event-driven pressure/prefetch planner; prediction is optional."""

    def __init__(
        self,
        graph: RuntimeCausalContextGraph,
        page_index: PageOwnershipIndex,
        classifier: ResidencyClassifier,
        frontier: CausalFrontierScheduler,
        shadow: ShadowController,
        config: TransferPlannerConfig | None = None,
        retry_guard: TransferAttemptGuard | None = None,
        bundle_builder: PhysicalBundleBuilder | None = None,
    ) -> None:
        self.graph = graph
        self.page_index = page_index
        self.classifier = classifier
        self.frontier = frontier
        self.shadow = shadow
        self.config = config or TransferPlannerConfig()
        self.retry_guard = retry_guard
        self.bundle_builder = bundle_builder or PhysicalBundleBuilder(graph, page_index)
        self._sequence = 0
        self._bundle_events: list[BundlePreviewEvent] = []
        self._last_bundle_fingerprints: dict[
            tuple[str, int, str, str], tuple[object, ...]
        ] = {}
        self._last_context_lease_fingerprints: dict[
            tuple[str, int], tuple[object, ...]
        ] = {}
        self._restore_focus_context_id: str | None = None

    def plan_next(
        self,
        *,
        now_ms: float,
        hbm_capacity_bytes: int,
        actual_hbm_used_bytes: int | None = None,
        reserved_hbm_bytes: int = 0,
        admission_required_bytes: int = 0,
        protected_context_id: str | None = None,
        allow_frontier_spill: bool = False,
        drop_unowned_enabled: bool = True,
        signals: ShadowSignals,
        predictions: dict[str, RemainingTimePrediction] | None = None,
        preferred_restore_context_ids: tuple[str, ...] = (),
    ) -> ControlCommand | None:
        if reserved_hbm_bytes < 0:
            raise ValueError("reserved_hbm_bytes must be non-negative")
        predictions = predictions or {}
        required_free = self.config.reserve_hbm_bytes + admission_required_bytes
        physical_used = max(
            self.page_index.gpu_bytes,
            actual_hbm_used_bytes if actual_hbm_used_bytes is not None else 0,
        )
        actual_free = max(0, hbm_capacity_bytes - physical_used)
        shortage = max(0, required_free - actual_free)
        if shortage > 0:
            command = self._pressure_command(
                now_ms,
                shortage,
                protected_context_id=protected_context_id,
                allow_frontier_spill=allow_frontier_spill,
                drop_unowned_enabled=drop_unowned_enabled,
                host_available_bytes=signals.host_free_bytes,
            )
            if command is not None:
                return command

        if self.config.prefetch_enabled:
            prefetch = self._prefetch_command(
                now_ms,
                actual_free
                - self.config.reserve_hbm_bytes
                - reserved_hbm_bytes,
                predictions,
                preferred_restore_context_ids=preferred_restore_context_ids,
            )
            if prefetch is not None:
                return prefetch
        shadow = self.shadow.plan(now_ms, signals, predictions)
        return self._materialize_shadow(shadow, now_ms)

    def plan_terminal_cleanup(
        self,
        *,
        now_ms: float,
        context_id: str,
        context_epoch: int,
        target_handles: tuple[PageHandle, ...],
    ) -> ControlCommand | None:
        if not target_handles:
            return None
        target_bytes = min(
            self.config.urgent_chunk_bytes,
            sum(
                page.size_bytes
                for handle in target_handles
                if (page := self.page_index.pages.get(handle)) is not None
                and page.cpu_resident
            ),
        )
        if target_bytes <= 0:
            return None
        command = self._command(
            CommandKind.DROP_TERMINAL_PRIVATE,
            now_ms,
            context_id=context_id,
            context_epoch=context_epoch,
            target_bytes=target_bytes,
            priority=2.0e9,
            metadata={"reason": "terminal_private_host_cleanup"},
            target_handles=target_handles,
        )
        return command if self._command_is_eligible(command, now_ms) else None

    def _pressure_command(
        self,
        now_ms: float,
        shortage: int,
        *,
        protected_context_id: str | None,
        allow_frontier_spill: bool,
        drop_unowned_enabled: bool,
        host_available_bytes: int,
    ) -> ControlCommand | None:
        if drop_unowned_enabled and any(
            not page.owner_contexts and page.gpu_resident
            for page in self.page_index.pages.values()
        ):
            return self._command(
                CommandKind.DROP_UNOWNED,
                now_ms,
                target_bytes=min(shortage, self.config.urgent_chunk_bytes),
                priority=1.0e9,
            )

        exclusive_victims: dict[str, tuple[tuple, ControlCommand]] = {}
        shared_victims: dict[str, tuple[tuple, ControlCommand]] = {}
        for context_id, context in self.graph.contexts.items():
            assessment = self.classifier.context(context_id, now_ms)
            if assessment.residency_class not in {
                ResidencyClass.PARKED,
                ResidencyClass.DEAD_UNOWNED,
            }:
                continue
            previews = self.bundle_builder.previews_for_context(
                CommandKind.OFFLOAD_CONTEXT,
                context_id,
                context.epoch,
                now_ms=now_ms,
                host_available_bytes=host_available_bytes,
            )
            self._record_bundle_previews(previews, now_ms)
            distance = max(
                (
                    self.frontier.ancestor_distance_to_active_leaf(item.invocation_id)
                    for item in self.graph.context_invocations(context_id)
                ),
                default=0,
            )
            dead = assessment.residency_class == ResidencyClass.DEAD_UNOWNED
            wait_tool = any(
                item.state == InvocationState.WAIT_TOOL
                for item in self.graph.context_invocations(context_id)
            )
            for preview in previews:
                bundle = preview.bundle
                if (
                    not preview.eligible
                    or bundle.marginal_reclaimable_bytes <= 0
                    or bundle.closure_bytes > self.config.urgent_chunk_bytes
                ):
                    continue
                reason = (
                    "context_exclusive_suffix_offload"
                    if bundle.scope == BundleScope.EXCLUSIVE_SUFFIX
                    else "global_shared_bundle_reclaim"
                )
                command = self._command(
                    CommandKind.OFFLOAD_CONTEXT,
                    now_ms,
                    context_id=context_id,
                    context_epoch=context.epoch,
                    target_bytes=bundle.marginal_reclaimable_bytes,
                    priority=1.0e8,
                    physical_bundle=preview.intent(),
                    metadata={
                        **self._bundle_metadata(preview),
                        "reason": reason,
                    },
                )
                if not self._command_is_eligible(command, now_ms):
                    continue
                copy_ratio = preview.copy_bytes / max(
                    bundle.marginal_reclaimable_bytes, 1
                )
                rank = (
                    -int(dead),
                    -int(preview.copy_bytes == 0),
                    -distance,
                    -int(wait_tool),
                    -min(shortage, bundle.marginal_reclaimable_bytes),
                    copy_ratio,
                    assessment.since_ms,
                    bundle.bundle_id,
                    context_id,
                )
                candidate_pool = (
                    exclusive_victims
                    if bundle.scope == BundleScope.EXCLUSIVE_SUFFIX
                    else shared_victims
                )
                existing = candidate_pool.get(bundle.bundle_id)
                if existing is None or rank < existing[0]:
                    candidate_pool[bundle.bundle_id] = (rank, command)
        victims = exclusive_victims or shared_victims
        if not victims:
            if not allow_frontier_spill or protected_context_id is None:
                return None
            return self._frontier_spill_command(
                now_ms,
                shortage,
                protected_context_id=protected_context_id,
                host_available_bytes=host_available_bytes,
            )
        return min(victims.values(), key=lambda item: item[0])[1]

    def _frontier_spill_command(
        self,
        now_ms: float,
        shortage: int,
        *,
        protected_context_id: str,
        host_available_bytes: int,
    ) -> ControlCommand | None:
        protected = self.graph.contexts.get(protected_context_id)
        protected_workflow_id = protected.workflow_id if protected is not None else None
        candidates: list[tuple[tuple, ControlCommand]] = []
        for context_id, context in self.graph.contexts.items():
            if context_id == protected_context_id:
                continue
            assessment = self.classifier.context(context_id, now_ms)
            if assessment.residency_class != ResidencyClass.IMMINENT:
                continue
            previews = self.bundle_builder.previews_for_context(
                CommandKind.OFFLOAD_CONTEXT,
                context_id,
                context.epoch,
                now_ms=now_ms,
                allow_ready_owners=True,
                protected_context_id=protected_context_id,
                host_available_bytes=host_available_bytes,
            )
            self._record_bundle_previews(previews, now_ms)
            for preview in previews:
                bundle = preview.bundle
                if (
                    not preview.eligible
                    or bundle.scope != BundleScope.EXCLUSIVE_SUFFIX
                    or bundle.marginal_reclaimable_bytes <= 0
                    or bundle.closure_bytes > self.config.urgent_chunk_bytes
                ):
                    continue
                metadata = {
                    **self._bundle_metadata(preview),
                    "reason": "admission_liveness_frontier_spill",
                    "allow_ready_owners": True,
                    "protected_context_id": protected_context_id,
                }
                command = self._command(
                    CommandKind.OFFLOAD_CONTEXT,
                    now_ms,
                    context_id=context_id,
                    context_epoch=context.epoch,
                    target_bytes=bundle.marginal_reclaimable_bytes,
                    priority=1.0e8,
                    physical_bundle=preview.intent(),
                    metadata=metadata,
                )
                if not self._command_is_eligible(command, now_ms):
                    continue
                rank = (
                    int(context.workflow_id == protected_workflow_id),
                    -min(shortage, bundle.marginal_reclaimable_bytes),
                    preview.copy_bytes / max(bundle.marginal_reclaimable_bytes, 1),
                    -assessment.since_ms,
                    bundle.bundle_id,
                )
                candidates.append((rank, command))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _prefetch_command(
        self,
        now_ms: float,
        available_bytes: int,
        predictions: dict[str, RemainingTimePrediction],
        *,
        preferred_restore_context_ids: tuple[str, ...] = (),
    ) -> ControlCommand | None:
        if available_bytes <= 0:
            return None
        preferred_rank = {
            context_id: rank
            for rank, context_id in enumerate(preferred_restore_context_ids)
        }
        candidates: list[tuple[tuple, ControlCommand]] = []
        for context_id, context in self.graph.contexts.items():
            if preferred_rank and context_id not in preferred_rank:
                continue
            assessment = self.classifier.context(context_id, now_ms)
            prediction = predictions.get(context_id)
            predicted_imminent = (
                prediction is not None
                and prediction.usable
                and prediction.p50_ms <= self.config.prefetch_horizon_ms
            )
            if assessment.residency_class != ResidencyClass.IMMINENT and not predicted_imminent:
                continue
            previews = self.bundle_builder.previews_for_context(
                CommandKind.PREFETCH_CONTEXT,
                context_id,
                context.epoch,
                now_ms=now_ms,
                device_available_bytes=available_bytes,
            )
            self._record_bundle_previews(previews, now_ms)
            for preview in previews:
                bytes_needed = preview.bundle.closure_bytes
                if (
                    not preview.eligible
                    or bytes_needed == 0
                    or bytes_needed > available_bytes
                    or bytes_needed > self.config.prefetch_chunk_bytes
                ):
                    continue
                command = self._command(
                    CommandKind.PREFETCH_CONTEXT,
                    now_ms,
                    context_id=context_id,
                    context_epoch=context.epoch,
                    target_bytes=bytes_needed,
                    priority=1.0e7,
                    physical_bundle=preview.intent(),
                    metadata=self._bundle_metadata(preview),
                )
                if not self._command_is_eligible(command, now_ms):
                    continue
                rank = (
                    preferred_rank.get(context_id, len(preferred_rank)),
                    -int(assessment.residency_class == ResidencyClass.IMMINENT),
                    prediction.p50_ms if prediction is not None else 0.0,
                    assessment.since_ms,
                    -bytes_needed,
                    preview.bundle.bundle_id,
                )
                candidates.append((rank, command))
        if not candidates:
            self._restore_focus_context_id = None
            return None
        focused = [
            item
            for item in candidates
            if item[1].context_id == self._restore_focus_context_id
        ]
        selected = min(focused or candidates, key=lambda item: item[0])[1]
        self._restore_focus_context_id = selected.context_id
        return selected

    def _command(
        self,
        kind: CommandKind,
        now_ms: float,
        *,
        context_id: str | None = None,
        context_epoch: int | None = None,
        target_bytes: int,
        priority: float,
        metadata: dict | None = None,
        physical_bundle: PhysicalBundleIntent | None = None,
        target_handles: tuple[PageHandle, ...] = (),
    ) -> ControlCommand:
        command = ControlCommand(
            command_id=f"reactive-{self._sequence}",
            kind=kind,
            created_ts_ms=now_ms,
            context_id=context_id,
            context_epoch=context_epoch,
            target_bytes=target_bytes,
            priority=priority,
            queue_class=CommandQueueClass.URGENT,
            metadata=metadata or {},
            physical_bundle=physical_bundle,
            target_handles=target_handles,
        )
        self._sequence += 1
        return command

    def drain_bundle_events(self) -> tuple[BundlePreviewEvent, ...]:
        events = tuple(self._bundle_events)
        self._bundle_events.clear()
        return events

    def _materialize_shadow(
        self, command: ControlCommand | None, now_ms: float
    ) -> ControlCommand | None:
        if command is None or command.context_id is None or command.context_epoch is None:
            return command
        previews = self.bundle_builder.previews_for_context(
            CommandKind.SHADOW_CONTEXT,
            command.context_id,
            command.context_epoch,
            now_ms=now_ms,
        )
        self._record_bundle_previews(previews, now_ms)
        candidates = [
            item
            for item in previews
            if item.eligible
            and item.bundle.scope == BundleScope.EXCLUSIVE_SUFFIX
            and item.bundle.closure_bytes <= command.target_bytes
            and item.bundle.closure_bytes <= self.config.urgent_chunk_bytes
        ]
        if not candidates:
            return None
        preview = min(
            candidates,
            key=lambda item: (
                item.copy_bytes,
                -item.bundle.gpu_bytes,
                item.bundle.bundle_id,
            ),
        )
        materialized = replace(
            command,
            target_bytes=preview.bundle.closure_bytes,
            metadata={**dict(command.metadata), **self._bundle_metadata(preview)},
            physical_bundle=preview.intent(),
        )
        return materialized if self._command_is_eligible(materialized, now_ms) else None

    def _record_bundle_previews(
        self,
        previews: tuple[PhysicalBundlePreview, ...],
        now_ms: float,
    ) -> None:
        detailed_count = 0
        omitted_count = 0
        omitted_closure_bytes = 0
        omitted_reclaimable_bytes = 0
        omitted_blockers: dict[str, int] = {}
        for preview in previews:
            bundle = preview.bundle
            identity = (
                preview.context_id,
                preview.context_epoch,
                preview.command_kind.value,
                bundle.bundle_id,
            )
            preview_state = (
                bundle.generation_fingerprint,
                preview.eligible,
                tuple(sorted(item.code.value for item in preview.blockers)),
                bundle.closure_bytes,
                preview.copy_bytes,
                bundle.marginal_reclaimable_bytes,
                bundle.locked_bytes,
                bundle.scope.value,
                bundle.exclusive_action_bytes,
                bundle.cross_context_action_bytes,
                bundle.foreign_owner_context_ids,
            )
            if self._last_bundle_fingerprints.get(identity) == preview_state:
                continue
            self._last_bundle_fingerprints[identity] = preview_state
            if (
                detailed_count
                >= self.config.bundle_preview_audit_max_detailed_per_cycle
            ):
                omitted_count += 1
                omitted_closure_bytes += bundle.closure_bytes
                omitted_reclaimable_bytes += bundle.marginal_reclaimable_bytes
                for blocker in preview.blockers:
                    code = blocker.code.value
                    omitted_blockers[code] = omitted_blockers.get(code, 0) + 1
                continue
            detailed_count += 1
            for owner_context_id in bundle.owner_context_ids:
                lease = self.bundle_builder.leases.context(
                    owner_context_id, now_ms=now_ms
                )
                condition = lease.condition
                lease_fingerprint = (
                    lease.kind.value,
                    condition.event_kind if condition is not None else None,
                    condition.subject_id if condition is not None else None,
                    condition.condition_id if condition is not None else None,
                    lease.reason,
                )
                lease_identity = (owner_context_id, lease.context_epoch)
                if (
                    self._last_context_lease_fingerprints.get(lease_identity)
                    != lease_fingerprint
                ):
                    self._last_context_lease_fingerprints[
                        lease_identity
                    ] = lease_fingerprint
                    self._bundle_events.append(
                        BundlePreviewEvent(
                            kind="context_lease_issued",
                            ts_ms=now_ms,
                            fields={
                                "context_id": owner_context_id,
                                "context_epoch": lease.context_epoch,
                                "workflow_id": lease.workflow_id,
                                "lease_kind": lease.kind.value,
                                "condition": (
                                    {
                                        "event_kind": condition.event_kind,
                                        "subject_id": condition.subject_id,
                                        "condition_id": condition.condition_id,
                                    }
                                    if condition is not None
                                    else None
                                ),
                                "confidence": lease.confidence,
                                "reason": lease.reason,
                            },
                        )
                    )
            self._bundle_events.append(
                BundlePreviewEvent(
                    kind="bundle_lease_aggregated",
                    ts_ms=now_ms,
                    fields={
                        "context_id": preview.context_id,
                        "context_epoch": preview.context_epoch,
                        "command_kind": preview.command_kind.value,
                        "bundle_id": bundle.bundle_id,
                        "owner_context_count": len(bundle.owner_context_ids),
                        "owner_context_digest": self._string_digest(
                            bundle.owner_context_ids,
                            person=b"bkv-owner-set",
                        ),
                        "bundle_scope": bundle.scope.value,
                        "exclusive_action_bytes": bundle.exclusive_action_bytes,
                        "cross_context_action_bytes": (
                            bundle.cross_context_action_bytes
                        ),
                        "foreign_owner_context_count": len(
                            bundle.foreign_owner_context_ids
                        ),
                        "foreign_owner_context_digest": self._string_digest(
                            bundle.foreign_owner_context_ids,
                            person=b"bkv-foreign-set",
                        ),
                        "strongest_lease_kind": (
                            bundle.lease.strongest_kind.value
                        ),
                        "condition_count": len(bundle.lease.conditions),
                        "condition_digest": self._string_digest(
                            tuple(
                                f"{item.event_kind}:{item.subject_id}:"
                                f"{item.condition_id}"
                                for item in bundle.lease.conditions
                            ),
                            person=b"bkv-condition",
                        ),
                    },
                )
            )
            self._bundle_events.append(
                BundlePreviewEvent(
                    kind="physical_bundle_preview",
                    ts_ms=now_ms,
                    fields={
                        "context_id": preview.context_id,
                        "context_epoch": preview.context_epoch,
                        "command_kind": preview.command_kind.value,
                        "bundle_id": bundle.bundle_id,
                        "generation_fingerprint": bundle.generation_fingerprint,
                        "owner_context_count": len(bundle.owner_context_ids),
                        "owner_context_digest": self._string_digest(
                            bundle.owner_context_ids,
                            person=b"bkv-owner-set",
                        ),
                        "bundle_scope": bundle.scope.value,
                        "exclusive_action_bytes": bundle.exclusive_action_bytes,
                        "cross_context_action_bytes": (
                            bundle.cross_context_action_bytes
                        ),
                        "foreign_owner_context_count": len(
                            bundle.foreign_owner_context_ids
                        ),
                        "foreign_owner_context_digest": self._string_digest(
                            bundle.foreign_owner_context_ids,
                            person=b"bkv-foreign-set",
                        ),
                        "closure_handle_count": len(bundle.handles),
                        "closure_handle_digest": self._handle_digest(
                            bundle.handles
                        ),
                        "physical_unique_bytes": bundle.physical_unique_bytes,
                        "gpu_bytes": bundle.gpu_bytes,
                        "cpu_bytes": bundle.cpu_bytes,
                        "closure_bytes": bundle.closure_bytes,
                        "copy_bytes": preview.copy_bytes,
                        "marginal_reclaimable_bytes": (
                            bundle.marginal_reclaimable_bytes
                        ),
                        "locked_bytes": bundle.locked_bytes,
                        "lease_kind": bundle.lease.strongest_kind.value,
                        "eligible": preview.eligible,
                        "blocker_codes": sorted(
                            {item.code.value for item in preview.blockers}
                        ),
                        "blocker_count": len(preview.blockers),
                        "blocker_histogram": {
                            code: sum(
                                item.code.value == code
                                for item in preview.blockers
                            )
                            for code in sorted(
                                {item.code.value for item in preview.blockers}
                            )
                        },
                        "blocker_required_bytes": sum(
                            item.required_bytes for item in preview.blockers
                        ),
                    },
                )
            )
        if omitted_count:
            self._bundle_events.append(
                BundlePreviewEvent(
                    kind="physical_bundle_preview_summary",
                    ts_ms=now_ms,
                    fields={
                        "detailed_count": detailed_count,
                        "omitted_count": omitted_count,
                        "omitted_closure_bytes": omitted_closure_bytes,
                        "omitted_reclaimable_bytes": omitted_reclaimable_bytes,
                        "omitted_blocker_counts": dict(
                            sorted(omitted_blockers.items())
                        ),
                    },
                )
            )

    @staticmethod
    def _string_digest(values: tuple[str, ...], *, person: bytes) -> str:
        digest = hashlib.blake2b(digest_size=16, person=person)
        for value in sorted(values):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _handle_digest(handles: tuple[PageHandle, ...]) -> str:
        digest = hashlib.blake2b(digest_size=16, person=b"bkv-handle-set")
        for handle in sorted(
            handles,
            key=lambda item: (item.page_id, item.allocation_generation),
        ):
            digest.update(str(handle.page_id).encode("ascii"))
            digest.update(b":")
            digest.update(str(handle.allocation_generation).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _bundle_metadata(preview: PhysicalBundlePreview) -> dict[str, object]:
        bundle = preview.bundle
        return {
            "physical_bundle_id": bundle.bundle_id,
            "physical_bundle_scope": bundle.scope.value,
            "physical_exclusive_action_bytes": bundle.exclusive_action_bytes,
            "physical_cross_context_action_bytes": (
                bundle.cross_context_action_bytes
            ),
            "physical_foreign_owner_context_ids": list(
                bundle.foreign_owner_context_ids
            ),
            "physical_generation_fingerprint": bundle.generation_fingerprint,
            "physical_closure_bytes": bundle.closure_bytes,
            "physical_unique_bytes": bundle.physical_unique_bytes,
            "physical_copy_bytes": preview.copy_bytes,
            "physical_marginal_reclaimable_bytes": (
                bundle.marginal_reclaimable_bytes
            ),
            "physical_locked_bytes": bundle.locked_bytes,
            "physical_lease_kind": bundle.lease.strongest_kind.value,
        }

    def _command_is_eligible(
        self, command: ControlCommand, now_ms: float
    ) -> bool:
        return self.retry_guard is None or self.retry_guard.command_is_eligible(
            command, now_ms=now_ms
        )
