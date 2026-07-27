from __future__ import annotations

from dataclasses import dataclass

from beliefkv.policy.reference.base import (
    AdmissionAction,
    AdmissionIntent,
    MetadataMode,
    MetadataRequirement,
    PolicyInput,
    PolicyOutput,
    ResidencyAction,
    ResidencyIntent,
    TransferDependency,
)
from beliefkv.policy.reference.common import (
    blocked_restore_bundles,
    execution_intent,
    fifo_frontier,
    required_restore_bundles,
)


@dataclass(frozen=True)
class ReactivePolicyConfig:
    reserve_bytes: int = 0

    def __post_init__(self) -> None:
        if self.reserve_bytes < 0:
            raise ValueError("reserve_bytes must be non-negative")


class ReactivePolicy:
    """Observed-state LRU/restore baseline on the common physical snapshot."""

    name = "reactive_reference"

    def __init__(self, config: ReactivePolicyConfig | None = None) -> None:
        self.config = config or ReactivePolicyConfig()

    def metadata_requirements(
        self, mode: MetadataMode
    ) -> tuple[MetadataRequirement, ...]:
        del mode
        return ()

    def decide(self, policy_input: PolicyInput) -> PolicyOutput:
        ordered = fifo_frontier(policy_input)
        available = max(
            0,
            policy_input.resources.hbm_available_bytes - self.config.reserve_bytes,
        )
        runnable_contexts = {item.context_id for item in ordered}
        residency: list[ResidencyIntent] = []
        dependencies: list[TransferDependency] = []
        admissions: list[AdmissionIntent] = []
        restored: set[str] = set()

        for invocation in ordered:
            restores = tuple(
                bundle
                for bundle in required_restore_bundles(
                    policy_input, invocation.context_id
                )
                if bundle.bundle_id not in restored
            )
            blocked_restores = blocked_restore_bundles(restores)
            if blocked_restores:
                admissions.append(
                    AdmissionIntent(
                        request_id=invocation.request_id,
                        action=AdmissionAction.DEFER,
                        reserved_bytes=0,
                        required_bundle_ids=tuple(
                            bundle.bundle_id for bundle in blocked_restores
                        ),
                        reason="required restore is physically blocked in this snapshot",
                    )
                )
                continue
            restore_bytes = sum(
                max(0, bundle.physical_unique_bytes - bundle.gpu_bytes)
                for bundle in restores
            )
            needed = invocation.startup_bytes + restore_bytes
            if needed <= available:
                required_ids: list[str] = []
                for bundle in restores:
                    index = len(residency)
                    target = max(0, bundle.physical_unique_bytes - bundle.gpu_bytes)
                    residency.append(
                        ResidencyIntent(
                            bundle_id=bundle.bundle_id,
                            action=ResidencyAction.PREFETCH_GPU,
                            target_bytes=target,
                            deadline_ms=policy_input.resources.ts_ms,
                            scenario_support=frozenset({"observed"}),
                            reason="reactive restore required by the FIFO head",
                        )
                    )
                    dependencies.append(
                        TransferDependency(
                            before_request_id=invocation.request_id,
                            residency_intent_index=index,
                            require_ack=True,
                        )
                    )
                    required_ids.append(bundle.bundle_id)
                    restored.add(bundle.bundle_id)
                action = (
                    AdmissionAction.RESTORE_THEN_ADMIT
                    if required_ids
                    else AdmissionAction.ADMIT
                )
                available -= needed
                admissions.append(
                    AdmissionIntent(
                        request_id=invocation.request_id,
                        action=action,
                        reserved_bytes=needed,
                        required_bundle_ids=tuple(required_ids),
                        reason="FIFO request fits observed HBM capacity",
                    )
                )
            else:
                admissions.append(
                    AdmissionIntent(
                        request_id=invocation.request_id,
                        action=AdmissionAction.DEFER,
                        reserved_bytes=0,
                        required_bundle_ids=tuple(
                            bundle.bundle_id for bundle in restores
                        ),
                        reason="FIFO request does not fit observed HBM capacity",
                    )
                )

        deferred_shortage = sum(
            next(
                item.startup_bytes
                for item in ordered
                if item.request_id == admission.request_id
            )
            for admission in admissions
            if admission.action == AdmissionAction.DEFER
        )
        host_free = policy_input.resources.host_free_bytes
        victims = sorted(
            (
                bundle
                for bundle in policy_input.physical_kv.bundles
                if bundle.gpu_bytes > 0
                and bundle.marginal_reclaimable_bytes > 0
                and bundle.locked_bytes == 0
                and not runnable_contexts.intersection(bundle.owner_context_ids)
            ),
            key=lambda bundle: (
                bundle.last_access_ms,
                -bundle.marginal_reclaimable_bytes,
                bundle.bundle_id,
            ),
        )
        reclaimed = 0
        existing = {item.bundle_id for item in residency}
        for bundle in victims:
            if reclaimed >= deferred_shortage or bundle.bundle_id in existing:
                break
            host_needed = max(0, bundle.gpu_bytes - bundle.cpu_bytes)
            if host_needed <= host_free:
                action = ResidencyAction.COMMIT_CPU
                host_free -= host_needed
                reason = "LRU inactive bundle selected for reactive CPU offload"
            else:
                action = ResidencyAction.DROP
                reason = "LRU inactive bundle selected for reactive recomputation"
            residency.append(
                ResidencyIntent(
                    bundle_id=bundle.bundle_id,
                    action=action,
                    target_bytes=bundle.marginal_reclaimable_bytes,
                    deadline_ms=policy_input.resources.ts_ms,
                    scenario_support=frozenset({"observed"}),
                    reason=reason,
                )
            )
            reclaimed += bundle.marginal_reclaimable_bytes

        return PolicyOutput(
            execution=execution_intent(
                policy_input,
                ordered,
                mode="reactive_fifo_lru",
                reason="FIFO execution with observed-capacity admission and LRU reclaim",
            ),
            admissions=tuple(admissions),
            residency=tuple(residency),
            dependencies=tuple(dependencies),
            policy_name=self.name,
            metadata_assumptions=(
                "observed:runnable_frontier",
                "observed:physical_kv",
                "observed:resources",
            ),
            input_snapshot_id=policy_input.snapshot_id,
            metadata_mode=policy_input.metadata_mode,
        )
