from __future__ import annotations

from math import inf

from beliefkv.core.types import (
    ContinuationBelief,
    DeviceState,
    KVAction,
    KVDecision,
    KVObjectMeta,
    PlannerConfig,
    RuntimeSnapshot,
)
from beliefkv.policy.cost_model import PCIeCostModel, estimate_recompute_ms
from beliefkv.policy.frontier import BeliefFrontier


class BeliefKVPlanner:
    """Plan KV residency actions from a belief frontier and runtime snapshot."""

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self.cost_model = PCIeCostModel(
            bandwidth_gbps=self.config.pcie_bandwidth_gbps,
            overhead_ms=self.config.transfer_overhead_ms,
        )

    def plan(
        self,
        kv_objects: list[KVObjectMeta],
        continuations: list[ContinuationBelief],
        snapshot: RuntimeSnapshot,
    ) -> list[KVDecision]:
        frontier = BeliefFrontier(continuations)
        scored = [self._score(kv, frontier, snapshot) for kv in kv_objects]

        gpu_budget = max(0, snapshot.hbm_capacity_bytes - self.config.reserve_hbm_bytes)
        must_keep_bytes = sum(
            kv.size_bytes
            for kv, decision in scored
            if kv.device_state == DeviceState.GPU and decision.action == KVAction.KEEP_GPU
        )
        remaining_budget = max(0, gpu_budget - must_keep_bytes)

        gpu_candidates = [
            (kv, decision)
            for kv, decision in scored
            if kv.device_state == DeviceState.GPU
            and decision.action == KVAction.NOOP
            and decision.priority > 0
        ]
        gpu_candidates.sort(key=lambda item: item[1].priority / max(1, item[0].size_bytes), reverse=True)

        keep_gpu: set[str] = set()
        for kv, _ in gpu_candidates:
            if kv.size_bytes <= remaining_budget:
                keep_gpu.add(kv.object_id)
                remaining_budget -= kv.size_bytes

        decisions: list[KVDecision] = []
        for kv, decision in scored:
            if kv.device_state == DeviceState.GPU and decision.action == KVAction.NOOP:
                if kv.object_id in keep_gpu:
                    decisions.append(
                        KVDecision(
                            object_id=kv.object_id,
                            action=KVAction.KEEP_GPU,
                            reason="selected_by_benefit_density",
                            priority=decision.priority,
                            next_use_p50_ms=decision.next_use_p50_ms,
                            reuse_probability=decision.reuse_probability,
                            expected_benefit_ms=decision.expected_benefit_ms,
                        )
                    )
                else:
                    decisions.append(self._gpu_victim_action(kv, decision, snapshot))
            else:
                decisions.append(decision)

        return decisions

    def _score(
        self,
        kv: KVObjectMeta,
        frontier: BeliefFrontier,
        snapshot: RuntimeSnapshot,
    ) -> tuple[KVObjectMeta, KVDecision]:
        reuse_probability, next_use_p50, confidence = frontier.match(kv)
        if reuse_probability == 0.0:
            reuse_probability = self.config.default_reuse_probability

        recompute_ms = kv.recompute_cost_ms
        if recompute_ms is None:
            recompute_ms = estimate_recompute_ms(
                kv.token_count, self.config.prefill_tokens_per_ms
            )

        d2h_ms = kv.d2h_cost_ms
        if d2h_ms is None:
            d2h_ms = self.cost_model.transfer_ms(kv.size_bytes)

        h2d_ms = kv.h2d_cost_ms
        if h2d_ms is None:
            h2d_ms = self.cost_model.transfer_ms(kv.size_bytes)

        expected_benefit = reuse_probability * max(0.0, recompute_ms - h2d_ms)
        urgency = 1.0 / max(1.0, next_use_p50)
        priority = expected_benefit * confidence * urgency

        in_active_decode_workflow = bool(kv.workflow_ids & snapshot.active_decode_workflows)
        if kv.is_active_decode or in_active_decode_workflow:
            if kv.device_state == DeviceState.CPU:
                action = KVAction.PREFETCH_GPU
                reason = "decode_protected_prefetch"
            elif kv.device_state == DeviceState.RAW_TEXT:
                action = KVAction.MATERIALIZE
                reason = "decode_protected_materialize"
            else:
                action = KVAction.KEEP_GPU
                reason = "decode_protected"
            return kv, KVDecision(
                object_id=kv.object_id,
                action=action,
                reason=reason,
                priority=1.0e30,
                next_use_p50_ms=next_use_p50,
                reuse_probability=reuse_probability,
                expected_benefit_ms=expected_benefit,
            )

        if (
            kv.device_state == DeviceState.CPU
            and next_use_p50 <= h2d_ms + self.config.prefetch_slack_ms
            and snapshot.hbm_free_bytes >= kv.size_bytes + self.config.reserve_hbm_bytes
        ):
            return kv, KVDecision(
                object_id=kv.object_id,
                action=KVAction.PREFETCH_GPU,
                reason="next_use_within_transfer_window",
                priority=priority,
                next_use_p50_ms=next_use_p50,
                reuse_probability=reuse_probability,
                expected_benefit_ms=expected_benefit,
            )

        if kv.device_state == DeviceState.RAW_TEXT:
            should_materialize = (
                reuse_probability >= self.config.min_branch_probability
                and next_use_p50 <= self.config.decode_protection_ms
            )
            return kv, KVDecision(
                object_id=kv.object_id,
                action=KVAction.MATERIALIZE if should_materialize else KVAction.NOOP,
                reason="raw_text_frontier_check",
                priority=priority,
                next_use_p50_ms=next_use_p50,
                reuse_probability=reuse_probability,
                expected_benefit_ms=expected_benefit,
            )

        return kv, KVDecision(
            object_id=kv.object_id,
            action=KVAction.NOOP,
            reason="candidate",
            priority=priority,
            next_use_p50_ms=next_use_p50,
            reuse_probability=reuse_probability,
            expected_benefit_ms=expected_benefit,
        )

    def _gpu_victim_action(
        self,
        kv: KVObjectMeta,
        decision: KVDecision,
        snapshot: RuntimeSnapshot,
    ) -> KVDecision:
        hbm_pressure_high = snapshot.hbm_pressure >= self.config.high_hbm_pressure
        reuse_low = decision.reuse_probability < self.config.min_branch_probability
        far_next_use = decision.next_use_p50_ms == inf or (
            decision.next_use_p50_ms > self.config.decode_protection_ms
        )

        if hbm_pressure_high and far_next_use and decision.expected_benefit_ms >= self.config.offload_min_benefit_ms:
            return KVDecision(
                object_id=kv.object_id,
                action=KVAction.OFFLOAD_CPU,
                reason="pressure_high_and_next_use_far",
                priority=decision.priority,
                next_use_p50_ms=decision.next_use_p50_ms,
                reuse_probability=decision.reuse_probability,
                expected_benefit_ms=decision.expected_benefit_ms,
            )

        if hbm_pressure_high and reuse_low:
            return KVDecision(
                object_id=kv.object_id,
                action=KVAction.RECOMPUTE_LATER,
                reason="low_survival_under_pressure",
                priority=decision.priority,
                next_use_p50_ms=decision.next_use_p50_ms,
                reuse_probability=decision.reuse_probability,
                expected_benefit_ms=decision.expected_benefit_ms,
            )

        return KVDecision(
            object_id=kv.object_id,
            action=KVAction.DROP_GPU,
            reason="outside_gpu_residency_budget",
            priority=decision.priority,
            next_use_p50_ms=decision.next_use_p50_ms,
            reuse_probability=decision.reuse_probability,
            expected_benefit_ms=decision.expected_benefit_ms,
        )
