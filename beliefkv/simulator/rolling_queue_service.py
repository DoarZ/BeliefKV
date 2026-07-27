from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from beliefkv.policy.reference import PolicyInput
from beliefkv.policy.scenario_physicalizer import ScenarioDemand
from beliefkv.policy.whatif_packer import ScenarioPlan
from beliefkv.simulator.queue_service import (
    CounterfactualSimulationError,
    CounterfactualSimulationResult,
    FrozenCounterfactualWorkload,
    FrozenRequestDemand,
    FrozenWorkflowDemand,
    QueueServiceModel,
    _causal_blocked_critical_path_ms,
)
from beliefkv.simulator.rolling_physical import (
    FutureRadixUse,
    ResidencyReplayMode,
    RollingMaterialization,
    RollingPhysicalReplayError,
    RollingRadixAllocator,
)


@dataclass
class _RollingRequestState:
    demand: FrozenRequestDemand
    future_use: FutureRadixUse
    prefill_remaining: int = 0
    prefill_total: int = 0
    decode_remaining: int = 0
    ready_at_ms: float | None = None
    admitted_at_ms: float | None = None
    first_service_ms: float | None = None
    finish_ms: float | None = None
    cache_hit_tokens: int = 0
    decoded_tokens: int = 0
    prefill_chunk_index: int = 0
    action_unlock_ms: float | None = None
    unique_commit_growth_tokens: int = 0


class RollingCounterfactualQueueServiceSimulator:
    """Recompute queue service and two-tier Radix state after every quantum."""

    def __init__(
        self,
        service_model: QueueServiceModel,
        *,
        residency_mode: ResidencyReplayMode,
    ) -> None:
        self.service_model = service_model
        self.residency_mode = residency_mode

    def simulate(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        workload: FrozenCounterfactualWorkload,
    ) -> CounterfactualSimulationResult:
        self._validate_inputs(policy_input, demand, plan, workload)
        bytes_per_token = int(workload.metadata["kv_bytes_per_token"])
        allocator = RollingRadixAllocator(
            hbm_capacity_bytes=policy_input.resources.hbm_capacity_bytes,
            hbm_fixed_bytes=policy_input.resources.hbm_reserved_bytes,
            host_capacity_bytes=policy_input.resources.host_free_bytes,
            bytes_per_token=bytes_per_token,
            d2h_bytes_per_ms=policy_input.resources.d2h_service_bytes_per_ms,
            h2d_bytes_per_ms=policy_input.resources.h2d_service_bytes_per_ms,
            transfer_setup_ms=policy_input.resources.transfer_setup_p50_ms,
            mode=self.residency_mode,
        )
        states = {
            request.request_id: _RollingRequestState(
                demand=request,
                future_use=FutureRadixUse(
                    request.request_id,
                    request.prompt_token_symbols,
                ),
                decode_remaining=request.output_tokens,
            )
            for request in workload.requests
        }
        workflows = {item.workflow_id: item for item in workload.workflows}
        rank = {
            request_id: index for index, request_id in enumerate(plan.execution_order)
        }
        now_ms = 0.0
        scheduler_steps = 0
        prefer_decode = False
        last_blockers: dict[str, str] = {}

        while any(state.finish_ms is None for state in states.values()):
            self._release_requests(states, workflows, now_ms)
            active_count = sum(
                state.admitted_at_ms is not None and state.finish_ms is None
                for state in states.values()
            )
            slots = max(0, self.service_model.max_decode_batch - active_count)
            ready = sorted(
                (
                    state
                    for state in states.values()
                    if state.ready_at_ms is not None
                    and state.admitted_at_ms is None
                    and state.finish_ms is None
                ),
                key=lambda state: self._priority_key(state, rank),
            )
            for state in ready:
                if slots <= 0:
                    break
                future = self._future_uses(states, rank, exclude={state.demand.request_id})
                try:
                    admission = allocator.admit(
                        request_id=state.demand.request_id,
                        prompt_token_symbols=state.demand.prompt_token_symbols,
                        now_ms=now_ms,
                        future_uses=future,
                    )
                except RollingPhysicalReplayError as error:
                    last_blockers[state.demand.request_id] = str(error)
                    continue
                now_ms = admission.ready_ms
                state.cache_hit_tokens = admission.match.logical_hit_tokens
                state.prefill_total = (
                    len(state.demand.prompt_token_symbols)
                    - state.cache_hit_tokens
                )
                state.prefill_remaining = state.prefill_total
                state.admitted_at_ms = now_ms
                last_blockers.pop(state.demand.request_id, None)
                slots -= 1

            for state in tuple(states.values()):
                if (
                    state.admitted_at_ms is not None
                    and state.finish_ms is None
                    and state.prefill_remaining == 0
                    and state.decode_remaining == 0
                ):
                    self._mark_first_service(state, now_ms)
                    now_ms = self._finish_request(
                        state,
                        now_ms,
                        allocator,
                        states,
                        rank,
                    )

            admitted = [
                state
                for state in states.values()
                if state.admitted_at_ms is not None and state.finish_ms is None
            ]
            prefill = sorted(
                (state for state in admitted if state.prefill_remaining > 0),
                key=lambda state: self._priority_key(state, rank),
            )
            decode = sorted(
                (
                    state
                    for state in admitted
                    if state.prefill_remaining == 0 and state.decode_remaining > 0
                ),
                key=lambda state: self._priority_key(state, rank),
            )
            progressed = False
            if prefill and (not decode or not prefer_decode):
                progressed, now_ms = self._serve_prefill(
                    prefill,
                    now_ms,
                    allocator,
                    states,
                    rank,
                )
                if progressed:
                    scheduler_steps += 1
                    prefer_decode = True
            if not progressed and decode:
                progressed, now_ms = self._serve_decode(
                    decode,
                    now_ms,
                    allocator,
                    states,
                    rank,
                )
                if progressed:
                    scheduler_steps += 1
                    prefer_decode = False
            if not progressed and prefill:
                progressed, now_ms = self._serve_prefill(
                    prefill,
                    now_ms,
                    allocator,
                    states,
                    rank,
                )
                if progressed:
                    scheduler_steps += 1
                    prefer_decode = True
            if progressed:
                continue

            next_release = self._next_release_time(states, workflows)
            if next_release is not None and next_release > now_ms + 1e-12:
                now_ms = next_release
                continue
            blocked = sorted(
                state.demand.request_id
                for state in states.values()
                if state.finish_ms is None
            )
            details = "; ".join(
                f"{request_id}={reason}"
                for request_id, reason in sorted(last_blockers.items())
            )
            raise CounterfactualSimulationError(
                "rolling allocator/dependency deadlock while requests remain: "
                + ",".join(blocked)
                + (f"; blockers: {details}" if details else "")
            )

        allocator.radix.assert_invariants()
        workflow_jct: dict[str, float] = {}
        for workflow in workload.workflows:
            terminal = max(
                states[request_id].finish_ms or 0.0
                for request_id in workflow.terminal_request_ids
            )
            completion = terminal + workflow.completion_delay_ms
            workflow_jct[workflow.workflow_id] = completion - workflow.release_ms
            now_ms = max(now_ms, completion)
        queue_wait = {
            request_id: max(
                0.0,
                (state.first_service_ms or state.finish_ms or 0.0)
                - (state.ready_at_ms or 0.0),
            )
            for request_id, state in states.items()
        }
        action_unlock = {
            request_id: state.action_unlock_ms
            for request_id, state in states.items()
            if state.action_unlock_ms is not None
        }
        return CounterfactualSimulationResult(
            workflow_jct_ms=MappingProxyType(dict(sorted(workflow_jct.items()))),
            request_finish_ms=MappingProxyType(
                {
                    request_id: float(state.finish_ms or 0.0)
                    for request_id, state in sorted(states.items())
                }
            ),
            request_queue_wait_ms=MappingProxyType(dict(sorted(queue_wait.items()))),
            request_action_unlock_ms=MappingProxyType(dict(sorted(action_unlock.items()))),
            causal_blocked_ms=_causal_blocked_critical_path_ms(workload),
            unhidden_stall_ms=sum(queue_wait.values()),
            final_ts_ms=now_ms,
            hbm_peak_bytes=allocator.hbm_peak_bytes,
            hbm_final_bytes=allocator.hbm_occupied_bytes,
            host_consumed_bytes=allocator.host_occupied_bytes,
            d2h_bytes=allocator.d2h_bytes,
            h2d_bytes=allocator.h2d_bytes,
            pcie_busy_ms=allocator.pcie_busy_ms,
            scheduler_steps=scheduler_steps,
            transition_hash=workload.transition_hash,
            radix_demand_recomputed=True,
            recomputed_cache_hit_tokens=MappingProxyType(
                {
                    request_id: state.cache_hit_tokens
                    for request_id, state in sorted(states.items())
                }
            ),
            recomputed_unique_growth_bytes=MappingProxyType(
                {
                    request_id: state.unique_commit_growth_tokens * bytes_per_token
                    for request_id, state in sorted(states.items())
                }
            ),
            rolling_physical_replay=True,
            physical_timeline=tuple(event.to_dict() for event in allocator.events),
        )

    def _serve_prefill(
        self,
        candidates: Sequence[_RollingRequestState],
        now_ms: float,
        allocator: RollingRadixAllocator,
        states: Mapping[str, _RollingRequestState],
        rank: Mapping[str, int],
    ) -> tuple[bool, float]:
        for state in candidates:
            tokens = min(
                state.prefill_remaining,
                self.service_model.prefill_chunk_tokens,
            )
            processed = state.prefill_total - state.prefill_remaining + tokens
            visible = state.cache_hit_tokens + processed
            path = state.demand.prompt_token_symbols[:visible]
            try:
                materialized = allocator.materialize_batch(
                    ((state.demand.request_id, path),),
                    now_ms=now_ms,
                    future_uses=self._future_uses(
                        states, rank, exclude={state.demand.request_id}
                    ),
                    reason="prefill_chunk",
                )
            except RollingPhysicalReplayError:
                continue
            self._record_growth(state, materialized)
            start_ms = materialized.ready_ms
            self._mark_first_service(state, start_ms)
            elapsed = self.service_model.prefill_elapsed_ms(
                tokens,
                chunk_index=state.prefill_chunk_index,
            )
            state.prefill_remaining -= tokens
            state.prefill_chunk_index += 1
            end_ms = start_ms + elapsed
            if state.prefill_remaining == 0 and state.decode_remaining == 0:
                end_ms = self._finish_request(
                    state, end_ms, allocator, states, rank
                )
            return True, end_ms
        return False, now_ms

    def _serve_decode(
        self,
        candidates: Sequence[_RollingRequestState],
        now_ms: float,
        allocator: RollingRadixAllocator,
        states: Mapping[str, _RollingRequestState],
        rank: Mapping[str, int],
    ) -> tuple[bool, float]:
        window = tuple(candidates[: self.service_model.max_decode_batch])
        attempts = [window[:size] for size in range(len(window), 0, -1)]
        attempts.extend((state,) for state in window[1:])
        seen: set[tuple[str, ...]] = set()
        for batch in attempts:
            identity = tuple(state.demand.request_id for state in batch)
            if identity in seen:
                continue
            seen.add(identity)
            progressed = {
                state.demand.request_id: min(
                    state.decode_remaining,
                    self.service_model.decode_quantum_tokens,
                )
                for state in batch
            }
            paths = []
            for state in batch:
                projected = state.decoded_tokens + progressed[state.demand.request_id]
                visible = min(
                    len(state.demand.cache_commit_token_symbols),
                    len(state.demand.prompt_token_symbols) + max(0, projected - 1),
                )
                paths.append(
                    (
                        state.demand.request_id,
                        state.demand.cache_commit_token_symbols[:visible],
                    )
                )
            try:
                materialized = allocator.materialize_batch(
                    tuple(paths),
                    now_ms=now_ms,
                    future_uses=self._future_uses(states, rank, exclude=set(identity)),
                    reason="decode_quantum",
                )
            except RollingPhysicalReplayError:
                continue
            for state in batch:
                self._record_growth(state, materialized)
            total_tokens = sum(progressed.values())
            elapsed = (
                self.service_model.decode_launch_ms
                + total_tokens / self.service_model.decode_rate(len(batch))
            )
            start_ms = materialized.ready_ms
            for state in batch:
                self._mark_first_service(state, start_ms)
                tokens = progressed[state.demand.request_id]
                state.decode_remaining -= tokens
                state.decoded_tokens += tokens
                boundary = state.demand.action_boundary_token_index
                if (
                    boundary is not None
                    and state.action_unlock_ms is None
                    and state.decoded_tokens >= boundary
                ):
                    fraction = (
                        0.0
                        if tokens == 0
                        else (
                            tokens - (state.decoded_tokens - boundary)
                        )
                        / tokens
                    )
                    state.action_unlock_ms = start_ms + elapsed * fraction
            end_ms = start_ms + elapsed
            for state in batch:
                if state.decode_remaining == 0:
                    end_ms = self._finish_request(
                        state, end_ms, allocator, states, rank
                    )
            return True, end_ms
        return False, now_ms

    def _finish_request(
        self,
        state: _RollingRequestState,
        now_ms: float,
        allocator: RollingRadixAllocator,
        states: Mapping[str, _RollingRequestState],
        rank: Mapping[str, int],
    ) -> float:
        if state.finish_ms is not None:
            raise CounterfactualSimulationError("request completed twice")
        materialized = allocator.materialize_batch(
            ((state.demand.request_id, state.demand.cache_commit_token_symbols),),
            now_ms=now_ms,
            future_uses=self._future_uses(
                states, rank, exclude={state.demand.request_id}
            ),
            reason="request_finish",
        )
        self._record_growth(state, materialized)
        allocator.complete_request(
            request_id=state.demand.request_id,
            context_id=state.demand.context_id,
            final_path=state.demand.cache_commit_token_symbols,
        )
        state.finish_ms = materialized.ready_ms
        if (
            state.demand.action_boundary_token_index is not None
            and state.action_unlock_ms is None
        ):
            state.action_unlock_ms = materialized.ready_ms
        return materialized.ready_ms

    @staticmethod
    def _record_growth(
        state: _RollingRequestState,
        materialized: RollingMaterialization,
    ) -> None:
        state.unique_commit_growth_tokens += materialized.unique_growth_tokens.get(
            state.demand.request_id, 0
        )

    @staticmethod
    def _validate_inputs(
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        workload: FrozenCounterfactualWorkload,
    ) -> None:
        if not plan.feasible:
            raise CounterfactualSimulationError("cannot simulate an infeasible plan")
        if plan.snapshot_id != policy_input.snapshot_id or demand.snapshot_id != policy_input.snapshot_id:
            raise CounterfactualSimulationError("rolling replay uses a stale snapshot")
        if not plan.physical_accounting_exact or not demand.physical_accounting_exact:
            raise CounterfactualSimulationError("rolling replay requires exact extent accounting")
        if not workload.prefix_identity_complete or not workload.initial_radix_state_known:
            raise CounterfactualSimulationError(
                "rolling replay requires complete token paths and a known initial Radix epoch"
            )
        if int(workload.metadata.get("kv_bytes_per_token", 0)) <= 0:
            raise CounterfactualSimulationError(
                "rolling replay requires kv_bytes_per_token metadata"
            )
        if (
            policy_input.physical_kv.gpu_bytes
            or policy_input.physical_kv.cpu_bytes
            or policy_input.physical_kv.bundles
            or policy_input.resources.hbm_used_bytes
        ):
            raise CounterfactualSimulationError(
                "rolling replay currently requires an empty cache-reset snapshot"
            )
        if plan.bundle_actions:
            raise CounterfactualSimulationError(
                "rolling replay regenerates physical actions and rejects static bundle actions"
            )
        request_ids = {item.request_id for item in workload.requests}
        unknown = set(plan.execution_order) - request_ids
        if unknown:
            raise CounterfactualSimulationError(
                f"execution plan references unknown requests: {unknown}"
            )

    @staticmethod
    def _priority_key(
        state: _RollingRequestState,
        rank: Mapping[str, int],
    ) -> tuple[float, float, str]:
        return (
            float(rank.get(state.demand.request_id, len(rank))),
            float(state.ready_at_ms or 0.0),
            state.demand.request_id,
        )

    @staticmethod
    def _mark_first_service(state: _RollingRequestState, now_ms: float) -> None:
        if state.first_service_ms is None:
            state.first_service_ms = now_ms
        if state.demand.action_boundary_token_index == 0:
            state.action_unlock_ms = state.action_unlock_ms or now_ms

    @staticmethod
    def _release_requests(
        states: Mapping[str, _RollingRequestState],
        workflows: Mapping[str, FrozenWorkflowDemand],
        now_ms: float,
    ) -> None:
        for state in states.values():
            if state.ready_at_ms is not None or state.finish_ms is not None:
                continue
            predecessors = [states[item] for item in state.demand.predecessor_request_ids]
            if any(item.finish_ms is None for item in predecessors):
                continue
            base = (
                max(float(item.finish_ms) for item in predecessors)
                if predecessors
                else workflows[state.demand.workflow_id].release_ms
            )
            ready = base + state.demand.release_delay_ms
            if ready <= now_ms + 1e-12:
                state.ready_at_ms = ready

    @staticmethod
    def _next_release_time(
        states: Mapping[str, _RollingRequestState],
        workflows: Mapping[str, FrozenWorkflowDemand],
    ) -> float | None:
        result = []
        for state in states.values():
            if state.ready_at_ms is not None or state.finish_ms is not None:
                continue
            predecessors = [states[item] for item in state.demand.predecessor_request_ids]
            if any(item.finish_ms is None for item in predecessors):
                continue
            base = (
                max(float(item.finish_ms) for item in predecessors)
                if predecessors
                else workflows[state.demand.workflow_id].release_ms
            )
            result.append(base + state.demand.release_delay_ms)
        return min(result) if result else None

    @staticmethod
    def _future_uses(
        states: Mapping[str, _RollingRequestState],
        rank: Mapping[str, int],
        *,
        exclude: set[str],
    ) -> tuple[FutureRadixUse, ...]:
        future = sorted(
            (
                state
                for state in states.values()
                if state.finish_ms is None and state.demand.request_id not in exclude
            ),
            key=lambda state: (
                rank.get(state.demand.request_id, len(rank)),
                state.demand.request_id,
            ),
        )
        return tuple(
            state.future_use
            for state in future
        )


__all__ = ["RollingCounterfactualQueueServiceSimulator"]
