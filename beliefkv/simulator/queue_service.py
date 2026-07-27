from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import dataclass, field
from hashlib import blake2b
from types import MappingProxyType
from typing import Mapping, Sequence

from beliefkv.policy.joint_oracle import (
    OracleArm,
    OracleCost,
    ResimulatedPlanEvaluation,
    ResimulationEvidence,
)
from beliefkv.policy.reference import PolicyInput, ResidencyAction
from beliefkv.policy.scenario_physicalizer import ScenarioDemand
from beliefkv.policy.whatif_packer import ScenarioPlan


class CounterfactualSimulationError(RuntimeError):
    """Raised when a counterfactual cannot be evaluated without fabrication."""


@dataclass(frozen=True)
class FrozenRequestDemand:
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    predecessor_request_ids: tuple[str, ...]
    release_delay_ms: float
    uncached_prompt_tokens: int
    output_tokens: int
    startup_bytes: int
    kv_growth_bytes: int
    required_bundle_ids: tuple[str, ...] = ()
    action_boundary_token_index: int | None = None
    tool_duration_ms: float = 0.0
    observed_cache_hit_tokens: int = 0
    prompt_token_symbols: tuple[int, ...] = ()
    cache_commit_token_symbols: tuple[int, ...] = ()
    partial_cache_commit_token_symbols: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in ("request_id", "workflow_id", "invocation_id", "context_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.context_epoch < 0:
            raise ValueError("context_epoch must be non-negative")
        if len(set(self.predecessor_request_ids)) != len(
            self.predecessor_request_ids
        ):
            raise ValueError("request predecessors must be unique")
        if len(set(self.required_bundle_ids)) != len(self.required_bundle_ids):
            raise ValueError("required bundle IDs must be unique")
        numeric = (
            self.release_delay_ms,
            self.uncached_prompt_tokens,
            self.output_tokens,
            self.startup_bytes,
            self.kv_growth_bytes,
            self.tool_duration_ms,
            self.observed_cache_hit_tokens,
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in numeric):
            raise ValueError("frozen request demand must be finite and non-negative")
        if self.kv_growth_bytes > self.startup_bytes:
            raise ValueError("KV growth cannot exceed the request startup reservation")
        if self.action_boundary_token_index is not None and not (
            0 <= self.action_boundary_token_index <= self.output_tokens
        ):
            raise ValueError("action boundary must lie within frozen output demand")
        if self.observed_cache_hit_tokens > (
            len(self.prompt_token_symbols)
            if self.prompt_token_symbols
            else self.uncached_prompt_tokens + self.observed_cache_hit_tokens
        ):
            raise ValueError("observed cache hit exceeds the prompt token path")
        for path in (
            self.prompt_token_symbols,
            self.cache_commit_token_symbols,
            *self.partial_cache_commit_token_symbols,
        ):
            if any(symbol < 0 or symbol >= 1 << 64 for symbol in path):
                raise ValueError("token symbols must be unsigned 64-bit integers")
        if self.cache_commit_token_symbols and self.prompt_token_symbols:
            shared = min(
                len(self.prompt_token_symbols),
                len(self.cache_commit_token_symbols),
            )
            if (
                self.cache_commit_token_symbols[:shared]
                != self.prompt_token_symbols[:shared]
            ):
                raise ValueError("cache commit path does not extend the request prompt")
        for partial in self.partial_cache_commit_token_symbols:
            if (
                self.cache_commit_token_symbols
                and self.cache_commit_token_symbols[: len(partial)] != partial
            ):
                raise ValueError("partial cache commit is not a final-path prefix")
        object.__setattr__(
            self,
            "predecessor_request_ids",
            tuple(sorted(self.predecessor_request_ids)),
        )
        object.__setattr__(
            self, "required_bundle_ids", tuple(sorted(self.required_bundle_ids))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "workflow_id": self.workflow_id,
            "invocation_id": self.invocation_id,
            "context_id": self.context_id,
            "context_epoch": self.context_epoch,
            "predecessor_request_ids": list(self.predecessor_request_ids),
            "release_delay_ms": self.release_delay_ms,
            "uncached_prompt_tokens": self.uncached_prompt_tokens,
            "output_tokens": self.output_tokens,
            "startup_bytes": self.startup_bytes,
            "kv_growth_bytes": self.kv_growth_bytes,
            "required_bundle_ids": list(self.required_bundle_ids),
            "action_boundary_token_index": self.action_boundary_token_index,
            "tool_duration_ms": self.tool_duration_ms,
            "observed_cache_hit_tokens": self.observed_cache_hit_tokens,
            "prompt_token_symbols_b64": _encode_token_symbols(
                self.prompt_token_symbols
            ),
            "cache_commit_token_symbols_b64": _encode_token_symbols(
                self.cache_commit_token_symbols
            ),
            "partial_cache_commit_token_symbols_b64": [
                _encode_token_symbols(item)
                for item in self.partial_cache_commit_token_symbols
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "FrozenRequestDemand":
        return cls(
            request_id=str(raw["request_id"]),
            workflow_id=str(raw["workflow_id"]),
            invocation_id=str(raw["invocation_id"]),
            context_id=str(raw["context_id"]),
            context_epoch=int(raw["context_epoch"]),
            predecessor_request_ids=tuple(
                str(item) for item in raw.get("predecessor_request_ids", ())
            ),
            release_delay_ms=float(raw["release_delay_ms"]),
            uncached_prompt_tokens=int(raw["uncached_prompt_tokens"]),
            output_tokens=int(raw["output_tokens"]),
            startup_bytes=int(raw["startup_bytes"]),
            kv_growth_bytes=int(raw["kv_growth_bytes"]),
            required_bundle_ids=tuple(
                str(item) for item in raw.get("required_bundle_ids", ())
            ),
            action_boundary_token_index=(
                int(raw["action_boundary_token_index"])
                if raw.get("action_boundary_token_index") is not None
                else None
            ),
            tool_duration_ms=float(raw.get("tool_duration_ms", 0.0)),
            observed_cache_hit_tokens=int(
                raw.get("observed_cache_hit_tokens", 0)
            ),
            prompt_token_symbols=_decode_token_symbols(
                raw.get("prompt_token_symbols_b64")
            ),
            cache_commit_token_symbols=_decode_token_symbols(
                raw.get("cache_commit_token_symbols_b64")
            ),
            partial_cache_commit_token_symbols=tuple(
                _decode_token_symbols(item)
                for item in raw.get(
                    "partial_cache_commit_token_symbols_b64", ()
                )
            ),
        )


@dataclass(frozen=True)
class FrozenWorkflowDemand:
    workflow_id: str
    release_ms: float
    terminal_request_ids: tuple[str, ...]
    completion_delay_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.workflow_id or not self.terminal_request_ids:
            raise ValueError("workflow and terminal request IDs must be non-empty")
        if len(set(self.terminal_request_ids)) != len(self.terminal_request_ids):
            raise ValueError("terminal request IDs must be unique")
        if (
            not math.isfinite(self.release_ms)
            or not math.isfinite(self.completion_delay_ms)
            or self.release_ms < 0
            or self.completion_delay_ms < 0
        ):
            raise ValueError("workflow timing must be finite and non-negative")
        object.__setattr__(
            self, "terminal_request_ids", tuple(sorted(self.terminal_request_ids))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_id": self.workflow_id,
            "release_ms": self.release_ms,
            "terminal_request_ids": list(self.terminal_request_ids),
            "completion_delay_ms": self.completion_delay_ms,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "FrozenWorkflowDemand":
        return cls(
            workflow_id=str(raw["workflow_id"]),
            release_ms=float(raw["release_ms"]),
            terminal_request_ids=tuple(
                str(item) for item in raw.get("terminal_request_ids", ())
            ),
            completion_delay_ms=float(raw.get("completion_delay_ms", 0.0)),
        )


@dataclass(frozen=True)
class FrozenCounterfactualWorkload:
    trace_id: str
    transition_hash: str
    trace_sensitivity: str
    requests: tuple[FrozenRequestDemand, ...]
    workflows: tuple[FrozenWorkflowDemand, ...]
    semantic_events_frozen: bool = True
    token_demand_frozen: bool = True
    tool_duration_frozen: bool = True
    future_physical_growth_exact: bool = False
    prefix_identity_complete: bool = False
    initial_radix_state_known: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id or not self.transition_hash:
            raise ValueError("frozen workload IDs must be non-empty")
        if self.trace_sensitivity not in {
            "schedule_invariant",
            "timing_sensitive",
            "semantic_race_sensitive",
        }:
            raise ValueError("invalid trace sensitivity")
        request_by_id = {item.request_id: item for item in self.requests}
        if len(request_by_id) != len(self.requests):
            raise ValueError("frozen request IDs must be unique")
        workflow_by_id = {item.workflow_id: item for item in self.workflows}
        if len(workflow_by_id) != len(self.workflows):
            raise ValueError("frozen workflow IDs must be unique")
        for request in self.requests:
            if request.workflow_id not in workflow_by_id:
                raise ValueError(f"request has unknown workflow: {request.request_id}")
            unknown = set(request.predecessor_request_ids) - set(request_by_id)
            if unknown:
                raise ValueError(
                    f"request {request.request_id} has unknown predecessors: {unknown}"
                )
            if self.prefix_identity_complete and (
                not request.prompt_token_symbols
                or not request.cache_commit_token_symbols
            ):
                raise ValueError(
                    "complete prefix identity requires prompt and final cache paths"
                )
        for workflow in self.workflows:
            for request_id in workflow.terminal_request_ids:
                request = request_by_id.get(request_id)
                if request is None or request.workflow_id != workflow.workflow_id:
                    raise ValueError("workflow terminal request identity is invalid")
        _assert_acyclic(request_by_id)
        object.__setattr__(
            self, "requests", tuple(sorted(self.requests, key=lambda item: item.request_id))
        )
        object.__setattr__(
            self,
            "workflows",
            tuple(sorted(self.workflows, key=lambda item: item.workflow_id)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "trace_id": self.trace_id,
            "transition_hash": self.transition_hash,
            "trace_sensitivity": self.trace_sensitivity,
            "semantic_events_frozen": self.semantic_events_frozen,
            "token_demand_frozen": self.token_demand_frozen,
            "tool_duration_frozen": self.tool_duration_frozen,
            "future_physical_growth_exact": self.future_physical_growth_exact,
            "prefix_identity_complete": self.prefix_identity_complete,
            "initial_radix_state_known": self.initial_radix_state_known,
            "requests": [item.to_dict() for item in self.requests],
            "workflows": [item.to_dict() for item in self.workflows],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, object]
    ) -> "FrozenCounterfactualWorkload":
        if int(raw.get("schema_version", 0)) not in {1, 2}:
            raise ValueError("unsupported frozen workload schema")
        return cls(
            trace_id=str(raw["trace_id"]),
            transition_hash=str(raw["transition_hash"]),
            trace_sensitivity=str(raw["trace_sensitivity"]),
            semantic_events_frozen=bool(raw.get("semantic_events_frozen", False)),
            token_demand_frozen=bool(raw.get("token_demand_frozen", False)),
            tool_duration_frozen=bool(raw.get("tool_duration_frozen", False)),
            future_physical_growth_exact=bool(
                raw.get("future_physical_growth_exact", False)
            ),
            prefix_identity_complete=bool(
                raw.get("prefix_identity_complete", False)
            ),
            initial_radix_state_known=bool(
                raw.get("initial_radix_state_known", False)
            ),
            requests=tuple(
                FrozenRequestDemand.from_dict(item)
                for item in raw.get("requests", ())  # type: ignore[arg-type]
            ),
            workflows=tuple(
                FrozenWorkflowDemand.from_dict(item)
                for item in raw.get("workflows", ())  # type: ignore[arg-type]
            ),
            metadata=dict(raw.get("metadata", {})),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class QueueServiceModel:
    model_id: str
    prefill_tokens_per_ms: float
    decode_tokens_per_ms: float
    decode_batch_efficiency: tuple[float, ...] = (1.0,)
    max_decode_batch: int = 8
    prefill_chunk_tokens: int = 2_048
    decode_quantum_tokens: int = 8
    prefill_launch_ms: float = 0.0
    decode_launch_ms: float = 0.0
    prefill_first_chunk_curve: tuple[tuple[int, float], ...] = ()
    prefill_continuation_chunk_curve: tuple[tuple[int, float], ...] = ()
    calibrated: bool = False
    calibration_source: str = "unspecified"
    physical_model_id: str = "exact_bundle_allocator_v1"

    def __post_init__(self) -> None:
        if not self.model_id or not self.physical_model_id:
            raise ValueError("service and physical model IDs must be non-empty")
        if self.prefill_tokens_per_ms <= 0 or self.decode_tokens_per_ms <= 0:
            raise ValueError("service rates must be positive")
        if (
            self.max_decode_batch <= 0
            or self.prefill_chunk_tokens <= 0
            or self.decode_quantum_tokens <= 0
        ):
            raise ValueError("scheduler batch and quantum limits must be positive")
        if not self.decode_batch_efficiency or any(
            not math.isfinite(value) or value <= 0
            for value in self.decode_batch_efficiency
        ):
            raise ValueError("decode batch efficiency values must be positive")
        if self.prefill_launch_ms < 0 or self.decode_launch_ms < 0:
            raise ValueError("launch costs must be non-negative")
        self._validate_curve(self.prefill_first_chunk_curve, "first prefill")
        self._validate_curve(
            self.prefill_continuation_chunk_curve,
            "continuation prefill",
        )

    @staticmethod
    def _validate_curve(
        curve: tuple[tuple[int, float], ...], name: str
    ) -> None:
        if not curve:
            return
        tokens = [item[0] for item in curve]
        if tokens != sorted(tokens) or len(tokens) != len(set(tokens)):
            raise ValueError(f"{name} curve token points must be unique and sorted")
        if any(
            token_count <= 0
            or not math.isfinite(elapsed_ms)
            or elapsed_ms <= 0
            for token_count, elapsed_ms in curve
        ):
            raise ValueError(f"{name} curve points must be positive")

    def decode_rate(self, batch_size: int) -> float:
        if batch_size <= 0:
            raise ValueError("decode batch size must be positive")
        index = min(batch_size, len(self.decode_batch_efficiency)) - 1
        return self.decode_tokens_per_ms * self.decode_batch_efficiency[index]

    def prefill_elapsed_ms(self, tokens: int, *, chunk_index: int) -> float:
        if tokens <= 0 or chunk_index < 0:
            raise ValueError("prefill demand and chunk index must be non-negative")
        curve = (
            self.prefill_first_chunk_curve
            if chunk_index == 0
            else self.prefill_continuation_chunk_curve
        )
        if curve:
            return _interpolate_service_curve(
                curve,
                tokens,
                proportional_below_minimum=chunk_index > 0,
                fallback_floor_ms=self.prefill_launch_ms,
            )
        return self.prefill_launch_ms + tokens / self.prefill_tokens_per_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "model_id": self.model_id,
            "prefill_tokens_per_ms": self.prefill_tokens_per_ms,
            "decode_tokens_per_ms": self.decode_tokens_per_ms,
            "decode_batch_efficiency": list(self.decode_batch_efficiency),
            "max_decode_batch": self.max_decode_batch,
            "prefill_chunk_tokens": self.prefill_chunk_tokens,
            "decode_quantum_tokens": self.decode_quantum_tokens,
            "prefill_launch_ms": self.prefill_launch_ms,
            "decode_launch_ms": self.decode_launch_ms,
            "prefill_first_chunk_curve": [
                [tokens, elapsed_ms]
                for tokens, elapsed_ms in self.prefill_first_chunk_curve
            ],
            "prefill_continuation_chunk_curve": [
                [tokens, elapsed_ms]
                for tokens, elapsed_ms in self.prefill_continuation_chunk_curve
            ],
            "calibrated": self.calibrated,
            "calibration_source": self.calibration_source,
            "physical_model_id": self.physical_model_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "QueueServiceModel":
        if int(raw.get("schema_version", 0)) not in {1, 2}:
            raise ValueError("unsupported queue service model schema")
        return cls(
            model_id=str(raw["model_id"]),
            prefill_tokens_per_ms=float(raw["prefill_tokens_per_ms"]),
            decode_tokens_per_ms=float(raw["decode_tokens_per_ms"]),
            decode_batch_efficiency=tuple(
                float(item) for item in raw.get("decode_batch_efficiency", ())
            ),
            max_decode_batch=int(raw["max_decode_batch"]),
            prefill_chunk_tokens=int(raw["prefill_chunk_tokens"]),
            decode_quantum_tokens=int(raw["decode_quantum_tokens"]),
            prefill_launch_ms=float(raw.get("prefill_launch_ms", 0.0)),
            decode_launch_ms=float(raw.get("decode_launch_ms", 0.0)),
            prefill_first_chunk_curve=tuple(
                (int(item[0]), float(item[1]))
                for item in raw.get("prefill_first_chunk_curve", ())
            ),
            prefill_continuation_chunk_curve=tuple(
                (int(item[0]), float(item[1]))
                for item in raw.get("prefill_continuation_chunk_curve", ())
            ),
            calibrated=bool(raw.get("calibrated", False)),
            calibration_source=str(raw.get("calibration_source", "unspecified")),
            physical_model_id=str(
                raw.get("physical_model_id", "exact_bundle_allocator_v1")
            ),
        )


@dataclass(frozen=True)
class CounterfactualSimulationResult:
    workflow_jct_ms: Mapping[str, float]
    request_finish_ms: Mapping[str, float]
    request_queue_wait_ms: Mapping[str, float]
    request_action_unlock_ms: Mapping[str, float]
    causal_blocked_ms: float
    unhidden_stall_ms: float
    final_ts_ms: float
    hbm_peak_bytes: int
    hbm_final_bytes: int
    host_consumed_bytes: int
    d2h_bytes: int
    h2d_bytes: int
    pcie_busy_ms: float
    scheduler_steps: int
    transition_hash: str
    radix_demand_recomputed: bool
    recomputed_cache_hit_tokens: Mapping[str, int]
    recomputed_unique_growth_bytes: Mapping[str, int]
    rolling_physical_replay: bool = False
    physical_timeline: tuple[Mapping[str, object], ...] = ()


@dataclass
class _RequestState:
    demand: FrozenRequestDemand
    prefill_remaining: int
    decode_remaining: int
    ready_at_ms: float | None = None
    admitted_at_ms: float | None = None
    first_service_ms: float | None = None
    finish_ms: float | None = None
    decoded_tokens: int = 0
    prefill_chunk_index: int = 0
    action_unlock_ms: float | None = None
    radix_initialized: bool = False
    cache_hit_tokens: int = 0
    prefill_total: int = 0
    startup_reserved_bytes: int = 0
    unique_commit_growth_tokens: int = 0


@dataclass
class _BundleState:
    size_bytes: int
    gpu_bytes: int
    cpu_bytes: int


@dataclass(frozen=True)
class _PhysicalEvent:
    ts_ms: float
    bundle_id: str
    gpu_delta_bytes: int
    cpu_delta_bytes: int
    direction: str
    transfer_bytes: int
    hbm_reservation_delta_bytes: int = 0


@dataclass
class _PhysicalSchedule:
    bundles: dict[str, _BundleState]
    events: list[_PhysicalEvent]
    hbm_used_bytes: int
    fixed_reserved_bytes: int
    hbm_transfer_reserved_bytes: int
    host_free_bytes: int
    host_initial_free_bytes: int
    d2h_bytes: int
    h2d_bytes: int
    pcie_busy_ms: float


class CounterfactualQueueServiceSimulator:
    """Recompute queue, token service, PCIe and allocator state from frozen demand."""

    def __init__(self, service_model: QueueServiceModel) -> None:
        self.service_model = service_model

    def simulate(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        workload: FrozenCounterfactualWorkload,
    ) -> CounterfactualSimulationResult:
        if not plan.feasible:
            raise CounterfactualSimulationError("cannot simulate an infeasible plan")
        if plan.snapshot_id != policy_input.snapshot_id:
            raise CounterfactualSimulationError("plan uses a stale physical snapshot")
        if demand.snapshot_id != policy_input.snapshot_id:
            raise CounterfactualSimulationError("demand uses a stale physical snapshot")
        if not plan.physical_accounting_exact or not demand.physical_accounting_exact:
            raise CounterfactualSimulationError(
                "counterfactual timing requires exact extent accounting"
            )
        request_by_id = {item.request_id: item for item in workload.requests}
        unknown_execution = set(plan.execution_order) - set(request_by_id)
        if unknown_execution:
            raise CounterfactualSimulationError(
                f"execution plan references unknown requests: {unknown_execution}"
            )

        physical = self._physical_schedule(policy_input, plan)
        radix_exact = (
            workload.prefix_identity_complete
            and workload.initial_radix_state_known
            and policy_input.physical_kv.gpu_bytes == 0
            and policy_input.physical_kv.cpu_bytes == 0
            and policy_input.resources.hbm_used_bytes == 0
        )
        radix = None
        if radix_exact:
            from beliefkv.simulator.token_radix import TokenRadixReplay

            radix = TokenRadixReplay()
        states = {
            request.request_id: _RequestState(
                demand=request,
                prefill_remaining=request.uncached_prompt_tokens,
                decode_remaining=request.output_tokens,
                cache_hit_tokens=request.observed_cache_hit_tokens,
                prefill_total=request.uncached_prompt_tokens,
                startup_reserved_bytes=request.startup_bytes,
            )
            for request in workload.requests
        }
        workflow_by_id = {item.workflow_id: item for item in workload.workflows}
        initial_required = set(demand.required_gpu_bundles)
        execution_set = set(plan.execution_order)
        rank = {request_id: index for index, request_id in enumerate(plan.execution_order)}
        request_reserved_bytes = 0
        hbm_peak = (
            physical.hbm_used_bytes
            + physical.fixed_reserved_bytes
            + physical.hbm_transfer_reserved_bytes
        )
        event_index = 0
        now_ms = 0.0
        scheduler_steps = 0
        prefer_decode = False

        while (
            any(state.finish_ms is None for state in states.values())
            or event_index < len(physical.events)
        ):
            while (
                event_index < len(physical.events)
                and physical.events[event_index].ts_ms <= now_ms + 1e-12
            ):
                event = physical.events[event_index]
                bundle = physical.bundles[event.bundle_id]
                bundle.gpu_bytes += event.gpu_delta_bytes
                bundle.cpu_bytes += event.cpu_delta_bytes
                physical.hbm_used_bytes += event.gpu_delta_bytes
                physical.hbm_transfer_reserved_bytes += (
                    event.hbm_reservation_delta_bytes
                )
                self._check_allocator(
                    policy_input,
                    physical,
                    request_reserved_bytes,
                )
                event_index += 1

            self._release_requests(states, workflow_by_id, now_ms)
            if radix is not None:
                for state in states.values():
                    if state.ready_at_ms is None or state.radix_initialized:
                        continue
                    hit = radix.match(state.demand.prompt_token_symbols)
                    state.cache_hit_tokens = hit
                    state.prefill_remaining = (
                        len(state.demand.prompt_token_symbols) - hit
                    )
                    state.prefill_total = state.prefill_remaining
                    kv_bytes_per_token = int(
                        workload.metadata.get("kv_bytes_per_token", 0)
                    )
                    if kv_bytes_per_token <= 0:
                        raise CounterfactualSimulationError(
                            "exact Radix replay requires kv_bytes_per_token metadata"
                        )
                    state.startup_reserved_bytes = max(
                        0,
                        state.demand.startup_bytes
                        + (
                            state.demand.observed_cache_hit_tokens - hit
                        )
                        * kv_bytes_per_token,
                    )
                    state.radix_initialized = True
            ordered_ready = sorted(
                (
                    state
                    for state in states.values()
                    if state.ready_at_ms is not None
                    and state.admitted_at_ms is None
                    and state.finish_ms is None
                ),
                key=lambda state: self._priority_key(state, rank),
            )
            for state in ordered_ready:
                required = set(state.demand.required_bundle_ids)
                if state.demand.request_id in execution_set:
                    required.update(initial_required)
                if not self._bundles_ready(required, physical.bundles):
                    continue
                startup = state.startup_reserved_bytes
                available = (
                    policy_input.resources.hbm_capacity_bytes
                    - physical.hbm_used_bytes
                    - physical.fixed_reserved_bytes
                    - physical.hbm_transfer_reserved_bytes
                    - request_reserved_bytes
                )
                if startup > available:
                    continue
                state.admitted_at_ms = now_ms
                request_reserved_bytes += startup

            for state in states.values():
                if (
                    state.admitted_at_ms is not None
                    and state.finish_ms is None
                    and state.prefill_remaining == 0
                    and state.decode_remaining == 0
                ):
                    self._mark_first_service(state, now_ms)
                    request_reserved_bytes = self._complete_request(
                        state,
                        now_ms,
                        request_reserved_bytes,
                        physical,
                        policy_input,
                        growth_bytes=self._request_growth_bytes(
                            state, workload, radix
                        ),
                    )

            admitted = [
                state
                for state in states.values()
                if state.admitted_at_ms is not None and state.finish_ms is None
            ]
            prefill = [state for state in admitted if state.prefill_remaining > 0]
            decode = [
                state
                for state in admitted
                if state.prefill_remaining == 0 and state.decode_remaining > 0
            ]
            if prefill and (not decode or not prefer_decode):
                state = min(prefill, key=lambda item: self._priority_key(item, rank))
                tokens = min(
                    state.prefill_remaining,
                    self.service_model.prefill_chunk_tokens,
                )
                elapsed = self.service_model.prefill_elapsed_ms(
                    tokens,
                    chunk_index=state.prefill_chunk_index,
                )
                self._mark_first_service(state, now_ms)
                state.prefill_remaining -= tokens
                state.prefill_chunk_index += 1
                if radix is not None:
                    processed = state.prefill_total - state.prefill_remaining
                    visible_tokens = state.cache_hit_tokens + processed
                    state.unique_commit_growth_tokens += radix.insert(
                        state.demand.prompt_token_symbols[:visible_tokens]
                    )
                now_ms += elapsed
                scheduler_steps += 1
                prefer_decode = True
                if state.prefill_remaining == 0 and state.decode_remaining == 0:
                    request_reserved_bytes = self._complete_request(
                        state,
                        now_ms,
                        request_reserved_bytes,
                        physical,
                        policy_input,
                        growth_bytes=self._request_growth_bytes(
                            state, workload, radix
                        ),
                    )
            elif decode:
                batch = sorted(
                    decode, key=lambda item: self._priority_key(item, rank)
                )[: self.service_model.max_decode_batch]
                progressed = {
                    state.demand.request_id: min(
                        state.decode_remaining,
                        self.service_model.decode_quantum_tokens,
                    )
                    for state in batch
                }
                total_tokens = sum(progressed.values())
                elapsed = (
                    self.service_model.decode_launch_ms
                    + total_tokens / self.service_model.decode_rate(len(batch))
                )
                for state in batch:
                    self._mark_first_service(state, now_ms)
                    tokens = progressed[state.demand.request_id]
                    state.decode_remaining -= tokens
                    state.decoded_tokens += tokens
                    boundary = state.demand.action_boundary_token_index
                    if (
                        boundary is not None
                        and state.action_unlock_ms is None
                        and state.decoded_tokens >= boundary
                    ):
                        fraction = 0.0 if tokens == 0 else (
                            tokens - (state.decoded_tokens - boundary)
                        ) / tokens
                        state.action_unlock_ms = now_ms + elapsed * fraction
                now_ms += elapsed
                scheduler_steps += 1
                prefer_decode = False
                for state in batch:
                    if state.decode_remaining == 0:
                        request_reserved_bytes = self._complete_request(
                            state,
                            now_ms,
                            request_reserved_bytes,
                            physical,
                            policy_input,
                            growth_bytes=self._request_growth_bytes(
                                state, workload, radix
                            ),
                        )
            else:
                next_times = []
                if event_index < len(physical.events):
                    next_times.append(physical.events[event_index].ts_ms)
                next_release = self._next_release_time(states, workflow_by_id)
                if next_release is not None:
                    next_times.append(next_release)
                future = [value for value in next_times if value > now_ms + 1e-12]
                if not future:
                    blocked = sorted(
                        state.demand.request_id
                        for state in states.values()
                        if state.finish_ms is None
                    )
                    raise CounterfactualSimulationError(
                        "allocator/dependency deadlock while requests remain: "
                        + ",".join(blocked)
                    )
                now_ms = min(future)

            hbm_peak = max(
                hbm_peak,
                physical.hbm_used_bytes
                + physical.fixed_reserved_bytes
                + physical.hbm_transfer_reserved_bytes
                + request_reserved_bytes,
            )

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
        causal_blocked = _causal_blocked_critical_path_ms(workload)
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
            causal_blocked_ms=causal_blocked,
            unhidden_stall_ms=sum(queue_wait.values()),
            final_ts_ms=now_ms,
            hbm_peak_bytes=hbm_peak,
            hbm_final_bytes=(
                physical.hbm_used_bytes + physical.fixed_reserved_bytes
            ),
            host_consumed_bytes=(
                physical.host_initial_free_bytes - physical.host_free_bytes
            ),
            d2h_bytes=physical.d2h_bytes,
            h2d_bytes=physical.h2d_bytes,
            pcie_busy_ms=physical.pcie_busy_ms,
            scheduler_steps=scheduler_steps,
            transition_hash=workload.transition_hash,
            radix_demand_recomputed=radix_exact,
            recomputed_cache_hit_tokens=MappingProxyType(
                {
                    request_id: state.cache_hit_tokens
                    for request_id, state in sorted(states.items())
                }
            ),
            recomputed_unique_growth_bytes=MappingProxyType(
                {
                    request_id: (
                        state.unique_commit_growth_tokens
                        * int(workload.metadata.get("kv_bytes_per_token", 0))
                        if radix_exact
                        else state.demand.kv_growth_bytes
                    )
                    for request_id, state in sorted(states.items())
                }
            ),
        )

    @staticmethod
    def _request_growth_bytes(
        state: _RequestState,
        workload: FrozenCounterfactualWorkload,
        radix: object | None,
    ) -> int:
        if radix is None:
            return state.demand.kv_growth_bytes
        insert = getattr(radix, "insert")
        state.unique_commit_growth_tokens += int(
            insert(state.demand.cache_commit_token_symbols)
        )
        return state.unique_commit_growth_tokens * int(
            workload.metadata["kv_bytes_per_token"]
        )

    def _physical_schedule(
        self, policy_input: PolicyInput, plan: ScenarioPlan
    ) -> _PhysicalSchedule:
        resources = policy_input.resources
        bundles = {
            item.bundle_id: _BundleState(
                item.physical_unique_bytes,
                item.gpu_bytes,
                item.cpu_bytes,
            )
            for item in policy_input.physical_kv.bundles
        }
        snapshots = {item.bundle_id: item for item in policy_input.physical_kv.bundles}
        unknown = set(plan.bundle_actions) - set(bundles)
        if unknown:
            raise CounterfactualSimulationError(
                f"plan references unknown physical bundles: {unknown}"
            )
        hbm_used = resources.hbm_used_bytes
        host_free = resources.host_free_bytes
        events: list[_PhysicalEvent] = []
        d2h_jobs: list[tuple[str, int, int, bool]] = []
        h2d_jobs: list[tuple[str, int]] = []

        for bundle_id, action in sorted(plan.bundle_actions.items()):
            snapshot = snapshots[bundle_id]
            if not snapshot.actionable and action != ResidencyAction.KEEP:
                raise CounterfactualSimulationError(
                    f"plan acts on blocked bundle {bundle_id}: {snapshot.blocker_codes}"
                )
            bundle = bundles[bundle_id]
            if action == ResidencyAction.KEEP:
                continue
            if action in {ResidencyAction.DROP, ResidencyAction.RECOMPUTE}:
                reclaimed = snapshot.marginal_reclaimable_bytes
                if reclaimed > bundle.gpu_bytes:
                    raise CounterfactualSimulationError("drop exceeds GPU residency")
                bundle.gpu_bytes -= reclaimed
                hbm_used -= reclaimed
                continue
            if action in {ResidencyAction.COMMIT_CPU, ResidencyAction.PREPARE_HOST}:
                transfer = max(0, bundle.gpu_bytes - bundle.cpu_bytes)
                reclaim = (
                    snapshot.marginal_reclaimable_bytes
                    if action == ResidencyAction.COMMIT_CPU
                    else 0
                )
                if transfer > host_free:
                    raise CounterfactualSimulationError("D2H exceeds Host capacity")
                host_free -= transfer
                d2h_jobs.append(
                    (bundle_id, transfer, reclaim, action == ResidencyAction.COMMIT_CPU)
                )
                continue
            if action == ResidencyAction.PREFETCH_GPU:
                transfer = max(0, bundle.size_bytes - bundle.gpu_bytes)
                if transfer > bundle.cpu_bytes:
                    raise CounterfactualSimulationError(
                        f"H2D source is incomplete for bundle {bundle_id}"
                    )
                h2d_jobs.append((bundle_id, transfer))
                continue
            raise CounterfactualSimulationError(f"unsupported action: {action.value}")

        cursor = 0.0
        d2h_bytes = 0
        h2d_bytes = 0
        total_h2d = sum(transfer for _, transfer in h2d_jobs)
        total_reclaim = sum(reclaim for _, _, reclaim, commit in d2h_jobs if commit)
        initial_h2d_reservation = max(0, total_h2d - total_reclaim)
        deferred_h2d_reservation = total_h2d - initial_h2d_reservation
        for index, (bundle_id, transfer, reclaim, commit) in enumerate(d2h_jobs):
            duration = self._transfer_ms(
                transfer,
                resources.d2h_service_bytes_per_ms,
                resources.transfer_setup_p50_ms,
            )
            cursor += duration
            d2h_bytes += transfer
            events.append(
                _PhysicalEvent(
                    cursor,
                    bundle_id,
                    -reclaim if commit else 0,
                    transfer,
                    "d2h",
                    transfer,
                    (
                        deferred_h2d_reservation
                        if index == len(d2h_jobs) - 1
                        else 0
                    ),
                )
            )
        hbm_transfer_reserved = initial_h2d_reservation
        for bundle_id, transfer in h2d_jobs:
            duration = self._transfer_ms(
                transfer,
                resources.h2d_service_bytes_per_ms,
                resources.transfer_setup_p50_ms,
            )
            cursor += duration
            h2d_bytes += transfer
            hbm_transfer_reserved += transfer
            events.append(
                _PhysicalEvent(
                    cursor,
                    bundle_id,
                    transfer,
                    0,
                    "h2d",
                    transfer,
                    -transfer,
                )
            )
        if d2h_bytes != plan.d2h_bytes or h2d_bytes != plan.h2d_bytes:
            raise CounterfactualSimulationError(
                "plan byte totals disagree with exact bundle replay: "
                f"d2h={d2h_bytes}/{plan.d2h_bytes}, "
                f"h2d={h2d_bytes}/{plan.h2d_bytes}"
            )
        physical = _PhysicalSchedule(
            bundles=bundles,
            events=events,
            hbm_used_bytes=hbm_used,
            fixed_reserved_bytes=resources.hbm_reserved_bytes,
            hbm_transfer_reserved_bytes=hbm_transfer_reserved,
            host_free_bytes=host_free,
            host_initial_free_bytes=resources.host_free_bytes,
            d2h_bytes=d2h_bytes,
            h2d_bytes=h2d_bytes,
            pcie_busy_ms=cursor,
        )
        self._check_allocator(policy_input, physical, 0)
        return physical

    @staticmethod
    def _transfer_ms(bytes_: int, rate: float, setup_ms: float) -> float:
        if bytes_ <= 0:
            return 0.0
        if rate <= 0:
            raise CounterfactualSimulationError("transfer service rate is unavailable")
        return setup_ms + bytes_ / rate

    @staticmethod
    def _check_allocator(
        policy_input: PolicyInput,
        physical: _PhysicalSchedule,
        request_reserved_bytes: int,
    ) -> None:
        occupied = (
            physical.hbm_used_bytes
            + physical.fixed_reserved_bytes
            + physical.hbm_transfer_reserved_bytes
            + request_reserved_bytes
        )
        if occupied > policy_input.resources.hbm_capacity_bytes:
            raise CounterfactualSimulationError(
                f"counterfactual allocator exceeds HBM by "
                f"{occupied - policy_input.resources.hbm_capacity_bytes} bytes"
            )
        if physical.hbm_used_bytes < 0 or physical.hbm_transfer_reserved_bytes < 0:
            raise CounterfactualSimulationError("counterfactual HBM ledger is negative")

    @staticmethod
    def _bundles_ready(
        required_bundle_ids: set[str], bundles: Mapping[str, _BundleState]
    ) -> bool:
        for bundle_id in required_bundle_ids:
            bundle = bundles.get(bundle_id)
            if bundle is None:
                raise CounterfactualSimulationError(
                    f"request requires unknown bundle: {bundle_id}"
                )
            if bundle.gpu_bytes < bundle.size_bytes:
                return False
        return True

    @staticmethod
    def _release_requests(
        states: Mapping[str, _RequestState],
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
        states: Mapping[str, _RequestState],
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
    def _priority_key(
        state: _RequestState, rank: Mapping[str, int]
    ) -> tuple[float, float, str]:
        return (
            float(rank.get(state.demand.request_id, len(rank))),
            float(state.ready_at_ms or 0.0),
            state.demand.request_id,
        )

    @staticmethod
    def _mark_first_service(state: _RequestState, now_ms: float) -> None:
        if state.first_service_ms is None:
            state.first_service_ms = now_ms
        if state.demand.action_boundary_token_index == 0:
            state.action_unlock_ms = state.action_unlock_ms or now_ms

    @staticmethod
    def _complete_request(
        state: _RequestState,
        now_ms: float,
        request_reserved_bytes: int,
        physical: _PhysicalSchedule,
        policy_input: PolicyInput,
        *,
        growth_bytes: int,
    ) -> int:
        if state.finish_ms is not None:
            raise CounterfactualSimulationError("request completed twice")
        state.finish_ms = now_ms
        if (
            state.demand.action_boundary_token_index is not None
            and state.action_unlock_ms is None
        ):
            state.action_unlock_ms = now_ms
        request_reserved_bytes -= state.startup_reserved_bytes
        physical.hbm_used_bytes += growth_bytes
        CounterfactualQueueServiceSimulator._check_allocator(
            policy_input,
            physical,
            request_reserved_bytes,
        )
        return request_reserved_bytes


class FrozenTracePlanEvaluator:
    """Bridge the queue/service resimulator to JointPlanOracle's evidence contract."""

    def __init__(
        self,
        workload: FrozenCounterfactualWorkload,
        service_model: QueueServiceModel,
    ) -> None:
        self.workload = workload
        self.service_model = service_model
        self.simulator = CounterfactualQueueServiceSimulator(service_model)

    def evaluate(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        *,
        trace_sensitivity: str,
    ) -> ResimulatedPlanEvaluation:
        if trace_sensitivity != self.workload.trace_sensitivity:
            raise CounterfactualSimulationError(
                "oracle candidate sensitivity differs from the frozen workload"
            )
        result = self.simulator.simulate(
            policy_input,
            demand,
            plan,
            self.workload,
        )
        workflow_jct = tuple(result.workflow_jct_ms.values())
        action_unlock = tuple(result.request_action_unlock_ms.values())
        return ResimulatedPlanEvaluation(
            cost=OracleCost(
                workflow_jct_ms=sum(workflow_jct) / len(workflow_jct),
                causal_blocked_ms=result.causal_blocked_ms,
                unhidden_stall_ms=result.unhidden_stall_ms,
                action_unlock_ms=(
                    sum(action_unlock) / len(action_unlock)
                    if action_unlock
                    else None
                ),
            ),
            evidence=ResimulationEvidence(
                schedule_recomputed=True,
                queue_service_recomputed=True,
                physical_actions_recomputed=(
                    self.workload.future_physical_growth_exact
                    or result.radix_demand_recomputed
                ),
                allocator_recomputed=True,
                service_model_calibrated=self.service_model.calibrated,
                semantic_events_frozen=self.workload.semantic_events_frozen,
                token_demand_frozen=self.workload.token_demand_frozen,
                tool_duration_frozen=self.workload.tool_duration_frozen,
                transition_hash=self.workload.transition_hash,
                service_model_id=self.service_model.model_id,
                physical_model_id=(
                    "token_radix_bundle_allocator_v2"
                    if result.radix_demand_recomputed
                    else self.service_model.physical_model_id
                ),
            ),
        )


class RollingFrozenTracePlanEvaluator:
    """Evaluate O0-O3 with arm-specific rolling Radix residency policies."""

    def __init__(
        self,
        workload: FrozenCounterfactualWorkload,
        service_model: QueueServiceModel,
    ) -> None:
        self.workload = workload
        self.service_model = service_model
        self.last_results: dict[OracleArm, CounterfactualSimulationResult] = {}
        self.simulation_count = 0
        self._result_cache: dict[
            tuple[object, ...],
            tuple[CounterfactualSimulationResult, ResimulatedPlanEvaluation],
        ] = {}

    def evaluate(
        self,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        *,
        trace_sensitivity: str,
    ) -> ResimulatedPlanEvaluation:
        return self._evaluate(
            OracleArm.O0_SEPARATE,
            policy_input,
            demand,
            plan,
            trace_sensitivity=trace_sensitivity,
        )

    def evaluate_arm(
        self,
        arm: OracleArm,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        *,
        trace_sensitivity: str,
    ) -> ResimulatedPlanEvaluation:
        return self._evaluate(
            arm,
            policy_input,
            demand,
            plan,
            trace_sensitivity=trace_sensitivity,
        )

    def _evaluate(
        self,
        arm: OracleArm,
        policy_input: PolicyInput,
        demand: ScenarioDemand,
        plan: ScenarioPlan,
        *,
        trace_sensitivity: str,
    ) -> ResimulatedPlanEvaluation:
        if trace_sensitivity != self.workload.trace_sensitivity:
            raise CounterfactualSimulationError(
                "oracle candidate sensitivity differs from the frozen workload"
            )
        from beliefkv.simulator.rolling_physical import ResidencyReplayMode
        from beliefkv.simulator.rolling_queue_service import (
            RollingCounterfactualQueueServiceSimulator,
        )

        mode = (
            ResidencyReplayMode.HINDSIGHT_NEXT_USE
            if arm in {OracleArm.O2_KV, OracleArm.O3_JOINT}
            else ResidencyReplayMode.REACTIVE_LRU
        )
        cache_key = (
            policy_input.snapshot_id,
            demand.scenario_id,
            mode.value,
            tuple(plan.execution_order),
            tuple(plan.admission_actions.items()),
            trace_sensitivity,
        )
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            result, evaluation = cached
            self.last_results[arm] = result
            return evaluation
        result = RollingCounterfactualQueueServiceSimulator(
            self.service_model,
            residency_mode=mode,
        ).simulate(policy_input, demand, plan, self.workload)
        self.simulation_count += 1
        self.last_results[arm] = result
        workflow_jct = tuple(result.workflow_jct_ms.values())
        action_unlock = tuple(result.request_action_unlock_ms.values())
        evaluation = ResimulatedPlanEvaluation(
            cost=OracleCost(
                workflow_jct_ms=sum(workflow_jct) / len(workflow_jct),
                causal_blocked_ms=result.causal_blocked_ms,
                unhidden_stall_ms=result.unhidden_stall_ms,
                action_unlock_ms=(
                    sum(action_unlock) / len(action_unlock)
                    if action_unlock
                    else None
                ),
            ),
            evidence=ResimulationEvidence(
                schedule_recomputed=True,
                queue_service_recomputed=True,
                physical_actions_recomputed=True,
                allocator_recomputed=True,
                service_model_calibrated=self.service_model.calibrated,
                semantic_events_frozen=self.workload.semantic_events_frozen,
                token_demand_frozen=self.workload.token_demand_frozen,
                tool_duration_frozen=self.workload.tool_duration_frozen,
                transition_hash=self.workload.transition_hash,
                service_model_id=self.service_model.model_id,
                physical_model_id="rolling_tiered_token_radix_allocator_v3",
            ),
        )
        self._result_cache[cache_key] = (result, evaluation)
        return evaluation


def frozen_transition_hash(records: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        [dict(item) for item in records],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return blake2b(
        payload.encode("utf-8"), digest_size=16, person=b"beliefkv-trans"
    ).hexdigest()


def _encode_token_symbols(symbols: tuple[int, ...]) -> str | None:
    if not symbols:
        return None
    packed = struct.pack(f"<{len(symbols)}Q", *symbols)
    return base64.b64encode(packed).decode("ascii")


def _decode_token_symbols(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        raise ValueError("encoded token symbols must be a base64 string or null")
    try:
        packed = base64.b64decode(value, validate=True)
    except Exception as error:
        raise ValueError("encoded token symbols are not valid base64") from error
    if len(packed) % 8:
        raise ValueError("encoded token symbol bytes are not uint64 aligned")
    return (
        tuple(struct.unpack(f"<{len(packed) // 8}Q", packed))
        if packed
        else ()
    )


def _interpolate_service_curve(
    curve: tuple[tuple[int, float], ...],
    tokens: int,
    *,
    proportional_below_minimum: bool,
    fallback_floor_ms: float,
) -> float:
    if len(curve) == 1:
        point_tokens, point_ms = curve[0]
        if tokens < point_tokens and proportional_below_minimum:
            return max(1e-9, point_ms * tokens / point_tokens)
        return max(fallback_floor_ms, point_ms * tokens / point_tokens)
    first_tokens, first_ms = curve[0]
    if tokens <= first_tokens:
        if proportional_below_minimum:
            return max(1e-9, first_ms * tokens / first_tokens)
        second_tokens, second_ms = curve[1]
        slope = (second_ms - first_ms) / (second_tokens - first_tokens)
        return max(fallback_floor_ms, first_ms + slope * (tokens - first_tokens))
    for (left_tokens, left_ms), (right_tokens, right_ms) in zip(curve, curve[1:]):
        if tokens <= right_tokens:
            fraction = (tokens - left_tokens) / (right_tokens - left_tokens)
            return left_ms + fraction * (right_ms - left_ms)
    left_tokens, left_ms = curve[-2]
    right_tokens, right_ms = curve[-1]
    slope = max(0.0, (right_ms - left_ms) / (right_tokens - left_tokens))
    return max(fallback_floor_ms, right_ms + slope * (tokens - right_tokens))


def _assert_acyclic(requests: Mapping[str, FrozenRequestDemand]) -> None:
    indegree = {request_id: 0 for request_id in requests}
    successors = {request_id: [] for request_id in requests}
    for request in requests.values():
        indegree[request.request_id] = len(request.predecessor_request_ids)
        for predecessor in request.predecessor_request_ids:
            successors[predecessor].append(request.request_id)
    ready = sorted(request_id for request_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        request_id = ready.pop(0)
        visited += 1
        for successor in sorted(successors[request_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if visited != len(requests):
        raise ValueError("frozen request dependencies contain a cycle")


def _causal_blocked_critical_path_ms(
    workload: FrozenCounterfactualWorkload,
) -> float:
    requests = {item.request_id: item for item in workload.requests}
    remaining = set(requests)
    external_path: dict[str, float] = {}
    while remaining:
        progressed = False
        for request_id in sorted(tuple(remaining)):
            request = requests[request_id]
            if any(item not in external_path for item in request.predecessor_request_ids):
                continue
            predecessor_path = max(
                (external_path[item] for item in request.predecessor_request_ids),
                default=0.0,
            )
            external_path[request_id] = predecessor_path + request.release_delay_ms
            remaining.remove(request_id)
            progressed = True
        if not progressed:
            raise ValueError("frozen workload contains an unresolved dependency cycle")
    return sum(
        max(external_path[item] for item in workflow.terminal_request_ids)
        + workflow.completion_delay_ms
        for workflow in workload.workflows
    )
