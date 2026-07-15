from __future__ import annotations

from dataclasses import dataclass

from beliefkv.policy.causal_frontier import CausalFrontierScheduler
from beliefkv.policy.workflow_fairness import WorkflowFairScheduler
from beliefkv.runtime.page_index import PageOwnershipIndex


@dataclass(frozen=True)
class AdmissionRequest:
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    submitted_ts_ms: float
    uncached_prompt_tokens: int
    expected_output_tokens: int
    kv_bytes_per_token: int
    fixed_overhead_bytes: int = 0
    prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.request_id, "request_id"),
            (self.workflow_id, "workflow_id"),
            (self.invocation_id, "invocation_id"),
            (self.context_id, "context_id"),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.context_epoch < 0 or self.submitted_ts_ms < 0:
            raise ValueError("context epoch and submit time must be non-negative")
        if min(
            self.uncached_prompt_tokens,
            self.expected_output_tokens,
            self.fixed_overhead_bytes,
        ) < 0:
            raise ValueError("admission token/byte estimates must be non-negative")
        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative or null")
        if self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")

    @property
    def estimated_incremental_bytes(self) -> int:
        token_bytes = (
            self.uncached_prompt_tokens + self.expected_output_tokens
        ) * self.kv_bytes_per_token
        return max(0, token_bytes + self.fixed_overhead_bytes)

    @property
    def estimated_working_set_bytes(self) -> int:
        prompt_tokens = (
            self.prompt_tokens
            if self.prompt_tokens is not None
            else self.uncached_prompt_tokens
        )
        return max(
            0,
            (prompt_tokens + self.expected_output_tokens) * self.kv_bytes_per_token
            + self.fixed_overhead_bytes,
        )


@dataclass(frozen=True)
class AdmissionDecision:
    request_id: str
    admitted: bool
    reason: str
    reserved_bytes: int = 0


class AdmissionController:
    """Workflow-fair admission based only on physical bytes and reservations."""

    def __init__(
        self,
        page_index: PageOwnershipIndex,
        fairness: WorkflowFairScheduler,
        frontier: CausalFrontierScheduler,
        *,
        reserve_hbm_bytes: int,
        max_borrow_fraction: float = 0.25,
    ) -> None:
        if reserve_hbm_bytes < 0:
            raise ValueError("reserve_hbm_bytes must be non-negative")
        if not 0 <= max_borrow_fraction <= 1:
            raise ValueError("max_borrow_fraction must be in [0, 1]")
        self.page_index = page_index
        self.fairness = fairness
        self.frontier = frontier
        self.reserve_hbm_bytes = reserve_hbm_bytes
        self.max_borrow_fraction = max_borrow_fraction
        self._pending: dict[str, AdmissionRequest] = {}
        self._reserved: dict[str, int] = {}

    def enqueue(self, request: AdmissionRequest) -> None:
        if request.request_id in self._pending or request.request_id in self._reserved:
            raise ValueError(f"duplicate request id: {request.request_id}")
        self._pending[request.request_id] = request
        self.fairness.register(request.workflow_id)

    def cancel(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
        self._reserved.pop(request_id, None)

    @property
    def reserved_bytes(self) -> int:
        return sum(self._reserved.values())

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def decide_next(
        self,
        hbm_capacity_bytes: int,
        *,
        actual_hbm_used_bytes: int | None = None,
        external_workflow_charges: dict[str, float] | None = None,
        allow_reserve_borrow: bool = False,
        preferred_request_id: str | None = None,
        force_preferred_progress: bool = False,
    ) -> AdmissionDecision | None:
        if not self._pending:
            return None
        allocatable = max(0, hbm_capacity_bytes - self.reserve_hbm_bytes)
        physical_used = max(
            self.page_index.gpu_bytes,
            actual_hbm_used_bytes if actual_hbm_used_bytes is not None else 0,
        )
        free = max(0, allocatable - physical_used - self.reserved_bytes)
        charges = self.page_index.workflow_gpu_charges()
        for workflow_id, charge in (external_workflow_charges or {}).items():
            charges[workflow_id] = charges.get(workflow_id, 0.0) + max(0.0, charge)
        workflows = {item.workflow_id for item in self._pending.values()}
        shares = self.fairness.fair_memory_shares(workflows, allocatable)

        def fitting_requests(
            available_bytes: int,
        ) -> dict[str, list[AdmissionRequest]]:
            result: dict[str, list[AdmissionRequest]] = {}
            for request in self._pending.values():
                if request.estimated_incremental_bytes <= available_bytes:
                    result.setdefault(request.workflow_id, []).append(request)
            return result

        preferred = (
            self._pending.get(preferred_request_id)
            if preferred_request_id is not None
            else None
        )
        if preferred is not None:
            preferred_reason = "admission_liveness_target"
            preferred_free = free
            if (
                preferred.estimated_incremental_bytes > preferred_free
                and allow_reserve_borrow
            ):
                preferred_free = max(
                    0,
                    hbm_capacity_bytes - physical_used - self.reserved_bytes,
                )
                preferred_reason = "admission_liveness_reserve_borrow"
            if preferred.estimated_incremental_bytes > preferred_free:
                if (
                    force_preferred_progress
                    and preferred.estimated_working_set_bytes <= hbm_capacity_bytes
                ):
                    self._pending.pop(preferred.request_id)
                    self._reserved[preferred.request_id] = (
                        preferred.estimated_incremental_bytes
                    )
                    return AdmissionDecision(
                        preferred.request_id,
                        True,
                        "admission_liveness_native_reclaim",
                        preferred.estimated_incremental_bytes,
                    )
                return AdmissionDecision(
                    preferred.request_id,
                    False,
                    (
                        "request_exceeds_hbm_capacity"
                        if force_preferred_progress
                        and preferred.estimated_working_set_bytes > hbm_capacity_bytes
                        else "insufficient_actual_hbm"
                    ),
                )
            self._pending.pop(preferred.request_id)
            self._reserved[preferred.request_id] = (
                preferred.estimated_incremental_bytes
            )
            return AdmissionDecision(
                preferred.request_id,
                True,
                preferred_reason,
                preferred.estimated_incremental_bytes,
            )

        fitting_by_workflow = fitting_requests(free)
        borrowed_reserve = False
        if not fitting_by_workflow and allow_reserve_borrow:
            hard_free = max(
                0,
                hbm_capacity_bytes - physical_used - self.reserved_bytes,
            )
            fitting_by_workflow = fitting_requests(hard_free)
            borrowed_reserve = bool(fitting_by_workflow)
        if not fitting_by_workflow:
            request = min(
                self._pending.values(),
                key=lambda item: (item.submitted_ts_ms, item.request_id),
            )
            return AdmissionDecision(request.request_id, False, "insufficient_actual_hbm")

        eligible_workflows = set(fitting_by_workflow)
        under_soft_share = {
            workflow_id
            for workflow_id, requests in fitting_by_workflow.items()
            if any(
                charges.get(workflow_id, 0.0) + item.estimated_incremental_bytes
                <= shares[workflow_id]
                for item in requests
            )
        }
        if under_soft_share:
            eligible_workflows = under_soft_share
        else:
            borrow_limit = allocatable * self.max_borrow_fraction
            borrowable = {
                workflow_id
                for workflow_id, requests in fitting_by_workflow.items()
                if any(
                    charges.get(workflow_id, 0.0) + item.estimated_incremental_bytes
                    <= shares[workflow_id] + borrow_limit
                    for item in requests
                )
            }
            if borrowable:
                eligible_workflows = borrowable

        selected_workflow = self.fairness.select(
            eligible_workflows,
            memory_charges=charges,
            hbm_capacity_bytes=hbm_capacity_bytes,
        )
        assert selected_workflow is not None
        requests = fitting_by_workflow[selected_workflow]
        frontier_order = {
            item.invocation_id: index
            for index, item in enumerate(self.frontier.candidates(selected_workflow))
        }
        selected = min(
            requests,
            key=lambda item: (
                frontier_order.get(item.invocation_id, 1 << 30),
                item.submitted_ts_ms,
                item.request_id,
            ),
        )
        self._pending.pop(selected.request_id)
        self._reserved[selected.request_id] = selected.estimated_incremental_bytes
        return AdmissionDecision(
            selected.request_id,
            True,
            (
                "engine_idle_reserve_borrow"
                if borrowed_reserve
                else "workflow_fair_causal_frontier"
            ),
            selected.estimated_incremental_bytes,
        )

    def acknowledge(self, request_id: str) -> int:
        try:
            return self._reserved.pop(request_id)
        except KeyError as exc:
            raise KeyError(f"request has no admission reservation: {request_id}") from exc

    def pending_requests(self) -> list[AdmissionRequest]:
        return sorted(
            self._pending.values(),
            key=lambda item: (item.submitted_ts_ms, item.request_id),
        )
