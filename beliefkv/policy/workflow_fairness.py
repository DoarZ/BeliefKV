from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkflowAccount:
    workflow_id: str
    weight: float = 1.0
    attained_service_ms: float = 0.0
    dispatch_count: int = 0

    @property
    def virtual_runtime(self) -> float:
        return self.attained_service_ms / self.weight


class WorkflowFairScheduler:
    """Root-workflow weighted fair queue independent of agent fanout."""

    def __init__(self, *, memory_penalty_ms: float = 5.0) -> None:
        self.accounts: dict[str, WorkflowAccount] = {}
        self.memory_penalty_ms = memory_penalty_ms
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def register(self, workflow_id: str, *, weight: float = 1.0) -> None:
        if weight <= 0:
            raise ValueError("workflow weight must be positive")
        account = self.accounts.get(workflow_id)
        if account is None:
            self.accounts[workflow_id] = WorkflowAccount(workflow_id, weight)
            self._revision += 1
        elif account.weight != weight:
            account.weight = weight
            self._revision += 1

    def charge_service(self, workflow_id: str, service_ms: float) -> None:
        if service_ms < 0:
            raise ValueError("service_ms must be non-negative")
        self.register(workflow_id)
        account = self.accounts[workflow_id]
        if service_ms == 0:
            return
        account.attained_service_ms += service_ms
        account.dispatch_count += 1
        self._revision += 1

    def select(
        self,
        runnable_workflows: set[str] | list[str] | tuple[str, ...],
        *,
        memory_charges: dict[str, float] | None = None,
        hbm_capacity_bytes: int = 0,
    ) -> str | None:
        runnable = set(runnable_workflows)
        if not runnable:
            return None
        for workflow_id in runnable:
            self.register(workflow_id)
        charges = memory_charges or {}

        def key(workflow_id: str) -> tuple[float, float, int, str]:
            account = self.accounts[workflow_id]
            memory_share = (
                charges.get(workflow_id, 0.0) / hbm_capacity_bytes
                if hbm_capacity_bytes > 0
                else 0.0
            )
            effective_vruntime = (
                account.virtual_runtime + memory_share * self.memory_penalty_ms
            )
            return (
                effective_vruntime,
                memory_share,
                account.dispatch_count,
                workflow_id,
            )

        return min(runnable, key=key)

    def fair_memory_shares(
        self, active_workflows: set[str], allocatable_bytes: int
    ) -> dict[str, float]:
        if not active_workflows:
            return {}
        for workflow_id in active_workflows:
            self.register(workflow_id)
        total_weight = sum(self.accounts[item].weight for item in active_workflows)
        return {
            workflow_id: allocatable_bytes
            * self.accounts[workflow_id].weight
            / total_weight
            for workflow_id in active_workflows
        }

    def ordered(
        self,
        runnable_workflows: set[str] | list[str] | tuple[str, ...],
        *,
        memory_charges: dict[str, float] | None = None,
        hbm_capacity_bytes: int = 0,
    ) -> list[str]:
        remaining = set(runnable_workflows)
        result: list[str] = []
        while remaining:
            selected = self.select(
                remaining,
                memory_charges=memory_charges,
                hbm_capacity_bytes=hbm_capacity_bytes,
            )
            assert selected is not None
            result.append(selected)
            remaining.remove(selected)
        return result
