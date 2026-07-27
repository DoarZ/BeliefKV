from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass, field
from typing import Any

from beliefkv.control.controller import BeliefKVController, ControllerTickResult
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.admission import AdmissionRequest
from beliefkv.runtime.page_index import PageIndexError
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandStatus,
    PageHandle,
    PhysicalPageAction,
    PhysicalResidency,
    ResolvedCommand,
)
from beliefkv.simulator.schema import (
    SimulationEvent,
    SimulationEventKind,
    SimulationScenario,
)


class SimulationError(RuntimeError):
    pass


@dataclass(order=True)
class _Scheduled:
    ts_ms: float
    phase: int
    sequence: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False)


@dataclass(frozen=True)
class _TransferCompletion:
    resolved: ResolvedCommand
    accepted_handles: tuple[PageHandle, ...]
    status: CommandStatus
    actual_bytes: int
    reason: str = ""


@dataclass
class SimulationMetrics:
    workflow_start_ms: dict[str, float] = field(default_factory=dict)
    workflow_end_ms: dict[str, float] = field(default_factory=dict)
    first_admission_denial_ms: dict[str, float] = field(default_factory=dict)
    admission_stall_ms: dict[str, float] = field(default_factory=dict)
    command_count: dict[str, int] = field(default_factory=dict)
    command_bytes: dict[str, int] = field(default_factory=dict)
    shadow_prepared_bytes: int = 0
    useful_shadow_bytes: int = 0
    urgent_d2h_bytes: int = 0
    urgent_h2d_bytes: int = 0
    dropped_bytes: int = 0
    peak_gpu_bytes: int = 0
    peak_cpu_bytes: int = 0
    planner_ticks: int = 0
    local_rejections: int = 0

    def summary(self, final_ts_ms: float, controller: BeliefKVController) -> dict[str, Any]:
        completion = {
            workflow_id: self.workflow_end_ms[workflow_id] - start
            for workflow_id, start in self.workflow_start_ms.items()
            if workflow_id in self.workflow_end_ms
        }
        shadow_resident = sum(
            page.size_bytes
            for page in controller.page_index.pages.values()
            if page.residency == PhysicalResidency.DUAL_CLEAN
        )
        wasted_shadow = max(
            0,
            self.shadow_prepared_bytes
            - self.useful_shadow_bytes
            - shadow_resident,
        )
        return {
            "final_ts_ms": final_ts_ms,
            "workflow_completion_ms": completion,
            "admission_stall_ms": dict(sorted(self.admission_stall_ms.items())),
            "command_count": dict(sorted(self.command_count.items())),
            "command_bytes": dict(sorted(self.command_bytes.items())),
            "shadow_prepared_bytes": self.shadow_prepared_bytes,
            "useful_shadow_bytes": self.useful_shadow_bytes,
            "resident_clean_shadow_bytes": shadow_resident,
            "wasted_shadow_bytes": wasted_shadow,
            "urgent_d2h_bytes": self.urgent_d2h_bytes,
            "urgent_h2d_bytes": self.urgent_h2d_bytes,
            "dropped_bytes": self.dropped_bytes,
            "peak_gpu_bytes": self.peak_gpu_bytes,
            "peak_cpu_bytes": self.peak_cpu_bytes,
            "final_gpu_bytes": controller.page_index.gpu_bytes,
            "final_cpu_bytes": controller.page_index.cpu_bytes,
            "planner_ticks": self.planner_ticks,
            "local_rejections": self.local_rejections,
        }


@dataclass(frozen=True)
class SimulationResult:
    scenario_name: str
    summary: dict[str, Any]
    event_log: tuple[dict[str, Any], ...]
    final_graph: dict[str, object]


class PageLevelSimulator:
    """Deterministic page-level HBM/PCIe replay engine."""

    _EXTERNAL_PHASE = 0
    _TRANSFER_PHASE = 1

    def __init__(
        self,
        config: BeliefKVConfig | None = None,
        *,
        controller: BeliefKVController | None = None,
    ) -> None:
        self.controller = controller or BeliefKVController(config)
        self.config = self.controller.config
        self.metrics = SimulationMetrics()
        self.event_log: list[dict[str, Any]] = []
        self._queue: list[_Scheduled] = []
        self._sequence = 0
        self._cancelled_commands: set[str] = set()
        self._pcie_busy_until_ms = 0.0
        self._admitted_at: dict[str, float] = {}
        self._has_run = False

    def run(self, scenario: SimulationScenario) -> SimulationResult:
        if self._has_run:
            raise RuntimeError("PageLevelSimulator instances are single-use")
        self._has_run = True
        for event in scenario.events:
            self._schedule(
                event.ts_ms,
                self._EXTERNAL_PHASE,
                "external",
                event,
                sequence=event.sequence,
            )
        final_ts = 0.0
        while self._queue:
            scheduled = heapq.heappop(self._queue)
            final_ts = scheduled.ts_ms
            if scheduled.kind == "external":
                self._process_external(scheduled.payload)
            elif scheduled.kind == "transfer_complete":
                self._complete_transfer(scheduled.ts_ms, scheduled.payload)
            else:
                raise SimulationError(f"unknown scheduled kind: {scheduled.kind}")
            self._drive_controller(scheduled.ts_ms)
            self._record_capacity(scheduled.ts_ms)
        return SimulationResult(
            scenario_name=scenario.name,
            summary=self.metrics.summary(final_ts, self.controller),
            event_log=tuple(self.event_log),
            final_graph=self.controller.graph.snapshot(),
        )

    def _process_external(self, event: SimulationEvent) -> None:
        payload = dict(event.payload)
        if event.kind == SimulationEventKind.RUNTIME:
            raw = dict(payload.get("event", payload))
            raw.setdefault("ts_ms", event.ts_ms)
            raw.setdefault("event_id", f"runtime-{event.sequence}")
            runtime_event = RuntimeEvent.from_dict(raw)
            self.controller.process_runtime_event(runtime_event)
            if runtime_event.kind == RuntimeEventKind.WORKFLOW_START:
                self.metrics.workflow_start_ms[runtime_event.workflow_id] = event.ts_ms
            elif runtime_event.kind == RuntimeEventKind.WORKFLOW_END:
                self.metrics.workflow_end_ms[runtime_event.workflow_id] = event.ts_ms
            self._log(event.ts_ms, "runtime", self._runtime_event_payload(runtime_event))
        elif event.kind == SimulationEventKind.CACHE_INSERT:
            self._cache_insert(event.ts_ms, payload)
        elif event.kind == SimulationEventKind.CACHE_BIND:
            handles = tuple(self._handle(item) for item in payload["pages"])
            self.controller.page_index.bind_pages(
                str(payload["context_id"]),
                int(payload["context_epoch"]),
                handles,
                replace=bool(payload.get("replace", False)),
            )
            self._log(event.ts_ms, "cache_bind", payload)
        elif event.kind == SimulationEventKind.PAGE_FREE:
            handle = self._handle(payload)
            self.controller.page_index.free_page(handle)
            self._log(event.ts_ms, "page_free", {"page": asdict(handle)})
        elif event.kind == SimulationEventKind.LOCK_CHANGE:
            handle = self._handle(payload)
            self.controller.page_index.set_engine_lock(handle, int(payload["value"]))
            self._log(event.ts_ms, "lock_change", payload)
        elif event.kind == SimulationEventKind.READER_CHANGE:
            handle = self._handle(payload)
            self.controller.page_index.set_active_readers(handle, int(payload["value"]))
            self._log(event.ts_ms, "reader_change", payload)
        elif event.kind == SimulationEventKind.REQUEST_SUBMIT:
            request = AdmissionRequest(**payload)
            self.controller.submit_request(request)
            self._log(event.ts_ms, "request_submit", asdict(request))
        elif event.kind == SimulationEventKind.ADMISSION_ACK:
            request_id = str(payload["request_id"])
            released = self.controller.acknowledge_admission(request_id)
            self._log(
                event.ts_ms,
                "admission_ack",
                {"request_id": request_id, "released_reservation_bytes": released},
            )
        elif event.kind == SimulationEventKind.SERVICE_CHARGE:
            self.controller.fairness.charge_service(
                str(payload["workflow_id"]), float(payload["service_ms"])
            )
            self._log(event.ts_ms, "service_charge", payload)
        elif event.kind == SimulationEventKind.SIGNAL:
            self.controller.update_signals(**payload)
            self._log(event.ts_ms, "signal", payload)
        elif event.kind == SimulationEventKind.TICK:
            self._log(event.ts_ms, "explicit_tick", {})
        else:
            raise SimulationError(f"unsupported external event: {event.kind}")

    def _cache_insert(self, ts_ms: float, payload: dict[str, Any]) -> None:
        handle = self._handle(payload)
        parent = self._handle(payload["parent"]) if payload.get("parent") else None
        page = self.controller.page_index.register_page(
            handle,
            size_bytes=int(payload["size_bytes"]),
            residency=PhysicalResidency(
                payload.get("residency", PhysicalResidency.GPU_ONLY.value)
            ),
            radix_depth=int(payload.get("radix_depth", 0)),
            parent=parent,
            sealed=bool(payload.get("sealed", True)),
            last_access_ms=ts_ms,
        )
        for binding in payload.get("bindings", []):
            self.controller.page_index.bind_pages(
                str(binding["context_id"]),
                int(binding["context_epoch"]),
                [handle],
            )
        if self.controller.page_index.gpu_bytes > self.config.hbm_capacity_bytes:
            raise SimulationError(
                f"CACHE_INSERT caused HBM OOM: {self.controller.page_index.gpu_bytes} "
                f"> {self.config.hbm_capacity_bytes}"
            )
        if self.controller.page_index.cpu_bytes > self.config.host_capacity_bytes:
            raise SimulationError("CACHE_INSERT caused host KV capacity overflow")
        self._log(
            ts_ms,
            "cache_insert",
            {
                "page": asdict(handle),
                "size_bytes": page.size_bytes,
                "residency": page.residency.value,
                "bindings": list(payload.get("bindings", [])),
            },
        )

    def _drive_controller(self, ts_ms: float) -> None:
        tick = self.controller.tick(ts_ms)
        self.metrics.planner_ticks += 1
        self._record_tick(tick)
        for command_id in tick.cancel_command_ids:
            self._cancelled_commands.add(command_id)
        for ack in tick.local_acks:
            self.metrics.local_rejections += 1
            self._log(ts_ms, "local_ack", self._ack_payload(ack))
        if tick.admission is not None:
            decision = tick.admission
            if decision.admitted:
                self._admitted_at[decision.request_id] = ts_ms
                denied_at = self.metrics.first_admission_denial_ms.pop(
                    decision.request_id, None
                )
                if denied_at is not None:
                    self.metrics.admission_stall_ms[decision.request_id] = ts_ms - denied_at
            else:
                self.metrics.first_admission_denial_ms.setdefault(
                    decision.request_id, ts_ms
                )
        if tick.transfer is not None:
            self._start_transfer(ts_ms, tick.transfer)

    def _start_transfer(self, ts_ms: float, resolved: ResolvedCommand) -> None:
        command = resolved.command
        accepted: list[PageHandle] = []
        host_available = max(
            0, self.config.host_capacity_bytes - self.controller.page_index.cpu_bytes
        )
        hbm_available = max(
            0, self.config.hbm_capacity_bytes - self.controller.page_index.gpu_bytes
        )
        transfer_bytes = 0
        actual_bytes = 0
        for item in resolved.page_actions:
            requires_host = item.action == PhysicalPageAction.START_D2H
            requires_hbm = item.action == PhysicalPageAction.START_H2D
            if requires_host and item.size_bytes > host_available:
                continue
            if requires_hbm and item.size_bytes > hbm_available:
                continue
            accepted.append(item.handle)
            actual_bytes += item.size_bytes
            if requires_host:
                host_available -= item.size_bytes
                transfer_bytes += item.size_bytes
            elif requires_hbm:
                hbm_available -= item.size_bytes
                transfer_bytes += item.size_bytes
        if accepted:
            self.controller.mark_command_started(command.command_id, accepted)
            status = (
                CommandStatus.COMPLETED
                if len(accepted) == len(resolved.page_actions)
                else CommandStatus.PARTIAL
            )
            reason = ""
        else:
            status = CommandStatus.REJECTED
            reason = "simulated_capacity_rejection"
        start_ms = max(ts_ms, self._pcie_busy_until_ms)
        duration_ms = self.controller.cost_model.transfer_ms(transfer_bytes)
        completion_ms = start_ms + duration_ms
        if transfer_bytes:
            self._pcie_busy_until_ms = completion_ms
            self.controller.update_signals(pcie_utilization=1.0)
        completion = _TransferCompletion(
            resolved=resolved,
            accepted_handles=tuple(accepted),
            status=status,
            actual_bytes=actual_bytes,
            reason=reason,
        )
        self._schedule(
            completion_ms,
            self._TRANSFER_PHASE,
            "transfer_complete",
            completion,
        )
        self._log(
            ts_ms,
            "command_start",
            {
                "command_id": command.command_id,
                "kind": command.kind.value,
                "accepted_pages": [asdict(item) for item in accepted],
                "resolved_bytes": resolved.resolved_bytes,
                "actual_bytes": actual_bytes,
                "completion_ts_ms": completion_ms,
            },
        )

    def _complete_transfer(self, ts_ms: float, completion: _TransferCompletion) -> None:
        command = completion.resolved.command
        cancelled = command.command_id in self._cancelled_commands
        if cancelled and command.kind != CommandKind.SHADOW_CONTEXT:
            self._cancelled_commands.discard(command.command_id)
            status = CommandStatus.CANCELLED
            handles: tuple[PageHandle, ...] = ()
            actual_bytes = 0
            reason = "cancelled_by_runtime_wakeup"
        else:
            if cancelled:
                self._cancelled_commands.discard(command.command_id)
            status = completion.status
            handles = completion.accepted_handles
            actual_bytes = completion.actual_bytes
            reason = (
                "current_nonpreemptible_chunk_completed_after_cancel"
                if cancelled
                else completion.reason
            )
        ack = CommandAck(
            command_id=command.command_id,
            status=status,
            completed_ts_ms=ts_ms,
            actual_bytes=actual_bytes,
            page_handles=handles,
            reason=reason,
        )
        self.controller.acknowledge_command(ack)
        self.controller.update_signals(pcie_utilization=0.0)
        self._update_command_metrics(completion.resolved, ack)
        self._log(ts_ms, "command_ack", self._ack_payload(ack))

    def _update_command_metrics(
        self, resolved: ResolvedCommand, ack: CommandAck
    ) -> None:
        kind = resolved.command.kind.value
        self.metrics.command_count[kind] = self.metrics.command_count.get(kind, 0) + 1
        self.metrics.command_bytes[kind] = (
            self.metrics.command_bytes.get(kind, 0) + ack.actual_bytes
        )
        completed = set(ack.page_handles)
        for action in resolved.page_actions:
            if action.handle not in completed:
                continue
            if resolved.command.kind == CommandKind.SHADOW_CONTEXT:
                self.metrics.shadow_prepared_bytes += action.size_bytes
            elif action.action == PhysicalPageAction.COMMIT_CPU:
                self.metrics.useful_shadow_bytes += action.size_bytes
            elif action.action == PhysicalPageAction.START_D2H:
                self.metrics.urgent_d2h_bytes += action.size_bytes
            elif action.action == PhysicalPageAction.START_H2D:
                self.metrics.urgent_h2d_bytes += action.size_bytes
            elif action.action == PhysicalPageAction.DROP:
                self.metrics.dropped_bytes += action.size_bytes
            elif action.action == PhysicalPageAction.DROP_HOST:
                self.metrics.dropped_bytes += action.size_bytes

    def _record_tick(self, tick: ControllerTickResult) -> None:
        payload: dict[str, Any] = {
            "gpu_bytes": self.controller.page_index.gpu_bytes,
            "cpu_bytes": self.controller.page_index.cpu_bytes,
            "urgent_queue_depth": self.controller.command_queue.urgent_count,
            "shadow_queue_depth": self.controller.command_queue.shadow_count,
        }
        if tick.admission is not None:
            payload["admission"] = asdict(tick.admission)
        if tick.transfer is not None:
            payload["transfer"] = {
                "command_id": tick.transfer.command.command_id,
                "kind": tick.transfer.command.kind.value,
                "resolved_bytes": tick.transfer.resolved_bytes,
                "reason": tick.transfer.reason,
            }
        if tick.cancel_command_ids:
            payload["cancel_command_ids"] = list(tick.cancel_command_ids)
        self._log(tick.now_ms, "controller_tick", payload)

    def _record_capacity(self, ts_ms: float) -> None:
        gpu = self.controller.page_index.gpu_bytes
        cpu = self.controller.page_index.cpu_bytes
        self.metrics.peak_gpu_bytes = max(self.metrics.peak_gpu_bytes, gpu)
        self.metrics.peak_cpu_bytes = max(self.metrics.peak_cpu_bytes, cpu)
        if gpu > self.config.hbm_capacity_bytes:
            raise SimulationError(f"HBM invariant violated at {ts_ms} ms")
        if cpu > self.config.host_capacity_bytes:
            raise SimulationError(f"host capacity invariant violated at {ts_ms} ms")
        self.controller.page_index.assert_consistent()

    def _schedule(
        self,
        ts_ms: float,
        phase: int,
        kind: str,
        payload: Any,
        *,
        sequence: int | None = None,
    ) -> None:
        actual_sequence = self._sequence if sequence is None else sequence
        self._sequence = max(self._sequence + 1, actual_sequence + 1)
        heapq.heappush(
            self._queue,
            _Scheduled(ts_ms, phase, actual_sequence, kind, payload),
        )

    @staticmethod
    def _handle(raw: dict[str, Any]) -> PageHandle:
        page_id = raw.get("page_id", raw.get("id"))
        generation = raw.get("allocation_generation", raw.get("generation", 0))
        if page_id is None:
            raise ValueError("page payload requires page_id")
        return PageHandle(int(page_id), int(generation))

    def _log(self, ts_ms: float, kind: str, payload: dict[str, Any]) -> None:
        self.event_log.append({"ts_ms": ts_ms, "kind": kind, "payload": payload})

    @staticmethod
    def _ack_payload(ack: CommandAck) -> dict[str, Any]:
        return {
            "command_id": ack.command_id,
            "status": ack.status.value,
            "actual_bytes": ack.actual_bytes,
            "page_handles": [asdict(item) for item in ack.page_handles],
            "reason": ack.reason,
        }

    @staticmethod
    def _runtime_event_payload(event: RuntimeEvent) -> dict[str, Any]:
        payload = asdict(event)
        payload["kind"] = event.kind.value
        payload["confidence"] = event.confidence.value
        for key in ("relation_type", "context_mode", "execution_mode"):
            value = getattr(event, key)
            payload[key] = value.value if value is not None else None
        return payload
