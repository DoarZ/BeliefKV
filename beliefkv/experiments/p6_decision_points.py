from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Iterable, Mapping

from beliefkv.control.causal_graph import InvocationState, RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.predictor.composer import observed_boundary_action
from beliefkv.predictor.frontier_belief import BeliefScopeBuilder


DECISION_POINT_SCHEMA_VERSION = 2
DEFAULT_DECODE_QUANTUM_TOKENS = 32
DEFAULT_HBM_PRESSURE_RATIO = 0.85

_EVENT_TRIGGERS = {
    RuntimeEventKind.LLM_SUBMIT,
    RuntimeEventKind.LLM_RESULT,
    RuntimeEventKind.TOOL_START,
    RuntimeEventKind.TOOL_END,
    RuntimeEventKind.SPAWN,
    RuntimeEventKind.HANDOFF,
    RuntimeEventKind.RETURN,
    RuntimeEventKind.MESSAGE,
    RuntimeEventKind.JOIN_WAIT,
    RuntimeEventKind.JOIN_SATISFIED,
    RuntimeEventKind.JOIN_TIMEOUT,
    RuntimeEventKind.REACTIVATE,
    RuntimeEventKind.CALL_CENSORED,
}


def build_frontier_decision_points(
    traces: Iterable[Iterable[Mapping[str, Any]]],
    *,
    calls: Iterable[Mapping[str, Any]],
    service_rows: Iterable[Mapping[str, Any]],
    audit_records: Iterable[Mapping[str, Any]],
    transfer_records: Iterable[Mapping[str, Any]],
    run_id: str,
    workflow_metadata: Mapping[str, Mapping[str, Any]],
    decode_quantum_tokens: int = DEFAULT_DECODE_QUANTUM_TOKENS,
    hbm_pressure_ratio: float = DEFAULT_HBM_PRESSURE_RATIO,
) -> list[dict[str, Any]]:
    """Replay observed state and sample only semantically meaningful decision points."""

    if decode_quantum_tokens <= 0:
        raise ValueError("decode decision quantum must be positive")
    if not 0 < hbm_pressure_ratio <= 1:
        raise ValueError("HBM pressure ratio must be in (0, 1]")
    events = sorted(
        (RuntimeEvent.from_dict(raw) for trace in traces for raw in trace),
        key=lambda item: (item.ts_ms, item.workflow_id, item.event_id),
    )
    call_rows = [dict(item) for item in calls]
    service = sorted(
        (dict(item) for item in service_rows),
        key=lambda item: (
            float(item.get("batch_service_complete_ts_ms") or 0.0),
            str(item.get("request_id") or ""),
        ),
    )
    boundaries = _future_boundaries(events)
    calls_by_invocation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calls_by_request: dict[str, dict[str, Any]] = {}
    for call in call_rows:
        invocation_id = str(call.get("invocation_id") or "")
        request_id = str(call.get("request_id") or "")
        if invocation_id:
            calls_by_invocation[invocation_id].append(call)
        if request_id:
            calls_by_request[request_id] = call
    for values in calls_by_invocation.values():
        values.sort(key=lambda item: float(item.get("submit_ts_ms") or 0.0))
    service_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in service:
        request_id = str(row.get("request_id") or "")
        if request_id:
            service_by_request[request_id].append(row)

    resource_samples = _resource_samples(audit_records, hbm_pressure_ratio)
    triggers = _event_triggers(events)
    triggers.extend(_decode_triggers(service, decode_quantum_tokens))
    triggers.extend(resource_samples["triggers"])
    triggers.extend(_transfer_triggers(transfer_records))
    triggers.sort(key=lambda item: (item["ts_ms"], item["priority"], item["trigger_id"]))

    graph = RuntimeCausalContextGraph(strict_timestamps=False)
    event_index = 0
    active_request: dict[str, str] = {}
    boundary_history: dict[str, list[str]] = defaultdict(list)
    latest_resource: Mapping[str, Any] | None = None
    resource_timeline = resource_samples["all"]
    resource_index = 0
    rows: list[dict[str, Any]] = []
    scope_builder = BeliefScopeBuilder()
    for trigger in triggers:
        ts_ms = float(trigger["ts_ms"])
        while event_index < len(events) and events[event_index].ts_ms <= ts_ms:
            event = events[event_index]
            graph.apply(event)
            if event.kind == RuntimeEventKind.LLM_SUBMIT and event.invocation_id:
                request_id = str(event.attributes.get("request_id") or "")
                if request_id:
                    active_request[event.invocation_id] = request_id
            elif event.kind == RuntimeEventKind.LLM_RESULT and event.invocation_id:
                active_request.pop(event.invocation_id, None)
            observed_action = observed_boundary_action(event)
            if event.invocation_id and observed_action is not None:
                boundary_history[event.invocation_id].append(observed_action)
            event_index += 1
        while (
            resource_index < len(resource_timeline)
            and float(resource_timeline[resource_index].get("ts_ms") or 0.0) <= ts_ms
        ):
            latest_resource = resource_timeline[resource_index]
            resource_index += 1

        workflow_ids = _trigger_workflows(trigger, graph)
        for workflow_id in workflow_ids:
            active = tuple(
                sorted(
                    item.invocation_id
                    for item in graph.invocations.values()
                    if item.workflow_id == workflow_id and not item.state.terminal
                )
            )
            if not active:
                continue
            scope = scope_builder.build(graph, active)
            invocation_features = []
            labels = []
            active_tools = [
                item
                for item in graph.invocations.values()
                if item.state == InvocationState.WAIT_TOOL
            ]
            active_tool_count = len(active_tools)
            active_tool_family_counts = Counter(
                item.active_tool_family or "unknown" for item in active_tools
            )
            for invocation_id in scope.invocation_ids:
                invocation = graph.invocations[invocation_id]
                request_id = active_request.get(invocation_id)
                tool_family = invocation.active_tool_family or "unknown"
                invocation_features.append(
                    _invocation_features(
                        invocation,
                        ts_ms=ts_ms,
                        context_epoch=graph.contexts[invocation.context_id].epoch,
                        request_id=request_id,
                        boundary_history=boundary_history.get(invocation_id, ()),
                        service_by_request=service_by_request,
                        calls_by_request=calls_by_request,
                        active_tool_count=active_tool_count,
                        backend_pressure=(
                            f"active_family:{active_tool_family_counts[tool_family]}"
                            if tool_family != "unknown"
                            else "unknown"
                        ),
                    )
                )
                labels.append(
                    _invocation_label(
                        invocation_id,
                        ts_ms=ts_ms,
                        request_id=request_id,
                        boundaries=boundaries,
                        calls_by_invocation=calls_by_invocation,
                        calls_by_request=calls_by_request,
                        service_by_request=service_by_request,
                    )
                )
            metadata = workflow_metadata.get(workflow_id, {})
            digest = hashlib.blake2b(
                f"{run_id}|{trigger['trigger_id']}|{workflow_id}".encode(),
                digest_size=16,
                person=b"bkv-p6-decision",
            ).hexdigest()
            rows.append(
                {
                    "schema_version": DECISION_POINT_SCHEMA_VERSION,
                    "row_type": "frontier_decision_point",
                    "decision_id": f"decision-{digest}",
                    "run_id": run_id,
                    "timestamp_ms": ts_ms,
                    "trigger_kind": trigger["kind"],
                    "trigger_id": trigger["trigger_id"],
                    "trigger_request_id": trigger.get("request_id"),
                    "trigger_attributes": trigger.get("attributes", {}),
                    "workflow_id": workflow_id,
                    "instance_id": metadata.get("instance_id"),
                    "project": metadata.get("project"),
                    "base_commit": metadata.get("base_commit"),
                    "split": metadata.get("split"),
                    "episode_group_id": (
                        f"{run_id}:{workflow_id}:{trigger.get('request_id') or 'runtime'}"
                    ),
                    "graph_version": graph.graph_version,
                    "scope": _scope_payload(scope),
                    "invocations": invocation_features,
                    "observed_resources": _resource_payload(latest_resource),
                    "batch": trigger.get("batch"),
                    "labels": labels,
                    "label_horizon": {
                        "boundary": "next observable runtime boundary",
                        "max_transitions_per_invocation": 1,
                        "timing_conversion": (
                            "none; candidate schedule and physical state are required"
                        ),
                    },
                    "censor_reasons": sorted(
                        {
                            str(item["censor_reason"])
                            for item in labels
                            if item.get("censor_reason")
                        }
                    ),
                    "training_eligible": any(
                        item.get("next_boundary_kind") is not None for item in labels
                    ),
                }
            )
    return rows


def _event_triggers(events: Iterable[RuntimeEvent]) -> list[dict[str, Any]]:
    return [
        {
            "ts_ms": event.ts_ms,
            "priority": 0,
            "kind": event.kind.value,
            "trigger_id": event.event_id,
            "workflow_id": event.workflow_id,
            "request_id": event.attributes.get("request_id"),
            "attributes": {
                key: event.attributes.get(key)
                for key in (
                    "tool_call_id",
                    "tool_name",
                    "tool_family",
                    "backend_class",
                    "prompt_semantic_sha256",
                    "sampling_seed",
                    "status",
                    "censor_reason",
                )
                if event.attributes.get(key) is not None
            },
        }
        for event in events
        if event.kind in _EVENT_TRIGGERS
    ]


def _decode_triggers(
    service: Iterable[Mapping[str, Any]], quantum: int
) -> list[dict[str, Any]]:
    cumulative: dict[str, int] = defaultdict(int)
    next_mark: dict[str, int] = defaultdict(lambda: quantum)
    triggers: list[dict[str, Any]] = []
    for row in service:
        if row.get("phase") != "decode" or not row.get("training_eligible"):
            continue
        request_id = str(row.get("request_id") or "")
        workflow_id = str(row.get("workflow_id") or "")
        if not request_id or not workflow_id:
            continue
        cumulative[request_id] += max(0, int(row.get("token_delta") or 0))
        while cumulative[request_id] >= next_mark[request_id]:
            mark = next_mark[request_id]
            triggers.append(
                {
                    "ts_ms": float(row["batch_service_complete_ts_ms"]),
                    "priority": 1,
                    "kind": "decode_quantum",
                    "trigger_id": f"decode:{request_id}:{mark}",
                    "workflow_id": workflow_id,
                    "request_id": request_id,
                    "batch": {
                        "phase": "decode",
                        "batch_size": row.get("batch_size"),
                        "sequence_tokens_before": row.get("sequence_tokens_before"),
                        "generated_tokens": mark,
                        "quantum_tokens": quantum,
                    },
                }
            )
            next_mark[request_id] += quantum
    return triggers


def _resource_samples(
    records: Iterable[Mapping[str, Any]], threshold: float
) -> dict[str, list[dict[str, Any]]]:
    all_samples = sorted(
        (dict(item) for item in records if item.get("event") == "resource_snapshot"),
        key=lambda item: float(item.get("ts_ms") or 0.0),
    )
    triggers: list[dict[str, Any]] = []
    pressured = False
    for index, item in enumerate(all_samples):
        capacity = int(item.get("hbm_capacity_bytes") or 0)
        used = int(item.get("hbm_used_bytes") or 0)
        current = capacity > 0 and used / capacity >= threshold
        if current and not pressured:
            triggers.append(
                {
                    "ts_ms": float(item["ts_ms"]),
                    "priority": 2,
                    "kind": "hbm_pressure_crossing",
                    "trigger_id": f"resource:{index}:{item.get('sequence', index)}",
                    "workflow_id": None,
                }
            )
        pressured = current
    return {"all": all_samples, "triggers": triggers}


def _transfer_triggers(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, item in enumerate(records):
        if item.get("status") != "completed" or item.get("complete_ts_ms") is None:
            continue
        context_ids = {
            str(value)
            for value in item.get("owner_context_ids", ())
            if value
        }
        if item.get("context_id"):
            context_ids.add(str(item["context_id"]))
        result.append(
            {
                "ts_ms": float(item["complete_ts_ms"]),
                "priority": 3,
                "kind": "transfer_completion",
                "trigger_id": f"transfer:{item.get('command_id') or index}",
                "workflow_id": item.get("workflow_id"),
                "owner_context_ids": sorted(context_ids),
            }
        )
    return result


def _trigger_workflows(
    trigger: Mapping[str, Any], graph: RuntimeCausalContextGraph
) -> tuple[str, ...]:
    workflow_id = str(trigger.get("workflow_id") or "")
    if workflow_id:
        return (workflow_id,) if workflow_id in graph.workflows else ()
    owner_workflows = {
        graph.contexts[context_id].workflow_id
        for context_id in trigger.get("owner_context_ids", ())
        if context_id in graph.contexts
    }
    if trigger.get("kind") == "transfer_completion":
        return tuple(sorted(owner_workflows))
    return tuple(
        sorted(
            workflow.workflow_id
            for workflow in graph.workflows.values()
            if workflow.end_ts_ms is None
        )
    )


def _scope_payload(scope: Any) -> dict[str, Any]:
    def atom_payload(atom: Any) -> dict[str, Any]:
        return {
            "atom_id": atom.atom_id,
            "kind": atom.kind.value,
            "invocation_ids": list(atom.invocation_ids),
            "join_ids": list(atom.join_ids),
            "blocker_set_ids": list(atom.blocker_set_ids),
            "estimated_model_cost": atom.estimated_model_cost,
        }

    return {
        "scope_id": scope.scope_id,
        "graph_version": scope.graph_version,
        "closure_complete": True,
        "physical_blocker_coverage": "unavailable_in_agent_trace",
        "included_atoms": [atom_payload(item) for item in scope.included_atoms],
        "other_atoms": [atom_payload(item) for item in scope.other_atoms],
        "modeled_invocation_ids": list(scope.invocation_ids),
        "residual_invocation_ids": list(scope.residual_invocation_ids),
        "modeled_cost": scope.modeled_cost,
    }


def _invocation_features(
    invocation: Any,
    *,
    ts_ms: float,
    context_epoch: int,
    request_id: str | None,
    boundary_history: Iterable[str],
    service_by_request: Mapping[str, list[dict[str, Any]]],
    calls_by_request: Mapping[str, Mapping[str, Any]],
    active_tool_count: int,
    backend_pressure: str,
) -> dict[str, Any]:
    observed_service = [
        item
        for item in service_by_request.get(str(request_id or ""), ())
        if float(item.get("batch_service_complete_ts_ms") or 0.0) <= ts_ms
    ]
    request = calls_by_request.get(str(request_id or ""), {})
    latest_service = observed_service[-1] if observed_service else {}
    observed_output_tokens = sum(
        max(0, int(item.get("token_delta") or 0))
        for item in observed_service
        if item.get("phase") == "decode"
    )
    current_sequence_tokens = (
        int(latest_service.get("sequence_tokens_before") or 0)
        + max(0, int(latest_service.get("token_delta") or 0))
        if latest_service
        else int(request.get("context_tokens") or 0)
    )
    return {
        "invocation_id": invocation.invocation_id,
        "agent_definition_id": invocation.agent_definition_id,
        "context_id": invocation.context_id,
        "context_epoch": context_epoch,
        "state": invocation.state.value,
        "state_elapsed_ms": max(0.0, ts_ms - invocation.updated_ts_ms),
        "llm_round": invocation.llm_round,
        "request_id": request_id,
        "boundary_history": list(boundary_history)[-8:],
        "context_tokens": request.get("context_tokens"),
        "current_sequence_tokens": current_sequence_tokens,
        "prompt_tokens": request.get("prompt_tokens"),
        "cache_hit_tokens": request.get("cache_hit_tokens"),
        "active_tool_family": invocation.active_tool_family,
        "active_tool_count": active_tool_count,
        "backend_pressure": backend_pressure,
        "active_tool_elapsed_ms": (
            max(0.0, ts_ms - invocation.active_tool_start_ms)
            if invocation.active_tool_start_ms is not None
            else None
        ),
        "unfinished_child_count": len(invocation.blocking_child_ids),
        "child_count": len(invocation.child_invocation_ids),
        "observed_output_tokens": observed_output_tokens,
        "diagnostics": {
            "last_observed_batch_size": latest_service.get("batch_size"),
            "observed_gpu_service_ms": sum(
                float(item.get("batch_service_elapsed_ms") or 0.0)
                for item in observed_service
            ),
            "gpu_service_semantics": (
                "shared_batch_runtime_observation; excluded_from_frontier_fit"
            ),
        },
    }


def _invocation_label(
    invocation_id: str,
    *,
    ts_ms: float,
    request_id: str | None,
    boundaries: Mapping[str, list[dict[str, Any]]],
    calls_by_invocation: Mapping[str, list[dict[str, Any]]],
    calls_by_request: Mapping[str, dict[str, Any]],
    service_by_request: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    boundary = next(
        (
            item
            for item in boundaries.get(invocation_id, ())
            if float(item["timestamp_ms"]) > ts_ms
        ),
        None,
    )
    request = calls_by_request.get(str(request_id or ""), {})
    remaining_service = [
        item
        for item in service_by_request.get(str(request_id or ""), ())
        if float(item.get("batch_service_complete_ts_ms") or 0.0) > ts_ms
    ]
    next_call = next(
        (
            item
            for item in calls_by_invocation.get(invocation_id, ())
            if float(item.get("submit_ts_ms") or 0.0) > ts_ms
        ),
        None,
    )
    current_context_tokens = request.get("context_tokens")
    next_prompt_tokens = next_call.get("prompt_tokens") if next_call else None
    return {
        "invocation_id": invocation_id,
        "next_boundary_kind": boundary.get("kind") if boundary else None,
        "next_boundary_timestamp_ms": boundary.get("timestamp_ms") if boundary else None,
        "next_boundary_delay_ms": (
            max(0.0, float(boundary["timestamp_ms"]) - ts_ms) if boundary else None
        ),
        "next_boundary_status": boundary.get("status") if boundary else None,
        "remaining_output_tokens": (
            sum(
                max(0, int(item.get("token_delta") or 0))
                for item in remaining_service
                if item.get("phase") == "decode"
            )
            if request_id
            else None
        ),
        "reentry_prompt_delta_tokens": (
            max(0, int(next_prompt_tokens) - int(current_context_tokens))
            if next_prompt_tokens is not None and current_context_tokens is not None
            else None
        ),
        "next_output_tokens": next_call.get("output_tokens") if next_call else None,
        "censored": bool(request.get("censored", False)),
        "censor_reason": request.get("censor_reason"),
        "demand_label_semantics": (
            "token demand only; observed shared-batch time is diagnostic"
        ),
    }


def _future_boundaries(events: Iterable[RuntimeEvent]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not event.invocation_id:
            continue
        kind: str | None = None
        status: str | None = None
        if event.kind == RuntimeEventKind.LLM_RESULT:
            actions = [
                str(item) for item in event.attributes.get("structured_action_kinds", ())
            ]
            kind = actions[0] if actions else "final"
            status = "censored" if event.attributes.get("censored") else "observed"
        elif event.kind in {
            RuntimeEventKind.TOOL_END,
            RuntimeEventKind.SPAWN,
            RuntimeEventKind.HANDOFF,
            RuntimeEventKind.RETURN,
            RuntimeEventKind.MESSAGE,
            RuntimeEventKind.JOIN_SATISFIED,
            RuntimeEventKind.JOIN_TIMEOUT,
            RuntimeEventKind.REACTIVATE,
            RuntimeEventKind.CALL_CENSORED,
        }:
            kind = event.kind.value
            status = (
                "censored"
                if event.kind == RuntimeEventKind.CALL_CENSORED
                else str(event.attributes.get("status") or "observed")
            )
        if kind is not None:
            result[event.invocation_id].append(
                {"timestamp_ms": event.ts_ms, "kind": kind, "status": status}
            )
    return result


def _resource_payload(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {
            "availability": "unavailable",
            "hbm_used_bytes": None,
            "hbm_capacity_bytes": None,
            "host_used_bytes": None,
            "host_capacity_bytes": None,
            "inflight_command_count": None,
            "pcie_utilization": None,
        }
    return {
        "availability": "observed_resource_snapshot",
        "snapshot_ts_ms": record.get("ts_ms"),
        "hbm_used_bytes": record.get("hbm_used_bytes"),
        "hbm_capacity_bytes": record.get("hbm_capacity_bytes"),
        "host_used_bytes": record.get("host_used_bytes"),
        "host_capacity_bytes": record.get("host_capacity_bytes"),
        "inflight_command_count": record.get("inflight_command_count"),
        "pcie_utilization": record.get("pcie_utilization"),
        "running_request_count": record.get("running_request_count"),
        "engine_locked_gpu_bytes": record.get("engine_locked_gpu_bytes"),
        "migratable_gpu_bytes": record.get("migratable_gpu_bytes"),
    }
