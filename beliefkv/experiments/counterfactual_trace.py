from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, TextIO

from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.simulator.queue_service import (
    FrozenCounterfactualWorkload,
    FrozenRequestDemand,
    FrozenWorkflowDemand,
    frozen_transition_hash,
)


class CounterfactualTraceError(ValueError):
    """Raised when a runtime trace cannot support causal counterfactual replay."""


@dataclass(frozen=True)
class _ObservedRequest:
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    arrival_ts_ms: float
    finish_ts_ms: float
    prompt_tokens: int
    cache_hit_tokens: int
    output_tokens: int
    action_boundary_token_index: int | None


@dataclass(frozen=True)
class _ToolInterval:
    invocation_id: str
    start_ts_ms: float
    end_ts_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ts_ms - self.start_ts_ms


@dataclass(frozen=True)
class _RequestTokenPath:
    prompt: tuple[int, ...]
    partial_commits: tuple[tuple[int, ...], ...]
    final_commit: tuple[int, ...]


@dataclass(frozen=True)
class CounterfactualTraceBuildResult:
    workload: FrozenCounterfactualWorkload
    request_count: int
    workflow_count: int
    dependency_edge_count: int
    semantic_edge_count: int
    tool_interval_count: int
    arrival_source_counts: Mapping[str, int]
    request_physical_delta_coverage: float
    request_prefix_identity_coverage: float
    initial_radix_state_known: bool

    def summary(self) -> dict[str, object]:
        return {
            "trace_id": self.workload.trace_id,
            "transition_hash": self.workload.transition_hash,
            "trace_sensitivity": self.workload.trace_sensitivity,
            "request_count": self.request_count,
            "workflow_count": self.workflow_count,
            "dependency_edge_count": self.dependency_edge_count,
            "semantic_edge_count": self.semantic_edge_count,
            "tool_interval_count": self.tool_interval_count,
            "arrival_source_counts": dict(self.arrival_source_counts),
            "request_physical_delta_coverage": (
                self.request_physical_delta_coverage
            ),
            "request_prefix_identity_coverage": (
                self.request_prefix_identity_coverage
            ),
            "semantic_events_frozen": self.workload.semantic_events_frozen,
            "token_demand_frozen": self.workload.token_demand_frozen,
            "tool_duration_frozen": self.workload.tool_duration_frozen,
            "future_physical_growth_exact": (
                self.workload.future_physical_growth_exact
            ),
            "prefix_identity_complete": self.workload.prefix_identity_complete,
            "initial_radix_state_known": self.initial_radix_state_known,
        }


class CounterfactualTraceBuilder:
    """Build a request DAG without carrying original GPU queue/service time."""

    _ARRIVAL_EVENTS = (
        "request_received",
        "request_deferred",
        "request_admitted",
        "request_started",
    )

    def build(
        self,
        runtime_event_path: Path,
        runtime_audit_path: Path,
        *,
        workflow_ids: Iterable[str] | None = None,
        trace_sensitivity: str | None = None,
        exact_kv_growth_bytes_by_request: Mapping[str, int] | None = None,
        request_token_trace_path: Path | None = None,
    ) -> CounterfactualTraceBuildResult:
        events = self._runtime_events(runtime_event_path)
        selected_workflows = {
            str(item) for item in (workflow_ids or ()) if str(item)
        }
        if selected_workflows:
            available_workflows = {
                event.workflow_id for event in events if event.workflow_id
            }
            missing_workflows = selected_workflows - available_workflows
            if missing_workflows:
                raise CounterfactualTraceError(
                    "requested workflows are absent from the runtime trace: "
                    f"{sorted(missing_workflows)}"
                )
            events = tuple(
                event
                for event in events
                if event.workflow_id in selected_workflows
            )
        audit = tuple(_read_jsonl(runtime_audit_path))
        observed, arrival_sources, kv_bytes_per_token = self._requests(events, audit)
        if not observed:
            raise CounterfactualTraceError("trace contains no complete LLM requests")
        request_by_id = {item.request_id: item for item in observed}
        token_paths, initial_radix_state_known = (
            self._request_token_paths(
                request_token_trace_path,
                request_ids=set(request_by_id),
            )
            if request_token_trace_path is not None
            else ({}, False)
        )
        for request_id, path in token_paths.items():
            request = request_by_id[request_id]
            if len(path.prompt) != request.prompt_tokens:
                raise CounterfactualTraceError(
                    f"prompt token trace length mismatch for {request_id}"
                )
            expected_cache_tokens = max(
                0, request.prompt_tokens + request.output_tokens - 1
            )
            if len(path.final_commit) != expected_cache_tokens:
                raise CounterfactualTraceError(
                    f"final cache token trace length mismatch for {request_id}"
                )
        workflows = {item.workflow_id for item in observed}
        starts, ends = self._workflow_bounds(events, workflows)
        predecessors: dict[str, set[str]] = {
            request_id: set() for request_id in request_by_id
        }
        semantic_edges: set[tuple[str, str, str]] = set()
        by_invocation: dict[str, list[_ObservedRequest]] = defaultdict(list)
        for request in observed:
            by_invocation[request.invocation_id].append(request)
        for requests in by_invocation.values():
            requests.sort(key=lambda item: (item.arrival_ts_ms, item.request_id))
            for previous, current in zip(requests, requests[1:]):
                if current.arrival_ts_ms + 1e-3 < previous.finish_ts_ms:
                    raise CounterfactualTraceError(
                        "overlapping requests mutate one invocation context: "
                        f"{previous.request_id}, {current.request_id}"
                    )
                predecessors[current.request_id].add(previous.request_id)

        self._add_parent_edges(
            events,
            by_invocation,
            predecessors,
            semantic_edges,
        )
        self._add_communication_edges(
            events,
            by_invocation,
            predecessors,
            semantic_edges,
        )
        self._add_join_edges(
            events,
            by_invocation,
            predecessors,
            semantic_edges,
        )
        tools = self._tool_intervals(events)
        observed_allocator_growth = self._observed_allocator_growth(audit)
        tools_by_invocation: dict[str, list[_ToolInterval]] = defaultdict(list)
        for tool in tools:
            tools_by_invocation[tool.invocation_id].append(tool)

        exact_growth = exact_kv_growth_bytes_by_request is not None
        if exact_growth:
            missing = set(request_by_id) - set(exact_kv_growth_bytes_by_request or {})
            extra = set(exact_kv_growth_bytes_by_request or {}) - set(request_by_id)
            if missing or extra:
                raise CounterfactualTraceError(
                    "exact KV growth mapping must cover the request trace exactly: "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
        context_tokens: dict[str, int] = defaultdict(int)
        frozen_requests: list[FrozenRequestDemand] = []
        for request in sorted(
            observed, key=lambda item: (item.arrival_ts_ms, item.request_id)
        ):
            predecessor_ids = tuple(sorted(predecessors[request.request_id]))
            dependency_finish = max(
                (
                    request_by_id[item].finish_ts_ms
                    for item in predecessor_ids
                ),
                default=starts[request.workflow_id],
            )
            release_delay = request.arrival_ts_ms - dependency_finish
            if release_delay < -1e-3:
                raise CounterfactualTraceError(
                    f"request {request.request_id} arrives before a causal predecessor"
                )
            release_delay = max(0.0, release_delay)
            uncached = request.prompt_tokens - request.cache_hit_tokens
            startup_bytes = (
                uncached + request.output_tokens
            ) * kv_bytes_per_token
            total_context_tokens = request.prompt_tokens + request.output_tokens
            logical_growth_tokens = max(
                0,
                total_context_tokens - context_tokens[request.context_id],
            )
            context_tokens[request.context_id] = max(
                context_tokens[request.context_id], total_context_tokens
            )
            inferred_growth = min(
                startup_bytes,
                logical_growth_tokens * kv_bytes_per_token,
            )
            kv_growth = (
                int((exact_kv_growth_bytes_by_request or {})[request.request_id])
                if exact_growth
                else observed_allocator_growth.get(
                    request.request_id, inferred_growth
                )
            )
            startup_bytes = max(startup_bytes, kv_growth)
            if kv_growth < 0 or kv_growth > startup_bytes:
                raise CounterfactualTraceError(
                    f"invalid physical growth for request {request.request_id}"
                )
            tool_duration = sum(
                item.duration_ms
                for item in tools_by_invocation.get(request.invocation_id, ())
                if item.start_ts_ms + 1e-3 >= dependency_finish
                and item.end_ts_ms <= request.arrival_ts_ms + 1e-3
            )
            frozen_requests.append(
                FrozenRequestDemand(
                    request_id=request.request_id,
                    workflow_id=request.workflow_id,
                    invocation_id=request.invocation_id,
                    context_id=request.context_id,
                    context_epoch=request.context_epoch,
                    predecessor_request_ids=predecessor_ids,
                    release_delay_ms=release_delay,
                    uncached_prompt_tokens=uncached,
                    output_tokens=request.output_tokens,
                    startup_bytes=startup_bytes,
                    kv_growth_bytes=kv_growth,
                    action_boundary_token_index=(
                        request.action_boundary_token_index
                    ),
                    tool_duration_ms=tool_duration,
                    observed_cache_hit_tokens=request.cache_hit_tokens,
                    prompt_token_symbols=(
                        token_paths[request.request_id].prompt
                        if request.request_id in token_paths
                        else ()
                    ),
                    cache_commit_token_symbols=(
                        token_paths[request.request_id].final_commit
                        if request.request_id in token_paths
                        else ()
                    ),
                    partial_cache_commit_token_symbols=(
                        token_paths[request.request_id].partial_commits
                        if request.request_id in token_paths
                        else ()
                    ),
                )
            )

        successor_count: Counter[str] = Counter()
        for request in frozen_requests:
            successor_count.update(request.predecessor_request_ids)
        frozen_workflows = []
        origin = min(starts.values())
        for workflow_id in sorted(workflows):
            workflow_requests = [
                item for item in observed if item.workflow_id == workflow_id
            ]
            terminals = tuple(
                sorted(
                    item.request_id
                    for item in workflow_requests
                    if successor_count[item.request_id] == 0
                )
            )
            if not terminals:
                raise CounterfactualTraceError(
                    f"workflow has no terminal request: {workflow_id}"
                )
            latest_finish = max(item.finish_ts_ms for item in workflow_requests)
            completion_delay = ends[workflow_id] - latest_finish
            if completion_delay < -1e-3:
                raise CounterfactualTraceError(
                    f"workflow ends before its final request: {workflow_id}"
                )
            frozen_workflows.append(
                FrozenWorkflowDemand(
                    workflow_id=workflow_id,
                    release_ms=starts[workflow_id] - origin,
                    terminal_request_ids=terminals,
                    completion_delay_ms=max(0.0, completion_delay),
                )
            )

        transition_records = self._transition_records(events)
        transition_hash = frozen_transition_hash(transition_records)
        sensitivity = trace_sensitivity or self._trace_sensitivity(events)
        trace_id = frozen_transition_hash(
            (
                {
                    "transition_hash": transition_hash,
                    "request_ids": sorted(request_by_id),
                    "workflow_ids": sorted(workflows),
                },
            )
        )
        workload = FrozenCounterfactualWorkload(
            trace_id=trace_id,
            transition_hash=transition_hash,
            trace_sensitivity=sensitivity,
            requests=tuple(frozen_requests),
            workflows=tuple(frozen_workflows),
            semantic_events_frozen=True,
            token_demand_frozen=True,
            tool_duration_frozen=True,
            future_physical_growth_exact=exact_growth,
            prefix_identity_complete=(
                bool(token_paths) and len(token_paths) == len(request_by_id)
            ),
            initial_radix_state_known=initial_radix_state_known,
            metadata={
                "runtime_event_path": str(runtime_event_path.expanduser().resolve()),
                "runtime_audit_path": str(runtime_audit_path.expanduser().resolve()),
                "selected_workflow_ids": sorted(selected_workflows),
                "kv_bytes_per_token": kv_bytes_per_token,
                "arrival_semantics": (
                    "earliest request_received/deferred/admitted/started audit event"
                ),
                "release_delay_semantics": (
                    "arrival minus latest causal predecessor finish; original GPU queue "
                    "and request service are excluded"
                ),
                "future_physical_growth_semantics": (
                    "exact per-request physical unique bytes"
                    if exact_growth
                    else "logical context delta estimate; timing evidence must fail closed"
                ),
                "request_physical_delta_coverage": (
                    len(set(request_by_id).intersection(observed_allocator_growth))
                    / len(request_by_id)
                ),
                "request_token_trace_path": (
                    str(request_token_trace_path.expanduser().resolve())
                    if request_token_trace_path is not None
                    else None
                ),
                "prefix_identity_semantics": (
                    "exact run-local token equality path; token-ID mapping discarded"
                    if token_paths
                    else "unavailable"
                ),
                "observed_allocator_growth_semantics": (
                    "exact allocation demand under the observed cache-hit outcome; "
                    "candidate policies must still recompute cache hits"
                ),
                "observed_request_order": [
                    item.request_id
                    for item in sorted(
                        observed,
                        key=lambda item: (item.arrival_ts_ms, item.request_id),
                    )
                ],
                "observed_request_order_semantics": (
                    "ascending causal arrival timestamp with request-ID tie break; "
                    "this is not GPU completion order"
                ),
            },
        )
        return CounterfactualTraceBuildResult(
            workload=workload,
            request_count=len(frozen_requests),
            workflow_count=len(frozen_workflows),
            dependency_edge_count=sum(
                len(item.predecessor_request_ids) for item in frozen_requests
            ),
            semantic_edge_count=len(semantic_edges),
            tool_interval_count=len(tools),
            arrival_source_counts=dict(sorted(arrival_sources.items())),
            request_physical_delta_coverage=(
                len(set(request_by_id).intersection(observed_allocator_growth))
                / len(request_by_id)
            ),
            request_prefix_identity_coverage=(
                len(token_paths) / len(request_by_id)
            ),
            initial_radix_state_known=initial_radix_state_known,
        )

    @staticmethod
    def _request_token_paths(
        path: Path,
        *,
        request_ids: set[str],
    ) -> tuple[dict[str, _RequestTokenPath], bool]:
        records: list[
            tuple[int, str, str | None, tuple[int, ...], int | None]
        ] = []
        run_ids = set()
        previous_sequence = 0
        for raw in _read_jsonl(path):
            if int(raw.get("schema_version", 0)) != 1:
                raise CounterfactualTraceError(
                    "unsupported request token trace schema"
                )
            sequence = int(raw.get("sequence", 0))
            if sequence <= previous_sequence:
                raise CounterfactualTraceError(
                    "request token trace sequence is not strictly increasing"
                )
            previous_sequence = sequence
            run_ids.add(str(raw.get("run_id", "")))
            event = str(raw.get("event", ""))
            if event not in {
                "cache_reset",
                "request_prompt",
                "cache_partial_commit",
                "cache_final_commit",
            }:
                raise CounterfactualTraceError(
                    f"unsupported request token trace event: {event}"
                )
            symbols = _decode_token_trace_symbols(raw)
            request_id = raw.get("request_id")
            if event != "cache_reset" and (
                not isinstance(request_id, str) or not request_id
            ):
                raise CounterfactualTraceError(
                    "request token trace event lacks request_id"
                )
            records.append(
                (
                    sequence,
                    event,
                    request_id if isinstance(request_id, str) else None,
                    symbols,
                    (
                        int(raw["chunk_index"])
                        if event == "cache_partial_commit"
                        else None
                    ),
                )
            )
        if len(run_ids) != 1 or "" in run_ids:
            raise CounterfactualTraceError(
                "request token trace must contain exactly one run_id"
            )

        selected_records = [item for item in records if item[2] in request_ids]
        if not selected_records:
            raise CounterfactualTraceError(
                "request token trace has no records for the selected requests"
            )
        first_sequence = min(item[0] for item in selected_records)
        last_sequence = max(item[0] for item in selected_records)
        reset_sequence = max(
            (
                sequence
                for sequence, event, _, _, _ in records
                if event == "cache_reset" and sequence < first_sequence
            ),
            default=None,
        )
        if any(
            event == "cache_reset" and first_sequence <= sequence <= last_sequence
            for sequence, event, _, _, _ in records
        ):
            raise CounterfactualTraceError(
                "cache reset inside the selected request interval"
            )
        segment_start = reset_sequence if reset_sequence is not None else 0
        interference = sorted(
            {
                request_id
                for sequence, event, request_id, _, _ in records
                if segment_start < sequence <= last_sequence
                and event != "cache_reset"
                and request_id not in request_ids
                and request_id is not None
            }
        )
        if interference:
            raise CounterfactualTraceError(
                "unselected requests alter the Radix state inside the selected "
                f"token-trace segment: {interference}"
            )

        prompts: dict[str, tuple[int, ...]] = {}
        finals: dict[str, tuple[int, ...]] = {}
        partials: dict[str, list[tuple[int, ...]]] = defaultdict(list)
        for _, event, request_id, symbols, chunk_index in selected_records:
            assert request_id is not None
            if event == "request_prompt":
                if request_id in prompts:
                    raise CounterfactualTraceError(
                        f"duplicate request prompt token trace: {request_id}"
                    )
                prompts[request_id] = symbols
            elif event == "cache_partial_commit":
                assert chunk_index is not None
                if chunk_index != len(partials[request_id]):
                    raise CounterfactualTraceError(
                        f"non-contiguous partial cache trace: {request_id}"
                    )
                partials[request_id].append(symbols)
            elif event == "cache_final_commit":
                if request_id in finals:
                    raise CounterfactualTraceError(
                        f"duplicate final cache token trace: {request_id}"
                    )
                finals[request_id] = symbols
        complete_ids = set(prompts).intersection(finals)
        missing = request_ids - complete_ids
        if missing:
            raise CounterfactualTraceError(
                "request token trace must exactly cover completed requests: "
                f"missing={sorted(missing)}, extra=[]"
            )
        result = {}
        for request_id in sorted(request_ids):
            prompt = prompts[request_id]
            final = finals[request_id]
            shared = min(len(prompt), len(final))
            if prompt[:shared] != final[:shared]:
                raise CounterfactualTraceError(
                    f"final cache path does not extend prompt: {request_id}"
                )
            request_partials = tuple(partials.get(request_id, ()))
            if any(final[: len(item)] != item for item in request_partials):
                raise CounterfactualTraceError(
                    f"partial cache path does not extend to final: {request_id}"
                )
            result[request_id] = _RequestTokenPath(
                prompt=prompt,
                partial_commits=request_partials,
                final_commit=final,
            )
        return result, reset_sequence is not None

    def _requests(
        self,
        events: tuple[RuntimeEvent, ...],
        audit: tuple[dict[str, object], ...],
    ) -> tuple[tuple[_ObservedRequest, ...], Counter[str], int]:
        submits: dict[str, RuntimeEvent] = {}
        results: dict[str, RuntimeEvent] = {}
        structured_actions: dict[
            tuple[str, str, str, int], RuntimeEvent
        ] = {}
        for event in events:
            if event.kind == RuntimeEventKind.STRUCTURED_ACTION:
                if (
                    event.invocation_id is None
                    or event.context_id is None
                    or event.context_epoch is None
                ):
                    raise CounterfactualTraceError(
                        f"structured action identity is incomplete: {event.event_id}"
                    )
                action_key = (
                    event.workflow_id,
                    event.invocation_id,
                    event.context_id,
                    event.context_epoch,
                )
                if action_key in structured_actions:
                    raise CounterfactualTraceError(
                        "multiple structured actions bind to one invocation epoch: "
                        f"{action_key}"
                    )
                structured_actions[action_key] = event
                continue
            if event.kind not in {
                RuntimeEventKind.LLM_SUBMIT,
                RuntimeEventKind.LLM_RESULT,
            }:
                continue
            request_id = event.attributes.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise CounterfactualTraceError(
                    f"{event.event_id} lacks an exact request_id"
                )
            target = (
                submits
                if event.kind == RuntimeEventKind.LLM_SUBMIT
                else results
            )
            if request_id in target:
                raise CounterfactualTraceError(
                    f"duplicate {event.kind.value} for request {request_id}"
                )
            target[request_id] = event
        if set(submits) != set(results):
            raise CounterfactualTraceError(
                "LLM submit/result request sets differ: "
                f"{sorted(set(submits) ^ set(results))}"
            )

        audit_by_request: dict[str, list[dict[str, object]]] = defaultdict(list)
        kv_bytes_per_token: int | None = None
        for raw in audit:
            if raw.get("event") == "runtime_initialized":
                value = raw.get("kv_bytes_per_token")
                if value is not None:
                    kv_bytes_per_token = int(value)
            request_id = raw.get("request_id")
            if isinstance(request_id, str):
                audit_by_request[request_id].append(raw)
        if kv_bytes_per_token is None or kv_bytes_per_token <= 0:
            raise CounterfactualTraceError(
                "runtime audit lacks a positive kv_bytes_per_token"
            )

        source_counts: Counter[str] = Counter()
        observed = []
        for request_id, submit in submits.items():
            result = results[request_id]
            records = audit_by_request.get(request_id, ())
            arrival, source = self._arrival(records, fallback=submit.ts_ms)
            finish = self._audit_timestamp(
                records,
                "request_finished",
                fallback=result.ts_ms,
            )
            source_counts[source] += 1
            prompt_tokens = self._integer_attribute(submit, "prompt_tokens")
            cache_hit_tokens = self._integer_attribute(submit, "cache_hit_tokens")
            output_tokens = self._integer_attribute(result, "output_tokens")
            if cache_hit_tokens > prompt_tokens:
                raise CounterfactualTraceError(
                    f"cache hit exceeds prompt demand: {request_id}"
                )
            if (
                submit.invocation_id is None
                or submit.context_id is None
                or submit.context_epoch is None
            ):
                raise CounterfactualTraceError(
                    f"request identity is incomplete: {request_id}"
                )
            action = structured_actions.get(
                (
                    submit.workflow_id,
                    submit.invocation_id,
                    submit.context_id,
                    submit.context_epoch,
                )
            )
            action_source = action or result
            action_output_tokens = action_source.attributes.get("output_tokens")
            if action is not None and action_output_tokens is not None and (
                int(action_output_tokens) != output_tokens
            ):
                raise CounterfactualTraceError(
                    f"structured action output demand mismatch: {request_id}"
                )
            boundary = action_source.attributes.get(
                "action_boundary_token_index"
            )
            observed.append(
                _ObservedRequest(
                    request_id=request_id,
                    workflow_id=submit.workflow_id,
                    invocation_id=submit.invocation_id,
                    context_id=submit.context_id,
                    context_epoch=submit.context_epoch,
                    arrival_ts_ms=arrival,
                    finish_ts_ms=finish,
                    prompt_tokens=prompt_tokens,
                    cache_hit_tokens=cache_hit_tokens,
                    output_tokens=output_tokens,
                    action_boundary_token_index=(
                        int(boundary) if boundary is not None else output_tokens
                    ),
                )
            )
        return tuple(observed), source_counts, kv_bytes_per_token

    @staticmethod
    def _workflow_bounds(
        events: tuple[RuntimeEvent, ...], workflows: set[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        starts: dict[str, float] = {}
        ends: dict[str, float] = {}
        for event in events:
            if event.kind == RuntimeEventKind.WORKFLOW_START:
                if event.workflow_id in starts:
                    raise CounterfactualTraceError("workflow starts more than once")
                starts[event.workflow_id] = event.ts_ms
            elif event.kind == RuntimeEventKind.WORKFLOW_END:
                if event.workflow_id in ends:
                    raise CounterfactualTraceError("workflow ends more than once")
                ends[event.workflow_id] = event.ts_ms
        if set(starts) != workflows or set(ends) != workflows:
            raise CounterfactualTraceError(
                "request workflows do not match complete workflow boundaries"
            )
        return starts, ends

    @staticmethod
    def _add_parent_edges(
        events: tuple[RuntimeEvent, ...],
        by_invocation: Mapping[str, list[_ObservedRequest]],
        predecessors: dict[str, set[str]],
        semantic_edges: set[tuple[str, str, str]],
    ) -> None:
        for event in events:
            if (
                event.kind != RuntimeEventKind.INVOCATION_CREATE
                or event.parent_invocation_id is None
                or event.invocation_id is None
            ):
                continue
            children = by_invocation.get(event.invocation_id, ())
            if not children:
                continue
            child = min(children, key=lambda item: item.arrival_ts_ms)
            parent = _latest_finished_before(
                by_invocation.get(event.parent_invocation_id, ()),
                child.arrival_ts_ms,
            )
            if parent is None:
                raise CounterfactualTraceError(
                    f"child {event.invocation_id} has no completed parent request"
                )
            predecessors[child.request_id].add(parent.request_id)
            semantic_edges.add((parent.request_id, child.request_id, "spawn"))

    @staticmethod
    def _add_communication_edges(
        events: tuple[RuntimeEvent, ...],
        by_invocation: Mapping[str, list[_ObservedRequest]],
        predecessors: dict[str, set[str]],
        semantic_edges: set[tuple[str, str, str]],
    ) -> None:
        for event in events:
            if event.kind not in {RuntimeEventKind.HANDOFF, RuntimeEventKind.MESSAGE}:
                continue
            if event.invocation_id is None or event.target_invocation_id is None:
                raise CounterfactualTraceError(
                    f"{event.kind.value} lacks source or target identity"
                )
            source = _latest_finished_before(
                by_invocation.get(event.invocation_id, ()), event.ts_ms + 1e-3
            )
            target = _first_arriving_after(
                by_invocation.get(event.target_invocation_id, ()), event.ts_ms - 1e-3
            )
            if source is None or target is None:
                raise CounterfactualTraceError(
                    f"cannot bind {event.kind.value} to request identities"
                )
            predecessors[target.request_id].add(source.request_id)
            semantic_edges.add(
                (source.request_id, target.request_id, event.kind.value)
            )

    @staticmethod
    def _add_join_edges(
        events: tuple[RuntimeEvent, ...],
        by_invocation: Mapping[str, list[_ObservedRequest]],
        predecessors: dict[str, set[str]],
        semantic_edges: set[tuple[str, str, str]],
    ) -> None:
        members: dict[str, tuple[str, ...]] = {}
        waiter: dict[str, str] = {}
        for event in events:
            if event.kind == RuntimeEventKind.JOIN_CREATE and event.join_id:
                members[event.join_id] = event.member_invocation_ids
            elif (
                event.kind == RuntimeEventKind.JOIN_WAIT
                and event.join_id
                and event.invocation_id
            ):
                waiter[event.join_id] = event.invocation_id
            elif event.kind == RuntimeEventKind.JOIN_SATISFIED and event.join_id:
                parent_id = waiter.get(event.join_id)
                if parent_id is None or event.join_id not in members:
                    raise CounterfactualTraceError(
                        f"join lifecycle is incomplete: {event.join_id}"
                    )
                target = _first_arriving_after(
                    by_invocation.get(parent_id, ()), event.ts_ms - 1e-3
                )
                if target is None:
                    continue
                for member_id in members[event.join_id]:
                    source = _latest_finished_before(
                        by_invocation.get(member_id, ()), event.ts_ms + 1e-3
                    )
                    if source is None:
                        raise CounterfactualTraceError(
                            f"join member has no completed request: {member_id}"
                        )
                    predecessors[target.request_id].add(source.request_id)
                    semantic_edges.add(
                        (source.request_id, target.request_id, "join_all")
                    )

    @staticmethod
    def _tool_intervals(events: tuple[RuntimeEvent, ...]) -> tuple[_ToolInterval, ...]:
        starts: dict[str, RuntimeEvent] = {}
        result = []
        for event in events:
            if event.kind not in {RuntimeEventKind.TOOL_START, RuntimeEventKind.TOOL_END}:
                continue
            tool_call_id = event.attributes.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise CounterfactualTraceError("tool event lacks tool_call_id")
            if event.kind == RuntimeEventKind.TOOL_START:
                if tool_call_id in starts:
                    raise CounterfactualTraceError("tool call starts more than once")
                starts[tool_call_id] = event
                continue
            start = starts.pop(tool_call_id, None)
            if start is None or start.invocation_id != event.invocation_id:
                raise CounterfactualTraceError("tool lifecycle identity mismatch")
            start_ts_ms = _source_ts_ms(start)
            end_ts_ms = _source_ts_ms(event)
            duration = end_ts_ms - start_ts_ms
            declared = event.attributes.get("duration_ms")
            if declared is not None and abs(float(declared) - duration) > 1e-3:
                raise CounterfactualTraceError("tool duration disagrees with timestamps")
            if event.invocation_id is None or duration < 0:
                raise CounterfactualTraceError("tool interval is invalid")
            result.append(
                _ToolInterval(event.invocation_id, start_ts_ms, end_ts_ms)
            )
        if starts:
            raise CounterfactualTraceError(
                f"unterminated tool calls: {sorted(starts)}"
            )
        return tuple(result)

    @staticmethod
    def _observed_allocator_growth(
        audit: tuple[dict[str, object], ...]
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        for raw in audit:
            if raw.get("event") != "request_physical_delta":
                continue
            request_id = raw.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise CounterfactualTraceError(
                    "request_physical_delta lacks request identity"
                )
            if not bool(raw.get("allocator_growth_exact", False)):
                continue
            bytes_ = int(raw["allocator_growth_bytes_upper_bound"])
            if bytes_ < 0 or request_id in result:
                raise CounterfactualTraceError(
                    f"invalid or duplicate request physical delta: {request_id}"
                )
            result[request_id] = bytes_
        return result

    @staticmethod
    def _transition_records(
        events: tuple[RuntimeEvent, ...]
    ) -> tuple[dict[str, object], ...]:
        excluded = {RuntimeEventKind.LLM_SUBMIT, RuntimeEventKind.TOOL_START}
        attribute_names = (
            "structured_action_kinds",
            "structured_action_names",
            "outcome",
            "mode",
            "tool_name",
            "exception_type",
        )
        records = []
        for event in events:
            if event.kind in excluded:
                continue
            records.append(
                {
                    "kind": event.kind.value,
                    "workflow_id": event.workflow_id,
                    "invocation_id": event.invocation_id,
                    "target_invocation_id": event.target_invocation_id,
                    "parent_invocation_id": event.parent_invocation_id,
                    "join_id": event.join_id,
                    "member_invocation_ids": list(event.member_invocation_ids),
                    "relation_type": (
                        event.relation_type.value if event.relation_type else None
                    ),
                    "context_mode": (
                        event.context_mode.value if event.context_mode else None
                    ),
                    "attributes": {
                        name: event.attributes[name]
                        for name in attribute_names
                        if name in event.attributes
                    },
                }
            )
        return tuple(records)

    @staticmethod
    def _trace_sensitivity(events: tuple[RuntimeEvent, ...]) -> str:
        values = {
            str(event.attributes["trace_sensitivity"])
            for event in events
            if event.attributes.get("trace_sensitivity")
        }
        if len(values) > 1:
            raise CounterfactualTraceError(
                f"trace declares conflicting sensitivities: {sorted(values)}"
            )
        return next(iter(values), "timing_sensitive")

    @staticmethod
    def _runtime_events(path: Path) -> tuple[RuntimeEvent, ...]:
        result = []
        for sequence, raw in enumerate(_read_jsonl(path), start=1):
            try:
                result.append(RuntimeEvent.from_dict(raw))
            except (KeyError, TypeError, ValueError) as error:
                raise CounterfactualTraceError(
                    f"invalid runtime event at line {sequence}: {error}"
                ) from error
        result.sort(key=lambda item: (item.ts_ms, item.event_id))
        return tuple(result)

    @classmethod
    def _arrival(
        cls,
        records: Iterable[Mapping[str, object]],
        *,
        fallback: float,
    ) -> tuple[float, str]:
        by_kind: dict[str, list[float]] = defaultdict(list)
        for raw in records:
            event = str(raw.get("event", ""))
            if event in cls._ARRIVAL_EVENTS:
                by_kind[event].append(float(raw["ts_ms"]))
        for event in cls._ARRIVAL_EVENTS:
            if by_kind[event]:
                return min(by_kind[event]), event
        return fallback, "runtime_llm_submit_fallback"

    @staticmethod
    def _audit_timestamp(
        records: Iterable[Mapping[str, object]],
        event: str,
        *,
        fallback: float,
    ) -> float:
        values = [
            float(raw["ts_ms"])
            for raw in records
            if raw.get("event") == event
        ]
        return max(values) if values else fallback

    @staticmethod
    def _integer_attribute(event: RuntimeEvent, name: str) -> int:
        value = event.attributes.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CounterfactualTraceError(
                f"{event.event_id} lacks integer {name}"
            )
        result = int(value)
        if result < 0 or not math.isclose(float(value), result):
            raise CounterfactualTraceError(
                f"{event.event_id} has invalid integer {name}"
            )
        return result


def _latest_finished_before(
    requests: Iterable[_ObservedRequest], ts_ms: float
) -> _ObservedRequest | None:
    eligible = [item for item in requests if item.finish_ts_ms <= ts_ms + 1e-3]
    return max(eligible, key=lambda item: item.finish_ts_ms) if eligible else None


def _first_arriving_after(
    requests: Iterable[_ObservedRequest], ts_ms: float
) -> _ObservedRequest | None:
    eligible = [item for item in requests if item.arrival_ts_ms >= ts_ms - 1e-3]
    return min(eligible, key=lambda item: item.arrival_ts_ms) if eligible else None


def _source_ts_ms(event: RuntimeEvent) -> float:
    value = event.attributes.get("beliefkv_source_ts_ms")
    return float(value) if value is not None else event.ts_ms


def _decode_token_trace_symbols(raw: Mapping[str, object]) -> tuple[int, ...]:
    if (
        raw.get("token_encoding")
        != "run_local_random_u64_bijection+uint64_le_base64"
    ):
        raise CounterfactualTraceError("unsupported request token trace encoding")
    encoded = raw.get("token_symbols_b64")
    if not isinstance(encoded, str):
        raise CounterfactualTraceError("request token trace lacks encoded symbols")
    try:
        packed = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise CounterfactualTraceError(
            "request token trace contains invalid base64"
        ) from error
    token_count = int(raw.get("token_count", -1))
    if token_count < 0 or len(packed) != token_count * 8:
        raise CounterfactualTraceError("request token trace length is inconsistent")
    digest = hashlib.blake2b(
        packed,
        digest_size=16,
        person=b"bk-token-trace",
    ).hexdigest()
    if digest != raw.get("token_symbols_blake2b"):
        raise CounterfactualTraceError("request token trace checksum mismatch")
    return (
        tuple(struct.unpack(f"<{token_count}Q", packed))
        if token_count
        else ()
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    source = path.expanduser().resolve()
    stream: TextIO
    if source.suffix == ".gz":
        stream = gzip.open(source, "rt", encoding="utf-8")
    else:
        stream = source.open(encoding="utf-8")
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise CounterfactualTraceError(
                    f"{source}:{line_number}: record must be an object"
                )
            yield raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a queue/service counterfactual workload from runtime traces."
    )
    parser.add_argument("--runtime-events", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--request-token-trace", type=Path)
    parser.add_argument("--workflow-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-sensitivity",
        choices=(
            "schedule_invariant",
            "timing_sensitive",
            "semantic_race_sensitive",
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = CounterfactualTraceBuilder().build(
        args.runtime_events,
        args.runtime_audit,
        workflow_ids=args.workflow_id,
        trace_sensitivity=args.trace_sensitivity,
        request_token_trace_path=args.request_token_trace,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(result.workload.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
