from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


VALID_RUNTIMES = frozenset({"qwen_code", "codex"})
VALID_STRUCTURES = frozenset({"sequential", "parallelizable"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MatrixTask:
    task_id: str
    block: str
    structure: str
    prompt: str
    required_marker_groups: tuple[tuple[str, ...], ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MatrixTask":
        task_id = str(raw.get("id", "")).strip()
        block = str(raw.get("block", "")).strip()
        structure = str(raw.get("structure", "")).strip()
        prompt = str(raw.get("prompt", "")).strip()
        if not task_id or not block or not prompt:
            raise ValueError("matrix task requires non-empty id, block, and prompt")
        if structure not in VALID_STRUCTURES:
            raise ValueError(f"unsupported task structure: {structure}")
        groups = tuple(
            tuple(str(marker).strip() for marker in group if str(marker).strip())
            for group in raw.get("required_marker_groups", ())
        )
        if not groups or any(not group for group in groups):
            raise ValueError(f"task {task_id} requires non-empty marker groups")
        return cls(task_id, block, structure, prompt, groups)


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    model: str
    repository: Path
    runtimes: tuple[str, ...]
    policies: tuple[tuple[str, str], ...]
    common_suffix: str
    tasks: tuple[MatrixTask, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base_directory: Path) -> "MatrixSpec":
        if int(raw.get("schema_version", 0)) != 1:
            raise ValueError("runtime decision matrix requires schema_version=1")
        name = str(raw.get("name", "")).strip()
        model = str(raw.get("model", "")).strip()
        repository = Path(str(raw.get("repository", ""))).expanduser()
        if not repository.is_absolute():
            repository = (base_directory / repository).resolve()
        else:
            repository = repository.resolve()
        runtimes = tuple(str(value) for value in raw.get("runtimes", ()))
        if set(runtimes) != VALID_RUNTIMES or len(runtimes) != len(VALID_RUNTIMES):
            raise ValueError("matrix must contain qwen_code and codex exactly once")
        policy_raw = raw.get("policies")
        if not isinstance(policy_raw, dict) or set(policy_raw) != {
            "natural",
            "policy_guided",
        }:
            raise ValueError("matrix requires natural and policy_guided policies")
        policies = tuple((str(key), str(value).strip()) for key, value in policy_raw.items())
        common_suffix = str(raw.get("common_suffix", "")).strip()
        tasks = tuple(MatrixTask.from_dict(item) for item in raw.get("tasks", ()))
        task_ids = [task.task_id for task in tasks]
        if not name or not model or not common_suffix or not tasks:
            raise ValueError("matrix metadata, suffix, and tasks must be non-empty")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("matrix task ids must be unique")
        blocks: dict[str, set[str]] = defaultdict(set)
        for task in tasks:
            blocks[task.block].add(task.structure)
        if any(structures != VALID_STRUCTURES for structures in blocks.values()):
            raise ValueError("each task block requires one task per structure")
        if not repository.is_dir():
            raise ValueError(f"matrix repository does not exist: {repository}")
        return cls(
            name,
            model,
            repository,
            runtimes,
            policies,
            common_suffix,
            tasks,
        )

    @classmethod
    def load(cls, path: Path) -> "MatrixSpec":
        resolved = path.expanduser().resolve()
        return cls.from_dict(
            json.loads(resolved.read_text(encoding="utf-8")),
            base_directory=resolved.parent,
        )


@dataclass(frozen=True)
class MatrixCondition:
    order: int
    condition_id: str
    runtime: str
    policy: str
    task: MatrixTask
    prompt: str


def build_prompt(spec: MatrixSpec, task: MatrixTask, policy: str) -> str:
    policies = dict(spec.policies)
    if policy not in policies:
        raise ValueError(f"unknown prompt policy: {policy}")
    parts = []
    if policies[policy]:
        parts.append(f"Delegation policy:\n{policies[policy]}")
    parts.extend((f"Task:\n{task.prompt}", spec.common_suffix))
    return "\n\n".join(parts) + "\n"


def build_conditions(spec: MatrixSpec) -> tuple[MatrixCondition, ...]:
    conditions: list[MatrixCondition] = []
    order = 0
    for task_index, task in enumerate(spec.tasks):
        for policy_index, (policy, _text) in enumerate(spec.policies):
            runtimes = (
                spec.runtimes
                if (task_index + policy_index) % 2 == 0
                else tuple(reversed(spec.runtimes))
            )
            for runtime in runtimes:
                condition_id = f"{task.task_id}__{policy}__{runtime}"
                conditions.append(
                    MatrixCondition(
                        order=order,
                        condition_id=condition_id,
                        runtime=runtime,
                        policy=policy,
                        task=task,
                        prompt=build_prompt(spec, task, policy),
                    )
                )
                order += 1
    return tuple(conditions)


def decode_json_document(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("runtime output is empty")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                value, end = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if not stripped[index + end :].strip():
                return value
        raise


def decode_qwen_records(text: str) -> list[dict[str, Any]]:
    document: Any = None
    try:
        document = decode_json_document(text)
    except json.JSONDecodeError:
        pass
    if isinstance(document, list):
        if any(not isinstance(item, dict) for item in document):
            raise ValueError("Qwen Code JSON output must contain only objects")
        return document
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if isinstance(document, dict) and len(nonempty_lines) == 1:
        return [document]

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid Qwen stream-json record on line {line_number}: {error}"
            ) from error
        if not isinstance(item, dict):
            raise ValueError(
                f"Qwen stream-json record {line_number} is not an object"
            )
        records.append(item)
    if not records:
        raise ValueError("Qwen Code output has no JSON records")
    return records


def summarize_qwen_records(records: Any) -> dict[str, Any]:
    if not isinstance(records, list):
        raise ValueError("Qwen Code JSON output must be a list")
    results = [item for item in records if item.get("type") == "result"]
    if not results:
        raise ValueError("Qwen Code output has no result record")
    result = results[-1]
    stats = result.get("stats") or {}
    tools = stats.get("tools") or {}
    by_name = tools.get("byName") or {}
    agent = by_name.get("agent") or {}
    tool_names: dict[str, str] = {}
    tool_attempts: Counter[str] = Counter()
    tool_results: dict[str, bool] = {}
    for record in records:
        message = record.get("message") or {}
        for block in message.get("content") or ():
            if block.get("type") == "tool_use":
                tool_id = str(block.get("id") or "")
                tool_name = str(block.get("name") or "")
                if tool_id and tool_name and tool_id not in tool_names:
                    tool_names[tool_id] = tool_name
                    tool_attempts[tool_name] += 1
            elif block.get("type") == "tool_result":
                tool_id = str(block.get("tool_use_id") or "")
                if tool_id:
                    tool_results[tool_id] = not bool(block.get("is_error"))
    stream_tool_calls = sum(tool_attempts.values())
    stream_tool_failures = sum(not success for success in tool_results.values())
    stream_agent_attempts = tool_attempts["agent"]
    stream_agent_successes = sum(
        tool_names.get(tool_id) == "agent" and success
        for tool_id, success in tool_results.items()
    )
    models = stats.get("models") or {}
    api_requests = 0
    for model_stats in models.values():
        api_requests += int(((model_stats.get("api") or {}).get("totalRequests") or 0))
    usage = result.get("usage") or {}
    decisions = tools.get("totalDecisions") or {}
    permission_denials = result.get("permission_denials") or ()
    raw_error = result.get("error") or ""
    runtime_error = (
        str(raw_error.get("message") or raw_error)
        if isinstance(raw_error, dict)
        else str(raw_error)
    )
    return {
        "runtime_success": result.get("subtype") == "success",
        "runtime_status": str(result.get("subtype") or "unknown"),
        "runtime_error": runtime_error,
        "duration_ms": float(result.get("duration_ms") or 0.0),
        "turn_count": int(result.get("num_turns") or 0),
        "request_count": api_requests,
        "tool_call_count": int(tools.get("totalCalls") or stream_tool_calls),
        "tool_failure_count": int(
            tools.get("totalFail") or stream_tool_failures
        ),
        "permission_rejection_count": max(
            int(decisions.get("reject") or 0), len(permission_denials)
        ),
        "spawn_count": int(agent.get("success") or stream_agent_successes),
        "spawn_attempt_count": int(
            agent.get("count") or stream_agent_attempts
        ),
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "cached_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "final_text": str(result.get("result") or ""),
    }


def evaluate_markers(
    text: str, marker_groups: Iterable[Iterable[str]]
) -> dict[str, Any]:
    normalized = text.casefold()
    groups = [tuple(group) for group in marker_groups]
    hits = [
        any(str(marker).casefold() in normalized for marker in group)
        for group in groups
    ]
    return {
        "marker_group_hits": hits,
        "marker_groups_hit": sum(hits),
        "marker_groups_total": len(groups),
        "marker_coverage": sum(hits) / len(groups) if groups else 1.0,
        "content_gate_passed": all(hits),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row.get(key) or 0.0) for row in rows) if rows else 0.0


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        cells[(row["runtime"], row["policy"], row["structure"])].append(row)
    cell_rows = []
    for (runtime, policy, structure), samples in sorted(cells.items()):
        count = len(samples)
        cell_rows.append(
            {
                "runtime": runtime,
                "policy": policy,
                "structure": structure,
                "n": count,
                "spawn_run_count": sum(int(row["spawn_count"] > 0) for row in samples),
                "spawn_rate": sum(int(row["spawn_count"] > 0) for row in samples)
                / count,
                "mean_spawn_count": _mean(samples, "spawn_count"),
                "runtime_success_rate": sum(
                    int(bool(row["runtime_success"])) for row in samples
                )
                / count,
                "task_success_rate": sum(
                    int(bool(row["task_success"])) for row in samples
                )
                / count,
                "mean_marker_coverage": _mean(samples, "marker_coverage"),
                "mean_duration_ms": _mean(samples, "duration_ms"),
                "mean_turn_count": _mean(samples, "turn_count"),
                "mean_request_count": _mean(samples, "request_count"),
                "mean_prompt_tokens": _mean(samples, "prompt_tokens"),
                "mean_completion_tokens": _mean(samples, "completion_tokens"),
                "mean_max_resident_tokens": _mean(
                    samples, "max_resident_tokens"
                ),
                "mean_max_resident_pressure": _mean(
                    samples, "max_resident_pressure"
                ),
            }
        )
    by_cell = {
        (item["runtime"], item["policy"], item["structure"]): item
        for item in cell_rows
    }
    prompt_effects = []
    for runtime in sorted({row["runtime"] for row in materialized}):
        for structure in sorted({row["structure"] for row in materialized}):
            natural = by_cell.get((runtime, "natural", structure))
            guided = by_cell.get((runtime, "policy_guided", structure))
            if natural and guided:
                prompt_effects.append(
                    {
                        "runtime": runtime,
                        "structure": structure,
                        "spawn_rate_delta": guided["spawn_rate"]
                        - natural["spawn_rate"],
                        "task_success_rate_delta": guided["task_success_rate"]
                        - natural["task_success_rate"],
                        "duration_ms_delta": guided["mean_duration_ms"]
                        - natural["mean_duration_ms"],
                    }
                )
    structure_effects = []
    for runtime in sorted({row["runtime"] for row in materialized}):
        for policy in sorted({row["policy"] for row in materialized}):
            sequential = by_cell.get((runtime, policy, "sequential"))
            parallel = by_cell.get((runtime, policy, "parallelizable"))
            if sequential and parallel:
                structure_effects.append(
                    {
                        "runtime": runtime,
                        "policy": policy,
                        "spawn_rate_delta": parallel["spawn_rate"]
                        - sequential["spawn_rate"],
                        "task_success_rate_delta": parallel["task_success_rate"]
                        - sequential["task_success_rate"],
                    }
                )
    return {
        "schema_version": 1,
        "run_count": len(materialized),
        "cell_count": len(cell_rows),
        "replicates_per_cell": sorted({item["n"] for item in cell_rows}),
        "statistical_inference": "descriptive_only_two_task_blocks",
        "cells": cell_rows,
        "prompt_policy_effects": prompt_effects,
        "task_structure_effects": structure_effects,
    }
