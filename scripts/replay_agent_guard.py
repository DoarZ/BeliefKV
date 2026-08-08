#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.agent_protocol import (
    AgentLoopGuardMiddleware,
    LoopGuardPolicy,
    WorkflowCompletion,
    analyze_agent_history,
)


def _message(raw: dict[str, Any]) -> BaseMessage:
    common = {
        "content": raw.get("content", ""),
        "additional_kwargs": raw.get("additional_kwargs") or {},
        "response_metadata": raw.get("response_metadata") or {},
        "id": raw.get("id"),
        "name": raw.get("name"),
    }
    kind = raw.get("message_type") or raw.get("type")
    if kind == "human":
        return HumanMessage(**common)
    if kind == "ai":
        return AIMessage(
            **common,
            tool_calls=raw.get("tool_calls") or [],
            invalid_tool_calls=raw.get("invalid_tool_calls") or [],
            usage_metadata=raw.get("usage_metadata"),
        )
    if kind == "tool":
        return ToolMessage(
            **common,
            tool_call_id=str(raw.get("tool_call_id") or ""),
            status=raw.get("status", "success"),
            artifact=raw.get("artifact"),
        )
    raise ValueError(f"unsupported trajectory message type: {kind}")


def load_trajectory(path: Path) -> list[BaseMessage]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("trajectory must be a JSON array")
    return [_message(dict(item)) for item in raw]


def replay_requests_recovery(path: Path) -> dict[str, Any]:
    messages = load_trajectory(path)
    duplicate_index = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage)
        and (message.additional_kwargs or {}).get("beliefkv_error_class")
        == "duplicate_suppressed"
    )
    successful_recovery_index = next(
        index
        for index in range(duplicate_index + 1, len(messages))
        if isinstance(messages[index], ToolMessage)
        and messages[index].name == "execute"
        and messages[index].status == "success"
    )
    policy = LoopGuardPolicy(enforce_graph_step_budget=False)
    trigger_prefix = messages[: duplicate_index + 1]
    trigger_snapshot = analyze_agent_history(trigger_prefix, policy)
    baseline = trigger_snapshot.progress_keys
    guard = AgentLoopGuardMiddleware(
        policy=policy,
        completion_schema=WorkflowCompletion,
        completion_instruction="Return WorkflowCompletion.",
        audit=None,
        scope="deterministic-replay",
    )
    recovered = guard.before_model(
        {
            "messages": messages[: successful_recovery_index + 1],
            "guard_phase": "RECOVERY",
            "guard_forcing_completion": True,
            "guard_ever_intervened": True,
            "guard_reason": "repeated_failed_tool_call",
            "guard_recovery_attempt": 1,
            "guard_progress_keys": baseline,
            "guard_recovery_baseline_keys": baseline,
        },
        runtime=None,
    )
    passed = bool(
        trigger_snapshot.physical_failure_count == 1
        and trigger_snapshot.suppressed_repeat_intent_count == 1
        and trigger_snapshot.repeated_failed_calls == 1
        and trigger_snapshot.reason is None
        and recovered is not None
        and recovered.get("guard_phase") == "NORMAL"
        and recovered.get("guard_forcing_completion") is False
    )
    return {
        "trajectory": str(path),
        "duplicate_message_index": duplicate_index,
        "successful_recovery_message_index": successful_recovery_index,
        "trigger_prefix": trigger_snapshot.to_dict(),
        "recovery_update": recovered,
        "step_62_false_positive_removed": trigger_snapshot.reason is None,
        "recovery_returns_to_normal": bool(
            recovered and recovered.get("guard_phase") == "NORMAL"
        ),
        "passed": passed,
    }


def replay_history_summary(path: Path) -> dict[str, Any]:
    messages = load_trajectory(path)
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(enforce_graph_step_budget=False),
        completion_schema=WorkflowCompletion,
        completion_instruction="Return WorkflowCompletion.",
        audit=None,
        scope="default-policy-replay",
    )
    state: dict[str, Any] = {"messages": []}
    interventions: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        state["messages"] = messages[: index + 1]
        if not isinstance(message, ToolMessage):
            continue
        update = guard.before_model(state, runtime=None)
        if update is None:
            continue
        state.update(update)
        if update.get("guard_forcing_completion", False) or update.get(
            "guard_phase"
        ) in {"SUSPECT", "RECOVERY", "FINALIZE"}:
            interventions.append({"message_index": index, "update": update})
    snapshot = analyze_agent_history(
        messages,
        LoopGuardPolicy(enforce_graph_step_budget=False),
        completion_tool_names=frozenset({"WorkflowCompletion"}),
    )
    return {
        "trajectory": str(path),
        "message_count": len(messages),
        "snapshot": snapshot.to_dict(),
        "semantic_effect_guard": (
            "credible progress certificates, not distinct command strings or output hashes"
        ),
        "default_policy_interventions": interventions,
        "passed": not interventions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay BeliefKV loop-guard traces")
    parser.add_argument("trajectory", type=Path)
    parser.add_argument(
        "--scenario",
        choices=("requests-recovery", "history-summary"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trajectory = args.trajectory.expanduser().resolve()
    report = (
        replay_requests_recovery(trajectory)
        if args.scenario == "requests-recovery"
        else replay_history_summary(trajectory)
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
