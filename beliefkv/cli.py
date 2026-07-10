from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from beliefkv.core.types import (
    ContinuationBelief,
    DeviceState,
    KVObjectMeta,
    PlannerConfig,
    RuntimeSnapshot,
)
from beliefkv.policy.planner import BeliefKVPlanner
from beliefkv.runtime.sglang_adapter import BASE_SGLANG_VERSION, required_hooks


def _load_config(raw: dict) -> PlannerConfig:
    return PlannerConfig(**raw) if raw else PlannerConfig()


def _load_kv_object(raw: dict) -> KVObjectMeta:
    return KVObjectMeta(
        object_id=raw["object_id"],
        workflow_ids=frozenset(raw.get("workflow_ids", [])),
        agent_ids=frozenset(raw.get("agent_ids", [])),
        branch_ids=frozenset(raw.get("branch_ids", [])),
        token_count=raw["token_count"],
        size_bytes=raw["size_bytes"],
        device_state=DeviceState(raw["device_state"]),
        is_shared_prefix=raw.get("is_shared_prefix", False),
        is_branch_delta=raw.get("is_branch_delta", False),
        is_active_decode=raw.get("is_active_decode", False),
        last_access_ms=raw.get("last_access_ms", 0.0),
        recompute_cost_ms=raw.get("recompute_cost_ms"),
        d2h_cost_ms=raw.get("d2h_cost_ms"),
        h2d_cost_ms=raw.get("h2d_cost_ms"),
    )


def _load_snapshot(raw: dict) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        now_ms=raw["now_ms"],
        hbm_capacity_bytes=raw["hbm_capacity_bytes"],
        hbm_used_bytes=raw["hbm_used_bytes"],
        active_decode_workflows=frozenset(raw.get("active_decode_workflows", [])),
    )


def _load_belief(raw: dict) -> ContinuationBelief:
    return ContinuationBelief(**raw)


def cmd_plan(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.snapshot).read_text())
    planner = BeliefKVPlanner(_load_config(data.get("config", {})))
    decisions = planner.plan(
        kv_objects=[_load_kv_object(item) for item in data["kv_objects"]],
        continuations=[_load_belief(item) for item in data.get("continuations", [])],
        snapshot=_load_snapshot(data["snapshot"]),
    )
    print(
        json.dumps(
            [asdict(item) for item in decisions],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_hooks(_: argparse.Namespace) -> None:
    payload = {
        "base_sglang_version": BASE_SGLANG_VERSION,
        "required_hooks": [asdict(hook) for hook in required_hooks()],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(prog="beliefkv")
    sub = parser.add_subparsers(required=True)

    plan = sub.add_parser("plan", help="plan KV actions for a JSON snapshot")
    plan.add_argument("snapshot")
    plan.set_defaults(func=cmd_plan)

    hooks = sub.add_parser("hooks", help="print runtime hook points")
    hooks.set_defaults(func=cmd_hooks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
