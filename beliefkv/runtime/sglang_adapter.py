from __future__ import annotations

from dataclasses import dataclass


BASE_SGLANG_VERSION = "0.5.2rc1"


@dataclass(frozen=True)
class RuntimeHook:
    name: str
    purpose: str
    expected_file: str


def required_hooks() -> list[RuntimeHook]:
    return [
        RuntimeHook(
            name="request_metadata",
            purpose="carry workflow, agent, branch, and policy hint fields",
            expected_file="python/sglang/srt/managers/io_struct.py",
        ),
        RuntimeHook(
            name="scheduler_snapshot",
            purpose="export waiting queue, active decode queue, and HBM pressure",
            expected_file="python/sglang/srt/managers/scheduler.py",
        ),
        RuntimeHook(
            name="kv_object_index",
            purpose="map prefix-cache nodes to workflow and branch metadata",
            expected_file="python/sglang/srt/mem_cache/radix_cache.py",
        ),
        RuntimeHook(
            name="hierarchical_residency_action",
            purpose="apply keep, offload, prefetch, and recompute decisions",
            expected_file="python/sglang/srt/mem_cache/hiradix_cache.py",
        ),
        RuntimeHook(
            name="runtime_flags",
            purpose="add BeliefKV enable and policy configuration arguments",
            expected_file="python/sglang/srt/server_args.py",
        ),
    ]
