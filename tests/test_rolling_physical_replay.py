from __future__ import annotations

import pytest

from beliefkv.simulator.rolling_physical import (
    FutureRadixUse,
    ResidencyReplayMode,
    RollingPhysicalReplayError,
    RollingRadixAllocator,
)


def _allocator(
    *,
    capacity_tokens: int,
    mode: ResidencyReplayMode = ResidencyReplayMode.REACTIVE_LRU,
    host_tokens: int = 100,
) -> RollingRadixAllocator:
    return RollingRadixAllocator(
        hbm_capacity_bytes=capacity_tokens * 10,
        hbm_fixed_bytes=0,
        host_capacity_bytes=host_tokens * 10,
        bytes_per_token=10,
        d2h_bytes_per_ms=10,
        h2d_bytes_per_ms=20,
        transfer_setup_ms=1,
        mode=mode,
    )


def test_rolling_allocator_replans_after_lock_and_evicts_only_peer_suffix() -> None:
    allocator = _allocator(capacity_tokens=5)
    allocator.radix.materialize_gpu((1, 2, 3, 4), context_id="context-a")
    allocator.radix.materialize_gpu((1, 2, 3, 5), context_id="context-b")
    admission = allocator.admit(
        request_id="request-a",
        prompt_token_symbols=(1, 2, 3, 4, 99),
        now_ms=0,
        future_uses=(FutureRadixUse("request-b", (1, 2, 3, 5, 98)),),
    )

    ready_ms = allocator.materialize(
        request_id="request-a",
        path=(1, 2, 3, 4, 6),
        now_ms=admission.ready_ms,
        future_uses=(FutureRadixUse("request-b", (1, 2, 3, 5, 98)),),
    )

    assert ready_ms == pytest.approx(2.0)
    assert allocator.radix.gpu_tokens == 5
    assert allocator.radix.cpu_tokens == 1
    assert [event.kind for event in allocator.events] == [
        "D2H_COMMIT",
        "MATERIALIZE",
    ]
    assert allocator.events[0].root_path == (1, 2, 3, 5)
    assert allocator.events[0].transfer_bytes == 10
    assert max(event.hbm_bytes_after for event in allocator.events) <= 50


def test_hindsight_next_use_avoids_lru_eviction_of_near_reuse() -> None:
    def run(mode: ResidencyReplayMode) -> RollingRadixAllocator:
        allocator = _allocator(capacity_tokens=4, mode=mode)
        allocator.radix.materialize_gpu((1, 10), context_id="near")
        allocator.radix.materialize_gpu((2, 20), context_id="dead")
        admission = allocator.admit(
            request_id="new",
            prompt_token_symbols=(3,),
            now_ms=0,
            future_uses=(FutureRadixUse("near-use", (1, 10, 99)),),
        )
        allocator.materialize(
            request_id="new",
            path=(3,),
            now_ms=admission.ready_ms,
            future_uses=(FutureRadixUse("near-use", (1, 10, 99)),),
        )
        return allocator

    reactive = run(ResidencyReplayMode.REACTIVE_LRU)
    oracle = run(ResidencyReplayMode.HINDSIGHT_NEXT_USE)

    assert reactive.events[0].root_path == (1,)
    assert reactive.events[0].kind == "D2H_COMMIT"
    assert oracle.events[0].root_path == (2,)
    assert oracle.events[0].kind == "DROP"
    assert oracle.d2h_bytes == 0


def test_rolling_allocator_fails_closed_when_active_path_owns_all_hbm() -> None:
    allocator = _allocator(capacity_tokens=3)
    allocator.radix.materialize_gpu((1, 2, 3), context_id="context")
    admission = allocator.admit(
        request_id="request",
        prompt_token_symbols=(1, 2, 3, 9),
        now_ms=0,
    )

    with pytest.raises(RollingPhysicalReplayError, match="no closure-safe bundle"):
        allocator.materialize(
            request_id="request",
            path=(1, 2, 3, 4),
            now_ms=admission.ready_ms,
        )


def test_shadow_capacity_rejection_has_no_side_effect() -> None:
    allocator = _allocator(capacity_tokens=4, host_tokens=1)
    allocator.radix.materialize_gpu((1, 2), context_id="context")

    with pytest.raises(RollingPhysicalReplayError, match="Host capacity"):
        allocator.prepare_host((1,), now_ms=0)

    assert allocator.radix.gpu_tokens == 2
    assert allocator.radix.cpu_tokens == 0
    assert allocator.events == []


def test_batch_materialization_counts_shared_prefix_once() -> None:
    allocator = _allocator(capacity_tokens=3)
    allocator.admit(request_id="left", prompt_token_symbols=(1,), now_ms=0)
    allocator.admit(request_id="right", prompt_token_symbols=(1,), now_ms=0)

    result = allocator.materialize_batch(
        (("left", (1, 2)), ("right", (1, 3))),
        now_ms=0,
    )

    assert result.ready_ms == 0
    assert sum(result.unique_growth_tokens.values()) == 3
    assert allocator.radix.gpu_tokens == 3
    assert allocator.hbm_occupied_bytes == 30
