from __future__ import annotations

import pytest

from beliefkv.simulator.queue_service import (
    FrozenCounterfactualWorkload,
    FrozenRequestDemand,
    FrozenWorkflowDemand,
)
from beliefkv.simulator.token_radix import (
    TieredTokenRadix,
    TieredTokenRadixError,
    TokenRadixReplay,
    TokenRadixReplayError,
)


def _request(
    request_id: str,
    prompt: tuple[int, ...],
    commit: tuple[int, ...],
    *,
    observed_hit: int,
    predecessors: tuple[str, ...] = (),
) -> FrozenRequestDemand:
    output_tokens = len(commit) - len(prompt) + 1
    return FrozenRequestDemand(
        request_id=request_id,
        workflow_id="workflow",
        invocation_id=f"invocation-{request_id}",
        context_id=f"context-{request_id}",
        context_epoch=0,
        predecessor_request_ids=predecessors,
        release_delay_ms=0,
        uncached_prompt_tokens=len(prompt) - observed_hit,
        output_tokens=output_tokens,
        startup_bytes=10_000,
        kv_growth_bytes=1_000,
        observed_cache_hit_tokens=observed_hit,
        prompt_token_symbols=prompt,
        cache_commit_token_symbols=commit,
        partial_cache_commit_token_symbols=(prompt,),
    )


def _workload() -> FrozenCounterfactualWorkload:
    prompt = (10, 20, 30, 40)
    commit = (*prompt, 50, 60)
    requests = (
        _request("first", prompt, commit, observed_hit=0),
        _request(
            "second",
            prompt,
            commit,
            observed_hit=3,
            predecessors=("first",),
        ),
    )
    return FrozenCounterfactualWorkload(
        trace_id="trace",
        transition_hash="transition",
        trace_sensitivity="schedule_invariant",
        requests=requests,
        workflows=(
            FrozenWorkflowDemand(
                workflow_id="workflow",
                release_ms=0,
                terminal_request_ids=("second",),
            ),
        ),
        prefix_identity_complete=True,
    )


def test_replay_recomputes_sglang_one_token_miss_and_unique_growth() -> None:
    result = TokenRadixReplay().replay_completion_order(
        _workload(),
        ("first", "second"),
        initial_state_known=True,
        model_partial_commits=True,
    )

    assert result.request_demands["first"].cache_hit_tokens == 0
    assert result.request_demands["second"].cache_hit_tokens == 3
    assert result.request_demands["second"].allocator_growth_tokens == 3
    assert result.request_demands["second"].unique_commit_growth_tokens == 0
    assert result.final_unique_cache_tokens == 6
    assert result.observed_hit_mismatch_count == 0
    assert result.valid_for_exact_prefix_demand


def test_replay_rejects_non_topological_completion_order() -> None:
    with pytest.raises(TokenRadixReplayError, match="dependencies"):
        TokenRadixReplay().replay_completion_order(
            _workload(),
            ("second", "first"),
            initial_state_known=True,
        )


def test_replay_does_not_claim_concurrent_exactness_without_partial_commits() -> None:
    result = TokenRadixReplay().replay_completion_order(
        _workload(),
        ("first", "second"),
        initial_state_known=True,
    )

    assert not result.valid_for_exact_prefix_demand


def _tiered_shared_radix() -> TieredTokenRadix:
    radix = TieredTokenRadix(bytes_per_token=10)
    radix.materialize_gpu((1, 2, 3, 4), context_id="context-a")
    radix.materialize_gpu((1, 2, 3, 5), context_id="context-b")
    return radix


def test_tiered_radix_accounts_shared_tokens_once_and_preserves_one_token_miss() -> None:
    radix = _tiered_shared_radix()

    match = radix.match((1, 2, 3, 4), touch=True)

    assert radix.unique_tokens == 5
    assert radix.gpu_bytes == 50
    assert match.logical_hit_tokens == 3
    assert match.gpu_hit_tokens == 3
    assert match.restore_tokens == 0


def test_tiered_radix_d2h_uses_descendant_closure_and_h2d_restores_ancestors() -> None:
    radix = _tiered_shared_radix()

    offload = radix.commit_cpu((1, 2, 3))
    match = radix.match((1, 2, 3, 4))
    restore = radix.restore_prefix((1, 2, 3, 4), match.logical_hit_tokens)

    assert offload.affected_tokens == 3
    assert offload.transfer_tokens == 3
    assert offload.gpu_delta_tokens == -3
    assert radix.gpu_tokens == 3
    assert radix.cpu_tokens == 3
    assert match.logical_hit_tokens == 3
    assert match.gpu_hit_tokens == 2
    assert restore.transfer_tokens == 1
    sibling = radix.match((1, 2, 3, 5, 6))
    assert sibling.logical_hit_tokens == 4
    assert sibling.gpu_hit_tokens == 3


def test_tiered_radix_lock_blocks_ancestor_bundle_but_not_peer_suffix() -> None:
    radix = _tiered_shared_radix()
    radix.lock_prefix("request-a", (1, 2, 3, 4), 3)

    bundles = radix.evictable_bundles()

    assert bundles
    assert any(bundle.root_path == (1, 2, 3, 5) for bundle in bundles)
    assert all(bundle.root_path != (1,) for bundle in bundles)
    with pytest.raises(TieredTokenRadixError, match="active locks"):
        radix.commit_cpu((1,))

    radix.unlock_request("request-a")
    assert any(bundle.root_path == (1,) for bundle in radix.evictable_bundles())


def test_tiered_radix_finds_bundles_on_contexts_deeper_than_python_recursion() -> None:
    radix = TieredTokenRadix(bytes_per_token=10)
    path = tuple(range(2_048))
    radix.materialize_gpu(path, context_id="deep-context")

    bundles = radix.evictable_bundles()

    assert any(bundle.root_path == (0,) for bundle in bundles)
    assert any(bundle.root_path == path for bundle in bundles)
    assert all(bundle.gpu_tokens > 0 for bundle in bundles)


def test_tiered_radix_drop_removes_both_tiers_without_stale_accounting() -> None:
    radix = _tiered_shared_radix()
    radix.prepare_host((1, 2, 3, 5))

    mutation = radix.drop_subtree((1, 2, 3, 5))

    assert mutation.gpu_delta_tokens == -1
    assert mutation.cpu_delta_tokens == -1
    assert radix.unique_tokens == 4
    assert radix.gpu_tokens == 4
    assert radix.cpu_tokens == 0
