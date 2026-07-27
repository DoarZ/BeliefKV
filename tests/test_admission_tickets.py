from __future__ import annotations

from dataclasses import replace

from beliefkv.policy.admission import (
    AdmissionCompileBudget,
    AdmissionRequest,
    AdmissionSideState,
    AdmissionTicketCompiler,
    ObservedAdmissionCandidate,
    ObservedAdmissionScheduler,
    ObservedAdmissionSnapshot,
    VisibleAdmissionIndex,
)


def _request(
    request_id: str,
    *,
    workflow_id: str = "wf",
    prompt_tokens: int = 2,
    output_tokens: int = 1,
) -> AdmissionRequest:
    return AdmissionRequest(
        request_id=request_id,
        workflow_id=workflow_id,
        invocation_id=f"inv-{request_id}",
        context_id=f"ctx-{request_id}",
        context_epoch=0,
        submitted_ts_ms=0,
        uncached_prompt_tokens=prompt_tokens,
        expected_output_tokens=output_tokens,
        kv_bytes_per_token=10,
        prompt_tokens=prompt_tokens,
    )


def _compile(
    index: VisibleAdmissionIndex,
    request_ids: tuple[str, ...],
    *,
    epoch: int = 1,
    hbm_bytes: int = 1_000,
    prefill_tokens: int = 1_000,
):
    return AdmissionTicketCompiler().compile(
        epoch=epoch,
        now_ms=10,
        ordered_request_ids=request_ids,
        entries={entry.request.request_id: entry for entry in index.entries()},
        budget=AdmissionCompileBudget(
            max_prefill_tokens=prefill_tokens,
            max_requests=32,
            max_candidates=64,
            available_hbm_bytes=hbm_bytes,
        ),
        source="observed_reactive",
        reason="bounded_fallback",
    )


def _candidate(
    request_id: str,
    *,
    workflow_id: str = "wf",
    causal_rank: int = 3,
    unblock_depth: int = 0,
    frontier_rank: int = 0,
    fair_rank: int = 0,
    wait_ms: float = 10.0,
    incremental_bytes: int = 100,
    starvation: bool = False,
    policy_eligible: bool = True,
    native_index: int = 0,
) -> ObservedAdmissionCandidate:
    return ObservedAdmissionCandidate(
        request_id=request_id,
        workflow_id=workflow_id,
        invocation_id=f"inv-{request_id}",
        native_index=native_index,
        causal_rank=causal_rank,
        unblock_depth=unblock_depth,
        frontier_rank=frontier_rank,
        workflow_fair_rank=fair_rank,
        wait_ms=wait_ms,
        estimated_incremental_bytes=incremental_bytes,
        starvation=starvation,
        policy_eligible=policy_eligible,
    )


def _snapshot(
    *,
    running: int = 2,
    radix_locked: int = 200,
    running_private: int = 100,
    native_hbm: int = 500,
    native_requests: int = 8,
) -> ObservedAdmissionSnapshot:
    return ObservedAdmissionSnapshot(
        hbm_capacity_bytes=1_000,
        reserve_hbm_bytes=100,
        native_available_hbm_bytes=native_hbm,
        native_max_requests=native_requests,
        running_request_count=running,
        radix_locked_bytes=radix_locked,
        running_private_bytes=running_private,
    )


def test_observed_scheduler_orders_causal_progress_before_soft_fairness() -> None:
    scheduler = ObservedAdmissionScheduler(
        active_kv_high_watermark_ratio=0.8,
        minimum_active_requests=1,
    )
    decision = scheduler.decide(
        (
            _candidate(
                "a-normal",
                workflow_id="a",
                causal_rank=3,
                frontier_rank=1,
                fair_rank=1,
                native_index=0,
            ),
            _candidate(
                "b-normal",
                workflow_id="b",
                causal_rank=3,
                frontier_rank=0,
                fair_rank=0,
                native_index=1,
            ),
            _candidate(
                "a-straggler",
                workflow_id="a",
                causal_rank=0,
                unblock_depth=2,
                fair_rank=1,
                native_index=2,
            ),
        ),
        _snapshot(),
    )

    assert decision.ordered_request_ids == (
        "a-straggler",
        "b-normal",
        "a-normal",
    )
    assert decision.mode == "active_kv_bounded"
    assert decision.active_kv_budget_bytes == 720
    assert decision.active_growth_budget_bytes == 420


def test_observed_scheduler_holds_new_growth_above_active_kv_watermark() -> None:
    scheduler = ObservedAdmissionScheduler(
        active_kv_high_watermark_ratio=0.8,
        minimum_active_requests=1,
    )
    candidates = tuple(
        _candidate(f"r{index}", native_index=index) for index in range(4)
    )
    decision = scheduler.decide(
        candidates,
        _snapshot(running=4, radix_locked=800, running_private=100),
    )

    assert decision.mode == "active_kv_pressure_hold"
    assert decision.max_new_requests == 4
    assert decision.active_growth_budget_bytes == 0
    assert decision.active_kv_headroom_bytes == 0


def test_observed_scheduler_floor_is_bounded_and_work_conserving() -> None:
    scheduler = ObservedAdmissionScheduler(
        active_kv_high_watermark_ratio=0.8,
        minimum_active_requests=2,
    )
    candidates = tuple(
        _candidate(f"r{index}", native_index=index) for index in range(6)
    )
    decision = scheduler.decide(
        candidates,
        _snapshot(
            running=0,
            radix_locked=800,
            running_private=100,
            native_hbm=350,
        ),
    )

    assert decision.mode == "work_conserving_floor"
    assert decision.max_new_requests == 2
    assert decision.active_growth_budget_bytes == 350


def test_observed_scheduler_does_not_cap_one_workflow_to_one_ticket() -> None:
    scheduler = ObservedAdmissionScheduler(
        active_kv_high_watermark_ratio=1.0,
        minimum_active_requests=0,
    )
    candidates = tuple(
        _candidate(
            request_id,
            workflow_id="fanout",
            frontier_rank=index,
            native_index=index,
        )
        for index, request_id in enumerate(("a", "b", "c"))
    )
    decision = scheduler.decide(candidates, _snapshot())

    assert decision.ordered_request_ids == ("a", "b", "c")
    assert decision.max_new_requests == 3


def test_observed_scheduler_places_explicit_blockers_after_visible_work() -> None:
    scheduler = ObservedAdmissionScheduler(
        active_kv_high_watermark_ratio=1.0,
        minimum_active_requests=0,
    )
    decision = scheduler.decide(
        (
            _candidate(
                "restore",
                starvation=True,
                policy_eligible=False,
                native_index=0,
            ),
            _candidate("ready", native_index=1),
        ),
        _snapshot(),
    )

    assert decision.ordered_request_ids == ("ready", "restore")


def test_one_workflow_can_receive_multiple_tickets_without_reservation() -> None:
    index = VisibleAdmissionIndex()
    for request_id in ("a", "b", "c"):
        index.register(_request(request_id))

    result = _compile(index, ("a", "b", "c"))

    assert [ticket.request_id for ticket in result.tickets] == ["a", "b", "c"]
    assert index.reserved_bytes == 0


def test_wait_restore_skips_only_the_dependent_request() -> None:
    index = VisibleAdmissionIndex()
    index.register(_request("restore"), bundle_generations={"bundle": "g1"})
    index.register(_request("ready"))
    index.set_wait_restore("restore", ("bundle",), reason="h2d_inflight")

    result = _compile(index, ("restore", "ready"))

    assert [ticket.request_id for ticket in result.tickets] == ["ready"]
    assert ("restore", AdmissionSideState.WAIT_RESTORE.value) in result.skipped
    assert index.reserved_bytes == 0


def test_expired_epoch_and_local_prefix_change_invalidate_ticket() -> None:
    index = VisibleAdmissionIndex()
    index.register(_request("a"), bundle_generations={"bundle": "g1"})
    ticket = _compile(index, ("a",), epoch=3).tickets[0]

    expired = index.validate_ticket(ticket, epoch=4)
    assert not expired.valid
    assert expired.reasons == ("epoch_expired",)

    index.observe_prefix(
        "a",
        uncached_prompt_tokens=1,
        bundle_generations={"bundle": "g2"},
    )
    stale = index.validate_ticket(ticket, epoch=3)
    assert not stale.valid
    assert "prefix_generation" in stale.reasons
    assert "bundle_generation:bundle" in stale.reasons


def test_bundle_change_invalidates_only_the_dependent_ticket() -> None:
    index = VisibleAdmissionIndex()
    index.register(_request("a"), bundle_generations={"bundle-a": "g1"})
    index.register(_request("b"), bundle_generations={"bundle-b": "g1"})
    result = _compile(index, ("a", "b"), epoch=7)
    by_request = result.by_request_id

    a_validation = index.validate_ticket(
        by_request["a"],
        epoch=7,
        bundle_generations={"bundle-a": "g2"},
    )
    b_validation = index.validate_ticket(
        by_request["b"],
        epoch=7,
        bundle_generations={"bundle-b": "g1"},
    )

    assert not a_validation.valid
    assert a_validation.reasons == ("bundle_generation:bundle-a",)
    assert b_validation.valid


def test_prompt_change_does_not_invalidate_an_unrelated_request() -> None:
    index = VisibleAdmissionIndex()
    index.register(_request("a"))
    index.register(_request("b"))
    result = _compile(index, ("a", "b"), epoch=8)

    index.update_request(
        replace(_request("a"), uncached_prompt_tokens=3, prompt_tokens=3),
        prompt_changed=True,
    )

    assert not index.validate_ticket(result.by_request_id["a"], epoch=8).valid
    assert index.validate_ticket(result.by_request_id["b"], epoch=8).valid


def test_compiler_issues_a_chunk_ticket_for_an_oversized_prompt() -> None:
    index = VisibleAdmissionIndex()
    index.register(_request("large", prompt_tokens=20, output_tokens=0))
    index.register(_request("small", prompt_tokens=2, output_tokens=0))

    result = _compile(
        index,
        ("large", "small"),
        hbm_bytes=1_000,
        prefill_tokens=10,
    )

    assert [ticket.request_id for ticket in result.tickets] == ["large"]
    assert result.tickets[0].estimated_prefill_tokens == 10
    assert result.tickets[0].estimated_incremental_bytes == 200
    assert ("small", "prefill_token_budget") in result.skipped


def test_compiler_continues_after_a_request_exceeds_hbm_budget() -> None:
    index = VisibleAdmissionIndex()
    index.register(_request("large", prompt_tokens=20, output_tokens=0))
    index.register(_request("small", prompt_tokens=2, output_tokens=0))

    result = _compile(
        index,
        ("large", "small"),
        hbm_bytes=50,
        prefill_tokens=10,
    )

    assert [ticket.request_id for ticket in result.tickets] == ["small"]
    assert ("large", "bounded_hbm_budget") in result.skipped


def test_transition_barrier_fails_closed_until_reopened() -> None:
    index = VisibleAdmissionIndex()
    index.register(_request("a"), transition_generation=1)
    index.set_policy_blocked("a", reason="transition_open")

    blocked = _compile(index, ("a",), epoch=1)
    assert not blocked.tickets
    assert blocked.skipped == (("a", "policy_blocked"),)

    index.set_transition_generation("a", 2)
    index.set_visible("a")
    visible = _compile(index, ("a",), epoch=2)
    assert [ticket.request_id for ticket in visible.tickets] == ["a"]
