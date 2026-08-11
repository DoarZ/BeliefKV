from beliefkv.policy.predictive_attribution import PredictiveActionAttributionLedger


def test_prepare_is_useful_only_after_transfer_completion() -> None:
    events = []
    ledger = PredictiveActionAttributionLedger(
        lambda event, _now, outcome: events.append((event, outcome.intent_id))
    )
    ledger.register(
        intent_id="i1",
        action="prepare_host",
        context_id="ctx",
        context_epoch=2,
        command_id="c1",
        now_ms=1,
    )
    ledger.transfer_terminal("i1", completed=True, actual_bytes=10, now_ms=2)
    consumed = ledger.consume_context(
        "ctx", context_epoch=2, now_ms=3, reason="pressure_commit"
    )
    assert consumed[0].state == "useful"
    assert events == [("registered", "i1"), ("prepared", "i1"), ("useful", "i1")]


def test_pending_prepare_is_too_late_and_terminal_prepare_is_wasted() -> None:
    ledger = PredictiveActionAttributionLedger()
    ledger.register(
        intent_id="late",
        action="prepare_host",
        context_id="ctx",
        context_epoch=0,
        command_id="c1",
        now_ms=1,
    )
    assert ledger.consume_context(
        "ctx", context_epoch=0, now_ms=2, reason="reactive_d2h"
    )[0].state == "too_late"
    ledger.register(
        intent_id="waste",
        action="prepare_host",
        context_id="other",
        context_epoch=0,
        command_id="c2",
        now_ms=3,
    )
    ledger.transfer_terminal("waste", completed=True, actual_bytes=20, now_ms=4)
    assert ledger.terminal_context(
        "other", context_epoch=0, now_ms=5, reason="context_terminal"
    )[0].state == "wasted"
