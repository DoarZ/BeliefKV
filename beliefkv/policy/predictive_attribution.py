from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TERMINAL_OUTCOMES = frozenset({"useful", "wasted", "too_late", "censored", "failed"})


@dataclass
class PredictiveActionOutcome:
    intent_id: str
    action: str
    context_id: str
    context_epoch: int
    command_id: str
    created_ts_ms: float
    state: str = "pending_transfer"
    transfer_completed_ts_ms: float | None = None
    terminal_ts_ms: float | None = None
    actual_bytes: int = 0
    reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_OUTCOMES


class PredictiveActionAttributionLedger:
    """Small runtime ledger for causal attribution of predictive KV actions."""

    def __init__(
        self,
        emit: Callable[[str, float, PredictiveActionOutcome], None] | None = None,
    ) -> None:
        self._by_intent: dict[str, PredictiveActionOutcome] = {}
        self._emit = emit

    def register(
        self,
        *,
        intent_id: str,
        action: str,
        context_id: str,
        context_epoch: int,
        command_id: str,
        now_ms: float,
    ) -> None:
        if intent_id in self._by_intent:
            raise ValueError(f"predictive intent already attributed: {intent_id}")
        outcome = PredictiveActionOutcome(
            intent_id=intent_id,
            action=action,
            context_id=context_id,
            context_epoch=context_epoch,
            command_id=command_id,
            created_ts_ms=now_ms,
        )
        self._by_intent[intent_id] = outcome
        self._notify("registered", now_ms, outcome)

    def transfer_terminal(
        self,
        intent_id: str,
        *,
        completed: bool,
        actual_bytes: int,
        now_ms: float,
        reason: str | None = None,
    ) -> None:
        outcome = self._by_intent.get(intent_id)
        if outcome is None or outcome.terminal:
            return
        outcome.actual_bytes = max(0, actual_bytes)
        outcome.transfer_completed_ts_ms = now_ms
        outcome.reason = reason
        if completed:
            outcome.state = "prepared"
            self._notify("prepared", now_ms, outcome)
        else:
            self._finish(outcome, "failed", now_ms, reason or "transfer_failed")

    def consume_context(
        self,
        context_id: str,
        *,
        context_epoch: int | None,
        now_ms: float,
        reason: str,
    ) -> tuple[PredictiveActionOutcome, ...]:
        consumed = []
        for outcome in self._by_intent.values():
            if (
                outcome.context_id != context_id
                or outcome.terminal
                or (
                    context_epoch is not None
                    and outcome.context_epoch != context_epoch
                )
            ):
                continue
            final = "useful" if outcome.state == "prepared" else "too_late"
            self._finish(outcome, final, now_ms, reason)
            consumed.append(outcome)
        return tuple(consumed)

    def terminal_context(
        self,
        context_id: str,
        *,
        context_epoch: int | None,
        now_ms: float,
        reason: str,
    ) -> tuple[PredictiveActionOutcome, ...]:
        wasted = []
        for outcome in self._by_intent.values():
            if (
                outcome.context_id == context_id
                and not outcome.terminal
                and (
                    context_epoch is None
                    or outcome.context_epoch == context_epoch
                )
            ):
                self._finish(outcome, "wasted", now_ms, reason)
                wasted.append(outcome)
        return tuple(wasted)

    def censor_all(self, *, now_ms: float, reason: str) -> None:
        for outcome in self._by_intent.values():
            if not outcome.terminal:
                self._finish(outcome, "censored", now_ms, reason)

    def outcomes(self) -> tuple[PredictiveActionOutcome, ...]:
        return tuple(self._by_intent.values())

    def _finish(
        self,
        outcome: PredictiveActionOutcome,
        state: str,
        now_ms: float,
        reason: str,
    ) -> None:
        outcome.state = state
        outcome.terminal_ts_ms = now_ms
        outcome.reason = reason
        self._notify(state, now_ms, outcome)

    def _notify(
        self, event: str, now_ms: float, outcome: PredictiveActionOutcome
    ) -> None:
        if self._emit is not None:
            self._emit(event, now_ms, outcome)
