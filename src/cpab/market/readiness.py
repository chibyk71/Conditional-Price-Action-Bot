"""Global readiness state for market-data series."""

from __future__ import annotations

from typing import Optional

from cpab.models import ReadinessState
from cpab.logging_setup import get_logger

logger = get_logger("cpab.market.readiness")


class ReadinessManager:
    """
    Simple global readiness:
    INITIALIZING -> READY | FAILED
    READY -> DEGRADED -> RECOVERING -> READY | FAILED
    """

    def __init__(self) -> None:
        self._state = ReadinessState.INITIALIZING
        self._reason = "startup"
        self._series_ok: dict[tuple[str, str], bool] = {}

    @property
    def state(self) -> ReadinessState:
        return self._state

    @property
    def reason(self) -> str:
        return self._reason

    def register_series(self, symbol: str, timeframe: str) -> None:
        key = (symbol, timeframe)
        if key not in self._series_ok:
            self._series_ok[key] = False

    def set_series(self, symbol: str, timeframe: str, ok: bool, detail: str = "") -> None:
        key = (symbol, timeframe)
        prev = self._series_ok.get(key)
        self._series_ok[key] = ok
        if prev is not None and prev != ok:
            logger.info(
                "Series %s/%s readiness=%s %s",
                symbol,
                timeframe,
                ok,
                detail,
            )
        self._recompute(detail or f"series {symbol}/{timeframe} ok={ok}")

    def _recompute(self, trigger: str) -> None:
        if not self._series_ok:
            self._transition(ReadinessState.INITIALIZING, trigger)
            return
        all_ok = all(self._series_ok.values())
        any_ok = any(self._series_ok.values())
        if self._state == ReadinessState.INITIALIZING:
            if all_ok:
                self._transition(ReadinessState.READY, trigger)
            # stay INITIALIZING until all required series ready
            return
        if self._state == ReadinessState.READY:
            if not all_ok:
                self._transition(ReadinessState.DEGRADED, trigger)
            return
        if self._state == ReadinessState.DEGRADED:
            if all_ok:
                self._transition(ReadinessState.RECOVERING, trigger)
            elif not any_ok:
                self._transition(ReadinessState.FAILED, trigger)
            return
        if self._state == ReadinessState.RECOVERING:
            if all_ok:
                self._transition(ReadinessState.READY, trigger)
            elif not any_ok:
                self._transition(ReadinessState.FAILED, trigger)
            return
        if self._state == ReadinessState.FAILED:
            if all_ok:
                self._transition(ReadinessState.RECOVERING, trigger)
            return

    def _transition(self, new_state: ReadinessState, reason: str) -> None:
        if new_state == self._state and reason == self._reason:
            return
        old = self._state
        self._state = new_state
        self._reason = reason
        logger.info("Readiness %s -> %s reason=%s", old.value, new_state.value, reason)

    def mark_failed(self, reason: str) -> None:
        self._transition(ReadinessState.FAILED, reason)

    def can_evaluate(self) -> bool:
        return self._state == ReadinessState.READY

    def snapshot(self) -> dict:
        return {
            "state": self._state.value,
            "reason": self._reason,
            "series": {f"{s}/{t}": ok for (s, t), ok in self._series_ok.items()},
        }
