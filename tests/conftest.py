"""Shared fixtures — deterministic simulated candles, no live API."""

from __future__ import annotations

import sys

import pytest

from cpab.models import Candle


def make_candle(
    symbol: str,
    timeframe: str,
    epoch: int,
    o: float,
    h: float,
    l: float,
    c: float,
    *,
    is_closed: bool = True,
    source: str = "provider",
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        epoch=epoch,
        open=o,
        high=h,
        low=l,
        close=c,
        is_closed=is_closed,
        source=source,
    )


def series_1m(
    symbol: str = "R_100",
    start_epoch: int = 1_700_000_000,
    n: int = 20,
    base: float = 100.0,
) -> list[Candle]:
    out = []
    price = base
    for i in range(n):
        o = price
        delta = 0.5 if i % 2 == 0 else -0.3
        c = o + delta
        h = max(o, c) + 0.2
        l = min(o, c) - 0.2
        out.append(
            make_candle(symbol, "1m", start_epoch + i * 60, o, h, l, c, is_closed=True)
        )
        price = c
    return out


@pytest.fixture
def sample_1m() -> list[Candle]:
    return series_1m()


def pytest_sessionstart(session):
    """A15: best-effort Phase 8B evidence bootstrap.

    Must not raise on missing Git or environment metadata so ordinary pytest
    works outside a checkout. Acceptance/meta tests still fail closed via
    validate_run_identity when mandatory evidence cannot be established.
    """
    from cpab.replay.evidence import begin_acceptance_run

    pytest_args = list(sys.argv)
    try:
        begin_acceptance_run(note="pytest_sessionstart", pytest_args=pytest_args)
    except Exception as exc:  # noqa: BLE001 — session must always start
        # Record a minimal unavailable run so meta-tests fail closed, not INTERNALERROR.
        try:
            begin_acceptance_run(note=f"pytest_sessionstart_error:{type(exc).__name__}", pytest_args=pytest_args or ["pytest"])
        except Exception:
            pass
