"""Shared fixtures — deterministic simulated candles, no live API."""

from __future__ import annotations

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
    """Ascending 1m closed candles with mild oscillation."""
    out = []
    price = base
    for i in range(n):
        o = price
        # alternate slight up/down
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
    """A15: Phase 8B evidence must come from this run, not a checked-in manifest.

    Fail closed: if clearing fails, abort the session so a stale manifest
    cannot certify acceptance.
    """
    from cpab.replay.evidence import clear_evidence
    clear_evidence()
