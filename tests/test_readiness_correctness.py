"""Regression tests for PR #1 review: gaps, poll depth, confirmed revisions."""

from __future__ import annotations

import pytest

from cpab.adapters.base import MarketDataAdapter
from cpab.market.cache import CandleCache
from cpab.market.readiness import ReadinessManager
from cpab.market.service import MarketDataService
from cpab.models import ReadinessState
from conftest import make_candle


# ---------------------------------------------------------------------------
# 1. Gaps must not yield READY
# ---------------------------------------------------------------------------

class GappedAdapter(MarketDataAdapter):
    """Returns min_candles bars but with a missing interval in the middle."""

    def __init__(self, n: int = 50, gap_at: int = 25, gran: int = 60):
        self.n = n
        self.gap_at = gap_at
        self.gran = gran
        self._connected = False

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False

    @property
    def connected(self):
        return self._connected

    async def fetch_candles(self, symbol, granularity, count, *, end="latest"):
        base = 1_700_000_000
        base = base - (base % granularity)
        out = []
        price = 100.0
        i = 0
        produced = 0
        # produce `count` candles but skip one slot so epochs have a gap
        while produced < count:
            if i == self.gap_at:
                i += 1  # skip one step → gap
                continue
            o = price
            c = price + 0.1
            out.append(
                {
                    "epoch": base + i * granularity,
                    "open": o,
                    "high": c + 0.05,
                    "low": o - 0.05,
                    "close": c,
                }
            )
            price = c
            produced += 1
            i += 1
        return out


@pytest.mark.asyncio
async def test_gapped_history_does_not_become_ready():
    adapter = GappedAdapter(n=50, gap_at=10, gran=60)
    rm = ReadinessManager()
    svc = MarketDataService(
        adapter,
        symbols=["R_100"],
        timeframes=[{"name": "1m", "granularity": 60}],
        min_candles=50,
        readiness=rm,
        confirm_lag_bars=0,
    )
    await adapter.connect()
    await svc.initialize()
    assert rm.state != ReadinessState.READY
    assert not rm.can_evaluate()
    assert rm.snapshot()["series"]["R_100/1m"] is False


# ---------------------------------------------------------------------------
# 2. Poll must not restore READY with shallow history
# ---------------------------------------------------------------------------

class ShallowThenPollAdapter(MarketDataAdapter):
    """Init returns full history; subsequent polls return only a few bars."""

    def __init__(self, full_n: int = 50, poll_n: int = 5, gran: int = 60):
        self.full_n = full_n
        self.poll_n = poll_n
        self.gran = gran
        self._connected = False
        self.calls = 0

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False

    @property
    def connected(self):
        return self._connected

    async def fetch_candles(self, symbol, granularity, count, *, end="latest"):
        self.calls += 1
        n = self.full_n if self.calls == 1 else min(count, self.poll_n)
        base = 1_700_000_000
        base = base - (base % granularity)
        # For poll calls, return the *latest* n bars so overlap works
        start_i = 0 if self.calls == 1 else max(0, self.full_n - n)
        out = []
        price = 100.0 + start_i * 0.1
        for i in range(start_i, start_i + n):
            o = price
            c = price + 0.1
            out.append(
                {
                    "epoch": base + i * granularity,
                    "open": o,
                    "high": c + 0.05,
                    "low": o - 0.05,
                    "close": c,
                }
            )
            price = c
        return out


@pytest.mark.asyncio
async def test_poll_requires_full_min_completed_not_shallow():
    """After init READY, clearing cache to shallow depth must not stay/become READY via poll."""
    adapter = ShallowThenPollAdapter(full_n=50, poll_n=5)
    rm = ReadinessManager()
    svc = MarketDataService(
        adapter,
        symbols=["R_100"],
        timeframes=[{"name": "1m", "granularity": 60}],
        min_candles=50,
        poll_lookback_bars=5,
        readiness=rm,
        confirm_lag_bars=0,
    )
    await adapter.connect()
    await svc.initialize()
    assert rm.state == ReadinessState.READY

    # Simulate data loss / restart of cache with only shallow bars
    cache = svc.get_cache("R_100", "1m")
    cache.clear()
    # Seed only 10 completed candles (well below min_candles=50)
    base = 1_700_000_000
    for i in range(10):
        cache.upsert(
            make_candle("R_100", "1m", base + i * 60, 100, 101, 99, 100.5, is_closed=True)
        )
    assert not cache.has_min_completed(50)

    await svc.poll_once()
    # Poll must NOT restore READY with only ~10–15 bars
    assert not rm.can_evaluate()
    assert rm.snapshot()["series"]["R_100/1m"] is False


# ---------------------------------------------------------------------------
# 3. Confirmed candle revision quarantines and degrades
# ---------------------------------------------------------------------------

def test_confirmed_revision_quarantines_and_excludes_from_completed():
    cache = CandleCache("R_100", "1m", 60)
    c1 = make_candle("R_100", "1m", 1000, 1.0, 2.0, 0.5, 1.5, is_closed=True)
    assert cache.upsert(c1) == "inserted"
    assert len(cache.completed()) == 1

    # Provider revises the closed bar
    c1_rev = make_candle("R_100", "1m", 1000, 1.0, 2.5, 0.5, 1.8, is_closed=True)
    status = cache.upsert(c1_rev)
    assert status == "revised_confirmed"
    assert 1000 in cache.quarantine_epochs
    # Quarantined bar must not appear in completed() for detectors
    assert cache.completed() == []
    assert cache.has_min_completed(1) is False
    assert cache.is_series_healthy(1) is False


@pytest.mark.asyncio
async def test_confirmed_revision_during_poll_degrades_readiness():
    class RevisingAdapter(MarketDataAdapter):
        def __init__(self):
            self._connected = False
            self.calls = 0
            self.base = 1_700_000_000 - (1_700_000_000 % 60)

        async def connect(self):
            self._connected = True

        async def close(self):
            self._connected = False

        @property
        def connected(self):
            return self._connected

        async def fetch_candles(self, symbol, granularity, count, *, end="latest"):
            self.calls += 1
            out = []
            price = 100.0
            n = 50 if self.calls == 1 else 5
            start = 0 if self.calls == 1 else 45
            for i in range(start, start + n):
                o = price
                # On second call, revise an interior closed bar (epoch at i=45)
                if self.calls > 1 and i == 45:
                    c = price + 0.5  # different close
                    h = c + 0.05
                else:
                    c = price + 0.1
                    h = c + 0.05
                out.append(
                    {
                        "epoch": self.base + i * granularity,
                        "open": o,
                        "high": h,
                        "low": o - 0.05,
                        "close": c,
                    }
                )
                price = c if not (self.calls > 1 and i == 45) else price + 0.1
            return out

    adapter = RevisingAdapter()
    rm = ReadinessManager()
    svc = MarketDataService(
        adapter,
        symbols=["R_100"],
        timeframes=[{"name": "1m", "granularity": 60}],
        min_candles=50,
        poll_lookback_bars=5,
        readiness=rm,
        confirm_lag_bars=0,
    )
    await adapter.connect()
    await svc.initialize()
    assert rm.state == ReadinessState.READY

    await svc.poll_once()
    # Confirmed revision should block evaluation
    assert not rm.can_evaluate()
    assert rm.snapshot()["series"]["R_100/1m"] is False
    cache = svc.get_cache("R_100", "1m")
    assert cache.confirmed_revision_count >= 1
    assert len(cache.quarantine_epochs) >= 1
