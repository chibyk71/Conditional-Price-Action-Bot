"""2, 3, 11. Historical init, multi symbol/tf, API errors via mock."""

from __future__ import annotations

import pytest

from cpab.adapters.base import MarketDataAdapter
from cpab.market.readiness import ReadinessManager
from cpab.market.service import MarketDataService
from cpab.models import ReadinessState


class MockAdapter(MarketDataAdapter):
    def __init__(self, responses=None, fail_symbols=None):
        self.responses = responses or {}
        self.fail_symbols = set(fail_symbols or [])
        self._connected = False
        self.fetch_calls = []

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False

    @property
    def connected(self):
        return self._connected

    async def fetch_candles(self, symbol, granularity, count, *, end="latest"):
        self.fetch_calls.append((symbol, granularity, count, end))
        if symbol in self.fail_symbols:
            raise RuntimeError("simulated API error")
        key = (symbol, granularity)
        if key in self.responses:
            return self.responses[key]
        # generate count candles
        base = 1_700_000_000
        base = base - (base % granularity)
        out = []
        price = 100.0
        for i in range(count):
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
async def test_initialize_sufficient_history():
    adapter = MockAdapter()
    rm = ReadinessManager()
    svc = MarketDataService(
        adapter,
        symbols=["R_100", "R_75"],
        timeframes=[{"name": "1m", "granularity": 60}, {"name": "5m", "granularity": 300}],
        min_candles=50,
        readiness=rm,
        confirm_lag_bars=0,
    )
    await adapter.connect()
    await svc.initialize()
    assert rm.state == ReadinessState.READY
    assert len(svc.get_cache("R_100", "1m")) >= 50
    assert len(svc.get_cache("R_75", "5m")) >= 50


@pytest.mark.asyncio
async def test_initialize_insufficient_history():
    adapter = MockAdapter()

    async def short_fetch(symbol, granularity, count, *, end="latest"):
        return [
            {"epoch": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        ]

    adapter.fetch_candles = short_fetch  # type: ignore
    rm = ReadinessManager()
    svc = MarketDataService(
        adapter,
        symbols=["R_100"],
        timeframes=[{"name": "1m", "granularity": 60}],
        min_candles=50,
        readiness=rm,
    )
    await adapter.connect()
    await svc.initialize()
    assert rm.state != ReadinessState.READY
    assert not rm.can_evaluate()


@pytest.mark.asyncio
async def test_api_error_marks_degraded():
    adapter = MockAdapter(fail_symbols={"R_100"})
    rm = ReadinessManager()
    svc = MarketDataService(
        adapter,
        symbols=["R_100"],
        timeframes=[{"name": "1m", "granularity": 60}],
        min_candles=10,
        readiness=rm,
    )
    await adapter.connect()
    await svc.initialize()
    assert rm.state == ReadinessState.FAILED or not rm.can_evaluate()


@pytest.mark.asyncio
async def test_poll_reconcile():
    adapter = MockAdapter()
    rm = ReadinessManager()
    svc = MarketDataService(
        adapter,
        symbols=["R_100"],
        timeframes=[{"name": "1m", "granularity": 60}, {"name": "5m", "granularity": 300}],
        min_candles=30,
        poll_lookback_bars=5,
        readiness=rm,
        confirm_lag_bars=0,
    )
    await adapter.connect()
    await svc.initialize()
    n_before = len(svc.get_cache("R_100", "1m"))
    await svc.poll_once()
    assert len(svc.get_cache("R_100", "1m")) >= n_before - 5  # overlap upsert
