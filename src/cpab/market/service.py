"""Market data service: init history, poll, reconcile, aggregate HTF, readiness."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from cpab.adapters.base import MarketDataAdapter
from cpab.market.aggregation import aggregate_from_1m
from cpab.market.cache import CandleCache
from cpab.market.readiness import ReadinessManager
from cpab.market.validation import (
    mark_closed_by_lag,
    normalize_provider_candle,
    validate_candle_sequence,
)
from cpab.models import Candle
from cpab.logging_setup import get_logger

logger = get_logger("cpab.market.service")


class MarketDataService:
    def __init__(
        self,
        adapter: MarketDataAdapter,
        symbols: list[str],
        timeframes: list[dict[str, Any]],
        *,
        min_candles: int = 300,
        poll_interval_sec: float = 8.0,
        poll_lookback_bars: int = 5,
        confirm_lag_bars: int = 1,
        readiness: Optional[ReadinessManager] = None,
    ) -> None:
        self.adapter = adapter
        self.symbols = symbols
        self.timeframes = timeframes  # [{name, granularity}, ...]
        self.min_candles = min_candles
        self.poll_interval_sec = poll_interval_sec
        self.poll_lookback_bars = poll_lookback_bars
        self.confirm_lag_bars = confirm_lag_bars
        self.readiness = readiness or ReadinessManager()

        self._caches: dict[tuple[str, str], CandleCache] = {}
        self._tf_by_name = {t["name"]: t for t in timeframes}
        self._gran_by_name = {t["name"]: int(t["granularity"]) for t in timeframes}
        self._1m_name = self._find_1m_name()
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None

        for sym in symbols:
            for tf in timeframes:
                name = tf["name"]
                gran = int(tf["granularity"])
                self._caches[(sym, name)] = CandleCache(sym, name, gran, max_size=min_candles + 100)
                self.readiness.register_series(sym, name)

    def _find_1m_name(self) -> Optional[str]:
        for t in self.timeframes:
            if int(t["granularity"]) == 60:
                return t["name"]
        return None

    def get_cache(self, symbol: str, timeframe: str) -> CandleCache:
        return self._caches[(symbol, timeframe)]

    def completed_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        return self.get_cache(symbol, timeframe).completed()

    async def initialize(self) -> None:
        """Fetch independent history for every required symbol/timeframe."""
        logger.info(
            "Initializing market data for symbols=%s timeframes=%s min=%s",
            self.symbols,
            [t["name"] for t in self.timeframes],
            self.min_candles,
        )
        server_time = int(time.time())
        failures = 0
        for sym in self.symbols:
            for tf in self.timeframes:
                name = tf["name"]
                gran = int(tf["granularity"])
                try:
                    raw = await self.adapter.fetch_candles(sym, gran, self.min_candles, end="latest")
                    candles = []
                    for r in raw:
                        c = normalize_provider_candle(r, sym, name, is_closed=True, source="provider")
                        if c:
                            candles.append(c)
                    candles.sort(key=lambda x: x.epoch)
                    vr = validate_candle_sequence(candles, expected_step=gran, allow_gaps=True)
                    if len(vr.candles) < self.min_candles:
                        logger.error(
                            "Insufficient history %s/%s got=%s need=%s issues=%s",
                            sym,
                            name,
                            len(vr.candles),
                            self.min_candles,
                            [i.code for i in vr.issues],
                        )
                        self.readiness.set_series(sym, name, False, "insufficient history")
                        failures += 1
                        continue
                    # Mark closed/forming by lag
                    marked = mark_closed_by_lag(
                        vr.candles, gran, server_time, self.confirm_lag_bars
                    )
                    cache = self.get_cache(sym, name)
                    cache.clear()
                    cache.upsert_many(marked)
                    completed_n = len(cache.completed())
                    ok = completed_n >= self.min_candles - self.confirm_lag_bars - 2
                    self.readiness.set_series(
                        sym,
                        name,
                        ok,
                        f"history loaded completed={completed_n}",
                    )
                    logger.info(
                        "History ready %s/%s total=%s completed=%s",
                        sym,
                        name,
                        len(cache),
                        completed_n,
                    )
                except Exception as e:
                    logger.exception("History init failed %s/%s: %s", sym, name, e)
                    self.readiness.set_series(sym, name, False, str(e))
                    failures += 1

        if failures == len(self.symbols) * len(self.timeframes):
            self.readiness.mark_failed("all series failed to initialize")
        logger.info("Initialize complete readiness=%s", self.readiness.snapshot())

    async def poll_once(self) -> None:
        """Poll recent 1m candles with overlap; reconcile; rebuild HTF from 1m."""
        if not self._1m_name:
            logger.warning("No 1m timeframe configured; skip poll aggregation path")
            await self._poll_all_timeframes()
            return

        server_time = int(time.time())
        gran_1m = 60
        for sym in self.symbols:
            try:
                raw = await self.adapter.fetch_candles(
                    sym, gran_1m, self.poll_lookback_bars, end="latest"
                )
                candles = []
                for r in raw:
                    c = normalize_provider_candle(
                        r, sym, self._1m_name, is_closed=True, source="provider"
                    )
                    if c:
                        candles.append(c)
                candles.sort(key=lambda x: x.epoch)
                marked = mark_closed_by_lag(
                    candles, gran_1m, server_time, self.confirm_lag_bars
                )
                cache_1m = self.get_cache(sym, self._1m_name)
                counts = cache_1m.upsert_many(marked)
                if counts.get("revised"):
                    logger.info(
                        "Poll reconcile %s/1m inserted=%s revised=%s unchanged=%s",
                        sym,
                        counts["inserted"],
                        counts["revised"],
                        counts["unchanged"],
                    )
                # Validate completed length
                ok = cache_1m.has_min_completed(max(10, self.min_candles // 10))
                self.readiness.set_series(
                    sym, self._1m_name, ok, f"poll completed={len(cache_1m.completed())}"
                )

                # Rebuild HTF from local 1m completed
                one_min = cache_1m.completed()
                for tf in self.timeframes:
                    name = tf["name"]
                    gran = int(tf["granularity"])
                    if gran == 60:
                        continue
                    htf = aggregate_from_1m(
                        one_min, gran, name, sym, require_complete=True
                    )
                    # Also merge any recent provider HTF if we poll them separately
                    htf_cache = self.get_cache(sym, name)
                    # Upsert aggregated; keep provider history for depth
                    for c in htf:
                        htf_cache.upsert(c)
                    # Ensure HTF still has enough completed
                    htf_ok = htf_cache.has_min_completed(max(5, self.min_candles // 20))
                    self.readiness.set_series(
                        sym, name, htf_ok, f"htf completed={len(htf_cache.completed())}"
                    )
            except Exception as e:
                logger.error("Poll failed for %s: %s", sym, e)
                for tf in self.timeframes:
                    self.readiness.set_series(sym, tf["name"], False, f"poll error: {e}")
                try:
                    if hasattr(self.adapter, "reconnect"):
                        await self.adapter.reconnect()  # type: ignore
                except Exception as re:
                    logger.error("Reconnect failed: %s", re)

    async def _poll_all_timeframes(self) -> None:
        server_time = int(time.time())
        for sym in self.symbols:
            for tf in self.timeframes:
                name = tf["name"]
                gran = int(tf["granularity"])
                try:
                    raw = await self.adapter.fetch_candles(
                        sym, gran, self.poll_lookback_bars, end="latest"
                    )
                    candles = [
                        c
                        for r in raw
                        if (c := normalize_provider_candle(r, sym, name)) is not None
                    ]
                    candles.sort(key=lambda x: x.epoch)
                    marked = mark_closed_by_lag(
                        candles, gran, server_time, self.confirm_lag_bars
                    )
                    counts = self.get_cache(sym, name).upsert_many(marked)
                    logger.debug("Poll %s/%s %s", sym, name, counts)
                    ok = self.get_cache(sym, name).has_min_completed(5)
                    self.readiness.set_series(sym, name, ok, "poll")
                except Exception as e:
                    logger.error("Poll %s/%s failed: %s", sym, name, e)
                    self.readiness.set_series(sym, name, False, str(e))

    async def start_polling(self) -> None:
        self._running = True

        async def _loop() -> None:
            while self._running:
                try:
                    await self.poll_once()
                except Exception as e:
                    logger.exception("Poll loop error: %s", e)
                await asyncio.sleep(self.poll_interval_sec)

        self._poll_task = asyncio.create_task(_loop())
        logger.info("Polling started interval=%ss", self.poll_interval_sec)

    async def stop_polling(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("Polling stopped")
