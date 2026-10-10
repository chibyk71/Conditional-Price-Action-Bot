"""In-memory rolling candle caches per (symbol, timeframe)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional

from cpab.models import Candle
from cpab.logging_setup import get_logger

logger = get_logger("cpab.market.cache")


class CandleCache:
    """
    Ordered by epoch ascending. Supports upsert by epoch (reconciliation),
    forming vs completed separation, and history window limits.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        granularity: int,
        max_size: int = 500,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.granularity = granularity
        self.max_size = max_size
        self._by_epoch: OrderedDict[int, Candle] = OrderedDict()
        self._revisions: int = 0

    def __len__(self) -> int:
        return len(self._by_epoch)

    def clear(self) -> None:
        self._by_epoch.clear()

    def upsert(self, candle: Candle) -> str:
        """
        Insert or update by epoch.
        Returns: 'inserted' | 'unchanged' | 'revised'
        """
        if candle.symbol != self.symbol or candle.timeframe != self.timeframe:
            raise ValueError(
                f"Cache mismatch: expected {self.symbol}/{self.timeframe}, "
                f"got {candle.symbol}/{candle.timeframe}"
            )
        existing = self._by_epoch.get(candle.epoch)
        if existing is None:
            self._by_epoch[candle.epoch] = candle
            self._trim()
            return "inserted"
        if existing.ohlc_tuple() == candle.ohlc_tuple() and existing.is_closed == candle.is_closed:
            return "unchanged"
        # revision
        self._revisions += 1
        self._by_epoch[candle.epoch] = candle
        logger.debug(
            "Candle revised %s %s epoch=%s closed=%s",
            self.symbol,
            self.timeframe,
            candle.epoch,
            candle.is_closed,
        )
        return "revised"

    def upsert_many(self, candles: Iterable[Candle]) -> dict[str, int]:
        counts = {"inserted": 0, "unchanged": 0, "revised": 0}
        for c in candles:
            counts[self.upsert(c)] += 1
        return counts

    def _trim(self) -> None:
        while len(self._by_epoch) > self.max_size:
            self._by_epoch.popitem(last=False)

    def get(self, epoch: int) -> Optional[Candle]:
        return self._by_epoch.get(epoch)

    def completed(self) -> list[Candle]:
        return [c for c in self._by_epoch.values() if c.is_closed]

    def forming(self) -> list[Candle]:
        return [c for c in self._by_epoch.values() if not c.is_closed]

    def all_sorted(self) -> list[Candle]:
        return list(self._by_epoch.values())

    def latest_completed(self) -> Optional[Candle]:
        for c in reversed(self._by_epoch.values()):
            if c.is_closed:
                return c
        return None

    def latest_epoch(self) -> Optional[int]:
        if not self._by_epoch:
            return None
        return next(reversed(self._by_epoch))

    def has_min_completed(self, n: int) -> bool:
        return sum(1 for c in self._by_epoch.values() if c.is_closed) >= n

    @property
    def revision_count(self) -> int:
        return self._revisions
