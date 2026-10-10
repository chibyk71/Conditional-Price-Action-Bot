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

    Upsert status values:
      inserted | unchanged | revised | revised_confirmed

    revised_confirmed means an already-closed candle's OHLC changed — callers
    must degrade readiness and must not treat that bar as stable for signals.
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
        self._confirmed_revisions: int = 0
        # Epochs of closed candles that were revised and not yet re-confirmed.
        self._quarantine: set[int] = set()

    def __len__(self) -> int:
        return len(self._by_epoch)

    def clear(self) -> None:
        self._by_epoch.clear()
        self._quarantine.clear()

    def upsert(self, candle: Candle) -> str:
        """
        Insert or update by epoch.
        Returns: 'inserted' | 'unchanged' | 'revised' | 'revised_confirmed'
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
            # Identical OHLC on a closed bar can clear quarantine after a re-poll.
            if candle.is_closed and candle.epoch in self._quarantine:
                self._quarantine.discard(candle.epoch)
            return "unchanged"

        self._revisions += 1
        was_confirmed = existing.is_closed
        ohlc_changed = existing.ohlc_tuple() != candle.ohlc_tuple()

        if was_confirmed and ohlc_changed:
            # Do not silently replace a confirmed bar for signal use.
            self._confirmed_revisions += 1
            self._quarantine.add(candle.epoch)
            self._by_epoch[candle.epoch] = candle
            logger.warning(
                "Confirmed candle revised %s %s epoch=%s — quarantined",
                self.symbol,
                self.timeframe,
                candle.epoch,
            )
            return "revised_confirmed"

        self._by_epoch[candle.epoch] = candle
        if candle.is_closed:
            self._quarantine.discard(candle.epoch)
        logger.debug(
            "Candle revised %s %s epoch=%s closed=%s",
            self.symbol,
            self.timeframe,
            candle.epoch,
            candle.is_closed,
        )
        return "revised"

    def upsert_many(self, candles: Iterable[Candle]) -> dict[str, int]:
        counts = {"inserted": 0, "unchanged": 0, "revised": 0, "revised_confirmed": 0}
        for c in candles:
            status = self.upsert(c)
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _trim(self) -> None:
        while len(self._by_epoch) > self.max_size:
            epoch, _ = self._by_epoch.popitem(last=False)
            self._quarantine.discard(epoch)

    def get(self, epoch: int) -> Optional[Candle]:
        return self._by_epoch.get(epoch)

    def completed(self) -> list[Candle]:
        """Completed candles excluding quarantined (confirmed-then-revised) epochs."""
        return [
            c
            for c in self._by_epoch.values()
            if c.is_closed and c.epoch not in self._quarantine
        ]

    def forming(self) -> list[Candle]:
        return [c for c in self._by_epoch.values() if not c.is_closed]

    def all_sorted(self) -> list[Candle]:
        return list(self._by_epoch.values())

    def latest_completed(self) -> Optional[Candle]:
        for c in reversed(self._by_epoch.values()):
            if c.is_closed and c.epoch not in self._quarantine:
                return c
        return None

    def latest_epoch(self) -> Optional[int]:
        if not self._by_epoch:
            return None
        return next(reversed(self._by_epoch))

    def has_min_completed(self, n: int) -> bool:
        return sum(1 for c in self._by_epoch.values() if c.is_closed and c.epoch not in self._quarantine) >= n

    def has_contiguous_completed(self, n: int) -> bool:
        """True if the latest n completed (non-quarantined) bars form a gap-free sequence."""
        completed = self.completed()
        if len(completed) < n:
            return False
        window = completed[-n:]
        step = self.granularity
        for i in range(1, len(window)):
            if window[i].epoch - window[i - 1].epoch != step:
                return False
        return True

    def is_series_healthy(self, min_candles: int) -> bool:
        """Enough contiguous completed history and no open quarantine."""
        if self._quarantine:
            return False
        return self.has_contiguous_completed(min_candles)

    @property
    def revision_count(self) -> int:
        return self._revisions

    @property
    def confirmed_revision_count(self) -> int:
        return self._confirmed_revisions

    @property
    def quarantine_epochs(self) -> set[int]:
        return set(self._quarantine)
