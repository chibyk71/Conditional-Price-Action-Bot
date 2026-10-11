"""Higher-timeframe construction from consecutive validated 1m candles."""

from __future__ import annotations

from typing import Optional

from cpab.models import Candle
from cpab.logging_setup import get_logger

logger = get_logger("cpab.market.aggregation")


def aggregate_from_1m(
    one_min: list[Candle],
    target_granularity: int,
    target_timeframe: str,
    symbol: str,
    *,
    require_complete: bool = True,
) -> list[Candle]:
    """
    Build non-overlapping HTF candles from 1m bars aligned to UTC multiples of target_granularity.

    Incomplete buckets are rejected (not invented) when require_complete is True.
    Only uses closed 1m candles.
    """
    if target_granularity % 60 != 0:
        raise ValueError("target_granularity must be a multiple of 60")
    bars_needed = target_granularity // 60
    closed = [c for c in one_min if c.is_closed and c.symbol == symbol]
    if not closed:
        return []

    # Bucket by open epoch of the HTF interval
    buckets: dict[int, list[Candle]] = {}
    for c in closed:
        bucket_start = c.epoch - (c.epoch % target_granularity)
        buckets.setdefault(bucket_start, []).append(c)

    result: list[Candle] = []
    for start in sorted(buckets.keys()):
        group = sorted(buckets[start], key=lambda x: x.epoch)
        if require_complete:
            expected_epochs = [start + i * 60 for i in range(bars_needed)]
            present = {c.epoch for c in group}
            if not all(e in present for e in expected_epochs):
                logger.debug(
                    "Incomplete HTF bucket %s %s start=%s present=%s needed=%s",
                    symbol,
                    target_timeframe,
                    start,
                    sorted(present),
                    expected_epochs,
                )
                continue
            # Use only the expected bars in order
            by_ep = {c.epoch: c for c in group}
            ordered = [by_ep[e] for e in expected_epochs]
        else:
            ordered = group
            if not ordered:
                continue

        o = ordered[0].open
        h = max(c.high for c in ordered)
        l = min(c.low for c in ordered)
        cl = ordered[-1].close
        result.append(
            Candle(
                symbol=symbol,
                timeframe=target_timeframe,
                epoch=start,
                open=o,
                high=h,
                low=l,
                close=cl,
                is_closed=True,
                source="aggregated",
            )
        )
    return result


def align_epoch(epoch: int, granularity: int) -> int:
    return epoch - (epoch % granularity)
