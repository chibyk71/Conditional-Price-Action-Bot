"""
Pure candle-geometry primitives for CPAB Phase 1.

Operates on the existing Candle model. No network, logging, detectors,
observations, registry, or market-state side effects.

Anatomy definitions (valid candle):
  range      = high - low
  real_body  = abs(close - open)
  upper_wick = high - max(open, close)
  lower_wick = min(open, close) - low

Direction is independent of doji classification:
  bullish  when close > open
  bearish  when close < open
  neutral  when close == open

Zero-range candles (high == low) yield range/body/wicks = 0 and
ratio / doji results of None (no division by zero).

Malformed OHLC is the responsibility of the upstream market-data validator;
geometry remains safe for valid zero-range candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

from cpab.models import Candle


class OverlapInterval(NamedTuple):
    """Shared price interval between two candles. Width may be zero (point overlap)."""

    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class CandleAnatomy:
    """Immutable complete anatomy snapshot of a single candle."""

    range: float
    real_body: float
    upper_wick: float
    lower_wick: float
    body_ratio: Optional[float]
    upper_wick_ratio: Optional[float]
    lower_wick_ratio: Optional[float]
    direction: str  # "bullish" | "bearish" | "neutral"
    is_doji: Optional[bool]


def candle_range(candle: Candle) -> float:
    return candle.high - candle.low


def real_body(candle: Candle) -> float:
    return abs(candle.close - candle.open)


def upper_wick(candle: Candle) -> float:
    return candle.high - max(candle.open, candle.close)


def lower_wick(candle: Candle) -> float:
    return min(candle.open, candle.close) - candle.low


def body_ratio(candle: Candle) -> Optional[float]:
    r = candle_range(candle)
    if r == 0.0:
        return None
    return real_body(candle) / r


def upper_wick_ratio(candle: Candle) -> Optional[float]:
    r = candle_range(candle)
    if r == 0.0:
        return None
    return upper_wick(candle) / r


def lower_wick_ratio(candle: Candle) -> Optional[float]:
    r = candle_range(candle)
    if r == 0.0:
        return None
    return lower_wick(candle) / r


def candle_direction(candle: Candle) -> str:
    """Direction from open/close only; independent of doji classification."""
    if candle.close > candle.open:
        return "bullish"
    if candle.close < candle.open:
        return "bearish"
    return "neutral"


def is_doji(candle: Candle, threshold: float = 0.10) -> Optional[bool]:
    """
    True when body-to-range ratio <= threshold.
    threshold must be in [0.0, 1.0]; otherwise ValueError.
    Returns None for zero-range candles (ratio unavailable).
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"doji threshold must be in [0.0, 1.0], got {threshold!r}")
    ratio = body_ratio(candle)
    if ratio is None:
        return None
    return ratio <= threshold


def analyze_candle(candle: Candle, doji_threshold: float = 0.10) -> CandleAnatomy:
    """
    Full anatomy snapshot. Prefer this when multiple measurements are needed.
    doji_threshold is validated by is_doji.
    """
    r = candle_range(candle)
    body = real_body(candle)
    uw = upper_wick(candle)
    lw = lower_wick(candle)
    if r == 0.0:
        br = uwr = lwr = None
        doji: Optional[bool] = None
    else:
        br = body / r
        uwr = uw / r
        lwr = lw / r
        if not (0.0 <= doji_threshold <= 1.0):
            raise ValueError(
                f"doji threshold must be in [0.0, 1.0], got {doji_threshold!r}"
            )
        doji = br <= doji_threshold
    return CandleAnatomy(
        range=r,
        real_body=body,
        upper_wick=uw,
        lower_wick=lw,
        body_ratio=br,
        upper_wick_ratio=uwr,
        lower_wick_ratio=lwr,
        direction=candle_direction(candle),
        is_doji=doji,
    )


def candle_overlap(a: Candle, b: Candle) -> Optional[OverlapInterval]:
    """
    Shared price interval of A and B.

    lower = max(A.low, B.low)
    upper = min(A.high, B.high)

    Returns None when lower > upper (no overlap).
    Returns an OverlapInterval when lower <= upper, including zero-width
    (point) overlap when lower == upper. Detectors that need positive width
    must check interval.width > 0 themselves.
    """
    lower = max(a.low, b.low)
    upper = min(a.high, b.high)
    if lower > upper:
        return None
    return OverlapInterval(low=lower, high=upper)


def range_contains(a: Candle, b: Candle) -> bool:
    """
    Full-range containment: True when A's entire range is contained in B's.

    A is contained within B when:
      A.high <= B.high and A.low >= B.low

    Boundary equality is included. Does not test body containment.
    Argument order: range_contains(inner, outer) → True if inner ⊆ outer.
    """
    return a.high <= b.high and a.low >= b.low
