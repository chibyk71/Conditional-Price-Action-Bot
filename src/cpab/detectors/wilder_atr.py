"""
Pure-Python Wilder Average True Range (ATR) helper for CPAB Phase 5D.

No external dependency. Deterministic. Used exclusively for pole-strength
qualification of flags and pennants.

Contract (locked for Phase 5D):
  - Method: Wilder's ATR
  - Lookback: exactly N completed candles immediately preceding the pole
  - True range: max(H-L, |H-prev_close|, |L-prev_close|)
  - Init: arithmetic mean of the first N true-range values
  - Smoothing: prev_ATR * (N-1)/N + TR * 1/N
  - The N candles plus the close that precedes them (to seed the first TR)
    must belong to one continuous segment.
  - Pole candles never contribute.
  - Gap / insufficient history / invalid OHLC → None (pole cannot qualify).
"""

from __future__ import annotations

from typing import Optional, Sequence

from cpab.models import Candle

_EPS = 1e-12


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def _valid_ohlc(c: Candle) -> bool:
    if not all(_finite(v) for v in (c.open, c.high, c.low, c.close)):
        return False
    if c.high < c.low:
        return False
    if c.high < max(c.open, c.close) or c.low > min(c.open, c.close):
        return False
    return True


def true_range(high: float, low: float, prev_close: float) -> float:
    """Single true-range value. Assumes finite inputs."""
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )


def wilder_atr(
    candles: Sequence[Candle],
    *,
    lookback: int = 14,
    step: Optional[int] = None,
) -> Optional[float]:
    """
    Compute Wilder ATR over a continuous window of exactly `lookback` candles.

    Requirements
    ------------
    - `candles` must contain at least lookback + 1 bars (the extra bar supplies
      the previous close for the first true range).
    - The final `lookback` candles are the ATR window; the candle immediately
      before them supplies prev_close for the first TR.
    - All bars in the supplied sequence must be consecutive when `step` is given
      (epoch differences exactly equal to step).
    - Every candle must have valid OHLC.
    - Returns None on any contract violation (insufficient history, gap,
      invalid OHLC). Never silently substitutes or skips.

    Returns
    -------
    The ATR value after the final smoothing step, or None.
    """
    if lookback < 1:
        return None
    # Need lookback candles for TRs + one preceding close.
    need = lookback + 1
    if len(candles) < need:
        return None

    window = list(candles[-need:])
    if step is not None:
        for i in range(1, len(window)):
            if window[i].epoch - window[i - 1].epoch != step:
                return None

    for c in window:
        if not _valid_ohlc(c):
            return None

    # First TR uses close of the candle that precedes the ATR window.
    prev_close = window[0].close
    trs: list[float] = []
    for c in window[1:]:
        tr = true_range(c.high, c.low, prev_close)
        if not _finite(tr) or tr < 0:
            return None
        trs.append(tr)
        prev_close = c.close

    if len(trs) != lookback:
        return None

    # Wilder initialization: simple average of the first lookback TRs.
    atr = sum(trs) / lookback
    if not _finite(atr):
        return None

    # Subsequent smoothing (already applied if we only have exactly lookback;
    # the loop is a no-op when len(trs) == lookback, which is the intended case).
    # For a pure lookback window the value after init *is* the ATR.
    # The recurrence is shown for completeness / future multi-step use:
    # for tr in trs[lookback:]:
    #     atr = atr * (lookback - 1) / lookback + tr / lookback
    return float(atr)


def atr_before_pole(
    segment: Sequence[Candle],
    pole_start_idx: int,
    *,
    lookback: int = 14,
    step: Optional[int] = None,
) -> Optional[float]:
    """
    ATR baseline for a pole that begins at segment[pole_start_idx].

    Uses the lookback candles immediately preceding the pole plus the close
    that precedes those lookback candles. Pole candles are never included.
    """
    if pole_start_idx < 0 or pole_start_idx >= len(segment):
        return None
    # Need lookback + 1 candles before the pole start.
    need_before = lookback + 1
    if pole_start_idx < need_before:
        return None
    prefix = segment[pole_start_idx - need_before : pole_start_idx]
    return wilder_atr(prefix, lookback=lookback, step=step)
