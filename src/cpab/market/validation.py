"""Candle normalization and validation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from cpab.models import Candle, CandleValidationIssue, CandleValidationResult


def _finite_positive(n: Any) -> bool:
    return isinstance(n, (int, float)) and n == n and abs(n) != float("inf") and n > 0


def normalize_provider_candle(
    raw: dict[str, Any],
    symbol: str,
    timeframe: str,
    *,
    is_closed: bool = True,
    source: str = "provider",
) -> Optional[Candle]:
    """Convert provider candle dict to internal Candle. Returns None if unusable."""
    try:
        epoch = int(raw.get("epoch") or raw.get("time") or 0)
        o = float(raw["open"])
        h = float(raw["high"])
        l = float(raw["low"])
        c = float(raw["close"])
    except (KeyError, TypeError, ValueError):
        return None
    if epoch <= 0:
        return None
    if not all(_finite_positive(x) for x in (o, h, l, c)):
        return None
    if h < l or h < o or h < c or l > o or l > c:
        return None
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


def validate_candle_sequence(
    candles: Iterable[Candle],
    *,
    expected_step: Optional[int] = None,
    allow_gaps: bool = False,
) -> CandleValidationResult:
    """
    Validate ordering, duplicates, OHLC consistency.
    Does not silently repair; reports issues.
    """
    items = list(candles)
    issues: list[CandleValidationIssue] = []
    if not items:
        return CandleValidationResult(
            ok=False,
            candles=[],
            issues=[CandleValidationIssue(code="EMPTY", message="Empty candle sequence")],
        )

    seen: set[int] = set()
    prev_epoch: Optional[int] = None
    valid: list[Candle] = []

    for i, c in enumerate(items):
        if not isinstance(c, Candle):
            issues.append(
                CandleValidationIssue(code="INVALID_TYPE", message=f"Index {i} not a Candle", index=i)
            )
            continue

        if not all(_finite_positive(x) for x in (c.open, c.high, c.low, c.close)):
            issues.append(
                CandleValidationIssue(
                    code="INVALID_OHLC",
                    message=f"Non-positive OHLC at epoch {c.epoch}",
                    index=i,
                    epoch=c.epoch,
                )
            )
            continue

        if c.high < c.low or c.high < c.open or c.high < c.close or c.low > c.open or c.low > c.close:
            issues.append(
                CandleValidationIssue(
                    code="OHLC_INCONSISTENT",
                    message=f"OHLC inconsistency at epoch {c.epoch}",
                    index=i,
                    epoch=c.epoch,
                )
            )
            continue

        if c.epoch in seen:
            issues.append(
                CandleValidationIssue(
                    code="DUPLICATE",
                    message=f"Duplicate epoch {c.epoch}",
                    index=i,
                    epoch=c.epoch,
                )
            )
            continue
        seen.add(c.epoch)

        if prev_epoch is not None:
            if c.epoch < prev_epoch:
                issues.append(
                    CandleValidationIssue(
                        code="OUT_OF_ORDER",
                        message=f"Epoch {c.epoch} before previous {prev_epoch}",
                        index=i,
                        epoch=c.epoch,
                    )
                )
                continue
            if expected_step and not allow_gaps and (c.epoch - prev_epoch) != expected_step:
                issues.append(
                    CandleValidationIssue(
                        code="GAP",
                        message=f"Gap between {prev_epoch} and {c.epoch} (expected step {expected_step})",
                        index=i,
                        epoch=c.epoch,
                    )
                )
                # still accept the candle; gap is recorded

        prev_epoch = c.epoch
        valid.append(c)

    ok = len(valid) > 0 and not any(
        i.code in ("EMPTY", "INVALID_TYPE") for i in issues
    )
    # Sequence is usable if we have valid candles; gaps/duplicates are issues but not hard fail
    # unless nothing remains.
    return CandleValidationResult(ok=len(valid) > 0, candles=valid, issues=issues)


def mark_closed_by_lag(
    candles: list[Candle],
    granularity: int,
    server_time: int,
    confirm_lag_bars: int = 1,
) -> list[Candle]:
    """
    Return new list with is_closed set by epoch + granularity + lag <= server_time.
    Forming candles remain is_closed=False.
    """
    lag = confirm_lag_bars * granularity
    out: list[Candle] = []
    for c in candles:
        closed = (c.epoch + granularity + lag) <= server_time
        if c.is_closed == closed:
            out.append(c)
        else:
            out.append(
                Candle(
                    symbol=c.symbol,
                    timeframe=c.timeframe,
                    epoch=c.epoch,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    is_closed=closed,
                    source=c.source,
                )
            )
    return out
