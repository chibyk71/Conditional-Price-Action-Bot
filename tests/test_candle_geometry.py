"""Unit tests for pure candle-geometry primitives (Phase 1 Milestone 1)."""
from __future__ import annotations
import pytest
from cpab.candle_geometry import (
    CandleAnatomy, analyze_candle, body_ratio, candle_direction, candle_overlap,
    candle_range, is_doji, lower_wick, lower_wick_ratio, range_contains,
    real_body, upper_wick, upper_wick_ratio,
)
from cpab.models import Candle

def _c(o, h, l, c, epoch=1_700_000_000, is_closed=True):
    return Candle(symbol="R_100", timeframe="1m", epoch=epoch, open=o, high=h, low=l, close=c, is_closed=is_closed)

def test_bullish_anatomy():
    c = _c(10.0, 13.0, 9.0, 12.0)
    assert candle_range(c) == 4.0 and real_body(c) == 2.0
    assert upper_wick(c) == 1.0 and lower_wick(c) == 1.0
    assert body_ratio(c) == pytest.approx(0.5)
    assert candle_direction(c) == "bullish"

def test_bearish_anatomy():
    c = _c(12.0, 13.0, 9.0, 10.0)
    assert real_body(c) == 2.0 and candle_direction(c) == "bearish"

def test_neutral_anatomy():
    c = _c(10.0, 12.0, 8.0, 10.0)
    assert real_body(c) == 0.0 and candle_direction(c) == "neutral"

def test_analyze_candle_snapshot_immutable():
    snap = analyze_candle(_c(10.0, 13.0, 9.0, 12.0))
    assert isinstance(snap, CandleAnatomy) and snap.range == 4.0 and snap.is_doji is False
    with pytest.raises(Exception):
        snap.range = 99.0  # type: ignore[misc]

def test_zero_range_candle():
    c = _c(5.0, 5.0, 5.0, 5.0)
    assert candle_range(c) == 0.0 and body_ratio(c) is None and is_doji(c) is None
    snap = analyze_candle(c)
    assert snap.body_ratio is None and snap.is_doji is None

def test_doji_at_threshold():
    c = _c(10.0, 15.0, 5.0, 11.0)
    assert body_ratio(c) == pytest.approx(0.1)
    assert is_doji(c, 0.10) is True and is_doji(c, 0.09) is False
    assert is_doji(c, 0.0) is False and is_doji(c, 1.0) is True

def test_doji_bullish_still_bullish():
    c = _c(10.0, 15.0, 5.0, 10.5)
    assert candle_direction(c) == "bullish" and is_doji(c, 0.10) is True

def test_doji_invalid_threshold():
    c = _c(10.0, 12.0, 8.0, 11.0)
    with pytest.raises(ValueError):
        is_doji(c, -0.01)
    with pytest.raises(ValueError):
        is_doji(c, 1.01)

def test_positive_width_overlap():
    ov = candle_overlap(_c(10,12,8,11), _c(9,11,7,10))
    assert ov is not None and ov.low == 8.0 and ov.high == 11.0 and ov.width == 3.0

def test_point_overlap():
    ov = candle_overlap(_c(10,12,10,11), _c(8,10,7,9))
    assert ov is not None and ov.width == 0.0

def test_no_overlap():
    assert candle_overlap(_c(10,12,10,11), _c(7,9,6,8)) is None

def test_range_contains_true():
    assert range_contains(_c(10,14,6,11), _c(10,15,5,12)) is True
    assert range_contains(_c(10,15,5,12), _c(10,14,6,11)) is False

def test_range_contains_equal_boundaries():
    assert range_contains(_c(10,12,8,11), _c(9,12,8,10)) is True

def test_range_contains_non():
    assert range_contains(_c(10,13,8,11), _c(9,12,9,10)) is False
