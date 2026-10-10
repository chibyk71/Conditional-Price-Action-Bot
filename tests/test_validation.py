"""1. Candle normalization and OHLC validation."""

from cpab.market.validation import (
    mark_closed_by_lag,
    normalize_provider_candle,
    validate_candle_sequence,
)
from cpab.models import Candle
from conftest import make_candle, series_1m


def test_normalize_valid():
    raw = {"epoch": 1000, "open": 1.0, "high": 1.5, "low": 0.9, "close": 1.2}
    c = normalize_provider_candle(raw, "R_100", "1m")
    assert c is not None
    assert c.epoch == 1000
    assert c.ohlc_tuple() == (1.0, 1.5, 0.9, 1.2)


def test_normalize_invalid_ohlc():
    raw = {"epoch": 1000, "open": 1.0, "high": 0.5, "low": 0.9, "close": 1.2}
    assert normalize_provider_candle(raw, "R_100", "1m") is None


def test_normalize_missing_fields():
    assert normalize_provider_candle({"epoch": 1}, "R_100", "1m") is None


def test_validate_ordered_ok():
    candles = series_1m(n=5)
    vr = validate_candle_sequence(candles, expected_step=60)
    assert vr.ok
    assert len(vr.candles) == 5
    assert not any(i.code == "DUPLICATE" for i in vr.issues)


def test_validate_duplicate():
    candles = series_1m(n=3)
    dup = make_candle("R_100", "1m", candles[1].epoch, 1, 2, 0.5, 1.5)
    mixed = [candles[0], candles[1], dup, candles[2]]
    vr = validate_candle_sequence(mixed, expected_step=60, allow_gaps=True)
    assert any(i.code == "DUPLICATE" for i in vr.issues)
    assert len(vr.candles) == 3


def test_validate_out_of_order():
    candles = series_1m(n=3)
    bad = [candles[0], candles[2], candles[1]]
    vr = validate_candle_sequence(bad, expected_step=60, allow_gaps=True)
    assert any(i.code == "OUT_OF_ORDER" for i in vr.issues)


def test_validate_gap():
    candles = series_1m(n=3)
    # remove middle conceptually by shifting last
    gapped = [candles[0], candles[1], make_candle(
        "R_100", "1m", candles[1].epoch + 180, 1, 2, 0.5, 1.5
    )]
    vr = validate_candle_sequence(gapped, expected_step=60, allow_gaps=False)
    assert any(i.code == "GAP" for i in vr.issues)


def test_mark_closed_by_lag():
    start = 1_700_000_000
    c1 = make_candle("R_100", "1m", start, 1, 2, 0.5, 1.5, is_closed=True)
    c2 = make_candle("R_100", "1m", start + 60, 1, 2, 0.5, 1.5, is_closed=True)
    # server just after second bar open + lag
    server = start + 60 + 60 + 60  # +1 lag bar after second closed
    marked = mark_closed_by_lag([c1, c2], 60, server, confirm_lag_bars=1)
    assert marked[0].is_closed is True
    assert marked[1].is_closed is True
    # if server is early, latest is forming
    early = start + 60 + 30
    marked2 = mark_closed_by_lag([c1, c2], 60, early, confirm_lag_bars=1)
    assert marked2[1].is_closed is False
