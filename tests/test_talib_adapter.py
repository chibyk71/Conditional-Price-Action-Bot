"""TA-Lib adapter: raw preservation, insufficient history, forming rejection, failures."""
from __future__ import annotations
import pytest
from cpab.detectors.talib_adapter import (
    AdapterOutcomeKind, TalibPatternDetector, candles_to_ohlc_arrays,
    evaluate_raw, min_history_for, raw_to_observation,
)
from cpab.models import Candle

def _c(o, h, l, c, epoch=1_700_000_000, is_closed=True):
    return Candle(symbol="R_100", timeframe="1m", epoch=epoch, open=float(o), high=float(h), low=float(l), close=float(c), is_closed=is_closed)

def _engulfing_pair(start=1_700_000_000):
    return [_c(10.0, 10.2, 9.0, 9.2, start), _c(9.1, 10.5, 9.0, 10.3, start + 60)]

def _known_positive_engulfing(start=1_700_000_000):
    """Series verified to produce CDLENGULFING == +100 on the last bar under TA-Lib."""
    return [
        _c(10.0, 10.5, 9.8, 10.2, start),
        _c(10.2, 10.4, 9.9, 10.0, start + 60),
        _c(10.0, 10.2, 9.5, 9.6, start + 120),
        _c(9.5, 9.8, 9.0, 9.1, start + 180),
        _c(9.0, 10.8, 8.9, 10.6, start + 240),
    ]

def test_min_history_pattern_specific():
    assert min_history_for("CDLENGULFING") == 2
    assert min_history_for("engulfing") == 2
    assert min_history_for("CDLDOJI") == 1
    assert min_history_for("CDLHAMMER") == 1
    assert min_history_for("CDLMORNINGSTAR") == 3
    assert min_history_for("CDLEVENINGSTAR") == 3
    assert min_history_for("CDL_UNKNOWN_XYZ") == 3

def test_insufficient_history_respects_declared_minimum():
    raw = evaluate_raw([_c(10, 11, 9, 10.5)], "CDLENGULFING")
    assert raw.kind == AdapterOutcomeKind.INSUFFICIENT_HISTORY
    pair = _engulfing_pair()
    raw2 = evaluate_raw(pair, "CDLMORNINGSTAR")
    assert raw2.kind == AdapterOutcomeKind.INSUFFICIENT_HISTORY
    assert "3" in (raw2.message or "")

def test_forming_candle_rejected():
    pair = _engulfing_pair()
    pair[1] = _c(9.1, 10.5, 9.0, 10.3, pair[1].epoch, is_closed=False)
    assert evaluate_raw(pair, "CDLENGULFING").kind == AdapterOutcomeKind.FORMING_CANDLE
    assert TalibPatternDetector("CDLENGULFING").detect(pair) == []

def test_raw_matches_direct_talib():
    import talib
    series = _known_positive_engulfing()
    o, h, l, c = candles_to_ohlc_arrays(series)
    direct = int(talib.CDLENGULFING(o, h, l, c)[-1])
    raw = evaluate_raw(series, "CDLENGULFING")
    assert raw.raw_talib_value == direct
    assert raw.candle_epoch == series[-1].epoch

def test_raw_zero_no_observation():
    candles = [_c(10, 10.1, 9.9, 10.0, 1_700_000_000 + i * 60) for i in range(5)]
    raw = evaluate_raw(candles, "CDLENGULFING")
    assert raw.raw_talib_value == 0 or raw.kind == AdapterOutcomeKind.NO_PATTERN
    assert TalibPatternDetector("CDLENGULFING").detect(candles) == []

def test_positive_observation_preserves_raw_and_identity():
    """Must not skip: fixture is known to produce a non-zero CDLENGULFING result."""
    import talib
    series = _known_positive_engulfing()
    o, h, l, c = candles_to_ohlc_arrays(series)
    direct = int(talib.CDLENGULFING(o, h, l, c)[-1])
    assert direct != 0, "fixture must produce a non-zero TA-Lib result"

    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:positive-test")
    raw = det.evaluate_raw(series)
    assert raw.kind == AdapterOutcomeKind.PATTERN
    assert raw.raw_talib_value == direct
    assert raw.candle_epoch == series[-1].epoch

    obs_list = det.detect(series)
    assert len(obs_list) == 1
    obs = obs_list[0]
    assert obs.payload.get("raw_talib_value") == direct
    assert obs.candle_epoch == series[-1].epoch
    assert obs.detector_config_id == "cfg:positive-test"
    assert obs.direction == ("bullish" if direct > 0 else "bearish")
    assert obs.detector_id == det.name

def test_doji_raw_magnitude():
    import talib
    candles = [_c(10.0, 12.0, 8.0, 10.05, 1_700_000_000), _c(10.0, 12.0, 8.0, 10.02, 1_700_000_060)]
    o, h, l, c = candles_to_ohlc_arrays(candles)
    direct = int(talib.CDLDOJI(o, h, l, c)[-1])
    raw = evaluate_raw(candles, "CDLDOJI")
    assert raw.raw_talib_value == direct

def test_unknown_pattern_failure():
    candles = [_c(10, 11, 9, 10.5, 1_700_000_000 + i * 60) for i in range(5)]
    raw = evaluate_raw(candles, "CDL_NOT_A_REAL_PATTERN_XYZ")
    assert raw.kind == AdapterOutcomeKind.EXECUTION_FAILURE
    assert TalibPatternDetector("CDL_NOT_A_REAL_PATTERN_XYZ").detect(candles) == []

def test_raw_to_observation_none_for_zero():
    from cpab.detectors.talib_adapter import AdapterRawResult
    zero = AdapterRawResult(kind=AdapterOutcomeKind.NO_PATTERN, pattern="CDLENGULFING", raw_talib_value=0, candle_epoch=1)
    assert raw_to_observation(zero, _engulfing_pair()[-1], detector_id="x") is None
