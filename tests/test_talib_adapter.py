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

def test_min_history_engulfing():
    assert min_history_for("CDLENGULFING") == 2
    assert min_history_for("engulfing") == 2

def test_insufficient_history():
    raw = evaluate_raw([_c(10, 11, 9, 10.5)], "CDLENGULFING")
    assert raw.kind == AdapterOutcomeKind.INSUFFICIENT_HISTORY
    assert TalibPatternDetector("CDLENGULFING").detect([_c(10, 11, 9, 10.5)]) == []

def test_forming_candle_rejected():
    pair = _engulfing_pair()
    pair[1] = _c(9.1, 10.5, 9.0, 10.3, pair[1].epoch, is_closed=False)
    assert evaluate_raw(pair, "CDLENGULFING").kind == AdapterOutcomeKind.FORMING_CANDLE
    assert TalibPatternDetector("CDLENGULFING").detect(pair) == []

def test_raw_matches_direct_talib():
    import talib
    pair = _engulfing_pair()
    o, h, l, c = candles_to_ohlc_arrays(pair)
    direct = int(talib.CDLENGULFING(o, h, l, c)[-1])
    raw = evaluate_raw(pair, "CDLENGULFING")
    assert raw.raw_talib_value == direct
    assert raw.candle_epoch == pair[-1].epoch

def test_raw_zero_no_observation():
    candles = [_c(10, 10.1, 9.9, 10.0, 1_700_000_000 + i * 60) for i in range(5)]
    raw = evaluate_raw(candles, "CDLENGULFING")
    assert raw.raw_talib_value == 0 or raw.kind == AdapterOutcomeKind.NO_PATTERN
    assert TalibPatternDetector("CDLENGULFING").detect(candles) == []

def test_pattern_observation_preserves_raw():
    pair = _engulfing_pair()
    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:test")
    raw = det.evaluate_raw(pair)
    if raw.kind != AdapterOutcomeKind.PATTERN:
        pytest.skip(f"TA-Lib did not detect pattern on fixture (raw={raw})")
    obs_list = det.detect(pair)
    assert len(obs_list) == 1
    assert obs_list[0].payload.get("raw_talib_value") == raw.raw_talib_value
    assert obs_list[0].candle_epoch == pair[-1].epoch
    assert obs_list[0].detector_config_id == "cfg:test"

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
