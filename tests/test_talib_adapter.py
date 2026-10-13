"""TA-Lib adapter: lookback-based min history, raw preservation, failures."""
from __future__ import annotations
import pytest
from talib.abstract import Function
from cpab.detectors.talib_adapter import (
    AdapterOutcomeKind, TalibPatternDetector, candles_to_ohlc_arrays,
    evaluate_raw, evaluate_raw_series, min_history_for, talib_lookback,
    resolve_pattern_name, raw_to_observation,
)
from cpab.models import Candle

def _c(o, h, l, c, epoch=1_700_000_000, is_closed=True):
    return Candle(symbol="R_100", timeframe="1m", epoch=epoch, open=float(o), high=float(h), low=float(l), close=float(c), is_closed=is_closed)

def _known_positive_engulfing(start=1_700_000_000):
    return [
        _c(10.0, 10.5, 9.8, 10.2, start),
        _c(10.2, 10.4, 9.9, 10.0, start + 60),
        _c(10.0, 10.2, 9.5, 9.6, start + 120),
        _c(9.5, 9.8, 9.0, 9.1, start + 180),
        _c(9.0, 10.8, 8.9, 10.6, start + 240),
    ]

def test_min_history_matches_talib_lookback_plus_one():
    for name in ["CDLENGULFING", "CDLDOJI", "CDLHAMMER", "CDLMORNINGSTAR",
                 "CDLEVENINGSTAR", "CDLHARAMI", "CDLPIERCING", "CDLSHOOTINGSTAR"]:
        lb = int(Function(name).lookback)
        assert talib_lookback(name) == lb
        assert min_history_for(name) == lb + 1

def test_unsupported_pattern_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_pattern_name("CDL_NOT_A_REAL_PATTERN_XYZ")
    with pytest.raises(ValueError, match="Unsupported"):
        min_history_for("CDL_NOT_A_REAL_PATTERN_XYZ")
    raw = evaluate_raw([_c(10, 11, 9, 10.5)] * 20, "CDL_NOT_A_REAL_PATTERN_XYZ")
    assert raw.kind == AdapterOutcomeKind.EXECUTION_FAILURE

def test_insufficient_history_uses_talib_derived_minimum():
    need = min_history_for("CDLDOJI")
    assert need == int(Function("CDLDOJI").lookback) + 1
    short = [_c(10, 12, 8, 10.05, 1_700_000_000 + i * 60) for i in range(need - 1)]
    raw = evaluate_raw(short, "CDLDOJI")
    assert raw.kind == AdapterOutcomeKind.INSUFFICIENT_HISTORY
    assert str(need) in (raw.message or "")

    need_e = min_history_for("CDLENGULFING")
    short_e = [_c(10, 11, 9, 10.5)]
    assert len(short_e) < need_e
    assert evaluate_raw(short_e, "CDLENGULFING").kind == AdapterOutcomeKind.INSUFFICIENT_HISTORY

def test_forming_candle_rejected():
    series = _known_positive_engulfing()
    series[-1] = _c(9.0, 10.8, 8.9, 10.6, series[-1].epoch, is_closed=False)
    assert evaluate_raw(series, "CDLENGULFING").kind == AdapterOutcomeKind.FORMING_CANDLE

def test_raw_matches_direct_talib():
    import talib
    series = _known_positive_engulfing()
    o, h, l, c = candles_to_ohlc_arrays(series)
    direct = int(talib.CDLENGULFING(o, h, l, c)[-1])
    raw = evaluate_raw(series, "CDLENGULFING")
    assert raw.raw_talib_value == direct
    assert raw.candle_epoch == series[-1].epoch

def test_positive_observation_preserves_raw_and_identity():
    import talib
    series = _known_positive_engulfing()
    o, h, l, c = candles_to_ohlc_arrays(series)
    direct = int(talib.CDLENGULFING(o, h, l, c)[-1])
    assert direct != 0
    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:positive-test")
    raw = det.evaluate_raw(series)
    assert raw.kind == AdapterOutcomeKind.PATTERN
    assert raw.raw_talib_value == direct
    obs_list = det.detect(series)
    assert len(obs_list) == 1
    obs = obs_list[0]
    assert obs.payload.get("raw_talib_value") == direct
    assert obs.candle_epoch == series[-1].epoch
    assert obs.detector_config_id == "cfg:positive-test"
    assert obs.direction == ("bullish" if direct > 0 else "bearish")

def test_raw_to_observation_none_for_zero():
    from cpab.detectors.talib_adapter import AdapterRawResult
    zero = AdapterRawResult(kind=AdapterOutcomeKind.NO_PATTERN, pattern="CDLENGULFING", raw_talib_value=0, candle_epoch=1)
    assert raw_to_observation(zero, _known_positive_engulfing()[-1], detector_id="x") is None
