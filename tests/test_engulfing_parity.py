"""Side-by-side: EngulfingDetector vs TA-Lib CDLENGULFING. No forced parity."""
from __future__ import annotations
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.talib_adapter import AdapterOutcomeKind, TalibPatternDetector, evaluate_raw
from cpab.models import Candle

def _c(o, h, l, c, epoch, is_closed=True):
    return Candle(symbol="R_100", timeframe="1m", epoch=epoch, open=float(o), high=float(h), low=float(l), close=float(c), is_closed=is_closed)

def _classic_bullish(start=1_700_000_000):
    return [_c(10.0, 10.2, 9.0, 9.2, start), _c(9.1, 10.5, 9.0, 10.3, start + 60)]

def _classic_bearish(start=1_700_000_000):
    return [_c(9.2, 10.2, 9.0, 10.0, start), _c(10.1, 10.3, 8.8, 8.9, start + 60)]

def _no_engulf(start=1_700_000_000):
    return [_c(10.0, 10.5, 9.5, 10.2, start), _c(10.2, 10.4, 10.0, 10.3, start + 60)]

def test_handwritten_unchanged_bullish():
    obs = EngulfingDetector().detect(_classic_bullish())
    assert len(obs) == 1 and obs[0].type == "bullish_engulfing" and obs[0].detector_id == "engulfing"

def test_side_by_side_classic_bullish():
    pair = _classic_bullish()
    hw = EngulfingDetector().detect(pair)
    talib_raw = evaluate_raw(pair, "CDLENGULFING")
    talib_obs = TalibPatternDetector("CDLENGULFING").detect(pair)
    assert len(hw) == 1
    if talib_raw.kind == AdapterOutcomeKind.PATTERN and (talib_raw.raw_talib_value or 0) > 0:
        assert talib_obs and talib_obs[0].direction == "bullish"
        assert talib_obs[0].payload.get("raw_talib_value") == talib_raw.raw_talib_value

def test_side_by_side_classic_bearish():
    pair = _classic_bearish()
    hw = EngulfingDetector().detect(pair)
    assert len(hw) == 1 and hw[0].direction == "bearish"
    talib_raw = evaluate_raw(pair, "CDLENGULFING")
    if talib_raw.kind == AdapterOutcomeKind.PATTERN and (talib_raw.raw_talib_value or 0) < 0:
        obs = TalibPatternDetector("CDLENGULFING").detect(pair)
        assert obs and obs[0].direction == "bearish"

def test_side_by_side_no_pattern():
    pair = _no_engulf()
    assert EngulfingDetector().detect(pair) == []
    raw = evaluate_raw(pair, "CDLENGULFING")
    assert raw.raw_talib_value == 0 or raw.kind == AdapterOutcomeKind.NO_PATTERN

def test_engulfing_module_not_migrated():
    from cpab.detectors import engulfing as eng_mod
    from cpab.detectors import talib_adapter as ta_mod
    assert eng_mod.EngulfingDetector is not ta_mod.TalibPatternDetector
    assert EngulfingDetector().name == "engulfing"
