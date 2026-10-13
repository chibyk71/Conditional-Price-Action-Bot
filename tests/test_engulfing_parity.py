"""Side-by-side: EngulfingDetector vs TA-Lib CDLENGULFING. No forced parity."""
from __future__ import annotations
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.talib_adapter import AdapterOutcomeKind, TalibPatternDetector, evaluate_raw
from cpab.models import Candle

def _c(o, h, l, c, epoch, is_closed=True):
    return Candle(symbol="R_100", timeframe="1m", epoch=epoch, open=float(o), high=float(h), low=float(l), close=float(c), is_closed=is_closed)

def _classic_bullish(start=1_700_000_000):
    return [_c(10.0, 10.2, 9.0, 9.2, start), _c(9.1, 10.5, 9.0, 10.3, start + 60)]

def test_handwritten_unchanged_bullish():
    obs = EngulfingDetector().detect(_classic_bullish())
    assert len(obs) == 1 and obs[0].type == "bullish_engulfing" and obs[0].detector_id == "engulfing"

def test_engulfing_module_not_migrated():
    from cpab.detectors import engulfing as eng_mod
    from cpab.detectors import talib_adapter as ta_mod
    assert eng_mod.EngulfingDetector is not ta_mod.TalibPatternDetector
