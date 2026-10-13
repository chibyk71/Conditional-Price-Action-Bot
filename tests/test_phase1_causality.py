"""Candle-by-candle replay: no future leakage, stable IDs, forming excluded."""
from __future__ import annotations
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.talib_adapter import TalibPatternDetector
from cpab.models import Candle

def _c(o, h, l, c, epoch, is_closed=True):
    return Candle(symbol="R_100", timeframe="1m", epoch=epoch, open=float(o), high=float(h), low=float(l), close=float(c), is_closed=is_closed)

def _series_with_engulfing():
    base = 1_700_000_000
    return [
        _c(10.0, 10.3, 9.8, 10.1, base),
        _c(10.1, 10.4, 9.9, 10.0, base + 60),
        _c(10.0, 10.2, 9.0, 9.2, base + 120),
        _c(9.1, 10.5, 9.0, 10.3, base + 180),
    ]

def test_replay_no_future_leakage_handwritten():
    series = _series_with_engulfing()
    det = EngulfingDetector()
    emitted_by_step = []
    for i in range(1, len(series) + 1):
        prefix = series[:i]
        obs = det.detect(prefix)
        emitted_by_step.append(obs)
        for o in obs:
            assert o.candle_epoch <= prefix[-1].epoch
            assert any(c.epoch == o.candle_epoch for c in prefix)
    final_obs = emitted_by_step[-1]
    if final_obs:
        final_ids = {o.id for o in final_obs}
        for step_obs in emitted_by_step:
            for o in step_obs:
                if o.candle_epoch == final_obs[0].candle_epoch:
                    assert o.id in final_ids or o.id == final_obs[0].id

def test_replay_no_future_leakage_talib():
    series = _series_with_engulfing()
    det = TalibPatternDetector("CDLENGULFING")
    prior_ids = {}
    for i in range(1, len(series) + 1):
        prefix = series[:i]
        for o in det.detect(prefix):
            assert o.candle_epoch == prefix[-1].epoch
            if o.candle_epoch in prior_ids:
                assert prior_ids[o.candle_epoch] == o.id
            prior_ids[o.candle_epoch] = o.id

def test_forming_never_contributes():
    series = _series_with_engulfing()
    last = series[-1]
    series[-1] = _c(last.open, last.high, last.low, last.close, last.epoch, is_closed=False)
    assert EngulfingDetector().detect(series) == []
    assert TalibPatternDetector("CDLENGULFING").detect(series) == []

def test_repeated_detect_idempotent_ids():
    series = _series_with_engulfing()
    det = EngulfingDetector()
    a, b = det.detect(series), det.detect(series)
    assert [o.id for o in a] == [o.id for o in b]
    det_t = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:fixed")
    assert [o.id for o in det_t.detect(series)] == [o.id for o in det_t.detect(series)]
