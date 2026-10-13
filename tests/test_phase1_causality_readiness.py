"""
Causality + readiness acceptance tests (Phase 1).

- Incremental replay: previously emitted observations remain unchanged as future
  candles arrive.
- Engine readiness gate blocks the TA-Lib adapter when not READY.
"""
from __future__ import annotations

from cpab.core.engine import ConditionEngine
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.talib_adapter import TalibPatternDetector
from cpab.market.readiness import ReadinessManager
from cpab.models import Candle, ConditionDef, ReadinessState


def _c(o, h, l, c, epoch, is_closed=True):
    return Candle(
        symbol="R_100", timeframe="1m", epoch=epoch,
        open=float(o), high=float(h), low=float(l), close=float(c),
        is_closed=is_closed,
    )


def _known_positive_series(start=1_700_000_000):
    """Series that yields CDLENGULFING +100 on the last bar."""
    return [
        _c(10.0, 10.5, 9.8, 10.2, start),
        _c(10.2, 10.4, 9.9, 10.0, start + 60),
        _c(10.0, 10.2, 9.5, 9.6, start + 120),
        _c(9.5, 9.8, 9.0, 9.1, start + 180),
        _c(9.0, 10.8, 8.9, 10.6, start + 240),
    ]


def test_replay_prior_observations_unchanged_talib():
    series = _known_positive_series()
    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:replay")
    prior = {}

    for i in range(1, len(series) + 1):
        prefix = series[:i]
        obs_list = det.detect(prefix)
        for o in obs_list:
            assert o.candle_epoch == prefix[-1].epoch
            if o.candle_epoch in prior:
                old = prior[o.candle_epoch]
                assert old.id == o.id
                assert old.type == o.type
                assert old.direction == o.direction
                assert old.payload.get("raw_talib_value") == o.payload.get("raw_talib_value")
                assert old.detector_config_id == o.detector_config_id
            prior[o.candle_epoch] = o

    assert prior, "expected at least one emission on the positive fixture"
    final = det.detect(series)
    for o in final:
        assert o.candle_epoch in prior
        assert prior[o.candle_epoch].id == o.id


def test_replay_prior_observations_unchanged_handwritten():
    series = _known_positive_series()
    det = EngulfingDetector()
    prior = {}
    for i in range(1, len(series) + 1):
        prefix = series[:i]
        for o in det.detect(prefix):
            if o.candle_epoch in prior:
                assert prior[o.candle_epoch].id == o.id
            prior[o.candle_epoch] = o
    final = det.detect(series)
    for o in final:
        if o.candle_epoch in prior:
            assert prior[o.candle_epoch].id == o.id


class _FakeMarket:
    def __init__(self, candles):
        self._candles = list(candles)

    def completed_candles(self, symbol, timeframe):
        return [c for c in self._candles if c.is_closed and c.symbol == symbol and c.timeframe == timeframe]


def test_readiness_blocks_talib_adapter_via_engine():
    series = _known_positive_series()
    market = _FakeMarket(series)
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    assert rm.state == ReadinessState.INITIALIZING
    assert not rm.can_evaluate()

    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:gate")
    cond = ConditionDef(
        id="ex",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(market, [det], [cond], rm)
    sigs = engine.evaluate_once()
    assert sigs == []
    assert engine.last_observations == []
    assert engine.retained_observations == []


def test_readiness_ready_allows_talib_detection():
    series = _known_positive_series()
    market = _FakeMarket(series)
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    rm.set_series("R_100", "1m", True)
    assert rm.state == ReadinessState.READY
    assert rm.can_evaluate()

    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:gate")
    cond = ConditionDef(
        id="ex",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(market, [det], [cond], rm)
    engine.evaluate_once()
    assert len(engine.last_observations) >= 1
    assert engine.last_observations[0].payload.get("raw_talib_value") != 0
