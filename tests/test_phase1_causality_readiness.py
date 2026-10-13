"""
Causality + readiness acceptance tests (Phase 1).

Causality: for each historical bar, the TA-Lib raw value computed from the
prefix ending at that bar equals the value at that same index when the full
series (with future candles) is evaluated.
"""
from __future__ import annotations
from cpab.core.engine import ConditionEngine
from cpab.detectors.talib_adapter import (
    TalibPatternDetector, evaluate_raw, evaluate_raw_series, min_history_for,
    AdapterOutcomeKind, raw_to_observation,
)
from cpab.market.readiness import ReadinessManager
from cpab.models import Candle, ConditionDef, ReadinessState

def _c(o, h, l, c, epoch, is_closed=True):
    return Candle(symbol="R_100", timeframe="1m", epoch=epoch, open=float(o), high=float(h), low=float(l), close=float(c), is_closed=is_closed)

def _known_positive_series(start=1_700_000_000):
    return [
        _c(10.0, 10.5, 9.8, 10.2, start),
        _c(10.2, 10.4, 9.9, 10.0, start + 60),
        _c(10.0, 10.2, 9.5, 9.6, start + 120),
        _c(9.5, 9.8, 9.0, 9.1, start + 180),
        _c(9.0, 10.8, 8.9, 10.6, start + 240),
    ]

def test_historical_talib_results_stable_with_future_candles():
    series = _known_positive_series()
    pattern = "CDLENGULFING"
    need = min_history_for(pattern)
    full = evaluate_raw_series(series, pattern)
    assert len(full) == len(series)

    for i in range(len(series)):
        prefix = series[: i + 1]
        if len(prefix) < need:
            raw = evaluate_raw(prefix, pattern)
            assert raw.kind == AdapterOutcomeKind.INSUFFICIENT_HISTORY
            continue
        prefix_arr = evaluate_raw_series(prefix, pattern)
        assert int(prefix_arr[i]) == int(full[i]), (
            f"bar {i} epoch={series[i].epoch}: prefix raw={int(prefix_arr[i])} "
            f"!= full raw={int(full[i])} (future leakage)"
        )

def test_historical_observation_fields_match_prefix_result():
    series = _known_positive_series()
    pattern = "CDLENGULFING"
    need = min_history_for(pattern)
    det_id = "talib:cdlengulfing"
    cfg = "cfg:causality"

    for i in range(need - 1, len(series)):
        prefix = series[: i + 1]
        raw_prefix = evaluate_raw(prefix, pattern)
        full_arr = evaluate_raw_series(series, pattern)
        assert raw_prefix.raw_talib_value is not None or raw_prefix.kind == AdapterOutcomeKind.NO_PATTERN
        if raw_prefix.kind == AdapterOutcomeKind.PATTERN:
            assert int(full_arr[i]) == raw_prefix.raw_talib_value
            obs = raw_to_observation(
                raw_prefix, series[i], detector_id=det_id, detector_config_id=cfg
            )
            assert obs is not None
            assert obs.candle_epoch == series[i].epoch
            assert obs.payload.get("raw_talib_value") == raw_prefix.raw_talib_value
            obs2 = raw_to_observation(
                raw_prefix, series[i], detector_id=det_id, detector_config_id=cfg
            )
            assert obs2 is not None and obs2.id == obs.id

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
    cond = ConditionDef(id="ex", symbol="R_100", timeframe="1m",
                        require_observations=[{"type": "bullish_engulfing", "min_count": 1}])
    engine = ConditionEngine(market, [det], [cond], rm)
    assert engine.evaluate_once() == []
    assert engine.last_observations == []
    assert engine.retained_observations == []

def test_readiness_ready_allows_talib_detection():
    series = _known_positive_series()
    market = _FakeMarket(series)
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    rm.set_series("R_100", "1m", True)
    assert rm.state == ReadinessState.READY
    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:gate")
    cond = ConditionDef(id="ex", symbol="R_100", timeframe="1m",
                        require_observations=[{"type": "bullish_engulfing", "min_count": 1}])
    engine = ConditionEngine(market, [det], [cond], rm)
    engine.evaluate_once()
    assert len(engine.last_observations) >= 1
    assert engine.last_observations[0].payload.get("raw_talib_value") != 0
