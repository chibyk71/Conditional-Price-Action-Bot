"""7, 9, 10, 12. Forming vs completed, conditions, signals, dedup."""

from cpab.conditions.evaluator import ConditionEvaluator
from cpab.core.engine import ConditionEngine
from cpab.detectors.engulfing import EngulfingDetector
from cpab.market.cache import CandleCache
from cpab.market.readiness import ReadinessManager
from cpab.models import ConditionDef, ReadinessState
from conftest import make_candle


class FakeMarket:
    def __init__(self, candles):
        self._candles = candles

    def completed_candles(self, symbol, timeframe):
        return [c for c in self._candles if c.is_closed and c.symbol == symbol and c.timeframe == timeframe]

    def series_ok(self, symbol, timeframe) -> bool:
        """Default healthy so engine series-health gate allows evaluation."""
        return True


def _engulfing_pair(symbol="R_100", tf="1m", start=1_700_000_000):
    # bearish then bullish engulfing
    prev = make_candle(symbol, tf, start, 10.0, 10.2, 9.0, 9.2, is_closed=True)  # bear
    curr = make_candle(symbol, tf, start + 60, 9.1, 10.5, 9.0, 10.3, is_closed=True)  # bull engulfs
    return [prev, curr]


def test_engulfing_on_completed_only():
    det = EngulfingDetector()
    pair = _engulfing_pair()
    obs = det.detect(pair)
    assert len(obs) == 1
    assert obs[0].type == "bullish_engulfing"
    assert obs[0].candle_epoch == pair[1].epoch

    # forming latest should not be used by detector if filtered
    forming = pair[:]
    forming[1] = make_candle(
        "R_100", "1m", pair[1].epoch, 9.1, 10.5, 9.0, 10.3, is_closed=False
    )
    assert det.detect(forming) == []


def test_condition_evaluation():
    det = EngulfingDetector()
    pair = _engulfing_pair()
    obs = det.detect(pair)
    cond = ConditionDef(
        id="ex",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    ev = ConditionEvaluator()
    result = ev.evaluate(cond, obs, candle_epoch=pair[1].epoch)
    assert result.met is True
    assert result.supporting_observation_ids


def test_signal_generation_and_dedup():
    pair = _engulfing_pair()
    market = FakeMarket(pair)
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    rm.set_series("R_100", "1m", True)
    assert rm.state == ReadinessState.READY

    cond = ConditionDef(
        id="ex_bull",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(market, [EngulfingDetector()], [cond], rm)
    sigs1 = engine.evaluate_once()
    assert len(sigs1) == 1
    s = sigs1[0]
    assert s.condition_id == "ex_bull"
    assert s.symbol == "R_100"
    assert s.timeframe == "1m"
    assert s.candle_epoch == pair[1].epoch
    assert s.supporting_observations
    assert s.reasons

    # second evaluation same epoch → no duplicate
    sigs2 = engine.evaluate_once()
    assert sigs2 == []


def test_readiness_blocks_signals():
    pair = _engulfing_pair()
    market = FakeMarket(pair)
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    # stay INITIALIZING / not READY
    cond = ConditionDef(
        id="ex",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(market, [EngulfingDetector()], [cond], rm)
    assert engine.evaluate_once() == []
