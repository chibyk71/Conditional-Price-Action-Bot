"""Phase 6B single-candle impulse detector tests."""
import math
import pytest
from cpab.detectors import ImpulseDetector
from cpab.detectors.behavior_volatility import ATR_PERIOD, MIN_HISTORY_IMPULSE, VARIANT_ATR_NORMALIZED_CANDLE
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.models import Candle

TF, START = 60, 1_700_000_000

def c(i, o, h, l, close, *, closed=True, symbol="BV_100", timeframe="1m", epoch=None):
    return Candle(symbol, timeframe, START+i*TF if epoch is None else epoch,
                  o, h, l, close, closed)

def series(candidate=None, *, timeframe="1m"):
    bars = [c(i,100,101,99,100,timeframe=timeframe) for i in range(15)]
    bars.append(candidate(15) if candidate else c(15,100,103,100,102,timeframe=timeframe))
    return bars

def test_bullish_bearish_direction_and_boundary_equality():
    det = ImpulseDetector()
    bull = det.detect_all_historical(series())
    bear = det.detect_all_historical(series(lambda i: c(i,102,102,99,100)))
    assert [(o.type,o.direction) for o in bull] == [("bullish_impulse","bullish")]
    assert [(o.type,o.direction) for o in bear] == [("bearish_impulse","bearish")]
    p = bull[0].payload
    assert p["body_atr_ratio"] == pytest.approx(1.0)
    assert p["true_range_atr_ratio"] == pytest.approx(1.5)
    boundary = [c(i,100,101.5,98.5,100) for i in range(15)] + [c(15,100,105,100,103)]
    obs = det.detect_all_historical(boundary)
    assert len(obs) == 1
    assert obs[0].payload["body_range_ratio"] == pytest.approx(0.60)

@pytest.mark.parametrize("candidate", [
    lambda i: c(i,100,103,100,101.99),
    lambda i: c(i,100,102.999,100,102),
    lambda i: c(i,100,103.34,100,102),
])
def test_each_threshold_just_below_boundary_rejects(candidate):
    assert ImpulseDetector().detect_all_historical(series(candidate)) == []

def test_large_range_alone_zero_range_and_neutral_candles_reject():
    det = ImpulseDetector()
    assert det.detect_all_historical(series(lambda i:c(i,100,110,90,100.5))) == []
    assert det.detect_all_historical(series(lambda i:c(i,95,110,90,105))) == []
    assert det.detect_all_historical(series(lambda i:c(i,100,100,100,100))) == []
    assert det.detect_all_historical(series(lambda i:c(i,100,103,100,100))) == []

def test_warmup_closed_only_and_unsupported_timeframe():
    det = ImpulseDetector()
    bars = series()
    assert det.min_history == MIN_HISTORY_IMPULSE == ATR_PERIOD+2 == 16
    assert det.detect_all_historical(bars[:-1]) == []
    open_bars = series(lambda i:c(i,100,103,100,102,closed=False))
    assert det.detect_all_historical(open_bars) == []
    assert det.detect_all_historical(series(lambda i: c(i,100,103,100,102,timeframe="unknown"), timeframe="unknown")) == []

def test_future_candle_does_not_change_candidate_atr_or_identity():
    bars = series()
    det = ImpulseDetector()
    first = det.detect_all_historical(bars)[0]
    extended = bars + [c(16,100,120,80,100)]
    later = [o for o in det.detect_all_historical(extended) if o.event_time == bars[-1].epoch][0]
    assert first.id == later.id
    assert first.payload == later.payload
    assert first.payload["preceding_atr"] == pytest.approx(2.0)

def test_gap_and_invalid_ohlc_break_continuity_and_warmup():
    bars = series()
    last = bars[-1]
    bars[-1] = Candle(last.symbol,last.timeframe,last.epoch+TF,last.open,last.high,last.low,last.close,True)
    assert ImpulseDetector().detect_all_historical(bars) == []
    bars = series()
    bad = bars[7]
    bars[7] = Candle(bad.symbol,bad.timeframe,bad.epoch,100,90,110,100,True)
    assert ImpulseDetector().detect_all_historical(bars) == []

def test_nonfinite_and_mixed_symbol_timeframe_are_rejected():
    bars = series()
    bad = bars[-1]
    bars[-1] = Candle(bad.symbol,bad.timeframe,bad.epoch,100,math.inf,100,102,True)
    assert ImpulseDetector().detect_all_historical(bars) == []
    for field, value in [("symbol","OTHER"),("timeframe","5m")]:
        mixed = series()
        old = mixed[3]
        mixed[3] = Candle(value if field=="symbol" else old.symbol,
                          value if field=="timeframe" else old.timeframe,
                          old.epoch,old.open,old.high,old.low,old.close,True)
        assert ImpulseDetector().detect_all_historical(mixed) == []

def test_observation_identity_timing_and_registry_config():
    bars = series()
    det = ImpulseDetector()
    obs = det.detect_all_historical(bars)
    assert len(obs) == 1
    o = obs[0]
    assert o.id == det.detect_all_historical(bars)[0].id
    assert o.candle_epoch == o.event_time == o.confirmed_at == bars[-1].epoch
    assert not o.developing and o.variant == VARIANT_ATR_NORMALIZED_CANDLE
    assert o.reference_ids == () and o.lifecycle_state is None
    assert o.detector_config_id == deterministic_config_id(det.config_dict())
    registry = build_default_registry()
    assert registry.is_supported("impulse")
    created, info = registry.create_detector("impulse")
    assert created.detector_config_id == info.detector_config_id == det.detector_config_id
    assert created.min_history == info.min_history == 16
    with pytest.raises(ValueError, match="cannot override fixed parameter"):
        registry.create_detector("impulse", config={"body_atr_min": 2.0})
