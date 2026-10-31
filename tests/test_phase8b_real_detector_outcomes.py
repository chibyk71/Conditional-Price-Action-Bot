"""Phase 8B A6: mandatory outcome fixtures using real detectors."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from cpab.conditions.expression import (
    AndExpression, ObservationRequirement, OrExpression, ThenExpression, ThenStep,
)
from cpab.conditions.lifecycle import LifecycleState
from cpab.conditions.schema import SCHEMA_VERSION_V1, ExpiryConfig, SetupDef
from cpab.conditions.validation import parse_setup
from cpab.detectors.base import PriceActionDetector
from cpab.detectors.behavior_volatility import ImpulseDetector
from cpab.detectors.breaker_block import BreakerBlockDetector
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.market_structure import MarketStructureDetector
from cpab.models import Candle, Observation
from cpab.replay.driver import ReplayRunConfig, run_replay
from cpab.replay.evidence import A6_EVIDENCE, record_a6
from cpab.replay.live_style import LiveStyleConfig, run_live_style
from conftest import make_candle

SYMBOL, TF, TF_SEC, START = "R_100", "1m", 60, 1_700_000_000
FIXTURES_8A = Path(__file__).resolve().parent / "fixtures" / "phase8a"
F2_START, F2_STEP = 1_699_992_000, 900

def _talib_ok():
    try:
        from talib.abstract import Function
        _ = int(Function("CDLENGULFING").lookback)
        return True
    except Exception:
        return False

def _mark(fid, ok, **extra):
    status = "real_env_passed" if ok and _talib_ok() else ("stub_only" if ok else "real_env_failed")
    record_a6(fid, status, extra=extra or None)

def _c(epoch, o, h, l, cl, timeframe=TF):
    return Candle(symbol=SYMBOL, timeframe=timeframe, epoch=epoch, open=o, high=h, low=l, close=cl, is_closed=True)

def _ohlc(rows, start=START, step=TF_SEC, timeframe=TF):
    return [_c(start + i * step, o, h, l, cl, timeframe=timeframe) for i, (o, h, l, cl) in enumerate(rows)]

class ScriptedDetector(PriceActionDetector):
    def __init__(self, by_epoch):
        self._by_epoch = {k: list(v) for k, v in by_epoch.items()}
        self._emitted = set()
    @property
    def name(self):
        return "scripted"
    def detect(self, candles):
        if not candles:
            return []
        latest = candles[-1].epoch
        if latest in self._emitted:
            return []
        self._emitted.add(latest)
        return list(self._by_epoch.pop(latest, []))

def _obs(typ, epoch, variant=None):
    return Observation(id=f"{typ}_{variant or 'nv'}_{epoch}", type=typ, symbol=SYMBOL, timeframe=TF,
                       candle_epoch=epoch, event_time=epoch, confirmed_at=epoch, detected_at=float(epoch),
                       developing=False, variant=variant)

def _neutrals(n, start=START):
    return [make_candle(SYMBOL, TF, start + i * TF_SEC, 100.0, 100.2, 99.8, 100.0, is_closed=True) for i in range(n)]

def test_a6_phase8a_completion():
    setup = parse_setup(json.loads((FIXTURES_8A / "phase8a_completion.json").read_text()))
    candles = _neutrals(10)
    e0, e1 = candles[5].epoch, candles[6].epoch
    det = ScriptedDetector({e0: [_obs("structure_break", e0, "bos")], e1: [_obs("zone_interaction", e1, "rejection")]})
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[det], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=6))
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.COMPLETED.value and len(r.signals) == 1
    assert ok
    det2 = ScriptedDetector({e0: [_obs("structure_break", e0, "bos")], e1: [_obs("zone_interaction", e1, "rejection")]})
    r2 = run_replay(ReplayRunConfig(setups=[setup], detectors=[det2], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=6))
    assert r.signals[0].id == r2.signals[0].id
    _mark("phase8a-completion", ok)

def test_a6_phase8a_invalidation():
    setup = parse_setup(json.loads((FIXTURES_8A / "phase8a_invalidation.json").read_text()))
    candles = _neutrals(8)
    e0 = candles[5].epoch
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[ScriptedDetector({e0: [_obs("structure_break", e0, "choch")]})], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=6))
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.INVALIDATED.value and r.signals == []
    assert ok
    _mark("phase8a-invalidation", ok)

def test_a6_phase8a_expiry():
    setup = parse_setup(json.loads((FIXTURES_8A / "phase8a_expiry.json").read_text()))
    candles = _neutrals(10)
    e0 = candles[5].epoch
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[ScriptedDetector({e0: [_obs("structure_break", e0, "bos")]})], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=6))
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.EXPIRED.value and r.signals == []
    assert ok
    _mark("phase8a-expiry", ok)

def _f1_rows():
    base = [(10.0,10.5,9.5,10.0),(10.0,12.0,9.8,11.0),(11.0,11.2,10.5,10.8),(11.0,11.2,9.8,10.0),(10.0,10.5,9.9,10.2),(10.2,13.5,10.0,13.0),(13.0,13.2,12.5,12.8),(12.8,12.9,9.0,9.5)]
    base += [(9.5,9.6,9.4,9.5)] * 20
    base.append((9.5,9.6,5.0,5.2))
    return base

def test_a6_f1_completion_real():
    """F1 completion: real BreakerBlock + Impulse; transition at cycle 98 (padded)."""
    setup = SetupDef(schema_version=SCHEMA_VERSION_V1, setup_id="f1_completion_real", symbol=SYMBOL, timeframe=TF, enabled=True,
        expression=ThenExpression(steps=(
            ThenStep(expression=ObservationRequirement(type="breaker_block", variant="bullish_to_bearish", min_count=1)),
            ThenStep(expression=ObservationRequirement(type="bearish_impulse", variant="atr_normalized_candle_v1", min_count=1)),
        )))
    # Pad 70 neutral bars so impulse lands at index 98 (base impulse was at 28).
    pad = [(10.0, 10.1, 9.9, 10.0)] * 70
    candles = _ohlc(pad + _f1_rows())
    dets = [BreakerBlockDetector(left_strength=1, right_strength=1), ImpulseDetector()]
    assert any(o.type == "breaker_block" for o in dets[0].detect_all_historical(candles, history_complete=True))
    imp = [o for o in dets[1].detect(candles) if o.type == "bearish_impulse"]
    assert imp, "impulse must emit"
    imp_idx = (imp[0].candle_epoch - START) // TF_SEC
    assert imp_idx == 98, f"expected impulse at idx 98, got {imp_idx}"
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=dets, symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=60))
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.COMPLETED.value and len(r.signals) == 1
    assert ok, r.final_setup_states
    tc = r.transition_cycle(setup.setup_id, LifecycleState.COMPLETED.value)
    assert tc == 98, f"expected transition at 98, got {tc}"
    r2 = run_replay(ReplayRunConfig(setups=[setup], detectors=[BreakerBlockDetector(left_strength=1, right_strength=1), ImpulseDetector()], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=60))
    assert r.signals[0].id == r2.signals[0].id
    _mark("F1-completion-real", ok, transition_cycle=tc, impulse_idx=imp_idx)

def test_a6_f1_invalidation_real():
    # Expression never completes (bullish impulse absent); invalidation is the
    # real bearish breaker produced by BreakerBlockDetector.
    setup = SetupDef(schema_version=SCHEMA_VERSION_V1, setup_id="f1_invalidation_real", symbol=SYMBOL, timeframe=TF, enabled=True,
        expression=ObservationRequirement(type="bullish_impulse", variant="atr_normalized_candle_v1", min_count=1),
        invalidation=ObservationRequirement(type="breaker_block", variant="bullish_to_bearish", min_count=1))
    candles = _ohlc(_f1_rows())
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[BreakerBlockDetector(left_strength=1, right_strength=1), ImpulseDetector()], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=5))
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.INVALIDATED.value and r.signals == []
    assert ok, r.final_setup_states
    _mark("F1-invalidation-real", ok)

def test_a6_f1_expiry_real():
    setup = SetupDef(schema_version=SCHEMA_VERSION_V1, setup_id="f1_expiry_real", symbol=SYMBOL, timeframe=TF, enabled=True,
        expression=ThenExpression(steps=(
            ThenStep(expression=ObservationRequirement(type="breaker_block", variant="bullish_to_bearish", min_count=1)),
            ThenStep(expression=ObservationRequirement(type="bearish_impulse", variant="atr_normalized_candle_v1", min_count=1)),
        )),
        expiry=ExpiryConfig(max_age_candles=1))
    rows = [(10.0,10.5,9.5,10.0),(10.0,12.0,9.8,11.0),(11.0,11.2,10.5,10.8),(11.0,11.2,9.8,10.0),(10.0,10.5,9.9,10.2),(10.2,13.5,10.0,13.0),(13.0,13.2,12.5,12.8),(12.8,12.9,9.0,9.5)] + [(9.5,9.55,9.45,9.5)] * 10
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[BreakerBlockDetector(left_strength=1, right_strength=1), ImpulseDetector()], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=_ohlc(rows), arm_count=5))
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.EXPIRED.value and r.signals == []
    assert ok, r.final_setup_states
    _mark("F1-expiry-real", ok)

def _ms(highs, lows, start=F2_START, step=F2_STEP):
    out = []
    for i, (h, l) in enumerate(zip(highs, lows)):
        mid = (h + l) / 2
        out.append(_c(start + i * step, mid, h, l, mid, timeframe="15m"))
    return out

def test_a6_f2_completion_real():
    """Frozen F2 timeline: HL event idx 75, confirm 76, COMPLETED at 76."""
    # Core HL series from market_structure tests (event relative idx 3, conf 4).
    # Pad 72 neutral bars so absolute indices are 75/76.
    pad_h, pad_l = [12.0] * 72, [11.0] * 72
    core_h = [12.0, 11.0, 12.5, 12.0, 13.0, 12.5]
    core_l = [10.0, 8.0, 9.0, 8.5, 9.5, 9.0]
    candles = _ms(pad_h + core_h, pad_l + core_l)
    # Place bullish engulfing at indices 68-69 so it is already accepted
    # before HL confirms at 76 → COMPLETED transition at exactly 76.
    candles[68] = _c(F2_START + 68 * F2_STEP, 10.0, 10.2, 9.0, 9.2, timeframe="15m")
    candles[69] = _c(F2_START + 69 * F2_STEP, 9.1, 10.5, 9.0, 10.3, timeframe="15m")
    ms = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    eng = EngulfingDetector()
    hist = ms.detect_all_historical(candles)
    hl = [o for o in hist if o.variant == "hl"]
    assert hl, "HL must emit from real MarketStructureDetector"
    hl0 = hl[0]
    event_idx = (hl0.candle_epoch - F2_START) // F2_STEP
    conf_idx = (hl0.confirmed_at - F2_START) // F2_STEP
    assert event_idx == 75, f"frozen event idx expected 75, got {event_idx}"
    assert conf_idx == 76, f"frozen confirm idx expected 76, got {conf_idx}"
    eng_obs = eng.detect(candles[:70])
    assert any(o.type == "bullish_engulfing" for o in eng_obs), "engulfing must emit by idx 69"
    setup = SetupDef(schema_version=SCHEMA_VERSION_V1, setup_id="f2_completion_real", symbol=SYMBOL, timeframe="15m", enabled=True,
        expression=AndExpression(children=(
            ObservationRequirement(type="market_structure", variant="hl", min_count=1),
            OrExpression(children=(
                ObservationRequirement(type="bullish_engulfing", min_count=1),
                ObservationRequirement(type="bullish_impulse", variant="atr_normalized_candle_v1", min_count=1),
            )),
        )))
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[ms, eng], symbol=SYMBOL, timeframe="15m", timeframe_seconds=F2_STEP, candles=candles, arm_count=60))
    # PENDING at 75 (HL not yet confirmed)
    st75 = next((c.setup_states.get(setup.setup_id) for c in r.cycles if c.cycle_index == 75), None)
    assert st75 == LifecycleState.PENDING.value, f"expected PENDING at 75, got {st75}"
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.COMPLETED.value and len(r.signals) == 1
    assert ok, r.final_setup_states
    # Completion must not occur before confirmation cycle 76
    for c in r.cycles:
        if c.cycle_index < 76:
            assert c.setup_states.get(setup.setup_id) != LifecycleState.COMPLETED.value
    tc = r.transition_cycle(setup.setup_id, LifecycleState.COMPLETED.value)
    assert tc == 76, f"frozen transition cycle expected 76, got {tc}"
    r2 = run_replay(ReplayRunConfig(setups=[setup], detectors=[MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict"), EngulfingDetector()], symbol=SYMBOL, timeframe="15m", timeframe_seconds=F2_STEP, candles=candles, arm_count=60))
    assert r.signals[0].id == r2.signals[0].id
    _mark("F2-completion-real", ok, event_idx=75, confirm_idx=76, transition_cycle=tc)

def test_a6_f2_invalidation_real():
    """Frozen F2 invalidation: LL event idx 92, confirm 93, INVALIDATED at 93."""
    pad_h, pad_l = [12.0] * 89, [11.5] * 89
    core_h = [14.0, 13.0, 12.0, 11.0, 10.0, 9.0]
    core_l = [12.0, 10.0, 11.0, 8.0, 9.0, 7.0]
    candles = _ms(pad_h + core_h, pad_l + core_l)
    ms = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    hist = ms.detect_all_historical(candles)
    ll = [o for o in hist if o.variant == "ll"]
    assert ll, "LL must emit"
    event_idx = (ll[0].candle_epoch - F2_START) // F2_STEP
    conf_idx = (ll[0].confirmed_at - F2_START) // F2_STEP
    assert event_idx == 92, f"frozen event idx expected 92, got {event_idx}"
    assert conf_idx == 93, f"frozen confirm idx expected 93, got {conf_idx}"
    setup = SetupDef(schema_version=SCHEMA_VERSION_V1, setup_id="f2_invalidation_real", symbol=SYMBOL, timeframe="15m", enabled=True,
        expression=ObservationRequirement(type="market_structure", variant="hl", min_count=1),
        invalidation=ObservationRequirement(type="market_structure", variant="ll", min_count=1))
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[ms], symbol=SYMBOL, timeframe="15m", timeframe_seconds=F2_STEP, candles=candles, arm_count=60))
    st92 = next((c.setup_states.get(setup.setup_id) for c in r.cycles if c.cycle_index == 92), None)
    assert st92 != LifecycleState.INVALIDATED.value, f"must not invalidate at 92, got {st92}"
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.INVALIDATED.value and r.signals == []
    assert ok, r.final_setup_states
    tc = r.transition_cycle(setup.setup_id, LifecycleState.INVALIDATED.value)
    assert tc is not None and tc >= 93, f"transition cycle {tc}"
    for c in r.cycles:
        if c.cycle_index < 93:
            assert c.setup_states.get(setup.setup_id) != LifecycleState.INVALIDATED.value
    _mark("F2-invalidation-real", ok, event_idx=92, confirm_idx=93, transition_cycle=tc)

def test_a6_f3_completion_real():
    setup = SetupDef(schema_version=SCHEMA_VERSION_V1, setup_id="f3_completion_real", symbol=SYMBOL, timeframe=TF, enabled=True,
        expression=ObservationRequirement(type="bullish_engulfing", min_count=1))
    prefix = [make_candle(SYMBOL, TF, START + i * TF_SEC, 100.0, 100.2, 99.8, 100.0, is_closed=True) for i in range(30)]
    ep = START + 30 * TF_SEC
    candles = prefix + [make_candle(SYMBOL, TF, ep, 10.0, 10.2, 9.0, 9.2, is_closed=True), make_candle(SYMBOL, TF, ep+TF_SEC, 9.1, 10.5, 9.0, 10.3, is_closed=True)]
    det = EngulfingDetector()
    assert any(o.type == "bullish_engulfing" for o in det.detect(candles))
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[det], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=20))
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.COMPLETED.value and len(r.signals) == 1
    assert ok and r.transition_cycle(setup.setup_id, LifecycleState.COMPLETED.value) is not None
    live = run_live_style(LiveStyleConfig(setups=[setup], detectors=[EngulfingDetector()], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=20))
    assert live.final_setup_states[setup.setup_id] == LifecycleState.COMPLETED.value
    _mark("F3-completion-real", ok)

def test_a6_f3_invalidation_real():
    setup = SetupDef(schema_version=SCHEMA_VERSION_V1, setup_id="f3_invalidation_real", symbol=SYMBOL, timeframe=TF, enabled=True,
        expression=ObservationRequirement(type="bearish_engulfing", min_count=1),
        invalidation=ObservationRequirement(type="bullish_engulfing", min_count=1))
    prefix = [make_candle(SYMBOL, TF, START + i * TF_SEC, 100.0, 100.2, 99.8, 100.0, is_closed=True) for i in range(20)]
    ep = START + 20 * TF_SEC
    candles = prefix + [make_candle(SYMBOL, TF, ep, 10.0, 10.2, 9.0, 9.2, is_closed=True), make_candle(SYMBOL, TF, ep+TF_SEC, 9.1, 10.5, 9.0, 10.3, is_closed=True)]
    r = run_replay(ReplayRunConfig(setups=[setup], detectors=[EngulfingDetector()], symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC, candles=candles, arm_count=15))
    ok = r.final_setup_states[setup.setup_id] == LifecycleState.INVALIDATED.value and r.signals == []
    assert ok, r.final_setup_states
    _mark("F3-invalidation-real", ok)
