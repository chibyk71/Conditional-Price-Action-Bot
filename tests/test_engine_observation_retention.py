"""
Engine-level integration: observation retention enables multi-component setups.

Proves the live ConditionEngine path (not a hand-built evaluator input):
  swing confirmed @ bar 13 + zone_retest @ 15 + bearish_engulfing @ 16
  → signal at bar 16 only; no re-fire at bar 17.
"""

from __future__ import annotations

from typing import Optional

from cpab.core.engine import ConditionEngine
from cpab.core.observation_store import ObservationStore
from cpab.detectors.base import PriceActionDetector
from cpab.detectors.engulfing import EngulfingDetector
from cpab.market.readiness import ReadinessManager
from cpab.models import Candle, ConditionDef, Observation, ReadinessState
from conftest import make_candle


class FakeMarket:
    def __init__(self) -> None:
        self._candles: dict[tuple[str, str], list[Candle]] = {}

    def set(self, symbol: str, timeframe: str, candles: list[Candle]) -> None:
        self._candles[(symbol, timeframe)] = list(candles)

    def completed_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        return list(self._candles[(symbol, timeframe)])


class ScriptedDetector(PriceActionDetector):
    def __init__(self, name: str, observation: Observation, emit_at: int) -> None:
        self._name = name
        self._observation = observation
        self._emit_at = emit_at
        self._emitted = False

    @property
    def name(self) -> str:
        return self._name

    def detect(self, candles: list[Candle]) -> list[Observation]:
        if self._emitted or not candles:
            return []
        latest = candles[-1]
        if latest.epoch == self._emit_at:
            self._emitted = True
            return [self._observation]
        return []


def _ready_manager(symbol: str = "R_100", timeframe: str = "1m") -> ReadinessManager:
    rm = ReadinessManager()
    rm.register_series(symbol, timeframe)
    rm.set_series(symbol, timeframe, True, "test ready")
    assert rm.state == ReadinessState.READY
    assert rm.can_evaluate() is True
    return rm


def _bars(base: int = 1_700_000_000, gran: int = 60, n: int = 20) -> list[Candle]:
    out = []
    for i in range(n):
        o = 100.0 - i * 0.1
        c = o - 0.2
        out.append(
            make_candle(
                "R_100", "1m", base + i * gran, o, o + 0.15, c - 0.15, c, is_closed=True
            )
        )
    return out


def test_store_accept_dedup_and_query():
    store = ObservationStore(max_size=10, max_age_seconds=None)
    a = Observation.create(
        type="swing_high", symbol="R_100", timeframe="1m", candle_epoch=100, confirmed_at=100
    )
    b = Observation.create(
        type="zone_retest", symbol="R_100", timeframe="1m", candle_epoch=200, confirmed_at=200
    )
    other = Observation.create(
        type="swing_high", symbol="R_75", timeframe="1m", candle_epoch=100, confirmed_at=100
    )
    assert store.accept([a, b, other]) == 3
    assert store.accept([a]) == 0
    assert len(store.query("R_100", "1m")) == 2
    assert len(store.query("R_75", "1m")) == 1


def test_store_max_size_drops_oldest():
    store = ObservationStore(max_size=2, max_age_seconds=None)
    obs = [
        Observation.create(
            type="x", symbol="R_100", timeframe="1m", candle_epoch=100 + i, confirmed_at=100 + i
        )
        for i in range(5)
    ]
    store.accept(obs)
    assert len(store) == 2
    anchors = sorted(o.candle_epoch for o in store.all())
    assert anchors == [103, 104]


def test_store_prune_by_age():
    store = ObservationStore(max_size=100, max_age_seconds=300)
    old = Observation.create(
        type="x", symbol="R_100", timeframe="1m", candle_epoch=1000, confirmed_at=1000
    )
    recent = Observation.create(
        type="y", symbol="R_100", timeframe="1m", candle_epoch=1400, confirmed_at=1400
    )
    store.accept([old, recent])
    removed = store.prune(as_of_epoch=1400)
    assert removed == 1
    assert [o.type for o in store.all()] == ["y"]


def test_engine_multi_component_retention_activates_once():
    base = 1_700_000_000
    gran = 60
    bars = _bars(base=base, gran=gran, n=20)
    swing_epoch = base + 13 * gran
    retest_epoch = base + 15 * gran
    engulf_epoch = base + 16 * gran
    swing = Observation.create(
        type="swing_high", symbol="R_100", timeframe="1m",
        candle_epoch=swing_epoch, event_time=base + 10 * gran, confirmed_at=swing_epoch,
        direction="bearish", detector_id="scripted_swing",
    )
    retest = Observation.create(
        type="zone_retest", symbol="R_100", timeframe="1m",
        candle_epoch=retest_epoch, event_time=retest_epoch, confirmed_at=retest_epoch,
        detector_id="scripted_retest",
    )
    engulf = Observation.create(
        type="bearish_engulfing", symbol="R_100", timeframe="1m",
        candle_epoch=engulf_epoch, event_time=engulf_epoch, confirmed_at=engulf_epoch,
        direction="bearish", detector_id="scripted_engulf",
    )
    market = FakeMarket()
    readiness = _ready_manager()
    detectors = [
        ScriptedDetector("scripted_swing", swing, swing_epoch),
        ScriptedDetector("scripted_retest", retest, retest_epoch),
        ScriptedDetector("scripted_engulf", engulf, engulf_epoch),
    ]
    cond = ConditionDef(
        id="multi_setup", symbol="R_100", timeframe="1m",
        require_observations=[
            {"type": "swing_high", "min_count": 1},
            {"type": "zone_retest", "min_count": 1},
            {"type": "bearish_engulfing", "min_count": 1},
        ],
    )
    engine = ConditionEngine(
        market, detectors, [cond], readiness,
        observation_store=ObservationStore(max_size=100, max_age_seconds=None),
    )

    def run_through(up_to_index: int) -> list:
        market.set("R_100", "1m", bars[: up_to_index + 1])
        return engine.evaluate_once()

    assert run_through(12) == []
    assert len(engine.store) == 0
    assert run_through(13) == []
    assert len(engine.store) == 1
    assert run_through(14) == []
    assert len(engine.store) == 1
    assert run_through(15) == []
    assert len(engine.store) == 2
    signals_16 = run_through(16)
    assert len(signals_16) == 1
    sig = signals_16[0]
    assert sig.condition_id == "multi_setup"
    assert sig.candle_epoch == engulf_epoch
    types = {o.type for o in sig.supporting_observations}
    assert types == {"swing_high", "zone_retest", "bearish_engulfing"}
    assert len(engine.store) == 3
    signals_17 = run_through(17)
    assert signals_17 == []
    assert len(engine.store) == 3


def test_engine_blocks_when_not_ready():
    market = FakeMarket()
    market.set("R_100", "1m", _bars(n=5))
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    engine = ConditionEngine(market, [EngulfingDetector()], [], rm)
    assert engine.evaluate_once() == []
    assert len(engine.store) == 0


def test_engine_retained_obs_not_treated_as_new_activation():
    base = 1_700_000_000
    gran = 60
    bars = _bars(base=base, gran=gran, n=10)
    emit_at = base + 5 * gran
    obs = Observation.create(
        type="swing_high", symbol="R_100", timeframe="1m",
        candle_epoch=emit_at, confirmed_at=emit_at, detector_id="scripted",
    )
    market = FakeMarket()
    readiness = _ready_manager()
    det = ScriptedDetector("scripted", obs, emit_at)
    cond = ConditionDef(
        id="swing_only", symbol="R_100", timeframe="1m",
        require_observations=[{"type": "swing_high", "min_count": 1}],
    )
    engine = ConditionEngine(
        market, [det], [cond], readiness,
        observation_store=ObservationStore(max_size=50, max_age_seconds=None),
    )
    market.set("R_100", "1m", bars[:6])
    s5 = engine.evaluate_once()
    assert len(s5) == 1
    assert s5[0].candle_epoch == emit_at
    market.set("R_100", "1m", bars[:7])
    s6 = engine.evaluate_once()
    assert s6 == []
    assert len(engine.store) == 1


def test_repeated_evaluate_once_does_not_pollute_store_with_duplicate_engulfing():
    """Same candle pair re-evaluated must not insert multiple logical engulfing copies."""
    base = 1_700_000_000
    gran = 60
    prev = make_candle("R_100", "1m", base, 10.0, 10.2, 9.0, 9.2, is_closed=True)
    curr = make_candle("R_100", "1m", base + gran, 9.1, 10.5, 9.0, 10.3, is_closed=True)
    market = FakeMarket()
    market.set("R_100", "1m", [prev, curr])
    readiness = _ready_manager()
    cond = ConditionDef(
        id="engulf_only", symbol="R_100", timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(
        market, [EngulfingDetector()], [cond], readiness,
        observation_store=ObservationStore(max_size=50, max_age_seconds=None),
    )
    first = engine.evaluate_once()
    assert len(first) == 1
    assert len(engine.store) == 1
    stored_id = engine.store.all()[0].id
    for _ in range(5):
        signals = engine.evaluate_once()
        assert signals == []
        assert len(engine.store) == 1
        assert engine.store.all()[0].id == stored_id
    obs = EngulfingDetector().detect([prev, curr])
    assert len(obs) == 1
    assert obs[0].id == stored_id


def test_stable_event_id_distinct_for_different_bars():
    a = Observation.create(
        type="bullish_engulfing", symbol="R_100", timeframe="1m",
        candle_epoch=100, confirmed_at=100, detector_id="engulfing", direction="bullish",
    )
    b = Observation.create(
        type="bullish_engulfing", symbol="R_100", timeframe="1m",
        candle_epoch=160, confirmed_at=160, detector_id="engulfing", direction="bullish",
    )
    assert a.id != b.id
    a2 = Observation.create(
        type="bullish_engulfing", symbol="R_100", timeframe="1m",
        candle_epoch=100, confirmed_at=100, detector_id="engulfing", direction="bullish",
    )
    assert a.id == a2.id
