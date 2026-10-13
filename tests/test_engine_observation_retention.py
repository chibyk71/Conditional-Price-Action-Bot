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


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeMarket:
    """Minimal market surface used by ConditionEngine.evaluate_once."""

    def __init__(self) -> None:
        self._candles: dict[tuple[str, str], list[Candle]] = {}

    def set(self, symbol: str, timeframe: str, candles: list[Candle]) -> None:
        self._candles[(symbol, timeframe)] = list(candles)

    def completed_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        return list(self._candles[(symbol, timeframe)])


class ScriptedDetector(PriceActionDetector):
    """
    Emits a fixed observation the first time detect() sees a candle list whose
    latest epoch equals emit_at. Subsequent calls with the same or later epochs
    do not re-emit (simulates one-shot concept detectors).
    """

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
    """Closed 1m candles; bar index i has epoch base + i * gran."""
    out = []
    for i in range(n):
        # Mild bearish drift so engulfing can form on a chosen pair if needed
        o = 100.0 - i * 0.1
        c = o - 0.2
        out.append(
            make_candle(
                "R_100",
                "1m",
                base + i * gran,
                o,
                o + 0.15,
                c - 0.15,
                c,
                is_closed=True,
            )
        )
    return out


# ---------------------------------------------------------------------------
# ObservationStore unit checks
# ---------------------------------------------------------------------------


def test_store_accept_dedup_and_query():
    store = ObservationStore(max_size=10, max_age_seconds=None)
    a = Observation.create(
        type="swing_high",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=100,
        confirmed_at=100,
    )
    b = Observation.create(
        type="zone_retest",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=200,
        confirmed_at=200,
    )
    other = Observation.create(
        type="swing_high",
        symbol="R_75",
        timeframe="1m",
        candle_epoch=100,
        confirmed_at=100,
    )
    assert store.accept([a, b, other]) == 3
    assert store.accept([a]) == 0  # same id, no double insert
    assert len(store.query("R_100", "1m")) == 2
    assert len(store.query("R_75", "1m")) == 1


def test_store_max_size_drops_oldest():
    store = ObservationStore(max_size=2, max_age_seconds=None)
    obs = [
        Observation.create(
            type="x",
            symbol="R_100",
            timeframe="1m",
            candle_epoch=100 + i,
            confirmed_at=100 + i,
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


# ---------------------------------------------------------------------------
# Engine integration: multi-component via retention
# ---------------------------------------------------------------------------


def test_engine_multi_component_retention_activates_once():
    """
    Live engine path:
      - Scripted swing emitted when latest bar is 13
      - Scripted retest emitted when latest bar is 15
      - Scripted engulfing emitted when latest bar is 16
      - Condition requires all three
      - Signal only at bar 16; bar 17 does not re-fire
    Uses ObservationStore retention; does not hand-build evaluator input.
    """
    base = 1_700_000_000
    gran = 60
    bars = _bars(base=base, gran=gran, n=20)

    swing_epoch = base + 13 * gran
    retest_epoch = base + 15 * gran
    engulf_epoch = base + 16 * gran

    swing = Observation.create(
        type="swing_high",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=swing_epoch,
        event_time=base + 10 * gran,
        confirmed_at=swing_epoch,
        direction="bearish",
        detector_id="scripted_swing",
    )
    retest = Observation.create(
        type="zone_retest",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=retest_epoch,
        event_time=retest_epoch,
        confirmed_at=retest_epoch,
        detector_id="scripted_retest",
    )
    engulf = Observation.create(
        type="bearish_engulfing",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=engulf_epoch,
        event_time=engulf_epoch,
        confirmed_at=engulf_epoch,
        direction="bearish",
        detector_id="scripted_engulf",
    )

    market = FakeMarket()
    readiness = _ready_manager()
    detectors: list[PriceActionDetector] = [
        ScriptedDetector("scripted_swing", swing, swing_epoch),
        ScriptedDetector("scripted_retest", retest, retest_epoch),
        ScriptedDetector("scripted_engulf", engulf, engulf_epoch),
    ]
    cond = ConditionDef(
        id="multi_setup",
        symbol="R_100",
        timeframe="1m",
        require_observations=[
            {"type": "swing_high", "min_count": 1},
            {"type": "zone_retest", "min_count": 1},
            {"type": "bearish_engulfing", "min_count": 1},
        ],
    )
    engine = ConditionEngine(
        market,
        detectors,
        [cond],
        readiness,
        observation_store=ObservationStore(max_size=100, max_age_seconds=None),
    )

    # Advance the market one bar at a time through the critical window
    def run_through(up_to_index: int) -> list:
        market.set("R_100", "1m", bars[: up_to_index + 1])
        return engine.evaluate_once()

    # Before swing confirms
    assert run_through(12) == []
    assert len(engine.store) == 0

    # Swing confirms at bar 13 — incomplete setup
    assert run_through(13) == []
    assert len(engine.store) == 1

    # Bar 14 — still incomplete
    assert run_through(14) == []
    assert len(engine.store) == 1

    # Retest at bar 15 — still incomplete (missing engulfing)
    assert run_through(15) == []
    assert len(engine.store) == 2

    # Activation bar 16 — all three available; signal fires once
    signals_16 = run_through(16)
    assert len(signals_16) == 1
    sig = signals_16[0]
    assert sig.condition_id == "multi_setup"
    assert sig.candle_epoch == engulf_epoch
    types = {o.type for o in sig.supporting_observations}
    assert types == {"swing_high", "zone_retest", "bearish_engulfing"}
    assert len(engine.store) == 3

    # Bar 17 — priors still retained, nothing newly confirmed → no re-fire
    signals_17 = run_through(17)
    assert signals_17 == []
    assert len(engine.store) == 3  # retention kept priors


def test_engine_blocks_when_not_ready():
    market = FakeMarket()
    market.set("R_100", "1m", _bars(n=5))
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    # Never marked ready
    engine = ConditionEngine(market, [EngulfingDetector()], [], rm)
    assert engine.evaluate_once() == []
    assert len(engine.store) == 0


def test_engine_retained_obs_not_treated_as_new_activation():
    """
    After a single-obs condition fires on its confirmation bar, the next bar
    must not re-fire solely because the observation remains in the store.
    """
    base = 1_700_000_000
    gran = 60
    bars = _bars(base=base, gran=gran, n=10)
    emit_at = base + 5 * gran
    obs = Observation.create(
        type="swing_high",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=emit_at,
        confirmed_at=emit_at,
        detector_id="scripted",
    )
    market = FakeMarket()
    readiness = _ready_manager()
    det = ScriptedDetector("scripted", obs, emit_at)
    cond = ConditionDef(
        id="swing_only",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "swing_high", "min_count": 1}],
    )
    engine = ConditionEngine(
        market,
        [det],
        [cond],
        readiness,
        observation_store=ObservationStore(max_size=50, max_age_seconds=None),
    )

    market.set("R_100", "1m", bars[:6])  # through bar 5
    s5 = engine.evaluate_once()
    assert len(s5) == 1
    assert s5[0].candle_epoch == emit_at

    market.set("R_100", "1m", bars[:7])  # bar 6
    s6 = engine.evaluate_once()
    assert s6 == []
    assert len(engine.store) == 1
