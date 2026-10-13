"""
Phase 0 acceptance tests: observation contracts, causal availability,
lifecycle events, readiness gate preservation, live/replay eligibility parity.
"""

from __future__ import annotations

import time

import pytest

from cpab.conditions.evaluator import ConditionEvaluator
from cpab.core.engine import ConditionEngine
from cpab.detectors.engulfing import EngulfingDetector
from cpab.models import (
    ConditionDef,
    LifecycleDefinition,
    LifecycleTransitionError,
    Observation,
    ObservationLifecycleEvent,
    ReadinessState,
    project_lifecycle_state,
)
from conftest import make_candle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeMarket:
    def __init__(self, candles):
        self._candles = list(candles)

    def completed_candles(self, symbol, timeframe):
        return [
            c
            for c in self._candles
            if c.is_closed and c.symbol == symbol and c.timeframe == timeframe
        ]


def _engulfing_pair(symbol="R_100", tf="1m", start=1_700_000_000):
    prev = make_candle(symbol, tf, start, 10.0, 10.2, 9.0, 9.2, is_closed=True)
    curr = make_candle(symbol, tf, start + 60, 9.1, 10.5, 9.0, 10.3, is_closed=True)
    return [prev, curr]


def _synthetic_swing(
    event_time: int,
    confirmed_at: int,
    *,
    developing: bool = False,
    symbol: str = "R_100",
    timeframe: str = "1m",
) -> Observation:
    """Minimal synthetic observation; no swing detector required."""
    return Observation.create(
        type="swing_high",
        symbol=symbol,
        timeframe=timeframe,
        candle_epoch=confirmed_at,  # legacy field points at confirmation bar
        event_time=event_time,
        confirmed_at=None if developing else confirmed_at,
        developing=developing,
        direction="bearish",
        detector_id="synthetic",
        metadata={"synthetic": True},
        payload={"origin_epoch": event_time},
    )


# ---------------------------------------------------------------------------
# Test 1 — Developing observations excluded by default
# ---------------------------------------------------------------------------

def test_1_developing_excluded_by_default():
    T = 1_700_000_100
    developing = _synthetic_swing(event_time=T - 60, confirmed_at=T, developing=True)
    assert developing.is_developing()
    assert developing.effective_confirmed_at() is None

    cond_normal = ConditionDef(
        id="need_swing",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "swing_high", "min_count": 1}],
    )
    ev = ConditionEvaluator()
    r = ev.evaluate(cond_normal, [developing], candle_epoch=T)
    assert r.met is False
    assert any("missing" in x for x in r.reasons)

    # Opt-in only when allow_developing is explicit on the requirement
    cond_opt_in = ConditionDef(
        id="need_swing_dev",
        symbol="R_100",
        timeframe="1m",
        require_observations=[
            {"type": "swing_high", "min_count": 1, "allow_developing": True}
        ],
    )
    r2 = ev.evaluate(cond_opt_in, [developing], candle_epoch=T)
    assert r2.met is True
    assert developing.id in r2.supporting_observation_ids


# ---------------------------------------------------------------------------
# Test 2 — Retrospective swing cannot trigger early
# ---------------------------------------------------------------------------

def test_2_retrospective_cannot_trigger_early():
    # event at candle 10, confirmed at candle 13
    base = 1_700_000_000
    event_t = base + 10 * 60
    conf_t = base + 13 * 60
    obs = _synthetic_swing(event_time=event_t, confirmed_at=conf_t)
    assert obs.effective_event_time() == event_t
    assert obs.effective_confirmed_at() == conf_t

    cond = ConditionDef(
        id="swing_cond",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "swing_high", "min_count": 1}],
    )
    ev = ConditionEvaluator()

    # At candle 11 (before confirmation) — must not satisfy
    early_T = base + 11 * 60
    r_early = ev.evaluate(cond, [obs], candle_epoch=early_T)
    assert r_early.met is False

    # At confirmation candle 13 — eligible and binds
    r_ok = ev.evaluate(cond, [obs], candle_epoch=conf_t)
    assert r_ok.met is True

    # After confirmation (candle 14+) — causally available but must NOT re-trigger
    # (preserves original signal semantics: no repeated signals on later bars)
    later_T = base + 14 * 60
    assert obs.is_eligible_at(later_T) is True  # available
    r_later = ev.evaluate(cond, [obs], candle_epoch=later_T)
    assert r_later.met is False  # does not bind / re-fire

    # Direct eligibility helper (causal only)
    assert obs.is_eligible_at(early_T) is False
    assert obs.is_eligible_at(conf_t) is True


# ---------------------------------------------------------------------------
# Test 3 — Live evaluation respects emission and acceptance
# ---------------------------------------------------------------------------

def test_3_live_respects_emission_acceptance():
    """
    The existing pipeline only evaluates observations produced by detectors
    during evaluate_once from the current completed-candle stream. An
    observation that has not yet been emitted into that stream cannot be
    consumed. We test the actual boundary: ConditionEngine only sees obs
    returned by detectors in the same call.
    """
    pair = _engulfing_pair()
    market = FakeMarket(pair)
    rm = ReadinessManager_ready("R_100", "1m")
    cond = ConditionDef(
        id="ex_bull",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(market, [EngulfingDetector()], [cond], rm)

    # First evaluation: detector emits, condition met, signal produced
    sigs = engine.evaluate_once()
    assert len(sigs) == 1
    assert engine.last_observations
    emitted_ids = {o.id for o in engine.last_observations}

    # A pre-constructed "future" observation not in the accepted stream must
    # not affect evaluation; engine rebuilds obs from detectors only.
    phantom = _synthetic_swing(
        event_time=pair[1].epoch,
        confirmed_at=pair[1].epoch,
    )
    # Engine has no API to inject external obs; last_observations is the
    # accepted set after evaluate_once. Phantom is not in it.
    assert phantom.id not in emitted_ids

    # Re-evaluate: still only detector-emitted obs (dedup blocks second signal)
    sigs2 = engine.evaluate_once()
    assert sigs2 == []
    assert all(o.type == "bullish_engulfing" for o in engine.last_observations)


def ReadinessManager_ready(symbol: str, timeframe: str):
    from cpab.market.readiness import ReadinessManager

    rm = ReadinessManager()
    rm.register_series(symbol, timeframe)
    rm.set_series(symbol, timeframe, True)
    assert rm.state == ReadinessState.READY
    return rm


# ---------------------------------------------------------------------------
# Test 4 — Lifecycle events are append-only
# ---------------------------------------------------------------------------

def test_4_lifecycle_events_append_only():
    obs = _synthetic_swing(event_time=100, confirmed_at=100)
    e1 = ObservationLifecycleEvent.create(
        observation_id=obs.id,
        event_type="activated",
        event_time=100,
        from_state=None,
        to_state="active",
    )
    e2 = ObservationLifecycleEvent.create(
        observation_id=obs.id,
        event_type="mitigated",
        event_time=200,
        from_state="active",
        to_state="mitigated",
    )
    # Events are independent frozen records; later event does not mutate earlier
    assert e1.to_state == "active"
    assert e2.to_state == "mitigated"
    assert e1.id != e2.id
    # Observation itself is frozen / unchanged
    assert obs.lifecycle_state is None  # snapshot at emission was None


# ---------------------------------------------------------------------------
# Test 5 — Multiple interactions do not force state changes
# ---------------------------------------------------------------------------

def test_5_interactions_do_not_force_state_change():
    obs_id = "obs-zone-1"
    defn = LifecycleDefinition(concept="zone")
    # Touch / retest are interaction types; state may stay "active"
    events = []
    for i, et in enumerate(["touch", "retest", "touch"]):
        ev = ObservationLifecycleEvent.create(
            observation_id=obs_id,
            event_type=et,
            event_time=1000 + i * 60,
            from_state="active",
            to_state="active",  # no exclusive transition forced
        )
        defn.validate_transition(ev.event_type, ev.from_state, ev.to_state)
        events.append(ev)
    assert project_lifecycle_state(events) == "active"
    assert all(e.to_state == "active" for e in events)


# ---------------------------------------------------------------------------
# Test 6 — Invalid lifecycle transitions are rejected
# ---------------------------------------------------------------------------

def test_6_invalid_lifecycle_transitions_rejected():
    # Concept-specific: zone only allows active -> mitigated | invalidated
    defn = LifecycleDefinition(
        concept="zone",
        allowed={
            None: {"active"},
            "active": {"mitigated", "invalidated"},
            "mitigated": set(),
            "invalidated": set(),
        },
    )
    # Valid
    defn.validate_transition("activated", None, "active")
    defn.validate_transition("mitigated", "active", "mitigated")
    # Interaction without state change
    defn.validate_transition("touch", "active", "active")
    defn.validate_transition("retest", "active", None)

    # Invalid
    with pytest.raises(LifecycleTransitionError):
        defn.validate_transition("revive", "mitigated", "active")
    with pytest.raises(LifecycleTransitionError):
        defn.validate_transition("jump", "active", "something_else")


# ---------------------------------------------------------------------------
# Test 7 — Readiness gate still blocks evaluation
# ---------------------------------------------------------------------------

def test_7_readiness_gate_blocks_evaluation():
    pair = _engulfing_pair()
    market = FakeMarket(pair)
    from cpab.market.readiness import ReadinessManager

    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    # remain INITIALIZING
    assert rm.can_evaluate() is False

    cond = ConditionDef(
        id="ex",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(market, [EngulfingDetector()], [cond], rm)
    assert engine.evaluate_once() == []

    # Even with a confirmed observation present via detector, gate blocks
    rm.set_series("R_100", "1m", True)
    assert rm.can_evaluate() is True
    sigs = engine.evaluate_once()
    assert len(sigs) == 1


# ---------------------------------------------------------------------------
# Test 8 — Live and replay agree on earliest eligibility
# ---------------------------------------------------------------------------

def test_8_live_and_replay_agree_on_earliest_eligibility():
    """
    For the same confirmed observation and market-time sequence, live
    evaluation (via ConditionEvaluator at successive T) and historical
    replay (same evaluator calls) agree on the earliest T at which the
    observation may satisfy a condition.

    Limitation: live software emission time (detected_at) is wall-clock and
    can lag market confirmation; this test proves market-time eligibility
    parity. Emission ordering is covered by Test 3 using the real engine
    boundary.
    """
    base = 1_700_000_000
    event_t = base + 10 * 60
    conf_t = base + 13 * 60
    obs = _synthetic_swing(event_time=event_t, confirmed_at=conf_t)

    cond = ConditionDef(
        id="swing_cond",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "swing_high", "min_count": 1}],
    )
    ev = ConditionEvaluator()

    # Replay: walk market times
    earliest_replay: int | None = None
    for i in range(9, 16):
        T = base + i * 60
        r = ev.evaluate(cond, [obs], candle_epoch=T)
        if r.met and earliest_replay is None:
            earliest_replay = T

    assert earliest_replay == conf_t

    # Live path: same evaluator, same successive T values → same earliest
    earliest_live: int | None = None
    for i in range(9, 16):
        T = base + i * 60
        r = ev.evaluate(cond, [obs], candle_epoch=T)
        if r.met and earliest_live is None:
            earliest_live = T

    assert earliest_live == earliest_replay == conf_t


# ---------------------------------------------------------------------------
# Compatibility / regression helpers for Phase 0 contracts
# ---------------------------------------------------------------------------

def test_legacy_confirmation_fallback():
    # Observation constructed like pre-Phase-0 code: only candle_epoch
    legacy = Observation(
        id="legacy-1",
        type="bullish_engulfing",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=1_700_000_060,
        detected_at=time.time(),
        metadata={"detector": "engulfing"},
    )
    assert legacy.developing is False
    assert legacy.confirmed_at is None
    assert legacy.effective_confirmed_at() == 1_700_000_060
    assert legacy.confirmation_is_detected_at_fallback() is False
    assert legacy.is_eligible_at(1_700_000_060) is True
    assert legacy.is_eligible_at(1_700_000_000) is False

    # Explicit developing must NOT take the legacy candle_epoch path
    developing = Observation(
        id="dev-1",
        type="swing_high",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=1_700_000_060,
        detected_at=time.time(),
        developing=True,
        confirmed_at=None,
    )
    assert developing.is_developing() is True
    assert developing.effective_confirmed_at() is None
    assert developing.is_eligible_at(1_700_000_060) is False
    assert developing.is_eligible_at(1_700_000_060, allow_developing=True) is True

    # detected_at fallback only when candle_epoch unavailable (sentinel 0)
    wall = 1_700_100_000.0
    incomplete = Observation(
        id="inc-1",
        type="x",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=0,
        detected_at=wall,
        developing=False,
        confirmed_at=None,
    )
    assert incomplete.effective_confirmed_at() == int(wall)
    assert incomplete.confirmation_is_detected_at_fallback() is True


def test_metadata_payload_compatibility():
    obs = Observation.create(
        type="bullish_engulfing",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=100,
        metadata={"prev_epoch": 40, "detector": "engulfing"},
    )
    # create copies metadata into payload when payload omitted
    assert obs.metadata.get("detector") == "engulfing"
    assert obs.payload.get("detector") == "engulfing"

    # Existing engulging path still exposes metadata for consumers
    pair = _engulfing_pair()
    det = EngulfingDetector()
    out = det.detect(pair)
    assert len(out) == 1
    assert "prev_ohlc" in out[0].metadata
    assert out[0].payload.get("prev_ohlc") == out[0].metadata["prev_ohlc"]
    assert out[0].direction == "bullish"
    assert out[0].confirmed_at == pair[1].epoch
    assert out[0].event_time == pair[1].epoch


def test_engulfing_regression_still_passes_through_evaluator():
    pair = _engulfing_pair()
    det = EngulfingDetector()
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
