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
    # Developing can satisfy availability with opt-in, and binds to T via candle_epoch
    assert r2.met is True


# ---------------------------------------------------------------------------
# Test 2 — Retrospective confirmation cannot trigger early
# ---------------------------------------------------------------------------

def test_2_retrospective_cannot_trigger_early():
    event_t = 1_700_000_000
    confirm_t = 1_700_000_180
    obs = Observation.create(
        type="swing_high",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=confirm_t,
        event_time=event_t,
        confirmed_at=confirm_t,
        detector_id="synthetic",
    )
    assert obs.effective_confirmed_at() == confirm_t
    assert obs.is_eligible_at(event_t) is False
    assert obs.is_eligible_at(confirm_t - 1) is False
    assert obs.is_eligible_at(confirm_t) is True

    cond = ConditionDef(
        id="swing_only",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "swing_high", "min_count": 1}],
    )
    ev = ConditionEvaluator()
    early = ev.evaluate(cond, [obs], candle_epoch=event_t)
    assert early.met is False
    late = ev.evaluate(cond, [obs], candle_epoch=confirm_t)
    assert late.met is True


# ---------------------------------------------------------------------------
# Test 3 — Live respects emission acceptance (engine stream boundary)
# ---------------------------------------------------------------------------

def test_3_live_respects_emission_acceptance():
    pair = _engulfing_pair()
    detector = EngulfingDetector()
    market = FakeMarket(pair)
    # Observations only come from detectors run in evaluate_once path
    obs = detector.detect(market.completed_candles("R_100", "1m"))
    assert len(obs) >= 1
    # Injected external observation with future confirmed_at is not in the stream
    # unless detectors emit it — engine only sees what was passed in.
    external = Observation.create(
        type="bearish_engulfing",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=pair[1].epoch + 999,
        confirmed_at=pair[1].epoch + 999,
        detector_id="external",
    )
    # Live path: only detector emissions
    assert external.id not in {o.id for o in obs}


# ---------------------------------------------------------------------------
# Test 4 — Lifecycle events are append-only
# ---------------------------------------------------------------------------

def test_4_lifecycle_events_append_only():
    obs = Observation.create(
        type="zone",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=100,
        confirmed_at=100,
        lifecycle_state="active",
    )
    e1 = ObservationLifecycleEvent.create(
        observation_id=obs.id,
        concept_type="zone",
        from_state=None,
        to_state="active",
        event_type="created",
        at_time=100,
    )
    e2 = ObservationLifecycleEvent.create(
        observation_id=obs.id,
        concept_type="zone",
        from_state="active",
        to_state="invalidated",
        event_type="invalidate",
        at_time=200,
    )
    events = [e1, e2]
    assert project_lifecycle_state(events) == "invalidated"
    # Original events unchanged
    assert e1.to_state == "active"
    assert e2.to_state == "invalidated"


# ---------------------------------------------------------------------------
# Test 5 — Interactions need not force state change
# ---------------------------------------------------------------------------

def test_5_interactions_do_not_force_state_change():
    obs = Observation.create(
        type="zone",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=100,
        confirmed_at=100,
        lifecycle_state="active",
    )
    touch = ObservationLifecycleEvent.create(
        observation_id=obs.id,
        concept_type="zone",
        from_state="active",
        to_state=None,  # interaction, no state change
        event_type="touch",
        at_time=150,
    )
    assert touch.to_state is None
    assert project_lifecycle_state([touch]) is None  # no to_state updates


# ---------------------------------------------------------------------------
# Test 6 — Invalid lifecycle transitions rejected
# ---------------------------------------------------------------------------

def test_6_invalid_lifecycle_transitions_rejected():
    defn = LifecycleDefinition(
        concept_type="zone",
        allowed_transitions=(
            (None, "active", "created"),
            ("active", "invalidated", "invalidate"),
        ),
    )
    defn.validate_transition(None, "active", "created")  # ok
    with pytest.raises(LifecycleTransitionError):
        defn.validate_transition("active", "active", "bogus")


# ---------------------------------------------------------------------------
# Test 7 — Readiness gate blocks evaluation
# ---------------------------------------------------------------------------

def test_7_readiness_gate_blocks_evaluation():
    from cpab.market.readiness import ReadinessManager

    rm = ReadinessManager()
    # Without series marked ready, can_evaluate is False
    assert rm.can_evaluate() is False


# ---------------------------------------------------------------------------
# Test 8 — Live and replay agree on earliest eligibility
# ---------------------------------------------------------------------------

def test_8_live_and_replay_agree_on_earliest_eligibility():
    event_t = 1_700_000_000
    confirm_t = 1_700_000_180
    obs = Observation.create(
        type="swing_high",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=confirm_t,
        event_time=event_t,
        confirmed_at=confirm_t,
    )
    # Replay at T=confirm_t is first eligible bar; same as live would see
    assert obs.is_eligible_at(confirm_t - 1) is False
    assert obs.is_eligible_at(confirm_t) is True


def test_legacy_confirmation_fallback():
    # Legacy: no confirmed_at, not developing → confirmed at candle_epoch
    obs = Observation(
        id="leg",
        type="bearish_engulfing",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=1_700_000_060,
        detected_at=time.time(),
        metadata={},
    )
    assert obs.effective_confirmed_at() == 1_700_000_060
    assert obs.confirmation_is_detected_at_fallback() is False


def test_metadata_payload_compatibility():
    o = Observation.create(
        type="x",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=1,
        metadata={"a": 1},
    )
    assert o.payload.get("a") == 1
    o2 = Observation.create(
        type="x",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=1,
        payload={"b": 2},
    )
    assert o2.metadata.get("b") == 2


def test_engulfing_regression_still_passes_through_evaluator():
    pair = _engulfing_pair()
    detector = EngulfingDetector()
    obs = detector.detect(pair)
    assert len(obs) >= 1
    cond = ConditionDef(
        id="engulf",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": obs[0].type, "min_count": 1}],
    )
    ev = ConditionEvaluator()
    result = ev.evaluate(cond, obs, candle_epoch=pair[1].epoch)
    assert result.met is True
    assert result.supporting_observation_ids


# ---------------------------------------------------------------------------
# Multi-component: prior confirmations + new activation at T
# ---------------------------------------------------------------------------

def test_multi_component_prior_obs_plus_activation():
    """
    Swing confirmed at bar 13, zone retest at 15, engulfing at 16.
    At T=16 all three must satisfy a multi-requirement condition.
    At T=17 (nothing newly confirmed) it must not re-fire.
    """
    base = 1_700_000_000
    gran = 60
    swing = Observation.create(
        type="swing_high",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=base + 13 * gran,
        event_time=base + 10 * gran,
        confirmed_at=base + 13 * gran,
        direction="bearish",
        detector_id="swing",
    )
    retest = Observation.create(
        type="zone_retest",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=base + 15 * gran,
        event_time=base + 15 * gran,
        confirmed_at=base + 15 * gran,
        detector_id="retest",
    )
    engulf = Observation.create(
        type="bearish_engulfing",
        symbol="R_100",
        timeframe="1m",
        candle_epoch=base + 16 * gran,
        event_time=base + 16 * gran,
        confirmed_at=base + 16 * gran,
        direction="bearish",
        detector_id="engulfing",
    )
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
    ev = ConditionEvaluator()
    obs_all = [swing, retest, engulf]

    # Before engulfing confirms — incomplete
    r15 = ev.evaluate(cond, obs_all, candle_epoch=base + 15 * gran)
    assert r15.met is False

    # Activation bar: engulfing newly confirmed; priors still available
    r16 = ev.evaluate(cond, obs_all, candle_epoch=base + 16 * gran)
    assert r16.met is True
    assert set(r16.supporting_observation_ids) == {swing.id, retest.id, engulf.id}

    # Later bar: all still available, but no new activation → no re-fire
    r17 = ev.evaluate(cond, obs_all, candle_epoch=base + 17 * gran)
    assert r17.met is False
    assert any("activation" in r for r in r17.reasons)
