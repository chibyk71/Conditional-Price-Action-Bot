"""Phase 7D: setup-engine integration — config, lifecycle, signals, safety."""

from __future__ import annotations

from typing import Optional

import pytest

from cpab.conditions.expression import (
    ObservationRequirement,
    ThenExpression,
    ThenStep,
)
from cpab.conditions.lifecycle import LifecycleState, new_pending_instance
from cpab.conditions.schema import ExpiryConfig, SCHEMA_VERSION_V1, SetupDef
from cpab.conditions.validation import SetupValidationError, parse_setups
from cpab.config import conditions_from_config, setups_from_config
from cpab.core.engine import ConditionEngine, _deterministic_setup_signal_id
from cpab.core.observation_store import ObservationStore
from cpab.detectors.base import PriceActionDetector
from cpab.market.readiness import ReadinessManager
from cpab.models import ConditionDef, Observation, ReadinessState, Signal
from conftest import make_candle

SYMBOL = "R_100"
TF = "1m"
TF_SEC = 60
START = 1_700_000_000


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeMarket:
    """Minimal market: completed candles + optional health / granularity."""

    def __init__(
        self,
        candles,
        *,
        granularity: int = TF_SEC,
        series_healthy: bool = True,
    ):
        self._candles = list(candles)
        self._granularity = granularity
        self._series_healthy = series_healthy
        self._gran_by_name = {TF: granularity}

    def completed_candles(self, symbol, timeframe):
        return [
            c
            for c in self._candles
            if c.is_closed and c.symbol == symbol and c.timeframe == timeframe
        ]

    def timeframe_seconds(self, timeframe: str) -> Optional[int]:
        if timeframe == TF:
            return self._granularity
        return None

    def series_ok(self, symbol, timeframe) -> bool:
        return self._series_healthy

    def set_candles(self, candles) -> None:
        self._candles = list(candles)

    def set_healthy(self, ok: bool) -> None:
        self._series_healthy = ok


class ScriptedDetector(PriceActionDetector):
    """Emits pre-scripted observations keyed by evaluation candle epoch.

    Mapping: candle_epoch -> list of Observation to emit when that epoch is
    the latest completed candle. Observations are only returned once (consumed).
    """

    def __init__(self, by_epoch: dict[int, list[Observation]]):
        self._by_epoch = {k: list(v) for k, v in by_epoch.items()}
        self._emitted_epochs: set[int] = set()

    @property
    def name(self) -> str:
        return "scripted"

    def detect(self, candles: list) -> list[Observation]:
        if not candles:
            return []
        latest = candles[-1].epoch
        if latest in self._emitted_epochs:
            return []
        obs = self._by_epoch.get(latest, [])
        self._emitted_epochs.add(latest)
        return list(obs)


def _obs(
    typ: str,
    candle_epoch: int,
    *,
    confirmed_at: int | None = None,
    oid: str | None = None,
    developing: bool = False,
) -> Observation:
    conf = None if developing else (confirmed_at if confirmed_at is not None else candle_epoch)
    return Observation(
        id=oid or f"{typ}_{candle_epoch}",
        type=typ,
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=candle_epoch,
        event_time=candle_epoch,
        confirmed_at=conf,
        detected_at=float(candle_epoch),
        developing=developing,
    )


def _candles_up_to(end_epoch: int, *, start: int = START, step: int = TF_SEC):
    out = []
    e = start
    while e <= end_epoch:
        out.append(
            make_candle(SYMBOL, TF, e, 100.0, 101.0, 99.0, 100.5, is_closed=True)
        )
        e += step
    return out


def _ready_rm() -> ReadinessManager:
    rm = ReadinessManager()
    rm.register_series(SYMBOL, TF)
    rm.set_series(SYMBOL, TF, True)
    assert rm.state == ReadinessState.READY
    return rm


def _setup(
    expression,
    *,
    setup_id: str = "s1",
    enabled: bool = True,
    invalidation=None,
    expiry=None,
) -> SetupDef:
    return SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id=setup_id,
        symbol=SYMBOL,
        timeframe=TF,
        enabled=enabled,
        expression=expression,
        invalidation=invalidation,
        expiry=expiry,
    )


def _engine(
    market,
    *,
    setups=None,
    conditions=None,
    detectors=None,
    readiness=None,
) -> ConditionEngine:
    return ConditionEngine(
        market,
        detectors or [],
        conditions or [],
        readiness or _ready_rm(),
        setups=setups or [],
        timeframe_seconds_by_name={TF: TF_SEC},
    )


# ---------------------------------------------------------------------------
# 1. Valid setup configuration loading and validation
# ---------------------------------------------------------------------------


def test_valid_setup_config_loading():
    cfg = {
        "setups": [
            {
                "schema_version": 1,
                "setup_id": "simple_a",
                "symbol": SYMBOL,
                "timeframe": TF,
                "enabled": True,
                "expression": {"type": "a"},
            }
        ]
    }
    setups = setups_from_config(cfg)
    assert len(setups) == 1
    assert setups[0].setup_id == "simple_a"
    assert setups[0].enabled is True


def test_invalid_setup_config_raises():
    cfg = {
        "setups": [
            {
                "schema_version": 1,
                "setup_id": "bad",
                "symbol": SYMBOL,
                "timeframe": TF,
                "enabled": True,
                # missing expression
            }
        ]
    }
    with pytest.raises(SetupValidationError):
        setups_from_config(cfg)


# ---------------------------------------------------------------------------
# 2. Legacy conditions remain backward-compatible
# ---------------------------------------------------------------------------


def test_legacy_conditions_backward_compatible():
    cfg = {
        "conditions": [
            {
                "id": "legacy_engulf",
                "symbol": SYMBOL,
                "timeframe": TF,
                "enabled": True,
                "require_observations": [{"type": "bullish_engulfing", "min_count": 1}],
            }
        ],
        "setups": [
            {
                "schema_version": 1,
                "setup_id": "setup_a",
                "symbol": SYMBOL,
                "timeframe": TF,
                "enabled": True,
                "expression": {"type": "a"},
            }
        ],
    }
    conds = conditions_from_config(cfg)
    setups = setups_from_config(cfg)
    assert len(conds) == 1
    assert conds[0].id == "legacy_engulf"
    assert len(setups) == 1
    # Legacy path must not be reinterpreted as SetupDef
    assert not hasattr(conds[0], "expression")


def test_legacy_condition_still_emits_via_engine():
    ep = START + 60
    candles = _candles_up_to(ep)
    market = FakeMarket(candles)
    det = ScriptedDetector({ep: [_obs("bullish_engulfing", ep)]})
    cond = ConditionDef(
        id="legacy_engulf",
        symbol=SYMBOL,
        timeframe=TF,
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = _engine(market, conditions=[cond], detectors=[det])
    sigs = engine.evaluate_once()
    assert len(sigs) == 1
    assert sigs[0].condition_id == "legacy_engulf"
    assert "kind" not in (sigs[0].metadata or {}) or sigs[0].metadata.get("kind") != "setup"


# ---------------------------------------------------------------------------
# 3. Simple expression completes a setup and emits a signal
# ---------------------------------------------------------------------------


def test_simple_expression_completes_and_emits_signal():
    ep = START + 60
    candles = _candles_up_to(ep)
    market = FakeMarket(candles)
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="simple_a")
    engine = _engine(market, setups=[setup], detectors=[det])
    sigs = engine.evaluate_once()
    assert len(sigs) == 1
    s = sigs[0]
    assert s.condition_id == "simple_a"
    assert s.symbol == SYMBOL
    assert s.timeframe == TF
    assert s.candle_epoch == ep
    assert s.supporting_observations
    assert s.supporting_observations[0].type == "a"
    assert s.metadata.get("kind") == "setup"
    assert s.metadata.get("lifecycle_state") == LifecycleState.COMPLETED.value
    assert s.id.startswith("sigsetup:")
    inst = engine.get_instance("simple_a")
    assert inst is not None
    assert inst.state == LifecycleState.COMPLETED


# ---------------------------------------------------------------------------
# 4. Multi-step THEN completion across evaluation candles
# ---------------------------------------------------------------------------


def test_then_completion_across_candles():
    ep0 = START
    ep1 = START + 60
    ep2 = START + 120
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="then_ab",
    )
    market = FakeMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {
            ep0: [_obs("a", ep0)],
            ep1: [_obs("b", ep1)],
        }
    )
    engine = _engine(market, setups=[setup], detectors=[det])

    # Candle 0: step a advances → IN_PROGRESS, no signal
    sigs0 = engine.evaluate_once()
    assert sigs0 == []
    assert engine.get_instance("then_ab").state == LifecycleState.IN_PROGRESS

    # Candle 1: step b completes → signal
    market.set_candles(_candles_up_to(ep1))
    sigs1 = engine.evaluate_once()
    assert len(sigs1) == 1
    assert sigs1[0].condition_id == "then_ab"
    assert engine.get_instance("then_ab").state == LifecycleState.COMPLETED
    assert {o.type for o in sigs1[0].supporting_observations} >= {"a", "b"}

    # Later candle: no re-fire
    market.set_candles(_candles_up_to(ep2))
    assert engine.evaluate_once() == []


# ---------------------------------------------------------------------------
# 5. IN_PROGRESS produces no signal
# ---------------------------------------------------------------------------


def test_in_progress_no_signal():
    ep0 = START
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="ip",
    )
    market = FakeMarket(_candles_up_to(ep0))
    det = ScriptedDetector({ep0: [_obs("a", ep0)]})
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("ip").state == LifecycleState.IN_PROGRESS


# ---------------------------------------------------------------------------
# 6. Invalidation and expiry produce no signals
# ---------------------------------------------------------------------------


def test_invalidation_no_signal():
    ep0 = START
    ep1 = START + 60
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="inv",
        invalidation=ObservationRequirement(type="x"),
    )
    market = FakeMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {
            ep0: [_obs("a", ep0)],
            ep1: [_obs("x", ep1)],
        }
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("inv").state == LifecycleState.IN_PROGRESS

    market.set_candles(_candles_up_to(ep1))
    sigs = engine.evaluate_once()
    assert sigs == []
    assert engine.get_instance("inv").state == LifecycleState.INVALIDATED


def test_expiry_no_signal():
    ep0 = START
    # max_age_candles=1 → expires on the next contiguous candle with no completion
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="exp",
        expiry=ExpiryConfig(max_age_candles=1),
    )
    market = FakeMarket(_candles_up_to(ep0))
    det = ScriptedDetector({ep0: [_obs("a", ep0)]})
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("exp").state == LifecycleState.IN_PROGRESS

    ep1 = START + 60
    market.set_candles(_candles_up_to(ep1))
    # No "b" observation — should expire, not complete
    sigs = engine.evaluate_once()
    assert sigs == []
    assert engine.get_instance("exp").state == LifecycleState.EXPIRED


# ---------------------------------------------------------------------------
# 7. Readiness blocking setup evaluation
# ---------------------------------------------------------------------------


def test_readiness_blocks_setup_evaluation():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="blocked")
    rm = ReadinessManager()
    rm.register_series(SYMBOL, TF)
    # stay not READY
    engine = _engine(market, setups=[setup], detectors=[det], readiness=rm)
    assert engine.evaluate_once() == []
    # Instance remains PENDING (never evaluated)
    assert engine.get_instance("blocked").state == LifecycleState.PENDING


# ---------------------------------------------------------------------------
# 8. Unhealthy / discontinuous candles do not advance state
# ---------------------------------------------------------------------------


def test_unhealthy_series_does_not_advance():
    ep0 = START
    ep1 = START + 60
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="unhealthy",
    )
    market = FakeMarket(_candles_up_to(ep0), series_healthy=True)
    det = ScriptedDetector(
        {
            ep0: [_obs("a", ep0)],
            ep1: [_obs("b", ep1)],
        }
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("unhealthy").state == LifecycleState.IN_PROGRESS

    market.set_candles(_candles_up_to(ep1))
    market.set_healthy(False)
    sigs = engine.evaluate_once()
    assert sigs == []
    # Diagnostic-only: still IN_PROGRESS, last_evaluation not advanced to ep1
    inst = engine.get_instance("unhealthy")
    assert inst.state == LifecycleState.IN_PROGRESS
    assert inst.last_evaluation_candle == ep0


def test_unhealthy_cycle_observations_not_retained_for_later_setup():
    """Observations produced while series is unhealthy must not enter the store
    and must not later complete a setup after health recovers.
    """
    ep0 = START
    ep1 = START + 60
    ep2 = START + 120
    setup = _setup(ObservationRequirement(type="a"), setup_id="stale_obs")
    market = FakeMarket(_candles_up_to(ep0), series_healthy=False)
    # Detector would emit "a" on the unhealthy bar if detectors were allowed to run
    det = ScriptedDetector(
        {
            ep0: [_obs("a", ep0)],
            ep1: [_obs("a", ep1)],
            ep2: [],  # no new evidence on recovery bar
        }
    )
    engine = _engine(market, setups=[setup], detectors=[det])

    # Unhealthy: no detect, no store, no completion
    assert engine.evaluate_once() == []
    assert engine.get_instance("stale_obs").state == LifecycleState.PENDING
    assert all(o.type != "a" for o in engine.store.query(SYMBOL, TF))

    # Still unhealthy on next bar with scripted "a" — still must not retain
    market.set_candles(_candles_up_to(ep1))
    assert engine.evaluate_once() == []
    assert all(o.type != "a" for o in engine.store.query(SYMBOL, TF))
    assert engine.get_instance("stale_obs").state == LifecycleState.PENDING

    # Health recovers but detector has no new emission for ep2; prior unhealthy
    # observations must not be available to complete the setup.
    market.set_candles(_candles_up_to(ep2))
    market.set_healthy(True)
    sigs = engine.evaluate_once()
    assert sigs == []
    assert engine.get_instance("stale_obs").state != LifecycleState.COMPLETED
    assert all(o.type != "a" for o in engine.store.query(SYMBOL, TF))


def test_legacy_condition_no_signal_when_series_unhealthy():
    """Legacy conditions must not emit when series health fails the gate."""
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep), series_healthy=False)
    det = ScriptedDetector({ep: [_obs("bullish_engulfing", ep)]})
    cond = ConditionDef(
        id="legacy_unhealthy",
        symbol=SYMBOL,
        timeframe=TF,
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = _engine(market, conditions=[cond], detectors=[det])
    assert engine.evaluate_once() == []
    # Observation from the unhealthy cycle must not be retained
    assert all(o.type != "bullish_engulfing" for o in engine.store.query(SYMBOL, TF))

    # Recovery without a new detector emission still yields no legacy signal
    market.set_candles(_candles_up_to(ep + 60))
    market.set_healthy(True)
    # Scripted detector already consumed ep; no emission on ep+60
    assert engine.evaluate_once() == []


class _RaisingCacheMarket:
    """Market double that can raise from get_cache / is_series_healthy / series_ok."""

    def __init__(
        self,
        candles,
        *,
        raise_get_cache: bool = False,
        cache_none: bool = False,
        raise_is_healthy: bool = False,
        raise_series_ok: bool = False,
        omit_series_ok: bool = False,
        omit_get_cache: bool = False,
    ):
        self._candles = list(candles)
        self._raise_get_cache = raise_get_cache
        self._cache_none = cache_none
        self._raise_is_healthy = raise_is_healthy
        self._raise_series_ok = raise_series_ok
        self._omit_series_ok = omit_series_ok
        self._omit_get_cache = omit_get_cache
        self._gran_by_name = {TF: TF_SEC}

    def completed_candles(self, symbol, timeframe):
        return [
            c
            for c in self._candles
            if c.is_closed and c.symbol == symbol and c.timeframe == timeframe
        ]

    def timeframe_seconds(self, timeframe: str):
        return TF_SEC if timeframe == TF else None

    def get_cache(self, symbol, timeframe):
        if self._omit_get_cache:
            raise AttributeError("get_cache omitted")  # not used if deleted
        if self._raise_get_cache:
            raise RuntimeError("cache lookup failed")
        if self._cache_none:
            return None

        class _Cache:
            def __init__(self, parent):
                self._parent = parent

            def is_series_healthy(self, min_candles=1):
                if self._parent._raise_is_healthy:
                    raise RuntimeError("health check exploded")
                return True

            def quarantine_epochs(self):
                return set()

        return _Cache(self)

    def series_ok(self, symbol, timeframe):
        if self._raise_series_ok:
            raise RuntimeError("series_ok exploded")
        return True


def _strip_attr(obj, name: str):
    """Remove an attribute so getattr(..., None) treats it as missing."""
    if hasattr(obj, name):
        try:
            delattr(obj, name)
        except Exception:
            # Bound methods on class — override instance dict with non-callable
            object.__setattr__(obj, name, None)


def test_get_cache_exception_does_not_advance():
    ep0 = START
    ep1 = START + 60
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="cache_exc",
    )
    market = _RaisingCacheMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {ep0: [_obs("a", ep0)], ep1: [_obs("b", ep1)]}
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("cache_exc").state == LifecycleState.IN_PROGRESS
    prior = engine.get_instance("cache_exc").last_evaluation_candle

    market._candles = _candles_up_to(ep1)
    market._raise_get_cache = True
    assert engine.evaluate_once() == []
    inst = engine.get_instance("cache_exc")
    assert inst.state == LifecycleState.IN_PROGRESS
    assert inst.last_evaluation_candle == prior == ep0


def test_is_series_healthy_exception_does_not_advance():
    ep0 = START
    ep1 = START + 60
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="health_exc",
    )
    market = _RaisingCacheMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {ep0: [_obs("a", ep0)], ep1: [_obs("b", ep1)]}
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("health_exc").state == LifecycleState.IN_PROGRESS

    market._candles = _candles_up_to(ep1)
    market._raise_is_healthy = True
    # Prefer cache path; strip series_ok so failure is not masked
    object.__setattr__(market, "series_ok", None)
    assert engine.evaluate_once() == []
    inst = engine.get_instance("health_exc")
    assert inst.state == LifecycleState.IN_PROGRESS
    assert inst.last_evaluation_candle == ep0


def test_get_cache_none_does_not_advance():
    ep0 = START
    ep1 = START + 60
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="cache_none",
    )
    market = _RaisingCacheMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {ep0: [_obs("a", ep0)], ep1: [_obs("b", ep1)]}
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("cache_none").state == LifecycleState.IN_PROGRESS

    market._candles = _candles_up_to(ep1)
    market._cache_none = True
    object.__setattr__(market, "series_ok", None)
    assert engine.evaluate_once() == []
    inst = engine.get_instance("cache_none")
    assert inst.state == LifecycleState.IN_PROGRESS
    assert inst.last_evaluation_candle == ep0


def test_series_ok_exception_does_not_advance():
    ep0 = START
    ep1 = START + 60
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="series_ok_exc",
    )
    # No get_cache — only series_ok path
    market = FakeMarket(_candles_up_to(ep0), series_healthy=True)
    det = ScriptedDetector(
        {ep0: [_obs("a", ep0)], ep1: [_obs("b", ep1)]}
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("series_ok_exc").state == LifecycleState.IN_PROGRESS

    market.set_candles(_candles_up_to(ep1))

    def _boom(symbol, timeframe):
        raise RuntimeError("series_ok exploded")

    market.series_ok = _boom  # type: ignore[method-assign]
    assert engine.evaluate_once() == []
    inst = engine.get_instance("series_ok_exc")
    assert inst.state == LifecycleState.IN_PROGRESS
    assert inst.last_evaluation_candle == ep0


def test_missing_health_interface_does_not_advance():
    """No get_cache and no series_ok → cannot verify health → no advancement."""
    ep0 = START
    ep1 = START + 60
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="no_health",
    )

    class BareMarket:
        def __init__(self, candles):
            self._candles = list(candles)
            self._gran_by_name = {TF: TF_SEC}

        def completed_candles(self, symbol, timeframe):
            return [
                c
                for c in self._candles
                if c.is_closed and c.symbol == symbol and c.timeframe == timeframe
            ]

        def timeframe_seconds(self, timeframe: str):
            return TF_SEC if timeframe == TF else None

    # First evaluation also requires health; with no interface, stays PENDING
    market = BareMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {ep0: [_obs("a", ep0)], ep1: [_obs("b", ep1)]}
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    inst = engine.get_instance("no_health")
    assert inst.state == LifecycleState.PENDING
    assert inst.last_evaluation_candle is None

    # Still cannot advance on next bar
    market._candles = _candles_up_to(ep1)
    assert engine.evaluate_once() == []
    inst = engine.get_instance("no_health")
    assert inst.state == LifecycleState.PENDING
    assert inst.last_evaluation_candle is None


def test_discontinuous_candle_does_not_advance():
    ep0 = START
    # Skip ep0+60; jump to ep0+120
    gap_epoch = START + 120
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="gap",
    )
    market = FakeMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {
            ep0: [_obs("a", ep0)],
            gap_epoch: [_obs("b", gap_epoch)],
        }
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("gap").state == LifecycleState.IN_PROGRESS

    # Present only gap candle as latest (discontinuous from engine's last accepted)
    market.set_candles(_candles_up_to(gap_epoch))
    # completed_candles returns full series; lifecycle checks continuity vs last accepted
    sigs = engine.evaluate_once()
    assert sigs == []
    inst = engine.get_instance("gap")
    # Still IN_PROGRESS; discontinuous event rejected diagnostically
    assert inst.state == LifecycleState.IN_PROGRESS
    assert inst.last_evaluation_candle == ep0


# ---------------------------------------------------------------------------
# 9. Causal evidence; developing rejected by default
# ---------------------------------------------------------------------------


def test_developing_evidence_rejected_by_default():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    # Only developing observation for type a
    det = ScriptedDetector({ep: [_obs("a", ep, developing=True)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="dev")
    engine = _engine(market, setups=[setup], detectors=[det])
    sigs = engine.evaluate_once()
    assert sigs == []
    # May accept evaluation candle but expression does not qualify
    inst = engine.get_instance("dev")
    assert inst.state != LifecycleState.COMPLETED


# ---------------------------------------------------------------------------
# 10. Deterministic instance identity and signal deduplication
# ---------------------------------------------------------------------------


def test_deterministic_instance_identity_and_signal_id():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="id_test")
    engine = _engine(market, setups=[setup], detectors=[det])
    expected_inst = new_pending_instance(setup, instance_key="default")
    assert engine.get_instance("id_test").instance_id == expected_inst.instance_id

    sigs = engine.evaluate_once()
    assert len(sigs) == 1
    expected_sid = _deterministic_setup_signal_id(expected_inst.instance_id, ep)
    assert sigs[0].id == expected_sid

    # Same inputs → same signal id (determinism)
    market2 = FakeMarket(_candles_up_to(ep))
    det2 = ScriptedDetector({ep: [_obs("a", ep)]})
    engine2 = _engine(market2, setups=[setup], detectors=[det2])
    sigs2 = engine2.evaluate_once()
    assert sigs2[0].id == expected_sid


# ---------------------------------------------------------------------------
# 11. Repeated evaluations do not produce duplicate completion signals
# ---------------------------------------------------------------------------


def test_no_duplicate_completion_signals():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="dedup")
    engine = _engine(market, setups=[setup], detectors=[det])
    assert len(engine.evaluate_once()) == 1
    # Same candle again
    assert engine.evaluate_once() == []
    # Advance market time; terminal immutable
    market.set_candles(_candles_up_to(ep + 60))
    assert engine.evaluate_once() == []
    assert engine.get_instance("dedup").state == LifecycleState.COMPLETED


# ---------------------------------------------------------------------------
# 12. Multiple setups maintain independent lifecycle state
# ---------------------------------------------------------------------------


def test_multiple_setups_independent_state():
    ep0 = START
    ep1 = START + 60
    s_then = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="multi_then",
    )
    s_simple = _setup(ObservationRequirement(type="c"), setup_id="multi_simple")
    market = FakeMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {
            ep0: [_obs("a", ep0), _obs("c", ep0)],
            ep1: [_obs("b", ep1)],
        }
    )
    engine = _engine(market, setups=[s_then, s_simple], detectors=[det])

    sigs0 = engine.evaluate_once()
    # simple completes; then is in progress
    assert len(sigs0) == 1
    assert sigs0[0].condition_id == "multi_simple"
    assert engine.get_instance("multi_simple").state == LifecycleState.COMPLETED
    assert engine.get_instance("multi_then").state == LifecycleState.IN_PROGRESS

    market.set_candles(_candles_up_to(ep1))
    sigs1 = engine.evaluate_once()
    assert len(sigs1) == 1
    assert sigs1[0].condition_id == "multi_then"
    assert engine.get_instance("multi_then").state == LifecycleState.COMPLETED
    # simple stays completed, no second signal
    assert engine.get_instance("multi_simple").state == LifecycleState.COMPLETED


# ---------------------------------------------------------------------------
# 13. Disabled setups never activate or complete
# ---------------------------------------------------------------------------


def test_disabled_setup_never_activates():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="off", enabled=False)
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("off").state == LifecycleState.PENDING


# ---------------------------------------------------------------------------
# Shared store / no duplicate infrastructure
# ---------------------------------------------------------------------------


def test_reuses_single_observation_store():
    store = ObservationStore(max_size=50, max_age_seconds=None)
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="store")
    engine = ConditionEngine(
        market,
        [det],
        [],
        _ready_rm(),
        setups=[setup],
        observation_store=store,
        timeframe_seconds_by_name={TF: TF_SEC},
    )
    engine.evaluate_once()
    assert engine.store is store
    assert len(store.query(SYMBOL, TF)) >= 1


def test_setup_and_legacy_share_store_and_can_both_signal():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector(
        {ep: [_obs("a", ep), _obs("bullish_engulfing", ep)]}
    )
    setup = _setup(ObservationRequirement(type="a"), setup_id="both_setup")
    cond = ConditionDef(
        id="both_legacy",
        symbol=SYMBOL,
        timeframe=TF,
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = _engine(market, setups=[setup], conditions=[cond], detectors=[det])
    sigs = engine.evaluate_once()
    ids = {s.condition_id for s in sigs}
    assert ids == {"both_setup", "both_legacy"}
