"""Phase 7E: end-to-end observational setup monitoring snapshot."""

from __future__ import annotations

from typing import Optional

import pytest

from cpab.conditions.expression import (
    ObservationRequirement,
    ThenExpression,
    ThenStep,
)
from cpab.conditions.lifecycle import LifecycleState
from cpab.conditions.schema import SCHEMA_VERSION_V1, SetupDef
from cpab.core.engine import ConditionEngine
from cpab.core.monitoring import MonitoringSnapshot, empty_monitoring_snapshot
from cpab.detectors.base import PriceActionDetector
from cpab.market.readiness import ReadinessManager
from cpab.models import ConditionDef, Observation, ReadinessState
from conftest import make_candle

SYMBOL = "R_100"
TF = "1m"
TF_SEC = 60
START = 1_700_000_000


class FakeMarket:
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
    def __init__(self, by_epoch: dict[int, list[Observation]]):
        self._by_epoch = {k: list(v) for k, v in by_epoch.items()}

    @property
    def name(self) -> str:
        return "scripted"

    def detect(self, candles):
        if not candles:
            return []
        ep = candles[-1].epoch
        return list(self._by_epoch.pop(ep, []))


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
) -> SetupDef:
    return SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id=setup_id,
        symbol=SYMBOL,
        timeframe=TF,
        enabled=enabled,
        expression=expression,
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


def test_empty_snapshot_before_any_cycle():
    market = FakeMarket(_candles_up_to(START))
    engine = _engine(market, setups=[_setup(ObservationRequirement(type="a"))])
    snap = engine.last_monitoring_snapshot
    assert isinstance(snap, MonitoringSnapshot)
    assert snap.cycle_recorded is False
    assert snap.evaluation_blocked is False
    assert snap.signal_count == 0


def test_readiness_blocked_cycle_records_snapshot():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="blocked")
    rm = ReadinessManager()
    rm.register_series(SYMBOL, TF)
    engine = _engine(market, setups=[setup], detectors=[det], readiness=rm)

    assert engine.evaluate_once() == []
    snap = engine.last_monitoring_snapshot
    assert snap.cycle_recorded is True
    assert snap.evaluation_blocked is True
    assert snap.block_reason is not None
    assert snap.readiness_state != ReadinessState.READY.value
    assert snap.signal_count == 0
    assert snap.detector_emitted_observation_ids == ()
    assert snap.newly_accepted_observation_ids == ()
    assert len(snap.setups) == 1
    assert snap.setups[0].setup_id == "blocked"
    assert snap.setups[0].state == LifecycleState.PENDING.value
    assert snap.setups[0].transitioned is False
    assert engine.get_instance("blocked").state == LifecycleState.PENDING


def test_healthy_completion_snapshot_exposes_transition_and_signal():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="simple_a")
    engine = _engine(market, setups=[setup], detectors=[det])

    sigs = engine.evaluate_once()
    assert len(sigs) == 1
    snap = engine.last_monitoring_snapshot
    assert snap.cycle_recorded is True
    assert snap.evaluation_blocked is False
    assert snap.block_reason is None
    assert snap.readiness_state == ReadinessState.READY.value
    assert snap.signal_count == 1
    assert snap.signal_ids == (sigs[0].id,)
    assert snap.newly_accepted_observation_ids == snap.detector_emitted_observation_ids
    assert len(snap.series) == 1
    assert snap.series[0].data_ok is True
    assert snap.series[0].evaluation_candle_epoch == ep
    assert len(snap.setups) == 1
    sv = snap.setups[0]
    assert sv.setup_id == "simple_a"
    assert sv.state == LifecycleState.COMPLETED.value
    assert sv.transitioned is True
    assert sv.diagnostic_only is False
    assert sv.instance_id
    assert sv.config_id
    assert sv.last_evaluation_candle == ep


def test_unhealthy_series_snapshot_is_diagnostic():
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
    assert engine.evaluate_once() == []
    snap = engine.last_monitoring_snapshot
    assert snap.cycle_recorded is True
    assert snap.evaluation_blocked is False
    assert snap.signal_count == 0
    assert snap.detector_emitted_observation_ids == ()
    assert snap.newly_accepted_observation_ids == ()
    assert len(snap.series) == 1
    assert snap.series[0].data_ok is False
    assert snap.series[0].diagnostics
    assert snap.series[0].evaluation_candle_epoch == ep1
    sv = snap.setups[0]
    assert sv.setup_id == "unhealthy"
    assert sv.state == LifecycleState.IN_PROGRESS.value
    assert sv.diagnostic_only is True
    assert sv.transitioned is False
    assert engine.get_instance("unhealthy").last_evaluation_candle == ep0


def test_duplicate_emission_not_newly_accepted():
    ep0 = START
    ep1 = START + 60
    oid = "shared_obs"
    market = FakeMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {
            ep0: [_obs("a", ep0, oid=oid)],
            ep1: [_obs("a", ep1, oid=oid)],
        }
    )
    setup = _setup(ObservationRequirement(type="a"), setup_id="dup")
    engine = _engine(market, setups=[setup], detectors=[det])
    engine.evaluate_once()
    snap0 = engine.last_monitoring_snapshot
    assert oid in snap0.detector_emitted_observation_ids
    assert oid in snap0.newly_accepted_observation_ids

    market.set_candles(_candles_up_to(ep1))
    engine.evaluate_once()
    snap1 = engine.last_monitoring_snapshot
    assert oid in snap1.detector_emitted_observation_ids
    assert oid not in snap1.newly_accepted_observation_ids


def test_monitoring_is_observational_no_state_mutation_on_read():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector({ep: [_obs("a", ep)]})
    setup = _setup(ObservationRequirement(type="a"), setup_id="obs_only")
    engine = _engine(market, setups=[setup], detectors=[det])
    engine.evaluate_once()
    inst_before = engine.get_instance("obs_only")
    state_before = inst_before.state
    last_eval_before = inst_before.last_evaluation_candle
    store_len_before = len(engine.store)

    snap = engine.last_monitoring_snapshot
    assert snap.cycle_recorded is True
    assert engine.get_instance("obs_only").state == state_before
    assert engine.get_instance("obs_only").last_evaluation_candle == last_eval_before
    assert len(engine.store) == store_len_before


def test_legacy_condition_signal_appears_in_snapshot():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
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
    snap = engine.last_monitoring_snapshot
    assert snap.signal_count == 1
    assert snap.signal_ids == (sigs[0].id,)
    assert snap.evaluation_blocked is False


def test_multiple_setups_independent_in_snapshot():
    ep = START + 60
    market = FakeMarket(_candles_up_to(ep))
    det = ScriptedDetector(
        {ep: [_obs("a", ep), _obs("b", ep, oid="b_only")]}
    )
    s_a = _setup(ObservationRequirement(type="a"), setup_id="sa")
    s_b = _setup(ObservationRequirement(type="b"), setup_id="sb")
    engine = _engine(market, setups=[s_a, s_b], detectors=[det])
    sigs = engine.evaluate_once()
    assert len(sigs) == 2
    snap = engine.last_monitoring_snapshot
    assert snap.signal_count == 2
    ids = {sv.setup_id: sv for sv in snap.setups}
    assert set(ids) == {"sa", "sb"}
    assert ids["sa"].state == LifecycleState.COMPLETED.value
    assert ids["sb"].state == LifecycleState.COMPLETED.value
    assert ids["sa"].transitioned is True
    assert ids["sb"].transitioned is True


def test_empty_monitoring_snapshot_helper():
    s = empty_monitoring_snapshot(readiness_state="READY", readiness_reason="ok")
    assert s.cycle_recorded is False
    assert s.readiness_state == "READY"
    assert s.readiness_reason == "ok"
    assert s.signal_count == 0


def test_snapshot_overwritten_each_cycle():
    ep0 = START
    ep1 = START + 60
    market = FakeMarket(_candles_up_to(ep0))
    det = ScriptedDetector(
        {
            ep0: [_obs("a", ep0)],
            ep1: [_obs("b", ep1)],
        }
    )
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        setup_id="then_snap",
    )
    engine = _engine(market, setups=[setup], detectors=[det])
    engine.evaluate_once()
    snap0 = engine.last_monitoring_snapshot
    assert snap0.setups[0].state == LifecycleState.IN_PROGRESS.value
    assert snap0.signal_count == 0

    market.set_candles(_candles_up_to(ep1))
    sigs = engine.evaluate_once()
    assert len(sigs) == 1
    snap1 = engine.last_monitoring_snapshot
    assert snap1 is not snap0
    assert snap1.setups[0].state == LifecycleState.COMPLETED.value
    assert snap1.signal_count == 1
    assert snap1.series[0].evaluation_candle_epoch == ep1
