"""Phase 8A: setup catalogue fixtures + consumer E2E via existing engine path.

Loads the three canonical fixtures through setups_from_config / parse_setup
and drives completion, invalidation, and expiry with injected observations.
Does not claim live detector production of catalogue types.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpab.conditions.lifecycle import LifecycleState
from cpab.conditions.validation import SetupValidationError, parse_setup
from cpab.config import setups_from_config
from cpab.core.engine import ConditionEngine
from cpab.detectors.base import PriceActionDetector
from cpab.market.readiness import ReadinessManager
from cpab.models import Observation, ReadinessState
from conftest import make_candle

SYMBOL = "R_100"
TF = "1m"
TF_SEC = 60
START = 1_700_000_000

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "phase8a"


class FakeMarket:
    def __init__(self, candles, *, granularity: int = TF_SEC, series_healthy: bool = True):
        self._candles = list(candles)
        self._granularity = granularity
        self._series_healthy = series_healthy

    def completed_candles(self, symbol, timeframe):
        return [c for c in self._candles if c.is_closed and c.symbol == symbol and c.timeframe == timeframe]

    def timeframe_seconds(self, timeframe: str):
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
        self._emitted_epochs.add(latest)
        return list(self._by_epoch.pop(latest, []))


def _obs(typ: str, candle_epoch: int, *, variant: str | None = None, confirmed_at: int | None = None, oid: str | None = None, developing: bool = False) -> Observation:
    conf = None if developing else (confirmed_at if confirmed_at is not None else candle_epoch)
    return Observation(id=oid or f"{typ}_{variant or 'nv'}_{candle_epoch}", type=typ, symbol=SYMBOL, timeframe=TF, candle_epoch=candle_epoch, event_time=candle_epoch, confirmed_at=conf, detected_at=float(candle_epoch), developing=developing, variant=variant)


def _candles_up_to(end_epoch: int, *, start: int = START, step: int = TF_SEC):
    out = []
    e = start
    while e <= end_epoch:
        out.append(make_candle(SYMBOL, TF, e, 100.0, 101.0, 99.0, 100.5, is_closed=True))
        e += step
    return out


def _ready_rm() -> ReadinessManager:
    rm = ReadinessManager()
    rm.register_series(SYMBOL, TF)
    rm.set_series(SYMBOL, TF, True)
    assert rm.state == ReadinessState.READY
    return rm


def _engine(market, *, setups=None, detectors=None, readiness=None) -> ConditionEngine:
    return ConditionEngine(market, detectors or [], [], readiness or _ready_rm(), setups=setups or [], timeframe_seconds_by_name={TF: TF_SEC})


def _load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _setup_from_fixture(name: str):
    return parse_setup(_load_fixture(name))


@pytest.mark.parametrize("fixture_id", ["phase8a_completion", "phase8a_invalidation", "phase8a_expiry"])
def test_fixture_loads_via_parse_setup(fixture_id):
    setup = _setup_from_fixture(fixture_id)
    assert setup.setup_id == fixture_id
    assert setup.schema_version == 1
    assert setup.symbol == SYMBOL
    assert setup.timeframe == TF
    assert setup.enabled is True
    assert setup.expression is not None


def test_fixtures_load_via_setups_from_config():
    payloads = [_load_fixture("phase8a_completion"), _load_fixture("phase8a_invalidation"), _load_fixture("phase8a_expiry")]
    setups = setups_from_config({"setups": payloads})
    assert [s.setup_id for s in setups] == ["phase8a_completion", "phase8a_invalidation", "phase8a_expiry"]
    by_id = {s.setup_id: s for s in setups}
    assert by_id["phase8a_completion"].invalidation is None
    assert by_id["phase8a_completion"].expiry is None
    assert by_id["phase8a_invalidation"].invalidation is not None
    assert by_id["phase8a_expiry"].expiry is not None
    assert by_id["phase8a_expiry"].expiry.max_age_candles == 1


def test_fixture_identifiers_and_terminal_intent_unchanged():
    assert (_setup_from_fixture("phase8a_completion")).setup_id == "phase8a_completion"
    inv = _setup_from_fixture("phase8a_invalidation")
    assert inv.invalidation is not None
    assert inv.invalidation.type == "structure_break"
    assert inv.invalidation.variant == "choch"
    exp = _setup_from_fixture("phase8a_expiry")
    assert exp.expiry is not None
    assert exp.expiry.max_age_candles == 1


def test_phase8a_completion_e2e():
    setup = _setup_from_fixture("phase8a_completion")
    ep0, ep1, ep2 = START, START + TF_SEC, START + 2 * TF_SEC
    det = ScriptedDetector({ep1: [_obs("structure_break", ep1, variant="bos")], ep2: [_obs("zone_interaction", ep2, variant="rejection")]})
    market = FakeMarket(_candles_up_to(ep0))
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_completion").state == LifecycleState.PENDING
    market.set_candles(_candles_up_to(ep1))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_completion").state == LifecycleState.IN_PROGRESS
    market.set_candles(_candles_up_to(ep2))
    sigs = engine.evaluate_once()
    assert len(sigs) == 1
    assert sigs[0].metadata.get("lifecycle_state") == LifecycleState.COMPLETED.value
    assert engine.get_instance("phase8a_completion").state == LifecycleState.COMPLETED
    snap = engine.last_monitoring_snapshot
    assert snap.cycle_recorded is True and snap.evaluation_blocked is False and snap.signal_count == 1
    assert snap.setups[0].setup_id == "phase8a_completion"
    assert snap.setups[0].state == LifecycleState.COMPLETED.value and snap.setups[0].transitioned is True
    market.set_candles(_candles_up_to(ep2 + TF_SEC))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_completion").state == LifecycleState.COMPLETED


def test_phase8a_invalidation_e2e():
    setup = _setup_from_fixture("phase8a_invalidation")
    ep0, ep1, ep2 = START, START + TF_SEC, START + 2 * TF_SEC
    det = ScriptedDetector({ep1: [_obs("structure_break", ep1, variant="bos")], ep2: [_obs("structure_break", ep2, variant="choch")]})
    market = FakeMarket(_candles_up_to(ep0))
    engine = _engine(market, setups=[setup], detectors=[det])
    market.set_candles(_candles_up_to(ep1))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_invalidation").state == LifecycleState.IN_PROGRESS
    market.set_candles(_candles_up_to(ep2))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_invalidation").state == LifecycleState.INVALIDATED
    snap = engine.last_monitoring_snapshot
    assert snap.signal_count == 0 and snap.setups[0].state == LifecycleState.INVALIDATED.value and snap.setups[0].transitioned is True
    market.set_candles(_candles_up_to(ep2 + TF_SEC))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_invalidation").state == LifecycleState.INVALIDATED


def test_phase8a_invalidation_precedence_over_completion():
    setup = _setup_from_fixture("phase8a_invalidation")
    ep0, ep1, ep2 = START, START + TF_SEC, START + 2 * TF_SEC
    det = ScriptedDetector({ep1: [_obs("structure_break", ep1, variant="bos")], ep2: [_obs("zone_interaction", ep2, variant="rejection"), _obs("structure_break", ep2, variant="choch", oid="choch_same")]})
    market = FakeMarket(_candles_up_to(ep0))
    engine = _engine(market, setups=[setup], detectors=[det])
    market.set_candles(_candles_up_to(ep1))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_invalidation").state == LifecycleState.IN_PROGRESS
    market.set_candles(_candles_up_to(ep2))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_invalidation").state == LifecycleState.INVALIDATED


def test_phase8a_expiry_e2e():
    """max_age_candles=1: first THEN step, no second step, next candle → EXPIRED."""
    setup = _setup_from_fixture("phase8a_expiry")
    assert setup.expiry is not None
    assert setup.expiry.max_age_candles == 1

    ep0 = START
    ep1 = START + TF_SEC  # first THEN step qualifies; becomes anchor
    ep2 = START + 2 * TF_SEC  # next contiguous completed candle; elapsed=1 >= 1

    det = ScriptedDetector(
        {
            ep1: [_obs("structure_break", ep1, variant="bos")],
            # deliberately no zone_interaction/rejection on any candle
        }
    )
    market = FakeMarket(_candles_up_to(ep0))
    engine = _engine(market, setups=[setup], detectors=[det])

    # Start THEN sequence (step 1 only)
    market.set_candles(_candles_up_to(ep1))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_expiry").state == LifecycleState.IN_PROGRESS

    # Second step never arrives; contiguous next candle expires under max_age_candles=1
    market.set_candles(_candles_up_to(ep2))
    sigs = engine.evaluate_once()
    assert sigs == []
    assert engine.get_instance("phase8a_expiry").state == LifecycleState.EXPIRED

    snap = engine.last_monitoring_snapshot
    assert snap.signal_count == 0
    assert snap.setups[0].state == LifecycleState.EXPIRED.value
    assert snap.setups[0].transitioned is True

    # Terminal immutability
    market.set_candles(_candles_up_to(ep2 + TF_SEC))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_expiry").state == LifecycleState.EXPIRED


def test_phase8a_unhealthy_series_does_not_advance():
    setup = _setup_from_fixture("phase8a_completion")
    ep0, ep1 = START, START + TF_SEC
    det = ScriptedDetector({ep1: [_obs("structure_break", ep1, variant="bos")]})
    market = FakeMarket(_candles_up_to(ep0), series_healthy=True)
    engine = _engine(market, setups=[setup], detectors=[det])
    market.set_candles(_candles_up_to(ep1))
    market.set_healthy(False)
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_completion").state == LifecycleState.PENDING


def test_phase8a_discontinuous_candle_does_not_advance():
    setup = _setup_from_fixture("phase8a_completion")
    ep0, ep1, ep3 = START, START + TF_SEC, START + 3 * TF_SEC
    det = ScriptedDetector({ep1: [_obs("structure_break", ep1, variant="bos")], ep3: [_obs("zone_interaction", ep3, variant="rejection")]})
    market = FakeMarket(_candles_up_to(ep0))
    engine = _engine(market, setups=[setup], detectors=[det])
    market.set_candles(_candles_up_to(ep1))
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_completion").state == LifecycleState.IN_PROGRESS
    gap_candles = _candles_up_to(ep1) + [make_candle(SYMBOL, TF, ep3, 100.0, 101.0, 99.0, 100.5, is_closed=True)]
    market.set_candles(gap_candles)
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_completion").state != LifecycleState.COMPLETED


def test_phase8a_developing_evidence_rejected_by_default():
    setup = _setup_from_fixture("phase8a_completion")
    ep1 = START + TF_SEC
    det = ScriptedDetector({ep1: [_obs("structure_break", ep1, variant="bos", developing=True)]})
    market = FakeMarket(_candles_up_to(ep1))
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_completion").state != LifecycleState.IN_PROGRESS


def test_phase8a_duplicate_observation_idempotent():
    setup = _setup_from_fixture("phase8a_completion")
    ep0, ep1, ep2 = START, START + TF_SEC, START + 2 * TF_SEC
    bos = _obs("structure_break", ep1, variant="bos", oid="bos_dup")
    rej = _obs("zone_interaction", ep2, variant="rejection", oid="rej_dup")
    det = ScriptedDetector({ep1: [bos], ep2: [rej, rej]})
    market = FakeMarket(_candles_up_to(ep0))
    engine = _engine(market, setups=[setup], detectors=[det])
    market.set_candles(_candles_up_to(ep1))
    assert engine.evaluate_once() == []
    market.set_candles(_candles_up_to(ep2))
    assert len(engine.evaluate_once()) == 1
    assert engine.get_instance("phase8a_completion").state == LifecycleState.COMPLETED
    assert engine.evaluate_once() == []


def test_phase8a_exact_type_and_variant_matching():
    setup = _setup_from_fixture("phase8a_completion")
    ep1 = START + TF_SEC
    det = ScriptedDetector({ep1: [_obs("sr_zone", ep1, variant="support"), _obs("structure_break", ep1, variant="choch")]})
    market = FakeMarket(_candles_up_to(ep1))
    engine = _engine(market, setups=[setup], detectors=[det])
    assert engine.evaluate_once() == []
    assert engine.get_instance("phase8a_completion").state == LifecycleState.PENDING


def test_main_py_still_only_wires_engulfing_detector():
    main_src = (Path(__file__).resolve().parents[1] / "src" / "cpab" / "main.py").read_text(encoding="utf-8")
    assert "EngulfingDetector()" in main_src
    assert "detectors = [EngulfingDetector()]" in main_src
    for forbidden in ("StructureBreakDetector", "ZoneInteractionDetector", "OrderBlockDetector", "auto_wire", "detectors_from_setups"):
        assert forbidden not in main_src
