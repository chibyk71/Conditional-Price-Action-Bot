"""Phase 8B A6: independent outcome fixtures via the replay driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpab.conditions.lifecycle import LifecycleState
from cpab.conditions.validation import parse_setup
from cpab.config import setups_from_config
from cpab.core.engine import ConditionEngine
from cpab.detectors.base import PriceActionDetector
from cpab.market.readiness import ReadinessManager
from cpab.models import Observation, ReadinessState
from cpab.replay.driver import ReplayRunConfig, run_replay
from conftest import make_candle

SYMBOL = "R_100"
TF = "1m"
TF_SEC = 60
START = 1_700_000_000
FIXTURES_8A = Path(__file__).resolve().parent / "fixtures" / "phase8a"


class ScriptedDetector(PriceActionDetector):
    """Emit pre-scripted observations keyed by evaluation candle epoch."""

    def __init__(self, by_epoch: dict[int, list[Observation]]):
        self._by_epoch = {k: list(v) for k, v in by_epoch.items()}
        self._emitted: set[int] = set()

    @property
    def name(self) -> str:
        return "scripted"

    def detect(self, candles: list) -> list[Observation]:
        if not candles:
            return []
        latest = candles[-1].epoch
        if latest in self._emitted:
            return []
        self._emitted.add(latest)
        return list(self._by_epoch.pop(latest, []))


def _obs(
    typ: str,
    epoch: int,
    *,
    variant: str | None = None,
    confirmed_at: int | None = None,
    oid: str | None = None,
) -> Observation:
    conf = confirmed_at if confirmed_at is not None else epoch
    return Observation(
        id=oid or f"{typ}_{variant or 'nv'}_{epoch}",
        type=typ,
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=epoch,
        event_time=epoch,
        confirmed_at=conf,
        detected_at=float(epoch),
        developing=False,
        variant=variant,
    )


def _candles(n: int, *, start: int = START, step: int = TF_SEC) -> list:
    out = []
    for i in range(n):
        e = start + i * step
        out.append(make_candle(SYMBOL, TF, e, 100.0, 101.0, 99.0, 100.5, is_closed=True))
    return out


def _ready() -> ReadinessManager:
    rm = ReadinessManager()
    rm.register_series(SYMBOL, TF)
    rm.set_series(SYMBOL, TF, True)
    assert rm.state == ReadinessState.READY
    return rm


def test_phase8a_completion_via_replay_independent_run():
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    # Arm with enough candles; inject bos then rejection on successive cycles.
    # Epochs: START, START+60, ...
    candles = _candles(10)
    e0 = candles[5].epoch
    e1 = candles[6].epoch
    det = ScriptedDetector(
        {
            e0: [_obs("structure_break", e0, variant="bos")],
            e1: [_obs("zone_interaction", e1, variant="rejection")],
        }
    )
    result = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[det],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=6,  # first eval at candle index 5
        )
    )
    assert result.final_setup_states[setup.setup_id] == LifecycleState.COMPLETED.value
    assert len(result.signals) == 1
    # Deterministic signal id across a second independent run.
    det2 = ScriptedDetector(
        {
            e0: [_obs("structure_break", e0, variant="bos")],
            e1: [_obs("zone_interaction", e1, variant="rejection")],
        }
    )
    result2 = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[det2],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=6,
        )
    )
    assert result.signals[0].id == result2.signals[0].id
    assert result.final_setup_states == result2.final_setup_states


def test_phase8a_invalidation_via_replay():
    raw = json.loads((FIXTURES_8A / "phase8a_invalidation.json").read_text())
    setup = parse_setup(raw)
    candles = _candles(8)
    e0 = candles[5].epoch
    # Invalidation expression uses structure_break/choch — emit that.
    det = ScriptedDetector(
        {e0: [_obs("structure_break", e0, variant="choch")]}
    )
    result = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[det],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=6,
        )
    )
    assert result.final_setup_states[setup.setup_id] == LifecycleState.INVALIDATED.value
    assert result.signals == []


def test_phase8a_expiry_via_replay():
    raw = json.loads((FIXTURES_8A / "phase8a_expiry.json").read_text())
    setup = parse_setup(raw)
    # max_age_candles=1: after first accepted eval, next cycle without progress expires.
    candles = _candles(10)
    e0 = candles[5].epoch
    # Emit only the first THEN step so the second never arrives.
    det = ScriptedDetector(
        {e0: [_obs("structure_break", e0, variant="bos")]}
    )
    result = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[det],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=6,
        )
    )
    assert result.final_setup_states[setup.setup_id] == LifecycleState.EXPIRED.value
    assert result.signals == []


def test_replay_fresh_state_isolation():
    """A8: interleaved independent runs do not leak state."""
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    candles = _candles(10)
    e0 = candles[5].epoch
    e1 = candles[6].epoch

    def _run():
        det = ScriptedDetector(
            {
                e0: [_obs("structure_break", e0, variant="bos")],
                e1: [_obs("zone_interaction", e1, variant="rejection")],
            }
        )
        return run_replay(
            ReplayRunConfig(
                setups=[setup],
                detectors=[det],
                symbol=SYMBOL,
                timeframe=TF,
                timeframe_seconds=TF_SEC,
                candles=candles,
                arm_count=6,
            )
        )

    r1 = _run()
    r2 = _run()
    assert r1.final_setup_states == r2.final_setup_states
    assert [s.id for s in r1.signals] == [s.id for s in r2.signals]


def test_replay_records_per_cycle_fields():
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    candles = _candles(8)
    det = ScriptedDetector({})
    result = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[det],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
        )
    )
    assert result.cycles
    rec = result.cycles[0]
    assert rec.cycle_index == 4  # arm_count=5 → first cycle is index 4
    assert rec.candle_epoch is not None
    assert rec.candle_ohlc is not None
    assert rec.effective_candle_epochs
    assert setup.setup_id in rec.setup_states


# ---------------------------------------------------------------------------
# A6 F2 delayed-confirmation timelines (frozen indices)
# ---------------------------------------------------------------------------

F2_START = 1_699_992_000  # contract: epoch = 1699992000 + 900*i
F2_STEP = 900
F2_ARM = 60


def _f2_epoch(idx: int) -> int:
    return F2_START + idx * F2_STEP


def _f2_candles(n: int) -> list:
    return [
        make_candle(SYMBOL, "15m", _f2_epoch(i), 100.0, 101.0, 99.0, 100.5, is_closed=True)
        for i in range(n)
    ]


def _f2_obs(typ: str, event_idx: int, *, variant: str | None = None, confirmed_idx: int | None = None) -> Observation:
    event_ep = _f2_epoch(event_idx)
    conf_ep = _f2_epoch(confirmed_idx) if confirmed_idx is not None else event_ep
    return Observation(
        id=f"{typ}_{variant or 'nv'}_{event_ep}_{conf_ep}",
        type=typ,
        symbol=SYMBOL,
        timeframe="15m",
        candle_epoch=event_ep,
        event_time=event_ep,
        confirmed_at=conf_ep,
        detected_at=float(conf_ep),
        developing=False,
        variant=variant,
    )


def _f2_setup_completion():
    from cpab.conditions.expression import AndExpression, OrExpression, ObservationRequirement
    from cpab.conditions.schema import ExpiryConfig, SCHEMA_VERSION_V1, SetupDef
    return SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id="f2_bullish_higher_low_trigger",
        symbol=SYMBOL,
        timeframe="15m",
        enabled=True,
        expression=AndExpression(
            children=(
                ObservationRequirement(type="market_structure", variant="hl", min_count=1),
                OrExpression(
                    children=(
                        ObservationRequirement(type="bullish_engulfing", min_count=1),
                        ObservationRequirement(type="bullish_impulse", variant="atr_normalized_candle_v1", min_count=1),
                    )
                ),
            )
        ),
        invalidation=ObservationRequirement(type="market_structure", variant="ll", min_count=1),
        expiry=ExpiryConfig(max_age_candles=60),
        name="F2 completion",
    )


def test_f2_completion_delayed_confirmation_at_76():
    """Frozen: event at 75, confirm at 76, COMPLETED at 76; PENDING at 75."""
    setup = _f2_setup_completion()
    candles = _f2_candles(90)
    # hl confirms at 76; engulfing on same candle 75 (available earlier)
    by_epoch = {
        _f2_epoch(64): [_f2_obs("bullish_impulse", 64, variant="atr_normalized_candle_v1")],
        _f2_epoch(75): [_f2_obs("bullish_engulfing", 75)],
        # market_structure/hl: event 75, confirmed_at 76 — only consumable at 76
        _f2_epoch(76): [
            _f2_obs("market_structure", 75, variant="hl", confirmed_idx=76),
        ],
    }
    det = ScriptedDetector(by_epoch)
    # Override detector to use 15m symbol path — ScriptedDetector ignores tf of candles
    result = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[det],
            symbol=SYMBOL,
            timeframe="15m",
            timeframe_seconds=F2_STEP,
            candles=candles,
            arm_count=F2_ARM,
        )
    )
    # At cycle 75 must still be PENDING (hl not yet confirmed)
    state_75 = None
    state_76 = None
    for rec in result.cycles:
        if rec.cycle_index == 75:
            state_75 = rec.setup_states.get(setup.setup_id)
        if rec.cycle_index == 76:
            state_76 = rec.setup_states.get(setup.setup_id)
    assert state_75 == LifecycleState.PENDING.value, f"expected PENDING at 75, got {state_75}"
    assert state_76 == LifecycleState.COMPLETED.value, f"expected COMPLETED at 76, got {state_76}"
    assert len(result.signals) == 1
    # Impulse at 64 must not complete alone
    for rec in result.cycles:
        if rec.cycle_index == 64:
            assert rec.setup_states.get(setup.setup_id) != LifecycleState.COMPLETED.value


def test_f2_invalidation_delayed_confirmation_at_93():
    """Frozen: ll event 92, confirm 93, INVALIDATED at 93; not at 92."""
    setup = _f2_setup_completion()
    candles = _f2_candles(100)
    by_epoch = {
        _f2_epoch(92): [
            _f2_obs("bullish_engulfing", 92),
            _f2_obs("bullish_impulse", 92, variant="atr_normalized_candle_v1"),
        ],
        _f2_epoch(93): [
            _f2_obs("market_structure", 92, variant="ll", confirmed_idx=93),
        ],
    }
    det = ScriptedDetector(by_epoch)
    result = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[det],
            symbol=SYMBOL,
            timeframe="15m",
            timeframe_seconds=F2_STEP,
            candles=candles,
            arm_count=F2_ARM,
        )
    )
    state_92 = state_93 = None
    for rec in result.cycles:
        if rec.cycle_index == 92:
            state_92 = rec.setup_states.get(setup.setup_id)
        if rec.cycle_index == 93:
            state_93 = rec.setup_states.get(setup.setup_id)
    assert state_92 != LifecycleState.INVALIDATED.value, f"must not invalidate at 92, got {state_92}"
    assert state_93 == LifecycleState.INVALIDATED.value, f"expected INVALIDATED at 93, got {state_93}"
    assert result.signals == []
