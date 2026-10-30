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
    assert rec.cycle_index == 0
    assert rec.candle_epoch is not None
    assert rec.candle_ohlc is not None
    assert rec.effective_candle_epochs
    assert setup.setup_id in rec.setup_states
