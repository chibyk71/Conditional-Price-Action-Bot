"""Phase 8B A6: Phase 8A injected-fixture outcome tests (sole writers for phase8a-* keys).

F1/F2/F3 seed-based A6 outcomes live in tests/test_phase8b_seed_fixtures.py only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpab.conditions.lifecycle import LifecycleState
from cpab.conditions.validation import parse_setup
from cpab.detectors.base import PriceActionDetector
from cpab.models import Candle, Observation
from cpab.replay.driver import ReplayRunConfig, run_replay
from cpab.replay.evidence import record_a6
from cpab.replay.real_talib import real_talib_available as _real_talib_available
from conftest import make_candle

SYMBOL, TF, TF_SEC, START = "R_100", "1m", 60, 1_700_000_000
FIXTURES_8A = Path(__file__).resolve().parent / "fixtures" / "phase8a"


def _talib_ok() -> bool:
    return _real_talib_available()


def _mark(fid, ok, **extra):
    status = "real_env_passed" if ok and _talib_ok() else ("stub_only" if ok else "real_env_failed")
    record_a6(fid, status, extra=extra or None)


def _c(epoch, o, h, l, cl, timeframe=TF):
    return Candle(
        symbol=SYMBOL, timeframe=timeframe, epoch=epoch,
        open=o, high=h, low=l, close=cl, is_closed=True,
    )


def _neutrals(n: int):
    return [_c(START + i * TF_SEC, 100.0, 100.2, 99.8, 100.0) for i in range(n)]


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
    import time
    return Observation(
        id=f"{typ}_{variant or 'nv'}_{epoch}",
        type=typ,
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=epoch,
        detected_at=float(epoch),
        confirmed_at=epoch,
        variant=variant,
        detector_id="scripted",
        detector_config_id="scripted",
    )


def test_a6_phase8a_completion():
    fid = "phase8a-completion"
    try:
        setup = parse_setup(json.loads((FIXTURES_8A / "phase8a_completion.json").read_text()))
        candles = _neutrals(8)
        e0, e1 = candles[5].epoch, candles[6].epoch
        det = ScriptedDetector({
            e0: [_obs("structure_break", e0, "bos")],
            e1: [_obs("zone_interaction", e1, "rejection")],
        })
        r = run_replay(ReplayRunConfig(
            setups=[setup], detectors=[det], symbol=SYMBOL, timeframe=TF,
            timeframe_seconds=TF_SEC, candles=candles, arm_count=6,
        ))
        ok = (
            r.final_setup_states[setup.setup_id] == LifecycleState.COMPLETED.value
            and len(r.signals) == 1
        )
        # Deterministic signal id check
        det2 = ScriptedDetector({
            e0: [_obs("structure_break", e0, "bos")],
            e1: [_obs("zone_interaction", e1, "rejection")],
        })
        r2 = run_replay(ReplayRunConfig(
            setups=[setup], detectors=[det2], symbol=SYMBOL, timeframe=TF,
            timeframe_seconds=TF_SEC, candles=candles, arm_count=6,
        ))
        assert ok
        assert r.signals[0].id == r2.signals[0].id
        _mark(fid, True)
    except Exception as exc:
        _mark(fid, False, error=str(exc))
        raise


def test_a6_phase8a_invalidation():
    fid = "phase8a-invalidation"
    try:
        setup = parse_setup(json.loads((FIXTURES_8A / "phase8a_invalidation.json").read_text()))
        candles = _neutrals(8)
        e0 = candles[5].epoch
        r = run_replay(ReplayRunConfig(
            setups=[setup],
            detectors=[ScriptedDetector({e0: [_obs("structure_break", e0, "choch")]})],
            symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC,
            candles=candles, arm_count=6,
        ))
        ok = (
            r.final_setup_states[setup.setup_id] == LifecycleState.INVALIDATED.value
            and r.signals == []
        )
        assert ok
        _mark(fid, True)
    except Exception as exc:
        _mark(fid, False, error=str(exc))
        raise


def test_a6_phase8a_expiry():
    fid = "phase8a-expiry"
    try:
        setup = parse_setup(json.loads((FIXTURES_8A / "phase8a_expiry.json").read_text()))
        candles = _neutrals(10)
        e0 = candles[5].epoch
        r = run_replay(ReplayRunConfig(
            setups=[setup],
            detectors=[ScriptedDetector({e0: [_obs("structure_break", e0, "bos")]})],
            symbol=SYMBOL, timeframe=TF, timeframe_seconds=TF_SEC,
            candles=candles, arm_count=6,
        ))
        ok = (
            r.final_setup_states[setup.setup_id] == LifecycleState.EXPIRED.value
            and r.signals == []
        )
        assert ok
        _mark(fid, True)
    except Exception as exc:
        _mark(fid, False, error=str(exc))
        raise
