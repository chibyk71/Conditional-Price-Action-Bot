"""Phase 8B R7–R8 / A10–A11: effective-input parity and first-divergence."""

from __future__ import annotations

import json
from pathlib import Path

from cpab.conditions.validation import parse_setup
from cpab.detectors.base import PriceActionDetector
from cpab.models import Observation
from cpab.replay.driver import ReplayRunConfig, run_replay
from cpab.replay.parity import ENGINE_DIVERGENCE, INPUT_DIVERGENCE, compare_replay_runs
from conftest import make_candle

SYMBOL = "R_100"
TF = "1m"
TF_SEC = 60
START = 1_700_000_000
FIXTURES_8A = Path(__file__).resolve().parent / "fixtures" / "phase8a"


class ScriptedDetector(PriceActionDetector):
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


def _obs(typ: str, epoch: int, *, variant: str | None = None) -> Observation:
    return Observation(
        id=f"{typ}_{variant or 'nv'}_{epoch}",
        type=typ,
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=epoch,
        event_time=epoch,
        confirmed_at=epoch,
        detected_at=float(epoch),
        developing=False,
        variant=variant,
    )


def _candles(n: int) -> list:
    return [
        make_candle(SYMBOL, TF, START + i * TF_SEC, 100.0, 101.0, 99.0, 100.5, is_closed=True)
        for i in range(n)
    ]


def test_identical_runs_match():
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    candles = _candles(12)
    e0 = candles[5].epoch
    e1 = candles[6].epoch
    script = {
        e0: [_obs("structure_break", e0, variant="bos")],
        e1: [_obs("zone_interaction", e1, variant="rejection")],
    }

    def _run():
        return run_replay(
            ReplayRunConfig(
                setups=[setup],
                detectors=[ScriptedDetector(script)],
                symbol=SYMBOL,
                timeframe=TF,
                timeframe_seconds=TF_SEC,
                candles=candles,
                arm_count=6,
            )
        )

    a = _run()
    b = _run()
    cmp = compare_replay_runs(a, b)
    assert cmp.matching
    assert not cmp.input_divergence
    assert not cmp.engine_divergence
    assert cmp.first_divergence is None


def test_input_divergence_detected_before_engine():
    """Injected effective-input difference → INPUT_DIVERGENCE, not engine claim."""
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    candles = _candles(10)
    cfg = ReplayRunConfig(
        setups=[setup],
        detectors=[ScriptedDetector({})],
        symbol=SYMBOL,
        timeframe=TF,
        timeframe_seconds=TF_SEC,
        candles=candles,
        arm_count=5,
    )
    a = run_replay(cfg)
    # Run B with an unhealthy fault so data_ok / readiness differ.
    cfg_b = ReplayRunConfig(
        setups=[setup],
        detectors=[ScriptedDetector({})],
        symbol=SYMBOL,
        timeframe=TF,
        timeframe_seconds=TF_SEC,
        candles=candles,
        arm_count=5,
        fault_schedule={4: {"series_healthy": False}},
    )
    b = run_replay(cfg_b)
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.input_divergence
    assert cmp.first_divergence is not None
    assert cmp.first_divergence.kind == INPUT_DIVERGENCE
    assert "effective_input" in cmp.first_divergence.field


def test_first_divergence_early_input_before_cycle_count():
    """Early per-cycle input mismatch is reported before a later cycle-count mismatch."""
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    candles_a = _candles(10)
    candles_b = _candles(12)  # longer series -> more cycles
    cfg_a = ReplayRunConfig(
        setups=[setup],
        detectors=[ScriptedDetector({})],
        symbol=SYMBOL,
        timeframe=TF,
        timeframe_seconds=TF_SEC,
        candles=candles_a,
        arm_count=5,
        fault_schedule={5: {"series_healthy": False}},
    )
    cfg_b = ReplayRunConfig(
        setups=[setup],
        detectors=[ScriptedDetector({})],
        symbol=SYMBOL,
        timeframe=TF,
        timeframe_seconds=TF_SEC,
        candles=candles_b,
        arm_count=5,
        # no fault - diverges on data_ok at cycle 5 AND has more cycles
    )
    a = run_replay(cfg_a)
    b = run_replay(cfg_b)
    assert len(a.cycles) != len(b.cycles)
    cmp = compare_replay_runs(a, b)
    assert cmp.input_divergence
    assert cmp.first_divergence is not None
    assert cmp.first_divergence.field != "cycle_count", cmp.to_dict()
    assert cmp.first_divergence.cycle_index < min(len(a.cycles), len(b.cycles))
    assert "effective_input" in cmp.first_divergence.field


def test_first_divergence_pure_length_mismatch():
    """When all common cycles match, a pure length mismatch is still reported."""
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    short = _candles(8)
    long = _candles(12)
    a = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[ScriptedDetector({})],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=short,
            arm_count=5,
        )
    )
    b = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[ScriptedDetector({})],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=long,
            arm_count=5,
        )
    )
    assert len(a.cycles) < len(b.cycles)
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.input_divergence
    assert cmp.first_divergence is not None
    assert cmp.first_divergence.field == "cycle_count"
    assert cmp.first_divergence.expected == len(a.cycles)
    assert cmp.first_divergence.actual == len(b.cycles)

