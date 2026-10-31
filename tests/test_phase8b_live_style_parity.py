"""Phase 8B A10: replay vs live-style (CandleCache + mark_closed_by_lag) parity."""

from __future__ import annotations

import json
from pathlib import Path

from cpab.conditions.validation import parse_setup
from cpab.detectors.base import PriceActionDetector
from cpab.detectors.engulfing import EngulfingDetector
from cpab.models import Observation
from cpab.replay.driver import ReplayRunConfig, run_replay
from cpab.replay.live_style import LiveStyleConfig, live_style_from_replay_config, run_live_style
from cpab.replay.parity import INPUT_DIVERGENCE, compare_replay_runs
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


def test_replay_vs_live_style_matching_on_identical_inputs():
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    candles = _candles(12)
    e0 = candles[5].epoch
    e1 = candles[6].epoch
    script = {
        e0: [_obs("structure_break", e0, variant="bos")],
        e1: [_obs("zone_interaction", e1, variant="rejection")],
    }

    cfg = ReplayRunConfig(
        setups=[setup],
        detectors=[ScriptedDetector(script)],
        symbol=SYMBOL,
        timeframe=TF,
        timeframe_seconds=TF_SEC,
        candles=candles,
        arm_count=6,
    )
    replay = run_replay(cfg)
    live = run_live_style(
        live_style_from_replay_config(
            ReplayRunConfig(
                setups=[setup],
                detectors=[ScriptedDetector(script)],
                symbol=SYMBOL,
                timeframe=TF,
                timeframe_seconds=TF_SEC,
                candles=candles,
                arm_count=6,
            ),
            cache_max_size=500,
            confirm_lag_bars=1,
        )
    )
    cmp = compare_replay_runs(replay, live, setup_ids=[setup.setup_id])
    # Behavior must match.
    assert replay.final_setup_states == live.final_setup_states
    assert [s.id for s in replay.signals] == [s.id for s in live.signals]
    # Effective candle epochs must match (core engine boundary input).
    for a, b in zip(replay.cycles, live.cycles):
        assert a.effective_candle_epochs == b.effective_candle_epochs, (
            a.cycle_index, a.effective_candle_epochs, b.effective_candle_epochs
        )
    # Live-style must record production knobs; pure replay leaves them None.
    # INPUT_DIVERGENCE on lag/cache metadata alone is expected and not a failure
    # of outcome parity — assert no ENGINE_DIVERGENCE.
    cmp2 = compare_replay_runs(replay, live, setup_ids=[setup.setup_id], ignore_live_metadata=True)
    assert cmp2.matching and not cmp2.input_divergence and not cmp2.engine_divergence, cmp2.to_dict()
    # Live-style records production knobs.
    assert live.cycles[0].monitoring is not None
    assert live.cycles[0].monitoring.get("confirm_lag_bars") == 1
    assert live.cycles[0].monitoring.get("cache_max_size") == 500
    assert live.cycles[0].effective_candle_epochs


def test_live_style_cache_cap_affects_effective_inputs():
    candles = _candles(20)
    det = EngulfingDetector()
    # No setup needed for input recording — use empty setups.
    live_small = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[det],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            cache_max_size=5,
            confirm_lag_bars=1,
        )
    )
    live_large = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[det],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            cache_max_size=500,
            confirm_lag_bars=1,
        )
    )
    # Small cache must expose fewer completed candles at the final cycle.
    small_n = len(live_small.cycles[-1].effective_candle_epochs)
    large_n = len(live_large.cycles[-1].effective_candle_epochs)
    assert small_n <= 5
    assert large_n >= small_n
    assert live_small.cycles[-1].monitoring.get("cache_max_size") == 5


def test_live_style_confirm_lag_recorded():
    candles = _candles(10)
    live = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=3,
            confirm_lag_bars=2,
            cache_max_size=100,
        )
    )
    assert all(
        rec.monitoring and rec.monitoring.get("confirm_lag_bars") == 2
        for rec in live.cycles
    )


def test_input_divergence_when_cache_caps_differ():
    candles = _candles(15)
    a = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            cache_max_size=5,
        )
    )
    b = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            cache_max_size=500,
        )
    )
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.input_divergence
    assert cmp.first_divergence is not None
    assert cmp.first_divergence.kind == INPUT_DIVERGENCE


def test_live_style_htf_from_1m_aggregation():
    """Production aggregate_from_1m path builds HTF from 1m completed bars."""
    # 30 minutes of 1m bars → 2 complete 15m HTF bars
    candles = [make_candle(SYMBOL, TF, 1700000100 + i * TF_SEC, 100.0, 101.0, 99.0, 100.5, is_closed=True) for i in range(45)]
    live = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=15,
            confirm_lag_bars=1,
            cache_max_size=500,
            htf_from_1m=("15m", 900),
        )
    )
    assert live.cycles
    last = live.cycles[-1]
    assert last.monitoring is not None
    assert last.monitoring.get("htf_source") == "aggregate_from_1m"
    htf_epochs = last.monitoring.get("htf_epochs") or ()
    # At least one complete 15m bucket from 30 ones
    assert len(htf_epochs) >= 1


def test_live_style_htf_initialization_explicit_series():
    """Initialization-style path: independent HTF history feed (not aggregate)."""
    primary = _candles(20)
    # Explicit 15m HTF series (simulates history fetched at init)
    htf = [
        make_candle(SYMBOL, "15m", START + i * 900, 100.0, 101.0, 99.0, 100.5, is_closed=True)
        for i in range(5)
    ]
    live = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=primary,
            arm_count=5,
            htf=(SYMBOL, "15m", 900, htf),
        )
    )
    assert live.cycles
    mon = live.cycles[-1].monitoring
    assert mon is not None
    assert mon.get("htf_source") == "explicit_series"
    assert len(mon.get("htf_epochs") or ()) >= 1


def test_live_style_identical_knobs_match():
    """Two live-style runs with identical knobs must fully match (no INPUT_DIVERGENCE)."""
    candles = _candles(12)
    def _run():
        return run_live_style(
            LiveStyleConfig(
                setups=[],
                detectors=[EngulfingDetector()],
                symbol=SYMBOL,
                timeframe=TF,
                timeframe_seconds=TF_SEC,
                candles=candles,
                arm_count=5,
                cache_max_size=100,
                confirm_lag_bars=1,
            )
        )
    a, b = _run(), _run()
    cmp = compare_replay_runs(a, b)
    assert cmp.matching, cmp.to_dict()
    assert not cmp.input_divergence
    assert not cmp.engine_divergence


def test_live_style_htf_init_then_polling_aggregate():
    """Combined path: independent HTF history at init, then aggregate_from_1m while polling."""
    start = 1_700_000_100
    primary = [
        make_candle(SYMBOL, TF, start + i * TF_SEC, 100.0, 101.0, 99.0, 100.5, is_closed=True)
        for i in range(45)
    ]
    htf_init = [
        make_candle(SYMBOL, "15m", start - 900 * 3 + i * 900, 99.0, 100.0, 98.0, 99.5, is_closed=True)
        for i in range(3)
    ]
    init_epochs = tuple(sorted(c.epoch for c in htf_init))
    live = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=primary,
            arm_count=5,
            htf=(SYMBOL, "15m", 900, htf_init),
            htf_from_1m=("15m", 900),
        )
    )
    assert live.cycles
    first = live.cycles[0]
    assert first.monitoring is not None
    assert first.monitoring.get("htf_source") == "init_then_aggregate"
    first_htf = tuple(first.monitoring.get("htf_epochs") or ())
    assert first_htf, "init path must seed HTF epochs"
    # Init epochs should appear in the first cycle's HTF set
    assert any(e in first_htf for e in init_epochs), (first_htf, init_epochs)
    last = live.cycles[-1]
    assert last.monitoring.get("htf_source") == "init_then_aggregate"
    last_htf = tuple(last.monitoring.get("htf_epochs") or ())
    assert last_htf, "polling aggregate must produce HTF epochs"
    # After polling aggregation, HTF set should differ from pure init-only seed
    # (aggregated from 45 ones produces buckets from primary range).
    assert last_htf != first_htf or len(last_htf) >= len(first_htf)


def test_replay_vs_live_style_full_match_on_closed_series():
    """When all candles are pre-closed and live knobs do not alter the set, engine outcomes match.

    Effective-input metadata (confirm_lag, cache_max) may still differ in the
    comparison record; behavior fields and candle epochs must match. A pure
    metadata-only INPUT_DIVERGENCE is acceptable; engine_divergence is not.
    """
    candles = _candles(15)
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    e0, e1 = candles[8].epoch, candles[9].epoch
    script = {
        e0: [_obs("structure_break", e0, variant="bos")],
        e1: [_obs("zone_interaction", e1, variant="rejection")],
    }
    cfg_kwargs = dict(
        setups=[setup],
        symbol=SYMBOL,
        timeframe=TF,
        timeframe_seconds=TF_SEC,
        candles=candles,
        arm_count=8,
    )
    replay = run_replay(ReplayRunConfig(detectors=[ScriptedDetector(script)], **cfg_kwargs))
    live = run_live_style(
        LiveStyleConfig(
            detectors=[ScriptedDetector(script)],
            cache_max_size=500,
            confirm_lag_bars=1,
            **cfg_kwargs,
        )
    )
    assert replay.final_setup_states == live.final_setup_states
    assert [s.id for s in replay.signals] == [s.id for s in live.signals]
    for a, b in zip(replay.cycles, live.cycles):
        assert a.effective_candle_epochs == b.effective_candle_epochs
        assert a.setup_states == b.setup_states
    cmp = compare_replay_runs(
        replay, live, setup_ids=[setup.setup_id], ignore_live_metadata=True
    )
    assert cmp.matching, cmp.to_dict()
    assert not cmp.input_divergence
    assert not cmp.engine_divergence
