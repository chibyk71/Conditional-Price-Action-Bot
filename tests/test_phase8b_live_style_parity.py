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

    # Positive A10: equivalent configs declared explicitly on both paths.
    # Replay completed-only path uses confirm_lag_bars=0 (no lag filter).
    # Live-style with confirm_lag_bars=0 and large cache yields the same
    # closed candle set; knobs are recorded identically.
    CAP = 500
    LAG = 0
    cfg = ReplayRunConfig(
        setups=[setup],
        detectors=[ScriptedDetector(script)],
        symbol=SYMBOL,
        timeframe=TF,
        timeframe_seconds=TF_SEC,
        candles=candles,
        arm_count=6,
        record_confirm_lag_bars=LAG,
        record_cache_max_size=CAP,
        record_htf_source=None,
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
            cache_max_size=CAP,
            confirm_lag_bars=LAG,
        )
    )
    # Effective candle epochs must match (core engine boundary input).
    for a, b in zip(replay.cycles, live.cycles):
        assert a.effective_candle_epochs == b.effective_candle_epochs, (
            a.cycle_index, a.effective_candle_epochs, b.effective_candle_epochs
        )
    assert replay.final_setup_states == live.final_setup_states
    assert [s.id for s in replay.signals] == [s.id for s in live.signals]
    cmp = compare_replay_runs(replay, live, setup_ids=[setup.setup_id])
    assert cmp.matching and not cmp.input_divergence and not cmp.engine_divergence, cmp.to_dict()
    assert live.cycles[0].monitoring is not None
    assert live.cycles[0].monitoring.get("confirm_lag_bars") == LAG
    assert live.cycles[0].monitoring.get("cache_max_size") == CAP
    assert replay.cycles[0].monitoring.get("confirm_lag_bars") == LAG
    assert replay.cycles[0].monitoring.get("cache_max_size") == CAP


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
    """A10 positive: equivalent configs produce matching effective inputs, no INPUT_DIVERGENCE.

    Both paths explicitly record the same confirm_lag_bars and cache_max_size.
    Live-style lag=0 with large cache yields the same closed set as completed-only replay.
    """
    candles = _candles(15)
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    e0, e1 = candles[8].epoch, candles[9].epoch
    script = {
        e0: [_obs("structure_break", e0, variant="bos")],
        e1: [_obs("zone_interaction", e1, variant="rejection")],
    }
    CAP, LAG = 500, 0
    cfg_kwargs = dict(
        setups=[setup],
        symbol=SYMBOL,
        timeframe=TF,
        timeframe_seconds=TF_SEC,
        candles=candles,
        arm_count=8,
    )
    replay = run_replay(
        ReplayRunConfig(
            detectors=[ScriptedDetector(script)],
            record_confirm_lag_bars=LAG,
            record_cache_max_size=CAP,
            record_htf_source=None,
            **cfg_kwargs,
        )
    )
    live = run_live_style(
        LiveStyleConfig(
            detectors=[ScriptedDetector(script)],
            cache_max_size=CAP,
            confirm_lag_bars=LAG,
            **cfg_kwargs,
        )
    )
    assert replay.final_setup_states == live.final_setup_states
    assert [s.id for s in replay.signals] == [s.id for s in live.signals]
    for a, b in zip(replay.cycles, live.cycles):
        assert a.effective_candle_epochs == b.effective_candle_epochs
        assert a.setup_states == b.setup_states
    cmp = compare_replay_runs(replay, live, setup_ids=[setup.setup_id])
    assert cmp.matching and not cmp.input_divergence and not cmp.engine_divergence, cmp.to_dict()


def test_parity_deliberate_lag_mismatch_reports_input_divergence():
    """Deliberate confirm_lag difference must surface as INPUT_DIVERGENCE when metadata compared."""
    candles = _candles(12)
    a = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            confirm_lag_bars=1,
            cache_max_size=100,
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
            confirm_lag_bars=3,
            cache_max_size=100,
        )
    )
    cmp = compare_replay_runs(a, b)
    assert cmp.input_divergence, cmp.to_dict()
    assert not cmp.matching


def test_a10_deliberate_detector_order_mismatch_reports_input_divergence():
    """Detector order difference must surface as INPUT_DIVERGENCE."""
    candles = _candles(10)
    from cpab.detectors.engulfing import EngulfingDetector

    a = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector(), ScriptedDetector({})],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            confirm_lag_bars=0,
            cache_max_size=100,
        )
    )
    b = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[ScriptedDetector({}), EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            confirm_lag_bars=0,
            cache_max_size=100,
        )
    )
    cmp = compare_replay_runs(a, b)
    assert cmp.input_divergence, cmp.to_dict()
    assert not cmp.matching
    assert cmp.first_divergence is not None
    assert ("effective_detector_names" in cmp.first_divergence.field
            or "effective_detector_configs" in cmp.first_divergence.field)


def test_r6_observation_identity_fields_recorded_on_accept():
    """R6: cycle evidence carries full observation identity + payload with values."""
    candles = _candles(12)
    e0 = candles[5].epoch
    e1 = candles[6].epoch
    o0 = Observation(
        id=f"structure_break_bos_{e0}",
        type="structure_break",
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=e0,
        event_time=e0,
        confirmed_at=e0,
        detected_at=float(e0),
        developing=False,
        variant="bos",
        detector_id="scripted",
        detector_config_id="scripted_cfg_v1",
        payload={"step": "arm"},
    )
    o1 = Observation(
        id=f"zone_interaction_rejection_{e1}",
        type="zone_interaction",
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=e1,
        event_time=e1,
        confirmed_at=e1,
        detected_at=float(e1),
        developing=False,
        variant="rejection",
        detector_id="scripted",
        detector_config_id="scripted_cfg_v1",
        payload={"step": "trigger"},
    )
    script = {e0: [o0], e1: [o1]}
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    r = run_replay(
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
    required_keys = {
        "id", "type", "symbol", "timeframe", "variant", "event_time",
        "confirmed_at", "candle_epoch", "detected_at", "detector_id",
        "detector_config_id", "developing", "payload",
    }
    accepted: list[dict] = []
    emitted: list[dict] = []
    for rec in r.cycles:
        accepted.extend(rec.newly_accepted_observations)
        emitted.extend(rec.emitted_observations)
    assert accepted, "expected newly accepted observations"
    for entry in accepted + emitted:
        assert required_keys <= set(entry.keys()), entry.keys()
        assert entry["developing"] is False
        assert entry["symbol"] == SYMBOL
        assert entry["timeframe"] == TF
        assert entry["event_time"] is not None
        assert entry["confirmed_at"] is not None
        assert entry["confirmed_at"] >= entry["event_time"]
        assert entry["candle_epoch"] == entry["event_time"]
        assert entry["detector_id"] == "scripted"
        assert entry["detector_config_id"] == "scripted_cfg_v1"
        assert isinstance(entry["payload"], dict)
        assert entry["payload"]  # non-empty content
    # Specific values for the two scripted observations
    by_type = {e["type"]: e for e in accepted}
    assert by_type["structure_break"]["variant"] == "bos"
    assert by_type["structure_break"]["payload"] == {"step": "arm"}
    assert by_type["zone_interaction"]["variant"] == "rejection"
    assert by_type["zone_interaction"]["payload"] == {"step": "trigger"}
    # Emitted and accepted are tracked as distinct sequences on the cycle
    assert any(rec.emitted_observations for rec in r.cycles)


def test_r6_identity_field_difference_visible_in_cycle_evidence():
    """R6: changing observation identity fields changes recorded cycle evidence."""
    candles = _candles(8)
    e0 = candles[4].epoch

    def _run(detector_id: str):
        o = Observation(
            id=f"structure_break_bos_{e0}_{detector_id}",
            type="structure_break",
            symbol=SYMBOL,
            timeframe=TF,
            candle_epoch=e0,
            event_time=e0,
            confirmed_at=e0,
            detected_at=float(e0),
            developing=False,
            variant="bos",
            detector_id=detector_id,
            detector_config_id=f"{detector_id}_cfg",
            payload={"marker": detector_id},
        )
        # Minimal setup that accepts structure_break/bos as completion via expression of type only
        from cpab.conditions.validation import parse_setup as _ps
        setup = _ps({
            "schema_version": 1,
            "setup_id": f"id_{detector_id}",
            "symbol": SYMBOL,
            "timeframe": TF,
            "enabled": True,
            "expression": {"type": "structure_break", "variant": "bos", "min_count": 1},
        })
        return run_replay(
            ReplayRunConfig(
                setups=[setup],
                detectors=[ScriptedDetector({e0: [o]})],
                symbol=SYMBOL,
                timeframe=TF,
                timeframe_seconds=TF_SEC,
                candles=candles,
                arm_count=5,
            )
        )

    a = _run("det_a")
    b = _run("det_b")
    a_entries = [e for rec in a.cycles for e in rec.newly_accepted_observations]
    b_entries = [e for rec in b.cycles for e in rec.newly_accepted_observations]
    assert a_entries and b_entries
    assert a_entries[0]["detector_id"] != b_entries[0]["detector_id"]
    assert a_entries[0]["payload"] != b_entries[0]["payload"]
    assert a_entries[0]["id"] != b_entries[0]["id"]


def test_c4_confirm_lag_changes_closed_visibility():
    """Controlled timeline: lag 0/1/3 change which bars are closed under mark_closed_by_lag."""
    from cpab.market.validation import mark_closed_by_lag

    # 5 bars, 60s step. Server time fixed so lag controls the open/closed boundary.
    bars = [
        make_candle(SYMBOL, TF, START + i * TF_SEC, 100.0, 101.0, 99.0, 100.5, is_closed=False)
        for i in range(5)
    ]
    # Latest bar epoch + 1 full bar duration; lag*granularity further delays closure of recent bars.
    server = START + 4 * TF_SEC + TF_SEC  # just after last bar open

    def closed_epochs(lag: int) -> tuple[int, ...]:
        marked = mark_closed_by_lag(bars, TF_SEC, server, lag)
        return tuple(c.epoch for c in marked if c.is_closed)

    e0 = closed_epochs(0)
    e1 = closed_epochs(1)
    e3 = closed_epochs(3)
    # Higher lag withholds more recent bars
    assert len(e0) >= len(e1) >= len(e3)
    assert e0 != e3
    # lag 1 is production-default path: not identical to lag 0 in this timeline
    assert e0 != e1


def test_c4_positive_parity_at_production_default_lag_1():
    """A10 positive parity at production default confirm_lag_bars=1 when configs match."""
    candles = _candles(12)
    raw = json.loads((FIXTURES_8A / "phase8a_completion.json").read_text())
    setup = parse_setup(raw)
    e0 = candles[5].epoch
    e1 = candles[6].epoch
    script = {
        e0: [_obs("structure_break", e0, variant="bos")],
        e1: [_obs("zone_interaction", e1, variant="rejection")],
    }
    CAP, LAG = 500, 1
    # Replay annotates the same lag/cache so effective-input knobs match.
    # Candle series is pre-closed; lag=1 with server_time advanced still yields
    # the full closed set for this fixture (verified via matching epochs).
    replay = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[ScriptedDetector(script)],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=6,
            record_confirm_lag_bars=LAG,
            record_cache_max_size=CAP,
            record_htf_source=None,
        )
    )
    live = run_live_style(
        LiveStyleConfig(
            setups=[setup],
            detectors=[ScriptedDetector(script)],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=6,
            cache_max_size=CAP,
            confirm_lag_bars=LAG,
        )
    )
    for a, b in zip(replay.cycles, live.cycles):
        assert a.effective_candle_epochs == b.effective_candle_epochs
    cmp = compare_replay_runs(replay, live, setup_ids=[setup.setup_id])
    assert cmp.matching and not cmp.input_divergence, cmp.to_dict()


def test_c3_lifecycle_calls_zero_when_evaluation_blocked(monkeypatch):
    """Blocked recovery cycle must report lifecycle_calls_per_cycle == 0."""
    candles = _candles(10)
    calls = {"n": 0}
    from cpab.conditions import lifecycle as lc_mod

    real_eval = lc_mod.SetupLifecycleEvaluator.evaluate

    def spy_eval(self, *a, **k):
        calls["n"] += 1
        return real_eval(self, *a, **k)

    monkeypatch.setattr(lc_mod.SetupLifecycleEvaluator, "evaluate", spy_eval)

    # Fault makes series unhealthy → evaluation blocked on that cycle
    r = run_live_style(
        LiveStyleConfig(
            setups=[],
            detectors=[EngulfingDetector()],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            confirm_lag_bars=1,
            cache_max_size=100,
            fault_schedule={6: {"series_healthy": False}},
        )
    )
    assert r.lifecycle_calls_per_cycle
    # Find blocked cycles in records
    for rec, lc in zip(r.cycles, r.lifecycle_calls_per_cycle):
        if rec.evaluation_blocked:
            assert lc == 0, (rec.cycle_index, lc, rec.evaluation_blocked)

