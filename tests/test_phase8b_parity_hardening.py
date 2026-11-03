"""Phase 8B correction pass: B1 no-git, B2 missing metadata, B3 cycle indices, C1 configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpab.conditions.validation import parse_setup
from cpab.detectors.base import PriceActionDetector
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.market_structure import MarketStructureDetector
from cpab.models import Observation
from cpab.replay.driver import CycleRecord, ReplayResult, ReplayRunConfig, detector_fingerprint, run_replay
from cpab.replay.parity import (
    ENGINE_DIVERGENCE,
    INPUT_DIVERGENCE,
    MISSING,
    compare_replay_runs,
)
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
        detector_id="scripted",
        detector_config_id="scripted_cfg",
        payload={"marker": typ},
    )


def _candles(n: int) -> list:
    return [
        make_candle(SYMBOL, TF, START + i * TF_SEC, 100.0, 101.0, 99.0, 100.5, is_closed=True)
        for i in range(n)
    ]


def _empty_rec(cycle_index: int, **kw) -> CycleRecord:
    defaults = dict(
        cycle_index=cycle_index,
        candle_epoch=None,
        candle_ohlc=None,
        readiness_state="READY",
        data_ok=True,
        data_diagnostics=(),
        evaluation_blocked=False,
        emitted_observation_ids=(),
        newly_accepted_observation_ids=(),
        setup_states={},
        setup_previous_states={},
        setup_reasons={},
        setup_anchors={},
        setup_diagnostic_only={},
        signal_ids=(),
        monitoring=None,
        effective_candle_epochs=(),
        effective_candle_ohlcs=(),
        effective_detector_names=(),
        effective_detector_configs=(),
    )
    defaults.update(kw)
    return CycleRecord(**defaults)


def _result(cycles: list[CycleRecord]) -> ReplayResult:
    return ReplayResult(
        cycles=cycles,
        final_setup_states={},
        signals=[],
        type_report_warnings=(),
        required_types=(),
        wired_types=(),
    )


# --- B1: no-Git portability ---

def test_b1_begin_acceptance_run_without_git(tmp_path, monkeypatch):
    """Ordinary evidence bootstrap must not raise when git is unavailable."""
    from cpab.replay import evidence as ev

    def _boom(*a, **k):
        raise FileNotFoundError("git")

    prev_id = ev.current_run_id()
    prev_path = tmp_path / "ev.json"
    monkeypatch.setattr(ev.subprocess, "check_output", _boom)
    monkeypatch.setenv("CPAB_PHASE8B_EVIDENCE_PATH", str(prev_path))
    try:
        run = ev.begin_acceptance_run(note="no_git", pytest_args=["pytest", "-q"])
        assert run["commit"] == "unknown"
        assert run["working_tree_clean"] is False
        with pytest.raises(AssertionError, match="commit|working tree"):
            ev.validate_run_identity({"run": run, "a6": {}, "stall": {}})
    finally:
        # Restore session run_id so later A15 meta-tests still bind to sessionstart.
        ev._CURRENT_RUN_ID = prev_id


def test_b1_pytest_sessionstart_survives_git_failure(monkeypatch, tmp_path):
    from cpab.replay import evidence as ev
    import conftest as cf

    def _boom(*a, **k):
        raise FileNotFoundError("git")

    prev_id = ev.current_run_id()
    monkeypatch.setattr(ev.subprocess, "check_output", _boom)
    monkeypatch.setenv("CPAB_PHASE8B_EVIDENCE_PATH", str(tmp_path / "ev2.json"))
    try:
        class Sess:
            pass
        cf.pytest_sessionstart(Sess())  # must not raise
    finally:
        ev._CURRENT_RUN_ID = prev_id


# --- B2: missing metadata fail closed ---

def test_b2_missing_vs_known_metadata_diverges():
    """Missing metadata never equals known values (fail closed).

    Both sides omitting the same key yields MISSING==MISSING (equal). The defect
    under review was inventing defaults so a missing side matched a known side.
    """
    known = {
        "confirm_lag_bars": 1,
        "cache_max_size": 500,
        "htf_epochs": (),
        "htf_source": None,
        "htf_ohlcs": (),
        "completed_count": 0,
    }
    a = _result([_empty_rec(5, monitoring=None)])
    b = _result([_empty_rec(5, monitoring=dict(known))])
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.input_divergence
    assert cmp.first_divergence is not None
    assert cmp.first_divergence.kind == INPUT_DIVERGENCE
    assert MISSING in (cmp.first_divergence.expected, cmp.first_divergence.actual)


def test_b2_both_sides_missing_fail_closed():
    """MISSING vs MISSING is INPUT_DIVERGENCE — absence is not proof of match."""
    a = _result([_empty_rec(5, monitoring=None)])
    b = _result([_empty_rec(5, monitoring=None)])
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.input_divergence
    assert cmp.first_divergence is not None
    assert cmp.first_divergence.kind == INPUT_DIVERGENCE
    assert "effective_input." in cmp.first_divergence.field
    assert cmp.first_divergence.expected == MISSING
    assert cmp.first_divergence.actual == MISSING


def test_b2_htf_none_not_equal_empty_tuple():
    mon_none = {
        "confirm_lag_bars": 0,
        "cache_max_size": 100,
        "htf_epochs": (),
        "htf_source": None,
        "htf_ohlcs": None,  # explicit null
        "completed_count": 1,
    }
    mon_empty = dict(mon_none)
    mon_empty["htf_ohlcs"] = ()
    a = _result([_empty_rec(3, monitoring=mon_none)])
    b = _result([_empty_rec(3, monitoring=mon_empty)])
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.input_divergence
    assert cmp.first_divergence.field == "effective_input.htf_ohlcs"
    assert cmp.first_divergence.expected is None
    assert cmp.first_divergence.actual == ()


def test_b2_htf_empty_not_equal_populated():
    mon_empty = {
        "confirm_lag_bars": 0,
        "cache_max_size": 100,
        "htf_epochs": (),
        "htf_source": None,
        "htf_ohlcs": (),
        "completed_count": 1,
    }
    mon_pop = dict(mon_empty)
    mon_pop["htf_ohlcs"] = ((START, 1.0, 2.0, 0.5, 1.5, 1.0),)
    mon_pop["htf_epochs"] = (START,)
    mon_pop["htf_source"] = "explicit_series"
    a = _result([_empty_rec(3, monitoring=mon_empty)])
    b = _result([_empty_rec(3, monitoring=mon_pop)])
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.input_divergence
    assert "htf" in cmp.first_divergence.field


# --- B3: actual cycle indices ---

def test_b3_reports_actual_cycle_index_not_list_position():
    mon = {
        "confirm_lag_bars": 0,
        "cache_max_size": 100,
        "htf_epochs": (),
        "htf_source": None,
        "htf_ohlcs": (),
        "completed_count": 1,
    }
    # Non-zero starting indices: list pos 0 is cycle_index 59
    a = _result([
        _empty_rec(59, monitoring=mon, data_ok=True),
        _empty_rec(60, monitoring=mon, data_ok=True),
    ])
    b = _result([
        _empty_rec(59, monitoring=mon, data_ok=True),
        _empty_rec(60, monitoring=mon, data_ok=False),  # diverge on second
    ])
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.first_divergence is not None
    assert cmp.first_divergence.cycle_index == 60  # actual, not list pos 1
    assert cmp.first_divergence.field == "effective_input.data_ok"
    assert cmp.first_divergence.expected is True
    assert cmp.first_divergence.actual is False
    assert cmp.first_divergence.kind == INPUT_DIVERGENCE


def test_b3_engine_divergence_exact_first():
    mon = {
        "confirm_lag_bars": 0,
        "cache_max_size": 100,
        "htf_epochs": (),
        "htf_source": None,
        "htf_ohlcs": (),
        "completed_count": 1,
    }
    a = _result([
        _empty_rec(10, monitoring=mon, signal_ids=("s1",), setup_states={"x": "ARMED"}),
    ])
    b = _result([
        _empty_rec(10, monitoring=mon, signal_ids=("s2",), setup_states={"x": "ARMED"}),
    ])
    cmp = compare_replay_runs(a, b)
    assert not cmp.matching
    assert cmp.engine_divergence
    assert not cmp.input_divergence
    assert cmp.first_divergence.kind == ENGINE_DIVERGENCE
    assert cmp.first_divergence.cycle_index == 10
    assert cmp.first_divergence.field == "signal_ids"
    assert cmp.first_divergence.expected == ("s1",)
    assert cmp.first_divergence.actual == ("s2",)


def test_b3_input_before_engine_when_both_differ():
    mon_a = {
        "confirm_lag_bars": 0,
        "cache_max_size": 100,
        "htf_epochs": (),
        "htf_source": None,
        "htf_ohlcs": (),
        "completed_count": 1,
    }
    mon_b = dict(mon_a)
    mon_b["confirm_lag_bars"] = 3
    a = _result([_empty_rec(7, monitoring=mon_a, signal_ids=("s1",))])
    b = _result([_empty_rec(7, monitoring=mon_b, signal_ids=("s2",))])
    cmp = compare_replay_runs(a, b)
    assert cmp.first_divergence.kind == INPUT_DIVERGENCE
    assert cmp.input_divergence
    # Engine divergence should not be claimed for the same cycle once inputs differ
    assert not any(
        d.kind == ENGINE_DIVERGENCE and d.cycle_index == 7 for d in cmp.divergences
    )


# --- C1: detector config fingerprints ---

def test_c1_detector_fingerprint_stable_and_ordered():
    d1 = MarketStructureDetector(left_strength=1, right_strength=1)
    d2 = MarketStructureDetector(left_strength=2, right_strength=2)
    f1 = detector_fingerprint(d1)
    f2 = detector_fingerprint(d2)
    assert f1 != f2
    assert "market_structure" in f1
    assert detector_fingerprint(d1) == f1  # stable


def test_c1_parity_detects_config_difference():
    candles = _candles(8)
    a = run_replay(
        ReplayRunConfig(
            setups=[],
            detectors=[MarketStructureDetector(left_strength=1, right_strength=1)],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            record_confirm_lag_bars=0,
            record_cache_max_size=100,
        )
    )
    b = run_replay(
        ReplayRunConfig(
            setups=[],
            detectors=[MarketStructureDetector(left_strength=3, right_strength=3)],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=5,
            record_confirm_lag_bars=0,
            record_cache_max_size=100,
        )
    )
    cmp = compare_replay_runs(a, b)
    assert cmp.input_divergence
    assert any("effective_detector_configs" in d.field for d in cmp.divergences)


def test_c1_parity_matching_configs():
    candles = _candles(8)
    def _run():
        return run_replay(
            ReplayRunConfig(
                setups=[],
                detectors=[EngulfingDetector()],
                symbol=SYMBOL,
                timeframe=TF,
                timeframe_seconds=TF_SEC,
                candles=candles,
                arm_count=5,
                record_confirm_lag_bars=0,
                record_cache_max_size=100,
            )
        )
    cmp = compare_replay_runs(_run(), _run())
    assert cmp.matching
    assert not cmp.input_divergence
