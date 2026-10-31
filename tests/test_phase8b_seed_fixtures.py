"""Phase 8B C1: contract seed fixtures integrity + A6 outcomes on real detectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cpab.conditions.lifecycle import LifecycleState
from cpab.conditions.validation import parse_setup
from cpab.detectors.behavior_volatility import ImpulseDetector
from cpab.detectors.breaker_block import BreakerBlockDetector
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.market_structure import MarketStructureDetector
from cpab.detectors.zone_polarity_flip import ZonePolarityFlipDetector
from cpab.models import Candle
from cpab.replay.driver import ReplayRunConfig, run_replay
from cpab.replay.evidence import record_a6
from cpab.replay.real_talib import real_talib_available, real_talib_info

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "phase8b"
START = 1_699_992_000
STEP = 900
ARM = 60


def _load_manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


def _load_candles(rel: str) -> list[Candle]:
    data = json.loads((FIXTURES / rel).read_text(encoding="utf-8"))
    symbol = data["symbol"]
    tf = data["timeframe"]
    out: list[Candle] = []
    for row in data["candles"]:
        epoch, o, h, l, c = row
        out.append(
            Candle(
                symbol=symbol,
                timeframe=tf,
                epoch=int(epoch),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                is_closed=True,
            )
        )
    return out


def _load_setup(name: str):
    raw = json.loads((FIXTURES / "setups" / f"{name}.json").read_text(encoding="utf-8"))
    return parse_setup(raw)


def test_seed_fixture_sha256_matches_manifest():
    m = _load_manifest()
    for case_id, meta in m["cases"].items():
        path = FIXTURES / meta["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == meta["sha256"], (case_id, digest, meta["sha256"])
    for setup_name in (
        "fx_bearish_breaker_then_impulse",
        "fx_bullish_higher_low_trigger",
        "fx_bearish_flip_then_engulfing",
    ):
        p = FIXTURES / "setups" / f"{setup_name}.json"
        assert p.is_file()


def _mark(fid: str, ok: bool, **extra):
    status = "real_env_passed" if ok and real_talib_available() else (
        "stub_only" if ok else "real_env_failed"
    )
    record_a6(fid, status, extra=extra or None)


def _run(setup, detectors, candles):
    return run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=list(detectors),
            symbol=setup.symbol,
            timeframe=setup.timeframe,
            timeframe_seconds=STEP,
            candles=candles,
            arm_count=ARM,
        )
    )


def test_a6_f1_completion_seed3():
    meta = _load_manifest()["cases"]["F1_completion_seed3"]
    candles = _load_candles(meta["file"])
    setup = _load_setup(meta["setup"])
    dets = [
        BreakerBlockDetector(left_strength=1, right_strength=1),
        ImpulseDetector(),
    ]
    r = _run(setup, dets, candles)
    expected_t = meta["expected_transition_idx"]
    tc = r.transition_cycle(setup.setup_id, LifecycleState.COMPLETED.value)
    terminal = r.final_setup_states.get(setup.setup_id)
    observed = {"terminal": terminal, "transition_idx": tc}
    ok = terminal == meta["expected_terminal"] and tc == expected_t and len(r.signals) == 1
    if not ok:
        # C1.4: do not rewrite expected indices under real TA-Lib drift.
        print(f"A6 INDEX DRIFT F1-completion expected={expected_t}/{meta['expected_terminal']} observed={observed}")
    assert ok, observed
    _mark("F1-completion-real", ok, **observed)


def test_a6_f1_invalidation_seed1():
    meta = _load_manifest()["cases"]["F1_invalidation_seed1"]
    candles = _load_candles(meta["file"])
    setup = _load_setup(meta["setup"])
    dets = [
        BreakerBlockDetector(left_strength=1, right_strength=1),
        ImpulseDetector(),
    ]
    r = _run(setup, dets, candles)
    expected_t = meta["expected_transition_idx"]
    tc = r.transition_cycle(setup.setup_id, LifecycleState.INVALIDATED.value)
    terminal = r.final_setup_states.get(setup.setup_id)
    observed = {"terminal": terminal, "transition_idx": tc}
    ok = terminal == meta["expected_terminal"] and tc == expected_t and r.signals == []
    if not ok:
        print(f"A6 INDEX DRIFT F1-invalidation expected={expected_t} observed={observed}")
    assert ok, observed
    _mark("F1-invalidation-real", ok, **observed)


def test_a6_f1_expiry_seed4():
    meta = _load_manifest()["cases"]["F1_expiry_seed4"]
    candles = _load_candles(meta["file"])
    setup = _load_setup(meta["setup"])
    dets = [
        BreakerBlockDetector(left_strength=1, right_strength=1),
        ImpulseDetector(),
    ]
    r = _run(setup, dets, candles)
    expected_t = meta["expected_transition_idx"]
    tc = r.transition_cycle(setup.setup_id, LifecycleState.EXPIRED.value)
    terminal = r.final_setup_states.get(setup.setup_id)
    observed = {"terminal": terminal, "transition_idx": tc}
    ok = terminal == meta["expected_terminal"] and tc == expected_t and r.signals == []
    if not ok:
        print(f"A6 INDEX DRIFT F1-expiry expected={expected_t} observed={observed}")
    assert ok, observed
    _mark("F1-expiry-real", ok, **observed)


def test_a6_f2_completion_seed4():
    meta = _load_manifest()["cases"]["F2_completion_seed4"]
    candles = _load_candles(meta["file"])
    setup = _load_setup(meta["setup"])
    dets = [
        MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict"),
        EngulfingDetector(),
        ImpulseDetector(),
    ]
    ms = dets[0]
    hist = ms.detect_all_historical(candles)
    hl = [o for o in hist if o.variant == "hl"]
    assert hl, "HL must emit on seed series"
    # Prefer the frozen 75/76 HL when present among multiple swings.
    chosen = None
    for o in hl:
        ei = (o.candle_epoch - START) // STEP
        ci = (o.confirmed_at - START) // STEP
        if ei == 75 and ci == 76:
            chosen = o
            break
    if chosen is None:
        chosen = hl[0]
    event_idx = (chosen.candle_epoch - START) // STEP
    conf_idx = (chosen.confirmed_at - START) // STEP
    r = _run(setup, dets, candles)
    expected_t = meta["expected_transition_idx"]
    tc = r.transition_cycle(setup.setup_id, LifecycleState.COMPLETED.value)
    terminal = r.final_setup_states.get(setup.setup_id)
    observed = {
        "terminal": terminal,
        "transition_idx": tc,
        "hl_event_idx": event_idx,
        "hl_confirm_idx": conf_idx,
    }
    ok = (
        terminal == meta["expected_terminal"]
        and tc == expected_t
        and event_idx == 75
        and conf_idx == 76
        and len(r.signals) == 1
    )
    if not ok:
        print(f"A6 INDEX DRIFT F2-completion expected t=76 event=75/76 observed={observed}")
    # Causal: not completed before confirmation cycle 76
    for c in r.cycles:
        if c.cycle_index < 76:
            assert c.setup_states.get(setup.setup_id) != LifecycleState.COMPLETED.value
    assert ok, observed
    _mark("F2-completion-real", ok, **observed)


def test_a6_f2_invalidation_seed7():
    meta = _load_manifest()["cases"]["F2_invalidation_seed7"]
    candles = _load_candles(meta["file"])
    setup = _load_setup(meta["setup"])
    dets = [
        MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict"),
        EngulfingDetector(),
        ImpulseDetector(),
    ]
    hist = dets[0].detect_all_historical(candles)
    ll = [o for o in hist if o.variant == "ll"]
    assert ll, "LL must emit"
    chosen = None
    for o in ll:
        ei = (o.candle_epoch - START) // STEP
        ci = (o.confirmed_at - START) // STEP
        if ei == 92 and ci == 93:
            chosen = o
            break
    if chosen is None:
        chosen = ll[0]
    event_idx = (chosen.candle_epoch - START) // STEP
    conf_idx = (chosen.confirmed_at - START) // STEP
    r = _run(setup, dets, candles)
    expected_t = meta["expected_transition_idx"]
    tc = r.transition_cycle(setup.setup_id, LifecycleState.INVALIDATED.value)
    terminal = r.final_setup_states.get(setup.setup_id)
    observed = {
        "terminal": terminal,
        "transition_idx": tc,
        "ll_event_idx": event_idx,
        "ll_confirm_idx": conf_idx,
    }
    term_ok = terminal == meta["expected_terminal"] and tc == expected_t and r.signals == []
    idx_ok = event_idx == 92 and conf_idx == 93
    if not idx_ok:
        print(
            f"A6 INDEX DRIFT F2-invalidation event/confirm expected=92/93 "
            f"observed={event_idx}/{conf_idx} (terminal ok={term_ok}, t={tc})"
        )
    for c in r.cycles:
        if c.cycle_index < 93:
            assert c.setup_states.get(setup.setup_id) != LifecycleState.INVALIDATED.value
    # Frozen transition index is mandatory; swing indices reported if drifted.
    assert term_ok, observed
    _mark("F2-invalidation-real", term_ok, **observed, swing_idx_ok=idx_ok)


def test_a6_f3_completion_seed1():
    meta = _load_manifest()["cases"]["F3_completion_seed1"]
    candles = _load_candles(meta["file"])
    setup = _load_setup(meta["setup"])
    dets = [ZonePolarityFlipDetector(), EngulfingDetector()]
    r = _run(setup, dets, candles)
    expected_t = meta["expected_transition_idx"]
    tc = r.transition_cycle(setup.setup_id, LifecycleState.COMPLETED.value)
    terminal = r.final_setup_states.get(setup.setup_id)
    observed = {"terminal": terminal, "transition_idx": tc}
    ok = terminal == meta["expected_terminal"] and tc == expected_t and len(r.signals) == 1
    if not ok:
        print(f"A6 INDEX DRIFT F3-completion expected={expected_t} observed={observed}")
    assert ok, observed
    _mark("F3-completion-real", ok, **observed)


def test_a6_f3_invalidation_seed23():
    meta = _load_manifest()["cases"]["F3_invalidation_seed23"]
    candles = _load_candles(meta["file"])
    setup = _load_setup(meta["setup"])
    dets = [ZonePolarityFlipDetector(), EngulfingDetector()]
    r = _run(setup, dets, candles)
    expected_t = meta["expected_transition_idx"]
    tc = r.transition_cycle(setup.setup_id, LifecycleState.INVALIDATED.value)
    terminal = r.final_setup_states.get(setup.setup_id)
    observed = {"terminal": terminal, "transition_idx": tc}
    ok = terminal == meta["expected_terminal"] and tc == expected_t and r.signals == []
    if not ok:
        print(f"A6 INDEX DRIFT F3-invalidation expected={expected_t} observed={observed}")
    assert ok, observed
    _mark("F3-invalidation-real", ok, **observed)


def test_a6_phase8a_fixtures_still_mark_evidence():
    """Merged Phase 8A injected fixtures remain mandatory A6 keys."""
    from tests.test_phase8b_real_detector_outcomes import (
        test_a6_phase8a_completion,
        test_a6_phase8a_invalidation,
        test_a6_phase8a_expiry,
    )
    # Invoke directly so evidence is recorded when this module runs alone after sessionstart.
    test_a6_phase8a_completion()
    test_a6_phase8a_invalidation()
    test_a6_phase8a_expiry()


def test_detector_emitted_types_registry_covers_phase8b_detectors():
    from cpab.conditions.analyst import DETECTOR_EMITTED_TYPES, detector_wiring_status

    dets = [
        EngulfingDetector(),
        ImpulseDetector(),
        MarketStructureDetector(left_strength=1, right_strength=1),
        BreakerBlockDetector(left_strength=1, right_strength=1),
        ZonePolarityFlipDetector(),
    ]
    status = detector_wiring_status(dets)
    for d in dets:
        assert status[type(d).__name__] == "declared"
        assert type(d).__name__ in DETECTOR_EMITTED_TYPES


def test_real_talib_info_exposes_extension():
    if not real_talib_available():
        pytest.skip("real TA-Lib unavailable")
    info = real_talib_info()
    assert info["talib_version"]
    assert info["talib_file"]
    assert ".so" in info["talib_extension"] or ".pyd" in info["talib_extension"] or ".dylib" in info["talib_extension"]
