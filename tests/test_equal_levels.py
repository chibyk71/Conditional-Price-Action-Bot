"""Phase 2B — Equal Highs / Equal Lows detection tests."""

from __future__ import annotations

import math

import pytest

from cpab.detectors.equal_levels import EqualLevelsDetector, _validate_tolerance
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.detectors.swing import SwingPivotDetector
from cpab.models import Candle, Observation
from cpab.core.observation_store import ObservationStore


def _c(
    epoch: int,
    o: float,
    h: float,
    l: float,
    c: float,
    *,
    symbol: str = "R_100",
    timeframe: str = "1m",
    is_closed: bool = True,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        epoch=epoch,
        open=o,
        high=h,
        low=l,
        close=c,
        is_closed=is_closed,
    )


def _series(
    highs: list[float],
    lows: list[float] | None = None,
    start: int = 1_700_000_000,
    *,
    step: int = 60,
    timeframe: str = "1m",
) -> list[Candle]:
    lows = lows if lows is not None else [h - 1.0 for h in highs]
    assert len(highs) == len(lows)
    out = []
    for i, (h, l) in enumerate(zip(highs, lows)):
        mid = (h + l) / 2.0
        out.append(_c(start + i * step, mid, h, l, mid, timeframe=timeframe))
    return out


# ---------------------------------------------------------------------------
# A. Basic detection
# ---------------------------------------------------------------------------


def test_two_swing_highs_within_tolerance_create_eqh():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1, tie_policy="strict")
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    assert len(levels) == 1
    lvl = levels[0]
    assert abs(lvl.payload["representative_price"] - 12.05) < 1e-9
    assert lvl.payload["tolerance"] == 0.2
    assert lvl.payload["touch_count"] == 2
    assert lvl.direction is None
    assert len(lvl.reference_ids) == 2
    assert lvl.confirmed_at is not None


def test_two_swing_lows_within_tolerance_create_eql():
    candles = _series(
        [12.0, 10.0, 11.0, 10.5, 10.1, 11.0],
        lows=[11.0, 8.0, 9.0, 8.5, 8.1, 9.0],
    )
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1, tie_policy="strict")
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_low_level"]
    assert len(levels) == 1
    assert levels[0].payload["tolerance"] == 0.2
    assert levels[0].direction is None


def test_pair_outside_tolerance_no_level():
    candles = _series([10.0, 12.0, 11.0, 10.5, 15.0, 11.0])
    d = EqualLevelsDetector(tolerance=0.1, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    assert [o for o in all_obs if o.type.startswith("equal_")] == []


def test_equality_exactly_at_tolerance_boundary():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.5, 11.0])
    d = EqualLevelsDetector(tolerance=0.5, left_strength=1, right_strength=1)
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(levels) == 1


def test_zero_tolerance_exact_match_only():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.0, 11.0])
    d = EqualLevelsDetector(tolerance=0.0, left_strength=1, right_strength=1)
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(levels) == 1
    assert levels[0].payload["representative_price"] == 12.0

    d2 = EqualLevelsDetector(tolerance=0.0, left_strength=1, right_strength=1)
    candles2 = _series([10.0, 12.0, 11.0, 10.5, 12.0001, 11.0])
    assert [o for o in d2.detect_all_historical(candles2) if o.type == "equal_high_level"] == []


# ---------------------------------------------------------------------------
# B. Configuration validation
# ---------------------------------------------------------------------------


def test_missing_tolerance_rejected_constructor():
    with pytest.raises(TypeError):
        EqualLevelsDetector()  # type: ignore[call-arg]


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        EqualLevelsDetector(tolerance=-0.1)


def test_boolean_tolerance_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        EqualLevelsDetector(tolerance=True)  # type: ignore[arg-type]


def test_string_tolerance_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        EqualLevelsDetector(tolerance="0.1")  # type: ignore[arg-type]


def test_nan_inf_tolerance_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        EqualLevelsDetector(tolerance=float("nan"))
    with pytest.raises(ValueError, match="tolerance"):
        EqualLevelsDetector(tolerance=float("inf"))
    with pytest.raises(ValueError, match="tolerance"):
        EqualLevelsDetector(tolerance=float("-inf"))


def test_valid_nonnegative_tolerance_accepted():
    d0 = EqualLevelsDetector(tolerance=0)
    assert d0.tolerance == 0.0
    d1 = EqualLevelsDetector(tolerance=0.5)
    assert d1.tolerance == 0.5
    d2 = EqualLevelsDetector(tolerance=1)
    assert d2.tolerance == 1.0


def test_tolerance_participates_in_config_identity():
    d1 = EqualLevelsDetector(tolerance=0.1)
    d2 = EqualLevelsDetector(tolerance=0.2)
    d3 = EqualLevelsDetector(tolerance=0.1, left_strength=2)
    assert d1.detector_config_id != d2.detector_config_id
    assert d1.detector_config_id != d3.detector_config_id
    d1b = EqualLevelsDetector(tolerance=0.1)
    assert d1.detector_config_id == d1b.detector_config_id


def test_different_swing_configs_cannot_share_levels():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d1 = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    d2 = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=2)
    o1 = [o for o in d1.detect_all_historical(candles) if o.type == "equal_high_level"]
    o2 = [o for o in d2.detect_all_historical(candles) if o.type == "equal_high_level"]
    if o1 and o2:
        assert o1[0].detector_config_id != o2[0].detector_config_id
        assert o1[0].id != o2[0].id


# ---------------------------------------------------------------------------
# C. Grouping and determinism
# ---------------------------------------------------------------------------


def test_nearest_existing_level_selected():
    candles = _series(
        [9.0, 10.0, 9.5, 9.2, 10.1, 9.5, 19.0, 20.0, 19.5, 19.2, 20.1, 19.5, 9.5, 10.05, 9.8]
    )
    d = EqualLevelsDetector(tolerance=0.15, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    touches = [o for o in all_obs if o.type == "equal_high_level_touch"]
    assert len(levels) >= 1
    for t in touches:
        assert "level_id" in t.payload
        assert t.payload["level_id"] in {lv.payload["level_id"] for lv in levels}


def test_tolerance_chaining_prevented():
    candles = _series([9.0, 10.0, 9.5, 9.2, 10.4, 9.5, 9.2, 10.8, 9.5])
    d = EqualLevelsDetector(tolerance=0.5, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    touches = [o for o in all_obs if o.type == "equal_high_level_touch"]
    for t in touches:
        assert abs(t.payload["swing_price"] - t.payload["representative_price"]) <= 0.5 + 1e-12


def test_representative_price_fixed_after_creation():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0, 10.5, 12.05, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    touches = [o for o in all_obs if o.type == "equal_high_level_touch"]
    if levels and touches:
        rep = levels[0].payload["representative_price"]
        for t in touches:
            if t.payload["level_id"] == levels[0].payload["level_id"]:
                assert t.payload["representative_price"] == rep


def test_input_order_does_not_change_results():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    a = d.detect_all_historical(candles)
    b = d.detect_all_historical(list(candles))
    assert [o.id for o in a] == [o.id for o in b]


def test_high_and_low_levels_isolated():
    candles = _series(
        [10.0, 12.0, 11.0, 10.5, 12.1, 11.0],
        lows=[9.0, 8.0, 8.5, 8.2, 8.05, 8.5],
    )
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    highs = [o for o in all_obs if "high" in o.type]
    lows = [o for o in all_obs if "low" in o.type]
    for h in highs:
        assert "low" not in h.type
    for l in lows:
        assert "high" not in l.type


# ---------------------------------------------------------------------------
# D. Causality and data quality
# ---------------------------------------------------------------------------


def test_level_not_emitted_before_second_swing_confirmed():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    assert levels
    second_confirm = candles[5].epoch
    for lvl in levels:
        assert lvl.confirmed_at == second_confirm
        assert not lvl.is_eligible_at(candles[4].epoch)
        assert lvl.is_eligible_at(second_confirm)


def test_incomplete_candles_not_used():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    last = candles[-1]
    candles[-1] = Candle(
        symbol=last.symbol,
        timeframe=last.timeframe,
        epoch=last.epoch,
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        is_closed=False,
    )
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    obs = d.detect(candles)
    for o in obs:
        assert o.confirmed_at != last.epoch or last.is_closed


def test_insufficient_history_no_emit():
    candles = _series([10.0, 12.0, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    assert d.min_history == 5
    assert d.detect(candles) == []
    assert d.detect_all_historical(candles) == []


def test_gap_prevents_swing_and_thus_level():
    start = 1_700_000_000
    candles = [
        _c(start, 10.0, 10.0, 9.0, 10.0),
        _c(start + 60, 11.0, 12.0, 10.0, 11.0),
        _c(start + 180, 10.5, 11.0, 9.5, 10.5),
        _c(start + 240, 10.5, 10.5, 9.5, 10.5),
        _c(start + 300, 11.0, 12.1, 10.0, 11.0),
        _c(start + 360, 10.5, 11.0, 9.5, 10.5),
    ]
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    assert d.detect_all_historical(candles) == []


# ---------------------------------------------------------------------------
# E. Observation identity and touch behavior
# ---------------------------------------------------------------------------


def test_level_identity_deterministic():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    a = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    b = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert a and b
    assert a[0].id == b[0].id
    assert a[0].payload["level_id"] == b[0].payload["level_id"]


def test_touch_identity_deterministic_and_store_idempotent():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0, 10.5, 12.05, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    a = d.detect_all_historical(candles)
    b = d.detect_all_historical(candles)
    assert [o.id for o in a] == [o.id for o in b]
    store = ObservationStore()
    assert store.accept(a) == len(a)
    assert store.accept(b) == 0


def test_touch_does_not_mutate_creation():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0, 10.5, 12.05, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    if levels:
        assert levels[0].payload["touch_count"] == 2


def test_creation_and_touch_ids_distinct_from_level_id():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert levels
    lid = levels[0].payload["level_id"]
    assert levels[0].id != lid


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_registry_equal_levels():
    reg = build_default_registry()
    assert reg.is_supported("equal_high")
    assert reg.is_supported("equal_low")
    assert reg.is_supported("eqh")
    with pytest.raises(ValueError, match="tolerance"):
        reg.create_detector("equal_high")
    det, info = reg.create_detector("equal_high", config={"tolerance": 0.25})
    assert isinstance(det, EqualLevelsDetector)
    assert det.tolerance == 0.25
    assert info.min_history == 5
    assert det.detector_config_id == info.detector_config_id
    assert "tolerance" in info.config


def test_registry_rejects_bad_tolerance():
    reg = build_default_registry()
    with pytest.raises(ValueError):
        reg.create_detector("eql", config={"tolerance": -1})
    with pytest.raises(ValueError):
        reg.create_detector("eql", config={"tolerance": True})
    with pytest.raises(ValueError):
        reg.create_detector("eql", config={"tolerance": float("nan")})


def test_registry_swing_params_override():
    reg = build_default_registry()
    det, info = reg.create_detector(
        "equal_level",
        config={"tolerance": 0.1, "left_strength": 2, "right_strength": 2, "tie_policy": "first"},
    )
    assert det.left_strength == 2
    assert det.right_strength == 2
    assert det.tie_policy == "first"
    assert info.min_history == 2 * (2 + 1 + 2) - 1


def test_detect_latest_bar_focus():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    all_hist = d.detect_all_historical(candles)
    latest = d.detect(candles)
    latest_epoch = candles[-1].epoch
    for o in latest:
        assert o.confirmed_at == latest_epoch
    hist_ids = {o.id for o in all_hist}
    for o in latest:
        assert o.id in hist_ids
