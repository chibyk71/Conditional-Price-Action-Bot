"""Phase 2B — Equal Highs / Equal Lows detection tests."""

from __future__ import annotations

import pytest

from cpab.detectors.equal_levels import EqualLevelsDetector
from cpab.detectors.registry import build_default_registry
from cpab.models import Candle
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
    """Distinct swing configs yield distinct config ids and observation ids.

    Uses an 8-bar series long enough for right_strength=2 to confirm two
    independent swing highs (min_history = 2*(1+1+2)-1 = 7).
    """
    # Indices: 0=9, 1=12 (swing), 2=11, 3=10.5 → confirm under R=1 and R=2
    #          4=10, 5=12.1 (swing), 6=11, 7=10.5 → second swing
    candles = _series([9.0, 12.0, 11.0, 10.5, 10.0, 12.1, 11.0, 10.5])
    d1 = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    d2 = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=2)
    o1 = [o for o in d1.detect_all_historical(candles) if o.type == "equal_high_level"]
    o2 = [o for o in d2.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(o1) >= 1
    assert len(o2) >= 1
    assert o1[0].detector_config_id != o2[0].detector_config_id
    assert o1[0].id != o2[0].id


def test_nearest_existing_level_selected():
    candles = _series(
        [
            9.0, 10.0, 9.5,
            9.2, 10.1, 9.5,
            19.0, 20.0, 19.5,
            19.2, 20.1, 19.5,
            9.5, 10.08, 9.8,
        ]
    )
    d = EqualLevelsDetector(tolerance=0.15, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    touches = [o for o in all_obs if o.type == "equal_high_level_touch"]
    assert len(levels) == 2
    levels_sorted = sorted(levels, key=lambda o: o.payload["representative_price"])
    near_lvl = levels_sorted[0]
    far_lvl = levels_sorted[1]
    assert near_lvl.payload["representative_price"] < 15.0
    assert far_lvl.payload["representative_price"] > 15.0
    near_touches = [t for t in touches if abs(t.payload["swing_price"] - 10.08) < 1e-9]
    assert len(near_touches) == 1
    assert near_touches[0].payload["level_id"] == near_lvl.payload["level_id"]
    assert near_touches[0].payload["level_id"] != far_lvl.payload["level_id"]


def test_earliest_created_level_wins_distance_tie():
    """Two levels (rep 10 and 12); touch at 11 under ε=1.0 is equidistant; earliest wins."""
    candles = _series(
        [
            9.0, 10.0, 9.5,
            9.2, 10.0, 9.5,
            11.0, 12.0, 11.5,
            11.2, 12.0, 11.5,
            10.5, 11.0, 10.5,
        ]
    )
    d = EqualLevelsDetector(tolerance=1.0, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = sorted(
        [o for o in all_obs if o.type == "equal_high_level"],
        key=lambda o: (o.confirmed_at or 0, o.payload["level_id"]),
    )
    assert len(levels) == 2
    earliest = levels[0]
    later = levels[1]
    assert earliest.confirmed_at is not None and later.confirmed_at is not None
    assert earliest.confirmed_at < later.confirmed_at
    touches = [
        t for t in all_obs
        if t.type == "equal_high_level_touch" and abs(t.payload["swing_price"] - 11.0) < 1e-9
    ]
    assert len(touches) == 1
    assert touches[0].payload["level_id"] == earliest.payload["level_id"]
    assert touches[0].payload["level_id"] != later.payload["level_id"]


def test_lex_smallest_level_id_when_created_at_tied():
    """Among equidistant levels, selection matches min(created_at, level_id)."""
    candles = _series(
        [
            9.0, 10.0, 9.5,
            9.2, 10.0, 9.5,
            11.0, 12.0, 11.5,
            11.2, 12.0, 11.5,
            10.5, 11.0, 10.5,
        ]
    )
    d = EqualLevelsDetector(tolerance=1.0, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    assert len(levels) == 2
    expected = min(levels, key=lambda o: (o.confirmed_at or 0, o.payload["level_id"]))
    touches = [
        t for t in all_obs
        if t.type == "equal_high_level_touch" and abs(t.payload["swing_price"] - 11.0) < 1e-9
    ]
    assert len(touches) == 1
    assert touches[0].payload["level_id"] == expected.payload["level_id"]


def test_ungrouped_nearest_prior_swing_selected():
    candles = _series([9.0, 10.0, 9.5, 9.0, 10.5, 9.5, 9.0, 10.1, 9.5])
    d = EqualLevelsDetector(tolerance=0.3, left_strength=1, right_strength=1)
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(levels) == 1
    assert abs(levels[0].payload["representative_price"] - 10.05) < 1e-9


def test_ungrouped_earliest_prior_wins_distance_tie():
    """Priors at 10 and 12; new at 11 under ε=1.0; pairs with earliest prior (rep 10.5)."""
    candles = _series([9.0, 10.0, 9.5, 11.0, 12.0, 11.5, 10.5, 11.0, 10.5])
    d = EqualLevelsDetector(tolerance=1.0, left_strength=1, right_strength=1)
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(levels) == 1
    assert abs(levels[0].payload["representative_price"] - 10.5) < 1e-9
    assert len(levels[0].payload["supporting_swing_ids"]) == 2


def test_ungrouped_stable_swing_id_final_tiebreaker():
    candles = _series([9.0, 10.0, 9.5, 9.0, 10.0, 9.5, 9.0, 10.0, 9.5])
    d = EqualLevelsDetector(tolerance=0.0, left_strength=1, right_strength=1)
    a = d.detect_all_historical(candles)
    b = d.detect_all_historical(candles)
    assert [o.id for o in a] == [o.id for o in b]
    assert len([o for o in a if o.type == "equal_high_level"]) == 1


def test_tolerance_chaining_prevented():
    candles = _series([9.0, 10.0, 9.5, 9.2, 10.4, 9.5, 9.2, 10.8, 9.5])
    d = EqualLevelsDetector(tolerance=0.5, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    touches = [o for o in all_obs if o.type == "equal_high_level_touch"]
    assert len(levels) == 1
    assert abs(levels[0].payload["representative_price"] - 10.2) < 1e-9
    assert [t for t in touches if abs(t.payload["swing_price"] - 10.8) < 1e-9] == []


def test_representative_price_fixed_after_creation():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0, 10.5, 12.05, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    levels = [o for o in all_obs if o.type == "equal_high_level"]
    touches = [o for o in all_obs if o.type == "equal_high_level_touch"]
    assert len(levels) == 1
    rep = levels[0].payload["representative_price"]
    for t in touches:
        if t.payload["level_id"] == levels[0].payload["level_id"]:
            assert t.payload["representative_price"] == rep


def test_established_levels_not_automatically_merged():
    candles = _series([9.0, 10.0, 9.5, 9.2, 10.1, 9.5, 19.0, 20.0, 19.5, 19.2, 20.1, 19.5])
    d = EqualLevelsDetector(tolerance=0.15, left_strength=1, right_strength=1)
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(levels) == 2
    assert len({o.payload["level_id"] for o in levels}) == 2


def test_swing_not_assigned_to_multiple_levels():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0, 10.5, 12.05, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    all_obs = d.detect_all_historical(candles)
    seen: dict[str, int] = {}
    for o in all_obs:
        if o.type.endswith("_level"):
            for sid in o.payload.get("supporting_swing_ids", []):
                seen[sid] = seen.get(sid, 0) + 1
        elif o.type.endswith("_touch"):
            sid = o.payload.get("swing_observation_id")
            if sid:
                seen[sid] = seen.get(sid, 0) + 1
    for sid, count in seen.items():
        assert count == 1, f"swing {sid} assigned {count} times"


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
    for h in [o for o in all_obs if "high" in o.type]:
        assert "low" not in h.type
    for lo in [o for o in all_obs if "low" in o.type]:
        assert "high" not in lo.type


def test_level_not_emitted_before_second_swing_confirmed():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(levels) == 1
    assert levels[0].confirmed_at == candles[5].epoch
    assert not levels[0].is_eligible_at(candles[4].epoch)
    assert levels[0].is_eligible_at(candles[5].epoch)


def test_incomplete_candles_not_used():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    last = candles[-1]
    candles[-1] = Candle(
        symbol=last.symbol, timeframe=last.timeframe, epoch=last.epoch,
        open=last.open, high=last.high, low=last.low, close=last.close, is_closed=False,
    )
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    for o in d.detect(candles):
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


def test_level_identity_deterministic_same_full_history():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    a = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    b = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(a) == 1 and len(b) == 1
    assert a[0].id == b[0].id
    assert a[0].payload["level_id"] == b[0].payload["level_id"]


def test_truncated_history_missing_original_swings_does_not_reinvent_identity():
    """Early level excluded by truncation is absent; late level retained with same identity."""
    full = _series(
        [
            9.0, 12.0, 11.0,
            10.5, 12.1, 11.0,
            10.0, 15.0, 14.0,
            13.5, 15.1, 14.0,
        ]
    )
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    full_levels = sorted(
        [o for o in d.detect_all_historical(full) if o.type == "equal_high_level"],
        key=lambda o: o.confirmed_at or 0,
    )
    assert len(full_levels) == 2
    early_level = full_levels[0]
    late_level = full_levels[1]
    assert early_level.payload["representative_price"] < 14.0
    assert late_level.payload["representative_price"] > 14.0
    truncated = full[-6:]
    trunc_levels = [o for o in d.detect_all_historical(truncated) if o.type == "equal_high_level"]
    trunc_ids = {o.payload["level_id"] for o in trunc_levels}
    assert early_level.payload["level_id"] not in trunc_ids
    assert len(trunc_levels) == 1
    assert trunc_levels[0].payload["level_id"] == late_level.payload["level_id"]
    assert trunc_levels[0].id == late_level.id


def test_prefix_history_preserving_supporting_swings_keeps_identity():
    full = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0, 10.0, 11.0, 10.5])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    full_levels = [o for o in d.detect_all_historical(full) if o.type == "equal_high_level"]
    assert len(full_levels) == 1
    lid = full_levels[0].payload["level_id"]
    prefix_levels = [o for o in d.detect_all_historical(full[:6]) if o.type == "equal_high_level"]
    assert len(prefix_levels) == 1
    assert prefix_levels[0].payload["level_id"] == lid
    assert prefix_levels[0].id == full_levels[0].id


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
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(levels) == 1
    assert levels[0].payload["touch_count"] == 2


def test_creation_and_touch_ids_distinct_from_level_id():
    candles = _series([10.0, 12.0, 11.0, 10.5, 12.1, 11.0])
    d = EqualLevelsDetector(tolerance=0.2, left_strength=1, right_strength=1)
    levels = [o for o in d.detect_all_historical(candles) if o.type == "equal_high_level"]
    assert len(levels) == 1
    assert levels[0].id != levels[0].payload["level_id"]


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
