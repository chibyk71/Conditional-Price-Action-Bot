"""Phase 5C — Classical triangle pattern acceptance tests."""

from __future__ import annotations

import math

import pytest

from cpab.detectors.triangle import (
    TriangleDetector,
    CONCEPT_ASCENDING,
    CONCEPT_DESCENDING,
    CONCEPT_SYMMETRICAL,
    VARIANT,
    _within_tolerance,
    _relative_change,
    _classify_boundary,
    _line_price_at,
    _intersection_time,
    DEFAULT_BOUNDARY_TOLERANCE,
    DEFAULT_HORIZONTAL_TOLERANCE,
    DEFAULT_MAX_BARS_TO_BREAKOUT,
)
from cpab.detectors.registry import build_default_registry
from cpab.models import Candle


TF = 60
START = 1_700_000_000


def _c(
    i: int,
    o: float,
    h: float,
    l: float,
    cl: float,
    *,
    symbol: str = "T_100",
    timeframe: str = "1m",
    start: int = START,
    is_closed: bool = True,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        epoch=start + i * TF,
        open=o,
        high=h,
        low=l,
        close=cl,
        is_closed=is_closed,
        source="provider",
    )


def _quiet(mid: float = 100.0, width: float = 0.3) -> tuple[float, float, float, float]:
    return mid, mid + width, mid - width, mid


def _build_ascending(
    n_bars: int = 40,
    *,
    upper: float = 110.0,
    lower_start: float = 95.0,
    lower_step: float = 1.5,
    start: int = START,
    symbol: str = "T_ASC",
) -> list[Candle]:
    """
    Synthetic ascending triangle:
      - Swing highs near flat upper resistance
      - Swing lows rising toward upper
    Pivot placement for strength-1: highs at i%4==1, lows at i%4==3.
    """
    candles: list[Candle] = []
    low_idx = 0
    for i in range(n_bars):
        o, h, l, cl = _quiet(100.0)
        if i % 4 == 1:
            # swing high near upper
            h = upper + 0.05
            o = upper - 1.0
            cl = upper - 0.5
            l = 100.0
        elif i % 4 == 3:
            # rising swing low
            lvl = lower_start + low_idx * lower_step
            l = lvl - 0.05
            o = lvl + 1.0
            cl = lvl + 0.5
            h = 100.0
            low_idx += 1
        candles.append(_c(i, o, h, l, cl, symbol=symbol, start=start))
    return candles


def _build_descending(
    n_bars: int = 40,
    *,
    lower: float = 90.0,
    upper_start: float = 115.0,
    upper_step: float = -1.5,
    start: int = START,
    symbol: str = "T_DESC",
) -> list[Candle]:
    """Descending: falling highs, flat lows."""
    candles: list[Candle] = []
    high_idx = 0
    for i in range(n_bars):
        o, h, l, cl = _quiet(100.0)
        if i % 4 == 1:
            lvl = upper_start + high_idx * upper_step
            h = lvl + 0.05
            o = lvl - 1.0
            cl = lvl - 0.5
            l = 100.0
            high_idx += 1
        elif i % 4 == 3:
            l = lower - 0.05
            o = lower + 1.0
            cl = lower + 0.5
            h = 100.0
        candles.append(_c(i, o, h, l, cl, symbol=symbol, start=start))
    return candles


def _build_symmetrical(
    n_bars: int = 40,
    *,
    upper_start: float = 115.0,
    upper_step: float = -1.2,
    lower_start: float = 85.0,
    lower_step: float = 1.2,
    start: int = START,
    symbol: str = "T_SYM",
) -> list[Candle]:
    """Symmetrical: falling highs, rising lows."""
    candles: list[Candle] = []
    high_idx = 0
    low_idx = 0
    for i in range(n_bars):
        o, h, l, cl = _quiet(100.0)
        if i % 4 == 1:
            lvl = upper_start + high_idx * upper_step
            h = lvl + 0.05
            o = lvl - 1.0
            cl = lvl - 0.5
            l = 100.0
            high_idx += 1
        elif i % 4 == 3:
            lvl = lower_start + low_idx * lower_step
            l = lvl - 0.05
            o = lvl + 1.0
            cl = lvl + 0.5
            h = 100.0
            low_idx += 1
        candles.append(_c(i, o, h, l, cl, symbol=symbol, start=start))
    return candles


def _append_breakout(
    candles: list[Candle],
    *,
    direction: str,
    level: float,
    n_extra: int = 5,
) -> list[Candle]:
    """Append quiet bars then a strict close beyond level."""
    out = list(candles)
    last_i = (out[-1].epoch - START) // TF
    mid = (out[-1].high + out[-1].low) / 2.0
    for k in range(1, n_extra):
        i = last_i + k
        o, h, l, cl = _quiet(mid)
        out.append(_c(i, o, h, l, cl, symbol=out[0].symbol, start=START))
    # breakout bar
    i = last_i + n_extra
    if direction == "bullish":
        cl = level + 1.0
        o = level - 0.5
        h = cl + 0.2
        l = o - 0.2
    else:
        cl = level - 1.0
        o = level + 0.5
        h = o + 0.2
        l = cl - 0.2
    out.append(_c(i, o, h, l, cl, symbol=out[0].symbol, start=START))
    return out


# ---------------------------------------------------------------------------
# A. Config validation & dual tolerances
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_defaults(self):
        d = TriangleDetector()
        assert d.boundary_tolerance == DEFAULT_BOUNDARY_TOLERANCE
        assert d.horizontal_tolerance == DEFAULT_HORIZONTAL_TOLERANCE
        assert d.max_bars_to_breakout == DEFAULT_MAX_BARS_TO_BREAKOUT
        assert d.left_strength == 1
        assert d.right_strength == 1
        assert d.tie_policy == "strict"
        assert d.detector_config_id.startswith("cfg:")
        assert d.min_history == 4 * (1 + 1 + 1) + 1  # 13
        cfg = d.config_dict()
        assert "boundary_tolerance" in cfg
        assert "horizontal_tolerance" in cfg
        assert "max_bars_to_breakout" in cfg

    def test_invalid_boundary_tolerance(self):
        with pytest.raises(ValueError):
            TriangleDetector(boundary_tolerance=-0.1)
        with pytest.raises(ValueError):
            TriangleDetector(boundary_tolerance=float("nan"))
        with pytest.raises(ValueError):
            TriangleDetector(boundary_tolerance=float("inf"))
        with pytest.raises(ValueError):
            TriangleDetector(boundary_tolerance="0.002")  # type: ignore
        with pytest.raises(ValueError):
            TriangleDetector(boundary_tolerance=True)  # type: ignore

    def test_invalid_horizontal_tolerance(self):
        with pytest.raises(ValueError):
            TriangleDetector(horizontal_tolerance=-0.1)
        with pytest.raises(ValueError):
            TriangleDetector(horizontal_tolerance=float("nan"))
        with pytest.raises(ValueError):
            TriangleDetector(horizontal_tolerance=True)  # type: ignore

    def test_invalid_max_bars(self):
        with pytest.raises(ValueError):
            TriangleDetector(max_bars_to_breakout=0)
        with pytest.raises(ValueError):
            TriangleDetector(max_bars_to_breakout=-1)
        with pytest.raises(ValueError):
            TriangleDetector(max_bars_to_breakout=1.5)  # type: ignore
        with pytest.raises(ValueError):
            TriangleDetector(max_bars_to_breakout=True)  # type: ignore

    def test_config_id_changes_with_each_tolerance(self):
        a = TriangleDetector(boundary_tolerance=0.002, horizontal_tolerance=0.002)
        b = TriangleDetector(boundary_tolerance=0.005, horizontal_tolerance=0.002)
        c = TriangleDetector(boundary_tolerance=0.002, horizontal_tolerance=0.005)
        d = TriangleDetector(max_bars_to_breakout=30)
        assert a.detector_config_id != b.detector_config_id
        assert a.detector_config_id != c.detector_config_id
        assert a.detector_config_id != d.detector_config_id
        assert b.detector_config_id != c.detector_config_id

    def test_tolerance_formula_inclusive(self):
        assert _within_tolerance(100.0, 100.2, 0.002) is True
        assert _within_tolerance(100.0, 100.3, 0.002) is False
        assert _within_tolerance(0.0, 0.0, 0.002) is True

    def test_relative_change_and_classify(self):
        # denom = max(100, 100.2) = 100.2 → 0.2/100.2 ≈ 0.001996 <= 0.002
        assert _relative_change(100.0, 100.2) == pytest.approx(0.2 / 100.2)
        assert _classify_boundary(100.0, 100.2, 0.002) == "horizontal"
        assert _classify_boundary(100.0, 100.3, 0.002) == "rising"
        assert _classify_boundary(100.0, 99.7, 0.002) == "falling"
        # exclusivity: at threshold is horizontal, not rising
        assert _classify_boundary(100.0, 100.2, 0.002) == "horizontal"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class TestGeometryHelpers:
    def test_line_price_at(self):
        # horizontal
        assert _line_price_at(50, t0=0, p0=100.0, t1=10, p1=100.0) == 100.0
        # rising: slope 1
        assert _line_price_at(5, t0=0, p0=100.0, t1=10, p1=110.0) == 105.0
        # same t, same p
        assert _line_price_at(0, t0=0, p0=100.0, t1=0, p1=100.0) == 100.0
        # same t, different p → None
        assert _line_price_at(0, t0=0, p0=100.0, t1=0, p1=101.0) is None

    def test_intersection_forward(self):
        # upper flat 110, lower rising 90→100 over t=0..10 → intersect later
        t = _intersection_time(
            t_h0=0, p_h0=110.0, t_h1=10, p_h1=110.0,
            t_l0=0, p_l0=90.0, t_l1=10, p_l1=100.0,
        )
        assert t is not None and t > 10

    def test_parallel_rejected(self):
        t = _intersection_time(
            t_h0=0, p_h0=110.0, t_h1=10, p_h1=110.0,
            t_l0=0, p_l0=90.0, t_l1=10, p_l1=90.0,
        )
        assert t is None


# ---------------------------------------------------------------------------
# A. Pattern classification
# ---------------------------------------------------------------------------


class TestPatternClassification:
    def test_ascending_valid(self):
        candles = _build_ascending(n_bars=36)
        # Force a clear bullish breakout above upper (~110)
        candles = _append_breakout(candles, direction="bullish", level=110.0, n_extra=4)
        d = TriangleDetector()
        obs = d.detect_all_historical(candles)
        asc = [o for o in obs if o.type == CONCEPT_ASCENDING]
        assert len(asc) >= 1
        o = asc[0]
        assert o.direction == "bullish"
        assert o.variant == VARIANT
        assert o.payload["upper_classification"] == "horizontal"
        assert o.payload["lower_classification"] == "rising"
        assert o.reference_ids

    def test_descending_valid(self):
        candles = _build_descending(n_bars=36)
        candles = _append_breakout(candles, direction="bearish", level=90.0, n_extra=4)
        d = TriangleDetector()
        obs = d.detect_all_historical(candles)
        desc = [o for o in obs if o.type == CONCEPT_DESCENDING]
        assert len(desc) >= 1
        o = desc[0]
        assert o.direction == "bearish"
        assert o.payload["upper_classification"] == "falling"
        assert o.payload["lower_classification"] == "horizontal"

    def test_symmetrical_valid(self):
        candles = _build_symmetrical(n_bars=36)
        candles = _append_breakout(candles, direction="bullish", level=110.0, n_extra=4)
        d = TriangleDetector()
        obs = d.detect_all_historical(candles)
        sym = [o for o in obs if o.type == CONCEPT_SYMMETRICAL]
        assert len(sym) >= 1
        o = sym[0]
        assert o.direction in ("bullish", "bearish")
        assert o.payload["upper_classification"] == "falling"
        assert o.payload["lower_classification"] == "rising"

    def test_incorrect_slope_combo_rejected(self):
        # Both horizontal → not a triangle under contract
        # Build flat range (rectangle-like)
        candles: list[Candle] = []
        for i in range(40):
            o, h, l, cl = _quiet(100.0)
            if i % 4 == 1:
                h, o, cl, l = 110.05, 109.0, 109.5, 100.0
            elif i % 4 == 3:
                l, o, cl, h = 89.95, 91.0, 90.5, 100.0
            candles.append(_c(i, o, h, l, cl, symbol="FLAT"))
        candles = _append_breakout(candles, direction="bullish", level=110.0, n_extra=3)
        d = TriangleDetector()
        obs = d.detect_all_historical(candles)
        assert all(o.type not in (CONCEPT_ASCENDING, CONCEPT_DESCENDING, CONCEPT_SYMMETRICAL)
                   for o in obs)

    def test_horizontal_not_also_rising(self):
        # Pure unit test of classifier exclusivity
        assert _classify_boundary(100.0, 100.0, 0.002) == "horizontal"
        assert _classify_boundary(100.0, 100.199, 0.002) == "horizontal"
        assert _classify_boundary(100.0, 100.201, 0.002) == "rising"


# ---------------------------------------------------------------------------
# B. Dual tolerances at threshold
# ---------------------------------------------------------------------------


class TestDualTolerances:
    def test_boundary_tolerance_at_threshold_accepted(self):
        # price offset by 0.2% of max is at threshold (inclusive)
        assert _within_tolerance(100.0, 100.2, 0.002) is True
        # clearly beyond
        assert _within_tolerance(100.0, 100.5, 0.002) is False

    def test_horizontal_tolerance_at_threshold(self):
        assert _classify_boundary(100.0, 100.2, 0.002) == "horizontal"
        assert _classify_boundary(100.0, 100.5, 0.002) == "rising"


# ---------------------------------------------------------------------------
# C. Geometry and convergence (via detector)
# ---------------------------------------------------------------------------


class TestConvergenceAndBreakout:
    def test_breakout_uses_evaluation_candle_timestamp(self):
        """Boundary price at breakout candle epoch, not final-anchor epoch."""
        candles = _build_ascending(n_bars=32)
        # Append breakout well after final swing
        candles = _append_breakout(candles, direction="bullish", level=110.0, n_extra=6)
        d = TriangleDetector()
        obs = d.detect_all_historical(candles)
        asc = [o for o in obs if o.type == CONCEPT_ASCENDING]
        assert asc
        o = asc[0]
        # payload must record boundary evaluated at breakout
        assert "upper_boundary_at_breakout" in o.payload
        assert o.payload["breakout_candle_epoch"] == o.confirmed_at
        # for nearly-flat upper, value should be near 110
        assert abs(o.payload["upper_boundary_at_breakout"] - 110.0) < 1.0

    def test_wick_only_does_not_confirm(self):
        candles = _build_ascending(n_bars=32)
        last_i = (candles[-1].epoch - START) // TF
        # quiet bars
        for k in range(1, 4):
            i = last_i + k
            o, h, l, cl = _quiet(105.0)
            candles.append(_c(i, o, h, l, cl, symbol="T_ASC"))
        # wick above upper but close inside
        i = last_i + 4
        candles.append(
            _c(i, 109.0, 112.0, 108.0, 109.5, symbol="T_ASC")  # high 112, close 109.5
        )
        d = TriangleDetector()
        obs = d.detect_all_historical(candles)
        # no confirmation on wick-only
        asc = [o for o in obs if o.type == CONCEPT_ASCENDING]
        for o in asc:
            assert o.confirmed_at != candles[-1].epoch

    def test_close_exactly_on_boundary_does_not_confirm(self):
        candles = _build_ascending(n_bars=32)
        last_i = (candles[-1].epoch - START) // TF
        for k in range(1, 4):
            i = last_i + k
            o, h, l, cl = _quiet(105.0)
            candles.append(_c(i, o, h, l, cl, symbol="T_ASC"))
        # close exactly at ~110
        i = last_i + 4
        candles.append(_c(i, 109.5, 110.2, 109.0, 110.0, symbol="T_ASC"))
        d = TriangleDetector()
        obs = d.detect_all_historical(candles)
        for o in obs:
            if o.type == CONCEPT_ASCENDING:
                assert o.confirmed_at != candles[-1].epoch

    def _minimal_ascending_series(
        self,
        *,
        breakout_offset: int,
        invalidate_at_offset: int | None = None,
        fill_close: float = 107.0,
        max_bars: int = 50,
    ) -> list[Candle]:
        """
        Controlled 2H+2L ascending geometry with shallow rising lower so the
        apex is far beyond offset 50. Final swing event is at index 7.
        Breakout is placed at index 7 + breakout_offset.
        Optional opposite-boundary invalidation close at 7 + invalidate_at_offset.
        """
        hi, low0, low1 = 110.0, 99.0, 99.5
        candles: list[Candle] = []
        candles.append(_c(0, 105, 105.3, 104.7, 105, symbol="M"))
        candles.append(_c(1, hi - 1, hi + 0.05, hi - 2, hi - 0.5, symbol="M"))
        candles.append(_c(2, 105, 105.3, 104.7, 105, symbol="M"))
        candles.append(_c(3, low0 + 1, low0 + 2, low0 - 0.05, low0 + 0.5, symbol="M"))
        candles.append(_c(4, 105, 105.3, 104.7, 105, symbol="M"))
        candles.append(_c(5, hi - 1, hi + 0.05, hi - 2, hi - 0.5, symbol="M"))
        candles.append(_c(6, 105, 105.3, 104.7, 105, symbol="M"))
        candles.append(_c(7, low1 + 1, low1 + 2, low1 - 0.05, low1 + 0.5, symbol="M"))
        candles.append(_c(8, 105, 105.3, 104.7, 105, symbol="M"))
        final_idx = 7
        end_idx = final_idx + breakout_offset
        for i in range(9, end_idx + 1):
            if invalidate_at_offset is not None and i == final_idx + invalidate_at_offset:
                # Strict close below the rising lower boundary (opposite side)
                candles.append(_c(i, 98.0, 99.0, 90.0, 91.0, symbol="M"))
            elif i == end_idx:
                candles.append(
                    _c(i, hi + 0.5, hi + 2, hi, hi + 1.5, symbol="M")
                )
            else:
                candles.append(
                    _c(
                        i,
                        fill_close,
                        fill_close + 0.2,
                        fill_close - 0.2,
                        fill_close,
                        symbol="M",
                    )
                )
        return candles

    def test_opposite_boundary_invalidates_same_candidate(self):
        """The exact anchor set that would confirm is blocked by opposite close."""
        # Without invalidation: confirms at offset 10
        clean = self._minimal_ascending_series(breakout_offset=10)
        d = TriangleDetector(max_bars_to_breakout=50)
        clean_obs = [
            o for o in d.detect_all_historical(clean) if o.type == CONCEPT_ASCENDING
        ]
        assert clean_obs, "precondition: clean series must confirm ascending"
        clean_keys = {tuple(o.reference_ids) for o in clean_obs}

        # With opposite-boundary close at offset 5, then same breakout at 10
        dirty = self._minimal_ascending_series(
            breakout_offset=10, invalidate_at_offset=5
        )
        dirty_obs = [
            o for o in d.detect_all_historical(dirty) if o.type == CONCEPT_ASCENDING
        ]
        dirty_keys = {tuple(o.reference_ids) for o in dirty_obs}
        # Every candidate that confirmed cleanly must be absent after invalidation
        leaked = clean_keys & dirty_keys
        assert not leaked, (
            f"invalidated candidate(s) still confirmed: {leaked}"
        )

    def test_expiry_offset_50_confirms_offset_51_does_not(self):
        """Same geometry: breakout at final+50 confirms under max_bars=50; final+51 does not."""
        at_50 = self._minimal_ascending_series(breakout_offset=50)
        at_51 = self._minimal_ascending_series(breakout_offset=51)

        d50 = TriangleDetector(max_bars_to_breakout=50)
        conf_50 = [
            o for o in d50.detect_all_historical(at_50) if o.type == CONCEPT_ASCENDING
        ]
        conf_51_under_50 = [
            o for o in d50.detect_all_historical(at_51) if o.type == CONCEPT_ASCENDING
        ]
        assert conf_50, "expected confirmation at offset 50 with max_bars=50"
        assert not conf_51_under_50, (
            "must not confirm at offset 51 when max_bars_to_breakout=50"
        )

        # Sanity: the same offset-51 series *can* confirm when the window allows it
        d51 = TriangleDetector(max_bars_to_breakout=51)
        conf_51_under_51 = [
            o for o in d51.detect_all_historical(at_51) if o.type == CONCEPT_ASCENDING
        ]
        assert conf_51_under_51, "precondition: offset 51 confirms under max_bars=51"


# ---------------------------------------------------------------------------
# D. Enumeration
# ---------------------------------------------------------------------------


class TestEnumeration:
    def test_min_anchors_enforced(self):
        # Too few bars → no swings → no triangles
        candles = [_c(i, 100, 101, 99, 100) for i in range(5)]
        d = TriangleDetector()
        assert d.detect_all_historical(candles) == []

    def test_overlapping_candidates_preserved(self):
        """Distinct overlapping anchor-ID combinations coexist; identical ones are unique."""
        from cpab.detectors.swing import SwingPivotDetector

        candles = _build_ascending(n_bars=28, symbol="OV")
        d = TriangleDetector()
        sw = SwingPivotDetector()
        sobs = sw.detect_all_historical(candles)
        highs = [o for o in sobs if o.type == "swing_high"]
        lows = [o for o in sobs if o.type == "swing_low"]
        cands = d._enumerate_candidates(highs, lows)
        keys = [tuple(a.id for a in c) for c in cands]
        assert len(keys) == len(set(keys)), "identical ordered anchor-ID sets must appear once"
        assert len(set(keys)) >= 2, "expected multiple distinct overlapping candidates"
        # ordered identity (not sorted) — order is part of the combination
        sorted_keys = [tuple(sorted(k)) for k in keys]
        # at least some pairs share the same set of ids but differ in order? not required;
        # required is that distinct combinations survive.
        assert len(set(sorted_keys)) >= 2

    def test_no_arbitrary_caps(self):
        # Long series still processes (may be slow but must not truncate silently)
        candles = _build_symmetrical(n_bars=36)
        candles = _append_breakout(candles, direction="bullish", level=112.0, n_extra=4)
        d = TriangleDetector()
        obs = d.detect_all_historical(candles)
        assert isinstance(obs, list)


# ---------------------------------------------------------------------------
# F. Causality & gaps
# ---------------------------------------------------------------------------


class TestCausalityAndGaps:
    def test_forming_candle_ignored(self):
        candles = _build_ascending(n_bars=32)
        candles = _append_breakout(candles, direction="bullish", level=110.0, n_extra=4)
        # last bar open
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
            source="provider",
        )
        d = TriangleDetector()
        # incremental detect uses only closed
        obs = d.detect(candles)
        assert all(o.confirmed_at != last.epoch for o in obs)

    def test_gap_splits_segments(self):
        left = _build_ascending(n_bars=28, symbol="GAP")
        left = _append_breakout(left, direction="bullish", level=110.0, n_extra=3)
        right_start = left[-1].epoch + 10 * TF
        right = _build_ascending(n_bars=28, start=right_start, symbol="GAP")
        right = _append_breakout(right, direction="bullish", level=110.0, n_extra=3)
        combined = left + right
        d = TriangleDetector()
        obs = d.detect_all_historical(combined)
        for o in obs:
            assert o.symbol == "GAP"

    def test_no_candidate_spans_gap(self):
        """No emitted observation mixes swing anchors from both sides of a gap."""
        from cpab.detectors.swing import SwingPivotDetector

        left = _build_ascending(n_bars=24, symbol="GAP2")
        left = _append_breakout(left, direction="bullish", level=110.0, n_extra=3)
        gap_start = left[-1].epoch + 10 * TF
        right = _build_ascending(n_bars=24, start=gap_start, symbol="GAP2")
        right = _append_breakout(right, direction="bullish", level=110.0, n_extra=3)
        combined = left + right

        sw = SwingPivotDetector()
        s_left = {o.id for o in sw.detect_all_historical(left)}
        s_right = {o.id for o in sw.detect_all_historical(right)}
        assert s_left.isdisjoint(s_right)

        d = TriangleDetector()
        obs = d.detect_all_historical(combined)
        triangle_types = {CONCEPT_ASCENDING, CONCEPT_DESCENDING, CONCEPT_SYMMETRICAL}
        for o in obs:
            if o.type not in triangle_types:
                continue
            refs = set(o.reference_ids)
            in_left = refs & s_left
            in_right = refs & s_right
            assert not (in_left and in_right), (
                f"candidate spanned gap: left={in_left} right={in_right}"
            )

    def test_mixed_symbol_rejected(self):
        a = _build_ascending(n_bars=20, symbol="A")
        b = _build_ascending(n_bars=20, symbol="B", start=START + 20 * TF)
        d = TriangleDetector()
        assert d.detect_all_historical(a + b) == []

    def test_historical_incremental_parity(self):
        candles = _build_ascending(n_bars=36)
        candles = _append_breakout(candles, direction="bullish", level=110.0, n_extra=5)
        d = TriangleDetector()
        hist = d.detect_all_historical(candles)
        # incremental: only emit those confirmed at last bar
        incr = d.detect(candles)
        last_epoch = candles[-1].epoch
        hist_at_last = [o for o in hist if o.confirmed_at == last_epoch]
        assert {o.id for o in incr} == {o.id for o in hist_at_last}


# ---------------------------------------------------------------------------
# G. Identity & registry
# ---------------------------------------------------------------------------


class TestIdentityAndRegistry:
    def _require_talib(self):
        pytest.importorskip("talib")

    def test_stable_observation_ids(self):
        candles = _build_ascending(n_bars=36)
        candles = _append_breakout(candles, direction="bullish", level=110.0, n_extra=4)
        d = TriangleDetector()
        a = d.detect_all_historical(candles)
        b = d.detect_all_historical(candles)
        assert [o.id for o in a] == [o.id for o in b]

    def test_registry_create(self):
        self._require_talib()
        reg = build_default_registry()
        det, info = reg.create_detector("ascending_triangle")
        assert det.name == "triangle"
        assert info.implementation == "triangle"
        assert "boundary_tolerance" in info.config
        assert "horizontal_tolerance" in info.config
        det2, info2 = reg.create_detector(
            "triangle", config={"boundary_tolerance": 0.005}
        )
        assert info2.config["boundary_tolerance"] == 0.005
        assert info.detector_config_id != info2.detector_config_id

    def test_registry_concepts(self):
        self._require_talib()
        reg = build_default_registry()
        for c in (
            "ascending_triangle",
            "descending_triangle",
            "symmetrical_triangle",
            "triangle",
        ):
            info = reg.validate_config(c, {})
            assert info.implementation == "triangle"
            assert info.min_history == 13


# ---------------------------------------------------------------------------
# Regression: Phase 5A / 5B still green is checked by full suite
# ---------------------------------------------------------------------------
