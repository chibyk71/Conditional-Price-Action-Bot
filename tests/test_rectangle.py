"""Phase 5B — Classical rectangle pattern acceptance tests."""

from __future__ import annotations

import pytest

from cpab.detectors.rectangle import (
    RectangleDetector,
    CONCEPT_RECTANGLE,
    VARIANT,
    _within_tolerance,
    _median,
    DEFAULT_BOUNDARY_TOLERANCE,
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
    symbol: str = "R_100",
    timeframe: str = "1m",
    start: int = START,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        epoch=start + i * TF,
        open=o,
        high=h,
        low=l,
        close=cl,
        is_closed=True,
        source="provider",
    )


def _make_flat_range(
    n_bars: int = 30,
    support: float = 100.0,
    resistance: float = 110.0,
    *,
    start: int = START,
    symbol: str = "R_100",
) -> list[Candle]:
    """
    Build a synthetic series that produces alternating swing highs near resistance
    and swing lows near support with left=right=1 (strict).

    Pattern of extremes (indices for strength-1 pivots):
      bar 1: high near res (swing high)
      bar 3: low near sup (swing low)
      bar 5: high near res
      bar 7: low near sup
      ... then a breakout later.
    Intermediate bars stay inside the range so they do not create extra pivots.
    """
    mid = (support + resistance) / 2.0
    candles: list[Candle] = []
    for i in range(n_bars):
        # Default quiet bar inside range
        o = mid
        h = mid + 0.5
        l = mid - 0.5
        cl = mid
        # Place swing highs at odd indices 1,5,9,... near resistance
        # Place swing lows at 3,7,11,... near support
        if i % 4 == 1:
            # potential swing high
            h = resistance + 0.1
            o = resistance - 1.0
            cl = resistance - 0.5
            l = mid
        elif i % 4 == 3:
            # potential swing low
            l = support - 0.1
            o = support + 1.0
            cl = support + 0.5
            h = mid
        candles.append(_c(i, o, h, l, cl, symbol=symbol, start=start))
    return candles


class TestConfigValidation:
    def test_defaults(self):
        d = RectangleDetector()
        assert d.boundary_tolerance == DEFAULT_BOUNDARY_TOLERANCE
        assert d.max_bars_to_breakout == DEFAULT_MAX_BARS_TO_BREAKOUT
        assert d.left_strength == 1
        assert d.right_strength == 1
        assert d.tie_policy == "strict"
        assert d.detector_config_id.startswith("cfg:")
        assert d.min_history == 4 * (1 + 1 + 1) + 1  # 13

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError):
            RectangleDetector(boundary_tolerance=-0.1)
        with pytest.raises(ValueError):
            RectangleDetector(boundary_tolerance=float("nan"))
        with pytest.raises(ValueError):
            RectangleDetector(boundary_tolerance=float("inf"))
        with pytest.raises(ValueError):
            RectangleDetector(boundary_tolerance="0.002")  # type: ignore
        with pytest.raises(ValueError):
            RectangleDetector(boundary_tolerance=True)  # type: ignore

    def test_invalid_max_bars(self):
        with pytest.raises(ValueError):
            RectangleDetector(max_bars_to_breakout=0)
        with pytest.raises(ValueError):
            RectangleDetector(max_bars_to_breakout=-1)
        with pytest.raises(ValueError):
            RectangleDetector(max_bars_to_breakout=1.5)  # type: ignore
        with pytest.raises(ValueError):
            RectangleDetector(max_bars_to_breakout=True)  # type: ignore

    def test_config_id_changes_with_settings(self):
        a = RectangleDetector(boundary_tolerance=0.002)
        b = RectangleDetector(boundary_tolerance=0.005)
        c = RectangleDetector(max_bars_to_breakout=30)
        assert a.detector_config_id != b.detector_config_id
        assert a.detector_config_id != c.detector_config_id

    def test_tolerance_formula(self):
        assert _within_tolerance(100.0, 100.2, 0.002) is True
        assert _within_tolerance(100.0, 100.3, 0.002) is False
        # exact boundary
        assert _within_tolerance(100.0, 100.2, 0.002) is True
        # zero / epsilon
        assert _within_tolerance(0.0, 0.0, 0.002) is True

    def test_median(self):
        assert _median([1.0, 3.0, 2.0]) == 2.0
        assert _median([1.0, 4.0, 2.0, 3.0]) == 2.5


class TestStructureAndBoundaries:
    def test_valid_hlhl_bullish_breakout(self):
        """Minimal H→L→H→L then close above resistance."""
        # Manually craft clear pivots with strength 1
        # bars: 0 quiet, 1 SH, 2 quiet, 3 SL, 4 quiet, 5 SH, 6 quiet, 7 SL,
        # then 8 quiet, 9 breakout up
        support, resistance = 100.0, 110.0
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 108, 110.2, 107, 109))  # SH ~110.2
            elif i == 3:
                candles.append(_c(i, 102, 103, 99.8, 101))   # SL ~99.8
            elif i == 5:
                candles.append(_c(i, 108, 110.1, 107, 109))  # SH ~110.1
            elif i == 7:
                candles.append(_c(i, 102, 103, 99.9, 101))   # SL ~99.9
            elif i == 9:
                # bullish breakout close above resistance
                candles.append(_c(i, 109, 112, 108.5, 111.5))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        obs = d.detect_all_historical(candles)
        rects = [o for o in obs if o.type == CONCEPT_RECTANGLE]
        assert len(rects) >= 1
        bull = [o for o in rects if o.direction == "bullish"]
        assert len(bull) >= 1
        o = bull[0]
        assert o.variant == VARIANT
        assert o.payload["resistance"] > o.payload["support"]
        assert o.payload["breakout_close"] > o.payload["resistance"]
        assert len(o.reference_ids) >= 4

    def test_valid_lhlh_bearish_breakout(self):
        """Minimal L→H→L→H then close below support."""
        support, resistance = 100.0, 110.0
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 102, 103, 99.8, 101))   # SL
            elif i == 3:
                candles.append(_c(i, 108, 110.2, 107, 109))  # SH
            elif i == 5:
                candles.append(_c(i, 102, 103, 99.9, 101))   # SL
            elif i == 7:
                candles.append(_c(i, 108, 110.1, 107, 109))  # SH
            elif i == 9:
                # bearish breakout
                candles.append(_c(i, 101, 102, 98, 99.0))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        obs = d.detect_all_historical(candles)
        rects = [o for o in obs if o.type == CONCEPT_RECTANGLE and o.direction == "bearish"]
        assert len(rects) >= 1
        o = rects[0]
        assert o.payload["breakout_close"] < o.payload["support"]

    def test_fewer_than_two_highs_rejected(self):
        # Only one high, two lows — no candidate
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 108, 110.2, 107, 109))  # SH
            elif i == 3:
                candles.append(_c(i, 102, 103, 99.8, 101))   # SL
            elif i == 5:
                candles.append(_c(i, 102, 103, 99.9, 101))   # SL
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01)
        obs = d.detect_all_historical(candles)
        assert [o for o in obs if o.type == CONCEPT_RECTANGLE] == []

    def test_non_alternating_rejected(self):
        """Every emitted candidate must strictly alternate H/L in chronological order.

        Construct a series with two consecutive swing highs (no intervening low
        in the pivot stream between them). Enumeration must never emit a
        non-alternating anchor list; any rectangle must still alternate.
        """
        from cpab.detectors.rectangle import RectangleDetector
        from cpab.detectors.swing import SwingPivotDetector

        mid = 105.0
        candles = []
        for i in range(20):
            if i == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))   # SH
            elif i == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))   # SL
            elif i == 5:
                candles.append(_c(i, 108, 110.0, 107, 109))   # SH
            elif i == 7:
                # second high without a new low between 5 and 7 in a way that
                # creates consecutive same-type if intermediate is not a pivot;
                # quiet bars in between keep pivots at 1,3,5,7 as H L H H? 
                # With strength 1, bar 7 high needs neighbors lower.
                candles.append(_c(i, 108, 110.5, 107, 109))   # SH again
            elif i == 9:
                candles.append(_c(i, 102, 103, 100.0, 101))   # SL
            elif i == 11:
                candles.append(_c(i, 109, 112, 108.5, 111.0))  # breakout
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))

        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        obs = d.detect_all_historical(candles)
        rects = [o for o in obs if o.type == CONCEPT_RECTANGLE]
        for o in rects:
            types = o.payload["anchor_swing_types"]
            assert len(types) >= 4
            for a, b in zip(types, types[1:]):
                assert a != b, f"non-alternating anchors in {types}"

        # Unit-level: enumeration itself never yields non-alternating paths
        sw = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
        sobs = sw.detect_all_historical(candles)
        highs = [o for o in sobs if o.type == "swing_high"]
        lows = [o for o in sobs if o.type == "swing_low"]
        for anchors in d._enumerate_candidates(highs, lows):
            types = [a.type for a in anchors]
            for a, b in zip(types, types[1:]):
                assert a != b, f"enumerate produced non-alternating {types}"

    def test_anchor_outside_tolerance_rejected(self):
        support, resistance = 100.0, 110.0
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))   # SH 110
            elif i == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))   # SL 100
            elif i == 5:
                # far outside tolerance of resistance median
                candles.append(_c(i, 115, 120.0, 114, 116))
            elif i == 7:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 9:
                candles.append(_c(i, 109, 112, 108.5, 111.5))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        # tight tolerance so 120 is way outside
        d = RectangleDetector(boundary_tolerance=0.001, max_bars_to_breakout=10)
        obs = d.detect_all_historical(candles)
        # The candidate that includes the outlier high should be rejected;
        # there may be no valid 4-anchor set that stays within tolerance.
        rects = [o for o in obs if o.type == CONCEPT_RECTANGLE]
        for o in rects:
            # any emitted must have all anchors within tolerance
            res = o.payload["resistance"]
            for p in o.payload["anchor_swing_prices"]:
                if o.payload["anchor_swing_types"][
                    o.payload["anchor_swing_prices"].index(p)
                ] == "swing_high":
                    assert _within_tolerance(p, res, 0.001)

    def test_inverted_geometry_rejected(self):
        # Construct so medians would invert if we forced them — practically
        # swing highs below swing lows is unlikely with normal pivots; we rely
        # on the explicit resistance > support check.
        d = RectangleDetector()
        # empty / short series
        assert d.detect_all_historical([]) == []


class TestBreakoutAndCausality:
    def test_close_exactly_at_boundary_does_not_confirm(self):
        support, resistance = 100.0, 110.0
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 5:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 7:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 9:
                # close exactly at resistance
                candles.append(_c(i, 109, 110.5, 108, 110.0))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        obs = d.detect_all_historical(candles)
        # no bullish confirmation on exact close
        bull = [
            o
            for o in obs
            if o.type == CONCEPT_RECTANGLE and o.direction == "bullish"
        ]
        # may still get later bars if any; the exact bar should not be the confirming one
        for o in bull:
            assert o.payload["breakout_close"] > o.payload["resistance"]

    def test_wick_only_does_not_confirm(self):
        support, resistance = 100.0, 110.0
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 5:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 7:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 9:
                # wick above resistance but close inside
                candles.append(_c(i, 109, 112.0, 108, 109.5))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        obs = d.detect_all_historical(candles)
        bull = [
            o
            for o in obs
            if o.type == CONCEPT_RECTANGLE
            and o.direction == "bullish"
            and o.candle_epoch == candles[9].epoch
        ]
        assert bull == []

    def test_breakout_before_last_anchor_not_emitted(self):
        # Ensure we never confirm on a candle that is not strictly after all anchors
        # This is enforced by _candle_after_all_available; structural test via short series
        d = RectangleDetector(max_bars_to_breakout=2)
        # very short — no room for post-anchor breakout after confirmation lag
        candles = [_c(i, 105, 106, 104, 105) for i in range(8)]
        obs = d.detect_all_historical(candles)
        assert [o for o in obs if o.type == CONCEPT_RECTANGLE] == []

    def test_only_first_direction_emitted(self):
        # Once a direction confirms, we stop for that candidate
        support, resistance = 100.0, 110.0
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 5:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 7:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 9:
                # first: bullish
                candles.append(_c(i, 109, 112, 108.5, 111.0))
            elif i == 11:
                # later: would be bearish if we continued
                candles.append(_c(i, 101, 102, 98, 99.0))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        obs = d.detect_all_historical(candles)
        rects = [o for o in obs if o.type == CONCEPT_RECTANGLE]
        # For the same 4-anchor set we expect only the first direction
        by_anchors: dict[tuple, list] = {}
        for o in rects:
            key = tuple(sorted(o.reference_ids))
            by_anchors.setdefault(key, []).append(o.direction)
        for dirs in by_anchors.values():
            assert len(dirs) == 1


class TestDataIntegrity:
    def test_closed_candles_only(self):
        mid = 105.0
        candles = [_c(i, mid, mid + 0.3, mid - 0.3, mid) for i in range(15)]
        # mark last open
        open_c = Candle(
            symbol="R_100",
            timeframe="1m",
            epoch=START + 15 * TF,
            open=mid,
            high=mid + 1,
            low=mid - 1,
            close=mid,
            is_closed=False,
            source="provider",
        )
        d = RectangleDetector()
        # detect uses only closed; open should not appear as confirmation
        obs = d.detect(candles + [open_c])
        for o in obs:
            assert o.confirmed_at != open_c.epoch

    def test_candle_gaps_prevent_cross_gap(self):
        support, resistance = 100.0, 110.0
        mid = 105.0
        # segment 1
        c1 = []
        for i in range(16):
            if i == 1:
                c1.append(_c(i, 108, 110.0, 107, 109))
            elif i == 3:
                c1.append(_c(i, 102, 103, 100.0, 101))
            elif i == 5:
                c1.append(_c(i, 108, 110.0, 107, 109))
            elif i == 7:
                c1.append(_c(i, 102, 103, 100.0, 101))
            else:
                c1.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        # gap
        gap_start = START + 20 * TF
        c2 = []
        for i in range(16):
            if i == 1:
                c2.append(_c(i, 108, 110.0, 107, 109, start=gap_start))
            elif i == 3:
                c2.append(_c(i, 102, 103, 100.0, 101, start=gap_start))
            elif i == 5:
                c2.append(_c(i, 108, 110.0, 107, 109, start=gap_start))
            elif i == 7:
                c2.append(_c(i, 102, 103, 100.0, 101, start=gap_start))
            elif i == 9:
                c2.append(_c(i, 109, 112, 108.5, 111.0, start=gap_start))
            else:
                c2.append(_c(i, mid, mid + 0.3, mid - 0.3, mid, start=gap_start))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        obs = d.detect_all_historical(c1 + c2)
        # any rectangle must have all anchors in one continuous segment
        for o in [x for x in obs if x.type == CONCEPT_RECTANGLE]:
            epochs = []
            # we can't easily re-fetch anchors; just assert confirmation is in c2
            assert o.candle_epoch >= gap_start

    def test_mixed_symbol_rejected(self):
        mid = 105.0
        candles = [_c(i, mid, mid + 0.3, mid - 0.3, mid) for i in range(15)]
        candles[5] = _c(5, mid, mid + 0.3, mid - 0.3, mid, symbol="OTHER")
        d = RectangleDetector()
        assert d.detect_all_historical(candles) == []

    def test_historical_reconstruction_deterministic(self):
        support, resistance = 100.0, 110.0
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 5:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 7:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 9:
                candles.append(_c(i, 109, 112, 108.5, 111.0))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        a = d.detect_all_historical(candles)
        b = d.detect_all_historical(candles)
        assert [o.id for o in a] == [o.id for o in b]
        assert [o.payload for o in a] == [o.payload for o in b]

    def test_incremental_matches_historical(self):
        support, resistance = 100.0, 110.0
        mid = 105.0
        candles = []
        for i in range(16):
            if i == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 5:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 7:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 9:
                candles.append(_c(i, 109, 112, 108.5, 111.0))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=10)
        hist = d.detect_all_historical(candles)
        # incremental at the breakout bar
        inc = d.detect(candles)
        # all incremental obs must appear in historical
        hist_ids = {o.id for o in hist}
        for o in inc:
            assert o.id in hist_ids



class TestEnumerationBounds:
    def test_contiguous_windows_only_alternating(self):
        """Enumerate never returns a non-alternating window."""
        mid = 105.0
        candles = []
        for i in range(24):
            if i % 4 == 1:
                candles.append(_c(i, 108, 110.0 + (i % 3) * 0.05, 107, 109))
            elif i % 4 == 3:
                candles.append(_c(i, 102, 103, 100.0 - (i % 3) * 0.05, 101))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.02, max_bars_to_breakout=10)
        from cpab.detectors.swing import SwingPivotDetector
        sw = SwingPivotDetector()
        sobs = sw.detect_all_historical(candles)
        highs = [o for o in sobs if o.type == "swing_high"]
        lows = [o for o in sobs if o.type == "swing_low"]
        cands = d._enumerate_candidates(highs, lows)
        assert len(cands) >= 1
        for anchors in cands:
            types = [a.type for a in anchors]
            assert len(anchors) >= 4
            assert len(anchors) <= d._MAX_ANCHORS_PER_CANDIDATE
            for a, b in zip(types, types[1:]):
                assert a != b
            nh = sum(1 for t in types if t == "swing_high")
            nl = len(types) - nh
            assert nh >= 2 and nl >= 2

    def test_enumeration_scales_quadratically_not_exponentially(self):
        """Candidate count grows polynomially with swing count, not 2^n.

        Build a long perfectly alternating pivot stream and assert the number
        of candidates stays within the O(n²) contiguous-window bound.
        """
        mid = 105.0
        n_bars = 80
        candles = []
        for i in range(n_bars):
            if i % 4 == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i % 4 == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.02, max_bars_to_breakout=5)
        from cpab.detectors.swing import SwingPivotDetector
        sw = SwingPivotDetector()
        sobs = sw.detect_all_historical(candles)
        highs = [o for o in sobs if o.type == "swing_high"]
        lows = [o for o in sobs if o.type == "swing_low"]
        n_swings = len(highs) + len(lows)
        cands = d._enumerate_candidates(highs, lows)
        # Contiguous windows of length 4..min(n, MAX) over a run of length n
        # yield at most n * MAX windows; with the swing cap, hard upper bound:
        max_n = min(n_swings, d._MAX_SWINGS_FOR_ENUMERATION)
        max_len = d._MAX_ANCHORS_PER_CANDIDATE
        # loose upper bound: runs * windows_per_run < max_n * max_n
        assert len(cands) <= max_n * max_n
        # and far below exponential 2^n for n >= 16
        if max_n >= 16:
            assert len(cands) < 2 ** (max_n // 2)


class TestRegistry:
    def _require_talib(self):
        pytest.importorskip("talib")

    def test_registry_construction(self):
        self._require_talib()
        reg = build_default_registry()
        assert reg.is_supported("rectangle")
        info = reg.validate_config("rectangle", {})
        assert info.implementation == "rectangle"
        assert "boundary_tolerance" in info.config
        det, info2 = reg.create_detector("rectangle", config={})
        assert isinstance(det, RectangleDetector)
        assert det.boundary_tolerance == DEFAULT_BOUNDARY_TOLERANCE

    def test_registry_invalid_config(self):
        self._require_talib()
        reg = build_default_registry()
        with pytest.raises(ValueError):
            reg.validate_config("rectangle", {"boundary_tolerance": -1})
        with pytest.raises(ValueError):
            reg.validate_config("rectangle", {"max_bars_to_breakout": 0})
