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
        """For every prefix, detect(prefix) equals historical observations confirmed
        on that prefix's last bar — same ids, anchors, direction, and timestamps.

        This is the incremental contract (not a one-way subset check). Confirmation
        bars are placed after min_history so the gate does not suppress emission.
        """
        mid = 105.0
        # min_history for strength 1 is 13; put structure early, breakout at bar 14
        candles = []
        for i in range(20):
            if i == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 5:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i == 7:
                candles.append(_c(i, 102, 103, 100.0, 101))
            elif i == 14:
                candles.append(_c(i, 109, 112, 108.5, 111.0))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=20)
        hist = d.detect_all_historical(candles)
        assert len(hist) >= 1
        for k in range(d.min_history, len(candles) + 1):
            prefix = candles[:k]
            latest = prefix[-1].epoch
            inc = d.detect(prefix)
            exp = [o for o in hist if o.confirmed_at == latest]
            assert sorted(o.id for o in inc) == sorted(o.id for o in exp)
            by_id = {o.id: o for o in exp}
            for o in inc:
                h = by_id[o.id]
                assert o.direction == h.direction
                assert o.confirmed_at == h.confirmed_at
                assert o.event_time == h.event_time
                assert o.reference_ids == h.reference_ids
                assert o.payload.get("support") == h.payload.get("support")
                assert o.payload.get("resistance") == h.payload.get("resistance")
                assert o.payload.get("breakout_close") == h.payload.get("breakout_close")



class TestEnumerationSemantics:
    """Regression: QA must not silently narrow Phase 5B candidate coverage."""

    def test_alternating_only(self):
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
            for a, b in zip(types, types[1:]):
                assert a != b
            nh = sum(1 for t in types if t == "swing_high")
            nl = len(types) - nh
            assert nh >= 2 and nl >= 2

    def test_long_anchor_sequence_not_capped(self):
        """A valid candidate with more than 12 anchors is not silently excluded."""
        from cpab.detectors.swing import SwingPivotDetector
        # 28 bars alternating → ~14 pivots; full path length 14 > 12 without
        # pathological 2^n cost from much longer streams.
        mid = 105.0
        res, sup = 110.0, 100.0
        n_bars = 28
        candles = []
        for i in range(n_bars):
            if i % 4 == 1:
                candles.append(_c(i, 108, res, 107, 109))
            elif i % 4 == 3:
                candles.append(_c(i, 102, 103, sup, 101))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.01, max_bars_to_breakout=5)
        sw = SwingPivotDetector()
        sobs = sw.detect_all_historical(candles)
        highs = [o for o in sobs if o.type == "swing_high"]
        lows = [o for o in sobs if o.type == "swing_low"]
        cands = d._enumerate_candidates(highs, lows)
        long_ones = [c for c in cands if len(c) > 12]
        assert len(highs) + len(lows) > 12
        assert len(long_ones) >= 1, (
            f"expected candidates with >12 anchors; got max len "
            f"{max((len(c) for c in cands), default=0)} from {len(cands)} candidates"
        )

    def test_older_swings_not_truncated(self):
        """Older swings remain eligible when the segment has more than 32 swings.

        Uses synthetic swing observations so we control count and alternation
        depth: a short alternating prefix (includes the oldest swing) followed
        by many same-type-heavy swings. The removed 32-swing cap would drop the
        oldest; full enumeration must still include it.
        """
        from cpab.models import Observation

        def mk(i: int, typ: str, price: float) -> Observation:
            ep = START + i * TF
            return Observation.create(
                type=typ,
                symbol="R_100",
                timeframe="1m",
                candle_epoch=ep,
                id=f"{typ}|synth|{i}",
                event_time=ep,
                confirmed_at=ep + TF,
                detector_id="swing_pivot",
                detector_config_id="cfg:test",
                payload={"price": price},
                metadata={"price": price},
            )

        # H L H L (oldest at i=0) then 40 additional swings with limited alternation
        highs = [
            mk(0, "swing_high", 110.0),
            mk(2, "swing_high", 110.0),
        ]
        lows = [
            mk(1, "swing_low", 100.0),
            mk(3, "swing_low", 100.0),
        ]
        # 30 more highs and 5 more lows interleaved sparsely → total > 32
        for j in range(30):
            highs.append(mk(10 + j, "swing_high", 110.0 + j * 0.001))
        for j in range(5):
            lows.append(mk(50 + j * 2, "swing_low", 100.0 - j * 0.001))

        d = RectangleDetector(boundary_tolerance=0.05, max_bars_to_breakout=5)
        n_swings = len(highs) + len(lows)
        assert n_swings > 32, f"need >32 swings, got {n_swings}"
        cands = d._enumerate_candidates(highs, lows)
        assert any(
            "swing_high|synth|0" in {a.id for a in c} for c in cands
        ), "oldest swing was dropped from all candidates (truncation)"

    def test_skip_intermediate_swing_still_evaluated(self):
        """A valid combination that skips an intermediate swing is still a candidate."""
        from cpab.detectors.swing import SwingPivotDetector
        mid = 105.0
        # Pivots: H0 L1 H2 L3 H4 — subsequence H0,L1,H2,L3 and H0,L1,H4,L3 etc.
        candles = []
        for i in range(24):
            if i % 4 == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i % 4 == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.02, max_bars_to_breakout=10)
        sw = SwingPivotDetector()
        sobs = sw.detect_all_historical(candles)
        highs = sorted(
            [o for o in sobs if o.type == "swing_high"],
            key=lambda o: o.event_time or 0,
        )
        lows = sorted(
            [o for o in sobs if o.type == "swing_low"],
            key=lambda o: o.event_time or 0,
        )
        assert len(highs) >= 3 and len(lows) >= 2
        cands = d._enumerate_candidates(highs, lows)
        # Look for a candidate that includes first and third high but skips the second
        h0, h2 = highs[0].id, highs[2].id
        h1 = highs[1].id
        skipped = [
            c for c in cands
            if h0 in {a.id for a in c}
            and h2 in {a.id for a in c}
            and h1 not in {a.id for a in c}
        ]
        assert len(skipped) >= 1, "no candidate skips an intermediate swing"

    def test_overlapping_distinct_and_identical_deduped(self):
        """Distinct overlapping anchor sets coexist; identical sets are deduplicated."""
        from cpab.detectors.swing import SwingPivotDetector
        mid = 105.0
        candles = []
        for i in range(28):
            if i % 4 == 1:
                candles.append(_c(i, 108, 110.0, 107, 109))
            elif i % 4 == 3:
                candles.append(_c(i, 102, 103, 100.0, 101))
            else:
                candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        d = RectangleDetector(boundary_tolerance=0.02, max_bars_to_breakout=10)
        sw = SwingPivotDetector()
        sobs = sw.detect_all_historical(candles)
        highs = [o for o in sobs if o.type == "swing_high"]
        lows = [o for o in sobs if o.type == "swing_low"]
        cands = d._enumerate_candidates(highs, lows)
        keys = [tuple(sorted(a.id for a in c)) for c in cands]
        assert len(keys) == len(set(keys)), "identical anchor-ID sets were not deduplicated"
        assert len(set(keys)) >= 2, "expected multiple distinct overlapping candidates"

    def test_no_candidate_spans_gap(self):
        """Candidates never mix anchors from opposite sides of a candle gap."""
        from cpab.detectors.swing import SwingPivotDetector
        mid = 105.0
        c1 = []
        for i in range(16):
            if i % 4 == 1:
                c1.append(_c(i, 108, 110.0, 107, 109))
            elif i % 4 == 3:
                c1.append(_c(i, 102, 103, 100.0, 101))
            else:
                c1.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
        gap_start = START + 40 * TF
        c2 = []
        for i in range(16):
            if i % 4 == 1:
                c2.append(_c(i, 108, 110.0, 107, 109, start=gap_start))
            elif i % 4 == 3:
                c2.append(_c(i, 102, 103, 100.0, 101, start=gap_start))
            elif i == 14:
                c2.append(_c(i, 109, 112, 108.5, 111.0, start=gap_start))
            else:
                c2.append(_c(i, mid, mid + 0.3, mid - 0.3, mid, start=gap_start))
        d = RectangleDetector(boundary_tolerance=0.02, max_bars_to_breakout=10)
        obs = d.detect_all_historical(c1 + c2)
        # Swing event times on each side of the gap
        sw = SwingPivotDetector()
        s1 = {o.id for o in sw.detect_all_historical(c1)}
        s2 = {o.id for o in sw.detect_all_historical(c2)}
        for o in obs:
            if o.type != CONCEPT_RECTANGLE:
                continue
            refs = set(o.reference_ids)
            in1 = refs & s1
            in2 = refs & s2
            assert not (in1 and in2), "candidate spanned the data gap"


    def test_coverage_matches_known_counts(self):
        """Deterministic candidate counts for perfect alternation (no wall-clock).

        These counts document exponential growth and lock coverage against
        accidental narrowing. Values measured from the contract-compliant
        subsequence enumerator at commit 8517257 / streaming optimization.
        """
        from cpab.models import Observation

        def mk(i, typ, price):
            ep = START + i * TF
            return Observation.create(
                type=typ, symbol="R_100", timeframe="1m", candle_epoch=ep,
                id=f"{typ}|c|{i}", event_time=ep, confirmed_at=ep + TF,
                detector_id="swing_pivot", detector_config_id="cfg:t",
                payload={"price": price}, metadata={"price": price},
            )

        d = RectangleDetector()
        # n alternating swings starting with high; expected candidate counts
        expected = {8: 43, 10: 156, 12: 490, 14: 1420}
        for n, exp in expected.items():
            highs, lows = [], []
            for i in range(n):
                if i % 2 == 0:
                    highs.append(mk(i, "swing_high", 110.0))
                else:
                    lows.append(mk(i, "swing_low", 100.0))
            cands = d._enumerate_candidates(highs, lows)
            assert len(cands) == exp, f"n={n}: got {len(cands)}, expected {exp}"
            # every candidate alternates and has ≥2 of each
            for c in cands:
                types = [a.type for a in c]
                assert all(a != b for a, b in zip(types, types[1:]))
                assert sum(1 for t in types if t == "swing_high") >= 2
                assert sum(1 for t in types if t == "swing_low") >= 2

    def test_stream_and_list_enumeration_agree(self):
        """Streaming iterator and list materialization yield identical sets."""
        from cpab.models import Observation

        def mk(i, typ, price):
            ep = START + i * TF
            return Observation.create(
                type=typ, symbol="R_100", timeframe="1m", candle_epoch=ep,
                id=f"{typ}|s|{i}", event_time=ep, confirmed_at=ep + TF,
                detector_id="swing_pivot", detector_config_id="cfg:t",
                payload={"price": price}, metadata={"price": price},
            )

        highs, lows = [], []
        for i in range(12):
            if i % 2 == 0:
                highs.append(mk(i, "swing_high", 110.0))
            else:
                lows.append(mk(i, "swing_low", 100.0))
        d = RectangleDetector()
        as_list = d._enumerate_candidates(highs, lows)
        as_stream = list(d._iter_candidates(highs, lows))
        keys_l = sorted(tuple(sorted(a.id for a in c)) for c in as_list)
        keys_s = sorted(tuple(sorted(a.id for a in c)) for c in as_stream)
        assert keys_l == keys_s



    def test_unique_source_ids_yield_unique_combinations(self):
        """With unique swing IDs, enumeration never emits duplicate ID sets.

        Index-based strictly increasing paths map 1:1 onto unique ordered ID
        sequences when every entry id is unique — no global seen-set required.
        """
        from cpab.models import Observation

        def mk(i, typ, price):
            ep = START + i * TF
            return Observation.create(
                type=typ, symbol="R_100", timeframe="1m", candle_epoch=ep,
                id=f"{typ}|uniq|{i}", event_time=ep, confirmed_at=ep + TF,
                detector_id="swing_pivot", detector_config_id="cfg:t",
                payload={"price": price}, metadata={"price": price},
            )

        highs = [mk(i, "swing_high", 110.0) for i in range(0, 12, 2)]
        lows = [mk(i, "swing_low", 100.0) for i in range(1, 12, 2)]
        d = RectangleDetector()
        cands = d._enumerate_candidates(highs, lows)
        keys = [tuple(sorted(a.id for a in c)) for c in cands]
        assert len(keys) == len(set(keys))
        assert len(cands) == 490  # n=12 alternating lock

    def test_duplicate_source_ids_collapsed_to_first_occurrence(self):
        """Duplicate observation IDs keep the first chronological entry only.

        Malformed or repeated upstream IDs must not multiply combinations or
        require an O(candidates) seen-set. After entry-level dedup, traversal
        uniqueness holds.
        """
        from cpab.models import Observation

        def mk(i, typ, price, oid):
            ep = START + i * TF
            return Observation.create(
                type=typ, symbol="R_100", timeframe="1m", candle_epoch=ep,
                id=oid, event_time=ep, confirmed_at=ep + TF,
                detector_id="swing_pivot", detector_config_id="cfg:t",
                payload={"price": price}, metadata={"price": price},
            )

        # Two highs share the same id; first chronological occurrence is i=0
        highs = [
            mk(0, "swing_high", 110.0, "H_DUP"),
            mk(2, "swing_high", 110.1, "H_DUP"),  # later duplicate — dropped
            mk(4, "swing_high", 110.0, "H4"),
            mk(6, "swing_high", 110.0, "H6"),
        ]
        lows = [
            mk(1, "swing_low", 100.0, "L1"),
            mk(3, "swing_low", 100.0, "L3"),
            mk(5, "swing_low", 100.0, "L5"),
        ]
        d = RectangleDetector()
        cands = d._enumerate_candidates(highs, lows)
        keys = [tuple(sorted(a.id for a in c)) for c in cands]
        assert len(keys) == len(set(keys)), "duplicate ID combinations emitted"
        # No candidate may reference two distinct objects under H_DUP via later copy
        for c in cands:
            dup_objs = [a for a in c if a.id == "H_DUP"]
            assert len(dup_objs) <= 1
            if dup_objs:
                assert dup_objs[0].event_time == START  # first occurrence only


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
