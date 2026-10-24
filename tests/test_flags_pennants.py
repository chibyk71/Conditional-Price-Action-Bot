"""Phase 5D — Flags and pennants acceptance tests."""

from __future__ import annotations

import math

import pytest

from cpab.detectors.flags_pennants import (
    FlagsPennantsDetector,
    CONCEPT_BULLISH_FLAG,
    CONCEPT_BEARISH_FLAG,
    CONCEPT_BULLISH_PENNANT,
    CONCEPT_BEARISH_PENNANT,
    VARIANT,
    DEFAULT_MIN_POLE_BARS,
    DEFAULT_MAX_POLE_BARS,
    DEFAULT_POLE_ATR_LOOKBACK,
    DEFAULT_POLE_STRENGTH_MULT,
    DEFAULT_MAX_POLE_RETRACEMENT,
    DEFAULT_MIN_CONSOLIDATION_BARS,
    DEFAULT_BOUNDARY_TOLERANCE,
    DEFAULT_PARALLEL_SLOPE_TOLERANCE,
    DEFAULT_MAX_BARS_TO_BREAKOUT,
    _are_parallel,
    _slope,
    _within_tolerance,
)
from cpab.detectors.wilder_atr import (
    true_range,
    wilder_atr,
    atr_before_pole,
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
    symbol: str = "FP_100",
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


# ---------------------------------------------------------------------------
# ATR unit tests
# ---------------------------------------------------------------------------


def test_true_range_basic():
    assert true_range(12.0, 10.0, 11.0) == 2.0
    assert true_range(12.0, 10.0, 9.0) == 3.0  # |H - prev|
    assert true_range(12.0, 10.0, 13.0) == 3.0  # |L - prev|


def test_wilder_atr_hand_calculated():
    """Hand-calculated Wilder ATR with lookback=3 for determinism."""
    # Candles: need lookback+1 = 4 bars
    # prev closes seed TRs
    bars = [
        _c(0, 10, 11, 9, 10),   # seed close=10
        _c(1, 10, 12, 9.5, 11),  # TR = max(2.5, 2, 0.5) = 2.5
        _c(2, 11, 13, 10.5, 12), # TR = max(2.5, 2, 0.5) = 2.5
        _c(3, 12, 14, 11.5, 13), # TR = max(2.5, 2, 0.5) = 2.5
    ]
    atr = wilder_atr(bars, lookback=3, step=TF)
    assert atr is not None
    assert abs(atr - 2.5) < 1e-9


def test_wilder_atr_insufficient_history():
    bars = [_c(i, 10, 11, 9, 10) for i in range(5)]
    assert wilder_atr(bars, lookback=14) is None


def test_wilder_atr_gap_rejected():
    bars = [_c(i, 10, 11, 9, 10) for i in range(16)]
    # Introduce gap
    bad = list(bars)
    bad[8] = _c(8, 10, 11, 9, 10, start=START + 1000)  # far epoch
    assert wilder_atr(bad, lookback=14, step=TF) is None


def test_wilder_atr_invalid_ohlc_rejected():
    bars = [_c(i, 10, 11, 9, 10) for i in range(16)]
    bad = list(bars)
    bad[10] = Candle(
        symbol="X",
        timeframe="1m",
        epoch=START + 10 * TF,
        open=10,
        high=8,  # high < low
        low=12,
        close=10,
        is_closed=True,
        source="provider",
    )
    assert wilder_atr(bad, lookback=14, step=TF) is None


def test_atr_before_pole_excludes_pole():
    # Build 20 quiet bars then a pole of 3 bars with large range.
    quiet = [_c(i, 100, 100.5, 99.5, 100) for i in range(20)]
    pole = [
        _c(20, 100, 110, 99, 109),
        _c(21, 109, 120, 108, 119),
        _c(22, 119, 130, 118, 129),
    ]
    seg = quiet + pole
    atr = atr_before_pole(seg, 20, lookback=14, step=TF)
    assert atr is not None
    # ATR from quiet bars should be small (~1.0 range)
    assert atr < 2.0
    # Pole itself has huge ranges; if included ATR would be large.


def test_atr_identical_input_identical_output():
    bars = [_c(i, 10 + i * 0.1, 11 + i * 0.1, 9 + i * 0.1, 10 + i * 0.1) for i in range(20)]
    a1 = wilder_atr(bars, lookback=14, step=TF)
    a2 = wilder_atr(bars, lookback=14, step=TF)
    assert a1 is not None and a1 == a2


# ---------------------------------------------------------------------------
# Parallelism helpers
# ---------------------------------------------------------------------------


def test_are_parallel_identical_slopes():
    assert _are_parallel(0.001, 0.001, 0.002)
    assert _are_parallel(-0.001, -0.001, 0.002)


def test_are_parallel_near_zero():
    assert _are_parallel(1e-15, -1e-15, 0.002)
    assert _are_parallel(0.0, 0.0, 0.002)


def test_are_parallel_just_inside_outside():
    # relative diff clearly inside tolerance
    assert _are_parallel(0.5, 0.5 - 0.0009, 0.002)
    # clearly outside
    assert not _are_parallel(0.5, 0.5 - 0.002, 0.002)


def test_are_parallel_nonparallel():
    assert not _are_parallel(0.01, -0.01, 0.002)


# ---------------------------------------------------------------------------
# Detector config validation
# ---------------------------------------------------------------------------


def test_detector_config_validation():
    d = FlagsPennantsDetector()
    assert d.name == "flags_pennants"
    assert d.detector_config_id.startswith("cfg:")
    with pytest.raises(ValueError):
        FlagsPennantsDetector(min_pole_bars=1)
    with pytest.raises(ValueError):
        FlagsPennantsDetector(pole_strength_mult=-1.0)
    with pytest.raises(ValueError):
        FlagsPennantsDetector(min_pole_bars=5, max_pole_bars=3)


def test_config_id_deterministic():
    d1 = FlagsPennantsDetector()
    d2 = FlagsPennantsDetector()
    assert d1.detector_config_id == d2.detector_config_id
    d3 = FlagsPennantsDetector(pole_strength_mult=2.0)
    assert d3.detector_config_id != d1.detector_config_id


# ---------------------------------------------------------------------------
# Synthetic pattern builders
# ---------------------------------------------------------------------------


def _quiet_series(n: int, mid: float = 100.0, width: float = 0.4, start_i: int = 0):
    return [
        _c(start_i + i, mid, mid + width, mid - width, mid) for i in range(n)
    ]


def _build_bullish_flag(
    *,
    atr_bars: int = 16,
    pole_bars: int = 4,
    cons_bars: int = 8,
    breakout: bool = True,
    retracement_ok: bool = True,
) -> list[Candle]:
    """
    Quiet ATR baseline → strong bullish pole → mild downward-sloping parallel
    consolidation with alternating swings → optional breakout above upper.
    """
    candles: list[Candle] = []
    # ATR baseline (quiet)
    for i in range(atr_bars):
        candles.append(_c(i, 100, 100.6, 99.4, 100.1))

    # Bullish pole: large upward displacement
    pole_start = atr_bars
    price = 100.0
    for j in range(pole_bars):
        i = pole_start + j
        o = price
        cl = price + 3.0
        h = cl + 0.3
        l = o - 0.2
        candles.append(_c(i, o, h, l, cl))
        price = cl
    pole_end_price = price

    # Consolidation: mild downward parallel channel with H/L swings
    # For strength-1 pivots: alternate highs (i%4==1) and lows (i%4==3)
    cons_start = atr_bars + pole_bars
    upper0 = pole_end_price - 0.5
    lower0 = pole_end_price - 2.5
    slope = -0.15  # mild downward
    for j in range(cons_bars):
        i = cons_start + j
        t = j
        mid_u = upper0 + slope * t
        mid_l = lower0 + slope * t
        if j % 4 == 1:  # swing high near upper
            o = mid_u - 0.2
            h = mid_u + 0.15
            l = mid_u - 0.6
            cl = mid_u - 0.1
        elif j % 4 == 3:  # swing low near lower
            o = mid_l + 0.2
            h = mid_l + 0.6
            l = mid_l - 0.15
            cl = mid_l + 0.1
        else:
            mid = (mid_u + mid_l) / 2
            o, h, l, cl = mid, mid + 0.3, mid - 0.3, mid
        if not retracement_ok:
            # Force deep retracement on every bar ( > 0.50 of pole displacement)
            l = min(l, pole_end_price - (pole_end_price - 100.0) * 0.7)
        candles.append(_c(i, o, h, l, cl))

    # Extra quiet bars so last consolidation swing can confirm before breakout
    for k in range(2):
        i = cons_start + cons_bars + k
        mid = (upper0 + slope * (cons_bars - 1) + lower0 + slope * (cons_bars - 1)) / 2
        candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
    if breakout:
        i = cons_start + cons_bars + 2
        last_u = upper0 + slope * (cons_bars - 1)
        candles.append(
            _c(i, last_u, last_u + 2.0, last_u - 0.5, last_u + 1.5)
        )
    return candles


def _build_bearish_flag(
    *,
    atr_bars: int = 16,
    pole_bars: int = 4,
    cons_bars: int = 8,
    breakout: bool = True,
) -> list[Candle]:
    candles: list[Candle] = []
    for i in range(atr_bars):
        candles.append(_c(i, 100, 100.6, 99.4, 99.9))

    pole_start = atr_bars
    price = 100.0
    for j in range(pole_bars):
        i = pole_start + j
        o = price
        cl = price - 3.0
        h = o + 0.2
        l = cl - 0.3
        candles.append(_c(i, o, h, l, cl))
        price = cl
    pole_end_price = price

    cons_start = atr_bars + pole_bars
    upper0 = pole_end_price + 2.5
    lower0 = pole_end_price + 0.5
    slope = 0.15  # mild upward
    for j in range(cons_bars):
        i = cons_start + j
        mid_u = upper0 + slope * j
        mid_l = lower0 + slope * j
        if j % 4 == 1:
            o = mid_u - 0.2
            h = mid_u + 0.15
            l = mid_u - 0.6
            cl = mid_u - 0.1
        elif j % 4 == 3:
            o = mid_l + 0.2
            h = mid_l + 0.6
            l = mid_l - 0.15
            cl = mid_l + 0.1
        else:
            mid = (mid_u + mid_l) / 2
            o, h, l, cl = mid, mid + 0.3, mid - 0.3, mid
        candles.append(_c(i, o, h, l, cl))

    for k in range(2):
        i = cons_start + cons_bars + k
        mid = (upper0 + slope * (cons_bars - 1) + lower0 + slope * (cons_bars - 1)) / 2
        candles.append(_c(i, mid, mid + 0.3, mid - 0.3, mid))
    if breakout:
        i = cons_start + cons_bars + 2
        last_l = lower0 + slope * (cons_bars - 1)
        candles.append(
            _c(i, last_l, last_l + 0.5, last_l - 2.0, last_l - 1.5)
        )
    return candles


def _build_bullish_pennant(
    *,
    atr_bars: int = 16,
    pole_bars: int = 4,
    cons_bars: int = 8,
    breakout: bool = True,
) -> list[Candle]:
    """Converging upper-down / lower-up after bullish pole."""
    candles: list[Candle] = []
    for i in range(atr_bars):
        candles.append(_c(i, 100, 100.6, 99.4, 100.1))

    pole_start = atr_bars
    price = 100.0
    for j in range(pole_bars):
        i = pole_start + j
        o = price
        cl = price + 3.0
        h = cl + 0.3
        l = o - 0.2
        candles.append(_c(i, o, h, l, cl))
        price = cl
    pole_end_price = price

    cons_start = atr_bars + pole_bars
    # Upper descending, lower ascending → converge
    upper0 = pole_end_price - 0.3
    lower0 = pole_end_price - 3.0
    for j in range(cons_bars):
        i = cons_start + j
        mid_u = upper0 - 0.25 * j
        mid_l = lower0 + 0.25 * j
        if j % 4 == 1:
            o = mid_u - 0.15
            h = mid_u + 0.1
            l = mid_u - 0.5
            cl = mid_u - 0.05
        elif j % 4 == 3:
            o = mid_l + 0.15
            h = mid_l + 0.5
            l = mid_l - 0.1
            cl = mid_l + 0.05
        else:
            mid = (mid_u + mid_l) / 2
            o, h, l, cl = mid, mid + 0.25, mid - 0.25, mid
        candles.append(_c(i, o, h, l, cl))

    for k in range(2):
        i = cons_start + cons_bars + k
        mid = (upper0 - 0.25 * (cons_bars - 1) + lower0 + 0.25 * (cons_bars - 1)) / 2
        candles.append(_c(i, mid, mid + 0.25, mid - 0.25, mid))
    if breakout:
        i = cons_start + cons_bars + 2
        last_u = upper0 - 0.25 * (cons_bars - 1)
        candles.append(
            _c(i, last_u, last_u + 2.0, last_u - 0.4, last_u + 1.5)
        )
    return candles


# ---------------------------------------------------------------------------
# Integration / pattern detection
# ---------------------------------------------------------------------------


def test_bullish_flag_detected():
    candles = _build_bullish_flag()
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,  # easier qualification on synthetic data
        max_pole_retracement=0.8,
    )
    obs = det.detect(candles)
    flags = [o for o in obs if o.type == CONCEPT_BULLISH_FLAG]
    assert len(flags) >= 1
    o = flags[0]
    assert o.direction == "bullish"
    assert o.variant == VARIANT
    assert o.detector_id == "flags_pennants"
    assert o.detector_config_id.startswith("cfg:")
    assert o.confirmed_at is not None
    assert o.reference_ids
    assert o.payload.get("is_flag") is True
    assert o.payload.get("pole_direction") == "bullish"


def test_bearish_flag_detected():
    candles = _build_bearish_flag()
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,
        max_pole_retracement=0.8,
    )
    obs = det.detect(candles)
    flags = [o for o in obs if o.type == CONCEPT_BEARISH_FLAG]
    assert len(flags) >= 1
    assert flags[0].direction == "bearish"


def test_bullish_pennant_detected():
    candles = _build_bullish_pennant()
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,
        max_pole_retracement=0.8,
    )
    obs = det.detect(candles)
    pens = [o for o in obs if o.type == CONCEPT_BULLISH_PENNANT]
    assert len(pens) >= 1
    assert pens[0].direction == "bullish"
    assert pens[0].payload.get("is_flag") is False


def test_no_breakout_no_emission():
    candles = _build_bullish_flag(breakout=False)
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,
        max_pole_retracement=0.8,
        max_bars_to_breakout=5,
    )
    obs = det.detect(candles)
    assert all(o.type not in ALL_CONCEPTS for o in obs) or len(
        [o for o in obs if o.type in (
            CONCEPT_BULLISH_FLAG, CONCEPT_BEARISH_FLAG,
            CONCEPT_BULLISH_PENNANT, CONCEPT_BEARISH_PENNANT,
        )]
    ) == 0


def test_retracement_rejects_deep():
    candles = _build_bullish_flag(retracement_ok=False)
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,
        max_pole_retracement=0.50,
    )
    obs = det.detect(candles)
    flags = [o for o in obs if o.type == CONCEPT_BULLISH_FLAG]
    # Deep retracement should prevent confirmation
    assert len(flags) == 0


def test_gap_isolation():
    # Two segments; pattern only in second should not see first ATR
    a = _build_bullish_flag()
    gap_start = a[-1].epoch + 10_000
    b = _build_bullish_flag()
    for c in b:
        # shift epochs
        object.__setattr__(c, "epoch", c.epoch - START + gap_start)
    combined = a + b
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,
        max_pole_retracement=0.8,
    )
    obs = det.detect(combined)
    # Should still detect (each segment independent)
    assert any(o.type == CONCEPT_BULLISH_FLAG for o in obs)


def test_registry_integration():
    pytest.importorskip("talib")
    reg = build_default_registry()
    for concept in (
        "bullish_flag",
        "bearish_flag",
        "bullish_pennant",
        "bearish_pennant",
        "flags_pennants",
    ):
        info = reg.validate_config(concept, {})
        assert info.detector_id == "flags_pennants"
        assert info.detector_config_id.startswith("cfg:")
        det, _ = reg.create_detector(concept)
        assert isinstance(det, FlagsPennantsDetector)


def test_pole_duration_bounds():
    # Too short pole should not qualify
    candles = _build_bullish_flag(pole_bars=1)
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,
        min_pole_bars=2,
        max_pole_bars=10,
        max_pole_retracement=0.8,
    )
    obs = det.detect(candles)
    assert not any(o.type == CONCEPT_BULLISH_FLAG for o in obs)


def test_zero_displacement_rejected():
    # Flat pole
    candles = _quiet_series(30)
    det = FlagsPennantsDetector(left_strength=1, right_strength=1)
    obs = det.detect(candles)
    assert not any(
        o.type
        in (
            CONCEPT_BULLISH_FLAG,
            CONCEPT_BEARISH_FLAG,
            CONCEPT_BULLISH_PENNANT,
            CONCEPT_BEARISH_PENNANT,
        )
        for o in obs
    )


# Needed by test_no_breakout
ALL_CONCEPTS = (
    CONCEPT_BULLISH_FLAG,
    CONCEPT_BEARISH_FLAG,
    CONCEPT_BULLISH_PENNANT,
    CONCEPT_BEARISH_PENNANT,
)


def test_strength_threshold_equality():
    """displacement == pole_strength_mult * ATR must qualify; just below must not."""
    from cpab.detectors.wilder_atr import atr_before_pole

    lookback = 14
    # lookback+1 quiet bars so ATR window is fully pre-pole. H-L=1.0, prev close
    # aligned → true range = 1.0 → Wilder ATR = 1.0 exactly.
    prefix = [_c(i, 100.0, 100.5, 99.5, 100.0) for i in range(lookback + 1)]
    # pole will start at index lookback+1; ATR uses prefix only.
    atr = atr_before_pole(
        prefix + [_c(lookback + 1, 100, 101, 99, 101)],
        lookback + 1,
        lookback=lookback,
        step=TF,
    )
    assert atr is not None and abs(atr - 1.0) < 1e-12

    mult = 1.5
    threshold = mult * atr
    pole_start = lookback + 1  # index of first pole bar

    def make_pole(disp: float):
        # 3-bar pole: close[start]=100, close[end]=100+disp so displacement is exact.
        start_close = 100.0
        end_close = start_close + disp
        bars = list(prefix)
        # Bar 0 of pole: close stays at start_close so displacement is measured correctly.
        bars.append(_c(pole_start, start_close, start_close + 0.2, start_close - 0.1, start_close))
        bars.append(
            _c(
                pole_start + 1,
                start_close,
                start_close + abs(disp) * 0.5 + 0.1,
                start_close - 0.1,
                start_close + disp / 2,
            )
        )
        bars.append(
            _c(
                pole_start + 2,
                start_close + disp / 2,
                end_close + 0.1,
                start_close + disp / 2 - 0.1,
                end_close,
            )
        )
        return bars

    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=mult,
        min_pole_bars=3,
        max_pole_bars=3,
        pole_atr_lookback=lookback,
    )

    # Exact equality: must qualify
    eq_candles = make_pole(threshold)
    eq_poles = [
        p for p in det._iter_poles(eq_candles, step=TF)
        if p["start_idx"] == pole_start and p["end_idx"] == pole_start + 2
    ]
    assert len(eq_poles) == 1, f"expected exact-threshold pole, got {eq_poles}"
    assert abs(eq_poles[0]["displacement"] - threshold) < 1e-9
    assert eq_poles[0]["displacement"] >= eq_poles[0]["threshold"] - 1e-12

    # Immediately below threshold: must not qualify
    below = threshold * 0.999
    below_candles = make_pole(below)
    below_poles = [
        p for p in det._iter_poles(below_candles, step=TF)
        if p["start_idx"] == pole_start and p["end_idx"] == pole_start + 2
    ]
    assert len(below_poles) == 0, f"below-threshold pole must be rejected, got {below_poles}"


def test_deterministic_ids():
    candles = _build_bullish_flag()
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,
        max_pole_retracement=0.8,
    )
    o1 = det.detect(candles)
    o2 = det.detect(candles)
    ids1 = sorted(o.id for o in o1 if o.type == CONCEPT_BULLISH_FLAG)
    ids2 = sorted(o.id for o in o2 if o.type == CONCEPT_BULLISH_FLAG)
    assert ids1 == ids2


def test_registry_min_history_matches_detector():
    """Registry metadata min_history must match FlagsPennantsDetector.min_history."""
    det = FlagsPennantsDetector()
    # Direct detector property
    assert det.min_history == (
        DEFAULT_POLE_ATR_LOOKBACK
        + 1
        + DEFAULT_MIN_POLE_BARS
        + DEFAULT_MIN_CONSOLIDATION_BARS
        + 1  # left default
        + 1  # right default
        + 4
    )
    # Registry path (skip if talib missing at import)
    talib = pytest.importorskip("talib")
    reg = build_default_registry()
    info = reg.validate_config("bullish_flag", {})
    assert info.min_history == det.min_history
    # Creating detector via registry yields same requirement
    det2, info2 = reg.create_detector("bullish_flag")
    assert det2.min_history == info2.min_history == det.min_history


def test_no_lookahead_retracement_after_final_anchor():
    """
    Retracement qualification must not use candles after the final selected
    anchor. A deep wick after the final anchor must not change whether an
    earlier-confirmed candidate qualifies.
    """
    # Build a normal bullish flag that confirms.
    base = _build_bullish_flag(cons_bars=8, breakout=True)
    det = FlagsPennantsDetector(
        left_strength=1,
        right_strength=1,
        pole_strength_mult=1.0,
        max_pole_retracement=0.50,
    )
    obs_base = [o for o in det.detect(base) if o.type == CONCEPT_BULLISH_FLAG]
    assert len(obs_base) >= 1
    confirmed_ids = {o.id for o in obs_base}

    # Append a candle with an extremely deep low *after* the pattern.
    # If retracement used this future candle, max retracement would exceed 0.50
    # and the earlier confirmation would be lost (lookahead).
    last = base[-1]
    deep = _c(
        len(base),
        last.close,
        last.close + 0.2,
        50.0,  # deep low far below pole
        last.close,
        start=START,
    )
    # Fix epoch relative to last
    deep = Candle(
        symbol=last.symbol,
        timeframe=last.timeframe,
        epoch=last.epoch + TF,
        open=last.close,
        high=last.close + 0.2,
        low=50.0,
        close=last.close,
        is_closed=True,
        source="provider",
    )
    extended = base + [deep]
    obs_ext = [o for o in det.detect(extended) if o.type == CONCEPT_BULLISH_FLAG]
    ext_ids = {o.id for o in obs_ext}
    # Previously confirmed observations must still be present (no retrospective
    # rejection from future deep wick).
    assert confirmed_ids.issubset(ext_ids), (
        f"lookahead changed confirmation set: lost {confirmed_ids - ext_ids}"
    )
