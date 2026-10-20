"""Phase 4B — Fair Value Gap lifecycle tracking tests."""

from __future__ import annotations

import pytest

from cpab.detectors.fair_value_gap import FairValueGapDetector
from cpab.detectors.fair_value_gap_lifecycle import FairValueGapLifecycleDetector
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.models import Candle, Observation
from cpab.core.observation_store import ObservationStore


def _c(
    epoch: int,
    o: float,
    h: float,
    l: float,
    cl: float,
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
        close=cl,
        is_closed=is_closed,
    )


START = 1_700_000_000
STEP = 60


def _series(
    ohlc_list: list[tuple[float, float, float, float]],
    *,
    start: int = START,
    step: int = STEP,
    timeframe: str = "1m",
    symbol: str = "R_100",
    last_closed: bool = True,
) -> list[Candle]:
    """Build consecutive closed candles from OHLC tuples (o,h,l,c)."""
    out: list[Candle] = []
    for i, (o, h, l, c) in enumerate(ohlc_list):
        closed = True if i < len(ohlc_list) - 1 else last_closed
        out.append(
            _c(
                start + i * step,
                o,
                h,
                l,
                c,
                symbol=symbol,
                timeframe=timeframe,
                is_closed=closed,
            )
        )
    return out


def _bullish_fvg_base() -> list[tuple[float, float, float, float]]:
    """
    Formation: c1 high=10, c3 low=12 → bullish FVG [10, 12].
    Returns three formation candles OHLC.
    """
    return [
        (9.0, 10.0, 8.0, 9.5),    # c1
        (9.5, 11.0, 9.0, 10.5),   # c2
        (12.2, 13.5, 12.0, 12.8),  # c3 formation
    ]


def _bearish_fvg_base() -> list[tuple[float, float, float, float]]:
    """
    Formation: c1 low=12, c3 high=10.5 → bearish FVG [10.5, 12].
    OHLC must be sane (high >= open/close).
    """
    return [
        (13.0, 14.0, 12.0, 12.5),  # c1
        (12.0, 13.0, 11.0, 11.5),  # c2
        (10.0, 10.5, 9.0, 9.5),    # c3 high=10.5 < c1.low=12
    ]


# ---------------------------------------------------------------------------
# 1. Bullish / bearish first touch
# ---------------------------------------------------------------------------


def test_bullish_first_touch():
    base = _bullish_fvg_base()
    # Touch: wick into gap from above — low=11.5 in [10,12]
    candles = _series(base + [(13.0, 13.5, 11.5, 12.5)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    touches = [o for o in obs if o.variant == "first_touch"]
    assert len(touches) == 1
    t = touches[0]
    assert t.type == "fair_value_gap_lifecycle"
    assert t.direction == "bullish"
    assert t.event_time == candles[3].epoch
    assert t.confirmed_at == candles[3].epoch
    assert len(t.reference_ids) == 1
    assert t.payload["lower_bound"] == 10.0
    assert t.payload["upper_bound"] == 12.0
    assert t.payload["penetration_depth"] == pytest.approx(0.5)  # 12 - 11.5


def test_bearish_first_touch():
    base = _bearish_fvg_base()
    # Touch: high=11.0 into [10.5, 12]
    candles = _series(base + [(9.0, 11.0, 8.5, 9.2)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    touches = [o for o in obs if o.variant == "first_touch"]
    assert len(touches) == 1
    t = touches[0]
    assert t.direction == "bearish"
    assert t.payload["penetration_depth"] == pytest.approx(0.5)  # 11.0 - 10.5


# ---------------------------------------------------------------------------
# 2. Wick-only and exact-boundary contact
# ---------------------------------------------------------------------------


def test_wick_only_and_exact_boundary_contact():
    base = _bullish_fvg_base()
    # Exact upper boundary: low == U == 12
    candles = _series(base + [(13.0, 13.5, 12.0, 12.8)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    touches = [o for o in obs if o.variant == "first_touch"]
    assert len(touches) == 1
    assert touches[0].payload["penetration_depth"] == pytest.approx(0.0)

    # Exact lower boundary contact via full mit path tested elsewhere.
    # Boundary-only low == L counts as intersect and full mit.
    candles2 = _series(base + [(13.0, 13.5, 10.0, 12.5)])
    obs2 = det.detect_all_historical(candles2)
    assert any(o.variant == "first_touch" for o in obs2)
    assert any(o.variant == "full_mitigation" for o in obs2)


# ---------------------------------------------------------------------------
# 3–5. Partial penetration depth, fill ratio, distinct IDs, no duplicate
# ---------------------------------------------------------------------------


def test_partial_penetration_depth_and_fill_ratio():
    base = _bullish_fvg_base()
    # Penetrate to low=11 → depth=1, fill=0.5
    candles = _series(base + [(13.0, 13.5, 11.0, 12.5)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    partials = [o for o in obs if o.variant == "partial_mitigation"]
    assert len(partials) == 1
    p = partials[0]
    assert p.payload["penetration_depth"] == pytest.approx(1.0)
    assert p.payload["fill_ratio"] == pytest.approx(0.5)


def test_multiple_increasing_penetration_distinct_ids():
    base = _bullish_fvg_base()
    # First: low=11.5 depth=0.5; second: low=11.0 depth=1.0
    candles = _series(
        base
        + [
            (13.0, 13.5, 11.5, 12.5),
            (12.5, 13.0, 11.0, 12.0),
        ]
    )
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    partials = [o for o in obs if o.variant == "partial_mitigation"]
    assert len(partials) == 2
    assert partials[0].payload["penetration_depth"] == pytest.approx(0.5)
    assert partials[1].payload["penetration_depth"] == pytest.approx(1.0)
    assert partials[0].id != partials[1].id
    assert partials[0].confirmed_at != partials[1].confirmed_at


def test_no_duplicate_partial_when_penetration_does_not_increase():
    base = _bullish_fvg_base()
    # First: low=11 depth=1; second: low=11.2 depth=0.8 (no increase)
    candles = _series(
        base
        + [
            (13.0, 13.5, 11.0, 12.5),
            (12.5, 13.0, 11.2, 12.2),
        ]
    )
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    partials = [o for o in obs if o.variant == "partial_mitigation"]
    assert len(partials) == 1
    assert partials[0].payload["penetration_depth"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6–9. Full mitigation, invalidation, same-candle ordering
# ---------------------------------------------------------------------------


def test_full_mitigation_by_wick_exact_far_boundary():
    base = _bullish_fvg_base()
    # low == L == 10
    candles = _series(base + [(13.0, 13.5, 10.0, 12.0)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    full = [o for o in obs if o.variant == "full_mitigation"]
    assert len(full) == 1
    assert full[0].payload["fill_ratio"] == pytest.approx(1.0)
    # No partial on the full-mit candle
    partials_same = [
        o
        for o in obs
        if o.variant == "partial_mitigation" and o.confirmed_at == candles[3].epoch
    ]
    assert partials_same == []


def test_close_based_invalidation_equality_does_not_invalidate():
    base = _bullish_fvg_base()
    # close == L == 10 → not invalidation; low=10.5 → partial only
    candles = _series(base + [(12.5, 13.0, 10.5, 10.0)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    assert not any(o.variant == "invalidation" for o in obs)

    # close < L
    candles2 = _series(base + [(12.5, 13.0, 9.5, 9.5)])
    obs2 = det.detect_all_historical(candles2)
    inv = [o for o in obs2 if o.variant == "invalidation"]
    assert len(inv) == 1
    assert inv[0].direction == "bullish"


def test_full_mitigation_and_invalidation_same_candle_ordering():
    base = _bullish_fvg_base()
    # low <= L and close < L: full mit + invalidation
    candles = _series(base + [(12.0, 12.5, 9.0, 9.5)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    # Filter events on the interaction candle
    events = [o for o in obs if o.confirmed_at == candles[3].epoch]
    variants = [o.variant for o in events]
    assert "first_touch" in variants
    assert "full_mitigation" in variants
    assert "invalidation" in variants
    assert "partial_mitigation" not in variants
    # Order: first_touch before full before invalidation
    idx = {v: variants.index(v) for v in ("first_touch", "full_mitigation", "invalidation")}
    assert idx["first_touch"] < idx["full_mitigation"] < idx["invalidation"]


def test_no_partial_when_full_mitigation_same_candle():
    base = _bullish_fvg_base()
    candles = _series(base + [(13.0, 13.5, 9.5, 12.0)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    assert any(o.variant == "full_mitigation" for o in obs)
    assert not any(
        o.variant == "partial_mitigation" and o.confirmed_at == candles[3].epoch
        for o in obs
    )


# ---------------------------------------------------------------------------
# 10–11. No events on formation / forming; no events after closure
# ---------------------------------------------------------------------------


def test_no_events_on_formation_or_forming_candles():
    base = _bullish_fvg_base()
    # Only formation candles
    candles = _series(base)
    det = FairValueGapLifecycleDetector()
    assert det.detect_all_historical(candles) == []

    # Forming last bar with touch geometry
    candles2 = _series(
        base + [(13.0, 13.5, 11.0, 12.5)],
        last_closed=False,
    )
    # detect() should ignore forming bar
    assert det.detect(candles2) == []
    # historical also filters is_closed
    assert det.detect_all_historical(candles2) == []


def test_no_events_after_lifecycle_closure():
    base = _bullish_fvg_base()
    # Full mit on bar 4, then deeper penetration on bar 5 should emit nothing new
    candles = _series(
        base
        + [
            (13.0, 13.5, 10.0, 12.0),  # full mit
            (12.0, 12.5, 9.0, 11.0),   # would be deeper but closed
        ]
    )
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    late = [o for o in obs if o.confirmed_at == candles[4].epoch]
    assert late == []


# ---------------------------------------------------------------------------
# 12. Gap handling
# ---------------------------------------------------------------------------


def test_gap_handling_skip_and_resume():
    base = _bullish_fvg_base()
    # Continuous touch after formation, then a gap, then a continuous pair after gap
    # that would increase penetration — state should carry; post-gap continuous
    # candle (with its predecessor) may be evaluated.
    form = base  # epochs 0,1,2
    # bar3 continuous: touch depth 0.5
    # bar4 missing (gap)
    # bar5 and bar6 at epochs after gap — we build with explicit epochs
    s = START
    candles = [
        _c(s, *base[0]),
        _c(s + STEP, *base[1]),
        _c(s + 2 * STEP, *base[2]),
        _c(s + 3 * STEP, 13.0, 13.5, 11.5, 12.5),  # touch
        # gap: skip s+4*STEP
        _c(s + 5 * STEP, 12.5, 13.0, 11.2, 12.0),  # discontinuous from prev
        _c(s + 6 * STEP, 12.0, 12.5, 11.0, 11.8),  # continuous with bar5
    ]
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    # First touch and partial on bar3 preserved
    assert any(o.variant == "first_touch" and o.confirmed_at == s + 3 * STEP for o in obs)
    # bar5 discontinuous → skipped
    assert not any(o.confirmed_at == s + 5 * STEP for o in obs)
    # bar6 continuous with bar5 → may emit partial if depth increased vs max from bar3
    # bar3 depth=0.5; bar6 low=11.0 depth=1.0 → partial
    partials_late = [
        o for o in obs if o.variant == "partial_mitigation" and o.confirmed_at == s + 6 * STEP
    ]
    assert len(partials_late) == 1
    assert partials_late[0].payload["penetration_depth"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 13. Malformed, unknown TF, mixed series, overlapping FVGs
# ---------------------------------------------------------------------------


def test_malformed_unknown_tf_mixed_overlapping():
    det = FairValueGapLifecycleDetector()
    base = _bullish_fvg_base()

    # Unknown timeframe
    bad_tf = _series(base + [(13.0, 13.5, 11.0, 12.5)], timeframe="unknown_tf")
    assert det.detect_all_historical(bad_tf) == []

    # Mixed symbols
    candles = _series(base + [(13.0, 13.5, 11.0, 12.5)])
    candles[2] = _c(candles[2].epoch, 12.2, 13.5, 12.0, 12.8, symbol="OTHER")
    assert det.detect_all_historical(candles) == []

    # Malformed OHLC on interaction candle (high < low) — skipped, no crash
    candles2 = _series(base + [(13.0, 10.0, 11.5, 12.0)])  # high < low
    # _ohlc_sane rejects; no lifecycle events
    assert det.detect_all_historical(candles2) == []

    # Overlapping / sequential FVGs tracked independently.
    # FVG1 at epochs 0-2 [10,12]; post-formation touch; then FVG2 far above.
    ohlc = [
        (9.0, 10.0, 8.0, 9.5),       # 0
        (9.5, 11.0, 9.0, 10.5),      # 1
        (12.2, 13.5, 12.0, 12.8),    # 2 FVG1 [10,12]
        (13.0, 13.5, 11.5, 12.5),    # 3 touch FVG1
        (19.0, 20.0, 18.5, 19.5),    # 4
        (19.5, 21.0, 19.0, 20.5),    # 5
        (22.2, 23.5, 22.0, 22.8),    # 6 FVG2 [20,22]
        (23.0, 23.5, 21.0, 22.5),    # 7 touch FVG2
    ]
    candles3 = _series(ohlc)
    fvg_det = FairValueGapDetector()
    fvgs = fvg_det.detect_all_historical(candles3)
    assert len(fvgs) >= 2
    obs = det.detect_all_historical(candles3)
    sources = {o.reference_ids[0] for o in obs if o.reference_ids}
    assert len(sources) >= 2


# ---------------------------------------------------------------------------
# 14–15. Stable IDs, idempotency, historical/incremental parity
# ---------------------------------------------------------------------------


def test_stable_ids_and_idempotency():
    base = _bullish_fvg_base()
    candles = _series(
        base
        + [
            (13.0, 13.5, 11.5, 12.5),
            (12.5, 13.0, 11.0, 12.0),
        ]
    )
    det = FairValueGapLifecycleDetector()
    a = det.detect_all_historical(candles)
    b = det.detect_all_historical(candles)
    assert [o.id for o in a] == [o.id for o in b]
    assert [o.variant for o in a] == [o.variant for o in b]

    store = ObservationStore()
    assert store.accept(a) == len(a)
    assert store.accept(b) == 0  # idempotent re-accept
    ids = [o.id for o in a]
    assert len(ids) == len(set(ids))


def test_historical_incremental_parity():
    base = _bullish_fvg_base()
    full = _series(
        base
        + [
            (13.0, 13.5, 11.5, 12.5),
            (12.5, 13.0, 11.0, 12.0),
            (12.0, 12.5, 9.5, 9.8),  # full mit + inv
        ]
    )
    det = FairValueGapLifecycleDetector()
    hist = det.detect_all_historical(full)

    # Incremental: walk growing prefixes, collect detect() emissions
    incremental: list[Observation] = []
    for n in range(1, len(full) + 1):
        incremental.extend(det.detect(full[:n]))

    # Same multiset of events; emission order may follow FVG iteration vs time.
    assert sorted(o.id for o in hist) == sorted(o.id for o in incremental)


# ---------------------------------------------------------------------------
# 16. Registry
# ---------------------------------------------------------------------------


def test_registry_capability_and_config():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("TA-Lib not installed in this environment")
        raise
    info = reg.validate_config("fair_value_gap_lifecycle", {})
    assert info.detector_id == "fair_value_gap_lifecycle"
    assert info.min_history == 4
    assert info.config == {}
    assert info.detector_config_id == deterministic_config_id({})

    # Alias
    info2 = reg.validate_config("fvg_lifecycle", {})
    assert info2.detector_id == "fair_value_gap_lifecycle"

    det, info3 = reg.create_detector("fair_value_gap_lifecycle")
    assert det.name == "fair_value_gap_lifecycle"
    assert det.min_history == 4
    assert isinstance(det, FairValueGapLifecycleDetector)

    # Unknown params rejected
    with pytest.raises(ValueError):
        reg.validate_config("fair_value_gap_lifecycle", {"min_size": 1.0})


# ---------------------------------------------------------------------------
# 17. Phase 4A regression
# ---------------------------------------------------------------------------


def test_phase4a_behavior_unchanged():
    """FairValueGapDetector still forms the same observations."""
    base = _bullish_fvg_base()
    candles = _series(base + [(13.0, 13.5, 11.0, 12.5)])
    fvg_det = FairValueGapDetector()
    fvgs = fvg_det.detect_all_historical(candles)
    assert len(fvgs) == 1
    assert fvgs[0].type == "fair_value_gap"
    assert fvgs[0].direction == "bullish"
    assert fvgs[0].payload["lower_bound"] == 10.0
    assert fvgs[0].payload["upper_bound"] == 12.0

    # Registry still resolves fvg
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("TA-Lib not installed in this environment")
        raise
    info = reg.validate_config("fvg", {})
    assert info.detector_id == "fair_value_gap"
    assert info.min_history == 3


def test_bearish_full_mitigation_and_invalidation():
    base = _bearish_fvg_base()
    # FVG [10.5, 12]; full mit: high >= 12; inv: close > 12
    candles = _series(base + [(11.0, 12.5, 10.8, 12.2)])
    det = FairValueGapLifecycleDetector()
    obs = det.detect_all_historical(candles)
    variants = [o.variant for o in obs if o.confirmed_at == candles[3].epoch]
    assert "first_touch" in variants
    assert "full_mitigation" in variants
    assert "invalidation" in variants
    assert "partial_mitigation" not in variants
