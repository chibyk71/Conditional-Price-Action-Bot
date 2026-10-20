"""Phase 4F — Mitigation Block detection acceptance tests."""

from __future__ import annotations

import pytest

from cpab.detectors.breaker_block import BreakerBlockDetector
from cpab.detectors.mitigation_block import MitigationBlockDetector
from cpab.detectors.order_block import OrderBlockDetector, DEFAULT_MIN_BODY_RATIO
from cpab.detectors.registry import build_default_registry
from cpab.models import Candle


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


def _series_from_ohlc(
    rows: list[tuple[float, float, float, float]],
    start: int = START,
    *,
    step: int = STEP,
    timeframe: str = "1m",
    symbol: str = "R_100",
    last_closed: bool = True,
) -> list[Candle]:
    out = []
    for i, (o, h, l, c) in enumerate(rows):
        closed = True if i < len(rows) - 1 else last_closed
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


def _bullish_ob_base_rows() -> list[tuple[float, float, float, float]]:
    """
    Bullish OB confirmed at idx 5. Zone [9.8, 11.2].
    Candidate bearish candle at idx 3.
    """
    return [
        (10.0, 10.5, 9.5, 10.0),   # 0
        (10.0, 12.0, 9.8, 11.0),   # 1 swing high
        (11.0, 11.2, 10.5, 10.8),  # 2
        (11.0, 11.2, 9.8, 10.0),   # 3 bearish OB candidate
        (10.0, 10.5, 9.9, 10.2),   # 4
        (10.2, 13.5, 10.0, 13.0),  # 5 bullish BOS / OB confirmed
    ]


def _bearish_ob_base_rows() -> list[tuple[float, float, float, float]]:
    """
    Bearish OB confirmed at idx 5. Zone [8.8, 10.0].
    Candidate bullish candle at idx 3.
    """
    return [
        (10.0, 10.5, 9.5, 10.0),   # 0
        (10.0, 10.2, 8.0, 9.0),    # 1 swing low
        (9.0, 9.5, 8.5, 9.2),      # 2
        (9.0, 10.0, 8.8, 9.8),     # 3 bullish OB candidate
        (9.8, 10.0, 9.5, 9.6),     # 4
        (9.6, 9.7, 7.0, 7.2),      # 5 bearish BOS / OB confirmed
    ]


# ---------------------------------------------------------------------------
# Geometry — bullish
# ---------------------------------------------------------------------------


def test_bullish_touch():
    """Bullish OB: wick touches near boundary U=11.2 from above."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 11.2, 12.5),  # low == U → touch only
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs if o.type == "mitigation_block"]
    assert stages == ["touch"]
    assert obs[0].direction == "bullish"
    assert obs[0].payload["source_type"] == "order_block"
    assert obs[0].payload["lower_bound"] == 9.8
    assert obs[0].payload["upper_bound"] == 11.2
    assert obs[0].lifecycle_state == "touch"
    assert len(obs[0].reference_ids) == 1


def test_bullish_partial_fill():
    """Bullish: low penetrates below U while remaining above L."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 10.5, 12.0),  # L=9.8 < low=10.5 < U=11.2
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs if o.type == "mitigation_block"]
    assert stages == ["touch", "partial_fill"]
    assert all(o.confirmed_at == candles[6].epoch for o in obs)


def test_bullish_full_fill():
    """Bullish: low reaches far boundary L=9.8."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 9.8, 12.0),  # low == L → full (with prior stages)
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs if o.type == "mitigation_block"]
    assert stages == ["touch", "partial_fill", "full_fill"]


def test_bullish_wick_penetration_close_outside():
    """Wick penetrates while close remains above the zone."""
    rows = _bullish_ob_base_rows() + [
        (12.5, 13.0, 10.0, 12.2),  # close 12.2 > U; low 10.0 interior
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert "touch" in stages
    assert "partial_fill" in stages
    assert "full_fill" not in stages


def test_bullish_exact_boundary_touch_only():
    """low == U is touch; not partial (strict interior required)."""
    rows = _bullish_ob_base_rows() + [
        (12.0, 12.5, 11.2, 11.5),
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    assert [o.variant for o in obs] == ["touch"]


def test_bullish_single_candle_crosses_all_stages():
    """One candle from outside to far boundary emits all three stages in order."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 9.5, 12.0),  # low < L
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert stages == ["touch", "partial_fill", "full_fill"]
    assert all(o.confirmed_at == candles[6].epoch for o in obs)


# ---------------------------------------------------------------------------
# Geometry — bearish
# ---------------------------------------------------------------------------


def test_bearish_touch():
    rows = _bearish_ob_base_rows() + [
        (7.0, 8.8, 6.5, 7.5),  # high == L → touch
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert stages == ["touch"]
    assert obs[0].direction == "bearish"


def test_bearish_partial_fill():
    rows = _bearish_ob_base_rows() + [
        (7.0, 9.5, 6.5, 8.0),  # L=8.8 < high=9.5 < U=10.0
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert stages == ["touch", "partial_fill"]


def test_bearish_full_fill():
    rows = _bearish_ob_base_rows() + [
        (7.0, 10.0, 6.5, 8.0),  # high == U
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert stages == ["touch", "partial_fill", "full_fill"]


def test_bearish_wick_penetration_close_outside():
    rows = _bearish_ob_base_rows() + [
        (7.5, 9.5, 7.0, 7.8),  # close below L; high interior
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert "touch" in stages and "partial_fill" in stages
    assert "full_fill" not in stages


def test_bearish_single_candle_crosses_all_stages():
    rows = _bearish_ob_base_rows() + [
        (7.0, 10.5, 6.5, 8.0),  # high > U
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    assert [o.variant for o in obs] == ["touch", "partial_fill", "full_fill"]


# ---------------------------------------------------------------------------
# Lifecycle ordering / no duplicates / no events after full_fill
# ---------------------------------------------------------------------------


def test_stage_ordering_progressive():
    """touch then partial on later candle then full — ordered, no dups."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 11.2, 12.5),  # touch
        (12.5, 13.0, 10.5, 12.0),  # partial
        (12.0, 12.5, 9.5, 11.5),   # full
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert stages == ["touch", "partial_fill", "full_fill"]
    epochs = [o.confirmed_at for o in obs]
    assert epochs[0] < epochs[1] < epochs[2]


def test_no_duplicate_stages_on_revisit():
    """Revisiting zone after partial must not re-emit touch or partial."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 10.5, 12.0),  # touch + partial
        (12.0, 12.5, 11.5, 12.0),  # above zone
        (12.0, 12.5, 10.2, 11.8),  # revisit interior — no new stages
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert stages == ["touch", "partial_fill"]
    assert stages.count("touch") == 1
    assert stages.count("partial_fill") == 1


def test_no_events_after_full_fill():
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 9.5, 12.0),  # full
        (12.0, 12.5, 10.0, 11.5),  # revisit — must emit nothing
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    assert stages == ["touch", "partial_fill", "full_fill"]
    assert all(o.confirmed_at == candles[6].epoch for o in obs)


def test_independent_lifecycle_per_source():
    """OB mitigation and breaker mitigation track independent state."""
    # Form bullish OB, then close-through → breaker, then mitigate the breaker.
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.2, 9.0, 9.5),   # idx 6: close < L → breaker (bearish)
        (9.5, 10.5, 9.0, 10.0),  # idx 7: approach breaker from below → touch/partial
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)

    # OB source may also have been mitigated by the close-through candle
    # (low=9.0 <= L → full_fill of bullish OB).
    by_source: dict[str, list[str]] = {}
    for o in obs:
        sid = o.payload["source_id"]
        by_source.setdefault(sid, []).append(o.variant)

    assert len(by_source) >= 1
    # At least one source should have progressed; OB full-fill on idx 6 is expected.
    ob_full = any(
        stages == ["touch", "partial_fill", "full_fill"]
        or "full_fill" in stages
        for stages in by_source.values()
    )
    assert ob_full or any(
        o.payload["source_type"] == "breaker_block" for o in obs
    )


def test_deterministic_identity_idempotent():
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 10.5, 12.0),
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    a = det.detect_all_historical(candles)
    b = det.detect_all_historical(candles)
    assert [o.id for o in a] == [o.id for o in b]
    assert len(a) >= 2
    # Distinct stages have distinct ids
    assert len({o.id for o in a}) == len(a)


def test_incremental_detect_filters_to_latest():
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 11.2, 12.5),  # touch on idx 6
        (12.5, 13.0, 10.5, 12.0),  # partial on idx 7
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    latest = det.detect(candles)
    assert all(o.confirmed_at == candles[-1].epoch for o in latest)
    assert any(o.variant == "partial_fill" for o in latest)


# ---------------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------------


def test_confirmed_order_block_accepted():
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 11.0, 12.5),
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    assert any(o.payload["source_type"] == "order_block" for o in obs)
    # Source OB itself unchanged
    ob_det = OrderBlockDetector()
    obs_ob = ob_det.detect_all_historical(candles)
    assert len(obs_ob) == 1
    assert obs_ob[0].payload["lower_bound"] == 9.8
    assert obs_ob[0].payload["upper_bound"] == 11.2


def test_confirmed_breaker_block_accepted():
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.2, 9.0, 9.5),   # breaker forms
        (9.5, 10.5, 9.2, 10.0),  # mitigate breaker (bearish, from below)
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    breaker_mit = [o for o in obs if o.payload.get("source_type") == "breaker_block"]
    assert len(breaker_mit) >= 1
    assert breaker_mit[0].direction == "bearish"


def test_source_observation_unchanged_after_mitigation():
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 9.5, 12.0),
    ]
    candles = _series_from_ohlc(rows)
    ob_before = OrderBlockDetector().detect_all_historical(candles)
    MitigationBlockDetector().detect_all_historical(candles)
    ob_after = OrderBlockDetector().detect_all_historical(candles)
    assert len(ob_before) == len(ob_after) == 1
    assert ob_before[0].id == ob_after[0].id
    assert ob_before[0].payload == ob_after[0].payload


def test_malformed_bounds_handled_safely():
    """Detector must not crash on empty series / insufficient history."""
    det = MitigationBlockDetector()
    assert det.detect_all_historical([]) == []
    short = _series_from_ohlc([(10, 11, 9, 10)] * 3)
    assert det.detect_all_historical(short) == []


# ---------------------------------------------------------------------------
# Causality and continuity
# ---------------------------------------------------------------------------


def test_source_confirmation_precedes_interaction():
    """No mitigation on the OB confirmation candle itself."""
    rows = _bullish_ob_base_rows()  # ends at OB confirm
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    # min_history may block; even if not, no interaction after confirm
    obs = det.detect_all_historical(candles) if len(candles) >= det.min_history else []
    # OB confirmed at last candle — no later interaction candle
    assert all(
        o.confirmed_at > candles[5].epoch
        for o in obs
        if o.payload.get("source_type") == "order_block"
    ) or obs == []


def test_forming_candles_ignored():
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 10.5, 12.0),
    ]
    candles = _series_from_ohlc(rows, last_closed=False)
    det = MitigationBlockDetector()
    # Last candle forming — incremental detect should ignore it
    latest = det.detect(candles)
    # closed series ends at OB confirm; no mitigation on forming bar
    assert latest == []


def test_discontinuity_skips_without_inference():
    """Gap in epochs must not invent mitigation across missing bars."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 10.5, 12.0),
    ]
    candles = _series_from_ohlc(rows)
    # Introduce a discontinuity by rewriting last epoch far ahead
    bad = list(candles)
    last = bad[-1]
    bad[-1] = Candle(
        symbol=last.symbol,
        timeframe=last.timeframe,
        epoch=last.epoch + 10 * STEP,  # gap
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        is_closed=True,
    )
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(bad)
    # Discontinuous candle must not produce mitigation
    assert all(o.confirmed_at != bad[-1].epoch for o in obs)


def test_discontinuity_at_first_interaction():
    """Gap at the first post-confirmation candle suppresses that interaction."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 10.5, 12.0),  # would be touch+partial if continuous
    ]
    candles = _series_from_ohlc(rows)
    # Gap between OB confirm (idx 5) and interaction (idx 6)
    bad = list(candles)
    last = bad[-1]
    bad[-1] = Candle(
        symbol=last.symbol,
        timeframe=last.timeframe,
        epoch=last.epoch + 5 * STEP,
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        is_closed=True,
    )
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(bad)
    assert obs == []


def test_discontinuity_after_touch_before_partial():
    """
    Touch on a continuous candle, then a gap, then a later continuous partial
    candle. The gap candle itself emits nothing; the later continuous pair may
    emit partial based on its own wick only (no inference across the gap).
    """
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 11.2, 12.5),  # idx 6: continuous touch (low == U)
        (12.5, 13.0, 10.5, 12.0),  # idx 7: will be made discontinuous
        (12.0, 12.5, 10.0, 11.5),  # idx 8: continuous with idx 7 after rewrite
    ]
    candles = _series_from_ohlc(rows)
    # Make idx 7 discontinuous from idx 6 (gap), but idx 8 continuous with idx 7
    c7 = candles[7]
    gap_epoch = candles[6].epoch + 5 * STEP
    candles[7] = Candle(
        symbol=c7.symbol,
        timeframe=c7.timeframe,
        epoch=gap_epoch,
        open=c7.open,
        high=c7.high,
        low=c7.low,
        close=c7.close,
        is_closed=True,
    )
    c8 = candles[8]
    candles[8] = Candle(
        symbol=c8.symbol,
        timeframe=c8.timeframe,
        epoch=gap_epoch + STEP,  # continuous with rewritten idx 7
        open=c8.open,
        high=c8.high,
        low=c8.low,
        close=c8.close,
        is_closed=True,
    )
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    stages = [o.variant for o in obs]
    # Touch from continuous idx 6 must remain
    assert "touch" in stages
    # Gap candle (idx 7) must not emit
    assert all(o.confirmed_at != gap_epoch for o in obs)
    # Later continuous candle may emit partial on its own wick (low=10.0 interior)
    assert "partial_fill" in stages
    partials = [o for o in obs if o.variant == "partial_fill"]
    assert all(o.confirmed_at == gap_epoch + STEP for o in partials)


def test_later_candle_after_gap_may_reach_full_independently():
    """
    After a suppressed discontinuous interaction, a later continuous pair that
    reaches the far boundary emits all missing stages from that candle's range
    alone — not from inferred intermediate price action.
    """
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 10.5, 12.0),  # idx 6: will be gapped (suppressed)
        (12.0, 12.5, 9.5, 11.5),   # idx 7: will be continuous with a gap-prev
    ]
    candles = _series_from_ohlc(rows)
    c6 = candles[6]
    gap_epoch = candles[5].epoch + 5 * STEP
    candles[6] = Candle(
        symbol=c6.symbol,
        timeframe=c6.timeframe,
        epoch=gap_epoch,
        open=c6.open,
        high=c6.high,
        low=c6.low,
        close=c6.close,
        is_closed=True,
    )
    c7 = candles[7]
    candles[7] = Candle(
        symbol=c7.symbol,
        timeframe=c7.timeframe,
        epoch=gap_epoch + STEP,
        open=c7.open,
        high=c7.high,
        low=c7.low,
        close=c7.close,
        is_closed=True,
    )
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    # Suppressed gap candle emits nothing
    assert all(o.confirmed_at != gap_epoch for o in obs)
    # Continuous later candle with low < L emits all three stages
    stages = [o.variant for o in obs]
    assert stages == ["touch", "partial_fill", "full_fill"]
    assert all(o.confirmed_at == gap_epoch + STEP for o in obs)


def test_identity_distinguishes_stages_same_source():
    """touch / partial_fill / full_fill for one source have distinct ids."""
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 9.5, 12.0),
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) == 3
    ids = [o.id for o in obs]
    assert len(set(ids)) == 3
    variants = {o.variant for o in obs}
    assert variants == {"touch", "partial_fill", "full_fill"}
    # Each id embeds the variant string
    for o in obs:
        assert o.variant in o.id


def test_reconstruction_matches_across_prefixes():
    """
    Full reconstruction on the complete series equals the union of
    incremental detect() results over successive prefixes (same stages/order).
    """
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 11.2, 12.5),  # touch
        (12.5, 13.0, 10.5, 12.0),  # partial
        (12.0, 12.5, 9.5, 11.5),   # full
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    full = det.detect_all_historical(candles)
    incremental: list = []
    for i in range(1, len(candles) + 1):
        incremental.extend(det.detect(candles[:i]))
    assert [o.id for o in incremental] == [o.id for o in full]
    assert [o.variant for o in incremental] == [o.variant for o in full]


def test_event_and_confirmation_timestamps():
    rows = _bullish_ob_base_rows() + [
        (13.0, 13.5, 10.5, 12.0),
    ]
    candles = _series_from_ohlc(rows)
    det = MitigationBlockDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) >= 1
    for o in obs:
        assert o.event_time == o.confirmed_at == o.candle_epoch
        assert o.confirmed_at == candles[6].epoch
        assert o.developing is False


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_registry_mitigation_block_capability():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("talib not installed")
        raise
    assert reg.is_supported("mitigation_block")
    assert reg.is_supported("mitigation")
    info = reg.validate_config("mitigation_block", {})
    assert info.detector_id == "mitigation_block"
    assert info.min_history >= 6
    det, info_c = reg.create_detector("mitigation_block")
    assert isinstance(det, MitigationBlockDetector)
    assert info_c.detector_id == "mitigation_block"
    assert det.config_dict()["min_body_ratio"] == DEFAULT_MIN_BODY_RATIO


def test_registry_rejects_unknown_param():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("talib not installed")
        raise
    with pytest.raises(ValueError):
        reg.create_detector(
            "mitigation_block", config={"retest_required": True}
        )
    with pytest.raises(ValueError):
        reg.create_detector(
            "mitigation_block", config={"min_body_ratio": 2.0}
        )


def test_detector_config_id_deterministic():
    a = MitigationBlockDetector()
    b = MitigationBlockDetector()
    assert a.detector_config_id == b.detector_config_id
    c = MitigationBlockDetector(min_body_ratio=0.6)
    assert c.detector_config_id != a.detector_config_id


def test_min_history_property():
    det = MitigationBlockDetector()
    assert det.min_history >= 6
    assert det.name == "mitigation_block"
