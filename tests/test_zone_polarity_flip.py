"""Phase 3D — zone polarity flip detection from Phase 3B breaks."""

from __future__ import annotations

import pytest

from cpab.detectors.zone_polarity_flip import ZonePolarityFlipDetector
from cpab.detectors.zone_interaction import ZoneInteractionDetector
from cpab.detectors.support_resistance import SupportResistanceZoneDetector
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.models import Candle, Observation


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
    bars: list[tuple[float, float, float, float]],
    start: int = 1_700_000_000,
    *,
    step: int = 60,
    timeframe: str = "1m",
    symbol: str = "R_100",
) -> list[Candle]:
    """bars: list of (open, high, low, close)."""
    out = []
    for i, (o, h, l, c) in enumerate(bars):
        out.append(
            _c(
                start + i * step,
                o,
                h,
                l,
                c,
                timeframe=timeframe,
                symbol=symbol,
            )
        )
    return out


def _support_zone_series(
    interaction_bars: list[tuple[float, float, float, float]],
    *,
    start: int = 1_700_000_000,
    step: int = 60,
) -> list[Candle]:
    """
    Bars 0-2 form a swing_low at bar 1 (low=100), confirmed at bar 2.
    Support zone level P=100 becomes available at bar 2 epoch.
    """
    base = [
        (101.0, 102.0, 100.5, 101.0),  # 0 left
        (101.0, 101.5, 100.0, 100.5),  # 1 swing low at 100
        (100.5, 101.0, 100.2, 100.8),  # 2 confirm
    ]
    return _series(base + interaction_bars, start=start, step=step)


def _resistance_zone_series(
    interaction_bars: list[tuple[float, float, float, float]],
    *,
    start: int = 1_700_000_000,
    step: int = 60,
) -> list[Candle]:
    """
    Bars 0-2 form a swing_high at bar 1 (high=110), confirmed at bar 2.
    Resistance zone level P=110 becomes available at bar 2 epoch.
    """
    base = [
        (108.0, 109.0, 107.5, 108.5),  # 0 left
        (108.5, 110.0, 108.0, 109.0),  # 1 swing high at 110
        (109.0, 109.5, 108.5, 109.0),  # 2 confirm
    ]
    return _series(base + interaction_bars, start=start, step=step)


# ===========================================================================
# Core polarity-flip emission
# ===========================================================================


def test_support_bearish_break_emits_support_to_resistance():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert len(flips) == 1
    f = flips[0]
    assert f.type == "zone_polarity_flip"
    assert f.variant == "support_to_resistance"
    assert f.direction == "bearish"
    assert f.payload["source_zone_variant"] == "support"
    assert f.payload["level_price"] == 100.0


def test_resistance_bullish_break_emits_resistance_to_support():
    candles = _resistance_zone_series([(109.5, 111.0, 109.0, 110.5)])
    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert len(flips) == 1
    f = flips[0]
    assert f.type == "zone_polarity_flip"
    assert f.variant == "resistance_to_support"
    assert f.direction == "bullish"
    assert f.payload["source_zone_variant"] == "resistance"
    assert f.payload["level_price"] == 110.0


def test_original_zone_price_preserved():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert flips[0].payload["level_price"] == 100.0


def test_timing_matches_source_break():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    ix = ZoneInteractionDetector()
    breaks = [
        o
        for o in ix.detect_all_historical(candles)
        if o.variant == "break"
    ]
    assert len(breaks) == 1
    brk = breaks[0]

    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert len(flips) == 1
    f = flips[0]
    assert f.event_time == brk.event_time
    assert f.confirmed_at == brk.confirmed_at
    assert f.candle_epoch == brk.candle_epoch


def test_reference_ids_order_zone_then_break():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    ix = ZoneInteractionDetector()
    breaks = [o for o in ix.detect_all_historical(candles) if o.variant == "break"]
    assert len(breaks) == 1
    brk = breaks[0]
    zone_id = brk.reference_ids[0]

    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert len(flips) == 1
    f = flips[0]
    assert f.reference_ids == (zone_id, brk.id)


# ===========================================================================
# Identity / uniqueness
# ===========================================================================


def test_different_source_zones_same_price_distinct_flip_ids():
    """Two distinct support zones that happen to share a price produce distinct flips."""
    # Build a series with two separate swing lows both at 100, each broken.
    # Bar layout (left=1, right=1):
    # 0 left, 1 low@100, 2 confirm+left for next, 3 low@100, 4 confirm,
    # then two break candles that each cross after their zone is eligible.
    start = 1_700_000_000
    step = 60
    bars = [
        (101.0, 102.0, 100.5, 101.0),  # 0
        (101.0, 101.5, 100.0, 100.5),  # 1 swing low A
        (100.5, 101.0, 100.2, 100.8),  # 2 confirm A
        (101.0, 101.5, 100.0, 100.6),  # 3 swing low B (same price)
        (100.6, 101.0, 100.3, 100.7),  # 4 confirm B
        (100.5, 100.8, 98.0, 99.0),    # 5 break both eligible zones
    ]
    candles = _series(bars, start=start, step=step)
    zones = SupportResistanceZoneDetector().detect_all_historical(candles)
    supports = [z for z in zones if z.variant == "support"]
    assert len(supports) >= 2

    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert len(flips) >= 2
    ids = {f.id for f in flips}
    assert len(ids) == len(flips)
    zone_ids_in_flips = {f.reference_ids[0] for f in flips}
    assert len(zone_ids_in_flips) >= 2


def test_different_break_observations_no_id_collision():
    candles = _support_zone_series(
        [
            (100.5, 101.0, 98.5, 99.0),  # first break
            (99.0, 99.5, 98.0, 98.5),    # remains beyond — no second break
        ]
    )
    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    # Only one break → only one flip
    assert len(flips) == 1


def test_no_duplicate_flip_for_same_source_zone_break():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert len(flips) == 1
    # Reconstruct again — same IDs, still one
    flips2 = det.detect_all_historical(candles)
    assert len(flips2) == 1
    assert flips[0].id == flips2[0].id


# ===========================================================================
# Negative cases (must not emit)
# ===========================================================================


def test_non_break_interactions_never_emit_flips():
    # Touch + rejection only
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.8)])
    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert flips == []


def test_wick_only_penetration_never_emits_flip():
    candles = _support_zone_series([(100.5, 101.0, 99.0, 100.5)])
    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert flips == []


def test_remaining_beyond_level_no_new_flip():
    candles = _support_zone_series(
        [
            (100.5, 101.0, 98.5, 99.0),  # break
            (99.0, 99.5, 98.0, 98.5),    # remains beyond
        ]
    )
    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert len(flips) == 1


def test_gap_suppresses_flip_when_phase3b_suppresses_break():
    # Insert a gap so Phase 3B will not emit a break on the discontinuous candle.
    start = 1_700_000_000
    step = 60
    # Base support zone (bars 0-2 continuous), then a large gap before the
    # would-be break candle.
    base = [
        (101.0, 102.0, 100.5, 101.0),
        (101.0, 101.5, 100.0, 100.5),
        (100.5, 101.0, 100.2, 100.8),
    ]
    candles = _series(base, start=start, step=step)
    # Append a candle after a multi-step gap
    gap_epoch = start + 2 * step + 5 * step  # skip 4 bars
    candles.append(
        _c(gap_epoch, 100.5, 101.0, 98.5, 99.0)
    )
    ix = ZoneInteractionDetector()
    breaks = [o for o in ix.detect_all_historical(candles) if o.variant == "break"]
    assert breaks == []

    det = ZonePolarityFlipDetector()
    flips = det.detect_all_historical(candles)
    assert flips == []


def test_forming_candles_cannot_trigger_flip():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    # Mark last candle as forming
    candles[-1] = Candle(
        symbol=candles[-1].symbol,
        timeframe=candles[-1].timeframe,
        epoch=candles[-1].epoch,
        open=candles[-1].open,
        high=candles[-1].high,
        low=candles[-1].low,
        close=candles[-1].close,
        is_closed=False,
    )
    det = ZonePolarityFlipDetector()
    # Incremental path should ignore forming candle
    flips = det.detect(candles)
    assert flips == []
    # Historical also filters to closed only
    flips_h = det.detect_all_historical(candles)
    assert flips_h == []


def test_empty_input():
    det = ZonePolarityFlipDetector()
    assert det.detect([]) == []
    assert det.detect_all_historical([]) == []


def test_mixed_symbol_timeframe_returns_empty():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    # Corrupt one candle
    bad = list(candles)
    bad[1] = Candle(
        symbol="OTHER",
        timeframe=bad[1].timeframe,
        epoch=bad[1].epoch,
        open=bad[1].open,
        high=bad[1].high,
        low=bad[1].low,
        close=bad[1].close,
        is_closed=True,
    )
    det = ZonePolarityFlipDetector()
    assert det.detect_all_historical(bad) == []


# ===========================================================================
# Historical / incremental consistency
# ===========================================================================


def test_repeated_historical_detection_identical():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    det = ZonePolarityFlipDetector()
    a = det.detect_all_historical(candles)
    b = det.detect_all_historical(candles)
    assert len(a) == 1
    assert a[0].id == b[0].id
    assert a[0].event_time == b[0].event_time
    assert a[0].confirmed_at == b[0].confirmed_at
    assert a[0].reference_ids == b[0].reference_ids


def test_incremental_emits_only_latest_confirmed():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    det = ZonePolarityFlipDetector()
    # Full history has the flip; incremental on full history should emit it
    # because confirmed_at == latest closed epoch.
    flips = det.detect(candles)
    assert len(flips) == 1
    assert flips[0].confirmed_at == candles[-1].epoch

    # On a prefix that ends before the break, no flip.
    prefix = candles[:-1]
    assert det.detect(prefix) == []


def test_historical_and_incremental_agree():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    det = ZonePolarityFlipDetector()
    hist = det.detect_all_historical(candles)
    incr = det.detect(candles)
    assert len(hist) == 1
    assert len(incr) == 1
    assert hist[0].id == incr[0].id
    assert hist[0].confirmed_at == incr[0].confirmed_at


# ===========================================================================
# Registry
# ===========================================================================


def test_registry_capability_lookup_and_factory():
    pytest.importorskip("talib")
    reg = build_default_registry()
    assert reg.is_supported("zone_polarity_flip")
    info = reg.validate_config("zone_polarity_flip", {})
    assert info.detector_id == "zone_polarity_flip"
    assert info.min_history == 4  # left=1 + 1 + right=1 + 1
    det, _info = reg.create_detector("zone_polarity_flip")
    assert isinstance(det, ZonePolarityFlipDetector)
    assert det.left_strength == 1
    assert det.right_strength == 1
    assert det.tie_policy == "strict"


def test_registry_config_validation_and_deterministic_id():
    pytest.importorskip("talib")
    reg = build_default_registry()
    cfg = {"left_strength": 2, "right_strength": 1, "tie_policy": "strict"}
    info = reg.validate_config("zone_polarity_flip", cfg)
    expected_id = deterministic_config_id(
        {"left_strength": 2, "right_strength": 1, "tie_policy": "strict"}
    )
    assert info.detector_config_id == expected_id
    assert info.min_history == 2 + 1 + 1 + 1

    det, _info = reg.create_detector("zone_polarity_flip", config=cfg)
    assert det.left_strength == 2
    assert det.detector_config_id == expected_id


def test_registry_rejects_unknown_params():
    pytest.importorskip("talib")
    reg = build_default_registry()
    with pytest.raises(ValueError):
        reg.validate_config("zone_polarity_flip", {"unknown_param": 1})
