"""Phase 3B — zone interaction detection (touch, rejection, break)."""

from __future__ import annotations

import pytest

from cpab.detectors.zone_interaction import ZoneInteractionDetector
from cpab.detectors.support_resistance import SupportResistanceZoneDetector
from cpab.detectors.registry import build_default_registry, deterministic_config_id
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
    bars: list[tuple[float, float, float, float]],
    start: int = 1_700_000_000,
    *,
    step: int = 60,
    timeframe: str = "1m",
) -> list[Candle]:
    """bars: list of (open, high, low, close)."""
    out = []
    for i, (o, h, l, c) in enumerate(bars):
        out.append(_c(start + i * step, o, h, l, c, timeframe=timeframe))
    return out


# ---------------------------------------------------------------------------
# Fixtures that produce a confirmed support or resistance zone early,
# then additional interaction candles.
#
# With left=1, right=1:
#   Support from swing_low at index 1 needs bars 0,1,2 (confirm at bar 2).
#   Resistance from swing_high at index 1 needs bars 0,1,2 (confirm at bar 2).
# ---------------------------------------------------------------------------


def _support_zone_series(
    interaction_bars: list[tuple[float, float, float, float]],
    *,
    start: int = 1_700_000_000,
    step: int = 60,
) -> list[Candle]:
    """
    Bars 0-2 form a swing_low at bar 1 (low=100), confirmed at bar 2.
    Support zone level P=100 becomes available at bar 2 epoch.
    interaction_bars are appended after bar 2.
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


def _zone_id_for(candles: list[Candle], variant: str) -> str:
    zones = SupportResistanceZoneDetector().detect_all_historical(candles)
    matching = [z for z in zones if z.variant == variant]
    assert matching, f"expected a {variant} zone"
    return matching[0].id


# ===========================================================================
# Touch
# ===========================================================================


def test_touch_exact_level_contact():
    # Candle with low=high=close=open=100 (exact touch of support 100)
    candles = _support_zone_series([(100.0, 100.0, 100.0, 100.0)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    touches = [o for o in obs if o.variant == "touch"]
    assert len(touches) == 1
    t = touches[0]
    assert t.type == "zone_interaction"
    assert t.direction is None
    assert t.payload["level_price"] == 100.0
    assert t.payload["zone_variant"] == "support"


def test_touch_wick_only_contact():
    # low dips to 99.5, close above 100 — wick touches support
    candles = _support_zone_series([(101.0, 102.0, 99.5, 101.5)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    touches = [o for o in obs if o.variant == "touch"]
    assert len(touches) == 1
    assert touches[0].payload["low"] == 99.5


def test_touch_candle_crossing_level():
    # open above, close below support — still a touch (and a break)
    candles = _support_zone_series([(101.0, 101.5, 98.0, 99.0)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    touches = [o for o in obs if o.variant == "touch"]
    assert len(touches) == 1


def test_no_touch_when_entirely_above():
    candles = _support_zone_series([(101.0, 102.0, 100.5, 101.5)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    touches = [o for o in obs if o.variant == "touch"]
    assert touches == []


def test_no_touch_when_entirely_below():
    candles = _support_zone_series([(98.0, 99.0, 97.0, 98.5)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    touches = [o for o in obs if o.variant == "touch"]
    assert touches == []


def test_touch_resistance_zone():
    candles = _resistance_zone_series([(109.0, 110.5, 108.5, 109.5)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    touches = [o for o in obs if o.variant == "touch"]
    assert len(touches) == 1
    assert touches[0].payload["zone_variant"] == "resistance"
    assert touches[0].payload["level_price"] == 110.0


def test_forming_candle_excluded():
    candles = _support_zone_series([(100.0, 101.0, 99.0, 100.5)])
    # Mark last as forming
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
    d = ZoneInteractionDetector()
    assert d.detect(candles) == []
    assert d.detect_all_historical(candles) == []


# ===========================================================================
# Rejection
# ===========================================================================


def test_valid_support_rejection():
    # touch support (low <= 100) and close > 100
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.8)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    rejections = [o for o in obs if o.variant == "rejection"]
    assert len(rejections) == 1
    r = rejections[0]
    assert r.direction == "bullish"
    assert r.payload["zone_variant"] == "support"


def test_valid_resistance_rejection():
    candles = _resistance_zone_series([(109.5, 110.5, 109.0, 109.2)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    rejections = [o for o in obs if o.variant == "rejection"]
    assert len(rejections) == 1
    r = rejections[0]
    assert r.direction == "bearish"
    assert r.payload["zone_variant"] == "resistance"


def test_close_exactly_at_level_not_rejection():
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.0)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    rejections = [o for o in obs if o.variant == "rejection"]
    assert rejections == []
    touches = [o for o in obs if o.variant == "touch"]
    assert len(touches) == 1


def test_close_on_wrong_side_not_rejection():
    # Support touch but close below (break path, not rejection)
    candles = _support_zone_series([(100.5, 101.0, 99.0, 99.5)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    rejections = [o for o in obs if o.variant == "rejection"]
    assert rejections == []


def test_touch_and_rejection_emitted_together_distinct_ids():
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.8)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    touches = [o for o in obs if o.variant == "touch"]
    rejections = [o for o in obs if o.variant == "rejection"]
    assert len(touches) == 1 and len(rejections) == 1
    assert touches[0].id != rejections[0].id
    assert touches[0].reference_ids == rejections[0].reference_ids


# ===========================================================================
# Break
# ===========================================================================


def test_support_break_downward():
    # prev (confirm bar) close 100.8 >= 100; current close 99.0 < 100
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    breaks = [o for o in obs if o.variant == "break"]
    assert len(breaks) == 1
    b = breaks[0]
    assert b.direction == "bearish"
    assert b.payload["zone_variant"] == "support"
    assert b.payload["previous_close"] == 100.8
    assert b.payload["close"] == 99.0


def test_resistance_break_upward():
    candles = _resistance_zone_series([(109.5, 111.0, 109.0, 110.5)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    breaks = [o for o in obs if o.variant == "break"]
    assert len(breaks) == 1
    b = breaks[0]
    assert b.direction == "bullish"
    assert b.payload["zone_variant"] == "resistance"


def test_wick_only_penetration_not_break():
    # low below support but close still above
    candles = _support_zone_series([(100.5, 101.0, 99.0, 100.5)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    breaks = [o for o in obs if o.variant == "break"]
    assert breaks == []


def test_previous_close_exactly_at_level_then_strict_crossing():
    # Need prev close == 100. Confirm bar low must stay above 100 so the
    # swing at bar 1 remains strictly lower under tie_policy=strict.
    start = 1_700_000_000
    candles = [
        _c(start, 101.0, 102.0, 100.5, 101.0),
        _c(start + 60, 101.0, 101.5, 100.0, 100.5),  # swing low 100
        _c(start + 120, 100.5, 101.0, 100.2, 100.0),  # confirm, close at level
        _c(start + 180, 100.0, 100.2, 98.5, 99.0),  # break
    ]
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    breaks = [o for o in obs if o.variant == "break"]
    assert len(breaks) == 1
    assert breaks[0].direction == "bearish"
    assert breaks[0].payload["previous_close"] == 100.0


def test_current_close_exactly_at_level_does_not_break():
    candles = _support_zone_series([(100.5, 101.0, 99.0, 100.0)])
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    breaks = [o for o in obs if o.variant == "break"]
    assert breaks == []


def test_remains_beyond_level_no_repeated_break():
    # First interaction candle breaks; second stays below — only one break
    candles = _support_zone_series(
        [
            (100.5, 101.0, 98.5, 99.0),  # break
            (99.0, 99.5, 98.0, 98.5),  # remains below
        ]
    )
    d = ZoneInteractionDetector()
    all_obs = d.detect_all_historical(candles)
    breaks = [o for o in all_obs if o.variant == "break"]
    assert len(breaks) == 1


def test_zone_available_while_price_already_beyond_no_false_break():
    """
    After a zone is broken, remaining beyond must not re-emit.
    Separately: when the first candle after a gap leaves price already beyond
    with no continuous prior, no break is invented.
    """
    start = 1_700_000_000
    # Valid swing low at 100 confirmed at bar 2 (close still above).
    # Bar 3 breaks. Bars 4–5 remain beyond — only one break total.
    candles = [
        _c(start, 101.0, 102.0, 100.5, 101.0),
        _c(start + 60, 101.0, 101.5, 100.0, 100.5),
        _c(start + 120, 100.5, 101.0, 100.2, 100.8),  # confirm zone
        _c(start + 180, 100.5, 100.8, 98.0, 98.5),  # break
        _c(start + 240, 98.5, 99.0, 97.5, 98.0),  # remains beyond
        _c(start + 300, 98.0, 98.5, 97.0, 97.5),  # remains beyond
    ]
    d = ZoneInteractionDetector()
    all_obs = d.detect_all_historical(candles)
    breaks = [o for o in all_obs if o.variant == "break"]
    assert len(breaks) == 1
    assert breaks[0].confirmed_at == start + 180

    # Gap then already-beyond: discontinuous prior suppresses break evaluation
    gapped = [
        _c(start, 101.0, 102.0, 100.5, 101.0),
        _c(start + 60, 101.0, 101.5, 100.0, 100.5),
        _c(start + 120, 100.5, 101.0, 100.2, 100.8),
        # gap
        _c(start + 240, 98.0, 98.5, 97.0, 97.5),  # already beyond, discontinuous
    ]
    breaks2 = [o for o in d.detect_all_historical(gapped) if o.variant == "break"]
    assert breaks2 == []


def test_missing_discontinuous_previous_suppresses_crossing():
    start = 1_700_000_000
    candles = [
        _c(start, 101.0, 102.0, 100.5, 101.0),
        _c(start + 60, 101.0, 101.5, 100.0, 100.5),
        _c(start + 120, 100.5, 101.0, 100.2, 100.8),
        # gap: skip one bar
        _c(start + 240, 100.5, 101.0, 98.5, 99.0),
    ]
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    breaks = [o for o in obs if o.variant == "break"]
    assert breaks == []


def test_no_break_from_forming_candles():
    candles = _support_zone_series([(100.5, 101.0, 98.5, 99.0)])
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
    d = ZoneInteractionDetector()
    assert d.detect(candles) == []


# ===========================================================================
# Identity and causality
# ===========================================================================


def test_each_interaction_references_source_zone_id():
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.8)])
    zone_id = _zone_id_for(candles, "support")
    d = ZoneInteractionDetector()
    obs = d.detect(candles)
    assert obs
    for o in obs:
        assert o.reference_ids == (zone_id,)
        assert o.payload["source_zone_confirmed_at"] is not None


def test_two_zones_same_price_distinct_observations():
    # Two support zones at the same price from successive swings
    start = 1_700_000_000
    # Build two swing lows both at 100
    candles = [
        _c(start, 101.0, 102.0, 100.5, 101.0),
        _c(start + 60, 101.0, 101.5, 100.0, 100.5),  # swing low 1
        _c(start + 120, 100.5, 101.0, 100.2, 100.8),  # confirm 1
        _c(start + 180, 100.8, 101.2, 100.3, 100.9),
        _c(start + 240, 100.9, 101.0, 100.0, 100.4),  # swing low 2 at 100
        _c(start + 300, 100.4, 100.8, 100.1, 100.6),  # confirm 2
        _c(start + 360, 100.5, 101.0, 99.5, 100.7),  # touch both
    ]
    d = ZoneInteractionDetector(left_strength=1, right_strength=1, tie_policy="all")
    obs = d.detect_all_historical(candles)
    touches = [o for o in obs if o.variant == "touch" and o.confirmed_at == start + 360]
    # At least two distinct source zones at price 100
    zone_ids = {o.reference_ids[0] for o in touches}
    assert len(zone_ids) >= 2
    touch_ids = {o.id for o in touches}
    assert len(touch_ids) == len(touches)


def test_interactions_not_before_zone_confirmed_at():
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.8)])
    d = ZoneInteractionDetector()
    all_obs = d.detect_all_historical(candles)
    zones = SupportResistanceZoneDetector().detect_all_historical(candles)
    support = [z for z in zones if z.variant == "support"][0]
    for o in all_obs:
        assert support.confirmed_at is not None
        assert o.confirmed_at is not None
        assert o.confirmed_at >= support.confirmed_at
        assert support.confirmed_at <= o.candle_epoch


def test_deterministic_ids_across_reconstruction():
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.8)])
    d = ZoneInteractionDetector()
    a = d.detect_all_historical(candles)
    b = d.detect_all_historical(candles)
    assert [o.id for o in a] == [o.id for o in b]
    store = ObservationStore()
    assert store.accept(a) == len(a)
    assert store.accept(b) == 0


def test_touch_rejection_break_identities_distinct():
    # Candle that touches, rejects, and would also satisfy... wait, rejection
    # has close above so not a break. Separate candles.
    candles = _support_zone_series(
        [
            (100.5, 101.0, 99.5, 100.8),  # touch + rejection
            (100.8, 101.0, 98.5, 99.0),  # break
        ]
    )
    d = ZoneInteractionDetector()
    all_obs = d.detect_all_historical(candles)
    variants = {(o.variant, o.id) for o in all_obs}
    ids = [o.id for o in all_obs]
    assert len(ids) == len(set(ids))
    assert any(o.variant == "touch" for o in all_obs)
    assert any(o.variant == "rejection" for o in all_obs)
    assert any(o.variant == "break" for o in all_obs)


def test_mixed_symbol_timeframe_rejected():
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.8)])
    candles[0] = Candle(
        symbol="OTHER",
        timeframe=candles[0].timeframe,
        epoch=candles[0].epoch,
        open=candles[0].open,
        high=candles[0].high,
        low=candles[0].low,
        close=candles[0].close,
        is_closed=True,
    )
    d = ZoneInteractionDetector()
    assert d.detect(candles) == []
    assert d.detect_all_historical(candles) == []


# ===========================================================================
# Consistency and integration
# ===========================================================================


def test_historical_and_latest_agree():
    candles = _support_zone_series(
        [
            (100.5, 101.0, 99.5, 100.8),
            (100.8, 101.0, 98.5, 99.0),
        ]
    )
    d = ZoneInteractionDetector()
    all_obs = d.detect_all_historical(candles)
    latest_epoch = candles[-1].epoch
    expected = [o for o in all_obs if o.confirmed_at == latest_epoch]
    inc = d.detect(candles)
    assert [o.id for o in inc] == [o.id for o in expected]


def test_progressive_history_no_duplicate_breaks():
    candles = _support_zone_series(
        [
            (100.5, 101.0, 98.5, 99.0),  # break
            (99.0, 99.5, 98.0, 98.5),
            (98.5, 99.0, 97.5, 98.0),
        ]
    )
    d = ZoneInteractionDetector()
    seen_break_ids: set[str] = set()
    for n in range(4, len(candles) + 1):
        subset = candles[:n]
        obs = d.detect_all_historical(subset)
        breaks = [o for o in obs if o.variant == "break"]
        for b in breaks:
            seen_break_ids.add(b.id)
    assert len(seen_break_ids) == 1


def test_registry_accepts_valid_config():
    reg = build_default_registry()
    assert reg.is_supported("zone_interaction")
    assert reg.is_supported("zone_touch")
    assert reg.is_supported("zone_rejection")
    assert reg.is_supported("zone_break")
    det, info = reg.create_detector("zone_interaction")
    assert isinstance(det, ZoneInteractionDetector)
    assert info.min_history == 4
    assert det.min_history == 4
    assert info.detector_config_id.startswith("cfg:")
    assert det.detector_config_id == info.detector_config_id

    det2, info2 = reg.create_detector(
        "zone_interaction",
        config={"left_strength": 2, "right_strength": 1, "tie_policy": "first"},
    )
    assert det2.left_strength == 2
    assert det2.right_strength == 1
    assert det2.tie_policy == "first"
    assert info2.min_history == 5
    assert info2.detector_config_id == deterministic_config_id(
        {"left_strength": 2, "right_strength": 1, "tie_policy": "first"}
    )


def test_registry_rejects_unsupported_params():
    reg = build_default_registry()
    with pytest.raises(ValueError, match="unknown params"):
        reg.validate_config("zone_interaction", {"left_strength": 1, "width": 0.5})
    with pytest.raises(ValueError, match="left_strength"):
        reg.create_detector("zone_interaction", config={"left_strength": 1.5})
    with pytest.raises(ValueError, match="tie_policy"):
        reg.create_detector("zone_interaction", config={"tie_policy": "nearest"})


def test_prior_phases_still_registered():
    reg = build_default_registry()
    for concept in (
        "engulfing",
        "swing_high",
        "equal_high",
        "market_structure",
        "structure_break",
        "support_resistance_zone",
        "zone_interaction",
    ):
        assert reg.is_supported(concept), concept


def test_invalid_strength_raises():
    with pytest.raises(ValueError, match="left_strength"):
        ZoneInteractionDetector(left_strength=0)
    with pytest.raises(ValueError, match="right_strength"):
        ZoneInteractionDetector(right_strength=-1)
    with pytest.raises(ValueError, match="tie_policy"):
        ZoneInteractionDetector(tie_policy="middle")


def test_insufficient_history_no_emit():
    candles = _series(
        [
            (101.0, 102.0, 100.5, 101.0),
            (101.0, 101.5, 100.0, 100.5),
            (100.5, 101.0, 100.2, 100.8),
        ]
    )
    d = ZoneInteractionDetector()
    # min_history = 4; only 3 bars
    assert d.detect(candles) == []
    assert d.detect_all_historical(candles) == []


def test_zone_level_price_rejects_missing_and_nonfinite():
    """Malformed zone payloads must not fabricate interactions at price 0.0."""
    from cpab.detectors.zone_interaction import _zone_level_price

    def _zone(payload: dict) -> Observation:
        return Observation.create(
            type="support_resistance_zone",
            symbol="R_100",
            timeframe="1m",
            candle_epoch=1_700_000_000,
            confirmed_at=1_700_000_120,
            variant="support",
            detector_id="support_resistance_zone",
            payload=payload,
            id="synthetic-zone",
        )

    assert _zone_level_price(_zone({"source_price": 100.0})) == 100.0
    assert _zone_level_price(_zone({"lower_bound": 99.5, "upper_bound": 99.5})) == 99.5
    assert _zone_level_price(_zone({})) is None
    assert _zone_level_price(_zone({"source_price": float("nan")})) is None
    assert _zone_level_price(_zone({"source_price": float("inf")})) is None
    assert _zone_level_price(_zone({"source_price": True})) is None
    assert _zone_level_price(_zone({"source_price": "100"})) is None


def test_malformed_zone_skipped_no_zero_price_interaction(monkeypatch):
    """A zone missing level price is skipped; no touch/rejection/break at 0.0."""
    candles = _support_zone_series([(100.5, 101.0, 99.5, 100.8)])
    d = ZoneInteractionDetector()

    real_zones = SupportResistanceZoneDetector().detect_all_historical(candles)
    assert real_zones  # fixture produces a valid support zone

    bad = Observation.create(
        type="support_resistance_zone",
        symbol=candles[0].symbol,
        timeframe=candles[0].timeframe,
        candle_epoch=candles[1].epoch,
        confirmed_at=candles[2].epoch,
        variant="support",
        detector_id="support_resistance_zone",
        payload={},  # no source_price / bounds
        id="malformed-zone-no-price",
    )

    def _fake_detect_all(_candles):
        return list(real_zones) + [bad]

    monkeypatch.setattr(d._zones, "detect_all_historical", _fake_detect_all)
    obs = d.detect_all_historical(candles)
    # Valid zone still interacts; malformed zone produces nothing at price 0
    assert any(o.reference_ids == (real_zones[0].id,) for o in obs)
    assert not any(o.reference_ids == (bad.id,) for o in obs)
    assert not any(o.payload.get("level_price") == 0.0 for o in obs)
