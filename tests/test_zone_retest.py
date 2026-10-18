"""Phase 3C — zone retest detection (first confirmed retest after break)."""

from __future__ import annotations

import pytest

from cpab.detectors.zone_retest import ZoneRetestDetector
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
# Fixtures: confirmed support/resistance zone, then break, then retest path.
#
# With left=1, right=1:
#   Support from swing_low at index 1 needs bars 0,1,2 (confirm at bar 2).
#   Resistance from swing_high at index 1 needs bars 0,1,2 (confirm at bar 2).
# ---------------------------------------------------------------------------


def _support_break_series(
    post_break_bars: list[tuple[float, float, float, float]],
    *,
    start: int = 1_700_000_000,
    step: int = 60,
) -> list[Candle]:
    """
    Bars 0-2: swing_low at 100, confirmed at bar 2.
    Bar 3: break support (close strictly below 100).
    post_break_bars appended after the break.
    """
    base = [
        (101.0, 102.0, 100.5, 101.0),  # 0 left
        (101.0, 101.5, 100.0, 100.5),  # 1 swing low at 100
        (100.5, 101.0, 100.2, 100.8),  # 2 confirm
        (100.5, 100.8, 98.5, 99.0),    # 3 break (close < 100)
    ]
    return _series(base + post_break_bars, start=start, step=step)


def _resistance_break_series(
    post_break_bars: list[tuple[float, float, float, float]],
    *,
    start: int = 1_700_000_000,
    step: int = 60,
) -> list[Candle]:
    """
    Bars 0-2: swing_high at 110, confirmed at bar 2.
    Bar 3: break resistance (close strictly above 110).
    """
    base = [
        (108.0, 109.0, 107.5, 108.5),  # 0 left
        (108.5, 110.0, 108.0, 109.0),  # 1 swing high at 110
        (109.0, 109.5, 108.5, 109.0),  # 2 confirm
        (109.5, 111.0, 109.0, 110.5),  # 3 break (close > 110)
    ]
    return _series(base + post_break_bars, start=start, step=step)


# ===========================================================================
# 1. Valid bearish retest of broken support
# ===========================================================================


def test_valid_bearish_retest_of_broken_support():
    # After break at 99, approach from below: contact level and close below.
    candles = _support_break_series([
        (99.0, 99.5, 98.0, 98.5),   # remain below
        (98.5, 100.2, 98.2, 99.2),  # contact (high>=100), close < 100 → retest
    ])
    d = ZoneRetestDetector()
    obs = d.detect(candles)
    retests = [o for o in obs if o.type == "zone_retest"]
    assert len(retests) == 1
    r = retests[0]
    assert r.variant == "retest"
    assert r.direction == "bearish"
    assert r.developing is False
    assert r.payload["zone_variant"] == "support"
    assert r.payload["level_price"] == 100.0
    assert r.payload["close"] == 99.2
    assert len(r.reference_ids) == 2


# ===========================================================================
# 2. Valid bullish retest of broken resistance
# ===========================================================================


def test_valid_bullish_retest_of_broken_resistance():
    candles = _resistance_break_series([
        (110.5, 111.0, 110.2, 110.8),  # remain above
        (110.8, 111.2, 109.8, 110.6),  # contact (low<=110), close > 110 → retest
    ])
    d = ZoneRetestDetector()
    obs = d.detect(candles)
    retests = [o for o in obs if o.type == "zone_retest"]
    assert len(retests) == 1
    r = retests[0]
    assert r.variant == "retest"
    assert r.direction == "bullish"
    assert r.payload["zone_variant"] == "resistance"
    assert r.payload["level_price"] == 110.0


# ===========================================================================
# 3. Touch without a preceding confirmed break
# ===========================================================================


def test_touch_without_preceding_break_no_retest():
    # Support zone only; touch but no break → no retest.
    base = [
        (101.0, 102.0, 100.5, 101.0),
        (101.0, 101.5, 100.0, 100.5),
        (100.5, 101.0, 100.2, 100.8),
        (100.5, 101.0, 99.5, 100.5),  # touch, close above — rejection, not break
    ]
    candles = _series(base)
    d = ZoneRetestDetector()
    obs = d.detect_all_historical(candles)
    retests = [o for o in obs if o.type == "zone_retest"]
    assert retests == []


# ===========================================================================
# 4. Contact on the break candle itself
# ===========================================================================


def test_contact_on_break_candle_not_retest():
    # Break candle already contacts the level (range includes 100).
    # Retest must be on a subsequent candle.
    candles = _support_break_series([])  # only through break
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert retests == []

    # Explicit post-break contact on a later candle is required.
    candles2 = _support_break_series([
        (99.0, 100.5, 98.5, 99.0),  # contact + close below → valid retest
    ])
    all_obs2 = d.detect_all_historical(candles2)
    retests2 = [o for o in all_obs2 if o.type == "zone_retest"]
    assert len(retests2) == 1
    # Retest candle is after the break candle
    assert retests2[0].candle_epoch > candles2[3].epoch


# ===========================================================================
# 5. Contact with close on the original side
# ===========================================================================


def test_contact_close_on_original_side_invalidates():
    # After break, candle contacts level but closes above (original side)
    # → invalidates the pending sequence; no retest.
    candles = _support_break_series([
        (99.0, 100.5, 98.5, 100.5),  # contact, close > 100 → invalidate
        (100.5, 101.0, 99.5, 99.5),  # later contact+close below — still no retest
    ])
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert retests == []


# ===========================================================================
# 6. Close exactly at the level
# ===========================================================================


def test_close_exactly_at_level_is_neutral():
    # Contact with close == level: neither confirm nor invalidate.
    # Subsequent valid retest should still emit.
    candles = _support_break_series([
        (99.0, 100.5, 98.5, 100.0),  # contact, close exactly at level — neutral
        (100.0, 100.2, 98.8, 99.2),  # contact, close < 100 → retest
    ])
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert len(retests) == 1
    assert retests[0].payload["close"] == 99.2


def test_close_exactly_at_level_alone_no_retest():
    candles = _support_break_series([
        (99.0, 100.5, 98.5, 100.0),  # contact, close at level only
    ])
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert retests == []


# ===========================================================================
# 7. Price remaining beyond the level without contact
# ===========================================================================


def test_remain_beyond_without_contact_no_retest():
    candles = _support_break_series([
        (99.0, 99.5, 98.0, 98.5),
        (98.5, 99.0, 97.5, 98.0),  # never touches 100
    ])
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert retests == []


# ===========================================================================
# 8. First confirmed retest consumes the permitted event
# ===========================================================================


def test_first_retest_consumes_permitted_event():
    candles = _support_break_series([
        (99.0, 100.2, 98.5, 99.2),  # first retest
        (99.2, 100.3, 98.8, 99.0),  # second contact+close below — must not emit
    ])
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert len(retests) == 1


# ===========================================================================
# 9. Invalidation prevents subsequent retest emission for that break
# ===========================================================================


def test_invalidation_prevents_subsequent_retest():
    candles = _support_break_series([
        (99.0, 99.5, 98.0, 98.5),
        (98.5, 101.0, 98.2, 100.8),  # close > 100 → invalidate
        (100.8, 101.0, 99.0, 99.2),   # would-be retest after invalidation
    ])
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert retests == []


# ===========================================================================
# 10. Gaps before or during the approach
# ===========================================================================


def test_gap_during_approach_invalidates():
    start = 1_700_000_000
    step = 60
    # Build continuous through break, then gap before retest contact.
    base = [
        (101.0, 102.0, 100.5, 101.0),
        (101.0, 101.5, 100.0, 100.5),
        (100.5, 101.0, 100.2, 100.8),
        (100.5, 100.8, 98.5, 99.0),  # break at start+180
    ]
    candles = _series(base, start=start, step=step)
    # Gap: skip one period, then contact candle.
    gap_epoch = start + 5 * step  # skip start+240
    candles.append(
        _c(gap_epoch, 99.0, 100.5, 98.5, 99.2)
    )
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert retests == []


def test_gap_before_break_does_not_create_false_retest():
    # Gap between zone confirm and break is handled by interaction detector
    # (no break without continuous prior). Ensure no retest either.
    start = 1_700_000_000
    candles = [
        _c(start, 101.0, 102.0, 100.5, 101.0),
        _c(start + 60, 101.0, 101.5, 100.0, 100.5),
        _c(start + 120, 100.5, 101.0, 100.2, 100.8),
        # gap
        _c(start + 240, 100.5, 100.8, 98.5, 99.0),
        _c(start + 300, 99.0, 100.2, 98.5, 99.2),
    ]
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert retests == []


# ===========================================================================
# 11. Forming candles are excluded
# ===========================================================================


def test_forming_candles_excluded():
    closed = _support_break_series([
        (99.0, 100.2, 98.5, 99.2),
    ])
    # Append a forming candle that would otherwise look like another retest.
    forming = _c(
        closed[-1].epoch + 60,
        99.2, 100.5, 98.8, 99.0,
        is_closed=False,
    )
    candles = closed + [forming]
    d = ZoneRetestDetector()
    # detect should only consider closed; retest is on the last closed bar
    obs = d.detect(candles)
    retests = [o for o in obs if o.type == "zone_retest"]
    assert len(retests) == 1
    assert retests[0].candle_epoch == closed[-1].epoch

    # detect_all_historical likewise ignores forming
    all_obs = d.detect_all_historical(candles)
    retests_all = [o for o in all_obs if o.type == "zone_retest"]
    assert len(retests_all) == 1


# ===========================================================================
# 12. Zone confirmation and break causality
# ===========================================================================


def test_zone_and_break_causality():
    candles = _support_break_series([
        (99.0, 100.2, 98.5, 99.2),
    ])
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert len(retests) == 1
    r = retests[0]

    # Source zone confirmed before break; break before retest.
    assert r.payload["source_zone_confirmed_at"] is not None
    assert r.payload["source_zone_confirmed_at"] < r.payload["break_event_time"]
    assert r.payload["break_event_time"] < r.event_time
    assert r.confirmed_at == r.event_time == r.candle_epoch


# ===========================================================================
# 13. Multiple source zones at the same price
# ===========================================================================


def test_multiple_source_zones_same_price_distinct_retests():
    """
    Two distinct support zones at the same level (from different swings)
    each produce their own break and retest with distinct identities.
    """
    start = 1_700_000_000
    step = 60
    # SL1 at 100 (bars 0-2), SL2 at 100 (bars 4-5), then one break candle
    # closes below 100 and one post-break retest candle. Both zones share
    # the level; both breaks can confirm on the same retest candle.
    bars = [
        (101.0, 102.0, 100.5, 101.0),  # 0
        (101.0, 101.5, 100.0, 100.5),  # 1 SL1=100
        (100.5, 101.5, 100.2, 101.0),  # 2 confirm SL1
        (101.0, 102.0, 100.8, 101.5),  # 3
        (101.5, 102.0, 100.0, 100.8),  # 4 SL2=100
        (100.8, 101.2, 100.3, 100.9),  # 5 confirm SL2
        (100.9, 101.0, 98.5, 99.0),    # 6 break both supports (close < 100)
        (99.0, 100.2, 98.5, 99.2),     # 7 retest contact + close < 100
    ]
    candles = _series(bars, start=start, step=step)

    # Preconditions: two distinct support zones at 100, two breaks.
    zones = SupportResistanceZoneDetector().detect_all_historical(candles)
    supports = [
        z for z in zones
        if z.variant == "support" and z.payload.get("source_price") == 100.0
    ]
    assert len(supports) >= 2, f"expected >=2 support zones at 100, got {len(supports)}"
    zone_ids = {z.id for z in supports}
    assert len(zone_ids) >= 2

    interactions = ZoneInteractionDetector().detect_all_historical(candles)
    breaks = [o for o in interactions if o.variant == "break"]
    breaks_at_100 = [
        b for b in breaks
        if b.reference_ids and b.reference_ids[0] in zone_ids
    ]
    assert len(breaks_at_100) >= 2, f"expected >=2 breaks of same-price zones, got {len(breaks_at_100)}"

    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert len(retests) >= 2, f"expected >=2 distinct retests, got {len(retests)}"

    retest_ids = {r.id for r in retests}
    assert len(retest_ids) == len(retests)

    source_zone_ids = {r.reference_ids[0] for r in retests}
    break_ids = {r.reference_ids[1] for r in retests}
    assert len(source_zone_ids) >= 2
    assert len(break_ids) >= 2
    for r in retests:
        assert len(r.reference_ids) == 2
        assert r.payload["level_price"] == 100.0
        assert r.direction == "bearish"


# ===========================================================================
# 14. Distinct source breaks produce distinct event identities
# ===========================================================================


def test_distinct_breaks_distinct_identities():
    """
    Two sequential breaks (support then resistance) each produce a retest
    whose identity incorporates that break; the retest IDs differ.
    """
    start = 1_700_000_000
    step = 60
    # Support path: SL at 100 → break → retest (bars 0-4)
    # Then resistance path: SH at 110 → break → retest (bars 5-9)
    bars = [
        # Support zone + break + retest
        (101.0, 102.0, 100.5, 101.0),  # 0
        (101.0, 101.5, 100.0, 100.5),  # 1 SL=100
        (100.5, 101.0, 100.2, 100.8),  # 2 confirm
        (100.5, 100.8, 98.5, 99.0),    # 3 break support
        (99.0, 100.2, 98.5, 99.2),     # 4 retest support (bearish)
        # Climb and form resistance
        (99.2, 108.5, 99.0, 108.0),    # 5
        (108.0, 110.0, 107.5, 109.0),  # 6 SH=110
        (109.0, 109.5, 108.5, 109.0),  # 7 confirm resistance
        (109.5, 111.0, 109.0, 110.5),  # 8 break resistance
        (110.5, 111.0, 109.8, 110.6),  # 9 retest resistance (bullish)
    ]
    candles = _series(bars, start=start, step=step)
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    retests = [o for o in all_obs if o.type == "zone_retest"]
    assert len(retests) >= 2, f"expected >=2 retests from distinct breaks, got {len(retests)}"

    by_dir = {r.direction: r for r in retests}
    assert "bearish" in by_dir
    assert "bullish" in by_dir

    bear = by_dir["bearish"]
    bull = by_dir["bullish"]
    assert bear.payload["break_observation_id"] != bull.payload["break_observation_id"]
    assert bear.id != bull.id
    assert bear.reference_ids[1] == bear.payload["break_observation_id"]
    assert bull.reference_ids[1] == bull.payload["break_observation_id"]
    assert bear.payload["break_observation_id"] in bear.id
    assert bull.payload["break_observation_id"] in bull.id


# ===========================================================================
# 15. Deterministic IDs and idempotent storage
# ===========================================================================


def test_deterministic_ids_and_idempotent_storage():
    candles = _support_break_series([
        (99.0, 100.2, 98.5, 99.2),
    ])
    d = ZoneRetestDetector()
    a = d.detect_all_historical(candles)
    b = d.detect_all_historical(candles)
    assert len(a) == len(b) >= 1
    assert {o.id for o in a} == {o.id for o in b}

    store = ObservationStore()
    store.accept(a)
    store.accept(b)
    # Idempotent: same IDs stored once
    stored = [o for o in store.all() if o.type == "zone_retest"]
    assert len(stored) == len(a)


# ===========================================================================
# 16. Historical / latest detection consistency
# ===========================================================================


def test_historical_latest_consistency():
    candles = _support_break_series([
        (99.0, 99.5, 98.0, 98.5),
        (98.5, 100.2, 98.2, 99.2),  # retest on last bar
    ])
    d = ZoneRetestDetector()
    all_obs = d.detect_all_historical(candles)
    latest = d.detect(candles)
    latest_epochs = {o.confirmed_at for o in latest}
    assert latest_epochs == {candles[-1].epoch}
    # Every latest observation appears in historical
    hist_ids = {o.id for o in all_obs}
    for o in latest:
        assert o.id in hist_ids


# ===========================================================================
# 17. Progressive historical reconstruction
# ===========================================================================


def test_progressive_historical_reconstruction():
    full = _support_break_series([
        (99.0, 99.5, 98.0, 98.5),
        (98.5, 100.2, 98.2, 99.2),
    ])
    d = ZoneRetestDetector()
    # Grow the series candle by candle; once retest confirms, it stays.
    seen_ids: set[str] = set()
    for n in range(1, len(full) + 1):
        prefix = full[:n]
        obs = d.detect_all_historical(prefix)
        for o in obs:
            if o.type == "zone_retest":
                seen_ids.add(o.id)
    final = d.detect_all_historical(full)
    final_ids = {o.id for o in final if o.type == "zone_retest"}
    assert final_ids == seen_ids
    assert len(final_ids) == 1


# ===========================================================================
# 18. Mixed symbol/timeframe rejection
# ===========================================================================


def test_mixed_symbol_timeframe_rejection():
    candles = _support_break_series([
        (99.0, 100.2, 98.5, 99.2),
    ])
    # Corrupt one candle's symbol
    mixed = list(candles)
    last = mixed[-1]
    mixed[-1] = Candle(
        symbol="OTHER",
        timeframe=last.timeframe,
        epoch=last.epoch,
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        is_closed=True,
    )
    d = ZoneRetestDetector()
    assert d.detect_all_historical(mixed) == []
    assert d.detect(mixed) == []


# ===========================================================================
# 19. Registry creation, configuration validation, and minimum history
# ===========================================================================


def test_registry_zone_retest():
    reg = build_default_registry()
    assert reg.is_supported("zone_retest")

    info = reg.validate_config("zone_retest", {})
    assert info.implementation == "zone_retest"
    assert info.min_history == 5  # left=1,right=1 → 1+1+1+1+1
    assert info.detector_config_id.startswith("cfg:")

    det, info_c = reg.create_detector("zone_retest")
    assert isinstance(det, ZoneRetestDetector)
    assert det.min_history == 5
    assert det.left_strength == 1
    assert det.right_strength == 1

    # Custom strengths
    info2 = reg.validate_config(
        "zone_retest",
        {"left_strength": 2, "right_strength": 2, "tie_policy": "strict"},
    )
    assert info2.min_history == 2 + 1 + 2 + 1 + 1  # 7
    det2, _ = reg.create_detector(
        "zone_retest",
        config={"left_strength": 2, "right_strength": 2},
    )
    assert det2.min_history == 7

    # Fixed config rejection
    with pytest.raises(ValueError):
        reg.validate_config("zone_retest", {"left_strength": "bad"})


def test_registry_config_id_deterministic():
    reg = build_default_registry()
    a = reg.validate_config("zone_retest", {"left_strength": 1, "right_strength": 1})
    b = reg.validate_config("zone_retest", {"left_strength": 1, "right_strength": 1})
    assert a.detector_config_id == b.detector_config_id
    c = reg.validate_config("zone_retest", {"left_strength": 2, "right_strength": 1})
    assert c.detector_config_id != a.detector_config_id


# ===========================================================================
# 20. Regression: existing Phase 3A/3B behavior unchanged
# ===========================================================================


def test_regression_phase3a_3b_unaffected():
    candles = _support_break_series([
        (99.0, 100.2, 98.5, 99.2),
    ])
    zones = SupportResistanceZoneDetector().detect_all_historical(candles)
    assert any(z.variant == "support" for z in zones)

    interactions = ZoneInteractionDetector().detect_all_historical(candles)
    breaks = [o for o in interactions if o.variant == "break"]
    assert len(breaks) == 1
    assert breaks[0].direction == "bearish"

    # Retest still works alongside
    retests = ZoneRetestDetector().detect_all_historical(candles)
    assert len([o for o in retests if o.type == "zone_retest"]) == 1


# ===========================================================================
# Payload / reference_ids contract
# ===========================================================================


def test_payload_and_reference_ids_contract():
    candles = _support_break_series([
        (99.0, 100.2, 98.5, 99.2),
    ])
    d = ZoneRetestDetector()
    retests = [o for o in d.detect_all_historical(candles) if o.type == "zone_retest"]
    assert len(retests) == 1
    r = retests[0]
    p = r.payload
    assert p["zone_type"] == "support_resistance_zone"
    assert p["zone_variant"] == "support"
    assert p["level_price"] == 100.0
    assert "source_zone_event_time" in p
    assert "source_zone_confirmed_at" in p
    assert "break_observation_id" in p
    assert "break_event_time" in p
    # Retest candle is the post-break bar: (99.0, 100.2, 98.5, 99.2)
    assert p["open"] == 99.0
    assert p["high"] == 100.2
    assert p["low"] == 98.5
    assert p["close"] == 99.2
    assert "previous_close" in p
    assert len(r.reference_ids) == 2
    assert r.reference_ids[0]  # zone id
    assert r.reference_ids[1] == p["break_observation_id"]
