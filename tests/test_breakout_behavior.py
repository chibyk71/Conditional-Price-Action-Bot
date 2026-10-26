"""Phase 6D/6E — Breakout Behavior and follow-through regression tests."""

from __future__ import annotations

import pytest

from cpab.detectors.breakout_behavior import BreakoutBehaviorDetector
from cpab.detectors.registry import build_default_registry
from cpab.models import Candle


START = 1_700_000_000
STEP = 60


def c(i: int, o: float, h: float, l: float, close: float, *, epoch: int | None = None) -> Candle:
    return Candle(
        symbol="BO_100",
        timeframe="1m",
        epoch=START + i * STEP if epoch is None else epoch,
        open=o,
        high=h,
        low=l,
        close=close,
        is_closed=True,
    )


def breakout_bars(closes: list[float], *, gap_at: int | None = None, invalid_at: int | None = None) -> list[Candle]:
    """A swing high at bar 1 (105), confirmed by bar 2."""
    values = [
        (100, 102, 99, 100),
        (103, 105, 102, 104),
        (101, 103, 100, 101),
    ]
    for i, close in enumerate(closes, start=3):
        previous = values[-1][3]
        values.append((previous, max(previous, close) + 1, min(previous, close) - 1, close))
    bars = [c(i, *row) for i, row in enumerate(values)]
    if gap_at is not None:
        bar = bars[gap_at]
        bars[gap_at] = Candle(
            bar.symbol, bar.timeframe, bar.epoch + STEP, bar.open, bar.high, bar.low, bar.close, True
        )
    if invalid_at is not None:
        bar = bars[invalid_at]
        bars[invalid_at] = Candle(
            bar.symbol, bar.timeframe, bar.epoch, bar.open, bar.low, bar.low, bar.close, True
        )
    return bars


def swing_high_id(events) -> str:
    for event in events:
        if event.payload.get("source_type") == "swing_high":
            return event.payload["source_id"]
    raise AssertionError("expected a swing-high sourced event")


def events_for_source(events, source_id: str):
    return [event for event in events if event.payload.get("source_id") == source_id]


def test_breakout_confirmation_and_failure_use_completed_closes():
    detector = BreakoutBehaviorDetector(failure_policy="close_back_across")
    events = detector.detect_all_historical(breakout_bars([106, 104]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)

    assert [event.variant for event in source_events] == [
        "breakout_confirmed",
        "breakout_failed",
    ]
    assert source_events[0].direction == "bullish"
    assert source_events[0].payload["boundary"] == 105.0
    assert source_events[0].reference_ids == (source_id,)
    assert source_events[0].lifecycle_state == "active"
    assert source_events[1].lifecycle_state == "failed"
    assert source_events[0].confirmed_at == START + 3 * STEP
    assert source_events[1].confirmed_at == START + 4 * STEP


def test_bearish_breakout_of_confirmed_swing_low():
    bars = [
        c(0, 100, 101, 98, 100),
        c(1, 96, 99, 95, 96),
        c(2, 99, 101, 97, 99),
        c(3, 96, 98, 93, 94),
        c(4, 94, 97, 92, 96),
    ]
    detector = BreakoutBehaviorDetector(failure_policy="close_back_across")
    events = detector.detect_all_historical(bars)
    low_events = [
        event for event in events
        if event.payload.get("source_type") == "swing_low"
        and event.payload.get("boundary") == 95.0
    ]
    assert [event.variant for event in low_events] == [
        "breakout_confirmed",
        "breakout_failed",
    ]
    assert all(event.direction == "bearish" for event in low_events)


def test_source_confirmation_candle_cannot_confirm_its_own_breakout():
    detector = BreakoutBehaviorDetector()
    # Bar 2 confirms the swing; only bar 3 or later can trigger a breakout.
    events = detector.detect_all_historical(breakout_bars([101, 106]))
    source_id = swing_high_id(events)
    confirmation = events_for_source(events, source_id)[0]
    assert confirmation.confirmed_at == START + 4 * STEP


def test_equality_does_not_confirm_or_fail():
    detector = BreakoutBehaviorDetector(failure_policy="close_back_across")
    events = detector.detect_all_historical(breakout_bars([105, 105, 104]))
    assert not any(
        event.variant in ("breakout_confirmed", "breakout_failed")
        and event.payload.get("source_type") == "swing_high"
        and event.payload.get("boundary") == 105.0
        for event in events
    )


def test_confirmed_failure_requires_consecutive_close_back_and_resets_on_equality():
    detector = BreakoutBehaviorDetector(failure_policy="confirmed_close_back_across")
    events = detector.detect_all_historical(breakout_bars([106, 104, 105, 104, 103]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)

    assert [event.variant for event in source_events] == [
        "breakout_confirmed",
        "breakout_failed",
    ]
    assert source_events[-1].confirmed_at == START + 7 * STEP


@pytest.mark.parametrize("policy", ["close_back_across", "confirmed_close_back_across"])
def test_gap_silently_abandons_active_lifecycle(policy):
    detector = BreakoutBehaviorDetector(failure_policy=policy)
    bars = breakout_bars([106, 104], gap_at=4)
    events = detector.detect_all_historical(bars)
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    assert [event.variant for event in source_events] == ["breakout_confirmed"]


@pytest.mark.parametrize("policy", ["close_back_across", "confirmed_close_back_across"])
def test_invalid_ohlc_silently_abandons_active_lifecycle(policy):
    detector = BreakoutBehaviorDetector(failure_policy=policy)
    bars = breakout_bars([106, 104], invalid_at=4)
    events = detector.detect_all_historical(bars)
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    assert [event.variant for event in source_events] == ["breakout_confirmed"]


def test_confirmed_failure_resets_when_breakout_side_is_reclaimed():
    detector = BreakoutBehaviorDetector(failure_policy="confirmed_close_back_across")
    events = detector.detect_all_historical(breakout_bars([106, 104, 106]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    assert [event.variant for event in source_events] == ["breakout_confirmed"]


def test_forming_candles_are_ignored():
    detector = BreakoutBehaviorDetector()
    bars = breakout_bars([106])
    last = bars[-1]
    bars[-1] = Candle(
        last.symbol, last.timeframe, last.epoch,
        last.open, last.high, last.low, last.close, False
    )
    assert detector.detect_all_historical(bars) == []


def test_source_identity_keeps_equal_price_swing_and_zone_distinct():
    detector = BreakoutBehaviorDetector()
    events = detector.detect_all_historical(breakout_bars([106, 104]))
    confirmed = [event for event in events if event.variant == "breakout_confirmed" and event.payload["boundary"] == 105.0]
    assert {event.payload["source_family"] for event in confirmed} == {
        "swing_structure",
        "support_resistance_zone",
    }
    assert len({event.payload["source_id"] for event in confirmed}) == 2
    assert len({event.id for event in confirmed}) == 2


def test_failure_policy_changes_config_identity_and_registry_validation():
    first = BreakoutBehaviorDetector(failure_policy="close_back_across")
    second = BreakoutBehaviorDetector(failure_policy="confirmed_close_back_across")
    assert first.detector_config_id != second.detector_config_id

    pytest.importorskip("talib")
    registry = build_default_registry()
    info = registry.validate_config(
        "breakout_behavior", {"failure_policy": "confirmed_close_back_across"}
    )
    assert info.config["failure_policy"] == "confirmed_close_back_across"
    with pytest.raises(ValueError):
        registry.validate_config("breakout_behavior", {"failure_policy": "sticky"})
    detector, info = registry.create_detector(
        "breakout_behavior", config={"failure_policy": "confirmed_close_back_across"}
    )
    assert detector.detector_config_id == info.detector_config_id


def test_incremental_and_historical_results_are_deterministic():
    bars = breakout_bars([106, 104, 106, 104])
    detector = BreakoutBehaviorDetector()
    historical = detector.detect_all_historical(bars)
    repeated = detector.detect_all_historical(bars)
    assert [(event.id, event.variant) for event in historical] == [
        (event.id, event.variant) for event in repeated
    ]
    latest = detector.detect(bars)
    assert all(event.confirmed_at == bars[-1].epoch for event in latest)



# ---------------------------------------------------------------------------
# Phase 6E — breakout follow-through (opt-in)
# ---------------------------------------------------------------------------


def test_bullish_follow_through_strict_beyond_confirmation_high():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    # Confirmation at close 106; helper sets high = max(prev, close)+1 = 107.
    # Next close 107 equals conf high → no FT; close 108 → FT.
    events = detector.detect_all_historical(breakout_bars([106, 107, 108]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    variants = [e.variant for e in source_events]
    assert "breakout_confirmed" in variants
    assert "breakout_follow_through" in variants
    ft = next(e for e in source_events if e.variant == "breakout_follow_through")
    conf = next(e for e in source_events if e.variant == "breakout_confirmed")
    assert ft.direction == "bullish"
    assert ft.payload["originating_breakout_id"] == conf.id
    assert conf.id in ft.reference_ids
    assert ft.payload["confirmation_high"] == 107.0


def test_bearish_follow_through_strict_beyond_confirmation_low():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    bars = [
        c(0, 100, 102, 99, 100),
        c(1, 97, 98, 95, 96),   # swing low 95
        c(2, 97, 99, 96, 98),   # confirms swing low
        c(3, 96, 97, 94, 94),   # breakout close < 95
        c(4, 93, 94, 92, 92.5), # equal-ish to conf low? conf low was 94
        c(5, 92, 93, 90, 91),   # close 91 < conf low 94 → FT
    ]
    events = detector.detect_all_historical(bars)
    low_events = [
        e for e in events
        if e.payload.get("source_type") == "swing_low"
        and e.payload.get("boundary") == 95.0
    ]
    variants = [e.variant for e in low_events]
    assert "breakout_confirmed" in variants
    assert "breakout_follow_through" in variants
    assert all(e.direction == "bearish" for e in low_events if e.variant != "breakout_failed")


def test_equality_with_confirmation_extreme_does_not_qualify():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    # conf close 106 → high 107; subsequent close 107 == high → no FT
    events = detector.detect_all_historical(breakout_bars([106, 107]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    assert [e.variant for e in source_events] == ["breakout_confirmed"]


def test_no_follow_through_when_disabled_or_without_confirmation():
    disabled = BreakoutBehaviorDetector(enable_follow_through=False)
    events = disabled.detect_all_historical(breakout_bars([106, 110]))
    assert not any(e.variant == "breakout_follow_through" for e in events)

    # No close beyond boundary → no confirmation → no FT
    enabled = BreakoutBehaviorDetector(enable_follow_through=True)
    events = enabled.detect_all_historical(breakout_bars([101, 102, 103]))
    assert not any(
        e.variant in ("breakout_confirmed", "breakout_follow_through")
        and e.payload.get("source_type") == "swing_high"
        and e.payload.get("boundary") == 105.0
        for e in events
    )


def test_no_follow_through_on_confirmation_candle_itself():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    # Single breakout bar only — confirmation candle cannot also be FT.
    events = detector.detect_all_historical(breakout_bars([106]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    assert [e.variant for e in source_events] == ["breakout_confirmed"]


def test_follow_through_after_multiple_subsequent_candles():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    # conf at 106 (high 107); 107 equal (no), 106.5 still below, then 109 FT
    events = detector.detect_all_historical(breakout_bars([106, 107, 106.5, 109]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    variants = [e.variant for e in source_events]
    assert variants.count("breakout_follow_through") == 1
    conf = next(e for e in source_events if e.variant == "breakout_confirmed")
    ft = next(e for e in source_events if e.variant == "breakout_follow_through")
    assert ft.confirmed_at > conf.confirmed_at


def test_failure_before_follow_through_prevents_later_ft():
    detector = BreakoutBehaviorDetector(
        failure_policy="close_back_across",
        enable_follow_through=True,
    )
    # confirm 106, fail 104, later 110 would have been FT but lifecycle ended
    events = detector.detect_all_historical(breakout_bars([106, 104, 110]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    assert [e.variant for e in source_events] == [
        "breakout_confirmed",
        "breakout_failed",
    ]


def test_failure_after_follow_through_still_possible():
    detector = BreakoutBehaviorDetector(
        failure_policy="close_back_across",
        enable_follow_through=True,
    )
    # confirm 106 (high 107), FT at 109, then fail at 104
    events = detector.detect_all_historical(breakout_bars([106, 109, 104]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    assert [e.variant for e in source_events] == [
        "breakout_confirmed",
        "breakout_follow_through",
        "breakout_failed",
    ]


def test_exactly_one_follow_through_per_originating_breakout():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    events = detector.detect_all_historical(breakout_bars([106, 109, 111, 113]))
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    variants = [e.variant for e in source_events]
    assert variants.count("breakout_follow_through") == 1


def test_follow_through_independent_sources_same_price():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    events = detector.detect_all_historical(breakout_bars([106, 109]))
    confirmed = [
        e for e in events
        if e.variant == "breakout_confirmed" and e.payload["boundary"] == 105.0
    ]
    fts = [
        e for e in events
        if e.variant == "breakout_follow_through" and e.payload["boundary"] == 105.0
    ]
    assert len(confirmed) == 2
    assert len(fts) == 2
    assert len({e.payload["source_id"] for e in fts}) == 2
    assert len({e.id for e in fts}) == 2
    for ft in fts:
        assert ft.payload["originating_breakout_id"] in {c.id for c in confirmed}


def test_gap_abandons_follow_through_tracking():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    bars = breakout_bars([106, 109], gap_at=4)
    events = detector.detect_all_historical(bars)
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    # gap on the would-be FT bar abandons tracking; only confirmation remains
    assert [e.variant for e in source_events] == ["breakout_confirmed"]


def test_invalid_ohlc_abandons_follow_through_tracking():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    bars = breakout_bars([106, 109], invalid_at=4)
    events = detector.detect_all_historical(bars)
    source_id = swing_high_id(events)
    source_events = events_for_source(events, source_id)
    assert [e.variant for e in source_events] == ["breakout_confirmed"]


def test_forming_candle_excluded_from_follow_through():
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    bars = breakout_bars([106, 109])
    last = bars[-1]
    bars[-1] = Candle(
        last.symbol, last.timeframe, last.epoch,
        last.open, last.high, last.low, last.close, False,
    )
    events = detector.detect_all_historical(bars)
    # forming last bar ignored → only confirmation from completed prefix
    assert not any(e.variant == "breakout_follow_through" for e in events)


def test_enable_follow_through_changes_config_identity_when_true():
    off = BreakoutBehaviorDetector(enable_follow_through=False)
    on = BreakoutBehaviorDetector(enable_follow_through=True)
    assert off.detector_config_id != on.detector_config_id
    # Default / disabled preserves legacy Phase 6D identity shape
    legacy = BreakoutBehaviorDetector()
    assert "enable_follow_through" not in legacy.config_dict()
    assert "enable_follow_through" not in off.config_dict()
    assert on.config_dict()["enable_follow_through"] is True

    pytest.importorskip("talib")
    registry = build_default_registry()
    info_off = registry.validate_config("breakout_behavior", {})
    assert "enable_follow_through" not in info_off.config or info_off.config.get(
        "enable_follow_through"
    ) is not True
    info_on = registry.validate_config(
        "breakout_behavior", {"enable_follow_through": True}
    )
    assert info_on.config.get("enable_follow_through") is True
    assert info_off.detector_config_id != info_on.detector_config_id

    with pytest.raises(ValueError):
        registry.validate_config(
            "breakout_behavior", {"enable_follow_through": "yes"}
        )

    detector, info = registry.create_detector(
        "breakout_behavior", config={"enable_follow_through": True}
    )
    assert detector.detector_config_id == info.detector_config_id
    assert detector.config_dict().get("enable_follow_through") is True


def test_follow_through_ids_deterministic_and_parity():
    bars = breakout_bars([106, 109, 104])
    detector = BreakoutBehaviorDetector(enable_follow_through=True)
    historical = detector.detect_all_historical(bars)
    repeated = detector.detect_all_historical(bars)
    assert [(e.id, e.variant) for e in historical] == [
        (e.id, e.variant) for e in repeated
    ]
    assert any(e.variant == "breakout_follow_through" for e in historical)
    latest = detector.detect(bars)
    assert all(e.confirmed_at == bars[-1].epoch for e in latest)


def test_default_disabled_preserves_phase6d_event_behavior():
    """When follow-through is off, Phase 6D events and IDs match prior semantics."""
    bars = breakout_bars([106, 104])
    off = BreakoutBehaviorDetector(enable_follow_through=False)
    default = BreakoutBehaviorDetector()
    events_off = off.detect_all_historical(bars)
    events_default = default.detect_all_historical(bars)
    assert [(e.id, e.variant) for e in events_off] == [
        (e.id, e.variant) for e in events_default
    ]
    assert not any(e.variant == "breakout_follow_through" for e in events_off)
    assert off.detector_config_id == default.detector_config_id
