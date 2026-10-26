"""Phase 6D — Breakout Behavior regression tests."""

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
