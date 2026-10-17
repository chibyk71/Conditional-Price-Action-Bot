"""Phase 2D follow-up: history completeness and review acceptance tests."""

from __future__ import annotations

import pytest

from cpab.detectors.structure_break import StructureBreakDetector
from cpab.detectors.swing import SwingPivotDetector
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.models import Candle
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


def _series_from_ohlc(
    rows: list[tuple[float, float, float, float]],
    start: int = 1_700_000_000,
    *,
    step: int = 60,
    timeframe: str = "1m",
    symbol: str = "R_100",
) -> list[Candle]:
    out = []
    for i, (o, h, l, c) in enumerate(rows):
        out.append(
            _c(start + i * step, o, h, l, c, symbol=symbol, timeframe=timeframe)
        )
    return out


def _bullish_break_fixture() -> list[Candle]:
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.0, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 10.5, 10.0, 10.1),
        (10.1, 13.5, 10.0, 13.0),
    ]
    return _series_from_ohlc(rows)


def _multi_strength_break_fixture() -> list[Candle]:
    """
    Series that produces a bullish structure break under both
    left_strength=1,right_strength=1 and left_strength=2,right_strength=1.

    SH candidate near the middle with two left bars below and one right bar
    below, then a later close above the SH.
    """
    rows = [
        (9.5, 9.8, 9.0, 9.5),    # left2 for L=2
        (9.5, 10.0, 9.2, 9.8),   # left1
        (9.8, 12.0, 9.5, 11.0),  # SH = 12.0
        (11.0, 11.2, 10.0, 10.5),  # right / confirm
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 10.5, 10.0, 10.1),
        (10.1, 13.5, 10.0, 13.0),  # close-cross above 12
    ]
    return _series_from_ohlc(rows)


def test_truncated_history_defaults_to_unknown_prior():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    bull = [o for o in obs if o.direction == "bullish"]
    assert len(bull) >= 1
    assert bull[0].payload["prior_structural_state"] == "unknown"
    assert bull[0].payload["history_complete"] is False
    assert bull[0].payload["resulting_structural_state"] == "bullish"
    assert bull[0].variant == "bos"


def test_history_complete_true_uses_neutral_prior():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles, history_complete=True)
    assert obs[0].payload["prior_structural_state"] == "neutral"
    assert obs[0].payload["history_complete"] is True


def test_history_complete_modes_have_distinct_stable_ids():
    """
    Same objective crossing under history_complete True vs False differs in
    prior_structural_state payload; stable IDs must differ so ObservationStore
    first-write-wins cannot silently drop one mode.
    """
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    complete = d.detect_all_historical(candles, history_complete=True)
    truncated = d.detect_all_historical(candles, history_complete=False)
    assert complete and truncated
    assert complete[0].id != truncated[0].id
    assert complete[0].payload["prior_structural_state"] == "neutral"
    assert truncated[0].payload["prior_structural_state"] == "unknown"
    assert complete[0].payload["history_complete"] is True
    assert truncated[0].payload["history_complete"] is False

    store = ObservationStore()
    assert store.accept(complete) == len(complete)
    # Truncated-mode records must still insert (distinct ids).
    assert store.accept(truncated) == len(truncated)
    # Re-accept is idempotent for each mode.
    assert store.accept(complete) == 0
    assert store.accept(truncated) == 0


def test_unconfirmed_swing_cannot_trigger_break():
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 13.0, 10.5, 12.5),
    ]
    candles = _series_from_ohlc(rows)
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
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    assert d.detect_all_historical(candles, history_complete=True) == []
    closed_only = [c for c in candles if c.is_closed]
    assert d.detect_all_historical(closed_only, history_complete=True) == []


def test_newly_eligible_level_already_beyond_without_crossing_no_break():
    """
    Under strict pivot geometry the confirmation window cannot close beyond the
    swing high (right-side highs must be strictly below the candidate). The
    reachable case of "level eligible while price is already beyond without a
    qualifying continuous close cross" is therefore: SH confirms while price is
    still on/at the level side, a discontinuous bar jumps beyond (evaluation
    suppressed), then continuous bars remain beyond with no prev<=level &
    cur>level pair.
    """
    start = 1_700_000_000
    step = 60
    # Continuous prefix: SH at 12.0 confirms on the third bar; closes stay <= 12.
    prefix_rows = [
        (10.0, 10.5, 9.5, 10.0),   # left
        (10.0, 12.0, 9.8, 11.0),   # swing high = 12.0 (valid OHLC)
        (11.0, 11.2, 10.0, 10.5),  # right / confirmation; close still below level
    ]
    candles = _series_from_ohlc(prefix_rows, start=start, step=step)

    swings = SwingPivotDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles
    )
    sh = [o for o in swings if o.type == "swing_high"]
    assert sh, "fixture must produce a swing high"
    assert any(float(o.payload.get("price", 0.0)) == 12.0 for o in sh)

    # Discontinuous jump beyond the level (gap suppresses break evaluation).
    last = candles[-1].epoch
    gapped = _c(last + 5 * step, 13.0, 14.0, 12.8, 13.5)
    # Resume continuous series already beyond the level: prev.close > 12 always.
    resume = [
        _c(gapped.epoch + step, 13.5, 14.0, 13.0, 13.8),
        _c(gapped.epoch + 2 * step, 13.8, 14.2, 13.5, 14.0),
    ]
    full = candles + [gapped] + resume

    # All candles satisfy OHLC: low <= open/close <= high, low <= high.
    for c in full:
        assert c.low <= c.high
        assert c.low <= c.open <= c.high
        assert c.low <= c.close <= c.high

    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(full, history_complete=True)
    # No bullish structure break against the SH=12 level (no qualifying continuous cross).
    assert not any(
        o.direction == "bullish" and float(o.payload.get("broken_level_price", 0.0)) == 12.0
        for o in obs
    )
    # Stronger: no bullish structure-break at all on this series.
    assert not any(o.direction == "bullish" for o in obs)


def test_registry_integration_structure_break():
    reg = build_default_registry()
    assert reg.is_supported("structure_break")
    assert reg.is_supported("bos")
    assert reg.is_supported("choch")
    det, info = reg.create_detector("structure_break")
    assert isinstance(det, StructureBreakDetector)
    assert info.min_history == 4
    assert det.min_history == 4
    assert det.detector_config_id == info.detector_config_id
    det2, info2 = reg.create_detector(
        "bos",
        config={"left_strength": 2, "right_strength": 2, "tie_policy": "first"},
    )
    assert det2.left_strength == 2
    assert info2.min_history == 6
    assert info2.detector_config_id == deterministic_config_id(
        {"left_strength": 2, "right_strength": 2, "tie_policy": "first"}
    )
    with pytest.raises(ValueError, match="unknown params"):
        reg.validate_config("structure_break", {"left_strength": 1, "foo": 1})
    with pytest.raises(ValueError):
        reg.validate_config("structure_break", {"left_strength": 0})


def test_timeframe_isolation():
    c1m = _bullish_break_fixture()
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.0, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 10.5, 10.0, 10.1),
        (10.1, 13.5, 10.0, 13.0),
    ]
    c5m = _series_from_ohlc(rows, timeframe="5m", step=300)
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    o1 = d.detect_all_historical(c1m, history_complete=True)
    o5 = d.detect_all_historical(c5m, history_complete=True)
    assert o1 and o5
    assert all(o.timeframe == "1m" for o in o1)
    assert all(o.timeframe == "5m" for o in o5)
    assert {o.id for o in o1}.isdisjoint({o.id for o in o5})


def test_detector_config_isolation_stable_ids():
    candles = _multi_strength_break_fixture()
    d1 = StructureBreakDetector(left_strength=1, right_strength=1, tie_policy="strict")
    d2 = StructureBreakDetector(left_strength=2, right_strength=1, tie_policy="strict")
    a = d1.detect_all_historical(candles, history_complete=True)
    b = d2.detect_all_historical(candles, history_complete=True)
    assert a, "fixture must emit under left_strength=1"
    assert b, "fixture must emit under left_strength=2"
    assert a[0].detector_config_id != b[0].detector_config_id
    assert a[0].id != b[0].id
    store = ObservationStore()
    assert store.accept(a) == len(a)
    assert store.accept(a) == 0
    assert store.accept(b) == len(b)
    assert store.accept(b) == 0


def test_multiple_crossed_levels_at_most_one_each_deterministic_order():
    rows = [
        (10.0, 10.2, 9.5, 10.0),
        (10.0, 11.0, 9.8, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 10.5, 9.9, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.5, 10.8),
        (10.8, 10.9, 10.4, 10.5),
        (10.5, 13.5, 10.4, 13.0),
        (13.0, 13.2, 12.5, 12.8),
        (12.8, 14.0, 12.5, 13.5),
    ]
    candles = _series_from_ohlc(rows)
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles, history_complete=True)
    bull = [o for o in obs if o.direction == "bullish"]
    assert len(bull) == 2
    assert [o.payload["broken_level_price"] for o in bull] == [11.0, 12.0]
    assert len({o.reference_ids[0] for o in bull}) == 2


def test_sequential_state_transitions_same_candle_order():
    rows = [
        (10.0, 10.2, 9.5, 10.0),
        (10.0, 11.0, 9.8, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 10.5, 9.9, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.5, 10.8),
        (10.8, 10.9, 10.4, 10.5),
        (10.5, 13.5, 10.4, 13.0),
    ]
    candles = _series_from_ohlc(rows)
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles, history_complete=True)
    bull = [
        o
        for o in obs
        if o.confirmed_at == candles[7].epoch and o.direction == "bullish"
    ]
    assert len(bull) == 2
    assert bull[0].payload["prior_structural_state"] == "neutral"
    assert bull[0].payload["resulting_structural_state"] == "bullish"
    assert bull[1].payload["prior_structural_state"] == "bullish"
    assert bull[1].payload["resulting_structural_state"] == "bullish"
    assert bull[0].variant == "bos"
    assert bull[1].variant == "bos"
