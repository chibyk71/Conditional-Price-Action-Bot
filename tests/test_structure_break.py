"""Phase 2D — BOS / CHoCH structure-break acceptance tests."""

from __future__ import annotations

import pytest

from cpab.detectors.structure_break import StructureBreakDetector
from cpab.detectors.swing import SwingPivotDetector
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


def _bearish_break_fixture() -> list[Candle]:
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 10.2, 8.0, 9.0),
        (9.0, 9.5, 8.5, 9.2),
        (9.2, 9.8, 9.0, 9.5),
        (9.5, 9.6, 9.0, 9.1),
        (9.1, 9.2, 7.0, 7.5),
    ]
    return _series_from_ohlc(rows)


def test_bullish_close_crossing_confirmed_swing_high():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    bull = [o for o in obs if o.direction == "bullish"]
    assert len(bull) >= 1
    o = bull[0]
    assert o.type == "structure_break"
    assert o.variant == "bos"
    assert o.payload["broken_level_price"] == 12.0
    assert o.developing is False
    assert o.reference_ids
    assert o.payload["prior_structural_state"] == "neutral"
    assert o.payload["resulting_structural_state"] == "bullish"


def test_bearish_close_crossing_confirmed_swing_low():
    candles = _bearish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    bear = [o for o in obs if o.direction == "bearish"]
    assert len(bear) >= 1
    o = bear[0]
    assert o.type == "structure_break"
    assert o.variant == "bos"
    assert o.payload["broken_level_price"] == 8.0
    assert o.payload["resulting_structural_state"] == "bearish"


def test_wick_only_penetration_no_break():
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.0, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 10.5, 10.0, 10.1),
        (10.1, 13.5, 10.0, 11.5),
    ]
    candles = _series_from_ohlc(rows)
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    assert not any(o.direction == "bullish" for o in obs)


def test_close_exactly_at_level_no_break():
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.0, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 10.5, 10.0, 10.1),
        (10.1, 12.5, 10.0, 12.0),
    ]
    candles = _series_from_ohlc(rows)
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    assert not any(o.direction == "bullish" for o in obs)


def test_crossing_from_equality_to_strictly_beyond():
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.0, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 12.0, 10.0, 12.0),
        (12.0, 13.0, 11.5, 12.5),
    ]
    candles = _series_from_ohlc(rows)
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    bull = [o for o in obs if o.direction == "bullish"]
    assert len(bull) == 1
    assert bull[0].payload["previous_close"] == 12.0
    assert bull[0].payload["breakout_close"] == 12.5


def test_remaining_beyond_level_no_repeated_events():
    candles = _bullish_break_fixture()
    last_epoch = candles[-1].epoch
    extra = [
        _c(last_epoch + 60, 13.0, 13.5, 12.5, 13.2),
        _c(last_epoch + 120, 13.2, 14.0, 13.0, 13.5),
    ]
    candles = candles + extra
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    bull = [o for o in obs if o.direction == "bullish"]
    assert len(bull) == 1


def test_level_generates_at_most_one_event():
    candles = _bullish_break_fixture()
    last = candles[-1].epoch
    more = [
        _c(last + 60, 13.0, 13.2, 11.0, 11.5),
        _c(last + 120, 11.5, 14.0, 11.0, 13.5),
    ]
    candles = candles + more
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    bull = [o for o in obs if o.direction == "bullish"]
    assert len(bull) == 1


def test_first_bullish_is_bos_establishes_bullish():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    o = [x for x in obs if x.direction == "bullish"][0]
    assert o.variant == "bos"
    assert o.payload["prior_structural_state"] == "neutral"
    assert o.payload["resulting_structural_state"] == "bullish"


def test_first_bearish_is_bos_establishes_bearish():
    candles = _bearish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    o = [x for x in obs if x.direction == "bearish"][0]
    assert o.variant == "bos"
    assert o.payload["resulting_structural_state"] == "bearish"


def test_same_direction_break_is_bos():
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.0, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 11.0, 10.0, 10.5),
        (10.5, 14.0, 10.0, 13.0),
        (13.0, 13.5, 12.5, 13.0),
        (13.0, 15.0, 12.8, 14.0),
        (14.0, 14.5, 13.5, 14.0),
        (14.0, 14.2, 13.8, 14.0),
        (14.0, 16.0, 13.9, 15.5),
    ]
    candles = _series_from_ohlc(rows)
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    bull = [o for o in obs if o.direction == "bullish"]
    assert len(bull) >= 2
    assert bull[0].variant == "bos"
    assert bull[1].variant == "bos"
    assert bull[1].payload["prior_structural_state"] == "bullish"


def test_opposite_direction_break_is_choch():
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.0, 10.5),
        (10.5, 10.8, 10.0, 10.2),
        (10.2, 10.5, 10.0, 10.1),
        (10.1, 13.5, 10.0, 13.0),
        (13.0, 13.2, 12.5, 12.8),
        (12.8, 13.0, 11.0, 11.5),
        (11.5, 12.0, 11.2, 11.8),
        (11.8, 12.0, 11.5, 11.6),
        (11.6, 11.8, 10.0, 10.5),
    ]
    candles = _series_from_ohlc(rows)
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    assert any(o.direction == "bullish" and o.variant == "bos" for o in obs)
    choch = [o for o in obs if o.variant == "choch"]
    assert len(choch) >= 1
    assert choch[0].direction == "bearish"
    assert choch[0].payload["prior_structural_state"] == "bullish"
    assert choch[0].payload["resulting_structural_state"] == "bearish"


def test_one_candle_crosses_multiple_eligible_levels_same_direction():
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
    obs = d.detect_all_historical(candles)
    bull = [o for o in obs if o.direction == "bullish" and o.confirmed_at == candles[7].epoch]
    assert len(bull) == 2
    prices = [o.payload["broken_level_price"] for o in bull]
    assert prices[0] == 11.0
    assert prices[1] == 12.0
    assert bull[0].id != bull[1].id


def test_gap_suppresses_break_evaluation():
    candles = _bullish_break_fixture()
    rows_base = candles[:-1]
    gapped = _c(rows_base[-1].epoch + 300, 10.1, 13.5, 10.0, 13.0)
    candles = rows_base + [gapped]
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    assert not any(o.confirmed_at == gapped.epoch for o in obs)


def test_gap_retains_structural_state_and_resumes():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs1 = d.detect_all_historical(candles)
    assert any(o.variant == "bos" and o.direction == "bullish" for o in obs1)

    last = candles[-1].epoch
    gap_c = _c(last + 600, 13.0, 13.2, 12.8, 13.0)
    cont = [
        gap_c,
        _c(gap_c.epoch + 60, 13.0, 13.1, 12.5, 12.6),
        _c(gap_c.epoch + 120, 12.6, 12.8, 11.0, 11.5),
        _c(gap_c.epoch + 180, 11.5, 12.0, 11.2, 11.6),
        _c(gap_c.epoch + 240, 11.6, 11.8, 11.4, 11.5),
        _c(gap_c.epoch + 300, 11.5, 11.6, 10.0, 10.5),
    ]
    full = candles + cont
    obs = d.detect_all_historical(full)
    assert not any(o.confirmed_at == gap_c.epoch for o in obs)
    choch = [o for o in obs if o.variant == "choch" and o.direction == "bearish"]
    assert len(choch) >= 1


def test_forming_candles_never_trigger():
    candles = _bullish_break_fixture()
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
    assert d.detect(candles) == []


def test_mixed_symbols_skip():
    candles = _bullish_break_fixture()
    candles[2] = Candle(
        symbol="OTHER",
        timeframe=candles[2].timeframe,
        epoch=candles[2].epoch,
        open=candles[2].open,
        high=candles[2].high,
        low=candles[2].low,
        close=candles[2].close,
        is_closed=True,
    )
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    assert d.detect_all_historical(candles) == []


def test_historical_reconstruction_deterministic():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    a = d.detect_all_historical(candles)
    b = d.detect_all_historical(candles)
    assert [o.id for o in a] == [o.id for o in b]


def test_incremental_agrees_with_historical_on_latest():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    hist = d.detect_all_historical(candles)
    incr = d.detect(candles)
    latest = candles[-1].epoch
    hist_latest = [o for o in hist if o.confirmed_at == latest]
    assert [o.id for o in incr] == [o.id for o in hist_latest]


def test_store_idempotent():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect(candles)
    store = ObservationStore()
    assert store.accept(obs) == len(obs)
    assert store.accept(obs) == 0


def test_observation_contract_fields():
    candles = _bullish_break_fixture()
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    assert obs
    o = obs[0]
    assert o.type == "structure_break"
    assert o.variant in ("bos", "choch")
    assert o.direction in ("bullish", "bearish")
    assert o.developing is False
    assert o.detector_id == "structure_break"
    assert o.detector_config_id
    assert len(o.reference_ids) == 1
    for key in (
        "broken_level_price",
        "break_direction",
        "event_variant",
        "source_swing_id",
        "prior_structural_state",
        "resulting_structural_state",
    ):
        assert key in o.payload


def test_config_identity_distinct():
    d1 = StructureBreakDetector(left_strength=1, right_strength=1, tie_policy="strict")
    d2 = StructureBreakDetector(left_strength=2, right_strength=1, tie_policy="strict")
    assert d1.detector_config_id != d2.detector_config_id
    d1b = StructureBreakDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert d1.detector_config_id == d1b.detector_config_id


def test_min_history():
    d = StructureBreakDetector(left_strength=1, right_strength=1)
    assert d.min_history == 4
    d2 = StructureBreakDetector(left_strength=2, right_strength=2)
    assert d2.min_history == 6


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        StructureBreakDetector(left_strength=0)
    with pytest.raises(ValueError):
        StructureBreakDetector(tie_policy="middle")


def test_empty_input():
    d = StructureBreakDetector()
    assert d.detect([]) == []
    assert d.detect_all_historical([]) == []
