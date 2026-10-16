"""Phase 2C — HH/HL/LH/LL market-structure classification tests."""

from __future__ import annotations

import pytest

from cpab.detectors.market_structure import MarketStructureDetector
from cpab.detectors.swing import SwingPivotDetector
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
    highs: list[float],
    lows: list[float] | None = None,
    start: int = 1_700_000_000,
    *,
    step: int = 60,
    timeframe: str = "1m",
) -> list[Candle]:
    lows = lows if lows is not None else [h - 1.0 for h in highs]
    assert len(highs) == len(lows)
    out = []
    for i, (h, l) in enumerate(zip(highs, lows)):
        mid = (h + l) / 2.0
        out.append(_c(start + i * step, mid, h, l, mid, timeframe=timeframe))
    return out


def test_hh_higher_high():
    candles = _series(
        [10.0, 12.0, 11.0, 10.5, 15.0, 14.0],
        lows=[9.0, 11.0, 10.0, 9.5, 13.0, 12.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    hh = [o for o in obs if o.variant == "hh"]
    assert len(hh) >= 1
    o = hh[0]
    assert o.type == "market_structure"
    assert o.payload["classification"] == "hh"
    assert o.payload["source_price"] == 15.0
    assert o.payload["reference_price"] == 12.0
    assert o.developing is False
    assert len(o.reference_ids) == 2


def test_lh_lower_high():
    # Strict 1/1 pivots: SH at 15 (idx 1), SH at 14 (idx 4) → LH
    candles = _series(
        [10.0, 15.0, 14.0, 13.0, 14.0, 13.0],
        lows=[9.0, 12.0, 11.0, 10.0, 11.0, 10.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    lh = [o for o in obs if o.variant == "lh"]
    assert len(lh) >= 1
    o = lh[0]
    assert o.payload["classification"] == "lh"
    assert o.payload["source_price"] < o.payload["reference_price"]
    assert o.payload["source_price"] == 14.0
    assert o.payload["reference_price"] == 15.0


def test_hl_higher_low():
    # Strict 1/1 pivots: SL at 8.0 (idx 1), SL at 8.5 (idx 3) → HL
    candles = _series(
        [12.0, 11.0, 12.5, 12.0, 13.0, 12.5],
        lows=[10.0, 8.0, 9.0, 8.5, 9.5, 9.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    hl = [o for o in obs if o.variant == "hl"]
    assert len(hl) >= 1
    o = hl[0]
    assert o.payload["classification"] == "hl"
    assert o.payload["source_price"] > o.payload["reference_price"]
    assert o.payload["source_price"] == 8.5
    assert o.payload["reference_price"] == 8.0


def test_ll_lower_low():
    candles = _series(
        [14.0, 13.0, 12.0, 11.0, 10.0, 9.0],
        lows=[12.0, 10.0, 11.0, 8.0, 9.0, 7.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    ll = [o for o in obs if o.variant == "ll"]
    assert len(ll) >= 1
    o = ll[0]
    assert o.payload["classification"] == "ll"
    assert o.payload["source_price"] < o.payload["reference_price"]


def test_first_high_no_classification():
    candles = _series([10.0, 12.0, 11.0])
    d = MarketStructureDetector(left_strength=1, right_strength=1)
    assert d.detect_all_historical(candles) == []
    assert d.detect(candles) == []


def test_first_low_no_classification():
    candles = _series([12.0, 10.0, 11.0], lows=[11.0, 8.0, 9.0])
    d = MarketStructureDetector(left_strength=1, right_strength=1)
    assert d.detect_all_historical(candles) == []


def test_insufficient_history_no_emit():
    candles = _series([10.0, 12.0])
    d = MarketStructureDetector(left_strength=1, right_strength=1)
    assert d.detect(candles) == []
    assert d.min_history == 5


def test_min_history_formula():
    """min_history = left + 2*right + 2 (earliest two same-type unequal swings)."""
    assert MarketStructureDetector(left_strength=1, right_strength=1).min_history == 5
    assert MarketStructureDetector(left_strength=2, right_strength=2).min_history == 8
    assert MarketStructureDetector(left_strength=2, right_strength=1).min_history == 6
    assert MarketStructureDetector(left_strength=1, right_strength=2).min_history == 7


def test_earliest_two_swing_highs_l2_r2_eight_bars():
    """
    left=2, right=2: two strict swing highs on exactly 8 bars must yield HH.

    highs: [1, 2, 5, 3, 4, 6, 3, 2]
      SH at idx 2 (price 5, confirmed at 4)
      SH at idx 5 (price 6, confirmed at 7) → HH

    Prior formula required 9 bars and would miss this classification.
    """
    highs = [1.0, 2.0, 5.0, 3.0, 4.0, 6.0, 3.0, 2.0]
    lows = [h - 0.5 for h in highs]
    candles = _series(highs, lows)
    assert len(candles) == 8

    d = MarketStructureDetector(left_strength=2, right_strength=2, tie_policy="strict")
    assert d.min_history == 8

    swing = SwingPivotDetector(left_strength=2, right_strength=2, tie_policy="strict")
    swing_highs = [o for o in swing.detect_all_historical(candles) if o.type == "swing_high"]
    assert len(swing_highs) == 2
    prices = sorted(float(o.payload["price"]) for o in swing_highs)
    assert prices == [5.0, 6.0]

    hist = d.detect_all_historical(candles)
    hh = [o for o in hist if o.variant == "hh"]
    assert len(hh) == 1
    assert hh[0].payload["source_price"] == 6.0
    assert hh[0].payload["reference_price"] == 5.0
    assert hh[0].confirmed_at == candles[7].epoch

    # Incremental: classification becomes available at the second swing's confirmation bar
    incr = d.detect(candles)
    assert len(incr) == 1
    assert incr[0].variant == "hh"
    assert incr[0].confirmed_at == candles[-1].epoch
    assert incr[0].id == hh[0].id


def test_equal_highs_no_classification_reference_advances():
    """
    Strict 1/1 swing highs:
      H1=100 → no classification (first)
      H2=105 → hh vs 100
      H3=105 → equal: no classification; reference advances to H3
      H4=103 → lh vs 105 (not vs 100)
    """
    highs = [
        98.0, 100.0, 99.0, 101.0, 105.0, 104.0, 102.0, 105.0, 104.0, 101.0, 103.0, 102.0,
    ]
    lows = [h - 2.0 for h in highs]
    candles = _series(highs, lows)
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")

    swing = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    swing_highs = sorted(
        [o for o in swing.detect_all_historical(candles) if o.type == "swing_high"],
        key=lambda o: (
            o.confirmed_at if o.confirmed_at is not None else 0,
            o.event_time if o.event_time is not None else o.candle_epoch,
            o.id,
        ),
    )
    swing_prices = [float(o.payload["price"]) for o in swing_highs]
    assert swing_prices == [100.0, 105.0, 105.0, 103.0]

    obs = d.detect_all_historical(candles)
    high_class = [
        (o.variant, o.payload["source_price"], o.payload["reference_price"])
        for o in obs
        if o.type == "market_structure" and o.variant in ("hh", "lh")
    ]
    # Equal consecutive highs must not emit a classification
    assert all(sp != rp for _, sp, rp in high_class)
    assert ("hh", 105.0, 100.0) in high_class
    # Reference advanced past the equal high: 103 is compared to 105, not 100
    assert ("lh", 103.0, 105.0) in high_class
    assert ("lh", 103.0, 100.0) not in high_class


def test_equal_lows_no_classification_reference_advances():
    """
    Strict 1/1 swing lows:
      L1=8.0 → no classification (first)
      L2=8.0 → equal: no classification; reference advances to L2
      L3=7.0 → ll vs 8.0 (the advanced equal reference)
    """
    highs = [12.0, 11.0, 13.0, 12.0, 11.0, 10.0, 11.0]
    lows = [10.0, 8.0, 9.0, 8.0, 9.0, 7.0, 8.0]
    candles = _series(highs, lows)
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")

    swing = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    swing_lows = sorted(
        [o for o in swing.detect_all_historical(candles) if o.type == "swing_low"],
        key=lambda o: (
            o.confirmed_at if o.confirmed_at is not None else 0,
            o.event_time if o.event_time is not None else o.candle_epoch,
            o.id,
        ),
    )
    swing_prices = [float(o.payload["price"]) for o in swing_lows]
    assert swing_prices == [8.0, 8.0, 7.0]

    obs = d.detect_all_historical(candles)
    low_class = [
        (o.variant, o.payload["source_price"], o.payload["reference_price"])
        for o in obs
        if o.type == "market_structure" and o.variant in ("hl", "ll")
    ]
    assert all(sp != rp for _, sp, rp in low_class)
    # Equal consecutive lows emit nothing; next unequal compares against advanced ref
    assert ("ll", 7.0, 8.0) in low_class
    # Only one low-side classification from this three-swing stream
    assert len(low_class) == 1


def test_highs_and_lows_independent():
    candles = _series(
        [10.0, 14.0, 13.0, 12.0, 16.0, 15.0],
        lows=[9.0, 11.0, 8.0, 10.0, 12.0, 11.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    for o in obs:
        assert o.variant in ("hh", "lh", "hl", "ll")


def test_classification_timing_inherits_source_swing():
    candles = _series(
        [10.0, 12.0, 11.0, 10.5, 15.0, 14.0],
        lows=[9.0, 11.0, 10.0, 9.5, 13.0, 12.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles)
    for o in obs:
        assert o.confirmed_at is not None
        assert o.event_time is not None
        assert o.candle_epoch == o.event_time
        assert o.developing is False
        assert o.detected_at > 0
        assert not o.is_eligible_at(o.confirmed_at - 1)
        assert o.is_eligible_at(o.confirmed_at)


def test_incremental_emits_only_latest_confirmation_boundary():
    candles = _series(
        [10.0, 12.0, 11.0, 10.5, 15.0, 14.0, 13.0, 16.0, 15.0],
        lows=[9.0, 11.0, 10.0, 9.5, 13.0, 12.0, 11.0, 14.0, 13.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1)
    hist = d.detect_all_historical(candles)
    incr = d.detect(candles)
    latest = candles[-1].epoch
    for o in incr:
        assert o.confirmed_at == latest
    hist_ids = {o.id for o in hist if o.confirmed_at == latest}
    incr_ids = {o.id for o in incr}
    assert incr_ids == hist_ids


def test_repeated_detect_idempotent():
    candles = _series([10.0, 12.0, 11.0, 10.5, 15.0, 14.0])
    d = MarketStructureDetector(left_strength=1, right_strength=1)
    a = d.detect(candles)
    b = d.detect(candles)
    assert [o.id for o in a] == [o.id for o in b]
    store = ObservationStore()
    assert store.accept(a) == len(a)
    assert store.accept(b) == 0


def test_observation_contract():
    candles = _series([10.0, 12.0, 11.0, 10.5, 15.0, 14.0])
    cfg = {"left_strength": 1, "right_strength": 1, "tie_policy": "strict"}
    d = MarketStructureDetector(**cfg)
    obs = d.detect_all_historical(candles)
    assert obs
    o = obs[0]
    assert o.type == "market_structure"
    assert o.variant in ("hh", "lh", "hl", "ll")
    assert o.direction is None
    assert o.developing is False
    assert o.detector_id == "market_structure"
    assert o.detector_config_id == deterministic_config_id(cfg)
    assert len(o.reference_ids) == 2
    assert "classification" in o.payload
    assert "source_price" in o.payload
    assert "reference_price" in o.payload
    assert o.detector_id != "swing_pivot"


def test_config_identity_distinct():
    d1 = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    d2 = MarketStructureDetector(left_strength=2, right_strength=1, tie_policy="strict")
    d3 = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="first")
    assert d1.detector_config_id != d2.detector_config_id
    assert d1.detector_config_id != d3.detector_config_id
    d1b = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert d1.detector_config_id == d1b.detector_config_id


def test_registry_integration():
    reg = build_default_registry()
    assert reg.is_supported("market_structure")
    assert reg.is_supported("hh")
    assert reg.is_supported("hl")
    det, info = reg.create_detector("market_structure")
    assert isinstance(det, MarketStructureDetector)
    assert info.min_history == 5
    assert det.min_history == 5
    assert det.detector_config_id == info.detector_config_id

    det2, info2 = reg.create_detector(
        "hh",
        config={"left_strength": 2, "right_strength": 2, "tie_policy": "first"},
    )
    assert det2.left_strength == 2
    assert det2.right_strength == 2
    assert det2.tie_policy == "first"
    assert info2.min_history == 8  # left + 2*right + 2
    assert info2.detector_config_id == deterministic_config_id(
        {"left_strength": 2, "right_strength": 2, "tie_policy": "first"}
    )


def test_registry_rejects_unknown_param():
    reg = build_default_registry()
    with pytest.raises(ValueError, match="unknown params"):
        reg.validate_config("market_structure", {"left_strength": 1, "foo": 1})


def test_mixed_symbol_skips():
    candles = _series([10.0, 12.0, 11.0, 10.5, 15.0, 14.0])
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
    d = MarketStructureDetector(left_strength=1, right_strength=1)
    assert d.detect_all_historical(candles) == []


def test_empty_input():
    d = MarketStructureDetector()
    assert d.detect([]) == []
    assert d.detect_all_historical([]) == []


def test_invalid_strength_raises():
    with pytest.raises(ValueError, match="left_strength"):
        MarketStructureDetector(left_strength=0)
    with pytest.raises(ValueError, match="tie_policy"):
        MarketStructureDetector(tie_policy="middle")
