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
    candles = _series(
        [10.0, 15.0, 14.0, 13.0, 12.0, 11.0],
        lows=[9.0, 13.0, 12.0, 11.0, 10.0, 9.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    lh = [o for o in obs if o.variant == "lh"]
    assert len(lh) >= 1
    o = lh[0]
    assert o.payload["classification"] == "lh"
    assert o.payload["source_price"] < o.payload["reference_price"]


def test_hl_higher_low():
    candles = _series(
        [12.0, 11.0, 13.0, 12.0, 14.0, 13.0],
        lows=[10.0, 8.0, 9.0, 7.5, 10.0, 9.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    hl = [o for o in obs if o.variant == "hl"]
    assert len(hl) >= 1
    o = hl[0]
    assert o.payload["classification"] == "hl"
    assert o.payload["source_price"] > o.payload["reference_price"]


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


def test_equal_highs_no_classification_reference_advances():
    highs = [
        98.0, 100.0, 99.0, 101.0, 105.0, 104.0, 102.0, 105.0, 104.0, 101.0, 103.0, 102.0,
    ]
    lows = [h - 2.0 for h in highs]
    candles = _series(highs, lows)
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    classifications = [
        (o.variant, o.payload["source_price"], o.payload["reference_price"])
        for o in obs
        if o.type == "market_structure"
    ]
    for v, sp, rp in classifications:
        assert sp != rp
    lh_103 = [c for c in classifications if c[0] == "lh" and c[1] == 103.0]
    if lh_103:
        assert lh_103[0][2] == 105.0


def test_equal_lows_no_classification():
    candles = _series(
        [12.0, 11.0, 13.0, 12.0, 11.0, 10.0],
        lows=[10.0, 8.0, 9.0, 8.0, 9.0, 7.0],
    )
    d = MarketStructureDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect_all_historical(candles)
    for o in obs:
        assert o.payload["source_price"] != o.payload["reference_price"]


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
    assert info2.min_history == 9
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
