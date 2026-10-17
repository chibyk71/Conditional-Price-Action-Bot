"""Phase 3A — support / resistance zones from confirmed swings."""

from __future__ import annotations

import pytest

from cpab.detectors.support_resistance import SupportResistanceZoneDetector
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


def test_confirmed_swing_high_produces_resistance_zone():
    candles = _series([10.0, 12.0, 11.0])
    d = SupportResistanceZoneDetector(left_strength=1, right_strength=1)
    obs = d.detect(candles)
    zones = [o for o in obs if o.type == "support_resistance_zone"]
    assert len(zones) == 1
    z = zones[0]
    assert z.variant == "resistance"
    assert z.direction is None
    assert z.payload["source_price"] == 12.0
    assert z.payload["lower_bound"] == 12.0
    assert z.payload["upper_bound"] == 12.0


def test_confirmed_swing_low_produces_support_zone():
    candles = _series([12.0, 10.0, 11.0], lows=[11.0, 8.0, 9.0])
    d = SupportResistanceZoneDetector(left_strength=1, right_strength=1)
    obs = d.detect(candles)
    zones = [o for o in obs if o.type == "support_resistance_zone"]
    assert len(zones) == 1
    z = zones[0]
    assert z.variant == "support"
    assert z.payload["source_price"] == 8.0
    assert z.payload["lower_bound"] == 8.0
    assert z.payload["upper_bound"] == 8.0


def test_bounds_equal_exact_source_swing_price():
    candles = _series([10.0, 15.5, 11.0])
    d = SupportResistanceZoneDetector()
    z = d.detect(candles)[0]
    assert z.payload["lower_bound"] == 15.5
    assert z.payload["upper_bound"] == 15.5
    assert z.payload["lower_bound"] == z.payload["source_price"]
    assert z.payload["lower_bound"] <= z.payload["upper_bound"]


def test_zone_event_time_matches_source_swing():
    candles = _series([10.0, 12.0, 11.0])
    swing = SwingPivotDetector().detect(candles)[0]
    zone = SupportResistanceZoneDetector().detect(candles)[0]
    assert zone.event_time == swing.event_time
    assert zone.candle_epoch == swing.event_time


def test_zone_confirmed_at_matches_source_swing():
    candles = _series([10.0, 12.0, 11.0])
    swing = SwingPivotDetector().detect(candles)[0]
    zone = SupportResistanceZoneDetector().detect(candles)[0]
    assert zone.confirmed_at == swing.confirmed_at
    assert zone.confirmed_at == candles[2].epoch


def test_source_swing_id_in_reference_ids():
    candles = _series([10.0, 12.0, 11.0])
    swing = SwingPivotDetector().detect(candles)[0]
    zone = SupportResistanceZoneDetector().detect(candles)[0]
    assert swing.id in zone.reference_ids
    assert zone.reference_ids == (swing.id,)


def test_preserves_symbol_and_timeframe():
    candles = _series([10.0, 12.0, 11.0], step=300, timeframe="5m")
    zone = SupportResistanceZoneDetector().detect(candles)[0]
    assert zone.symbol == "R_100"
    assert zone.timeframe == "5m"


def test_zone_not_eligible_before_source_swing():
    candles = _series([10.0, 12.0, 11.0])
    zone = SupportResistanceZoneDetector().detect(candles)[0]
    assert not zone.is_eligible_at(candles[1].epoch)
    assert zone.is_eligible_at(candles[2].epoch)
    assert zone.effective_confirmed_at() == candles[2].epoch


def test_multiple_swings_separate_zones_including_same_price():
    candles = _series([10.0, 12.0, 12.0, 11.0, 10.0])
    d = SupportResistanceZoneDetector(left_strength=1, right_strength=1, tie_policy="all")
    all_zones = d.detect_all_historical(candles)
    resistance = [z for z in all_zones if z.variant == "resistance"]
    assert len(resistance) >= 2
    prices = [z.payload["source_price"] for z in resistance]
    assert 12.0 in prices
    ids = {z.id for z in resistance}
    assert len(ids) == len(resistance)
    same_price = [z for z in resistance if z.payload["source_price"] == 12.0]
    assert len(same_price) >= 2
    assert same_price[0].id != same_price[1].id


def test_stable_identity_and_idempotent_store():
    candles = _series([10.0, 12.0, 11.0])
    d = SupportResistanceZoneDetector(left_strength=1, right_strength=1, tie_policy="strict")
    o1 = d.detect(candles)
    o2 = d.detect(candles)
    assert len(o1) == 1 and len(o2) == 1
    assert o1[0].id == o2[0].id
    store = ObservationStore()
    assert store.accept(o1) == 1
    assert store.accept(o2) == 0


def test_insufficient_history_no_emit():
    candles = _series([10.0, 12.0])
    d = SupportResistanceZoneDetector(left_strength=1, right_strength=1)
    assert d.detect(candles) == []
    assert d.detect_all_historical(candles) == []


def test_gap_prevents_zone():
    start = 1_700_000_000
    candles = [
        _c(start, 10.0, 10.0, 9.0, 10.0),
        _c(start + 60, 11.0, 12.0, 10.0, 11.0),
        _c(start + 180, 10.5, 11.0, 9.5, 10.5),
    ]
    d = SupportResistanceZoneDetector(left_strength=1, right_strength=1)
    assert d.detect(candles) == []
    assert d.detect_all_historical(candles) == []


def test_invalid_strength_raises():
    with pytest.raises(ValueError, match="left_strength"):
        SupportResistanceZoneDetector(left_strength=0)
    with pytest.raises(ValueError, match="right_strength"):
        SupportResistanceZoneDetector(right_strength=-1)
    with pytest.raises(ValueError, match="tie_policy"):
        SupportResistanceZoneDetector(tie_policy="middle")


def test_registry_construction_and_validation():
    reg = build_default_registry()
    assert reg.is_supported("support_resistance_zone")
    assert reg.is_supported("support_zone")
    assert reg.is_supported("resistance")
    det, info = reg.create_detector("support_resistance_zone")
    assert isinstance(det, SupportResistanceZoneDetector)
    assert info.min_history == 3
    assert det.min_history == 3
    assert info.detector_config_id.startswith("cfg:")
    assert det.detector_config_id == info.detector_config_id

    det2, info2 = reg.create_detector(
        "support",
        config={"left_strength": 2, "right_strength": 1, "tie_policy": "first"},
    )
    assert det2.left_strength == 2
    assert det2.right_strength == 1
    assert det2.tie_policy == "first"
    assert info2.min_history == 4
    assert info2.detector_config_id == deterministic_config_id(
        {"left_strength": 2, "right_strength": 1, "tie_policy": "first"}
    )


def test_registry_rejects_unknown_and_invalid_config():
    reg = build_default_registry()
    with pytest.raises(ValueError, match="unknown params"):
        reg.validate_config("support_resistance_zone", {"left_strength": 1, "width": 0.5})
    with pytest.raises(ValueError, match="left_strength"):
        reg.create_detector("support_resistance_zone", config={"left_strength": 1.5})
    with pytest.raises(ValueError, match="left_strength"):
        reg.create_detector("support_resistance_zone", config={"left_strength": True})
    with pytest.raises(ValueError, match="tie_policy"):
        reg.create_detector("support_resistance_zone", config={"tie_policy": "nearest"})


def test_incremental_only_latest_confirmation():
    candles = _series([10.0, 11.0, 13.0, 12.0])
    d = SupportResistanceZoneDetector(left_strength=1, right_strength=1)
    all_z = d.detect_all_historical(candles)
    assert len(all_z) >= 1
    inc = d.detect(candles)
    assert all(z.confirmed_at == candles[-1].epoch for z in inc)
    assert len(inc) == len([z for z in all_z if z.confirmed_at == candles[-1].epoch])


def test_zero_width_no_silent_widening():
    candles = _series([10.0, 12.0, 11.0])
    z = SupportResistanceZoneDetector().detect(candles)[0]
    assert z.payload["lower_bound"] == z.payload["upper_bound"]
    assert "width" not in z.payload
    assert "tolerance" not in z.payload
    assert "atr" not in z.payload


def test_prior_phases_still_registered():
    reg = build_default_registry()
    for concept in (
        "engulfing",
        "swing_high",
        "equal_high",
        "market_structure",
        "structure_break",
        "support_resistance_zone",
    ):
        assert reg.is_supported(concept), concept
