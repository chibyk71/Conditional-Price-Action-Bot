"""Phase 2A — swing pivot detector tests."""

from __future__ import annotations

import pytest

from cpab.detectors.swing import SwingPivotDetector, timeframe_seconds
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
    """Build closed candles; body near mid of high/low."""
    lows = lows if lows is not None else [h - 1.0 for h in highs]
    assert len(highs) == len(lows)
    out = []
    for i, (h, l) in enumerate(zip(highs, lows)):
        mid = (h + l) / 2.0
        out.append(_c(start + i * step, mid, h, l, mid, timeframe=timeframe))
    return out


def test_invalid_zero_strength_raises():
    with pytest.raises(ValueError, match="left_strength"):
        SwingPivotDetector(left_strength=0, right_strength=1)
    with pytest.raises(ValueError, match="right_strength"):
        SwingPivotDetector(left_strength=1, right_strength=0)


def test_invalid_negative_strength_raises():
    with pytest.raises(ValueError, match="left_strength"):
        SwingPivotDetector(left_strength=-1, right_strength=1)
    with pytest.raises(ValueError, match="right_strength"):
        SwingPivotDetector(left_strength=1, right_strength=-2)


def test_invalid_float_strength_raises():
    with pytest.raises(ValueError, match="left_strength"):
        SwingPivotDetector(left_strength=1.5, right_strength=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="right_strength"):
        SwingPivotDetector(left_strength=1, right_strength=2.0)  # type: ignore[arg-type]


def test_invalid_bool_strength_raises():
    with pytest.raises(ValueError, match="left_strength"):
        SwingPivotDetector(left_strength=True, right_strength=1)  # type: ignore[arg-type]


def test_invalid_tie_policy_raises():
    with pytest.raises(ValueError, match="tie_policy"):
        SwingPivotDetector(tie_policy="middle")


def test_defaults_1_1_strict():
    d = SwingPivotDetector()
    assert d.left_strength == 1
    assert d.right_strength == 1
    assert d.tie_policy == "strict"
    assert d.min_history == 3
    assert d.detector_config_id.startswith("cfg:")


def test_valid_swing_high_1_1():
    candles = _series([10.0, 12.0, 11.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect(candles)
    highs = [o for o in obs if o.type == "swing_high"]
    assert len(highs) == 1
    assert highs[0].payload["price"] == 12.0
    assert highs[0].event_time == candles[1].epoch
    assert highs[0].confirmed_at == candles[2].epoch
    assert highs[0].variant == "pivot_lr"
    assert highs[0].direction is None
    assert highs[0].detector_id == "swing_pivot"


def test_valid_swing_low_1_1():
    candles = _series([12.0, 10.0, 11.0], lows=[11.0, 8.0, 9.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    obs = d.detect(candles)
    lows = [o for o in obs if o.type == "swing_low"]
    assert len(lows) == 1
    assert lows[0].payload["price"] == 8.0
    assert lows[0].event_time == candles[1].epoch
    assert lows[0].confirmed_at == candles[2].epoch


def test_asymmetric_strength_2_1():
    candles = _series([10.0, 11.0, 15.0, 12.0])
    d = SwingPivotDetector(left_strength=2, right_strength=1, tie_policy="strict")
    assert d.min_history == 4
    obs = d.detect(candles)
    highs = [o for o in obs if o.type == "swing_high"]
    assert len(highs) == 1
    assert highs[0].payload["price"] == 15.0
    assert highs[0].payload["left_strength"] == 2
    assert highs[0].payload["right_strength"] == 1


def test_candidate_fails_against_left():
    candles = _series([13.0, 12.0, 11.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert d.detect(candles) == []


def test_candidate_fails_against_right():
    candles = _series([10.0, 12.0, 13.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert d.detect(candles) == []


def test_missing_interval_prevents_swing():
    """Epochs 0, 60, 180 for 1m (missing 120) must not produce a swing."""
    start = 1_700_000_000
    candles = [
        _c(start, 10.0, 10.0, 9.0, 10.0),
        _c(start + 60, 11.0, 12.0, 10.0, 11.0),
        _c(start + 180, 10.5, 11.0, 9.5, 10.5),
    ]
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert d.detect(candles) == []
    assert d.detect_all_historical(candles) == []


def test_correct_spacing_still_qualifies():
    candles = _series([10.0, 12.0, 11.0], step=60, timeframe="1m")
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert len([o for o in d.detect(candles) if o.type == "swing_high"]) == 1


def test_unknown_timeframe_rejects_window():
    candles = _series([10.0, 12.0, 11.0], timeframe="weird_tf")
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert d.detect(candles) == []


def test_timeframe_seconds_known_labels():
    assert timeframe_seconds("1m") == 60
    assert timeframe_seconds("5m") == 300
    assert timeframe_seconds("1h") == 3600
    assert timeframe_seconds("nope") is None


def test_strict_rejects_equal_highs():
    candles = _series([10.0, 12.0, 12.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert [o for o in d.detect(candles) if o.type == "swing_high"] == []


def test_strict_rejects_equal_lows():
    candles = _series([12.0, 10.0, 11.0], lows=[9.0, 8.0, 8.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert [o for o in d.detect(candles) if o.type == "swing_low"] == []


def test_first_selects_earliest_equal_high_plateau():
    candles = _series([10.0, 12.0, 12.0, 11.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="first")
    all_obs = d.detect_all_historical(candles)
    highs = [o for o in all_obs if o.type == "swing_high"]
    assert len(highs) == 1
    assert highs[0].event_time == candles[1].epoch
    assert highs[0].payload["price"] == 12.0


def test_first_selects_earliest_equal_low_plateau():
    candles = _series(
        [12.0, 11.0, 11.0, 12.0],
        lows=[10.0, 8.0, 8.0, 9.0],
    )
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="first")
    all_obs = d.detect_all_historical(candles)
    lows = [o for o in all_obs if o.type == "swing_low"]
    assert len(lows) == 1
    assert lows[0].event_time == candles[1].epoch
    assert lows[0].payload["price"] == 8.0


def test_all_emits_independent_tied_candidates():
    candles = _series([10.0, 12.0, 12.0, 11.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="all")
    all_obs = d.detect_all_historical(candles)
    highs = sorted([o for o in all_obs if o.type == "swing_high"], key=lambda o: o.event_time)
    assert len(highs) == 2
    assert highs[0].event_time == candles[1].epoch
    assert highs[1].event_time == candles[2].epoch
    assert highs[0].id != highs[1].id


def test_non_contiguous_equals_not_one_plateau():
    candles = _series([12.0, 10.0, 12.0, 11.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="first")
    all_obs = d.detect_all_historical(candles)
    highs = [o for o in all_obs if o.type == "swing_high"]
    assert len(highs) == 1
    assert highs[0].event_time == candles[2].epoch


def test_first_no_premature_selection_near_edge():
    candles = _series([12.0, 12.0, 12.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="first")
    assert d.detect(candles) == []
    assert d.detect_all_historical(candles) == []


def test_overlapping_windows_respect_tie_policy():
    candles = _series([10.0, 15.0, 14.0, 13.0, 12.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    all_obs = d.detect_all_historical(candles)
    highs = [o for o in all_obs if o.type == "swing_high"]
    assert len(highs) == 1
    assert highs[0].payload["price"] == 15.0


def test_no_confirmation_before_right_side_closed():
    candles = _series([10.0, 12.0, 11.0])
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
    d = SwingPivotDetector(left_strength=1, right_strength=1)
    assert d.detect(candles) == []


def test_correct_timestamps():
    candles = _series([10.0, 12.0, 11.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1)
    obs = d.detect(candles)
    assert len(obs) == 1
    o = obs[0]
    assert o.event_time == candles[1].epoch
    assert o.confirmed_at == candles[2].epoch
    assert o.candle_epoch == candles[1].epoch
    assert o.detected_at > 0
    assert o.developing is False


def test_confirmed_at_eligibility_against_phase0_T():
    """Swing not eligible before confirming candle open epoch (Phase 0 T)."""
    candles = _series([10.0, 12.0, 11.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1)
    obs = d.detect(candles)[0]
    assert not obs.is_eligible_at(candles[1].epoch)
    assert obs.is_eligible_at(candles[2].epoch)
    assert obs.is_eligible_at(candles[2].epoch + 60)
    assert obs.effective_confirmed_at() == candles[2].epoch


def test_gap_in_window_prevents_confirmation():
    candles = _series([10.0, 12.0, 11.0])
    bad = list(candles)
    bad[1] = Candle(
        symbol=bad[1].symbol,
        timeframe=bad[1].timeframe,
        epoch=bad[0].epoch,
        open=bad[1].open,
        high=bad[1].high,
        low=bad[1].low,
        close=bad[1].close,
        is_closed=True,
    )
    d = SwingPivotDetector(left_strength=1, right_strength=1)
    assert d.detect(bad) == []


def test_insufficient_history_no_emit():
    candles = _series([10.0, 12.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1)
    assert d.detect(candles) == []


def test_forming_candle_not_used_as_confirmation():
    candles = _series([10.0, 12.0, 11.0, 10.5])
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
    d = SwingPivotDetector(left_strength=1, right_strength=1)
    obs = d.detect(candles)
    assert len([o for o in obs if o.type == "swing_high"]) == 1


def test_observation_payload_and_null_direction():
    candles = _series([10.0, 12.0, 11.0])
    cfg = {"left_strength": 1, "right_strength": 1, "tie_policy": "strict"}
    d = SwingPivotDetector(**cfg)
    obs = d.detect(candles)[0]
    assert obs.direction is None
    assert obs.variant == "pivot_lr"
    assert obs.payload["price"] == 12.0
    assert obs.payload["left_strength"] == 1
    assert obs.payload["right_strength"] == 1
    assert obs.payload["tie_policy"] == "strict"
    assert obs.reference_ids == ()
    assert obs.detector_config_id == deterministic_config_id(cfg)


def test_stable_identity_and_idempotent_store():
    candles = _series([10.0, 12.0, 11.0])
    d = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    o1 = d.detect(candles)
    o2 = d.detect(candles)
    assert len(o1) == 1 and len(o2) == 1
    assert o1[0].id == o2[0].id
    store = ObservationStore()
    assert store.accept(o1) == 1
    assert store.accept(o2) == 0


def test_direct_construction_distinct_config_identity():
    """Directly constructed detectors with different configs must not share identity."""
    candles = _series([10.0, 12.0, 11.0])
    d1 = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    d2 = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="all")
    d3 = SwingPivotDetector(left_strength=2, right_strength=1, tie_policy="strict")
    assert d1.detector_config_id != d2.detector_config_id
    assert d1.detector_config_id != d3.detector_config_id
    d1b = SwingPivotDetector(left_strength=1, right_strength=1, tie_policy="strict")
    assert d1.detector_config_id == d1b.detector_config_id
    o1 = d1.detect(candles)
    o2 = d2.detect(candles)
    assert o1 and o2
    assert o1[0].id != o2[0].id
    assert o1[0].detector_config_id is not None
    assert o2[0].detector_config_id is not None


def test_registry_integration():
    reg = build_default_registry()
    assert reg.is_supported("swing_high")
    assert reg.is_supported("swing_low")
    assert reg.is_supported("swing")
    det, info = reg.create_detector("swing_high")
    assert isinstance(det, SwingPivotDetector)
    assert info.min_history == 3
    assert det.min_history == 3
    assert info.detector_config_id.startswith("cfg:")
    assert det.detector_config_id == info.detector_config_id

    det2, info2 = reg.create_detector(
        "swing_low",
        config={"left_strength": 2, "right_strength": 3, "tie_policy": "first"},
    )
    assert det2.left_strength == 2
    assert det2.right_strength == 3
    assert det2.tie_policy == "first"
    assert info2.min_history == 6
    assert info2.detector_config_id == deterministic_config_id(
        {"left_strength": 2, "right_strength": 3, "tie_policy": "first"}
    )


def test_registry_rejects_unknown_param():
    reg = build_default_registry()
    with pytest.raises(ValueError, match="unknown params"):
        reg.validate_config("swing", {"left_strength": 1, "foo": 1})


def test_registry_rejects_float_and_bool_strength():
    reg = build_default_registry()
    with pytest.raises(ValueError, match="left_strength"):
        reg.create_detector("swing", config={"left_strength": 1.5})
    with pytest.raises(ValueError, match="left_strength"):
        reg.create_detector("swing", config={"left_strength": True})
    with pytest.raises(ValueError, match="right_strength"):
        reg.create_detector("swing", config={"right_strength": 0})


def test_engulfing_still_registered():
    reg = build_default_registry()
    det, info = reg.create_detector("engulfing")
    assert info.min_history == 2
    assert det.name == "engulfing"


def test_readiness_gate_blocks_engine_evaluation():
    """When readiness is not READY, ConditionEngine must not run detectors."""
    from unittest.mock import MagicMock
    from cpab.core.engine import ConditionEngine
    from cpab.market.readiness import ReadinessManager
    from cpab.models import ReadinessState, ConditionDef

    readiness = ReadinessManager()
    assert readiness.state == ReadinessState.INITIALIZING
    assert not readiness.can_evaluate()

    market = MagicMock()
    market.completed_candles.return_value = _series([10.0, 12.0, 11.0])

    det = SwingPivotDetector(left_strength=1, right_strength=1)
    engine = ConditionEngine(
        market=market,
        detectors=[det],
        conditions=[
            ConditionDef(
                id="c1",
                symbol="R_100",
                timeframe="1m",
                enabled=True,
                require_observations=[{"type": "swing_high", "min_count": 1}],
            )
        ],
        readiness=readiness,
    )
    signals = engine.evaluate_once()
    assert signals == []
    assert engine.last_observations == []
    market.completed_candles.assert_not_called()
