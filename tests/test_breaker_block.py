"""Phase 4E — Breaker Block detection acceptance tests."""

from __future__ import annotations

import pytest

from cpab.detectors.breaker_block import BreakerBlockDetector
from cpab.detectors.order_block import OrderBlockDetector
from cpab.detectors.registry import build_default_registry
from cpab.models import Candle


def _c(
    epoch: int,
    o: float,
    h: float,
    l: float,
    cl: float,
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
        close=cl,
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


def _bullish_ob_base_rows() -> list[tuple[float, float, float, float]]:
    """
    Series that produces a bullish Order Block at idx 5 (BOS).
    Candidate (bearish) at idx 3: high=11.2, low=9.8 → zone [9.8, 11.2].
    """
    return [
        (10.0, 10.5, 9.5, 10.0),   # 0
        (10.0, 12.0, 9.8, 11.0),   # 1 swing high
        (11.0, 11.2, 10.5, 10.8),  # 2
        (11.0, 11.2, 9.8, 10.0),   # 3 bearish OB candidate
        (10.0, 10.5, 9.9, 10.2),   # 4
        (10.2, 13.5, 10.0, 13.0),  # 5 bullish BOS / OB confirmed
    ]


def _bearish_ob_base_rows() -> list[tuple[float, float, float, float]]:
    """
    Series that produces a bearish Order Block at idx 5.
    Candidate (bullish) at idx 3: high=10.0, low=8.8 → zone [8.8, 10.0].
    """
    return [
        (10.0, 10.5, 9.5, 10.0),   # 0
        (10.0, 10.2, 8.0, 9.0),    # 1 swing low
        (9.0, 9.5, 8.5, 9.2),      # 2
        (9.0, 10.0, 8.8, 9.8),     # 3 bullish OB candidate
        (9.8, 10.0, 9.5, 9.6),     # 4
        (9.6, 9.7, 7.0, 7.2),      # 5 bearish BOS / OB confirmed
    ]


def _with_extra(
    base: list[tuple[float, float, float, float]],
    extras: list[tuple[float, float, float, float]],
) -> list[Candle]:
    return _series_from_ohlc(base + extras)


# ---------------------------------------------------------------------------
# Direction and boundaries
# ---------------------------------------------------------------------------


def test_bullish_ob_close_below_lower_emits_bearish_breaker():
    # Zone [9.8, 11.2]; close 9.5 < 9.8
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [
            (13.0, 13.2, 12.5, 12.8),  # 6 inside / above
            (12.8, 12.9, 9.0, 9.5),    # 7 close below L
        ],
    )
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert len(obs) == 1
    bb = obs[0]
    assert bb.type == "breaker_block"
    assert bb.variant == "bullish_to_bearish"
    assert bb.direction == "bearish"
    assert bb.payload["lower_bound"] == pytest.approx(9.8)
    assert bb.payload["upper_bound"] == pytest.approx(11.2)
    assert bb.payload["close"] == pytest.approx(9.5)


def test_bearish_ob_close_above_upper_emits_bullish_breaker():
    # Zone [8.8, 10.0]; close 10.5 > 10.0
    candles = _with_extra(
        _bearish_ob_base_rows(),
        [
            (7.2, 7.5, 7.0, 7.3),     # 6
            (7.3, 11.0, 7.2, 10.5),   # 7 close above U
        ],
    )
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert len(obs) == 1
    bb = obs[0]
    assert bb.variant == "bearish_to_bullish"
    assert bb.direction == "bullish"
    assert bb.payload["lower_bound"] == pytest.approx(8.8)
    assert bb.payload["upper_bound"] == pytest.approx(10.0)
    assert bb.payload["close"] == pytest.approx(10.5)


def test_close_exactly_at_lower_does_not_trigger():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [
            (13.0, 13.2, 9.8, 9.8),  # close == L = 9.8
        ],
    )
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert obs == []


def test_close_exactly_at_upper_does_not_trigger():
    candles = _with_extra(
        _bearish_ob_base_rows(),
        [
            (7.2, 10.0, 7.0, 10.0),  # close == U = 10.0
        ],
    )
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert obs == []


def test_wick_breach_close_inside_no_emission():
    # Wick below L but close still above L
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [
            (13.0, 13.2, 9.0, 10.0),  # low < L, close 10.0 > L
        ],
    )
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert obs == []


def test_preserves_original_ob_zone_boundaries():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [(13.0, 13.2, 9.0, 9.5)],
    )
    ob_d = OrderBlockDetector(left_strength=1, right_strength=1)
    obs_ob = ob_d.detect_all_historical(candles, history_complete=True)
    assert obs_ob
    ob = obs_ob[0]

    bb_d = BreakerBlockDetector(left_strength=1, right_strength=1)
    obs_bb = bb_d.detect_all_historical(candles, history_complete=True)
    assert obs_bb
    bb = obs_bb[0]
    assert bb.payload["lower_bound"] == ob.payload["lower_bound"]
    assert bb.payload["upper_bound"] == ob.payload["upper_bound"]
    assert bb.payload["size"] == ob.payload["upper_bound"] - ob.payload["lower_bound"]


def test_source_order_block_observation_unchanged():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [(13.0, 13.2, 9.0, 9.5)],
    )
    before = OrderBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    after = OrderBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert len(before) == len(after)
    assert [o.id for o in before] == [o.id for o in after]
    assert [o.payload for o in before] == [o.payload for o in after]


# ---------------------------------------------------------------------------
# Confirmation and provenance
# ---------------------------------------------------------------------------


def test_no_emission_before_qualifying_candle_completed():
    base = _bullish_ob_base_rows()
    # Only up to OB confirmation — no post-OB candle
    candles = _series_from_ohlc(base)
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert obs == []


def test_emission_on_qualifying_candle_confirmation():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [(13.0, 13.2, 9.0, 9.5)],
    )
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert obs
    assert obs[0].confirmed_at == candles[-1].epoch
    assert obs[0].event_time == candles[-1].epoch
    assert obs[0].candle_epoch == candles[-1].epoch
    assert obs[0].developing is False


def test_source_ob_reference_retained():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [(13.0, 13.2, 9.0, 9.5)],
    )
    obs_ob = OrderBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    obs_bb = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert obs_ob and obs_bb
    assert obs_bb[0].reference_ids == (obs_ob[0].id,)
    assert obs_bb[0].payload["source_ob_id"] == obs_ob[0].id
    assert obs_bb[0].payload["source_ob_direction"] == "bullish"


def test_invalidation_candle_ohlc_provenance():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [(13.0, 13.2, 9.0, 9.5)],
    )
    inv = candles[-1]
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert obs
    p = obs[0].payload
    assert p["invalidation_candle_epoch"] == inv.epoch
    assert p["open"] == inv.open
    assert p["high"] == inv.high
    assert p["low"] == inv.low
    assert p["close"] == inv.close


# ---------------------------------------------------------------------------
# Determinism and data integrity
# ---------------------------------------------------------------------------


def test_reprocessing_idempotent():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [(13.0, 13.2, 9.0, 9.5)],
    )
    d = BreakerBlockDetector(left_strength=1, right_strength=1)
    a = d.detect_all_historical(candles, history_complete=True)
    b = d.detect_all_historical(candles, history_complete=True)
    assert [o.id for o in a] == [o.id for o in b]
    assert [o.payload for o in a] == [o.payload for o in b]


def test_historical_matches_incremental_latest():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [(13.0, 13.2, 9.0, 9.5)],
    )
    d = BreakerBlockDetector(left_strength=1, right_strength=1)
    hist = d.detect_all_historical(candles, history_complete=True)
    incr = d.detect(candles, history_complete=True)
    latest = candles[-1].epoch
    assert all(o.confirmed_at == latest for o in incr)
    assert {o.id for o in incr} == {o.id for o in hist if o.confirmed_at == latest}


def test_stable_ids_for_identical_inputs():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [(13.0, 13.2, 9.0, 9.5)],
    )
    d1 = BreakerBlockDetector(left_strength=1, right_strength=1)
    d2 = BreakerBlockDetector(left_strength=1, right_strength=1)
    assert [o.id for o in d1.detect_all_historical(candles, history_complete=True)] == [
        o.id for o in d2.detect_all_historical(candles, history_complete=True)
    ]


def test_gap_does_not_infer_transition():
    """Discontinuous bar after OB is skipped; continuous later failure still works
    only if predecessor is continuous — a pure gap mid-path skips that bar."""
    base = _bullish_ob_base_rows()
    start = 1_700_000_000
    candles = _series_from_ohlc(base)
    # Append a gapped candle (epoch jump) that would otherwise fail the zone
    gap = _c(
        start + len(base) * 60 + 300,  # 5-minute gap instead of 1m
        13.0,
        13.2,
        9.0,
        9.5,
    )
    gapped = candles + [gap]
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        gapped, history_complete=True
    )
    # Gap → no continuous predecessor → no breaker
    assert obs == []


def test_missing_source_ob_produces_no_breaker():
    # Series with no structure suitable for OB
    rows = [
        (10.0, 10.2, 9.8, 10.0),
        (10.0, 10.3, 9.9, 10.1),
        (10.1, 10.4, 10.0, 10.2),
        (10.2, 10.5, 10.1, 10.3),
        (10.3, 10.6, 10.2, 10.4),
        (10.4, 10.7, 10.3, 10.5),
    ]
    candles = _series_from_ohlc(rows)
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert obs == []


def test_one_breaker_per_order_block_first_failure_only():
    candles = _with_extra(
        _bullish_ob_base_rows(),
        [
            (13.0, 13.2, 9.0, 9.5),   # first failure
            (9.5, 9.6, 8.0, 8.5),    # second close below L
        ],
    )
    obs = BreakerBlockDetector(left_strength=1, right_strength=1).detect_all_historical(
        candles, history_complete=True
    )
    assert len(obs) == 1
    assert obs[0].confirmed_at == candles[-2].epoch


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_breaker_block_capability():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("talib not installed")
        raise
    assert reg.is_supported("breaker_block")
    assert reg.is_supported("breaker")
    info = reg.validate_config("breaker_block", {})
    assert info.detector_id == "breaker_block"
    det, info_c = reg.create_detector("breaker_block")
    assert isinstance(det, BreakerBlockDetector)
    assert info_c.detector_id == "breaker_block"


def test_registry_rejects_unknown_param():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("talib not installed")
        raise
    with pytest.raises(ValueError):
        reg.create_detector("breaker_block", config={"retest_required": True})
    with pytest.raises(ValueError):
        reg.create_detector("breaker_block", config={"min_body_ratio": 2.0})
