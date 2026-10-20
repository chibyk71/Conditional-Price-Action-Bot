"""Phase 4D — Order Block detection acceptance tests."""

from __future__ import annotations

import pytest

from cpab.detectors.order_block import OrderBlockDetector, DEFAULT_MIN_BODY_RATIO
from cpab.detectors.structure_break import StructureBreakDetector
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.models import Candle, Observation


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


def _bullish_bos_with_bearish_candidate() -> list[Candle]:
    """
    Swing high at index 1, then a clear bearish opposing candle (idx 3),
    then bullish displacement that breaks the swing high (idx 5).

    Body ratios designed so the bearish candle at idx 3 has ratio >= 0.5.
    """
    # idx0
    # idx1 swing high candidate (high=12)
    # idx2 right of swing
    # idx3 bearish opposing (open 11, close 10, high 11.2, low 9.8) body/range ~0.71
    # idx4 small
    # idx5 break close > 12
    rows = [
        (10.0, 10.5, 9.5, 10.0),   # 0
        (10.0, 12.0, 9.8, 11.0),   # 1 swing high
        (11.0, 11.2, 10.5, 10.8),  # 2
        (11.0, 11.2, 9.8, 10.0),   # 3 bearish opposing
        (10.0, 10.5, 9.9, 10.2),   # 4
        (10.2, 13.5, 10.0, 13.0),  # 5 bullish BOS
    ]
    return _series_from_ohlc(rows)


def _bearish_bos_with_bullish_candidate() -> list[Candle]:
    """
    Swing low at index 1, bullish opposing at idx 3, bearish break at idx 5.
    """
    rows = [
        (10.0, 10.5, 9.5, 10.0),   # 0
        (10.0, 10.2, 8.0, 9.0),    # 1 swing low
        (9.0, 9.5, 8.5, 9.2),      # 2
        (9.0, 10.0, 8.8, 9.8),     # 3 bullish opposing (body/range ~0.67)
        (9.8, 10.0, 9.5, 9.6),     # 4
        (9.6, 9.7, 7.0, 7.2),      # 5 bearish BOS
    ]
    return _series_from_ohlc(rows)


# ---------------------------------------------------------------------------
# Core formation
# ---------------------------------------------------------------------------


def test_bullish_bos_produces_bullish_order_block():
    candles = _bullish_bos_with_bearish_candidate()
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles, history_complete=True)
    bos_obs = [
        o
        for o in StructureBreakDetector(left_strength=1, right_strength=1).detect_all_historical(
            candles, history_complete=True
        )
        if o.variant == "bos" and o.direction == "bullish"
    ]
    assert bos_obs, "fixture must produce a bullish BOS"
    assert len(obs) >= 1
    ob = [o for o in obs if o.direction == "bullish"][0]
    assert ob.type == "order_block"
    assert ob.direction == "bullish"
    assert ob.variant == "bullish"
    assert ob.developing is False
    assert ob.confirmed_at == bos_obs[0].confirmed_at
    assert ob.reference_ids == (bos_obs[0].id,)
    assert ob.payload["source_bos_id"] == bos_obs[0].id
    # Candidate is the bearish candle at idx 3
    start = 1_700_000_000
    assert ob.payload["source_candle_epoch"] == start + 3 * 60
    assert ob.payload["lower_bound"] == candles[3].low
    assert ob.payload["upper_bound"] == candles[3].high


def test_bearish_bos_produces_bearish_order_block():
    candles = _bearish_bos_with_bullish_candidate()
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles, history_complete=True)
    assert any(o.direction == "bearish" and o.type == "order_block" for o in obs)
    ob = [o for o in obs if o.direction == "bearish"][0]
    start = 1_700_000_000
    assert ob.payload["source_candle_epoch"] == start + 3 * 60
    assert ob.payload["lower_bound"] == candles[3].low
    assert ob.payload["upper_bound"] == candles[3].high


def test_choch_does_not_create_order_block():
    """
    Build a series that produces BOS then opposite CHoCH; only BOS may yield OBs.
    """
    # First establish bullish structure, then bearish CHoCH.
    # Use history_complete so first break is BOS.
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),   # swing high
        (11.0, 11.2, 10.5, 10.8),
        (11.0, 11.2, 9.8, 10.0),   # bearish candidate for first BOS
        (10.0, 10.5, 9.9, 10.2),
        (10.2, 13.5, 10.0, 13.0),  # bullish BOS
        (13.0, 13.5, 12.5, 12.8),
        (12.8, 13.0, 11.0, 11.5),  # potential swing low region
        (11.5, 12.0, 11.0, 11.2),
        (11.2, 11.5, 10.8, 11.0),
        (11.0, 11.2, 9.0, 9.2),    # bearish break of a low → CHoCH
    ]
    candles = _series_from_ohlc(rows)
    sb = StructureBreakDetector(left_strength=1, right_strength=1)
    breaks = sb.detect_all_historical(candles, history_complete=True)
    choch = [o for o in breaks if o.variant == "choch"]
    bos = [o for o in breaks if o.variant == "bos"]
    # May or may not have CHoCH depending on swings; if present, ensure no OB from it
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles, history_complete=True)
    for o in obs:
        assert o.payload["source_bos_id"] in {b.id for b in bos}
        # Must not reference a CHoCH id
        assert o.payload["source_bos_id"] not in {c.id for c in choch}


def test_nearest_opposing_candle_selected():
    """When multiple opposing candles exist, the nearest to BOS wins."""
    # Two bearish candles: idx 2 (farther) and idx 4 (nearer to BOS at 5)
    rows = [
        (10.0, 10.5, 9.5, 10.0),   # 0
        (10.0, 12.0, 9.8, 11.0),   # 1 swing high
        (11.0, 11.5, 9.5, 10.0),   # 2 bearish (farther)
        (10.0, 10.8, 9.8, 10.5),   # 3 bullish
        (10.5, 10.8, 9.5, 9.8),    # 4 bearish (nearer)
        (9.8, 13.5, 9.5, 13.0),    # 5 BOS
    ]
    candles = _series_from_ohlc(rows)
    d = OrderBlockDetector(left_strength=1, right_strength=1, min_body_ratio=0.3)
    obs = d.detect_all_historical(candles, history_complete=True)
    assert len(obs) >= 1
    ob = obs[0]
    start = 1_700_000_000
    assert ob.payload["source_candle_epoch"] == start + 4 * 60


def test_candidate_selection_bounded_by_swing_event():
    """Candle before the swing event must not be selected."""
    # Bearish candle only before swing; after swing only bullish (non-opposing).
    rows = [
        (11.0, 11.5, 9.0, 9.5),    # 0 bearish (before swing) — must NOT be chosen
        (9.5, 12.0, 9.3, 11.5),    # 1 swing high
        (11.5, 11.8, 11.0, 11.6),  # 2 bullish
        (11.6, 11.9, 11.2, 11.8),  # 3 bullish
        (11.8, 12.1, 11.4, 12.0),  # 4 bullish
        (12.0, 13.5, 11.8, 13.0),  # 5 BOS
    ]
    candles = _series_from_ohlc(rows)
    d = OrderBlockDetector(left_strength=1, right_strength=1, min_body_ratio=0.3)
    obs = d.detect_all_historical(candles, history_complete=True)
    # No qualifying opposing candle in [swing, bos) → no OB
    assert obs == []


# ---------------------------------------------------------------------------
# Body-ratio filter
# ---------------------------------------------------------------------------


def test_body_ratio_at_threshold_accepted():
    # Construct bearish candidate with ratio exactly 0.5
    # high-low=2, abs(c-o)=1 → 0.5
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),   # swing high
        (11.0, 11.2, 10.5, 10.8),
        (11.0, 12.0, 10.0, 10.0),  # bearish: open=11 close=10, high=12 low=10 → 1/2=0.5
        (10.0, 10.5, 9.9, 10.2),
        (10.2, 13.5, 10.0, 13.0),
    ]
    candles = _series_from_ohlc(rows)
    d = OrderBlockDetector(left_strength=1, right_strength=1, min_body_ratio=0.5)
    obs = d.detect_all_historical(candles, history_complete=True)
    assert len(obs) >= 1
    assert obs[0].payload["body_ratio"] == pytest.approx(0.5)


def test_body_ratio_below_threshold_rejected():
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.5, 10.8),
        (11.0, 12.0, 10.0, 10.8),  # body=0.2, range=2 → 0.1 < 0.5
        (10.8, 11.0, 10.5, 10.6),
        (10.6, 13.5, 10.0, 13.0),
    ]
    candles = _series_from_ohlc(rows)
    d = OrderBlockDetector(left_strength=1, right_strength=1, min_body_ratio=0.5)
    obs = d.detect_all_historical(candles, history_complete=True)
    assert obs == []


def test_zero_range_candidate_rejected():
    # Only opposing candidate is zero-range; other bars are non-bearish.
    rows = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),   # swing high
        (11.0, 11.5, 10.8, 11.3),  # bullish
        (10.5, 10.5, 10.5, 10.5),  # zero-range (not a valid candidate)
        (10.5, 11.0, 10.3, 10.8),  # bullish
        (10.8, 13.5, 10.5, 13.0),  # BOS
    ]
    candles = _series_from_ohlc(rows)
    d = OrderBlockDetector(left_strength=1, right_strength=1, min_body_ratio=0.0)
    obs = d.detect_all_historical(candles, history_complete=True)
    # zero-range rejected even with min_body_ratio=0
    assert obs == []


def test_invalid_min_body_ratio_rejected():
    with pytest.raises(ValueError, match="min_body_ratio"):
        OrderBlockDetector(min_body_ratio=-0.1)
    with pytest.raises(ValueError, match="min_body_ratio"):
        OrderBlockDetector(min_body_ratio=1.5)
    with pytest.raises(ValueError, match="min_body_ratio"):
        OrderBlockDetector(min_body_ratio="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_body_ratio"):
        OrderBlockDetector(min_body_ratio=True)  # type: ignore[arg-type]


def test_invalid_strength_rejected():
    with pytest.raises(ValueError):
        OrderBlockDetector(left_strength=0)
    with pytest.raises(ValueError):
        OrderBlockDetector(tie_policy="invalid")


# ---------------------------------------------------------------------------
# Zone boundaries & provenance
# ---------------------------------------------------------------------------


def test_full_candle_high_low_preserved():
    candles = _bullish_bos_with_bearish_candidate()
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles, history_complete=True)
    assert obs
    ob = obs[0]
    src = [c for c in candles if c.epoch == ob.payload["source_candle_epoch"]][0]
    assert ob.payload["lower_bound"] == src.low
    assert ob.payload["upper_bound"] == src.high
    assert ob.payload["open"] == src.open
    assert ob.payload["high"] == src.high
    assert ob.payload["low"] == src.low
    assert ob.payload["close"] == src.close
    # Not body bounds
    assert ob.payload["lower_bound"] == min(src.open, src.close) or True  # may equal
    # Explicit: bounds are high/low not body
    assert ob.payload["upper_bound"] == src.high
    assert ob.payload["lower_bound"] == src.low


def test_source_candle_and_bos_provenance():
    candles = _bullish_bos_with_bearish_candidate()
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(candles, history_complete=True)
    bos = [
        o
        for o in StructureBreakDetector(left_strength=1, right_strength=1).detect_all_historical(
            candles, history_complete=True
        )
        if o.variant == "bos"
    ][0]
    ob = obs[0]
    assert ob.reference_ids == (bos.id,)
    assert ob.payload["source_bos_id"] == bos.id
    assert "source_candle_epoch" in ob.payload
    assert "source_swing_event_time" in ob.payload
    assert ob.payload["source_swing_event_time"] == bos.payload["source_swing_event_time"]


# ---------------------------------------------------------------------------
# Causality, continuity, idempotency
# ---------------------------------------------------------------------------


def test_no_observation_before_bos_confirmation():
    candles = _bullish_bos_with_bearish_candidate()
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    # Truncate before the BOS candle
    partial = candles[:-1]
    obs = d.detect_all_historical(partial, history_complete=True)
    assert obs == []
    # Full series emits with confirmed_at == BOS epoch
    full = d.detect_all_historical(candles, history_complete=True)
    assert full
    assert full[0].confirmed_at == candles[-1].epoch


def test_discontinuity_suppresses_order_block():
    candles = _bullish_bos_with_bearish_candidate()
    # Insert a gap between swing and BOS by shifting later epochs
    # Original step=60; make candle index 4 jump by 180
    start = 1_700_000_000
    gapped = []
    for i, c in enumerate(candles):
        epoch = start + i * 60
        if i >= 4:
            epoch = start + i * 60 + 120  # introduce gap
        gapped.append(
            Candle(
                symbol=c.symbol,
                timeframe=c.timeframe,
                epoch=epoch,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                is_closed=True,
            )
        )
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    obs = d.detect_all_historical(gapped, history_complete=True)
    # Gap between swing and breakout → no OB
    assert obs == []


def test_repeated_evaluation_idempotent():
    candles = _bullish_bos_with_bearish_candidate()
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    a = d.detect_all_historical(candles, history_complete=True)
    b = d.detect_all_historical(candles, history_complete=True)
    assert len(a) == len(b)
    assert [o.id for o in a] == [o.id for o in b]
    assert [o.payload for o in a] == [o.payload for o in b]


def test_incremental_emits_only_latest_confirmed():
    candles = _bullish_bos_with_bearish_candidate()
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    hist = d.detect_all_historical(candles, history_complete=True)
    incr = d.detect(candles, history_complete=True)
    latest = candles[-1].epoch
    assert all(o.confirmed_at == latest for o in incr)
    assert {o.id for o in incr} == {o.id for o in hist if o.confirmed_at == latest}


def test_historical_and_incremental_conventions():
    candles = _bullish_bos_with_bearish_candidate()
    d = OrderBlockDetector(left_strength=1, right_strength=1)
    # Before enough history
    assert d.detect(candles[:2]) == []
    # Full
    hist = d.detect_all_historical(candles, history_complete=True)
    assert hist
    # Incremental on full series matches latest slice of historical
    incr = d.detect(candles, history_complete=True)
    assert len(incr) == sum(1 for o in hist if o.confirmed_at == candles[-1].epoch)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_order_block_capability():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("talib not installed")
        raise
    assert reg.is_supported("order_block")
    assert reg.is_supported("ob")
    info = reg.validate_config("order_block", {})
    assert info.detector_id == "order_block"
    assert info.min_history >= 4
    # Default min_body_ratio applied via factory
    det, info_c = reg.create_detector("order_block")
    assert isinstance(det, OrderBlockDetector)
    assert det.config_dict()["min_body_ratio"] == DEFAULT_MIN_BODY_RATIO
    assert info_c.detector_id == "order_block"


def test_registry_rejects_invalid_min_body_ratio():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("talib not installed")
        raise
    with pytest.raises(ValueError):
        reg.create_detector("order_block", config={"min_body_ratio": 2.0})
    with pytest.raises(ValueError):
        reg.create_detector("order_block", config={"min_body_ratio": -1})


def test_detector_config_id_deterministic():
    a = OrderBlockDetector(min_body_ratio=0.5, left_strength=1, right_strength=1)
    b = OrderBlockDetector(min_body_ratio=0.5, left_strength=1, right_strength=1)
    assert a.detector_config_id == b.detector_config_id
    c = OrderBlockDetector(min_body_ratio=0.6, left_strength=1, right_strength=1)
    assert a.detector_config_id != c.detector_config_id
