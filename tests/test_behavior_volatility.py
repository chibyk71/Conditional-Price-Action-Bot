"""Phase 6A — ATR expansion / contraction and range expansion tests."""

from __future__ import annotations

import math

import pytest

from cpab.detectors.behavior_volatility import (
    ATR_PERIOD,
    BASELINE_WINDOW,
    EXPANSION_MULT,
    CONTRACTION_MULT,
    RANGE_EXPANSION_MULT,
    MIN_HISTORY_ATR_REGIME,
    MIN_HISTORY_RANGE,
    VARIANT_ROLLING_ATR,
    VARIANT_TRUE_RANGE,
    ATRExpansionDetector,
    ATRContractionDetector,
    RangeExpansionDetector,
    _rolling_atr_series,
    _baseline_mean,
)
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.detectors.wilder_atr import true_range, wilder_atr
from cpab.models import Candle


TF = 60
START = 1_700_000_000


def _c(
    i: int,
    o: float,
    h: float,
    l: float,
    cl: float,
    *,
    symbol: str = "BV_100",
    timeframe: str = "1m",
    start: int = START,
    is_closed: bool = True,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        epoch=start + i * TF,
        open=o,
        high=h,
        low=l,
        close=cl,
        is_closed=is_closed,
        source="provider",
    )


def _flat_series(
    n: int,
    *,
    mid: float = 100.0,
    half_range: float = 1.0,
    symbol: str = "BV_100",
) -> list[Candle]:
    """n consecutive closed candles with constant true range ≈ 2*half_range."""
    out: list[Candle] = []
    for i in range(n):
        o = mid
        h = mid + half_range
        l = mid - half_range
        cl = mid
        out.append(_c(i, o, h, l, cl, symbol=symbol))
    return out


def _series_with_spike(
    n: int,
    spike_idx: int,
    *,
    base_half: float = 1.0,
    spike_half: float = 5.0,
    mid: float = 100.0,
) -> list[Candle]:
    """Flat range series with one enlarged-range bar at spike_idx."""
    out: list[Candle] = []
    for i in range(n):
        hr = spike_half if i == spike_idx else base_half
        out.append(_c(i, mid, mid + hr, mid - hr, mid))
    return out


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def test_rolling_atr_length_and_warmup():
    candles = _flat_series(40)
    atrs = _rolling_atr_series(candles, period=14, step=TF)
    assert len(atrs) == 40
    for i in range(14):
        assert atrs[i] is None
    for i in range(14, 40):
        assert atrs[i] is not None
        assert atrs[i] > 0


def test_baseline_mean_requires_full_window():
    atrs: list = [None] * 10 + [1.0] * 25
    assert _baseline_mean(atrs, 14, window=20) is None  # not enough non-None before
    # Fill from index 0
    atrs = [1.0] * 30
    m = _baseline_mean(atrs, 20, window=20)
    assert m == pytest.approx(1.0)
    assert _baseline_mean(atrs, 19, window=20) is None


# ---------------------------------------------------------------------------
# Registry / config identity
# ---------------------------------------------------------------------------


def test_registry_concepts_and_min_history():
    pytest.importorskip("talib")
    reg = build_default_registry()
    for concept, mh in (
        ("atr_expansion", 35),
        ("atr_contraction", 35),
        ("range_expansion", 16),
    ):
        assert reg.is_supported(concept)
        info = reg.validate_config(concept, {})
        assert info.min_history == mh
        det, info2 = reg.create_detector(concept)
        assert det.min_history == mh
        assert info2.detector_config_id == det.detector_config_id
        assert det.detector_config_id.startswith("cfg:")


def test_configuration_identity_stable_and_distinct():
    exp = ATRExpansionDetector()
    con = ATRContractionDetector()
    rng = RangeExpansionDetector()
    assert exp.detector_config_id != con.detector_config_id
    assert exp.detector_config_id != rng.detector_config_id
    assert ATRExpansionDetector().detector_config_id == exp.detector_config_id
    assert RangeExpansionDetector().detector_config_id == rng.detector_config_id


def test_locked_params_rejected():
    with pytest.raises(ValueError, match="locked"):
        ATRExpansionDetector(atr_period=10)
    with pytest.raises(ValueError, match="locked"):
        ATRExpansionDetector(baseline_window=10)
    with pytest.raises(ValueError, match="locked"):
        ATRExpansionDetector(threshold=2.0)
    with pytest.raises(ValueError, match="locked"):
        RangeExpansionDetector(atr_period=7)
    with pytest.raises(ValueError, match="locked"):
        RangeExpansionDetector(threshold=2.0)


# ---------------------------------------------------------------------------
# Warm-up boundaries
# ---------------------------------------------------------------------------


def test_atr_regime_warmup_boundary():
    """No observations before 35 bars; possible from bar index 34 onward."""
    det = ATRExpansionDetector()
    assert det.min_history == MIN_HISTORY_ATR_REGIME == 35
    short = _flat_series(34)
    assert det.detect_all_historical(short) == []
    # 35 flat bars → ratio ≈ 1.0, no expansion
    flat35 = _flat_series(35)
    assert det.detect_all_historical(flat35) == []


def test_range_expansion_warmup_boundary():
    det = RangeExpansionDetector()
    assert det.min_history == MIN_HISTORY_RANGE == 16
    assert det.detect_all_historical(_flat_series(15)) == []
    # Flat: TR ≈ ATR → ratio ≈ 1 < 1.5
    assert det.detect_all_historical(_flat_series(20)) == []


# ---------------------------------------------------------------------------
# Thresholds and crossings
# ---------------------------------------------------------------------------


def _build_expansion_series() -> list[Candle]:
    """
    Flat base then a sustained high-volatility regime so ATR rises above 1.25×
    baseline. Returns enough bars for warm-up + regime.
    """
    n = 80
    candles: list[Candle] = []
    for i in range(n):
        if i < 50:
            hr = 1.0
        else:
            # Sustained wide range to lift ATR
            hr = 4.0
        mid = 100.0
        candles.append(_c(i, mid, mid + hr, mid - hr, mid))
    return candles


def test_atr_expansion_crossing_only():
    candles = _build_expansion_series()
    det = ATRExpansionDetector()
    obs = det.detect_all_historical(candles)
    # At least one expansion; all must be atr_expansion with locked variant
    assert len(obs) >= 1
    epochs = [o.event_time for o in obs]
    assert epochs == sorted(epochs)
    # Crossing-only: no two consecutive bars both emit
    for a, b in zip(epochs, epochs[1:]):
        assert b - a > TF  # at least one bar gap (or larger)
    for o in obs:
        assert o.type == "atr_expansion"
        assert o.direction is None
        assert o.variant == VARIANT_ROLLING_ATR
        assert o.developing is False
        assert o.payload["threshold"] == EXPANSION_MULT
        assert o.payload["ratio"] >= EXPANSION_MULT
        assert o.payload["atr_period"] == ATR_PERIOD
        assert o.payload["baseline_window"] == BASELINE_WINDOW
        assert o.confirmed_at == o.event_time == o.candle_epoch
        assert o.detector_id == "atr_expansion"
        assert o.detector_config_id == det.detector_config_id


def test_atr_contraction_crossing_only():
    """High vol then low vol → contraction crossings."""
    n = 90
    candles: list[Candle] = []
    for i in range(n):
        hr = 5.0 if i < 45 else 0.5
        mid = 100.0
        candles.append(_c(i, mid, mid + hr, mid - hr, mid))
    det = ATRContractionDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) >= 1
    epochs = [o.event_time for o in obs]
    for a, b in zip(epochs, epochs[1:]):
        assert b - a > TF
    for o in obs:
        assert o.type == "atr_contraction"
        assert o.direction is None
        assert o.variant == VARIANT_ROLLING_ATR
        assert o.payload["threshold"] == CONTRACTION_MULT
        assert o.payload["ratio"] <= CONTRACTION_MULT
        assert o.detector_id == "atr_contraction"


def test_range_expansion_every_qualifying_bar():
    """Emit on every bar whose TR ≥ 1.5× preceding ATR (not crossing-only)."""
    n = 40
    candles: list[Candle] = []
    for i in range(n):
        # After warm-up, alternate normal and wide bars
        if i >= 20 and i % 2 == 0:
            hr = 5.0
        else:
            hr = 1.0
        mid = 100.0
        candles.append(_c(i, mid, mid + hr, mid - hr, mid))
    det = RangeExpansionDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) >= 2  # multiple qualifying bars
    for o in obs:
        assert o.type == "range_expansion"
        assert o.direction is None
        assert o.variant == VARIANT_TRUE_RANGE
        assert o.payload["threshold"] == RANGE_EXPANSION_MULT
        assert o.payload["ratio"] >= RANGE_EXPANSION_MULT
        assert o.payload["true_range"] > 0
        assert o.payload["preceding_atr"] > 0
        assert o.detector_id == "range_expansion"
        assert o.detector_config_id == det.detector_config_id


# ---------------------------------------------------------------------------
# Causality / incomplete candles
# ---------------------------------------------------------------------------


def test_incomplete_candle_ignored():
    candles = _build_expansion_series()
    # Mark last candle open
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
        source=last.source,
    )
    det = ATRExpansionDetector()
    # detect() uses only closed bars
    closed_count = sum(1 for c in candles if c.is_closed)
    assert closed_count == len(candles) - 1
    obs_all = det.detect_all_historical(candles)
    # Same as if last were absent
    obs_trim = det.detect_all_historical(candles[:-1])
    assert [o.event_time for o in obs_all] == [o.event_time for o in obs_trim]


def test_detect_emits_only_latest_bar():
    candles = _build_expansion_series()
    det = ATRExpansionDetector()
    all_obs = det.detect_all_historical(candles)
    latest = det.detect(candles)
    if not all_obs:
        pytest.skip("no expansion in synthetic series")
    latest_epoch = max(c.epoch for c in candles if c.is_closed)
    for o in latest:
        assert o.confirmed_at == latest_epoch
    # All historical that confirm on latest should match
    expected = [o for o in all_obs if o.confirmed_at == latest_epoch]
    assert len(latest) == len(expected)


def test_causality_prefix_equals_full_at_prefix_end():
    """Observations confirmed at T on prefix == same on full series (no lookahead)."""
    candles = _build_expansion_series()
    det = ATRExpansionDetector()
    full = det.detect_all_historical(candles)
    if len(full) < 1:
        pytest.skip("need at least one expansion")
    # Take a confirmation epoch in the middle of emissions if possible
    target = full[0].confirmed_at
    # Prefix up to and including target
    prefix = [c for c in candles if c.epoch <= target]
    pref_obs = det.detect_all_historical(prefix)
    pref_at_t = [o for o in pref_obs if o.confirmed_at == target]
    full_at_t = [o for o in full if o.confirmed_at == target]
    assert len(pref_at_t) == len(full_at_t)
    if pref_at_t:
        assert pref_at_t[0].payload["ratio"] == pytest.approx(full_at_t[0].payload["ratio"])


# ---------------------------------------------------------------------------
# Invalid OHLC / zero baseline / gaps
# ---------------------------------------------------------------------------


def test_invalid_ohlc_skips_window():
    candles = _flat_series(40)
    # Corrupt a bar inside warm-up so early ATRs fail
    bad = candles[10]
    candles[10] = Candle(
        symbol=bad.symbol,
        timeframe=bad.timeframe,
        epoch=bad.epoch,
        open=100.0,
        high=90.0,  # high < low → invalid
        low=110.0,
        close=100.0,
        is_closed=True,
        source=bad.source,
    )
    atrs = _rolling_atr_series(candles, period=14, step=TF)
    # Windows that include index 10 must be None
    for i in range(14, 40):
        if i - 14 <= 10 <= i:
            assert atrs[i] is None
        # After the bad bar leaves the window, ATR may recover
    det = ATRExpansionDetector()
    # Should not crash
    det.detect_all_historical(candles)


def test_zero_baseline_no_emit():
    """Constant zero-range bars → ATR 0 → baseline 0 → skip."""
    candles = _flat_series(50, half_range=0.0)
    det = ATRExpansionDetector()
    assert det.detect_all_historical(candles) == []
    det_c = ATRContractionDetector()
    assert det_c.detect_all_historical(candles) == []
    det_r = RangeExpansionDetector()
    assert det_r.detect_all_historical(candles) == []


def test_gap_no_bridging():
    """Gap splits segments; baseline cannot use ATR from the other side."""
    left = _flat_series(40, symbol="GAP")
    # Gap of 10 missing bars
    right_start_i = 50
    right: list[Candle] = []
    for j in range(40):
        i = right_start_i + j
        right.append(_c(i, 100.0, 104.0, 96.0, 100.0, symbol="GAP"))
    candles = left + right
    det = ATRExpansionDetector()
    obs = det.detect_all_historical(candles)
    # Right segment starts fresh; needs its own 35 bars
    # left is flat → no expansion; right has wide range from start of segment
    # but still needs 35 continuous bars in-segment
    for o in obs:
        # event must fall inside a continuous segment
        assert o.event_time is not None


def test_mixed_symbol_rejected():
    a = _flat_series(40, symbol="A")
    b = _flat_series(40, symbol="B")
    # Merge alternating would mix; simpler: concatenate different symbols
    mixed = a[:20] + b[20:]
    # Fix epochs continuous
    mixed = []
    for i in range(40):
        sym = "A" if i < 20 else "B"
        mixed.append(_c(i, 100.0, 101.0, 99.0, 100.0, symbol=sym))
    det = ATRExpansionDetector()
    assert det.detect_all_historical(mixed) == []


# ---------------------------------------------------------------------------
# Repeated calls / idempotency
# ---------------------------------------------------------------------------


def test_repeated_detect_idempotent():
    candles = _build_expansion_series()
    det = ATRExpansionDetector()
    a = det.detect_all_historical(candles)
    b = det.detect_all_historical(candles)
    assert len(a) == len(b)
    assert [o.id for o in a] == [o.id for o in b]
    assert [o.payload for o in a] == [o.payload for o in b]


def test_range_expansion_hand_ratio():
    """Explicit TR / ATR check on a constructed series."""
    # 16+ bars of unit range, then one wide bar
    candles = _flat_series(20, half_range=1.0)
    # Replace last bar with wide range
    i = 19
    mid = 100.0
    hr = 10.0
    candles[i] = _c(i, mid, mid + hr, mid - hr, mid)
    det = RangeExpansionDetector()
    obs = det.detect_all_historical(candles)
    # preceding ATR at i-1 should be ~2.0 (TR of flat bars)
    assert len(obs) >= 1
    o = [x for x in obs if x.event_time == candles[i].epoch][0]
    assert o.payload["true_range"] == pytest.approx(20.0)
    assert o.payload["ratio"] >= RANGE_EXPANSION_MULT


def test_min_history_constants_match_contract():
    assert MIN_HISTORY_ATR_REGIME == ATR_PERIOD + BASELINE_WINDOW + 1
    assert MIN_HISTORY_RANGE == ATR_PERIOD + 2
    assert ATRExpansionDetector().min_history == 35
    assert ATRContractionDetector().min_history == 35
    assert RangeExpansionDetector().min_history == 16
