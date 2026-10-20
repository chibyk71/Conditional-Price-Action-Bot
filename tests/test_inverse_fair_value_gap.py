"""Phase 4C — Inverse Fair Value Gap (IFVG) detection tests."""

from __future__ import annotations

import pytest

from cpab.detectors.fair_value_gap import FairValueGapDetector
from cpab.detectors.fair_value_gap_lifecycle import FairValueGapLifecycleDetector
from cpab.detectors.inverse_fair_value_gap import InverseFairValueGapDetector
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.models import Candle
from cpab.core.observation_store import ObservationStore


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


START = 1_700_000_000
STEP = 60


def _series(
    ohlc_list: list[tuple[float, float, float, float]],
    *,
    start: int = START,
    step: int = STEP,
    timeframe: str = "1m",
    symbol: str = "R_100",
    last_closed: bool = True,
) -> list[Candle]:
    out: list[Candle] = []
    for i, (o, h, l, c) in enumerate(ohlc_list):
        closed = True if i < len(ohlc_list) - 1 else last_closed
        out.append(
            _c(
                start + i * step,
                o,
                h,
                l,
                c,
                symbol=symbol,
                timeframe=timeframe,
                is_closed=closed,
            )
        )
    return out


def _bullish_fvg_base() -> list[tuple[float, float, float, float]]:
    """c1 high=10, c3 low=12 → bullish FVG [10, 12]."""
    return [
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    ]


def _bearish_fvg_base() -> list[tuple[float, float, float, float]]:
    """c1 low=12, c3 high=10.5 → bearish FVG [10.5, 12]."""
    return [
        (13.0, 14.0, 12.0, 12.5),
        (12.0, 13.0, 11.0, 11.5),
        (10.0, 10.5, 9.0, 9.5),
    ]


# ---------------------------------------------------------------------------
# 1–3. Bullish/bearish invalidation → IFVG direction and variant
# ---------------------------------------------------------------------------


def test_bullish_fvg_invalidation_produces_bearish_ifvg():
    base = _bullish_fvg_base()
    # close < L=10
    candles = _series(base + [(12.0, 12.5, 9.0, 9.5)])
    det = InverseFairValueGapDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) == 1
    o = obs[0]
    assert o.type == "inverse_fair_value_gap"
    assert o.variant == "bullish_to_bearish"
    assert o.direction == "bearish"
    assert o.event_time == candles[3].epoch
    assert o.confirmed_at == candles[3].epoch
    assert o.developing is False


def test_bearish_fvg_invalidation_produces_bullish_ifvg():
    base = _bearish_fvg_base()
    # close > U=12
    candles = _series(base + [(11.0, 12.5, 10.8, 12.2)])
    det = InverseFairValueGapDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) == 1
    o = obs[0]
    assert o.variant == "bearish_to_bullish"
    assert o.direction == "bullish"
    assert o.event_time == candles[3].epoch


def test_direction_and_variant_mapping():
    """Explicit mapping table for both polarities."""
    det = InverseFairValueGapDetector()

    bull = det.detect_all_historical(
        _series(_bullish_fvg_base() + [(12.0, 12.5, 9.0, 9.5)])
    )
    assert bull[0].variant == "bullish_to_bearish"
    assert bull[0].direction == "bearish"

    bear = det.detect_all_historical(
        _series(_bearish_fvg_base() + [(11.0, 12.5, 10.8, 12.2)])
    )
    assert bear[0].variant == "bearish_to_bullish"
    assert bear[0].direction == "bullish"


# ---------------------------------------------------------------------------
# 4. Equality does not invert
# ---------------------------------------------------------------------------


def test_equality_at_boundary_does_not_invert():
    base = _bullish_fvg_base()
    # close == L=10 → Phase 4B does not invalidate
    candles = _series(base + [(12.5, 13.0, 10.0, 10.0)])
    det = InverseFairValueGapDetector()
    assert det.detect_all_historical(candles) == []

    # Lifecycle should also have no invalidation
    life = FairValueGapLifecycleDetector()
    inv = [o for o in life.detect_all_historical(candles) if o.variant == "invalidation"]
    assert inv == []


# ---------------------------------------------------------------------------
# 5–6. Original bounds preserved; source FVG reference; identity
# ---------------------------------------------------------------------------


def test_original_bounds_preserved_and_source_reference():
    base = _bullish_fvg_base()
    candles = _series(base + [(12.0, 12.5, 9.0, 9.5)])
    fvg_det = FairValueGapDetector()
    fvgs = fvg_det.detect_all_historical(candles)
    assert len(fvgs) == 1
    fvg = fvgs[0]

    det = InverseFairValueGapDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) == 1
    o = obs[0]
    assert o.payload["lower_bound"] == 10.0
    assert o.payload["upper_bound"] == 12.0
    assert o.payload["source_fvg_direction"] == "bullish"
    assert o.reference_ids == (fvg.id,)
    assert o.payload["source_fvg_id"] == fvg.id
    assert fvg.id in o.id


def test_deterministic_identity_and_idempotency():
    candles = _series(_bullish_fvg_base() + [(12.0, 12.5, 9.0, 9.5)])
    det = InverseFairValueGapDetector()
    a = det.detect_all_historical(candles)
    b = det.detect_all_historical(candles)
    assert [o.id for o in a] == [o.id for o in b]
    assert len(a) == 1

    store = ObservationStore()
    assert store.accept(a) == 1
    assert store.accept(b) == 0


# ---------------------------------------------------------------------------
# 7–8. No pre-close; formation candle cannot invalidate
# ---------------------------------------------------------------------------


def test_no_ifvg_before_close_or_on_formation_candle():
    base = _bullish_fvg_base()
    # Only formation
    det = InverseFairValueGapDetector()
    assert det.detect_all_historical(_series(base)) == []

    # Forming invalidation candle
    candles = _series(base + [(12.0, 12.5, 9.0, 9.5)], last_closed=False)
    assert det.detect(candles) == []
    assert det.detect_all_historical(candles) == []


# ---------------------------------------------------------------------------
# 9–10. Continuity / malformed OHLC follow Phase 4B conventions
# ---------------------------------------------------------------------------


def test_missing_timeframe_step_no_inference():
    base = _bullish_fvg_base()
    s = START
    # Gap between formation and putative invalidation candle
    candles = [
        _c(s, *base[0]),
        _c(s + STEP, *base[1]),
        _c(s + 2 * STEP, *base[2]),
        # skip s+3*STEP
        _c(s + 4 * STEP, 12.0, 12.5, 9.0, 9.5),
    ]
    det = InverseFairValueGapDetector()
    # Discontinuous → lifecycle skips → no IFVG
    assert det.detect_all_historical(candles) == []


def test_malformed_ohlc_follows_existing_conventions():
    base = _bullish_fvg_base()
    # high < low on interaction candle
    candles = _series(base + [(12.0, 8.0, 9.5, 9.0)])
    det = InverseFairValueGapDetector()
    assert det.detect_all_historical(candles) == []

    bad_tf = _series(base + [(12.0, 12.5, 9.0, 9.5)], timeframe="unknown_tf")
    assert det.detect_all_historical(bad_tf) == []


# ---------------------------------------------------------------------------
# 11. Multiple overlapping FVGs independent
# ---------------------------------------------------------------------------


def test_multiple_overlapping_fvgs_independent():
    ohlc = [
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),  # FVG1 [10,12]
        (13.0, 13.5, 11.5, 12.5),  # touch FVG1 only
        (19.0, 20.0, 18.5, 19.5),
        (19.5, 21.0, 19.0, 20.5),
        (22.2, 23.5, 22.0, 22.8),  # FVG2 [20,22]
        (23.0, 23.5, 19.0, 19.5),  # invalidates FVG2 (close < 20); may also hit FVG1
    ]
    candles = _series(ohlc)
    fvg_det = FairValueGapDetector()
    fvgs = fvg_det.detect_all_historical(candles)
    assert len(fvgs) >= 2

    det = InverseFairValueGapDetector()
    obs = det.detect_all_historical(candles)
    # At least FVG2 should invert on the last bar (close 19.5 < 20)
    assert len(obs) >= 1
    sources = {o.reference_ids[0] for o in obs if o.reference_ids}
    fvg_ids = {f.id for f in fvgs}
    assert sources.issubset(fvg_ids)
    assert len(sources) == len(obs)  # one IFVG per source invalidation


# ---------------------------------------------------------------------------
# 12. Registry + Phase 4A/4B regression
# ---------------------------------------------------------------------------


def test_registry_ifvg():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("TA-Lib not installed in this environment")
        raise
    info = reg.validate_config("inverse_fair_value_gap", {})
    assert info.detector_id == "inverse_fair_value_gap"
    assert info.min_history == 4
    assert info.config == {}
    assert info.detector_config_id == deterministic_config_id({})

    info2 = reg.validate_config("ifvg", {})
    assert info2.detector_id == "inverse_fair_value_gap"

    det, _ = reg.create_detector("inverse_fair_value_gap")
    assert isinstance(det, InverseFairValueGapDetector)
    assert det.min_history == 4

    with pytest.raises(ValueError):
        reg.validate_config("inverse_fair_value_gap", {"min_size": 1.0})


def test_phase4a_and_4b_unchanged():
    base = _bullish_fvg_base()
    candles = _series(base + [(12.0, 12.5, 9.0, 9.5)])

    fvg_det = FairValueGapDetector()
    fvgs = fvg_det.detect_all_historical(candles)
    assert len(fvgs) == 1
    assert fvgs[0].type == "fair_value_gap"
    assert fvgs[0].payload["lower_bound"] == 10.0

    life = FairValueGapLifecycleDetector()
    life_obs = life.detect_all_historical(candles)
    inv = [o for o in life_obs if o.variant == "invalidation"]
    assert len(inv) == 1
    assert inv[0].type == "fair_value_gap_lifecycle"

    # IFVG count matches invalidation count
    ifvg = InverseFairValueGapDetector().detect_all_historical(candles)
    assert len(ifvg) == len(inv)


def test_historical_incremental_parity():
    candles = _series(
        _bullish_fvg_base()
        + [
            (13.0, 13.5, 11.5, 12.5),
            (12.0, 12.5, 9.0, 9.5),
        ]
    )
    det = InverseFairValueGapDetector()
    hist = det.detect_all_historical(candles)
    incremental = []
    for n in range(1, len(candles) + 1):
        incremental.extend(det.detect(candles[:n]))
    assert sorted(o.id for o in hist) == sorted(o.id for o in incremental)
