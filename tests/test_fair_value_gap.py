"""Phase 4A — Fair Value Gap (FVG) formation detector tests."""

from __future__ import annotations

import pytest

from cpab.detectors.fair_value_gap import FairValueGapDetector
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.models import Candle, Observation
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


def _triple(
    c1_ohlc: tuple[float, float, float, float],
    c2_ohlc: tuple[float, float, float, float],
    c3_ohlc: tuple[float, float, float, float],
    *,
    start: int = START,
    step: int = STEP,
    timeframe: str = "1m",
    symbol: str = "R_100",
    c3_closed: bool = True,
) -> list[Candle]:
    """Three consecutive candles; OHLC = (open, high, low, close)."""
    e1, e2, e3 = start, start + step, start + 2 * step
    return [
        _c(e1, *c1_ohlc, symbol=symbol, timeframe=timeframe),
        _c(e2, *c2_ohlc, symbol=symbol, timeframe=timeframe),
        _c(e3, *c3_ohlc, symbol=symbol, timeframe=timeframe, is_closed=c3_closed),
    ]


# ---------------------------------------------------------------------------
# 1–2. Bullish / bearish bounds and payload
# ---------------------------------------------------------------------------


def test_bullish_fvg_bounds_and_payload():
    # c1 high=10, c3 low=12 → gap [10, 12]
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),   # c1
        (9.5, 11.0, 9.0, 10.5),  # c2 middle (any colour)
        (12.2, 13.5, 12.0, 12.8),  # c3
    )
    det = FairValueGapDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) == 1
    o = obs[0]
    assert o.type == "fair_value_gap"
    assert o.direction == "bullish"
    assert o.variant == "bullish"
    assert o.developing is False
    assert o.event_time == candles[2].epoch
    assert o.confirmed_at == candles[2].epoch
    assert o.candle_epoch == candles[2].epoch
    assert o.payload["lower_bound"] == 10.0
    assert o.payload["upper_bound"] == 12.0
    assert o.payload["size"] == 2.0
    assert o.payload["candle_1_epoch"] == candles[0].epoch
    assert o.payload["candle_2_epoch"] == candles[1].epoch
    assert o.payload["candle_3_epoch"] == candles[2].epoch
    assert o.reference_ids == ()
    assert o.detector_id == "fair_value_gap"
    assert o.detector_config_id == det.detector_config_id


def test_bearish_fvg_bounds_and_payload():
    # c1 low=12, c3 high=10 → gap [10, 12]
    candles = _triple(
        (13.0, 14.0, 12.0, 12.5),  # c1
        (12.0, 13.0, 11.0, 11.5),  # c2
        (11.0, 10.0, 9.0, 9.5),    # c3 high=10 < c1.low=12
    )
    # Fix OHLC: high must be >= open/close. Use valid bearish displacement.
    candles = _triple(
        (13.0, 14.0, 12.0, 12.5),
        (12.0, 13.0, 11.0, 11.5),
        (11.5, 11.8, 9.0, 9.5),  # high=11.8 < c1.low=12
    )
    det = FairValueGapDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) == 1
    o = obs[0]
    assert o.direction == "bearish"
    assert o.variant == "bearish"
    assert o.payload["lower_bound"] == 11.8  # c3.high
    assert o.payload["upper_bound"] == 12.0  # c1.low
    assert o.payload["size"] == pytest.approx(0.2)
    assert o.payload["candle_3_epoch"] == candles[2].epoch
    assert o.event_time == candles[2].epoch
    assert o.confirmed_at == candles[2].epoch


# ---------------------------------------------------------------------------
# 3. Equal boundaries do not qualify
# ---------------------------------------------------------------------------


def test_no_fvg_when_boundaries_equal():
    # c1.high == c3.low → not upper > lower
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (10.0, 11.0, 10.0, 10.5),  # low == 10 == c1.high
    )
    det = FairValueGapDetector()
    assert det.detect_all_historical(candles) == []
    assert det.detect(candles) == []


# ---------------------------------------------------------------------------
# 4. Middle-candle colour does not affect geometric detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "c2_ohlc",
    [
        (9.5, 11.0, 9.0, 10.5),  # bullish body
        (10.5, 11.0, 9.0, 9.5),  # bearish body
        (10.0, 11.0, 9.0, 10.0),  # doji-ish
    ],
)
def test_middle_candle_colour_irrelevant(c2_ohlc):
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        c2_ohlc,
        (12.2, 13.5, 12.0, 12.8),
    )
    det = FairValueGapDetector()
    obs = det.detect_all_historical(candles)
    assert len(obs) == 1
    assert obs[0].direction == "bullish"
    assert obs[0].payload["lower_bound"] == 10.0
    assert obs[0].payload["upper_bound"] == 12.0


# ---------------------------------------------------------------------------
# 5. Forming candle 3 does not produce an observation
# ---------------------------------------------------------------------------


def test_forming_candle_3_no_observation():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
        c3_closed=False,
    )
    det = FairValueGapDetector()
    assert det.detect(candles) == []
    assert det.detect_all_historical(candles) == []


# ---------------------------------------------------------------------------
# 6. Missing candles / incorrect timeframe steps suppress formation
# ---------------------------------------------------------------------------


def test_gap_in_epochs_suppresses_formation():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )
    # Skip middle spacing: set c3 epoch to +3 steps
    candles[2] = _c(
        START + 3 * STEP,
        11.0,
        13.0,
        12.0,
        12.5,
    )
    det = FairValueGapDetector()
    assert det.detect_all_historical(candles) == []


def test_non_consecutive_step_suppresses():
    # 5m series with 1m-like OHLC but wrong step relative to label
    candles = [
        _c(START, 9.0, 10.0, 8.0, 9.5, timeframe="5m"),
        _c(START + 60, 9.5, 11.0, 9.0, 10.5, timeframe="5m"),  # 60s not 300s
        _c(START + 120, 12.2, 13.5, 12.0, 12.8, timeframe="5m"),
    ]
    det = FairValueGapDetector()
    assert det.detect_all_historical(candles) == []


# ---------------------------------------------------------------------------
# 7. Unknown timeframe durations rejected safely
# ---------------------------------------------------------------------------


def test_unknown_timeframe_rejected():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
        timeframe="weird_tf",
    )
    det = FairValueGapDetector()
    assert det.detect_all_historical(candles) == []
    assert det.detect(candles) == []


# ---------------------------------------------------------------------------
# 8. Mixed symbols or timeframes rejected
# ---------------------------------------------------------------------------


def test_mixed_symbols_rejected():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )
    candles[1] = _c(
        candles[1].epoch,
        9.5,
        11.0,
        9.0,
        10.5,
        symbol="R_75",
    )
    det = FairValueGapDetector()
    assert det.detect_all_historical(candles) == []


def test_mixed_timeframes_rejected():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )
    candles[0] = _c(
        candles[0].epoch,
        9.0,
        10.0,
        8.0,
        9.5,
        timeframe="5m",
    )
    det = FairValueGapDetector()
    assert det.detect_all_historical(candles) == []


# ---------------------------------------------------------------------------
# 9. Repeated detection produces stable observation IDs
# ---------------------------------------------------------------------------


def test_stable_observation_ids():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )
    det = FairValueGapDetector()
    a = det.detect_all_historical(candles)
    b = det.detect_all_historical(candles)
    assert len(a) == 1 and len(b) == 1
    assert a[0].id == b[0].id
    assert a[0].id == det.detect(candles)[0].id

    store = ObservationStore(max_size=100)
    assert store.accept([a[0]]) == 1
    assert store.accept([b[0]]) == 0
    retained = [o for o in store.query("R_100", "1m") if o.type == "fair_value_gap"]
    assert len(retained) == 1


# ---------------------------------------------------------------------------
# 10. Consecutive / overlapping FVGs remain separate observations
# ---------------------------------------------------------------------------


def test_consecutive_fvgs_are_separate():
    # Four candles → two overlapping triples both can form bullish FVG
    # c0 high < c2 low, and c1 high < c3 low
    candles = [
        _c(START, 9.0, 10.0, 8.0, 9.5),
        _c(START + STEP, 10.0, 11.0, 9.5, 10.5),
        _c(START + 2 * STEP, 12.2, 13.5, 12.0, 12.8),  # low 12 > c0.high 10
        _c(START + 3 * STEP, 14.2, 15.5, 14.0, 14.8),  # low 14 > c1.high 11
    ]
    det = FairValueGapDetector()
    hist = det.detect_all_historical(candles)
    assert len(hist) == 2
    assert hist[0].id != hist[1].id
    assert hist[0].payload["candle_3_epoch"] == candles[2].epoch
    assert hist[1].payload["candle_3_epoch"] == candles[3].epoch
    # Incremental on full series only emits the latest
    incr = det.detect(candles)
    assert len(incr) == 1
    assert incr[0].id == hist[1].id


# ---------------------------------------------------------------------------
# 11. Historical and incremental agree on confirmed formations
# ---------------------------------------------------------------------------


def test_historical_and_incremental_agree():
    series = [
        _c(START + i * STEP, 9.0 + i, 10.0 + i, 8.0 + i, 9.5 + i)
        for i in range(6)
    ]
    # Force a clear bullish gap at the end: adjust last three
    series[-3] = _c(START + 3 * STEP, 9.0, 10.0, 8.0, 9.5)
    series[-2] = _c(START + 4 * STEP, 9.5, 11.0, 9.0, 10.5)
    series[-1] = _c(START + 5 * STEP, 12.2, 13.5, 12.0, 12.8)

    det = FairValueGapDetector()
    hist = det.detect_all_historical(series)
    # Incremental only reports those confirmed on the latest bar
    incr = det.detect(series)
    latest = series[-1].epoch
    hist_on_latest = [o for o in hist if o.confirmed_at == latest]
    assert [o.id for o in incr] == [o.id for o in hist_on_latest]
    assert len(incr) >= 1


# ---------------------------------------------------------------------------
# 12. Later candles cannot alter an existing FVG's bounds or timestamps
# ---------------------------------------------------------------------------


def test_later_candles_do_not_mutate_prior_fvg():
    base = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )
    det = FairValueGapDetector()
    first = det.detect_all_historical(base)
    assert len(first) == 1
    original = first[0]

    extended = base + [
        _c(START + 3 * STEP, 12.5, 14.0, 12.0, 13.0),
        _c(START + 4 * STEP, 13.0, 15.0, 12.5, 14.0),
    ]
    later = det.detect_all_historical(extended)
    matching = [o for o in later if o.payload["candle_3_epoch"] == original.payload["candle_3_epoch"]]
    assert len(matching) == 1
    o = matching[0]
    assert o.id == original.id
    assert o.payload["lower_bound"] == original.payload["lower_bound"]
    assert o.payload["upper_bound"] == original.payload["upper_bound"]
    assert o.event_time == original.event_time
    assert o.confirmed_at == original.confirmed_at


# ---------------------------------------------------------------------------
# 13. Malformed OHLC handled safely
# ---------------------------------------------------------------------------


def test_malformed_ohlc_skipped():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )
    # Invert high/low on middle candle
    bad = list(candles)
    bad[1] = Candle(
        symbol="R_100",
        timeframe="1m",
        epoch=candles[1].epoch,
        open=10.0,
        high=9.0,  # high < low
        low=11.0,
        close=10.0,
        is_closed=True,
    )
    det = FairValueGapDetector()
    assert det.detect_all_historical(bad) == []


def test_nan_ohlc_skipped():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )
    bad = list(candles)
    bad[0] = Candle(
        symbol="R_100",
        timeframe="1m",
        epoch=candles[0].epoch,
        open=float("nan"),
        high=10.0,
        low=8.0,
        close=9.5,
        is_closed=True,
    )
    det = FairValueGapDetector()
    assert det.detect_all_historical(bad) == []


# ---------------------------------------------------------------------------
# 14. Registry registration, min_history, config identity
# ---------------------------------------------------------------------------


def test_registry_fair_value_gap():
    try:
        reg = build_default_registry()
    except ModuleNotFoundError as e:
        if "talib" in str(e).lower():
            pytest.skip("TA-Lib not installed in this environment")
        raise
    assert reg.is_supported("fair_value_gap")
    assert reg.is_supported("fvg")
    info = reg.validate_config("fair_value_gap", {})
    assert info.detector_id == "fair_value_gap"
    assert info.min_history == 3
    assert info.implementation == "fair_value_gap"
    assert info.config == {}
    expected_cfg_id = deterministic_config_id({})
    assert info.detector_config_id == expected_cfg_id

    det, info2 = reg.create_detector("fair_value_gap")
    assert isinstance(det, FairValueGapDetector)
    assert det.min_history == 3
    assert det.detector_config_id == expected_cfg_id
    assert info2.detector_config_id == expected_cfg_id

    # Unknown params rejected
    with pytest.raises(ValueError, match="unknown"):
        reg.validate_config("fair_value_gap", {"min_size": 1.0})


def test_min_history_insufficient():
    det = FairValueGapDetector()
    assert det.min_history == 3
    two = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )[:2]
    assert det.detect(two) == []
    assert det.detect_all_historical(two) == []


def test_incremental_only_latest_bar():
    candles = _triple(
        (9.0, 10.0, 8.0, 9.5),
        (9.5, 11.0, 9.0, 10.5),
        (12.2, 13.5, 12.0, 12.8),
    )
    det = FairValueGapDetector()
    obs = det.detect(candles)
    assert len(obs) == 1
    assert obs[0].confirmed_at == candles[2].epoch
