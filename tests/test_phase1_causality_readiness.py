"""
Causality + readiness acceptance tests (Phase 1).

Causality:
  - Raw: for each historical bar i, evaluate_raw_series(prefix[:i+1])[i]
    equals evaluate_raw_series(full)[i].
  - Observation: derive AdapterRawResult + Observation independently from the
    prefix array value and from the full-series array value at the same bar;
    compare type, direction, candle identity, detector_config_id, payload, id.
  - Zero raw: neither path emits an observation.
"""
from __future__ import annotations

from cpab.core.engine import ConditionEngine
from cpab.detectors.talib_adapter import (
    AdapterOutcomeKind,
    AdapterRawResult,
    TalibPatternDetector,
    evaluate_raw,
    evaluate_raw_series,
    min_history_for,
    raw_to_observation,
    resolve_pattern_name,
)
from cpab.market.readiness import ReadinessManager
from cpab.models import Candle, ConditionDef, ReadinessState


def _c(o, h, l, c, epoch, is_closed=True):
    return Candle(
        symbol="R_100",
        timeframe="1m",
        epoch=epoch,
        open=float(o),
        high=float(h),
        low=float(l),
        close=float(c),
        is_closed=is_closed,
    )


def _known_positive_engulfing(start=1_700_000_000):
    """Five-bar series; last bar yields CDLENGULFING +100 under TA-Lib 0.8.x."""
    return [
        _c(10.0, 10.5, 9.8, 10.2, start),
        _c(10.2, 10.4, 9.9, 10.0, start + 60),
        _c(10.0, 10.2, 9.5, 9.6, start + 120),
        _c(9.5, 9.8, 9.0, 9.1, start + 180),
        _c(9.0, 10.8, 8.9, 10.6, start + 240),
    ]


def _long_series(n=20, start=1_700_000_000):
    """
    Neutral-ish series long enough for patterns with lookback 10–12.
    Includes a near-doji bar near the end for CDLDOJI coverage.
    """
    candles = []
    for i in range(n):
        base = 10.0 + 0.05 * (i % 3)
        if i == n - 2:
            candles.append(_c(base, base + 2.0, base - 2.0, base + 0.02, start + i * 60))
        else:
            candles.append(_c(base, base + 0.4, base - 0.3, base + 0.15, start + i * 60))
    return candles


def _raw_result_from_value(val: int, pattern: str, candle: Candle) -> AdapterRawResult:
    """Build AdapterRawResult independently from a series-array integer at one bar."""
    resolved = resolve_pattern_name(pattern)
    if val == 0:
        return AdapterRawResult(
            kind=AdapterOutcomeKind.NO_PATTERN,
            pattern=resolved,
            raw_talib_value=0,
            candle_epoch=candle.epoch,
        )
    return AdapterRawResult(
        kind=AdapterOutcomeKind.PATTERN,
        pattern=resolved,
        raw_talib_value=val,
        candle_epoch=candle.epoch,
    )


def _assert_observation_causality(series, pattern: str, det_id: str, cfg: str) -> None:
    need = min_history_for(pattern)
    full_arr = evaluate_raw_series(series, pattern)
    assert len(full_arr) == len(series)

    for i in range(len(series)):
        prefix = series[: i + 1]
        if len(prefix) < need:
            raw = evaluate_raw(prefix, pattern)
            assert raw.kind == AdapterOutcomeKind.INSUFFICIENT_HISTORY
            continue

        prefix_arr = evaluate_raw_series(prefix, pattern)
        prefix_val = int(prefix_arr[i])
        full_val = int(full_arr[i])
        assert prefix_val == full_val, (
            f"{pattern} bar {i} epoch={series[i].epoch}: "
            f"prefix={prefix_val} != full={full_val}"
        )

        # Independently derive results from each series array value
        raw_from_prefix = _raw_result_from_value(prefix_val, pattern, series[i])
        raw_from_full = _raw_result_from_value(full_val, pattern, series[i])
        assert raw_from_prefix.raw_talib_value == raw_from_full.raw_talib_value
        assert raw_from_prefix.kind == raw_from_full.kind

        obs_prefix = raw_to_observation(
            raw_from_prefix, series[i], detector_id=det_id, detector_config_id=cfg
        )
        obs_full = raw_to_observation(
            raw_from_full, series[i], detector_id=det_id, detector_config_id=cfg
        )

        if prefix_val == 0:
            assert obs_prefix is None and obs_full is None
        else:
            assert obs_prefix is not None and obs_full is not None
            assert obs_prefix.id == obs_full.id
            assert obs_prefix.type == obs_full.type
            assert obs_prefix.direction == obs_full.direction
            assert obs_prefix.candle_epoch == obs_full.candle_epoch == series[i].epoch
            assert obs_prefix.detector_config_id == obs_full.detector_config_id == cfg
            assert obs_prefix.payload == obs_full.payload
            assert obs_prefix.payload.get("raw_talib_value") == prefix_val


def test_historical_talib_results_stable_engulfing():
    _assert_observation_causality(
        _known_positive_engulfing(),
        "CDLENGULFING",
        "talib:cdlengulfing",
        "cfg:causality-engulfing",
    )


def test_historical_observation_causality_engulfing_positive_and_zero():
    """Explicit check that the positive last bar and earlier zero bars are consistent."""
    series = _known_positive_engulfing()
    pattern = "CDLENGULFING"
    need = min_history_for(pattern)
    full_arr = evaluate_raw_series(series, pattern)
    assert int(full_arr[-1]) != 0
    eligible_zeros = [
        i for i in range(need - 1, len(series) - 1) if int(full_arr[i]) == 0
    ]
    assert eligible_zeros, "fixture should include zero-result bars for zero-path check"
    _assert_observation_causality(
        series, pattern, "talib:cdlengulfing", "cfg:causality-engulfing"
    )


def test_historical_causality_longer_lookback_patterns():
    """
    Exercise patterns whose min_history is 11–13 (doji, hammer, morning star).
    Fixture has 20 bars so valid evaluation points exist beyond the lookback.
    """
    series = _long_series(20)
    for pattern, det_id, cfg in [
        ("CDLDOJI", "talib:cdldoji", "cfg:causality-doji"),
        ("CDLHAMMER", "talib:cdlhammer", "cfg:causality-hammer"),
        ("CDLMORNINGSTAR", "talib:cdlmorningstar", "cfg:causality-morning"),
    ]:
        need = min_history_for(pattern)
        assert len(series) >= need
        _assert_observation_causality(series, pattern, det_id, cfg)


class _FakeMarket:
    def __init__(self, candles):
        self._candles = list(candles)

    def completed_candles(self, symbol, timeframe):
        return [
            c
            for c in self._candles
            if c.is_closed and c.symbol == symbol and c.timeframe == timeframe
        ]

    def series_ok(self, symbol, timeframe) -> bool:
        """Default healthy so engine series-health gate allows evaluation."""
        return True


def test_readiness_blocks_talib_adapter_via_engine():
    series = _known_positive_engulfing()
    market = _FakeMarket(series)
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    assert rm.state == ReadinessState.INITIALIZING
    assert not rm.can_evaluate()
    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:gate")
    cond = ConditionDef(
        id="ex",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(market, [det], [cond], rm)
    assert engine.evaluate_once() == []
    assert engine.last_observations == []
    assert engine.retained_observations == []


def test_readiness_ready_allows_talib_detection():
    series = _known_positive_engulfing()
    market = _FakeMarket(series)
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    rm.set_series("R_100", "1m", True)
    assert rm.state == ReadinessState.READY
    det = TalibPatternDetector("CDLENGULFING", detector_config_id="cfg:gate")
    cond = ConditionDef(
        id="ex",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    engine = ConditionEngine(market, [det], [cond], rm)
    engine.evaluate_once()
    assert len(engine.last_observations) >= 1
    assert engine.last_observations[0].payload.get("raw_talib_value") != 0
