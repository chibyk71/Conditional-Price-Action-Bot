"""Phase 5A — Classical reversal pattern acceptance tests."""

from __future__ import annotations

import pytest

from cpab.detectors.classical_reversal import (
    ClassicalReversalDetector,
    CONCEPT_DOUBLE_TOP,
    CONCEPT_DOUBLE_BOTTOM,
    CONCEPT_HEAD_AND_SHOULDERS,
    CONCEPT_INVERSE_HEAD_AND_SHOULDERS,
    VARIANT,
    _similar,
    _similarity_deviation,
    DEFAULT_SIMILARITY_TOLERANCE,
    DEFAULT_MAX_BARS_TO_BREAKOUT,
)
from cpab.detectors.registry import build_default_registry
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
    symbol: str = "R_100",
    timeframe: str = "1m",
    start: int = START,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        epoch=start + i * TF,
        open=o,
        high=h,
        low=l,
        close=cl,
        is_closed=True,
        source="provider",
    )


class TestConfigValidation:
    def test_defaults(self):
        d = ClassicalReversalDetector()
        assert d.similarity_tolerance == DEFAULT_SIMILARITY_TOLERANCE
        assert d.max_bars_to_breakout == DEFAULT_MAX_BARS_TO_BREAKOUT
        assert d.left_strength == 1
        assert d.right_strength == 1
        assert d.tie_policy == "strict"
        assert d.detector_config_id.startswith("cfg:")

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError):
            ClassicalReversalDetector(similarity_tolerance=-0.1)
        with pytest.raises(ValueError):
            ClassicalReversalDetector(similarity_tolerance=float("nan"))
        with pytest.raises(ValueError):
            ClassicalReversalDetector(similarity_tolerance="0.002")  # type: ignore

    def test_invalid_max_bars(self):
        with pytest.raises(ValueError):
            ClassicalReversalDetector(max_bars_to_breakout=0)
        with pytest.raises(ValueError):
            ClassicalReversalDetector(max_bars_to_breakout=-1)
        with pytest.raises(ValueError):
            ClassicalReversalDetector(max_bars_to_breakout=1.5)  # type: ignore
        with pytest.raises(ValueError):
            ClassicalReversalDetector(max_bars_to_breakout=True)  # type: ignore

    def test_similarity_formula(self):
        assert _similar(100.0, 100.2, 0.002) is True
        assert _similar(100.0, 100.2, 0.001) is False
        assert _similar(100.0, 100.15, 0.002) is True
        assert _similar(100.0, 100.25, 0.002) is False
        dev = _similarity_deviation(100.0, 100.2)
        assert abs(dev - (0.2 / 100.2)) < 1e-15


def _double_top_series(
    *,
    peak_a: float = 110.0,
    trough_b: float = 100.0,
    peak_c: float = 110.1,
    breakout_close: float = 99.0,
    extra_after: int = 2,
) -> list[Candle]:
    mid_h = min(peak_a, peak_c) - 4.0
    mid_l = trough_b + 4.0
    candles = [
        _c(0, 105, mid_h, mid_l, 105),
        _c(1, 105, mid_h, mid_l, 105),
        _c(2, 105, peak_a, mid_l, 105),
        _c(3, 105, mid_h, mid_l, 105),
        _c(4, 105, mid_h, mid_l, 105),
        _c(5, 105, mid_h, trough_b, 105),
        _c(6, 105, mid_h, mid_l, 105),
        _c(7, 105, mid_h, mid_l, 105),
        _c(8, 105, peak_c, mid_l, 105),
        _c(9, 105, mid_h, mid_l, 105),
        _c(10, 105, mid_h, mid_l, 105),
        _c(11, trough_b + 1, trough_b + 2, breakout_close - 1, breakout_close),
    ]
    for i in range(extra_after):
        candles.append(_c(12 + i, 105, mid_h, mid_l, 105))
    return candles


class TestDoubleTop:
    def test_valid_double_top(self):
        candles = _double_top_series()
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) >= 1
        t = tops[0]
        assert t.direction == "bearish"
        assert t.variant == VARIANT
        assert t.confirmed_at is not None
        assert t.event_time == t.confirmed_at
        assert t.payload["neckline_type"] == "horizontal"
        assert t.payload["neckline_at_breakout"] == pytest.approx(100.0)
        assert len(t.reference_ids) == 3
        assert t.detector_config_id == det.detector_config_id

    def test_similarity_at_boundary(self):
        candles = _double_top_series(peak_a=110.0, peak_c=110.22)
        det = ClassicalReversalDetector(similarity_tolerance=0.002, max_bars_to_breakout=20)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) >= 1

    def test_similarity_outside(self):
        candles = _double_top_series(peak_a=110.0, peak_c=112.0)
        det = ClassicalReversalDetector(similarity_tolerance=0.002, max_bars_to_breakout=20)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) == 0

    def test_wick_only_does_not_confirm(self):
        candles = _double_top_series(breakout_close=100.5)
        candles[11] = _c(11, 101.0, 101.5, 99.0, 100.5)
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) == 0

    def test_reject_close_above_higher_peak(self):
        candles = _double_top_series(breakout_close=99.0)
        higher = max(110.0, 110.1)
        candles[10] = _c(10, higher + 0.5, higher + 1.0, higher, higher + 0.8)
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) == 0

    def test_peaks_must_be_above_trough(self):
        candles = _double_top_series(peak_a=99.0, trough_b=100.0, peak_c=99.5)
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) == 0


def _double_bottom_series(
    *,
    trough_a: float = 90.0,
    peak_b: float = 100.0,
    trough_c: float = 90.1,
    breakout_close: float = 101.0,
    extra_after: int = 2,
) -> list[Candle]:
    mid_h = peak_b - 4.0
    mid_l = max(trough_a, trough_c) + 4.0
    candles = [
        _c(0, 95, mid_h, mid_l, 95),
        _c(1, 95, mid_h, mid_l, 95),
        _c(2, 95, mid_h, trough_a, 95),
        _c(3, 95, mid_h, mid_l, 95),
        _c(4, 95, mid_h, mid_l, 95),
        _c(5, 95, peak_b, mid_l, 95),
        _c(6, 95, mid_h, mid_l, 95),
        _c(7, 95, mid_h, mid_l, 95),
        _c(8, 95, mid_h, trough_c, 95),
        _c(9, 95, mid_h, mid_l, 95),
        _c(10, 95, mid_h, mid_l, 95),
        _c(11, peak_b - 1, breakout_close + 0.5, peak_b - 2, breakout_close),
    ]
    for i in range(extra_after):
        candles.append(_c(12 + i, 95, mid_h, mid_l, 95))
    return candles


class TestDoubleBottom:
    def test_valid_double_bottom(self):
        candles = _double_bottom_series()
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        bottoms = [
            o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_BOTTOM
        ]
        assert len(bottoms) >= 1
        b = bottoms[0]
        assert b.direction == "bullish"
        assert b.variant == VARIANT
        assert b.payload["neckline_at_breakout"] == pytest.approx(100.0)

    def test_wick_only_does_not_confirm(self):
        candles = _double_bottom_series(breakout_close=99.0)
        candles[11] = _c(11, 99.5, 101.5, 99.0, 99.0)
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        bottoms = [
            o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_BOTTOM
        ]
        assert len(bottoms) == 0


def _hs_series(
    *,
    shoulder_a: float = 108.0,
    trough_b: float = 100.0,
    head: float = 115.0,
    trough_d: float = 101.0,
    shoulder_e: float = 108.15,
    breakout_close: float = 99.0,
) -> list[Candle]:
    mid_h = min(shoulder_a, shoulder_e) - 3.0
    mid_l = max(trough_b, trough_d) + 3.0
    candles = [
        _c(0, 105, mid_h, mid_l, 105),
        _c(1, 105, mid_h, mid_l, 105),
        _c(2, 105, shoulder_a, mid_l, 105),
        _c(3, 105, mid_h, mid_l, 105),
        _c(4, 105, mid_h, mid_l, 105),
        _c(5, 105, mid_h, trough_b, 105),
        _c(6, 105, mid_h, mid_l, 105),
        _c(7, 105, mid_h, mid_l, 105),
        _c(8, 105, head, mid_l, 105),
        _c(9, 105, mid_h, mid_l, 105),
        _c(10, 105, mid_h, mid_l, 105),
        _c(11, 105, mid_h, trough_d, 105),
        _c(12, 105, mid_h, mid_l, 105),
        _c(13, 105, mid_h, mid_l, 105),
        _c(14, 105, shoulder_e, mid_l, 105),
        _c(15, 105, mid_h, mid_l, 105),
        _c(16, 105, mid_h, mid_l, 105),
        _c(17, 100.5, 101.0, breakout_close - 0.5, breakout_close),
        _c(18, 105, mid_h, mid_l, 105),
        _c(19, 105, mid_h, mid_l, 105),
    ]
    return candles


class TestHeadAndShoulders:
    def test_valid_hs(self):
        candles = _hs_series()
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        hs = [
            o
            for o in det.detect_all_historical(candles)
            if o.type == CONCEPT_HEAD_AND_SHOULDERS
        ]
        assert len(hs) >= 1
        h = hs[0]
        assert h.direction == "bearish"
        assert h.variant == VARIANT
        assert h.payload["neckline_type"] == "sloped"
        assert "neckline_at_breakout" in h.payload
        assert len(h.reference_ids) == 5

    def test_rising_neckline(self):
        candles = _hs_series(trough_b=99.0, trough_d=102.0, breakout_close=98.0)
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        hs = [
            o
            for o in det.detect_all_historical(candles)
            if o.type == CONCEPT_HEAD_AND_SHOULDERS
        ]
        assert len(hs) >= 1

    def test_falling_neckline(self):
        candles = _hs_series(trough_b=102.0, trough_d=99.0, breakout_close=95.0)
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        hs = [
            o
            for o in det.detect_all_historical(candles)
            if o.type == CONCEPT_HEAD_AND_SHOULDERS
        ]
        assert len(hs) >= 1

    def test_head_not_higher_rejects(self):
        candles = _hs_series(shoulder_a=110.0, head=109.0, shoulder_e=110.0)
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        hs = [
            o
            for o in det.detect_all_historical(candles)
            if o.type == CONCEPT_HEAD_AND_SHOULDERS
        ]
        assert len(hs) == 0

    def test_shoulders_not_similar_rejects(self):
        candles = _hs_series(shoulder_a=108.0, shoulder_e=120.0)
        det = ClassicalReversalDetector(similarity_tolerance=0.002, max_bars_to_breakout=20)
        hs = [
            o
            for o in det.detect_all_historical(candles)
            if o.type == CONCEPT_HEAD_AND_SHOULDERS
        ]
        assert len(hs) == 0

    def test_close_above_head_rejects(self):
        candles = _hs_series(breakout_close=99.0)
        candles[16] = _c(16, 116.0, 117.0, 115.5, 116.5)
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        hs = [
            o
            for o in det.detect_all_historical(candles)
            if o.type == CONCEPT_HEAD_AND_SHOULDERS
        ]
        assert len(hs) == 0


def _ihs_series(
    *,
    shoulder_a: float = 92.0,
    peak_b: float = 100.0,
    head: float = 85.0,
    peak_d: float = 101.0,
    shoulder_e: float = 92.15,
    breakout_close: float = 103.0,
) -> list[Candle]:
    mid_h = min(peak_b, peak_d) - 3.0
    mid_l = max(shoulder_a, shoulder_e) + 3.0
    candles = [
        _c(0, 95, mid_h, mid_l, 95),
        _c(1, 95, mid_h, mid_l, 95),
        _c(2, 95, mid_h, shoulder_a, 95),
        _c(3, 95, mid_h, mid_l, 95),
        _c(4, 95, mid_h, mid_l, 95),
        _c(5, 95, peak_b, mid_l, 95),
        _c(6, 95, mid_h, mid_l, 95),
        _c(7, 95, mid_h, mid_l, 95),
        _c(8, 95, mid_h, head, 95),
        _c(9, 95, mid_h, mid_l, 95),
        _c(10, 95, mid_h, mid_l, 95),
        _c(11, 95, peak_d, mid_l, 95),
        _c(12, 95, mid_h, mid_l, 95),
        _c(13, 95, mid_h, mid_l, 95),
        _c(14, 95, mid_h, shoulder_e, 95),
        _c(15, 95, mid_h, mid_l, 95),
        _c(16, 95, mid_h, mid_l, 95),
        _c(17, 100.5, breakout_close + 0.5, 100.0, breakout_close),
        _c(18, 95, mid_h, mid_l, 95),
        _c(19, 95, mid_h, mid_l, 95),
    ]
    return candles


class TestInverseHeadAndShoulders:
    def test_valid_ihs(self):
        candles = _ihs_series()
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        ihs = [
            o
            for o in det.detect_all_historical(candles)
            if o.type == CONCEPT_INVERSE_HEAD_AND_SHOULDERS
        ]
        assert len(ihs) >= 1
        h = ihs[0]
        assert h.direction == "bullish"
        assert h.variant == VARIANT
        assert h.payload["neckline_type"] == "sloped"
        assert len(h.reference_ids) == 5


class TestCausalityAndExpiry:
    def test_confirmation_timing(self):
        candles = _double_top_series()
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) >= 1
        assert tops[0].confirmed_at == candles[11].epoch
        assert tops[0].event_time == candles[11].epoch

    def test_expiry_boundary(self):
        candles = _double_top_series(extra_after=5)
        det = ClassicalReversalDetector(max_bars_to_breakout=1)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) == 0

    def test_gap_rejects_series(self):
        candles = _double_top_series()
        c = candles[6]
        candles[6] = Candle(
            symbol=c.symbol,
            timeframe=c.timeframe,
            epoch=c.epoch + 5 * TF,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            is_closed=True,
            source=c.source,
        )
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        assert det.detect_all_historical(candles) == []

    def test_incremental_matches_historical_on_latest(self):
        candles = _double_top_series()
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        hist = det.detect_all_historical(candles)
        incr = det.detect(candles)
        latest = candles[-1].epoch
        hist_latest = [o for o in hist if o.confirmed_at == latest]
        assert len(incr) == len(hist_latest)
        assert {o.id for o in incr} == {o.id for o in hist_latest}

    def test_stable_ids_and_idempotency(self):
        candles = _double_top_series()
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        a = det.detect_all_historical(candles)
        b = det.detect_all_historical(candles)
        assert [o.id for o in a] == [o.id for o in b]
        tops = [o for o in a if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) >= 1
        assert tops[0].id.startswith("double_top|")

    def test_duplicate_anchors_deduped(self):
        candles = _double_top_series()
        det = ClassicalReversalDetector(max_bars_to_breakout=20)
        tops = [o for o in det.detect_all_historical(candles) if o.type == CONCEPT_DOUBLE_TOP]
        ids = [o.id for o in tops]
        assert len(ids) == len(set(ids))


class TestRegistry:
    @pytest.fixture(autouse=True)
    def _require_talib(self):
        pytest.importorskip("talib")

    def test_concepts_registered(self):
        reg = build_default_registry()
        for concept in (
            "double_top",
            "double_bottom",
            "head_and_shoulders",
            "inverse_head_and_shoulders",
            "classical_reversal",
        ):
            assert reg.is_supported(concept)

    def test_create_detector_defaults(self):
        reg = build_default_registry()
        det, info = reg.create_detector("double_top")
        assert isinstance(det, ClassicalReversalDetector)
        assert info.implementation == "classical_reversal"
        assert info.config["similarity_tolerance"] == DEFAULT_SIMILARITY_TOLERANCE
        assert info.config["max_bars_to_breakout"] == DEFAULT_MAX_BARS_TO_BREAKOUT
        assert info.min_history >= 16

    def test_create_with_custom_config(self):
        reg = build_default_registry()
        det, info = reg.create_detector(
            "head_and_shoulders",
            config={"similarity_tolerance": 0.01, "max_bars_to_breakout": 30},
        )
        assert det.similarity_tolerance == 0.01
        assert det.max_bars_to_breakout == 30
        assert info.detector_config_id == det.detector_config_id

    def test_invalid_config_rejected(self):
        reg = build_default_registry()
        with pytest.raises(ValueError):
            reg.create_detector("double_top", config={"similarity_tolerance": -1})
        with pytest.raises(ValueError):
            reg.create_detector("double_top", config={"max_bars_to_breakout": 0})
        with pytest.raises(ValueError):
            reg.create_detector("double_top", config={"unknown_param": 1})

    def test_detector_emits_via_registry(self):
        reg = build_default_registry()
        det, _ = reg.create_detector("double_top", config={"max_bars_to_breakout": 20})
        candles = _double_top_series()
        obs = det.detect_all_historical(candles)
        tops = [o for o in obs if o.type == CONCEPT_DOUBLE_TOP]
        assert len(tops) >= 1
