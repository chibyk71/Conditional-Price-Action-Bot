from __future__ import annotations
import pytest
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.detectors.talib_adapter import TalibPatternDetector, min_history_for
from talib.abstract import Function

def test_create_detector_matches_talib_lookback():
    det, info = build_default_registry().create_detector("CDLENGULFING")
    assert isinstance(det, TalibPatternDetector)
    expected = int(Function("CDLENGULFING").lookback) + 1
    assert det.min_history == info.min_history == expected == min_history_for("CDLENGULFING")
    assert det.pattern == info.pattern == "CDLENGULFING"
    assert det._detector_config_id == info.detector_config_id

def test_pattern_override_rejected():
    with pytest.raises(ValueError, match="override|Invalid"):
        build_default_registry().validate_config("CDLENGULFING", {"pattern": "CDLDOJI"})

def test_unsupported_concept_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        build_default_registry().validate_config("not_a_real_pattern_xyz", {})

def test_doji_min_history_from_talib():
    expected = int(Function("CDLDOJI").lookback) + 1
    assert build_default_registry().min_history("CDLDOJI") == expected
