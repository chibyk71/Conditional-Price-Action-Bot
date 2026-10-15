from __future__ import annotations
import pytest
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.detectors.talib_adapter import TalibPatternDetector, min_history_for
from cpab.detectors.swing import SwingPivotDetector
from talib.abstract import Function


def test_create_detector_matches_talib_lookback():
    det, info = build_default_registry().create_detector("CDLENGULFING")
    assert isinstance(det, TalibPatternDetector)
    expected = int(Function("CDLENGULFING").lookback) + 1
    assert det.min_history == info.min_history == expected == min_history_for("CDLENGULFING")
    assert det.pattern == info.pattern == "CDLENGULFING"
    assert det._detector_config_id == info.detector_config_id


def test_pattern_override_rejected():
    """Registered TA-Lib pattern is fixed; a different pattern value must raise."""
    with pytest.raises(ValueError, match="override|Invalid"):
        build_default_registry().validate_config("CDLENGULFING", {"pattern": "CDLDOJI"})


def test_pattern_same_value_accepted():
    """Supplying the registered pattern value is allowed (not an override)."""
    info = build_default_registry().validate_config(
        "CDLENGULFING", {"pattern": "CDLENGULFING"}
    )
    assert info.pattern == "CDLENGULFING"
    assert info.config.get("pattern") == "CDLENGULFING"


def test_unsupported_concept_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        build_default_registry().validate_config("not_a_real_pattern_xyz", {})


def test_doji_min_history_from_talib():
    expected = int(Function("CDLDOJI").lookback) + 1
    assert build_default_registry().min_history("CDLDOJI") == expected


def test_swing_config_override_accepted():
    """Swing left/right/tie_policy are overridable (not fixed registered params)."""
    reg = build_default_registry()
    det, info = reg.create_detector(
        "swing_high",
        config={"left_strength": 2, "right_strength": 3, "tie_policy": "first"},
    )
    assert isinstance(det, SwingPivotDetector)
    assert det.left_strength == 2
    assert det.right_strength == 3
    assert det.tie_policy == "first"
    assert info.min_history == 6
    assert info.detector_config_id == deterministic_config_id(
        {"left_strength": 2, "right_strength": 3, "tie_policy": "first"}
    )


def test_swing_default_config():
    reg = build_default_registry()
    det, info = reg.create_detector("swing")
    assert isinstance(det, SwingPivotDetector)
    assert det.left_strength == 1
    assert det.right_strength == 1
    assert det.tie_policy == "strict"
    assert info.min_history == 3
