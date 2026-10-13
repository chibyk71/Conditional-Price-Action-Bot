"""Capability registry and deterministic detector_config_id."""
from __future__ import annotations
import pytest
from cpab.detectors.registry import CapabilityRegistry, build_default_registry, deterministic_config_id
from cpab.detectors.talib_adapter import TalibPatternDetector

def test_deterministic_config_id_order_independent():
    a = deterministic_config_id({"pattern": "CDLENGULFING", "x": 1})
    b = deterministic_config_id({"x": 1, "pattern": "CDLENGULFING"})
    assert a == b and a.startswith("cfg:")

def test_different_configs_different_ids():
    assert deterministic_config_id({"pattern": "CDLENGULFING"}) != deterministic_config_id({"pattern": "CDLDOJI"})

def test_defaults_consistent():
    assert deterministic_config_id({}) == deterministic_config_id({})

def test_supported_concepts():
    concepts = build_default_registry().supported_concepts()
    assert "engulfing" in concepts and "bullish_engulfing" in concepts

def test_unsupported_concept_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        build_default_registry().validate_config("not_a_real_pattern_xyz", {})

def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="unknown|Invalid"):
        build_default_registry().validate_config("engulfing", {"bogus_param": 123})

def test_min_history_reported():
    reg = build_default_registry()
    assert reg.min_history("engulfing") == 2
    assert reg.min_history("CDLENGULFING") >= 2

def test_create_detector_talib():
    det, info = build_default_registry().create_detector("CDLENGULFING")
    assert isinstance(det, TalibPatternDetector)
    assert info.min_history >= 2 and info.detector_config_id.startswith("cfg:")
    assert info.implementation == "talib"

def test_no_silent_substitution():
    with pytest.raises(ValueError):
        build_default_registry().create_detector("completely_unknown_concept")

def test_equivalent_config_same_id_via_registry():
    reg = build_default_registry()
    i1 = reg.validate_config("CDLENGULFING", {"pattern": "CDLENGULFING"})
    i2 = reg.validate_config("CDLENGULFING", {"pattern": "CDLENGULFING"})
    assert i1.detector_config_id == i2.detector_config_id
