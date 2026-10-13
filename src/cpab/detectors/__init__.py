"""Price-action detectors: market facts only, no trade decisions."""

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.talib_adapter import TalibPatternDetector, evaluate_raw, AdapterOutcomeKind
from cpab.detectors.registry import (
    CapabilityRegistry,
    build_default_registry,
    deterministic_config_id,
)

__all__ = [
    "PriceActionDetector",
    "EngulfingDetector",
    "TalibPatternDetector",
    "evaluate_raw",
    "AdapterOutcomeKind",
    "CapabilityRegistry",
    "build_default_registry",
    "deterministic_config_id",
]
