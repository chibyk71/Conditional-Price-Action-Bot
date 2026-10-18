"""Price-action detectors: market facts only, no trade decisions."""

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.engulfing import EngulfingDetector
from cpab.detectors.talib_adapter import TalibPatternDetector, evaluate_raw, AdapterOutcomeKind
from cpab.detectors.registry import (
    CapabilityRegistry,
    build_default_registry,
    deterministic_config_id,
)
from cpab.detectors.swing import SwingPivotDetector
from cpab.detectors.equal_levels import EqualLevelsDetector
from cpab.detectors.market_structure import MarketStructureDetector
from cpab.detectors.structure_break import StructureBreakDetector
from cpab.detectors.support_resistance import SupportResistanceZoneDetector
from cpab.detectors.zone_interaction import ZoneInteractionDetector
from cpab.detectors.zone_retest import ZoneRetestDetector

__all__ = [
    "PriceActionDetector",
    "EngulfingDetector",
    "TalibPatternDetector",
    "SwingPivotDetector",
    "EqualLevelsDetector",
    "MarketStructureDetector",
    "StructureBreakDetector",
    "SupportResistanceZoneDetector",
    "ZoneInteractionDetector",
    "ZoneRetestDetector",
    "evaluate_raw",
    "AdapterOutcomeKind",
    "CapabilityRegistry",
    "build_default_registry",
    "deterministic_config_id",
]
