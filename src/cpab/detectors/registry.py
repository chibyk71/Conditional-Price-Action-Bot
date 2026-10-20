"""
Minimal detector capability registry for CPAB Phase 1+.

Registered capabilities have deterministic configuration identity.
Unsupported patterns / invalid params are rejected (no silent substitution).

Phase 2A: swing_pivot accepts left_strength, right_strength, tie_policy as
overridable configuration. TA-Lib pattern values remain fixed at registration.

Phase 2B: equal_levels accepts tolerance (required), left_strength, right_strength,
tie_policy. Tolerance has no silent default.

Phase 2C: market_structure accepts left_strength, right_strength, tie_policy
(forwarded to internal SwingPivotDetector).

Phase 2D: structure_break accepts left_strength, right_strength, tie_policy
(forwarded to internal SwingPivotDetector for breakable levels).

Phase 3A: support_resistance_zone accepts left_strength, right_strength, tie_policy
(forwarded to internal SwingPivotDetector for zone source swings).

Phase 3B: zone_interaction accepts left_strength, right_strength, tie_policy
(forwarded to internal SupportResistanceZoneDetector for source zones).

Phase 3C: zone_retest accepts left_strength, right_strength, tie_policy
(forwarded to internal SupportResistanceZoneDetector / ZoneInteractionDetector).

Phase 3D: zone_polarity_flip accepts left_strength, right_strength, tie_policy
(forwarded to internal ZoneInteractionDetector / SupportResistanceZoneDetector).

Phase 4A: fair_value_gap — no configuration parameters (geometric formation only).

Phase 4B: fair_value_gap_lifecycle — no configuration parameters; composes Phase 4A FVG.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.talib_adapter import (
    TalibPatternDetector,
    min_history_for,
    resolve_pattern_name,
    _PATTERN_ALIASES,
    _SUPPORTED_PATTERNS,
)


def deterministic_config_id(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"cfg:{digest}"


@dataclass(frozen=True)
class CapabilityInfo:
    concept: str
    detector_id: str
    detector_config_id: str
    min_history: int
    pattern: Optional[str] = None
    implementation: str = ""
    config: dict[str, Any] = field(default_factory=dict)


# Factory may be zero-arg or accept a single config dict.
DetectorFactory = Callable[..., PriceActionDetector]


def _factory_accepts_config(factory: DetectorFactory) -> bool:
    """
    Inspect the factory signature to decide whether to pass effective config.

    Avoids catching TypeError from inside the factory body, which would mask
    real construction failures.
    """
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if not params:
        return False
    first = params[0]
    if first.kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ):
        return False
    return True


@dataclass
class CapabilityRegistry:
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    _concept_index: dict[str, list[str]] = field(default_factory=dict)

    def register(
        self,
        detector_id: str,
        *,
        concepts: list[str],
        factory: DetectorFactory,
        config: Optional[dict[str, Any]] = None,
        accepted_params: Optional[list[str]] = None,
        min_history: Optional[int] = None,
        implementation: str = "",
        pattern: Optional[str] = None,
    ) -> str:
        cfg = dict(config or {})
        if accepted_params is not None:
            unknown = set(cfg) - set(accepted_params)
            if unknown:
                raise ValueError(
                    f"Invalid configuration for {detector_id}: unknown params {sorted(unknown)}"
                )
        config_id = deterministic_config_id(cfg)
        self._entries[detector_id] = {
            "detector_id": detector_id,
            "concepts": list(concepts),
            "factory": factory,
            "config": cfg,
            "detector_config_id": config_id,
            "accepted_params": list(accepted_params) if accepted_params is not None else None,
            "min_history": min_history,
            "implementation": implementation,
            "pattern": pattern,
        }
        for concept in concepts:
            self._concept_index.setdefault(concept, [])
            if detector_id not in self._concept_index[concept]:
                self._concept_index[concept].append(detector_id)
        return config_id

    def supported_concepts(self) -> list[str]:
        return sorted(self._concept_index.keys())

    def is_supported(self, concept: str, *, config: Optional[dict[str, Any]] = None) -> bool:
        if concept not in self._concept_index:
            return False
        if config is None:
            return True
        try:
            self.validate_config(concept, config)
            return True
        except ValueError:
            return False

    def validate_config(self, concept: str, config: dict[str, Any]) -> CapabilityInfo:
        """
        Validate user config against registered detectors for ``concept``.

        Semantics:
          - ``accepted_params`` (when set) is the whitelist of allowed keys.
          - Keys present in the registered ``config`` are **fixed**: supplying a
            different value raises ValueError (cannot override).
          - Keys in ``accepted_params`` but not fixed in registered config may be
            supplied by the caller and become part of the effective config.
          - Swing / equal_levels / market_structure / structure_break /
            support_resistance_zone / zone_interaction / zone_retest apply documented defaults
            for missing overridable keys (except equal_levels tolerance, which
            is required).
        """
        if concept not in self._concept_index:
            raise ValueError(f"Unsupported concept: {concept!r}")
        detector_ids = self._concept_index[concept]
        last_error: Optional[str] = None
        for did in detector_ids:
            entry = self._entries[did]
            accepted = entry.get("accepted_params")
            registered = dict(entry["config"])

            if accepted is not None:
                unknown = set(config) - set(accepted)
                if unknown:
                    last_error = (
                        f"Invalid configuration for concept {concept!r}: "
                        f"unknown params {sorted(unknown)}"
                    )
                    continue

            conflict = False
            for key, val in config.items():
                if key in registered and registered[key] != val:
                    last_error = (
                        f"Invalid configuration for concept {concept!r}: "
                        f"cannot override fixed parameter {key!r} "
                        f"(registered={registered[key]!r}, got={val!r})"
                    )
                    conflict = True
                    break
            if conflict:
                continue

            effective = dict(registered)
            if accepted is not None:
                accepted_set = set(accepted)
                for key, val in config.items():
                    if key in accepted_set and key not in registered:
                        effective[key] = val

            impl = entry.get("implementation") or ""

            if impl == "swing_pivot":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy

                defaults = {
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                mh = left + 1 + right
            elif impl == "equal_levels":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy
                from cpab.detectors.equal_levels import _validate_tolerance

                if "tolerance" not in config and "tolerance" not in effective:
                    last_error = (
                        f"Invalid configuration for concept {concept!r}: "
                        f"tolerance is required"
                    )
                    continue
                tol_raw = config.get("tolerance", effective.get("tolerance"))
                try:
                    tol = _validate_tolerance(tol_raw)
                except ValueError as e:
                    last_error = str(e)
                    continue

                defaults = {
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted and key != "tolerance":
                            merged[key] = val
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "tolerance": tol,
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                s = left + 1 + right
                mh = 2 * s - 1
            elif impl == "market_structure":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy

                defaults = {
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                mh = left + 2 * right + 2
            elif impl == "structure_break":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy

                defaults = {
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                mh = left + 1 + right + 1
            elif impl == "support_resistance_zone":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy

                defaults = {
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                mh = left + 1 + right
            elif impl == "zone_interaction":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy

                defaults = {
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                mh = left + 1 + right + 1
            elif impl == "zone_retest":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy

                defaults = {
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                # zone_interaction min_history + 1 post-break candle
                mh = left + 1 + right + 1 + 1
            elif impl == "zone_polarity_flip":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy

                defaults = {
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                # Same bound as zone_interaction (zone + prior/current pair).
                mh = left + 1 + right + 1
            elif impl == "fair_value_gap":
                # Phase 4A: no configuration; geometric three-candle formation only.
                effective = {}
                mh = 3
            elif impl == "fair_value_gap_lifecycle":
                # Phase 4B: no configuration; composition of Phase 4A + post-formation bars.
                effective = {}
                mh = 4
            else:
                mh = entry["min_history"]
                if mh is None and entry.get("pattern"):
                    mh = min_history_for(entry["pattern"])

            config_id = deterministic_config_id(effective)
            return CapabilityInfo(
                concept=concept,
                detector_id=did,
                detector_config_id=config_id,
                min_history=mh if mh is not None else 2,
                pattern=entry.get("pattern"),
                implementation=entry.get("implementation") or "",
                config=effective,
            )
        if last_error:
            raise ValueError(last_error)
        raise ValueError(f"No valid detector configuration for concept {concept!r}")

    def min_history(self, concept: str, *, config: Optional[dict[str, Any]] = None) -> int:
        info = self.validate_config(concept, config or {})
        return info.min_history

    def create_detector(
        self,
        concept: str,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> tuple[PriceActionDetector, CapabilityInfo]:
        info = self.validate_config(concept, config or {})
        entry = self._entries[info.detector_id]
        factory = entry["factory"]
        effective = info.config

        if _factory_accepts_config(factory):
            det = factory(effective)
        else:
            det = factory()

        if isinstance(det, TalibPatternDetector):
            det._detector_config_id = info.detector_config_id
            if info.pattern and det.pattern != info.pattern:
                raise RuntimeError(
                    f"Factory pattern mismatch: detector has {det.pattern!r}, "
                    f"registry expects {info.pattern!r}"
                )
        if hasattr(det, "_detector_config_id") and getattr(det, "_detector_config_id") is None:
            det._detector_config_id = info.detector_config_id  # type: ignore[attr-defined]
        if hasattr(det, "_detector_config_id"):
            det._detector_config_id = info.detector_config_id  # type: ignore[attr-defined]
        return det, info

    def get_entry(self, detector_id: str) -> Optional[dict[str, Any]]:
        return self._entries.get(detector_id)


def build_default_registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()

    def _engulfing_factory(_cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.engulfing import EngulfingDetector

        return EngulfingDetector()

    reg.register(
        "engulfing",
        concepts=["bullish_engulfing", "bearish_engulfing", "engulfing"],
        factory=_engulfing_factory,
        config={},
        accepted_params=[],
        min_history=2,
        implementation="engulfing",
    )

    for alias, cdl_name in _PATTERN_ALIASES.items():
        pattern = cdl_name
        concepts = [alias, pattern.lower(), pattern]
        base = pattern.lower().replace("cdl", "")
        concepts.extend([f"bullish_{base}", f"bearish_{base}"])
        mh = min_history_for(pattern)

        def _make_factory(p: str = pattern) -> DetectorFactory:
            def factory(_cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
                return TalibPatternDetector(pattern=p, detector_id=f"talib:{p.lower()}")

            return factory

        reg.register(
            f"talib:{pattern.lower()}",
            concepts=list(dict.fromkeys(concepts)),
            factory=_make_factory(),
            config={"pattern": pattern},
            accepted_params=["pattern"],
            min_history=mh,
            implementation="talib",
            pattern=pattern,
        )

    def _swing_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.swing import SwingPivotDetector

        c = dict(cfg or {})
        return SwingPivotDetector(
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="swing_pivot",
        )

    reg.register(
        "swing_pivot",
        concepts=["swing_high", "swing_low", "swing"],
        factory=_swing_factory,
        config={},
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=3,
        implementation="swing_pivot",
    )

    def _equal_levels_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.equal_levels import EqualLevelsDetector

        c = dict(cfg or {})
        if "tolerance" not in c:
            raise ValueError("tolerance is required for equal_levels")
        return EqualLevelsDetector(
            tolerance=c["tolerance"],
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="equal_levels",
        )

    reg.register(
        "equal_levels",
        concepts=["equal_high", "equal_low", "equal_level", "eqh", "eql"],
        factory=_equal_levels_factory,
        config={},
        accepted_params=["tolerance", "left_strength", "right_strength", "tie_policy"],
        min_history=5,
        implementation="equal_levels",
    )

    def _market_structure_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.market_structure import MarketStructureDetector

        c = dict(cfg or {})
        return MarketStructureDetector(
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="market_structure",
        )

    reg.register(
        "market_structure",
        concepts=[
            "market_structure",
            "hh",
            "hl",
            "lh",
            "ll",
            "higher_high",
            "higher_low",
            "lower_high",
            "lower_low",
        ],
        factory=_market_structure_factory,
        config={},
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=5,
        implementation="market_structure",
    )

    def _structure_break_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.structure_break import StructureBreakDetector

        c = dict(cfg or {})
        return StructureBreakDetector(
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="structure_break",
        )

    reg.register(
        "structure_break",
        concepts=[
            "structure_break",
            "bos",
            "choch",
            "break_of_structure",
            "change_of_character",
        ],
        factory=_structure_break_factory,
        config={},
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=4,
        implementation="structure_break",
    )

    def _sr_zone_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.support_resistance import SupportResistanceZoneDetector

        c = dict(cfg or {})
        return SupportResistanceZoneDetector(
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="support_resistance_zone",
        )

    reg.register(
        "support_resistance_zone",
        concepts=[
            "support_resistance_zone",
            "sr_zone",
        ],
        factory=_sr_zone_factory,
        config={},
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=3,
        implementation="support_resistance_zone",
    )

    def _zone_interaction_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.zone_interaction import ZoneInteractionDetector

        c = dict(cfg or {})
        return ZoneInteractionDetector(
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="zone_interaction",
        )

    reg.register(
        "zone_interaction",
        concepts=[
            "zone_interaction",
            "zone_touch",
            "zone_rejection",
            "zone_break",
        ],
        factory=_zone_interaction_factory,
        config={},
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=4,
        implementation="zone_interaction",
    )

    def _zone_retest_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.zone_retest import ZoneRetestDetector

        c = dict(cfg or {})
        return ZoneRetestDetector(
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="zone_retest",
        )

    reg.register(
        "zone_retest",
        concepts=[
            "zone_retest",
        ],
        factory=_zone_retest_factory,
        config={},
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=5,
        implementation="zone_retest",
    )

    def _zone_polarity_flip_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.zone_polarity_flip import ZonePolarityFlipDetector

        c = dict(cfg or {})
        return ZonePolarityFlipDetector(
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="zone_polarity_flip",
        )

    reg.register(
        "zone_polarity_flip",
        concepts=[
            "zone_polarity_flip",
        ],
        factory=_zone_polarity_flip_factory,
        config={},
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=4,
        implementation="zone_polarity_flip",
    )

    def _fair_value_gap_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.fair_value_gap import FairValueGapDetector

        # Phase 4A: no accepted configuration parameters.
        return FairValueGapDetector(detector_id="fair_value_gap")

    reg.register(
        "fair_value_gap",
        concepts=[
            "fair_value_gap",
            "fvg",
        ],
        factory=_fair_value_gap_factory,
        config={},
        accepted_params=[],
        min_history=3,
        implementation="fair_value_gap",
    )

    def _fair_value_gap_lifecycle_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.fair_value_gap_lifecycle import FairValueGapLifecycleDetector

        return FairValueGapLifecycleDetector(detector_id="fair_value_gap_lifecycle")

    reg.register(
        "fair_value_gap_lifecycle",
        concepts=[
            "fair_value_gap_lifecycle",
            "fvg_lifecycle",
        ],
        factory=_fair_value_gap_lifecycle_factory,
        config={},
        accepted_params=[],
        min_history=4,
        implementation="fair_value_gap_lifecycle",
    )

    return reg
