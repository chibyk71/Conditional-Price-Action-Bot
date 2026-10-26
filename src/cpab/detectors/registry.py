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

Phase 4C: inverse_fair_value_gap — no configuration; composes Phase 4B invalidations.

Phase 4D: order_block — min_body_ratio (default 0.5) plus left/right_strength,
tie_policy forwarded to StructureBreakDetector; composes confirmed BOS only.

Phase 4E: breaker_block — same OB config forwarded; composes confirmed Order
Blocks; emits on strict close-through of the OB zone (no retest/BOS required).

Phase 4F: mitigation_block — same OB/breaker config forwarded; tracks progressive
wick-based touch / partial_fill / full_fill against confirmed Order Blocks and
Breaker Blocks. Does not mutate sources.

Phase 5A: classical_reversal — similarity_tolerance (default 0.002),
max_bars_to_breakout (default 50), plus left/right_strength and tie_policy
forwarded to SwingPivotDetector. Emits double_top, double_bottom,
head_and_shoulders, inverse_head_and_shoulders.

Phase 5B: rectangle — boundary_tolerance (default 0.002),
max_bars_to_breakout (default 50), plus left/right_strength and tie_policy.

Phase 5C: triangle — boundary_tolerance (default 0.002),
horizontal_tolerance (default 0.002), max_bars_to_breakout (default 50),
plus left/right_strength and tie_policy. Emits ascending_triangle,
descending_triangle, symmetrical_triangle.

Phase 5D: flags_pennants — pole (min/max bars, ATR lookback, strength mult,
max retracement), consolidation (min/max bars), boundary_tolerance,
parallel_slope_tolerance, max_bars_to_breakout, plus left/right_strength
and tie_policy. Emits bullish_flag, bearish_flag, bullish_pennant,
bearish_pennant.

Phase 6A: atr_expansion, atr_contraction, range_expansion — locked ATR period 14,
baseline mean of preceding 20 rolling ATRs; expansion 1.25× / contraction 0.75×
crossing-only; range expansion TR ≥ 1.5× preceding ATR (every qualifying bar).
Phase 6B: impulse — body/preceding ATR ≥ 1.0, TR/preceding ATR ≥ 1.5,
body/range ≥ 0.60; emits bullish/bearish observations for every qualifying bar.
Warm-up 35 / 35 / 16 / 16. Variants rolling_atr_v1, true_range_v1, and
atr_normalized_candle_v1. No lifecycle.

Phase 6C: pullback — detector_id pullback, variants structure_based and impulse_based;
tracks pullback start, continuation, and termination (trend_resumption, structural_failure, invalidation).

Phase 6D: breakout_behavior — confirmed swing and zero-width S/R boundaries;
strict causal confirmation; close_back_across or confirmed_close_back_across failure.
Phase 6E: optional enable_follow_through (default False) emits breakout_follow_through
after a later completed candle closes strictly beyond the confirmation candle high/low.
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
            elif impl == "inverse_fair_value_gap":
                # Phase 4C: no configuration; maps Phase 4B invalidations to IFVG.
                effective = {}
                mh = 4
            elif impl == "classical_reversal":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy
                from cpab.detectors.classical_reversal import (
                    _validate_similarity_tolerance,
                    _validate_max_bars_to_breakout,
                    DEFAULT_SIMILARITY_TOLERANCE,
                    DEFAULT_MAX_BARS_TO_BREAKOUT,
                )

                defaults = {
                    "similarity_tolerance": DEFAULT_SIMILARITY_TOLERANCE,
                    "max_bars_to_breakout": DEFAULT_MAX_BARS_TO_BREAKOUT,
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
                tol = _validate_similarity_tolerance(
                    merged.get("similarity_tolerance", DEFAULT_SIMILARITY_TOLERANCE)
                )
                max_bars = _validate_max_bars_to_breakout(
                    merged.get("max_bars_to_breakout", DEFAULT_MAX_BARS_TO_BREAKOUT)
                )
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "similarity_tolerance": tol,
                    "max_bars_to_breakout": max_bars,
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                s = left + 1 + right
                mh = 5 * s + 1
            elif impl == "rectangle":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy
                from cpab.detectors.rectangle import (
                    _validate_boundary_tolerance,
                    _validate_max_bars_to_breakout,
                    DEFAULT_BOUNDARY_TOLERANCE,
                    DEFAULT_MAX_BARS_TO_BREAKOUT,
                )

                defaults = {
                    "boundary_tolerance": DEFAULT_BOUNDARY_TOLERANCE,
                    "max_bars_to_breakout": DEFAULT_MAX_BARS_TO_BREAKOUT,
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
                tol = _validate_boundary_tolerance(
                    merged.get("boundary_tolerance", DEFAULT_BOUNDARY_TOLERANCE)
                )
                max_bars = _validate_max_bars_to_breakout(
                    merged.get("max_bars_to_breakout", DEFAULT_MAX_BARS_TO_BREAKOUT)
                )
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "boundary_tolerance": tol,
                    "max_bars_to_breakout": max_bars,
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                s = left + 1 + right
                mh = 4 * s + 1
            elif impl == "triangle":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy
                from cpab.detectors.triangle import (
                    _validate_tolerance,
                    _validate_max_bars_to_breakout,
                    DEFAULT_BOUNDARY_TOLERANCE,
                    DEFAULT_HORIZONTAL_TOLERANCE,
                    DEFAULT_MAX_BARS_TO_BREAKOUT,
                )

                defaults = {
                    "boundary_tolerance": DEFAULT_BOUNDARY_TOLERANCE,
                    "horizontal_tolerance": DEFAULT_HORIZONTAL_TOLERANCE,
                    "max_bars_to_breakout": DEFAULT_MAX_BARS_TO_BREAKOUT,
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
                btol = _validate_tolerance(
                    "boundary_tolerance",
                    merged.get("boundary_tolerance", DEFAULT_BOUNDARY_TOLERANCE),
                )
                htol = _validate_tolerance(
                    "horizontal_tolerance",
                    merged.get("horizontal_tolerance", DEFAULT_HORIZONTAL_TOLERANCE),
                )
                max_bars = _validate_max_bars_to_breakout(
                    merged.get("max_bars_to_breakout", DEFAULT_MAX_BARS_TO_BREAKOUT)
                )
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength(
                    "right_strength", merged.get("right_strength", 1)
                )
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "boundary_tolerance": btol,
                    "horizontal_tolerance": htol,
                    "max_bars_to_breakout": max_bars,
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                s = left + 1 + right
                mh = 4 * s + 1
            elif impl == "flags_pennants":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy
                from cpab.detectors.flags_pennants import (
                    DEFAULT_MIN_POLE_BARS,
                    DEFAULT_MAX_POLE_BARS,
                    DEFAULT_POLE_ATR_LOOKBACK,
                    DEFAULT_POLE_STRENGTH_MULT,
                    DEFAULT_MAX_POLE_RETRACEMENT,
                    DEFAULT_MIN_CONSOLIDATION_BARS,
                    DEFAULT_MAX_CONSOLIDATION_BARS,
                    DEFAULT_BOUNDARY_TOLERANCE,
                    DEFAULT_PARALLEL_SLOPE_TOLERANCE,
                    DEFAULT_MAX_BARS_TO_BREAKOUT,
                )

                defaults = {
                    "min_pole_bars": DEFAULT_MIN_POLE_BARS,
                    "max_pole_bars": DEFAULT_MAX_POLE_BARS,
                    "pole_atr_lookback": DEFAULT_POLE_ATR_LOOKBACK,
                    "pole_strength_mult": DEFAULT_POLE_STRENGTH_MULT,
                    "max_pole_retracement": DEFAULT_MAX_POLE_RETRACEMENT,
                    "min_consolidation_bars": DEFAULT_MIN_CONSOLIDATION_BARS,
                    "max_consolidation_bars": DEFAULT_MAX_CONSOLIDATION_BARS,
                    "boundary_tolerance": DEFAULT_BOUNDARY_TOLERANCE,
                    "parallel_slope_tolerance": DEFAULT_PARALLEL_SLOPE_TOLERANCE,
                    "max_bars_to_breakout": DEFAULT_MAX_BARS_TO_BREAKOUT,
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
                    "min_pole_bars": int(merged["min_pole_bars"]),
                    "max_pole_bars": int(merged["max_pole_bars"]),
                    "pole_atr_lookback": int(merged["pole_atr_lookback"]),
                    "pole_strength_mult": float(merged["pole_strength_mult"]),
                    "max_pole_retracement": float(merged["max_pole_retracement"]),
                    "min_consolidation_bars": int(merged["min_consolidation_bars"]),
                    "max_consolidation_bars": int(merged["max_consolidation_bars"]),
                    "boundary_tolerance": float(merged["boundary_tolerance"]),
                    "parallel_slope_tolerance": float(
                        merged["parallel_slope_tolerance"]
                    ),
                    "max_bars_to_breakout": int(merged["max_bars_to_breakout"]),
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                # Align with FlagsPennantsDetector.min_history: use *min*
                # pole/consolidation durations. max_bars_to_breakout is a
                # waiting window, not required history before detection can start.
                mh = (
                    int(merged["pole_atr_lookback"])
                    + 1
                    + int(merged["min_pole_bars"])
                    + int(merged["min_consolidation_bars"])
                    + left
                    + right
                    + 4
                )
            elif impl == "breakout_behavior":
                from cpab.detectors.breakout_behavior import (
                    _validate_enable_follow_through,
                    _validate_failure_policy,
                )
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy

                defaults = {
                    "failure_policy": "close_back_across",
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                    "enable_follow_through": False,
                }
                merged = dict(defaults)
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                failure_policy = _validate_failure_policy(merged.get("failure_policy"))
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength("right_strength", merged.get("right_strength", 1))
                tie = _validate_tie_policy(merged.get("tie_policy", "strict"))
                enable_ft = _validate_enable_follow_through(
                    merged.get("enable_follow_through", False)
                )
                # Keep default Phase 6D config identity: omit disabled flag.
                effective = {
                    "failure_policy": failure_policy,
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": tie,
                }
                if enable_ft:
                    effective["enable_follow_through"] = True
                mh = left + 1 + right
            elif impl == "pullback":
                from cpab.detectors.swing import _validate_strength, _validate_tie_policy
                from cpab.detectors.pullback import ALLOWED_VARIANTS

                defaults = {
                    "variant": "structure_based",
                    "left_strength": 1,
                    "right_strength": 1,
                    "tie_policy": "strict",
                }
                merged = dict(defaults)
                if concept in ("structure_based", "structure_based_pullback", "pullback:structure_based"):
                    merged["variant"] = "structure_based"
                elif concept in ("impulse_based", "impulse_based_pullback", "pullback:impulse_based"):
                    merged["variant"] = "impulse_based"
                merged.update(effective)
                if accepted is not None:
                    for key, val in config.items():
                        if key in accepted:
                            merged[key] = val
                var = merged.get("variant", "structure_based")
                if var not in ALLOWED_VARIANTS:
                    raise ValueError(f"variant must be one of {sorted(ALLOWED_VARIANTS)}, got {var!r}")
                left = _validate_strength("left_strength", merged.get("left_strength", 1))
                right = _validate_strength("right_strength", merged.get("right_strength", 1))
                policy = _validate_tie_policy(merged.get("tie_policy", "strict"))
                effective = {
                    "variant": var,
                    "left_strength": left,
                    "right_strength": right,
                    "tie_policy": policy,
                }
                if var == "impulse_based":
                    mh = 17
                elif var == "structure_based":
                    mh = left + 2 * right + 3
                else:
                    mh = min(17, left + 2 * right + 3)
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

    def _inverse_fair_value_gap_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.inverse_fair_value_gap import InverseFairValueGapDetector

        return InverseFairValueGapDetector(detector_id="inverse_fair_value_gap")

    reg.register(
        "inverse_fair_value_gap",
        concepts=[
            "inverse_fair_value_gap",
            "ifvg",
        ],
        factory=_inverse_fair_value_gap_factory,
        config={},
        accepted_params=[],
        min_history=4,
        implementation="inverse_fair_value_gap",
    )

    def _order_block_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.order_block import OrderBlockDetector

        c = dict(cfg or {})
        return OrderBlockDetector(
            min_body_ratio=c.get("min_body_ratio", 0.5),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="order_block",
        )

    reg.register(
        "order_block",
        concepts=[
            "order_block",
            "ob",
        ],
        factory=_order_block_factory,
        config={},
        accepted_params=[
            "min_body_ratio",
            "left_strength",
            "right_strength",
            "tie_policy",
        ],
        min_history=4,
        implementation="order_block",
    )

    def _breaker_block_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.breaker_block import BreakerBlockDetector

        c = dict(cfg or {})
        return BreakerBlockDetector(
            min_body_ratio=c.get("min_body_ratio", 0.5),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="breaker_block",
        )

    reg.register(
        "breaker_block",
        concepts=[
            "breaker_block",
            "breaker",
        ],
        factory=_breaker_block_factory,
        config={},
        accepted_params=[
            "min_body_ratio",
            "left_strength",
            "right_strength",
            "tie_policy",
        ],
        min_history=5,
        implementation="breaker_block",
    )

    def _mitigation_block_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.mitigation_block import MitigationBlockDetector

        c = dict(cfg or {})
        return MitigationBlockDetector(
            min_body_ratio=c.get("min_body_ratio", 0.5),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="mitigation_block",
        )

    reg.register(
        "mitigation_block",
        concepts=[
            "mitigation_block",
            "mitigation",
        ],
        factory=_mitigation_block_factory,
        config={},
        accepted_params=[
            "min_body_ratio",
            "left_strength",
            "right_strength",
            "tie_policy",
        ],
        min_history=6,
        implementation="mitigation_block",
    )

    def _classical_reversal_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.classical_reversal import ClassicalReversalDetector

        c = dict(cfg or {})
        return ClassicalReversalDetector(
            similarity_tolerance=c.get("similarity_tolerance", 0.002),
            max_bars_to_breakout=c.get("max_bars_to_breakout", 50),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="classical_reversal",
        )

    reg.register(
        "classical_reversal",
        concepts=[
            "double_top",
            "double_bottom",
            "head_and_shoulders",
            "inverse_head_and_shoulders",
            "classical_reversal",
        ],
        factory=_classical_reversal_factory,
        config={},
        accepted_params=[
            "similarity_tolerance",
            "max_bars_to_breakout",
            "left_strength",
            "right_strength",
            "tie_policy",
        ],
        min_history=16,
        implementation="classical_reversal",
    )

    def _rectangle_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.rectangle import RectangleDetector

        c = dict(cfg or {})
        return RectangleDetector(
            boundary_tolerance=c.get("boundary_tolerance", 0.002),
            max_bars_to_breakout=c.get("max_bars_to_breakout", 50),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="rectangle",
        )

    reg.register(
        "rectangle",
        concepts=[
            "rectangle",
        ],
        factory=_rectangle_factory,
        config={},
        accepted_params=[
            "boundary_tolerance",
            "max_bars_to_breakout",
            "left_strength",
            "right_strength",
            "tie_policy",
        ],
        min_history=13,
        implementation="rectangle",
    )

    def _triangle_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.triangle import TriangleDetector

        c = dict(cfg or {})
        return TriangleDetector(
            boundary_tolerance=c.get("boundary_tolerance", 0.002),
            horizontal_tolerance=c.get("horizontal_tolerance", 0.002),
            max_bars_to_breakout=c.get("max_bars_to_breakout", 50),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="triangle",
        )

    reg.register(
        "triangle",
        concepts=[
            "ascending_triangle",
            "descending_triangle",
            "symmetrical_triangle",
            "triangle",
        ],
        factory=_triangle_factory,
        config={},
        accepted_params=[
            "boundary_tolerance",
            "horizontal_tolerance",
            "max_bars_to_breakout",
            "left_strength",
            "right_strength",
            "tie_policy",
        ],
        min_history=13,
        implementation="triangle",
    )

    def _flags_pennants_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.flags_pennants import FlagsPennantsDetector

        c = dict(cfg or {})
        return FlagsPennantsDetector(
            min_pole_bars=c.get("min_pole_bars", 2),
            max_pole_bars=c.get("max_pole_bars", 10),
            pole_atr_lookback=c.get("pole_atr_lookback", 14),
            pole_strength_mult=c.get("pole_strength_mult", 1.5),
            max_pole_retracement=c.get("max_pole_retracement", 0.50),
            min_consolidation_bars=c.get("min_consolidation_bars", 3),
            max_consolidation_bars=c.get("max_consolidation_bars", 20),
            boundary_tolerance=c.get("boundary_tolerance", 0.002),
            parallel_slope_tolerance=c.get("parallel_slope_tolerance", 0.002),
            max_bars_to_breakout=c.get("max_bars_to_breakout", 50),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="flags_pennants",
        )

    reg.register(
        "flags_pennants",
        concepts=[
            "bullish_flag",
            "bearish_flag",
            "bullish_pennant",
            "bearish_pennant",
            "flags_pennants",
        ],
        factory=_flags_pennants_factory,
        config={},
        accepted_params=[
            "min_pole_bars",
            "max_pole_bars",
            "pole_atr_lookback",
            "pole_strength_mult",
            "max_pole_retracement",
            "min_consolidation_bars",
            "max_consolidation_bars",
            "boundary_tolerance",
            "parallel_slope_tolerance",
            "max_bars_to_breakout",
            "left_strength",
            "right_strength",
            "tie_policy",
        ],
        min_history=26,
        implementation="flags_pennants",
    )

    from cpab.detectors.behavior_volatility import (
        ATR_PERIOD,
        BASELINE_WINDOW,
        EXPANSION_MULT,
        CONTRACTION_MULT,
        RANGE_EXPANSION_MULT,
        VARIANT_ROLLING_ATR,
        VARIANT_TRUE_RANGE,
    )

    def _atr_expansion_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.behavior_volatility import ATRExpansionDetector

        return ATRExpansionDetector(detector_id="atr_expansion")

    reg.register(
        "atr_expansion",
        concepts=["atr_expansion"],
        factory=_atr_expansion_factory,
        config={
            "atr_period": ATR_PERIOD,
            "baseline_window": BASELINE_WINDOW,
            "crossing": "above",
            "threshold": EXPANSION_MULT,
            "variant": VARIANT_ROLLING_ATR,
        },
        accepted_params=[
            "atr_period",
            "baseline_window",
            "crossing",
            "threshold",
            "variant",
        ],
        min_history=35,
        implementation="behavior_volatility",
    )

    def _atr_contraction_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.behavior_volatility import ATRContractionDetector

        return ATRContractionDetector(detector_id="atr_contraction")

    reg.register(
        "atr_contraction",
        concepts=["atr_contraction"],
        factory=_atr_contraction_factory,
        config={
            "atr_period": ATR_PERIOD,
            "baseline_window": BASELINE_WINDOW,
            "crossing": "below",
            "threshold": CONTRACTION_MULT,
            "variant": VARIANT_ROLLING_ATR,
        },
        accepted_params=[
            "atr_period",
            "baseline_window",
            "crossing",
            "threshold",
            "variant",
        ],
        min_history=35,
        implementation="behavior_volatility",
    )

    def _range_expansion_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.behavior_volatility import RangeExpansionDetector

        return RangeExpansionDetector(detector_id="range_expansion")

    reg.register(
        "range_expansion",
        concepts=["range_expansion"],
        factory=_range_expansion_factory,
        config={
            "atr_period": ATR_PERIOD,
            "threshold": RANGE_EXPANSION_MULT,
            "variant": VARIANT_TRUE_RANGE,
        },
        accepted_params=["atr_period", "threshold", "variant"],
        min_history=16,
        implementation="behavior_volatility",
    )


    from cpab.detectors.behavior_volatility import (
        IMPULSE_BODY_ATR_MIN,
        IMPULSE_TRUE_RANGE_ATR_MIN,
        IMPULSE_BODY_RANGE_MIN,
        VARIANT_ATR_NORMALIZED_CANDLE,
        MIN_HISTORY_IMPULSE,
    )

    def _impulse_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.behavior_volatility import ImpulseDetector

        return ImpulseDetector(detector_id="impulse")

    reg.register(
        "impulse",
        concepts=["impulse", "bullish_impulse", "bearish_impulse"],
        factory=_impulse_factory,
        config={
            "atr_period": ATR_PERIOD,
            "body_atr_min": IMPULSE_BODY_ATR_MIN,
            "true_range_atr_min": IMPULSE_TRUE_RANGE_ATR_MIN,
            "body_range_min": IMPULSE_BODY_RANGE_MIN,
            "variant": VARIANT_ATR_NORMALIZED_CANDLE,
        },
        accepted_params=[
            "atr_period",
            "body_atr_min",
            "true_range_atr_min",
            "body_range_min",
            "variant",
        ],
        min_history=MIN_HISTORY_IMPULSE,
        implementation="behavior_volatility",
    )

    def _pullback_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.pullback import PullbackDetector

        c = dict(cfg or {})
        return PullbackDetector(
            variant=c.get("variant", "structure_based"),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            detector_id="pullback",
        )

    reg.register(
        "pullback",
        concepts=[
            "pullback",
            "structure_based",
            "impulse_based",
            "structure_based_pullback",
            "impulse_based_pullback",
            "pullback:structure_based",
            "pullback:impulse_based",
        ],
        factory=_pullback_factory,
        config={},
        accepted_params=[
            "variant",
            "left_strength",
            "right_strength",
            "tie_policy",
        ],
        min_history=6,
        implementation="pullback",
    )

    def _breakout_behavior_factory(
        cfg: Optional[dict[str, Any]] = None,
    ) -> PriceActionDetector:
        from cpab.detectors.breakout_behavior import BreakoutBehaviorDetector

        c = dict(cfg or {})
        return BreakoutBehaviorDetector(
            failure_policy=c.get("failure_policy", "close_back_across"),
            left_strength=c.get("left_strength", 1),
            right_strength=c.get("right_strength", 1),
            tie_policy=c.get("tie_policy", "strict"),
            enable_follow_through=c.get("enable_follow_through", False),
            detector_id="breakout_behavior",
        )

    reg.register(
        "breakout_behavior",
        concepts=["breakout_behavior", "breakout"],
        factory=_breakout_behavior_factory,
        config={},
        accepted_params=[
            "failure_policy",
            "left_strength",
            "right_strength",
            "tie_policy",
            "enable_follow_through",
        ],
        min_history=3,
        implementation="breakout_behavior",
    )

    return reg
