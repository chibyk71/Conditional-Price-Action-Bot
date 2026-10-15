"""
Minimal detector capability registry for CPAB Phase 1+.

Registered capabilities have deterministic configuration identity.
Unsupported patterns / invalid params are rejected (no silent substitution).

Phase 2A: swing_pivot accepts left_strength, right_strength, tie_policy as
overridable configuration. TA-Lib pattern values remain fixed at registration.
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
          - Swing applies documented defaults for missing overridable keys.
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

            # Fixed parameters: any key in registered config is immutable.
            # accepted_params only controls which keys are *allowed*, not which
            # may change a registered value (Phase 1 contract).
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
                    # Only merge non-fixed accepted keys (not already constrained
                    # by registered). Fixed keys already matched above.
                    if key in accepted_set and key not in registered:
                        effective[key] = val

            # Swing: overridable params live only in accepted_params (registered
            # config is empty). Apply defaults, then user overrides, then validate.
            if entry.get("implementation") == "swing_pivot":
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

        # Explicit signature check — do not catch TypeError from factory body.
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

        # pattern is a *fixed* registered value (cannot override).
        # accepted_params=["pattern"] only allows the key to appear; its value
        # must match the registered pattern (Phase 1 contract).
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

    # --- Phase 2A: swing pivot ---
    # Registered config is empty so left/right/tie_policy are not fixed params.
    # Defaults and validation are applied in validate_config for swing_pivot.
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
        config={},  # no fixed params — strengths/tie_policy are overridable
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=3,
        implementation="swing_pivot",
    )

    return reg
