"""
Minimal detector capability registry for CPAB Phase 1+.

Registered capabilities have deterministic configuration identity.
Unsupported patterns / invalid params are rejected (no silent substitution).

Phase 2A: swing_pivot accepts left_strength, right_strength, tie_policy overrides
within accepted_params; effective config drives detector construction and config_id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

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

            # Fixed params: keys present in registered that are NOT in accepted_params
            # cannot be overridden. Keys in accepted_params may be overridden.
            accepted_set = set(accepted) if accepted is not None else set()
            conflict = False
            for key, val in config.items():
                if key in registered and registered[key] != val and key not in accepted_set:
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
                for key, val in config.items():
                    if key in accepted_set:
                        effective[key] = val

            config_id = deterministic_config_id(effective)
            mh = entry["min_history"]
            if mh is None and entry.get("pattern"):
                mh = min_history_for(entry["pattern"])
            # Swing: min_history from strengths when present
            if entry.get("implementation") == "swing_pivot":
                left = int(effective.get("left_strength", 1))
                right = int(effective.get("right_strength", 1))
                mh = left + 1 + right
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

        det: PriceActionDetector
        # Prefer config-aware factory (Phase 2A swing); fall back to zero-arg.
        try:
            det = factory(effective)
        except TypeError:
            det = factory()

        if isinstance(det, TalibPatternDetector):
            det._detector_config_id = info.detector_config_id
            if info.pattern and det.pattern != info.pattern:
                raise RuntimeError(
                    f"Factory pattern mismatch: detector has {det.pattern!r}, "
                    f"registry expects {info.pattern!r}"
                )
        # Swing and other config-aware detectors should already embed config_id;
        # force-set if attribute exists and is still None.
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
    def _swing_factory(cfg: Optional[dict[str, Any]] = None) -> PriceActionDetector:
        from cpab.detectors.swing import SwingPivotDetector

        c = dict(cfg or {})
        left = int(c.get("left_strength", 1))
        right = int(c.get("right_strength", 1))
        policy = str(c.get("tie_policy", "strict"))
        effective = {
            "left_strength": left,
            "right_strength": right,
            "tie_policy": policy,
        }
        return SwingPivotDetector(
            left_strength=left,
            right_strength=right,
            tie_policy=policy,
            detector_id="swing_pivot",
            detector_config_id=deterministic_config_id(effective),
        )

    default_swing_cfg = {
        "left_strength": 1,
        "right_strength": 1,
        "tie_policy": "strict",
    }
    reg.register(
        "swing_pivot",
        concepts=["swing_high", "swing_low", "swing"],
        factory=_swing_factory,
        config=default_swing_cfg,
        accepted_params=["left_strength", "right_strength", "tie_policy"],
        min_history=3,
        implementation="swing_pivot",
    )

    return reg
