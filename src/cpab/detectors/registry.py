"""
Minimal detector capability registry for CPAB Phase 1.

Answers:
  1. Which concepts/patterns are supported?
  2. Which detector implementation + effective config provide the capability?
  3. What minimum history does the capability require?

Not a plugin system. No dynamic imports, no DI framework.
Unsupported concepts and invalid configurations fail during validation
(before market monitoring), with no silent substitution.

Each registered capability has an immutable effective configuration.
Caller-supplied config may only supply keys already present with the same
values (or omit keys to accept defaults). Overriding a fixed pattern or other
immutable parameter is rejected. create_detector always constructs from the
registered configuration so detector_config_id, pattern, and min_history
describe the instance that actually runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.talib_adapter import (
    TalibPatternDetector,
    min_history_for,
    _PATTERN_ALIASES,
)


def deterministic_config_id(config: dict[str, Any]) -> str:
    """
    Deterministic detector_config_id from effective configuration.
    Equivalent effective configs → same ID; order-independent.
    Compatible with Phase 0 Observation identity (plain string token).
    """
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


@dataclass
class CapabilityRegistry:
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    _concept_index: dict[str, list[str]] = field(default_factory=dict)

    def register(
        self,
        detector_id: str,
        *,
        concepts: list[str],
        factory: Callable[[], PriceActionDetector],
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
        Validate concept + config against a registered capability.

        Registered configs are immutable: caller keys must not introduce unknown
        params, and any provided value for a registered key must equal the
        registered default (no silent override of pattern or other fixed params).
        The returned CapabilityInfo always reflects the registered effective config.
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

            # Reject attempts to override immutable registered values
            for key, val in config.items():
                if key in registered and registered[key] != val:
                    last_error = (
                        f"Invalid configuration for concept {concept!r}: "
                        f"cannot override fixed parameter {key!r} "
                        f"(registered={registered[key]!r}, got={val!r})"
                    )
                    break
            else:
                # effective config is always the registered config (immutable)
                effective = dict(registered)
                config_id = deterministic_config_id(effective)
                mh = entry["min_history"]
                if mh is None and entry.get("pattern"):
                    mh = min_history_for(entry["pattern"])
                return CapabilityInfo(
                    concept=concept,
                    detector_id=did,
                    detector_config_id=config_id,
                    min_history=mh if mh is not None else 2,
                    pattern=entry.get("pattern"),
                    implementation=entry.get("implementation") or "",
                )
            continue
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
        """
        Instantiate the detector for a validated concept.
        The detector is always built from the registered factory/config so
        pattern, min_history, and detector_config_id match the running instance.
        """
        info = self.validate_config(concept, config or {})
        entry = self._entries[info.detector_id]
        det = entry["factory"]()
        # Attach the validated (registered) config identity
        if isinstance(det, TalibPatternDetector):
            det._detector_config_id = info.detector_config_id
            # Ensure pattern matches registration
            if info.pattern and det.pattern != info.pattern:
                raise RuntimeError(
                    f"Factory pattern mismatch: detector has {det.pattern!r}, "
                    f"registry expects {info.pattern!r}"
                )
        return det, info

    def get_entry(self, detector_id: str) -> Optional[dict[str, Any]]:
        return self._entries.get(detector_id)


def build_default_registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()

    def _engulfing_factory() -> PriceActionDetector:
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

        def _make_factory(p: str = pattern) -> Callable[[], PriceActionDetector]:
            def factory() -> PriceActionDetector:
                return TalibPatternDetector(pattern=p, detector_id=f"talib:{p.lower()}")
            return factory

        reg.register(
            f"talib:{pattern.lower()}",
            concepts=list(dict.fromkeys(concepts)),
            factory=_make_factory(),
            config={"pattern": pattern},
            accepted_params=["pattern"],
            min_history=min_history_for(pattern),
            implementation="talib",
            pattern=pattern,
        )

    return reg
