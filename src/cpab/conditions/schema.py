"""Version-1 setup schema (declarative configuration only).

Phase 7A defines and validates setup configuration. Phase 7C adds validated
invalidation expression and explicit candle-based expiry configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from cpab.conditions.expression import Expression


SCHEMA_VERSION_V1 = 1

# Fields permitted on a version-1 setup (required + optional + reserved).
V1_REQUIRED_FIELDS = frozenset(
    {"schema_version", "setup_id", "symbol", "timeframe", "enabled", "expression"}
)
V1_OPTIONAL_FIELDS = frozenset({"name", "description", "invalidation", "expiry"})
V1_ALLOWED_FIELDS = V1_REQUIRED_FIELDS | V1_OPTIONAL_FIELDS


@dataclass(frozen=True)
class ExpiryConfig:
    """Explicit candle-based expiry (Phase 7C).

    Anchored to the instance's first accepted evaluation candle.
    Expires inclusively when elapsed valid candles >= max_age_candles.
    """

    max_age_candles: int

    def to_dict(self) -> dict[str, Any]:
        return {"max_age_candles": self.max_age_candles}


@dataclass(frozen=True)
class SetupDef:
    """Validated version-1 setup definition.

    invalidation: optional ObservationRequirement / AND / OR expression (no THEN).
    expiry: optional explicit max_age_candles configuration.
    """

    schema_version: int
    setup_id: str
    symbol: str
    timeframe: str
    enabled: bool
    expression: Expression
    name: Optional[str] = None
    description: Optional[str] = None
    invalidation: Optional[Expression] = None
    expiry: Optional[ExpiryConfig] = None

    def to_behavior_dict(self) -> dict[str, Any]:
        """Behavior-affecting fields only (excludes presentation-only name/description)."""
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "setup_id": self.setup_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "enabled": self.enabled,
            "expression": self.expression.to_dict(),
        }
        if self.invalidation is not None:
            d["invalidation"] = self.invalidation.to_dict()
        if self.expiry is not None:
            d["expiry"] = self.expiry.to_dict()
        return d
