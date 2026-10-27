"""Version-1 setup schema (declarative configuration only).

Phase 7A defines and validates setup configuration. It does not evaluate
expressions, manage instances, or generate signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class SetupDef:
    """Validated version-1 setup definition.

    Reserved fields (invalidation, expiry) are stored when present and
    structurally validated, but have no runtime behavior in Phase 7A.
    """

    schema_version: int
    setup_id: str
    symbol: str
    timeframe: str
    enabled: bool
    expression: Expression
    name: Optional[str] = None
    description: Optional[str] = None
    invalidation: Optional[dict[str, Any]] = None
    expiry: Optional[dict[str, Any]] = None

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
            d["invalidation"] = self.invalidation
        if self.expiry is not None:
            d["expiry"] = self.expiry
        return d
