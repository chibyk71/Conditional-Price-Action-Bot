"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cpab.conditions.schema import SetupDef
from cpab.conditions.validation import SetupValidationError, parse_setups
from cpab.models import ConditionDef


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data


def conditions_from_config(cfg: dict[str, Any]) -> list[ConditionDef]:
    """Load legacy ConditionDef entries from the ``conditions`` key.

    Preserves existing defaults and matching behavior. Does not interpret
    legacy definitions as version-1 expression-based setups.
    """
    out: list[ConditionDef] = []
    for raw in cfg.get("conditions") or []:
        if not isinstance(raw, dict):
            raise SetupValidationError(
                f"legacy condition entry must be a mapping, got {type(raw).__name__}",
                path="conditions",
            )
        if "id" not in raw or not isinstance(raw["id"], str) or not raw["id"].strip():
            raise SetupValidationError(
                "legacy condition missing non-empty 'id'",
                path="conditions",
            )
        if "symbol" not in raw or not isinstance(raw["symbol"], str) or not raw["symbol"].strip():
            raise SetupValidationError(
                f"legacy condition {raw.get('id')!r} missing non-empty 'symbol'",
                path="conditions",
                setup_id=str(raw.get("id")),
            )
        if "timeframe" not in raw or not isinstance(raw["timeframe"], str) or not raw["timeframe"].strip():
            raise SetupValidationError(
                f"legacy condition {raw.get('id')!r} missing non-empty 'timeframe'",
                path="conditions",
                setup_id=str(raw.get("id")),
            )
        out.append(
            ConditionDef(
                id=raw["id"].strip(),
                symbol=raw["symbol"].strip(),
                timeframe=raw["timeframe"].strip(),
                enabled=bool(raw.get("enabled", True)),
                require_observations=list(raw.get("require_observations") or []),
            )
        )
    return out


def setups_from_config(cfg: dict[str, Any]) -> list[SetupDef]:
    """Load and strictly validate version-1 setups from the ``setups`` key.

    Empty or missing ``setups`` yields an empty list. Disabled setups are
    still fully validated. Unknown top-level keys under each setup are rejected.
    """
    return parse_setups(cfg.get("setups"))


def legacy_condition_as_observation_list(cond: ConditionDef) -> list[dict[str, Any]]:
    """Return a shallow copy of require_observations for identity/tests.

    Does not build an expression tree or change evaluation semantics.
    """
    return [dict(r) for r in cond.require_observations]
