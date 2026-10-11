"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cpab.models import ConditionDef


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data


def conditions_from_config(cfg: dict[str, Any]) -> list[ConditionDef]:
    out: list[ConditionDef] = []
    for raw in cfg.get("conditions") or []:
        out.append(
            ConditionDef(
                id=raw["id"],
                symbol=raw["symbol"],
                timeframe=raw["timeframe"],
                enabled=bool(raw.get("enabled", True)),
                require_observations=list(raw.get("require_observations") or []),
            )
        )
    return out
