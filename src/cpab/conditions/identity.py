"""Deterministic configuration identity for validated SetupDef instances."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cpab.conditions.schema import SetupDef


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 safe."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def setup_config_identity(setup: SetupDef) -> str:
    """Return a stable configuration identity for a validated setup.

    Rules:
    - Includes schema_version and all behavior-affecting fields.
    - Excludes presentation-only fields (name, description).
    - setup_id is part of the identity so two setups with different ids but
      identical other fields remain distinct; the setup_id itself is stable
      across material changes (callers keep setup_id; identity changes).
    - Equivalent configs (key order / YAML formatting) produce the same digest.
    - No timestamps, randomness, or nondeterministic iteration order.
    """
    payload = setup.to_behavior_dict()
    canonical = _canonical_json(payload)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"setupcfg:{digest[:24]}"


def expression_config_identity(expression_dict: dict[str, Any]) -> str:
    """Identity helper for an expression dict alone (tests / diagnostics)."""
    canonical = _canonical_json(expression_dict)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"expr:{digest[:16]}"
