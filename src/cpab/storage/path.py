"""Database path resolution for ``storage.database_path``.

Locked behaviour (§17.2):
  - The ``storage.database_path`` key must be present and a non-empty string.
  - Missing, empty, null, or non-string values raise ``ConfigurationError``.
  - No fallback path is used.
  - A leading ``~`` is expanded.
  - Relative paths resolve against the absolute parent directory of the
    selected YAML configuration file (not the CWD or repository root).
  - Absolute paths are used as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_DATABASE_PATH = "../data/cpab_signals.sqlite3"


class ConfigurationError(ValueError):
    """Raised when configuration is invalid and no fallback is safe."""


def resolve_database_path(
    cfg: dict[str, Any], config_path: str | Path
) -> Path:
    """Resolve the effective absolute database path from configuration."""
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent

    storage_cfg = cfg.get("storage")
    if not isinstance(storage_cfg, dict):
        raise ConfigurationError(
            "storage.database_path is required: 'storage' mapping is missing "
            "or not a mapping in configuration"
        )

    raw = storage_cfg.get("database_path")
    if raw is None:
        raise ConfigurationError(
            "storage.database_path is required: key is missing or null in "
            "configuration"
        )
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(
            "storage.database_path must be a non-empty string, got "
            f"{type(raw).__name__}"
        )

    candidate = Path(raw.strip())
    candidate = candidate.expanduser()

    if candidate.is_absolute():
        return candidate
    return (config_dir / candidate).resolve()
