"""Phase 8C — storage.database_path resolution (A01–A06).

Locked behaviour (§17.2):
  - Configuration key: storage.database_path
  - Default value: ../data/cpab_signals.sqlite3
  - Relative paths resolve against the absolute parent directory of the
    selected YAML configuration file (not CWD or repo root).
  - Expand a leading ~.
  - No environment-variable interpolation or override.
  - Keep --config / -c as the configuration selector; no separate DB-path CLI
    override.
  - Missing, empty, null, or non-string values produce an explicit
    configuration error. No fallback path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cpab.config import database_path_from_config
from cpab.storage.path import ConfigurationError, resolve_database_path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


# -- A01: default value resolves relative to config file's parent dir ------

def test_default_database_path_resolves_against_config_parent():
    cfg = {"storage": {"database_path": "../data/cpab_signals.sqlite3"}}
    resolved = resolve_database_path(cfg, str(DEFAULT_CONFIG))
    assert resolved.is_absolute()
    assert resolved == (CONFIG_DIR / "../data/cpab_signals.sqlite3").resolve()
    assert resolved.name == "cpab_signals.sqlite3"


def test_default_value_is_documented():
    """The documented default is ../data/cpab_signals.sqlite3."""
    from cpab.storage.path import DEFAULT_DATABASE_PATH
    assert DEFAULT_DATABASE_PATH == "../data/cpab_signals.sqlite3"


# -- A02: custom config directory (different from default config dir) ------

def test_custom_config_directory_different_from_default(tmp_path):
    """Place YAML in a directory different from the default config dir.

    Verify both the resolved database path and the startup diagnostic path.
    """
    custom_dir = tmp_path / "custom_config"
    custom_dir.mkdir()
    custom_cfg = custom_dir / "custom.yaml"
    custom_cfg.write_text("storage:\n  database_path: \"data/signals.sqlite3\"\n")

    cfg = {"storage": {"database_path": "data/signals.sqlite3"}}
    resolved = resolve_database_path(cfg, str(custom_cfg))
    expected = (custom_dir / "data/signals.sqlite3").resolve()
    assert resolved == expected
    assert resolved != (CONFIG_DIR / "../data/cpab_signals.sqlite3").resolve()
    assert resolved.parent == custom_dir / "data"


# -- A03: leading ~ expansion ---------------------------------------------

def test_tilde_expansion(tmp_path):
    cfg = {"storage": {"database_path": "~/cpab_signals.sqlite3"}}
    resolved = resolve_database_path(cfg, str(DEFAULT_CONFIG))
    assert resolved == Path.home() / "cpab_signals.sqlite3"
    assert resolved.is_absolute()


# -- A04: absolute path used as-is ----------------------------------------

def test_absolute_path_used_as_is(tmp_path):
    abs_db = tmp_path / "abs" / "signals.sqlite3"
    cfg = {"storage": {"database_path": str(abs_db)}}
    resolved = resolve_database_path(cfg, str(DEFAULT_CONFIG))
    assert resolved == abs_db.resolve()


# -- A05: no env var interpolation ----------------------------------------

def test_no_env_var_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("CPAB_DB", "/tmp/env_interpolated.sqlite3")
    cfg = {"storage": {"database_path": "$CPAB_DB/signals.sqlite3"}}
    # Literal $CPAB_DB is not interpolated; used as a path segment.
    resolved = resolve_database_path(cfg, str(DEFAULT_CONFIG))
    assert "$CPAB_DB" in str(resolved)


# -- A06: missing / empty / null / non-string values produce errors ------

def test_missing_storage_key_raises():
    with pytest.raises(ConfigurationError):
        resolve_database_path({}, str(DEFAULT_CONFIG))


def test_missing_database_path_key_raises():
    with pytest.raises(ConfigurationError):
        resolve_database_path({"storage": {}}, str(DEFAULT_CONFIG))


def test_null_database_path_raises():
    with pytest.raises(ConfigurationError):
        resolve_database_path({"storage": {"database_path": None}}, str(DEFAULT_CONFIG))


def test_empty_database_path_raises():
    with pytest.raises(ConfigurationError):
        resolve_database_path({"storage": {"database_path": ""}}, str(DEFAULT_CONFIG))


def test_whitespace_only_database_path_raises():
    with pytest.raises(ConfigurationError):
        resolve_database_path({"storage": {"database_path": "   "}}, str(DEFAULT_CONFIG))


def test_non_string_database_path_raises():
    with pytest.raises(ConfigurationError):
        resolve_database_path({"storage": {"database_path": 123}}, str(DEFAULT_CONFIG))


def test_storage_not_mapping_raises():
    with pytest.raises(ConfigurationError):
        resolve_database_path({"storage": "not-a-mapping"}, str(DEFAULT_CONFIG))


# -- config-level wrapper --------------------------------------------------

def test_config_wrapper_raises_on_invalid():
    with pytest.raises(ConfigurationError):
        database_path_from_config({}, str(DEFAULT_CONFIG))
