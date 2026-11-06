"""Phase 8C — SQLite database, PRAGMAs, and schema (A07–A12).

Locked behaviour (§17.1 / §17.2):
  - SQLite is the sole authoritative signal store.
  - Python standard-library sqlite3.
  - Journal mode: DELETE.
  - Synchronous mode: FULL.
  - Foreign keys: ON.
  - Busy timeout: 5 seconds.
  - Initial schema version: 1.
  - Initial serializer/fingerprint version: 1.
  - Create missing parent directories only as required for a new database.
  - Never overwrite, delete, or recreate an invalid existing database.
  - Fail explicitly for inaccessible paths and database-open failures.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from cpab.storage.database import (
    SCHEMA_VERSION,
    SERIALIZER_VERSION,
    FINGERPRINT_VERSION,
    connect,
    get_pragma,
)
from cpab.storage.path import ConfigurationError


EXPECTED_TABLES = {"meta", "signals", "observations", "signal_observations"}
EXPECTED_INDEXES = {
    "idx_signals_symbol_timeframe",
    "idx_signals_candle_epoch",
    "idx_signals_condition_id",
    "idx_observations_symbol_timeframe",
    "idx_observations_type",
    "idx_signal_observations_observation_id",
}


@contextmanager
def _connect_and_close(db_path):
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def test_journal_mode_delete(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        assert get_pragma(conn, "journal_mode") == "delete"


def test_synchronous_full(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        assert get_pragma(conn, "synchronous") == "2"


def test_foreign_keys_on(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        assert get_pragma(conn, "foreign_keys") == "1"


def test_busy_timeout_5_seconds(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        assert get_pragma(conn, "busy_timeout") == "5000"



def test_schema_version_is_one(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        assert SCHEMA_VERSION == 1
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "1"


def test_serializer_version_is_one(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        assert SERIALIZER_VERSION == 1
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'serializer_version'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "1"


def test_fingerprint_version_is_one(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        assert FINGERPRINT_VERSION == 1
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'fingerprint_version'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "1"


def test_tables_created(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert EXPECTED_TABLES <= tables


def test_signals_columns(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(signals)")}
        expected = {
            "id", "condition_id", "symbol", "timeframe", "candle_epoch",
            "generated_at", "reasons", "metadata", "fingerprint",
        }
        assert expected <= cols


def test_observations_columns(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(observations)")}
        expected = {
            "id", "type", "symbol", "timeframe", "candle_epoch", "detected_at",
            "event_time", "confirmed_at", "developing", "direction", "variant",
            "detector_id", "detector_config_id", "metadata", "payload",
            "lifecycle_state",
        }
        assert expected <= cols


def test_signal_observations_columns_and_fk(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(signal_observations)")}
        assert {"signal_id", "observation_id"} <= cols
        fks = conn.execute("PRAGMA foreign_key_list(signal_observations)").fetchall()
        fk_tables = {r["table"] for r in fks}
        assert fk_tables == {"signals", "observations"}


def test_indexes_created(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    with _connect_and_close(db) as conn:
        indexes = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert EXPECTED_INDEXES <= indexes


def test_creates_parent_directory_for_new_database(tmp_path):
    new_db = str(tmp_path / "new_subdir" / "signals.sqlite3")
    assert not os.path.exists(tmp_path / "new_subdir")
    connect(new_db)
    assert os.path.exists(tmp_path / "new_subdir")
    assert os.path.exists(new_db)


def test_invalid_path_raises(tmp_path):
    not_a_file = tmp_path / "adir"
    not_a_file.mkdir()
    with pytest.raises(ConfigurationError):
        connect(str(not_a_file))
