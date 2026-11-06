"""SQLite database connection, PRAGMAs, and schema initialization.

Durability settings are locked by §17.1 / §17.2:
  - Journal mode: DELETE
  - Synchronous mode: FULL
  - Foreign keys: ON
  - Busy timeout: 5 seconds
  - Explicit write transaction: BEGIN IMMEDIATE

Schema version: 1. Serializer version: 1. Fingerprint version: 1.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from cpab.storage.path import ConfigurationError

SCHEMA_VERSION = 1
SERIALIZER_VERSION = 1
FINGERPRINT_VERSION = 1

PRAGMA_STATEMENTS = (
    "PRAGMA journal_mode = DELETE",
    "PRAGMA synchronous = FULL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 5000",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    condition_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    candle_epoch INTEGER NOT NULL,
    generated_at REAL NOT NULL,
    reasons TEXT NOT NULL,
    metadata TEXT NOT NULL,
    fingerprint TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    candle_epoch INTEGER NOT NULL,
    detected_at REAL NOT NULL,
    event_time INTEGER,
    confirmed_at INTEGER,
    developing INTEGER NOT NULL DEFAULT 0,
    direction TEXT,
    variant TEXT,
    detector_id TEXT,
    detector_config_id TEXT,
    metadata TEXT,
    payload TEXT,
    lifecycle_state TEXT
);

CREATE TABLE IF NOT EXISTS signal_observations (
    signal_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    PRIMARY KEY (signal_id, observation_id),
    FOREIGN KEY (signal_id) REFERENCES signals(id),
    FOREIGN KEY (observation_id) REFERENCES observations(id)
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_timeframe
    ON signals(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_signals_candle_epoch
    ON signals(candle_epoch);
CREATE INDEX IF NOT EXISTS idx_signals_condition_id
    ON signals(condition_id);
CREATE INDEX IF NOT EXISTS idx_observations_symbol_timeframe
    ON observations(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_observations_type
    ON observations(type);
CREATE INDEX IF NOT EXISTS idx_signal_observations_observation_id
    ON signal_observations(observation_id);
"""

META_SQL = """
INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with locked durability settings.

    Creates missing parent directories only when the database file itself is
    being created for the first time. Never overwrites, deletes, or recreates
    an existing database file.
    """
    db_path = Path(db_path)

    if db_path.exists() and not db_path.is_file():
        raise ConfigurationError(
            f"Database path is not a regular file: {db_path}"
        )

    is_new = not db_path.exists()
    if is_new:
        parent = db_path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
    except sqlite3.Error as exc:
        raise ConfigurationError(
            f"Failed to open database {db_path}: {exc}"
        ) from exc

    conn.row_factory = sqlite3.Row

    try:
        for stmt in PRAGMA_STATEMENTS:
            conn.execute(stmt)
        conn.executescript(SCHEMA_SQL)
        for key, value in (
            ("schema_version", str(SCHEMA_VERSION)),
            ("serializer_version", str(SERIALIZER_VERSION)),
            ("fingerprint_version", str(FINGERPRINT_VERSION)),
        ):
            conn.execute(META_SQL, (key, value))
        conn.commit()
    except sqlite3.Error as exc:
        conn.close()
        raise ConfigurationError(
            f"Failed to initialize database {db_path}: {exc}"
        ) from exc

    return conn


def get_pragma(conn: sqlite3.Connection, name: str) -> Optional[str]:
    """Return the effective value of a PRAGMA as a string."""
    row = conn.execute(f"PRAGMA {name}").fetchone()
    if row is None:
        return None
    return str(row[0])
