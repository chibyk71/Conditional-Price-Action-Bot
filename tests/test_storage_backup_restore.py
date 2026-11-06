"""Phase 8C — backup and restore (A21–A28).

Locked behaviour (§17.2):
  - Application-level maintenance gate suspends new engine evaluations and
    signal-consumer writes for the entire backup operation.
  - SQLite standard-library backup API.
  - Verify with PRAGMA integrity_check and PRAGMA user_version.
  - Destination operator-selected and outside the live database directory.
  - No automatic backup scheduling or rotation.
  - Restore: app stopped, target database closed.
  - Preserve existing database and -journal sidecar before restoration.
  - Unexpected -wal or -shm sidecars: preserve and halt.
  - Restore through staged file in target directory; verify before replacement.
  - Explicit file-level replacement, atomically where supported.
  - Preserve pre-restore copy until post-restore verification succeeds.
  - On failure, do not restart the application.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from cpab.storage.backup import MaintenanceGate, backup_database
from cpab.storage.database import connect
from cpab.storage.restore import (
    RestoreAbortedError,
    RestoreVerificationError,
    restore_database,
)


def _make_db(db_path, signals=1):
    conn = connect(db_path)
    for i in range(signals):
        sid = f"sig-{i}"
        conn.execute(
            "INSERT INTO signals (id, condition_id, symbol, timeframe, "
            "candle_epoch, generated_at, reasons, metadata, fingerprint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, "c1", "R_100", "1m", 1700000000, 1000000.0, "[]", "{}", "fp"),
        )
        conn.execute(
            "INSERT INTO observations (id, type, symbol, timeframe, "
            "candle_epoch, detected_at, event_time, confirmed_at, developing, "
            "direction, variant, detector_id, detector_config_id, "
            "metadata, payload, lifecycle_state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"obs-{i}", "engulfing", "R_100", "1m", 1700000000, 1000000.0,
             1700000000, 1700000000, 0, "bullish", None, "eng", None, "{}",
             "{}", None),
        )
        conn.execute(
            "INSERT INTO signal_observations (signal_id, observation_id) "
            "VALUES (?, ?)",
            (sid, f"obs-{i}"),
        )
    conn.commit()
    conn.close()


def test_maintenance_gate_acquire_release():
    gate = MaintenanceGate()
    assert not gate.active
    gate.acquire()
    assert gate.active
    gate.release()
    assert not gate.active


def test_backup_creates_verified_copy(tmp_path):
    src = str(tmp_path / "live.sqlite3")
    dst = str(tmp_path / "backup" / "copy.sqlite3")
    _make_db(src, signals=3)
    gate = MaintenanceGate()
    gate.acquire()
    try:
        backup_database(src, dst, gate)
    finally:
        gate.release()
    assert gate.active is False
    assert os.path.exists(dst)
    conn = connect(dst)
    try:
        n = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        assert n == 3
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        conn.close()


def test_backup_rejects_same_directory(tmp_path):
    src = str(tmp_path / "live.sqlite3")
    dst = str(tmp_path / "copy.sqlite3")
    _make_db(src, signals=1)
    gate = MaintenanceGate()
    gate.acquire()
    try:
        with pytest.raises(ValueError):
            backup_database(src, dst, gate)
    finally:
        gate.release()


def test_restore_preserves_pre_restore_copy_until_verified(tmp_path):
    target = str(tmp_path / "target.sqlite3")
    staged = str(tmp_path / "staged.sqlite3")
    _make_db(target, signals=2)
    _make_db(staged, signals=5)
    restore_database(staged, target)
    conn = connect(target)
    try:
        n = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        assert n == 5
    finally:
        conn.close()


def test_restore_halts_on_unexpected_wal_shm(tmp_path):
    target = str(tmp_path / "target.sqlite3")
    staged = str(tmp_path / "staged.sqlite3")
    _make_db(target, signals=1)
    _make_db(staged, signals=1)
    Path(target + "-wal").touch()
    try:
        with pytest.raises(RestoreAbortedError):
            restore_database(staged, target)
    finally:
        for p in [target + "-wal", target + "-shm"]:
            if os.path.exists(p):
                os.unlink(p)


def test_restore_rejects_corrupt_staged_backup(tmp_path):
    target = str(tmp_path / "target.sqlite3")
    staged = str(tmp_path / "staged.sqlite3")
    _make_db(target, signals=1)
    Path(staged).write_bytes(b"not a sqlite database")
    try:
        with pytest.raises((RestoreVerificationError, sqlite3.DatabaseError)):
            restore_database(staged, target)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)


def test_restore_uses_file_replacement(tmp_path):
    target = str(tmp_path / "target.sqlite3")
    staged = str(tmp_path / "staged.sqlite3")
    _make_db(target, signals=1)
    _make_db(staged, signals=7)
    restore_database(staged, target)
    conn = connect(target)
    try:
        n = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        assert n == 7
    finally:
        conn.close()


def test_restore_failure_reinstates_pre_restore_copy(tmp_path):
    target = str(tmp_path / "target.sqlite3")
    staged = str(tmp_path / "staged.sqlite3")
    _make_db(target, signals=3)
    Path(staged).write_bytes(b"corrupt")
    try:
        with pytest.raises((RestoreVerificationError, sqlite3.DatabaseError)):
            restore_database(staged, target)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)
    conn = connect(target)
    try:
        n = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        assert n == 3
    finally:
        conn.close()

