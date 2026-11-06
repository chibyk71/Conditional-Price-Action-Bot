"""Application-maintenance-gated SQLite backup and restore.

The application-level gate (not SQLite read-lock behaviour) establishes the
CPAB write-pause boundary for backup. External direct database writers remain
the operator's responsibility.

Locked behaviour (§17.2):
  - Suspend new engine evaluations and signal-consumer writes for the entire
    backup operation.
  - Use the SQLite standard-library backup API.
  - Resume only after successful completion or explicit failure handling.
  - Verify with ``PRAGMA integrity_check`` and ``PRAGMA user_version``.
  - Destination must be operator-selected and outside the live database
    directory.
  - No automatic backup scheduling or rotation.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from cpab.logging_setup import get_logger

logger = get_logger("cpab.storage.backup")


class MaintenanceGate:
    """Application-level maintenance gate.

    When ``active`` is True, new engine evaluations and signal-consumer writes
    must be suspended. The gate is cooperative: the engine and store consult
    it before proceeding. It does not rely on SQLite read locks.
    """

    def __init__(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def acquire(self) -> None:
        self._active = True
        logger.info("Maintenance gate acquired: write-pause boundary active")

    def release(self) -> None:
        self._active = False
        logger.info("Maintenance gate released: write-pause boundary ended")


def _integrity_ok(conn: sqlite3.Connection) -> bool:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    return row is not None and str(row[0]).lower() == "ok"


def _user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def backup_database(
    source_db: str | Path,
    destination_db: str | Path,
    gate: MaintenanceGate,
) -> None:
    """Back up the live database to an operator-selected destination.

    The gate must be acquired by the caller before invoking this function and
    released afterward (even on failure). This function verifies the
    destination copy with ``PRAGMA integrity_check`` and
    ``PRAGMA user_version``.
    """
    source_db = Path(source_db)
    destination_db = Path(destination_db)

    # Destination must be outside the live database directory.
    source_dir = source_db.resolve().parent
    dest_dir = destination_db.resolve().parent
    if dest_dir == source_dir:
        raise ValueError(
            "Backup destination must be outside the live database directory "
            f"({source_dir}); got {dest_dir}"
        )

    # Create destination parent directory if missing.
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)

    # Remove any stale destination file so we start from a clean copy.
    if destination_db.exists():
        destination_db.unlink()

    source_conn: Optional[sqlite3.Connection] = None
    dest_conn: Optional[sqlite3.Connection] = None

    try:
        source_conn = sqlite3.connect(str(source_db), timeout=5.0)
        source_conn.execute("PRAGMA busy_timeout = 5000")
        dest_conn = sqlite3.connect(str(destination_db), timeout=5.0)
        dest_conn.execute("PRAGMA busy_timeout = 5000")

        # SQLite standard-library backup API.
        source_conn.backup(dest_conn)

        # Verify the staged/copied destination.
        if not _integrity_ok(dest_conn):
            raise RuntimeError(
                f"Backup verification failed: integrity_check on "
                f"{destination_db} returned non-ok"
            )

        source_version = _user_version(source_conn)
        dest_version = _user_version(dest_conn)
        if source_version != dest_version:
            raise RuntimeError(
                f"Backup verification failed: user_version mismatch "
                f"(source={source_version}, dest={dest_version})"
            )

        dest_conn.commit()
        logger.info(
            "Backup verified: %s -> %s (user_version=%s)",
            source_db, destination_db, dest_version,
        )
    except BaseException:
        # On failure, remove the incomplete destination so it is not mistaken
        # for a valid backup.
        if destination_db.exists():
            try:
                destination_db.unlink()
            except OSError:
                logger.debug("Could not remove incomplete backup", exc_info=True)
        raise
    finally:
        if dest_conn is not None:
            dest_conn.close()
        if source_conn is not None:
            source_conn.close()
