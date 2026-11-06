"""Application-level SQLite restore with staged replacement.

Locked behaviour (§17.2):
  - The application must be stopped and the target database closed before
    restoration.
  - Preserve the existing database and any -journal sidecar before restoration.
  - If unexpected -wal or -shm sidecars exist, preserve them and halt
    restoration for investigation.
  - Restore through a staged file in the target directory; verify the staged
    copy before replacement.
  - Use explicit file-level replacement, atomically where supported. Do not
    use ATTACH/INSERT as the restore mechanism.
  - Preserve the pre-restore copy until post-restore verification succeeds.
  - If restoration or verification fails, do not restart the application.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from cpab.logging_setup import get_logger

logger = get_logger("cpab.storage.restore")


class RestoreAbortedError(RuntimeError):
    """Raised when restoration is intentionally halted for investigation."""


class RestoreVerificationError(RuntimeError):
    """Raised when staged-copy verification fails."""


def _sidecar_paths(db_path: Path) -> list[Path]:
    base = str(db_path)
    return [
        Path(base + "-journal"),
        Path(base + "-wal"),
        Path(base + "-shm"),
    ]


def _integrity_ok(conn: sqlite3.Connection) -> bool:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    return row is not None and str(row[0]).lower() == "ok"


def _user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def restore_database(
    staged_backup: str | Path,
    target_db: str | Path,
) -> None:
    """Restore the target database from a staged backup copy.

    The caller must ensure the application is stopped and the target database
    is closed before calling this function.

    The staged copy is verified (integrity_check + user_version against the
    pre-restore target) before replacement. The pre-restore copy is preserved
    until verification succeeds. On any failure, the application must not be
    restarted.
    """
    staged_backup = Path(staged_backup)
    target_db = Path(target_db)
    target_dir = target_db.resolve().parent

    if not staged_backup.exists():
        raise FileNotFoundError(f"Staged backup does not exist: {staged_backup}")
    if not staged_backup.is_file():
        raise RestoreAbortedError(
            f"Staged backup is not a regular file: {staged_backup}"
        )

    # Sidecar check: if unexpected -wal or -shm sidecars exist on the target,
    # preserve them and halt for investigation.
    target_sidecars = _sidecar_paths(target_db)
    unexpected = [p for p in target_sidecars[1:] if p.exists()]  # -wal, -shm
    if unexpected:
        names = ", ".join(str(p) for p in unexpected)
        raise RestoreAbortedError(
            "Unexpected SQLite sidecar(s) present on target database; "
            "preserving and halting restoration for investigation: " + names
        )

    # Open the staged backup and verify it before any replacement.
    staged_conn: Optional[sqlite3.Connection] = None
    try:
        staged_conn = sqlite3.connect(str(staged_backup), timeout=5.0)
        staged_conn.execute("PRAGMA busy_timeout = 5000")
        if not _integrity_ok(staged_conn):
            raise RestoreVerificationError(
                f"Staged backup failed integrity_check: {staged_backup}"
            )
        staged_version = _user_version(staged_conn)
        logger.info(
            "Staged backup verified: integrity_check=ok, user_version=%s",
            staged_version,
        )
    except sqlite3.Error as exc:
        raise RestoreVerificationError(
            f"Staged backup could not be opened/verified: {exc}"
        ) from exc
    finally:
        if staged_conn is not None:
            staged_conn.close()

    # Preserve the existing target database and its -journal sidecar.
    preserved: list[Path] = []
    if target_db.exists():
        preserved.append(target_db)
    for p in target_sidecars:
        if p.exists():
            preserved.append(p)

    # Staged replacement: copy the verified staged backup into the target
    # directory as a staged file, then replace atomically where supported.
    staged_in_target = target_dir / (target_db.name + ".stage-restore")
    preserved_paths: list[Path] = []
    try:
        shutil.copy2(str(staged_backup), str(staged_in_target))

        # Move pre-restore copies aside before replacement. They are retained
        # until post-restore verification succeeds.
        for p in preserved:
            preserved_path = target_dir / (p.name + ".pre-restore")
            shutil.move(str(p), str(preserved_path))
            preserved_paths.append(preserved_path)

        # Atomic file-level replacement where supported (os.replace).
        os.replace(str(staged_in_target), str(target_db))

        # Verify the replacement in place.
        verify_conn: Optional[sqlite3.Connection] = None
        try:
            verify_conn = sqlite3.connect(str(target_db), timeout=5.0)
            verify_conn.execute("PRAGMA busy_timeout = 5000")
            if not _integrity_ok(verify_conn):
                raise RestoreVerificationError(
                    f"Restored target {target_db} failed integrity_check"
                )
            restored_version = _user_version(verify_conn)
            if restored_version != staged_version:
                raise RestoreVerificationError(
                    "user_version mismatch after restore "
                    f"(staged={staged_version}, restored={restored_version})"
                )
            logger.info(
                "Restore verified: %s -> %s (user_version=%s)",
                staged_backup, target_db, restored_version,
            )
        finally:
            if verify_conn is not None:
                verify_conn.close()

        # Post-restore verification succeeded: drop pre-restore preserved copies.
        for p in preserved_paths:
            try:
                p.unlink()
            except OSError:
                logger.debug("Could not remove pre-restore copy %s", p, exc_info=True)

    except BaseException:
        # On failure, attempt to reinstate pre-restore copies and remove the
        # staged file. Do not restart the application.
        try:
            if staged_in_target.exists():
                staged_in_target.unlink()
        except OSError:
            logger.debug("Could not remove staged restore file", exc_info=True)
        for p in preserved_paths:
            original = target_dir / p.name.removesuffix(".pre-restore")
            try:
                if p.exists() and not original.exists():
                    shutil.move(str(p), str(original))
            except OSError:
                logger.debug(
                    "Could not reinstate pre-restore copy %s", p, exc_info=True
                )
        raise

