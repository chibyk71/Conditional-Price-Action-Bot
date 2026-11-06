"""Application-level signal store backed by SQLite.

:class:`SignalStore` is the persistence consumer at the signal-output boundary.
It persists signals together with their supporting observations in a single
explicit ``BEGIN IMMEDIATE`` write transaction, using idempotent observation
inserts so the same observation can be referenced by many signals.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any
from pathlib import Path

from cpab.logging_setup import get_logger
from cpab.models import Observation, Signal
from cpab.storage.database import connect
from cpab.storage.serializer import (
    compute_fingerprint,
    serialize_observation,
    serialize_signal,
)

logger = get_logger("cpab.storage.store")

# Locked retry policy for persistence write failures.
_MAX_RETRIES = 3
_BASE_BACKOFF_SEC = 0.25

_INSERT_SIGNAL = """
INSERT OR REPLACE INTO signals
    (id, condition_id, symbol, timeframe, candle_epoch, generated_at, reasons, metadata, fingerprint)
VALUES (:id, :condition_id, :symbol, :timeframe, :candle_epoch, :generated_at, :reasons, :metadata, :fingerprint)
"""

_INSERT_OBSERVATION = """
INSERT OR IGNORE INTO observations
    (id, type, symbol, timeframe, candle_epoch, detected_at, event_time, confirmed_at,
     developing, direction, variant, detector_id, detector_config_id, metadata, payload, lifecycle_state)
VALUES (:id, :type, :symbol, :timeframe, :candle_epoch, :detected_at, :event_time, :confirmed_at,
        :developing, :direction, :variant, :detector_id, :detector_config_id, :metadata, :payload, :lifecycle_state)
"""

_INSERT_SIGNAL_OBS = """
INSERT OR IGNORE INTO signal_observations (signal_id, observation_id)
VALUES (:signal_id, :observation_id)
"""


class SignalStore:
    """SQLite-backed store for signals and their supporting observations.

    All writes occur within an explicit ``BEGIN IMMEDIATE`` transaction so
    concurrent writers serialize. Observation inserts are idempotent
    (``INSERT OR IGNORE``) so re-persisting the same observation is safe.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "SignalStore":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def open(self) -> None:
        if self._conn is not None:
            return
        self._conn = connect(self.db_path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SignalStore is not open; call open() first")
        return self._conn

    def save_signals(self, signals: list[Signal]) -> int:
        """Persist a batch of signals with their supporting observations."""
        if self._conn is None:
            raise RuntimeError("SignalStore is not open; call open() first")

        last_error: sqlite3.Error | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._save_signals_once(signals)
            except sqlite3.OperationalError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    sleep_for = _BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                    logger.warning(
                        "SignalStore write attempt %s/%s failed (%s); retrying "
                        "in %.2fs", attempt, _MAX_RETRIES, exc, sleep_for
                    )
                    time.sleep(sleep_for)
                else:
                    logger.error(
                        "SignalStore write failed after %s attempts: %s",
                        _MAX_RETRIES, exc,
                    )
        assert last_error is not None
        raise last_error

    def _save_signals_once(self, signals: list[Signal]) -> int:
        """Execute one BEGIN IMMEDIATE write transaction for a batch."""
        if not signals:
            return 0

        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            written = 0
            for sig in signals:
                obs_list = list(sig.supporting_observations)
                fingerprint = compute_fingerprint(sig, obs_list)

                signal_row = serialize_signal(sig)
                signal_row["fingerprint"] = fingerprint
                conn.execute(_INSERT_SIGNAL, signal_row)

                for obs in obs_list:
                    conn.execute(_INSERT_OBSERVATION, serialize_observation(obs))
                    conn.execute(
                        _INSERT_SIGNAL_OBS,
                        {"signal_id": sig.id, "observation_id": obs.id},
                    )
                written += 1

            conn.commit()
            logger.info(
                "Persisted %s signal(s) to %s", written, self.db_path
            )
            return written
        except BaseException:
            try:
                conn.rollback()
            except sqlite3.Error:
                logger.debug("Rollback failed during error handling", exc_info=True)
            raise

    def signal_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()
        return int(row["c"]) if row is not None else 0

    def observation_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM observations").fetchone()
        return int(row["c"]) if row is not None else 0

    def get_signal(self, signal_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM signals WHERE id = ?", (signal_id,)
        ).fetchone()

    def get_schema_version(self) -> int:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row["value"]) if row is not None else 0
