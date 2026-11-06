"""Phase 8C — SignalStore persistence (A17–A20).

Locked behaviour (§17.1 / integration):
  - SQLite is the sole authoritative signal store.
  - Explicit write transaction: BEGIN IMMEDIATE.
  - Retain returned eligible batches in memory until committed.
  - Do not call evaluate_once() again while a batch remains unpersisted.
  - Retry policy on transient write failures.
  - JSON and JSONL are exports only, never primary persistence.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from unittest.mock import patch

from unittest.mock import patch

import pytest

from cpab.models import Observation, Signal
from cpab.storage.store import SignalStore


def _make_observation(oid="obs-1", **overrides):
    data = dict(
        id=oid, type="bullish_engulfing", symbol="R_100", timeframe="1m",
        candle_epoch=1700000000, detected_at=time.time(),
        event_time=1700000000, confirmed_at=1700000000, developing=False,
        direction="bullish", variant=None, detector_id="engulfing",
        detector_config_id=None, metadata={}, payload={}, lifecycle_state=None,
    )
    data.update(overrides)
    return Observation(**data)


def _make_signal(sid="sig-1", **overrides):
    obs = _make_observation()
    data = dict(
        id=sid, condition_id="cond-1", symbol="R_100", timeframe="1m",
        candle_epoch=1700000000, generated_at=time.time(),
        supporting_observations=[obs], reasons=["r1"], metadata={},
    )
    data.update(overrides)
    return Signal(**data)


# -- A17: save_signals persists signals + observations -------------------

def test_save_signals_persists_signal_and_observations(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    store = SignalStore(db)
    store.open()
    try:
        sig = _make_signal()
        n = store.save_signals([sig])
        assert n == 1
        assert store.signal_count() == 1
        assert store.observation_count() == 1

        row = store.get_signal(sig.id)
        assert row is not None
        assert row["condition_id"] == sig.condition_id
        assert row["symbol"] == sig.symbol
        assert row["fingerprint"] is not None
    finally:
        store.close()


def test_save_signals_idempotent_observations(tmp_path):
    """Same observation referenced by multiple signals is stored once."""
    db = str(tmp_path / "store.sqlite3")
    store = SignalStore(db)
    store.open()
    try:
        obs = _make_observation()
        sig_a = _make_signal(sid="sig-a", supporting_observations=[obs])
        sig_b = _make_signal(sid="sig-b", supporting_observations=[obs])
        store.save_signals([sig_a, sig_b])
        assert store.signal_count() == 2
        assert store.observation_count() == 1  # same obs, stored once
    finally:
        store.close()


def test_save_signals_empty_batch(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    store = SignalStore(db)
    store.open()
    try:
        assert store.save_signals([]) == 0
        assert store.signal_count() == 0
    finally:
        store.close()


# -- A18: retry on transient failure -------------------------------------

def test_retry_on_operational_error(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    store = SignalStore(db)
    store.open()
    try:
        sig = _make_signal()
        call_count = {"n": 0}
        original = store._save_signals_once

        def flaky(signals):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return original(signals)

        with patch.object(store, "_save_signals_once", side_effect=flaky):
            n = store.save_signals([sig])
            assert n == 1
            assert call_count["n"] >= 2
    finally:
        store.close()


# -- A19: one-shot failure path -> nonzero exit (covered in integration) --

# -- A20: store schema version -------------------------------------------

def test_store_schema_version(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    store = SignalStore(db)
    store.open()
    try:
        assert store.get_schema_version() == 1
    finally:
        store.close()
