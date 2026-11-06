"""Phase 8C — serializer and fingerprint (A13–A16).

Locked behaviour (§17.1):
  - Exact fingerprint exclusion list: Signal.generated_at and each supporting
    Observation.detected_at.
  - No other exclusions or indexes without an approved contract change.
"""

from __future__ import annotations

import time

import pytest

from cpab.models import Observation, Signal
from cpab.storage.serializer import (
    compute_fingerprint,
    serialize_observation,
    serialize_signal,
)


def _make_observation(oid="obs-1", detected_at=1000000.0, **overrides):
    data = dict(
        id=oid, type="bullish_engulfing", symbol="R_100", timeframe="1m",
        candle_epoch=1700000000, detected_at=detected_at,
        event_time=1700000000, confirmed_at=1700000000, developing=False,
        direction="bullish", variant=None, detector_id="engulfing",
        detector_config_id=None, metadata={"k": "v"}, payload={"p": 1},
        lifecycle_state=None,
    )
    data.update(overrides)
    return Observation(**data)


def _make_signal(sid="sig-1", generated_at=1000000.0, **overrides):
    data = dict(
        id=sid, condition_id="cond-1", symbol="R_100", timeframe="1m",
        candle_epoch=1700000000, generated_at=generated_at,
        supporting_observations=[], reasons=["r1"], metadata={"m": 1},
    )
    data.update(overrides)
    return Signal(**data)


# -- A13: serializer round-trip ------------------------------------------

def test_serialize_signal_round_trip():
    obs = _make_observation()
    sig = _make_signal(supporting_observations=[obs])
    row = serialize_signal(sig)
    assert row["id"] == sig.id
    assert row["condition_id"] == sig.condition_id
    assert row["symbol"] == sig.symbol
    assert row["timeframe"] == sig.timeframe
    assert row["candle_epoch"] == sig.candle_epoch
    assert row["generated_at"] == sig.generated_at
    import json
    assert json.loads(row["reasons"]) == sig.reasons
    assert json.loads(row["metadata"]) == sig.metadata


def test_serialize_observation_round_trip():
    obs = _make_observation()
    row = serialize_observation(obs)
    assert row["id"] == obs.id
    assert row["type"] == obs.type
    assert row["symbol"] == obs.symbol
    assert row["timeframe"] == obs.timeframe
    assert row["candle_epoch"] == obs.candle_epoch
    assert row["detected_at"] == obs.detected_at
    assert row["event_time"] == obs.event_time
    assert row["confirmed_at"] == obs.confirmed_at
    assert row["developing"] == int(obs.developing)


# -- A14: fingerprint excludes Signal.generated_at -----------------------

def test_fingerprint_excludes_generated_at():
    obs = _make_observation()
    sig_a = _make_signal(generated_at=1000000.0, supporting_observations=[obs])
    sig_b = _make_signal(generated_at=999999999.0, supporting_observations=[obs])
    fp_a = compute_fingerprint(sig_a, [obs])
    fp_b = compute_fingerprint(sig_b, [obs])
    assert fp_a == fp_b
    assert len(fp_a) == 64  # SHA-256 hex


# -- A15: fingerprint excludes each supporting Observation.detected_at ---

def test_fingerprint_excludes_detected_at():
    obs_a = _make_observation(detected_at=1000000.0)
    obs_b = _make_observation(detected_at=999999999.0)
    sig = _make_signal()
    fp_a = compute_fingerprint(sig, [obs_a])
    fp_b = compute_fingerprint(sig, [obs_b])
    assert fp_a == fp_b


# -- A16: fingerprint is deterministic and sensitive to included fields --

def test_fingerprint_deterministic():
    obs = _make_observation()
    sig = _make_signal(supporting_observations=[obs])
    fp1 = compute_fingerprint(sig, [obs])
    fp2 = compute_fingerprint(sig, [obs])
    assert fp1 == fp2


def test_fingerprint_changes_with_signal_fields():
    obs = _make_observation()
    sig_a = _make_signal(condition_id="cond-a", supporting_observations=[obs])
    sig_b = _make_signal(condition_id="cond-b", supporting_observations=[obs])
    assert compute_fingerprint(sig_a, [obs]) != compute_fingerprint(sig_b, [obs])


def test_fingerprint_changes_with_observation_fields():
    obs_a = _make_observation(oid="obs-a", direction="bullish")
    obs_b = _make_observation(oid="obs-b", direction="bearish")
    sig = _make_signal()
    assert compute_fingerprint(sig, [obs_a]) != compute_fingerprint(sig, [obs_b])
