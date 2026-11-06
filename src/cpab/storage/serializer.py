"""Serialization and fingerprinting for signals and observations.

The fingerprint is a SHA-256 digest computed over the durable (non-transient)
fields of a signal and its supporting observations. Per §17.1 the fingerprint
exclusion list is preserved exactly:

  - Signal.generated_at  (wall-clock emission time — not part of identity)
  - each supporting Observation.detected_at  (wall-clock detection time)

JSON fields are serialized with ``sort_keys=True`` for deterministic output.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cpab.models import Observation, Signal


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def serialize_signal(sig: Signal) -> dict[str, Any]:
    """Serialize a :class:`Signal` into a flat dict of SQL column values."""
    return {
        "id": sig.id,
        "condition_id": sig.condition_id,
        "symbol": sig.symbol,
        "timeframe": sig.timeframe,
        "candle_epoch": int(sig.candle_epoch),
        "generated_at": float(sig.generated_at),
        "reasons": _json(sig.reasons),
        "metadata": _json(sig.metadata),
    }


def serialize_observation(obs: Observation) -> dict[str, Any]:
    """Serialize an :class:`Observation` into a flat dict of SQL column values."""
    return {
        "id": obs.id,
        "type": obs.type,
        "symbol": obs.symbol,
        "timeframe": obs.timeframe,
        "candle_epoch": int(obs.candle_epoch),
        "detected_at": float(obs.detected_at),
        "event_time": obs.event_time,
        "confirmed_at": obs.confirmed_at,
        "developing": int(bool(obs.developing)),
        "direction": obs.direction,
        "variant": obs.variant,
        "detector_id": obs.detector_id,
        "detector_config_id": obs.detector_config_id,
        "metadata": _json(obs.metadata) if obs.metadata else None,
        "payload": _json(obs.payload) if obs.payload else None,
        "lifecycle_state": obs.lifecycle_state,
    }


def compute_fingerprint(sig: Signal, supporting: list[Observation]) -> str:
    """Compute a deterministic SHA-256 fingerprint.

    Excludes ``Signal.generated_at`` and each supporting
    ``Observation.detected_at`` per the locked exclusion list in §17.1.
    """
    h = hashlib.sha256()

    # Signal fields (generated_at excluded).
    h.update(b"sig|")
    h.update(sig.id.encode("utf-8"))
    h.update(b"|")
    h.update(sig.condition_id.encode("utf-8"))
    h.update(b"|")
    h.update(sig.symbol.encode("utf-8"))
    h.update(b"|")
    h.update(sig.timeframe.encode("utf-8"))
    h.update(b"|")
    h.update(str(int(sig.candle_epoch)).encode("ascii"))
    h.update(b"|")
    h.update(_json(sig.reasons).encode("utf-8"))
    h.update(b"|")
    h.update(_json(sig.metadata).encode("utf-8"))
    h.update(b"|")

    # Supporting observations (detected_at excluded), ordered by id for stability.
    for obs in sorted(supporting, key=lambda o: o.id):
        h.update(b"obs|")
        h.update(obs.id.encode("utf-8"))
        h.update(b"|")
        h.update(obs.type.encode("utf-8"))
        h.update(b"|")
        h.update(obs.symbol.encode("utf-8"))
        h.update(b"|")
        h.update(obs.timeframe.encode("utf-8"))
        h.update(b"|")
        h.update(str(int(obs.candle_epoch)).encode("ascii"))
        h.update(b"|")
        # detected_at deliberately excluded.
        h.update(b"|")
        h.update(
            str(obs.event_time).encode("ascii")
            if obs.event_time is not None
            else b""
        )
        h.update(b"|")
        h.update(
            str(obs.confirmed_at).encode("ascii")
            if obs.confirmed_at is not None
            else b""
        )
        h.update(b"|")
        h.update(str(int(bool(obs.developing))).encode("ascii"))
        h.update(b"|")
        h.update((obs.direction or "").encode("utf-8"))
        h.update(b"|")
        h.update((obs.variant or "").encode("utf-8"))
        h.update(b"|")
        h.update((obs.detector_id or "").encode("utf-8"))
        h.update(b"|")
        h.update((obs.detector_config_id or "").encode("utf-8"))
        h.update(b"|")
        h.update(_json(obs.metadata).encode("utf-8") if obs.metadata else b"")
        h.update(b"|")
        h.update(_json(obs.payload).encode("utf-8") if obs.payload else b"")
        h.update(b"|")
        h.update((obs.lifecycle_state or "").encode("utf-8"))
        h.update(b"|")

    return h.hexdigest()
