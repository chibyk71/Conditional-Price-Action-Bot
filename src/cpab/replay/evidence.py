"""Shared Phase 8B acceptance evidence registries (A15).

Evidence is written both in-memory and to a durable JSON manifest so acceptance
does not depend on pytest collection order within a single process, and a
complete record survives the test run for review.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Stall matrix: (control, state_at_s, fault_class, d) -> status
STALL_EVIDENCE: dict[tuple, str] = {}

# A6 outcome fixtures: fixture_id -> status
A6_EVIDENCE: dict[str, str] = {}

A6_MANDATORY_KEYS: tuple[str, ...] = (
    "phase8a-completion",
    "phase8a-invalidation",
    "phase8a-expiry",
    "F1-completion-real",
    "F1-invalidation-real",
    "F1-expiry-real",
    "F2-completion-real",
    "F2-invalidation-real",
    "F3-completion-real",
    "F3-invalidation-real",
)

# Durable manifest path (overridable for CI).
_DEFAULT_MANIFEST = Path(
    os.environ.get(
        "CPAB_PHASE8B_EVIDENCE_PATH",
        # Default outside the repo tree so a checked-in success file cannot
        # independently certify acceptance. Override for durable CI artifacts.
        str(Path("/tmp") / "cpab_phase8b_a15_evidence.json"),
    )
)


def _manifest_path() -> Path:
    return Path(
        os.environ.get("CPAB_PHASE8B_EVIDENCE_PATH", str(_DEFAULT_MANIFEST))
    )


def record_a6(fixture_id: str, status: str, *, extra: dict[str, Any] | None = None) -> None:
    A6_EVIDENCE[fixture_id] = status
    _persist(fixture_id, status, kind="a6", extra=extra)


def record_stall(key: tuple, status: str, *, extra: dict[str, Any] | None = None) -> None:
    STALL_EVIDENCE[key] = status
    # Always persist under str(key) so meta-tests can load without memory.
    _persist(str(key), status, kind="stall", extra=extra)


def _persist(key: str, status: str, *, kind: str, extra: dict[str, Any] | None) -> None:
    path = _manifest_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"a6": {}, "stall": {}, "run": {}}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise RuntimeError(f"A15 evidence manifest invalid type: {path}")
        bucket = data.setdefault(kind, {})
        entry: dict[str, Any] = {"status": status}
        if extra:
            entry["extra"] = extra
        bucket[key] = entry
        # Also mirror full in-memory maps
        data["a6_memory"] = dict(A6_EVIDENCE)
        data["stall_memory"] = {str(k): v for k, v in STALL_EVIDENCE.items()}
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        # Surface persistence failures so A15 cannot silently degrade.
        raise RuntimeError(f"A15 evidence manifest write failed: {path}: {exc}") from exc


def load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {"a6": {}, "stall": {}, "run": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"A15 evidence manifest corrupt: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"A15 evidence manifest invalid type: {path}")
    return data


def begin_acceptance_run(*, commit: str | None = None, note: str = "") -> None:
    """Stamp the current acceptance run identity into the manifest."""
    import platform
    import time
    try:
        import talib
        talib_ver = getattr(talib, "__version__", "unknown")
    except Exception:
        talib_ver = "unavailable"
    clear_evidence()
    path = _manifest_path()
    data = {
        "a6": {},
        "stall": {},
        "run": {
            "started_at": time.time(),
            "commit": commit or os.environ.get("GIT_COMMIT", os.environ.get("GITHUB_SHA", "unknown")),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "talib": talib_ver,
            "note": note,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def clear_evidence() -> None:
    """Reset in-memory and on-disk evidence for a clean acceptance run."""
    STALL_EVIDENCE.clear()
    A6_EVIDENCE.clear()
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"a6": {}, "stall": {}}\n', encoding="utf-8")
