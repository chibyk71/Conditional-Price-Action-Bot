"""Shared Phase 8B acceptance evidence registries (A15).

Evidence is written both in-memory and to a durable JSON manifest so acceptance
does not depend on pytest collection order within a single process, and a
complete record survives the test run for review.

A15 rules (G2):
- Commit identity must be exact 40-hex git rev-parse HEAD; "unknown" is rejected.
- Working tree must be clean (git status --porcelain empty).
- Each run starts with a fresh manifest and a unique run_id.
- Meta-tests validate the on-disk manifest only (no in-memory copy fallback).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
import uuid
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

# Designated sole writers for each A6 key (module::function).
A6_DESIGNATED_WRITERS: dict[str, str] = {
    "phase8a-completion": "tests/test_phase8b_real_detector_outcomes.py::test_a6_phase8a_completion",
    "phase8a-invalidation": "tests/test_phase8b_real_detector_outcomes.py::test_a6_phase8a_invalidation",
    "phase8a-expiry": "tests/test_phase8b_real_detector_outcomes.py::test_a6_phase8a_expiry",
    "F1-completion-real": "tests/test_phase8b_seed_fixtures.py::test_a6_f1_completion_seed3",
    "F1-invalidation-real": "tests/test_phase8b_seed_fixtures.py::test_a6_f1_invalidation_seed1",
    "F1-expiry-real": "tests/test_phase8b_seed_fixtures.py::test_a6_f1_expiry_seed4",
    "F2-completion-real": "tests/test_phase8b_seed_fixtures.py::test_a6_f2_completion_seed4",
    "F2-invalidation-real": "tests/test_phase8b_seed_fixtures.py::test_a6_f2_invalidation_seed7",
    "F3-completion-real": "tests/test_phase8b_seed_fixtures.py::test_a6_f3_completion_seed1",
    "F3-invalidation-real": "tests/test_phase8b_seed_fixtures.py::test_a6_f3_invalidation_seed23",
}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_DEFAULT_MANIFEST = Path(
    os.environ.get(
        "CPAB_PHASE8B_EVIDENCE_PATH",
        str(Path("/tmp") / "cpab_phase8b_a15_evidence.json"),
    )
)

# Current run identity (set by begin_acceptance_run).
_CURRENT_RUN_ID: str | None = None


def _manifest_path() -> Path:
    return Path(os.environ.get("CPAB_PHASE8B_EVIDENCE_PATH", str(_DEFAULT_MANIFEST)))


def current_run_id() -> str | None:
    return _CURRENT_RUN_ID


def _git_commit() -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
    if not _COMMIT_RE.match(out):
        raise RuntimeError(f"A15: git rev-parse HEAD is not 40-hex: {out!r}")
    return out


def _git_porcelain() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
    )


def _dependency_set_hash() -> str:
    """SHA-256 of sorted pip freeze lines (deterministic)."""
    raw = subprocess.check_output(
        ["pip", "freeze"], text=True, stderr=subprocess.DEVNULL
    )
    lines = sorted(line.strip() for line in raw.splitlines() if line.strip())
    payload = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def begin_acceptance_run(
    *,
    commit: str | None = None,
    note: str = "",
    pytest_args: list[str] | None = None,
) -> dict[str, Any]:
    """Start a fresh A15 run: clear evidence, stamp run identity, reject dirty trees."""
    global _CURRENT_RUN_ID

    resolved_commit = commit or _git_commit()
    if resolved_commit == "unknown" or not _COMMIT_RE.match(resolved_commit):
        raise RuntimeError(
            f"A15: commit identity must be exact 40-hex SHA; got {resolved_commit!r}"
        )

    porcelain = _git_porcelain()
    clean = porcelain.strip() == ""
    dep_hash = _dependency_set_hash()

    talib_meta: dict[str, Any] = {}
    try:
        from cpab.replay.real_talib import real_talib_info

        talib_meta = real_talib_info()
    except Exception as exc:
        talib_meta = {"talib_error": str(exc)}

    run_id = str(uuid.uuid4())
    _CURRENT_RUN_ID = run_id

    STALL_EVIDENCE.clear()
    A6_EVIDENCE.clear()

    data = {
        "a6": {},
        "stall": {},
        "run": {
            "run_id": run_id,
            "started_at": time.time(),
            "commit": resolved_commit,
            "working_tree_clean": clean,
            "git_status_porcelain": porcelain,
            "dependency_set_sha256": dep_hash,
            "pytest_args": list(pytest_args or []),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "talib_version": talib_meta.get("talib_version"),
            "talib_file": talib_meta.get("talib_file"),
            "talib_extension": talib_meta.get("talib_extension"),
            "note": note,
        },
    }
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data["run"]


def record_a6(fixture_id: str, status: str, *, extra: dict[str, Any] | None = None) -> None:
    A6_EVIDENCE[fixture_id] = status
    _persist(fixture_id, status, kind="a6", extra=extra)


def record_stall(key: tuple, status: str, *, extra: dict[str, Any] | None = None) -> None:
    STALL_EVIDENCE[key] = status
    _persist(str(key), status, kind="stall", extra=extra)


def _persist(key: str, status: str, *, kind: str, extra: dict[str, Any] | None) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"a6": {}, "stall": {}, "run": {}}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError(f"A15 evidence manifest invalid type: {path}")
        data = loaded
    bucket = data.setdefault(kind, {})
    entry: dict[str, Any] = {
        "status": status,
        "run_id": _CURRENT_RUN_ID,
        "recorded_at": time.time(),
    }
    if extra:
        entry["extra"] = extra
    bucket[key] = entry
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {"a6": {}, "stall": {}, "run": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"A15 evidence manifest invalid type: {path}")
    return data


def clear_evidence() -> None:
    """Reset in-memory and on-disk evidence (does not stamp a new run)."""
    global _CURRENT_RUN_ID
    STALL_EVIDENCE.clear()
    A6_EVIDENCE.clear()
    _CURRENT_RUN_ID = None
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"a6": {}, "stall": {}, "run": {}}\n', encoding="utf-8")


def validate_run_identity(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fail closed if run identity is incomplete or invalid."""
    data = manifest if manifest is not None else load_manifest()
    run = data.get("run") or {}
    commit = run.get("commit")
    if not commit or not _COMMIT_RE.match(str(commit)):
        raise AssertionError(f"A15 invalid commit identity: {commit!r}")
    if run.get("working_tree_clean") is not True:
        raise AssertionError(
            f"A15 working tree is dirty or cleanliness unset: "
            f"clean={run.get('working_tree_clean')!r} porcelain={run.get('git_status_porcelain')!r}"
        )
    if not run.get("dependency_set_sha256"):
        raise AssertionError("A15 missing dependency_set_sha256")
    if not run.get("talib_version") or not run.get("talib_file"):
        raise AssertionError(f"A15 missing TA-Lib identity: {run}")
    if not run.get("run_id"):
        raise AssertionError("A15 missing run_id")
    if not run.get("python"):
        raise AssertionError("A15 missing python version")
    return run
