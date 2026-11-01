"""G2: A15 run-identity strictness tests (in-memory clones; do not wipe session manifest)."""
from __future__ import annotations

import copy
import re

import pytest

from cpab.replay import evidence as ev


def test_begin_acceptance_run_uses_actual_head():
    m = ev.load_manifest()
    run = m.get("run") or {}
    head = ev._git_commit()
    assert run.get("commit") == head
    assert re.fullmatch(r"[0-9a-f]{40}", str(run.get("commit") or ""))
    assert run.get("working_tree_clean") is True
    assert str(run.get("git_status_porcelain", "x")).strip() == ""
    assert re.fullmatch(r"[0-9a-f]{64}", str(run.get("dependency_set_sha256") or ""))
    assert isinstance(run.get("pytest_args"), list) and run["pytest_args"]
    assert run.get("talib_version")
    assert run.get("talib_file")
    assert run.get("talib_extension")
    assert run.get("run_id")
    assert run.get("python")
    ev.validate_run_identity(m)


def test_begin_acceptance_run_rejects_commit_kwarg():
    with pytest.raises(TypeError):
        ev.begin_acceptance_run(commit="a" * 40, note="x", pytest_args=["pytest"])  # type: ignore[call-arg]


def test_validate_run_identity_rejects_mismatched_commit():
    m = copy.deepcopy(ev.load_manifest())
    m.setdefault("run", {})["commit"] = "0" * 40
    with pytest.raises(AssertionError, match="does not match"):
        ev.validate_run_identity(m)


def test_validate_run_identity_rejects_unknown_commit():
    m = copy.deepcopy(ev.load_manifest())
    m.setdefault("run", {})["commit"] = "unknown"
    with pytest.raises(AssertionError, match="invalid commit"):
        ev.validate_run_identity(m)


def test_validate_run_identity_rejects_invalid_dep_hash():
    m = copy.deepcopy(ev.load_manifest())
    m.setdefault("run", {})["dependency_set_sha256"] = "not-a-hash"
    with pytest.raises(AssertionError, match="dependency_set_sha256"):
        ev.validate_run_identity(m)


def test_validate_run_identity_rejects_empty_pytest_args():
    m = copy.deepcopy(ev.load_manifest())
    m.setdefault("run", {})["pytest_args"] = []
    with pytest.raises(AssertionError, match="pytest_args"):
        ev.validate_run_identity(m)


def test_validate_run_identity_rejects_missing_talib_extension():
    m = copy.deepcopy(ev.load_manifest())
    m.setdefault("run", {})["talib_extension"] = None
    with pytest.raises(AssertionError, match="talib_extension"):
        ev.validate_run_identity(m)


def test_validate_run_identity_rejects_dirty_porcelain():
    m = copy.deepcopy(ev.load_manifest())
    m.setdefault("run", {})["git_status_porcelain"] = " M src/cpab/replay/evidence.py\n"
    m["run"]["working_tree_clean"] = False
    with pytest.raises(AssertionError):
        ev.validate_run_identity(m)
