"""Shared Phase 8B acceptance evidence registries (A15)."""

from __future__ import annotations

# Stall matrix: (control, state_at_s, fault_class, d) -> status
STALL_EVIDENCE: dict[tuple, str] = {}

# A6 outcome fixtures: fixture_id -> status
A6_EVIDENCE: dict[str, str] = {}

A6_MANDATORY_KEYS: tuple[str, ...] = (
    "F3-engulfing-completion-real",
    "F3-engulfing-completion-live-style",
)
