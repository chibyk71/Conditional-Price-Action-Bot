"""Shared Phase 8B acceptance evidence registries (A15)."""

from __future__ import annotations

# Stall matrix: (control, state_at_s, fault_class, d) -> status
STALL_EVIDENCE: dict[tuple, str] = {}

# A6 outcome fixtures: fixture_id -> status
A6_EVIDENCE: dict[str, str] = {}

# Full mandatory A6 set per revision 8 section 6b (completion/invalidation;
# F2/F3 expiry deferred by contract).
A6_MANDATORY_KEYS: tuple[str, ...] = (
    # Merged Phase 8A injected-observation outcomes (still required)
    "phase8a-completion",
    "phase8a-invalidation",
    "phase8a-expiry",
    # F1 real-detector
    "F1-completion-real",
    "F1-invalidation-real",
    "F1-expiry-real",
    # F2 real-detector with delayed confirmation
    "F2-completion-real",
    "F2-invalidation-real",
    # F3 real-detector
    "F3-completion-real",
    "F3-invalidation-real",
)
