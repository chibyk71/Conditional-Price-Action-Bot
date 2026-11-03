"""Report-only STALLED classification (Phase 8B section 3).

Does not alter lifecycle or continuity policy.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from cpab.replay.driver import CycleRecord, ReplayResult

CONTINUITY_DIAGNOSTIC = "CONTINUITY_DIAGNOSTIC"
STALLED = "STALLED"
STALL_CLEARED = "STALL_CLEARED"

# Exact engine format from lifecycle.py:
# "discontinuous evaluation candle {actual}; expected {expected} after last accepted
#  {last} (timeframe_seconds={tf})"
_CONTINUITY_RE = re.compile(
    r"discontinuous evaluation candle (\d+);\s*expected (\d+) after last accepted (\d+)"
    r"(?:\s*\(timeframe_seconds=(\d+)\))?",
    re.IGNORECASE,
)


def _continuity_match(reasons: tuple[str, ...]) -> Optional[re.Match[str]]:
    for r in reasons:
        m = _CONTINUITY_RE.search(r)
        if m:
            return m
    return None


def _is_continuity_diagnostic(rec: CycleRecord, setup_id: str) -> bool:
    """True only for diagnostic-only cycles with the engine continuity reason.

    Health-fault diagnostics (data_ok=False without epoch discontinuity) do NOT
    count as continuity diagnostics (contract section 3 / C2).
    Readiness-blocked cycles are never continuity diagnostics.
    """
    if rec.evaluation_blocked:
        return False
    if rec.injected_fault == "readiness_blocked":
        return False
    if not rec.setup_diagnostic_only.get(setup_id, False):
        return False
    reasons = rec.setup_reasons.get(setup_id, ())
    return _continuity_match(reasons) is not None


def _parse_expected_epoch(reasons: tuple[str, ...]) -> Optional[int]:
    m = _continuity_match(reasons)
    if m:
        return int(m.group(2))
    return None


def _parse_actual_epoch_from_reason(reasons: tuple[str, ...]) -> Optional[int]:
    m = _continuity_match(reasons)
    if m:
        return int(m.group(1))
    return None


def _parse_last_accepted_from_reason(reasons: tuple[str, ...]) -> Optional[int]:
    m = _continuity_match(reasons)
    if m:
        return int(m.group(3))
    return None


@dataclass
class StallReport:
    setup_id: str
    classification: str  # CONTINUITY_DIAGNOSTIC | STALLED | STALL_CLEARED | ""
    first_discontinuity_cycle: Optional[int] = None
    first_discontinuity_expected_epoch: Optional[int] = None
    first_discontinuity_actual_epoch: Optional[int] = None
    first_discontinuity_last_accepted_epoch: Optional[int] = None
    stalled_classified_at: Optional[int] = None
    subsequent_diagnostic_cycles: list[int] = field(default_factory=list)
    subsequent_diagnostic_count: int = 0
    blocked_cycles: list[int] = field(default_factory=list)
    injected_fault_cause: Optional[str] = None
    engine_observed_reasons: list[str] = field(default_factory=list)
    stall_cleared_cycle: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_stalled(
    result: ReplayResult,
    setup_id: str,
    *,
    injected_fault_cause: Optional[str] = None,
) -> StallReport:
    """Classify STALLED / CONTINUITY_DIAGNOSTIC / STALL_CLEARED from cycle records.

    Threshold (section 3):
      j = first cycle where lifecycle returns diagnostic_only with continuity reason
          (expected≠actual epoch). Health faults alone do not set j.
      STALLED at first k > j where instance non-terminal, a further candle arrived,
      and cycle k is also diagnostic_only for continuity.
      Readiness-blocked cycles neither start nor end the count.
    """
    report = StallReport(setup_id=setup_id, classification="")
    report.injected_fault_cause = injected_fault_cause

    j: Optional[int] = None
    k: Optional[int] = None
    subsequent: list[int] = []
    blocked: list[int] = []
    engine_reasons: list[str] = []

    for rec in result.cycles:
        if rec.evaluation_blocked or rec.injected_fault == "readiness_blocked":
            blocked.append(rec.cycle_index)
            continue

        is_cont = _is_continuity_diagnostic(rec, setup_id)
        state = rec.setup_states.get(setup_id, "PENDING")
        terminal = state in ("COMPLETED", "INVALIDATED", "EXPIRED")

        if is_cont:
            reasons = rec.setup_reasons.get(setup_id, ())
            engine_reasons.extend(reasons)
            if j is None:
                j = rec.cycle_index
                report.first_discontinuity_cycle = j
                report.first_discontinuity_actual_epoch = (
                    _parse_actual_epoch_from_reason(reasons) or rec.candle_epoch
                )
                report.first_discontinuity_last_accepted_epoch = (
                    _parse_last_accepted_from_reason(reasons)
                    or rec.setup_anchors.get(setup_id)
                )
                report.first_discontinuity_expected_epoch = _parse_expected_epoch(reasons)
                report.classification = CONTINUITY_DIAGNOSTIC
            elif k is None and not terminal:
                k = rec.cycle_index
                report.stalled_classified_at = k
                report.classification = STALLED
                subsequent.append(k)
            elif k is not None and not terminal:
                subsequent.append(rec.cycle_index)
        elif j is not None and k is not None and not terminal and not is_cont:
            # Recovery without continuity diagnostic → STALL_CLEARED
            if report.classification == STALLED:
                report.classification = STALL_CLEARED
                report.stall_cleared_cycle = rec.cycle_index

    report.subsequent_diagnostic_cycles = subsequent
    report.subsequent_diagnostic_count = len(subsequent)
    report.blocked_cycles = blocked
    report.engine_observed_reasons = engine_reasons
    return report
