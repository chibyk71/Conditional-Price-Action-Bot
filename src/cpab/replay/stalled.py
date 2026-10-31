"""Report-only STALLED classification (Phase 8B section 3).

Does not alter lifecycle or continuity policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from cpab.replay.driver import CycleRecord, ReplayResult

CONTINUITY_DIAGNOSTIC = "CONTINUITY_DIAGNOSTIC"
STALLED = "STALLED"
STALL_CLEARED = "STALL_CLEARED"


def _is_continuity_diagnostic(rec: CycleRecord, setup_id: str) -> bool:
    """True only when diagnostic-only *and* a continuity reason is present.

    Generic diagnostic-only states without continuity markers are not
    treated as continuity diagnostics (avoids over-classification).
    Readiness-blocked cycles are never continuity diagnostics.
    """
    if rec.evaluation_blocked:
        return False
    if not rec.setup_diagnostic_only.get(setup_id, False):
        return False
    reasons = rec.setup_reasons.get(setup_id, ())
    text = " ".join(reasons).lower()
    markers = (
        "continu",
        "expected epoch",
        "expected_epoch",
        "gap",
        "discontinuous",
        "timeframe_seconds",
        "out of order",
        "out-of-order",
        "missing candle",
        "series_ok reported false",
        "data continuity",
    )
    if any(m in text for m in markers):
        return True
    # Unhealthy series (data_ok=False) with diagnostic-only is continuity-
    # related when the engine surfaces health diagnostics.
    if not rec.data_ok and rec.data_diagnostics:
        return True
    return False


def _parse_expected_epoch(reasons: tuple[str, ...]) -> Optional[int]:
    """Best-effort extract of expected epoch from engine reason strings."""
    import re

    for r in reasons:
        m = re.search(r"expected(?:\s+epoch)?[=\s:]+(\d+)", r, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"expected_epoch[=\s:]+(\d+)", r, re.IGNORECASE)
        if m:
            return int(m.group(1))
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
      j = first cycle where lifecycle returns diagnostic_only with continuity reason.
      STALLED at first k > j where instance non-terminal, a further candle arrived,
      and cycle k is also diagnostic_only for continuity.
      Until then the report shows CONTINUITY_DIAGNOSTIC.
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
        if rec.evaluation_blocked or (
            rec.injected_fault == "readiness_blocked"
        ):
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
                report.first_discontinuity_actual_epoch = rec.candle_epoch
                report.first_discontinuity_last_accepted_epoch = rec.setup_anchors.get(
                    setup_id
                )
                report.first_discontinuity_expected_epoch = _parse_expected_epoch(
                    reasons
                )
                report.classification = CONTINUITY_DIAGNOSTIC
            elif k is None and not terminal:
                # First further diagnostic cycle after j → STALLED.
                k = rec.cycle_index
                report.stalled_classified_at = k
                report.classification = STALLED
                subsequent.append(k)
            elif k is not None and not terminal:
                subsequent.append(rec.cycle_index)
        elif j is not None and not terminal and not is_cont:
            # Unexpected return to non-diagnostic after j.
            report.classification = STALL_CLEARED
            report.stall_cleared_cycle = rec.cycle_index
            break

    report.subsequent_diagnostic_cycles = subsequent
    report.subsequent_diagnostic_count = len(subsequent)
    report.blocked_cycles = blocked
    report.engine_observed_reasons = list(dict.fromkeys(engine_reasons))
    return report
