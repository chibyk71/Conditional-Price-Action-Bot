"""Replay vs live-style parity with effective-input comparison (R7-R8, A10-A11)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from cpab.replay.driver import CycleRecord, ReplayResult

INPUT_DIVERGENCE = "INPUT_DIVERGENCE"
ENGINE_DIVERGENCE = "ENGINE_DIVERGENCE"

# Sentinel: required effective-input metadata was never recorded.
# Distinct from explicit None and from an empty sequence.
MISSING: str = "__MISSING__"


@dataclass
class Divergence:
    cycle_index: int
    field: str
    expected: Any
    actual: Any
    kind: str  # INPUT_DIVERGENCE | ENGINE_DIVERGENCE


@dataclass
class ParityComparison:
    matching: bool
    divergences: list[Divergence] = field(default_factory=list)
    first_divergence: Optional[Divergence] = None
    input_divergence: bool = False
    engine_divergence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "matching": self.matching,
            "input_divergence": self.input_divergence,
            "engine_divergence": self.engine_divergence,
            "first_divergence": asdict(self.first_divergence)
            if self.first_divergence
            else None,
            "divergences": [asdict(d) for d in self.divergences],
        }


_REQUIRED_INPUT_KEYS: tuple[str, ...] = (
    "effective_candle_epochs",
    "effective_candle_ohlcs",
    "effective_detector_names",
    "effective_detector_configs",
    "data_ok",
    "readiness_state",
    "evaluation_blocked",
    "confirm_lag_bars",
    "cache_max_size",
    "htf_epochs",
    "htf_source",
    "htf_ohlcs",
    "completed_count",
)


def _mon_get(mon: dict[str, Any] | None, key: str) -> Any:
    """Return recorded value or MISSING. Never invent defaults."""
    if mon is None or key not in mon:
        return MISSING
    return mon[key]


def _effective_inputs(rec: CycleRecord) -> dict[str, Any]:
    """A10: always record and compare effective inputs. No metadata suppression.

    Missing metadata (MISSING) never equals known values, empty values,
    or another MISSING. Absence of evidence on both sides is not proof of
    matching inputs — both-side MISSING is INPUT_DIVERGENCE.
    Explicit None, empty tuple (), and populated sequences are distinct.
    """
    mon = rec.monitoring if isinstance(rec.monitoring, dict) else None
    return {
        "effective_candle_epochs": rec.effective_candle_epochs
        if rec.effective_candle_epochs is not None
        else MISSING,
        "effective_candle_ohlcs": rec.effective_candle_ohlcs
        if rec.effective_candle_ohlcs is not None
        else MISSING,
        "effective_detector_names": rec.effective_detector_names
        if rec.effective_detector_names is not None
        else MISSING,
        "effective_detector_configs": getattr(rec, "effective_detector_configs", ())
        if getattr(rec, "effective_detector_configs", None) is not None
        else MISSING,
        "data_ok": rec.data_ok,
        "readiness_state": rec.readiness_state,
        "evaluation_blocked": rec.evaluation_blocked,
        "confirm_lag_bars": _mon_get(mon, "confirm_lag_bars"),
        "cache_max_size": _mon_get(mon, "cache_max_size"),
        "htf_epochs": _mon_get(mon, "htf_epochs"),
        "htf_source": _mon_get(mon, "htf_source"),
        "htf_ohlcs": _mon_get(mon, "htf_ohlcs"),
        "completed_count": _mon_get(mon, "completed_count"),
    }


def _behavior_fields(rec: CycleRecord) -> dict[str, Any]:
    return {
        "setup_states": dict(rec.setup_states),
        "signal_ids": rec.signal_ids,
        "emitted_observation_ids": rec.emitted_observation_ids,
        "newly_accepted_observation_ids": rec.newly_accepted_observation_ids,
        "setup_diagnostic_only": dict(rec.setup_diagnostic_only),
    }


def _report_cycle_index(a: CycleRecord, b: CycleRecord) -> int:
    """Prefer the left-run actual CycleRecord.cycle_index (not list position)."""
    return int(a.cycle_index)


def compare_replay_runs(
    run_a: ReplayResult,
    run_b: ReplayResult,
    *,
    setup_ids: Optional[Sequence[str]] = None,
) -> ParityComparison:
    """Compare two independent runs cycle-by-cycle.

    Ordering rule (first-divergence):
    1. Walk the common cycle range in chronological (list) order.
    2. Report the earliest actual divergence by CycleRecord.cycle_index and field.
    3. Only report a cycle-count / length mismatch if no earlier divergence exists.

    Missing required effective-input metadata is INPUT_DIVERGENCE (fail closed).
    """
    result = ParityComparison(matching=True)
    n = min(len(run_a.cycles), len(run_b.cycles))

    for i in range(n):
        a = run_a.cycles[i]
        b = run_b.cycles[i]
        reported_idx = _report_cycle_index(a, b)

        # Actual cycle-index identity difference is itself an input divergence.
        if a.cycle_index != b.cycle_index:
            d = Divergence(
                cycle_index=reported_idx,
                field="cycle_index",
                expected=a.cycle_index,
                actual=b.cycle_index,
                kind=INPUT_DIVERGENCE,
            )
            result.divergences.append(d)
            if result.first_divergence is None:
                result.first_divergence = d
            result.matching = False
            result.input_divergence = True
            continue

        a_in = _effective_inputs(a)
        b_in = _effective_inputs(b)
        input_mismatch = False
        for key in sorted(set(a_in) | set(b_in) | set(_REQUIRED_INPUT_KEYS)):
            av = a_in.get(key, MISSING)
            bv = b_in.get(key, MISSING)
            # Fail closed: any MISSING on either side is INPUT_DIVERGENCE,
            # including MISSING vs MISSING. Equal present values may match.
            missing_either = av is MISSING or bv is MISSING or av == MISSING or bv == MISSING
            if missing_either or av != bv:
                d = Divergence(
                    cycle_index=reported_idx,
                    field=f"effective_input.{key}",
                    expected=av,
                    actual=bv,
                    kind=INPUT_DIVERGENCE,
                )
                result.divergences.append(d)
                if result.first_divergence is None:
                    result.first_divergence = d
                result.matching = False
                result.input_divergence = True
                input_mismatch = True

        if input_mismatch:
            continue

        a_beh = _behavior_fields(a)
        b_beh = _behavior_fields(b)
        if setup_ids is not None:
            a_beh["setup_states"] = {
                k: v for k, v in a_beh["setup_states"].items() if k in setup_ids
            }
            b_beh["setup_states"] = {
                k: v for k, v in b_beh["setup_states"].items() if k in setup_ids
            }
            a_beh["setup_diagnostic_only"] = {
                k: v
                for k, v in a_beh["setup_diagnostic_only"].items()
                if k in setup_ids
            }
            b_beh["setup_diagnostic_only"] = {
                k: v
                for k, v in b_beh["setup_diagnostic_only"].items()
                if k in setup_ids
            }

        for key in sorted(set(a_beh) | set(b_beh)):
            if a_beh.get(key) != b_beh.get(key):
                d = Divergence(
                    cycle_index=reported_idx,
                    field=key,
                    expected=a_beh.get(key),
                    actual=b_beh.get(key),
                    kind=ENGINE_DIVERGENCE,
                )
                result.divergences.append(d)
                if result.first_divergence is None:
                    result.first_divergence = d
                result.matching = False
                result.engine_divergence = True

    if len(run_a.cycles) != len(run_b.cycles):
        # Only if no earlier divergence
        if result.first_divergence is None:
            # Use last common cycle's index + 1 when available; else n
            if n > 0:
                idx = max(run_a.cycles[n - 1].cycle_index, run_b.cycles[n - 1].cycle_index) + 1
            else:
                idx = 0
            d = Divergence(
                cycle_index=idx,
                field="cycle_count",
                expected=len(run_a.cycles),
                actual=len(run_b.cycles),
                kind=INPUT_DIVERGENCE,
            )
            result.divergences.append(d)
            result.first_divergence = d
            result.matching = False
            result.input_divergence = True
        else:
            # Still record length mismatch but do not replace first_divergence
            if n > 0:
                idx = max(run_a.cycles[n - 1].cycle_index, run_b.cycles[n - 1].cycle_index) + 1
            else:
                idx = 0
            result.divergences.append(
                Divergence(
                    cycle_index=idx,
                    field="cycle_count",
                    expected=len(run_a.cycles),
                    actual=len(run_b.cycles),
                    kind=INPUT_DIVERGENCE,
                )
            )
            result.matching = False
            result.input_divergence = True

    return result
