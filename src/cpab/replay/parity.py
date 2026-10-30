"""Replay vs live-style parity with effective-input comparison (R7–R8, A10–A11)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from cpab.replay.driver import CycleRecord, ReplayResult

INPUT_DIVERGENCE = "INPUT_DIVERGENCE"
ENGINE_DIVERGENCE = "ENGINE_DIVERGENCE"


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


def _effective_inputs(rec: CycleRecord) -> dict[str, Any]:
    return {
        "effective_candle_epochs": rec.effective_candle_epochs,
        "effective_detector_names": rec.effective_detector_names,
        "data_ok": rec.data_ok,
        "readiness_state": rec.readiness_state,
        "evaluation_blocked": rec.evaluation_blocked,
    }


def _behavior_fields(rec: CycleRecord) -> dict[str, Any]:
    return {
        "setup_states": dict(rec.setup_states),
        "signal_ids": rec.signal_ids,
        "emitted_observation_ids": rec.emitted_observation_ids,
        "newly_accepted_observation_ids": rec.newly_accepted_observation_ids,
        "setup_diagnostic_only": dict(rec.setup_diagnostic_only),
    }


def compare_replay_runs(
    run_a: ReplayResult,
    run_b: ReplayResult,
    *,
    setup_ids: Optional[Sequence[str]] = None,
) -> ParityComparison:
    """Compare two independent runs cycle-by-cycle.

    Effective inputs are compared first. Any input mismatch is reported as
    INPUT_DIVERGENCE and prevents an ENGINE_DIVERGENCE claim for that cycle.
    """
    result = ParityComparison(matching=True)
    n = min(len(run_a.cycles), len(run_b.cycles))
    if len(run_a.cycles) != len(run_b.cycles):
        d = Divergence(
            cycle_index=n,
            field="cycle_count",
            expected=len(run_a.cycles),
            actual=len(run_b.cycles),
            kind=INPUT_DIVERGENCE,
        )
        result.divergences.append(d)
        result.first_divergence = d
        result.matching = False
        result.input_divergence = True

    for i in range(n):
        a = run_a.cycles[i]
        b = run_b.cycles[i]
        a_in = _effective_inputs(a)
        b_in = _effective_inputs(b)
        input_mismatch = False
        for key in a_in:
            if a_in[key] != b_in[key]:
                d = Divergence(
                    cycle_index=i,
                    field=f"effective_input.{key}",
                    expected=a_in[key],
                    actual=b_in[key],
                    kind=INPUT_DIVERGENCE,
                )
                result.divergences.append(d)
                if result.first_divergence is None:
                    result.first_divergence = d
                result.matching = False
                result.input_divergence = True
                input_mismatch = True

        if input_mismatch:
            # Do not also claim engine inconsistency for this cycle.
            continue

        a_beh = _behavior_fields(a)
        b_beh = _behavior_fields(b)
        if setup_ids is not None:
            # Restrict state comparison to requested setups.
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

        for key in a_beh:
            if a_beh[key] != b_beh[key]:
                d = Divergence(
                    cycle_index=i,
                    field=key,
                    expected=a_beh[key],
                    actual=b_beh[key],
                    kind=ENGINE_DIVERGENCE,
                )
                result.divergences.append(d)
                if result.first_divergence is None:
                    result.first_divergence = d
                result.matching = False
                result.engine_divergence = True

    return result
