"""Phase 8B A12–A15: programmatic stall matrix against locked s/r/t/E values.

Separate F1-completion, F1-invalidation, and F1-expiry controls. Fault windows
use the exact contract coordinates. A15 requires a verified real-TA-Lib import.
"""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from cpab.conditions.expression import (
    ObservationRequirement,
    ThenExpression,
    ThenStep,
)
from cpab.conditions.lifecycle import LifecycleState
from cpab.conditions.schema import SCHEMA_VERSION_V1, ExpiryConfig, SetupDef
from cpab.detectors.base import PriceActionDetector
from cpab.models import Observation
from cpab.replay.driver import ReplayRunConfig, run_replay
from cpab.replay.stalled import classify_stalled
from conftest import make_candle
from cpab.detectors.breaker_block import BreakerBlockDetector
from cpab.detectors.behavior_volatility import ImpulseDetector

SYMBOL = "stpRNG"
TF = "15m"
TF_SEC = 900
START = 1_699_992_000
ARM = 60  # first evaluation at candle index 59

from cpab.replay.evidence import A6_EVIDENCE, A6_MANDATORY_KEYS, STALL_EVIDENCE as EVIDENCE, clear_evidence, load_manifest, record_a6, record_stall


from cpab.replay.real_talib import real_talib_available as _real_talib_available


class ScriptedDetector(PriceActionDetector):
    def __init__(self, by_epoch: dict[int, list[Observation]]):
        self._by_epoch = {k: list(v) for k, v in by_epoch.items()}
        self._emitted: set[int] = set()

    @property
    def name(self) -> str:
        return "scripted"

    def detect(self, candles: list) -> list[Observation]:
        if not candles:
            return []
        latest = candles[-1].epoch
        if latest in self._emitted:
            return []
        self._emitted.add(latest)
        return list(self._by_epoch.pop(latest, []))


def _obs(typ: str, epoch: int, *, variant: str | None = None) -> Observation:
    return Observation(
        id=f"{typ}_{variant or 'nv'}_{epoch}",
        type=typ,
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=epoch,
        event_time=epoch,
        confirmed_at=epoch,
        detected_at=float(epoch),
        developing=False,
        variant=variant,
    )


def _candles(n: int) -> list:
    return [
        make_candle(
            SYMBOL, TF, START + i * TF_SEC, 100.0, 101.0, 99.0, 100.5, is_closed=True
        )
        for i in range(n)
    ]


def _epoch(idx: int) -> int:
    return START + idx * TF_SEC


@dataclass(frozen=True)
class StallRow:
    control: str
    state_at_s: str
    d: int
    s: int
    r: int
    t: int
    E: int


def f1_rows() -> list[StallRow]:
    return [
        StallRow("F1-completion", "PENDING", 1, 70, 71, 98, 100),
        StallRow("F1-completion", "PENDING", 5, 70, 75, 98, 100),
        StallRow("F1-completion", "IN_PROGRESS", 1, 95, 96, 98, 100),
        StallRow("F1-completion", "IN_PROGRESS", 5, 95, 100, 98, 104),
        StallRow("F1-invalidation", "PENDING", 1, 70, 71, 129, 131),
        StallRow("F1-invalidation", "PENDING", 5, 70, 75, 129, 131),
        StallRow("F1-expiry", "PENDING", 1, 70, 71, 139, 141),
        StallRow("F1-expiry", "PENDING", 5, 70, 75, 139, 141),
    ]


FAULT_CLASSES = ("unhealthy", "readiness_blocked", "missing_candle")


def _e_rb(row: StallRow) -> int:
    """Readiness-class check-window end: E_rb = max(r+5, t+2) (R1 / Rev 9)."""
    return max(row.r + 5, row.t + 2)


def generate_stall_scenarios() -> list[dict[str, Any]]:
    out = []
    for row in f1_rows():
        for fc in FAULT_CLASSES:
            e_rb = _e_rb(row)
            out.append(
                {
                    "control": row.control,
                    "state_at_s": row.state_at_s,
                    "d": row.d,
                    "s": row.s,
                    "r": row.r,
                    "t": row.t,
                    "E": row.E,
                    # Readiness uses E_rb; unhealthy/missing_candle use E.
                    "E_rb": e_rb,
                    "fault_class": fc,
                    "key": (row.control, row.state_at_s, fc, row.d),
                }
            )
    assert len(out) == 24
    return out


def _f1_expression() -> ThenExpression:
    """Contract fx_bearish_breaker_then_impulse expression."""
    return ThenExpression(
        steps=(
            ThenStep(
                expression=ObservationRequirement(
                    type="breaker_block", variant="bullish_to_bearish", min_count=1
                )
            ),
            ThenStep(
                expression=ObservationRequirement(
                    type="bearish_impulse",
                    variant="atr_normalized_candle_v1",
                    min_count=1,
                )
            ),
        )
    )


def _setup_completion() -> SetupDef:
    return SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id="f1_completion_control",
        symbol=SYMBOL,
        timeframe=TF,
        enabled=True,
        expression=_f1_expression(),
        invalidation=ObservationRequirement(
            type="breaker_block", variant="bearish_to_bullish", min_count=1
        ),
        expiry=ExpiryConfig(max_age_candles=80),
        name="fx_bearish_breaker_then_impulse",
    )


def _setup_invalidation() -> SetupDef:
    # Same F1 setup; invalidation observation is injected instead of completion path.
    return SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id="f1_invalidation_control",
        symbol=SYMBOL,
        timeframe=TF,
        enabled=True,
        expression=_f1_expression(),
        invalidation=ObservationRequirement(
            type="breaker_block", variant="bearish_to_bullish", min_count=1
        ),
        expiry=ExpiryConfig(max_age_candles=80),
        name="fx_bearish_breaker_then_impulse",
    )


def _setup_expiry() -> SetupDef:
    return SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id="f1_expiry_control",
        symbol=SYMBOL,
        timeframe=TF,
        enabled=True,
        expression=_f1_expression(),
        invalidation=ObservationRequirement(
            type="breaker_block", variant="bearish_to_bullish", min_count=1
        ),
        expiry=ExpiryConfig(max_age_candles=80),
        name="fx_bearish_breaker_then_impulse",
    )


def _script_for_control(control: str) -> dict[int, list[Observation]]:
    """Scripted F1 observation timing at locked contract indices.

    Continuity isolation still uses ScriptedDetector; observation *types/variants*
    match the contract F1 setup (breaker_block / bearish_impulse).
    """
    if control == "F1-completion":
        return {
            _epoch(92): [
                _obs("breaker_block", _epoch(92), variant="bullish_to_bearish")
            ],
            _epoch(98): [
                _obs(
                    "bearish_impulse",
                    _epoch(98),
                    variant="atr_normalized_candle_v1",
                )
            ],
        }
    if control == "F1-invalidation":
        # Inject invalidating breaker at 129 (no completion path).
        return {
            _epoch(129): [
                _obs("breaker_block", _epoch(129), variant="bearish_to_bullish")
            ],
        }
    if control == "F1-expiry":
        # Breaker accepted near arming without impulse; max_age 80 → EXPIRED at 139.
        return {
            _epoch(59): [
                _obs("breaker_block", _epoch(59), variant="bullish_to_bearish")
            ],
        }
    raise ValueError(control)


def _setup_for(control: str) -> SetupDef:
    return {
        "F1-completion": _setup_completion,
        "F1-invalidation": _setup_invalidation,
        "F1-expiry": _setup_expiry,
    }[control]()


def _fault_schedule(s: int, d: int, fault_class: str) -> dict[int, dict]:
    sched: dict[int, dict] = {}
    for i in range(s, s + d):
        if fault_class == "unhealthy":
            sched[i] = {"series_healthy": False}
        elif fault_class == "readiness_blocked":
            sched[i] = {"readiness_ready": False}
        else:
            sched[i] = {"skip_candle": True}
    return sched



FIXTURES_8B = Path(__file__).resolve().parent / "fixtures" / "phase8b"
SEED_START = 1_699_992_000
SEED_STEP = 900


def _load_seed_candles(case_key: str) -> list:
    import json
    meta = json.loads((FIXTURES_8B / "manifest.json").read_text())["cases"][case_key]
    data = json.loads((FIXTURES_8B / meta["file"]).read_text())
    out = []
    for row in data["candles"]:
        epoch, o, h, l, c = row
        out.append(
            make_candle(
                data["symbol"],
                data["timeframe"],
                int(epoch),
                float(o),
                float(h),
                float(l),
                float(c),
                is_closed=True,
            )
        )
    return out


def _seed_case_for(control: str) -> str:
    return {
        "F1-completion": "F1_completion_seed3",
        "F1-invalidation": "F1_invalidation_seed1",
        "F1-expiry": "F1_expiry_seed4",
    }[control]


def _run(control: str, *, n: int, fault_schedule: Optional[dict] = None):
    """Run with real F1 detectors on contract seed series (C1.3)."""
    setup = _setup_for(control)
    candles = _load_seed_candles(_seed_case_for(control))
    # n is advisory; seed series is 220 candles — use full series for arming/eval.
    dets = [
        BreakerBlockDetector(left_strength=1, right_strength=1),
        ImpulseDetector(),
    ]
    return setup, run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=dets,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
            timeframe_seconds=SEED_STEP,
            candles=candles,
            arm_count=ARM,
            fault_schedule=fault_schedule or {},
        )
    )


def test_matrix_has_exactly_24_scenarios():
    assert len(generate_stall_scenarios()) == 24


def test_matrix_locked_coordinates():
    rows = f1_rows()
    assert rows[0] == StallRow("F1-completion", "PENDING", 1, 70, 71, 98, 100)
    assert rows[3].E == 104
    assert rows[4].t == 129
    assert rows[6].t == 139



def _assert_stall_window(
    *,
    control: str,
    s: int,
    r: int,
    t: int,
    E: int,
    d: int,
    fc: str,
    state_at_s: str,
    setup,
    clean,
    setup_f,
    faulted,
    E_rb: int | None = None,
) -> None:
    """Reusable A12–A14 assertions shared by matrix and mutation tests.

    Fault-class specific post-recovery rules (Rev 9 / R1):
    - unhealthy / missing_candle: j=r, STALLED r+1, window r..E
    - readiness_blocked: blocked [s,r], j=r+1, STALLED r+2, window r+1..E_rb
    """
    clean_terminal = clean.final_setup_states.get(setup.setup_id)
    if control == "F1-completion":
        assert clean_terminal == LifecycleState.COMPLETED.value, clean_terminal
        assert any(
            rec.cycle_index == t
            and rec.setup_states.get(setup.setup_id) == LifecycleState.COMPLETED.value
            for rec in clean.cycles
        ), t
    elif control == "F1-invalidation":
        assert clean_terminal == LifecycleState.INVALIDATED.value, clean_terminal
    elif control == "F1-expiry":
        assert clean_terminal == LifecycleState.EXPIRED.value, clean_terminal

    if state_at_s == "IN_PROGRESS":
        state_at_s_obs = None
        for rec in clean.cycles:
            if rec.cycle_index == s:
                state_at_s_obs = rec.setup_states.get(setup.setup_id)
                break
        assert state_at_s_obs == LifecycleState.IN_PROGRESS.value, (
            f"expected IN_PROGRESS at s={s}, got {state_at_s_obs}"
        )

    state_before = None
    anchor_before = None
    for rec in faulted.cycles:
        if rec.cycle_index == s - 1:
            state_before = rec.setup_states.get(setup_f.setup_id)
            anchor_before = rec.setup_anchors.get(setup_f.setup_id)
            break

    if fc == "readiness_blocked":
        # R1: blocked on [s, r] inclusive (injected [s,r-1] + RECOVERING at r).
        for i, rec in enumerate(faulted.cycles):
            if s <= rec.cycle_index <= r:
                assert rec.evaluation_blocked or rec.injected_fault == "readiness_blocked", (
                    f"cycle {rec.cycle_index} must be evaluation_blocked under R1"
                )
                assert faulted.lifecycle_calls_per_cycle[i] == 0, (
                    f"lifecycle calls must be 0 on blocked cycle {rec.cycle_index}"
                )

    if fc == "missing_candle":
        omit_ep = START + s * TF_SEC
        for rec in faulted.cycles:
            if rec.cycle_index > s:
                assert omit_ep not in rec.effective_candle_epochs, (
                    f"omitted candle epoch {omit_ep} reappeared at cycle {rec.cycle_index}"
                )

    # Fault interval [s, r-1]: state and anchor unchanged, no signal (except readiness: zero LC).
    if state_before is not None and fc != "readiness_blocked":
        for rec in faulted.cycles:
            if s <= rec.cycle_index < r:
                st = rec.setup_states.get(setup_f.setup_id)
                an = rec.setup_anchors.get(setup_f.setup_id)
                assert st == state_before, (rec.cycle_index, st, state_before)
                assert an == anchor_before, (rec.cycle_index, an, anchor_before)
                assert not rec.signal_ids

    # Post-recovery window: class-specific start/end.
    if fc == "readiness_blocked":
        win_start = r + 1
        win_end = E_rb if E_rb is not None else max(r + 5, t + 2)
    else:
        win_start = r
        win_end = E

    for rec in faulted.cycles:
        if not (win_start <= rec.cycle_index <= win_end):
            continue
        st = rec.setup_states.get(setup_f.setup_id)
        if st in (
            LifecycleState.COMPLETED.value,
            LifecycleState.INVALIDATED.value,
            LifecycleState.EXPIRED.value,
        ):
            raise AssertionError(
                f"terminal {st} at cycle {rec.cycle_index} inside {win_start}..{win_end} after {fc}"
            )
        if rec.signal_ids:
            raise AssertionError(
                f"signal at cycle {rec.cycle_index} inside {win_start}..{win_end} after {fc}"
            )
        if fc in ("unhealthy", "missing_candle"):
            assert rec.setup_diagnostic_only.get(setup_f.setup_id, False), (
                f"expected diagnostic_only at cycle {rec.cycle_index} in r..E after {fc}, state={st}"
            )
            reasons = " ".join(rec.setup_reasons.get(setup_f.setup_id, ()))
            if fc == "missing_candle":
                assert "discontinuous evaluation candle" in reasons, reasons
        if fc == "readiness_blocked":
            assert rec.setup_diagnostic_only.get(setup_f.setup_id, False), (
                f"post-recovery cycle {rec.cycle_index} must be diagnostic_only under R1"
            )
            reasons = " ".join(rec.setup_reasons.get(setup_f.setup_id, ()))
            assert "discontinuous evaluation candle" in reasons, reasons

    report = classify_stalled(faulted, setup_f.setup_id, injected_fault_cause=fc)
    assert report.injected_fault_cause == fc

    if fc == "readiness_blocked":
        # R1: injected [s, r-1]; effective blocked [s, r] including RECOVERING.
        expected_injected = list(range(s, r))
        expected_blocked = list(range(s, r + 1))
        injected_blocked = [
            rec.cycle_index
            for rec in faulted.cycles
            if rec.injected_fault == "readiness_blocked"
        ]
        assert injected_blocked == expected_injected, (injected_blocked, expected_injected)
        assert report.blocked_cycles == expected_blocked, (
            f"blocked_cycles expected {expected_blocked}, got {report.blocked_cycles}"
        )
        assert report.first_discontinuity_cycle == r + 1, (
            f"first_discontinuity_cycle expected {r + 1}, got {report.first_discontinuity_cycle}"
        )
        assert report.stalled_classified_at == r + 2, (
            f"stalled_classified_at expected {r + 2}, got {report.stalled_classified_at}"
        )
        assert report.first_discontinuity_expected_epoch is not None
        assert report.first_discontinuity_actual_epoch is not None
        assert (
            report.first_discontinuity_expected_epoch
            != report.first_discontinuity_actual_epoch
        )
        if report.first_discontinuity_last_accepted_epoch is not None:
            assert (
                report.first_discontinuity_expected_epoch
                == report.first_discontinuity_last_accepted_epoch + TF_SEC
            )
        for bc in report.blocked_cycles:
            assert bc not in report.subsequent_diagnostic_cycles
            # No blocked cycle may be diagnostic_only.
            for rec in faulted.cycles:
                if rec.cycle_index == bc:
                    assert not rec.setup_diagnostic_only.get(setup_f.setup_id, False), (
                        f"blocked cycle {bc} must not be diagnostic_only"
                    )
    elif fc in ("unhealthy", "missing_candle"):
        assert report.blocked_cycles == []
        assert report.first_discontinuity_cycle == r, (
            f"first_discontinuity_cycle expected {r}, got {report.first_discontinuity_cycle}"
        )
        assert report.stalled_classified_at == r + 1, (
            f"stalled_classified_at expected {r + 1}, got {report.stalled_classified_at}"
        )
        assert report.first_discontinuity_expected_epoch is not None
        assert report.first_discontinuity_actual_epoch is not None
        assert (
            report.first_discontinuity_expected_epoch
            != report.first_discontinuity_actual_epoch
        )
        if report.first_discontinuity_last_accepted_epoch is not None:
            assert (
                report.first_discontinuity_expected_epoch
                == report.first_discontinuity_last_accepted_epoch + TF_SEC
            )


@pytest.mark.parametrize(
    "scenario",
    generate_stall_scenarios(),
    ids=lambda s: f"{s['control']}_{s['state_at_s']}_{s['fault_class']}_d{s['d']}",
)
def test_stall_scenario_locked_window(scenario):
    """Full-scenario failure-recording boundary (G1).

    Every operation required to execute the scenario is inside the try block so
    setup/execution failures also produce durable real_env_failed evidence.
    """
    key = scenario["key"]
    fc = scenario["fault_class"]
    try:
        s, r, t, E = scenario["s"], scenario["r"], scenario["t"], scenario["E"]
        d = scenario["d"]
        control = scenario["control"]
        n = max(E, t) + 5

        setup, clean = _run(control, n=n)
        setup_f, faulted = _run(control, n=n, fault_schedule=_fault_schedule(s, d, fc))

        _assert_stall_window(
            control=control,
            s=s,
            r=r,
            t=t,
            E=E,
            d=d,
            fc=fc,
            state_at_s=scenario["state_at_s"],
            setup=setup,
            clean=clean,
            setup_f=setup_f,
            faulted=faulted,
            E_rb=scenario.get("E_rb"),
        )
        status = "real_env_passed" if _real_talib_available() else "stub_only"
        EVIDENCE[key] = status
        record_stall(key, status, extra={"fault_class": fc, "control": control})
    except Exception as exc:
        status = "real_env_failed"
        EVIDENCE[key] = status
        try:
            record_stall(
                key,
                status,
                extra={
                    "reason": str(exc),
                    "fault_class": fc,
                    "control": scenario.get("control"),
                    "exc_type": type(exc).__name__,
                },
            )
        except Exception as rec_exc:
            raise RuntimeError(
                f"A15: failed to record real_env_failed for {key!r}: {rec_exc}"
            ) from exc
        raise



def test_z_acceptance_meta_requires_real_env_for_all_24():
    """A15: every mandatory stall scenario must be real_env_passed. No exemptions.

    Validates the on-disk manifest only — does not copy in-memory EVIDENCE.
    """
    if not _real_talib_available():
        pytest.fail("A15 meta-test: real TA-Lib is not available")
    from cpab.replay.evidence import current_run_id, validate_run_identity

    manifest = load_manifest()
    run = validate_run_identity(manifest)
    run_id = run["run_id"]
    assert current_run_id() == run_id, (current_run_id(), run_id)
    stall = manifest.get("stall") or {}
    missing = []
    bad = []
    mismatched = []
    for s in generate_stall_scenarios():
        key = str(s["key"])
        entry = stall.get(key)
        if not isinstance(entry, dict):
            missing.append(key)
            continue
        if entry.get("run_id") != run_id:
            mismatched.append((key, entry.get("run_id")))
        status = entry.get("status")
        if status != "real_env_passed":
            bad.append((key, status))
    assert not missing, f"A15 stall missing from manifest: {missing}"
    assert not mismatched, f"A15 stall run_id mismatch: {mismatched}"
    assert not bad, f"A15 stall not real_env_passed: {bad}"



def test_z_acceptance_meta_a6_real_detector_evidence():
    """A15: durable manifest is sole authority for mandatory A6 fixture evidence.

    Does not copy in-memory A6_EVIDENCE into the manifest before validation.
    """
    if not _real_talib_available():
        pytest.fail("A15: real TA-Lib unavailable")
    from cpab.replay.evidence import A6_MANDATORY_KEYS, current_run_id, validate_run_identity

    manifest = load_manifest()
    run = validate_run_identity(manifest)
    run_id = run["run_id"]
    assert current_run_id() == run_id
    a6 = manifest.get("a6") or {}
    missing = []
    bad = []
    mismatched = []
    for key in A6_MANDATORY_KEYS:
        entry = a6.get(key)
        if not isinstance(entry, dict):
            missing.append(key)
            continue
        if entry.get("run_id") != run_id:
            mismatched.append((key, entry.get("run_id")))
        status = entry.get("status")
        if status != "real_env_passed":
            bad.append((key, status))
    assert not missing, f"A6 evidence missing from manifest: {missing}"
    assert not mismatched, f"A6 run_id mismatch: {mismatched}"
    assert not bad, f"A6 evidence not real_env_passed: {bad}"



def test_f1_completion_clean_control_real_detectors():
    setup, result = _run("F1-completion", n=110)
    assert result.final_setup_states.get(setup.setup_id) == LifecycleState.COMPLETED.value


def test_z_matrix_evidence_report_24_rows():
    """Complete 24-row matrix evidence from current-run manifest only (G2)."""
    from cpab.replay.evidence import current_run_id, validate_run_identity

    manifest = load_manifest()
    run = validate_run_identity(manifest)
    run_id = run["run_id"]
    assert current_run_id() == run_id
    stall = manifest.get("stall") or {}
    rows = []
    for s in generate_stall_scenarios():
        key = str(s["key"])
        entry = stall.get(key) or {}
        rows.append(
            {
                "key": key,
                "control": s["control"],
                "state_at_s": s["state_at_s"],
                "fault_class": s["fault_class"],
                "d": s["d"],
                "s": s["s"],
                "r": s["r"],
                "t": s["t"],
                "E": s["E"],
                "status": entry.get("status") if isinstance(entry, dict) else None,
                "run_id": entry.get("run_id") if isinstance(entry, dict) else None,
            }
        )
    assert len(rows) == 24
    missing = [
        row for row in rows
        if row["status"] != "real_env_passed" or row["run_id"] != run_id
    ]
    assert not missing, (
        f"matrix evidence incomplete (A15 requires all real_env_passed same run): {missing}"
    )



def _mutate_faulted_to_advance(faulted, setup_id: str, r: int, E: int, t: int, terminal: str) -> None:
    """Test-only: make faulted cycles look like lifecycle advanced through the fault (E2)."""
    for rec in faulted.cycles:
        if r <= rec.cycle_index <= E:
            # Clear diagnostic continuity and force terminal + signal at t.
            rec.setup_diagnostic_only[setup_id] = False
            rec.setup_reasons[setup_id] = ()
            if rec.cycle_index == t:
                object.__setattr__(
                    rec,
                    "setup_states",
                    {**rec.setup_states, setup_id: terminal},
                )
                object.__setattr__(rec, "signal_ids", ("mutated-signal",))


def test_mutation_missing_candle_helper_rejects_advance():
    """E2: real helper must fail when post-fault candle is accepted and setup completes."""
    control, s, r, t, E, d = "F1-completion", 70, 71, 98, 100, 1
    setup, clean = _run(control, n=E + 5)
    setup_f, faulted = _run(control, n=E + 5, fault_schedule=_fault_schedule(s, d, "missing_candle"))
    # Genuine faulted run must satisfy the helper.
    _assert_stall_window(
        control=control, s=s, r=r, t=t, E=E, d=d, fc="missing_candle",
        state_at_s="PENDING", setup=setup, clean=clean, setup_f=setup_f, faulted=faulted,
    )
    # Mutate to simulate accepting the post-fault candle and completing.
    _mutate_faulted_to_advance(
        faulted, setup_f.setup_id, r, E, t, LifecycleState.COMPLETED.value
    )
    with pytest.raises(AssertionError):
        _assert_stall_window(
            control=control, s=s, r=r, t=t, E=E, d=d, fc="missing_candle",
            state_at_s="PENDING", setup=setup, clean=clean, setup_f=setup_f, faulted=faulted,
        )


def test_mutation_unhealthy_helper_rejects_advance():
    """E2: real helper must fail when unhealthy path advances to terminal."""
    control, s, r, t, E, d = "F1-completion", 70, 71, 98, 100, 1
    setup, clean = _run(control, n=E + 5)
    setup_f, faulted = _run(control, n=E + 5, fault_schedule=_fault_schedule(s, d, "unhealthy"))
    _assert_stall_window(
        control=control, s=s, r=r, t=t, E=E, d=d, fc="unhealthy",
        state_at_s="PENDING", setup=setup, clean=clean, setup_f=setup_f, faulted=faulted,
    )
    _mutate_faulted_to_advance(
        faulted, setup_f.setup_id, r, E, t, LifecycleState.COMPLETED.value
    )
    with pytest.raises(AssertionError):
        _assert_stall_window(
            control=control, s=s, r=r, t=t, E=E, d=d, fc="unhealthy",
            state_at_s="PENDING", setup=setup, clean=clean, setup_f=setup_f, faulted=faulted,
        )


def test_mutation_readiness_helper_rejects_advance():
    """R1: genuine readiness run passes helper; mutated terminal is rejected."""
    control, s, r, t, E, d = "F1-completion", 70, 71, 98, 100, 1
    E_rb = max(r + 5, t + 2)
    setup, clean = _run(control, n=E_rb + 5)
    setup_f, faulted = _run(control, n=E_rb + 5, fault_schedule=_fault_schedule(s, d, "readiness_blocked"))

    # Genuine R1 run must pass the shared helper.
    _assert_stall_window(
        control=control, s=s, r=r, t=t, E=E, d=d, fc="readiness_blocked",
        state_at_s="PENDING", setup=setup, clean=clean, setup_f=setup_f, faulted=faulted,
        E_rb=E_rb,
    )

    # Mutation: force terminal in recovery window; helper must raise.
    _mutate_faulted_to_advance(
        faulted, setup_f.setup_id, r + 1, E_rb, t, LifecycleState.COMPLETED.value
    )
    with pytest.raises(AssertionError):
        _assert_stall_window(
            control=control, s=s, r=r, t=t, E=E, d=d, fc="readiness_blocked",
            state_at_s="PENDING", setup=setup, clean=clean, setup_f=setup_f, faulted=faulted,
            E_rb=E_rb,
        )


def test_readiness_sequence_ready_degraded_recovering_ready():
    """R1: pin real ReadinessManager sequence on d=1 and d=5 readiness faults."""
    for s, d, r in [(70, 1, 71), (70, 5, 75)]:
        setup_f, faulted = _run(
            "F1-completion", n=r + 10, fault_schedule=_fault_schedule(s, d, "readiness_blocked")
        )
        # Collect readiness states around the fault window.
        seq = []
        for rec in faulted.cycles:
            if s - 1 <= rec.cycle_index <= r + 2:
                seq.append((rec.cycle_index, rec.readiness_state, rec.evaluation_blocked))
        # Before fault: READY, not blocked.
        pre = [x for x in seq if x[0] == s - 1]
        assert pre and pre[0][1] == "READY" and pre[0][2] is False, seq
        # Injected cycles [s, r-1]: DEGRADED (or equivalent non-READY), blocked.
        for cyc in range(s, r):
            rows = [x for x in seq if x[0] == cyc]
            assert rows, (cyc, seq)
            assert rows[0][2] is True, (cyc, rows)
            assert rows[0][1] != "READY", (cyc, rows)
        # Cycle r: RECOVERING, still blocked, zero lifecycle calls.
        rows_r = [x for x in seq if x[0] == r]
        assert rows_r, seq
        assert rows_r[0][1] == "RECOVERING", (rows_r, seq)
        assert rows_r[0][2] is True, rows_r
        # r+1: READY (or eligible); first continuity diagnostic cycle.
        rows_rp1 = [x for x in seq if x[0] == r + 1]
        assert rows_rp1, seq
        assert rows_rp1[0][1] == "READY", (rows_rp1, seq)
        assert rows_rp1[0][2] is False, rows_rp1


def test_readiness_negative_underreports_recovering_cycle():
    """R1 negative: asserting blocked == [s,r-1] only must fail."""
    control, s, r, t, E, d = "F1-completion", 70, 71, 98, 100, 1
    E_rb = max(r + 5, t + 2)
    setup, clean = _run(control, n=E_rb + 5)
    setup_f, faulted = _run(control, n=E_rb + 5, fault_schedule=_fault_schedule(s, d, "readiness_blocked"))
    report = classify_stalled(faulted, setup_f.setup_id, injected_fault_cause="readiness_blocked")
    # Genuine blocked is [s, r]; under-report would be [s, r-1].
    assert report.blocked_cycles == list(range(s, r + 1))
    with pytest.raises(AssertionError):
        assert report.blocked_cycles == list(range(s, r))


def test_readiness_negative_stalled_at_r_plus_1():
    """R1 negative: STALLED at r+1 (Rev 8 timing) must fail under Rev 9."""
    control, s, r, t, E, d = "F1-completion", 70, 71, 98, 100, 1
    E_rb = max(r + 5, t + 2)
    setup_f, faulted = _run(control, n=E_rb + 5, fault_schedule=_fault_schedule(s, d, "readiness_blocked"))
    report = classify_stalled(faulted, setup_f.setup_id, injected_fault_cause="readiness_blocked")
    assert report.stalled_classified_at == r + 2
    with pytest.raises(AssertionError):
        assert report.stalled_classified_at == r + 1


def _record_scenario_outcome(key, fc, control, fn):
    """Minimal G1 recording boundary mirror for isolation tests."""
    try:
        fn()
        status = "real_env_passed" if _real_talib_available() else "stub_only"
        EVIDENCE[key] = status
        record_stall(key, status, extra={"fault_class": fc, "control": control})
    except Exception as exc:
        status = "real_env_failed"
        EVIDENCE[key] = status
        try:
            record_stall(
                key,
                status,
                extra={
                    "reason": str(exc),
                    "fault_class": fc,
                    "control": control,
                    "exc_type": type(exc).__name__,
                },
            )
        except Exception as rec_exc:
            raise RuntimeError(
                f"A15: failed to record real_env_failed for {key!r}: {rec_exc}"
            ) from exc
        raise


def test_stall_recording_on_clean_run_failure(monkeypatch):
    """G1: failure during clean _run records real_env_failed and re-raises."""
    import tests.test_phase8b_stall_matrix as mod
    from cpab.replay import evidence as ev

    key = ("__g1_clean_fail__", "PENDING", "unhealthy", 1)

    def boom(*args, **kwargs):
        raise RuntimeError("injected clean _run failure")

    monkeypatch.setattr(mod, "_run", boom)

    def body():
        mod._run("F1-completion", n=10)

    with pytest.raises(RuntimeError, match="injected clean"):
        _record_scenario_outcome(key, "unhealthy", "F1-completion", body)
    manifest = load_manifest()
    entry = (manifest.get("stall") or {}).get(str(key))
    assert isinstance(entry, dict), entry
    assert entry["status"] == "real_env_failed"
    assert entry.get("run_id") == ev.current_run_id()


def test_stall_recording_on_faulted_run_failure(monkeypatch):
    """G1: failure during faulted _run records real_env_failed and re-raises."""
    import tests.test_phase8b_stall_matrix as mod
    from cpab.replay import evidence as ev

    key = ("__g1_faulted_fail__", "PENDING", "unhealthy", 1)
    real_run = mod._run

    def selective(*args, **kwargs):
        if kwargs.get("fault_schedule") is not None:
            raise RuntimeError("injected faulted _run failure")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod, "_run", selective)

    def body():
        mod._run("F1-completion", n=10)
        mod._run("F1-completion", n=10, fault_schedule={"70": {"unhealthy": True}})

    with pytest.raises(RuntimeError, match="injected faulted"):
        _record_scenario_outcome(key, "unhealthy", "F1-completion", body)
    manifest = load_manifest()
    entry = (manifest.get("stall") or {}).get(str(key))
    assert isinstance(entry, dict)
    assert entry["status"] == "real_env_failed"
    assert entry.get("run_id") == ev.current_run_id()


def test_stall_recording_on_assertion_failure(monkeypatch):
    """G1: assertion failure inside helper records real_env_failed and re-raises."""
    import tests.test_phase8b_stall_matrix as mod
    from cpab.replay import evidence as ev

    key = ("__g1_assert_fail__", "PENDING", "unhealthy", 1)

    def boom(**kwargs):
        raise AssertionError("injected assertion failure")

    monkeypatch.setattr(mod, "_assert_stall_window", boom)

    def body():
        setup, clean = mod._run("F1-completion", n=110)
        setup_f, faulted = mod._run(
            "F1-completion", n=110, fault_schedule=mod._fault_schedule(70, 1, "unhealthy")
        )
        mod._assert_stall_window(
            control="F1-completion",
            s=70, r=71, t=98, E=100, d=1, fc="unhealthy",
            state_at_s="PENDING", setup=setup, clean=clean, setup_f=setup_f, faulted=faulted,
        )

    with pytest.raises(AssertionError, match="injected assertion"):
        _record_scenario_outcome(key, "unhealthy", "F1-completion", body)
    manifest = load_manifest()
    entry = (manifest.get("stall") or {}).get(str(key))
    assert isinstance(entry, dict)
    assert entry["status"] == "real_env_failed"
    assert entry.get("run_id") == ev.current_run_id()


def test_stall_failed_record_not_overwritten_as_passed():
    """G1: a real_env_failed record remains failed; meta requires real_env_passed."""
    from cpab.replay import evidence as ev
    key = ("__g1_probe__", "PENDING", "unhealthy", 1)
    record_stall(key, "real_env_failed", extra={"probe": True})
    entry = (load_manifest().get("stall") or {}).get(str(key))
    assert entry["status"] == "real_env_failed"
    assert entry.get("run_id") == ev.current_run_id()
