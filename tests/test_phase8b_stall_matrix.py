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


def generate_stall_scenarios() -> list[dict[str, Any]]:
    out = []
    for row in f1_rows():
        for fc in FAULT_CLASSES:
            out.append(
                {
                    "control": row.control,
                    "state_at_s": row.state_at_s,
                    "d": row.d,
                    "s": row.s,
                    "r": row.r,
                    "t": row.t,
                    "E": row.E,
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
) -> None:
    """Reusable A12–A14 assertions shared by matrix and mutation tests (E2)."""
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
        for i, rec in enumerate(faulted.cycles):
            if s <= rec.cycle_index < r:
                assert rec.evaluation_blocked or rec.injected_fault == "readiness_blocked"
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

    # r..E: no terminal, no signal; diagnostic continuity for unhealthy/missing.
    for rec in faulted.cycles:
        if not (r <= rec.cycle_index <= E):
            continue
        st = rec.setup_states.get(setup_f.setup_id)
        if st in (
            LifecycleState.COMPLETED.value,
            LifecycleState.INVALIDATED.value,
            LifecycleState.EXPIRED.value,
        ):
            raise AssertionError(f"terminal {st} at cycle {rec.cycle_index} inside r..E after {fc}")
        if rec.signal_ids:
            raise AssertionError(f"signal at cycle {rec.cycle_index} inside r..E after {fc}")
        if fc in ("unhealthy", "missing_candle"):
            assert rec.setup_diagnostic_only.get(setup_f.setup_id, False), (
                f"expected diagnostic_only at cycle {rec.cycle_index} in r..E after {fc}, state={st}"
            )
            reasons = " ".join(rec.setup_reasons.get(setup_f.setup_id, ()))
            if fc == "missing_candle":
                assert "discontinuous evaluation candle" in reasons, reasons
        if fc == "readiness_blocked":
            assert rec.setup_diagnostic_only.get(setup_f.setup_id, False), (
                f"post-block cycle {rec.cycle_index} must be diagnostic_only"
            )

    report = classify_stalled(faulted, setup_f.setup_id, injected_fault_cause=fc)
    assert report.injected_fault_cause == fc

    if fc == "readiness_blocked":
        expected_blocked = list(range(s, r))
        injected_blocked = [
            rec.cycle_index
            for rec in faulted.cycles
            if rec.injected_fault == "readiness_blocked"
        ]
        assert injected_blocked == expected_blocked, (injected_blocked, expected_blocked)
        # HOLD: strict — no RECOVERING tolerance; readiness stays real_env_failed until R1.
        assert report.blocked_cycles == expected_blocked, (
            f"blocked_cycles expected {expected_blocked}, got {report.blocked_cycles}"
        )
        assert report.first_discontinuity_cycle == r, report.first_discontinuity_cycle
        assert report.stalled_classified_at == r + 1, report.stalled_classified_at
        for bc in report.blocked_cycles:
            assert bc not in report.subsequent_diagnostic_cycles
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
    key = scenario["key"]
    s, r, t, E = scenario["s"], scenario["r"], scenario["t"], scenario["E"]
    d = scenario["d"]
    fc = scenario["fault_class"]
    control = scenario["control"]
    n = max(E, t) + 5

    setup, clean = _run(control, n=n)
    setup_f, faulted = _run(control, n=n, fault_schedule=_fault_schedule(s, d, fc))

    try:
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
        )
        status = "real_env_passed" if _real_talib_available() else "stub_only"
        EVIDENCE[key] = status
        record_stall(key, status)
    except AssertionError as exc:
        status = "real_env_failed"
        EVIDENCE[key] = status
        record_stall(
            key,
            status,
            extra={
                "reason": str(exc),
                "fault_class": fc,
            },
        )
        raise


def test_acceptance_meta_requires_real_env_for_all_24():
    """A15: every mandatory stall scenario must be real_env_passed. No exemptions (E1)."""
    if not _real_talib_available():
        pytest.fail("A15 meta-test: real TA-Lib is not available")
    for s in generate_stall_scenarios():
        key = s["key"]
        if key in EVIDENCE:
            record_stall(key, EVIDENCE[key])
    manifest = load_manifest()
    run = manifest.get("run") or {}
    assert run.get("commit"), f"A15 run identity missing: {run}"
    assert run.get("talib_version"), f"A15 missing talib_version: {run}"
    assert run.get("talib_file"), f"A15 missing talib_file: {run}"
    stall = manifest.get("stall") or {}
    missing = []
    bad = []
    for s in generate_stall_scenarios():
        key = str(s["key"])
        entry = stall.get(key)
        status = entry.get("status") if isinstance(entry, dict) else None
        if status is None:
            missing.append(key)
        elif status != "real_env_passed":
            bad.append((key, status))
    assert not missing, f"A15 stall missing from manifest: {missing}"
    assert not bad, f"A15 stall not real_env_passed: {bad}"


def test_acceptance_meta_a6_real_detector_evidence():
    """A15: durable manifest is sole authority for mandatory A6 fixture evidence."""
    if not _real_talib_available():
        pytest.fail("A15: real TA-Lib unavailable")
    for key, status in list(A6_EVIDENCE.items()):
        record_a6(key, status)
    manifest = load_manifest()
    run = manifest.get("run") or {}
    assert run.get("commit"), f"A15 run identity missing: {run}"
    assert run.get("talib_version"), f"A15 missing talib_version: {run}"
    assert run.get("talib_file"), f"A15 missing talib_file: {run}"
    a6 = manifest.get("a6") or {}
    required = [
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
    ]
    missing = []
    bad = []
    for key in required:
        entry = a6.get(key)
        status = entry.get("status") if isinstance(entry, dict) else None
        if status is None:
            missing.append(key)
        elif status != "real_env_passed":
            bad.append((key, status))
    assert not missing, f"A6 evidence missing from manifest: {missing}"
    assert not bad, f"A6 evidence not real_env_passed: {bad}"


def test_f1_completion_clean_control_real_detectors():
    setup, result = _run("F1-completion", n=110)
    assert result.final_setup_states.get(setup.setup_id) == LifecycleState.COMPLETED.value


def test_matrix_evidence_report_24_rows():
    """Complete 24-row matrix evidence summary from the current run (E1: all real_env_passed)."""
    for s in generate_stall_scenarios():
        key = s["key"]
        if key in EVIDENCE:
            record_stall(
                key,
                EVIDENCE[key],
                extra={
                    "control": s["control"],
                    "state_at_s": s["state_at_s"],
                    "fault_class": s["fault_class"],
                    "d": s["d"],
                    "s": s["s"],
                    "r": s["r"],
                    "t": s["t"],
                    "E": s["E"],
                },
            )
    manifest = load_manifest()
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
            }
        )
    assert len(rows) == 24
    missing = [row for row in rows if row["status"] != "real_env_passed"]
    assert not missing, f"matrix evidence incomplete (A15 requires all real_env_passed): {missing}"


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
    """E2: real helper must fail when readiness-blocked path advances to terminal.

    Under HOLD the genuine readiness run already fails the helper; mutation still
    verifies the helper rejects a terminal state in r..E.
    """
    control, s, r, t, E, d = "F1-completion", 70, 71, 98, 100, 1
    setup, clean = _run(control, n=E + 5)
    setup_f, faulted = _run(control, n=E + 5, fault_schedule=_fault_schedule(s, d, "readiness_blocked"))
    # Genuine readiness run fails under HOLD (RECOVERING lag) — that is expected.
    # Mutation: force terminal in r..E; helper must still raise on that violation.
    _mutate_faulted_to_advance(
        faulted, setup_f.setup_id, r, E, t, LifecycleState.COMPLETED.value
    )
    with pytest.raises(AssertionError):
        _assert_stall_window(
            control=control, s=s, r=r, t=t, E=E, d=d, fc="readiness_blocked",
            state_at_s="PENDING", setup=setup, clean=clean, setup_f=setup_f, faulted=faulted,
        )
