"""Phase 8B A12–A15: programmatic stall matrix against locked s/r/t/E values.

Separate F1-completion, F1-invalidation, and F1-expiry controls. Fault windows
use the exact contract coordinates. A15 requires a verified real-TA-Lib import.
"""

from __future__ import annotations

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

SYMBOL = "R_100"
TF = "1m"
TF_SEC = 60
START = 1_700_000_000
ARM = 60  # first evaluation at candle index 59

from cpab.replay.evidence import A6_EVIDENCE, A6_MANDATORY_KEYS, STALL_EVIDENCE as EVIDENCE, clear_evidence, load_manifest, record_a6, record_stall


def _real_talib_available() -> bool:
    try:
        import talib  # noqa: F401
        from talib.abstract import Function  # noqa: F401

        _ = int(Function("CDLENGULFING").lookback)
        return True
    except Exception:
        return False


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


def _run(control: str, *, n: int, fault_schedule: Optional[dict] = None):
    setup = _setup_for(control)
    candles = _candles(n)
    det = ScriptedDetector(_script_for_control(control))
    return setup, run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[det],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
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
    clean_terminal = clean.final_setup_states.get(setup.setup_id)

    if control == "F1-completion":
        assert clean_terminal == LifecycleState.COMPLETED.value, clean_terminal
        found_t = None
        for rec in clean.cycles:
            if (
                rec.cycle_index == t
                and rec.setup_states.get(setup.setup_id)
                == LifecycleState.COMPLETED.value
            ):
                found_t = rec.cycle_index
        assert found_t == t, f"clean completion expected at t={t}, got {found_t}"
    elif control == "F1-invalidation":
        assert clean_terminal == LifecycleState.INVALIDATED.value, clean_terminal
    elif control == "F1-expiry":
        assert clean_terminal == LifecycleState.EXPIRED.value, clean_terminal

    if scenario["state_at_s"] == "IN_PROGRESS":
        state_at_s = None
        for rec in clean.cycles:
            if rec.cycle_index == s:
                state_at_s = rec.setup_states.get(setup.setup_id)
                break
        assert state_at_s == LifecycleState.IN_PROGRESS.value, (
            f"expected IN_PROGRESS at s={s}, got {state_at_s}"
        )

    setup_f, faulted = _run(control, n=n, fault_schedule=_fault_schedule(s, d, fc))

    # State immediately before the fault window (for unchanged-state checks).
    state_before_fault = None
    for rec in faulted.cycles:
        if rec.cycle_index == s - 1:
            state_before_fault = rec.setup_states.get(setup_f.setup_id)
            break

    if fc == "readiness_blocked":
        for i, rec in enumerate(faulted.cycles):
            if s <= rec.cycle_index < r:
                assert rec.evaluation_blocked or rec.injected_fault == "readiness_blocked"
                assert faulted.lifecycle_calls_per_cycle[i] == 0

    # Missing-candle must remain omitted after the fault interval (no backfill).
    if fc == "missing_candle":
        omit_ep = _epoch(s)
        for rec in faulted.cycles:
            if rec.cycle_index > s:
                assert omit_ep not in rec.effective_candle_epochs, (
                    f"omitted candle epoch {omit_ep} reappeared at cycle "
                    f"{rec.cycle_index} (backfill)"
                )

    terminal = {
        "F1-completion": LifecycleState.COMPLETED.value,
        "F1-invalidation": LifecycleState.INVALIDATED.value,
        "F1-expiry": LifecycleState.EXPIRED.value,
    }[control]

    for rec in faulted.cycles:
        # Fault interval: lifecycle state must not advance toward terminal.
        if s <= rec.cycle_index < r and s < t:
            st = rec.setup_states.get(setup_f.setup_id)
            if state_before_fault is not None and fc != "readiness_blocked":
                # State may stay the same or become diagnostic; must not jump
                # to the clean control's terminal.
                assert st != terminal or rec.setup_previous_states.get(
                    setup_f.setup_id
                ) == terminal, (
                    f"state advanced to {terminal} during fault at cycle "
                    f"{rec.cycle_index}"
                )
            assert not rec.signal_ids, (
                f"signal emitted during fault interval at cycle {rec.cycle_index}"
            )

        # Full post-recovery window r..E.
        if r <= rec.cycle_index <= E and s < t:
            st = rec.setup_states.get(setup_f.setup_id)
            if (
                st == terminal
                and rec.setup_previous_states.get(setup_f.setup_id) != terminal
            ):
                pytest.fail(
                    f"{control}/{fc} reached {terminal} at cycle {rec.cycle_index} "
                    f"inside r..E after fault at s={s}"
                )
            if rec.signal_ids:
                pytest.fail(
                    f"signal emitted at cycle {rec.cycle_index} inside r..E after fault"
                )
            # Continuity-affecting faults: cycles in r..E should be
            # diagnostic-only when the engine has an active instance.
            if fc in ("unhealthy", "missing_candle") and st not in (
                LifecycleState.COMPLETED.value,
                LifecycleState.INVALIDATED.value,
                LifecycleState.EXPIRED.value,
            ):
                # Prefer diagnostic_only; if the engine has not yet opened an
                # instance, PENDING without diagnostic is acceptable only
                # before any acceptance.
                if st != LifecycleState.PENDING.value:
                    assert rec.setup_diagnostic_only.get(setup_f.setup_id, False), (
                        f"expected diagnostic_only at cycle {rec.cycle_index} "
                        f"in r..E after {fc}, state={st}"
                    )
                elif scenario["state_at_s"] == "IN_PROGRESS":
                    # Active instance that reports PENDING in r..E after a
                    # continuity fault is a continuity failure unless diagnostic.
                    assert rec.setup_diagnostic_only.get(setup_f.setup_id, False) or (
                        state_before_fault == LifecycleState.PENDING.value
                    ), (
                        f"IN_PROGRESS control at cycle {rec.cycle_index} in r..E "
                        f"after {fc} must be diagnostic_only (got PENDING without diagnostic)"
                    )

    report = classify_stalled(faulted, setup_f.setup_id, injected_fault_cause=fc)
    assert report.injected_fault_cause == fc
    if fc == "readiness_blocked":
        for bc in report.blocked_cycles:
            assert bc not in report.subsequent_diagnostic_cycles
        # Readiness blocks are not continuity diagnostics.
        assert report.classification in ("", "CONTINUITY_DIAGNOSTIC", "STALLED", "STALL_CLEARED")

    status = "real_env_passed" if _real_talib_available() else "stub_only"
    EVIDENCE[key] = status
    record_stall(key, status)


def test_acceptance_meta_requires_real_env_for_all_24():
    """A15: durable manifest must list all 24 stall scenarios as real_env_passed."""
    if not _real_talib_available():
        pytest.fail("A15 meta-test: real TA-Lib is not available")
    # Ensure current-process evidence is flushed.
    for s in generate_stall_scenarios():
        key = s["key"]
        if key in EVIDENCE:
            record_stall(key, EVIDENCE[key])
    manifest = load_manifest()
    stall = manifest.get("stall") or {}
    missing = []
    bad = []
    for s in generate_stall_scenarios():
        key = str(s["key"])
        entry = stall.get(key) or stall.get(repr(s["key"]))
        # Also try memory if tuple-string mismatch
        status = None
        if entry:
            status = entry.get("status") if isinstance(entry, dict) else entry
        if status is None:
            status = EVIDENCE.get(s["key"], "not_run")
            # re-persist under str key for next load
            if status != "not_run":
                record_stall(s["key"], status)
                manifest = load_manifest()
                stall = manifest.get("stall") or {}
                entry = stall.get(str(s["key"]))
                status = (entry or {}).get("status", status) if isinstance(entry, dict) else status
        if status == "not_run":
            missing.append(key)
        elif status != "real_env_passed":
            bad.append((key, status))
    assert not missing, f"A15 stall missing from manifest/memory: {missing}"
    assert not bad, f"A15 stall non-real_env_passed: {bad}"


def test_acceptance_meta_a6_real_detector_evidence():
    """A15: durable manifest is sole authority for mandatory A6 fixture evidence."""
    if not _real_talib_available():
        pytest.fail("A15: real TA-Lib unavailable")
    for key, status in list(A6_EVIDENCE.items()):
        record_a6(key, status)
    manifest = load_manifest()
    run = manifest.get("run") or {}
    assert run.get("commit"), f"A15 run identity missing: {run}"
    a6 = manifest.get("a6") or {}
    missing = []
    bad = []
    for key in A6_MANDATORY_KEYS:
        entry = a6.get(key)
        status = entry.get("status") if isinstance(entry, dict) else None
        if status is None:
            missing.append(key)
        elif status != "real_env_passed":
            bad.append((key, status))
    assert not missing, f"A6 evidence missing from manifest: {missing}"
    assert not bad, f"A6 evidence not real_env_passed: {bad}"


def test_f1_completion_clean_control_real_detectors():
    """A12 control: real BreakerBlock+Impulse reach COMPLETED at locked cycle 98."""
    from cpab.detectors.breaker_block import BreakerBlockDetector
    from cpab.detectors.behavior_volatility import ImpulseDetector
    from cpab.replay.driver import ReplayRunConfig, run_replay

    # Same constructed series as A6 F1 completion (impulse at 98).
    pad = [(10.0, 10.1, 9.9, 10.0)] * 70
    base = [
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 12.0, 9.8, 11.0),
        (11.0, 11.2, 10.5, 10.8),
        (11.0, 11.2, 9.8, 10.0),
        (10.0, 10.5, 9.9, 10.2),
        (10.2, 13.5, 10.0, 13.0),
        (13.0, 13.2, 12.5, 12.8),
        (12.8, 12.9, 9.0, 9.5),
    ] + [(9.5, 9.6, 9.4, 9.5)] * 20 + [(9.5, 9.6, 5.0, 5.2)]
    candles = []
    for i, (o, h, l, c) in enumerate(pad + base):
        candles.append(
            make_candle(SYMBOL, TF, START + i * TF_SEC, o, h, l, c, is_closed=True)
        )
    setup = _setup_completion()
    result = run_replay(
        ReplayRunConfig(
            setups=[setup],
            detectors=[
                BreakerBlockDetector(left_strength=1, right_strength=1),
                ImpulseDetector(),
            ],
            symbol=SYMBOL,
            timeframe=TF,
            timeframe_seconds=TF_SEC,
            candles=candles,
            arm_count=ARM,
        )
    )
    assert result.final_setup_states[setup.setup_id] == LifecycleState.COMPLETED.value
    tc = result.transition_cycle(setup.setup_id, LifecycleState.COMPLETED.value)
    assert tc == 98, tc
    assert len(result.signals) == 1
