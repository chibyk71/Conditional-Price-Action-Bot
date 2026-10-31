"""Thin replay driver around ConditionEngine.evaluate_once() (Phase 8B R1–R6)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from cpab.conditions.analyst import report_setup_types
from cpab.conditions.lifecycle import LifecycleState, SetupInstance
from cpab.conditions.schema import SetupDef
from cpab.core.engine import ConditionEngine
from cpab.core.monitoring import MonitoringSnapshot
from cpab.core.observation_store import ObservationStore
from cpab.detectors.base import PriceActionDetector
from cpab.market.readiness import ReadinessManager
from cpab.models import Candle, Observation, ReadinessState, Signal


@dataclass
class ReplayRunConfig:
    """Configuration for one independent replay run."""

    setups: Sequence[SetupDef]
    detectors: Sequence[PriceActionDetector]
    symbol: str
    timeframe: str
    timeframe_seconds: int
    # Full closed-candle series in causal order (all is_closed=True).
    candles: Sequence[Candle]
    # Arm after this many candles are present; first evaluation uses candles[0:arm_count].
    arm_count: int = 1
    readiness_ready: bool = True
    series_healthy: bool = True
    # Optional per-cycle overrides keyed by evaluation cycle index (0 = first evaluate).
    # Values: {"series_healthy": bool} | {"readiness_ready": bool} | {"skip_candle": True}
    fault_schedule: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class CycleRecord:
    """Per-cycle evidence required by R6."""

    cycle_index: int
    candle_epoch: Optional[int]
    candle_ohlc: Optional[tuple[float, float, float, float]]
    readiness_state: str
    data_ok: bool
    data_diagnostics: tuple[str, ...]
    evaluation_blocked: bool
    emitted_observation_ids: tuple[str, ...]
    newly_accepted_observation_ids: tuple[str, ...]
    setup_states: dict[str, str]  # setup_id -> LifecycleState value
    setup_previous_states: dict[str, str]
    setup_reasons: dict[str, tuple[str, ...]]
    setup_anchors: dict[str, Optional[int]]
    setup_diagnostic_only: dict[str, bool]
    signal_ids: tuple[str, ...]
    monitoring: Optional[dict[str, Any]] = None
    # Effective inputs actually visible to the engine this cycle (for parity).
    effective_candle_epochs: tuple[int, ...] = ()
    # Engine-boundary OHLC of each effective completed candle (epoch, o, h, l, c).
    effective_candle_ohlcs: tuple[tuple[int, float, float, float, float], ...] = ()
    effective_detector_names: tuple[str, ...] = ()
    injected_fault: Optional[str] = None


@dataclass
class ReplayResult:
    """Full result of one independent replay run."""

    cycles: list[CycleRecord]
    final_setup_states: dict[str, str]
    signals: list[Signal]
    type_report_warnings: tuple[str, ...]
    required_types: tuple[str, ...]
    wired_types: tuple[str, ...]
    # Lifecycle call count per cycle (0 when readiness blocked).
    lifecycle_calls_per_cycle: list[int] = field(default_factory=list)
    input_diagnostics: tuple[str, ...] = ()

    def terminal_state(self, setup_id: str) -> Optional[str]:
        return self.final_setup_states.get(setup_id)

    def transition_cycle(self, setup_id: str, state: str) -> Optional[int]:
        for rec in self.cycles:
            if rec.setup_states.get(setup_id) == state:
                prev = rec.setup_previous_states.get(setup_id)
                if prev != state:
                    return rec.cycle_index
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycles": [asdict(c) for c in self.cycles],
            "final_setup_states": dict(self.final_setup_states),
            "signal_ids": [s.id for s in self.signals],
            "type_report_warnings": list(self.type_report_warnings),
            "required_types": list(self.required_types),
            "wired_types": list(self.wired_types),
            "lifecycle_calls_per_cycle": list(self.lifecycle_calls_per_cycle),
            "input_diagnostics": list(self.input_diagnostics),
        }


class _ReplayMarket:
    """Minimal market adapter for causal one-candle-at-a-time replay."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        granularity: int,
        *,
        series_healthy: bool = True,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._granularity = granularity
        self._candles: list[Candle] = []
        self._series_healthy = series_healthy
        self._gran_by_name = {timeframe: granularity}

    def set_candles(self, candles: Sequence[Candle]) -> None:
        self._candles = [
            c
            for c in candles
            if c.is_closed and c.symbol == self.symbol and c.timeframe == self.timeframe
        ]

    def set_healthy(self, ok: bool) -> None:
        self._series_healthy = ok

    def completed_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        return list(self._candles)

    def timeframe_seconds(self, timeframe: str) -> Optional[int]:
        if timeframe == self.timeframe:
            return self._granularity
        return None

    def series_ok(self, symbol: str, timeframe: str) -> bool:
        return self._series_healthy


def run_replay(config: ReplayRunConfig) -> ReplayResult:
    """Execute one independent causal replay run with fresh state (R1–R6).

    Advances one completed candle per evaluation step. No forming or future
    candles are visible. Detector list order is preserved exactly as supplied.
    """
    if not config.candles:
        raise ValueError("replay requires a non-empty completed-candle series")
    if config.arm_count < 1:
        raise ValueError("arm_count must be >= 1")

    detectors = list(config.detectors)
    setups = list(config.setups)

    # Type wiring report (warning only for valid-but-unwired).
    warnings: list[str] = []
    required: set[str] = set()
    for setup in setups:
        rep = report_setup_types(setup, detectors)
        warnings.extend(rep.warnings)
        for st in rep.type_statuses:
            required.add(st.type)

    market = _ReplayMarket(
        config.symbol,
        config.timeframe,
        config.timeframe_seconds,
        series_healthy=config.series_healthy,
    )
    readiness = ReadinessManager()
    readiness.register_series(config.symbol, config.timeframe)
    if config.readiness_ready:
        readiness.set_series(config.symbol, config.timeframe, True)
    else:
        # Leave not ready / force blocked path via readiness state if available.
        pass

    store = ObservationStore()
    engine = ConditionEngine(
        market,
        detectors,
        [],
        readiness,
        setups=setups,
        observation_store=store,
        timeframe_seconds_by_name={config.timeframe: config.timeframe_seconds},
    )

    # Preserve caller order. Do not silently sort or drop forming bars — record
    # input diagnostics when non-closed or out-of-order bars are present.
    input_diagnostics: list[str] = []
    all_candles: list[Candle] = []
    prev_epoch: Optional[int] = None
    for c in config.candles:
        if c.symbol != config.symbol or c.timeframe != config.timeframe:
            input_diagnostics.append(
                f"skipped candle symbol/tf mismatch epoch={getattr(c, 'epoch', None)}"
            )
            continue
        if not c.is_closed:
            input_diagnostics.append(f"forming_candle_visible epoch={c.epoch}")
            # Still exclude from evaluation input (contract: completed only),
            # but the diagnostic is recorded on the result.
            continue
        if prev_epoch is not None and c.epoch < prev_epoch:
            input_diagnostics.append(
                f"out_of_order epoch={c.epoch} after {prev_epoch}"
            )
        if prev_epoch is not None and c.epoch == prev_epoch:
            input_diagnostics.append(f"duplicate_epoch epoch={c.epoch}")
        all_candles.append(c)
        prev_epoch = c.epoch

    cycles: list[CycleRecord] = []
    all_signals: list[Signal] = []
    lifecycle_calls: list[int] = []
    detector_names = tuple(
        getattr(d, "name", type(d).__name__) for d in detectors
    )
    # Epochs omitted by missing-candle faults stay omitted for the rest of the
    # run (A14: no backfill / no gap repair). Later candles still arrive.
    omitted_epochs: set[int] = set()

    # Arm: present first arm_count candles, then evaluate once per additional candle.
    # cycle_index = series index of the latest visible candle (contract-aligned:
    # first evaluation uses index arm_count-1, e.g. 59 when arm_count=60).
    n = len(all_candles)
    start_eval_at = max(config.arm_count, 1)
    for end_idx in range(start_eval_at, n + 1):
        cycle_idx = end_idx - 1
        fault = config.fault_schedule.get(cycle_idx, {})
        injected: Optional[str] = None

        # Apply fault schedule before building the visible series so a
        # missing-candle fault permanently omits that bar.
        series_ok = config.series_healthy
        if "series_healthy" in fault:
            series_ok = bool(fault["series_healthy"])
            if not series_ok:
                injected = "unhealthy"
        if fault.get("skip_candle"):
            # Permanently omit the candle that would have been the latest at
            # this cycle index. Subsequent cycles must not reintroduce it.
            omit_c = all_candles[cycle_idx] if cycle_idx < len(all_candles) else None
            if omit_c is not None:
                omitted_epochs.add(omit_c.epoch)
            injected = injected or "missing_candle"

        visible = [
            c
            for c in all_candles[:end_idx]
            if c.epoch not in omitted_epochs
        ]
        market.set_candles(visible)
        market.set_healthy(series_ok)

        readiness_ready = config.readiness_ready
        if "readiness_ready" in fault:
            readiness_ready = bool(fault["readiness_ready"])
            if not readiness_ready:
                injected = injected or "readiness_blocked"
        # Force readiness state.
        if readiness_ready:
            readiness.set_series(config.symbol, config.timeframe, True)
        else:
            # Mark series not ok so can_evaluate is false when possible.
            readiness.set_series(config.symbol, config.timeframe, False)

        # Snapshot instance states before evaluation.
        prev_states = {
            sid: inst.state.value for sid, inst in engine.setup_instances.items()
        }
        for setup in setups:
            if setup.setup_id not in prev_states:
                prev_states[setup.setup_id] = LifecycleState.PENDING.value

        signals = engine.evaluate_once()
        all_signals.extend(signals)
        mon: MonitoringSnapshot = engine.last_monitoring_snapshot

        # Lifecycle calls: 0 when readiness blocked the whole cycle.
        blocked = bool(mon.evaluation_blocked) if mon is not None else False
        if not readiness_ready:
            blocked = True
            lc_calls = 0
        else:
            lc_calls = 1  # one evaluate path per series when not blocked
        lifecycle_calls.append(lc_calls)

        latest = visible[-1] if visible else None
        ohlc = (
            (latest.open, latest.high, latest.low, latest.close)
            if latest is not None
            else None
        )
        # Collect setup state after evaluation.
        setup_states: dict[str, str] = {}
        setup_prev: dict[str, str] = {}
        setup_reasons: dict[str, tuple[str, ...]] = {}
        setup_anchors: dict[str, Optional[int]] = {}
        setup_diag: dict[str, bool] = {}
        for setup in setups:
            inst = engine.get_instance(setup.setup_id)
            if inst is None:
                setup_states[setup.setup_id] = LifecycleState.PENDING.value
                setup_prev[setup.setup_id] = prev_states.get(
                    setup.setup_id, LifecycleState.PENDING.value
                )
                setup_reasons[setup.setup_id] = ()
                setup_anchors[setup.setup_id] = None
                setup_diag[setup.setup_id] = False
            else:
                setup_states[setup.setup_id] = inst.state.value
                setup_prev[setup.setup_id] = prev_states.get(
                    setup.setup_id, LifecycleState.PENDING.value
                )
                setup_reasons[setup.setup_id] = tuple(inst.last_transition_reasons)
                setup_anchors[setup.setup_id] = inst.anchor_candle_epoch
                # Diagnostic-only inferred from monitoring setup views when present.
                diag = False
                if mon is not None:
                    for sv in mon.setups:
                        if sv.setup_id == setup.setup_id:
                            diag = bool(sv.diagnostic_only)
                            if sv.reasons:
                                setup_reasons[setup.setup_id] = tuple(sv.reasons)
                            break
                setup_diag[setup.setup_id] = diag

        data_ok = True
        data_diag: tuple[str, ...] = ()
        if mon is not None and mon.series:
            for sv in mon.series:
                if sv.symbol == config.symbol and sv.timeframe == config.timeframe:
                    data_ok = sv.data_ok
                    data_diag = tuple(sv.diagnostics)
                    break
        if not series_ok:
            data_ok = False
            if not data_diag:
                data_diag = ("series_ok reported False",)

        emitted_ids = tuple(mon.detector_emitted_observation_ids) if mon else ()
        accepted_ids = tuple(mon.newly_accepted_observation_ids) if mon else ()
        sig_ids = tuple(s.id for s in signals)

        rec = CycleRecord(
            cycle_index=cycle_idx,
            candle_epoch=latest.epoch if latest else None,
            candle_ohlc=ohlc,
            readiness_state=readiness.state.value,
            data_ok=data_ok,
            data_diagnostics=data_diag,
            evaluation_blocked=blocked,
            emitted_observation_ids=emitted_ids,
            newly_accepted_observation_ids=accepted_ids,
            setup_states=setup_states,
            setup_previous_states=setup_prev,
            setup_reasons=setup_reasons,
            setup_anchors=setup_anchors,
            setup_diagnostic_only=setup_diag,
            signal_ids=sig_ids,
            monitoring={
                "evaluation_blocked": mon.evaluation_blocked if mon else blocked,
                "signal_count": mon.signal_count if mon else len(signals),
            },
            effective_candle_epochs=tuple(c.epoch for c in visible),
            effective_candle_ohlcs=tuple(
                (c.epoch, float(c.open), float(c.high), float(c.low), float(c.close))
                for c in visible
            ),
            effective_detector_names=detector_names,
            injected_fault=injected,
        )
        cycles.append(rec)

    final_states = {
        sid: inst.state.value for sid, inst in engine.setup_instances.items()
    }
    for setup in setups:
        final_states.setdefault(setup.setup_id, LifecycleState.PENDING.value)

    # Wired types from detector name heuristics.
    from cpab.conditions.analyst import _detector_emittable_types

    wired = sorted(_detector_emittable_types(detectors))

    return ReplayResult(
        cycles=cycles,
        final_setup_states=final_states,
        signals=all_signals,
        type_report_warnings=tuple(dict.fromkeys(warnings)),
        required_types=tuple(sorted(required)),
        wired_types=tuple(wired),
        lifecycle_calls_per_cycle=lifecycle_calls,
        input_diagnostics=tuple(input_diagnostics),
    )
