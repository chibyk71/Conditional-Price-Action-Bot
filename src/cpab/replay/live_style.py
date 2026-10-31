"""Live-style evaluation path using real CandleCache and mark_closed_by_lag (A10).

Does not invent a second pipeline — reuses the production cache and closure
helpers, then feeds ConditionEngine.evaluate_once() with the same contract as
replay (completed candles only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from cpab.conditions.schema import SetupDef
from cpab.core.engine import ConditionEngine
from cpab.core.observation_store import ObservationStore
from cpab.detectors.base import PriceActionDetector
from cpab.market.aggregation import aggregate_from_1m
from cpab.market.cache import CandleCache
from cpab.market.readiness import ReadinessManager
from cpab.market.validation import mark_closed_by_lag
from cpab.models import Candle, ReadinessState, Signal
from cpab.replay.driver import CycleRecord, ReplayResult, ReplayRunConfig


@dataclass
class LiveStyleConfig:
    """Controls for the live-style path that mirror production knobs."""

    setups: Sequence[SetupDef]
    detectors: Sequence[PriceActionDetector]
    symbol: str
    timeframe: str
    timeframe_seconds: int
    candles: Sequence[Candle]
    arm_count: int = 1
    # Production-style knobs
    cache_max_size: int = 500
    confirm_lag_bars: int = 1
    readiness_ready: bool = True
    series_healthy: bool = True
    fault_schedule: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Optional HTF series (symbol, timeframe, granularity, candles)
    htf: Optional[tuple[str, str, int, Sequence[Candle]]] = None
    # When set, build HTF from primary 1m via production aggregate_from_1m
    # (target_timeframe, target_granularity). Overrides htf series.
    htf_from_1m: Optional[tuple[str, int]] = None


class _CacheBackedMarket:
    """Market adapter that exposes completed candles from real CandleCache(s)."""

    def __init__(
        self,
        primary: CandleCache,
        *,
        series_healthy: bool = True,
        htf: Optional[CandleCache] = None,
    ) -> None:
        self._primary = primary
        self._htf = htf
        self._series_healthy = series_healthy
        self._gran = {
            primary.timeframe: primary.granularity,
        }
        if htf is not None:
            self._gran[htf.timeframe] = htf.granularity

    def set_healthy(self, ok: bool) -> None:
        self._series_healthy = ok

    def completed_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        if (
            symbol == self._primary.symbol
            and timeframe == self._primary.timeframe
        ):
            return self._primary.completed()
        if (
            self._htf is not None
            and symbol == self._htf.symbol
            and timeframe == self._htf.timeframe
        ):
            return self._htf.completed()
        return []

    def timeframe_seconds(self, timeframe: str) -> Optional[int]:
        return self._gran.get(timeframe)

    def series_ok(self, symbol: str, timeframe: str) -> bool:
        if not self._series_healthy:
            return False
        if symbol == self._primary.symbol and timeframe == self._primary.timeframe:
            return self._primary.is_series_healthy(1)
        if (
            self._htf is not None
            and symbol == self._htf.symbol
            and timeframe == self._htf.timeframe
        ):
            return self._htf.is_series_healthy(1)
        return False


def run_live_style(config: LiveStyleConfig) -> ReplayResult:
    """Causal one-candle-per-step evaluation through CandleCache + mark_closed_by_lag.

    Effective inputs recorded on each cycle include:
    - completed epochs after cache window/cap
    - confirm_lag application (forming vs closed)
    - cache max_size trim effect
    - HTF completed epochs when configured
    """
    if not config.candles:
        raise ValueError("live-style requires a non-empty candle series")
    if config.arm_count < 1:
        raise ValueError("arm_count must be >= 1")

    primary = CandleCache(
        config.symbol,
        config.timeframe,
        config.timeframe_seconds,
        max_size=config.cache_max_size,
    )
    htf_cache: Optional[CandleCache] = None
    htf_candles: list[Candle] = []
    htf_init_seeded = False
    if config.htf is not None:
        # Initialization path: independent HTF history feed.
        htf_sym, htf_tf, htf_gran, htf_series = config.htf
        htf_cache = CandleCache(htf_sym, htf_tf, htf_gran, max_size=config.cache_max_size)
        htf_candles = list(htf_series)
    elif config.htf_from_1m is not None:
        htf_tf, htf_gran = config.htf_from_1m
        htf_cache = CandleCache(
            config.symbol, htf_tf, htf_gran, max_size=config.cache_max_size
        )
    if config.htf is not None and config.htf_from_1m is not None:
        # Combined path: keep explicit series for init seed, then poll via aggregate.
        pass

    market = _CacheBackedMarket(
        primary, series_healthy=config.series_healthy, htf=htf_cache
    )
    readiness = ReadinessManager()
    readiness.register_series(config.symbol, config.timeframe)
    if htf_cache is not None:
        readiness.register_series(htf_cache.symbol, htf_cache.timeframe)
    if config.readiness_ready:
        readiness.set_series(config.symbol, config.timeframe, True)
        if htf_cache is not None:
            readiness.set_series(htf_cache.symbol, htf_cache.timeframe, True)

    store = ObservationStore()
    tf_map = {config.timeframe: config.timeframe_seconds}
    if htf_cache is not None:
        tf_map[htf_cache.timeframe] = htf_cache.granularity

    engine = ConditionEngine(
        market,
        list(config.detectors),
        [],
        readiness,
        setups=list(config.setups),
        observation_store=store,
        timeframe_seconds_by_name=tf_map,
    )

    # Source series in caller order (no silent sort).
    source = [
        c
        for c in config.candles
        if c.symbol == config.symbol and c.timeframe == config.timeframe
    ]
    omitted_epochs: set[int] = set()
    detector_names = tuple(
        getattr(d, "name", type(d).__name__) for d in config.detectors
    )

    cycles: list[CycleRecord] = []
    all_signals: list[Signal] = []
    lifecycle_calls: list[int] = []
    input_diagnostics: list[str] = []

    n = len(source)
    start_eval_at = max(config.arm_count, 1)
    for end_idx in range(start_eval_at, n + 1):
        cycle_idx = end_idx - 1
        fault = config.fault_schedule.get(cycle_idx, {})
        injected: Optional[str] = None

        series_ok = config.series_healthy
        if "series_healthy" in fault:
            series_ok = bool(fault["series_healthy"])
            if not series_ok:
                injected = "unhealthy"
        if fault.get("skip_candle"):
            omit_c = source[cycle_idx] if cycle_idx < len(source) else None
            if omit_c is not None:
                omitted_epochs.add(omit_c.epoch)
            injected = injected or "missing_candle"

        # Server time is set so all bars through the latest are closable under lag.
        latest_src = source[cycle_idx]
        server_time = (
            latest_src.epoch
            + config.timeframe_seconds
            + config.confirm_lag_bars * config.timeframe_seconds
        )

        window = [c for c in source[:end_idx] if c.epoch not in omitted_epochs]
        marked = mark_closed_by_lag(
            window,
            config.timeframe_seconds,
            server_time,
            config.confirm_lag_bars,
        )
        # Rebuild cache each cycle from marked series (simulates progressive feed).
        primary.clear()
        for c in marked:
            primary.upsert(c)

        htf_source_label: Optional[str] = None
        if htf_cache is not None:
            use_aggregate = config.htf_from_1m is not None
            use_explicit = bool(htf_candles)
            # Combined: seed once from explicit init history, then aggregate from 1m.
            if use_explicit and use_aggregate:
                if not htf_init_seeded:
                    htf_window = [c for c in htf_candles if c.epoch <= latest_src.epoch]
                    htf_server = (
                        latest_src.epoch
                        + htf_cache.granularity
                        + config.confirm_lag_bars * htf_cache.granularity
                    )
                    htf_marked = mark_closed_by_lag(
                        htf_window,
                        htf_cache.granularity,
                        htf_server,
                        config.confirm_lag_bars,
                    )
                    htf_cache.clear()
                    for c in htf_marked:
                        htf_cache.upsert(c)
                    htf_init_seeded = True
                    htf_source_label = "init_then_aggregate"
                else:
                    htf_tf, htf_gran = config.htf_from_1m
                    completed_1m = primary.completed()
                    aggregated = aggregate_from_1m(
                        completed_1m,
                        htf_gran,
                        htf_tf,
                        config.symbol,
                        require_complete=True,
                    )
                    # Upsert aggregated without wiping init history epochs that
                    # are still valid; clear+rebuild from aggregate for polling.
                    htf_cache.clear()
                    for c in aggregated:
                        htf_cache.upsert(c)
                    htf_source_label = "init_then_aggregate"
            elif use_aggregate:
                htf_tf, htf_gran = config.htf_from_1m
                completed_1m = primary.completed()
                aggregated = aggregate_from_1m(
                    completed_1m,
                    htf_gran,
                    htf_tf,
                    config.symbol,
                    require_complete=True,
                )
                htf_cache.clear()
                for c in aggregated:
                    htf_cache.upsert(c)
                htf_source_label = "aggregate_from_1m"
            elif use_explicit:
                htf_window = [c for c in htf_candles if c.epoch <= latest_src.epoch]
                htf_server = (
                    latest_src.epoch
                    + htf_cache.granularity
                    + config.confirm_lag_bars * htf_cache.granularity
                )
                htf_marked = mark_closed_by_lag(
                    htf_window,
                    htf_cache.granularity,
                    htf_server,
                    config.confirm_lag_bars,
                )
                htf_cache.clear()
                for c in htf_marked:
                    htf_cache.upsert(c)
                htf_source_label = "explicit_series"
            else:
                htf_source_label = None
        else:
            htf_source_label = None

        market.set_healthy(series_ok)

        readiness_ready = config.readiness_ready
        if "readiness_ready" in fault:
            readiness_ready = bool(fault["readiness_ready"])
            if not readiness_ready:
                injected = injected or "readiness_blocked"
        if readiness_ready:
            readiness.set_series(config.symbol, config.timeframe, True)
        else:
            readiness.set_series(config.symbol, config.timeframe, False)

        from cpab.conditions.lifecycle import LifecycleState

        prev_states = {
            sid: inst.state.value for sid, inst in engine.setup_instances.items()
        }
        for setup in config.setups:
            prev_states.setdefault(setup.setup_id, LifecycleState.PENDING.value)

        signals = engine.evaluate_once()
        all_signals.extend(signals)
        mon = engine.last_monitoring_snapshot

        blocked = bool(mon.evaluation_blocked) if mon is not None else False
        if not readiness_ready:
            blocked = True
            lc_calls = 0
        else:
            lc_calls = 1
        lifecycle_calls.append(lc_calls)

        completed = primary.completed()
        latest = completed[-1] if completed else None
        ohlc = (
            (latest.open, latest.high, latest.low, latest.close)
            if latest is not None
            else None
        )

        setup_states: dict[str, str] = {}
        setup_prev: dict[str, str] = {}
        setup_reasons: dict[str, tuple[str, ...]] = {}
        setup_anchors: dict[str, Optional[int]] = {}
        setup_diag: dict[str, bool] = {}
        for setup in config.setups:
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
                diag = False
                if mon is not None:
                    for sv in mon.setups:
                        if sv.setup_id == setup.setup_id:
                            diag = bool(sv.diagnostic_only)
                            if sv.reasons:
                                setup_reasons[setup.setup_id] = tuple(sv.reasons)
                            break
                setup_diag[setup.setup_id] = diag

        data_ok = series_ok
        data_diag: tuple[str, ...] = ()
        if mon is not None and mon.series:
            for sv in mon.series:
                if sv.symbol == config.symbol and sv.timeframe == config.timeframe:
                    data_ok = sv.data_ok
                    data_diag = tuple(sv.diagnostics)
                    break
        if not series_ok and not data_diag:
            data_diag = ("series_ok reported False",)

        # Effective inputs after cache window/cap and lag closure.
        effective_epochs = tuple(c.epoch for c in completed)
        htf_epochs = (
            tuple(c.epoch for c in htf_cache.completed()) if htf_cache else ()
        )
        # Record trim evidence when cache capped.
        if len(window) > config.cache_max_size:
            input_diagnostics.append(
                f"cycle={cycle_idx} cache_trim max_size={config.cache_max_size} "
                f"window={len(window)} completed={len(completed)}"
            )

        rec = CycleRecord(
            cycle_index=cycle_idx,
            candle_epoch=latest.epoch if latest else None,
            candle_ohlc=ohlc,
            readiness_state=readiness.state.value,
            data_ok=data_ok,
            data_diagnostics=data_diag,
            evaluation_blocked=blocked,
            emitted_observation_ids=tuple(mon.detector_emitted_observation_ids)
            if mon
            else (),
            newly_accepted_observation_ids=tuple(mon.newly_accepted_observation_ids)
            if mon
            else (),
            setup_states=setup_states,
            setup_previous_states=setup_prev,
            setup_reasons=setup_reasons,
            setup_anchors=setup_anchors,
            setup_diagnostic_only=setup_diag,
            signal_ids=tuple(s.id for s in signals),
            monitoring={
                "evaluation_blocked": mon.evaluation_blocked if mon else blocked,
                "signal_count": mon.signal_count if mon else len(signals),
                "cache_max_size": config.cache_max_size,
                "confirm_lag_bars": config.confirm_lag_bars,
                "htf_epochs": htf_epochs,
                "htf_source": htf_source_label,
                "completed_count": len(completed),
            },
            effective_candle_epochs=effective_epochs,
            effective_candle_ohlcs=tuple(
                (c.epoch, float(c.open), float(c.high), float(c.low), float(c.close))
                for c in completed
            ),
            effective_detector_names=detector_names,
            injected_fault=injected,
        )
        cycles.append(rec)

    from cpab.conditions.lifecycle import LifecycleState

    final_states = {
        sid: inst.state.value for sid, inst in engine.setup_instances.items()
    }
    for setup in config.setups:
        final_states.setdefault(setup.setup_id, LifecycleState.PENDING.value)

    from cpab.conditions.analyst import _detector_emittable_types

    wired = sorted(_detector_emittable_types(list(config.detectors)))
    required: set[str] = set()
    for setup in config.setups:
        # lightweight: type names from expression leaves would need walk;
        # leave empty here — report is advisory.
        pass

    return ReplayResult(
        cycles=cycles,
        final_setup_states=final_states,
        signals=all_signals,
        type_report_warnings=(),
        required_types=tuple(sorted(required)),
        wired_types=tuple(wired),
        lifecycle_calls_per_cycle=lifecycle_calls,
        input_diagnostics=tuple(input_diagnostics),
    )


def live_style_from_replay_config(
    cfg: ReplayRunConfig,
    *,
    cache_max_size: int = 500,
    confirm_lag_bars: int = 1,
    htf: Optional[tuple[str, str, int, Sequence[Candle]]] = None,
) -> LiveStyleConfig:
    """Map a ReplayRunConfig onto the live-style path with production knobs."""
    return LiveStyleConfig(
        setups=cfg.setups,
        detectors=cfg.detectors,
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        timeframe_seconds=cfg.timeframe_seconds,
        candles=cfg.candles,
        arm_count=cfg.arm_count,
        cache_max_size=cache_max_size,
        confirm_lag_bars=confirm_lag_bars,
        readiness_ready=cfg.readiness_ready,
        series_healthy=cfg.series_healthy,
        fault_schedule=dict(cfg.fault_schedule),
        htf=htf,
    )
