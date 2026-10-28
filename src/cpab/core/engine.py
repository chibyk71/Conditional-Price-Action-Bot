"""Condition engine: detectors → observations → conditions / setups → signals."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional, Sequence

from cpab.conditions.evaluator import ConditionEvaluator
from cpab.conditions.lifecycle import (
    LifecycleState,
    SetupInstance,
    SetupLifecycleEvaluator,
    new_pending_instance,
)
from cpab.conditions.schema import SetupDef
from cpab.core.observation_store import ObservationStore
from cpab.detectors.base import PriceActionDetector
from cpab.market.readiness import ReadinessManager
from cpab.market.service import MarketDataService
from cpab.models import ConditionDef, Observation, Signal
from cpab.logging_setup import get_logger

logger = get_logger("cpab.core.engine")

# Default retention: keep up to 1000 observations; age out after ~7 days of
# market seconds so multi-component setups spanning many bars remain available
# without unbounded growth. Callers may override via observation_store=.
_DEFAULT_MAX_SIZE = 1000
_DEFAULT_MAX_AGE_SECONDS = 7 * 24 * 3600

# Default instance key when the engine owns a single instance per setup config.
_DEFAULT_INSTANCE_KEY = "default"


def _deterministic_setup_signal_id(instance_id: str, candle_epoch: int) -> str:
    """Stable signal id for a setup completion (no randomness / wall-clock)."""
    payload = f"{instance_id}:{candle_epoch}:completed"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sigsetup:{digest[:24]}"


class ConditionEngine:
    """
    Runs detectors on completed candles only when readiness is READY.
    Retains observations across cycles (ObservationStore) so multi-component
    conditions/setups can combine facts confirmed on different bars.
    Evaluates legacy conditions and Phase 7A–7C setups, emitting explainable
    signals only when a setup instance transitions to COMPLETED (or a legacy
    condition is met).
    Deduplicates signals for the same (condition_id, candle_epoch) [legacy] or
    (instance_id, candle_epoch) [setups].

    Activation vs retention:
      - Store retains prior confirmed observations for causal availability.
      - Evaluator activation still requires at least one supporting observation
        newly associated with evaluation bar T (confirmed_at == T, etc.).
      - Retained priors alone never re-fire a condition on a later bar.
      - Setup lifecycle state is owned by the engine (one default instance per
        setup_id unless callers inject pre-built instances).
    """

    def __init__(
        self,
        market: MarketDataService,
        detectors: list[PriceActionDetector],
        conditions: list[ConditionDef],
        readiness: ReadinessManager,
        *,
        setups: Optional[Sequence[SetupDef]] = None,
        observation_store: Optional[ObservationStore] = None,
        lifecycle_evaluator: Optional[SetupLifecycleEvaluator] = None,
        timeframe_seconds_by_name: Optional[dict[str, int]] = None,
    ) -> None:
        self.market = market
        self.detectors = detectors
        self.conditions = list(conditions)
        self.setups = list(setups or [])
        self.readiness = readiness
        self.evaluator = ConditionEvaluator()
        self.lifecycle = lifecycle_evaluator or SetupLifecycleEvaluator()
        self._emitted: set[tuple[str, int]] = set()  # (condition_id|instance_id, candle_epoch)
        self._last_observations: list[Observation] = []
        # Use explicit None check — ObservationStore defines __len__ so an
        # empty store is falsy and must not be replaced by a new instance.
        self.store = (
            observation_store
            if observation_store is not None
            else ObservationStore(
                max_size=_DEFAULT_MAX_SIZE,
                max_age_seconds=_DEFAULT_MAX_AGE_SECONDS,
            )
        )
        # Optional explicit map; otherwise derived from market when possible.
        self._tf_seconds: dict[str, int] = dict(timeframe_seconds_by_name or {})
        # setup_id -> SetupInstance (engine-owned lifecycle state)
        self._instances: dict[str, SetupInstance] = {}
        for s in self.setups:
            if s.setup_id not in self._instances:
                self._instances[s.setup_id] = new_pending_instance(
                    s, instance_key=_DEFAULT_INSTANCE_KEY
                )

    @property
    def setup_instances(self) -> dict[str, SetupInstance]:
        """Current lifecycle instances keyed by setup_id (copy of mapping)."""
        return dict(self._instances)

    def get_instance(self, setup_id: str) -> Optional[SetupInstance]:
        return self._instances.get(setup_id)

    def evaluate_once(self) -> list[Signal]:
        if not self.readiness.can_evaluate():
            logger.debug(
                "Evaluation blocked readiness=%s reason=%s",
                self.readiness.state.value,
                self.readiness.reason,
            )
            return []

        signals: list[Signal] = []
        cycle_new: list[Observation] = []

        # Union of (symbol, timeframe) from enabled legacy conditions and setups
        pairs: set[tuple[str, str]] = set()
        for c in self.conditions:
            if c.enabled:
                pairs.add((c.symbol, c.timeframe))
        for s in self.setups:
            # Disabled setups still need instance retention for inspection, but
            # we only pull market series for enabled ones (disabled is diagnostic).
            if s.enabled:
                pairs.add((s.symbol, s.timeframe))
        # Always include disabled setup series if we already have an instance
        # so continuity/diagnostic evaluation can still run when data is present.
        for s in self.setups:
            pairs.add((s.symbol, s.timeframe))

        for symbol, timeframe in pairs:
            try:
                candles = self.market.completed_candles(symbol, timeframe)
            except KeyError:
                logger.warning("No cache for %s/%s", symbol, timeframe)
                continue
            if not candles:
                continue

            # Detectors only see completed candles; emissions are new for this cycle
            new_obs: list[Observation] = []
            for det in self.detectors:
                emitted = det.detect(candles)
                new_obs.extend(emitted)
            cycle_new.extend(new_obs)

            # Retain newly emitted facts for future bars (deduped by id)
            self.store.accept(new_obs)

            latest = candles[-1]
            # Age-based prune relative to current market time
            self.store.prune(latest.epoch)

            # Evaluation set = all retained observations for this series
            eval_obs = self.store.query(symbol, timeframe)

            # --- Legacy conditions (unchanged semantics) ---
            relevant = [
                c
                for c in self.conditions
                if c.enabled and c.symbol == symbol and c.timeframe == timeframe
            ]
            for cond in relevant:
                result = self.evaluator.evaluate(
                    cond, eval_obs, candle_epoch=latest.epoch
                )
                if not result.met:
                    continue
                key = (cond.id, latest.epoch)
                if key in self._emitted:
                    logger.debug(
                        "Skip duplicate signal condition=%s epoch=%s",
                        cond.id,
                        latest.epoch,
                    )
                    continue
                supporting = [
                    o for o in eval_obs if o.id in result.supporting_observation_ids
                ]
                sig = Signal.from_eval(result, supporting)
                self._emitted.add(key)
                signals.append(sig)
                logger.info(
                    "SIGNAL %s symbol=%s tf=%s epoch=%s reasons=%s obs=%s",
                    sig.condition_id,
                    sig.symbol,
                    sig.timeframe,
                    sig.candle_epoch,
                    sig.reasons,
                    [o.type for o in supporting],
                )

            # --- Phase 7 setups (lifecycle-driven) ---
            relevant_setups = [
                s
                for s in self.setups
                if s.symbol == symbol and s.timeframe == timeframe
            ]
            tf_sec = self._resolve_timeframe_seconds(timeframe)
            data_ok, data_diagnostics = self._series_data_health(symbol, timeframe)

            for setup in relevant_setups:
                instance = self._instances.get(setup.setup_id)
                if instance is None:
                    instance = new_pending_instance(
                        setup, instance_key=_DEFAULT_INSTANCE_KEY
                    )
                    self._instances[setup.setup_id] = instance

                # Disabled setups: still call lifecycle (diagnostic-only, no advance)
                life_result = self.lifecycle.evaluate(
                    setup,
                    instance,
                    eval_obs,
                    candle_epoch=latest.epoch,
                    evaluation_time=latest.epoch,
                    timeframe_seconds=tf_sec,
                    data_ok=data_ok,
                    data_diagnostics=data_diagnostics,
                )
                # Always retain returned instance (terminal or progress)
                self._instances[setup.setup_id] = life_result.instance

                if life_result.diagnostic_only:
                    logger.debug(
                        "Setup %s diagnostic-only reasons=%s",
                        setup.setup_id,
                        life_result.reasons,
                    )
                    continue

                # Emit only on transition into COMPLETED
                if (
                    life_result.transitioned
                    and life_result.instance.state == LifecycleState.COMPLETED
                    and life_result.previous_state != LifecycleState.COMPLETED
                ):
                    dedup_key = (life_result.instance.instance_id, latest.epoch)
                    if dedup_key in self._emitted:
                        logger.debug(
                            "Skip duplicate setup signal instance=%s epoch=%s",
                            life_result.instance.instance_id,
                            latest.epoch,
                        )
                        continue
                    sig = self._signal_from_lifecycle(
                        setup, life_result, eval_obs, candle_epoch=latest.epoch
                    )
                    self._emitted.add(dedup_key)
                    # Also record under setup_id for callers inspecting legacy-style keys
                    self._emitted.add((setup.setup_id, latest.epoch))
                    signals.append(sig)
                    logger.info(
                        "SETUP_SIGNAL %s instance=%s symbol=%s tf=%s epoch=%s reasons=%s",
                        setup.setup_id,
                        life_result.instance.instance_id,
                        setup.symbol,
                        setup.timeframe,
                        latest.epoch,
                        sig.reasons,
                    )

        self._last_observations = cycle_new
        return signals

    def _signal_from_lifecycle(
        self,
        setup: SetupDef,
        life_result: Any,
        eval_obs: list[Observation],
        *,
        candle_epoch: int,
    ) -> Signal:
        """Build an explainable Signal from a COMPLETED lifecycle transition."""
        expr = life_result.expression_result
        supporting_ids: list[str] = []
        if expr is not None and expr.evidence:
            supporting_ids = [e.observation_id for e in expr.evidence]
        # Prefer committed THEN evidence when present on progress
        progress = life_result.instance.progress
        if progress is not None and getattr(progress, "committed_steps", None):
            for step in progress.committed_steps:
                for e in getattr(step, "evidence", ()) or ():
                    if e.observation_id not in supporting_ids:
                        supporting_ids.append(e.observation_id)

        supporting = [o for o in eval_obs if o.id in supporting_ids]
        reasons = list(life_result.reasons) or list(
            life_result.instance.last_transition_reasons
        )
        if not reasons:
            reasons = ["setup instance completed"]

        metadata: dict[str, Any] = {
            "kind": "setup",
            "setup_id": setup.setup_id,
            "instance_id": life_result.instance.instance_id,
            "config_id": life_result.instance.config_id,
            "lifecycle_state": life_result.instance.state.value,
            "previous_state": life_result.previous_state.value,
            "anchor_candle_epoch": life_result.instance.anchor_candle_epoch,
        }
        if setup.name:
            metadata["name"] = setup.name

        return Signal(
            id=_deterministic_setup_signal_id(
                life_result.instance.instance_id, candle_epoch
            ),
            condition_id=setup.setup_id,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
            candle_epoch=candle_epoch,
            generated_at=time.time(),
            supporting_observations=supporting,
            reasons=reasons,
            metadata=metadata,
        )

    def _resolve_timeframe_seconds(self, timeframe: str) -> Optional[int]:
        if timeframe in self._tf_seconds:
            return self._tf_seconds[timeframe]
        market = self.market
        # Prefer MarketDataService granularity map when available
        gran_map = getattr(market, "_gran_by_name", None)
        if isinstance(gran_map, dict) and timeframe in gran_map:
            try:
                return int(gran_map[timeframe])
            except (TypeError, ValueError):
                pass
        # Optional helper on market for tests
        getter = getattr(market, "timeframe_seconds", None)
        if callable(getter):
            try:
                v = getter(timeframe)
                if v is not None:
                    return int(v)
            except (TypeError, ValueError):
                pass
        return None

    def _series_data_health(
        self, symbol: str, timeframe: str
    ) -> tuple[bool, tuple[str, ...]]:
        """Return (data_ok, diagnostics) for lifecycle evaluation.

        Fail-closed: when series health cannot be verified (missing interface,
        lookup failure, or check exception), return data_ok=False so lifecycle
        state does not advance. Explicit unhealthy / quarantine also blocks.
        """
        market = self.market
        saw_health_interface = False

        get_cache = getattr(market, "get_cache", None)
        if callable(get_cache):
            saw_health_interface = True
            try:
                cache = get_cache(symbol, timeframe)
            except Exception as exc:
                return False, (
                    f"get_cache failed: {type(exc).__name__}: {exc}",
                )
            if cache is None:
                return False, (
                    "get_cache returned None; cannot verify series health",
                )

            is_healthy = getattr(cache, "is_series_healthy", None)
            if callable(is_healthy):
                min_c = getattr(market, "min_candles", 1) or 1
                try:
                    try:
                        ok = bool(is_healthy(min_c))
                    except TypeError:
                        ok = bool(is_healthy())
                except Exception as exc:
                    return False, (
                        f"is_series_healthy failed: {type(exc).__name__}: {exc}",
                    )
                if not ok:
                    detail = "series unhealthy"
                    q = getattr(cache, "quarantine_epochs", None)
                    if callable(q):
                        try:
                            epochs = q()
                            if epochs:
                                detail = (
                                    f"series unhealthy; quarantine="
                                    f"{sorted(epochs)[:5]}"
                                )
                        except Exception:
                            pass
                    return False, (detail,)

            q = getattr(cache, "quarantine_epochs", None)
            if callable(q):
                try:
                    epochs = q()
                except Exception as exc:
                    return False, (
                        f"quarantine_epochs failed: {type(exc).__name__}: {exc}",
                    )
                if epochs:
                    return False, (
                        f"quarantined epochs present: {sorted(epochs)[:5]}",
                    )

            # Cache path with a successful is_series_healthy check is sufficient.
            if callable(is_healthy):
                return True, ()
            # Cache present but no health method — try series_ok below; if that
            # is also absent, fall through to the unverifiable failure.

        series_ok = getattr(market, "series_ok", None)
        if callable(series_ok):
            saw_health_interface = True
            try:
                ok = bool(series_ok(symbol, timeframe))
            except Exception as exc:
                return False, (
                    f"series_ok failed: {type(exc).__name__}: {exc}",
                )
            if not ok:
                return False, ("series_ok reported False",)
            return True, ()

        if not saw_health_interface:
            return False, (
                "no series health interface available "
                "(get_cache/is_series_healthy or series_ok); "
                "cannot verify data health",
            )
        return False, (
            "cache present but no is_series_healthy or series_ok; "
            "cannot verify data health",
        )

    @property
    def last_observations(self) -> list[Observation]:
        """Observations newly emitted by detectors in the last evaluate_once."""
        return list(self._last_observations)

    @property
    def retained_observations(self) -> list[Observation]:
        """All observations currently held in the store."""
        return self.store.all()
