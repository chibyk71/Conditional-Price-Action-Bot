"""Condition engine: detectors → observations → conditions → signals."""

from __future__ import annotations

from typing import Optional

from cpab.conditions.evaluator import ConditionEvaluator
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


class ConditionEngine:
    """
    Runs detectors on completed candles only when readiness is READY.
    Retains observations across cycles (ObservationStore) so multi-component
    conditions can combine facts confirmed on different bars.
    Evaluates conditions and emits explainable signals.
    Deduplicates signals for the same (condition_id, candle_epoch).

    Activation vs retention:
      - Store retains prior confirmed observations for causal availability.
      - Evaluator activation still requires at least one supporting observation
        newly associated with evaluation bar T (confirmed_at == T, etc.).
      - Retained priors alone never re-fire a condition on a later bar.
    """

    def __init__(
        self,
        market: MarketDataService,
        detectors: list[PriceActionDetector],
        conditions: list[ConditionDef],
        readiness: ReadinessManager,
        *,
        observation_store: Optional[ObservationStore] = None,
    ) -> None:
        self.market = market
        self.detectors = detectors
        self.conditions = conditions
        self.readiness = readiness
        self.evaluator = ConditionEvaluator()
        self._emitted: set[tuple[str, int]] = set()  # (condition_id, candle_epoch)
        self._last_observations: list[Observation] = []
        self.store = observation_store or ObservationStore(
            max_size=_DEFAULT_MAX_SIZE,
            max_age_seconds=_DEFAULT_MAX_AGE_SECONDS,
        )

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

        # Group conditions by (symbol, timeframe)
        pairs = {(c.symbol, c.timeframe) for c in self.conditions if c.enabled}
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
            # (priors + just-accepted). Activation is still evaluator-owned.
            eval_obs = self.store.query(symbol, timeframe)

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

        self._last_observations = cycle_new
        return signals

    @property
    def last_observations(self) -> list[Observation]:
        """Observations newly emitted by detectors in the last evaluate_once."""
        return list(self._last_observations)

    @property
    def retained_observations(self) -> list[Observation]:
        """All observations currently held in the store."""
        return self.store.all()
