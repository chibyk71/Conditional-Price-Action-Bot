"""Condition engine: detectors → observations → conditions → signals."""

from __future__ import annotations

from typing import Optional

from cpab.conditions.evaluator import ConditionEvaluator
from cpab.detectors.base import PriceActionDetector
from cpab.market.readiness import ReadinessManager
from cpab.market.service import MarketDataService
from cpab.models import ConditionDef, Observation, Signal
from cpab.logging_setup import get_logger

logger = get_logger("cpab.core.engine")


class ConditionEngine:
    """
    Runs detectors on completed candles only when readiness is READY.
    Evaluates conditions and emits explainable signals.
    Deduplicates signals for the same (condition_id, candle_epoch).
    """

    def __init__(
        self,
        market: MarketDataService,
        detectors: list[PriceActionDetector],
        conditions: list[ConditionDef],
        readiness: ReadinessManager,
    ) -> None:
        self.market = market
        self.detectors = detectors
        self.conditions = conditions
        self.readiness = readiness
        self.evaluator = ConditionEvaluator()
        self._emitted: set[tuple[str, int]] = set()  # (condition_id, candle_epoch)
        self._last_observations: list[Observation] = []

    def evaluate_once(self) -> list[Signal]:
        if not self.readiness.can_evaluate():
            logger.debug(
                "Evaluation blocked readiness=%s reason=%s",
                self.readiness.state.value,
                self.readiness.reason,
            )
            return []

        signals: list[Signal] = []
        all_obs: list[Observation] = []

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

            # Detectors only see completed candles
            for det in self.detectors:
                obs = det.detect(candles)
                all_obs.extend(obs)

            latest = candles[-1]
            relevant = [
                c
                for c in self.conditions
                if c.enabled and c.symbol == symbol and c.timeframe == timeframe
            ]
            for cond in relevant:
                result = self.evaluator.evaluate(
                    cond, all_obs, candle_epoch=latest.epoch
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
                    o
                    for o in all_obs
                    if o.id in result.supporting_observation_ids
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

        self._last_observations = all_obs
        return signals

    @property
    def last_observations(self) -> list[Observation]:
        return list(self._last_observations)
