"""Condition evaluator: consumes observations, applies configurable rules."""

from __future__ import annotations

from typing import Iterable, Optional

from cpab.models import ConditionDef, ConditionEvalResult, Observation
from cpab.logging_setup import get_logger

logger = get_logger("cpab.conditions")


class ConditionEvaluator:
    """
    Evaluates ConditionDef against a list of observations for the relevant
    symbol/timeframe. Produces ConditionEvalResult; does not emit signals.
    """

    def evaluate(
        self,
        condition: ConditionDef,
        observations: Iterable[Observation],
        *,
        candle_epoch: Optional[int] = None,
    ) -> ConditionEvalResult:
        if not condition.enabled:
            return ConditionEvalResult(
                condition_id=condition.id,
                met=False,
                reasons=["condition disabled"],
                symbol=condition.symbol,
                timeframe=condition.timeframe,
                candle_epoch=candle_epoch,
            )

        obs_list = [
            o
            for o in observations
            if o.symbol == condition.symbol and o.timeframe == condition.timeframe
        ]
        if candle_epoch is not None:
            obs_list = [o for o in obs_list if o.candle_epoch == candle_epoch]

        supporting_ids: list[str] = []
        reasons: list[str] = []
        met = True

        for req in condition.require_observations:
            req_type = req.get("type")
            min_count = int(req.get("min_count", 1))
            matches = [o for o in obs_list if o.type == req_type]
            if len(matches) < min_count:
                met = False
                reasons.append(
                    f"missing observation type={req_type} need>={min_count} got={len(matches)}"
                )
            else:
                supporting_ids.extend(o.id for o in matches[:min_count])
                reasons.append(
                    f"observation type={req_type} satisfied count={len(matches)}"
                )

        if not condition.require_observations:
            met = False
            reasons.append("no require_observations configured")

        result = ConditionEvalResult(
            condition_id=condition.id,
            met=met,
            reasons=reasons,
            supporting_observation_ids=supporting_ids,
            symbol=condition.symbol,
            timeframe=condition.timeframe,
            candle_epoch=candle_epoch,
        )
        logger.debug(
            "Condition %s met=%s reasons=%s",
            condition.id,
            met,
            reasons,
        )
        return result
