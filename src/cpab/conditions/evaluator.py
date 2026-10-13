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

    Phase 0 causal + signal-binding rules:
      - Evaluation market time T is the provided candle_epoch (or evaluation_time).
      - Causal availability: a confirmed observation is available only if
        effective_confirmed_at <= T. Developing observations are excluded unless
        the requirement sets allow_developing: true.
      - Trigger binding (preserves original signal semantics): a confirmed
        observation satisfies a condition at T only when it becomes available
        on that bar — i.e. effective_confirmed_at() == T. This prevents an
        old observation from re-satisfying the same condition on every later
        candle (which would have produced signals the original equality filter
        never allowed). Developing (opt-in) observations bind via candle_epoch
        or event_time == T.
      - Legacy observations (no confirmed_at, developing=False) are treated as
        confirmed at candle_epoch, so binding reduces to the original
        candle_epoch == T behaviour.
    """

    def evaluate(
        self,
        condition: ConditionDef,
        observations: Iterable[Observation],
        *,
        candle_epoch: Optional[int] = None,
        evaluation_time: Optional[int] = None,
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

        # T: market evaluation time. Prefer explicit evaluation_time; else candle_epoch.
        T = evaluation_time if evaluation_time is not None else candle_epoch

        obs_list = [
            o
            for o in observations
            if o.symbol == condition.symbol and o.timeframe == condition.timeframe
        ]

        supporting_ids: list[str] = []
        reasons: list[str] = []
        met = True

        for req in condition.require_observations:
            req_type = req.get("type")
            min_count = int(req.get("min_count", 1))
            allow_developing = bool(req.get("allow_developing", False))

            candidates = [o for o in obs_list if o.type == req_type]
            if T is not None:
                matches = [
                    o
                    for o in candidates
                    if o.is_eligible_at(T, allow_developing=allow_developing)
                    and self._binds_to_evaluation_bar(o, T)
                ]
            else:
                # No market time: exclude developing unless opted in (non-causal callers)
                matches = [
                    o
                    for o in candidates
                    if (not o.is_developing()) or allow_developing
                ]

            if len(matches) < min_count:
                met = False
                reasons.append(
                    f"missing observation type={req_type} need>={min_count} "
                    f"got={len(matches)} (eligible at T={T})"
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

    @staticmethod
    def _binds_to_evaluation_bar(obs: Observation, T: int) -> bool:
        """
        Whether this observation is associated with evaluation bar T for
        condition triggering (distinct from causal availability).

        Confirmed: triggers on the confirmation bar only
        (effective_confirmed_at == T), matching original candle_epoch equality
        for legacy/engulfing and preventing re-fire on later bars.

        Developing (opt-in): binds when candle_epoch or event_time equals T.
        """
        if obs.is_developing():
            if obs.candle_epoch == T:
                return True
            if obs.event_time is not None and obs.event_time == T:
                return True
            return False
        conf = obs.effective_confirmed_at()
        return conf is not None and conf == T
