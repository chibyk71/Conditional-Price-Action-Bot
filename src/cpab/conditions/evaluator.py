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

    Phase 0 causal availability + condition activation (corrected contract):

      Evaluation market time T is candle_epoch or evaluation_time.

      1. Causal availability (per observation):
         - Confirmed: eligible iff effective_confirmed_at() <= T.
         - Developing: only if the requirement sets allow_developing: true.
         Prior confirmed observations remain available on later bars so
         multi-component setups can combine facts confirmed at different times.

      2. Requirement matching:
         Each require_observations entry is satisfied by causally available
         observations of the requested type (not by confirmation-bar equality
         alone). When selecting the supporting set (min_count), activation
         observations at T are preferred over older eligible matches so a
         new same-type fact on the current bar can activate the condition.

      3. Condition activation (prevents passive re-fire):
         When T is set, the condition is met only if every requirement is
         satisfied AND at least one supporting observation is newly associated
         with bar T (_is_activation_observation). That is:
           - confirmed with effective_confirmed_at() == T, or
           - developing (opt-in) with candle_epoch or event_time == T.
         This allows e.g. swing (confirmed earlier) + retest (earlier) +
         engulfing (confirmed at T) to complete at T, without re-satisfying
         on every subsequent bar when nothing new confirms.

      4. Legacy observations (no confirmed_at, developing=False) are treated as
         confirmed at candle_epoch, so a single-obs engulfing condition still
         activates only on that bar.

      5. Duplicate signal suppression across bars remains the engine's
         responsibility via (condition_id, candle_epoch) emission tracking.
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

        T = evaluation_time if evaluation_time is not None else candle_epoch

        obs_list = [
            o
            for o in observations
            if o.symbol == condition.symbol and o.timeframe == condition.timeframe
        ]

        supporting_ids: list[str] = []
        supporting_obs: list[Observation] = []
        reasons: list[str] = []
        met = True

        for req in condition.require_observations:
            req_type = req.get("type")
            req_variant = req.get("variant")
            min_count = int(req.get("min_count", 1))
            allow_developing = bool(req.get("allow_developing", False))

            # Strict type matching: type filters only observation.type.
            # Variant filters only when explicitly specified. No type↔variant
            # fallback (avoids e.g. type="hh" matching variant="hh" on another type).
            candidates = [
                o
                for o in obs_list
                if (req_type is None or o.type == req_type)
                and (req_variant is None or o.variant == req_variant)
                and (req_type is not None or req_variant is not None)
            ]
            if T is not None:
                matches = [
                    o
                    for o in candidates
                    if o.is_eligible_at(T, allow_developing=allow_developing)
                ]
                # Prefer activation observations at T so a new same-type fact on
                # the current bar is selected over an older eligible prior when
                # only min_count slots are taken into the supporting set.
                matches = sorted(
                    matches,
                    key=lambda o: (
                        0 if ConditionEvaluator._is_activation_observation(o, T) else 1,
                        o.effective_confirmed_at() is None,
                        o.effective_confirmed_at() or 0,
                        o.id,
                    ),
                )
            else:
                matches = [
                    o
                    for o in candidates
                    if (not o.is_developing()) or allow_developing
                ]

            if len(matches) < min_count:
                met = False
                desc = f"type={req_type}" if req_type else f"variant={req_variant}"
                if req_type and req_variant:
                    desc = f"type={req_type} variant={req_variant}"
                reasons.append(
                    f"missing observation {desc} need>={min_count} "
                    f"got={len(matches)} (eligible at T={T})"
                )
            else:
                chosen = matches[:min_count]
                supporting_ids.extend(o.id for o in chosen)
                supporting_obs.extend(chosen)
                desc = f"type={req_type}" if req_type else f"variant={req_variant}"
                if req_type and req_variant:
                    desc = f"type={req_type} variant={req_variant}"
                reasons.append(
                    f"observation {desc} satisfied count={len(matches)}"
                )

        if not condition.require_observations:
            met = False
            reasons.append("no require_observations configured")

        # Activation: when evaluating at a market time T, require at least one
        # supporting observation newly associated with T so multi-component
        # setups can use prior confirmations without re-firing every later bar.
        if met and T is not None and condition.require_observations:
            if not any(self._is_activation_observation(o, T) for o in supporting_obs):
                met = False
                reasons.append(
                    f"no activation observation at T={T} "
                    "(need at least one newly confirmed or developing bind on this bar)"
                )

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
    def _is_activation_observation(obs: Observation, T: int) -> bool:
        """
        True if this observation newly associates with evaluation bar T and can
        activate a multi-component condition (distinct from mere availability).

        Confirmed: effective_confirmed_at == T.
        Developing (opt-in path only reaches here if allowed): candle_epoch or
        event_time == T.
        """
        if obs.is_developing():
            if obs.candle_epoch == T:
                return True
            if obs.event_time is not None and obs.event_time == T:
                return True
            return False
        conf = obs.effective_confirmed_at()
        return conf is not None and conf == T

    # Kept as a named alias for clarity in tests/docs; same as activation.
    @staticmethod
    def _binds_to_evaluation_bar(obs: Observation, T: int) -> bool:
        return ConditionEvaluator._is_activation_observation(obs, T)
