"""Phase 7C: deterministic setup-instance lifecycle (invalidation & expiry).

Pure evaluation API — no persistence, scheduling, or engine integration.
Reuses Phase 7A SetupDef and Phase 7B ExpressionEvaluator / ThenProgress.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

from cpab.conditions.expression import Expression, ThenExpression
from cpab.conditions.expression_evaluator import (
    ExpressionEvalResult,
    ExpressionEvaluator,
    ThenProgress,
)
from cpab.conditions.identity import setup_config_identity
from cpab.conditions.schema import ExpiryConfig, SetupDef
from cpab.models import Observation


class LifecycleState(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            LifecycleState.COMPLETED,
            LifecycleState.INVALIDATED,
            LifecycleState.EXPIRED,
        )


def setup_instance_identity(config_id: str, instance_key: str) -> str:
    """Deterministic instance identity (no randomness or wall-clock).

    Distinct from setup configuration identity. Caller supplies a stable
    instance_key (e.g. ordinal or external correlation id).
    """
    if not config_id or not instance_key:
        raise ValueError("config_id and instance_key must be non-empty")
    payload = f"{config_id}:{instance_key}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"setupinst:{digest[:24]}"


@dataclass(frozen=True)
class SetupInstance:
    """Caller-owned lifecycle state for one setup instance."""

    instance_id: str
    config_id: str
    state: LifecycleState = LifecycleState.PENDING
    progress: Optional[ThenProgress] = None
    # First accepted evaluation candle (expiry anchor); set on first non-diagnostic eval.
    anchor_candle_epoch: Optional[int] = None
    last_evaluation_candle: Optional[int] = None
    # Reasons for the last accepted transition (empty if none).
    last_transition_reasons: tuple[str, ...] = ()


def new_pending_instance(setup: SetupDef, *, instance_key: str) -> SetupInstance:
    """Create a PENDING instance for a validated setup."""
    config_id = setup_config_identity(setup)
    return SetupInstance(
        instance_id=setup_instance_identity(config_id, instance_key),
        config_id=config_id,
        state=LifecycleState.PENDING,
        progress=ThenProgress() if isinstance(setup.expression, ThenExpression) else None,
    )


@dataclass(frozen=True)
class LifecycleEvalResult:
    """Result of one lifecycle evaluation step."""

    instance: SetupInstance
    transitioned: bool
    previous_state: LifecycleState
    reasons: tuple[str, ...] = ()
    expression_result: Optional[ExpressionEvalResult] = None
    invalidation_result: Optional[ExpressionEvalResult] = None
    # True when inputs were rejected (health / order) without state change.
    diagnostic_only: bool = False


class SetupLifecycleEvaluator:
    """Deterministic causal lifecycle evaluator for a single setup instance."""

    def __init__(self, expression_evaluator: Optional[ExpressionEvaluator] = None) -> None:
        self._expr = expression_evaluator or ExpressionEvaluator()

    def evaluate(
        self,
        setup: SetupDef,
        instance: SetupInstance,
        observations: Iterable[Observation],
        *,
        candle_epoch: int,
        evaluation_time: Optional[int] = None,
        timeframe_seconds: Optional[int] = None,
        data_ok: bool = True,
        data_diagnostics: Sequence[str] = (),
    ) -> LifecycleEvalResult:
        """Evaluate one lifecycle step.

        Precedence for a valid evaluation event:
          1. Terminal state → preserve
          2. Invalidation (and note concurrent expiry reasons)
          3. Expiry
          4. Expression activation / completion

        Missing/unhealthy/out-of-order/discontinuous data yields diagnostic_only
        without changing accepted lifecycle state.
        """
        previous = instance.state
        T = evaluation_time if evaluation_time is not None else candle_epoch
        reasons: list[str] = []

        # 1. Terminal immutability
        if instance.state.is_terminal:
            return LifecycleEvalResult(
                instance=instance,
                transitioned=False,
                previous_state=previous,
                reasons=("terminal state is immutable",),
            )

        # Config identity must match the instance's recorded config
        expected_config_id = setup_config_identity(setup)
        if instance.config_id != expected_config_id:
            return LifecycleEvalResult(
                instance=instance,
                transitioned=False,
                previous_state=previous,
                reasons=(
                    f"setup config identity mismatch: instance.config_id="
                    f"{instance.config_id!r} != setup {expected_config_id!r}",
                ),
                diagnostic_only=True,
            )

        # Disabled setups must not activate or complete
        if not setup.enabled:
            return LifecycleEvalResult(
                instance=instance,
                transitioned=False,
                previous_state=previous,
                reasons=("setup is disabled; lifecycle state unchanged",),
                diagnostic_only=True,
            )

        # Data health: do not invalidate/expire or advance on bad data
        if not data_ok:
            diag = tuple(data_diagnostics) or (
                "evaluation data not ok; lifecycle state unchanged",
            )
            return LifecycleEvalResult(
                instance=instance,
                transitioned=False,
                previous_state=previous,
                reasons=diag,
                diagnostic_only=True,
            )

        # Reject older evaluation events
        if (
            instance.last_evaluation_candle is not None
            and candle_epoch < instance.last_evaluation_candle
        ):
            return LifecycleEvalResult(
                instance=instance,
                transitioned=False,
                previous_state=previous,
                reasons=(
                    f"stale evaluation candle {candle_epoch} < "
                    f"last accepted {instance.last_evaluation_candle}",
                ),
                diagnostic_only=True,
            )

        # Idempotent reprocess of the same candle: no second mutation
        if (
            instance.last_evaluation_candle is not None
            and candle_epoch == instance.last_evaluation_candle
        ):
            return LifecycleEvalResult(
                instance=instance,
                transitioned=False,
                previous_state=previous,
                reasons=("idempotent: evaluation candle already accepted",),
            )

        # Continuity: next accepted candle must be exactly one step after last
        if (
            instance.last_evaluation_candle is not None
            and timeframe_seconds is not None
            and timeframe_seconds > 0
        ):
            expected = instance.last_evaluation_candle + timeframe_seconds
            if candle_epoch != expected:
                return LifecycleEvalResult(
                    instance=instance,
                    transitioned=False,
                    previous_state=previous,
                    reasons=(
                        f"discontinuous evaluation candle {candle_epoch}; "
                        f"expected {expected} after last accepted "
                        f"{instance.last_evaluation_candle} "
                        f"(timeframe_seconds={timeframe_seconds})",
                    ),
                    diagnostic_only=True,
                )

        # Expiry configured requires evaluable timeframe context
        if setup.expiry is not None and (
            timeframe_seconds is None or timeframe_seconds <= 0
        ):
            return LifecycleEvalResult(
                instance=instance,
                transitioned=False,
                previous_state=previous,
                reasons=(
                    "expiry configured but timeframe_seconds missing or "
                    "non-positive; cannot evaluate lifecycle safely",
                ),
                diagnostic_only=True,
            )

        obs_list = list(observations)

        # Establish / keep anchor on first accepted evaluation
        anchor = (
            instance.anchor_candle_epoch
            if instance.anchor_candle_epoch is not None
            else candle_epoch
        )

        # 2–3. Invalidation and expiry (evaluate both; invalidation wins)
        inv_result: Optional[ExpressionEvalResult] = None
        invalidated = False
        if setup.invalidation is not None:
            inv_result = self._expr.evaluate(
                setup.invalidation,
                obs_list,
                symbol=setup.symbol,
                timeframe=setup.timeframe,
                candle_epoch=candle_epoch,
                evaluation_time=T,
            )
            if inv_result.qualified:
                invalidated = True
                reasons.append("invalidation condition qualified")
                if inv_result.reasons:
                    reasons.extend(inv_result.reasons)

        expired = False
        exp_reasons: list[str] = []
        if setup.expiry is not None:
            expired, exp_reasons = self._check_expiry(
                setup.expiry,
                anchor_candle_epoch=anchor,
                candle_epoch=candle_epoch,
                timeframe_seconds=timeframe_seconds,
            )
            if expired:
                reasons.extend(exp_reasons)

        if invalidated:
            # INVALIDATED takes precedence; keep concurrent expiry reasons
            new_inst = replace(
                instance,
                state=LifecycleState.INVALIDATED,
                anchor_candle_epoch=anchor,
                last_evaluation_candle=candle_epoch,
                last_transition_reasons=tuple(reasons),
                progress=instance.progress,
            )
            return LifecycleEvalResult(
                instance=new_inst,
                transitioned=True,
                previous_state=previous,
                reasons=tuple(reasons),
                invalidation_result=inv_result,
            )

        if expired:
            new_inst = replace(
                instance,
                state=LifecycleState.EXPIRED,
                anchor_candle_epoch=anchor,
                last_evaluation_candle=candle_epoch,
                last_transition_reasons=tuple(reasons),
                progress=instance.progress,
            )
            return LifecycleEvalResult(
                instance=new_inst,
                transitioned=True,
                previous_state=previous,
                reasons=tuple(reasons),
                invalidation_result=inv_result,
            )

        # 4. Expression activation / completion
        progress = instance.progress
        if isinstance(setup.expression, ThenExpression):
            if progress is None:
                progress = ThenProgress()
            # Completed progress must not restart
            if progress.completed:
                new_inst = replace(
                    instance,
                    state=LifecycleState.COMPLETED,
                    anchor_candle_epoch=anchor,
                    last_evaluation_candle=candle_epoch,
                    progress=progress,
                    last_transition_reasons=("THEN progress already completed",),
                )
                # Only transition if we were not already COMPLETED (we are not terminal above)
                transitioned = previous != LifecycleState.COMPLETED
                return LifecycleEvalResult(
                    instance=new_inst,
                    transitioned=transitioned,
                    previous_state=previous,
                    reasons=new_inst.last_transition_reasons,
                    invalidation_result=inv_result,
                )

            expr_result = self._expr.evaluate(
                setup.expression,
                obs_list,
                symbol=setup.symbol,
                timeframe=setup.timeframe,
                candle_epoch=candle_epoch,
                evaluation_time=T,
                progress=progress,
            )
            new_progress = expr_result.progress if expr_result.progress is not None else progress

            if expr_result.advanced:
                if new_progress.completed or expr_result.qualified:
                    reasons.append("THEN sequence completed")
                    new_state = LifecycleState.COMPLETED
                else:
                    reasons.append(
                        f"THEN advanced to step[{new_progress.next_step_index}]"
                    )
                    new_state = LifecycleState.IN_PROGRESS
                if expr_result.reasons:
                    reasons.extend(expr_result.reasons)
                new_inst = replace(
                    instance,
                    state=new_state,
                    progress=new_progress,
                    anchor_candle_epoch=anchor,
                    last_evaluation_candle=candle_epoch,
                    last_transition_reasons=tuple(reasons),
                )
                return LifecycleEvalResult(
                    instance=new_inst,
                    transitioned=True,
                    previous_state=previous,
                    reasons=tuple(reasons),
                    expression_result=expr_result,
                    invalidation_result=inv_result,
                )

            # No advancement
            new_inst = replace(
                instance,
                anchor_candle_epoch=anchor,
                last_evaluation_candle=candle_epoch,
                progress=progress,
            )
            return LifecycleEvalResult(
                instance=new_inst,
                transitioned=False,
                previous_state=previous,
                reasons=("expression did not qualify or advance",) + tuple(expr_result.reasons),
                expression_result=expr_result,
                invalidation_result=inv_result,
            )

        # Non-THEN expression
        expr_result = self._expr.evaluate(
            setup.expression,
            obs_list,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
            candle_epoch=candle_epoch,
            evaluation_time=T,
        )
        if expr_result.qualified:
            reasons.append("expression qualified; instance completed")
            if expr_result.reasons:
                reasons.extend(expr_result.reasons)
            new_inst = replace(
                instance,
                state=LifecycleState.COMPLETED,
                anchor_candle_epoch=anchor,
                last_evaluation_candle=candle_epoch,
                last_transition_reasons=tuple(reasons),
            )
            return LifecycleEvalResult(
                instance=new_inst,
                transitioned=True,
                previous_state=previous,
                reasons=tuple(reasons),
                expression_result=expr_result,
                invalidation_result=inv_result,
            )

        new_inst = replace(
            instance,
            anchor_candle_epoch=anchor,
            last_evaluation_candle=candle_epoch,
        )
        return LifecycleEvalResult(
            instance=new_inst,
            transitioned=False,
            previous_state=previous,
            reasons=("expression did not qualify",) + tuple(expr_result.reasons),
            expression_result=expr_result,
            invalidation_result=inv_result,
        )

    def _check_expiry(
        self,
        expiry: ExpiryConfig,
        *,
        anchor_candle_epoch: int,
        candle_epoch: int,
        timeframe_seconds: Optional[int],
    ) -> tuple[bool, list[str]]:
        """Inclusive candle-based expiry from the instance anchor.

        elapsed = (candle_epoch - anchor) // timeframe_seconds
        expires when elapsed >= max_age_candles (inclusive boundary).

        Caller must supply positive timeframe_seconds; evaluate() rejects the
        event diagnostically when expiry is configured without it. Continuity
        from the last accepted candle is enforced in evaluate() before this runs.
        """
        if timeframe_seconds is None or timeframe_seconds <= 0:
            # Should not reach here when evaluate() gates correctly
            return False, [
                "expiry configured but timeframe_seconds missing or non-positive"
            ]
        if candle_epoch < anchor_candle_epoch:
            return False, ["evaluation candle before anchor; expiry not applied"]
        delta = candle_epoch - anchor_candle_epoch
        if delta % timeframe_seconds != 0:
            return False, [
                f"candle_epoch {candle_epoch} not aligned with anchor "
                f"{anchor_candle_epoch} and timeframe_seconds {timeframe_seconds}"
            ]
        elapsed = delta // timeframe_seconds
        if elapsed >= expiry.max_age_candles:
            return True, [
                f"expiry: elapsed_candles={elapsed} >= max_age_candles="
                f"{expiry.max_age_candles} (inclusive; anchor={anchor_candle_epoch})"
            ]
        return False, []
