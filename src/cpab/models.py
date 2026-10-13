"""Core domain models: candles, observations, conditions, signals, readiness, lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import time
import uuid


class ReadinessState(str, Enum):
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Candle:
    """Normalized internal candle. Epoch is open time in seconds (UTC)."""

    symbol: str
    timeframe: str
    epoch: int  # open time, seconds
    open: float
    high: float
    low: float
    close: float
    is_closed: bool = True
    source: str = "provider"  # provider | aggregated

    def key(self) -> tuple[str, str, int]:
        return (self.symbol, self.timeframe, self.epoch)

    def ohlc_tuple(self) -> tuple[float, float, float, float]:
        return (self.open, self.high, self.low, self.close)


@dataclass
class CandleValidationIssue:
    code: str
    message: str
    index: Optional[int] = None
    epoch: Optional[int] = None


@dataclass
class CandleValidationResult:
    ok: bool
    candles: list[Candle]
    issues: list[CandleValidationIssue] = field(default_factory=list)


@dataclass(frozen=True)
class Observation:
    """
    Structured market fact from a price-action detector. Not a trade decision.

    Timestamp semantics (Phase 0):
      - event_time: market/candle time at which the underlying event occurred.
      - confirmed_at: market/candle time at which sufficient evidence existed
        to confirm the observation; None for explicitly developing observations.
      - detected_at: wall-clock time at which the detector emitted the observation.
      - candle_epoch: legacy field. For engulging it identifies the confirming
        candle (same as event/confirm for that detector). Do not assume
        candle_epoch == event_time for every future concept.

    Legacy confirmation policy:
      - If confirmed_at is set → use it.
      - Else if developing is True → not confirmed (ineligible for normal conditions).
      - Else if candle_epoch is present → treat as confirmed at candle_epoch
        (legacy observations that pre-date confirmed_at).
      - Else fall back to detected_at only as a documented last resort; this is
        not a real market confirmation time.

    metadata vs payload:
      - metadata is retained for backward compatibility (engulfing tests/consumers).
      - payload is the canonical home for new concept-specific properties.
      - When both are supplied, payload is authoritative for new code; metadata
        is kept in sync by Observation.create when only one is provided.
    """

    id: str
    type: str
    symbol: str
    timeframe: str
    candle_epoch: int
    detected_at: float  # wall-clock unix
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 0 additive fields (all optional / safe defaults)
    event_time: Optional[int] = None
    confirmed_at: Optional[int] = None
    developing: bool = False
    direction: Optional[str] = None  # "bullish" | "bearish" | None
    variant: Optional[str] = None
    detector_id: Optional[str] = None
    detector_config_id: Optional[str] = None
    reference_ids: tuple[str, ...] = field(default_factory=tuple)
    lifecycle_state: Optional[str] = None  # snapshot at emission only
    payload: dict[str, Any] = field(default_factory=dict)

    def effective_event_time(self) -> int:
        """Market time of the event; falls back to candle_epoch for legacy."""
        if self.event_time is not None:
            return self.event_time
        return self.candle_epoch

    def effective_confirmed_at(self) -> Optional[int]:
        """
        Market confirmation time used for eligibility, applying the legacy policy.
        Returns None for explicitly developing observations.

        Policy order:
          1. developing=True → None (never treat as confirmed).
          2. confirmed_at set → that market time.
          3. else candle_epoch (legacy detectors always set this).
          4. else int(detected_at) as a last-resort documented fallback when
             candle_epoch is unavailable (e.g. incomplete deserialized record).
             This is wall-clock, NOT a real market confirmation time.
        """
        if self.developing:
            return None
        if self.confirmed_at is not None:
            return self.confirmed_at
        # candle_epoch is required on the dataclass for normal construction;
        # treat 0 as "unavailable" for incomplete records.
        if self.candle_epoch:
            return self.candle_epoch
        if self.detected_at:
            return int(self.detected_at)
        return None

    def confirmation_is_detected_at_fallback(self) -> bool:
        """True when effective_confirmed_at came from detected_at, not market time."""
        if self.developing or self.confirmed_at is not None:
            return False
        if self.candle_epoch:
            return False
        return bool(self.detected_at)

    def is_developing(self) -> bool:
        return bool(self.developing)

    def is_confirmed(self) -> bool:
        return self.effective_confirmed_at() is not None

    def is_eligible_at(self, market_time: int, *, allow_developing: bool = False) -> bool:
        """
        Causal availability at market evaluation time T.
        Confirmed obs eligible iff effective_confirmed_at <= T.
        Developing obs only if allow_developing is True (and still not treated as confirmed).

        Note: condition *triggering* additionally requires binding to the evaluation
        bar (see ConditionEvaluator._binds_to_evaluation_bar). Availability alone
        does not imply a signal on every subsequent candle.
        """
        if self.is_developing():
            return allow_developing
        conf = self.effective_confirmed_at()
        if conf is None:
            return False
        return conf <= market_time

    @staticmethod
    def create(
        type: str,
        symbol: str,
        timeframe: str,
        candle_epoch: int,
        metadata: Optional[dict[str, Any]] = None,
        *,
        event_time: Optional[int] = None,
        confirmed_at: Optional[int] = None,
        developing: bool = False,
        direction: Optional[str] = None,
        variant: Optional[str] = None,
        detector_id: Optional[str] = None,
        detector_config_id: Optional[str] = None,
        reference_ids: Optional[list[str] | tuple[str, ...]] = None,
        lifecycle_state: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> "Observation":
        """
        Factory preserving the original positional/keyword surface.
        New confirmed detectors should pass event_time and confirmed_at explicitly.
        Default: confirmed at candle_epoch (engulfing-compatible), not developing.
        """
        meta = dict(metadata or {})
        pay = dict(payload or {})
        # Keep metadata and payload from diverging on the create path when only one is given
        if payload is None and meta:
            pay = dict(meta)
        elif metadata is None and pay:
            meta = dict(pay)

        et = event_time if event_time is not None else candle_epoch
        # Explicit developing must not get a synthetic confirmed_at
        if developing:
            conf: Optional[int] = None
        elif confirmed_at is not None:
            conf = confirmed_at
        else:
            conf = candle_epoch

        return Observation(
            id=str(uuid.uuid4()),
            type=type,
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=candle_epoch,
            detected_at=time.time(),
            metadata=meta,
            event_time=et,
            confirmed_at=conf,
            developing=developing,
            direction=direction,
            variant=variant,
            detector_id=detector_id,
            detector_config_id=detector_config_id,
            reference_ids=tuple(reference_ids or ()),
            lifecycle_state=lifecycle_state,
            payload=pay,
        )


@dataclass(frozen=True)
class ObservationLifecycleEvent:
    """
    Append-only record of a lifecycle transition or interaction for an observation.
    Observations and prior lifecycle events are immutable; later changes append new events.
    Interaction events (touch, retest, ...) need not change lifecycle state.
    """

    id: str
    observation_id: str
    event_type: str  # e.g. mitigated, invalidated, retested, touch
    event_time: int  # market/candle time of the transition/interaction
    detected_at: float  # wall-clock emission
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    confirmed_at: Optional[int] = None
    payload: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        observation_id: str,
        event_type: str,
        event_time: int,
        *,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
        confirmed_at: Optional[int] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> "ObservationLifecycleEvent":
        return ObservationLifecycleEvent(
            id=str(uuid.uuid4()),
            observation_id=observation_id,
            event_type=event_type,
            event_time=event_time,
            detected_at=time.time(),
            from_state=from_state,
            to_state=to_state,
            confirmed_at=confirmed_at if confirmed_at is not None else event_time,
            payload=dict(payload or {}),
        )


class LifecycleTransitionError(ValueError):
    """Raised when a configured concept-specific transition is not allowed."""


@dataclass
class LifecycleDefinition:
    """
    Minimal concept-specific allowed transitions.
    Keys are from_state (None means initial / no prior state);
    values are sets of permitted to_state values.
    Interaction-only event types listed in interaction_types do not require a state change.
    """

    concept: str
    allowed: dict[Optional[str], set[str]] = field(default_factory=dict)
    interaction_types: set[str] = field(
        default_factory=lambda: {
            "touch",
            "approach",
            "entry",
            "rejection",
            "hold",
            "retest",
        }
    )

    def validate_transition(
        self,
        event_type: str,
        from_state: Optional[str],
        to_state: Optional[str],
    ) -> None:
        # Pure interactions may leave state unchanged
        if event_type in self.interaction_types and (
            to_state is None or to_state == from_state
        ):
            return
        if to_state is None:
            # No state change claimed; allowed only as interaction (handled above)
            if event_type not in self.interaction_types:
                raise LifecycleTransitionError(
                    f"{self.concept}: event_type={event_type} requires to_state"
                )
            return
        permitted = self.allowed.get(from_state, set())
        if to_state not in permitted:
            raise LifecycleTransitionError(
                f"{self.concept}: transition {from_state!r} -> {to_state!r} "
                f"not allowed for event_type={event_type}"
            )


def project_lifecycle_state(
    events: list[ObservationLifecycleEvent],
) -> Optional[str]:
    """
    Minimal read-model: latest non-null to_state from ordered events.
    No persistent projection store in Phase 0.
    """
    state: Optional[str] = None
    for ev in events:
        if ev.to_state is not None:
            state = ev.to_state
    return state


@dataclass
class ConditionDef:
    """Configurable condition definition."""

    id: str
    symbol: str
    timeframe: str
    enabled: bool = True
    require_observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ConditionEvalResult:
    condition_id: str
    met: bool
    reasons: list[str] = field(default_factory=list)
    supporting_observation_ids: list[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)
    symbol: str = ""
    timeframe: str = ""
    candle_epoch: Optional[int] = None


@dataclass
class Signal:
    """Explainable signal produced when a condition is satisfied. No trade execution."""

    id: str
    condition_id: str
    symbol: str
    timeframe: str
    candle_epoch: int
    generated_at: float
    supporting_observations: list[Observation]
    reasons: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "condition_id": self.condition_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "candle_epoch": self.candle_epoch,
            "generated_at": self.generated_at,
            "reasons": self.reasons,
            "supporting_observations": [
                {
                    "id": o.id,
                    "type": o.type,
                    "candle_epoch": o.candle_epoch,
                    "event_time": o.event_time,
                    "confirmed_at": o.confirmed_at,
                    "developing": o.developing,
                    "metadata": o.metadata,
                    "payload": o.payload,
                }
                for o in self.supporting_observations
            ],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_eval(
        result: ConditionEvalResult,
        observations: list[Observation],
    ) -> "Signal":
        return Signal(
            id=str(uuid.uuid4()),
            condition_id=result.condition_id,
            symbol=result.symbol,
            timeframe=result.timeframe,
            candle_epoch=result.candle_epoch or 0,
            generated_at=time.time(),
            supporting_observations=observations,
            reasons=list(result.reasons),
        )


def candle_to_dict(c: Candle) -> dict[str, Any]:
    return asdict(c)
