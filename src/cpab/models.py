"""Core domain models: candles, observations, conditions, signals, readiness."""

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
    """Structured market fact from a price-action detector. Not a trade decision."""

    id: str
    type: str
    symbol: str
    timeframe: str
    candle_epoch: int
    detected_at: float  # wall-clock unix
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        type: str,
        symbol: str,
        timeframe: str,
        candle_epoch: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Observation":
        return Observation(
            id=str(uuid.uuid4()),
            type=type,
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=candle_epoch,
            detected_at=time.time(),
            metadata=metadata or {},
        )


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
                    "metadata": o.metadata,
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
