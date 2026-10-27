"""Declarative expression model for Phase 7A setup configurations.

This module defines typed representations only. It does not evaluate
observations or execute trading logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass(frozen=True)
class ObservationRequirement:
    """Leaf: require a canonical observation type (and optional constraints)."""

    type: str
    variant: Optional[str] = None
    min_count: int = 1
    allow_developing: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "min_count": self.min_count}
        if self.variant is not None:
            d["variant"] = self.variant
        if self.allow_developing:
            d["allow_developing"] = True
        return d


@dataclass(frozen=True)
class AndExpression:
    """All child expressions are required. No implicit temporal ordering."""

    children: tuple["Expression", ...]

    def to_dict(self) -> dict[str, Any]:
        return {"op": "and", "children": [c.to_dict() for c in self.children]}


@dataclass(frozen=True)
class OrExpression:
    """At least one child expression is required."""

    children: tuple["Expression", ...]

    def to_dict(self) -> dict[str, Any]:
        return {"op": "or", "children": [c.to_dict() for c in self.children]}


@dataclass(frozen=True)
class ThenStep:
    """One ordered step in a THEN sequence.

    allow_same_candle=False (default) requires a strictly later evaluation
    candle than the previous step's qualifying observation. Setting True
    permits progression on the same candle (explicit opt-in only).
    """

    expression: "Expression"
    allow_same_candle: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = self.expression.to_dict()
        if self.allow_same_candle:
            d = {**d, "allow_same_candle": True}
        return d


@dataclass(frozen=True)
class ThenExpression:
    """Single ordered sequence. No branching or competing histories in 7A.

    Confirmation times must remain non-decreasing. Consecutive steps require
    strictly later evaluation candles by default; same-candle only via
    ThenStep.allow_same_candle.
    """

    steps: tuple[ThenStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"op": "then", "steps": [s.to_dict() for s in self.steps]}


Expression = Union[ObservationRequirement, AndExpression, OrExpression, ThenExpression]


def expression_to_canonical_dict(expr: Expression) -> dict[str, Any]:
    """Stable dict form for hashing (sorted keys applied by caller)."""
    return expr.to_dict()
