"""Detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cpab.models import Candle, Observation


class PriceActionDetector(ABC):
    """
    Receives normalized completed candle history and reports structured observations.
    Must not decide whether a complete trading setup exists.
    """

    @abstractmethod
    def detect(self, candles: list[Candle]) -> list[Observation]:
        """Analyze completed candles only; return zero or more observations."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
