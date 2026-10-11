"""Small adapter interface for market-data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class MarketDataAdapter(ABC):
    """Provider-specific communication lives behind this interface."""

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    @abstractmethod
    async def fetch_candles(
        self,
        symbol: str,
        granularity: int,
        count: int,
        *,
        end: str | int = "latest",
    ) -> list[dict[str, Any]]:
        """Return raw provider candle dicts (epoch, open, high, low, close)."""
        ...

    @property
    @abstractmethod
    def connected(self) -> bool:
        ...
