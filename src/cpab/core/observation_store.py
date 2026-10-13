"""Bounded observation retention across evaluation cycles."""

from __future__ import annotations

from typing import Iterable, Optional

from cpab.models import Observation
from cpab.logging_setup import get_logger

logger = get_logger("cpab.core.observation_store")


class ObservationStore:
    """
    In-memory store of observations retained across ConditionEngine.evaluate_once
    cycles so multi-component setups can combine facts confirmed on different bars.

    Phase 0 policy:
      - Indexed by observation id; first write wins (records are not mutated).
      - max_size hard cap; when exceeded, drop oldest by market time.
      - Optional max_age_seconds: drop observations whose market anchor is older
        than (as_of_epoch - max_age_seconds).
      - Query is filtered by symbol/timeframe only.
      - Retention does **not** make an observation "new". Activation remains the
        evaluator's responsibility (effective_confirmed_at == T, etc.).
    """

    def __init__(
        self,
        *,
        max_size: int = 1000,
        max_age_seconds: Optional[int] = None,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self._by_id: dict[str, Observation] = {}

    def __len__(self) -> int:
        return len(self._by_id)

    def accept(self, observations: Iterable[Observation]) -> int:
        """
        Insert new observations. Existing ids are left unchanged.
        Returns count of newly inserted records.
        """
        inserted = 0
        for obs in observations:
            if obs.id in self._by_id:
                continue
            self._by_id[obs.id] = obs
            inserted += 1
        if inserted:
            logger.debug("ObservationStore accepted %s new observation(s)", inserted)
        self._enforce_max_size()
        return inserted

    def query(self, symbol: str, timeframe: str) -> list[Observation]:
        """
        Return all retained observations for the series.

        Causal eligibility and activation are applied by ConditionEvaluator;
        the store does not filter on confirmation time here.
        """
        return [
            obs
            for obs in self._by_id.values()
            if obs.symbol == symbol and obs.timeframe == timeframe
        ]

    def prune(self, as_of_epoch: int) -> int:
        """
        Drop observations older than max_age_seconds relative to as_of_epoch.
        If max_age_seconds is None, no age-based pruning occurs.
        Returns number removed.
        """
        if self.max_age_seconds is None:
            return 0
        cutoff = as_of_epoch - self.max_age_seconds
        remove_ids = [
            oid
            for oid, obs in self._by_id.items()
            if self._market_anchor(obs) < cutoff
        ]
        for oid in remove_ids:
            del self._by_id[oid]
        if remove_ids:
            logger.debug(
                "ObservationStore pruned %s observation(s) older than %s",
                len(remove_ids),
                cutoff,
            )
        return len(remove_ids)

    def clear(self) -> None:
        self._by_id.clear()

    def all(self) -> list[Observation]:
        return list(self._by_id.values())

    @staticmethod
    def _market_anchor(obs: Observation) -> int:
        """Sort/prune key: confirmation time, else event, else candle_epoch."""
        conf = obs.effective_confirmed_at()
        if conf is not None:
            return conf
        if obs.event_time is not None:
            return obs.event_time
        return obs.candle_epoch or 0

    def _enforce_max_size(self) -> None:
        overflow = len(self._by_id) - self.max_size
        if overflow <= 0:
            return
        ordered = sorted(
            self._by_id.items(),
            key=lambda kv: self._market_anchor(kv[1]),
        )
        for oid, _ in ordered[:overflow]:
            del self._by_id[oid]
        logger.debug(
            "ObservationStore size cap: dropped %s oldest observation(s)", overflow
        )
