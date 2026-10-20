"""
Phase 4C — Inverse Fair Value Gap (IFVG) detection.

An IFVG forms immediately when an existing FVG is invalidated by a qualifying
completed-candle close (Phase 4B invalidation semantics):

  Bullish FVG → bearish IFVG (variant=bullish_to_bearish):
      close strictly below the original FVG lower bound.

  Bearish FVG → bullish IFVG (variant=bearish_to_bullish):
      close strictly above the original FVG upper bound.

Equality with the boundary does not qualify. No retest is required.
Original FVG bounds are preserved (not recalculated from the inversion candle).

This detector composes FairValueGapLifecycleDetector and maps its
variant="invalidation" observations to IFVG observations. It does not
reimplement formation or invalidation rules.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.fair_value_gap_lifecycle import FairValueGapLifecycleDetector
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.inverse_fair_value_gap")


def _stable_ifvg_id(
    *,
    symbol: str,
    timeframe: str,
    candle_epoch: int,
    event_time: int,
    confirmed_at: int,
    detector_id: str,
    detector_config_id: str,
    direction: str,
    variant: str,
    source_fvg_id: str,
    source_invalidation_id: str,
) -> str:
    """
    Deterministic identity including source FVG and invalidation event so
    repeated reconstruction is idempotent and overlapping FVGs stay distinct.
    """
    parts = [
        "inverse_fair_value_gap",
        symbol,
        timeframe,
        str(candle_epoch),
        f"e{event_time}",
        f"c{confirmed_at}",
        detector_id,
        detector_config_id,
        direction,
        variant,
        source_fvg_id,
        source_invalidation_id,
    ]
    return "|".join(parts)


class InverseFairValueGapDetector(PriceActionDetector):
    """
    IFVG observations derived exclusively from Phase 4B invalidation events.

    Configuration is empty. detector_config_id is deterministic.
    """

    def __init__(
        self,
        *,
        detector_id: str = "inverse_fair_value_gap",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._detector_id = detector_id
        self._lifecycle = FairValueGapLifecycleDetector(
            detector_id="fair_value_gap_lifecycle"
        )
        cfg: dict[str, Any] = {}
        if detector_config_id is not None:
            self._detector_config_id = detector_config_id
        else:
            canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
            self._detector_config_id = f"cfg:{digest}"

    @property
    def name(self) -> str:
        return self._detector_id

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def min_history(self) -> int:
        # Same bound as lifecycle: formation (3) + at least one post-formation bar.
        return 4

    def config_dict(self) -> dict[str, Any]:
        return {}

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: full reconstruction, then emit only IFVGs whose
        confirmed_at equals the latest closed bar's open epoch.
        """
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        all_obs = self._reconstruct(closed)
        latest_epoch = closed[-1].epoch
        return [o for o in all_obs if o.confirmed_at == latest_epoch]

    def detect_all_historical(self, candles: list[Candle]) -> list[Observation]:
        """Full historical reconstruction for tests / offline backfill."""
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        return self._reconstruct(closed)

    def _reconstruct(self, closed: list[Candle]) -> list[Observation]:
        if len(closed) < 4:
            return []

        # Phase 4B lifecycle already enforces continuity, OHLC sanity, and
        # invalidation rules. Consume only invalidation events.
        lifecycle_obs = self._lifecycle.detect_all_historical(closed)
        invalidations = [o for o in lifecycle_obs if o.variant == "invalidation"]

        out: list[Observation] = []
        seen_keys: set[str] = set()

        for inv in invalidations:
            source_fvg_id = inv.reference_ids[0] if inv.reference_ids else ""
            if not source_fvg_id:
                source_fvg_id = (inv.payload or {}).get("source_fvg_id") or ""
            if not source_fvg_id:
                logger.debug("Skipping invalidation %s: missing source FVG id", inv.id)
                continue

            payload = inv.payload or {}
            try:
                lower = float(payload["lower_bound"])
                upper = float(payload["upper_bound"])
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping invalidation %s: missing bounds", inv.id)
                continue

            source_direction = (
                payload.get("source_fvg_direction")
                or inv.direction
                or ""
            )
            if source_direction == "bullish":
                ifvg_variant = "bullish_to_bearish"
                ifvg_direction = "bearish"
            elif source_direction == "bearish":
                ifvg_variant = "bearish_to_bullish"
                ifvg_direction = "bullish"
            else:
                logger.debug(
                    "Skipping invalidation %s: unknown source direction %r",
                    inv.id,
                    source_direction,
                )
                continue

            # Deduplicate: one IFVG per source FVG + invalidation event.
            dedupe_key = f"{source_fvg_id}|{inv.id}"
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            event_time = inv.event_time if inv.event_time is not None else inv.candle_epoch
            confirmed_at = (
                inv.confirmed_at if inv.confirmed_at is not None else inv.candle_epoch
            )
            candle_epoch = inv.candle_epoch

            ifvg_payload: dict[str, Any] = {
                "source_fvg_id": source_fvg_id,
                "source_invalidation_id": inv.id,
                "source_fvg_direction": source_direction,
                "lower_bound": lower,
                "upper_bound": upper,
                "size": upper - lower,
                "invalidation_candle_epoch": candle_epoch,
                "open": payload.get("open"),
                "high": payload.get("high"),
                "low": payload.get("low"),
                "close": payload.get("close"),
            }
            if "source_fvg_candle_epoch" in payload:
                ifvg_payload["source_fvg_candle_epoch"] = payload["source_fvg_candle_epoch"]

            oid = _stable_ifvg_id(
                symbol=inv.symbol,
                timeframe=inv.timeframe,
                candle_epoch=candle_epoch,
                event_time=event_time,
                confirmed_at=confirmed_at,
                detector_id=self._detector_id,
                detector_config_id=self._detector_config_id,
                direction=ifvg_direction,
                variant=ifvg_variant,
                source_fvg_id=source_fvg_id,
                source_invalidation_id=inv.id,
            )

            out.append(
                Observation.create(
                    type="inverse_fair_value_gap",
                    symbol=inv.symbol,
                    timeframe=inv.timeframe,
                    candle_epoch=candle_epoch,
                    event_time=event_time,
                    confirmed_at=confirmed_at,
                    developing=False,
                    direction=ifvg_direction,
                    variant=ifvg_variant,
                    detector_id=self._detector_id,
                    detector_config_id=self._detector_config_id,
                    reference_ids=(source_fvg_id,),
                    payload=ifvg_payload,
                    metadata=ifvg_payload,
                    id=oid,
                )
            )

        return out
