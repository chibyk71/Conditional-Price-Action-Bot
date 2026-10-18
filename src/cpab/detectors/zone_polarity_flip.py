"""
Phase 3D — Zone polarity flip detection (potential support↔resistance change).

Emits candidate polarity-flip observations derived exclusively from existing
Phase 3B break observations. Does not reimplement zone construction or break
detection; composes SupportResistanceZoneDetector and ZoneInteractionDetector
for deterministic reconstruction.

A polarity flip is a *potential* change only. It does not assert that the zone
has successfully held as the opposite polarity, nor does it implement retests,
mitigation, invalidation, or a zone lifecycle state machine.

Locked rules (completed candles only):

  Support zone + bearish break  →  variant="support_to_resistance", direction="bearish"
  Resistance zone + bullish break → variant="resistance_to_support", direction="bullish"

  No other interaction variant or direction produces a flip.
  At most one flip per source-zone break.
  Timing (event_time / confirmed_at) is inherited from the source break.
  Price is the original zone level.

Causal / continuity rules are those of Phase 3B (zone eligibility, close-crossing,
gap suppression). Historical reconstruction and incremental latest-bar focus
match established detector conventions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.support_resistance import SupportResistanceZoneDetector
from cpab.detectors.swing import (
    _validate_strength,
    _validate_tie_policy,
)
from cpab.detectors.zone_interaction import ZoneInteractionDetector
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.zone_polarity_flip")


def _stable_flip_id(
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
    source_zone_id: str,
    source_break_id: str,
) -> str:
    """
    Deterministic identity including both source zone and source break so:
    * flips from different zones remain distinct even at the same price;
    * different break observations cannot collide;
    * repeated reconstruction yields identical IDs.
    """
    parts = [
        "zone_polarity_flip",
        symbol,
        timeframe,
        str(candle_epoch),
        f"e{event_time}",
        f"c{confirmed_at}",
        detector_id,
        detector_config_id,
        direction,
        variant,
        source_zone_id,
        source_break_id,
    ]
    return "|".join(parts)


class ZonePolarityFlipDetector(PriceActionDetector):
    """
    Potential zone polarity flips derived from Phase 3B break observations.

    Configuration is immutable after construction. Invalid swing parameters
    raise ValueError (no silent defaults or coercion for invalid values).

    Zone and interaction generation stay in Phases 3A/3B; this detector only
    maps eligible breaks to polarity-flip candidates.
    """

    def __init__(
        self,
        *,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "zone_polarity_flip",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._left = _validate_strength("left_strength", left_strength)
        self._right = _validate_strength("right_strength", right_strength)
        self._tie_policy = _validate_tie_policy(tie_policy)
        self._detector_id = detector_id

        # Compose Phase 3B interaction detector (which itself composes 3A zones).
        self._interactions = ZoneInteractionDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
            detector_id="zone_interaction",
        )

        cfg = {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
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
    def left_strength(self) -> int:
        return self._left

    @property
    def right_strength(self) -> int:
        return self._right

    @property
    def tie_policy(self) -> str:
        return self._tie_policy

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def min_history(self) -> int:
        """
        Same bound as ZoneInteractionDetector: enough for a confirmed zone
        plus a prior/current close pair for break evaluation.
        """
        return self._interactions.min_history

    def config_dict(self) -> dict[str, Any]:
        return {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: reconstruct all flips from closed history, then emit
        only those whose confirmed_at equals the latest closed bar's open epoch.
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
        if not closed:
            return []
        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe; skipping zone-polarity-flip detect")
            return []

        # Reconstruct Phase 3B interactions; only breaks are candidates.
        interactions = self._interactions.detect_all_historical(closed)
        breaks = [
            o
            for o in interactions
            if o.type == "zone_interaction" and o.variant == "break"
        ]

        out: list[Observation] = []
        seen_break_ids: set[str] = set()

        for brk in breaks:
            if brk.id in seen_break_ids:
                continue
            seen_break_ids.add(brk.id)

            zone_variant = (brk.payload or {}).get("zone_variant")
            if zone_variant is None:
                # Fallback: look at reference zone if payload incomplete.
                zone_variant = None

            direction = brk.direction
            level_price = (brk.payload or {}).get("level_price")
            if level_price is None:
                logger.debug("Skipping break %s: missing level_price", brk.id)
                continue

            # Map only the two valid polarity transitions.
            if zone_variant == "support" and direction == "bearish":
                flip_variant = "support_to_resistance"
                flip_direction = "bearish"
            elif zone_variant == "resistance" and direction == "bullish":
                flip_variant = "resistance_to_support"
                flip_direction = "bullish"
            else:
                # Non-break or wrong direction: never emit.
                continue

            source_zone_id = brk.reference_ids[0] if brk.reference_ids else ""
            if not source_zone_id:
                logger.debug("Skipping break %s: missing source zone id", brk.id)
                continue

            event_time = brk.event_time if brk.event_time is not None else brk.candle_epoch
            confirmed_at = brk.confirmed_at if brk.confirmed_at is not None else brk.candle_epoch
            candle_epoch = brk.candle_epoch

            payload: dict[str, Any] = {
                "zone_type": "support_resistance_zone",
                "source_zone_variant": zone_variant,
                "source_zone_id": source_zone_id,
                "source_break_id": brk.id,
                "level_price": float(level_price),
                "flip_candle_epoch": candle_epoch,
                "source_break_event_time": event_time,
                "source_break_confirmed_at": confirmed_at,
                "left_strength": self._left,
                "right_strength": self._right,
                "tie_policy": self._tie_policy,
            }
            # Preserve useful OHLC context from the break when present.
            for key in ("open", "high", "low", "close", "previous_close"):
                if key in (brk.payload or {}):
                    payload[key] = brk.payload[key]

            oid = _stable_flip_id(
                symbol=symbol,
                timeframe=timeframe,
                candle_epoch=candle_epoch,
                event_time=event_time,
                confirmed_at=confirmed_at,
                detector_id=self._detector_id,
                detector_config_id=self._detector_config_id,
                direction=flip_direction,
                variant=flip_variant,
                source_zone_id=source_zone_id,
                source_break_id=brk.id,
            )

            out.append(
                Observation.create(
                    type="zone_polarity_flip",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_epoch=candle_epoch,
                    event_time=event_time,
                    confirmed_at=confirmed_at,
                    developing=False,
                    direction=flip_direction,
                    variant=flip_variant,
                    detector_id=self._detector_id,
                    detector_config_id=self._detector_config_id,
                    reference_ids=(source_zone_id, brk.id),
                    payload=payload,
                    metadata=payload,
                    id=oid,
                )
            )

        return out
