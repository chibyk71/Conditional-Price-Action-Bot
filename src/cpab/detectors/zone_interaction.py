"""
Phase 3B — Zone interaction detection (touch, rejection, break).

Consumes Phase 3A support/resistance zone observations produced by
SupportResistanceZoneDetector. Does not reimplement zone construction;
composes the Phase 3A detector for zone reconstruction, matching the
StructureBreakDetector ↔ SwingPivotDetector pattern.

Locked interaction definitions (completed candles only):

  Touch
    candle.low <= P <= candle.high
    variant="touch", direction=None

  Rejection (requires touch + close strictly away on the expected side)
    Support:   touch and close > P  → rejection, bullish
    Resistance: touch and close < P → rejection, bearish
    Close exactly at P is not a rejection.
    A candle may emit both touch and rejection for the same zone (distinct IDs).

  Break (genuine close-crossing transition only)
    Support broken downward:
      prev.close >= P and cur.close < P → break, bearish
    Resistance broken upward:
      prev.close <= P and cur.close > P → break, bullish
    Wick-only penetration is not a break.
    Close exactly at P does not complete a break.
    Once a zone has broken, remaining beyond the level does not re-emit.
    A zone that becomes available while price is already beyond it does not
    emit a false break (requires a valid prior close on the level side).

Causal availability:
  zone.confirmed_at <= interaction_candle.epoch

Gap handling (Phase 2A / 2D convention):
  Continuous only when open epochs differ by exactly the timeframe step.
  Discontinuous previous candle suppresses break crossing evaluation.

Observation:
  type = "zone_interaction"
  variant = "touch" | "rejection" | "break"
  direction = None (touch) | "bullish" | "bearish"
  reference_ids = (source_zone_id,)

Historical reconstruction is pure. Incremental detect() emits only events
whose confirmed_at equals the latest closed bar's open epoch.

Retests and multi-candle interaction sequences are out of scope (Phase 3C).
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
    timeframe_seconds,
)
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.zone_interaction")


def _stable_interaction_id(
    *,
    symbol: str,
    timeframe: str,
    candle_epoch: int,
    event_time: int,
    confirmed_at: int,
    detector_id: str,
    detector_config_id: str,
    direction: Optional[str],
    variant: str,
    source_zone_id: str,
) -> str:
    """
    Deterministic identity including the source zone so:
    * interactions on the same candle against different zones remain distinct;
    * touch / rejection / break on the same zone+candle remain distinct;
    * repeated reconstruction yields identical IDs.
    """
    parts = [
        "zone_interaction",
        symbol,
        timeframe,
        str(candle_epoch),
        f"e{event_time}",
        f"c{confirmed_at}",
        detector_id,
        detector_config_id,
        direction or "",
        variant,
        source_zone_id,
    ]
    return "|".join(parts)


class ZoneInteractionDetector(PriceActionDetector):
    """
    Touch / rejection / break detector built on SupportResistanceZoneDetector.

    Configuration is immutable after construction. Invalid swing parameters
    raise ValueError (no silent defaults or coercion for invalid values).

    Zone generation stays in Phase 3A; this detector only evaluates interactions
    against zones reconstructed from the same candle history via the Phase 3A
    detector (composition, not independent zone algorithms).
    """

    def __init__(
        self,
        *,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "zone_interaction",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._left = _validate_strength("left_strength", left_strength)
        self._right = _validate_strength("right_strength", right_strength)
        self._tie_policy = _validate_tie_policy(tie_policy)
        self._detector_id = detector_id

        self._zones = SupportResistanceZoneDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
            detector_id="support_resistance_zone",
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
        Minimum closed candles for any interaction to become possible.

        Need at least one confirmed zone (same as Phase 3A min_history) plus
        room for a prior/current close pair on break evaluation. Touch and
        rejection need only the zone-eligible candle, but break needs a prior
        continuous candle; use the stricter bound so callers request enough.
        """
        return self._left + 1 + self._right + 1

    def config_dict(self) -> dict[str, Any]:
        return {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: reconstruct all interactions from closed history,
        then emit only those whose confirmed_at equals the latest closed bar's
        open epoch.
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
            logger.debug("Mixed symbol/timeframe; skipping zone-interaction detect")
            return []

        step = timeframe_seconds(timeframe)
        zone_obs = self._zones.detect_all_historical(closed)

        def zone_sort_key(o: Observation) -> tuple:
            return (
                o.confirmed_at if o.confirmed_at is not None else 0,
                o.event_time if o.event_time is not None else o.candle_epoch,
                o.id,
            )

        zones_sorted = sorted(zone_obs, key=zone_sort_key)
        broken: set[str] = set()
        out: list[Observation] = []

        for i, cur in enumerate(closed):
            eligible = [
                z
                for z in zones_sorted
                if z.confirmed_at is not None and z.confirmed_at <= cur.epoch
            ]

            continuous = False
            prev: Optional[Candle] = None
            if i >= 1:
                prev = closed[i - 1]
                if step is not None and step > 0 and cur.epoch - prev.epoch == step:
                    continuous = True

            for zone in eligible:
                level_price = float(
                    zone.payload.get(
                        "source_price",
                        zone.payload.get("lower_bound", zone.payload.get("upper_bound", 0.0)),
                    )
                )
                zone_variant = zone.variant  # "support" | "resistance"

                # --- Touch ---
                touches = cur.low <= level_price <= cur.high
                if touches:
                    out.append(
                        self._make_observation(
                            variant="touch",
                            direction=None,
                            zone=zone,
                            level_price=level_price,
                            candle=cur,
                            prev_close=prev.close if prev is not None else None,
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                    )

                    # --- Rejection (touch + close strictly away) ---
                    if zone_variant == "support" and cur.close > level_price:
                        out.append(
                            self._make_observation(
                                variant="rejection",
                                direction="bullish",
                                zone=zone,
                                level_price=level_price,
                                candle=cur,
                                prev_close=prev.close if prev is not None else None,
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                        )
                    elif zone_variant == "resistance" and cur.close < level_price:
                        out.append(
                            self._make_observation(
                                variant="rejection",
                                direction="bearish",
                                zone=zone,
                                level_price=level_price,
                                candle=cur,
                                prev_close=prev.close if prev is not None else None,
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                        )

                # --- Break (transition only; continuous prior required) ---
                if zone.id in broken:
                    continue
                if not continuous or prev is None:
                    continue

                is_break = False
                break_direction: Optional[str] = None
                if zone_variant == "support":
                    if prev.close >= level_price and cur.close < level_price:
                        is_break = True
                        break_direction = "bearish"
                elif zone_variant == "resistance":
                    if prev.close <= level_price and cur.close > level_price:
                        is_break = True
                        break_direction = "bullish"

                if is_break and break_direction is not None:
                    out.append(
                        self._make_observation(
                            variant="break",
                            direction=break_direction,
                            zone=zone,
                            level_price=level_price,
                            candle=cur,
                            prev_close=prev.close,
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                    )
                    broken.add(zone.id)

        return out

    def _make_observation(
        self,
        *,
        variant: str,
        direction: Optional[str],
        zone: Observation,
        level_price: float,
        candle: Candle,
        prev_close: Optional[float],
        symbol: str,
        timeframe: str,
    ) -> Observation:
        event_time = candle.epoch
        confirmed_at = candle.epoch
        source_zone_id = zone.id
        zone_variant = zone.variant

        payload: dict[str, Any] = {
            "zone_type": "support_resistance_zone",
            "zone_variant": zone_variant,
            "level_price": level_price,
            "interaction_candle_epoch": candle.epoch,
            "source_zone_event_time": zone.event_time
            if zone.event_time is not None
            else zone.candle_epoch,
            "source_zone_confirmed_at": zone.confirmed_at,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        if prev_close is not None:
            payload["previous_close"] = prev_close

        oid = _stable_interaction_id(
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=candle.epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            direction=direction,
            variant=variant,
            source_zone_id=source_zone_id,
        )

        return Observation.create(
            type="zone_interaction",
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=candle.epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            developing=False,
            direction=direction,
            variant=variant,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=(source_zone_id,),
            payload=payload,
            metadata=payload,
            id=oid,
        )
