"""
Phase 3C — Zone retest detection.

Detects the first confirmed retest following a confirmed support/resistance
zone break (Phase 3B). Composes SupportResistanceZoneDetector and
ZoneInteractionDetector for zone and break reconstruction; does not
reimplement zone construction or break definitions.

Locked event definition (completed candles only):

  Prerequisites
    1. Confirmed source zone (Phase 3A).
    2. Confirmed Phase 3B break of that zone.
    3. Subsequent continuous approach from the broken side.
    4. Contact with the zero-width zone level (range includes level).
    5. Completed candle closing strictly on the broken side.

  Broken support (bearish retest):
    Break:     close strictly below the support level.
    Contact:   candle range includes the level.
    Confirm:   candle closes strictly below the level.
    Direction: bearish.

  Broken resistance (bullish retest):
    Break:     close strictly above the resistance level.
    Contact:   candle range includes the level.
    Confirm:   candle closes strictly above the level.
    Direction: bullish.

  A wick touch without the closing condition is not a confirmed retest.
  A close exactly at the level is neutral: does not confirm, does not
  invalidate; the sequence remains pending subject to continuity.

Sequence invalidation
  A completed candle closing strictly on the original side of the level
  invalidates the pending break-and-retest sequence. No separate
  invalidation observation is emitted in V1. Once invalidated, that break
  cannot produce a retest.

  Gaps must never be bridged. Continuous only when open epochs differ by
  exactly the timeframe step (same convention as Phase 2A / 2D / 3B).

  A valid retest consumes the break's one permitted retest event.
  The break candle itself cannot be a retest candle.

Observation:
  type = "zone_retest"
  variant = "retest"
  direction = "bearish" | "bullish"
  reference_ids = (source_zone_id, break_observation_id)

Historical reconstruction is pure. Incremental detect() emits only events
whose confirmed_at equals the latest closed bar's open epoch.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.support_resistance import SupportResistanceZoneDetector
from cpab.detectors.swing import (
    _validate_strength,
    _validate_tie_policy,
    timeframe_seconds,
)
from cpab.detectors.zone_interaction import ZoneInteractionDetector, _zone_level_price
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.zone_retest")


def _stable_retest_id(
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
    break_observation_id: str,
) -> str:
    """
    Deterministic identity including source zone and break so:
    * retests from distinct breaks remain distinct even at the same price;
    * repeated reconstruction yields identical IDs.
    """
    parts = [
        "zone_retest",
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
        break_observation_id,
    ]
    return "|".join(parts)


class ZoneRetestDetector(PriceActionDetector):
    """
    First confirmed retest after a confirmed zone break.

    Configuration is immutable after construction. Invalid swing parameters
    raise ValueError (no silent defaults or coercion for invalid values).

    Zone and break generation stay in Phase 3A/3B; this detector only
    evaluates retest sequences against reconstructed zones and breaks from
    the same candle history (composition).
    """

    def __init__(
        self,
        *,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "zone_retest",
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
        Minimum closed candles for a retest to become possible.

        Need a confirmed zone, a subsequent break (needs prior continuous
        candle), and at least one post-break candle for contact+close.
        Use the stricter bound from the composed interaction detector plus
        one extra bar so a post-break retest candle can exist.
        """
        return self._interactions.min_history + 1

    def config_dict(self) -> dict[str, Any]:
        return {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: reconstruct all retests from closed history,
        then emit only those whose confirmed_at equals the latest closed
        bar's open epoch.
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
            logger.debug("Mixed symbol/timeframe; skipping zone-retest detect")
            return []

        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            logger.debug("Unknown timeframe step; skipping zone-retest detect")
            return []

        # Reconstruct zones and interactions (breaks) from the same history.
        zone_obs = self._zones.detect_all_historical(closed)
        interaction_obs = self._interactions.detect_all_historical(closed)

        zones_by_id: dict[str, Observation] = {z.id: z for z in zone_obs}
        breaks = [o for o in interaction_obs if o.variant == "break"]

        # Sort breaks chronologically for stable processing.
        def break_sort_key(o: Observation) -> tuple:
            return (
                o.confirmed_at if o.confirmed_at is not None else 0,
                o.event_time if o.event_time is not None else o.candle_epoch,
                o.id,
            )

        breaks_sorted = sorted(breaks, key=break_sort_key)

        # Index candles by epoch for quick lookup.
        epoch_to_idx: dict[int, int] = {c.epoch: i for i, c in enumerate(closed)}

        out: list[Observation] = []
        # Each break may produce at most one retest.
        consumed_breaks: set[str] = set()

        for brk in breaks_sorted:
            if brk.id in consumed_breaks:
                continue
            if not brk.reference_ids:
                continue
            source_zone_id = brk.reference_ids[0]
            zone = zones_by_id.get(source_zone_id)
            if zone is None:
                continue

            level_price = _zone_level_price(zone)
            if level_price is None:
                logger.debug(
                    "Skipping break %s: zone %s missing or non-finite level",
                    brk.id,
                    source_zone_id,
                )
                continue

            zone_variant = zone.variant  # "support" | "resistance"
            break_epoch = brk.candle_epoch
            break_idx = epoch_to_idx.get(break_epoch)
            if break_idx is None:
                continue

            # Scan candles strictly after the break candle.
            pending = True
            for j in range(break_idx + 1, len(closed)):
                if not pending:
                    break
                cur = closed[j]
                prev = closed[j - 1]

                # Continuity required for the approach sequence.
                if cur.epoch - prev.epoch != step:
                    # Gap: invalidate pending sequence for this break.
                    pending = False
                    break

                # Invalidation: close strictly on the original side.
                if zone_variant == "support":
                    # Original side is above the level.
                    if cur.close > level_price:
                        pending = False
                        break
                else:
                    # Resistance: original side is below the level.
                    if cur.close < level_price:
                        pending = False
                        break

                # Contact: range includes the level.
                contacts = cur.low <= level_price <= cur.high
                if not contacts:
                    # Still beyond without contact — sequence remains pending
                    # as long as we stay on the broken side (checked above)
                    # and continuity holds.
                    continue

                # Confirmation: close strictly on the broken side.
                confirmed = False
                retest_direction: Optional[str] = None
                if zone_variant == "support":
                    if cur.close < level_price:
                        confirmed = True
                        retest_direction = "bearish"
                else:
                    if cur.close > level_price:
                        confirmed = True
                        retest_direction = "bullish"

                # Close exactly at level: neutral — neither confirm nor invalidate.
                if not confirmed:
                    continue

                # Emit retest and consume the break.
                assert retest_direction is not None
                out.append(
                    self._make_observation(
                        direction=retest_direction,
                        zone=zone,
                        level_price=level_price,
                        break_obs=brk,
                        candle=cur,
                        prev_close=prev.close,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )
                consumed_breaks.add(brk.id)
                pending = False
                break

        return out

    def _make_observation(
        self,
        *,
        direction: str,
        zone: Observation,
        level_price: float,
        break_obs: Observation,
        candle: Candle,
        prev_close: float,
        symbol: str,
        timeframe: str,
    ) -> Observation:
        event_time = candle.epoch
        confirmed_at = candle.epoch
        source_zone_id = zone.id
        break_id = break_obs.id
        zone_variant = zone.variant

        payload: dict[str, Any] = {
            "zone_type": "support_resistance_zone",
            "zone_variant": zone_variant,
            "level_price": level_price,
            "source_zone_event_time": zone.event_time
            if zone.event_time is not None
            else zone.candle_epoch,
            "source_zone_confirmed_at": zone.confirmed_at,
            "break_observation_id": break_id,
            "break_event_time": break_obs.event_time
            if break_obs.event_time is not None
            else break_obs.candle_epoch,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "previous_close": prev_close,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

        oid = _stable_retest_id(
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=candle.epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            direction=direction,
            variant="retest",
            source_zone_id=source_zone_id,
            break_observation_id=break_id,
        )

        return Observation.create(
            type="zone_retest",
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=candle.epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            developing=False,
            direction=direction,
            variant="retest",
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=(source_zone_id, break_id),
            payload=payload,
            metadata=payload,
            id=oid,
        )
