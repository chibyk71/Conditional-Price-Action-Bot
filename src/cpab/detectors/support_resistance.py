"""
Phase 3A — Support / resistance zones from confirmed swing pivots.

Each confirmed swing_low yields a support zone; each confirmed swing_high yields
a resistance zone. Zones are zero-width: lower_bound == upper_bound == source
swing price. No merging, width, ATR, or tolerance is applied.

Reuses SwingPivotDetector for confirmed swing reconstruction. Does not implement
an independent pivot algorithm. Does not consume equal-high/low observations.

Observation convention:
  type = "support_resistance_zone"
  variant = "support" | "resistance"
  direction = None  (polarity is carried only by variant)

Timing inherits from the source swing:
  event_time / candle_epoch = source swing event_time
  confirmed_at = source swing confirmed_at

reference_ids = (source_swing_id,)

Incremental detect() emits only zones whose confirmed_at equals the latest
closed bar's open epoch (same latest-bar focus as Phase 2A–2D).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.swing import SwingPivotDetector, _validate_strength, _validate_tie_policy
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.support_resistance")


def _finite_bound(value: Any, name: str) -> float:
    """Require a finite numeric bound; reject bool, NaN, inf, non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return f


class SupportResistanceZoneDetector(PriceActionDetector):
    """
    Support and resistance zones derived exclusively from confirmed Phase 2A swings.

    Configuration is immutable after construction. Invalid swing parameters
    raise ValueError (no silent defaults or coercion for invalid values).
    """

    def __init__(
        self,
        *,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "support_resistance_zone",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._left = _validate_strength("left_strength", left_strength)
        self._right = _validate_strength("right_strength", right_strength)
        self._tie_policy = _validate_tie_policy(tie_policy)
        self._detector_id = detector_id

        self._swing = SwingPivotDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
            detector_id="swing_pivot",
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
        """Same as SwingPivotDetector: one confirmed swing requires left+1+right bars."""
        return self._left + 1 + self._right

    def config_dict(self) -> dict[str, Any]:
        return {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: reconstruct zones from closed history, then emit only
        those whose confirmed_at equals the latest closed bar's open epoch.
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
            logger.debug("Mixed symbol/timeframe; skipping support-resistance detect")
            return []

        swing_obs = self._swing.detect_all_historical(closed)
        out: list[Observation] = []
        for swing in swing_obs:
            zone = self._zone_from_swing(swing, symbol=symbol, timeframe=timeframe)
            if zone is not None:
                out.append(zone)
        return out

    def _zone_from_swing(
        self,
        swing: Observation,
        *,
        symbol: str,
        timeframe: str,
    ) -> Optional[Observation]:
        if swing.type == "swing_high":
            variant = "resistance"
        elif swing.type == "swing_low":
            variant = "support"
        else:
            return None

        raw_price = swing.payload.get("price", swing.metadata.get("price"))
        try:
            price = _finite_bound(raw_price, "source swing price")
        except ValueError:
            logger.debug("Non-finite swing price for %s; skip zone", swing.id)
            return None

        lower_bound = price
        upper_bound = price
        if lower_bound > upper_bound:
            # Defensive; zero-width zones should never violate this.
            return None

        event_time = swing.event_time if swing.event_time is not None else swing.candle_epoch
        confirmed_at = swing.confirmed_at
        if confirmed_at is None:
            # Developing swings are never emitted by SwingPivotDetector; skip defensively.
            return None

        payload: dict[str, Any] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "source_price": price,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

        return Observation.create(
            type="support_resistance_zone",
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=event_time,
            event_time=event_time,
            confirmed_at=confirmed_at,
            developing=False,
            direction=None,
            variant=variant,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=(swing.id,),
            payload=payload,
            metadata=payload,
        )
