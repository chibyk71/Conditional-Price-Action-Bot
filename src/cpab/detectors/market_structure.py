"""
Phase 2C — HH / HL / LH / LL market-structure classification.

Classifies each eligible confirmed swing relative to the immediately preceding
confirmed swing of the same type (highs vs highs, lows vs lows).

Outcomes:
  - HH — Higher High
  - LH — Lower High
  - HL — Higher Low
  - LL — Lower Low

Exact numeric equality produces no classification, but the current swing still
becomes the comparison reference for the next swing of that type (reference
advancement).

Reuses SwingPivotDetector for confirmed swing reconstruction. Does not implement
an independent pivot algorithm. Does not consume EqualLevelsDetector outputs.

Observation convention:
  type = "market_structure"
  variant = "hh" | "lh" | "hl" | "ll"

Timing inherits from the source (classified) swing. developing is always False.
reference_ids = (source_swing_id, reference_swing_id) in that order.

Incremental detect() emits only classifications whose source swing becomes
newly confirmable at the latest closed bar's open epoch.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.swing import SwingPivotDetector, _validate_strength, _validate_tie_policy
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.market_structure")


class MarketStructureDetector(PriceActionDetector):
    """
    HH/HL/LH/LL classifier built on SwingPivotDetector.

    Configuration is immutable after construction. Invalid swing parameters
    raise ValueError (no silent defaults or coercion for invalid values).
    """

    def __init__(
        self,
        *,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "market_structure",
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
        """
        Minimum closed candles for any classification to become confirmable.

        Requires at least two independently confirmable swings of the same type.
        Lower bound mirrors equal_levels: 2 * (left + 1 + right) - 1.
        """
        s = self._left + 1 + self._right
        return 2 * s - 1

    def config_dict(self) -> dict[str, Any]:
        return {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: reconstruct classifications from closed history, then
        emit only those whose confirmed_at equals the latest closed bar's open
        epoch (same latest-bar focus as SwingPivotDetector / EqualLevelsDetector).
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
            logger.debug("Mixed symbol/timeframe; skipping market-structure detect")
            return []

        swing_obs = self._swing.detect_all_historical(closed)
        if len(swing_obs) < 2:
            return []

        highs = [o for o in swing_obs if o.type == "swing_high"]
        lows = [o for o in swing_obs if o.type == "swing_low"]

        out: list[Observation] = []
        out.extend(self._classify_stream(highs, kind="high", symbol=symbol, timeframe=timeframe))
        out.extend(self._classify_stream(lows, kind="low", symbol=symbol, timeframe=timeframe))
        return out

    def _classify_stream(
        self,
        swings: list[Observation],
        *,
        kind: str,
        symbol: str,
        timeframe: str,
    ) -> list[Observation]:
        if len(swings) < 2:
            return []

        ordered = sorted(
            swings,
            key=lambda o: (
                o.confirmed_at if o.confirmed_at is not None else 0,
                o.event_time if o.event_time is not None else o.candle_epoch,
                o.id,
            ),
        )

        out: list[Observation] = []
        prev: Optional[Observation] = None

        for swing in ordered:
            if prev is None:
                prev = swing
                continue

            src_price = float(swing.payload.get("price", 0.0))
            ref_price = float(prev.payload.get("price", 0.0))
            ref_swing = prev

            label: Optional[str] = None
            if kind == "high":
                if src_price > ref_price:
                    label = "hh"
                elif src_price < ref_price:
                    label = "lh"
            else:
                if src_price > ref_price:
                    label = "hl"
                elif src_price < ref_price:
                    label = "ll"

            # Always advance reference (mandatory equal-price rule).
            prev = swing

            if label is None:
                continue

            out.append(
                self._make_observation(
                    label=label,
                    source=swing,
                    reference=ref_swing,
                    symbol=symbol,
                    timeframe=timeframe,
                )
            )
        return out

    def _make_observation(
        self,
        *,
        label: str,
        source: Observation,
        reference: Observation,
        symbol: str,
        timeframe: str,
    ) -> Observation:
        src_price = float(source.payload.get("price", 0.0))
        ref_price = float(reference.payload.get("price", 0.0))
        payload: dict[str, Any] = {
            "classification": label,
            "source_price": src_price,
            "reference_price": ref_price,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        return Observation.create(
            type="market_structure",
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=source.event_time if source.event_time is not None else source.candle_epoch,
            event_time=source.event_time if source.event_time is not None else source.candle_epoch,
            confirmed_at=source.confirmed_at,
            developing=False,
            direction=None,
            variant=label,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=(source.id, reference.id),
            payload=payload,
            metadata=payload,
        )
