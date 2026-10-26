"""Phase 6D — source-aware breakout behavior and failure lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.order_block import _ohlc_sane
from cpab.detectors.support_resistance import SupportResistanceZoneDetector
from cpab.detectors.swing import (
    SwingPivotDetector,
    _validate_strength,
    _validate_tie_policy,
    timeframe_seconds,
)
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.breakout_behavior")

FAILURE_POLICIES = frozenset({
    "close_back_across",
    "confirmed_close_back_across",
})


def _validate_failure_policy(value: Any) -> str:
    if not isinstance(value, str) or value not in FAILURE_POLICIES:
        raise ValueError(
            f"failure_policy must be one of {sorted(FAILURE_POLICIES)}, got {value!r}"
        )
    return value


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _stable_breakout_id(
    *,
    symbol: str,
    timeframe: str,
    candle_epoch: int,
    event_type: str,
    source_id: str,
    source_family: str,
    direction: str,
    boundary: float,
    detector_id: str,
    detector_config_id: str,
    failure_policy: str,
) -> str:
    """Stable source-aware identity; Observation.create omits source IDs."""
    identity = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_epoch": candle_epoch,
        "event_type": event_type,
        "source_id": source_id,
        "source_family": source_family,
        "direction": direction,
        "boundary": boundary,
        "detector_id": detector_id,
        "detector_config_id": detector_config_id,
        "failure_policy": failure_policy,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "breakout:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


class BreakoutBehaviorDetector(PriceActionDetector):
    """Observe breakouts of confirmed swing and zero-width S/R boundaries.

    A source becomes eligible only on candles strictly later than its
    confirmed_at epoch. Gaps and invalid OHLC candles split history into
    independent segments; lifecycle state never crosses a segment boundary.
    """

    def __init__(
        self,
        *,
        failure_policy: str = "close_back_across",
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "breakout_behavior",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._failure_policy = _validate_failure_policy(failure_policy)
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
        self._zones = SupportResistanceZoneDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
            detector_id="support_resistance_zone",
        )
        config = self.config_dict()
        if detector_config_id is not None:
            self._detector_config_id = detector_config_id
        else:
            canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
            self._detector_config_id = "cfg:" + hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()[:16]

    @property
    def name(self) -> str:
        return self._detector_id

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def min_history(self) -> int:
        return self._left + 1 + self._right

    def config_dict(self) -> dict[str, Any]:
        return {
            "failure_policy": self._failure_policy,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        all_events = self._reconstruct(closed)
        latest_epoch = closed[-1].epoch
        return [event for event in all_events if event.confirmed_at == latest_epoch]

    def detect_all_historical(self, candles: list[Candle]) -> list[Observation]:
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        return self._reconstruct(closed)

    def _segments(self, candles: list[Candle], step: int) -> list[list[Candle]]:
        """Split at gaps and invalid candles; never filter a break and bridge it."""
        segments: list[list[Candle]] = []
        current: list[Candle] = []
        previous_epoch: Optional[int] = None
        for candle in candles:
            if not _ohlc_sane(candle):
                if current:
                    segments.append(current)
                current = []
                previous_epoch = None
                continue
            if previous_epoch is not None and candle.epoch - previous_epoch != step:
                if current:
                    segments.append(current)
                current = []
            current.append(candle)
            previous_epoch = candle.epoch
        if current:
            segments.append(current)
        return segments

    def _reconstruct(self, closed: list[Candle]) -> list[Observation]:
        if not closed:
            return []
        symbol = closed[0].symbol
        timeframe = closed[0].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe; skipping breakout detection")
            return []
        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            logger.debug("Unknown timeframe %s; refusing to infer continuity", timeframe)
            return []

        output: list[Observation] = []
        for segment in self._segments(closed, step):
            if len(segment) < self.min_history:
                continue
            swings = self._swing.detect_all_historical(segment)
            zones = self._zones.detect_all_historical(segment)
            sources: list[tuple[Observation, str, float, str]] = []
            for source in swings:
                price = _finite_number(source.payload.get("price", source.metadata.get("price")))
                if price is None or source.confirmed_at is None or source.developing:
                    continue
                if source.type == "swing_high":
                    sources.append((source, "swing_structure", price, "bullish"))
                elif source.type == "swing_low":
                    sources.append((source, "swing_structure", price, "bearish"))

            for source in zones:
                if source.confirmed_at is None or source.developing:
                    continue
                variant = source.variant
                lower = _finite_number(source.payload.get("lower_bound"))
                upper = _finite_number(source.payload.get("upper_bound"))
                if lower is None or upper is None or lower != upper:
                    # V1 explicitly supports only existing zero-width S/R zones.
                    continue
                if variant == "resistance":
                    sources.append((source, "support_resistance_zone", lower, "bullish"))
                elif variant == "support":
                    sources.append((source, "support_resistance_zone", lower, "bearish"))

            # Stable ordering ensures deterministic output when sources share a price.
            sources.sort(key=lambda item: (
                item[0].confirmed_at if item[0].confirmed_at is not None else 0,
                item[0].id,
            ))
            for source, source_family, boundary, direction in sources:
                output.extend(self._track_source(
                    segment,
                    source=source,
                    source_family=source_family,
                    boundary=boundary,
                    direction=direction,
                ))
        output.sort(key=lambda o: (
            o.confirmed_at if o.confirmed_at is not None else 0,
            o.id,
        ))
        return output

    def _track_source(
        self,
        candles: list[Candle],
        *,
        source: Observation,
        source_family: str,
        boundary: float,
        direction: str,
    ) -> list[Observation]:
        source_confirmed_at = source.confirmed_at
        if source_confirmed_at is None:
            return []
        output: list[Observation] = []
        active = False
        pending_failure = False

        for candle in candles:
            # Strict causality: the source-confirming candle itself is ineligible.
            if candle.epoch <= source_confirmed_at:
                continue
            beyond = candle.close > boundary if direction == "bullish" else candle.close < boundary
            back_across = candle.close < boundary if direction == "bullish" else candle.close > boundary

            if not active:
                if beyond:
                    active = True
                    output.append(self._make_event(
                        candle=candle,
                        event_type="breakout_confirmed",
                        source=source,
                        source_family=source_family,
                        boundary=boundary,
                        direction=direction,
                    ))
                continue

            if self._failure_policy == "close_back_across":
                if back_across:
                    output.append(self._make_event(
                        candle=candle,
                        event_type="breakout_failed",
                        source=source,
                        source_family=source_family,
                        boundary=boundary,
                        direction=direction,
                    ))
                    break
            elif pending_failure:
                # Only the immediately following eligible completed candle counts.
                if back_across:
                    output.append(self._make_event(
                        candle=candle,
                        event_type="breakout_failed",
                        source=source,
                        source_family=source_family,
                        boundary=boundary,
                        direction=direction,
                    ))
                    break
                # Equality or reclaim of the breakout side resets the pending state.
                pending_failure = False
            elif back_across:
                pending_failure = True
        return output

    def _make_event(
        self,
        *,
        candle: Candle,
        event_type: str,
        source: Observation,
        source_family: str,
        boundary: float,
        direction: str,
    ) -> Observation:
        payload = {
            "event_type": event_type,
            "source_id": source.id,
            "source_family": source_family,
            "source_type": source.type,
            "source_variant": source.variant,
            "source_confirmed_at": source.confirmed_at,
            "boundary": boundary,
            "direction": direction,
            "failure_policy": self._failure_policy,
            "lifecycle_event": "confirmation" if event_type == "breakout_confirmed" else "failure",
            "lifecycle_state": "active" if event_type == "breakout_confirmed" else "failed",
        }
        event_id = _stable_breakout_id(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            candle_epoch=candle.epoch,
            event_type=event_type,
            source_id=source.id,
            source_family=source_family,
            direction=direction,
            boundary=boundary,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            failure_policy=self._failure_policy,
        )
        return Observation.create(
            type="breakout_behavior",
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            candle_epoch=candle.epoch,
            event_time=candle.epoch,
            confirmed_at=candle.epoch,
            developing=False,
            direction=direction,
            variant=event_type,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=(source.id,),
            lifecycle_state="active" if event_type == "breakout_confirmed" else "failed",
            payload=payload,
            metadata=payload,
            id=event_id,
        )
