"""
Phase 4D — Order Block detection.

Order Blocks are derived exclusively from confirmed BOS observations
(StructureBreakDetector, variant="bos"). CHoCH does not create Order Blocks.

Candidate selection (locked):
  - Bound the search to candles with epoch in [source_swing_event_time, bos_epoch).
  - Require exact timeframe continuity from the swing event candle through the
    BOS candle; any gap aborts OB formation for that BOS (no inference).
  - Scan backward from the candle immediately before the BOS candle toward the
    swing event candle; select the nearest opposing-color candle that passes
    the candidate-candle quality filter (min_body_ratio).
  - Opposing color: bullish BOS → bearish candle (close < open);
    bearish BOS → bullish candle (close > open).
  - Body ratio: abs(close - open) / (high - low) >= min_body_ratio.
  - Zero-range candles (high == low) are rejected.
  - If no qualifying candidate exists, no Order Block is emitted for that BOS.

Zone boundaries: full high–low of the candidate candle (not body).
Causal timing: confirmed_at equals the BOS confirmation epoch; never earlier.
Provenance: source candle epoch/OHLC in payload; triggering BOS via reference_ids.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.structure_break import StructureBreakDetector
from cpab.detectors.swing import (
    _validate_strength,
    _validate_tie_policy,
    timeframe_seconds,
)
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.order_block")

DEFAULT_MIN_BODY_RATIO = 0.5


def _validate_min_body_ratio(value: Any) -> float:
    """
    Validate min_body_ratio. Must be a finite number in [0.0, 1.0].
    Raise ValueError — no silent coercion.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"min_body_ratio must be a finite number in [0.0, 1.0], got {value!r}"
        )
    v = float(value)
    if v != v or abs(v) == float("inf"):
        raise ValueError(
            f"min_body_ratio must be a finite number in [0.0, 1.0], got {value!r}"
        )
    if v < 0.0 or v > 1.0:
        raise ValueError(
            f"min_body_ratio must be in [0.0, 1.0], got {value!r}"
        )
    return v


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def _ohlc_sane(c: Candle) -> bool:
    if not all(_finite(v) for v in (c.open, c.high, c.low, c.close)):
        return False
    if c.high < c.low:
        return False
    if c.high < c.open or c.high < c.close:
        return False
    if c.low > c.open or c.low > c.close:
        return False
    return True


def _body_ratio(c: Candle) -> Optional[float]:
    """Return body/range ratio, or None if zero-range or invalid."""
    if not _ohlc_sane(c):
        return None
    span = c.high - c.low
    if span == 0.0:
        return None
    return abs(c.close - c.open) / span


def _is_bearish(c: Candle) -> bool:
    return c.close < c.open


def _is_bullish(c: Candle) -> bool:
    return c.close > c.open


def _stable_ob_id(
    *,
    symbol: str,
    timeframe: str,
    candle_epoch: int,
    event_time: int,
    confirmed_at: int,
    detector_id: str,
    detector_config_id: str,
    direction: str,
    source_bos_id: str,
    source_candle_epoch: int,
) -> str:
    parts = [
        "order_block",
        symbol,
        timeframe,
        str(candle_epoch),
        f"e{event_time}",
        f"c{confirmed_at}",
        detector_id,
        detector_config_id,
        direction,
        source_bos_id,
        str(source_candle_epoch),
    ]
    return "|".join(parts)


class OrderBlockDetector(PriceActionDetector):
    """
    Order Block observations derived from confirmed BOS only.

    Configuration:
      - min_body_ratio (default 0.5): candidate-candle quality filter
      - left_strength, right_strength, tie_policy: forwarded to internal
        StructureBreakDetector / SwingPivotDetector
    """

    def __init__(
        self,
        *,
        min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "order_block",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._min_body_ratio = _validate_min_body_ratio(min_body_ratio)
        self._left = _validate_strength("left_strength", left_strength)
        self._right = _validate_strength("right_strength", right_strength)
        self._tie_policy = _validate_tie_policy(tie_policy)
        self._detector_id = detector_id

        self._structure = StructureBreakDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
            detector_id="structure_break",
        )

        cfg: dict[str, Any] = {
            "min_body_ratio": self._min_body_ratio,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        if detector_config_id is not None:
            self._detector_config_id = detector_config_id
        else:
            canonical = json.dumps(
                cfg, sort_keys=True, separators=(",", ":"), default=str
            )
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
        return self._structure.min_history

    def config_dict(self) -> dict[str, Any]:
        return {
            "min_body_ratio": self._min_body_ratio,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(
        self,
        candles: list[Candle],
        *,
        history_complete: bool = False,
    ) -> list[Observation]:
        """
        Incremental path: full reconstruction, then emit only OBs whose
        confirmed_at equals the latest closed bar's open epoch.
        """
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        all_obs = self._reconstruct(closed, history_complete=history_complete)
        latest_epoch = closed[-1].epoch
        return [o for o in all_obs if o.confirmed_at == latest_epoch]

    def detect_all_historical(
        self,
        candles: list[Candle],
        *,
        history_complete: bool = False,
    ) -> list[Observation]:
        """Full historical reconstruction for tests / offline backfill."""
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        return self._reconstruct(closed, history_complete=history_complete)

    def _reconstruct(
        self,
        closed: list[Candle],
        *,
        history_complete: bool,
    ) -> list[Observation]:
        if not closed:
            return []
        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe; skipping order-block detect")
            return []

        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            logger.debug("Unknown timeframe %r; skipping order-block detect", timeframe)
            return []

        # Index candles by epoch for O(1) lookup and ordered scan.
        by_epoch: dict[int, Candle] = {}
        ordered: list[Candle] = []
        for c in closed:
            if c.epoch in by_epoch:
                # Duplicate epoch — treat as discontinuity risk; keep first.
                continue
            by_epoch[c.epoch] = c
            ordered.append(c)
        ordered.sort(key=lambda c: c.epoch)
        epoch_to_idx = {c.epoch: i for i, c in enumerate(ordered)}

        bos_list = self._structure.detect_all_historical(
            closed, history_complete=history_complete
        )
        bos_only = [o for o in bos_list if o.variant == "bos"]

        out: list[Observation] = []
        seen: set[str] = set()

        for bos in bos_only:
            direction = bos.direction
            if direction not in ("bullish", "bearish"):
                continue

            bos_epoch = bos.confirmed_at if bos.confirmed_at is not None else bos.candle_epoch
            if bos_epoch is None or bos_epoch not in epoch_to_idx:
                continue

            payload = bos.payload or {}
            swing_event = payload.get("source_swing_event_time")
            if swing_event is None:
                swing_event = payload.get("source_swing_confirmed_at")
            if swing_event is None:
                logger.debug("BOS %s missing source_swing_event_time; skip", bos.id)
                continue
            try:
                swing_event = int(swing_event)
            except (TypeError, ValueError):
                continue
            if swing_event not in epoch_to_idx:
                # Swing pivot candle not present in this closed series.
                logger.debug(
                    "BOS %s swing event %s not in closed series; skip",
                    bos.id,
                    swing_event,
                )
                continue

            swing_idx = epoch_to_idx[swing_event]
            bos_idx = epoch_to_idx[bos_epoch]
            if bos_idx <= swing_idx:
                # No candles between swing and BOS (or inverted).
                continue

            # Continuity: every step from swing_idx through bos_idx must be exact.
            continuous = True
            for i in range(swing_idx + 1, bos_idx + 1):
                if ordered[i].epoch - ordered[i - 1].epoch != step:
                    continuous = False
                    break
            if not continuous:
                logger.debug(
                    "Discontinuity between swing %s and BOS %s; no OB",
                    swing_event,
                    bos_epoch,
                )
                continue

            # Scan nearest opposing qualifying candle: bos_idx-1 down to swing_idx.
            candidate: Optional[Candle] = None
            for i in range(bos_idx - 1, swing_idx - 1, -1):
                c = ordered[i]
                if direction == "bullish":
                    if not _is_bearish(c):
                        continue
                else:
                    if not _is_bullish(c):
                        continue
                ratio = _body_ratio(c)
                if ratio is None:
                    continue
                if ratio < self._min_body_ratio:
                    continue
                candidate = c
                break

            if candidate is None:
                continue

            dedupe = f"{bos.id}|{candidate.epoch}"
            if dedupe in seen:
                continue
            seen.add(dedupe)

            ratio = _body_ratio(candidate)
            assert ratio is not None

            event_time = bos_epoch
            confirmed_at = bos_epoch
            candle_epoch = bos_epoch

            lower_bound = candidate.low
            upper_bound = candidate.high

            ob_payload: dict[str, Any] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "size": upper_bound - lower_bound,
                "source_candle_epoch": candidate.epoch,
                "source_bos_id": bos.id,
                "source_swing_event_time": swing_event,
                "source_swing_id": payload.get("source_swing_id"),
                "broken_level_price": payload.get("broken_level_price"),
                "body_ratio": ratio,
                "min_body_ratio": self._min_body_ratio,
                "open": candidate.open,
                "high": candidate.high,
                "low": candidate.low,
                "close": candidate.close,
                "left_strength": self._left,
                "right_strength": self._right,
                "tie_policy": self._tie_policy,
            }

            oid = _stable_ob_id(
                symbol=symbol,
                timeframe=timeframe,
                candle_epoch=candle_epoch,
                event_time=event_time,
                confirmed_at=confirmed_at,
                detector_id=self._detector_id,
                detector_config_id=self._detector_config_id,
                direction=direction,
                source_bos_id=bos.id,
                source_candle_epoch=candidate.epoch,
            )

            out.append(
                Observation.create(
                    type="order_block",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_epoch=candle_epoch,
                    event_time=event_time,
                    confirmed_at=confirmed_at,
                    developing=False,
                    direction=direction,
                    variant=direction,
                    detector_id=self._detector_id,
                    detector_config_id=self._detector_config_id,
                    reference_ids=(bos.id,),
                    payload=ob_payload,
                    metadata=ob_payload,
                    id=oid,
                )
            )

        return out
