"""
Phase 4E — Breaker Block detection.

A Breaker Block forms immediately when an existing confirmed Order Block is
invalidated by a qualifying completed-candle close-through:

  Bullish Order Block [L, U] → bearish breaker (variant=bullish_to_bearish):
      close strictly below L.

  Bearish Order Block [L, U] → bullish breaker (variant=bearish_to_bullish):
      close strictly above U.

Equality with the boundary does not qualify. Wick-only breaches do not qualify.
No retest, BOS, or CHoCH is required.

Original Order Block zone bounds are preserved exactly (not recalculated).
This detector composes OrderBlockDetector; it does not reimplement OB formation
or mutate source OB observations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.order_block import OrderBlockDetector, DEFAULT_MIN_BODY_RATIO
from cpab.detectors.swing import timeframe_seconds
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.breaker_block")


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


def _is_ob_failure(direction: str, lower: float, upper: float, candle: Candle) -> bool:
    """
    Strict close-through invalidation of an Order Block zone.

    Bullish OB: close < lower (equality does not qualify).
    Bearish OB: close > upper (equality does not qualify).
    """
    if direction == "bullish":
        return candle.close < lower
    if direction == "bearish":
        return candle.close > upper
    return False


def _stable_breaker_id(
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
    source_ob_id: str,
) -> str:
    parts = [
        "breaker_block",
        symbol,
        timeframe,
        str(candle_epoch),
        f"e{event_time}",
        f"c{confirmed_at}",
        detector_id,
        detector_config_id,
        direction,
        variant,
        source_ob_id,
    ]
    return "|".join(parts)


class BreakerBlockDetector(PriceActionDetector):
    """
    Breaker Block observations derived exclusively from confirmed Order Blocks
    and a strict close-through of the OB zone on a completed candle.

    Configuration is forwarded to the internal OrderBlockDetector
    (min_body_ratio, left_strength, right_strength, tie_policy).
    No additional invalidation thresholds or retest options in V1.
    """

    def __init__(
        self,
        *,
        min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "breaker_block",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._detector_id = detector_id
        self._ob = OrderBlockDetector(
            min_body_ratio=min_body_ratio,
            left_strength=left_strength,
            right_strength=right_strength,
            tie_policy=tie_policy,
            detector_id="order_block",
        )
        cfg = self._ob.config_dict()
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
        # Need at least one completed bar after an Order Block can form.
        return self._ob.min_history + 1

    def config_dict(self) -> dict[str, Any]:
        return self._ob.config_dict()

    def detect(
        self,
        candles: list[Candle],
        *,
        history_complete: bool = False,
    ) -> list[Observation]:
        """
        Incremental path: full reconstruction, then emit only breakers whose
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
            logger.debug("Mixed symbol/timeframe; skipping breaker-block detect")
            return []

        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            logger.debug("Unknown timeframe %r; skipping breaker-block detect", timeframe)
            return []

        order_blocks = self._ob.detect_all_historical(
            closed, history_complete=history_complete
        )
        if not order_blocks:
            return []

        epoch_to_idx: dict[int, int] = {c.epoch: i for i, c in enumerate(closed)}

        out: list[Observation] = []
        seen: set[str] = set()

        for ob in order_blocks:
            if ob.type != "order_block":
                continue
            direction = ob.direction or ""
            if direction not in ("bullish", "bearish"):
                continue

            payload = ob.payload or {}
            try:
                lower = float(payload["lower_bound"])
                upper = float(payload["upper_bound"])
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping OB %s: missing bounds", ob.id)
                continue
            if not (upper > lower):
                continue

            ob_epoch = ob.confirmed_at if ob.confirmed_at is not None else ob.candle_epoch
            if ob_epoch is None:
                continue
            form_idx = epoch_to_idx.get(ob_epoch)
            if form_idx is None:
                continue

            # First strict close-through after the OB confirmation bar only.
            for j in range(form_idx + 1, len(closed)):
                cur = closed[j]
                prev = closed[j - 1]

                # Exact continuity with immediate predecessor required.
                if cur.epoch - prev.epoch != step:
                    continue

                if not _ohlc_sane(cur):
                    continue

                if not _is_ob_failure(direction, lower, upper, cur):
                    continue

                # One breaker per source OB (first failure only).
                dedupe = ob.id
                if dedupe in seen:
                    break
                seen.add(dedupe)

                if direction == "bullish":
                    variant = "bullish_to_bearish"
                    breaker_direction = "bearish"
                else:
                    variant = "bearish_to_bullish"
                    breaker_direction = "bullish"

                event_time = cur.epoch
                confirmed_at = cur.epoch
                candle_epoch = cur.epoch

                breaker_payload: dict[str, Any] = {
                    "source_ob_id": ob.id,
                    "source_ob_direction": direction,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "size": upper - lower,
                    "invalidation_candle_epoch": candle_epoch,
                    "open": cur.open,
                    "high": cur.high,
                    "low": cur.low,
                    "close": cur.close,
                }
                if "source_candle_epoch" in payload:
                    breaker_payload["source_ob_candle_epoch"] = payload[
                        "source_candle_epoch"
                    ]
                if "source_bos_id" in payload:
                    breaker_payload["source_bos_id"] = payload["source_bos_id"]

                oid = _stable_breaker_id(
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_epoch=candle_epoch,
                    event_time=event_time,
                    confirmed_at=confirmed_at,
                    detector_id=self._detector_id,
                    detector_config_id=self._detector_config_id,
                    direction=breaker_direction,
                    variant=variant,
                    source_ob_id=ob.id,
                )

                out.append(
                    Observation.create(
                        type="breaker_block",
                        symbol=symbol,
                        timeframe=timeframe,
                        candle_epoch=candle_epoch,
                        event_time=event_time,
                        confirmed_at=confirmed_at,
                        developing=False,
                        direction=breaker_direction,
                        variant=variant,
                        detector_id=self._detector_id,
                        detector_config_id=self._detector_config_id,
                        reference_ids=(ob.id,),
                        payload=breaker_payload,
                        metadata=breaker_payload,
                        id=oid,
                    )
                )
                break  # first failure only for this OB

        return out
