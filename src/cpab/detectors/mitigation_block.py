"""
Phase 4F — Mitigation Block detection.

Tracks progressive wick-based interactions with confirmed Order Blocks and
Breaker Blocks. Emits immutable lifecycle observations for touch, partial_fill,
and full_fill. Does not mutate source observations or their bounds.

Locked geometry for source-zone bounds [L, U] where L < U:

  Bullish source (approached from above):
    near = U, far = L
    touch: candle range first intersects the zone (inclusive boundaries)
    partial_fill: candle.low penetrates below U while remaining above L
    full_fill: candle.low reaches or crosses L (inclusive)

  Bearish source (approached from below):
    near = L, far = U
    touch: candle range first intersects the zone (inclusive boundaries)
    partial_fill: candle.high penetrates above L while remaining below U
    full_fill: candle.high reaches or crosses U (inclusive)

Stages emit at most once per source, in order. A single candle that crosses
multiple thresholds emits any missing earlier stages before full_fill.
After full_fill no further events are emitted for that source.

Causality / continuity:
  - Completed candles only; source must already be confirmed before the
    interaction candle; never forming bars.
  - Exact timeframe step between consecutive candles required to evaluate.
  - Discontinuous candles are skipped; per-source state is preserved;
    evaluation resumes on later continuous pairs. No interpolation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.breaker_block import BreakerBlockDetector
from cpab.detectors.order_block import (
    DEFAULT_MIN_BODY_RATIO,
    OrderBlockDetector,
    _ohlc_sane,
)
from cpab.detectors.swing import timeframe_seconds
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.mitigation_block")

STAGES = ("touch", "partial_fill", "full_fill")


def _stable_mitigation_id(
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
    source_id: str,
) -> str:
    """
    Deterministic identity including source observation so independent sources
    remain distinct and repeated reconstruction yields identical IDs.
    """
    parts = [
        "mitigation_block",
        symbol,
        timeframe,
        str(candle_epoch),
        f"e{event_time}",
        f"c{confirmed_at}",
        detector_id,
        detector_config_id,
        direction,
        variant,
        source_id,
    ]
    return "|".join(parts)


def _intersects_closed_interval(candle: Candle, lower: float, upper: float) -> bool:
    """True if candle range intersects closed interval [lower, upper]."""
    return candle.high >= lower and candle.low <= upper


def _is_partial_fill(direction: str, lower: float, upper: float, candle: Candle) -> bool:
    """
    Strict interior penetration without reaching the far boundary.
    Bullish: L < low < U
    Bearish: L < high < U
    """
    if direction == "bullish":
        low = float(candle.low)
        return lower < low < upper
    if direction == "bearish":
        high = float(candle.high)
        return lower < high < upper
    return False


def _is_full_fill(direction: str, lower: float, upper: float, candle: Candle) -> bool:
    """Inclusive far-boundary reach via wick extreme."""
    if direction == "bullish":
        return float(candle.low) <= lower
    if direction == "bearish":
        return float(candle.high) >= upper
    return False


class MitigationBlockDetector(PriceActionDetector):
    """
    Mitigation lifecycle events composed from confirmed Order Blocks and
    Breaker Blocks.

    Configuration is forwarded to the internal OrderBlockDetector /
    BreakerBlockDetector (min_body_ratio, left_strength, right_strength,
    tie_policy). No additional thresholds in V1.
    """

    def __init__(
        self,
        *,
        min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "mitigation_block",
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
        self._breaker = BreakerBlockDetector(
            min_body_ratio=min_body_ratio,
            left_strength=left_strength,
            right_strength=right_strength,
            tie_policy=tie_policy,
            detector_id="breaker_block",
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
        # Need at least one completed bar after a Breaker Block can form.
        return self._breaker.min_history + 1

    def config_dict(self) -> dict[str, Any]:
        return self._ob.config_dict()

    def detect(
        self,
        candles: list[Candle],
        *,
        history_complete: bool = False,
    ) -> list[Observation]:
        """
        Incremental path: full reconstruction, then emit only events whose
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
            logger.debug("Mixed symbol/timeframe; skipping mitigation-block detect")
            return []

        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            logger.debug(
                "Unknown timeframe %r; skipping mitigation-block detect", timeframe
            )
            return []

        order_blocks = self._ob.detect_all_historical(
            closed, history_complete=history_complete
        )
        breakers = self._breaker.detect_all_historical(
            closed, history_complete=history_complete
        )

        sources: list[Observation] = []
        for obs in order_blocks:
            if obs.type == "order_block" and obs.is_confirmed():
                sources.append(obs)
        for obs in breakers:
            if obs.type == "breaker_block" and obs.is_confirmed():
                sources.append(obs)

        if not sources:
            return []

        epoch_to_idx: dict[int, int] = {c.epoch: i for i, c in enumerate(closed)}

        out: list[Observation] = []

        for source in sources:
            direction = source.direction or ""
            if direction not in ("bullish", "bearish"):
                continue

            payload = source.payload or {}
            try:
                lower = float(payload["lower_bound"])
                upper = float(payload["upper_bound"])
            except (KeyError, TypeError, ValueError):
                logger.debug(
                    "Skipping source %s: missing or invalid bounds", source.id
                )
                continue
            if not (upper > lower):
                logger.debug(
                    "Skipping source %s: malformed bounds L=%s U=%s",
                    source.id,
                    lower,
                    upper,
                )
                continue

            conf = source.effective_confirmed_at()
            if conf is None:
                continue
            conf_idx = epoch_to_idx.get(conf)
            if conf_idx is None:
                # Confirmation candle not in series — cannot evaluate safely.
                continue

            # Per-source lifecycle state (reconstructed; deterministic).
            stages_done: set[str] = set()
            closed_lifecycle = False

            # Interaction candles strictly after source confirmation.
            for j in range(conf_idx + 1, len(closed)):
                if closed_lifecycle:
                    break

                cur = closed[j]
                prev = closed[j - 1]

                # Exact continuity with immediate predecessor required.
                if cur.epoch - prev.epoch != step:
                    continue

                if not _ohlc_sane(cur):
                    logger.debug(
                        "Malformed OHLC at epoch=%s; skip mitigation eval",
                        cur.epoch,
                    )
                    continue

                intersects = _intersects_closed_interval(cur, lower, upper)
                partial = _is_partial_fill(direction, lower, upper, cur)
                full = _is_full_fill(direction, lower, upper, cur)

                # Same-candle order: touch → partial_fill → full_fill.
                # Emit any missing earlier stages when jumping ahead.

                if "touch" not in stages_done and intersects:
                    out.append(
                        self._make_event(
                            source=source,
                            candle=cur,
                            variant="touch",
                            direction=direction,
                            lower=lower,
                            upper=upper,
                        )
                    )
                    stages_done.add("touch")

                if "partial_fill" not in stages_done and (
                    partial or full
                ):
                    # Partial is emitted when interior is penetrated, or when
                    # full_fill is reached on a candle that skipped partial
                    # (contract: emit missing earlier stages in order).
                    if intersects or partial or full:
                        out.append(
                            self._make_event(
                                source=source,
                                candle=cur,
                                variant="partial_fill",
                                direction=direction,
                                lower=lower,
                                upper=upper,
                            )
                        )
                        stages_done.add("partial_fill")

                if "full_fill" not in stages_done and full:
                    # Ensure touch/partial already emitted if somehow missed
                    # (e.g. first contact is a full_fill candle).
                    if "touch" not in stages_done:
                        out.append(
                            self._make_event(
                                source=source,
                                candle=cur,
                                variant="touch",
                                direction=direction,
                                lower=lower,
                                upper=upper,
                            )
                        )
                        stages_done.add("touch")
                    if "partial_fill" not in stages_done:
                        out.append(
                            self._make_event(
                                source=source,
                                candle=cur,
                                variant="partial_fill",
                                direction=direction,
                                lower=lower,
                                upper=upper,
                            )
                        )
                        stages_done.add("partial_fill")
                    out.append(
                        self._make_event(
                            source=source,
                            candle=cur,
                            variant="full_fill",
                            direction=direction,
                            lower=lower,
                            upper=upper,
                        )
                    )
                    stages_done.add("full_fill")
                    closed_lifecycle = True

        return out

    def _make_event(
        self,
        *,
        source: Observation,
        candle: Candle,
        variant: str,
        direction: str,
        lower: float,
        upper: float,
    ) -> Observation:
        event_time = candle.epoch
        confirmed_at = candle.epoch
        candle_epoch = candle.epoch
        source_id = source.id

        source_type = source.type
        payload: dict[str, Any] = {
            "source_id": source_id,
            "source_type": source_type,
            "lower_bound": lower,
            "upper_bound": upper,
            "size": upper - lower,
            "confirming_candle_epoch": candle.epoch,
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "source_direction": direction,
            "lifecycle_stage": variant,
        }
        # Preserve useful source provenance when present.
        src_payload = source.payload or {}
        if "source_candle_epoch" in src_payload:
            payload["source_zone_candle_epoch"] = src_payload["source_candle_epoch"]
        if source.candle_epoch is not None:
            payload["source_confirmed_candle_epoch"] = source.candle_epoch
        if source_type == "breaker_block" and "source_ob_id" in src_payload:
            payload["source_ob_id"] = src_payload["source_ob_id"]
        if source_type == "order_block" and "source_bos_id" in src_payload:
            payload["source_bos_id"] = src_payload["source_bos_id"]

        oid = _stable_mitigation_id(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            candle_epoch=candle_epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            direction=direction,
            variant=variant,
            source_id=source_id,
        )

        return Observation.create(
            type="mitigation_block",
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            candle_epoch=candle_epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            developing=False,
            direction=direction,
            variant=variant,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=(source_id,),
            lifecycle_state=variant,
            payload=payload,
            metadata=payload,
            id=oid,
        )
