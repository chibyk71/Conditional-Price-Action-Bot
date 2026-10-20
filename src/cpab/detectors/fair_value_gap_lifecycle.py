"""
Phase 4B — Fair Value Gap lifecycle tracking.

Composes Phase 4A FairValueGapDetector for formation; does not reimplement
geometric rules. Emits immutable lifecycle observations for first touch,
partial mitigation, full mitigation, and invalidation.

Locked rules (completed candles only, after the formation candle):

  Bullish FVG bounds [L, U]:
    near = U, far = L
    penetration depth = U - min(U, candle.low), capped at gap size
    fill_ratio = depth / (U - L)
    full mitigation: candle.low <= L
    invalidation: candle.close < L  (equality does not invalidate)

  Bearish FVG bounds [L, U]:
    near = L, far = U
    penetration depth = max(0, candle.high - L), capped at gap size
    fill_ratio = depth / (U - L)
    full mitigation: candle.high >= U
    invalidation: candle.close > U  (equality does not invalidate)

  First touch: first eligible candle whose range intersects [L, U]
  (boundary / wick contact counts).

  Partial mitigation: whenever maximum penetration so far increases, and the
  candle does not itself reach full mitigation. Distinct increases yield
  distinct observations.

  Same-candle order: first_touch, partial_mitigation, full_mitigation,
  invalidation. Full mitigation and invalidation are independent; both may
  emit on one candle. Lifecycle closes after full mitigation or invalidation
  (both allowed on the closing candle).

Causality / continuity:
  - Completed candles only; never the formation candle; never forming bars.
  - Exact timeframe step between consecutive candles required to evaluate a
    candle. Discontinuous candles are skipped; state and prior events are
    preserved; evaluation resumes on later continuous pairs.
  - No interpolation or bridging of missing bars.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.fair_value_gap import FairValueGapDetector, _ohlc_sane
from cpab.detectors.swing import timeframe_seconds
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.fair_value_gap_lifecycle")


def _stable_lifecycle_id(
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
) -> str:
    """
    Deterministic identity including source FVG so overlapping gaps remain
    independent and repeated reconstruction yields identical IDs.
    """
    parts = [
        "fair_value_gap_lifecycle",
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
    ]
    return "|".join(parts)


def _intersects_closed_interval(candle: Candle, lower: float, upper: float) -> bool:
    """True if candle range intersects closed interval [lower, upper]."""
    return candle.high >= lower and candle.low <= upper


def _penetration_depth(
    direction: str, lower: float, upper: float, candle: Candle
) -> float:
    """
    Penetration into the gap from the near boundary, capped at gap size.
    Non-negative.
    """
    gap = upper - lower
    if gap <= 0:
        return 0.0
    if direction == "bullish":
        # Near = U; depth = U - min(U, candle.low)
        depth = upper - min(upper, float(candle.low))
    else:
        # Near = L; depth = max(0, candle.high - L)
        depth = max(0.0, float(candle.high) - lower)
    if depth > gap:
        depth = gap
    if depth < 0:
        depth = 0.0
    return float(depth)


def _is_full_mitigation(direction: str, lower: float, upper: float, candle: Candle) -> bool:
    if direction == "bullish":
        return float(candle.low) <= lower
    return float(candle.high) >= upper


def _is_invalidation(direction: str, lower: float, upper: float, candle: Candle) -> bool:
    if direction == "bullish":
        return float(candle.close) < lower
    return float(candle.close) > upper


class FairValueGapLifecycleDetector(PriceActionDetector):
    """
    FVG lifecycle events composed from Phase 4A FairValueGapDetector formations.

    Configuration is empty (no filters). detector_config_id remains deterministic.
    """

    def __init__(
        self,
        *,
        detector_id: str = "fair_value_gap_lifecycle",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._detector_id = detector_id
        self._fvg = FairValueGapDetector(detector_id="fair_value_gap")
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
        # Formation needs 3; at least one post-formation candle for lifecycle.
        return 4

    def config_dict(self) -> dict[str, Any]:
        return {}

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: full reconstruction, then emit only events whose
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

        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe; skipping FVG lifecycle detect")
            return []

        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            logger.debug("Unknown timeframe %r; rejecting FVG lifecycle", timeframe)
            return []

        # Formations from Phase 4A (immutable, independent of lifecycle).
        fvgs = self._fvg.detect_all_historical(closed)
        if not fvgs:
            return []

        epoch_to_idx: dict[int, int] = {c.epoch: i for i, c in enumerate(closed)}

        out: list[Observation] = []

        for fvg in fvgs:
            form_epoch = fvg.candle_epoch
            form_idx = epoch_to_idx.get(form_epoch)
            if form_idx is None:
                continue

            direction = fvg.direction or fvg.variant or ""
            if direction not in ("bullish", "bearish"):
                continue

            payload = fvg.payload or {}
            try:
                lower = float(payload["lower_bound"])
                upper = float(payload["upper_bound"])
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping FVG %s: missing bounds", fvg.id)
                continue
            if not (upper > lower):
                continue
            gap_size = upper - lower

            max_pen = 0.0
            first_touch_done = False
            closed_lifecycle = False

            for j in range(form_idx + 1, len(closed)):
                if closed_lifecycle:
                    break

                cur = closed[j]
                prev = closed[j - 1]

                # Exact continuity with immediate predecessor required to evaluate.
                if cur.epoch - prev.epoch != step:
                    continue

                if not _ohlc_sane(cur):
                    logger.debug(
                        "Malformed OHLC at epoch=%s; skip lifecycle eval", cur.epoch
                    )
                    continue

                # --- evaluate this candle ---
                intersects = _intersects_closed_interval(cur, lower, upper)
                depth = _penetration_depth(direction, lower, upper, cur)
                full_mit = _is_full_mitigation(direction, lower, upper, cur)
                inv = _is_invalidation(direction, lower, upper, cur)

                # Same-candle event order: first_touch, partial, full, invalidation.
                # 1. First touch
                if not first_touch_done and intersects:
                    out.append(
                        self._make_event(
                            fvg=fvg,
                            candle=cur,
                            variant="first_touch",
                            direction=direction,
                            lower=lower,
                            upper=upper,
                            penetration_depth=depth,
                            fill_ratio=(depth / gap_size) if gap_size > 0 else 0.0,
                        )
                    )
                    first_touch_done = True

                # 2. Partial mitigation — only if penetration increases and
                #    this candle does not reach full mitigation.
                if (
                    not closed_lifecycle
                    and depth > max_pen
                    and not full_mit
                ):
                    max_pen = depth
                    fill = (depth / gap_size) if gap_size > 0 else 0.0
                    out.append(
                        self._make_event(
                            fvg=fvg,
                            candle=cur,
                            variant="partial_mitigation",
                            direction=direction,
                            lower=lower,
                            upper=upper,
                            penetration_depth=depth,
                            fill_ratio=fill,
                        )
                    )
                elif depth > max_pen:
                    # Full mit on this candle: advance max_pen but do not emit partial.
                    max_pen = depth

                # 3. Full mitigation
                if full_mit:
                    fill = 1.0
                    out.append(
                        self._make_event(
                            fvg=fvg,
                            candle=cur,
                            variant="full_mitigation",
                            direction=direction,
                            lower=lower,
                            upper=upper,
                            penetration_depth=min(depth, gap_size) if gap_size > 0 else depth,
                            fill_ratio=fill,
                        )
                    )
                    closed_lifecycle = True

                # 4. Invalidation (independent; may co-occur with full mit)
                if inv:
                    out.append(
                        self._make_event(
                            fvg=fvg,
                            candle=cur,
                            variant="invalidation",
                            direction=direction,
                            lower=lower,
                            upper=upper,
                            penetration_depth=depth,
                            fill_ratio=(depth / gap_size) if gap_size > 0 else 0.0,
                        )
                    )
                    closed_lifecycle = True

        return out

    def _make_event(
        self,
        *,
        fvg: Observation,
        candle: Candle,
        variant: str,
        direction: str,
        lower: float,
        upper: float,
        penetration_depth: float,
        fill_ratio: float,
    ) -> Observation:
        event_time = candle.epoch
        confirmed_at = candle.epoch
        candle_epoch = candle.epoch
        source_id = fvg.id

        payload: dict[str, Any] = {
            "source_fvg_id": source_id,
            "lower_bound": lower,
            "upper_bound": upper,
            "size": upper - lower,
            "confirming_candle_epoch": candle.epoch,
            "penetration_depth": float(penetration_depth),
            "fill_ratio": float(fill_ratio),
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "source_fvg_candle_epoch": fvg.candle_epoch,
            "source_fvg_direction": direction,
        }

        oid = _stable_lifecycle_id(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            candle_epoch=candle_epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            direction=direction,
            variant=variant,
            source_fvg_id=source_id,
        )

        return Observation.create(
            type="fair_value_gap_lifecycle",
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
            payload=payload,
            metadata=payload,
            id=oid,
        )
