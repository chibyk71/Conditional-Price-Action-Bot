"""
Phase 4A — Causal, closed-candle-only Fair Value Gap (FVG) formation detection.

Locked geometric contract (three consecutive completed candles, same symbol/TF):

  Bullish FVG:
    candle_1.high < candle_3.low
    lower_bound = candle_1.high
    upper_bound = candle_3.low
    direction = variant = "bullish"

  Bearish FVG:
    candle_1.low > candle_3.high
    lower_bound = candle_3.high
    upper_bound = candle_1.low
    direction = variant = "bearish"

  Require upper_bound > lower_bound (equal boundaries do not qualify).

No middle-candle colour filter, minimum gap size, ATR filter, or merging of
consecutive FVGs. Touch / mitigation / invalidation lifecycle is out of scope
for this phase; see docs/PHASE4A_FAIR_VALUE_GAP.md for intended future meanings.

Causality:
  - Only completed candles.
  - Exactly three consecutive bars with the expected timeframe step.
  - Unknown timeframe durations are rejected (no invented step).
  - Mixed symbols/timeframes or malformed OHLC suppress formation.
  - Incremental detect() emits only formations confirmed on the latest closed bar.
  - Historical reconstruction must not leak future information into earlier obs.

Observation contract:
  type = "fair_value_gap"
  event_time = confirmed_at = candle_epoch = candle_3.open epoch
  developing = false
  payload: lower_bound, upper_bound, size, candle_1_epoch, candle_2_epoch, candle_3_epoch
  reference_ids = []
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.swing import timeframe_seconds
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.fair_value_gap")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def _ohlc_sane(c: Candle) -> bool:
    """Reject non-finite or inverted OHLC using the same spirit as market validation."""
    if not all(_finite(v) for v in (c.open, c.high, c.low, c.close)):
        return False
    if c.high < c.low:
        return False
    if c.high < c.open or c.high < c.close:
        return False
    if c.low > c.open or c.low > c.close:
        return False
    return True


class FairValueGapDetector(PriceActionDetector):
    """
    Three-candle Fair Value Gap formation detector.

    Configuration is empty for Phase 4A (no filters). detector_config_id is still
    deterministic so the observation identity remains stable and distinct from
    any future configured variants.
    """

    def __init__(
        self,
        *,
        detector_id: str = "fair_value_gap",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._detector_id = detector_id
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
        return 3

    def config_dict(self) -> dict[str, Any]:
        return {}

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: reconstruct all confirmed FVGs from closed history,
        then emit only those whose confirmed_at equals the latest closed bar's
        open epoch (formation completed on candle 3).
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
        if len(closed) < 3:
            return []

        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe; skipping FVG detect")
            return []

        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            logger.debug("Unknown timeframe %r; rejecting FVG windows", timeframe)
            return []

        out: list[Observation] = []
        # Sliding window of three consecutive closed candles
        for i in range(len(closed) - 2):
            c1, c2, c3 = closed[i], closed[i + 1], closed[i + 2]
            if not self._triple_consecutive(c1, c2, c3, step=step):
                continue
            if not (_ohlc_sane(c1) and _ohlc_sane(c2) and _ohlc_sane(c3)):
                logger.debug(
                    "Malformed OHLC in FVG window epochs=%s,%s,%s; skip",
                    c1.epoch,
                    c2.epoch,
                    c3.epoch,
                )
                continue

            obs = self._try_fvg(c1, c2, c3)
            if obs is not None:
                out.append(obs)

        return out

    @staticmethod
    def _triple_consecutive(
        c1: Candle, c2: Candle, c3: Candle, *, step: int
    ) -> bool:
        """Require exact timeframe spacing; no gap bridging."""
        if c2.epoch - c1.epoch != step:
            return False
        if c3.epoch - c2.epoch != step:
            return False
        if c1.epoch >= c2.epoch or c2.epoch >= c3.epoch:
            return False
        return True

    def _try_fvg(
        self, c1: Candle, c2: Candle, c3: Candle
    ) -> Optional[Observation]:
        """
        Apply locked geometric rules. Middle candle colour is ignored.
        """
        # Bullish: gap above candle_1 high into candle_3 low
        if c1.high < c3.low:
            lower = float(c1.high)
            upper = float(c3.low)
            if not (upper > lower):
                return None
            direction = "bullish"
            variant = "bullish"
        # Bearish: gap below candle_1 low into candle_3 high
        elif c1.low > c3.high:
            lower = float(c3.high)
            upper = float(c1.low)
            if not (upper > lower):
                return None
            direction = "bearish"
            variant = "bearish"
        else:
            return None

        size = upper - lower
        event_time = c3.epoch
        confirmed_at = c3.epoch
        candle_epoch = c3.epoch

        payload: dict[str, Any] = {
            "lower_bound": lower,
            "upper_bound": upper,
            "size": size,
            "candle_1_epoch": c1.epoch,
            "candle_2_epoch": c2.epoch,
            "candle_3_epoch": c3.epoch,
        }

        return Observation.create(
            type="fair_value_gap",
            symbol=c3.symbol,
            timeframe=c3.timeframe,
            candle_epoch=candle_epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            developing=False,
            direction=direction,
            variant=variant,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=[],
            payload=payload,
            metadata=payload,
        )
