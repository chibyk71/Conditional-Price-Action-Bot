"""
Phase 2D — Break of Structure (BOS) and Change of Character (CHoCH).

Uses confirmed swing_high / swing_low observations (from SwingPivotDetector) as
breakable structural levels. Equal-high/equal-low levels are not breakable in V1.

Break confirmation (completed candles only):
  Bullish: previous_close <= swing_high_price AND current_close > swing_high_price
  Bearish: previous_close >= swing_low_price  AND current_close < swing_low_price

Equality is not a break. Wick-only penetration is not a break. Remaining beyond
a level does not generate repeated events. A newly confirmed level that is already
beyond the current close does not trigger a false break (requires a valid prior
close that was on/at the level side).

Structural state (independent per symbol / timeframe / effective config):
  Known-complete history (history_complete=True): start neutral; first break is BOS.
  Incomplete / truncated history (history_complete=False, default): start unknown;
  the first observed break still records the objective crossing and establishes
  in-window direction, but prior_structural_state is "unknown" so callers do not
  treat the label as absolute market-structure inception.
  Same-direction break → BOS; opposite-direction break → CHoCH.

Observation:
  type = "structure_break"
  variant = "bos" | "choch"
  direction = "bullish" | "bearish"

Event ordering when multiple levels cross on one candle:
  1. All bullish breaks first
  2. All bearish breaks second
  Within each direction: ordered by (level confirmed_at, level event_time, swing id)

Gap handling (Phase 2A convention):
  A candle is continuous with its predecessor only when open epochs differ by
  exactly the expected timeframe duration. On discontinuity, suppress break
  evaluation on that candle and retain structural state.

Historical reconstruction is pure: no persistent store. Incremental detect()
emits only events whose confirmed_at equals the latest closed bar's open epoch.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional, Union

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.swing import (
    SwingPivotDetector,
    _validate_strength,
    _validate_tie_policy,
    timeframe_seconds,
)
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.structure_break")

StructuralState = Literal["neutral", "bullish", "bearish"]
PriorState = Union[StructuralState, Literal["unknown"]]


def _stable_break_id(
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
    broken_swing_id: str,
    history_complete: bool,
) -> str:
    """
    Deterministic identity including the broken swing and history-completeness
    mode so:

    * multiple breaks on the same candle (different levels) remain distinct;
    * complete vs truncated reconstructions of the same objective crossing do
      not collide in ObservationStore (payloads differ on prior_structural_state).
    """
    hc = "hc1" if history_complete else "hc0"
    parts = [
        "structure_break",
        symbol,
        timeframe,
        str(candle_epoch),
        f"e{event_time}",
        f"c{confirmed_at}",
        detector_id,
        detector_config_id,
        direction,
        variant,
        broken_swing_id,
        hc,
    ]
    return "|".join(parts)


class StructureBreakDetector(PriceActionDetector):
    """
    BOS / CHoCH detector built on SwingPivotDetector.

    Configuration is immutable after construction. Invalid swing parameters
    raise ValueError (no silent defaults or coercion for invalid values).

    History completeness
    --------------------
    Reconstruction never invents structural events outside the supplied closed
    series. Whether the *starting* structural state is known is controlled by
    the keyword-only flag ``history_complete`` on detect / detect_all_historical:

    * ``history_complete=True`` — caller asserts the series covers structural
      history from a known neutral inception (or an equivalent full window).
      Starting state is neutral; the first break is BOS under the locked table.
    * ``history_complete=False`` (default) — starting state is **unknown**.
      Objective crossings still emit, but ``prior_structural_state`` is
      ``"unknown"`` until the first in-window break establishes direction.
      Subsequent same/opposite transitions within the window follow the locked
      table. Callers must not treat a first-break BOS label under unknown prior
      as proof that no earlier structural break existed outside the window.
    """

    def __init__(
        self,
        *,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "structure_break",
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
        Minimum closed candles for any break to become possible.

        Need at least one confirmed swing (left + 1 + right) plus one additional
        closed bar after the swing's confirmation bar so a prior/current close
        pair exists for crossing evaluation.
        """
        return self._left + 1 + self._right + 1

    def config_dict(self) -> dict[str, Any]:
        return {
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
        Incremental path: reconstruct all breaks from closed history, then emit
        only those whose confirmed_at equals the latest closed bar's open epoch.

        ``history_complete`` — see class docstring. Default False avoids treating
        a truncated rolling window as structural inception.
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
            logger.debug("Mixed symbol/timeframe; skipping structure-break detect")
            return []

        step = timeframe_seconds(timeframe)
        swing_obs = self._swing.detect_all_historical(closed)

        highs = [o for o in swing_obs if o.type == "swing_high"]
        lows = [o for o in swing_obs if o.type == "swing_low"]

        def level_sort_key(o: Observation) -> tuple:
            return (
                o.confirmed_at if o.confirmed_at is not None else 0,
                o.event_time if o.event_time is not None else o.candle_epoch,
                o.id,
            )

        highs_sorted = sorted(highs, key=level_sort_key)
        lows_sorted = sorted(lows, key=level_sort_key)

        broken: set[str] = set()
        # Known-complete → neutral; truncated → unknown (do not invent inception).
        state: Optional[StructuralState] = "neutral" if history_complete else None
        out: list[Observation] = []

        for i in range(1, len(closed)):
            prev = closed[i - 1]
            cur = closed[i]

            continuous = True
            if step is None or step <= 0:
                continuous = False
            elif cur.epoch - prev.epoch != step:
                continuous = False

            if not continuous:
                continue

            eligible_highs = [
                s
                for s in highs_sorted
                if s.id not in broken
                and s.confirmed_at is not None
                and s.confirmed_at <= cur.epoch
            ]
            eligible_lows = [
                s
                for s in lows_sorted
                if s.id not in broken
                and s.confirmed_at is not None
                and s.confirmed_at <= cur.epoch
            ]

            bullish_swings: list[tuple[Observation, float]] = []
            bearish_swings: list[tuple[Observation, float]] = []

            for swing in eligible_highs:
                level_price = float(swing.payload.get("price", 0.0))
                if prev.close <= level_price and cur.close > level_price:
                    bullish_swings.append((swing, level_price))

            for swing in eligible_lows:
                level_price = float(swing.payload.get("price", 0.0))
                if prev.close >= level_price and cur.close < level_price:
                    bearish_swings.append((swing, level_price))

            for swing, level_price in bullish_swings:
                variant, new_state, prior_label = self._classify_and_transition(
                    state, "bullish"
                )
                obs = self._make_observation(
                    direction="bullish",
                    swing=swing,
                    level_price=level_price,
                    candle=cur,
                    prev_close=prev.close,
                    prior_label=prior_label,
                    variant=variant,
                    resulting_state=new_state,
                    history_complete=history_complete,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                out.append(obs)
                broken.add(swing.id)
                state = new_state

            for swing, level_price in bearish_swings:
                variant, new_state, prior_label = self._classify_and_transition(
                    state, "bearish"
                )
                obs = self._make_observation(
                    direction="bearish",
                    swing=swing,
                    level_price=level_price,
                    candle=cur,
                    prev_close=prev.close,
                    prior_label=prior_label,
                    variant=variant,
                    resulting_state=new_state,
                    history_complete=history_complete,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                out.append(obs)
                broken.add(swing.id)
                state = new_state

        return out

    @staticmethod
    def _classify_and_transition(
        prior: Optional[StructuralState], break_direction: str
    ) -> tuple[str, StructuralState, PriorState]:
        """
        Map prior state + break direction → (variant, resulting state, prior label).

        prior is None when history is incomplete and no in-window break has
        established direction yet. The crossing still establishes direction;
        prior_label is "unknown" and variant is "bos" as in-window direction
        establishment (not a claim of absolute structural inception).
        """
        if prior is None:
            return "bos", break_direction, "unknown"  # type: ignore[return-value]
        if prior == "neutral":
            return "bos", break_direction, "neutral"  # type: ignore[return-value]
        if prior == break_direction:
            return "bos", prior, prior
        return "choch", break_direction, prior  # type: ignore[return-value]

    def _make_observation(
        self,
        *,
        direction: str,
        swing: Observation,
        level_price: float,
        candle: Candle,
        prev_close: float,
        prior_label: PriorState,
        variant: str,
        resulting_state: StructuralState,
        history_complete: bool,
        symbol: str,
        timeframe: str,
    ) -> Observation:
        event_time = candle.epoch
        confirmed_at = candle.epoch
        broken_swing_id = swing.id

        payload: dict[str, Any] = {
            "broken_level_price": level_price,
            "break_direction": direction,
            "event_variant": variant,
            "source_swing_id": broken_swing_id,
            "source_swing_event_time": swing.event_time
            if swing.event_time is not None
            else swing.candle_epoch,
            "source_swing_confirmed_at": swing.confirmed_at,
            "previous_close": prev_close,
            "breakout_close": candle.close,
            "prior_structural_state": prior_label,
            "resulting_structural_state": resulting_state,
            "history_complete": history_complete,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

        oid = _stable_break_id(
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=candle.epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            direction=direction,
            variant=variant,
            broken_swing_id=broken_swing_id,
            history_complete=history_complete,
        )

        return Observation.create(
            type="structure_break",
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
            reference_ids=(broken_swing_id,),
            payload=payload,
            metadata=payload,
            id=oid,
        )
