"""
Phase 6C — Pullback Behavior detection.

Tracks the lifecycle of pullbacks following confirmed market-structure moves
(structure-based) or confirmed directional impulse candles (impulse-based).

Capability:
  Detector ID: pullback
  Variants:
    - structure_based
    - impulse_based

Variant A: structure_based
  Source:
    - Established structural move derived from existing confirmed
      market-structure observations (HH for bullish, LL for bearish).
    - Uses only source observations confirmed and available before evaluation.
  Start:
    - Starts when a completed candle closes strictly beyond a previously confirmed
      swing level in the direction opposite the originating structural move.
      (Bullish: close < swing_high; Bearish: close > swing_low).
    - The swing must have been confirmed and available before the candidate candle's close.
    - Equality does not qualify.
  Continuation:
    - Active until termination. No duplicate continuation events on every candle.
  Trend-resumption termination:
    - Closes strictly beyond the most recent confirmed swing level in the move direction.
      (Bullish: close > most_recent_swing_high; Bearish: close < most_recent_swing_low).
  Structural-failure termination:
    - Closes strictly beyond the latest confirmed protective swing:
      Bullish: most recent confirmed higher low (close < protective_hl).
      Bearish: most recent confirmed lower high (close > protective_lh).
    - If protective swing cannot be identified unambiguously, do not invent one
      and do not emit structural-failure termination.
  Ambiguous termination:
    - If both qualify on the same candle, structural failure takes precedence.

Variant B: impulse_based
  Source:
    - Previously confirmed bullish_impulse or bearish_impulse observation.
  Start:
    - Starts when a subsequent completed candle closes against the impulse direction.
      (Bullish impulse: close < open; Bearish impulse: close > open).
    - Impulse must be confirmed and available before candidate evaluation.
    - No retracement-depth threshold in V1.
  Continuation:
    - Active until termination. No duplicate continuation events on every candle.
  Trend-resumption termination:
    - Bullish: close strictly beyond impulse candle's high (close > high).
    - Bearish: close strictly beyond impulse candle's low (close < low).
  Invalidation termination:
    - Bullish: close strictly below impulse candle's low (close < low).
    - Bearish: close strictly above impulse candle's high (close > high).
  Ambiguous termination:
    - Invalidation takes precedence over trend resumption.

Deduplication & Provenance:
  - One logical lifecycle per source and variant.
  - Terminated source cannot start another lifecycle in V1.
  - Distinct sources may have concurrent pullbacks.
  - Repeated processing is idempotent.
  - Provenance records only observations actually used.

Data continuity (gaps / invalid OHLC) — V1, aligned with zone_retest:
  - Gaps must never be bridged. Continuous only when successive open epochs
    differ by exactly the timeframe step (same as Phase 2A / 2D / 3B / 3C).
  - An invalid OHLC candle is treated as a hard segment boundary.
  - When continuity breaks while a source is pending or active, that source's
    lifecycle is abandoned for V1: internal tracking ends and the source cannot
    start or continue a pullback afterward.
  - No separate terminal observation is emitted for data-continuity abandonment
    (same convention as zone_retest sequence invalidation: "No separate
    invalidation observation is emitted in V1"). Price-based terminals remain
    the only emitted terminations: trend_resumption, structural_failure, and
    impulse invalidation.
  - Downstream consumers that observe a start (lifecycle_state=active) and need
    to know whether the pullback is still trackable must not assume perpetual
    activity: either require a subsequent price-based terminal event, or treat
    a gap / invalid candle in the closed series after the start as ending the
    active track (no emission required from this detector).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.behavior_volatility import ImpulseDetector
from cpab.detectors.market_structure import MarketStructureDetector
from cpab.detectors.order_block import _ohlc_sane
from cpab.detectors.swing import (
    SwingPivotDetector,
    _validate_strength,
    _validate_tie_policy,
    timeframe_seconds,
)
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.pullback")

ALLOWED_VARIANTS = frozenset({"structure_based", "impulse_based", "all"})


def _deterministic_config_id(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"cfg:{digest}"


def _stable_pullback_id(
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
    lifecycle_event: str,
) -> str:
    parts = [
        "pullback",
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
        lifecycle_event,
    ]
    return "|".join(parts)


class PullbackDetector(PriceActionDetector):
    """
    CPAB Phase 6C Pullback Detector supporting structure_based and impulse_based variants.
    """

    def __init__(
        self,
        *,
        variant: str = "all",
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "pullback",
        detector_config_id: Optional[str] = None,
    ) -> None:
        if variant not in ALLOWED_VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(ALLOWED_VARIANTS)}, got {variant!r}"
            )
        self._variant = variant
        self._left = _validate_strength("left_strength", left_strength)
        self._right = _validate_strength("right_strength", right_strength)
        self._tie_policy = _validate_tie_policy(tie_policy)
        self._detector_id = detector_id

        self._config: dict[str, Any] = {
            "variant": self._variant,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        if detector_config_id is not None:
            self._detector_config_id = detector_config_id
        else:
            self._detector_config_id = _deterministic_config_id(self._config)

        # Logical sub-variant config identities (shared swing params; variant-specific).
        # Observations embed the effective sub-variant config id so that variant="all"
        # produces the same observation identities as dedicated impulse_based /
        # structure_based detectors. Detector-level detector_config_id remains
        # distinct when variant="all".
        base_params = {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        self._variant_config_ids: dict[str, str] = {
            "structure_based": _deterministic_config_id(
                {**base_params, "variant": "structure_based"}
            ),
            "impulse_based": _deterministic_config_id(
                {**base_params, "variant": "impulse_based"}
            ),
        }
        # When constructed for a single logical variant, keep observation config
        # identity aligned with the detector's own config id (including any
        # explicit detector_config_id override).
        if self._variant in ("structure_based", "impulse_based"):
            self._variant_config_ids[self._variant] = self._detector_config_id

        # Internal composed detectors for reconstruction
        self._swing = SwingPivotDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
            detector_id="swing_pivot",
        )
        self._structure = MarketStructureDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
            detector_id="market_structure",
        )
        self._impulse = ImpulseDetector(detector_id="impulse")

    @property
    def name(self) -> str:
        return self._detector_id

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def variant(self) -> str:
        return self._variant

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
    def min_history(self) -> int:
        struct_min = self._left + 2 * self._right + 3
        impulse_min = 17
        if self._variant == "impulse_based":
            return impulse_min
        if self._variant == "structure_based":
            return struct_min
        return min(struct_min, impulse_min)

    def config_dict(self) -> dict[str, Any]:
        return dict(self._config)

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """Incremental detect: emits only observations confirmed at latest candle epoch."""
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        all_obs = self._reconstruct(closed)
        latest_epoch = closed[-1].epoch
        return [o for o in all_obs if o.confirmed_at == latest_epoch]

    def detect_all_historical(self, candles: list[Candle]) -> list[Observation]:
        """Full historical reconstruction for tests and backfill."""
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
            logger.debug("Mixed symbol/timeframe; skipping pullback detection")
            return []

        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            logger.debug("Unknown timeframe %r; rejecting pullback", timeframe)
            return []

        out: list[Observation] = []
        if self._variant in ("structure_based", "all"):
            out.extend(self._reconstruct_structure_based(closed, step=step))
        if self._variant in ("impulse_based", "all"):
            out.extend(self._reconstruct_impulse_based(closed, step=step))

        # Order chronologically and deterministically
        out.sort(
            key=lambda o: (
                o.confirmed_at if o.confirmed_at is not None else 0,
                o.event_time if o.event_time is not None else o.candle_epoch,
                0 if (o.lifecycle_state == "active") else 1,
                o.id,
            )
        )
        return out

    def _config_id_for_variant(self, variant: str) -> str:
        """Return the deterministic config id for a logical sub-variant.

        Ensures observations from variant='all' share identity with those from
        a dedicated detector for the same logical configuration.
        """
        return self._variant_config_ids.get(variant, self._detector_config_id)

    def _make_observation(
        self,
        *,
        candle: Candle,
        variant: str,
        direction: str,
        source_id: str,
        lifecycle_event: str,
        lifecycle_state: str,
        reference_ids: tuple[str, ...],
        payload: dict[str, Any],
    ) -> Observation:
        event_time = candle.epoch
        confirmed_at = candle.epoch
        candle_epoch = candle.epoch
        # Use effective sub-variant config identity so all/dedicated match.
        cfg_id = self._config_id_for_variant(variant)

        oid = _stable_pullback_id(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            candle_epoch=candle_epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            detector_id=self._detector_id,
            detector_config_id=cfg_id,
            direction=direction,
            variant=variant,
            source_id=source_id,
            lifecycle_event=lifecycle_event,
        )

        return Observation.create(
            type="pullback",
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            candle_epoch=candle_epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            developing=False,
            direction=direction,
            variant=variant,
            detector_id=self._detector_id,
            detector_config_id=cfg_id,
            reference_ids=reference_ids,
            lifecycle_state=lifecycle_state,
            payload=payload,
            metadata=payload,
            id=oid,
        )

    # -------------------------------------------------------------------------
    # Variant B: impulse_based
    # -------------------------------------------------------------------------

    def _reconstruct_impulse_based(
        self, closed: list[Candle], *, step: int
    ) -> list[Observation]:
        impulses = self._impulse.detect_all_historical(closed)
        if not impulses:
            return []

        epoch_to_candle: dict[int, Candle] = {c.epoch: c for c in closed}
        epoch_to_idx: dict[int, int] = {c.epoch: i for i, c in enumerate(closed)}

        out: list[Observation] = []

        for impulse in impulses:
            impulse_id = impulse.id
            direction = impulse.direction
            if direction not in ("bullish", "bearish"):
                continue

            form_candle = epoch_to_candle.get(impulse.candle_epoch)
            form_idx = epoch_to_idx.get(impulse.candle_epoch)
            if form_candle is None or form_idx is None:
                continue

            impulse_high = float(form_candle.high)
            impulse_low = float(form_candle.low)

            lifecycle_status: str = "pending"  # "pending" -> "active" -> "terminated"

            for j in range(form_idx + 1, len(closed)):
                if lifecycle_status == "terminated":
                    break

                cur = closed[j]
                prev = closed[j - 1]

                # Hard segment boundary (zone_retest V1 convention): gap or invalid OHLC
                # abandons pending/active tracking for this source with no terminal
                # observation. Do not bridge across broken data.
                if cur.epoch - prev.epoch != step:
                    if lifecycle_status in ("pending", "active"):
                        lifecycle_status = "terminated"
                    continue
                if not _ohlc_sane(cur):
                    if lifecycle_status in ("pending", "active"):
                        lifecycle_status = "terminated"
                    continue

                # Start evaluation
                if lifecycle_status == "pending":
                    is_against = (
                        (cur.close < cur.open)
                        if direction == "bullish"
                        else (cur.close > cur.open)
                    )
                    if is_against:
                        lifecycle_status = "active"
                        payload = {
                            "source_impulse_id": impulse_id,
                            "lifecycle_event": "start",
                            "lifecycle_state": "active",
                            "direction": direction,
                            "impulse_candle_epoch": form_candle.epoch,
                            "impulse_high": impulse_high,
                            "impulse_low": impulse_low,
                            "open": float(cur.open),
                            "high": float(cur.high),
                            "low": float(cur.low),
                            "close": float(cur.close),
                        }
                        out.append(
                            self._make_observation(
                                candle=cur,
                                variant="impulse_based",
                                direction=direction,
                                source_id=impulse_id,
                                lifecycle_event="start",
                                lifecycle_state="active",
                                reference_ids=(impulse_id,),
                                payload=payload,
                            )
                        )

                # Termination evaluation
                if lifecycle_status == "active":
                    if direction == "bullish":
                        is_resumption = cur.close > impulse_high
                        is_invalidation = cur.close < impulse_low
                    else:
                        is_resumption = cur.close < impulse_low
                        is_invalidation = cur.close > impulse_high

                    # Ambiguous termination: invalidation takes precedence
                    if is_invalidation:
                        lifecycle_status = "terminated"
                        payload = {
                            "source_impulse_id": impulse_id,
                            "lifecycle_event": "invalidation",
                            "lifecycle_state": "invalidation",
                            "termination_reason": "invalidation",
                            "direction": direction,
                            "impulse_candle_epoch": form_candle.epoch,
                            "impulse_high": impulse_high,
                            "impulse_low": impulse_low,
                            "open": float(cur.open),
                            "high": float(cur.high),
                            "low": float(cur.low),
                            "close": float(cur.close),
                        }
                        out.append(
                            self._make_observation(
                                candle=cur,
                                variant="impulse_based",
                                direction=direction,
                                source_id=impulse_id,
                                lifecycle_event="invalidation",
                                lifecycle_state="invalidation",
                                reference_ids=(impulse_id,),
                                payload=payload,
                            )
                        )
                    elif is_resumption:
                        lifecycle_status = "terminated"
                        payload = {
                            "source_impulse_id": impulse_id,
                            "lifecycle_event": "trend_resumption",
                            "lifecycle_state": "trend_resumption",
                            "termination_reason": "trend_resumption",
                            "direction": direction,
                            "impulse_candle_epoch": form_candle.epoch,
                            "impulse_high": impulse_high,
                            "impulse_low": impulse_low,
                            "open": float(cur.open),
                            "high": float(cur.high),
                            "low": float(cur.low),
                            "close": float(cur.close),
                        }
                        out.append(
                            self._make_observation(
                                candle=cur,
                                variant="impulse_based",
                                direction=direction,
                                source_id=impulse_id,
                                lifecycle_event="trend_resumption",
                                lifecycle_state="trend_resumption",
                                reference_ids=(impulse_id,),
                                payload=payload,
                            )
                        )

        return out

    # -------------------------------------------------------------------------
    # Variant A: structure_based
    # -------------------------------------------------------------------------

    def _reconstruct_structure_based(
        self, closed: list[Candle], *, step: int
    ) -> list[Observation]:
        swings = self._swing.detect_all_historical(closed)
        structure_moves = self._structure.detect_all_historical(closed)
        if not structure_moves or not swings:
            return []

        swing_by_id: dict[str, Observation] = {s.id: s for s in swings}
        epoch_to_idx: dict[int, int] = {c.epoch: i for i, c in enumerate(closed)}

        out: list[Observation] = []

        for move in structure_moves:
            # Established structural moves: HH for bullish, LL for bearish
            if move.variant not in ("hh", "ll"):
                continue
            direction = "bullish" if move.variant == "hh" else "bearish"
            move_id = move.id

            if not move.reference_ids:
                continue
            source_swing_id = move.reference_ids[0]
            source_swing = swing_by_id.get(source_swing_id)
            if source_swing is None:
                continue

            # Must evaluate subsequent candles strictly after move was confirmed
            move_conf = move.confirmed_at
            if move_conf is None:
                continue
            move_idx = epoch_to_idx.get(move_conf)
            if move_idx is None:
                continue

            lifecycle_status: str = "pending"

            for j in range(move_idx + 1, len(closed)):
                if lifecycle_status == "terminated":
                    break

                cur = closed[j]
                prev = closed[j - 1]

                # Hard segment boundary (zone_retest V1 convention): gap or invalid OHLC
                # abandons pending/active tracking for this source with no terminal
                # observation. Do not bridge across broken data.
                if cur.epoch - prev.epoch != step:
                    if lifecycle_status in ("pending", "active"):
                        lifecycle_status = "terminated"
                    continue
                if not _ohlc_sane(cur):
                    if lifecycle_status in ("pending", "active"):
                        lifecycle_status = "terminated"
                    continue

                # Confirmed swings available strictly before or at candidate candle's close
                # Causal contract: confirmed_at <= cur.epoch
                avail_swings = [
                    s
                    for s in swings
                    if s.confirmed_at is not None and s.confirmed_at <= cur.epoch
                ]

                # Start condition
                if lifecycle_status == "pending":
                    # Swing level used for start: confirmed swing level in direction opposite the move
                    # Bullish: candle closes strictly below a confirmed swing high (e.g. source swing or recent swing high)
                    # Bearish: candle closes strictly above a confirmed swing low
                    target_kind = "swing_high" if direction == "bullish" else "swing_low"
                    eligible_start_swings = [
                        s
                        for s in avail_swings
                        if s.type == target_kind
                        and s.event_time is not None
                        and source_swing.event_time is not None
                        and s.event_time >= source_swing.event_time
                    ]
                    if not eligible_start_swings:
                        eligible_start_swings = [source_swing]

                    # Pick the most recent eligible swing high/low
                    start_swing = max(
                        eligible_start_swings,
                        key=lambda s: (
                            s.event_time if s.event_time is not None else 0,
                            s.confirmed_at if s.confirmed_at is not None else 0,
                        ),
                    )
                    start_price = float(start_swing.payload.get("price", 0.0))

                    is_start = (
                        (cur.close < start_price)
                        if direction == "bullish"
                        else (cur.close > start_price)
                    )
                    if is_start:
                        lifecycle_status = "active"
                        payload = {
                            "originating_move_id": move_id,
                            "start_swing_id": start_swing.id,
                            "start_swing_price": start_price,
                            "lifecycle_event": "start",
                            "lifecycle_state": "active",
                            "direction": direction,
                            "open": float(cur.open),
                            "high": float(cur.high),
                            "low": float(cur.low),
                            "close": float(cur.close),
                        }
                        out.append(
                            self._make_observation(
                                candle=cur,
                                variant="structure_based",
                                direction=direction,
                                source_id=move_id,
                                lifecycle_event="start",
                                lifecycle_state="active",
                                reference_ids=(move_id, start_swing.id),
                                payload=payload,
                            )
                        )

                # Termination conditions
                if lifecycle_status == "active":
                    # 1. Structural failure:
                    # Latest confirmed protective swing:
                    #   Bullish: most recent confirmed higher low (HL)
                    #   Bearish: most recent confirmed lower high (LH)
                    # Use confirmed market-structure observations available up to cur.epoch
                    protective_swing = self._find_protective_swing(
                        direction=direction,
                        structure_moves=structure_moves,
                        swing_by_id=swing_by_id,
                        eval_epoch=cur.epoch,
                    )

                    is_failure = False
                    if protective_swing is not None:
                        prot_price = float(protective_swing.payload.get("price", 0.0))
                        is_failure = (
                            (cur.close < prot_price)
                            if direction == "bullish"
                            else (cur.close > prot_price)
                        )

                    # 2. Trend resumption:
                    # Most recent confirmed swing level in move direction
                    #   Bullish: most recent confirmed swing_high
                    #   Bearish: most recent confirmed swing_low
                    resumption_swing = self._find_resumption_swing(
                        direction=direction,
                        avail_swings=avail_swings,
                        move_source_swing=source_swing,
                    )
                    is_resumption = False
                    if resumption_swing is not None:
                        res_price = float(resumption_swing.payload.get("price", 0.0))
                        is_resumption = (
                            (cur.close > res_price)
                            if direction == "bullish"
                            else (cur.close < res_price)
                        )

                    # Ambiguous termination: structural failure takes precedence
                    if is_failure and protective_swing is not None:
                        lifecycle_status = "terminated"
                        prot_price = float(protective_swing.payload.get("price", 0.0))
                        payload = {
                            "originating_move_id": move_id,
                            "protective_swing_id": protective_swing.id,
                            "protective_swing_price": prot_price,
                            "lifecycle_event": "structural_failure",
                            "lifecycle_state": "structural_failure",
                            "termination_reason": "structural_failure",
                            "direction": direction,
                            "open": float(cur.open),
                            "high": float(cur.high),
                            "low": float(cur.low),
                            "close": float(cur.close),
                        }
                        out.append(
                            self._make_observation(
                                candle=cur,
                                variant="structure_based",
                                direction=direction,
                                source_id=move_id,
                                lifecycle_event="structural_failure",
                                lifecycle_state="structural_failure",
                                reference_ids=(move_id, protective_swing.id),
                                payload=payload,
                            )
                        )
                    elif is_resumption and resumption_swing is not None:
                        lifecycle_status = "terminated"
                        res_price = float(resumption_swing.payload.get("price", 0.0))
                        payload = {
                            "originating_move_id": move_id,
                            "resumption_swing_id": resumption_swing.id,
                            "resumption_swing_price": res_price,
                            "lifecycle_event": "trend_resumption",
                            "lifecycle_state": "trend_resumption",
                            "termination_reason": "trend_resumption",
                            "direction": direction,
                            "open": float(cur.open),
                            "high": float(cur.high),
                            "low": float(cur.low),
                            "close": float(cur.close),
                        }
                        out.append(
                            self._make_observation(
                                candle=cur,
                                variant="structure_based",
                                direction=direction,
                                source_id=move_id,
                                lifecycle_event="trend_resumption",
                                lifecycle_state="trend_resumption",
                                reference_ids=(move_id, resumption_swing.id),
                                payload=payload,
                            )
                        )

        return out

    def _find_protective_swing(
        self,
        *,
        direction: str,
        structure_moves: list[Observation],
        swing_by_id: dict[str, Observation],
        eval_epoch: int,
    ) -> Optional[Observation]:
        """
        Identify the latest confirmed protective swing available at evaluation:
          - Bullish: most recent confirmed higher low (HL).
          - Bearish: most recent confirmed lower high (LH).
        Eligibility is causal on confirmation only (confirmed_at <= eval_epoch).
        Swings formed after the originating move may qualify once confirmed.
        If ambiguous or missing, returns None.
        """
        target_variant = "hl" if direction == "bullish" else "lh"
        eligible_moves = [
            m
            for m in structure_moves
            if m.variant == target_variant
            and m.confirmed_at is not None
            and m.confirmed_at <= eval_epoch
        ]
        if not eligible_moves:
            return None

        # Latest by event_time, then confirmed_at (most recent protective structure)
        eligible_moves.sort(
            key=lambda m: (
                m.event_time if m.event_time is not None else 0,
                m.confirmed_at if m.confirmed_at is not None else 0,
            )
        )
        latest_ms = eligible_moves[-1]
        if not latest_ms.reference_ids:
            return None
        # source swing of HL/LH
        protective_swing = swing_by_id.get(latest_ms.reference_ids[0])
        return protective_swing

    def _find_resumption_swing(
        self,
        *,
        direction: str,
        avail_swings: list[Observation],
        move_source_swing: Observation,
    ) -> Optional[Observation]:
        """
        Identify the most recent confirmed swing level in the originating move's direction.
          - Bullish: most recent confirmed swing_high.
          - Bearish: most recent confirmed swing_low.
        """
        target_type = "swing_high" if direction == "bullish" else "swing_low"
        base_event_time = (
            move_source_swing.event_time
            if move_source_swing.event_time is not None
            else move_source_swing.candle_epoch
        )
        eligible = [
            s
            for s in avail_swings
            if s.type == target_type
            and (s.event_time is not None and s.event_time >= base_event_time)
        ]
        if not eligible:
            eligible = [move_source_swing]

        # Select the most recent confirmed swing
        return max(
            eligible,
            key=lambda s: (
                s.event_time if s.event_time is not None else 0,
                s.confirmed_at if s.confirmed_at is not None else 0,
            ),
        )
