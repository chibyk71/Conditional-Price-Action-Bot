"""
Phase 5A — Classical swing-based reversal patterns.

Detects four confirmed patterns from Phase 2A swing observations:

  - double_top              (bearish)
  - double_bottom           (bullish)
  - head_and_shoulders      (bearish)
  - inverse_head_and_shoulders (bullish)

All patterns:
  - Use only confirmed swing_high / swing_low observations.
  - Require exact timeframe continuity across evidence candles.
  - Confirm on a subsequent eligible completed candle with a strict close
    beyond the neckline (wick-only does not confirm).
  - Reject on structural invalidation before confirmation.
  - Never confirm retrospectively (breakout must occur after all anchors
    are confirmed/available).
  - Expiry measured in candles from the final anchor's candle; candidates
    whose window has already expired when the last anchor becomes available
    are discarded; no confirmation after max_bars_to_breakout.

Configuration (all participate in detector_config_id):
  - similarity_tolerance: relative (default 0.002)
      abs(P1-P2) / max(abs(P1), abs(P2), eps) <= tolerance
  - max_bars_to_breakout: positive int (default 50)
  - left_strength, right_strength, tie_policy: forwarded to SwingPivotDetector

Variant is always "swing_based_v1". Direction is bullish or bearish as specified.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.swing import (
    SwingPivotDetector,
    _validate_strength,
    _validate_tie_policy,
    timeframe_seconds,
)
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.classical_reversal")

DEFAULT_SIMILARITY_TOLERANCE = 0.002
DEFAULT_MAX_BARS_TO_BREAKOUT = 50
_EPS = 1e-12

CONCEPT_DOUBLE_TOP = "double_top"
CONCEPT_DOUBLE_BOTTOM = "double_bottom"
CONCEPT_HEAD_AND_SHOULDERS = "head_and_shoulders"
CONCEPT_INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"

VARIANT = "swing_based_v1"


def _validate_similarity_tolerance(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"similarity_tolerance must be a finite nonnegative number, got {value!r}"
        )
    f = float(value)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(
            f"similarity_tolerance must be a finite nonnegative number, got {value!r}"
        )
    return f


def _validate_max_bars_to_breakout(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"max_bars_to_breakout must be a positive integer, got {value!r}"
        )
    if value < 1:
        raise ValueError(
            f"max_bars_to_breakout must be >= 1, got {value}"
        )
    return value


def _similar(p1: float, p2: float, tolerance: float) -> bool:
    denom = max(abs(p1), abs(p2), _EPS)
    return abs(p1 - p2) / denom <= tolerance


def _similarity_deviation(p1: float, p2: float) -> float:
    denom = max(abs(p1), abs(p2), _EPS)
    return abs(p1 - p2) / denom


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def _swing_price(obs: Observation) -> Optional[float]:
    payload = obs.payload or {}
    p = payload.get("price")
    if p is None and obs.metadata:
        p = obs.metadata.get("price")
    if not _finite(p):
        return None
    return float(p)


def _neckline_at(
    t: int,
    *,
    t_b: int,
    p_b: float,
    t_d: int,
    p_d: float,
) -> float:
    """Linear neckline through (t_b, p_b) and (t_d, p_d) evaluated at time t."""
    if t_d == t_b:
        return p_b
    slope = (p_d - p_b) / (t_d - t_b)
    return p_b + slope * (t - t_b)


def _stable_pattern_id(
    *,
    concept: str,
    symbol: str,
    timeframe: str,
    candle_epoch: int,
    event_time: int,
    confirmed_at: int,
    detector_id: str,
    detector_config_id: str,
    direction: str,
    anchor_ids: tuple[str, ...],
) -> str:
    parts = [
        concept,
        symbol,
        timeframe,
        str(candle_epoch),
        f"e{event_time}",
        f"c{confirmed_at}",
        detector_id,
        detector_config_id,
        direction,
        VARIANT,
        *anchor_ids,
    ]
    return "|".join(parts)


class ClassicalReversalDetector(PriceActionDetector):
    """
    Swing-based classical reversal pattern detector.

    Configuration is immutable after construction. Invalid parameters raise
    ValueError (no silent coercion).
    """

    def __init__(
        self,
        *,
        similarity_tolerance: float = DEFAULT_SIMILARITY_TOLERANCE,
        max_bars_to_breakout: int = DEFAULT_MAX_BARS_TO_BREAKOUT,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "classical_reversal",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._similarity_tolerance = _validate_similarity_tolerance(similarity_tolerance)
        self._max_bars = _validate_max_bars_to_breakout(max_bars_to_breakout)
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
            "similarity_tolerance": self._similarity_tolerance,
            "max_bars_to_breakout": self._max_bars,
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
    def similarity_tolerance(self) -> float:
        return self._similarity_tolerance

    @property
    def max_bars_to_breakout(self) -> int:
        return self._max_bars

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
        Lower bound for the shortest pattern (double top/bottom) to become
        confirmable: three swings each needing left+1+right bars, plus one
        post-final-anchor bar for breakout. Longer patterns (H&S) simply need
        more history; insufficient series yields no emission.
        """
        s = self._left + 1 + self._right
        return 3 * s + 1

    def config_dict(self) -> dict[str, Any]:
        return {
            "similarity_tolerance": self._similarity_tolerance,
            "max_bars_to_breakout": self._max_bars,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: reconstruct all patterns, emit only those whose
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
        if not closed:
            return []
        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            return []

        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            return []

        # Process each continuous segment independently so a gap does not
        # suppress valid patterns wholly within an unbroken section. Patterns
        # still cannot bridge gaps: anchors and breakout must live in one segment.
        out: list[Observation] = []
        for segment in self._continuous_segments(closed, step=step):
            if len(segment) < self.min_history:
                continue
            out.extend(self._reconstruct_segment(segment, step=step))

        out.sort(
            key=lambda o: (
                o.confirmed_at or 0,
                o.event_time or 0,
                o.type,
                o.id,
            )
        )
        return out

    def _reconstruct_segment(
        self,
        closed: list[Candle],
        *,
        step: int,
    ) -> list[Observation]:
        """Detect patterns within one continuous closed-candle segment."""
        epoch_to_idx = {c.epoch: i for i, c in enumerate(closed)}

        swing_obs = self._swing.detect_all_historical(closed)
        highs = [o for o in swing_obs if o.type == "swing_high"]
        lows = [o for o in swing_obs if o.type == "swing_low"]

        def _key(o: Observation) -> tuple:
            return (o.event_time or o.candle_epoch, o.confirmed_at or 0, o.id)

        highs = sorted(highs, key=_key)
        lows = sorted(lows, key=_key)

        out: list[Observation] = []
        seen_anchor_keys: set[tuple] = set()

        out.extend(
            self._detect_double_tops(
                highs, lows, closed, epoch_to_idx, step, seen_anchor_keys
            )
        )
        out.extend(
            self._detect_double_bottoms(
                highs, lows, closed, epoch_to_idx, step, seen_anchor_keys
            )
        )
        out.extend(
            self._detect_head_and_shoulders(
                highs, lows, closed, epoch_to_idx, step, seen_anchor_keys
            )
        )
        out.extend(
            self._detect_inverse_head_and_shoulders(
                highs, lows, closed, epoch_to_idx, step, seen_anchor_keys
            )
        )
        return out

    @staticmethod
    def _continuous_segments(
        candles: list[Candle],
        *,
        step: int,
    ) -> list[list[Candle]]:
        """Split into maximal runs of exact timeframe spacing."""
        if not candles:
            return []
        segments: list[list[Candle]] = [[candles[0]]]
        for i in range(1, len(candles)):
            if candles[i].epoch - candles[i - 1].epoch == step:
                segments[-1].append(candles[i])
            else:
                segments.append([candles[i]])
        return segments

    @staticmethod
    def _series_consecutive(candles: list[Candle], *, step: int) -> bool:
        if len(candles) < 2:
            return True
        for i in range(1, len(candles)):
            if candles[i].epoch - candles[i - 1].epoch != step:
                return False
        return True

    def _candle_after_all_available(
        self,
        closed: list[Candle],
        epoch_to_idx: dict[int, int],
        *,
        final_anchor_event: int,
        all_confirmed_at: int,
        max_bars: int,
    ) -> list[tuple[int, Candle]]:
        """
        Eligible breakout candidates: completed candles strictly after the
        final anchor's candle, within max_bars, and only after all anchors
        are available (candle.epoch > all_confirmed_at).

        Contract: confirmation requires a *subsequent* completed candle after
        every anchor is confirmed/available. Equality with the latest
        anchor's confirmed_at is not eligible.
        """
        if final_anchor_event not in epoch_to_idx:
            return []
        final_idx = epoch_to_idx[final_anchor_event]
        eligible: list[tuple[int, Candle]] = []
        for offset in range(1, max_bars + 1):
            idx = final_idx + offset
            if idx >= len(closed):
                break
            c = closed[idx]
            # Must be strictly after all anchors are available.
            if c.epoch <= all_confirmed_at:
                continue
            eligible.append((offset, c))
        return eligible

    def _detect_double_tops(
        self,
        highs: list[Observation],
        lows: list[Observation],
        closed: list[Candle],
        epoch_to_idx: dict[int, int],
        step: int,
        seen: set[tuple],
    ) -> list[Observation]:
        out: list[Observation] = []
        for i, a in enumerate(highs):
            pa = _swing_price(a)
            if pa is None:
                continue
            ta = a.event_time if a.event_time is not None else a.candle_epoch
            ca = a.confirmed_at
            if ca is None:
                continue
            for j in range(i + 1, len(highs)):
                c_swing = highs[j]
                pc = _swing_price(c_swing)
                if pc is None:
                    continue
                tc = (
                    c_swing.event_time
                    if c_swing.event_time is not None
                    else c_swing.candle_epoch
                )
                cc = c_swing.confirmed_at
                if cc is None or tc <= ta:
                    continue
                if not _similar(pa, pc, self._similarity_tolerance):
                    continue
                # Find a low B strictly between A and C in event time.
                candidates_b = [
                    b
                    for b in lows
                    if (b.event_time if b.event_time is not None else b.candle_epoch)
                    > ta
                    and (b.event_time if b.event_time is not None else b.candle_epoch)
                    < tc
                    and b.confirmed_at is not None
                ]
                for b in candidates_b:
                    pb = _swing_price(b)
                    if pb is None:
                        continue
                    tb = b.event_time if b.event_time is not None else b.candle_epoch
                    cb = b.confirmed_at
                    if cb is None:
                        continue
                    if not (pa > pb and pc > pb):
                        continue

                    anchor_key = (
                        CONCEPT_DOUBLE_TOP,
                        a.id,
                        b.id,
                        c_swing.id,
                    )
                    if anchor_key in seen:
                        continue
                    seen.add(anchor_key)

                    all_conf = max(ca, cb, cc)
                    higher_peak = max(pa, pc)
                    neckline = pb  # horizontal at B

                    # Expiry check when all anchors available.
                    if tc not in epoch_to_idx:
                        continue
                    final_idx = epoch_to_idx[tc]
                    # If all_conf is already past the last eligible breakout bar, reject.
                    last_eligible_idx = final_idx + self._max_bars
                    if last_eligible_idx < len(closed):
                        last_eligible_epoch = closed[last_eligible_idx].epoch
                    else:
                        last_eligible_epoch = (
                            closed[-1].epoch
                            if closed
                            else all_conf
                        )
                    # Reject if window already expired at availability.
                    # Window starts at final_idx+1; if all_conf is after the
                    # last possible breakout candle epoch, expired.
                    if all_conf >= last_eligible_epoch:
                        continue

                    eligible = self._candle_after_all_available(
                        closed,
                        epoch_to_idx,
                        final_anchor_event=tc,
                        all_confirmed_at=all_conf,
                        max_bars=self._max_bars,
                    )
                    confirmed_obs = None
                    for offset, candle in eligible:
                        # Structural rejection: close above higher peak.
                        if candle.close > higher_peak:
                            break
                        # Confirmation: close strictly below neckline.
                        if candle.close < neckline:
                            confirmed_obs = self._make_obs(
                                concept=CONCEPT_DOUBLE_TOP,
                                direction="bearish",
                                breakout=candle,
                                anchors=(a, b, c_swing),
                                neckline_at_breakout=neckline,
                                similarity_dev=_similarity_deviation(pa, pc),
                                geometry={
                                    "peak_a_price": pa,
                                    "trough_b_price": pb,
                                    "peak_c_price": pc,
                                    "peak_a_event_time": ta,
                                    "trough_b_event_time": tb,
                                    "peak_c_event_time": tc,
                                    "higher_peak": higher_peak,
                                    "neckline_type": "horizontal",
                                },
                            )
                            break
                    if confirmed_obs is not None:
                        out.append(confirmed_obs)
        return out

    def _detect_double_bottoms(
        self,
        highs: list[Observation],
        lows: list[Observation],
        closed: list[Candle],
        epoch_to_idx: dict[int, int],
        step: int,
        seen: set[tuple],
    ) -> list[Observation]:
        out: list[Observation] = []
        for i, a in enumerate(lows):
            pa = _swing_price(a)
            if pa is None:
                continue
            ta = a.event_time if a.event_time is not None else a.candle_epoch
            ca = a.confirmed_at
            if ca is None:
                continue
            for j in range(i + 1, len(lows)):
                c_swing = lows[j]
                pc = _swing_price(c_swing)
                if pc is None:
                    continue
                tc = (
                    c_swing.event_time
                    if c_swing.event_time is not None
                    else c_swing.candle_epoch
                )
                cc = c_swing.confirmed_at
                if cc is None or tc <= ta:
                    continue
                if not _similar(pa, pc, self._similarity_tolerance):
                    continue
                candidates_b = [
                    b
                    for b in highs
                    if (b.event_time if b.event_time is not None else b.candle_epoch)
                    > ta
                    and (b.event_time if b.event_time is not None else b.candle_epoch)
                    < tc
                    and b.confirmed_at is not None
                ]
                for b in candidates_b:
                    pb = _swing_price(b)
                    if pb is None:
                        continue
                    tb = b.event_time if b.event_time is not None else b.candle_epoch
                    cb = b.confirmed_at
                    if cb is None:
                        continue
                    if not (pa < pb and pc < pb):
                        continue

                    anchor_key = (
                        CONCEPT_DOUBLE_BOTTOM,
                        a.id,
                        b.id,
                        c_swing.id,
                    )
                    if anchor_key in seen:
                        continue
                    seen.add(anchor_key)

                    all_conf = max(ca, cb, cc)
                    lower_trough = min(pa, pc)
                    neckline = pb

                    if tc not in epoch_to_idx:
                        continue
                    final_idx = epoch_to_idx[tc]
                    last_eligible_idx = final_idx + self._max_bars
                    if last_eligible_idx < len(closed):
                        last_eligible_epoch = closed[last_eligible_idx].epoch
                    else:
                        last_eligible_epoch = closed[-1].epoch if closed else all_conf
                    if all_conf >= last_eligible_epoch:
                        continue

                    eligible = self._candle_after_all_available(
                        closed,
                        epoch_to_idx,
                        final_anchor_event=tc,
                        all_confirmed_at=all_conf,
                        max_bars=self._max_bars,
                    )
                    confirmed_obs = None
                    for offset, candle in eligible:
                        if candle.close < lower_trough:
                            break
                        if candle.close > neckline:
                            confirmed_obs = self._make_obs(
                                concept=CONCEPT_DOUBLE_BOTTOM,
                                direction="bullish",
                                breakout=candle,
                                anchors=(a, b, c_swing),
                                neckline_at_breakout=neckline,
                                similarity_dev=_similarity_deviation(pa, pc),
                                geometry={
                                    "trough_a_price": pa,
                                    "peak_b_price": pb,
                                    "trough_c_price": pc,
                                    "trough_a_event_time": ta,
                                    "peak_b_event_time": tb,
                                    "trough_c_event_time": tc,
                                    "lower_trough": lower_trough,
                                    "neckline_type": "horizontal",
                                },
                            )
                            break
                    if confirmed_obs is not None:
                        out.append(confirmed_obs)
        return out

    def _detect_head_and_shoulders(
        self,
        highs: list[Observation],
        lows: list[Observation],
        closed: list[Candle],
        epoch_to_idx: dict[int, int],
        step: int,
        seen: set[tuple],
    ) -> list[Observation]:
        """
        Sequence: High A → Low B → High C → Low D → High E
        C strictly above A and E; A and E similar; neckline through B, D.
        """
        out: list[Observation] = []
        n_h = len(highs)
        if n_h < 3:
            return out
        for i_a in range(n_h):
            a = highs[i_a]
            pa = _swing_price(a)
            if pa is None or a.confirmed_at is None:
                continue
            ta = a.event_time if a.event_time is not None else a.candle_epoch
            for i_c in range(i_a + 1, n_h):
                c_sw = highs[i_c]
                pc = _swing_price(c_sw)
                if pc is None or c_sw.confirmed_at is None:
                    continue
                tc = (
                    c_sw.event_time
                    if c_sw.event_time is not None
                    else c_sw.candle_epoch
                )
                if tc <= ta or not (pc > pa):
                    continue
                for i_e in range(i_c + 1, n_h):
                    e = highs[i_e]
                    pe = _swing_price(e)
                    if pe is None or e.confirmed_at is None:
                        continue
                    te = e.event_time if e.event_time is not None else e.candle_epoch
                    if te <= tc or not (pc > pe):
                        continue
                    if not _similar(pa, pe, self._similarity_tolerance):
                        continue

                    # Lows B between A and C, D between C and E.
                    bs = [
                        b
                        for b in lows
                        if b.confirmed_at is not None
                        and ta
                        < (b.event_time if b.event_time is not None else b.candle_epoch)
                        < tc
                    ]
                    ds = [
                        d
                        for d in lows
                        if d.confirmed_at is not None
                        and tc
                        < (d.event_time if d.event_time is not None else d.candle_epoch)
                        < te
                    ]
                    for b in bs:
                        pb = _swing_price(b)
                        if pb is None:
                            continue
                        tb = b.event_time if b.event_time is not None else b.candle_epoch
                        for d in ds:
                            pd = _swing_price(d)
                            if pd is None:
                                continue
                            td = (
                                d.event_time
                                if d.event_time is not None
                                else d.candle_epoch
                            )
                            if td <= tb:
                                continue

                            anchor_key = (
                                CONCEPT_HEAD_AND_SHOULDERS,
                                a.id,
                                b.id,
                                c_sw.id,
                                d.id,
                                e.id,
                            )
                            if anchor_key in seen:
                                continue
                            seen.add(anchor_key)

                            all_conf = max(
                                a.confirmed_at,
                                b.confirmed_at,
                                c_sw.confirmed_at,
                                d.confirmed_at,
                                e.confirmed_at,
                            )
                            head_price = pc

                            if te not in epoch_to_idx:
                                continue
                            final_idx = epoch_to_idx[te]
                            last_eligible_idx = final_idx + self._max_bars
                            if last_eligible_idx < len(closed):
                                last_eligible_epoch = closed[last_eligible_idx].epoch
                            else:
                                last_eligible_epoch = (
                                    closed[-1].epoch if closed else all_conf
                                )
                            if all_conf >= last_eligible_epoch:
                                continue

                            eligible = self._candle_after_all_available(
                                closed,
                                epoch_to_idx,
                                final_anchor_event=te,
                                all_confirmed_at=all_conf,
                                max_bars=self._max_bars,
                            )
                            confirmed_obs = None
                            for offset, candle in eligible:
                                if candle.close > head_price:
                                    break
                                nl = _neckline_at(
                                    candle.epoch,
                                    t_b=tb,
                                    p_b=pb,
                                    t_d=td,
                                    p_d=pd,
                                )
                                if candle.close < nl:
                                    confirmed_obs = self._make_obs(
                                        concept=CONCEPT_HEAD_AND_SHOULDERS,
                                        direction="bearish",
                                        breakout=candle,
                                        anchors=(a, b, c_sw, d, e),
                                        neckline_at_breakout=nl,
                                        similarity_dev=_similarity_deviation(pa, pe),
                                        geometry={
                                            "shoulder_a_price": pa,
                                            "trough_b_price": pb,
                                            "head_c_price": pc,
                                            "trough_d_price": pd,
                                            "shoulder_e_price": pe,
                                            "shoulder_a_event_time": ta,
                                            "trough_b_event_time": tb,
                                            "head_c_event_time": tc,
                                            "trough_d_event_time": td,
                                            "shoulder_e_event_time": te,
                                            "neckline_type": "sloped",
                                            "neckline_b_price": pb,
                                            "neckline_d_price": pd,
                                        },
                                    )
                                    break
                            if confirmed_obs is not None:
                                out.append(confirmed_obs)
        return out

    def _detect_inverse_head_and_shoulders(
        self,
        highs: list[Observation],
        lows: list[Observation],
        closed: list[Candle],
        epoch_to_idx: dict[int, int],
        step: int,
        seen: set[tuple],
    ) -> list[Observation]:
        """
        Sequence: Low A → High B → Low C → High D → Low E
        C strictly below A and E; A and E similar; neckline through B, D.
        """
        out: list[Observation] = []
        n_l = len(lows)
        if n_l < 3:
            return out
        for i_a in range(n_l):
            a = lows[i_a]
            pa = _swing_price(a)
            if pa is None or a.confirmed_at is None:
                continue
            ta = a.event_time if a.event_time is not None else a.candle_epoch
            for i_c in range(i_a + 1, n_l):
                c_sw = lows[i_c]
                pc = _swing_price(c_sw)
                if pc is None or c_sw.confirmed_at is None:
                    continue
                tc = (
                    c_sw.event_time
                    if c_sw.event_time is not None
                    else c_sw.candle_epoch
                )
                if tc <= ta or not (pc < pa):
                    continue
                for i_e in range(i_c + 1, n_l):
                    e = lows[i_e]
                    pe = _swing_price(e)
                    if pe is None or e.confirmed_at is None:
                        continue
                    te = e.event_time if e.event_time is not None else e.candle_epoch
                    if te <= tc or not (pc < pe):
                        continue
                    if not _similar(pa, pe, self._similarity_tolerance):
                        continue

                    bs = [
                        b
                        for b in highs
                        if b.confirmed_at is not None
                        and ta
                        < (b.event_time if b.event_time is not None else b.candle_epoch)
                        < tc
                    ]
                    ds = [
                        d
                        for d in highs
                        if d.confirmed_at is not None
                        and tc
                        < (d.event_time if d.event_time is not None else d.candle_epoch)
                        < te
                    ]
                    for b in bs:
                        pb = _swing_price(b)
                        if pb is None:
                            continue
                        tb = b.event_time if b.event_time is not None else b.candle_epoch
                        for d in ds:
                            pd = _swing_price(d)
                            if pd is None:
                                continue
                            td = (
                                d.event_time
                                if d.event_time is not None
                                else d.candle_epoch
                            )
                            if td <= tb:
                                continue

                            anchor_key = (
                                CONCEPT_INVERSE_HEAD_AND_SHOULDERS,
                                a.id,
                                b.id,
                                c_sw.id,
                                d.id,
                                e.id,
                            )
                            if anchor_key in seen:
                                continue
                            seen.add(anchor_key)

                            all_conf = max(
                                a.confirmed_at,
                                b.confirmed_at,
                                c_sw.confirmed_at,
                                d.confirmed_at,
                                e.confirmed_at,
                            )
                            head_price = pc

                            if te not in epoch_to_idx:
                                continue
                            final_idx = epoch_to_idx[te]
                            last_eligible_idx = final_idx + self._max_bars
                            if last_eligible_idx < len(closed):
                                last_eligible_epoch = closed[last_eligible_idx].epoch
                            else:
                                last_eligible_epoch = (
                                    closed[-1].epoch if closed else all_conf
                                )
                            if all_conf >= last_eligible_epoch:
                                continue

                            eligible = self._candle_after_all_available(
                                closed,
                                epoch_to_idx,
                                final_anchor_event=te,
                                all_confirmed_at=all_conf,
                                max_bars=self._max_bars,
                            )
                            confirmed_obs = None
                            for offset, candle in eligible:
                                if candle.close < head_price:
                                    break
                                nl = _neckline_at(
                                    candle.epoch,
                                    t_b=tb,
                                    p_b=pb,
                                    t_d=td,
                                    p_d=pd,
                                )
                                if candle.close > nl:
                                    confirmed_obs = self._make_obs(
                                        concept=CONCEPT_INVERSE_HEAD_AND_SHOULDERS,
                                        direction="bullish",
                                        breakout=candle,
                                        anchors=(a, b, c_sw, d, e),
                                        neckline_at_breakout=nl,
                                        similarity_dev=_similarity_deviation(pa, pe),
                                        geometry={
                                            "shoulder_a_price": pa,
                                            "peak_b_price": pb,
                                            "head_c_price": pc,
                                            "peak_d_price": pd,
                                            "shoulder_e_price": pe,
                                            "shoulder_a_event_time": ta,
                                            "peak_b_event_time": tb,
                                            "head_c_event_time": tc,
                                            "peak_d_event_time": td,
                                            "shoulder_e_event_time": te,
                                            "neckline_type": "sloped",
                                            "neckline_b_price": pb,
                                            "neckline_d_price": pd,
                                        },
                                    )
                                    break
                            if confirmed_obs is not None:
                                out.append(confirmed_obs)
        return out

    def _make_obs(
        self,
        *,
        concept: str,
        direction: str,
        breakout: Candle,
        anchors: tuple[Observation, ...],
        neckline_at_breakout: float,
        similarity_dev: float,
        geometry: dict[str, Any],
    ) -> Observation:
        anchor_ids = tuple(a.id for a in anchors)
        event_time = breakout.epoch
        confirmed_at = breakout.epoch  # completed candle available at its open epoch
        payload: dict[str, Any] = {
            **geometry,
            "neckline_at_breakout": neckline_at_breakout,
            "similarity_deviation": similarity_dev,
            "similarity_tolerance": self._similarity_tolerance,
            "max_bars_to_breakout": self._max_bars,
            "breakout_candle_epoch": breakout.epoch,
            "breakout_ohlc": breakout.ohlc_tuple(),
            "anchor_swing_ids": list(anchor_ids),
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        obs_id = _stable_pattern_id(
            concept=concept,
            symbol=breakout.symbol,
            timeframe=breakout.timeframe,
            candle_epoch=breakout.epoch,
            event_time=event_time,
            confirmed_at=confirmed_at,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            direction=direction,
            anchor_ids=anchor_ids,
        )
        return Observation.create(
            type=concept,
            symbol=breakout.symbol,
            timeframe=breakout.timeframe,
            candle_epoch=breakout.epoch,
            id=obs_id,
            event_time=event_time,
            confirmed_at=confirmed_at,
            developing=False,
            direction=direction,
            variant=VARIANT,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=anchor_ids,
            payload=payload,
            metadata=payload,
        )
