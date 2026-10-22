"""
Phase 5B — Classical swing-based rectangle pattern.

Detects confirmed rectangle (range) patterns from Phase 2A swing observations.

  - type: rectangle
  - variant: swing_based_v1
  - bullish: breakout close strictly above resistance
  - bearish: breakout close strictly below support

Geometry:
  - At least two distinct swing highs and two distinct swing lows.
  - Anchors form an alternating sequence in chronological order
    (H→L→H→L… or L→H→L→H…).
  - Resistance = median of selected swing-high prices.
  - Support   = median of selected swing-low prices.
  - Even-sized sets use the arithmetic mean of the two central sorted values.
  - Every selected high must lie within boundary_tolerance of resistance;
    every selected low within boundary_tolerance of support
    (symmetric relative deviation, inclusive).
  - Resistance must be strictly greater than support.

Causality and expiry (aligned with Phase 5A):
  - Uses only confirmed swing observations.
  - Breakout eligible only on a subsequent completed candle after all anchors
    are available (candle.epoch > max(anchor.confirmed_at)).
  - Window measured in candles from the final anchor's candle;
    offsets 1 .. max_bars_to_breakout inclusive.
  - Continuous segments processed independently; no gap bridging.
  - At most one confirmation per anchor-ID combination (first qualifying direction).
  - Candidate enumeration: contiguous alternating windows within maximal alternating
    runs of the chronological swing stream (O(n²)); operational bounds
    _MAX_SWINGS_FOR_ENUMERATION and _MAX_ANCHORS_PER_CANDIDATE prevent pathological
    work on extremely long histories.

Configuration (all participate in detector_config_id):
  - boundary_tolerance: relative (default 0.002)
  - max_bars_to_breakout: positive int (default 50)
  - left_strength, right_strength, tie_policy: forwarded to SwingPivotDetector
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

logger = get_logger("cpab.detectors.rectangle")

DEFAULT_BOUNDARY_TOLERANCE = 0.002
DEFAULT_MAX_BARS_TO_BREAKOUT = 50
_EPS = 1e-12

CONCEPT_RECTANGLE = "rectangle"
VARIANT = "swing_based_v1"


def _validate_boundary_tolerance(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"boundary_tolerance must be a finite nonnegative number, got {value!r}"
        )
    f = float(value)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(
            f"boundary_tolerance must be a finite nonnegative number, got {value!r}"
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


def _within_tolerance(price: float, boundary: float, tolerance: float) -> bool:
    """Symmetric relative deviation; inclusive; epsilon guard."""
    denom = max(abs(price), abs(boundary), _EPS)
    return abs(price - boundary) / denom <= tolerance


def _median(prices: list[float]) -> float:
    """Median; for even n use arithmetic mean of the two central sorted values."""
    if not prices:
        raise ValueError("median of empty sequence")
    s = sorted(prices)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float(s[mid - 1] + s[mid]) / 2.0


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


def _swing_event_time(obs: Observation) -> Optional[int]:
    if obs.event_time is not None:
        return int(obs.event_time)
    if obs.candle_epoch is not None:
        return int(obs.candle_epoch)
    return None


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


class RectangleDetector(PriceActionDetector):
    """
    Swing-based classical rectangle (range) pattern detector.

    Configuration is immutable after construction. Invalid parameters raise
    ValueError (no silent coercion).
    """

    def __init__(
        self,
        *,
        boundary_tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
        max_bars_to_breakout: int = DEFAULT_MAX_BARS_TO_BREAKOUT,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "rectangle",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._boundary_tolerance = _validate_boundary_tolerance(boundary_tolerance)
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
            "boundary_tolerance": self._boundary_tolerance,
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
    def boundary_tolerance(self) -> float:
        return self._boundary_tolerance

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
        Lower bound for a minimal rectangle (four swings) to become confirmable:
        four swings each needing left+1+right bars, plus one post-final-anchor
        bar for breakout. Insufficient series yields no emission.
        """
        s = self._left + 1 + self._right
        return 4 * s + 1

    def config_dict(self) -> dict[str, Any]:
        return {
            "boundary_tolerance": self._boundary_tolerance,
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

    def _reconstruct_segment(
        self,
        closed: list[Candle],
        *,
        step: int,
    ) -> list[Observation]:
        """Detect rectangles within one continuous closed-candle segment."""
        epoch_to_idx = {c.epoch: i for i, c in enumerate(closed)}

        swing_obs = self._swing.detect_all_historical(closed)
        highs = [o for o in swing_obs if o.type == "swing_high"]
        lows = [o for o in swing_obs if o.type == "swing_low"]

        def _key(o: Observation) -> tuple:
            et = _swing_event_time(o) or 0
            return (et, o.confirmed_at or 0, o.id)

        highs = sorted(highs, key=_key)
        lows = sorted(lows, key=_key)

        candidates = self._enumerate_candidates(highs, lows)
        seen: set[tuple[str, ...]] = set()
        out: list[Observation] = []

        for anchors in candidates:
            anchor_ids = tuple(a.id for a in anchors)
            # Canonical identity: sorted for dedup key, chronological order preserved in payload
            dedup_key = tuple(sorted(anchor_ids))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            obs = self._try_confirm_candidate(
                anchors,
                closed=closed,
                epoch_to_idx=epoch_to_idx,
            )
            if obs is not None:
                out.append(obs)

        return out

    # Operational safety bound: number of swings considered for enumeration
    # within one continuous segment. Prevents pathological exponential work
    # on extremely long histories while covering realistic range structures.
    # Most-recent swings are retained when the bound is exceeded.
    _MAX_SWINGS_FOR_ENUMERATION = 32

    # Maximum anchors per single candidate window (contiguous alternating run).
    # Longer runs are split into overlapping windows of this length or less.
    _MAX_ANCHORS_PER_CANDIDATE = 12

    def _enumerate_candidates(
        self,
        highs: list[Observation],
        lows: list[Observation],
    ) -> list[list[Observation]]:
        """
        Enumerate rectangle candidates from the chronological swing stream.

        Strategy (bounded, O(n²) in the number of swings):
          1. Merge and sort highs/lows by event time.
          2. If more than _MAX_SWINGS_FOR_ENUMERATION, keep the most recent.
          3. Split into maximal *contiguous alternating runs* (a run breaks when
             two consecutive swings share the same type).
          4. Within each run of length >= 4, emit every contiguous window
             (subarray) of length 4 .. min(run_len, _MAX_ANCHORS_PER_CANDIDATE)
             that contains at least 2 highs and 2 lows.

        Contiguous windows preserve alternation by construction and match the
        natural definition of successive range touches. This replaces the prior
        exhaustive alternating-subsequence search, which could grow exponentially
        with swing count while producing many non-structural combinations.
        """
        all_swings: list[tuple[Observation, str]] = []
        for o in highs:
            et = _swing_event_time(o)
            if et is None or o.confirmed_at is None:
                continue
            if _swing_price(o) is None:
                continue
            all_swings.append((o, "H"))
        for o in lows:
            et = _swing_event_time(o)
            if et is None or o.confirmed_at is None:
                continue
            if _swing_price(o) is None:
                continue
            all_swings.append((o, "L"))

        all_swings.sort(
            key=lambda t: (
                _swing_event_time(t[0]) or 0,
                t[0].confirmed_at or 0,
                t[0].id,
            )
        )

        if len(all_swings) > self._MAX_SWINGS_FOR_ENUMERATION:
            all_swings = all_swings[-self._MAX_SWINGS_FOR_ENUMERATION :]

        # Split into maximal contiguous alternating runs.
        runs: list[list[tuple[Observation, str]]] = []
        if all_swings:
            current: list[tuple[Observation, str]] = [all_swings[0]]
            for item in all_swings[1:]:
                if item[1] == current[-1][1]:
                    # same type as previous → break run
                    if len(current) >= 4:
                        runs.append(current)
                    current = [item]
                else:
                    current.append(item)
            if len(current) >= 4:
                runs.append(current)

        candidates: list[list[Observation]] = []
        max_len = self._MAX_ANCHORS_PER_CANDIDATE
        for run in runs:
            n = len(run)
            # Contiguous windows [i, j) of length >= 4
            for i in range(n):
                for j in range(i + 4, min(n, i + max_len) + 1):
                    window = run[i:j]
                    nh = sum(1 for _, t in window if t == "H")
                    nl = len(window) - nh
                    if nh >= 2 and nl >= 2:
                        candidates.append([o for o, _ in window])

        return candidates

    def _try_confirm_candidate(
        self,
        anchors: list[Observation],
        *,
        closed: list[Candle],
        epoch_to_idx: dict[int, int],
    ) -> Optional[Observation]:
        """Validate geometry and look for the first qualifying breakout."""
        high_prices: list[float] = []
        low_prices: list[float] = []
        high_obs: list[Observation] = []
        low_obs: list[Observation] = []
        confs: list[int] = []
        final_event: Optional[int] = None

        for a in anchors:
            p = _swing_price(a)
            if p is None:
                return None
            et = _swing_event_time(a)
            ca = a.confirmed_at
            if et is None or ca is None:
                return None
            confs.append(int(ca))
            final_event = et if final_event is None else max(final_event, et)
            if a.type == "swing_high":
                high_prices.append(p)
                high_obs.append(a)
            else:
                low_prices.append(p)
                low_obs.append(a)

        if len(high_prices) < 2 or len(low_prices) < 2:
            return None
        if final_event is None or final_event not in epoch_to_idx:
            return None

        resistance = _median(high_prices)
        support = _median(low_prices)
        if not (resistance > support):
            return None

        # All selected highs/lows must qualify against the boundaries
        for p in high_prices:
            if not _within_tolerance(p, resistance, self._boundary_tolerance):
                return None
        for p in low_prices:
            if not _within_tolerance(p, support, self._boundary_tolerance):
                return None

        all_conf = max(confs)
        final_idx = epoch_to_idx[final_event]

        # Expiry check when all anchors become available
        last_eligible_idx = final_idx + self._max_bars
        if last_eligible_idx < len(closed):
            last_eligible_epoch = closed[last_eligible_idx].epoch
        else:
            last_eligible_epoch = closed[-1].epoch if closed else all_conf
        if all_conf >= last_eligible_epoch:
            return None

        eligible = self._candle_after_all_available(
            closed,
            epoch_to_idx,
            final_anchor_event=final_event,
            all_confirmed_at=all_conf,
            max_bars=self._max_bars,
        )

        for _offset, candle in eligible:
            # Bullish: close strictly above resistance
            if candle.close > resistance:
                return self._make_obs(
                    direction="bullish",
                    breakout=candle,
                    anchors=anchors,
                    support=support,
                    resistance=resistance,
                    high_obs=high_obs,
                    low_obs=low_obs,
                )
            # Bearish: close strictly below support
            if candle.close < support:
                return self._make_obs(
                    direction="bearish",
                    breakout=candle,
                    anchors=anchors,
                    support=support,
                    resistance=resistance,
                    high_obs=high_obs,
                    low_obs=low_obs,
                )
            # Wick-only or close exactly on boundary does not confirm; continue

        return None

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
        Offsets: 1 .. max_bars inclusive (from final_idx).
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
            if c.epoch <= all_confirmed_at:
                continue
            eligible.append((offset, c))
        return eligible

    def _make_obs(
        self,
        *,
        direction: str,
        breakout: Candle,
        anchors: list[Observation],
        support: float,
        resistance: float,
        high_obs: list[Observation],
        low_obs: list[Observation],
    ) -> Observation:
        # Chronological order of selected anchors
        ordered = sorted(
            anchors,
            key=lambda a: (
                _swing_event_time(a) or 0,
                a.confirmed_at or 0,
                a.id,
            ),
        )
        anchor_ids = tuple(a.id for a in ordered)
        event_time = breakout.epoch
        confirmed_at = breakout.epoch  # completed candle available at its open epoch

        swing_types = [a.type for a in ordered]
        swing_prices = [_swing_price(a) for a in ordered]
        high_ids = [a.id for a in high_obs]
        low_ids = [a.id for a in low_obs]

        payload: dict[str, Any] = {
            "support": support,
            "resistance": resistance,
            "boundary_tolerance": self._boundary_tolerance,
            "max_bars_to_breakout": self._max_bars,
            "anchor_swing_ids": list(anchor_ids),
            "anchor_swing_types": swing_types,
            "anchor_swing_prices": swing_prices,
            "high_swing_ids": high_ids,
            "low_swing_ids": low_ids,
            "breakout_direction": direction,
            "breakout_close": breakout.close,
            "breakout_candle_epoch": breakout.epoch,
            "breakout_ohlc": breakout.ohlc_tuple(),
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

        obs_id = _stable_pattern_id(
            concept=CONCEPT_RECTANGLE,
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
            type=CONCEPT_RECTANGLE,
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
