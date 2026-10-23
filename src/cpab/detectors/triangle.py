"""
Phase 5C — Classical swing-based triangle patterns.

Detects confirmed ascending, descending, and symmetrical triangles from
Phase 2A swing observations.

  - type: ascending_triangle | descending_triangle | symmetrical_triangle
  - variant: swing_based_v1
  - Ascending:   upper ≈ horizontal, lower rising; breakout close above upper
  - Descending:  upper falling, lower ≈ horizontal; breakout close below lower
  - Symmetrical: upper falling, lower rising; breakout either side

Geometry:
  - At least two distinct swing highs and two distinct swing lows.
  - Anchors form an alternating sequence in chronological order
    (H→L→H→L… or L→H→L→H…). Intermediate swings may be skipped.
  - Upper boundary = straight line through first and last selected swing highs
    (chronological).
  - Lower boundary = straight line through first and last selected swing lows.
  - Every selected high must lie within boundary_tolerance of the upper line
    evaluated at that swing's event time; every selected low within
    boundary_tolerance of the lower line (symmetric relative deviation,
    inclusive).
  - A boundary is approximately horizontal when
    abs(last - first) / max(|first|, |last|, eps) <= horizontal_tolerance.
    Horizontal classification is exclusive of rising/falling.
  - Upper and lower boundaries must have a finite intersection strictly after
    the final anchor's event time.

Causality and expiry (aligned with Phase 5A/5B):
  - Uses only confirmed swing observations.
  - Breakout / invalidation eligible only on a subsequent completed candle
    after all anchors are available (candle.epoch > max(anchor.confirmed_at)).
  - Window measured in candles from the final anchor's candle;
    offsets 1 .. max_bars_to_breakout inclusive.
  - Continuous segments processed independently; no gap bridging.
  - Opposite-boundary close invalidates before confirmation; an invalidated
    candidate never later emits a breakout.
  - Candidate enumeration: all alternating chronological subsequences with
    ≥2 highs and ≥2 lows. Safe suffix-count pruning; no silent truncation
    of history and no arbitrary anchor-count caps. Candidates are streamed
    into confirmation. Source swing IDs unique per entry; index-based
    traversal yields each anchor-ID combination at most once.
  - Worst-case runtime remains exponential in the number of alternating
    swings; peak memory is O(n swings + recursion depth + confirmed outputs).

Configuration (all participate in detector_config_id):
  - boundary_tolerance: relative (default 0.002)
  - horizontal_tolerance: relative (default 0.002)
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

logger = get_logger("cpab.detectors.triangle")

DEFAULT_BOUNDARY_TOLERANCE = 0.002
DEFAULT_HORIZONTAL_TOLERANCE = 0.002
DEFAULT_MAX_BARS_TO_BREAKOUT = 50
_EPS = 1e-12

CONCEPT_ASCENDING = "ascending_triangle"
CONCEPT_DESCENDING = "descending_triangle"
CONCEPT_SYMMETRICAL = "symmetrical_triangle"
VARIANT = "swing_based_v1"

# Pattern → (intended breakout side, opposite invalidation side)
# "upper" / "lower" refer to the boundary that must be crossed.
_PATTERN_INTENDED = {
    CONCEPT_ASCENDING: "upper",
    CONCEPT_DESCENDING: "lower",
    CONCEPT_SYMMETRICAL: "either",
}


def _validate_tolerance(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a finite nonnegative number, got {value!r}"
        )
    f = float(value)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(
            f"{name} must be a finite nonnegative number, got {value!r}"
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


def _relative_change(first: float, last: float) -> float:
    denom = max(abs(first), abs(last), _EPS)
    return abs(last - first) / denom


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


def _line_price_at(
    t: int,
    *,
    t0: int,
    p0: float,
    t1: int,
    p1: float,
) -> Optional[float]:
    """Linear interpolation / extrapolation through (t0, p0) and (t1, p1) at t."""
    if t1 == t0:
        # Degenerate: horizontal only if prices match; otherwise undefined slope
        if p0 == p1:
            return p0
        return None
    slope = (p1 - p0) / (t1 - t0)
    val = p0 + slope * (t - t0)
    if not _finite(val):
        return None
    return float(val)


def _classify_boundary(
    first_price: float,
    last_price: float,
    horizontal_tolerance: float,
) -> str:
    """
    Return 'horizontal', 'rising', or 'falling'.
    Horizontal is exclusive: if the relative change is within tolerance,
    the boundary is horizontal and must not also be rising/falling.
    """
    if _relative_change(first_price, last_price) <= horizontal_tolerance:
        return "horizontal"
    if last_price > first_price:
        return "rising"
    return "falling"


def _intersection_time(
    *,
    t_h0: int,
    p_h0: float,
    t_h1: int,
    p_h1: float,
    t_l0: int,
    p_l0: float,
    t_l1: int,
    p_l1: float,
) -> Optional[float]:
    """
    Time of finite intersection of upper line (h0→h1) and lower line (l0→l1).
    Returns None for parallel / coincident / non-finite cases.
    """
    # upper: p = p_h0 + s_h * (t - t_h0)
    # lower: p = p_l0 + s_l * (t - t_l0)
    if t_h1 == t_h0 or t_l1 == t_l0:
        return None
    s_h = (p_h1 - p_h0) / (t_h1 - t_h0)
    s_l = (p_l1 - p_l0) / (t_l1 - t_l0)
    if not (_finite(s_h) and _finite(s_l)):
        return None
    # p_h0 + s_h*(t - t_h0) = p_l0 + s_l*(t - t_l0)
    # s_h*t - s_l*t = p_l0 - p_h0 - s_l*t_l0 + s_h*t_h0
    ds = s_h - s_l
    if abs(ds) < _EPS:
        return None  # parallel
    t_int = (p_l0 - p_h0 - s_l * t_l0 + s_h * t_h0) / ds
    if not _finite(t_int):
        return None
    return float(t_int)


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


class TriangleDetector(PriceActionDetector):
    """
    Swing-based classical triangle pattern detector
    (ascending, descending, symmetrical).

    Configuration is immutable after construction. Invalid parameters raise
    ValueError (no silent coercion).
    """

    def __init__(
        self,
        *,
        boundary_tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
        horizontal_tolerance: float = DEFAULT_HORIZONTAL_TOLERANCE,
        max_bars_to_breakout: int = DEFAULT_MAX_BARS_TO_BREAKOUT,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "triangle",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._boundary_tolerance = _validate_tolerance(
            "boundary_tolerance", boundary_tolerance
        )
        self._horizontal_tolerance = _validate_tolerance(
            "horizontal_tolerance", horizontal_tolerance
        )
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
            "horizontal_tolerance": self._horizontal_tolerance,
            "max_bars_to_breakout": self._max_bars,
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
    def boundary_tolerance(self) -> float:
        return self._boundary_tolerance

    @property
    def horizontal_tolerance(self) -> float:
        return self._horizontal_tolerance

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
        Lower bound for a minimal triangle (four swings) to become confirmable:
        four swings each needing left+1+right bars, plus one post-final-anchor
        bar for breakout. Insufficient series yields no emission.
        """
        s = self._left + 1 + self._right
        return 4 * s + 1

    def config_dict(self) -> dict[str, Any]:
        return {
            "boundary_tolerance": self._boundary_tolerance,
            "horizontal_tolerance": self._horizontal_tolerance,
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
        """Detect triangles within one continuous closed-candle segment."""
        epoch_to_idx = {c.epoch: i for i, c in enumerate(closed)}

        swing_obs = self._swing.detect_all_historical(closed)
        highs = [o for o in swing_obs if o.type == "swing_high"]
        lows = [o for o in swing_obs if o.type == "swing_low"]

        def _key(o: Observation) -> tuple:
            et = _swing_event_time(o) or 0
            return (et, o.confirmed_at or 0, o.id)

        highs = sorted(highs, key=_key)
        lows = sorted(lows, key=_key)

        out: list[Observation] = []
        for anchors in self._iter_candidates(highs, lows):
            obs = self._try_confirm_candidate(
                anchors,
                closed=closed,
                epoch_to_idx=epoch_to_idx,
            )
            if obs is not None:
                out.append(obs)

        return out

    def _enumerate_candidates(
        self,
        highs: list[Observation],
        lows: list[Observation],
    ) -> list[list[Observation]]:
        """Materialize all candidates (tests / diagnostics). Same coverage as _iter_candidates."""
        return list(self._iter_candidates(highs, lows))

    def _iter_candidates(
        self,
        highs: list[Observation],
        lows: list[Observation],
    ):
        """
        Yield all alternating chronological subsequences with ≥2 highs and ≥2 lows.

        Semantics (aligned with Phase 5B):
          - Intermediate swings may be skipped.
          - Every swing in the continuous segment is eligible; history is never
            truncated by a fixed swing-count cap.
          - No maximum anchor-count restriction on a candidate.
          - Alternation is required in chronological order.
          - Each distinct anchor-ID combination is yielded at most once.

        Uniqueness: unique swing ids + strictly increasing index paths.
        Performance: suffix-count pruning; streaming yield.
        Worst-case runtime exponential in number of alternating swings.
        """
        raw: list[tuple[Observation, str, str]] = []
        for o in highs:
            et = _swing_event_time(o)
            if et is None or o.confirmed_at is None:
                continue
            if _swing_price(o) is None:
                continue
            raw.append((o, "H", o.id))
        for o in lows:
            et = _swing_event_time(o)
            if et is None or o.confirmed_at is None:
                continue
            if _swing_price(o) is None:
                continue
            raw.append((o, "L", o.id))

        raw.sort(
            key=lambda t: (
                _swing_event_time(t[0]) or 0,
                t[0].confirmed_at or 0,
                t[2],
            )
        )

        seen_ids: set[str] = set()
        entries: list[tuple[Observation, str, str]] = []
        for item in raw:
            if item[2] in seen_ids:
                continue
            seen_ids.add(item[2])
            entries.append(item)

        n = len(entries)
        if n < 4:
            return

        obs = [e[0] for e in entries]
        types = [e[1] for e in entries]
        ids = [e[2] for e in entries]

        suffix_h = [0] * (n + 1)
        suffix_l = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_h[i] = suffix_h[i + 1] + (1 if types[i] == "H" else 0)
            suffix_l[i] = suffix_l[i + 1] + (1 if types[i] == "L" else 0)

        path_idx: list[int] = []

        def rec(start: int, last_typ: str | None, nh: int, nl: int):
            if nh + suffix_h[start] < 2 or nl + suffix_l[start] < 2:
                return
            if len(path_idx) >= 4 and nh >= 2 and nl >= 2:
                yield [obs[i] for i in path_idx]
            for i in range(start, n):
                typ = types[i]
                if last_typ is not None and typ == last_typ:
                    continue
                next_nh = nh + (1 if typ == "H" else 0)
                next_nl = nl + (1 if typ == "L" else 0)
                if next_nh + suffix_h[i + 1] < 2 or next_nl + suffix_l[i + 1] < 2:
                    continue
                path_idx.append(i)
                yield from rec(i + 1, typ, next_nh, next_nl)
                path_idx.pop()

        yield from rec(0, None, 0, 0)

    def _try_confirm_candidate(
        self,
        anchors: list[Observation],
        *,
        closed: list[Candle],
        epoch_to_idx: dict[int, int],
    ) -> Optional[Observation]:
        """Validate geometry, classify, require forward convergence, then breakout."""
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
                high_obs.append(a)
            else:
                low_obs.append(a)

        if len(high_obs) < 2 or len(low_obs) < 2:
            return None
        if final_event is None or final_event not in epoch_to_idx:
            return None

        # Chronological order within highs / lows for first & last
        high_obs = sorted(
            high_obs,
            key=lambda a: (
                _swing_event_time(a) or 0,
                a.confirmed_at or 0,
                a.id,
            ),
        )
        low_obs = sorted(
            low_obs,
            key=lambda a: (
                _swing_event_time(a) or 0,
                a.confirmed_at or 0,
                a.id,
            ),
        )

        h0, h1 = high_obs[0], high_obs[-1]
        l0, l1 = low_obs[0], low_obs[-1]
        t_h0 = _swing_event_time(h0)
        t_h1 = _swing_event_time(h1)
        t_l0 = _swing_event_time(l0)
        t_l1 = _swing_event_time(l1)
        p_h0 = _swing_price(h0)
        p_h1 = _swing_price(h1)
        p_l0 = _swing_price(l0)
        p_l1 = _swing_price(l1)
        if None in (t_h0, t_h1, t_l0, t_l1, p_h0, p_h1, p_l0, p_l1):
            return None
        assert t_h0 is not None and t_h1 is not None
        assert t_l0 is not None and t_l1 is not None
        assert p_h0 is not None and p_h1 is not None
        assert p_l0 is not None and p_l1 is not None

        # Boundary fit for every selected anchor
        for a in high_obs:
            et = _swing_event_time(a)
            p = _swing_price(a)
            if et is None or p is None:
                return None
            bp = _line_price_at(et, t0=t_h0, p0=p_h0, t1=t_h1, p1=p_h1)
            if bp is None or not _within_tolerance(p, bp, self._boundary_tolerance):
                return None
        for a in low_obs:
            et = _swing_event_time(a)
            p = _swing_price(a)
            if et is None or p is None:
                return None
            bp = _line_price_at(et, t0=t_l0, p0=p_l0, t1=t_l1, p1=p_l1)
            if bp is None or not _within_tolerance(p, bp, self._boundary_tolerance):
                return None

        # Slope classification
        upper_cls = _classify_boundary(p_h0, p_h1, self._horizontal_tolerance)
        lower_cls = _classify_boundary(p_l0, p_l1, self._horizontal_tolerance)

        concept: Optional[str] = None
        if upper_cls == "horizontal" and lower_cls == "rising":
            concept = CONCEPT_ASCENDING
        elif upper_cls == "falling" and lower_cls == "horizontal":
            concept = CONCEPT_DESCENDING
        elif upper_cls == "falling" and lower_cls == "rising":
            concept = CONCEPT_SYMMETRICAL
        else:
            return None

        # Convergence: finite intersection strictly after final anchor
        t_int = _intersection_time(
            t_h0=t_h0,
            p_h0=p_h0,
            t_h1=t_h1,
            p_h1=p_h1,
            t_l0=t_l0,
            p_l0=p_l0,
            t_l1=t_l1,
            p_l1=p_l1,
        )
        if t_int is None or t_int <= final_event:
            return None

        all_conf = max(confs)
        final_idx = epoch_to_idx[final_event]

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

        intended = _PATTERN_INTENDED[concept]

        for _offset, candle in eligible:
            upper_at = _line_price_at(
                candle.epoch, t0=t_h0, p0=p_h0, t1=t_h1, p1=p_h1
            )
            lower_at = _line_price_at(
                candle.epoch, t0=t_l0, p0=p_l0, t1=t_l1, p1=p_l1
            )
            if upper_at is None or lower_at is None:
                return None  # malformed geometry
            if not (_finite(upper_at) and _finite(lower_at)):
                return None
            # If geometry is inverted or degenerate at evaluation time, reject
            if upper_at <= lower_at:
                return None

            close = candle.close
            broke_upper = close > upper_at
            broke_lower = close < lower_at

            # Same candle both sides → reject geometry rather than invent precedence
            if broke_upper and broke_lower:
                return None

            if intended == "upper":
                if broke_upper:
                    return self._make_obs(
                        concept=concept,
                        direction="bullish",
                        breakout=candle,
                        anchors=anchors,
                        high_obs=high_obs,
                        low_obs=low_obs,
                        upper_cls=upper_cls,
                        lower_cls=lower_cls,
                        upper_at_breakout=upper_at,
                        lower_at_breakout=lower_at,
                        t_int=t_int,
                    )
                if broke_lower:
                    # opposite-boundary invalidation
                    return None
            elif intended == "lower":
                if broke_lower:
                    return self._make_obs(
                        concept=concept,
                        direction="bearish",
                        breakout=candle,
                        anchors=anchors,
                        high_obs=high_obs,
                        low_obs=low_obs,
                        upper_cls=upper_cls,
                        lower_cls=lower_cls,
                        upper_at_breakout=upper_at,
                        lower_at_breakout=lower_at,
                        t_int=t_int,
                    )
                if broke_upper:
                    return None
            else:  # either (symmetrical)
                if broke_upper:
                    return self._make_obs(
                        concept=concept,
                        direction="bullish",
                        breakout=candle,
                        anchors=anchors,
                        high_obs=high_obs,
                        low_obs=low_obs,
                        upper_cls=upper_cls,
                        lower_cls=lower_cls,
                        upper_at_breakout=upper_at,
                        lower_at_breakout=lower_at,
                        t_int=t_int,
                    )
                if broke_lower:
                    return self._make_obs(
                        concept=concept,
                        direction="bearish",
                        breakout=candle,
                        anchors=anchors,
                        high_obs=high_obs,
                        low_obs=low_obs,
                        upper_cls=upper_cls,
                        lower_cls=lower_cls,
                        upper_at_breakout=upper_at,
                        lower_at_breakout=lower_at,
                        t_int=t_int,
                    )
            # Wick-only or close exactly on boundary does not confirm

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
        Eligible candles: completed bars strictly after the final anchor's
        candle, within max_bars, and only after all anchors are available
        (candle.epoch > all_confirmed_at). Offsets 1 .. max_bars inclusive.
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
        concept: str,
        direction: str,
        breakout: Candle,
        anchors: list[Observation],
        high_obs: list[Observation],
        low_obs: list[Observation],
        upper_cls: str,
        lower_cls: str,
        upper_at_breakout: float,
        lower_at_breakout: float,
        t_int: float,
    ) -> Observation:
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
        confirmed_at = breakout.epoch

        swing_types = [a.type for a in ordered]
        swing_prices = [_swing_price(a) for a in ordered]
        high_ids = [a.id for a in high_obs]
        low_ids = [a.id for a in low_obs]

        payload: dict[str, Any] = {
            "pattern": concept,
            "upper_classification": upper_cls,
            "lower_classification": lower_cls,
            "boundary_tolerance": self._boundary_tolerance,
            "horizontal_tolerance": self._horizontal_tolerance,
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
            "upper_boundary_at_breakout": upper_at_breakout,
            "lower_boundary_at_breakout": lower_at_breakout,
            "convergence_time": t_int,
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
