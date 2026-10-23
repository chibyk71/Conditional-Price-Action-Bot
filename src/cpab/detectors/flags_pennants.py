"""
Phase 5D — Swing-based flags and pennants.

Detects confirmed bullish_flag, bearish_flag, bullish_pennant, bearish_pennant
from Phase 2A swing observations after a qualifying impulse pole.

  - type: bullish_flag | bearish_flag | bullish_pennant | bearish_pennant
  - variant: swing_based_v1

Locked impulse-pole contract
  - Duration: 2 .. 10 candles inclusive (inclusive count of first and final).
  - Displacement = abs(close[end] - close[start]); zero displacement rejected.
  - Bullish pole: close[end] > close[start]; bearish: close[end] < close[start].
  - Strength: displacement >= pole_strength_mult * ATR.
  - ATR: pure Wilder, exactly 14 completed candles immediately preceding the
    pole (plus preceding close for first TR); never includes pole candles;
    continuous segment only; insufficient / gap / invalid OHLC → no qualify.
  - Pole eligible only after its final candle has closed.

Locked consolidation contract
  - Begins immediately after the pole's final candle (no intervening bars).
  - 3 .. 20 bars inclusive.
  - At least two distinct selected swing highs and two distinct swing lows.
  - Boundaries from first/last selected anchors (Phase 5C convention).
  - Boundary-fit relative tolerance (default 0.002).
  - Parallel-slope relative tolerance (default 0.002) for flags.
  - Pennants reuse Phase 5C triangle convergence (upper descending, lower
    ascending, finite intersection strictly after final anchor).

Flag direction
  - Bullish flag after bullish pole: boundaries slope downward or flat.
  - Bearish flag after bearish pole: boundaries slope upward or flat.
  - Flat (both |slopes| < eps) accepted as parallel.

Retracement
  - Max consolidation retracement normalized by pole close-to-close displacement
    must be <= max_pole_retracement (0.50).

Breakout / lifecycle (Phase 5 conventions)
  - Bullish: strict close above upper boundary.
  - Bearish: strict close below lower boundary.
  - Equality is not a breakout.
  - Eligible only when candle.epoch > max(anchor.confirmed_at).
  - Expiry: offsets 1 .. max_bars_to_breakout inclusive from final consolidation
    anchor candle.
  - Continuous segments independent; no gap bridging.
  - Exhaustive candidate enumeration; identical anchor combinations deduped
    only; overlapping non-identical candidates retained.
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
from cpab.detectors.wilder_atr import atr_before_pole
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.flags_pennants")

# Defaults (all participate in detector_config_id)
DEFAULT_MIN_POLE_BARS = 2
DEFAULT_MAX_POLE_BARS = 10
DEFAULT_POLE_ATR_LOOKBACK = 14
DEFAULT_POLE_STRENGTH_MULT = 1.5
DEFAULT_MAX_POLE_RETRACEMENT = 0.50
DEFAULT_MIN_CONSOLIDATION_BARS = 3
DEFAULT_MAX_CONSOLIDATION_BARS = 20
DEFAULT_BOUNDARY_TOLERANCE = 0.002
DEFAULT_PARALLEL_SLOPE_TOLERANCE = 0.002
DEFAULT_MAX_BARS_TO_BREAKOUT = 50
_EPS = 1e-12

VARIANT = "swing_based_v1"

CONCEPT_BULLISH_FLAG = "bullish_flag"
CONCEPT_BEARISH_FLAG = "bearish_flag"
CONCEPT_BULLISH_PENNANT = "bullish_pennant"
CONCEPT_BEARISH_PENNANT = "bearish_pennant"

ALL_CONCEPTS = (
    CONCEPT_BULLISH_FLAG,
    CONCEPT_BEARISH_FLAG,
    CONCEPT_BULLISH_PENNANT,
    CONCEPT_BEARISH_PENNANT,
)


def _validate_positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _validate_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    f = float(value)
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return f


def _validate_nonneg_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")
    f = float(value)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")
    return f


def _validate_max_bars_to_breakout(value: Any) -> int:
    return _validate_positive_int("max_bars_to_breakout", value, minimum=1)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def _within_tolerance(price: float, boundary: float, tolerance: float) -> bool:
    denom = max(abs(price), abs(boundary), _EPS)
    return abs(price - boundary) / denom <= tolerance


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
    if t1 == t0:
        if p0 == p1:
            return p0
        return None
    slope = (p1 - p0) / (t1 - t0)
    val = p0 + slope * (t - t0)
    if not _finite(val):
        return None
    return float(val)


def _slope(t0: int, p0: float, t1: int, p1: float) -> Optional[float]:
    if t1 == t0:
        return 0.0 if p0 == p1 else None
    s = (p1 - p0) / (t1 - t0)
    return float(s) if _finite(s) else None


def _are_parallel(
    s_u: Optional[float],
    s_l: Optional[float],
    parallel_tolerance: float,
) -> bool:
    """
    abs(s_u - s_l) / max(|s_u|, |s_l|, eps) <= parallel_tolerance.
    Both |slopes| < eps → parallel.
    """
    if s_u is None or s_l is None:
        return False
    if abs(s_u) < _EPS and abs(s_l) < _EPS:
        return True
    denom = max(abs(s_u), abs(s_l), _EPS)
    return abs(s_u - s_l) / denom <= parallel_tolerance


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
    if t_h1 == t_h0 or t_l1 == t_l0:
        return None
    s_h = (p_h1 - p_h0) / (t_h1 - t_h0)
    s_l = (p_l1 - p_l0) / (t_l1 - t_l0)
    if not (_finite(s_h) and _finite(s_l)):
        return None
    ds = s_h - s_l
    if abs(ds) < _EPS:
        return None
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
    pole_start_epoch: int,
    pole_end_epoch: int,
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
        f"p{pole_start_epoch}-{pole_end_epoch}",
        *anchor_ids,
    ]
    return "|".join(parts)


class FlagsPennantsDetector(PriceActionDetector):
    """
    Swing-based flag and pennant detector (Phase 5D).

    Configuration is immutable after construction. Invalid parameters raise
    ValueError (no silent coercion).
    """

    def __init__(
        self,
        *,
        min_pole_bars: int = DEFAULT_MIN_POLE_BARS,
        max_pole_bars: int = DEFAULT_MAX_POLE_BARS,
        pole_atr_lookback: int = DEFAULT_POLE_ATR_LOOKBACK,
        pole_strength_mult: float = DEFAULT_POLE_STRENGTH_MULT,
        max_pole_retracement: float = DEFAULT_MAX_POLE_RETRACEMENT,
        min_consolidation_bars: int = DEFAULT_MIN_CONSOLIDATION_BARS,
        max_consolidation_bars: int = DEFAULT_MAX_CONSOLIDATION_BARS,
        boundary_tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
        parallel_slope_tolerance: float = DEFAULT_PARALLEL_SLOPE_TOLERANCE,
        max_bars_to_breakout: int = DEFAULT_MAX_BARS_TO_BREAKOUT,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "flags_pennants",
    ) -> None:
        self._min_pole = _validate_positive_int("min_pole_bars", min_pole_bars, minimum=2)
        self._max_pole = _validate_positive_int("max_pole_bars", max_pole_bars, minimum=2)
        if self._min_pole > self._max_pole:
            raise ValueError(
                f"min_pole_bars ({self._min_pole}) > max_pole_bars ({self._max_pole})"
            )
        self._atr_lb = _validate_positive_int(
            "pole_atr_lookback", pole_atr_lookback, minimum=1
        )
        self._strength_mult = _validate_positive_float(
            "pole_strength_mult", pole_strength_mult
        )
        self._max_retr = _validate_nonneg_float(
            "max_pole_retracement", max_pole_retracement
        )
        self._min_cons = _validate_positive_int(
            "min_consolidation_bars", min_consolidation_bars, minimum=1
        )
        self._max_cons = _validate_positive_int(
            "max_consolidation_bars", max_consolidation_bars, minimum=1
        )
        if self._min_cons > self._max_cons:
            raise ValueError(
                f"min_consolidation_bars ({self._min_cons}) > "
                f"max_consolidation_bars ({self._max_cons})"
            )
        self._boundary_tol = _validate_nonneg_float(
            "boundary_tolerance", boundary_tolerance
        )
        self._parallel_tol = _validate_nonneg_float(
            "parallel_slope_tolerance", parallel_slope_tolerance
        )
        self._max_bars = _validate_max_bars_to_breakout(max_bars_to_breakout)
        self._left = _validate_strength("left_strength", left_strength)
        self._right = _validate_strength("right_strength", right_strength)
        self._tie_policy = _validate_tie_policy(tie_policy)
        self._detector_id = detector_id

        cfg = {
            "min_pole_bars": self._min_pole,
            "max_pole_bars": self._max_pole,
            "pole_atr_lookback": self._atr_lb,
            "pole_strength_mult": self._strength_mult,
            "max_pole_retracement": self._max_retr,
            "min_consolidation_bars": self._min_cons,
            "max_consolidation_bars": self._max_cons,
            "boundary_tolerance": self._boundary_tol,
            "parallel_slope_tolerance": self._parallel_tol,
            "max_bars_to_breakout": self._max_bars,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        self._detector_config_id = f"cfg:{digest}"
        self._config = cfg

        self._swing = SwingPivotDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
        )

    @property
    def name(self) -> str:
        return self._detector_id

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def min_history(self) -> int:
        # ATR lookback + min pole + min consolidation + swing strength + buffer.
        # max_bars_to_breakout is not required up-front; breakout may still be pending.
        return (
            self._atr_lb
            + 1
            + self._min_pole
            + self._min_cons
            + self._left
            + self._right
            + 4
        )

    def detect(self, candles: list[Candle]) -> list[Observation]:
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            return []
        step = timeframe_seconds(timeframe)
        if step is None:
            return []
        out: list[Observation] = []
        for segment in self._continuous_segments(closed, step=step):
            out.extend(self._reconstruct_segment(segment, step=step))
        return out

    @staticmethod
    def _continuous_segments(
        candles: list[Candle],
        *,
        step: int,
    ) -> list[list[Candle]]:
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
        if len(closed) < self._atr_lb + 1 + self._min_pole + self._min_cons:
            return []

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
        # Enumerate qualifying poles then consolidation candidates.
        for pole in self._iter_poles(closed, step=step):
            cons_start = pole["end_idx"] + 1
            if cons_start >= len(closed):
                continue
            for cons_end in range(
                cons_start + self._min_cons - 1,
                min(cons_start + self._max_cons, len(closed)),
            ):
                # Consolidation candles: closed[cons_start : cons_end+1]
                cons_candles = closed[cons_start : cons_end + 1]
                if len(cons_candles) < self._min_cons:
                    continue
                # Swings whose event times fall inside the consolidation window.
                cons_epochs = {c.epoch for c in cons_candles}
                cons_highs = [
                    o
                    for o in highs
                    if (_swing_event_time(o) or -1) in cons_epochs
                ]
                cons_lows = [
                    o
                    for o in lows
                    if (_swing_event_time(o) or -1) in cons_epochs
                ]
                if len(cons_highs) < 2 or len(cons_lows) < 2:
                    continue

                for anchors in self._iter_candidates(cons_highs, cons_lows):
                    obs = self._try_confirm(
                        pole=pole,
                        anchors=anchors,
                        closed=closed,
                        epoch_to_idx=epoch_to_idx,
                        cons_start=cons_start,
                        cons_end=cons_end,
                    )
                    if obs is not None:
                        out.append(obs)
        return out

    def _iter_poles(
        self,
        closed: list[Candle],
        *,
        step: int,
    ):
        """
        Yield qualifying poles: dict with start_idx, end_idx, direction,
        displacement, atr, start_close, end_close, start_epoch, end_epoch.
        """
        n = len(closed)
        # Pole needs ATR lookback + 1 before it, and pole itself.
        min_start = self._atr_lb + 1
        for start in range(min_start, n):
            for length in range(self._min_pole, self._max_pole + 1):
                end = start + length - 1
                if end >= n:
                    break
                # Inclusive duration check already by construction.
                start_c = closed[start]
                end_c = closed[end]
                disp = abs(end_c.close - start_c.close)
                if disp <= 0 or not _finite(disp):
                    continue
                if end_c.close > start_c.close:
                    direction = "bullish"
                elif end_c.close < start_c.close:
                    direction = "bearish"
                else:
                    continue

                atr = atr_before_pole(
                    closed,
                    start,
                    lookback=self._atr_lb,
                    step=step,
                )
                if atr is None or atr <= 0:
                    continue
                threshold = self._strength_mult * atr
                if disp < threshold:
                    continue  # strict? contract: >= ; equality qualifies
                # Contract: displacement >= pole_strength_mult * ATR
                # (equality at threshold qualifies)

                yield {
                    "start_idx": start,
                    "end_idx": end,
                    "direction": direction,
                    "displacement": disp,
                    "atr": atr,
                    "threshold": threshold,
                    "start_close": start_c.close,
                    "end_close": end_c.close,
                    "start_epoch": start_c.epoch,
                    "end_epoch": end_c.epoch,
                }

    def _iter_candidates(
        self,
        highs: list[Observation],
        lows: list[Observation],
    ):
        """
        Yield all alternating chronological subsequences with ≥2 highs and ≥2 lows.
        Same semantics as Phase 5C (intermediate swings may be skipped).
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

    def _try_confirm(
        self,
        *,
        pole: dict[str, Any],
        anchors: list[Observation],
        closed: list[Candle],
        epoch_to_idx: dict[int, int],
        cons_start: int,
        cons_end: int,
    ) -> Optional[Observation]:
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

        # Boundary fit
        for a in high_obs:
            et = _swing_event_time(a)
            p = _swing_price(a)
            if et is None or p is None:
                return None
            bp = _line_price_at(et, t0=t_h0, p0=p_h0, t1=t_h1, p1=p_h1)
            if bp is None or not _within_tolerance(p, bp, self._boundary_tol):
                return None
        for a in low_obs:
            et = _swing_event_time(a)
            p = _swing_price(a)
            if et is None or p is None:
                return None
            bp = _line_price_at(et, t0=t_l0, p0=p_l0, t1=t_l1, p1=p_l1)
            if bp is None or not _within_tolerance(p, bp, self._boundary_tol):
                return None

        s_u = _slope(t_h0, p_h0, t_h1, p_h1)
        s_l = _slope(t_l0, p_l0, t_l1, p_l1)

        pole_dir = pole["direction"]
        concept: Optional[str] = None
        is_flag = False

        # Flag geometry: approximately parallel; direction relative to pole
        if _are_parallel(s_u, s_l, self._parallel_tol):
            # Flat or counter-trend slope permitted.
            # Bullish flag: after bullish pole; boundaries downward or flat.
            # Bearish flag: after bearish pole; boundaries upward or flat.
            if pole_dir == "bullish":
                # Both slopes <= ~0 (downward or flat). Allow mild positive
                # within parallel tolerance already enforced; classify by
                # average slope sign relative to zero with eps.
                avg = ((s_u or 0.0) + (s_l or 0.0)) / 2.0
                if avg <= _EPS:  # downward or flat
                    concept = CONCEPT_BULLISH_FLAG
                    is_flag = True
            else:
                avg = ((s_u or 0.0) + (s_l or 0.0)) / 2.0
                if avg >= -_EPS:  # upward or flat
                    concept = CONCEPT_BEARISH_FLAG
                    is_flag = True

        # Pennant geometry: converging (upper descending, lower ascending)
        if concept is None:
            if (
                s_u is not None
                and s_l is not None
                and s_u < -_EPS
                and s_l > _EPS
            ):
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
                if t_int is not None and t_int > final_event:
                    if pole_dir == "bullish":
                        concept = CONCEPT_BULLISH_PENNANT
                    else:
                        concept = CONCEPT_BEARISH_PENNANT

        if concept is None:
            return None

        # Retracement check
        cons_candles = closed[cons_start : cons_end + 1]
        if pole_dir == "bullish":
            lowest = min(c.low for c in cons_candles)
            retr = (pole["end_close"] - lowest) / pole["displacement"]
        else:
            highest = max(c.high for c in cons_candles)
            retr = (highest - pole["end_close"]) / pole["displacement"]
        if not _finite(retr) or retr > self._max_retr + _EPS:
            return None

        all_conf = max(confs)
        final_idx = epoch_to_idx[final_event]

        eligible = self._candle_after_all_available(
            closed,
            epoch_to_idx,
            final_anchor_event=final_event,
            all_confirmed_at=all_conf,
            max_bars=self._max_bars,
        )

        bullish_pattern = concept in (
            CONCEPT_BULLISH_FLAG,
            CONCEPT_BULLISH_PENNANT,
        )

        for _offset, candle in eligible:
            upper_at = _line_price_at(
                candle.epoch, t0=t_h0, p0=p_h0, t1=t_h1, p1=p_h1
            )
            lower_at = _line_price_at(
                candle.epoch, t0=t_l0, p0=p_l0, t1=t_l1, p1=p_l1
            )
            if upper_at is None or lower_at is None:
                continue
            broke_upper = candle.close > upper_at
            broke_lower = candle.close < lower_at

            if bullish_pattern:
                if broke_upper:
                    return self._make_obs(
                        concept=concept,
                        direction="bullish",
                        breakout=candle,
                        anchors=anchors,
                        high_obs=high_obs,
                        low_obs=low_obs,
                        pole=pole,
                        upper_at=upper_at,
                        lower_at=lower_at,
                        retracement=retr,
                        is_flag=is_flag,
                        s_u=s_u,
                        s_l=s_l,
                    )
                if broke_lower:
                    # opposite-boundary invalidation
                    return None
            else:
                if broke_lower:
                    return self._make_obs(
                        concept=concept,
                        direction="bearish",
                        breakout=candle,
                        anchors=anchors,
                        high_obs=high_obs,
                        low_obs=low_obs,
                        pole=pole,
                        upper_at=upper_at,
                        lower_at=lower_at,
                        retracement=retr,
                        is_flag=is_flag,
                        s_u=s_u,
                        s_l=s_l,
                    )
                if broke_upper:
                    return None

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
        pole: dict[str, Any],
        upper_at: float,
        lower_at: float,
        retracement: float,
        is_flag: bool,
        s_u: Optional[float],
        s_l: Optional[float],
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

        payload: dict[str, Any] = {
            "pattern": concept,
            "is_flag": is_flag,
            "pole_direction": pole["direction"],
            "pole_start_epoch": pole["start_epoch"],
            "pole_end_epoch": pole["end_epoch"],
            "pole_displacement": pole["displacement"],
            "pole_atr": pole["atr"],
            "pole_strength_threshold": pole["threshold"],
            "retracement": retracement,
            "max_pole_retracement": self._max_retr,
            "boundary_tolerance": self._boundary_tol,
            "parallel_slope_tolerance": self._parallel_tol,
            "max_bars_to_breakout": self._max_bars,
            "upper_slope": s_u,
            "lower_slope": s_l,
            "anchor_swing_ids": list(anchor_ids),
            "high_swing_ids": [a.id for a in high_obs],
            "low_swing_ids": [a.id for a in low_obs],
            "breakout_direction": direction,
            "breakout_close": breakout.close,
            "breakout_candle_epoch": breakout.epoch,
            "breakout_ohlc": breakout.ohlc_tuple(),
            "upper_boundary_at_breakout": upper_at,
            "lower_boundary_at_breakout": lower_at,
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
            pole_start_epoch=pole["start_epoch"],
            pole_end_epoch=pole["end_epoch"],
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
