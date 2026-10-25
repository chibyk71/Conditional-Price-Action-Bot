"""
Phase 6A — Behavioral / volatility regime detectors.

Exactly three detectors:

  - atr_expansion   (variant: rolling_atr_v1)
  - atr_contraction (variant: rolling_atr_v1)
  - range_expansion (variant: true_range_v1)
  - impulse (variant: atr_normalized_candle_v1)

Locked contract
---------------
ATR period: 14 (Wilder, via wilder_atr.py unchanged).
ATR baseline: arithmetic mean of the preceding 20 rolling ATR values,
  excluding the current value.
Expansion threshold: 1.25× baseline, crossing-only.
Contraction threshold: 0.75× baseline, crossing-only.
Range expansion: current true range ≥ 1.5× the preceding ATR; emit for every
  qualifying completed candle (not crossing-only).
Warm-up: 35 bars for ATR regimes; 16 bars for range expansion.
Direction: None for all three.
Strict continuous-segment isolation; no gap bridging.
No lifecycle, trading signals, volume logic, or TA-Lib dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any, Optional, Sequence

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.swing import timeframe_seconds
from cpab.detectors.wilder_atr import true_range, wilder_atr, _valid_ohlc, _finite
from cpab.models import Candle, Observation

logger = logging.getLogger(__name__)

# Locked constants (participate in detector_config_id)
ATR_PERIOD = 14
BASELINE_WINDOW = 20
EXPANSION_MULT = 1.25
CONTRACTION_MULT = 0.75
RANGE_EXPANSION_MULT = 1.5

# Warm-up lengths (derived; documented for callers / registry)
# ATR regimes: first ATR at index 14; need 20 preceding ATRs → index 34 → 35 bars
MIN_HISTORY_ATR_REGIME = ATR_PERIOD + BASELINE_WINDOW + 1  # 35
# Range expansion: preceding ATR at index i-1 needs 15 bars ending there → i >= 15 → 16 bars
MIN_HISTORY_RANGE = ATR_PERIOD + 2  # 16

VARIANT_ROLLING_ATR = "rolling_atr_v1"
VARIANT_TRUE_RANGE = "true_range_v1"

_EPS = 1e-12


def _config_id(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"cfg:{digest}"


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


def _rolling_atr_series(
    segment: Sequence[Candle],
    *,
    period: int = ATR_PERIOD,
    step: Optional[int] = None,
) -> list[Optional[float]]:
    """
    Genuine Wilder-smoothed ATR series ending at each index.

    Seeding:
      Arithmetic mean of the first `period` true ranges in a continuous valid run.
      Requires `period + 1` consecutive valid candles (the first supplies prev_close).
      Index of the seed ATR is `period` bars after the start of the valid run.

    Smoothing:
      For subsequent valid bars in the continuous run:
        atr_i = atr_{i-1} * (period - 1) / period + tr_i * (1 / period)

    Gaps & Invalid OHLC:
      Any gap (when `step` is specified) or invalid OHLC terminates the continuous
      run. The invalid bar receives None. A fresh run must accumulate `period + 1`
      consecutive valid bars to seed a new ATR (invalid-window recovery).
    """
    n = len(segment)
    out: list[Optional[float]] = [None] * n
    if period < 1 or n < period + 1:
        return out

    run_start: Optional[int] = None
    last_atr: Optional[float] = None
    seed_trs: list[float] = []

    for i in range(n):
        c = segment[i]
        if not _valid_ohlc(c):
            run_start = None
            last_atr = None
            seed_trs = []
            continue

        if run_start is None:
            run_start = i
            last_atr = None
            seed_trs = []
            continue

        if step is not None and (c.epoch - segment[i - 1].epoch != step):
            run_start = i
            last_atr = None
            seed_trs = []
            continue

        tr = true_range(c.high, c.low, segment[i - 1].close)
        if not _finite(tr) or tr < 0:
            run_start = None
            last_atr = None
            seed_trs = []
            continue

        if last_atr is None:
            seed_trs.append(tr)
            if len(seed_trs) == period:
                initial_atr = sum(seed_trs) / period
                if _finite(initial_atr):
                    last_atr = float(initial_atr)
                    out[i] = last_atr
                    seed_trs = []
                else:
                    run_start = None
                    last_atr = None
                    seed_trs = []
        else:
            new_atr = (last_atr * (period - 1) + tr) / period
            if _finite(new_atr):
                last_atr = float(new_atr)
                out[i] = last_atr
            else:
                run_start = None
                last_atr = None
                seed_trs = []

    return out


def _baseline_mean(
    atrs: Sequence[Optional[float]],
    end_exclusive: int,
    window: int = BASELINE_WINDOW,
) -> Optional[float]:
    """
    Mean of the `window` ATR values immediately preceding `end_exclusive`.
    All values must be present and finite; otherwise None.
    """
    start = end_exclusive - window
    if start < 0:
        return None
    vals: list[float] = []
    for j in range(start, end_exclusive):
        v = atrs[j]
        if v is None or not _finite(v):
            return None
        vals.append(float(v))
    if len(vals) != window:
        return None
    mean = sum(vals) / window
    if not _finite(mean):
        return None
    return float(mean)


# ---------------------------------------------------------------------------
# Shared ATR-regime base (expansion / contraction)
# ---------------------------------------------------------------------------


class _ATRRegimeDetector(PriceActionDetector):
    """Base for atr_expansion and atr_contraction (crossing-only)."""

    _concept: str
    _threshold: float
    _crossing: str  # "above" | "below"

    def __init__(
        self,
        *,
        atr_period: int = ATR_PERIOD,
        baseline_window: int = BASELINE_WINDOW,
        threshold: Optional[float] = None,
        detector_id: Optional[str] = None,
    ) -> None:
        if atr_period != ATR_PERIOD:
            raise ValueError(
                f"atr_period is locked to {ATR_PERIOD}, got {atr_period!r}"
            )
        if baseline_window != BASELINE_WINDOW:
            raise ValueError(
                f"baseline_window is locked to {BASELINE_WINDOW}, got {baseline_window!r}"
            )
        thr = self._threshold if threshold is None else float(threshold)
        if thr != self._threshold:
            raise ValueError(
                f"threshold is locked to {self._threshold}, got {threshold!r}"
            )
        self._atr_period = ATR_PERIOD
        self._baseline_window = BASELINE_WINDOW
        self._thr = self._threshold
        self._detector_id = detector_id or self._concept

        cfg = {
            "atr_period": self._atr_period,
            "baseline_window": self._baseline_window,
            "threshold": self._thr,
            "crossing": self._crossing,
            "variant": VARIANT_ROLLING_ATR,
        }
        self._detector_config_id = _config_id(cfg)
        self._config = cfg

    @property
    def name(self) -> str:
        return self._detector_id

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def min_history(self) -> int:
        return MIN_HISTORY_ATR_REGIME

    def config_dict(self) -> dict[str, Any]:
        return dict(self._config)

    def detect(self, candles: list[Candle]) -> list[Observation]:
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        all_obs = self._reconstruct(closed)
        latest = closed[-1].epoch
        return [o for o in all_obs if o.confirmed_at == latest]

    def detect_all_historical(self, candles: list[Candle]) -> list[Observation]:
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        return self._reconstruct(closed)

    def _reconstruct(self, closed: list[Candle]) -> list[Observation]:
        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe; skipping %s", self._concept)
            return []
        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            return []
        out: list[Observation] = []
        for segment in _continuous_segments(closed, step=step):
            out.extend(self._reconstruct_segment(segment, step=step))
        return out

    def _reconstruct_segment(
        self,
        segment: list[Candle],
        *,
        step: int,
    ) -> list[Observation]:
        if len(segment) < self.min_history:
            return []
        atrs = _rolling_atr_series(segment, period=self._atr_period, step=step)
        out: list[Observation] = []
        # First evaluable index: period + baseline_window
        start_i = self._atr_period + self._baseline_window
        for i in range(start_i, len(segment)):
            cur = atrs[i]
            if cur is None or not _finite(cur):
                continue
            baseline = _baseline_mean(atrs, i, window=self._baseline_window)
            if baseline is None:
                continue
            # Zero / non-positive baseline: undefined ratio → skip
            if baseline <= _EPS:
                continue
            ratio = cur / baseline
            if not _finite(ratio):
                continue

            # Current regime membership
            if self._crossing == "above":
                now = ratio >= self._thr
            else:
                now = ratio <= self._thr
            if not now:
                continue

            # Crossing requires a *demonstrably opposite* previous evaluable ratio.
            # Unavailable / invalid previous state is unknown — not a crossing.
            prev_i = i - 1
            prev_in_regime: Optional[bool] = None
            if prev_i >= start_i:
                prev_atr = atrs[prev_i]
                prev_base = _baseline_mean(atrs, prev_i, window=self._baseline_window)
                if (
                    prev_atr is not None
                    and _finite(prev_atr)
                    and prev_base is not None
                    and prev_base > _EPS
                ):
                    prev_ratio = prev_atr / prev_base
                    if _finite(prev_ratio):
                        if self._crossing == "above":
                            prev_in_regime = prev_ratio >= self._thr
                        else:
                            prev_in_regime = prev_ratio <= self._thr

            if prev_in_regime is None or prev_in_regime:
                # Unknown previous state, or already in regime → no proven crossing
                continue

            c = segment[i]
            payload = {
                "atr": float(cur),
                "baseline": float(baseline),
                "ratio": float(ratio),
                "threshold": float(self._thr),
                "atr_period": self._atr_period,
                "baseline_window": self._baseline_window,
            }
            out.append(
                Observation.create(
                    type=self._concept,
                    symbol=c.symbol,
                    timeframe=c.timeframe,
                    candle_epoch=c.epoch,
                    event_time=c.epoch,
                    confirmed_at=c.epoch,
                    developing=False,
                    direction=None,
                    variant=VARIANT_ROLLING_ATR,
                    detector_id=self._detector_id,
                    detector_config_id=self._detector_config_id,
                    payload=payload,
                    metadata=payload,
                )
            )
        return out


class ATRExpansionDetector(_ATRRegimeDetector):
    """Emit atr_expansion on crossing of ATR ≥ 1.25× preceding 20-ATR mean."""

    _concept = "atr_expansion"
    _threshold = EXPANSION_MULT
    _crossing = "above"

    def __init__(
        self,
        *,
        atr_period: int = ATR_PERIOD,
        baseline_window: int = BASELINE_WINDOW,
        threshold: float = EXPANSION_MULT,
        detector_id: str = "atr_expansion",
    ) -> None:
        super().__init__(
            atr_period=atr_period,
            baseline_window=baseline_window,
            threshold=threshold,
            detector_id=detector_id,
        )


class ATRContractionDetector(_ATRRegimeDetector):
    """Emit atr_contraction on crossing of ATR ≤ 0.75× preceding 20-ATR mean."""

    _concept = "atr_contraction"
    _threshold = CONTRACTION_MULT
    _crossing = "below"

    def __init__(
        self,
        *,
        atr_period: int = ATR_PERIOD,
        baseline_window: int = BASELINE_WINDOW,
        threshold: float = CONTRACTION_MULT,
        detector_id: str = "atr_contraction",
    ) -> None:
        super().__init__(
            atr_period=atr_period,
            baseline_window=baseline_window,
            threshold=threshold,
            detector_id=detector_id,
        )


# ---------------------------------------------------------------------------
# Range expansion
# ---------------------------------------------------------------------------


class RangeExpansionDetector(PriceActionDetector):
    """
    Emit range_expansion when current true range ≥ 1.5× preceding ATR.

    Emitted for every qualifying completed candle (not crossing-only).
    """

    def __init__(
        self,
        *,
        atr_period: int = ATR_PERIOD,
        threshold: float = RANGE_EXPANSION_MULT,
        detector_id: str = "range_expansion",
    ) -> None:
        if atr_period != ATR_PERIOD:
            raise ValueError(
                f"atr_period is locked to {ATR_PERIOD}, got {atr_period!r}"
            )
        if float(threshold) != RANGE_EXPANSION_MULT:
            raise ValueError(
                f"threshold is locked to {RANGE_EXPANSION_MULT}, got {threshold!r}"
            )
        self._atr_period = ATR_PERIOD
        self._thr = RANGE_EXPANSION_MULT
        self._detector_id = detector_id
        cfg = {
            "atr_period": self._atr_period,
            "threshold": self._thr,
            "variant": VARIANT_TRUE_RANGE,
        }
        self._detector_config_id = _config_id(cfg)
        self._config = cfg

    @property
    def name(self) -> str:
        return self._detector_id

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def min_history(self) -> int:
        return MIN_HISTORY_RANGE

    def config_dict(self) -> dict[str, Any]:
        return dict(self._config)

    def detect(self, candles: list[Candle]) -> list[Observation]:
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        all_obs = self._reconstruct(closed)
        latest = closed[-1].epoch
        return [o for o in all_obs if o.confirmed_at == latest]

    def detect_all_historical(self, candles: list[Candle]) -> list[Observation]:
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        return self._reconstruct(closed)

    def _reconstruct(self, closed: list[Candle]) -> list[Observation]:
        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe; skipping range_expansion")
            return []
        step = timeframe_seconds(timeframe)
        if step is None or step <= 0:
            return []
        out: list[Observation] = []
        for segment in _continuous_segments(closed, step=step):
            out.extend(self._reconstruct_segment(segment, step=step))
        return out

    def _reconstruct_segment(
        self,
        segment: list[Candle],
        *,
        step: int,
    ) -> list[Observation]:
        if len(segment) < self.min_history:
            return []
        atrs = _rolling_atr_series(segment, period=self._atr_period, step=step)
        out: list[Observation] = []
        # Index i needs preceding ATR at i-1 → i >= period + 1
        for i in range(self._atr_period + 1, len(segment)):
            prev_atr = atrs[i - 1]
            if prev_atr is None or not _finite(prev_atr) or prev_atr <= _EPS:
                continue
            c = segment[i]
            if not _valid_ohlc(c):
                continue
            prev_c = segment[i - 1]
            if not _valid_ohlc(prev_c):
                continue
            # Continuity already guaranteed by segment, but re-check step for safety
            if c.epoch - prev_c.epoch != step:
                continue
            tr = true_range(c.high, c.low, prev_c.close)
            if not _finite(tr) or tr < 0:
                continue
            ratio = tr / prev_atr
            if not _finite(ratio) or ratio < self._thr:
                continue
            payload = {
                "true_range": float(tr),
                "preceding_atr": float(prev_atr),
                "ratio": float(ratio),
                "threshold": float(self._thr),
                "atr_period": self._atr_period,
            }
            out.append(
                Observation.create(
                    type="range_expansion",
                    symbol=c.symbol,
                    timeframe=c.timeframe,
                    candle_epoch=c.epoch,
                    event_time=c.epoch,
                    confirmed_at=c.epoch,
                    developing=False,
                    direction=None,
                    variant=VARIANT_TRUE_RANGE,
                    detector_id=self._detector_id,
                    detector_config_id=self._detector_config_id,
                    payload=payload,
                    metadata=payload,
                )
            )
        return out
