"""
TA-Lib pattern adapter for CPAB Phase 1.

Separates:
  1. Raw TA-Lib pattern recognition (integer result for the latest completed bar).
  2. Translation of a documented raw result into a canonical CPAB Observation.

TA-Lib is a required dependency. There is no silent fallback to hand-written
detectors. Forming candles are rejected at the adapter boundary. Insufficient
history yields an explicit non-emission outcome (no Observation). A raw zero
result means no pattern observation is emitted.

Minimum history is derived from TA-Lib's own lookback for each pattern:
  min_history = Function(pattern).lookback + 1
That is the number of completed candles required before the latest bar can be
a valid (non-unstable-period) evaluation of the pattern. Unsupported pattern
names are rejected rather than assigned an arbitrary default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from cpab.detectors.base import PriceActionDetector
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.talib_adapter")

# Stable aliases → TA-Lib function name (1:1). Only these aliases are accepted.
_PATTERN_ALIASES: dict[str, str] = {
    "engulfing": "CDLENGULFING",
    "doji": "CDLDOJI",
    "hammer": "CDLHAMMER",
    "shooting_star": "CDLSHOOTINGSTAR",
    "harami": "CDLHARAMI",
    "piercing": "CDLPIERCING",
    "morning_star": "CDLMORNINGSTAR",
    "evening_star": "CDLEVENINGSTAR",
}

# Supported CDL function names (explicit allow-list). Lookbacks are NOT hard-coded
# here; they are read from TA-Lib via Function(name).lookback at runtime.
_SUPPORTED_PATTERNS: frozenset[str] = frozenset(_PATTERN_ALIASES.values())


class AdapterOutcomeKind(str, Enum):
    PATTERN = "PATTERN"
    NO_PATTERN = "NO_PATTERN"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    FORMING_CANDLE = "FORMING_CANDLE"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"


@dataclass(frozen=True)
class AdapterRawResult:
    kind: AdapterOutcomeKind
    pattern: str
    raw_talib_value: Optional[int] = None
    candle_epoch: Optional[int] = None
    message: Optional[str] = None
    detail: Optional[str] = None


def resolve_pattern_name(pattern: str) -> str:
    """
    Resolve alias or CDL name to a canonical TA-Lib function name.
    Raises ValueError for unsupported patterns (no arbitrary default).
    """
    key = pattern.strip().upper()
    if key in _SUPPORTED_PATTERNS:
        return key
    alias = _PATTERN_ALIASES.get(pattern.strip().lower())
    if alias is not None:
        return alias
    cdl = f"CDL{key}" if not key.startswith("CDL") else key
    if cdl in _SUPPORTED_PATTERNS:
        return cdl
    raise ValueError(
        f"Unsupported TA-Lib pattern: {pattern!r}. "
        f"Supported: {sorted(_SUPPORTED_PATTERNS)}"
    )


def talib_lookback(pattern: str) -> int:
    """
    Computational lookback from TA-Lib itself (Function.lookback).
    This is the number of leading bars in the unstable period for the pattern.
    """
    from talib.abstract import Function

    name = resolve_pattern_name(pattern)
    return int(Function(name).lookback)


def min_history_for(pattern: str) -> int:
    """
    Minimum completed candles required to evaluate the *latest* bar.

    Derived from TA-Lib: lookback + 1 (lookback preceding bars + the signal bar).
    Raises ValueError for unsupported patterns.
    """
    return talib_lookback(pattern) + 1


def _get_talib_fn(pattern: str) -> Callable[..., np.ndarray]:
    import talib

    name = resolve_pattern_name(pattern)
    fn = getattr(talib, name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"TA-Lib function not available: {name!r}")
    return fn


def candles_to_ohlc_arrays(candles: list[Candle]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(candles)
    o = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)
    l = np.empty(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    for i, candle in enumerate(candles):
        o[i] = candle.open
        h[i] = candle.high
        l[i] = candle.low
        c[i] = candle.close
    return o, h, l, c


def evaluate_raw_series(candles: list[Candle], pattern: str) -> np.ndarray:
    """
    Full TA-Lib integer array for the series (one value per bar).
    Used by causality tests; production detect() uses evaluate_raw (latest only).
    Raises ValueError for unsupported patterns.
    """
    name = resolve_pattern_name(pattern)
    if any(not c.is_closed for c in candles):
        raise ValueError("Forming candles are not eligible for pattern evaluation")
    fn = _get_talib_fn(name)
    o, h, l, c = candles_to_ohlc_arrays(candles)
    arr = fn(o, h, l, c)
    return np.asarray(arr, dtype=np.int64)


def evaluate_raw(candles: list[Candle], pattern: str) -> AdapterRawResult:
    try:
        resolved = resolve_pattern_name(pattern)
    except ValueError as exc:
        return AdapterRawResult(
            kind=AdapterOutcomeKind.EXECUTION_FAILURE,
            pattern=pattern.strip().upper(),
            message=str(exc),
            detail=type(exc).__name__,
        )

    if any(not c.is_closed for c in candles):
        return AdapterRawResult(
            kind=AdapterOutcomeKind.FORMING_CANDLE,
            pattern=resolved,
            message="Forming candles are not eligible for pattern evaluation",
        )

    need = min_history_for(resolved)
    if len(candles) < need:
        return AdapterRawResult(
            kind=AdapterOutcomeKind.INSUFFICIENT_HISTORY,
            pattern=resolved,
            message=f"Need at least {need} completed candles (TA-Lib lookback+1), got {len(candles)}",
        )

    try:
        arr = evaluate_raw_series(candles, resolved)
        if arr is None or len(arr) == 0:
            return AdapterRawResult(
                kind=AdapterOutcomeKind.EXECUTION_FAILURE,
                pattern=resolved,
                message="TA-Lib returned empty result",
            )
        raw = int(arr[-1])
        latest = candles[-1]
        if raw == 0:
            return AdapterRawResult(
                kind=AdapterOutcomeKind.NO_PATTERN,
                pattern=resolved,
                raw_talib_value=0,
                candle_epoch=latest.epoch,
            )
        return AdapterRawResult(
            kind=AdapterOutcomeKind.PATTERN,
            pattern=resolved,
            raw_talib_value=raw,
            candle_epoch=latest.epoch,
        )
    except Exception as exc:
        logger.warning("TA-Lib evaluation failure pattern=%s: %s", resolved, exc, exc_info=True)
        return AdapterRawResult(
            kind=AdapterOutcomeKind.EXECUTION_FAILURE,
            pattern=resolved,
            message=str(exc),
            detail=type(exc).__name__,
        )


def _direction_from_raw(raw: int) -> Optional[str]:
    if raw > 0:
        return "bullish"
    if raw < 0:
        return "bearish"
    return None


def _observation_type(pattern: str, direction: Optional[str]) -> str:
    base = pattern.lower().replace("cdl", "")
    if direction == "bullish":
        return f"bullish_{base}"
    if direction == "bearish":
        return f"bearish_{base}"
    return base


def raw_to_observation(
    raw: AdapterRawResult,
    candle: Candle,
    *,
    detector_id: str,
    detector_config_id: Optional[str] = None,
) -> Optional[Observation]:
    if raw.kind != AdapterOutcomeKind.PATTERN or raw.raw_talib_value is None:
        return None
    if raw.raw_talib_value == 0:
        return None
    direction = _direction_from_raw(raw.raw_talib_value)
    obs_type = _observation_type(raw.pattern, direction)
    payload: dict[str, Any] = {
        "raw_talib_value": raw.raw_talib_value,
        "talib_pattern": raw.pattern,
        "detector": detector_id,
    }
    return Observation.create(
        type=obs_type,
        symbol=candle.symbol,
        timeframe=candle.timeframe,
        candle_epoch=candle.epoch,
        event_time=candle.epoch,
        confirmed_at=candle.epoch,
        developing=False,
        direction=direction,
        detector_id=detector_id,
        detector_config_id=detector_config_id,
        payload=payload,
        metadata=payload,
    )


class TalibPatternDetector(PriceActionDetector):
    def __init__(
        self,
        pattern: str = "CDLENGULFING",
        *,
        detector_id: Optional[str] = None,
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._pattern = resolve_pattern_name(pattern)
        self._detector_id = detector_id or f"talib:{self._pattern.lower()}"
        self._detector_config_id = detector_config_id

    @property
    def name(self) -> str:
        return self._detector_id

    @property
    def pattern(self) -> str:
        return self._pattern

    @property
    def min_history(self) -> int:
        return min_history_for(self._pattern)

    def evaluate_raw(self, candles: list[Candle]) -> AdapterRawResult:
        return evaluate_raw(candles, self._pattern)

    def detect(self, candles: list[Candle]) -> list[Observation]:
        raw = evaluate_raw(candles, self._pattern)
        if raw.kind == AdapterOutcomeKind.FORMING_CANDLE:
            logger.debug("Rejecting forming candles for %s", self._pattern)
            return []
        if raw.kind == AdapterOutcomeKind.INSUFFICIENT_HISTORY:
            logger.debug("Insufficient history for %s: %s", self._pattern, raw.message)
            return []
        if raw.kind == AdapterOutcomeKind.EXECUTION_FAILURE:
            logger.warning("Controlled failure for %s: %s", self._pattern, raw.message)
            return []
        if raw.kind == AdapterOutcomeKind.NO_PATTERN:
            return []
        latest = candles[-1]
        obs = raw_to_observation(
            raw,
            latest,
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
        )
        return [obs] if obs is not None else []
