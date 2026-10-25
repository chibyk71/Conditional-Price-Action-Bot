"""Phase 6A — ATR expansion / contraction and range expansion tests."""

from __future__ import annotations

import math

import pytest

from cpab.detectors.behavior_volatility import (
    ATR_PERIOD,
    BASELINE_WINDOW,
    EXPANSION_MULT,
    CONTRACTION_MULT,
    RANGE_EXPANSION_MULT,
    MIN_HISTORY_ATR_REGIME,
    MIN_HISTORY_RANGE,
    VARIANT_ROLLING_ATR,
    VARIANT_TRUE_RANGE,
    ATRExpansionDetector,
    ATRContractionDetector,
    RangeExpansionDetector,
    _rolling_atr_series,
    _baseline_mean,
)
from cpab.detectors.registry import build_default_registry, deterministic_config_id
from cpab.detectors.wilder_atr import true_range, wilder_atr
from cpab.models import Candle


TF = 60
START = 1_700_000_000


def _c(
    i: int,
    o: float,
    h: float,
    l: float,
    cl: float,
    *,
    symbol: str = "BV_100",
    timeframe: str = "1m",
    start: int = START,
    is_closed: bool = True,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        epoch=start + i * TF,
        open=o,
        high=h,
        low=l,
        close=cl,
        is_closed=is_closed,
        source="provider",
    )
