"""
Provisional deterministic example detector: bullish / bearish engulfing.

PROVISIONAL — not an approved trading strategy. Used as a testable vertical
slice for the observation → condition → signal pipeline.
Definition (explicit):
  Bullish engulfing on completed bar i (i >= 1):
    - candle i-1 is bearish (close < open)
    - candle i is bullish (close > open)
    - open_i <= close_{i-1}
    - close_i >= open_{i-1}
  Bearish is the mirror.
Only completed candles are considered.
"""

from __future__ import annotations

from cpab.detectors.base import PriceActionDetector
from cpab.models import Candle, Observation
from cpab.logging_setup import get_logger

logger = get_logger("cpab.detectors.engulfing")


class EngulfingDetector(PriceActionDetector):
    @property
    def name(self) -> str:
        return "engulfing"

    def detect(self, candles: list[Candle]) -> list[Observation]:
        closed = [c for c in candles if c.is_closed]
        if len(closed) < 2:
            return []
        out: list[Observation] = []
        # Only evaluate the latest completed pair to avoid re-emitting history
        prev, curr = closed[-2], closed[-1]
        if curr.symbol != prev.symbol or curr.timeframe != prev.timeframe:
            return []

        prev_bear = prev.close < prev.open
        prev_bull = prev.close > prev.open
        curr_bull = curr.close > curr.open
        curr_bear = curr.close < curr.open

        if prev_bear and curr_bull and curr.open <= prev.close and curr.close >= prev.open:
            obs = Observation.create(
                type="bullish_engulfing",
                symbol=curr.symbol,
                timeframe=curr.timeframe,
                candle_epoch=curr.epoch,
                metadata={
                    "prev_epoch": prev.epoch,
                    "prev_ohlc": prev.ohlc_tuple(),
                    "curr_ohlc": curr.ohlc_tuple(),
                    "detector": self.name,
                    "provisional": True,
                },
            )
            out.append(obs)
            logger.debug(
                "bullish_engulfing %s/%s epoch=%s",
                curr.symbol,
                curr.timeframe,
                curr.epoch,
            )
        elif prev_bull and curr_bear and curr.open >= prev.close and curr.close <= prev.open:
            obs = Observation.create(
                type="bearish_engulfing",
                symbol=curr.symbol,
                timeframe=curr.timeframe,
                candle_epoch=curr.epoch,
                metadata={
                    "prev_epoch": prev.epoch,
                    "prev_ohlc": prev.ohlc_tuple(),
                    "curr_ohlc": curr.ohlc_tuple(),
                    "detector": self.name,
                    "provisional": True,
                },
            )
            out.append(obs)
            logger.debug(
                "bearish_engulfing %s/%s epoch=%s",
                curr.symbol,
                curr.timeframe,
                curr.epoch,
            )
        return out
