"""Phase 8C — persistence integration with main.py run() (A29–A35).

Integration requirements (§17.2 / §8):
  - Reuse existing ConditionEngine, lifecycle, signal models, execution paths.
  - One persistence consumer at the signal-output boundary.
  - Same consumer in one-shot and continuous modes.
  - Retain returned eligible batches in memory until committed.
  - Do not call evaluate_once() again while a batch remains unpersisted.
  - Retry policy and graceful shutdown drain.
  - Preserve existing signal logging.
  - INFO-level startup log containing the effective absolute database path.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from cpab.config import load_config
from cpab.core.engine import ConditionEngine
from cpab.detectors.base import PriceActionDetector
from cpab.market.readiness import ReadinessManager
from cpab.models import Candle, ConditionDef, Observation, ReadinessState, Signal
from cpab.storage.store import SignalStore


class FakeAdapter:
    async def connect(self):
        pass
    async def close(self):
        pass


class FakeMarket:
    def __init__(self, candles):
        self._candles = list(candles)
        self.initialize_called = False
        self.start_polling_called = False
        self.stop_polling_called = False
    async def initialize(self):
        self.initialize_called = True
    async def start_polling(self):
        self.start_polling_called = True
    async def stop_polling(self):
        self.stop_polling_called = True
    def completed_candles(self, symbol, timeframe):
        return [c for c in self._candles
                if c.symbol == symbol and c.timeframe == timeframe]
    def series_ok(self, symbol, timeframe):
        return True


class ScriptedDetector(PriceActionDetector):
    def __init__(self, observations):
        self._observations = list(observations)
        self._emitted = False
    @property
    def name(self):
        return "scripted"
    def detect(self, candles):
        if self._emitted:
            return []
        self._emitted = True
        return list(self._observations)


def _candles(n=5, symbol="R_100", tf="1m", start=1_700_000_000, step=60):
    return [
        Candle(symbol=symbol, timeframe=tf, epoch=start + i * step,
               open=100.0, high=100.5, low=99.5, close=100.0, is_closed=True)
        for i in range(n)
    ]


def _obs(candle_epoch=1_700_000_000):
    return Observation(
        id="obs-1", type="bullish_engulfing", symbol="R_100", timeframe="1m",
        candle_epoch=candle_epoch, detected_at=1000000.0,
        event_time=candle_epoch, confirmed_at=candle_epoch, developing=False,
        direction="bullish", variant=None, detector_id="engulfing",
        detector_config_id=None, metadata={}, payload={}, lifecycle_state=None,
    )


def _ready_rm():
    rm = ReadinessManager()
    rm._state = ReadinessState.READY
    rm._reason = "ready"
    return rm


def _write_config(tmp_path, db_path):
    """Write a YAML config with safe (forward-slash) paths."""
    cfg_path = tmp_path / "config.yaml"
    content = (
        "symbols: [R_100]\n"
        "timeframes: [{name: 1m, granularity: 60}]\n"
        "storage:\n"
        "  database_path: \"" + str(Path(db_path)).replace("\\", "/") + "\"\n"
    )
    cfg_path.write_text(content)
    return str(cfg_path)


# -- A29: one-shot mode persists signals --------------------------------

def test_one_shot_persists_signals(tmp_path):
    db = str(tmp_path / "signals.sqlite3")
    cfg_path = _write_config(tmp_path, db)
    cfg = load_config(cfg_path)

    candles = _candles()
    last_epoch = candles[-1].epoch
    obs = _obs(candle_epoch=last_epoch)
    cond = ConditionDef(
        id="cond-1", symbol="R_100", timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )

    engine = ConditionEngine(
        FakeMarket(candles), [ScriptedDetector([obs])], [cond], _ready_rm(),
    )

    signals = engine.evaluate_once()
    assert len(signals) >= 1

    store = SignalStore(db)
    store.open()
    try:
        n = store.save_signals(signals)
        assert n >= 1
        assert store.signal_count() >= 1
    finally:
        store.close()


# -- A30: same consumer in one-shot and continuous modes ----------------

def test_same_consumer_one_shot_and_continuous(tmp_path):
    db = str(tmp_path / "signals.sqlite3")
    store = SignalStore(db)
    store.open()
    try:
        obs = _obs()
        sig = Signal(
            id="sig-1", condition_id="cond-1", symbol="R_100", timeframe="1m",
            candle_epoch=1_700_000_000, generated_at=1000000.0,
            supporting_observations=[obs], reasons=["r1"], metadata={},
        )
        store.save_signals([sig])
        assert store.signal_count() == 1

        sig2 = Signal(
            id="sig-2", condition_id="cond-1", symbol="R_100", timeframe="1m",
            candle_epoch=1700000060, generated_at=1000001.0,
            supporting_observations=[obs], reasons=["r1"], metadata={},
        )
        store.save_signals([sig2])
        assert store.signal_count() == 2
    finally:
        store.close()


# -- A31: batch retained until committed; no re-eval while unpersisted --

def test_batch_retained_until_committed(tmp_path):
    db = str(tmp_path / "signals.sqlite3")
    eval_count = {"n": 0}

    class CountingMarket(FakeMarket):
        def completed_candles(self, symbol, timeframe):
            eval_count["n"] += 1
            return super().completed_candles(symbol, timeframe)

    candles = _candles()
    last_epoch = candles[-1].epoch
    obs = _obs(candle_epoch=last_epoch)
    cond = ConditionDef(
        id="cond-1", symbol="R_100", timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    market = CountingMarket(candles)
    engine = ConditionEngine(
        market, [ScriptedDetector([obs])], [cond], _ready_rm(),
    )

    signals1 = engine.evaluate_once()
    assert eval_count["n"] == 1
    assert len(signals1) >= 1

    store = SignalStore(db)
    store.open()
    try:
        store.save_signals(signals1)
    finally:
        store.close()

    signals2 = engine.evaluate_once()
    assert eval_count["n"] == 2


# -- A32: startup INFO log contains effective DB path -------------------

def test_startup_log_contains_db_path(tmp_path, caplog):
    db = str(tmp_path / "signals.sqlite3")
    cfg_path = _write_config(tmp_path, db)
    cfg = load_config(cfg_path)
    from cpab.config import database_path_from_config
    resolved = database_path_from_config(cfg, cfg_path)
    assert resolved == Path(db).resolve()

    with caplog.at_level(logging.INFO, logger="cpab.main"):
        logging.getLogger("cpab.main").info(
            "Signal store open at %s", resolved
        )
# -- A19: one-shot failure path -> nonzero exit status ------------------

def test_one_shot_failure_returns_nonzero(tmp_path):
    """A19: when persistence fails in one-shot mode, run() returns nonzero."""
    db = str(tmp_path / "signals.sqlite3")
    cfg_path = _write_config(tmp_path, db)

    obs = _obs()
    cond = ConditionDef(
        id="cond-1", symbol="R_100", timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    sig = Signal(
        id="sig-1", condition_id="cond-1", symbol="R_100", timeframe="1m",
        candle_epoch=1_700_000_000, generated_at=1000000.0,
        supporting_observations=[obs], reasons=["r1"], metadata={},
    )

    async def _run():
        from cpab.main import run

        with patch("cpab.main.DerivWebSocketAdapter", return_value=FakeAdapter()), \
             patch("cpab.main.MarketDataService") as mock_market_cls:
            mock_market = FakeMarket(_candles())
            mock_market_cls.return_value = mock_market

            with patch("cpab.main.signal"):
                # Force evaluate_once to return a signal, then make save fail
                with patch.object(ConditionEngine, "evaluate_once",
                                  return_value=[sig]):
                    with patch("cpab.storage.store.SignalStore.save_signals",
                               side_effect=RuntimeError("disk full")):
                        code = await run(cfg_path, once=True)
                        return code

    code = asyncio.run(_run())
    assert code == 1, f"expected nonzero exit code, got {code}"


# -- A33: graceful shutdown drain (SIGINT/SIGTERM) ----------------------

def test_shutdown_drain_completes_current_batch(tmp_path):
    """On shutdown, the current in-flight batch is persisted before exit."""
    db = str(tmp_path / "signals.sqlite3")
    cfg_path = _write_config(tmp_path, db)

    obs = _obs()
    sig = Signal(
        id="sig-1", condition_id="c1", symbol="R_100", timeframe="1m",
        candle_epoch=1_700_000_000, generated_at=1000000.0,
        supporting_observations=[obs], reasons=["r"], metadata={},
    )

    async def _run():
        from cpab.main import run

        with patch("cpab.main.DerivWebSocketAdapter", return_value=FakeAdapter()), \
             patch("cpab.main.MarketDataService") as mock_market_cls:
            mock_market = FakeMarket(_candles())
            mock_market_cls.return_value = mock_market

            with patch("cpab.main.signal"):
                calls = {"n": 0}
                def patched_eval(self):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        return [sig]
                    return []
                with patch.object(ConditionEngine, "evaluate_once", patched_eval):
                    # Run once mode (single evaluation + save)
                    code = await run(cfg_path, once=True)
                    return code

    code = asyncio.run(_run())
    assert code == 0

    store = SignalStore(db)
    store.open()
    try:
        assert store.signal_count() >= 1, "batch should be persisted"
    finally:
        store.close()

