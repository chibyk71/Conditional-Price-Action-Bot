"""Entry point for Conditional Price-Action Bot V1.0."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from cpab.adapters.deriv_ws import DerivWebSocketAdapter
from cpab.config import conditions_from_config, load_config
from cpab.core.engine import ConditionEngine
from cpab.detectors.engulfing import EngulfingDetector
from cpab.logging_setup import get_logger, setup_logging
from cpab.market.readiness import ReadinessManager
from cpab.market.service import MarketDataService

logger = get_logger("cpab.main")


async def run(config_path: str, once: bool = False) -> int:
    cfg = load_config(config_path)
    log_cfg = cfg.get("logging") or {}
    setup_logging(level=log_cfg.get("level", "INFO"), fmt=log_cfg.get("format"))

    deriv_cfg = cfg.get("deriv") or {}
    adapter = DerivWebSocketAdapter(
        endpoint=deriv_cfg.get("endpoint", "wss://api.derivws.com/trading/v1/options/ws/public"),
        app_id=str(deriv_cfg.get("app_id", "1089")),
        request_timeout_sec=float(deriv_cfg.get("request_timeout_sec", 15)),
        max_connect_retries=int(deriv_cfg.get("max_connect_retries", 5)),
        reconnect_backoff_sec=float(deriv_cfg.get("reconnect_backoff_sec", 2.0)),
    )

    readiness = ReadinessManager()
    market = MarketDataService(
        adapter=adapter,
        symbols=list(cfg.get("symbols") or ["R_100"]),
        timeframes=list(cfg.get("timeframes") or [{"name": "1m", "granularity": 60}]),
        min_candles=int((cfg.get("history") or {}).get("min_candles", 300)),
        poll_interval_sec=float(deriv_cfg.get("poll_interval_sec", 8.0)),
        poll_lookback_bars=int(deriv_cfg.get("poll_lookback_bars", 5)),
        confirm_lag_bars=int(deriv_cfg.get("confirm_lag_bars", 1)),
        readiness=readiness,
    )

    conditions = conditions_from_config(cfg)
    detectors = [EngulfingDetector()]
    engine = ConditionEngine(market, detectors, conditions, readiness)

    stop = asyncio.Event()

    def _handle_sig(*_args):
        logger.info("Shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _handle_sig)
        except NotImplementedError:
            pass

    try:
        await adapter.connect()
        await market.initialize()
        if once:
            signals = engine.evaluate_once()
            for sig in signals:
                logger.info("Signal: %s", sig.to_dict())
            return 0

        await market.start_polling()
        logger.info("Bot running; readiness=%s", readiness.snapshot())
        while not stop.is_set():
            signals = engine.evaluate_once()
            for sig in signals:
                logger.info("Signal: %s", sig.to_dict())
            try:
                await asyncio.wait_for(stop.wait(), timeout=market.poll_interval_sec)
            except asyncio.TimeoutError:
                pass
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1
    finally:
        await market.stop_polling()
        await adapter.close()
        logger.info("Shutdown complete")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Conditional Price-Action Bot V1.0")
    parser.add_argument(
        "-c",
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "config" / "default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Initialize, evaluate once, exit (no polling loop)",
    )
    args = parser.parse_args(argv)
    code = asyncio.run(run(args.config, once=args.once))
    sys.exit(code)


if __name__ == "__main__":
    main()
