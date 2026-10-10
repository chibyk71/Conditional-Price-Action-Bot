"""Deriv WebSocket adapter (Options public endpoint verified by diagnostic)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from cpab.adapters.base import MarketDataAdapter
from cpab.logging_setup import get_logger

logger = get_logger("cpab.adapters.deriv")


class DerivWebSocketAdapter(MarketDataAdapter):
    """
    Isolates Deriv protocol details.

    Endpoint is configurable. Default follows diagnostic fallback:
    wss://api.derivws.com/trading/v1/options/ws/public
    because general v3 returned HTTP 520 in the validation environment.
    """

    def __init__(
        self,
        endpoint: str = "wss://api.derivws.com/trading/v1/options/ws/public",
        app_id: str = "1089",
        request_timeout_sec: float = 15.0,
        max_connect_retries: int = 5,
        reconnect_backoff_sec: float = 2.0,
    ) -> None:
        self.endpoint = endpoint
        self.app_id = app_id
        self.request_timeout_sec = request_timeout_sec
        self.max_connect_retries = max_connect_retries
        self.reconnect_backoff_sec = reconnect_backoff_sec
        self._ws: Any = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._recv_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        if self._ws is None:
            return False
        try:
            from websockets.protocol import State
            return self._ws.state is State.OPEN
        except Exception:
            return True  # assume open if we have a socket after connect

    def _url(self) -> str:
        # Options public does not require app_id query; general v3 does.
        if "app_id=" in self.endpoint:
            return self.endpoint
        if "ws.derivws.com" in self.endpoint or "binaryws.com" in self.endpoint:
            sep = "&" if "?" in self.endpoint else "?"
            return f"{self.endpoint}{sep}app_id={self.app_id}"
        return self.endpoint

    async def connect(self) -> None:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_connect_retries + 1):
            try:
                url = self._url()
                logger.info("Connecting to Deriv WS (attempt %s): %s", attempt, url)
                self._ws = await websockets.connect(
                    url,
                    open_timeout=self.request_timeout_sec,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                )
                self._recv_task = asyncio.create_task(self._recv_loop())
                logger.info("Deriv WS connected")
                return
            except Exception as e:
                last_err = e
                logger.warning("Connect attempt %s failed: %s", attempt, e)
                await asyncio.sleep(self.reconnect_backoff_sec * attempt)
        raise ConnectionError(f"Failed to connect after {self.max_connect_retries} attempts: {last_err}")

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        logger.info("Deriv WS closed")

    async def _ensure_connected(self) -> None:
        if not self.connected:
            await self.connect()

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Malformed JSON from Deriv")
                    continue
                req_id = msg.get("req_id")
                if req_id is not None and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        if "error" in msg:
                            err = msg["error"]
                            fut.set_exception(
                                RuntimeError(
                                    f"Deriv API error {err.get('code')}: {err.get('message')}"
                                )
                            )
                        else:
                            fut.set_result(msg)
                else:
                    logger.debug("Uncorrelated message msg_type=%s", msg.get("msg_type"))
        except ConnectionClosed as e:
            logger.warning("Deriv WS connection closed: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Recv loop error: %s", e)

    async def fetch_candles(
        self,
        symbol: str,
        granularity: int,
        count: int,
        *,
        end: str | int = "latest",
    ) -> list[dict[str, Any]]:
        async with self._lock:
            await self._ensure_connected()
            self._req_id += 1
            req_id = self._req_id
            # Options public: omit subscribe for one-shot; subscribe:0 is rejected
            payload: dict[str, Any] = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": count,
                "end": end,
                "req_id": req_id,
            }
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._pending[req_id] = fut
            assert self._ws is not None
            try:
                await self._ws.send(json.dumps(payload))
                msg = await asyncio.wait_for(fut, timeout=self.request_timeout_sec)
            except asyncio.TimeoutError:
                self._pending.pop(req_id, None)
                logger.error("Timeout waiting for candles %s gran=%s count=%s", symbol, granularity, count)
                raise
            except Exception:
                self._pending.pop(req_id, None)
                raise

            candles = msg.get("candles") or []
            if not isinstance(candles, list):
                raise RuntimeError(f"Unexpected candles payload type: {type(candles)}")
            logger.debug(
                "Fetched %s candles for %s gran=%s (req_id=%s)",
                len(candles),
                symbol,
                granularity,
                req_id,
            )
            return candles

    async def reconnect(self) -> None:
        logger.info("Reconnecting Deriv WS")
        await self.close()
        await self.connect()
