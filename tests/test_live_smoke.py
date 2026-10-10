"""Optional live-connection smoke test — not part of default suite."""

from __future__ import annotations

import os

import pytest

from cpab.adapters.deriv_ws import DerivWebSocketAdapter
from cpab.market.validation import normalize_provider_candle


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_fetch_r100_1m():
    if os.environ.get("CPAB_LIVE") != "1":
        pytest.skip("Set CPAB_LIVE=1 to run live smoke test")
    endpoint = os.environ.get(
        "DERIV_ENDPOINT",
        "wss://api.derivws.com/trading/v1/options/ws/public",
    )
    adapter = DerivWebSocketAdapter(endpoint=endpoint, app_id="1089", request_timeout_sec=20)
    try:
        await adapter.connect()
        raw = await adapter.fetch_candles("R_100", 60, 10, end="latest")
        assert len(raw) >= 5
        c = normalize_provider_candle(raw[0], "R_100", "1m")
        assert c is not None
        assert c.epoch > 0
    finally:
        await adapter.close()
