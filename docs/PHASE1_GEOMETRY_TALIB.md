# Phase 1 — Candle Geometry, TA-Lib Adapter, Capability Registry

## Candle geometry (`cpab.candle_geometry`)

Pure functions over the existing `Candle` model. No network, logging, detectors, or readiness side effects.

| Measurement | Definition |
|-------------|------------|
| range | `high - low` |
| real_body | `abs(close - open)` |
| upper_wick | `high - max(open, close)` |
| lower_wick | `min(open, close) - low` |
| direction | bullish / bearish / neutral from open vs close only (independent of doji) |
| ratios | relative to range; `None` when range == 0 |
| is_doji | body_ratio ≤ threshold (default 0.10); threshold ∈ [0.0, 1.0] or `ValueError`; `None` for zero-range |

Overlap: shared interval `[max(lows), min(highs)]`. Point overlap (width 0) is preserved; `None` means no overlap.
Containment: `range_contains(inner, outer)` is full-range only. Malformed OHLC remains the responsibility of `cpab.market.validation`.

## TA-Lib dependency

Required: Python `TA-Lib` and `numpy`. System: TA-Lib C library (e.g. build from [TA-Lib 0.6.x source](https://github.com/TA-Lib/ta-lib/releases)).

```bash
wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar xzf ta-lib-0.6.4-src.tar.gz && cd ta-lib-0.6.4
./configure --prefix=/usr && make && make install
pip install TA-Lib numpy
```

No optional-dependency path and no silent fallback to the hand-written detector.

## Adapter contract (`TalibPatternDetector`)

1. **Raw evaluation** (`evaluate_raw`): `AdapterRawResult` with kind
   `PATTERN | NO_PATTERN | INSUFFICIENT_HISTORY | FORMING_CANDLE | EXECUTION_FAILURE`.
   Preserves the exact integer for the latest completed bar (`raw_talib_value`).
2. **Observation mapping** (`raw_to_observation`): only for `PATTERN` with non-zero raw.
   Payload includes `raw_talib_value`. Uses existing `Observation.create` / stable identity.

Forming candles rejected at the adapter boundary. Insufficient history and raw `0` do not create Observations. Minimum history is pattern-specific (CDLENGULFING → 2).

## Capability registry

Minimal lookup: supported concepts, detector + config, min history.
`deterministic_config_id(config)` hashes canonical JSON (order-independent).
Unsupported concepts and invalid parameters raise `ValueError` during validation. No silent substitution.

## Engulfing comparison

`EngulfingDetector` is **unchanged**. Side-by-side tests compare it with `CDLENGULFING`. Definitions differ; parity is characterized, not forced. Migration is deferred.

## Causality

Incremental replay supplies only the completed-candle prefix at each step. Observations never use future bars; forming candles never contribute; repeated `detect()` is idempotent under stable IDs.
