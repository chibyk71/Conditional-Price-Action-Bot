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
| is_doji | body_ratio ≤ threshold (default 0.10); threshold ∈ [0.0, 1.0] or `ValueError` (validated even for zero-range); `None` for zero-range |

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

Forming candles rejected at the adapter boundary. Insufficient history and raw `0` do not create Observations.

### Minimum history (TA-Lib lookback)

Minimum history is **not** a hard-coded structural candle count. It is derived at runtime from the installed TA-Lib library:

```text
talib_lookback(pattern) = Function(pattern).lookback   # TA-Lib computational lookback
min_history_for(pattern) = talib_lookback(pattern) + 1  # preceding bars + signal bar
```

`Function(name).lookback` is TA-Lib's function-specific computational lookback under the library's default configuration: the number of preceding input bars required before the current bar can be evaluated. Adding 1 includes the signal bar, giving the minimum completed candles CPAB requires before evaluating the latest bar.

Lookback values depend on the TA-Lib implementation and configuration in use. Illustrative values under TA-Lib 0.8.x with default candle settings:

| Pattern | lookback | min_history (lookback+1) |
|---------|----------|--------------------------|
| CDLENGULFING | 2 | **3** |
| CDLDOJI | 10 | 11 |
| CDLHAMMER / CDLHARAMI / CDLPIERCING / CDLSHOOTINGSTAR | 11 | 12 |
| CDLMORNINGSTAR / CDLEVENINGSTAR | 12 | 13 |

Unsupported pattern names are rejected (`ValueError` / `EXECUTION_FAILURE`); they are not assigned an arbitrary default.

## Capability registry

Minimal lookup: supported concepts, detector + config, min history.
`deterministic_config_id(config)` hashes canonical JSON (order-independent).
Registered configuration is immutable (pattern override rejected).
Unsupported concepts and invalid parameters raise `ValueError` during validation. No silent substitution.

## Engulfing comparison

`EngulfingDetector` is **unchanged**. Side-by-side tests compare it with `CDLENGULFING`. Definitions differ; parity is characterized, not forced. Migration is deferred.

## Causality

For each historical bar `i` with sufficient history:

- `evaluate_raw_series(prefix[:i+1])[i] == evaluate_raw_series(full)[i]` (raw integer identity)
- Observations derived independently from the prefix result and from the full-series result at bar `i` match on type, direction, candle identity, `detector_config_id`, payload, and stable id
- Where the raw value is zero, neither path emits an observation

Incremental production `detect()` evaluates only the latest completed bar; forming candles never contribute.
