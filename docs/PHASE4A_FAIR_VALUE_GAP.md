# Phase 4A — Fair Value Gap (FVG) Formation Detection

## Scope

Causal, closed-candle-only **formation** of Fair Value Gaps. This phase emits
immutable geometric observations only. Lifecycle tracking (touch, mitigation,
invalidation) is explicitly out of scope and is documented below for future work.

## Locked geometric contract

Three consecutive, completed candles from the same symbol and timeframe:

### Bullish FVG

- Condition: `candle_1.high < candle_3.low`
- `lower_bound = candle_1.high`
- `upper_bound = candle_3.low`
- `direction = "bullish"`
- `variant = "bullish"`

### Bearish FVG

- Condition: `candle_1.low > candle_3.high`
- `lower_bound = candle_3.high`
- `upper_bound = candle_1.low`
- `direction = "bearish"`
- `variant = "bearish"`

Require `upper_bound > lower_bound`. Equal boundaries do not qualify.

**Not applied in Phase 4A:**

- Middle-candle colour / body filter
- Minimum gap size or ATR filter
- Merging of consecutive or overlapping FVGs
- Displacement filters
- Inverse FVGs

## Observation contract

| Field | Value |
|-------|--------|
| `type` | `"fair_value_gap"` |
| `event_time` | candle_3 open epoch |
| `confirmed_at` | candle_3 open epoch (emit only after candle 3 is completed) |
| `candle_epoch` | candle_3 open epoch |
| `developing` | `false` |
| `direction` / `variant` | `"bullish"` or `"bearish"` |
| `detector_id` | `"fair_value_gap"` |
| `detector_config_id` | deterministic id of empty config (`cfg:…`) |
| `reference_ids` | `[]` |

### Payload

- `lower_bound`, `upper_bound`, `size` (`upper_bound - lower_bound`)
- `candle_1_epoch`, `candle_2_epoch`, `candle_3_epoch`

Stable observation identity follows the established `Observation.stable_event_id`
convention (symbol, timeframe, candle_3 epoch, direction/variant, detector and
config identity).

## Data integrity and causality

- Exactly three consecutive candles with the expected timeframe step
  (`timeframe_seconds`; unknown labels reject the window).
- No gap bridging, reordering, or use of future / forming candles.
- Mixed symbols or timeframes suppress detection for the series.
- Malformed OHLC (non-finite, high < low, high/low inconsistent with open/close)
  skips that window.
- `detect()` emits only formations whose `confirmed_at` equals the latest closed
  bar’s open epoch.
- `detect_all_historical()` reconstructs all confirmed formations without
  leaking later information into earlier observations.

## Registry

- Concept keys: `fair_value_gap`, `fvg`
- Detector id: `fair_value_gap`
- `min_history = 3`
- Accepted params: none (empty config)
- Implementation tag: `fair_value_gap`

## Future lifecycle meanings (not implemented)

These terms are reserved for later phases. Phase 4A does **not** emit lifecycle
events or mutate FVG observations after formation.

| Term | Intended meaning (future) |
|------|---------------------------|
| **Touched** | Price has traded into the gap interval at least once after formation (wick or body contact with `(lower_bound, upper_bound)`), without requiring fill. |
| **Partially mitigated** | A portion of the gap has been filled by subsequent price action (e.g. price has crossed partway through the interval), but the opposite boundary has not yet been reached. |
| **Fully mitigated** | Price has fully traversed the gap (for a bullish FVG, traded down through to `lower_bound` or beyond; for bearish, up through to `upper_bound` or beyond), so the imbalance is considered filled. |
| **Invalidated** | Structural or rule-based cancellation of the FVG as a usable level (policy TBD: e.g. opposite-direction structure break, time decay, or explicit supersession by a newer overlapping gap). Distinct from full mitigation. |

Implementation of touch / mitigation / invalidation tracking, inverse FVGs,
filters, merging, and trading signals remains out of scope until a later phase
locks those contracts.
