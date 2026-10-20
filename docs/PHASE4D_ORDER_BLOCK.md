# Phase 4D — Order Block Detection

Causal, closed-candle-only Order Block observations derived exclusively from
confirmed **Break of Structure (BOS)** events. This phase does not implement
mitigation, breaker blocks, inverse order blocks, or trade logic.

## Capability

| Item | Value |
|------|--------|
| Detector id | `order_block` |
| Concepts / aliases | `order_block`, `ob` |
| `min_history` | Same as structure break (default 4) |
| Config | `min_body_ratio` (default `0.5`), `left_strength`, `right_strength`, `tie_policy` |

Swing parameters are forwarded to the internal `StructureBreakDetector`.
Invalid configuration values raise `ValueError` (no silent coercion).

## Qualifying event

- Only confirmed structure-break observations with `variant == "bos"` qualify.
- **CHoCH does not** create Order Blocks.
- BOS detection is reused; structure-break rules are not reimplemented.

## Candidate selection (displacement-leg rule)

For each qualifying BOS:

1. Read `source_swing_event_time` from the BOS payload. This is the open epoch
   of the broken swing pivot candle and aligns with `Candle.epoch` semantics.
2. Require the swing event candle to be present in the closed series.
3. Require **exact timeframe continuity** from the swing event candle through
   the BOS candle (same `timeframe_seconds` step). Any discontinuity aborts
   Order Block formation for that BOS — no inference across gaps.
4. Search candles with epoch in `[source_swing_event_time, bos_epoch)` by
   scanning **backward** from the candle immediately before the BOS candle
   toward the swing event candle.
5. Select the **nearest** opposing-color candle that passes the candidate
   quality filter:
   - Bullish BOS → bearish candle (`close < open`)
   - Bearish BOS → bullish candle (`close > open`)
6. If no qualifying candidate exists, no Order Block is emitted for that BOS.

The search is bounded by the broken swing’s event time; candles before the
swing event are never selected. No arbitrary fixed lookback is used.

## Candidate-candle quality filter

Parameter: `min_body_ratio` (default `0.5`).

```
abs(close - open) / (high - low) >= min_body_ratio
```

- This is a **candidate-candle quality filter**, not a displacement-strength measure.
- Zero-range candles (`high == low`) are rejected.
- Ratio must be a finite number in `[0.0, 1.0]`; out-of-range values raise
  `ValueError` at construction / registry validation.

## Zone boundaries

- `lower_bound` = candidate candle `low`
- `upper_bound` = candidate candle `high`
- Full high–low range is preserved; body boundaries are not substituted.

## Observation contract

| Field | Value |
|-------|--------|
| `type` | `"order_block"` |
| `variant` | `"bullish"` \| `"bearish"` (same as direction) |
| `direction` | Same as the triggering BOS |
| `event_time` / `confirmed_at` / `candle_epoch` | BOS confirmation open epoch |
| `developing` | `false` |
| `reference_ids` | Exactly the triggering BOS observation id |
| `detector_id` | `"order_block"` |
| `detector_config_id` | Deterministic from effective config |

### Payload

- `lower_bound`, `upper_bound`, `size`
- `source_candle_epoch`, `open`, `high`, `low`, `close` (candidate)
- `source_bos_id`, `source_swing_event_time`, `source_swing_id`
- `broken_level_price` (from BOS when available)
- `body_ratio`, `min_body_ratio`
- Swing config fields: `left_strength`, `right_strength`, `tie_policy`

Stable observation identity includes source BOS id, source candle epoch,
direction, detector, and config identity so reconstruction is idempotent.

## Causal timing and data continuity

- Observations are emitted only when the qualifying BOS is confirmed.
- Never available before BOS confirmation.
- Completed candles only; forming bars are ignored.
- Exact timeframe continuity required between swing event and BOS; gaps suppress
  that Order Block (no bridging or interpolation).
- Mixed symbol/timeframe series and unknown timeframe labels suppress detection.
- Malformed OHLC on a candidate skips that candle for selection.

## Historical vs incremental

- `detect_all_historical()` reconstructs all confirmed Order Blocks from closed history.
- `detect()` emits only events whose `confirmed_at` equals the latest closed bar’s
  open epoch.
- Repeated runs produce identical observations and stable IDs.
- `history_complete` is forwarded to the internal structure-break reconstruction.

## Explicit limitations

- Displacement strength is **not** measured (body ratio is a candidate quality filter only).
- No ATR or recent-average-body filters.
- No mitigation / invalidation lifecycle.
- No breaker blocks or inverse order blocks.
- CHoCH-triggered Order Blocks are out of scope.
- Strategy evaluation, trade execution, and multi-timeframe confluence remain out of scope.

## Provenance

- Candidate candle: `source_candle_epoch` and OHLC in payload.
- Triggering BOS: `reference_ids` and `source_bos_id`.
- Broken swing: `source_swing_event_time` / `source_swing_id` when present on the BOS.
