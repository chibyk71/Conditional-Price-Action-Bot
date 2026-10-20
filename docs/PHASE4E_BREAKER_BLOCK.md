# Phase 4E — Breaker Block Detection

Causal, closed-candle-only Breaker Block observations derived exclusively from
confirmed **Order Block** observations when a completed candle closes strictly
through the Order Block zone. This phase does not implement retests, mitigation
lifecycle beyond the first failure, or trade logic.

## Capability

| Item | Value |
|------|--------|
| Detector id | `breaker_block` |
| Concepts / aliases | `breaker_block`, `breaker` |
| `min_history` | Order Block min history + 1 (default 5) |
| Config | Forwarded to Order Block: `min_body_ratio`, `left_strength`, `right_strength`, `tie_policy` |

No speculative parameters (retest, BOS confirmation, custom thresholds) in V1.
Invalid configuration raises `ValueError` (no silent coercion).

## Source observations

- Only existing confirmed `order_block` observations are eligible.
- Source Order Block observations are **not** mutated.
- Zone boundaries are the source OB’s original high–low range.

## Invalidation rules (Breaker formation)

| Source OB | Condition (completed candle) | Breaker variant | Breaker direction |
|-----------|------------------------------|-----------------|-------------------|
| Bullish `[L, U]` | `close < L` | `bullish_to_bearish` | `bearish` |
| Bearish `[L, U]` | `close > U` | `bearish_to_bullish` | `bullish` |

- Equality with the boundary does **not** qualify.
- Wick-only breaches (range beyond boundary, close still inside/on zone) do **not** qualify.
- No retest, additional BOS, or CHoCH is required.
- At most one Breaker Block per source Order Block (first qualifying failure only).

## Observation contract

| Field | Value |
|-------|--------|
| `type` | `"breaker_block"` |
| `variant` | `bullish_to_bearish` \| `bearish_to_bullish` |
| `direction` | Opposite polarity of the source Order Block |
| `event_time` / `confirmed_at` / `candle_epoch` | Invalidating candle open epoch |
| `developing` | `false` |
| `reference_ids` | Exactly the source Order Block observation id |
| `detector_id` | `"breaker_block"` |
| `detector_config_id` | Deterministic from effective (forwarded) config |

### Payload

- `source_ob_id`, `source_ob_direction`
- `lower_bound`, `upper_bound`, `size` (original OB bounds, unchanged)
- `invalidation_candle_epoch` and confirming OHLC
- `source_ob_candle_epoch`, `source_bos_id` when present on the source OB

Stable observation identity includes source OB id, invalidation epoch,
direction/variant, detector, and config identity.

## Causal timing and continuity

- Emit only when the qualifying completed candle is confirmed.
- Never available before that candle closes.
- Exact timeframe continuity with the immediate predecessor is required to
  evaluate a candidate invalidation bar; discontinuous bars are skipped (no
  inference across gaps).
- Mixed symbol/timeframe series and unknown timeframe labels suppress detection.
- Malformed OHLC on a candidate bar skips that bar.

## Historical vs incremental

- `detect_all_historical()` reconstructs all confirmed Breaker Blocks from closed history.
- `detect()` emits only events whose `confirmed_at` equals the latest closed bar’s open epoch.
- Repeated runs produce identical observations and stable IDs.
- `history_complete` is forwarded to the internal Order Block (and structure-break) reconstruction.

## Explicit limitations

- No retest tracking after the breaker forms.
- No requirement for BOS/CHoCH confirmation of the failure.
- No mutation or lifecycle extension of the source Order Block observation.
- No ATR or custom invalidation-threshold configuration in V1.
- Strategy evaluation, alerts, and execution remain out of scope.

## Provenance

- Source Order Block: `reference_ids` and `source_ob_id`.
- Invalidation candle: epoch and OHLC in payload.
- Original zone bounds preserved exactly from the source OB.
