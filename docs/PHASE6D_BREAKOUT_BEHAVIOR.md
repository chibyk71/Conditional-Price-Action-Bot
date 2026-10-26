# Phase 6D — Breakout Behavior

## Scope

The `breakout_behavior` capability observes completed-candle breakouts of previously confirmed swing highs/lows and zero-width support/resistance zones. It emits observations only; it does not make trading decisions.

Supported failure policies:

- `close_back_across`: fail on the first completed close strictly back across the boundary.
- `confirmed_close_back_across`: a close-back enters pending failure; the next completed candle must also close strictly back across. Equality or a close back on the breakout side resets pending failure.

## Invariants

- A source is eligible only on candles with `candidate.epoch > source.confirmed_at`.
- Wick-only crosses and closes equal to the boundary do not confirm a breakout.
- Each source observation has its own lifecycle, even when multiple sources share a price.
- Gaps and invalid OHLC candles split history into independent segments. Tracking is abandoned silently at the segment boundary.
- Event IDs include source ID/family, direction, event type, boundary, detector/config identity, and failure policy.
- Historical reconstruction is deterministic; incremental detection emits events confirmed on the latest closed candle.

## Source mapping

- `swing_high` → bullish breakout above `payload.price`.
- `swing_low` → bearish breakout below `payload.price`.
- `support_resistance_zone` with `variant=resistance` → bullish breakout above its zero-width level.
- `support_resistance_zone` with `variant=support` → bearish breakout below its zero-width level.

V1 rejects missing/non-finite geometry and non-zero-width zones rather than inferring boundaries. Existing structure-break, zone-interaction, and chart-pattern breakout observations are not consumed as sources.

## Observation shape

The detector emits `type=breakout_behavior` with `variant=breakout_confirmed` or `variant=breakout_failed`. The payload records event type, source ID/family/type/variant, source confirmation time, boundary, direction, failure policy, and lifecycle event. `reference_ids` contains the exact source observation ID.

Follow-through and approach/test observations are deferred.
