# Phase 4B — Fair Value Gap Lifecycle Tracking

## Scope

Immutable lifecycle observations for Fair Value Gaps formed by Phase 4A.
Composes `FairValueGapDetector`; does not reimplement formation geometry.
Does not mutate source FVG observations. Does not integrate engine-wide
lifecycle state machines.

## Capability

| Item | Value |
|------|--------|
| Detector id | `fair_value_gap_lifecycle` |
| Concepts / aliases | `fair_value_gap_lifecycle`, `fvg_lifecycle` |
| `min_history` | 4 |
| Config | empty (no accepted params) |

Phase 4A `fair_value_gap` / `fvg` remains unchanged.

## Lifecycle event contract

| Field | Value |
|-------|--------|
| `type` | `"fair_value_gap_lifecycle"` |
| `variant` | `first_touch` \| `partial_mitigation` \| `full_mitigation` \| `invalidation` |
| `direction` | Inherited from source FVG |
| `reference_ids` | Exactly the source FVG observation id |
| `event_time` / `confirmed_at` | Confirming candle open epoch |
| `developing` | `false` |

Stable IDs incorporate source FVG id, event type (variant), event candle epoch,
detector id, and config identity.

### Payload

- `source_fvg_id`, `lower_bound`, `upper_bound`, `size`
- `confirming_candle_epoch`, `penetration_depth`, `fill_ratio`
- Confirming candle OHLC, `source_fvg_candle_epoch`, `source_fvg_direction`

## Locked rules

For bullish FVG `[L, U]`: near = U, far = L.
Penetration depth = `U - min(U, candle.low)`, capped at gap size.
Fill ratio = depth / (U − L).
Full mitigation: `candle.low <= L`. Invalidation: `candle.close < L`.

For bearish FVG `[L, U]`: near = L, far = U.
Penetration depth = `max(0, candle.high - L)`, capped at gap size.
Full mitigation: `candle.high >= U`. Invalidation: `candle.close > U`.

Equality at the far boundary does **not** invalidate.

- First touch once on first intersecting completed candle after formation.
- Partial mitigation on each strict increase of max penetration (not on full-mit candle).
- Full mitigation and invalidation independent; both may emit on one candle.
- Same-candle order: first_touch → partial_mitigation → full_mitigation → invalidation.
- Lifecycle closes after full mitigation or invalidation (both allowed on that candle).
- No events on the formation candle or from forming bars.
- No events after closure.

## Continuity and causality

- Completed candles only; exact timeframe step required to evaluate a candle.
- Discontinuous candles are skipped; prior events and penetration state preserved.
- Evaluation resumes on later pairs that are continuous with their predecessor.
- No interpolation or bridging of missing bars.
- Mixed symbols/timeframes and unknown timeframes reject the series.
- Malformed OHLC skips that candle’s evaluation.

## Out of scope

Inverse FVGs, merging, time-based expiry, ATR/size/displacement filters,
structure-based invalidation, trading signals, engine-wide lifecycle integration.
