# Phase 4C — Inverse Fair Value Gap (IFVG) Detection

## Scope

Immutable observations that mark the **inversion** of an existing Fair Value Gap
when Phase 4B invalidation occurs. Composes `FairValueGapLifecycleDetector`;
does not reimplement formation or invalidation geometry.

## Capability

| Item | Value |
|------|--------|
| Detector id | `inverse_fair_value_gap` |
| Concepts / aliases | `inverse_fair_value_gap`, `ifvg` |
| `min_history` | 4 |
| Config | empty (no accepted params) |

Phase 4A and Phase 4B public contracts remain unchanged.

## Locked semantics

An IFVG forms **immediately** when an existing FVG is invalidated by a
qualifying completed-candle close:

| Source FVG | Invalidation condition | IFVG variant | IFVG direction |
|------------|------------------------|--------------|----------------|
| Bullish `[L, U]` | `close < L` | `bullish_to_bearish` | `bearish` |
| Bearish `[L, U]` | `close > U` | `bearish_to_bullish` | `bullish` |

- Equality with the boundary does **not** qualify.
- No retest or rejection is required.
- Original FVG bounds are **preserved** (not recalculated from the inversion candle).

Invalidation rules are those of Phase 4B; this detector maps
`variant="invalidation"` lifecycle observations to IFVG observations.

## Observation contract

| Field | Value |
|-------|--------|
| `type` | `"inverse_fair_value_gap"` |
| `variant` | `bullish_to_bearish` \| `bearish_to_bullish` |
| `direction` | Opposite of the source FVG |
| `event_time` / `confirmed_at` | Invalidating candle open epoch |
| `developing` | `false` |
| `reference_ids` | Exactly the source FVG observation id |

### Payload

- `source_fvg_id`, `source_invalidation_id`, `source_fvg_direction`
- `lower_bound`, `upper_bound`, `size` (original FVG bounds)
- `invalidation_candle_epoch` and confirming OHLC when available
- `source_fvg_candle_epoch` when available

Stable IDs incorporate source FVG id, invalidation id, event candle epoch,
direction/variant, and detector/config identity.

## Causality and continuity

Inherited from Phase 4B lifecycle evaluation:

- Completed candles only; never before the invalidating candle closes.
- Exact timeframe continuity; discontinuous candles are not evaluated.
- No interpolation or bridging of missing bars.
- Malformed OHLC and unknown/mixed series follow existing conventions.
- Multiple overlapping FVGs are tracked independently.
- At most one IFVG per source FVG invalidation event.

## Out of scope

Retest/rejection, new lifecycle states, FVG merging, size/ATR/expiry filters,
structure filters, trading signals, engine/alert/execution integration.
