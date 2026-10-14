# Phase 2A — Swing Detection (Pivot L/R)

Configurable left/right-strength pivot swing highs and swing lows.

## Configuration

| Parameter | Type | Constraint | Default |
|-----------|------|------------|---------|
| `left_strength` | int | ≥ 1 | 1 |
| `right_strength` | int | ≥ 1 | 1 |
| `tie_policy` | str | `strict` \| `first` \| `all` | `strict` |

Invalid values raise `ValueError` at construction / registry validation. No silent substitution.

`detector_config_id` is the SHA-256 prefix of the canonical JSON of the effective
`{left_strength, right_strength, tie_policy}` map (via `deterministic_config_id`).

`min_history = left_strength + 1 + right_strength`.

## Pivot definition

For candidate candle **C**:

- **Swing high**: compare `C.high` to highs of the `left_strength` preceding candles
  and `right_strength` following candles.
- **Swing low**: same window, using lows.
- The window must contain exactly the expected number of **closed**, consecutive
  (strictly increasing epoch) candles. Gaps, forming bars, or out-of-order bars
  prevent confirmation. No interpolation or gap bridging.

Price comparisons use **exact equality** on stored floats (no epsilon / tolerance).

### Tie policies

| Policy | Rule |
|--------|------|
| `strict` | Candidate extreme must be strictly greater (high) / lower (low) than every other extreme in the window. Any equal extreme disqualifies. |
| `first` | Equals allowed. Among a contiguous plateau of exact equal extremes that are also ≥ / ≤ the rest of the window, emit only the **earliest** candle. Confirmation always waits for the full right-side window so the earliest representative is known before emission (no premature plateau selection). |
| `all` | Every candle that independently meets the ≥ / ≤ extreme condition emits its own observation (including tied plateaus). Each has its own `event_time` and stable identity. |

Non-contiguous equal prices are not one plateau.

## Confirmation and timestamps

A candidate is evaluable only when:

1. All required right-side candles have closed.
2. The complete left / candidate / right window is present and consecutive.
3. Market-data readiness permits detection (engine gate; detector itself only sees closed candles).

| Field | Value |
|-------|--------|
| `event_time` | Candidate candle open epoch |
| `candle_epoch` | Same as candidate epoch (legacy compatibility) |
| `confirmed_at` | Open epoch of the **final required right-side** candle (the bar at which the window became complete). Aligns with Phase 0 market-time `T` comparisons used by conditions. |
| `detected_at` | Wall-clock emission time |

Developing / incomplete candidates are never emitted as confirmed swings.

Missing or invalid candles in the window: no emission. Permanent unrecoverable gaps are handled by the existing market layer; this detector does not retain pending candidates across calls.

## Canonical Observation

| Field | Value |
|-------|--------|
| `type` | `swing_high` or `swing_low` |
| `variant` | `pivot_lr` |
| `direction` | `null` |
| `detector_id` | `swing_pivot` |
| `detector_config_id` | deterministic config id |
| `reference_ids` | empty (no separate candle-observation IDs in current architecture) |
| `payload.price` | candidate high (swing high) or low (swing low) — structural extremum for future comparison |
| `payload.left_strength` | configured |
| `payload.right_strength` | configured |
| `payload.tie_policy` | configured |

Stable identity uses the existing `Observation.stable_event_id` (includes type, symbol, timeframe, candle_epoch, event_time, confirmed_at, detector_id, detector_config_id, variant). Reprocessing the same completed series is idempotent via `ObservationStore`.

## Registry

Concepts: `swing_high`, `swing_low`, `swing`.

```python
from cpab.detectors import build_default_registry

reg = build_default_registry()
det, info = reg.create_detector(
    "swing_high",
    config={"left_strength": 2, "right_strength": 2, "tie_policy": "first"},
)
assert info.min_history == 5
assert det.min_history == 5
```

## Scope exclusions (later phases)

Equal-high/low grouping, HH/HL/LH/LL, BOS/CHoCH, structure sequencing, S/R or supply/demand, SMC/ICT, setup JSON evaluation.
