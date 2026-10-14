# Phase 2A — Swing Detection (Pivot L/R)

Configurable left/right-strength pivot swing highs and swing lows.

## Configuration

| Parameter | Type | Constraint | Default |
|-----------|------|------------|---------|
| `left_strength` | int | ≥ 1 (true `int`, not bool/float) | 1 |
| `right_strength` | int | ≥ 1 (true `int`, not bool/float) | 1 |
| `tie_policy` | str | `strict` \| `first` \| `all` | `strict` |

Invalid values raise `ValueError` at construction / registry validation. No silent
substitution or coercion (`1.5`, `True`, `"2"` are rejected).

`detector_config_id` is always derived from the effective
`{left_strength, right_strength, tie_policy}` map via `deterministic_config_id`,
including for detectors constructed **outside** the registry.

`min_history = left_strength + 1 + right_strength`.

## Pivot definition

For candidate candle **C**:

- **Swing high**: compare `C.high` to highs of the `left_strength` preceding candles
  and `right_strength` following candles.
- **Swing low**: same window, using lows.
- The window must contain exactly the expected number of **closed** candles with
  **exact timeframe spacing** (e.g. 60s for `1m`). Gaps, forming bars, out-of-order
  bars, or unknown timeframe labels prevent confirmation. No interpolation or gap
  bridging.

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

1. All required right-side candles have closed (`is_closed=True`).
2. The complete left / candidate / right window is present with exact timeframe step.
3. Market-data readiness permits detection (engine gate; detector itself only sees
   closed candles handed to it).

| Field | Value |
|-------|--------|
| `event_time` | Candidate candle open epoch |
| `candle_epoch` | Same as candidate epoch (legacy compatibility) |
| `confirmed_at` | Open epoch of the **final required right-side** candle |
| `detected_at` | Wall-clock emission time |

### Phase 0 reconciliation of `confirmed_at`

Phase 0 conditions evaluate causal availability with:

```text
effective_confirmed_at() <= T
```

where `T` is the **evaluation candle's open epoch**. Detectors only receive closed
candles. Therefore storing the confirming bar's **open epoch** as `confirmed_at` is
the correct causal boundary:

- The swing is **not** eligible on any bar before the confirming candle.
- The swing **is** first eligible on the confirming bar itself (`T == confirmed_at`),
  which is the bar whose close completed the right-side evidence.
- A forming (not-yet-closed) right-side candle never reaches the detector, so the
  swing cannot become eligible before its evidence is available.

This matches engulfing and TA-Lib adapters, which also use candle open epoch as
`confirmed_at`.

Developing / incomplete candidates are never emitted as confirmed swings.

Missing or invalid candles in the window: no emission. Permanent unrecoverable gaps
are handled by the existing market layer; this detector does not retain pending
candidates across calls.

## Canonical Observation

| Field | Value |
|-------|--------|
| `type` | `swing_high` or `swing_low` |
| `variant` | `pivot_lr` |
| `direction` | `null` |
| `detector_id` | `swing_pivot` |
| `detector_config_id` | deterministic config id (always set) |
| `reference_ids` | empty (no separate candle-observation IDs in current architecture) |
| `payload.price` | candidate high (swing high) or low (swing low) — structural extremum for future comparison |
| `payload.left_strength` | configured |
| `payload.right_strength` | configured |
| `payload.tie_policy` | configured |

Stable identity uses the existing `Observation.stable_event_id` (includes type,
symbol, timeframe, candle_epoch, event_time, confirmed_at, detector_id,
detector_config_id, variant). Reprocessing the same completed series is idempotent
via `ObservationStore`.

## Registry

Concepts: `swing_high`, `swing_low`, `swing`.

Factories are invoked via an explicit signature check (`inspect.signature`);
`TypeError` raised inside a factory body is **not** swallowed as a signature mismatch.

```python
from cpab.detectors import build_default_registry

reg = build_default_registry()
det, info = reg.create_detector(
    "swing_high",
    config={"left_strength": 2, "right_strength": 2, "tie_policy": "first"},
)
assert info.min_history == 5
assert det.min_history == 5
assert det.detector_config_id == info.detector_config_id
```

## Scope exclusions (later phases)

Equal-high/low grouping, HH/HL/LH/LL, BOS/CHoCH, structure sequencing, S/R or
supply/demand, SMC/ICT, setup JSON evaluation.
