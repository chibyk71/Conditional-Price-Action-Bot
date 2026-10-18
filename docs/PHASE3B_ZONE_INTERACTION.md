# Phase 3B — Zone Interaction Detection (Touch, Rejection, Break)

## Summary

`ZoneInteractionDetector` evaluates completed-candle interactions against
existing Phase 3A support/resistance zone observations.

| Interaction | Variant     | Direction   |
|-------------|-------------|-------------|
| Touch       | `touch`     | `null`      |
| Rejection   | `rejection` | `bullish` / `bearish` |
| Break       | `break`     | `bullish` / `bearish` |

Retests and multi-candle interaction sequences are **out of scope** and
reserved for Phase 3C.

## Integration design

The detector interface remains `detect(candles: list[Candle])`. Zone
observations are not passed as a second argument.

Following the established composition pattern (`StructureBreakDetector` →
`SwingPivotDetector`), `ZoneInteractionDetector` composes
`SupportResistanceZoneDetector` and reconstructs zones from the same closed
candle history. Zone *generation* stays in Phase 3A; this detector only
evaluates interactions against those reconstructed zone observations.

No independent zone algorithm is implemented here. No zone width or tolerance
is introduced (Phase 3A zones remain zero-width price levels).

## Interaction definitions

Let **P** be the source zone’s price level (`payload.source_price` /
`lower_bound` / `upper_bound`).

### Touch

A completed candle touches the zone when:

```
candle.low <= P <= candle.high
```

- Wick contact counts.
- Contact at exactly P counts.
- A candle entirely above or below P without touching does not count.
- `variant="touch"`, `direction=None`.

### Rejection

Requires a touch **and** a close strictly away from the level on the expected side.

| Zone        | Condition                         | Direction  |
|-------------|-----------------------------------|------------|
| Support     | touch and `close > P`             | `bullish`  |
| Resistance  | touch and `close < P`             | `bearish`  |

- Close exactly at P is **not** a rejection.
- A candle may emit both a touch and a rejection for the same source zone;
  they have distinct deterministic IDs.

### Break

Requires a genuine close-crossing transition (not merely remaining beyond).

| Zone        | Condition                                      | Direction  |
|-------------|------------------------------------------------|------------|
| Support     | `prev.close >= P` and `cur.close < P`          | `bearish`  |
| Resistance  | `prev.close <= P` and `cur.close > P`          | `bullish`  |

- Wick-only penetration is **not** a break.
- Close exactly at P does not complete a break; a later candle can break if
  its previous close satisfies the crossing rule.
- Once a genuine break is emitted for a source zone, later candles remaining
  beyond the level do **not** generate additional break events for that zone.
- Transition-only: a zone that becomes available while price is already beyond
  it does not invent a break without a valid prior/current continuous close pair
  that crosses.

## Causal availability

Interactions are generated only when the source zone is eligible:

```
zone.confirmed_at <= interaction_candle.epoch
```

- Process completed candles only (`is_closed=True`).
- Do not generate interactions using forming candles.
- Do not backdate an interaction before it could have been known.
- Mixed symbol/timeframe inputs are rejected (empty result), consistent with
  Phase 2A–3A conventions.
- `Candle.epoch` is candle-open time; `event_time` / `confirmed_at` for an
  interaction equal the interaction candle’s open epoch.

## Gap / discontinuity handling

A candle is continuous with its predecessor only when open epochs differ by
exactly the expected timeframe duration (`timeframe_seconds`). On discontinuity,
break crossing evaluation is suppressed for that candle. Touch and rejection
do not require a continuous prior candle.

## Observation contract

| Field | Value |
|-------|--------|
| `type` | `zone_interaction` |
| `variant` | `touch` \| `rejection` \| `break` |
| `direction` | `null` (touch) \| `bullish` \| `bearish` |
| `event_time` / `candle_epoch` / `confirmed_at` | Interaction candle open epoch |
| `reference_ids` | `(source_zone_observation_id,)` |
| `payload.zone_type` | `support_resistance_zone` |
| `payload.zone_variant` | `support` \| `resistance` |
| `payload.level_price` | Zone price P |
| `payload.interaction_candle_epoch` | Same as candle epoch |
| `payload.source_zone_event_time` | Source zone event time |
| `payload.source_zone_confirmed_at` | Source zone confirmed_at |
| `payload.open/high/low/close` | Interaction candle OHLC |
| `payload.previous_close` | Present when a prior continuous candle exists |

### Deterministic identity

`Observation.stable_event_id` does not include `reference_ids`. Therefore
interactions use a custom stable ID that includes the source zone observation
ID (same approach as Phase 2D structure breaks):

```
zone_interaction|{symbol}|{timeframe}|{candle_epoch}|e{event_time}|c{confirmed_at}|{detector_id}|{detector_config_id}|{direction}|{variant}|{source_zone_id}
```

This keeps observations for different source zones distinct even when they
share price and interact on the same candle, and keeps touch/rejection/break
identities distinct.

## Configuration

Same overridable parameters as `SupportResistanceZoneDetector` /
`SwingPivotDetector`:

| Parameter | Default | Constraint |
|-----------|---------|------------|
| `left_strength` | `1` | positive integer |
| `right_strength` | `1` | positive integer |
| `tie_policy` | `"strict"` | `strict` \| `first` \| `all` |

`detector_config_id = deterministic_config_id({left_strength, right_strength, tie_policy})`.

`min_history = left + 1 + right + 1` (one confirmed zone plus one additional
bar for a prior/current close pair on breaks).

## Registry concepts

- `zone_interaction` (canonical)
- `zone_touch`, `zone_rejection`, `zone_break` (aliases resolving to the same detector)

Polarized support/resistance capability names are not registered for this
detector; consumers filter on `observation.variant` and
`payload.zone_variant`.

## Historical vs incremental detection

- `detect_all_historical(candles)` reconstructs all interactions across eligible
  completed candles.
- `detect(candles)` emits only interactions whose `confirmed_at` equals the
  latest closed bar’s open epoch.
- Historical and latest-candle paths agree for the same market history.
- Break state is reconstructed deterministically from the available history
  (no persistent store between calls).

## Known limitations

- If history is truncated such that the first available candle after a zone’s
  confirmation already has a close beyond the level **and** the prior close
  also satisfies a crossing, a break is emitted once (objective transition in
  window). Callers that need absolute “never broken before window” certainty
  must supply complete history; the detector does not invent pre-window state.
- Retests are not detected (Phase 3C).

## Explicit non-scope

Retest detection, multi-candle interaction sequences, zone lifecycle
transitions (mitigation / invalidation), configurable zone widths, and trade
execution are deferred.
