# Phase 3D — Zone Polarity Flip Detection

## Summary

`ZonePolarityFlipDetector` emits *potential* zone polarity-flip observations
derived exclusively from existing Phase 3B break observations.

| Source break                         | Flip variant              | Direction |
|--------------------------------------|---------------------------|-----------|
| Support zone + bearish break         | `support_to_resistance`   | `bearish` |
| Resistance zone + bullish break      | `resistance_to_support`   | `bullish` |

A flip represents a **potential** polarity change only. It does **not** claim
that the zone has successfully held as the opposite polarity, nor does it
implement retests, mitigation, invalidation, or a general zone lifecycle.

## Observation contract

| Field           | Value |
|-----------------|--------|
| `type`          | `zone_polarity_flip` |
| `variant`       | `support_to_resistance` \| `resistance_to_support` |
| `direction`     | `bearish` \| `bullish` (respectively) |
| `event_time`    | Inherited from the source break |
| `confirmed_at`  | Inherited from the source break |
| `reference_ids` | `(source_zone_id, break_observation_id)` |
| `payload.level_price` | Original zone level price |
| `payload.source_zone_variant` | `"support"` \| `"resistance"` |
| `payload.source_zone_id` / `source_break_id` | Explicit references |

Common observation fields (`detector_id`, `detector_config_id`, etc.) follow
existing conventions.

## Relationship to Phase 3B

The detector **composes** `ZoneInteractionDetector` (which itself composes
`SupportResistanceZoneDetector`). It does not reimplement zone construction or
break detection.

Only observations with:

```
type == "zone_interaction" and variant == "break"
```

are considered. Mapping rules:

1. Support zone + bearish break → `support_to_resistance` / `bearish`
2. Resistance zone + bullish break → `resistance_to_support` / `bullish`
3. Any other interaction or direction → no flip
4. At most one flip per source-zone break

## Timing and gap rules

- Process completed candles only.
- Timing is taken from the source break; never backdated or independently
  recalculated.
- Phase 3B zone eligibility, close-crossing, and gap/discontinuity rules are
  reused unchanged. A gap that suppresses a Phase 3B break also suppresses the
  corresponding flip.
- Forming candles cannot trigger a flip.
- Empty and mixed symbol/timeframe inputs follow existing detector conventions
  (empty result).

## Deterministic identity

Stable IDs include (directly or via the established mechanism):

- Symbol and timeframe
- Event and confirmation epochs
- Detector / config identity
- Flip variant and direction
- Source zone observation ID
- Source break observation ID

No random values, runtime timestamps, object addresses, or mutable counters.

## Configuration and registry

| Parameter        | Default   | Constraint |
|------------------|-----------|------------|
| `left_strength`  | `1`       | positive integer |
| `right_strength` | `1`       | positive integer |
| `tie_policy`     | `"strict"`| `strict` \| `first` \| `all` |

Parameters are forwarded to the composed interaction / zone detectors.

- Canonical concept: `zone_polarity_flip`
- Detector ID: `zone_polarity_flip`
- `min_history` matches Phase 3B (`left + 1 + right + 1`)
- `detector_config_id = deterministic_config_id({left_strength, right_strength, tie_policy})`

## Meaning and limitations

A polarity flip is a **candidate** signal that a former support (resistance)
level has been broken and *may* later act as resistance (support). Downstream
logic (retest confirmation, mitigation, invalidation) is intentionally out of
scope.

## Explicit non-goals

- Independent break detection
- Retest detection (Phase 3C)
- Zone clustering, merging, width, or tolerance
- Persistent mutable zone state or lifecycle state machine
- Claims that the flipped polarity has been confirmed by subsequent price action
- Changes to Phase 3A/3B detector behavior or observation contracts
