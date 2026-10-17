# Phase 3A — Support / Resistance Zones from Confirmed Swings

## Summary

`SupportResistanceZoneDetector` emits zero-width support and resistance zone
observations derived exclusively from confirmed Phase 2A swing pivots.

| Source swing | Zone variant |
|--------------|--------------|
| `swing_low`  | `support`    |
| `swing_high` | `resistance` |

Equal-high/low levels, supply/demand, and order blocks are out of scope.

## Observation contract

| Field | Value |
|-------|--------|
| `type` | `support_resistance_zone` |
| `variant` | `support` \| `resistance` |
| `direction` | `null` (polarity is only in `variant`) |
| `event_time` / `candle_epoch` | Source swing `event_time` |
| `confirmed_at` | Source swing `confirmed_at` |
| `reference_ids` | `(source_swing_id,)` |
| `payload.lower_bound` | Source swing price |
| `payload.upper_bound` | Source swing price (zero-width) |
| `payload.source_price` | Same price |
| `payload.left_strength` / `right_strength` / `tie_policy` | Detector config |

No width, percentage, ATR, or tolerance configuration. Bounds must be finite and
`lower_bound <= upper_bound`.

## Configuration

Same overridable parameters as `SwingPivotDetector`:

| Parameter | Default | Constraint |
|-----------|---------|------------|
| `left_strength` | `1` | positive integer |
| `right_strength` | `1` | positive integer |
| `tie_policy` | `"strict"` | `strict` \| `first` \| `all` |

`detector_config_id = deterministic_config_id({left_strength, right_strength, tie_policy})`.

`min_history = left + 1 + right`.

## Causal correctness

Zones inherit timing from the source swing. A zone is never eligible before its
source swing (`is_eligible_at` uses `confirmed_at`). Gaps and non-consecutive
windows are handled by the underlying swing detector (no gap bridging).

## Overlapping / duplicate prices

Each confirmed swing produces its own zone observation. Zones at the same price
are not merged or suppressed; identity is deterministic per source swing via
`Observation.stable_event_id` (includes type, variant, detector/config identity,
and timing fields).

## Explicit non-scope

Zone touch/entry/exit, rejection, breaks, retests, polarity flips, mitigation,
invalidation, lifecycle transitions, equal-level-derived zones, configurable
widths, and trade execution are deferred.
