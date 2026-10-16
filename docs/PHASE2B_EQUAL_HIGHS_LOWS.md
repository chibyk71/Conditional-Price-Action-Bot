# Phase 2B — Equal Highs / Equal Lows (EQH / EQL)

Structural equal-high and equal-low levels formed from confirmed pivot swings.

## What EQH / EQL observations represent

- **equal_high_level** / **equal_low_level**: an immutable creation observation for a level defined by two distinct confirmed swing highs (or lows) whose prices satisfy the absolute tolerance.
- **equal_high_level_touch** / **equal_low_level_touch**: subsequent confirmed swings that qualify against an established level’s fixed representative price.

These are structural observations only. They do **not** generate trade signals, invalidate levels, or track lifecycle.

`direction` is always `null` (consistent with Phase 2A pure structural swings).

## Absolute tolerance

Equality condition for two swing prices \(P_1\), \(P_2\):

\[
|P_1 - P_2| \le \varepsilon
\]

Boundary is inclusive. \(\varepsilon\) is configured explicitly as `tolerance`; there is **no** silent default.

| Parameter | Type | Constraint | Default |
|-----------|------|------------|---------|
| `tolerance` | float | finite, ≥ 0 (true number, not bool/str/NaN/Inf) | **required** |
| `left_strength` | int | ≥ 1 | 1 |
| `right_strength` | int | ≥ 1 | 1 |
| `tie_policy` | str | `strict` \| `first` \| `all` | `strict` |

Invalid values raise `ValueError`. Zero tolerance permits exact-price matching only.

All of the above participate in `detector_config_id`. Different swing configurations cannot share levels or touch evidence.

## Grouping and tie-breaking

When a newly confirmed swing is processed:

1. **Existing levels** (same series, direction, detector config): qualify if `|price − representative| ≤ ε`. Select nearest distance; ties → earliest-created level; remaining ties → lexicographically smallest stable level ID.
2. **Ungrouped prior swings**: if no level qualifies, pair with the nearest eligible prior ungrouped swing (same ties: earliest confirmed_at, then stable swing observation ID).

Rules enforced:

- Representative price is the midpoint of the two supporting swing prices at creation and is **immutable**.
- Established levels are never automatically merged.
- A swing is never assigned to more than one level.
- **No tolerance chaining**: a new swing is compared only against a level’s fixed representative price, never against intermediate touch prices.

## Causal confirmation

A level becomes observable only when the **second** supporting swing is confirmed.

- `event_time` / `candle_epoch`: second swing’s event time (candidate open epoch).
- `confirmed_at`: second swing’s `confirmed_at` (open epoch of its final required right-side candle).
- Developing / incomplete candles are never used.
- Gaps are not bridged (inherited from SwingPivotDetector window continuity).

## Historical reconstruction and identity continuity

The detector is **stateless**: it reconstructs confirmed swings via `SwingPivotDetector.detect_all_historical` on the closed series it is given, then groups deterministically. There is no persistent level-state store.

### Minimum history (lower bound for any equal-level)

\[
\min\_history = 2 \times (\text{left} + 1 + \text{right}) - 1
\]

Derivation: one swing needs \(L+1+R\) closed bars; a second distinct confirmable pivot needs additional bars so that two full left/candidate/right windows fit under contiguous spacing.

### What is guaranteed

| Scenario | Guarantee |
|----------|-----------|
| Same full closed history, same effective config | Identical swing IDs, level IDs, creation/touch observations, supporting references |
| Prefix of a series that still contains both original supporting swings of a level | Same `level_id` and creation observation ID for that level |
| Truncated / rolling window that **excludes** the original supporting swings | That level is **not** recreated. Any new pairing of remaining swings receives a **different** `level_id` (derived from the new pair’s swing observation IDs). No invented identity for a missing historical level. |

### Explicit limitation (Option A)

Identity continuity for a previously established level requires that the supplied candle history still contains the original supporting swings. A rolling market-data window that has aged those bars out cannot reconstruct the original level identity under a pure reconstruction model without persistent state.

This is by design for Phase 2B:

- No persistent level-state storage was introduced.
- Insufficient or truncated history yields no emission of the original level (or a different identity if a new pair forms from remaining swings).
- Callers that need long-lived level continuity across truncated windows must retain the original supporting swings (or a future architectural store) outside this detector.

Tests cover both the same-history determinism case and the truncated-history non-reinvention case.

## Observation payload (creation)

| Field | Meaning |
|-------|---------|
| `level_id` | Stable deterministic level identity (distinct from observation id) |
| `representative_price` | Fixed midpoint of the two supporting prices |
| `tolerance` | Effective ε |
| `touch_count` | Always 2 at creation |
| `supporting_swing_ids` | Canonical ordered pair of swing observation ids |
| `left_strength`, `right_strength`, `tie_policy` | Effective swing config |

`reference_ids` holds the two supporting swing observation ids (sorted).

## Observation payload (touch)

| Field | Meaning |
|-------|---------|
| `level_id` | Referenced stable level identity |
| `representative_price` | Unchanged from creation |
| `swing_price` | Price of the new supporting swing |
| `swing_observation_id` | Id of the new swing observation |

`reference_ids` = `(level_id, swing_observation_id)`.

Touch observations are immutable and distinct from the creation observation. Repeated evaluation does not duplicate touches (`ObservationStore` first-write-wins).

## Registry

Concepts: `equal_high`, `equal_low`, `equal_level`, `eqh`, `eql`.

```python
from cpab.detectors import build_default_registry

reg = build_default_registry()
det, info = reg.create_detector("equal_high", config={"tolerance": 0.25})
assert info.min_history == 5
assert det.tolerance == 0.25
```

## Scope exclusions

- Level invalidation / breakout / wick or close-based invalidation
- Level expiration or lifecycle state machines
- Liquidity-sweep detection
- Buy/sell signal generation
- Persistent level-state storage
- Broad engine / observation-model redesign
