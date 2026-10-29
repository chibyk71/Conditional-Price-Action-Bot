# Phase 8A — Setup JSON Contract & Catalogue

## Purpose

Document the **verified emitted observation identifiers** that setup
expressions may reference, provide three canonical setup fixtures that
exercise completion / invalidation / expiry through the existing consumer
path, and record the **live detector wiring gap** in production entry.

This phase does **not** change schema, parser, evaluator, lifecycle,
monitoring, or `main.py` detector wiring.

## Matching semantics (unchanged)

Implementation reference: `ExpressionEvaluator` leaf matching in
`src/cpab/conditions/expression_evaluator.py`.

- Matching uses the **exact emitted `Observation.type` string**.
- If a requirement specifies `variant`, it must match the observation’s
  `variant` **exactly**.
- If a requirement omits `variant`, any variant (including `None`) of that
  type satisfies the leaf.
- **Registry concept aliases are not substituted.** A setup that requires
  `type: "sr_zone"` will not match an observation whose type is
  `support_resistance_zone`, even though the registry indexes both under the
  same detector capability.
- **Pullback lifecycle / termination reasons are not variants.** Values such
  as `trend_resumption`, `structural_failure`, and `invalidation` appear in
  pullback payloads / metadata; they are not the observation `variant` field
  and must not be written as `variant` in setup JSON.
- Prices, stops, targets, and similar fields in setup descriptions (or in
  observation payloads) are **descriptive information only**. They are not
  additional schema fields on `SetupDef` / `ObservationRequirement`.

## Detector configuration vs emitted identifiers

Some detectors accept **constructor / registry configuration** values that are
not themselves written onto emitted observations:

- **Pullback** constructor `variant`: `structure_based` | `impulse_based` | `all`.
  Observations always use `type=pullback` with emitted `variant` of
  `structure_based` or `impulse_based` only. Config `all` selects both
  reconstructors; it is never the observation variant string.
- Lifecycle / payload fields (e.g. pullback termination reasons) are not
  observation variants and must not appear as `variant` in setup JSON.

Setup requirements must reference **emitted** `type` / optional `variant`
pairs from the tables below, not detector config knobs or registry aliases.

## Catalogue coverage disclaimer

Catalogue coverage means: the type/variant pair is constructed at a known
site in detector code on this repository tip. It does **not** imply that
production currently instantiates the corresponding detector, or that the
pair is emitted on every live series. See [Live detector wiring gap](#live-detector-wiring-gap).

## Verified emitted type / variant pairs

Pairs are taken from observation construction sites (`Observation.create` /
equivalent) and detector module contracts. Registry aliases are listed only
where they differ from the emitted type, for clarity — they are **not**
valid requirement types unless a detector also emits that string.

### Candle patterns

| Emitted type | Variants | Construction / notes |
|---|---|---|
| `bullish_engulfing` | *(none)* | `EngulfingDetector`; also TA-Lib `CDLENGULFING` path via `_observation_type` |
| `bearish_engulfing` | *(none)* | Same |

**TA-Lib distinction:** Registry concepts include pattern aliases and names
such as `engulfing`, `CDLENGULFING`, and direction-qualified forms. The
adapter emits **direction-prefixed** types (`bullish_<base>` /
`bearish_<base>` from the resolved pattern name with the `CDL` prefix
stripped). Setup requirements must use the **emitted** type, not the
registry alias or raw TA-Lib function name, unless those strings are
themselves emitted (they are not for the default engulfing path).

### Swings and structure

| Emitted type | Variants | Construction / notes |
|---|---|---|
| `swing_high` | `pivot_lr` | `swing.py` |
| `swing_low` | `pivot_lr` | `swing.py` |
| `equal_high_level` | `equal_abs` | `equal_levels.py` |
| `equal_low_level` | `equal_abs` | `equal_levels.py` |
| `equal_high_level_touch` | `equal_abs` | `equal_levels.py` |
| `equal_low_level_touch` | `equal_abs` | `equal_levels.py` |
| `market_structure` | `hh`, `lh`, `hl`, `ll` | `market_structure.py` |
| `structure_break` | `bos`, `choch` | `structure_break.py` |

### Zones and zone lifecycle

| Emitted type | Variants | Construction / notes |
|---|---|---|
| `support_resistance_zone` | `support`, `resistance` | Emitted type is `support_resistance_zone` (not registry alias `sr_zone`) |
| `zone_interaction` | `touch`, `rejection`, `break` | `zone_interaction.py` |
| `zone_retest` | `retest` | `zone_retest.py` |
| `zone_polarity_flip` | `support_to_resistance`, `resistance_to_support` | `zone_polarity_flip.py` |

### Fair value gaps and blocks

| Emitted type | Variants | Construction / notes |
|---|---|---|
| `fair_value_gap` | `bullish`, `bearish` | `fair_value_gap.py` |
| `fair_value_gap_lifecycle` | `first_touch`, `partial_mitigation`, `full_mitigation`, `invalidation` | `fair_value_gap_lifecycle.py` |
| `inverse_fair_value_gap` | `bullish_to_bearish`, `bearish_to_bullish` | Derived from FVG lifecycle invalidations; not an `invalidation` variant on IFVG itself |
| `order_block` | `bullish`, `bearish` | `variant` equals direction; composed from confirmed BOS only |
| `breaker_block` | `bullish_to_bearish`, `bearish_to_bullish` | `breaker_block.py` |
| `mitigation_block` | `touch`, `partial_fill`, `full_fill` | `mitigation_block.py` |

### Classical patterns

| Emitted type | Variants | Construction / notes |
|---|---|---|
| `double_top` | `swing_based_v1` | Type is the concept name |
| `double_bottom` | `swing_based_v1` | |
| `head_and_shoulders` | `swing_based_v1` | |
| `inverse_head_and_shoulders` | `swing_based_v1` | |
| `rectangle` | `swing_based_v1` | |
| `ascending_triangle` | `swing_based_v1` | |
| `descending_triangle` | `swing_based_v1` | |
| `symmetrical_triangle` | `swing_based_v1` | |
| `bullish_flag` | `swing_based_v1` | |
| `bearish_flag` | `swing_based_v1` | |
| `bullish_pennant` | `swing_based_v1` | |
| `bearish_pennant` | `swing_based_v1` | |

### Behavior / volatility / impulse / pullback / breakout

| Emitted type | Variants | Construction / notes |
|---|---|---|
| `atr_expansion` | `rolling_atr_v1` | Crossing-only expansion |
| `atr_contraction` | `rolling_atr_v1` | Crossing-only contraction |
| `range_expansion` | `true_range_v1` | |
| `bullish_impulse` | `atr_normalized_candle_v1` | |
| `bearish_impulse` | `atr_normalized_candle_v1` | |
| `pullback` | `structure_based`, `impulse_based` | Emitted logical variants only. Detector constructor also accepts config value `all` (run both reconstructors); that config is **not** an observation `variant`. Termination reasons in payload (`trend_resumption`, `structural_failure`, `invalidation`) are **not** variants |
| `breakout_behavior` | `breakout_confirmed`, `breakout_failed`, `breakout_follow_through` | `variant` equals event type; follow-through is opt-in on the detector |

## Canonical fixtures

JSON fixtures under `tests/fixtures/phase8a/`:

| Fixture id | Intended terminal outcome |
|---|---|
| `phase8a_completion` | `COMPLETED` after THEN: `structure_break`/`bos` then `zone_interaction`/`rejection` |
| `phase8a_invalidation` | `INVALIDATED` after first THEN step via `structure_break`/`choch` invalidation expression |
| `phase8a_expiry` | `EXPIRED` under `max_age_candles` when the second THEN step never arrives |

Fixtures are loaded through `setups_from_config` / `parse_setup` (no parallel
parser). Consumer tests inject verified observations via the existing
`ScriptedDetector` / engine harness; they do **not** claim live detector
production of those observations.

## Live detector wiring gap

**Verified on this branch’s base and unchanged by this phase:**

`src/cpab/main.py` currently instantiates only:

```python
detectors = [EngulfingDetector()]
```

Production therefore does **not** emit the full documented catalogue through
the default entry point. Configured setups that require types other than
those produced by `EngulfingDetector` (for example `structure_break` or
`zone_interaction`) will not receive matching observations in live runs
until detector wiring is addressed in a **separately authorized** phase.

This phase:

- does **not** change the detector list in `main.py`;
- does **not** auto-wire detectors from setup JSON;
- does **not** imply that configured setups receive observations from
  detectors that are not instantiated.

## Non-goals (Phase 8A)

- No new schema, parser, evaluator, lifecycle, or monitoring system
- No Telegram, alerts, broker integration, or autotrading
- No invented observation types or variants
- No matching via registry aliases
- No unrelated cleanup or refactoring
