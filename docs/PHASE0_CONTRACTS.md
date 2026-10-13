# Phase 0 Domain Contracts

## Observation timestamps

| Field | Meaning |
|-------|---------|
| `event_time` | Market/candle time the underlying event occurred. |
| `confirmed_at` | Market/candle time sufficient evidence existed to confirm. `None` for developing. |
| `detected_at` | Wall-clock emission time. |
| `candle_epoch` | **Legacy.** For engulfing = confirming candle. Do not equate to `event_time` for all concepts. |

## Legacy confirmation policy

1. `developing=True` → not confirmed; ineligible for normal conditions.
2. `confirmed_at` set → use that market time.
3. Else treat as confirmed at `candle_epoch` (legacy path; always present for
   normal construction).
4. If `candle_epoch` is unavailable (incomplete record), fall back to
   `int(detected_at)` and mark via `confirmation_is_detected_at_fallback()` —
   this is wall-clock, **not** a real market confirmation time.
5. Do not invent retrospective confirmation times that were never recorded.
   Explicit developing and absent-confirmation legacy take different paths.

## Developing observations

- Represented by `developing=True` (and `confirmed_at is None`).
- Excluded from normal requirements by default.
- Opt-in per requirement: `"allow_developing": true` in `require_observations` entries.
- Opt-in does **not** treat them as confirmed.

## Causal availability vs condition activation

At market evaluation time `T`:

### Causal availability (per observation)

- Confirmed observation available iff `effective_confirmed_at() <= T`.
- Developing only if the requirement opts in (`allow_developing: true`).
- **Prior confirmed observations remain available on later bars.** This is
  required for multi-component setups (e.g. swing confirmed at bar 13, retest
  at 15, engulfing at 16 must all be queryable at T=16).

### Requirement matching

Each `require_observations` entry is satisfied by causally available
observations of the requested type. Matching does **not** require every
component to have `confirmed_at == T`. When selecting the supporting set
(`min_count`), activation observations at `T` are preferred over older eligible
matches so a new same-type fact on the current bar can activate the condition.

### Condition activation (signal semantics)

When `T` is set, a condition is **met** only if:

1. Every requirement is satisfied by available observations, **and**
2. At least one supporting observation is an **activation** observation at `T`:
   - confirmed with `effective_confirmed_at() == T`, or
   - developing (opt-in) with `candle_epoch == T` or `event_time == T`.

This preserves single-observation engulfing behaviour (activates only on its
confirmation bar) while allowing multi-component setups to combine earlier
confirmations with a new trigger fact on bar `T`.

It also prevents passive re-satisfaction on every later bar when nothing new
confirms: availability alone is not enough.

### Engine-level dedup

`ConditionEngine` still tracks emitted `(condition_id, candle_epoch)` pairs so
the same evaluation event is not emitted twice within a process.

### Observation identity (idempotent emissions)

- `Observation.create` uses a **stable event id** when `detector_id` is set
  (`type|symbol|timeframe|candle_epoch|…|detector_id|detector_config_id|…`),
  so repeated `detect()` calls on the same logical event produce the same id.
- `detector_config_id` is part of the identity: two configured instances of the
  same detector type can emit distinct observations on the same bar.
- Without `detector_id`, a random UUID is used (synthetic/test observations).
- Explicit `id=` overrides both.

### Observation retention (live engine)

`ConditionEngine` holds a bounded `ObservationStore`:

- Detector emissions from each `evaluate_once` are accepted into the store
  (deduped by observation id; first write wins). Stable ids make re-emission
  of the same logical event idempotent and prevent store pollution.
- Evaluation uses **all retained** observations for the series, not only the
  current cycle's detector output. This is required for multi-component setups
  whose components confirm on different bars.
- Retention does **not** invent activation: the evaluator still requires at
  least one supporting observation newly associated with bar `T`.
- Policy: `max_size` (default 1000) drops oldest by market anchor; optional
  `max_age_seconds` prunes relative to the current evaluation epoch.
- Readiness remains authoritative: if `can_evaluate()` is false, no detection,
  retention, or condition evaluation runs.

### Live stream boundary

New facts enter the store only via detector emissions inside `evaluate_once`.
There is no external injection path. Retained priors are available for
matching but are not re-emitted as new activation events.

## Observation immutability (documented scope)

- `Observation` and `ObservationLifecycleEvent` are **frozen dataclasses**:
  top-level fields cannot be reassigned.
- `metadata` and `payload` are plain `dict`s (and nested values are unconstrained).
  Consumers **must not mutate** these after construction; treat them as
  read-only by convention.
- Phase 0 does **not** enforce deep immutability (e.g. `MappingProxyType`).
  Callers that need a defensive copy should `dict(...)` at the boundary.

## metadata vs payload

- `metadata` retained for backward compatibility (existing engulfing consumers/tests).
- `payload` is the canonical field for new concept properties.
- `Observation.create` keeps them synchronized when only one is supplied.

## Lifecycle events

- `ObservationLifecycleEvent` is append-only. Never rewrite prior events or observations.
- `lifecycle_state` on an observation is a snapshot at emission only.
- Latest state is derived via `project_lifecycle_state(events)`.
- `LifecycleDefinition` validates concept-specific allowed transitions; interaction types (`touch`, `retest`, …) need not change state.
- Invalid configured transitions raise `LifecycleTransitionError`.

## Readiness

The existing `ReadinessManager.can_evaluate()` gate is unchanged and remains authoritative.
