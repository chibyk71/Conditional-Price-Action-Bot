# Phase 0 Domain Contracts

## Observation timestamps

| Field | Meaning |
|-------|---------|
| `event_time` | Market/candle time the underlying event occurred. |
| `confirmed_at` | Market/candle time sufficient evidence existed to confirm. `None` for developing. |
| `detected_at` | Wall-clock emission time. |
| `candle_epoch` | **Legacy.** For engulging = confirming candle. Do not equate to `event_time` for all concepts. |

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

## Causal eligibility and trigger binding

At market evaluation time `T`:

- **Causal availability:** confirmed observation available iff `effective_confirmed_at() <= T`.
  Developing only if the requirement opts in (`allow_developing: true`).
- **Trigger binding (signal semantics):** a confirmed observation satisfies a
  condition at `T` only when `effective_confirmed_at() == T` (the confirmation
  bar). This preserves the original `candle_epoch == evaluation_epoch` behaviour
  for engulging/legacy and prevents an old observation from re-firing the same
  condition on every later candle.
- Developing (opt-in) observations bind when `candle_epoch == T` or `event_time == T`.
- Live engine only consumes observations emitted into the accepted stream of the
  current `evaluate_once` (detectors → engine); no external injection path.

## metadata vs payload

- `metadata` retained for backward compatibility (existing engulging consumers/tests).
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
