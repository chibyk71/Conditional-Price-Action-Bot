# Phase 8B — Replay, Parity, and Watch-List Confidence

Contract: `docs/PHASE_8B_CONTRACT_REV8.md` (revision 8).

## Scope

Mandatory:
- Thin replay driver around existing `ConditionEngine.evaluate_once()`
- Effective-input parity reporting (`INPUT_DIVERGENCE`)
- Report-only `STALLED` classification
- Analyst-path load-time validation (nine timeframes, 33-pair allowlist)
- Independent outcome fixtures (merged 8A + F1–F3 completion/invalidation)
- Programmatic 24-scenario F1 stall matrix with A15 evidence tracking

Deferred (not implemented):
- F2/F3 expiry outcomes
- F2/F3 stall rows
- Seconds-based analyst timeframes
- O3 production cadence measurement
- O4 provider symbol/granularity verification
- Automatic detector wiring, alerts, CLI, second engine

## Analyst path

Module: `cpab.conditions.analyst` / `cpab.conditions.catalogue`

- Timeframes accepted (exact, case-sensitive, no whitespace):
  `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `1d`
- Default-deny allowlist of 33 `(type, variant)` pairs (see catalogue module)
- Variant-bearing types require an explicit variant
- Direction-ambiguous types rejected: `structure_break`, `zone_interaction`,
  `zone_retest`, `fair_value_gap_lifecycle`, `mitigation_block`, `breakout_behavior`
- Engine/test path (`parse_setup`) is unchanged so merged fixtures still load

## Replay

Module: `cpab.replay`

- `run_replay(ReplayRunConfig)` — one completed candle per step, fresh state
- Explicit detector list (order preserved); no auto-wiring
- Per-cycle records: candle, health/readiness, observations, lifecycle, signals
- `compare_replay_runs` — effective inputs first; `INPUT_DIVERGENCE` vs engine
- `classify_stalled` — report-only; readiness-blocked cycles excluded

## STALLED (report-only)

Not a lifecycle state. Continuity policy is unchanged.

- First continuity diagnostic → `CONTINUITY_DIAGNOSTIC`
- Later diagnostic with a further candle → `STALLED`
- Unexpected recovery → `STALL_CLEARED` (test failure signal)
- `injected_fault_cause` is harness-only; never claimed as engine diagnosis

## Synthetic fixtures

F1–F3 examples are synthetic integration fixtures. They are not market data,
provider-backed, or profitability evidence. F2/F3 expiry is not covered.

## Production wiring

`main.py` still instantiates only `EngulfingDetector()`. Valid-but-unwired
types produce warnings, not rejection.
