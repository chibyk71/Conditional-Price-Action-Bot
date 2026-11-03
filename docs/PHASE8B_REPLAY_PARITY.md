# Phase 8B — Replay, Parity, and Watch-List Confidence

Contract: `docs/PHASE_8B_CONTRACT_REV9.md` (Revision 9, approved).
Contract content SHA-256: `967bcd1150a02bd22c61b84a523a806add8ec90e1c4b302ff1ede513d4b7ab87`.

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

## Live-style harness limitations (C4)

The live-style path (`run_live_style`) models **cache capacity** and
**candle-closure marking** via production `CandleCache` and
`mark_closed_by_lag`.

It does **not** model lag-based withholding of recent candles as a distinct
visibility window under production server-time progression. Therefore:

- lag 0 / 1 / 3 results from the live-style harness must **not** be interpreted
  as demonstrating different recent-candle visibility under those lag settings;
- lag-based withholding has **not** been tested through `run_live_style`;
- unit-level checks of `mark_closed_by_lag` remain valid for closure marking
  alone and are not a substitute for full lag-withholding parity under live-style.

Positive replay/live-style parity claims apply only to configurations where
both paths record equivalent effective inputs that the harness can actually
represent (cache caps, annotated lag knobs, pre-closed series, HTF source labels).

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

## A15 acceptance runs

Acceptance evidence is generated only during a **full Phase 8B / full-suite pytest session**.
The evidence manifest is cleared at `pytest_sessionstart`. Meta-tests that require all
mandatory A6 keys and all 24 stall rows therefore **fail closed** if run in isolation
(empty manifest). Run `pytest tests/test_phase8b_*.py` or the full suite.

