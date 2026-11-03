# Phase 8B — Reconciled contract and acceptance criteria (REVIEW DRAFT, revision 8: section 6 made consistent with section 6b (the eight F1 rows are the entire mandatory stall-matrix scope; no F2/F3 stall rows in Phase 8B); revision 7: A15 enforcement rule, explicit mandatory vs deferred scope (section 6b), frozen F2 delayed-confirmation timelines; revision 6: four consistency conditions folded in (programmatic matrix with per-run evidence status, blocked vs diagnostic separation, effective-input parity, catalogue/allowlist tests); revision 5: stall matrix groups 2a/2b made consistent with A12-A14, undefined N replaced by r..E; revision 4: `Ns` dropped from analyst path, explicit s/r/t/E scenario table, evidence-boundary section)

Status: proposed for review. No implementation, branch or repository change is authorized or made.
Repo basis: `main` at `b93606e` (PR #32 merged). `src/` is identical to the earlier `e205ae4` tip I audited (PR #32 changed only docs and tests), so my source-based findings still apply.

## 0. Evidence labels
- **VERIFIED-STUB**: observed by me running real engine/lifecycle/detector code with a TA-Lib stub on synthetic candles. Preliminary evidence only. Does NOT satisfy real-environment acceptance (A15).
- **VERIFIED-SOURCE**: read directly from merged source, doc or tests.
- **PROPOSED**: requirement, not yet implemented or tested.

## 1. Verified behaviour (inputs to this contract)
| # | Finding | Label |
|---|---|---|
| V1 | Wire format = `parse_setups`; unknown top-level fields rejected; THEN top-level only; `allow_same_candle` not on step 0; expiry `{max_age_candles>=1}` only | SOURCE + STUB |
| V2 | Leaf matching is exact `type`, optional exact `variant`. Observation `direction` is not matchable; a bullish CHoCH satisfies a "choch" requirement | STUB |
| V3 | Non-canonical labels (`M15`, `H4`, `m15`, `h4`, `M30`) pass validation; detectors emit nothing (0 vs 65 swing observations); setup never fires | STUB |
| V4 | Unknown type (`equal_highs`) and registry concepts (`hh`) pass validation and never fire | STUB |
| V5 | `allow_developing: true` has no effect; developing observations never qualify | STUB |
| V6 | No history backfill: arming after an event misses it | STUB |
| V7 | Continuity: after the first accepted evaluation the next evaluated candle must be exactly +1 timeframe step. A faulted/missed candle is never accepted and is the one continuity then requires, so every later cycle is `diagnostic_only`. Reproduced for: series_ok False 1 cycle, series_ok False 5 cycles, readiness blocked 1 cycle, candle missing from series. All end PENDING, 0 signals, ~150 diagnostic cycles | STUB (gate_scenarios_summary.json) |
| V7b | Same stall when the fault occurs after IN_PROGRESS (fault at idx 95, control completes at 98): unhealthy, readiness-blocked and missing-candle runs all stay IN_PROGRESS with 0 signals, and invalidation/expiry also cannot fire afterwards | STUB |
| V7a | Caveat: live polling evaluates many times per candle interval, so a stall needs the fault to cover the whole interval in which that candle is latest. My harness evaluates once per candle | STUB / reasoning |
| V8 | Merged tests `test_phase8a_unhealthy_series_does_not_advance` and `..._discontinuous_candle_does_not_advance` assert only the immediate cycle, not recovery or permanence | SOURCE |
| V9 | `main.py` wires only `EngulfingDetector` (asserted by merged test) | SOURCE |
| V10 | Confirmation delay: swing, equal-level, market_structure, support_resistance observations have `confirmed_at` after the event (right_strength); others observed same-candle | STUB |
| V11 | rectangle / triangle / flags_pennants have exponential or slow worst-case runtime on synthetic data | STUB |
| V12 | Runs are deterministic (identical ids and states across repeats); this shows engine determinism only | STUB |

## 2. Fixtures
### 2.1 Merged Phase 8A fixtures — UNCHANGED
`phase8a_completion`, `phase8a_invalidation`, `phase8a_expiry` use `structure_break` (bos/choch) and `zone_interaction` (rejection). They are valid engine/lifecycle tests with injected observations. Under D1 they are NOT direction-safe analyst examples and must not be presented as such. The expiry fixture's outcome comes from the lifecycle anchor (max_age_candles=1), not from a missing second step.

### 2.2 Proposed additional analyst-facing examples (PROPOSED; merged fixtures not replaced)
Type/variant pairs checked against the merged `docs/PHASE8A_SETUP_CATALOGUE.md` (VERIFIED-SOURCE) and exercised through real detectors via the registry (VERIFIED-STUB).
| Proposal | Expression | Required detectors (must be supplied explicitly) | Confirmation timing | Production-wired? |
|---|---|---|---|---|
| `fx_bearish_breaker_then_impulse` (symbol stpRNG*, 15m) | THEN breaker_block/bullish_to_bearish → bearish_impulse; invalidation breaker_block/bearish_to_bullish; expiry 80 | breaker_block, impulse | all same-candle | No |
| `fx_bullish_higher_low_trigger` | AND[market_structure/hl, OR[bullish_engulfing, bullish_impulse]]; invalidation market_structure/ll; expiry 60 | market_structure, engulfing, impulse | market_structure/hl and /ll confirm LATER than the swing (right_strength); trigger candle same-candle | Only engulfing |
| `fx_bearish_flip_then_engulfing` | THEN zone_polarity_flip/support_to_resistance → bearish_engulfing (allow_same_candle true); invalidation zone_polarity_flip/resistance_to_support; expiry 40 | zone_polarity_flip, engulfing | same-candle | Only engulfing |
*Symbol unverified (Deriv `active_symbols`); timeframe canonical `15m`.
Conventions: `idx` = candle index in the series (epoch = START + idx*900). The engine is armed with candles 0..59 present; the first evaluation is idx 59; one candle is added per cycle. Each case below is an INDEPENDENT run with fresh engine, store, lifecycle and monitoring state. A setup reaches at most one terminal state per run, so outcomes are never combined in one run. Cases differ only in the candle sequence (generator seed); the setup JSON is identical across a fixture's cases.

#### Outcome test cases (VERIFIED-STUB unless marked; cycle numbers = idx of the transition)
| Case | Setup | Seed | Expected | Verified idx |
|---|---|---|---|---|
| F1-completion | fx_bearish_breaker_then_impulse | 3 | COMPLETED, 1 signal | IN_PROGRESS 92, COMPLETED 98 |
| F1-invalidation | same | 1 | INVALIDATED, no signal | 129 |
| F1-expiry | same | 4 | EXPIRED, no signal | 139 |
| F2-completion | fx_bullish_higher_low_trigger | 4 (replaces 2) | COMPLETED, 1 signal | 76 |
| F2-invalidation | same | 7 (replaces 1) | INVALIDATED | 93 |
| F2-expiry | same | none yet | not covered | gap: no seed found among 1-24 |
| F3-completion | fx_bearish_flip_then_engulfing | 1 | COMPLETED, 1 signal | IN_PROGRESS 72, COMPLETED 98 |
| F3-invalidation | same | 23 (replaces 2) | INVALIDATED | 79 |
| F3-expiry | same | none yet | not covered | gap |
Seed changes: the previously reported F2 seed 2 completed at idx 59 and F3 seed 2 invalidated at idx 60, i.e. on the first evaluation cycle, driven by warm-up history rather than by post-arming evidence. Seeds 4, 7 and 23 resolve later. (Seeds 4, 7, 23 were found by a seed scan on 24 seeds; F2/F3 outcomes are frequently decided near arming, so these fixtures are sensitive to warm-up.) Expiry coverage for F2/F3 is absent; merged fixtures already cover expiry, and a controlled candle sequence (not a seed search) is proposed to cover it. Earlier numbers in the first revision that listed several outcomes for one fixture referred to separate runs; this table replaces them.
Determinism: two repeated runs of the original cases gave identical states and signal ids (VERIFIED-STUB). Cases F2-completion (4), F2-invalidation (7) and F3-invalidation (23) were selected by scan in this revision and then repeat-run: identical states and signal ids (candle files `*_v2.json`). They are not frozen; they still need the real-environment re-check (A15).

Limits, stated explicitly:
- Real-detector use is stronger integration evidence than injected observations only because the detectors are supplied explicitly. It does not show they are wired in production.
- F2 involves a delayed-confirmation observation; its fixture must record the swing candle and the (later) confirmation candle, and evaluation outcomes must be asserted on the confirmation timeline.
- Candles are synthetic and seed-selected to reach outcomes; they are not market data and prove nothing about profitability.
- TA-Lib was stubbed; re-verification in the real environment is required before these become frozen fixtures.

## 3. STALLED (report-only)
Not a lifecycle state. No change to continuity policy; no reset, re-anchor or fabricated candle.
**Classification threshold (reporting rule only; no engine or lifecycle effect).** Let cycle j be the first cycle in which the lifecycle returns `diagnostic_only` with a continuity reason (expected epoch != actual epoch). The report labels the setup `STALLED` at the first cycle k > j such that: the instance is non-terminal, AND a further candle arrived at k, AND cycle k is also `diagnostic_only` for continuity. Until then (a continuity diagnostic at j with no further arriving candle) the report shows `CONTINUITY_DIAGNOSTIC`, not `STALLED`. After classification the report summarizes every later stalled cycle (count and list) but never re-labels or clears the first-discontinuity record. Cycles blocked by the readiness gate (lifecycle not called) neither start nor end the count; they are listed as blocked. If a setup ever returns to a non-diagnostic cycle after j, the report records `STALL_CLEARED` with the cycle and the test fails (source behaviour says this cannot happen; the report must not hide it if it does).
Required report fields: setup id; `first_discontinuity` (cycle j: expected epoch, actual epoch, last accepted epoch); `stalled_classified_at` (cycle k); count and list of subsequent diagnostic-only cycles; no signal inferred. The first observed discontinuity and the injected fault are separate fields (below).
Two distinct fields: `injected_fault_cause` (known to the test harness only: unhealthy / readiness_blocked / missing_candle / skipped_evaluation) and `engine_observed_reason` (the engine's continuity or health diagnostic text). The report must never claim the engine diagnosed a cause it cannot know.

## 4. Replay and parity requirements (PROPOSED)
R1 Thin driver around the existing `ConditionEngine.evaluate_once()`; no second engine, lifecycle, store, monitoring system or identity scheme.
R2 Explicit detector list; no inference from the catalogue; order preserved. Report: required types, types supported by supplied detectors, types actually emitted, and valid-but-unwired setups (warning, not rejection).
R3 Causal: one completed candle per step; no forming or future candle visible.
R4 Fresh engine, store, lifecycle and monitoring state per independent run; same setup and detector config across compared runs.
R5 Stable ordering: series order, detector list order, observation ordering key, signal order.
R6 Per-cycle record: candle epoch/OHLC, readiness/health result, data_ok, emitted and newly accepted observations (identity, payload), lifecycle state before/after, reason, anchor, signals, monitoring view.
R7 Parity: Run A (replay source) vs Run B (live-style: real `CandleCache`, completed-candle/confirm-lag marking via `mark_closed_by_lag`, HTF per the live path). Record and control: replay window size and cache cap (live cap is `min_candles + 100`), confirm-lag bars, HTF construction (live aggregates from 1m on poll; init fetches HTF independently), per-cycle readiness/health, detector list and order, initial state.
R7a For every parity comparison, record the effective inputs the engine actually saw (candles after window/cap, confirm-lag marking, HTF series source, readiness/health, detector list). If they differ, report INPUT_DIVERGENCE; do not claim an engine inconsistency until the input difference is ruled out.
R8 First-divergence report: cycle, field, expected, actual. A difference traced to effective-input differences is reported as INPUT_DIVERGENCE, not an engine inconsistency.
R9 Results in memory; optional deterministic JSON export; no database, CLI, alerts, broker or auto-wiring.

## 5. Load-time validation (PROPOSED; extends existing path, no parallel parser)
L1 Timeframe syntax (analyst/load path; the existing `timeframe_seconds()` and its `Ns` support stay unchanged for the engine/test path). The analyst path accepts exactly nine case-sensitive lowercase labels, no whitespace: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `1d`. Everything else is rejected with this list in the diagnostic, including `M15`, `H4`, `m15`, `h4`, `M30`, `m1`, `h1`, `1m ` and every seconds form (`30s`, `60s`, `15sec`). Seconds-based analyst setups are deferred until provider granularity is verified (O4); reconsidering them is a future contract change. Existing engine behaviour is unchanged so merged fixtures load as before. (Correction retained: earlier lists omitted `3m` and `2h`, which the engine does recognise.)
L2 Single authoritative catalogue: the code-defined data is the source of truth; `docs/PHASE8A_SETUP_CATALOGUE.md` becomes derived or is checked against it by a test (doc tables parsed and compared to the code data), so doc and code cannot drift; no second independently maintained list. Rules:  exact type; optional variant must belong to that type; unknown type or variant rejected listing valid variants. No alias normalization.
L3 Analyst-allowed V1 subset: an explicit positive allowlist of exact (type, variant) pairs. Anything not listed is rejected on the analyst path (default deny), with a message naming the nearest allowed entries. The allowlist is code-defined data alongside the catalogue (same single source, L2); the 33 entries are:
  - `bullish_engulfing` and `bearish_engulfing`: no variant
  - `bullish_impulse`, `bearish_impulse`: `atr_normalized_candle_v1`
  - `market_structure`: `hh`, `hl`, `lh`, `ll` (4)
  - `order_block`: `bullish`, `bearish` (2)
  - `breaker_block`: `bullish_to_bearish`, `bearish_to_bullish` (2)
  - `fair_value_gap`: `bullish`, `bearish` (2)
  - `inverse_fair_value_gap`: `bullish_to_bearish`, `bearish_to_bullish` (2)
  - `zone_polarity_flip`: `support_to_resistance`, `resistance_to_support` (2)
  - `swing_high`, `swing_low`: `pivot_lr`
  - `support_resistance_zone`: `support`, `resistance` (2)
  - `equal_high_level`, `equal_low_level`, `equal_high_level_touch`, `equal_low_level_touch`: `equal_abs`
  - `atr_expansion`, `atr_contraction`: `rolling_atr_v1`; `range_expansion`: `true_range_v1`
  - `double_top`, `double_bottom`, `head_and_shoulders`, `inverse_head_and_shoulders`: `swing_based_v1`
  Variant rule on the analyst path (wildcard guard): the variant must be stated explicitly and must exactly equal an allowlisted variant for that type; an omitted variant is rejected for every type that has one, because an omitted variant matches every direction/variant and would undermine D1. (For types with no variant, i.e. the engulfing pair, the variant must be omitted.) The general parser keeps accepting an omitted variant, as the engine does today, so merged fixtures and engine tests are unaffected. Identity consequence: an analyst setup with an explicit variant and an engine-path setup with the variant omitted hash differently; this is expected and documented.
  Rejected with an explanatory message because direction is ambiguous: structure_break, zone_interaction, zone_retest, fair_value_gap_lifecycle, mitigation_block, breakout_behavior. Also not allowed (default deny): pullback (stage unmatchable), rectangle, the triangles, flags and pennants (runtime), TA-Lib candlestick types (unverified at runtime). Engine/test loading of the full catalogue stays possible via a non-analyst path so merged fixtures keep loading.
  Caveat: whether the listed pattern types (double_top etc.) are acceptable to analysts at their runtime cost rests on my stub runs; A15 must confirm.
L4 Three separate report dimensions per setup type: catalogue-valid / analyst-allowed / wired in the supplied detector list. Valid-but-unwired = warning.
L5 Documentation must also state: direction unmatchable; delayed-confirmation types; no backfill; one-shot lifecycle; `allow_developing` ignored.

## 6. Reconciled acceptance criteria A1–A15 (PROPOSED)
Original A1–A15 list not available to me; this is a reconciliation against the criteria you supplied.
- **A1 Schema reuse**: all fixtures (merged and proposed) load via `parse_setup`/`setups_from_config`; unknown top-level field, nested THEN, THEN in invalidation, step-0 `allow_same_candle`, malformed expiry are rejected. No parallel parser.
- **A2 Identity**: config identity stable under key order; unchanged by name/description; changes when behavior fields change; signal and instance ids deterministic.
- **A3 Load-time validation**: L1–L2 enforced with actionable diagnostics (the nine accepted labels; valid variants listed); aliases, registry concepts and every non-canonical timeframe (`M15`, `H4`, `m1`, `30s`, `60s`, `15sec`) rejected on the analyst path; two separate tests pass: (i) every documented catalogue entry equals the authoritative code-defined catalogue (both directions); (ii) the analyst allowlist is checked independently: it is a subset of the code catalogue and equals the 33 pairs in L3.
- **A4 Analyst subset**: L3 enforced as a default-deny allowlist of the 33 exact pairs; an omitted variant, an unlisted variant and every direction-ambiguous type are rejected on the analyst path; the three proposed examples (F1-F3) are added to the analyst-facing docs only after real-environment verification (A15) and remain labelled synthetic integration fixtures; direction-ambiguous merged fixtures stay engine/lifecycle fixtures and are never promoted; merged fixtures still load through the engine path and are documented as engine/lifecycle fixtures, not analyst examples.
- **A5 Wiring report**: L4 three dimensions reported; unwired type warns, does not reject; production wiring unchanged (merged `main.py` test stays green).
- **A6 Outcomes (independent cases)**: every row of the outcome table is its own test case with a fresh engine and lifecycle state: merged fixtures with injected observations (completion, invalidation, expiry; each its own run) and the proposed fixtures with explicitly supplied real detectors. Each case asserts terminal state, transition cycle, signal count (1 deduplicated signal for completion, 0 otherwise) and deterministic signal id. No case expects more than one terminal outcome. For F2 (delayed confirmation) the three points are frozen below (VERIFIED-STUB with the TA-Lib stub; re-verified in A15; a real-TA-Lib run that moves any value invalidates the freeze and requires a contract amendment). Candle index i has epoch 1699992000 + 900*i; evaluation is armed after candle 59 and first evaluates candle 59.
  | Case | Seed | Observation | Swing/event candle (idx, epoch) | Confirmation candle (idx, epoch) | First cycle that can consume it | Terminal transition |
  |---|---|---|---|---|---|---|
  | F2-completion | 4 | market_structure/hl (+ bullish_engulfing on idx 75, same candle) | 75, 1700059500 | 76, 1700060400 | cycle idx 76 (epoch 1700060400) | COMPLETED at 76 |
  | F2-invalidation | 7 | market_structure/ll | 92, 1700074800 | 93, 1700075700 | cycle idx 93 (epoch 1700075700) | INVALIDATED at 93 |
  Causal assertions: in F2-completion the bullish_engulfing is already in the store at cycle 75 but the setup stays PENDING at cycle 75 and no signal is emitted, because market_structure/hl is confirmed only at 76; a bullish_impulse at idx 64 (OR branch) also does not complete the setup without the hl. The setup completes at 76 and never earlier, and the observation's confirmed_at is 76's epoch. In F2-invalidation the setup is not INVALIDATED at cycle 92 and becomes INVALIDATED at 93 (ll confirmed 93); a bullish_engulfing and bullish_impulse at idx 92 do not complete it (no hl).
- **A7 Determinism and ordering**: repeated runs produce identical per-cycle records, observation order, signal ids; stable series/detector ordering.
- **A8 State isolation**: independent runs start from fresh state; no state leaks between runs (test with interleaved runs).
- **A9 Causality and input validation**: no forming/future candle visible; duplicate, out-of-order, malformed and incomplete candle inputs handled per existing contracts and visible in diagnostics; first-write-wins observation acceptance preserved.
- **A10 Parity**: Identical outcomes from two runs of the same engine are not sufficient evidence of parity. Each run records the candles actually visible to the engine each cycle (after cache cap, windowing and closed-candle marking), plus HTF construction, readiness, health and detector order; any difference is reported as INPUT_DIVERGENCE before outcomes are compared. Run A vs Run B per R7 with window, cap, confirm-lag, HTF source, readiness and detector list recorded and controlled; matching per-cycle results; injected input difference is reported as INPUT_DIVERGENCE.
- **A11 First-divergence report**: an injected deliberate difference yields the correct first cycle, field, expected and actual.
- **Stall-test cycle rule (applies to A12–A14)**. Definitions: a clean control is the same setup, detectors and candle sequence with no injected fault; t = the cycle index at which the control reaches its terminal transition (COMPLETED, INVALIDATED or EXPIRED); a fault of duration d in {1, 5} cycles occupies cycles s, s+1, ..., s+d-1; r = s + d is the first recovery cycle (post-recovery checks begin at r, never inside the fault); the check window is cycles r through E, where E = max(r + 4, t + 2) (at least 5 post-recovery cycles, and through the control's terminal cycle plus 2, whichever ends later). Requirements: (1) every fault class (unhealthy data, readiness block, missing candle) is run at d = 1 and d = 5; (2) the matrix has exactly two groups: (2a) BEFORE IN_PROGRESS: the completion, invalidation and expiry controls, with the setup PENDING at s; (2b) AFTER IN_PROGRESS: only completion controls whose setup actually reaches IN_PROGRESS before s (F1-completion is the only such control in Phase 8B; F3-completion is deferred per section 6b). Invalidation and expiry controls that never enter IN_PROGRESS (F1-invalidation, F1-expiry) have no after-IN_PROGRESS row, and none is required; (3) s < t, so the fault precedes the transition it is meant to prevent (and for 2b, s >= the IN_PROGRESS transition cycle); (4) the three controls in 2a show each of the three transitions prevented by the existing continuity failure where the control would have produced it. Assertions in the check window: the faulted run never reaches the control's terminal state or emits a signal, while the control does at t; faulted cycles do not advance state; recovery does not repair continuity or re-anchor; every cycle from r through E is diagnostic-only; STALLED reported per section 3 without inventing an engine diagnosis.
  **Executable scenario table (required in the implementation prompt; every scenario states s, r, t, E explicitly).** Scenario matrix = fault class (unhealthy, readiness_blocked, missing_candle) x rows below, so 3 x 8 = 24 faulted runs plus one clean control per control row. Finding from my stub data: F1-invalidation and F1-expiry go PENDING straight to their terminal state and never enter IN_PROGRESS, so the "after IN_PROGRESS" variant exists only for F1-completion in Phase 8B (F3-completion, transitions 72/98, is deferred per section 6b).
  | Control | State at s | d | s | r = s+d | t | E = max(r+4, t+2) | row note (applies to the row only; no label means the row is verified across all three fault classes) |
  |---|---|---|---|---|---|---|---|
  | F1-completion | PENDING | 1 | 70 | 71 | 98 | 100 | d=1 VERIFIED-STUB |
  | F1-completion | PENDING | 5 | 70 | 75 | 98 | 100 | VERIFIED-STUB (unhealthy, 5 cycles); other classes PROPOSED |
  | F1-completion | IN_PROGRESS | 1 | 95 | 96 | 98 | 100 | VERIFIED-STUB |
  | F1-completion | IN_PROGRESS | 5 | 95 | 100 | 98 | 104 | PROPOSED, not run |
  | F1-invalidation | PENDING | 1 | 70 | 71 | 129 | 131 | PROPOSED, not run |
  | F1-invalidation | PENDING | 5 | 70 | 75 | 129 | 131 | PROPOSED, not run |
  | F1-expiry | PENDING | 1 | 70 | 71 | 139 | 141 | PROPOSED, not run |
  | F1-expiry | PENDING | 5 | 70 | 75 | 139 | 141 | PROPOSED, not run |
  **Programmatic generation (condition 1).** The implementation must generate the full matrix in code (every table row x every fault class = 24 scenarios; these eight F1 rows are the entire mandatory stall-matrix scope for Phase 8B), not hand-list scenarios. Each generated run records its own evidence status (`not_run`, `stub_only`, `real_env_passed`, `real_env_failed`) keyed by (control, state at s, fault class, d). A meta-test enumerates the required set and fails if any required scenario is absent, skipped or lacks a status; a scenario may not be marked `real_env_passed` except by a real-environment run (A15).
  No F2 or F3 stall rows exist in Phase 8B (F2 outcomes resolve near arming, F2/F3 expiry cases do not exist; section 2.2 gaps); they are deferred per section 6b and must not be added by the implementation agent. The 'status' column is conservative: only the specific stub runs recorded in V7/V7b (see section 1; their exact class and duration combinations are in `fixtures/gate_scenarios_summary.json`) count as VERIFIED-STUB, and any (class, duration) cell not in that file is PROPOSED. An implementer must not treat any cell as verified; A15 re-establishes all of them.
  Cycle bookkeeping per scenario: fault interval [s, r-1]. For unhealthy data these cycles are diagnostic-only (data_ok False path). For a readiness block these cycles are recorded as blocked and the lifecycle is not called (so no diagnostic is produced in the fault interval). For a missing candle, candle s.. s+d-1 are absent from the series. Post-recovery cycles r..E are the continuity diagnostics; the first such cycle is j = r, and STALLED is classified at r+1 (section 3). Blocked cycles and post-recovery continuity diagnostics are reported in separate lists and must not be merged or counted toward each other. Test assertion (A13 and A14): no readiness-blocked cycle appears in the diagnostic-only list or in the STALLED count, and the lifecycle call count in the fault interval is zero for readiness blocks.
  Concrete values, F1-completion (control: IN_PROGRESS at 92, COMPLETED t = 98; VERIFIED-STUB for d=1 and the 5-cycle PENDING case; the 5-cycle IN_PROGRESS case is PROPOSED and must be run): PENDING fault at s = 70: d = 1, r = 71, E = max(75, 100) = 100; d = 5 (cycles 70-74), r = 75, E = max(79, 100) = 100. IN_PROGRESS fault at s = 95: d = 1, r = 96, E = max(100, 100) = 100; d = 5 (cycles 95-99, which spans the control's completion at 98), r = 100, E = max(104, 100) = 104 (the earlier bound of 100 was insufficient for this case). For F1-invalidation (t = 129) and F1-expiry (t = 139) I have not run fault injections on those exact cases (my earlier stub observation that invalidation and expiry cannot fire after a continuity fault came from the completion scenarios); their s, r, t and E values are the ones in the table above, frozen by this contract; they are PROPOSED (not yet run) and A15 verifies them in the real environment. F2/F3 stall cases are not part of Phase 8B (section 6b).
- **A12 Permanent-stall regression (unhealthy data)**: series_ok False for 1 cycle and for 5 cycles over the matrix groups 2a and 2b of the stall-test cycle rule (the table rows), checked over r..E, each paired with a clean control; assert (a) fault cycle diagnostic-only, no state advance; (b) after health recovers every later cycle stays diagnostic-only with the continuity reason naming expected and actual epochs; (c) no signal and no terminal transition at any cycle throughout r..E, including the candles that complete, invalidate or expire the control at t; (d) report classifies STALLED with injected cause `unhealthy`, engine-observed reason recorded separately.
- **A13 Permanent-stall regression (readiness block)**: readiness blocked for 1 and 5 cycles, same matrix groups 2a/2b (table rows) and r..E window (blocked cycles visible in monitoring, lifecycle not called), then recovery: same (b)–(d) with cause `readiness_blocked`.
- **A14 Permanent-stall regression (missing candle) and STALLED semantics**: candle(s) absent from the series (1 and 5 consecutive), same matrix groups 2a/2b (table rows) and r..E window, then later candles arrive: same (a)–(d) with cause `missing_candle`; plus: lifecycle state set unchanged (no new state), no reset/re-anchor/fabricated candle, STALLED appears only in the replay report. Also a negative control: same series without fault completes with one signal. Existing continuity behavior is asserted, not modified.
- **A15 Real-environment gate**: full pytest suite (focused and full) passes with real TA-Lib and the project's actual test dependencies. Enforcement rule: the acceptance-mode meta-test FAILS unless every mandatory fixture case and every mandatory generated stall scenario (section 6b) has status `real_env_passed`. Any `not_run`, `stub_only` or `real_env_failed` status fails acceptance. Only a real-environment run can set `real_env_passed`; stub, synthetic-shim or developer-mode runs can set at most `stub_only`. A separate development mode may tolerate `not_run`/`stub_only` so the agent can iterate, but it cannot produce an acceptance result. Stub/synthetic results are not acceptance evidence.

## 6a. Evidence boundary (what is NOT yet established)
- F2 and F3 expiry outcome cases: do not exist (no seed found; controlled candle sequence proposed). Explicit gap.
- Fault injections on F1-invalidation and F1-expiry (and every F2/F3 stall row): not run. Explicit gap.
- F1-completion 5-cycle IN_PROGRESS fault: not run.
- The whole proposed fixture matrix is therefore NOT verified; only the cases marked VERIFIED-STUB were run, with a TA-Lib stub and synthetic candles, and A15 (real TA-Lib, full suite) must establish every result before acceptance.
- O3 (production cadence) and O4 (provider symbol/granularity) are unverified, are not reasons to change lifecycle behavior or delay contract lock, and the contract makes no live-market or provider-backed claim.

## 6b. Acceptance scope: mandatory vs deferred (frozen by this contract)
MANDATORY for acceptance (each must reach `real_env_passed`):
- A6 outcome cases: the three merged Phase 8A fixtures (completion, invalidation, expiry; injected observations); F1 completion, invalidation, expiry; F2 completion and invalidation; F3 completion and invalidation.
- Stall matrix: the eight F1 rows in section 6 x three fault classes = 24 generated scenarios, each with its own clean control.
- All A1-A14 criteria not tied to a deferred item below.
DEFERRED (explicitly outside Phase 8B acceptance; must not be added by the implementation agent; each needs a later contract amendment with frozen candles and indices):
- F2-expiry and F3-expiry cases. Expiry is covered by the merged expiry fixture and F1-expiry.
- F3-completion stall rows (including the after-IN_PROGRESS row) and all F2 stall rows.
- Seconds-based analyst timeframes; production-cadence measurement (O3); provider symbol/granularity verification (O4).
Because F2/F3 expiry and the deferred stall rows are out of scope, the analyst-facing example set may carry a note that F2/F3 expiry is not covered, and no document may claim full outcome coverage for F2/F3.

## 7. Exclusions
Telegram or alerts; broker/autotrade; automatic detector wiring; second engine/lifecycle/store/monitoring; schema redesign or trade-plan semantics (entry, SL, TP stay descriptive); direction field; lifecycle re-anchor/reset; silent gap repair; strategy or profitability claims; CLI; merging without explicit authorization.

## 8. Open items for the review
O1 Confirm the proposed fixtures are wanted as examples, and whether F2's delayed-confirmation timeline is acceptable in a frozen fixture.
O2 Confirm the A12–A14 stall-regression scope (settled: 1 and 5 cycle faults; check window r..E; no free parameter N).
O3 (decided in review: measurement task, not a policy change) Measure production polling/evaluation cadence with controlled timestamps or a deterministic harness that exercises the real loop logic, separate from A15: how long a candle is exposed to evaluation and whether repeated evaluations can observe a fault. No live-market experiment. Original note: live exposure of the stall depends on how often the production loop evaluates per candle; to be measured in the real environment rather than assumed.
O4 (decided in review: remains unverified; fixtures stay synthetic integration tests, not provider-backed) `stpRNG` symbol and Step Index provider granularity unverified.
