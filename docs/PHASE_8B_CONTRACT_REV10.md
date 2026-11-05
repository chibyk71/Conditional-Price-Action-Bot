# Phase 8B — R1 ratification addendum (operative as Revision 10)

Status: **OPERATIVE**. Owner formally approved R1 on **2026-11-06 (Africa/Lagos)** (2026-11-05 UTC) as a narrow amendment to Rev8. This document is the consolidated operative authority statement for Phase 8B. It does not authorize runtime, test, detector, lifecycle, or configuration changes.

Repository reference reviewed: `main` at `3e07dd3c84a7112ca08bd4741ce26ffc17a6e2be`.

## Authority and versioning

| Document | Role | Identity |
|---|---|---|
| Rev8 | Operative baseline (historical record, unchanged) | `docs/PHASE_8B_CONTRACT_REV8.md`, blob `640f4e6b4c26f6f425d910c5a696dc3e6d5cc2f0` |
| R1 | Approved narrow amendment to Rev8 for the readiness-blocked fault class only | Approved 2026-11-06 (Africa/Lagos); 2026-11-05 UTC |
| Rev9 | Historical record of the implemented R1 overlay (unchanged) | `docs/PHASE_8B_CONTRACT_REV9.md`, blob `91a0b88addbc3fdf88281ec569b0cec7f6c06742` |
| **Rev10** | **Operative ratification addendum** (this file) | `docs/PHASE_8B_CONTRACT_REV10.md` |

Relationship:

- **Rev8** remains the operative baseline text for all rules not amended by R1. The Rev8 file itself is unchanged. Its original header still reads “REVIEW DRAFT” / “proposed for review”; that wording is **historical metadata only**. The owner’s explicit confirmation of Rev8 as the operative baseline, together with this formal R1 ratification, **supersedes** that historical draft-status label for the operative Phase 8B acceptance baseline. Do not rewrite Rev8 to change its historical header.
- **R1** is approved **only** for the readiness-blocked fault class. It does not amend unhealthy-data or missing-candle timing.
- **Rev9** is preserved as the historical document that first recorded the implemented R1 overlay. Its historical contents are not silently rewritten.
- **Rev10** is the formal ratification that makes the operative relationship explicit. The technical body of the Phase 8B contract remains the Rev9 text (Rev8 + R1 overlay) as recorded at the blob identity above; this addendum supplies authority, provenance, R1 exact scope, and mandatory-acceptance restatements.

Do not rewrite the historical contents of Rev8 or Rev9. Do not treat this document as authorization to change the runtime recovery state machine.

## R1 amendment — exact scope (readiness-blocked only)

Recorded for ratification. All four changes apply exclusively to the readiness-blocked fault class:

1. One additional engine-blocked **RECOVERING** cycle at **r**.
2. First continuity diagnostic at **r+1**.
3. **STALLED** classification at **r+2**.
4. Readiness-blocked check window **r+1…E_rb**, where **E_rb = max(r+5, t+2)**.

Explicitly preserved from Rev8 (unchanged by R1):

- Unhealthy-data faults: first continuity diagnostic at **r**, STALLED at **r+1**, check window **r…E**.
- Missing-candle faults: first continuity diagnostic at **r**, STALLED at **r+1**, check window **r…E**.

R1 does **not** change the runtime recovery state machine. Reporting and acceptance windows for readiness-blocked scenarios are the only amendment.

Concrete readiness-class bookkeeping (from the implemented overlay recorded in Rev9):

- Injected fault interval remains **[s, r−1]** as for other classes.
- Cycle **r** is the additional engine-blocked RECOVERING cycle (lifecycle not called).
- Blocked cycles are exactly **[s, r]** (`blocked_cycles == list(range(s, r+1))`).
- No readiness-blocked cycle appears in the diagnostic-only list or the STALLED count.
- First continuity diagnostic **j = r+1**; STALLED classified at **r+2**.
- Check window **r+1…E_rb** with **E_rb = max(r+5, t+2)**.
- Per-cycle readiness sequence (READY → DEGRADED → … → RECOVERING → READY) is recorded and asserted.

Readiness-class window table and scenario bookkeeping remain as stated in Rev9 section 6 (R1 overlay) and A13.

## Provenance and review boundary

Exact source identities used for this ratification:

- Rev8: `docs/PHASE_8B_CONTRACT_REV8.md`, blob `640f4e6b4c26f6f425d910c5a696dc3e6d5cc2f0`
- Rev9: `docs/PHASE_8B_CONTRACT_REV9.md`, blob `91a0b88addbc3fdf88281ec569b0cec7f6c06742`
- Repository reference reviewed: `main` at `3e07dd3c84a7112ca08bd4741ce26ffc17a6e2be`
- Ratification date: **2026-11-06 (Africa/Lagos)**; corresponding UTC date **2026-11-05 UTC**
- Independent comparison: Rev8 vs Rev9 at the repository reference above (blob SHAs as listed). No claim is made that any unavailable historical draft (including any prior "Rev10") was reviewed as a source text for this ratification.
- Reviewer identity for this documentation reconciliation: independent documentation pass against the repository reference above; owner approval of R1 dated 2026-11-06 (Africa/Lagos).

**A15 evidence-status enum provenance (correction):** all four statuses — `not_run`, `stub_only`, `real_env_passed`, `real_env_failed` — are already present in **Rev8** (section 6 programmatic-generation condition and A15). They are **not** newly proposed in later revisions. Cite Rev8 as the source of the enum.

## Mandatory acceptance rules retained

The following rules remain mandatory and non-excludable under this operative document:

- Missing mandatory lifecycle parity fields produce **INCOMPLETE**, never PASS or exclusion.
- **pass → fail** is mandatory and non-excludable.
- Where a nine-row transition table appears, it is identified as the **proposed model** where appropriate, not misrepresented as a verified historical restatement.
- **A15 evidence status** is distinct from individual attempt verdicts.
- `real_env_passed` requires a passing attempt in a **verified real-TA-Lib environment**.
- All mandatory **A6** fixture cases and all **24** stall-matrix scenarios must pass in the **same** verified acceptance run.
- **Do not claim A15 has passed.** No durable A15 run manifest has yet been established at ratification.

## Incorporation by reference

Operative technical requirements for Phase 8B (fixtures, STALLED reporting, replay/parity requirements, load-time validation, acceptance criteria A1–A15, exclusions, and open/deferred items) are those recorded in:

- `docs/PHASE_8B_CONTRACT_REV9.md` (blob `91a0b88addbc3fdf88281ec569b0cec7f6c06742`), which is Rev8 plus the R1 readiness-blocked overlay,

as constrained by this addendum. In case of conflict on readiness-blocked timing, the R1 exact-scope section of this addendum controls. In case of conflict on unhealthy-data or missing-candle timing, Rev8 controls. Rev8 and Rev9 files remain intact historical records.

Pointers:

- Summary index: `docs/PHASE8B_REPLAY_PARITY.md` (updated to cite this operative document).
- Historical baseline: `docs/PHASE_8B_CONTRACT_REV8.md`.
- Historical R1 overlay record: `docs/PHASE_8B_CONTRACT_REV9.md`.
