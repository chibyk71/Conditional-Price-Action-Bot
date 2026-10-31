# Phase 8B — Proposed amendment: A6 constructed fixture series (not yet approved)

**Status:** PROPOSED — does not amend revision 8 until explicitly approved.  
**Related:** `docs/PHASE_8B_CONTRACT_REV8.md` section 6 / A6.

## Problem

Revision 8 freezes F1/F3 outcomes to generator **seeds** 3, 1, 4, 7, 23 on symbol `stpRNG*` / 15m. The seed-series generator that produced those VERIFIED-STUB candle sequences is **not present** in the repository or attached artifacts. Real-TA-Lib verification therefore cannot re-run those exact series.

## What the implementation currently does

| Case | Contract | Implementation at branch head |
|------|----------|-------------------------------|
| F1 completion | seed 3 @ 98 | Real BreakerBlock+Impulse; **constructed** 1m series; transition **98** |
| F1 invalidation | seed 1 @ 129 | Contract setup shape; constructed series |
| F1 expiry | seed 4 @ 139 | Contract setup (expiry 80); constructed series |
| F2 completion/invalidation | 75/76, 92/93 | Real MarketStructureDetector; **exact** frozen indices |
| F3 completion/invalidation | seed 1 @ 98, seed 23 @ 79 | Real EngulfingDetector substitutes (not zone_polarity_flip series) |

## Proposed amendment (for reviewer approval only)

1. Accept **constructed real-detector series** that:
   - use the **exact contract setup definitions** (`fx_bearish_breaker_then_impulse`, `fx_bullish_higher_low_trigger`, `fx_bearish_flip_then_engulfing`);
   - assert the **exact frozen transition cycles** (98, 129, 139, 76, 93, 98, 79);
   - run under real TA-Lib / real detectors where applicable.
2. Keep F2 timelines as already locked (real detector confirmation lag).
3. Document that original seed bytes remain a gap if/when the generator is recovered.

Until this amendment is approved, A6 F1/F3 seed identity remains an open acceptance item under revision 8 as written.

## Explicit non-claims

- No claim that constructed series equal seed-3/1/4/7/23 OHLC bytes.
- No change to stall-matrix locked `s,r,t,E` coordinates.
- No merge authorization implied.
