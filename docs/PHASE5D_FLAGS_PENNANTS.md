# Phase 5D — Flags and Pennants

## Concepts

- `bullish_flag`, `bearish_flag`, `bullish_pennant`, `bearish_pennant`
- Variant: `swing_based_v1`
- Detector id: `flags_pennants`

## Impulse pole (locked)

| Parameter | Default |
| --- | --- |
| min_pole_bars | 2 |
| max_pole_bars | 10 |
| pole_atr_lookback | 14 |
| pole_strength_mult | 1.5 |
| max_pole_retracement | 0.50 |

- Duration inclusive of first and final candle.
- Displacement = `abs(close[end] − close[start])`; zero rejected.
- Strength: `displacement >= pole_strength_mult * ATR`.
- ATR: pure Wilder, exactly 14 completed candles immediately preceding the pole (plus preceding close for first TR). Pole candles never contribute. Continuous segment only.

## Consolidation

| Parameter | Default |
| --- | --- |
| min_consolidation_bars | 3 |
| max_consolidation_bars | 20 |
| boundary_tolerance | 0.002 |
| parallel_slope_tolerance | 0.002 |
| max_bars_to_breakout | 50 |

- Begins immediately after the pole’s final candle.
- ≥2 distinct swing highs and ≥2 distinct swing lows (Phase 2A).
- Boundaries from first/last selected anchors (Phase 5C convention).

## Flag vs pennant

- **Flag**: approximately parallel boundaries (relative slope formula with ε=1e-12). Bullish after bullish pole (down/flat); bearish after bearish pole (up/flat).
- **Pennant**: upper descending, lower ascending, finite intersection strictly after final anchor (Phase 5C triangle geometry).

## Breakout / lifecycle

- Bullish: strict close above upper boundary.
- Bearish: strict close below lower boundary.
- Eligible only when `candle.epoch > max(anchor.confirmed_at)`.
- Expiry: offsets 1 … max_bars_to_breakout inclusive from final consolidation anchor.
- Opposite-boundary close invalidates; no later confirmation.
- Continuous segments independent; no gap bridging.

## Files

- `src/cpab/detectors/wilder_atr.py` — ATR helper
- `src/cpab/detectors/flags_pennants.py` — detector
- `tests/test_flags_pennants.py` — focused tests
- Registry + `__init__` updates
