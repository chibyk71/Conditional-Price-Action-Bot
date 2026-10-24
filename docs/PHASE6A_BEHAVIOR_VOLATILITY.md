# Phase 6A — Behavior / Volatility Detectors

## Concepts

| Concept | Detector id | Variant | Direction |
| --- | --- | --- | --- |
| `atr_expansion` | `atr_expansion` | `rolling_atr_v1` | None |
| `atr_contraction` | `atr_contraction` | `rolling_atr_v1` | None |
| `range_expansion` | `range_expansion` | `true_range_v1` | None |

## Locked contract

| Parameter | Value |
| --- | --- |
| ATR period | 14 (Wilder, `wilder_atr.py` unchanged) |
| ATR baseline | Mean of the preceding 20 rolling ATR values (exclude current) |
| Expansion threshold | 1.25× baseline, **crossing-only** |
| Contraction threshold | 0.75× baseline, **crossing-only** |
| Range expansion | Current true range ≥ 1.5× preceding ATR; emit every qualifying bar |
| Warm-up (ATR regimes) | 35 bars |
| Warm-up (range expansion) | 16 bars |

### Crossing-only semantics

An expansion (contraction) observation is emitted only on the completed candle
where the ratio first meets or exceeds (falls to or below) the threshold after
a bar that did not. Sustained regimes do not re-emit.

### Continuous segments

Segments are maximal runs of exact timeframe spacing. ATR windows and baselines
never bridge gaps. Invalid OHLC inside a window yields no ATR for that index.

### Zero / non-positive baseline

When the baseline mean is ≤ ε the ratio is undefined; no observation is emitted.

## Canonical payload

**atr_expansion / atr_contraction**

```text
atr, baseline, ratio, threshold, atr_period, baseline_window
```

**range_expansion**

```text
true_range, preceding_atr, ratio, threshold, atr_period
```

## Files

- `src/cpab/detectors/behavior_volatility.py` — three detectors
- `src/cpab/detectors/wilder_atr.py` — reused unchanged
- `tests/test_behavior_volatility.py` — focused regression suite
- Registry + `__init__` exports

## Out of scope

Lifecycle tracking, trading signals, volume logic, TA-Lib, and any modification
to prior-phase detectors.
