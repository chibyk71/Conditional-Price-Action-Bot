# Phase 5A — Classical Swing-Based Reversal Patterns

## Scope

Four confirmed patterns derived exclusively from Phase 2A swing observations:

| Concept | Direction | Anchors |
|---------|-----------|---------|
| `double_top` | bearish | SH A → SL B → SH C |
| `double_bottom` | bullish | SL A → SH B → SL C |
| `head_and_shoulders` | bearish | SH A → SL B → SH C → SL D → SH E |
| `inverse_head_and_shoulders` | bullish | SL A → SH B → SL C → SH D → SL E |

Variant is always `swing_based_v1`.

## Geometry

- **Similarity**: `abs(P1-P2) / max(abs(P1), abs(P2), ε) ≤ similarity_tolerance` (default 0.002).
- **Double top/bottom neckline**: horizontal at the middle swing price.
- **H&S / IH&S neckline**: straight line through the two intermediate swings; evaluated at the breakout candle’s open epoch. Slope may be rising or falling; no trend filter.
- **Confirmation**: subsequent completed candle closes strictly beyond the neckline (wick-only does not confirm).
- **Structural rejection**: for tops/H&S, close above the higher peak / head before confirmation; for bottoms/IH&S, close below the lower trough / head.

## Causality

- Uses only confirmed swing observations (never before `confirmed_at`).
- Breakout eligible only on a subsequent completed candle after all anchors are available (`candle.epoch > max(anchor.confirmed_at)`). Equality with the latest anchor availability timestamp does not confirm.
- Window measured in candles from the final anchor’s candle; max `max_bars_to_breakout` (default 50).
- No retrospective confirmation; no gap bridging across a pattern’s evidence path.
- Continuous segments are processed independently: a gap does not suppress valid patterns wholly within an unbroken section; exact timeframe continuity is required within each segment.
- Deduplication by identical anchor swing observation IDs.

## Configuration

Accepted params (all enter `detector_config_id`):

- `similarity_tolerance` (float ≥ 0, default 0.002)
- `max_bars_to_breakout` (positive int, default 50)
- `left_strength`, `right_strength`, `tie_policy` (forwarded to `SwingPivotDetector`)

Invalid configuration raises `ValueError` (no silent coercion).

## Observation payload (selected)

- Pattern prices and event times for each anchor
- `neckline_at_breakout`, `neckline_type` (`horizontal` | `sloped`)
- `similarity_deviation`, effective config
- `breakout_candle_epoch`, `breakout_ohlc`
- `anchor_swing_ids` (also in `reference_ids`)

`event_time` / `confirmed_at` = breakout candle open epoch.
