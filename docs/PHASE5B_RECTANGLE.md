# Phase 5B — Classical Swing-Based Rectangle Pattern

## Scope

Confirmed rectangle (range) patterns derived exclusively from Phase 2A swing observations.

| Concept | Direction | Anchors |
|---------|-----------|---------|
| `rectangle` | bullish | ≥2 SH + ≥2 SL, alternating chronological sequence; breakout close above resistance |
| `rectangle` | bearish | same structure; breakout close below support |

Variant is always `swing_based_v1`.

## Geometry

- **Minimum structure**: at least two distinct swing highs and two distinct swing lows forming an alternating sequence in chronological order (H→L→H→L… or L→H→L→H…). Longer alternating sequences are valid when they satisfy all other requirements.
- **Resistance**: median of the selected swing-high prices.
- **Support**: median of the selected swing-low prices.
- Even-sized sets use the arithmetic mean of the two central sorted values.
- **Boundary qualification**: every selected high must satisfy
  `abs(price − resistance) / max(abs(price), abs(resistance), ε) ≤ boundary_tolerance`
  (and likewise for lows vs support). Tolerance is inclusive.
- **Geometry**: resistance must be strictly greater than support.
- **Confirmation**: subsequent completed candle closes strictly beyond the boundary
  (close > resistance for bullish; close < support for bearish). Wick-only does not confirm.
  Close exactly on the boundary does not confirm.

## Causality

- Uses only confirmed swing observations (never before `confirmed_at`).
- Breakout eligible only on a subsequent completed candle after all anchors are available
  (`candle.epoch > max(anchor.confirmed_at)`). Equality with the latest anchor availability
  timestamp does not confirm.
- Window measured in candles from the final anchor’s candle; offsets `1 .. max_bars_to_breakout`
  inclusive. Candidates whose window has already expired when the last anchor becomes available
  are discarded; no confirmation after the deadline.
- No retrospective confirmation; no gap bridging across a pattern’s evidence path.
- Continuous segments are processed independently: a gap does not suppress valid patterns
  wholly within an unbroken section; exact timeframe continuity is required within each segment.
- Deduplication by identical sets of anchor swing observation IDs (order-insensitive for identity;
  chronological order preserved in the payload). Overlapping candidates with distinct anchors
  may coexist.
- At most one confirmation per anchor combination (first qualifying breakout direction).
- **Candidate formation**: all alternating chronological *subsequences* with ≥2 highs and
  ≥2 lows. Intermediate swings may be skipped. History within a continuous segment is never
  truncated by a fixed swing-count cap; there is no maximum anchor-count restriction.
  Safe suffix-count pruning drops only branches that cannot reach the minimum touch counts.
  Candidates are streamed into breakout confirmation (not retained as a full in-memory list).
  Worst-case combinatorial cost is exponential in the number of alternating swings in a
  continuous segment; that cost is inherent to exhaustive subsequence enumeration and is
  not mitigated by silently narrowing valid coverage. Practical measurements: perfect
  alternation of ~20 swings already yields tens of thousands of candidates; ~22 swings
  approach 10⁵ candidates and multi-hundred-millisecond enumeration on typical hardware.

## Configuration

Accepted params (all enter `detector_config_id`):

- `boundary_tolerance` (float ≥ 0, default 0.002)
- `max_bars_to_breakout` (positive int, default 50)
- `left_strength`, `right_strength`, `tie_policy` (forwarded to `SwingPivotDetector`)

Invalid configuration raises `ValueError` (no silent coercion).

## Observation payload (selected)

- `support`, `resistance`
- `boundary_tolerance`, `max_bars_to_breakout`
- `anchor_swing_ids` (chronological; also in `reference_ids`)
- `anchor_swing_types`, `anchor_swing_prices`
- `high_swing_ids`, `low_swing_ids`
- `breakout_direction`, `breakout_close`, `breakout_candle_epoch`, `breakout_ohlc`
- Swing-detector configuration (`left_strength`, `right_strength`, `tie_policy`)

`event_time` / `confirmed_at` = breakout candle open epoch.
`type` = `"rectangle"`, `variant` = `"swing_based_v1"`.
