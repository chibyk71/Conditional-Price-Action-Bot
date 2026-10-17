# Phase 2D — Break of Structure (BOS) and Change of Character (CHoCH)

Causal, deterministic detection of structural breaks using confirmed swing highs and lows.

## Operational definition (V1)

This is an explicit operational definition for V1. It does **not** require HH/HL or LH/LL sequence prerequisites.

### Breakable levels

- Confirmed `swing_high` observations are candidates for **bullish** breaks.
- Confirmed `swing_low` observations are candidates for **bearish** breaks.
- Equal-high / equal-low observations are **not** breakable levels in this phase.
- A level is eligible only after its swing has been confirmed (`confirmed_at <= breakout candle epoch`).
- Never use a swing that was confirmed after the breakout candle.
- Source swing IDs and causal timestamps are preserved in the observation.
- Every eligible, unbroken level crossed by a candle close is evaluated (not only the most recent).
- Each level may generate at most one break event within a reconstructed structural history.

### Break confirmation

Completed candles only. Forming candles never trigger events.

| Direction | Condition |
|-----------|-----------|
| Bullish | `previous_close <= swing_high_price` **and** `current_close > swing_high_price` |
| Bearish | `previous_close >= swing_low_price` **and** `current_close < swing_low_price` |

- Equality with the level is **not** a break.
- Wick-only penetration is **not** a break.
- Remaining beyond a level does **not** generate a new break.
- A newly confirmed level does not trigger a break merely because price is already beyond it without a valid prior-side close.
- No inference through missing candles.

### Structural state and history completeness

Independent reconstruction per `(symbol, timeframe, effective detector configuration)`.

Reconstruction **never invents** structural events outside the supplied closed series. Whether the starting structural state is *known* is controlled by the keyword-only flag `history_complete` on `detect` / `detect_all_historical`:

| `history_complete` | Starting state | First in-window break |
|--------------------|----------------|------------------------|
| `True` | **neutral** (caller asserts complete structural coverage from known inception) | BOS under the locked table; `prior_structural_state="neutral"` |
| `False` (default) | **unknown** | Objective crossing still emits; `prior_structural_state="unknown"`; variant is in-window direction establishment — **not** proof that no earlier break existed outside the window |

Default is `False` so a truncated rolling window is not silently treated as structural inception.

Directional states after establishment: **bullish**, **bearish**.

| Previous state | Break direction | Event variant | Resulting state |
|----------------|-----------------|---------------|-----------------|
| neutral / unknown (first) | bullish | BOS | bullish |
| neutral / unknown (first) | bearish | BOS | bearish |
| bullish | bullish | BOS | bullish |
| bullish | bearish | CHoCH | bearish |
| bearish | bearish | BOS | bearish |
| bearish | bullish | CHoCH | bullish |

Payload includes `history_complete` so consumers can distinguish known-neutral BOS from truncated-window direction establishment.

**Caller impact:** engines that supply a full history from series inception (or otherwise know prior state is neutral) should pass `history_complete=True`. Rolling or truncated windows should leave the default `False`.

### Multiple and simultaneous breaks

**Same direction, multiple levels:** each crossed level produces its own observation. Order within a direction is stable: `(level.confirmed_at, level.event_time, swing.id)`. State transitions are applied in that order on the same candle.

**Bullish and bearish on the same candle:**

1. Emit bullish break events first.
2. Emit bearish break events second.
3. Apply state transitions in exactly that order.

Note: with standard geometry (swing high price > swing low price), a single close cannot simultaneously satisfy both `close > high` and `close < low`. The ordering rule remains defined for completeness.

### Gap handling

A candle is continuous with its immediate predecessor only when their opening epochs differ by exactly the expected timeframe duration (Phase 2A `timeframe_seconds`).

When continuity is broken:

- Suppress break evaluation on the discontinuous candle.
- Do not infer an intervening crossing.
- Retain the previously established structural state.
- Resume evaluation when continuity is restored.

Unknown timeframe labels reject continuity (same as swing windows).

## Observation contract

| Field | Value |
|-------|--------|
| `type` | `structure_break` |
| `variant` | `bos` \| `choch` |
| `direction` | `bullish` \| `bearish` |
| `event_time` / `candle_epoch` / `confirmed_at` | breakout candle open epoch |
| `developing` | `False` |
| `reference_ids` | `(broken_swing_id,)` |
| `detector_id` | `structure_break` |
| `detector_config_id` | deterministic from effective swing config |

Payload includes: `broken_level_price`, `break_direction`, `event_variant`, `source_swing_id`, `source_swing_event_time`, `source_swing_confirmed_at`, `previous_close`, `breakout_close`, `prior_structural_state`, `resulting_structural_state`, `history_complete`, and swing config fields.

Stable event IDs include the broken swing id **and** the `history_complete` mode (`hc1` / `hc0`) so:

- multiple breaks on the same candle remain distinct;
- complete vs truncated reconstructions of the same objective crossing do not share an ID (payloads differ on `prior_structural_state`; ObservationStore first-write-wins cannot silently drop one mode).

ObservationStore insertion is idempotent per mode.

## Configuration

Same swing parameters as Phase 2A / 2C:

| Parameter | Default | Constraint |
|-----------|---------|------------|
| `left_strength` | 1 | positive int |
| `right_strength` | 1 | positive int |
| `tie_policy` | `strict` | `strict` \| `first` \| `all` |

`min_history = left + 1 + right + 1`.

No wick-based break mode or equal-level break mode in V1.

## Historical vs incremental

- `detect_all_historical()` reconstructs pure structural state from closed history (no persistent store).
- `detect()` emits only events whose `confirmed_at` equals the latest closed bar's open epoch.
- Repeated runs produce identical observations and stable IDs for the same `history_complete` flag.
- Past event classifications are never revised using future candles.

## Scope exclusions

Invalidation lifecycle, order-block / supply-demand zones, SMC premium-discount, multi-timeframe confluence, and trade signals remain out of scope.
