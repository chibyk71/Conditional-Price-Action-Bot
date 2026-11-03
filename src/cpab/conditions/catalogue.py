"""Authoritative observation catalogue and analyst allowlist (Phase 8B L2/L3).

Code-defined data is the single source of truth. Documentation must be derived
from or checked against this module in both directions.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Full catalogue: emitted type -> frozenset of variants (empty = no variant)
# Matches construction sites documented in docs/PHASE8A_SETUP_CATALOGUE.md.
# ---------------------------------------------------------------------------

CATALOGUE: dict[str, frozenset[str]] = {
    # Candle patterns
    "bullish_engulfing": frozenset(),
    "bearish_engulfing": frozenset(),
    # Swings and structure
    "swing_high": frozenset({"pivot_lr"}),
    "swing_low": frozenset({"pivot_lr"}),
    "equal_high_level": frozenset({"equal_abs"}),
    "equal_low_level": frozenset({"equal_abs"}),
    "equal_high_level_touch": frozenset({"equal_abs"}),
    "equal_low_level_touch": frozenset({"equal_abs"}),
    "market_structure": frozenset({"hh", "lh", "hl", "ll"}),
    "structure_break": frozenset({"bos", "choch"}),
    # Zones
    "support_resistance_zone": frozenset({"support", "resistance"}),
    "zone_interaction": frozenset({"touch", "rejection", "break"}),
    "zone_retest": frozenset({"retest"}),
    "zone_polarity_flip": frozenset({"support_to_resistance", "resistance_to_support"}),
    # FVG and blocks
    "fair_value_gap": frozenset({"bullish", "bearish"}),
    "fair_value_gap_lifecycle": frozenset(
        {"first_touch", "partial_mitigation", "full_mitigation", "invalidation"}
    ),
    "inverse_fair_value_gap": frozenset({"bullish_to_bearish", "bearish_to_bullish"}),
    "order_block": frozenset({"bullish", "bearish"}),
    "breaker_block": frozenset({"bullish_to_bearish", "bearish_to_bullish"}),
    "mitigation_block": frozenset({"touch", "partial_fill", "full_fill"}),
    # Classical patterns
    "double_top": frozenset({"swing_based_v1"}),
    "double_bottom": frozenset({"swing_based_v1"}),
    "head_and_shoulders": frozenset({"swing_based_v1"}),
    "inverse_head_and_shoulders": frozenset({"swing_based_v1"}),
    "rectangle": frozenset({"swing_based_v1"}),
    "ascending_triangle": frozenset({"swing_based_v1"}),
    "descending_triangle": frozenset({"swing_based_v1"}),
    "symmetrical_triangle": frozenset({"swing_based_v1"}),
    "bullish_flag": frozenset({"swing_based_v1"}),
    "bearish_flag": frozenset({"swing_based_v1"}),
    "bullish_pennant": frozenset({"swing_based_v1"}),
    "bearish_pennant": frozenset({"swing_based_v1"}),
    # Behavior / volatility / impulse / pullback / breakout
    "atr_expansion": frozenset({"rolling_atr_v1"}),
    "atr_contraction": frozenset({"rolling_atr_v1"}),
    "range_expansion": frozenset({"true_range_v1"}),
    "bullish_impulse": frozenset({"atr_normalized_candle_v1"}),
    "bearish_impulse": frozenset({"atr_normalized_candle_v1"}),
    "pullback": frozenset({"structure_based", "impulse_based"}),
    "breakout_behavior": frozenset(
        {"breakout_confirmed", "breakout_failed", "breakout_follow_through"}
    ),
}

# Direction-ambiguous types rejected on the analyst path (L3).
DIRECTION_AMBIGUOUS_TYPES: frozenset[str] = frozenset(
    {
        "structure_break",
        "zone_interaction",
        "zone_retest",
        "fair_value_gap_lifecycle",
        "mitigation_block",
        "breakout_behavior",
    }
)

# Additional types excluded from analyst path (default deny beyond allowlist).
# Listed for diagnostics; allowlist is authoritative.

# ---------------------------------------------------------------------------
# Analyst allowlist — exact 33 (type, variant|None) pairs from contract L3.
# ---------------------------------------------------------------------------

ANALYST_ALLOWLIST: frozenset[tuple[str, Optional[str]]] = frozenset(
    {
        ("bullish_engulfing", None),
        ("bearish_engulfing", None),
        ("bullish_impulse", "atr_normalized_candle_v1"),
        ("bearish_impulse", "atr_normalized_candle_v1"),
        ("market_structure", "hh"),
        ("market_structure", "hl"),
        ("market_structure", "lh"),
        ("market_structure", "ll"),
        ("order_block", "bullish"),
        ("order_block", "bearish"),
        ("breaker_block", "bullish_to_bearish"),
        ("breaker_block", "bearish_to_bullish"),
        ("fair_value_gap", "bullish"),
        ("fair_value_gap", "bearish"),
        ("inverse_fair_value_gap", "bullish_to_bearish"),
        ("inverse_fair_value_gap", "bearish_to_bullish"),
        ("zone_polarity_flip", "support_to_resistance"),
        ("zone_polarity_flip", "resistance_to_support"),
        ("swing_high", "pivot_lr"),
        ("swing_low", "pivot_lr"),
        ("support_resistance_zone", "support"),
        ("support_resistance_zone", "resistance"),
        ("equal_high_level", "equal_abs"),
        ("equal_low_level", "equal_abs"),
        ("equal_high_level_touch", "equal_abs"),
        ("equal_low_level_touch", "equal_abs"),
        ("atr_expansion", "rolling_atr_v1"),
        ("atr_contraction", "rolling_atr_v1"),
        ("range_expansion", "true_range_v1"),
        ("double_top", "swing_based_v1"),
        ("double_bottom", "swing_based_v1"),
        ("head_and_shoulders", "swing_based_v1"),
        ("inverse_head_and_shoulders", "swing_based_v1"),
    }
)

assert len(ANALYST_ALLOWLIST) == 33, f"allowlist size {len(ANALYST_ALLOWLIST)} != 33"

# Nine analyst timeframe labels (case-sensitive lowercase, no whitespace).
ANALYST_TIMEFRAMES: frozenset[str] = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}
)

ANALYST_TIMEFRAME_LIST: tuple[str, ...] = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "1d",
)


def is_catalogue_type(type_name: str) -> bool:
    return type_name in CATALOGUE


def catalogue_variants(type_name: str) -> frozenset[str]:
    return CATALOGUE.get(type_name, frozenset())


def is_catalogue_pair(type_name: str, variant: Optional[str]) -> bool:
    if type_name not in CATALOGUE:
        return False
    variants = CATALOGUE[type_name]
    if not variants:
        return variant is None
    return variant is not None and variant in variants


def is_analyst_allowed(type_name: str, variant: Optional[str]) -> bool:
    return (type_name, variant) in ANALYST_ALLOWLIST


def nearest_allowlist_entries(type_name: str, *, limit: int = 5) -> list[str]:
    """Human-readable nearest allowlist entries for diagnostics."""
    matches = [
        f"{t}" if v is None else f"{t}/{v}"
        for t, v in sorted(ANALYST_ALLOWLIST, key=lambda x: (x[0], x[1] or ""))
        if t == type_name or type_name in t or t in type_name
    ]
    if not matches:
        matches = [
            f"{t}" if v is None else f"{t}/{v}"
            for t, v in sorted(ANALYST_ALLOWLIST, key=lambda x: (x[0], x[1] or ""))[:limit]
        ]
    return matches[:limit]
