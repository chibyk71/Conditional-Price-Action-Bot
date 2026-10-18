"""
Minimal detector capability registry for CPAB Phase 1+.

Registered capabilities have deterministic configuration identity.
Unsupported patterns / invalid params are rejected (no silent substitution).

Phase 2A: swing_pivot accepts left_strength, right_strength, tie_policy as
overridable configuration. TA-Lib pattern values remain fixed at registration.

Phase 2B: equal_levels accepts tolerance (required), left_strength, right_strength,
tie_policy. Tolerance has no silent default.

Phase 2C: market_structure accepts left_strength, right_strength, tie_policy
(forwarded to internal SwingPivotDetector).

Phase 2D: structure_break accepts left_strength, right_strength, tie_policy
(forwarded to internal SwingPivotDetector for breakable levels).

Phase 3A: support_resistance_zone accepts left_strength, right_strength, tie_policy
(forwarded to internal SwingPivotDetector for zone source swings).

Phase 3B: zone_interaction accepts left_strength, right_strength, tie_policy
(forwarded to internal SupportResistanceZoneDetector for source zones).

Phase 3C: zone_retest accepts left_strength, right_strength, tie_policy
(forwarded to internal SupportResistanceZoneDetector / ZoneInteractionDetector).

Phase 3D: zone_polarity_flip accepts left_strength, right_strength, tie_policy
(forwarded to internal ZoneInteractionDetector / SupportResistanceZoneDetector).
"""
