"""
Phase 4F — Mitigation Block detection.

Tracks progressive wick-based interactions with confirmed Order Blocks and
Breaker Blocks. Emits immutable lifecycle observations for touch, partial_fill,
and full_fill. Does not mutate source observations or their bounds.
"""
