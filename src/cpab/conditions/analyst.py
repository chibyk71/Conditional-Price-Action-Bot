"""Phase 8B analyst-path load-time validation (L1–L5).

Extends the existing parser without replacing it. Engine/test path continues
to use parse_setup / parse_setups unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from cpab.conditions.catalogue import (
    ANALYST_ALLOWLIST,
    ANALYST_TIMEFRAME_LIST,
    ANALYST_TIMEFRAMES,
    CATALOGUE,
    DIRECTION_AMBIGUOUS_TYPES,
    is_analyst_allowed,
    is_catalogue_pair,
    is_catalogue_type,
    nearest_allowlist_entries,
)
from cpab.conditions.expression import (
    AndExpression,
    Expression,
    ObservationRequirement,
    OrExpression,
    ThenExpression,
)
from cpab.conditions.schema import SetupDef
from cpab.conditions.validation import SetupValidationError, parse_setup, parse_setups
from cpab.detectors.base import PriceActionDetector


@dataclass(frozen=True)
class TypeStatus:
    """Three independent dimensions for a required observation type."""

    type: str
    variant: Optional[str]
    catalogue_valid: bool
    analyst_allowed: bool
    wired: bool  # present in the supplied detector list's emission capability


@dataclass(frozen=True)
class AnalystValidationReport:
    setup: SetupDef
    type_statuses: tuple[TypeStatus, ...]
    warnings: tuple[str, ...]


def _iter_requirements(expr: Expression) -> Iterable[ObservationRequirement]:
    if isinstance(expr, ObservationRequirement):
        yield expr
    elif isinstance(expr, (AndExpression, OrExpression)):
        for child in expr.children:
            yield from _iter_requirements(child)
    elif isinstance(expr, ThenExpression):
        for step in expr.steps:
            yield from _iter_requirements(step.expression)


def validate_analyst_timeframe(timeframe: str, *, setup_id: Optional[str] = None) -> None:
    """Reject any timeframe not in the nine canonical labels."""
    if timeframe not in ANALYST_TIMEFRAMES:
        raise SetupValidationError(
            f"analyst path rejects timeframe {timeframe!r}; "
            f"accepted labels (case-sensitive): {list(ANALYST_TIMEFRAME_LIST)}",
            path="timeframe",
            setup_id=setup_id,
        )


def validate_analyst_requirement(
    req: ObservationRequirement,
    *,
    path: str,
    setup_id: Optional[str] = None,
) -> None:
    """Enforce catalogue membership and analyst allowlist on one leaf."""
    t = req.type
    v = req.variant

    if t in DIRECTION_AMBIGUOUS_TYPES:
        raise SetupValidationError(
            f"type {t!r} is direction-ambiguous and rejected on the analyst path; "
            f"use a direction-qualified type/variant from the allowlist "
            f"(examples: {nearest_allowlist_entries(t)})",
            path=path,
            setup_id=setup_id,
        )

    if not is_catalogue_type(t):
        raise SetupValidationError(
            f"unknown observation type {t!r} (not in catalogue); "
            f"nearest allowlist entries: {nearest_allowlist_entries(t)}",
            path=path,
            setup_id=setup_id,
        )

    variants = CATALOGUE[t]
    if variants:
        # Variant-bearing types require an explicit exact variant on analyst path.
        if v is None:
            raise SetupValidationError(
                f"type {t!r} requires an explicit variant on the analyst path; "
                f"valid variants: {sorted(variants)}",
                path=path,
                setup_id=setup_id,
            )
        if v not in variants:
            raise SetupValidationError(
                f"variant {v!r} is not valid for type {t!r}; "
                f"valid variants: {sorted(variants)}",
                path=path,
                setup_id=setup_id,
            )
    else:
        # Types with no variants must omit variant.
        if v is not None:
            raise SetupValidationError(
                f"type {t!r} does not accept a variant on the analyst path "
                f"(got variant={v!r})",
                path=path,
                setup_id=setup_id,
            )

    if not is_analyst_allowed(t, v):
        raise SetupValidationError(
            f"({t!r}, {v!r}) is not in the analyst allowlist (default deny); "
            f"nearest entries: {nearest_allowlist_entries(t)}",
            path=path,
            setup_id=setup_id,
        )


def parse_setup_analyst(raw: Any, *, index: Optional[int] = None) -> SetupDef:
    """Parse via the existing parser, then apply analyst-path restrictions."""
    # Reject whitespace / non-exact labels on the raw timeframe before the
    # engine parser strips strings (L1: no whitespace).
    if isinstance(raw, dict) and "timeframe" in raw:
        tf_raw = raw["timeframe"]
        early_id = None
        if isinstance(raw.get("setup_id"), str) and raw["setup_id"].strip():
            early_id = raw["setup_id"].strip()
        if not isinstance(tf_raw, str) or tf_raw != tf_raw.strip() or " " in tf_raw:
            raise SetupValidationError(
                f"analyst path rejects timeframe {tf_raw!r}; "
                f"accepted labels (case-sensitive, no whitespace): "
                f"{list(ANALYST_TIMEFRAME_LIST)}",
                path="timeframe",
                setup_id=early_id,
            )
        validate_analyst_timeframe(tf_raw, setup_id=early_id)
    setup = parse_setup(raw, index=index)
    validate_analyst_timeframe(setup.timeframe, setup_id=setup.setup_id)
    for req in _iter_requirements(setup.expression):
        validate_analyst_requirement(
            req, path="expression", setup_id=setup.setup_id
        )
    if setup.invalidation is not None:
        for req in _iter_requirements(setup.invalidation):
            validate_analyst_requirement(
                req, path="invalidation", setup_id=setup.setup_id
            )
    return setup


def parse_setups_analyst(raw_list: Any) -> list[SetupDef]:
    """Parse a list of setups under analyst-path rules."""
    if raw_list is None:
        return []
    if not isinstance(raw_list, list):
        raise SetupValidationError(
            f"'setups' must be a list, got {type(raw_list).__name__}",
            path="setups",
        )
    setups: list[SetupDef] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(raw_list):
        setup = parse_setup_analyst(raw, index=i)
        if setup.setup_id in seen:
            raise SetupValidationError(
                f"duplicate setup_id {setup.setup_id!r} "
                f"(first at index {seen[setup.setup_id]})",
                path=f"setups[{i}].setup_id",
                setup_id=setup.setup_id,
            )
        seen[setup.setup_id] = i
        setups.append(setup)
    return setups


def _detector_emittable_types(detectors: Sequence[PriceActionDetector]) -> set[str]:
    """Best-effort set of types the supplied detectors can emit.

    Uses detector class name heuristics and optional ``emits`` / ``name``
    attributes. This is a wiring report aid, not a capability registry.
    """
    emitted: set[str] = set()
    for det in detectors:
        name = getattr(det, "name", type(det).__name__).lower()
        # Common mappings from detector names to emitted types.
        mapping = {
            "engulfing": {"bullish_engulfing", "bearish_engulfing"},
            "impulse": {"bullish_impulse", "bearish_impulse"},
            "market_structure": {"market_structure"},
            "structure_break": {"structure_break"},
            "breaker": {"breaker_block"},
            "order_block": {"order_block"},
            "zone_polarity": {"zone_polarity_flip"},
            "swing": {"swing_high", "swing_low"},
            "support_resistance": {"support_resistance_zone"},
            "equal": {
                "equal_high_level",
                "equal_low_level",
                "equal_high_level_touch",
                "equal_low_level_touch",
            },
            "fair_value_gap": {"fair_value_gap"},
            "inverse_fair": {"inverse_fair_value_gap"},
            "atr_expansion": {"atr_expansion"},
            "atr_contraction": {"atr_contraction"},
            "range_expansion": {"range_expansion"},
            "classical": {
                "double_top",
                "double_bottom",
                "head_and_shoulders",
                "inverse_head_and_shoulders",
            },
        }
        for key, types in mapping.items():
            if key in name:
                emitted |= types
        # Explicit attribute if present.
        extra = getattr(det, "emitted_types", None)
        if extra:
            emitted |= set(extra)
    return emitted


def report_setup_types(
    setup: SetupDef,
    detectors: Sequence[PriceActionDetector] = (),
) -> AnalystValidationReport:
    """Report catalogue / analyst / wired dimensions without rejecting."""
    wired_types = _detector_emittable_types(detectors)
    statuses: list[TypeStatus] = []
    warnings: list[str] = []
    reqs = list(_iter_requirements(setup.expression))
    if setup.invalidation is not None:
        reqs.extend(_iter_requirements(setup.invalidation))
    for req in reqs:
        cat_ok = is_catalogue_pair(req.type, req.variant)
        an_ok = is_analyst_allowed(req.type, req.variant)
        wired = req.type in wired_types
        statuses.append(
            TypeStatus(
                type=req.type,
                variant=req.variant,
                catalogue_valid=cat_ok,
                analyst_allowed=an_ok,
                wired=wired,
            )
        )
        if cat_ok and an_ok and not wired:
            warnings.append(
                f"type {req.type!r} variant={req.variant!r} is catalogue-valid and "
                f"analyst-allowed but not wired in the supplied detector list"
            )
    return AnalystValidationReport(
        setup=setup,
        type_statuses=tuple(statuses),
        warnings=tuple(warnings),
    )


# Re-export engine-path parsers for callers that need both.
__all__ = [
    "TypeStatus",
    "AnalystValidationReport",
    "validate_analyst_timeframe",
    "validate_analyst_requirement",
    "parse_setup_analyst",
    "parse_setups_analyst",
    "report_setup_types",
    "parse_setup",
    "parse_setups",
]
