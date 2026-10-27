"""Strict validation for Phase 7A setup configurations and expressions."""

from __future__ import annotations

from typing import Any, Optional

from cpab.conditions.expression import (
    AndExpression,
    Expression,
    ObservationRequirement,
    OrExpression,
    ThenExpression,
    ThenStep,
)
from cpab.conditions.schema import (
    SCHEMA_VERSION_V1,
    V1_ALLOWED_FIELDS,
    SetupDef,
)


class SetupValidationError(ValueError):
    """Raised when a setup or expression fails strict validation."""

    def __init__(self, message: str, *, path: str = "", setup_id: Optional[str] = None):
        self.path = path
        self.setup_id = setup_id
        prefix_parts: list[str] = []
        if setup_id is not None:
            prefix_parts.append(f"setup_id={setup_id!r}")
        if path:
            prefix_parts.append(f"path={path}")
        prefix = f"[{', '.join(prefix_parts)}] " if prefix_parts else ""
        super().__init__(f"{prefix}{message}")


def _require_dict(raw: Any, path: str, setup_id: Optional[str] = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SetupValidationError(
            f"expected mapping, got {type(raw).__name__}",
            path=path,
            setup_id=setup_id,
        )
    return raw


def _require_non_empty_str(value: Any, path: str, setup_id: Optional[str] = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SetupValidationError(
            f"expected non-empty string, got {value!r}",
            path=path,
            setup_id=setup_id,
        )
    return value.strip()


def _validate_reserved_block(
    raw: Any,
    field_name: str,
    path: str,
    setup_id: Optional[str],
) -> Optional[dict[str, Any]]:
    """Reserved invalidation/expiry: null/absent ok; if present must be a mapping.

    Behavior is not implemented in 7A. Reject non-mappings and require that
    any present block is a plain dict (future schema_version may appear inside).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SetupValidationError(
            f"reserved field {field_name!r} must be a mapping or null, got {type(raw).__name__}",
            path=path,
            setup_id=setup_id,
        )
    return dict(raw)


def parse_expression(
    raw: Any,
    *,
    path: str = "expression",
    setup_id: Optional[str] = None,
    allow_same_candle_on_leaf: bool = False,
) -> Expression:
    """Parse and strictly validate an expression tree."""
    data = _require_dict(raw, path, setup_id)
    if "op" in data:
        op = data["op"]
        if not isinstance(op, str):
            raise SetupValidationError(
                f"op must be a string, got {type(op).__name__}",
                path=f"{path}.op",
                setup_id=setup_id,
            )
        op_norm = op.strip().lower()
        if op_norm == "and":
            return _parse_and(data, path=path, setup_id=setup_id)
        if op_norm == "or":
            return _parse_or(data, path=path, setup_id=setup_id)
        if op_norm == "then":
            return _parse_then(data, path=path, setup_id=setup_id)
        raise SetupValidationError(
            f"unknown operator {op!r}; expected and, or, or then",
            path=f"{path}.op",
            setup_id=setup_id,
        )
    if "type" not in data:
        raise SetupValidationError(
            "expression must have 'op' (and|or|then) or leaf 'type'",
            path=path,
            setup_id=setup_id,
        )
    return _parse_observation_requirement(
        data, path=path, setup_id=setup_id, allow_same_candle_key=allow_same_candle_on_leaf
    )


def _parse_observation_requirement(
    data: dict[str, Any],
    *,
    path: str,
    setup_id: Optional[str],
    allow_same_candle_key: bool = False,
) -> ObservationRequirement:
    allowed = {"type", "variant", "min_count", "allow_developing"}
    if allow_same_candle_key:
        allowed = allowed | {"allow_same_candle"}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise SetupValidationError(
            f"unknown field(s) on observation requirement: {sorted(unknown)}",
            path=path,
            setup_id=setup_id,
        )
    obs_type = _require_non_empty_str(data.get("type"), f"{path}.type", setup_id)
    variant = data.get("variant")
    if variant is not None:
        if not isinstance(variant, str) or not variant.strip():
            raise SetupValidationError(
                f"variant must be a non-empty string when present, got {variant!r}",
                path=f"{path}.variant",
                setup_id=setup_id,
            )
        variant = variant.strip()
    min_count_raw = data.get("min_count", 1)
    if isinstance(min_count_raw, bool) or not isinstance(min_count_raw, int):
        raise SetupValidationError(
            f"min_count must be a positive integer, got {min_count_raw!r}",
            path=f"{path}.min_count",
            setup_id=setup_id,
        )
    if min_count_raw < 1:
        raise SetupValidationError(
            f"min_count must be >= 1, got {min_count_raw}",
            path=f"{path}.min_count",
            setup_id=setup_id,
        )
    allow_dev = data.get("allow_developing", False)
    if not isinstance(allow_dev, bool):
        raise SetupValidationError(
            f"allow_developing must be a boolean, got {allow_dev!r}",
            path=f"{path}.allow_developing",
            setup_id=setup_id,
        )
    return ObservationRequirement(
        type=obs_type,
        variant=variant,
        min_count=min_count_raw,
        allow_developing=allow_dev,
    )


def _parse_and(
    data: dict[str, Any], *, path: str, setup_id: Optional[str]
) -> AndExpression:
    unknown = set(data.keys()) - {"op", "children"}
    if unknown:
        raise SetupValidationError(
            f"unknown field(s) on and expression: {sorted(unknown)}",
            path=path,
            setup_id=setup_id,
        )
    children_raw = data.get("children")
    if not isinstance(children_raw, list) or len(children_raw) == 0:
        raise SetupValidationError(
            "and expression requires a non-empty 'children' list",
            path=f"{path}.children",
            setup_id=setup_id,
        )
    children = tuple(
        parse_expression(c, path=f"{path}.children[{i}]", setup_id=setup_id)
        for i, c in enumerate(children_raw)
    )
    return AndExpression(children=children)


def _parse_or(
    data: dict[str, Any], *, path: str, setup_id: Optional[str]
) -> OrExpression:
    unknown = set(data.keys()) - {"op", "children"}
    if unknown:
        raise SetupValidationError(
            f"unknown field(s) on or expression: {sorted(unknown)}",
            path=path,
            setup_id=setup_id,
        )
    children_raw = data.get("children")
    if not isinstance(children_raw, list) or len(children_raw) == 0:
        raise SetupValidationError(
            "or expression requires a non-empty 'children' list",
            path=f"{path}.children",
            setup_id=setup_id,
        )
    children = tuple(
        parse_expression(c, path=f"{path}.children[{i}]", setup_id=setup_id)
        for i, c in enumerate(children_raw)
    )
    return OrExpression(children=children)


def _parse_then(
    data: dict[str, Any], *, path: str, setup_id: Optional[str]
) -> ThenExpression:
    unknown = set(data.keys()) - {"op", "steps"}
    if unknown:
        raise SetupValidationError(
            f"unknown field(s) on then expression: {sorted(unknown)}",
            path=path,
            setup_id=setup_id,
        )
    steps_raw = data.get("steps")
    if not isinstance(steps_raw, list) or len(steps_raw) == 0:
        raise SetupValidationError(
            "then expression requires a non-empty 'steps' list",
            path=f"{path}.steps",
            setup_id=setup_id,
        )
    steps: list[ThenStep] = []
    for i, step_raw in enumerate(steps_raw):
        step_path = f"{path}.steps[{i}]"
        step_data = _require_dict(step_raw, step_path, setup_id)
        allow_same = step_data.get("allow_same_candle", False)
        if not isinstance(allow_same, bool):
            raise SetupValidationError(
                f"allow_same_candle must be a boolean, got {allow_same!r}",
                path=f"{step_path}.allow_same_candle",
                setup_id=setup_id,
            )
        # First step has no predecessor; same-candle is meaningless there.
        if i == 0 and allow_same is True:
            raise SetupValidationError(
                "allow_same_candle cannot be true on the first THEN step "
                "(no preceding step to share a candle with)",
                path=f"{step_path}.allow_same_candle",
                setup_id=setup_id,
            )
        expr_data = {k: v for k, v in step_data.items() if k != "allow_same_candle"}
        expr = parse_expression(
            expr_data,
            path=step_path,
            setup_id=setup_id,
            allow_same_candle_on_leaf=False,
        )
        steps.append(ThenStep(expression=expr, allow_same_candle=allow_same))
    return ThenExpression(steps=tuple(steps))


def parse_setup(raw: Any, *, index: Optional[int] = None) -> SetupDef:
    """Parse and strictly validate a single version-1 setup definition."""
    path_prefix = f"setups[{index}]" if index is not None else "setup"
    data = _require_dict(raw, path_prefix)
    early_id: Optional[str] = None
    if isinstance(data.get("setup_id"), str) and data["setup_id"].strip():
        early_id = data["setup_id"].strip()
    unknown = set(data.keys()) - V1_ALLOWED_FIELDS
    if unknown:
        raise SetupValidationError(
            f"unknown field(s) not permitted by schema version 1: {sorted(unknown)}",
            path=path_prefix,
            setup_id=early_id,
        )
    if "schema_version" not in data:
        raise SetupValidationError(
            "missing required field 'schema_version'",
            path=f"{path_prefix}.schema_version",
            setup_id=early_id,
        )
    schema_version = data["schema_version"]
    # Reject bool/float/str before == 1 (True == 1 and 1.0 == 1 in Python).
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise SetupValidationError(
            f"schema_version must be an integer, got {schema_version!r} "
            f"({type(schema_version).__name__})",
            path=f"{path_prefix}.schema_version",
            setup_id=early_id,
        )
    if schema_version != SCHEMA_VERSION_V1:
        raise SetupValidationError(
            f"unsupported schema_version {schema_version!r}; only {SCHEMA_VERSION_V1} is supported",
            path=f"{path_prefix}.schema_version",
            setup_id=early_id,
        )
    for req in ("setup_id", "symbol", "timeframe", "enabled", "expression"):
        if req not in data:
            raise SetupValidationError(
                f"missing required field {req!r}",
                path=f"{path_prefix}.{req}",
                setup_id=early_id,
            )
    setup_id = _require_non_empty_str(data["setup_id"], f"{path_prefix}.setup_id", early_id)
    symbol = _require_non_empty_str(data["symbol"], f"{path_prefix}.symbol", setup_id)
    timeframe = _require_non_empty_str(
        data["timeframe"], f"{path_prefix}.timeframe", setup_id
    )
    enabled = data["enabled"]
    if not isinstance(enabled, bool):
        raise SetupValidationError(
            f"enabled must be a boolean, got {enabled!r}",
            path=f"{path_prefix}.enabled",
            setup_id=setup_id,
        )
    name = data.get("name")
    if name is not None:
        if not isinstance(name, str):
            raise SetupValidationError(
                f"name must be a string when present, got {type(name).__name__}",
                path=f"{path_prefix}.name",
                setup_id=setup_id,
            )
    description = data.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise SetupValidationError(
                f"description must be a string when present, got {type(description).__name__}",
                path=f"{path_prefix}.description",
                setup_id=setup_id,
            )
    invalidation = _validate_reserved_block(
        data.get("invalidation"), "invalidation", f"{path_prefix}.invalidation", setup_id
    )
    expiry = _validate_reserved_block(
        data.get("expiry"), "expiry", f"{path_prefix}.expiry", setup_id
    )
    expression = parse_expression(
        data["expression"], path=f"{path_prefix}.expression", setup_id=setup_id
    )
    return SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id=setup_id,
        symbol=symbol,
        timeframe=timeframe,
        enabled=enabled,
        expression=expression,
        name=name,
        description=description,
        invalidation=invalidation,
        expiry=expiry,
    )


def parse_setups(raw_list: Any) -> list[SetupDef]:
    """Parse a list of setups; enforce unique setup_id; validate disabled setups."""
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
        setup = parse_setup(raw, index=i)
        if setup.setup_id in seen:
            raise SetupValidationError(
                f"duplicate setup_id {setup.setup_id!r} (first at index {seen[setup.setup_id]})",
                path=f"setups[{i}].setup_id",
                setup_id=setup.setup_id,
            )
        seen[setup.setup_id] = i
        setups.append(setup)
    return setups
