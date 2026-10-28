"""Phase 7A: setup schema, expression model, validation, config identity."""

from __future__ import annotations

import copy

import pytest
import yaml

from cpab.conditions.expression import (
    AndExpression,
    ObservationRequirement,
    OrExpression,
    ThenExpression,
    ThenStep,
)
from cpab.conditions.identity import setup_config_identity
from cpab.conditions.schema import SCHEMA_VERSION_V1, SetupDef
from cpab.conditions.validation import (
    SetupValidationError,
    parse_expression,
    parse_setup,
    parse_setups,
)
from cpab.config import conditions_from_config, load_config, setups_from_config
from cpab.models import ConditionDef


def _leaf(t="bullish_engulfing", **kwargs):
    d = {"type": t}
    d.update(kwargs)
    return d


def test_valid_v1_minimal_setup():
    raw = {
        "schema_version": 1,
        "setup_id": "s1",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": _leaf(),
    }
    setup = parse_setup(raw)
    assert isinstance(setup, SetupDef)
    assert setup.schema_version == SCHEMA_VERSION_V1
    assert setup.setup_id == "s1"
    assert setup.enabled is True
    assert isinstance(setup.expression, ObservationRequirement)
    assert setup.expression.type == "bullish_engulfing"
    assert setup.expression.min_count == 1
    assert setup.expression.allow_developing is False
    assert setup.name is None
    assert setup.invalidation is None


def test_valid_optional_and_reserved_fields():
    raw = {
        "schema_version": 1,
        "setup_id": "s2",
        "symbol": "R_100",
        "timeframe": "5m",
        "enabled": False,
        "name": "Example",
        "description": "desc only",
        "invalidation": {"type": "structure_break"},
        "expiry": {"max_age_candles": 50},
        "expression": {
            "op": "and",
            "children": [
                _leaf("swing_high"),
                _leaf("zone_retest", min_count=1, allow_developing=False),
            ],
        },
    }
    setup = parse_setup(raw)
    assert setup.enabled is False
    assert setup.name == "Example"
    assert setup.description == "desc only"
    from cpab.conditions.expression import ObservationRequirement
    from cpab.conditions.schema import ExpiryConfig
    assert isinstance(setup.invalidation, ObservationRequirement)
    assert setup.invalidation.type == "structure_break"
    assert setup.expiry == ExpiryConfig(max_age_candles=50)
    assert isinstance(setup.expression, AndExpression)
    assert len(setup.expression.children) == 2


def test_valid_or_and_then():
    or_expr = parse_expression(
        {
            "op": "or",
            "children": [_leaf("bullish_engulfing"), _leaf("bullish_flag")],
        }
    )
    assert isinstance(or_expr, OrExpression)
    assert len(or_expr.children) == 2

    then_expr = parse_expression(
        {
            "op": "then",
            "steps": [
                _leaf("swing_high"),
                {"type": "zone_retest", "allow_same_candle": False},
                {
                    "type": "bullish_engulfing",
                    "allow_same_candle": True,
                },
            ],
        }
    )
    assert isinstance(then_expr, ThenExpression)
    assert len(then_expr.steps) == 3
    assert then_expr.steps[0].allow_same_candle is False
    assert then_expr.steps[2].allow_same_candle is True
    assert isinstance(then_expr.steps[2].expression, ObservationRequirement)


def test_nested_and_or_inside_then():
    expr = parse_expression(
        {
            "op": "then",
            "steps": [
                {
                    "op": "and",
                    "children": [_leaf("swing_high"), _leaf("equal_highs")],
                },
                {
                    "op": "or",
                    "children": [_leaf("bullish_engulfing"), _leaf("bullish_flag")],
                    "allow_same_candle": True,
                },
            ],
        }
    )
    assert isinstance(expr, ThenExpression)
    assert isinstance(expr.steps[0].expression, AndExpression)
    assert isinstance(expr.steps[1].expression, OrExpression)
    assert expr.steps[1].allow_same_candle is True


def test_leaf_defaults_and_variant():
    req = parse_expression(
        {"type": "fair_value_gap", "variant": "bullish", "min_count": 2}
    )
    assert isinstance(req, ObservationRequirement)
    assert req.variant == "bullish"
    assert req.min_count == 2
    assert req.allow_developing is False


def test_empty_and_children_rejected():
    with pytest.raises(SetupValidationError, match="non-empty"):
        parse_expression({"op": "and", "children": []})


def test_empty_or_children_rejected():
    with pytest.raises(SetupValidationError, match="non-empty"):
        parse_expression({"op": "or", "children": []})


def test_empty_then_steps_rejected():
    with pytest.raises(SetupValidationError, match="non-empty"):
        parse_expression({"op": "then", "steps": []})


def test_unknown_operator_rejected():
    with pytest.raises(SetupValidationError, match="unknown operator"):
        parse_expression({"op": "xor", "children": [_leaf()]})


def test_missing_leaf_type_rejected():
    with pytest.raises(SetupValidationError, match="type"):
        parse_expression({"min_count": 1})


def test_invalid_min_count_zero_negative_non_int():
    with pytest.raises(SetupValidationError, match="min_count"):
        parse_expression({"type": "x", "min_count": 0})
    with pytest.raises(SetupValidationError, match="min_count"):
        parse_expression({"type": "x", "min_count": -1})
    with pytest.raises(SetupValidationError, match="min_count"):
        parse_expression({"type": "x", "min_count": 1.5})
    with pytest.raises(SetupValidationError, match="min_count"):
        parse_expression({"type": "x", "min_count": True})


def test_invalid_allow_developing_type():
    with pytest.raises(SetupValidationError, match="allow_developing"):
        parse_expression({"type": "x", "allow_developing": "yes"})


def test_invalid_same_candle_type():
    with pytest.raises(SetupValidationError, match="allow_same_candle"):
        parse_expression(
            {
                "op": "then",
                "steps": [{"type": "x", "allow_same_candle": "true"}],
            }
        )


def test_allow_same_candle_rejected_on_first_then_step():
    with pytest.raises(SetupValidationError, match="first THEN step"):
        parse_expression(
            {
                "op": "then",
                "steps": [
                    {"type": "swing_high", "allow_same_candle": True},
                    {"type": "zone_retest"},
                ],
            }
        )


def test_allow_same_candle_accepted_on_later_then_step():
    expr = parse_expression(
        {
            "op": "then",
            "steps": [
                {"type": "swing_high"},
                {"type": "zone_retest", "allow_same_candle": True},
            ],
        }
    )
    assert isinstance(expr, ThenExpression)
    assert expr.steps[0].allow_same_candle is False
    assert expr.steps[1].allow_same_candle is True


def test_allow_same_candle_default_false_on_first_step():
    expr = parse_expression(
        {
            "op": "then",
            "steps": [
                {"type": "swing_high"},
                {"type": "zone_retest"},
            ],
        }
    )
    assert expr.steps[0].allow_same_candle is False
    assert expr.steps[1].allow_same_candle is False


def test_unknown_fields_on_leaf_rejected():
    with pytest.raises(SetupValidationError, match="unknown field"):
        parse_expression({"type": "x", "extra": 1})


def test_unknown_fields_on_and_rejected():
    with pytest.raises(SetupValidationError, match="unknown field"):
        parse_expression({"op": "and", "children": [_leaf()], "foo": 1})


def test_missing_required_fields():
    base = {
        "schema_version": 1,
        "setup_id": "s",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": _leaf(),
    }
    for key in ("schema_version", "setup_id", "symbol", "timeframe", "enabled", "expression"):
        bad = copy.deepcopy(base)
        del bad[key]
        with pytest.raises(SetupValidationError, match="missing required|schema_version"):
            parse_setup(bad)


def test_unsupported_schema_version():
    with pytest.raises(SetupValidationError, match="unsupported schema_version"):
        parse_setup(
            {
                "schema_version": 99,
                "setup_id": "s",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": True,
                "expression": _leaf(),
            }
        )


def test_schema_version_rejects_non_integer_types():
    """True == 1 and 1.0 == 1 in Python; require a real integer."""
    base = {
        "setup_id": "s",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": _leaf(),
    }
    for bad in (True, False, 1.0, "1"):
        raw = {**base, "schema_version": bad}
        with pytest.raises(SetupValidationError, match="schema_version must be an integer"):
            parse_setup(raw)
    # Integer 1 remains accepted
    ok = parse_setup({**base, "schema_version": 1})
    assert ok.schema_version == 1


def test_duplicate_setup_ids():
    raw = [
        {
            "schema_version": 1,
            "setup_id": "dup",
            "symbol": "R_100",
            "timeframe": "1m",
            "enabled": True,
            "expression": _leaf(),
        },
        {
            "schema_version": 1,
            "setup_id": "dup",
            "symbol": "R_75",
            "timeframe": "5m",
            "enabled": True,
            "expression": _leaf("swing_high"),
        },
    ]
    with pytest.raises(SetupValidationError, match="duplicate setup_id"):
        parse_setups(raw)


def test_unknown_setup_fields_rejected():
    with pytest.raises(SetupValidationError, match="unknown field"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "s",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": True,
                "expression": _leaf(),
                "priority": 1,
            }
        )


def test_malformed_reserved_fields():
    with pytest.raises(SetupValidationError, match="invalidation"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "s",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": True,
                "expression": _leaf(),
                "invalidation": "not-a-mapping",
            }
        )
    with pytest.raises(SetupValidationError, match="expiry"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "s",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": True,
                "expression": _leaf(),
                "expiry": 42,
            }
        )


def test_disabled_but_malformed_still_validated():
    with pytest.raises(SetupValidationError, match="min_count"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "disabled_bad",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": False,
                "expression": {"type": "x", "min_count": 0},
            }
        )


def test_empty_setup_id_rejected():
    with pytest.raises(SetupValidationError, match="non-empty"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "  ",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": True,
                "expression": _leaf(),
            }
        )


def test_enabled_must_be_bool():
    with pytest.raises(SetupValidationError, match="enabled"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "s",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": "true",
                "expression": _leaf(),
            }
        )


def test_validation_error_includes_path_and_setup_id():
    with pytest.raises(SetupValidationError) as ei:
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "path_check",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": True,
                "expression": {"type": "x", "min_count": -3},
            }
        )
    msg = str(ei.value)
    assert "path_check" in msg
    assert "min_count" in msg


def test_legacy_conditions_still_load_from_default_yaml():
    cfg = load_config("config/default.yaml")
    conditions = conditions_from_config(cfg)
    assert len(conditions) >= 1
    c = conditions[0]
    assert isinstance(c, ConditionDef)
    assert c.id == "example_bullish_engulfing_1m"
    assert c.symbol == "R_100"
    assert c.timeframe == "1m"
    assert c.enabled is True
    assert c.require_observations == [{"type": "bullish_engulfing", "min_count": 1}]


def test_legacy_defaults_preserved():
    cfg = {
        "conditions": [
            {
                "id": "legacy_only",
                "symbol": "R_100",
                "timeframe": "1m",
            }
        ]
    }
    conds = conditions_from_config(cfg)
    assert len(conds) == 1
    assert conds[0].enabled is True
    assert conds[0].require_observations == []


def test_legacy_not_silently_reinterpreted_as_setup():
    cfg = {
        "conditions": [
            {
                "id": "legacy",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": True,
                "require_observations": [{"type": "bullish_engulfing", "min_count": 1}],
            }
        ]
    }
    assert setups_from_config(cfg) == []
    conds = conditions_from_config(cfg)
    assert len(conds) == 1
    assert conds[0].require_observations[0]["type"] == "bullish_engulfing"


def test_setups_and_legacy_can_coexist():
    cfg = {
        "conditions": [
            {
                "id": "legacy",
                "symbol": "R_100",
                "timeframe": "1m",
                "require_observations": [{"type": "bullish_engulfing"}],
            }
        ],
        "setups": [
            {
                "schema_version": 1,
                "setup_id": "new_setup",
                "symbol": "R_100",
                "timeframe": "1m",
                "enabled": True,
                "expression": {"type": "swing_high"},
            }
        ],
    }
    assert len(conditions_from_config(cfg)) == 1
    assert len(setups_from_config(cfg)) == 1


def test_legacy_condition_eval_unchanged():
    from cpab.conditions.evaluator import ConditionEvaluator
    from cpab.models import Observation

    cond = ConditionDef(
        id="ex",
        symbol="R_100",
        timeframe="1m",
        require_observations=[{"type": "bullish_engulfing", "min_count": 1}],
    )
    obs = [
        Observation.create(
            type="bullish_engulfing",
            symbol="R_100",
            timeframe="1m",
            candle_epoch=100,
            confirmed_at=100,
        )
    ]
    result = ConditionEvaluator().evaluate(cond, obs, candle_epoch=100)
    assert result.met is True


def _sample_setup(**overrides):
    raw = {
        "schema_version": 1,
        "setup_id": "id_stable",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": {
            "op": "and",
            "children": [
                {"type": "swing_high"},
                {"type": "zone_retest", "min_count": 1},
            ],
        },
    }
    raw.update(overrides)
    return parse_setup(raw)


def test_equivalent_configs_same_identity():
    a = _sample_setup()
    b = _sample_setup()
    assert setup_config_identity(a) == setup_config_identity(b)


def test_yaml_key_order_does_not_change_identity():
    yaml_a = """
schema_version: 1
setup_id: order_test
symbol: R_100
timeframe: 1m
enabled: true
expression:
  type: swing_high
  min_count: 1
"""
    yaml_b = """
enabled: true
expression:
  min_count: 1
  type: swing_high
timeframe: 1m
symbol: R_100
setup_id: order_test
schema_version: 1
"""
    sa = parse_setup(yaml.safe_load(yaml_a))
    sb = parse_setup(yaml.safe_load(yaml_b))
    assert setup_config_identity(sa) == setup_config_identity(sb)


def test_material_change_changes_identity():
    a = _sample_setup()
    b = _sample_setup(expression={"type": "swing_low"})
    assert setup_config_identity(a) != setup_config_identity(b)
    c = _sample_setup(enabled=False)
    assert setup_config_identity(a) != setup_config_identity(c)
    d = _sample_setup(symbol="R_75")
    assert setup_config_identity(a) != setup_config_identity(d)


def test_presentation_only_does_not_change_identity():
    a = _sample_setup()
    b = _sample_setup(name="Pretty name", description="long text")
    assert setup_config_identity(a) == setup_config_identity(b)
    assert a.setup_id == b.setup_id == "id_stable"


def test_setup_id_stable_when_identity_changes():
    a = _sample_setup()
    b = _sample_setup(expression={"type": "bullish_engulfing"})
    assert a.setup_id == b.setup_id
    assert setup_config_identity(a) != setup_config_identity(b)


def test_reserved_fields_affect_identity_when_present():
    a = _sample_setup()
    b = _sample_setup(invalidation={"type": "structure_break"})
    assert setup_config_identity(a) != setup_config_identity(b)


def test_to_behavior_dict_excludes_name_description():
    s = _sample_setup(name="N", description="D")
    d = s.to_behavior_dict()
    assert "name" not in d
    assert "description" not in d
    assert "expression" in d
    assert d["setup_id"] == "id_stable"
