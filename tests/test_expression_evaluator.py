"""Phase 7B: focused tests for deterministic expression evaluation."""

from __future__ import annotations

import pytest

from cpab.conditions.expression import (
    AndExpression,
    ObservationRequirement,
    OrExpression,
    ThenExpression,
    ThenStep,
)
from cpab.conditions.expression_evaluator import (
    ExpressionEvaluator,
    ThenProgress,
)
from cpab.conditions.validation import parse_expression
from cpab.models import Observation


SYMBOL = "R_100"
TF = "1m"


def _obs(
    type: str,
    epoch: int,
    *,
    oid: str | None = None,
    variant: str | None = None,
    confirmed_at: int | None = None,
    developing: bool = False,
    symbol: str = SYMBOL,
    timeframe: str = TF,
) -> Observation:
    conf = None if developing else (confirmed_at if confirmed_at is not None else epoch)
    return Observation.create(
        type,
        symbol,
        timeframe,
        epoch,
        id=oid or f"{type}|{epoch}|{variant or ''}|{conf}",
        confirmed_at=conf,
        developing=developing,
        variant=variant,
        event_time=epoch,
        detector_id="test",
    )


@pytest.fixture
def ev() -> ExpressionEvaluator:
    return ExpressionEvaluator()


# ---------------------------------------------------------------------------
# Leaf evaluation
# ---------------------------------------------------------------------------


def test_leaf_type_match(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="bullish_engulfing")
    obs = [_obs("bullish_engulfing", 100), _obs("bearish_engulfing", 100, oid="other")]
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is True
    assert len(r.evidence) == 1
    assert r.evidence[0].observation_type == "bullish_engulfing"


def test_leaf_variant_only_when_specified(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="swing", variant="hh")
    obs = [
        _obs("swing", 100, variant="hl", oid="a"),
        _obs("swing", 100, variant="hh", oid="b"),
    ]
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is True
    assert r.evidence[0].observation_id == "b"

    expr_any = ObservationRequirement(type="swing")
    r2 = ev.evaluate(expr_any, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r2.qualified is True
    # earliest conf then id — both same conf; id "a" first
    assert r2.evidence[0].observation_id == "a"


def test_leaf_symbol_timeframe(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="bullish_engulfing")
    obs = [
        _obs("bullish_engulfing", 100, symbol="OTHER"),
        _obs("bullish_engulfing", 100, timeframe="5m", oid="tf"),
        _obs("bullish_engulfing", 100, oid="ok"),
    ]
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is True
    assert r.evidence[0].observation_id == "ok"


def test_leaf_min_count_and_duplicate_ids(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="touch", min_count=2)
    # Same logical id cannot count twice
    o1 = _obs("touch", 90, oid="dup")
    o2 = _obs("touch", 100, oid="dup")  # same id
    o3 = _obs("touch", 100, oid="unique")
    r = ev.evaluate(expr, [o1, o2, o3], symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    # Two distinct ids → qualifies; duplicate id does not double-count
    assert r.qualified is True
    assert len({e.observation_id for e in r.evidence}) == 2

    # Only one unique id → insufficient
    r_one = ev.evaluate(
        expr, [o1, o2], symbol=SYMBOL, timeframe=TF, candle_epoch=100
    )
    assert r_one.qualified is False


def test_leaf_min_count_insufficient(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="touch", min_count=3)
    obs = [_obs("touch", 100, oid="a"), _obs("touch", 100, oid="b")]
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is False


def test_leaf_developing_rejected_even_when_allow_developing(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="zone_touch", allow_developing=True)
    dev = _obs("zone_touch", 100, developing=True, oid="dev")
    conf = _obs("zone_touch", 100, oid="conf")
    r_dev = ev.evaluate(expr, [dev], symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r_dev.qualified is False
    r_ok = ev.evaluate(expr, [dev, conf], symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r_ok.qualified is True
    assert r_ok.evidence[0].observation_id == "conf"


def test_leaf_causality_and_future_rejection(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="swing_high")
    future = _obs("swing_high", 200, confirmed_at=200, oid="fut")
    past = _obs("swing_high", 50, confirmed_at=50, oid="past")
    # At T=100, future not eligible
    r = ev.evaluate(expr, [future, past], symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    # past is eligible but not activation at 100 → require_activation fails
    assert r.qualified is False

    # At T=50, past activates
    r2 = ev.evaluate(expr, [future, past], symbol=SYMBOL, timeframe=TF, candle_epoch=50)
    assert r2.qualified is True
    assert r2.evidence[0].observation_id == "past"

    # At T=200, both eligible; activation seed selects future so qualification succeeds
    r3 = ev.evaluate(expr, [future, past], symbol=SYMBOL, timeframe=TF, candle_epoch=200)
    assert r3.qualified is True
    assert r3.evidence[0].observation_id == "fut"

    # Only future at T=200
    r4 = ev.evaluate(expr, [future], symbol=SYMBOL, timeframe=TF, candle_epoch=200)
    assert r4.qualified is True


def test_leaf_activation_required_at_T(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="engulfing")
    prior = _obs("engulfing", 80, confirmed_at=80, oid="p")
    r = ev.evaluate(expr, [prior], symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is False
    assert any("activation" in x for x in r.reasons)


# ---------------------------------------------------------------------------
# AND / OR
# ---------------------------------------------------------------------------


def test_and_requires_every_child(ev: ExpressionEvaluator):
    expr = AndExpression(
        children=(
            ObservationRequirement(type="a"),
            ObservationRequirement(type="b"),
        )
    )
    r = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
    )
    assert r.qualified is False

    r2 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1"), _obs("b", 100, oid="b1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
    )
    assert r2.qualified is True
    ids = {e.observation_id for e in r2.evidence}
    assert ids == {"a1", "b1"}


def test_and_child_order_does_not_affect_qualification(ev: ExpressionEvaluator):
    expr1 = AndExpression(
        children=(
            ObservationRequirement(type="a"),
            ObservationRequirement(type="b"),
        )
    )
    expr2 = AndExpression(
        children=(
            ObservationRequirement(type="b"),
            ObservationRequirement(type="a"),
        )
    )
    obs = [_obs("a", 100, oid="a1"), _obs("b", 100, oid="b1")]
    r1 = ev.evaluate(expr1, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    r2 = ev.evaluate(expr2, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r1.qualified is True and r2.qualified is True
    assert {e.observation_id for e in r1.evidence} == {e.observation_id for e in r2.evidence}


def test_and_prior_only_no_repeated_activation(ev: ExpressionEvaluator):
    expr = AndExpression(
        children=(
            ObservationRequirement(type="swing"),
            ObservationRequirement(type="retest"),
        )
    )
    obs = [
        _obs("swing", 50, confirmed_at=50, oid="s"),
        _obs("retest", 60, confirmed_at=60, oid="r"),
    ]
    # At T=100 nothing new
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is False

    # New activation on one leg at T=100
    obs2 = obs + [_obs("retest", 100, confirmed_at=100, oid="r2")]
    r2 = ev.evaluate(expr, obs2, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r2.qualified is True


def test_or_declared_child_order(ev: ExpressionEvaluator):
    expr = OrExpression(
        children=(
            ObservationRequirement(type="a"),
            ObservationRequirement(type="b"),
        )
    )
    obs = [
        _obs("b", 100, oid="b1"),
        _obs("a", 100, oid="a1"),
    ]
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is True
    # First child in declared order is "a"
    assert r.evidence[0].observation_id == "a1"


def test_or_second_branch_when_first_fails(ev: ExpressionEvaluator):
    expr = OrExpression(
        children=(
            ObservationRequirement(type="a"),
            ObservationRequirement(type="b"),
        )
    )
    obs = [_obs("b", 100, oid="b1")]
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is True
    assert r.evidence[0].observation_id == "b1"


def test_evidence_earliest_confirmation_then_id(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="x", min_count=1)
    obs = [
        _obs("x", 100, confirmed_at=100, oid="z"),
        _obs("x", 90, confirmed_at=90, oid="a"),
        _obs("x", 90, confirmed_at=90, oid="b"),
    ]
    # Without activation gate we'd pick earliest; at T=90 activation works
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=90)
    assert r.qualified is True
    assert r.evidence[0].observation_id == "a"  # earliest conf, then id


def test_and_deterministic_dedup(ev: ExpressionEvaluator):
    expr = AndExpression(
        children=(
            ObservationRequirement(type="a"),
            ObservationRequirement(type="a", min_count=1),
        )
    )
    # Two requirements of same type need two distinct observations
    obs = [_obs("a", 100, oid="a1"), _obs("a", 100, oid="a2")]
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is True
    assert len({e.observation_id for e in r.evidence}) == 2


# ---------------------------------------------------------------------------
# THEN
# ---------------------------------------------------------------------------


def test_then_ordered_progression(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(
            ThenStep(expression=ObservationRequirement(type="step1")),
            ThenStep(expression=ObservationRequirement(type="step2")),
        )
    )
    progress = ThenProgress()
    # Step 1 at T=100
    r1 = ev.evaluate(
        expr,
        [_obs("step1", 100, oid="s1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=progress,
    )
    assert r1.qualified is False  # not complete yet
    assert r1.advanced is True
    assert r1.progress is not None
    assert r1.progress.next_step_index == 1
    assert r1.progress.completed is False
    assert len(r1.progress.committed_steps) == 1

    # Step 2 needs later candle by default
    r2 = ev.evaluate(
        expr,
        [
            _obs("step1", 100, oid="s1"),
            _obs("step2", 100, oid="s2_same"),
        ],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=r1.progress,
    )
    assert r2.advanced is False
    assert r2.progress == r1.progress

    r3 = ev.evaluate(
        expr,
        [
            _obs("step1", 100, oid="s1"),
            _obs("step2", 160, oid="s2"),
        ],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=160,
        progress=r1.progress,
    )
    assert r3.advanced is True
    assert r3.qualified is True
    assert r3.progress is not None
    assert r3.progress.completed is True
    assert r3.progress.next_step_index == 2
    assert len(r3.progress.committed_steps) == 2
    assert r3.progress.committed_steps[0].evidence[0].observation_id == "s1"
    assert r3.progress.committed_steps[1].evidence[0].observation_id == "s2"


def test_then_no_skip_steps(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(
            ThenStep(expression=ObservationRequirement(type="a")),
            ThenStep(expression=ObservationRequirement(type="b")),
            ThenStep(expression=ObservationRequirement(type="c")),
        )
    )
    # Only b and c present — cannot skip to them
    r = ev.evaluate(
        expr,
        [_obs("b", 100, oid="b"), _obs("c", 100, oid="c")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=ThenProgress(),
    )
    assert r.advanced is False
    assert r.progress is not None
    assert r.progress.next_step_index == 0


def test_then_first_evidence_committed(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(
            ThenStep(expression=ObservationRequirement(type="a")),
            ThenStep(expression=ObservationRequirement(type="b")),
        )
    )
    r1 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a_first")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=ThenProgress(),
    )
    # Later evaluation with another "a" must not replace committed evidence
    r2 = ev.evaluate(
        expr,
        [
            _obs("a", 100, oid="a_first"),
            _obs("a", 160, oid="a_later"),
            _obs("b", 160, oid="b1"),
        ],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=160,
        progress=r1.progress,
    )
    assert r2.progress is not None
    assert r2.progress.committed_steps[0].evidence[0].observation_id == "a_first"


def test_then_same_candle_opt_in(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(
            ThenStep(expression=ObservationRequirement(type="a")),
            ThenStep(
                expression=ObservationRequirement(type="b"),
                allow_same_candle=True,
            ),
        )
    )
    r1 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1"), _obs("b", 100, oid="b1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=ThenProgress(),
    )
    assert r1.advanced is True
    assert r1.progress is not None
    assert r1.progress.next_step_index == 1

    r2 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1"), _obs("b", 100, oid="b1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=r1.progress,
    )
    assert r2.advanced is True
    assert r2.qualified is True
    assert r2.progress is not None
    assert r2.progress.completed is True


def test_then_non_decreasing_confirmation(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(
            ThenStep(expression=ObservationRequirement(type="a")),
            ThenStep(expression=ObservationRequirement(type="b")),
        )
    )
    r1 = ev.evaluate(
        expr,
        [_obs("a", 100, confirmed_at=100, oid="a1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=ThenProgress(),
    )
    assert r1.progress is not None
    # Artificially raise last confirmation so a later-candle activation with
    # a lower confirmation time is rejected by the non-decreasing guard.
    # (Normal activation has conf == T which is non-decreasing with candles;
    # the guard still protects progress integrity for edge / reconstructed state.)
    from cpab.conditions.expression_evaluator import ThenStepProvenance, EvidenceItem

    inflated = ThenProgress(
        next_step_index=1,
        committed_steps=r1.progress.committed_steps,
        last_qualifying_evaluation_candle=100,
        last_qualifying_confirmation_time=500,
        completed=False,
    )
    r2 = ev.evaluate(
        expr,
        [
            _obs("a", 100, confirmed_at=100, oid="a1"),
            _obs("b", 160, confirmed_at=160, oid="b1"),
        ],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=160,
        progress=inflated,
    )
    assert r2.advanced is False
    assert any("confirmation time" in x for x in r2.reasons)


def test_then_prior_cannot_advance_without_current_event(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(
            ThenStep(expression=ObservationRequirement(type="a")),
            ThenStep(expression=ObservationRequirement(type="b")),
        )
    )
    r1 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=ThenProgress(),
    )
    # b confirmed earlier, evaluated later — prior only, no activation at 160
    r2 = ev.evaluate(
        expr,
        [
            _obs("a", 100, oid="a1"),
            _obs("b", 110, confirmed_at=110, oid="b_prior"),
        ],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=160,
        progress=r1.progress,
    )
    assert r2.advanced is False


def test_then_completion_stable_reeval(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(
            ThenStep(expression=ObservationRequirement(type="a")),
            ThenStep(expression=ObservationRequirement(type="b")),
        )
    )
    p = ThenProgress()
    r1 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=p,
    )
    r2 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1"), _obs("b", 160, oid="b1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=160,
        progress=r1.progress,
    )
    assert r2.progress is not None and r2.progress.completed is True

    r3 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1"), _obs("b", 160, oid="b1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=160,
        progress=r2.progress,
    )
    assert r3.advanced is False
    assert r3.qualified is True
    assert r3.progress == r2.progress

    # Later candle still no restart
    r4 = ev.evaluate(
        expr,
        [
            _obs("a", 100, oid="a1"),
            _obs("b", 160, oid="b1"),
            _obs("a", 200, oid="a2"),
            _obs("b", 200, oid="b2"),
        ],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=200,
        progress=r2.progress,
    )
    assert r4.advanced is False
    assert r4.progress is not None
    assert r4.progress.completed is True
    assert r4.progress.committed_steps[0].evidence[0].observation_id == "a1"


def test_then_reject_developing(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(ThenStep(expression=ObservationRequirement(type="a", allow_developing=True)),)
    )
    r = ev.evaluate(
        expr,
        [_obs("a", 100, developing=True, oid="dev")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=ThenProgress(),
    )
    assert r.advanced is False
    assert r.qualified is False


def test_parse_expression_then_evaluate(ev: ExpressionEvaluator):
    raw = {
        "op": "then",
        "steps": [
            {"type": "swing_high"},
            {"type": "bullish_engulfing", "allow_same_candle": False},
        ],
    }
    expr = parse_expression(raw)
    assert isinstance(expr, ThenExpression)
    r = ev.evaluate(
        expr,
        [_obs("swing_high", 100, oid="sh")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=ThenProgress(),
    )
    assert r.advanced is True
    assert r.progress is not None
    assert r.progress.next_step_index == 1


def test_idempotent_identical_inputs(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="x")
    obs = [_obs("x", 100, oid="x1")]
    r1 = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    r2 = ev.evaluate(expr, list(reversed(obs)), symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r1.qualified == r2.qualified
    assert [e.observation_id for e in r1.evidence] == [e.observation_id for e in r2.evidence]


# ---------------------------------------------------------------------------
# Review regression: order-independent AND, THEN context, nested THEN
# ---------------------------------------------------------------------------


def test_and_overlapping_requirements_order_independent(ev: ExpressionEvaluator):
    """Broad + restricted swing requirements must qualify regardless of child order."""
    obs = [
        _obs("swing", 100, variant="hh", oid="hh"),
        _obs("swing", 100, variant="hl", oid="hl"),
    ]
    expr_broad_first = AndExpression(
        children=(
            ObservationRequirement(type="swing"),
            ObservationRequirement(type="swing", variant="hh"),
        )
    )
    expr_restricted_first = AndExpression(
        children=(
            ObservationRequirement(type="swing", variant="hh"),
            ObservationRequirement(type="swing"),
        )
    )
    r1 = ev.evaluate(expr_broad_first, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    r2 = ev.evaluate(
        expr_restricted_first, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100
    )
    assert r1.qualified is True, r1.reasons
    assert r2.qualified is True, r2.reasons
    assert {e.observation_id for e in r1.evidence} == {"hh", "hl"}
    assert {e.observation_id for e in r2.evidence} == {"hh", "hl"}


def test_then_no_advance_without_evaluation_context_first_step(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(ThenStep(expression=ObservationRequirement(type="a")),)
    )
    # Prior confirmed obs present, but no candle_epoch / evaluation_time
    r = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1")],
        symbol=SYMBOL,
        timeframe=TF,
        progress=ThenProgress(),
    )
    assert r.advanced is False
    assert r.qualified is False
    assert r.progress is not None
    assert r.progress.next_step_index == 0
    assert any("evaluation context" in x for x in r.reasons)


def test_then_no_advance_without_evaluation_context_later_step(ev: ExpressionEvaluator):
    expr = ThenExpression(
        steps=(
            ThenStep(expression=ObservationRequirement(type="a")),
            ThenStep(expression=ObservationRequirement(type="b")),
        )
    )
    r1 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1")],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
        progress=ThenProgress(),
    )
    assert r1.advanced is True
    assert r1.progress is not None

    r2 = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1"), _obs("b", 160, oid="b1")],
        symbol=SYMBOL,
        timeframe=TF,
        # intentionally omit candle_epoch and evaluation_time
        progress=r1.progress,
    )
    assert r2.advanced is False
    assert r2.progress == r1.progress
    assert any("evaluation context" in x for x in r2.reasons)


def test_nested_then_rejected_by_validation():
    from cpab.conditions.validation import SetupValidationError, parse_expression

    with pytest.raises(SetupValidationError, match="nested THEN"):
        parse_expression(
            {
                "op": "and",
                "children": [
                    {"op": "then", "steps": [{"type": "a"}]},
                    {"type": "b"},
                ],
            }
        )
    with pytest.raises(SetupValidationError, match="nested THEN"):
        parse_expression(
            {
                "op": "then",
                "steps": [
                    {"op": "then", "steps": [{"type": "a"}]},
                    {"type": "b"},
                ],
            }
        )
    # AND/OR inside THEN remains valid
    expr = parse_expression(
        {
            "op": "then",
            "steps": [
                {"op": "and", "children": [{"type": "a"}, {"type": "b"}]},
                {"type": "c"},
            ],
        }
    )
    assert isinstance(expr, ThenExpression)


def test_leaf_no_qualify_without_evaluation_context(ev: ExpressionEvaluator):
    expr = ObservationRequirement(type="a")
    r = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1")],
        symbol=SYMBOL,
        timeframe=TF,
        # omit candle_epoch and evaluation_time
    )
    assert r.qualified is False
    assert any("evaluation context" in x for x in r.reasons)


def test_and_no_qualify_without_evaluation_context(ev: ExpressionEvaluator):
    expr = AndExpression(
        children=(
            ObservationRequirement(type="a"),
            ObservationRequirement(type="b"),
        )
    )
    r = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1"), _obs("b", 100, oid="b1")],
        symbol=SYMBOL,
        timeframe=TF,
    )
    assert r.qualified is False
    assert any("evaluation context" in x for x in r.reasons)


def test_or_no_qualify_without_evaluation_context(ev: ExpressionEvaluator):
    expr = OrExpression(
        children=(
            ObservationRequirement(type="a"),
            ObservationRequirement(type="b"),
        )
    )
    r = ev.evaluate(
        expr,
        [_obs("a", 100, oid="a1")],
        symbol=SYMBOL,
        timeframe=TF,
    )
    assert r.qualified is False
    assert any("evaluation context" in x for x in r.reasons)


def test_nested_and_propagates_activation_preference(ev: ExpressionEvaluator):
    """AND(AND(a, b)) must use current-event a as activation with prior b."""
    expr = AndExpression(
        children=(
            AndExpression(
                children=(
                    ObservationRequirement(type="a"),
                    ObservationRequirement(type="b"),
                )
            ),
        )
    )
    prior_a = _obs("a", 50, confirmed_at=50, oid="a_prior")
    prior_b = _obs("b", 60, confirmed_at=60, oid="b_prior")
    new_a = _obs("a", 100, confirmed_at=100, oid="a_new")

    for order in (
        [prior_a, prior_b, new_a],
        [new_a, prior_b, prior_a],
        [prior_b, prior_a, new_a],
        [new_a, prior_a, prior_b],
    ):
        r = ev.evaluate(expr, order, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
        assert r.qualified is True, (order, r.reasons)
        ids = {e.observation_id for e in r.evidence}
        assert "a_new" in ids, (order, ids)
        assert "b_prior" in ids, (order, ids)
        assert any(e.is_activation for e in r.evidence), (order, r.evidence)


def test_nested_and_prior_only_still_fails_activation(ev: ExpressionEvaluator):
    expr = AndExpression(
        children=(
            AndExpression(
                children=(
                    ObservationRequirement(type="a"),
                    ObservationRequirement(type="b"),
                )
            ),
        )
    )
    r = ev.evaluate(
        expr,
        [
            _obs("a", 50, confirmed_at=50, oid="a_prior"),
            _obs("b", 60, confirmed_at=60, oid="b_prior"),
        ],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
    )
    assert r.qualified is False


def test_and_broad_leaf_does_not_starve_nested_and(ev: ExpressionEvaluator):
    """Outer broad x + nested AND(x/hh, y) must find valid distinct allocation.

    Valid assignment: outer x uses x/hl; nested uses x/hh + y.
    """
    nested = AndExpression(
        children=(
            ObservationRequirement(type="x", variant="hh"),
            ObservationRequirement(type="y"),
        )
    )
    expr_broad_first = AndExpression(
        children=(
            ObservationRequirement(type="x"),
            nested,
        )
    )
    expr_nested_first = AndExpression(
        children=(
            nested,
            ObservationRequirement(type="x"),
        )
    )
    x_hh = _obs("x", 100, variant="hh", confirmed_at=100, oid="x_hh")
    x_hl = _obs("x", 50, variant="hl", confirmed_at=50, oid="x_hl")
    y_prior = _obs("y", 60, confirmed_at=60, oid="y_prior")

    for expr in (expr_broad_first, expr_nested_first):
        for order in (
            [x_hh, x_hl, y_prior],
            [y_prior, x_hl, x_hh],
            [x_hl, y_prior, x_hh],
            [x_hh, y_prior, x_hl],
        ):
            r = ev.evaluate(expr, order, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
            assert r.qualified is True, (expr, [o.id for o in order], r.reasons)
            ids = {e.observation_id for e in r.evidence}
            assert "x_hh" in ids
            assert "y_prior" in ids
            assert "x_hl" in ids
            assert any(e.is_activation for e in r.evidence)


def test_nested_and_does_not_starve_outer_specific_leaf(ev: ExpressionEvaluator):
    """Outer x/hh + nested AND(x any, y) must reconsider nested assignment.

    Valid: outer takes x/hh (activation); nested takes x/hl + y.
    Nested prefers activation first but outer backtracking forces the alternative.
    """
    nested = AndExpression(
        children=(
            ObservationRequirement(type="x"),
            ObservationRequirement(type="y"),
        )
    )
    expr_specific_first = AndExpression(
        children=(
            ObservationRequirement(type="x", variant="hh"),
            nested,
        )
    )
    expr_nested_first = AndExpression(
        children=(
            nested,
            ObservationRequirement(type="x", variant="hh"),
        )
    )
    x_hh = _obs("x", 100, variant="hh", confirmed_at=100, oid="x_hh")
    x_hl = _obs("x", 50, variant="hl", confirmed_at=50, oid="x_hl")
    y_prior = _obs("y", 60, confirmed_at=60, oid="y_prior")

    for expr in (expr_specific_first, expr_nested_first):
        for order in (
            [x_hh, x_hl, y_prior],
            [y_prior, x_hl, x_hh],
            [x_hl, y_prior, x_hh],
            [x_hh, y_prior, x_hl],
        ):
            r = ev.evaluate(expr, order, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
            assert r.qualified is True, (expr, [o.id for o in order], r.reasons)
            ids = {e.observation_id for e in r.evidence}
            assert ids == {"x_hh", "x_hl", "y_prior"}, ids
            assert any(e.is_activation and e.observation_id == "x_hh" for e in r.evidence)


def test_nested_or_preserves_declared_branch_order(ev: ExpressionEvaluator):
    """Under outer AND, nested OR prefers the first declared branch when both work."""
    # AND(OR(a, b), c) with both a and b available + c — should use a (first branch)
    expr = AndExpression(
        children=(
            OrExpression(
                children=(
                    ObservationRequirement(type="a"),
                    ObservationRequirement(type="b"),
                )
            ),
            ObservationRequirement(type="c"),
        )
    )
    obs = [
        _obs("a", 100, confirmed_at=100, oid="a1"),
        _obs("b", 100, confirmed_at=100, oid="b1"),
        _obs("c", 50, confirmed_at=50, oid="c1"),
    ]
    r = ev.evaluate(expr, obs, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r.qualified is True, r.reasons
    ids = {e.observation_id for e in r.evidence}
    assert "a1" in ids
    assert "c1" in ids
    assert "b1" not in ids

    # When only second OR branch works with outer (first branch starved):
    # AND(OR(a, b), a) with one a and one b — first branch takes a, outer fails;
    # must try second branch (b) so outer can take a.
    expr2 = AndExpression(
        children=(
            OrExpression(
                children=(
                    ObservationRequirement(type="a"),
                    ObservationRequirement(type="b"),
                )
            ),
            ObservationRequirement(type="a"),
        )
    )
    obs2 = [
        _obs("a", 100, confirmed_at=100, oid="a1"),
        _obs("b", 90, confirmed_at=90, oid="b1"),
    ]
    r2 = ev.evaluate(expr2, obs2, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
    assert r2.qualified is True, r2.reasons
    ids2 = {e.observation_id for e in r2.evidence}
    assert ids2 == {"a1", "b1"}


def test_and_or_activation_continues_past_prior_only_assignment(ev: ExpressionEvaluator):
    """AND(OR(a,b), c): prior a+c first must not block current b + prior c."""
    expr = AndExpression(
        children=(
            OrExpression(
                children=(
                    ObservationRequirement(type="a"),
                    ObservationRequirement(type="b"),
                )
            ),
            ObservationRequirement(type="c"),
        )
    )
    a_prior = _obs("a", 50, confirmed_at=50, oid="a_prior")
    b_new = _obs("b", 100, confirmed_at=100, oid="b_new")
    c_prior = _obs("c", 60, confirmed_at=60, oid="c_prior")

    for order in (
        [a_prior, b_new, c_prior],
        [c_prior, a_prior, b_new],
        [b_new, c_prior, a_prior],
        [a_prior, c_prior, b_new],
    ):
        r = ev.evaluate(expr, order, symbol=SYMBOL, timeframe=TF, candle_epoch=100)
        assert r.qualified is True, ([o.id for o in order], r.reasons)
        ids = {e.observation_id for e in r.evidence}
        assert "b_new" in ids, ids
        assert "c_prior" in ids, ids
        assert "a_prior" not in ids, ids
        assert any(e.is_activation for e in r.evidence)

    # Prior-only still cannot activate
    r_prior = ev.evaluate(
        expr,
        [a_prior, c_prior],
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=100,
    )
    assert r_prior.qualified is False
