"""Phase 7C: setup lifecycle, invalidation, and expiry."""

from __future__ import annotations

import pytest

from cpab.conditions.expression import (
    AndExpression,
    ObservationRequirement,
    OrExpression,
    ThenExpression,
    ThenStep,
)
from cpab.conditions.expression_evaluator import ThenProgress
from cpab.conditions.identity import setup_config_identity
from cpab.conditions.lifecycle import (
    LifecycleState,
    SetupLifecycleEvaluator,
    new_pending_instance,
    setup_instance_identity,
)
from cpab.conditions.schema import ExpiryConfig, SetupDef, SCHEMA_VERSION_V1
from cpab.conditions.validation import SetupValidationError, parse_setup
from cpab.models import Observation

SYMBOL = "R_100"
TF = "1m"
TF_SEC = 60


def _obs(
    typ: str,
    candle_epoch: int,
    *,
    confirmed_at: int | None = None,
    oid: str | None = None,
    variant: str | None = None,
) -> Observation:
    conf = confirmed_at if confirmed_at is not None else candle_epoch
    return Observation(
        id=oid or f"{typ}_{candle_epoch}",
        type=typ,
        symbol=SYMBOL,
        timeframe=TF,
        candle_epoch=candle_epoch,
        event_time=candle_epoch,
        confirmed_at=conf,
        detected_at=float(conf),
        variant=variant,
        developing=False,
    )


def _setup(
    expression,
    *,
    invalidation=None,
    expiry=None,
    setup_id: str = "s1",
) -> SetupDef:
    return SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id=setup_id,
        symbol=SYMBOL,
        timeframe=TF,
        enabled=True,
        expression=expression,
        invalidation=invalidation,
        expiry=expiry,
    )


@pytest.fixture
def life() -> SetupLifecycleEvaluator:
    return SetupLifecycleEvaluator()


def test_new_instance_pending_and_stable_identity():
    setup = _setup(ObservationRequirement(type="a"))
    a = new_pending_instance(setup, instance_key="1")
    b = new_pending_instance(setup, instance_key="1")
    c = new_pending_instance(setup, instance_key="2")
    assert a.state == LifecycleState.PENDING
    assert a.instance_id == b.instance_id
    assert a.instance_id != c.instance_id
    assert a.config_id == setup_config_identity(setup)
    assert a.instance_id.startswith("setupinst:")
    assert a.config_id.startswith("setupcfg:")


def test_non_then_qualifies_completes_immediately(life):
    setup = _setup(ObservationRequirement(type="a"))
    inst = new_pending_instance(setup, instance_key="1")
    r = life.evaluate(
        setup, inst, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    assert r.transitioned is True
    assert r.instance.state == LifecycleState.COMPLETED
    assert r.instance.anchor_candle_epoch == 100


def test_non_then_no_qualify_stays_pending(life):
    setup = _setup(ObservationRequirement(type="a"))
    inst = new_pending_instance(setup, instance_key="1")
    r = life.evaluate(setup, inst, [_obs("b", 100)], candle_epoch=100, timeframe_seconds=TF_SEC)
    assert r.transitioned is False
    assert r.instance.state == LifecycleState.PENDING
    assert r.instance.last_evaluation_candle == 100
    assert r.instance.anchor_candle_epoch == 100


def test_then_pending_to_in_progress_to_completed(life):
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        )
    )
    inst = new_pending_instance(setup, instance_key="1")
    r1 = life.evaluate(
        setup, inst, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    assert r1.transitioned is True
    assert r1.instance.state == LifecycleState.IN_PROGRESS
    assert r1.instance.progress is not None
    assert r1.instance.progress.next_step_index == 1

    r2 = life.evaluate(
        setup,
        r1.instance,
        [_obs("a", 100), _obs("b", 160)],
        candle_epoch=160,
        timeframe_seconds=TF_SEC,
    )
    assert r2.transitioned is True
    assert r2.instance.state == LifecycleState.COMPLETED
    assert r2.instance.progress is not None
    assert r2.instance.progress.completed is True


def test_then_no_advance_preserves_progress(life):
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        )
    )
    inst = new_pending_instance(setup, instance_key="1")
    r1 = life.evaluate(
        setup, inst, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    prog = r1.instance.progress
    r2 = life.evaluate(
        setup, r1.instance, [_obs("a", 100)], candle_epoch=160, timeframe_seconds=TF_SEC
    )
    assert r2.transitioned is False
    assert r2.instance.state == LifecycleState.IN_PROGRESS
    assert r2.instance.progress == prog


def test_invalidation_leaf_and_or(life):
    for inv in (
        ObservationRequirement(type="kill"),
        AndExpression(
            children=(
                ObservationRequirement(type="kill"),
                ObservationRequirement(type="flag"),
            )
        ),
        OrExpression(
            children=(
                ObservationRequirement(type="kill"),
                ObservationRequirement(type="other"),
            )
        ),
    ):
        setup = _setup(ObservationRequirement(type="a"), invalidation=inv)
        inst = new_pending_instance(setup, instance_key="x")
        obs = [_obs("kill", 100), _obs("flag", 100)]
        r = life.evaluate(setup, inst, obs, candle_epoch=100, timeframe_seconds=TF_SEC)
        assert r.instance.state == LifecycleState.INVALIDATED, inv
        assert r.transitioned is True


def test_invalidation_does_not_mutate_then_progress(life):
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        ),
        invalidation=ObservationRequirement(type="kill"),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r1 = life.evaluate(
        setup, inst, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    assert r1.instance.state == LifecycleState.IN_PROGRESS
    prog = r1.instance.progress
    r2 = life.evaluate(
        setup,
        r1.instance,
        [_obs("a", 100), _obs("kill", 160)],
        candle_epoch=160,
        timeframe_seconds=TF_SEC,
    )
    assert r2.instance.state == LifecycleState.INVALIDATED
    assert r2.instance.progress == prog


def test_expiry_inclusive_boundary(life):
    setup = _setup(
        ObservationRequirement(type="never"),
        expiry=ExpiryConfig(max_age_candles=3),
    )
    inst = new_pending_instance(setup, instance_key="1")
    # First accepted eval at 100 → anchor=100
    r = life.evaluate(setup, inst, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    assert r.instance.state == LifecycleState.PENDING
    assert r.instance.anchor_candle_epoch == 100

    # Step continuously: 160 (elapsed=1), 220 (elapsed=2) still pending
    for epoch in (160, 220):
        r = life.evaluate(
            setup, r.instance, [], candle_epoch=epoch, timeframe_seconds=TF_SEC
        )
        assert r.instance.state == LifecycleState.PENDING, epoch

    # elapsed=3 (>=3) inclusive at 280
    r = life.evaluate(
        setup, r.instance, [], candle_epoch=280, timeframe_seconds=TF_SEC
    )
    assert r.transitioned is True
    assert r.instance.state == LifecycleState.EXPIRED


def test_invalidation_beats_expiry(life):
    setup = _setup(
        ObservationRequirement(type="never"),
        invalidation=ObservationRequirement(type="kill"),
        expiry=ExpiryConfig(max_age_candles=1),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r0 = life.evaluate(setup, inst, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    r1 = life.evaluate(
        setup,
        r0.instance,
        [_obs("kill", 160)],
        candle_epoch=160,
        timeframe_seconds=TF_SEC,
    )
    assert r1.instance.state == LifecycleState.INVALIDATED


def test_invalidation_beats_completion_same_event(life):
    setup = _setup(
        ObservationRequirement(type="a"),
        invalidation=ObservationRequirement(type="a"),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r = life.evaluate(
        setup, inst, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    assert r.instance.state == LifecycleState.INVALIDATED


def test_terminal_immutable(life):
    setup = _setup(ObservationRequirement(type="a"))
    inst = new_pending_instance(setup, instance_key="1")
    r = life.evaluate(
        setup, inst, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    assert r.instance.state == LifecycleState.COMPLETED
    r2 = life.evaluate(
        setup, r.instance, [_obs("a", 160)], candle_epoch=160, timeframe_seconds=TF_SEC
    )
    assert r2.transitioned is False
    assert r2.instance.state == LifecycleState.COMPLETED
    assert "immutable" in r2.reasons[0]


def test_invalidation_then_rejected_at_validation():
    with pytest.raises(SetupValidationError, match="nested THEN|not permitted"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "s",
                "symbol": SYMBOL,
                "timeframe": TF,
                "enabled": True,
                "expression": {"type": "a"},
                "invalidation": {"op": "then", "steps": [{"type": "x"}]},
            }
        )


def test_expiry_malformed_rejected():
    with pytest.raises(SetupValidationError, match="max_age_candles"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "s",
                "symbol": SYMBOL,
                "timeframe": TF,
                "enabled": True,
                "expression": {"type": "a"},
                "expiry": {"bars": 10},
            }
        )
    with pytest.raises(SetupValidationError, match="positive"):
        parse_setup(
            {
                "schema_version": 1,
                "setup_id": "s",
                "symbol": SYMBOL,
                "timeframe": TF,
                "enabled": True,
                "expression": {"type": "a"},
                "expiry": {"max_age_candles": 0},
            }
        )


def test_unhealthy_data_no_state_change(life):
    setup = _setup(
        ObservationRequirement(type="a"),
        expiry=ExpiryConfig(max_age_candles=1),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r0 = life.evaluate(setup, inst, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    r1 = life.evaluate(
        setup,
        r0.instance,
        [_obs("a", 160)],
        candle_epoch=160,
        timeframe_seconds=TF_SEC,
        data_ok=False,
        data_diagnostics=("gap detected",),
    )
    assert r1.diagnostic_only is True
    assert r1.transitioned is False
    assert r1.instance.state == LifecycleState.PENDING
    assert r1.instance.last_evaluation_candle == 100


def test_stale_candle_rejected(life):
    setup = _setup(ObservationRequirement(type="a"))
    inst = new_pending_instance(setup, instance_key="1")
    r0 = life.evaluate(setup, inst, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    r1 = life.evaluate(
        setup, r0.instance, [_obs("a", 90)], candle_epoch=90, timeframe_seconds=TF_SEC
    )
    assert r1.diagnostic_only is True
    assert r1.instance.state == LifecycleState.PENDING


def test_idempotent_same_candle(life):
    setup = _setup(ObservationRequirement(type="never"))
    inst = new_pending_instance(setup, instance_key="1")
    r0 = life.evaluate(setup, inst, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    assert r0.instance.state == LifecycleState.PENDING
    r1 = life.evaluate(setup, r0.instance, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    assert r1.transitioned is False
    assert "idempotent" in r1.reasons[0]
    # Terminal path still immutable on same candle
    setup2 = _setup(ObservationRequirement(type="a"))
    inst2 = new_pending_instance(setup2, instance_key="2")
    done = life.evaluate(
        setup2, inst2, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    again = life.evaluate(
        setup2, done.instance, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    assert again.transitioned is False
    assert "immutable" in again.reasons[0]


def test_deterministic_replay(life):
    setup = _setup(
        ThenExpression(
            steps=(
                ThenStep(expression=ObservationRequirement(type="a")),
                ThenStep(expression=ObservationRequirement(type="b")),
            )
        )
    )
    events = [
        (100, [_obs("a", 100)]),
        (160, [_obs("a", 100), _obs("b", 160)]),
    ]

    def run():
        inst = new_pending_instance(setup, instance_key="replay")
        states = []
        for epoch, obs in events:
            r = life.evaluate(
                setup, inst, obs, candle_epoch=epoch, timeframe_seconds=TF_SEC
            )
            inst = r.instance
            states.append(
                (
                    inst.state,
                    inst.progress.next_step_index if inst.progress else None,
                    inst.progress.completed if inst.progress else None,
                    r.transitioned,
                )
            )
        return states

    assert run() == run()


def test_expiry_misaligned_epoch_does_not_expire(life):
    setup = _setup(
        ObservationRequirement(type="never"),
        expiry=ExpiryConfig(max_age_candles=1),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r0 = life.evaluate(setup, inst, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    # 150 is discontinuous / not next candle
    r1 = life.evaluate(
        setup, r0.instance, [], candle_epoch=150, timeframe_seconds=TF_SEC
    )
    assert r1.diagnostic_only is True
    assert r1.instance.state == LifecycleState.PENDING
    assert r1.instance.last_evaluation_candle == 100


def test_missing_timeframe_seconds_with_expiry_is_diagnostic(life):
    setup = _setup(
        ObservationRequirement(type="a"),
        expiry=ExpiryConfig(max_age_candles=5),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r = life.evaluate(
        setup, inst, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=None
    )
    assert r.diagnostic_only is True
    assert r.transitioned is False
    assert r.instance.state == LifecycleState.PENDING
    assert "timeframe_seconds" in r.reasons[0]


def test_skipped_candle_rejected_with_expiry_and_qualifying_expr(life):
    setup = _setup(
        ObservationRequirement(type="a"),
        expiry=ExpiryConfig(max_age_candles=10),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r0 = life.evaluate(setup, inst, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    # Skip 160; jump to 220 with qualifying observation
    r1 = life.evaluate(
        setup,
        r0.instance,
        [_obs("a", 220)],
        candle_epoch=220,
        timeframe_seconds=TF_SEC,
    )
    assert r1.diagnostic_only is True
    assert r1.transitioned is False
    assert r1.instance.state == LifecycleState.PENDING
    assert r1.instance.last_evaluation_candle == 100
    assert "discontinuous" in r1.reasons[0]


def test_config_id_mismatch_rejected(life):
    setup_a = _setup(ObservationRequirement(type="a"), setup_id="a")
    setup_b = _setup(ObservationRequirement(type="b"), setup_id="b")
    inst = new_pending_instance(setup_a, instance_key="1")
    r = life.evaluate(
        setup_b, inst, [_obs("b", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    assert r.diagnostic_only is True
    assert r.transitioned is False
    assert r.instance.state == LifecycleState.PENDING
    assert "mismatch" in r.reasons[0]


def test_simultaneous_invalidation_and_expiry_preserves_both_reasons(life):
    setup = _setup(
        ObservationRequirement(type="never"),
        invalidation=ObservationRequirement(type="kill"),
        expiry=ExpiryConfig(max_age_candles=1),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r0 = life.evaluate(setup, inst, [], candle_epoch=100, timeframe_seconds=TF_SEC)
    r1 = life.evaluate(
        setup,
        r0.instance,
        [_obs("kill", 160)],
        candle_epoch=160,
        timeframe_seconds=TF_SEC,
    )
    assert r1.instance.state == LifecycleState.INVALIDATED
    joined = " ".join(r1.reasons)
    assert "invalidation" in joined
    assert "expiry" in joined


def test_disabled_setup_does_not_transition(life):
    setup = SetupDef(
        schema_version=SCHEMA_VERSION_V1,
        setup_id="s1",
        symbol=SYMBOL,
        timeframe=TF,
        enabled=False,
        expression=ObservationRequirement(type="a"),
    )
    inst = new_pending_instance(setup, instance_key="1")
    r = life.evaluate(
        setup, inst, [_obs("a", 100)], candle_epoch=100, timeframe_seconds=TF_SEC
    )
    assert r.diagnostic_only is True
    assert r.transitioned is False
    assert r.instance.state == LifecycleState.PENDING
    assert "disabled" in r.reasons[0]
