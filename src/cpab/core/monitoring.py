"""Phase 7E: observational in-memory monitoring snapshot for setup cycles.

Snapshot semantics
------------------
``MonitoringSnapshot`` describes the **latest attempted evaluation cycle**,
including cycles blocked by readiness or fail-closed series health.

It is purely observational:
  - Built from state already produced by ConditionEngine.evaluate_once.
  - Never mutates engine, store, lifecycle, or readiness state.
  - Never emits signals or schedules work.

Detector-emitted vs newly accepted
----------------------------------
  - ``detector_emitted_observation_ids``: observations returned by detectors
    during the cycle (may include ids already retained from prior cycles).
  - ``newly_accepted_observation_ids``: subset that ObservationStore.accept
    actually inserted this cycle (first-write-wins; duplicates skipped).

Blocked cycles (readiness not READY) still produce a snapshot so operators
can inspect readiness and current setup identity/state without advancing
lifecycle or retaining evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class SeriesCycleView:
    """Per-series health and evaluation candle for one attempted cycle."""

    symbol: str
    timeframe: str
    data_ok: bool
    diagnostics: tuple[str, ...]
    evaluation_candle_epoch: Optional[int]


@dataclass(frozen=True)
class SetupCycleView:
    """Per-setup identity and lifecycle outcome for one attempted cycle."""

    setup_id: str
    instance_id: str
    config_id: str
    enabled: bool
    state: str
    previous_state: Optional[str]
    transitioned: bool
    diagnostic_only: bool
    reasons: tuple[str, ...]
    anchor_candle_epoch: Optional[int]
    last_evaluation_candle: Optional[int]


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Immutable view of the latest attempted ConditionEngine cycle.

    Always reflects the most recent ``evaluate_once`` invocation, whether that
    cycle was fully evaluated, health-gated, or readiness-blocked.
    """

    # True after the first evaluate_once call that recorded this snapshot.
    cycle_recorded: bool
    # Readiness gate blocked the entire cycle (no detect/store/eval).
    evaluation_blocked: bool
    block_reason: Optional[str]
    readiness_state: str
    readiness_reason: str
    series: tuple[SeriesCycleView, ...]
    setups: tuple[SetupCycleView, ...]
    detector_emitted_observation_ids: tuple[str, ...]
    newly_accepted_observation_ids: tuple[str, ...]
    signal_ids: tuple[str, ...]
    signal_count: int


def empty_monitoring_snapshot(
    *,
    readiness_state: str = "INITIALIZING",
    readiness_reason: str = "startup",
) -> MonitoringSnapshot:
    """Placeholder before any evaluate_once has run."""
    return MonitoringSnapshot(
        cycle_recorded=False,
        evaluation_blocked=False,
        block_reason=None,
        readiness_state=readiness_state,
        readiness_reason=readiness_reason,
        series=(),
        setups=(),
        detector_emitted_observation_ids=(),
        newly_accepted_observation_ids=(),
        signal_ids=(),
        signal_count=0,
    )


def build_setup_views(
    setups: Sequence,
    instances: dict,
    *,
    cycle_outcomes: Optional[dict[str, dict]] = None,
) -> tuple[SetupCycleView, ...]:
    """Build setup views from current instances and optional cycle outcomes.

    ``cycle_outcomes`` maps setup_id -> {
        previous_state, transitioned, diagnostic_only, reasons
    } captured during the cycle. Missing entries use instance state only
    (previous_state=None, transitioned=False, diagnostic_only=False).
    """
    outcomes = cycle_outcomes or {}
    views: list[SetupCycleView] = []
    for setup in setups:
        inst = instances.get(setup.setup_id)
        if inst is None:
            continue
        oc = outcomes.get(setup.setup_id, {})
        prev = oc.get("previous_state")
        views.append(
            SetupCycleView(
                setup_id=setup.setup_id,
                instance_id=inst.instance_id,
                config_id=inst.config_id,
                enabled=bool(getattr(setup, "enabled", True)),
                state=inst.state.value if hasattr(inst.state, "value") else str(inst.state),
                previous_state=(
                    prev.value if prev is not None and hasattr(prev, "value") else prev
                ),
                transitioned=bool(oc.get("transitioned", False)),
                diagnostic_only=bool(oc.get("diagnostic_only", False)),
                reasons=tuple(oc.get("reasons") or inst.last_transition_reasons or ()),
                anchor_candle_epoch=inst.anchor_candle_epoch,
                last_evaluation_candle=inst.last_evaluation_candle,
            )
        )
    return tuple(views)
