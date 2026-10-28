# Phase 7E — Setup Monitoring (Observational Snapshot)

## Scope

Lightweight, in-memory monitoring of the latest **attempted** evaluation cycle
on `ConditionEngine`, including cycles blocked by readiness or fail-closed
series health.

## Semantics

- Snapshot is recorded at the end of every `evaluate_once` call.
- **Blocked cycles** (readiness not READY) still produce a snapshot so operators
  can inspect readiness and current setup identity/state without advancing
  lifecycle or retaining evidence.
- **Detector-emitted** observation ids: returned by detectors this cycle.
- **Newly accepted** observation ids: subset actually inserted by
  `ObservationStore.accept` (first-write-wins; duplicates skipped).
- Monitoring is **observational only**: reading `last_monitoring_snapshot`
  never mutates engine, store, lifecycle, or readiness state and never emits
  signals.

## Surface

- `ConditionEngine.last_monitoring_snapshot` → `MonitoringSnapshot`
- Types: `MonitoringSnapshot`, `SeriesCycleView`, `SetupCycleView`
  in `cpab.core.monitoring`

## Non-goals

No persistence, external alerts, dashboard, broker integration, live trading,
new polling loop, or duplicate infrastructure.
