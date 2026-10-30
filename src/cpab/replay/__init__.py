"""Phase 8B replay, parity, and report-only STALLED classification."""

from cpab.replay.driver import (
    CycleRecord,
    ReplayResult,
    ReplayRunConfig,
    run_replay,
)
from cpab.replay.parity import (
    INPUT_DIVERGENCE,
    ParityComparison,
    compare_replay_runs,
)
from cpab.replay.stalled import (
    CONTINUITY_DIAGNOSTIC,
    STALL_CLEARED,
    STALLED,
    StallReport,
    classify_stalled,
)

__all__ = [
    "CycleRecord",
    "ReplayResult",
    "ReplayRunConfig",
    "run_replay",
    "INPUT_DIVERGENCE",
    "ParityComparison",
    "compare_replay_runs",
    "CONTINUITY_DIAGNOSTIC",
    "STALL_CLEARED",
    "STALLED",
    "StallReport",
    "classify_stalled",
]
