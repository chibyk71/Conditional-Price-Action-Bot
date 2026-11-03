"""Phase 8B replay, parity, and report-only STALLED classification."""

from cpab.replay.live_style import LiveStyleConfig, live_style_from_replay_config, run_live_style
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
    "LiveStyleConfig",
    "run_live_style",
    "live_style_from_replay_config",
    "INPUT_DIVERGENCE",
    "ParityComparison",
    "compare_replay_runs",
    "CONTINUITY_DIAGNOSTIC",
    "STALL_CLEARED",
    "STALLED",
    "StallReport",
    "classify_stalled",
]
