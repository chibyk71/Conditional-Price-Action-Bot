"""8. Global readiness blocking evaluation when degraded."""

from cpab.market.readiness import ReadinessManager
from cpab.models import ReadinessState


def test_initializing_to_ready():
    rm = ReadinessManager()
    assert rm.state == ReadinessState.INITIALIZING
    assert not rm.can_evaluate()
    rm.register_series("R_100", "1m")
    rm.register_series("R_100", "5m")
    rm.set_series("R_100", "1m", True)
    assert rm.state == ReadinessState.INITIALIZING  # not all yet
    rm.set_series("R_100", "5m", True)
    assert rm.state == ReadinessState.READY
    assert rm.can_evaluate()


def test_ready_to_degraded():
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    rm.set_series("R_100", "1m", True)
    assert rm.state == ReadinessState.READY
    rm.set_series("R_100", "1m", False, "poll error")
    assert rm.state == ReadinessState.DEGRADED
    assert not rm.can_evaluate()


def test_degraded_to_recovering_to_ready():
    rm = ReadinessManager()
    rm.register_series("R_100", "1m")
    rm.set_series("R_100", "1m", True)
    rm.set_series("R_100", "1m", False)
    assert rm.state == ReadinessState.DEGRADED
    rm.set_series("R_100", "1m", True)
    assert rm.state == ReadinessState.RECOVERING
    # still need recompute — set again triggers READY
    rm.set_series("R_100", "1m", True, "stable")
    assert rm.state == ReadinessState.READY
