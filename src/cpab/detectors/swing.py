"""
Phase 2A — Configurable left/right-strength pivot swing high / swing low detector.

Pivot definition (candidate candle C):
  - Swing high: C's high vs highs of left_strength preceding + right_strength following.
  - Swing low:  C's low  vs lows  of the same window.
  - Window must be consecutive, closed, valid candles with exact timeframe spacing;
    no gap bridging or interpolation.
  - Exact float equality only for plateaus (no epsilon / tolerance).

Tie policies:
  - strict: candidate extreme must be strictly greater (high) / lower (low) than every
    other extreme in the complete window.
  - first: equal extremes allowed; select only the earliest candle in a contiguous
    plateau of exactly equal extremes. Confirmation waits for full right-side data
    so the earliest representative is known before emission.
  - all: every independently qualifying candle emits its own observation (including ties).

Confirmation timestamps (Phase 0 reconciliation):
  - event_time / candle_epoch: candidate candle open epoch.
  - confirmed_at: open epoch of the final required right-side candle — the evaluation
    bar T at which the full window first became available.
  - detected_at: wall-clock emission time (Observation.create).

  Phase 0 conditions evaluate eligibility with effective_confirmed_at() <= T where T is
  the evaluation candle's open epoch. Using the confirming bar's open epoch is therefore
  the correct causal boundary: the swing is first eligible on the bar whose close
  completed the right-side evidence, and not on any earlier bar. It is never eligible
  while that confirming candle is still forming (detectors only receive is_closed=True
  candles). See docs/PHASE2A_SWING_DETECTION.md.

Developing / unconfirmed candidates are never emitted.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.swing")

TiePolicy = Literal["strict", "first", "all"]
_VALID_TIE_POLICIES = frozenset({"strict", "first", "all"})

# Common timeframe → seconds. Unknown labels reject the window rather than invent a step.
_TF_SECONDS: dict[str, int] = {
    "1m": 60,
    "m1": 60,
    "3m": 180,
    "5m": 300,
    "m5": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "h1": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
    "d1": 86400,
}


def timeframe_seconds(timeframe: str) -> Optional[int]:
    """
    Resolve a timeframe label to bar duration in seconds.

    Returns None when the label is not recognized. Callers that need a hard
    continuity check should reject the window rather than invent a step.
    """
    key = timeframe.strip().lower()
    if key in _TF_SECONDS:
        return _TF_SECONDS[key]
    m = re.fullmatch(r"(\d+)\s*s(?:ec(?:onds?)?)?", key)
    if m:
        return int(m.group(1))
    return None


def _validate_strength(name: str, value: Any) -> int:
    """Accept only real positive ints; reject bool, float, str, etc."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _validate_tie_policy(value: Any) -> TiePolicy:
    if not isinstance(value, str) or value not in _VALID_TIE_POLICIES:
        raise ValueError(
            f"tie_policy must be one of {sorted(_VALID_TIE_POLICIES)}, got {value!r}"
        )
    return value  # type: ignore[return-value]


class SwingPivotDetector(PriceActionDetector):
    """
    Left/right-strength pivot detector for swing highs and swing lows.

    Configuration is immutable after construction. Invalid strengths or tie_policy
    raise ValueError (no silent defaults or coercion for invalid values).

    Every instance has a deterministic detector_config_id derived from the effective
    configuration, whether constructed directly or via the registry.
    """

    def __init__(
        self,
        *,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "swing_pivot",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._left = _validate_strength("left_strength", left_strength)
        self._right = _validate_strength("right_strength", right_strength)
        self._tie_policy: TiePolicy = _validate_tie_policy(tie_policy)
        self._detector_id = detector_id
        cfg = {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        # Always derive a deterministic id from effective config when not supplied.
        # Inline hash matches registry.deterministic_config_id (avoid circular import).
        if detector_config_id is not None:
            self._detector_config_id = detector_config_id
        else:
            canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
            self._detector_config_id = f"cfg:{digest}"

    @property
    def name(self) -> str:
        return self._detector_id

    @property
    def left_strength(self) -> int:
        return self._left

    @property
    def right_strength(self) -> int:
        return self._right

    @property
    def tie_policy(self) -> TiePolicy:
        return self._tie_policy

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def min_history(self) -> int:
        """Minimum closed candles needed for any candidate to be evaluable."""
        return self._left + 1 + self._right

    def config_dict(self) -> dict[str, Any]:
        return {
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Evaluate completed candles only. Emits confirmed swing_high / swing_low
        observations for candidates whose full left/candidate/right window is present.

        Only candidates that become newly confirmable with the *latest* closed bar
        as the final right-side candle are emitted. This keeps incremental detect()
        aligned with engulfing/TA-Lib (latest-bar focus) and avoids re-emitting history.
        """
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []

        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe in candle list; skipping swing detect")
            return []

        step = timeframe_seconds(timeframe)
        cand_idx = len(closed) - 1 - self._right
        if cand_idx < self._left:
            return []

        window = closed[cand_idx - self._left : cand_idx + self._right + 1]
        if len(window) != self.min_history:
            return []

        if not self._window_consecutive(window, step=step):
            logger.debug(
                "Non-consecutive or gapped window for swing candidate epoch=%s; skip",
                closed[cand_idx].epoch,
            )
            return []

        candidate = closed[cand_idx]
        confirm_candle = closed[-1]

        out: list[Observation] = []
        for kind in ("high", "low"):
            if self._qualifies(window, cand_local_idx=self._left, kind=kind):
                out.append(
                    self._make_observation(
                        kind=kind,
                        candidate=candidate,
                        confirm_candle=confirm_candle,
                    )
                )
        return out

    def detect_all_historical(self, candles: list[Candle]) -> list[Observation]:
        """
        Evaluate every historically confirmable candidate in the series.
        Intended for tests and offline backfill; production path uses detect().
        """
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []

        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            return []

        step = timeframe_seconds(timeframe)
        out: list[Observation] = []
        for cand_idx in range(self._left, len(closed) - self._right):
            window = closed[cand_idx - self._left : cand_idx + self._right + 1]
            if len(window) != self.min_history:
                continue
            if not self._window_consecutive(window, step=step):
                continue
            candidate = closed[cand_idx]
            confirm_candle = closed[cand_idx + self._right]
            for kind in ("high", "low"):
                if self._qualifies(window, cand_local_idx=self._left, kind=kind):
                    out.append(
                        self._make_observation(
                            kind=kind,
                            candidate=candidate,
                            confirm_candle=confirm_candle,
                        )
                    )
        return out

    def _qualifies(
        self,
        window: list[Candle],
        *,
        cand_local_idx: int,
        kind: str,
    ) -> bool:
        """Return True if the candidate at cand_local_idx qualifies under tie_policy."""
        extremes = [c.high if kind == "high" else c.low for c in window]
        cand_ext = extremes[cand_local_idx]
        others = [extremes[i] for i in range(len(extremes)) if i != cand_local_idx]

        if self._tie_policy == "strict":
            if kind == "high":
                return all(cand_ext > e for e in others)
            return all(cand_ext < e for e in others)

        if self._tie_policy == "all":
            if kind == "high":
                return all(cand_ext >= e for e in others)
            return all(cand_ext <= e for e in others)

        if kind == "high":
            if not all(cand_ext >= e for e in others):
                return False
        else:
            if not all(cand_ext <= e for e in others):
                return False

        left = cand_local_idx
        while left > 0 and extremes[left - 1] == cand_ext:
            left -= 1
        return cand_local_idx == left

    @staticmethod
    def _window_consecutive(window: list[Candle], *, step: Optional[int]) -> bool:
        """
        Require consecutive candles with exact timeframe spacing.

        - Epochs must be strictly increasing by exactly ``step`` seconds.
        - When ``step`` is unknown, reject the window rather than assume spacing.
        """
        if len(window) < 2:
            return True
        if step is None or step <= 0:
            return False
        for i in range(1, len(window)):
            delta = window[i].epoch - window[i - 1].epoch
            if delta != step:
                return False
        return True

    def _make_observation(
        self,
        *,
        kind: str,
        candidate: Candle,
        confirm_candle: Candle,
    ) -> Observation:
        obs_type = "swing_high" if kind == "high" else "swing_low"
        price = candidate.high if kind == "high" else candidate.low
        payload: dict[str, Any] = {
            "price": price,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        return Observation.create(
            type=obs_type,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            candle_epoch=candidate.epoch,
            event_time=candidate.epoch,
            confirmed_at=confirm_candle.epoch,
            developing=False,
            direction=None,
            variant="pivot_lr",
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=(),
            payload=payload,
            metadata=payload,
        )
