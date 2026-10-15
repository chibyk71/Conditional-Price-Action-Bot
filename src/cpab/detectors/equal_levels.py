"""
Phase 2B — Equal Highs / Equal Lows (EQH / EQL) detection.

Reuses SwingPivotDetector for confirmed swing highs and lows. Groups two or
more distinct confirmed swings of the same direction whose prices satisfy an
absolute inclusive tolerance ε:

    |P1 - P2| <= ε

Configuration (all participate in detector_config_id):
  - tolerance: absolute price ε (finite, >= 0; required; no silent default)
  - left_strength, right_strength, tie_policy: forwarded to internal swing detector

Level creation is causal: the level is only confirmed when the *second*
supporting swing is confirmed (confirmed_at = that swing's confirmed_at).

Representative price at creation = median of the two supporting prices
(midpoint for two values). Immutable thereafter.

Subsequent qualifying swings emit separate touch observations; they do not
mutate the creation observation or the representative price.

Grouping is deterministic (nearest representative / nearest prior swing,
then earliest, then stable id). Tolerance chaining is prevented by comparing
new swings only against a level's fixed representative price.

Event types:
  - equal_high_level / equal_low_level          (creation)
  - equal_high_level_touch / equal_low_level_touch

direction is always null (structural, consistent with Phase 2A swings).

Historical window: same as the internal SwingPivotDetector's min_history for
the first confirmable swing, plus enough additional closed bars for a second
independently confirmable swing. In practice the detector reconstructs from
the full closed series available to it; if the original supporting swings
cannot be reconstructed, level creation is skipped (no invented identity).

Invalidation, lifecycle, breakouts, and signal generation are out of scope.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from cpab.detectors.base import PriceActionDetector
from cpab.detectors.swing import SwingPivotDetector, _validate_strength, _validate_tie_policy
from cpab.logging_setup import get_logger
from cpab.models import Candle, Observation

logger = get_logger("cpab.detectors.equal_levels")


def _validate_tolerance(value: Any) -> float:
    """Accept only finite nonnegative real numbers; reject bool, str, NaN, Inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"tolerance must be a finite nonnegative number, got {value!r}")
    f = float(value)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(f"tolerance must be a finite nonnegative number, got {value!r}")
    return f


def _median_two(a: float, b: float) -> float:
    return (a + b) / 2.0


def _stable_level_id(
    *,
    kind: str,
    symbol: str,
    timeframe: str,
    detector_config_id: str,
    swing_id_a: str,
    swing_id_b: str,
) -> str:
    """Deterministic level identity from the two supporting swing observation ids."""
    a, b = sorted([swing_id_a, swing_id_b])
    parts = [
        "eqlvl",
        kind,
        symbol,
        timeframe,
        detector_config_id,
        a,
        b,
    ]
    return "|".join(parts)


@dataclass
class _Level:
    """In-memory reconstruction of an established equal level (not persisted)."""

    level_id: str
    kind: str  # "high" | "low"
    representative_price: float
    created_confirmed_at: int
    supporting_swing_ids: tuple[str, str]
    creation_obs_id: str
    touch_swing_ids: set[str] = field(default_factory=set)


class EqualLevelsDetector(PriceActionDetector):
    """
    Equal-high / equal-low detector built on SwingPivotDetector.

    Configuration is immutable after construction. Invalid tolerance or swing
    parameters raise ValueError (no silent defaults for tolerance).
    """

    def __init__(
        self,
        *,
        tolerance: float,
        left_strength: int = 1,
        right_strength: int = 1,
        tie_policy: str = "strict",
        detector_id: str = "equal_levels",
        detector_config_id: Optional[str] = None,
    ) -> None:
        self._tolerance = _validate_tolerance(tolerance)
        self._left = _validate_strength("left_strength", left_strength)
        self._right = _validate_strength("right_strength", right_strength)
        self._tie_policy = _validate_tie_policy(tie_policy)
        self._detector_id = detector_id

        self._swing = SwingPivotDetector(
            left_strength=self._left,
            right_strength=self._right,
            tie_policy=self._tie_policy,
            detector_id="swing_pivot",
        )

        cfg = {
            "tolerance": self._tolerance,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
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
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def left_strength(self) -> int:
        return self._left

    @property
    def right_strength(self) -> int:
        return self._right

    @property
    def tie_policy(self) -> str:
        return self._tie_policy

    @property
    def detector_config_id(self) -> str:
        return self._detector_config_id

    @property
    def min_history(self) -> int:
        """
        Minimum closed candles for any equal-level to become confirmable.

        Derivation: a single swing needs left + 1 + right closed bars. A second
        distinct swing needs at least one additional candidate bar after the
        first swing's confirmation window. The lower bound used here is
        2 * (left + 1 + right) - 1, which is the smallest series length that
        can contain two non-identical pivot candidates each with a full
        left/candidate/right window under contiguous spacing. Callers that
        need full historical reconstruction of older levels must supply the
        entire relevant closed series; insufficient history simply yields no
        emission rather than invented identities.
        """
        s = self._left + 1 + self._right
        return 2 * s - 1

    def config_dict(self) -> dict[str, Any]:
        return {
            "tolerance": self._tolerance,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }

    def detect(self, candles: list[Candle]) -> list[Observation]:
        """
        Incremental path: reconstruct all levels from closed history, then
        emit only observations whose confirmed_at equals the latest closed
        bar's open epoch (same latest-bar focus as SwingPivotDetector).
        """
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        all_obs = self._reconstruct(closed)
        latest_epoch = closed[-1].epoch
        return [o for o in all_obs if o.confirmed_at == latest_epoch]

    def detect_all_historical(self, candles: list[Candle]) -> list[Observation]:
        """Full historical reconstruction for tests / offline backfill."""
        closed = [c for c in candles if c.is_closed]
        if len(closed) < self.min_history:
            return []
        return self._reconstruct(closed)

    def _reconstruct(self, closed: list[Candle]) -> list[Observation]:
        if not closed:
            return []
        symbol = closed[-1].symbol
        timeframe = closed[-1].timeframe
        if any(c.symbol != symbol or c.timeframe != timeframe for c in closed):
            logger.debug("Mixed symbol/timeframe; skipping equal-level detect")
            return []

        # Reconstruct confirmed swings with the internal pivot detector.
        swing_obs = self._swing.detect_all_historical(closed)
        if len(swing_obs) < 2:
            return []

        # Process highs and lows independently.
        highs = [o for o in swing_obs if o.type == "swing_high"]
        lows = [o for o in swing_obs if o.type == "swing_low"]

        out: list[Observation] = []
        out.extend(self._process_direction(highs, kind="high", symbol=symbol, timeframe=timeframe))
        out.extend(self._process_direction(lows, kind="low", symbol=symbol, timeframe=timeframe))
        return out

    def _process_direction(
        self,
        swings: list[Observation],
        *,
        kind: str,
        symbol: str,
        timeframe: str,
    ) -> list[Observation]:
        if len(swings) < 2:
            return []

        # Deterministic order: confirmed_at, then event_time, then stable id.
        ordered = sorted(
            swings,
            key=lambda o: (
                o.confirmed_at if o.confirmed_at is not None else 0,
                o.event_time if o.event_time is not None else o.candle_epoch,
                o.id,
            ),
        )

        levels: list[_Level] = []
        assigned: set[str] = set()  # swing ids already used in a level or touch
        out: list[Observation] = []

        for swing in ordered:
            price = float(swing.payload.get("price", 0.0))
            sid = swing.id

            if sid in assigned:
                continue

            # 1) Try attach to existing level (fixed representative price).
            qualifiers: list[tuple[float, int, str, _Level]] = []
            for lvl in levels:
                dist = abs(price - lvl.representative_price)
                if dist <= self._tolerance:
                    qualifiers.append((dist, lvl.created_confirmed_at, lvl.level_id, lvl))

            if qualifiers:
                # nearest, then earliest-created, then lex-smallest level_id
                qualifiers.sort(key=lambda t: (t[0], t[1], t[2]))
                chosen = qualifiers[0][3]
                touch_obs = self._make_touch(
                    kind=kind,
                    level=chosen,
                    swing=swing,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                out.append(touch_obs)
                chosen.touch_swing_ids.add(sid)
                assigned.add(sid)
                continue

            # 2) No existing level — try pair with ungrouped prior swing.
            priors = [
                s
                for s in ordered
                if s.id not in assigned
                and s.id != sid
                and (
                    (s.confirmed_at if s.confirmed_at is not None else 0),
                    s.event_time if s.event_time is not None else s.candle_epoch,
                    s.id,
                )
                < (
                    swing.confirmed_at if swing.confirmed_at is not None else 0,
                    swing.event_time if swing.event_time is not None else swing.candle_epoch,
                    swing.id,
                )
            ]
            candidates: list[tuple[float, int, str, Observation]] = []
            for prior in priors:
                pprice = float(prior.payload.get("price", 0.0))
                dist = abs(price - pprice)
                if dist <= self._tolerance:
                    conf = prior.confirmed_at if prior.confirmed_at is not None else 0
                    candidates.append((dist, conf, prior.id, prior))

            if not candidates:
                continue

            candidates.sort(key=lambda t: (t[0], t[1], t[2]))
            prior = candidates[0][3]
            prior_price = float(prior.payload.get("price", 0.0))
            rep = _median_two(prior_price, price)
            level_id = _stable_level_id(
                kind=kind,
                symbol=symbol,
                timeframe=timeframe,
                detector_config_id=self._detector_config_id,
                swing_id_a=prior.id,
                swing_id_b=sid,
            )
            creation = self._make_creation(
                kind=kind,
                level_id=level_id,
                representative_price=rep,
                swing_a=prior,
                swing_b=swing,
                symbol=symbol,
                timeframe=timeframe,
            )
            out.append(creation)
            levels.append(
                _Level(
                    level_id=level_id,
                    kind=kind,
                    representative_price=rep,
                    created_confirmed_at=creation.confirmed_at or 0,
                    supporting_swing_ids=tuple(sorted([prior.id, sid])),
                    creation_obs_id=creation.id,
                )
            )
            assigned.add(prior.id)
            assigned.add(sid)

        return out

    def _make_creation(
        self,
        *,
        kind: str,
        level_id: str,
        representative_price: float,
        swing_a: Observation,
        swing_b: Observation,
        symbol: str,
        timeframe: str,
    ) -> Observation:
        # Second swing is the later-confirmed one (causal boundary).
        by_conf = sorted(
            [swing_a, swing_b],
            key=lambda o: (
                o.confirmed_at if o.confirmed_at is not None else 0,
                o.event_time if o.event_time is not None else o.candle_epoch,
                o.id,
            ),
        )
        first, second = by_conf[0], by_conf[1]
        ref_ids = tuple(sorted([first.id, second.id]))

        obs_type = "equal_high_level" if kind == "high" else "equal_low_level"
        payload: dict[str, Any] = {
            "level_id": level_id,
            "representative_price": representative_price,
            "tolerance": self._tolerance,
            "touch_count": 2,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
            "supporting_swing_ids": list(ref_ids),
        }
        return Observation.create(
            type=obs_type,
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=second.event_time if second.event_time is not None else second.candle_epoch,
            event_time=second.event_time if second.event_time is not None else second.candle_epoch,
            confirmed_at=second.confirmed_at,
            developing=False,
            direction=None,
            variant="equal_abs",
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=ref_ids,
            payload=payload,
            metadata=payload,
        )

    def _make_touch(
        self,
        *,
        kind: str,
        level: _Level,
        swing: Observation,
        symbol: str,
        timeframe: str,
    ) -> Observation:
        obs_type = "equal_high_level_touch" if kind == "high" else "equal_low_level_touch"
        price = float(swing.payload.get("price", 0.0))
        payload: dict[str, Any] = {
            "level_id": level.level_id,
            "representative_price": level.representative_price,
            "tolerance": self._tolerance,
            "swing_price": price,
            "swing_observation_id": swing.id,
            "left_strength": self._left,
            "right_strength": self._right,
            "tie_policy": self._tie_policy,
        }
        return Observation.create(
            type=obs_type,
            symbol=symbol,
            timeframe=timeframe,
            candle_epoch=swing.event_time if swing.event_time is not None else swing.candle_epoch,
            event_time=swing.event_time if swing.event_time is not None else swing.candle_epoch,
            confirmed_at=swing.confirmed_at,
            developing=False,
            direction=None,
            variant="equal_abs",
            detector_id=self._detector_id,
            detector_config_id=self._detector_config_id,
            reference_ids=(level.level_id, swing.id),
            payload=payload,
            metadata=payload,
        )
