"""Phase 7B: deterministic evaluator for Phase 7A AND/OR/THEN expressions.

Consumes validated Expression trees and observations. Does not manage
setup-instance lifecycle, invalidation, expiry, or signal emission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Optional, Sequence, Union

from cpab.conditions.expression import (
    AndExpression,
    Expression,
    ObservationRequirement,
    OrExpression,
    ThenExpression,
    ThenStep,
)
from cpab.models import Observation


@dataclass(frozen=True)
class EvidenceItem:
    """One observation selected as supporting evidence for a requirement or branch."""

    observation_id: str
    observation_type: str
    confirmation_time: Optional[int]
    candle_epoch: int
    is_activation: bool = False

    @staticmethod
    def from_observation(obs: Observation, *, evaluation_time: Optional[int]) -> "EvidenceItem":
        conf = obs.effective_confirmed_at()
        is_act = False
        if evaluation_time is not None:
            is_act = ExpressionEvaluator._is_activation_observation(obs, evaluation_time)
        return EvidenceItem(
            observation_id=obs.id,
            observation_type=obs.type,
            confirmation_time=conf,
            candle_epoch=obs.candle_epoch,
            is_activation=is_act,
        )


@dataclass(frozen=True)
class BranchProvenance:
    """Provenance for a single child branch or leaf satisfaction."""

    path: str
    qualified: bool
    evidence: tuple[EvidenceItem, ...] = ()
    child_provenances: tuple["BranchProvenance", ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ThenStepProvenance:
    """Committed evidence for one completed THEN step."""

    step_index: int
    evaluation_candle: int
    confirmation_time: Optional[int]
    evidence: tuple[EvidenceItem, ...]
    branch: Optional[BranchProvenance] = None


@dataclass(frozen=True)
class ThenProgress:
    """Explicit, minimal caller-supplied / evaluator-returned THEN sequence state.

    Pure evaluator state — no persistence or lifecycle management.
    """

    next_step_index: int = 0
    committed_steps: tuple[ThenStepProvenance, ...] = ()
    last_qualifying_evaluation_candle: Optional[int] = None
    last_qualifying_confirmation_time: Optional[int] = None
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_step_index": self.next_step_index,
            "committed_steps": [
                {
                    "step_index": s.step_index,
                    "evaluation_candle": s.evaluation_candle,
                    "confirmation_time": s.confirmation_time,
                    "evidence": [
                        {
                            "observation_id": e.observation_id,
                            "observation_type": e.observation_type,
                            "confirmation_time": e.confirmation_time,
                            "candle_epoch": e.candle_epoch,
                            "is_activation": e.is_activation,
                        }
                        for e in s.evidence
                    ],
                }
                for s in self.committed_steps
            ],
            "last_qualifying_evaluation_candle": self.last_qualifying_evaluation_candle,
            "last_qualifying_confirmation_time": self.last_qualifying_confirmation_time,
            "completed": self.completed,
        }


@dataclass(frozen=True)
class ExpressionEvalResult:
    """Deterministic result of evaluating an expression."""

    qualified: bool
    reasons: tuple[str, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    provenance: Optional[BranchProvenance] = None
    progress: Optional[ThenProgress] = None
    # True when this evaluation newly advanced or completed a THEN step
    advanced: bool = False
    symbol: str = ""
    timeframe: str = ""
    evaluation_time: Optional[int] = None
    candle_epoch: Optional[int] = None


class ExpressionEvaluator:
    """Evaluate Phase 7A Expression trees against observations.

    Leaf rules (Phase 7B contract):
      - Exact observation-type match; variant only when the requirement specifies it.
      - Symbol and timeframe from the evaluation context.
      - min_count uses distinct observation IDs.
      - Developing observations never qualify as evidence (allow_developing is
        preserved on the schema but ignored for qualification).
      - Causal eligibility at evaluation time T; completed-candle evidence only.
      - No future / forming-candle evidence.

    AND / OR:
      - AND: all children qualify in the same context; prior eligible evidence
        may contribute; prior-only evidence does not cause repeated activation.
      - OR: first qualifying child in declared order wins.
      - Evidence selection: earliest confirmation time, then stable observation id.

    THEN:
      - Caller supplies / receives explicit ThenProgress.
      - Advance only the next pending step; never skip.
      - First qualifying evidence for a step is committed permanently.
      - Step advances only when its expression qualifies with evidence associated
        with the current evaluation event.
      - Default progression requires a strictly later evaluation candle;
        allow_same_candle=True opts later steps into same-candle progression.
      - Confirmation times are non-decreasing across committed steps.
      - No automatic restart or re-arm of completed sequences.
    """

    def evaluate(
        self,
        expression: Expression,
        observations: Iterable[Observation],
        *,
        symbol: str,
        timeframe: str,
        candle_epoch: Optional[int] = None,
        evaluation_time: Optional[int] = None,
        progress: Optional[ThenProgress] = None,
    ) -> ExpressionEvalResult:
        T = evaluation_time if evaluation_time is not None else candle_epoch
        obs_list = [
            o
            for o in observations
            if o.symbol == symbol and o.timeframe == timeframe
        ]
        # Deterministic order independent of arrival: by confirmation time, id
        obs_list = sorted(
            obs_list,
            key=lambda o: (
                o.effective_confirmed_at() is None,
                o.effective_confirmed_at() if o.effective_confirmed_at() is not None else 0,
                o.id,
            ),
        )

        if isinstance(expression, ThenExpression):
            return self._eval_then(
                expression,
                obs_list,
                symbol=symbol,
                timeframe=timeframe,
                T=T,
                candle_epoch=candle_epoch,
                progress=progress or ThenProgress(),
            )

        # Non-THEN expressions require a current evaluation event so prior
        # confirmed observations alone cannot activate the condition repeatedly.
        # Fail closed when both evaluation_time and candle_epoch are omitted.
        if T is None:
            reasons = (
                "evaluation context required (evaluation_time or candle_epoch); "
                "prior evidence alone cannot activate leaf/AND/OR without a "
                "current evaluation event",
            )
            return ExpressionEvalResult(
                qualified=False,
                reasons=reasons,
                evidence=(),
                provenance=BranchProvenance(
                    path="expression",
                    qualified=False,
                    reasons=reasons,
                ),
                progress=None,
                advanced=False,
                symbol=symbol,
                timeframe=timeframe,
                evaluation_time=None,
                candle_epoch=candle_epoch,
            )

        prov, qualified, evidence, reasons = self._eval_node(
            expression,
            obs_list,
            path="expression",
            T=T,
            require_activation=True,
            committed_ids=frozenset(),
        )
        return ExpressionEvalResult(
            qualified=qualified,
            reasons=tuple(reasons),
            evidence=tuple(evidence),
            provenance=prov,
            progress=None,
            advanced=False,
            symbol=symbol,
            timeframe=timeframe,
            evaluation_time=T,
            candle_epoch=candle_epoch,
        )

    def _eval_node(
        self,
        expression: Expression,
        obs_list: Sequence[Observation],
        *,
        path: str,
        T: Optional[int],
        require_activation: bool,
        committed_ids: frozenset[str],
        prefer_activation: bool = False,
    ) -> tuple[BranchProvenance, bool, list[EvidenceItem], list[str]]:
        if isinstance(expression, ObservationRequirement):
            return self._eval_leaf(
                expression,
                obs_list,
                path=path,
                T=T,
                require_activation=require_activation,
                committed_ids=committed_ids,
                prefer_activation=prefer_activation or require_activation,
            )
        if isinstance(expression, AndExpression):
            return self._eval_and(
                expression,
                obs_list,
                path=path,
                T=T,
                require_activation=require_activation,
                committed_ids=committed_ids,
                prefer_activation=prefer_activation or require_activation,
            )
        if isinstance(expression, OrExpression):
            return self._eval_or(
                expression,
                obs_list,
                path=path,
                T=T,
                require_activation=require_activation,
                committed_ids=committed_ids,
                prefer_activation=prefer_activation or require_activation,
            )
        if isinstance(expression, ThenExpression):
            # Nested THEN is not expected as a child of AND/OR in 7A schema usage,
            # but handle defensively: evaluate without mutating external progress.
            # Nested THEN without progress cannot advance; treat as not qualified.
            reasons = [f"{path}: nested THEN is not evaluable without dedicated progress"]
            prov = BranchProvenance(path=path, qualified=False, reasons=tuple(reasons))
            return prov, False, [], reasons
        reasons = [f"{path}: unknown expression type {type(expression).__name__}"]
        return BranchProvenance(path=path, qualified=False, reasons=tuple(reasons)), False, [], reasons

    def _eval_leaf(
        self,
        req: ObservationRequirement,
        obs_list: Sequence[Observation],
        *,
        path: str,
        T: Optional[int],
        require_activation: bool,
        committed_ids: frozenset[str],
        prefer_activation: bool = False,
    ) -> tuple[BranchProvenance, bool, list[EvidenceItem], list[str]]:
        reasons: list[str] = []
        # Exact type match; variant only when specified.
        candidates = [
            o
            for o in obs_list
            if o.type == req.type
            and (req.variant is None or o.variant == req.variant)
            and o.id not in committed_ids
        ]

        # Phase 7B: developing observations never qualify as evidence.
        # allow_developing is preserved on the schema but does not admit developing evidence.
        eligible: list[Observation] = []
        for o in candidates:
            if o.is_developing():
                continue
            if T is not None:
                if not o.is_eligible_at(T, allow_developing=False):
                    continue
            else:
                # No evaluation time: only confirmed (non-developing) observations.
                if not o.is_confirmed():
                    continue
            eligible.append(o)

        # Deduplicate by observation id so the same id cannot satisfy min_count twice.
        # Sort: earliest confirmation time, then stable id (deterministic).
        eligible = sorted(
            eligible,
            key=lambda o: (
                o.effective_confirmed_at() if o.effective_confirmed_at() is not None else 2**62,
                o.id,
            ),
        )
        seen_ids: set[str] = set()
        unique_eligible: list[Observation] = []
        for o in eligible:
            if o.id in seen_ids:
                continue
            seen_ids.add(o.id)
            unique_eligible.append(o)
        eligible = unique_eligible

        if len(eligible) < req.min_count:
            desc = f"type={req.type}"
            if req.variant is not None:
                desc += f" variant={req.variant}"
            reasons.append(
                f"{path}: missing {desc} need>={req.min_count} got={len(eligible)} "
                f"(eligible at T={T}; developing rejected)"
            )
            prov = BranchProvenance(path=path, qualified=False, reasons=tuple(reasons))
            return prov, False, [], reasons

        # When activation is preferred or required, seed with an activation observation
        # if any exist so new evidence can activate while prior evidence contributes.
        activations = (
            [o for o in eligible if self._is_activation_observation(o, T)]
            if (prefer_activation or require_activation) and T is not None
            else []
        )
        if require_activation and T is not None and not activations:
            desc = f"type={req.type}"
            if req.variant is not None:
                desc += f" variant={req.variant}"
            prior_ev = [
                EvidenceItem.from_observation(o, evaluation_time=T)
                for o in eligible[: req.min_count]
            ]
            reasons.append(
                f"{path}: {desc} satisfied by prior evidence only; "
                f"no activation observation at T={T}"
            )
            prov = BranchProvenance(
                path=path,
                qualified=False,
                evidence=tuple(prior_ev),
                reasons=tuple(reasons),
            )
            return prov, False, prior_ev, reasons

        if activations:
            chosen = [activations[0]]
            chosen_ids = {activations[0].id}
            for o in eligible:
                if len(chosen) >= req.min_count:
                    break
                if o.id in chosen_ids:
                    continue
                chosen.append(o)
                chosen_ids.add(o.id)
            if len(chosen) < req.min_count:
                desc = f"type={req.type}"
                if req.variant is not None:
                    desc += f" variant={req.variant}"
                reasons.append(
                    f"{path}: missing {desc} need>={req.min_count} got={len(chosen)} "
                    f"after activation seed"
                )
                prov = BranchProvenance(path=path, qualified=False, reasons=tuple(reasons))
                return prov, False, [], reasons
        else:
            chosen = eligible[: req.min_count]

        evidence = [EvidenceItem.from_observation(o, evaluation_time=T) for o in chosen]

        if require_activation and T is not None:
            if not any(self._is_activation_observation(o, T) for o in chosen):
                desc = f"type={req.type}"
                if req.variant is not None:
                    desc += f" variant={req.variant}"
                reasons.append(
                    f"{path}: {desc} satisfied by prior evidence only; "
                    f"no activation observation at T={T}"
                )
                prov = BranchProvenance(
                    path=path,
                    qualified=False,
                    evidence=tuple(evidence),
                    reasons=tuple(reasons),
                )
                return prov, False, evidence, reasons

        desc = f"type={req.type}"
        if req.variant is not None:
            desc += f" variant={req.variant}"
        reasons.append(f"{path}: {desc} satisfied count={len(chosen)}")
        prov = BranchProvenance(
            path=path,
            qualified=True,
            evidence=tuple(evidence),
            reasons=tuple(reasons),
        )
        return prov, True, evidence, reasons

    def _eligible_observations(
        self,
        req: ObservationRequirement,
        obs_list: Sequence[Observation],
        *,
        T: Optional[int],
        committed_ids: frozenset[str],
    ) -> list[Observation]:
        """All distinct eligible observations for a leaf requirement (no selection)."""
        candidates = [
            o
            for o in obs_list
            if o.type == req.type
            and (req.variant is None or o.variant == req.variant)
            and o.id not in committed_ids
        ]
        eligible: list[Observation] = []
        for o in candidates:
            if o.is_developing():
                continue
            if T is not None:
                if not o.is_eligible_at(T, allow_developing=False):
                    continue
            elif not o.is_confirmed():
                continue
            eligible.append(o)
        eligible = sorted(
            eligible,
            key=lambda o: (
                o.effective_confirmed_at() if o.effective_confirmed_at() is not None else 2**62,
                o.id,
            ),
        )
        seen: set[str] = set()
        unique: list[Observation] = []
        for o in eligible:
            if o.id in seen:
                continue
            seen.add(o.id)
            unique.append(o)
        return unique

    def _leaf_choice_sets(
        self,
        req: ObservationRequirement,
        obs_list: Sequence[Observation],
        *,
        T: Optional[int],
        used_ids: frozenset[str],
        prefer_activation: bool,
    ) -> list[tuple[Observation, ...]]:
        """Deterministic candidate evidence sets of size min_count for a leaf.

        When prefer_activation is set and activation observations exist, sets that
        include at least one activation are tried first; all combinations remain
        available so backtracking can recover a globally valid assignment.
        """
        eligible = self._eligible_observations(
            req, obs_list, T=T, committed_ids=used_ids
        )
        if len(eligible) < req.min_count:
            return []
        act_ids = set()
        if prefer_activation and T is not None:
            act_ids = {
                o.id for o in eligible if self._is_activation_observation(o, T)
            }
        combos = list(combinations(eligible, req.min_count))

        def _combo_key(combo: tuple[Observation, ...]) -> tuple:
            has_act = 0 if any(o.id in act_ids for o in combo) else 1
            return (has_act, tuple(o.id for o in combo))

        combos.sort(key=_combo_key)
        return combos

    def _flatten_evidence(self, prov: BranchProvenance) -> list[EvidenceItem]:
        """Flatten evidence from a provenance tree in stable depth-first order."""
        out: list[EvidenceItem] = []
        if prov.child_provenances:
            for child in prov.child_provenances:
                out.extend(self._flatten_evidence(child))
        else:
            out.extend(prov.evidence)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[EvidenceItem] = []
        for e in out:
            if e.observation_id not in seen:
                seen.add(e.observation_id)
                deduped.append(e)
        return deduped

    def _iter_assignments(
        self,
        expression: Expression,
        obs_list: Sequence[Observation],
        *,
        path: str,
        T: Optional[int],
        used_ids: frozenset[str],
        prefer_activation: bool,
    ):
        """Yield BranchProvenance for every valid assignment of an expression subtree.

        Nested AND assignments are fully enumerated so an outer sibling can force
        an alternative when the first nested choice conflicts. OR children are
        tried in declared order; assignments from earlier branches are yielded
        before later ones (first global success prefers earlier branches).
        Successful OR provenance records the selected branch index in path and
        child_provenances.
        """
        if isinstance(expression, ObservationRequirement):
            for combo in self._leaf_choice_sets(
                expression,
                obs_list,
                T=T,
                used_ids=used_ids,
                prefer_activation=prefer_activation,
            ):
                ev = [
                    EvidenceItem.from_observation(o, evaluation_time=T) for o in combo
                ]
                ids = ",".join(e.observation_id for e in ev)
                yield BranchProvenance(
                    path=path,
                    qualified=True,
                    evidence=tuple(ev),
                    reasons=(f"{path}: assigned evidence [{ids}]",),
                )
            return

        if isinstance(expression, AndExpression):
            yield from self._iter_and_assignments(
                expression.children,
                obs_list,
                path=path,
                T=T,
                used_ids=used_ids,
                prefer_activation=prefer_activation,
            )
            return

        if isinstance(expression, OrExpression):
            # Declared order: exhaust earlier branches' assignments before later
            # ones so the first successful global allocation prefers earlier
            # branches without forbidding later branches when earlier ones cannot
            # complete a joint outer AND.
            for i, child in enumerate(expression.children):
                child_path = f"{path}.or[{i}]"
                for child_prov in self._iter_assignments(
                    child,
                    obs_list,
                    path=child_path,
                    T=T,
                    used_ids=used_ids,
                    prefer_activation=prefer_activation,
                ):
                    yield BranchProvenance(
                        path=path,
                        qualified=True,
                        evidence=child_prov.evidence,
                        child_provenances=(child_prov,),
                        reasons=(f"{path}: OR satisfied by child[{i}]",)
                        + child_prov.reasons,
                    )
            return

        # Nested THEN unsupported under AND/OR
        return

    def _child_rank_key(
        self,
        index: int,
        child: Expression,
        obs_list: Sequence[Observation],
        *,
        T: Optional[int],
        used_ids: frozenset[str],
    ) -> tuple:
        """Order-independent constraint ranking for AND children."""
        if isinstance(child, ObservationRequirement):
            n_elig = len(
                self._eligible_observations(
                    child, obs_list, T=T, committed_ids=used_ids
                )
            )
            specific = 0 if child.variant is not None else 1
            return (
                specific,
                n_elig,
                child.min_count,
                child.type,
                child.variant or "",
                index,
            )
        if isinstance(child, AndExpression):
            # Nested AND is structurally constrained — rank before broad leaves.
            return (-1, 0, len(child.children), "AndExpression", "", index)
        if isinstance(child, OrExpression):
            return (0, 0, len(child.children), "OrExpression", "", index)
        return (9, 0, 0, type(child).__name__, "", index)

    def _iter_and_assignments(
        self,
        children: Sequence[Expression],
        obs_list: Sequence[Observation],
        *,
        path: str,
        T: Optional[int],
        used_ids: frozenset[str],
        prefer_activation: bool,
    ):
        """Yield BranchProvenance for each valid AND assignment.

        Search order is ranked by constraint tightness (not declaration order).
        child_provenances on the result are always in declared child order.
        Nested composites yield alternative assignments so outer siblings can
        force reconsideration when the first nested choice conflicts.
        """
        n = len(children)
        order = sorted(
            range(n),
            key=lambda i: self._child_rank_key(
                i, children[i], obs_list, T=T, used_ids=used_ids
            ),
        )
        assignment: dict[int, BranchProvenance] = {}

        def search(pos: int, current_used: frozenset[str]):
            if pos >= len(order):
                child_provs = tuple(assignment[i] for i in range(n))
                flat: list[EvidenceItem] = []
                for cp in child_provs:
                    flat.extend(self._flatten_evidence(cp))
                seen: set[str] = set()
                deduped: list[EvidenceItem] = []
                for e in flat:
                    if e.observation_id not in seen:
                        seen.add(e.observation_id)
                        deduped.append(e)
                yield BranchProvenance(
                    path=path,
                    qualified=True,
                    evidence=tuple(deduped),
                    child_provenances=child_provs,
                    reasons=(f"{path}: AND satisfied",),
                )
                return
            i = order[pos]
            child = children[i]
            child_path = f"{path}.and[{i}]"

            for child_prov in self._iter_assignments(
                child,
                obs_list,
                path=child_path,
                T=T,
                used_ids=current_used,
                prefer_activation=prefer_activation,
            ):
                assignment[i] = child_prov
                new_used = current_used | {
                    e.observation_id for e in self._flatten_evidence(child_prov)
                }
                yield from search(pos + 1, frozenset(new_used))
                del assignment[i]

        yield from search(0, used_ids)

    def _assign_and_children(
        self,
        children: Sequence[Expression],
        obs_list: Sequence[Observation],
        *,
        path: str,
        T: Optional[int],
        used_ids: frozenset[str],
        prefer_activation: bool,
        require_activation: bool = False,
    ) -> Optional[BranchProvenance]:
        """Return the first valid AND provenance satisfying activation when required.

        Continues through generator alternatives so a prior-only first assignment
        does not block a later assignment that includes current-event evidence.
        Among complete (distinct + activation) solutions, declared OR order is
        preserved because earlier OR branches are yielded first.
        """
        need_activation = require_activation and T is not None
        for prov in self._iter_and_assignments(
            children,
            obs_list,
            path=path,
            T=T,
            used_ids=used_ids,
            prefer_activation=prefer_activation,
        ):
            if need_activation and not any(e.is_activation for e in prov.evidence):
                continue
            return prov
        return None

    def _eval_and(
        self,
        expr: AndExpression,
        obs_list: Sequence[Observation],
        *,
        path: str,
        T: Optional[int],
        require_activation: bool,
        committed_ids: frozenset[str],
        prefer_activation: bool = False,
    ) -> tuple[BranchProvenance, bool, list[EvidenceItem], list[str]]:
        """Evaluate AND with backtracking distinct-evidence allocation.

        Finds a valid assignment of distinct observations across children
        (including nested AND/OR) when one exists. When activation is required,
        continues search until an assignment includes current-event evidence.
        Successful provenance includes child_provenances in declared order,
        including which OR branch was selected under nested OR children.
        """
        prefer = (require_activation or prefer_activation) and T is not None
        assigned_prov = self._assign_and_children(
            expr.children,
            obs_list,
            path=path,
            T=T,
            used_ids=committed_ids,
            prefer_activation=prefer,
            require_activation=require_activation,
        )

        if assigned_prov is None:
            # Diagnostic: independent eligibility failure messages
            reasons: list[str] = []
            child_provs: list[BranchProvenance] = []
            for i, child in enumerate(expr.children):
                child_path = f"{path}.and[{i}]"
                c_prov, c_ok, c_ev, c_reasons = self._eval_node(
                    child,
                    obs_list,
                    path=child_path,
                    T=T,
                    require_activation=False,
                    committed_ids=committed_ids,
                    prefer_activation=prefer,
                )
                child_provs.append(c_prov)
                reasons.extend(c_reasons)
            if require_activation and T is not None:
                reasons.append(
                    f"{path}: AND not satisfied (no valid assignment with "
                    f"activation evidence at T={T})"
                )
            else:
                reasons.append(
                    f"{path}: AND not satisfied (no valid distinct-evidence assignment)"
                )
            prov = BranchProvenance(
                path=path,
                qualified=False,
                child_provenances=tuple(child_provs),
                reasons=tuple(reasons),
            )
            return prov, False, [], reasons

        evidence = list(assigned_prov.evidence)
        return assigned_prov, True, evidence, list(assigned_prov.reasons)

    def _eval_or(
        self,
        expr: OrExpression,
        obs_list: Sequence[Observation],
        *,
        path: str,
        T: Optional[int],
        require_activation: bool,
        committed_ids: frozenset[str],
        prefer_activation: bool = False,
    ) -> tuple[BranchProvenance, bool, list[EvidenceItem], list[str]]:
        child_provs: list[BranchProvenance] = []
        all_reasons: list[str] = []
        # Propagate activation preference into nested children (e.g. AND under OR
        # under outer AND) without changing declared-order first-match semantics.
        prefer = prefer_activation or require_activation

        # Declared child order: first qualifying branch wins.
        for i, child in enumerate(expr.children):
            child_path = f"{path}.or[{i}]"
            c_prov, c_ok, c_ev, c_reasons = self._eval_node(
                child,
                obs_list,
                path=child_path,
                T=T,
                require_activation=require_activation,
                committed_ids=committed_ids,
                prefer_activation=prefer,
            )
            child_provs.append(c_prov)
            all_reasons.extend(c_reasons)
            if c_ok:
                reasons = all_reasons + [f"{path}: OR satisfied by child[{i}]"]
                prov = BranchProvenance(
                    path=path,
                    qualified=True,
                    evidence=tuple(c_ev),
                    child_provenances=tuple(child_provs),
                    reasons=tuple(reasons),
                )
                return prov, True, list(c_ev), reasons

        reasons = all_reasons + [f"{path}: OR not satisfied (no child qualified)"]
        prov = BranchProvenance(
            path=path,
            qualified=False,
            child_provenances=tuple(child_provs),
            reasons=tuple(reasons),
        )
        return prov, False, [], reasons

    def _eval_then(
        self,
        expr: ThenExpression,
        obs_list: Sequence[Observation],
        *,
        symbol: str,
        timeframe: str,
        T: Optional[int],
        candle_epoch: Optional[int],
        progress: ThenProgress,
    ) -> ExpressionEvalResult:
        steps = expr.steps
        if progress.completed or progress.next_step_index >= len(steps):
            # Idempotent: re-evaluation of completed sequence preserves state.
            return ExpressionEvalResult(
                qualified=True,
                reasons=("THEN sequence already completed",),
                evidence=tuple(
                    e for s in progress.committed_steps for e in s.evidence
                ),
                provenance=None,
                progress=progress,
                advanced=False,
                symbol=symbol,
                timeframe=timeframe,
                evaluation_time=T,
                candle_epoch=candle_epoch,
            )

        step_idx = progress.next_step_index
        step = steps[step_idx]
        reasons: list[str] = []

        # A THEN step may advance only when the caller supplies enough
        # current-event context to verify activation and record a valid
        # evaluation candle. Omitting both evaluation_time and candle_epoch
        # must not allow prior-only evidence to advance the sequence.
        eval_candle = candle_epoch if candle_epoch is not None else T
        if T is None or eval_candle is None:
            reasons.append(
                f"THEN step[{step_idx}]: evaluation context required "
                f"(evaluation_time or candle_epoch); cannot advance without "
                f"a current evaluation event"
            )
            return ExpressionEvalResult(
                qualified=False,
                reasons=tuple(reasons),
                progress=progress,
                advanced=False,
                symbol=symbol,
                timeframe=timeframe,
                evaluation_time=T,
                candle_epoch=candle_epoch,
            )

        # Candle progression constraint (steps after the first).
        if step_idx > 0 and progress.last_qualifying_evaluation_candle is not None:
            last_candle = progress.last_qualifying_evaluation_candle
            if not step.allow_same_candle:
                if eval_candle <= last_candle:
                    reasons.append(
                        f"THEN step[{step_idx}]: requires strictly later evaluation "
                        f"candle than {last_candle}; got {eval_candle}"
                    )
                    return ExpressionEvalResult(
                        qualified=False,
                        reasons=tuple(reasons),
                        progress=progress,
                        advanced=False,
                        symbol=symbol,
                        timeframe=timeframe,
                        evaluation_time=T,
                        candle_epoch=candle_epoch,
                    )
            else:
                if eval_candle < last_candle:
                    reasons.append(
                        f"THEN step[{step_idx}]: evaluation candle {eval_candle} "
                        f"is before last qualifying candle {last_candle}"
                    )
                    return ExpressionEvalResult(
                        qualified=False,
                        reasons=tuple(reasons),
                        progress=progress,
                        advanced=False,
                        symbol=symbol,
                        timeframe=timeframe,
                        evaluation_time=T,
                        candle_epoch=candle_epoch,
                    )

        # Committed observation ids must not be reused to satisfy a later step.
        committed_ids = frozenset(
            e.observation_id for s in progress.committed_steps for e in s.evidence
        )

        # Pending step must qualify with evidence associated with the current event.
        # require_activation=True ensures prior-only evidence cannot advance the step.
        prov, ok, evidence, node_reasons = self._eval_node(
            step.expression,
            obs_list,
            path=f"then.steps[{step_idx}]",
            T=T,
            require_activation=(T is not None),
            committed_ids=committed_ids,
        )
        reasons.extend(node_reasons)

        if not ok:
            return ExpressionEvalResult(
                qualified=False,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                provenance=prov,
                progress=progress,
                advanced=False,
                symbol=symbol,
                timeframe=timeframe,
                evaluation_time=T,
                candle_epoch=candle_epoch,
            )

        # Confirmation times must be non-decreasing across committed steps.
        step_conf = self._max_confirmation(evidence)
        if (
            progress.last_qualifying_confirmation_time is not None
            and step_conf is not None
            and step_conf < progress.last_qualifying_confirmation_time
        ):
            reasons.append(
                f"THEN step[{step_idx}]: confirmation time {step_conf} is before "
                f"last committed confirmation {progress.last_qualifying_confirmation_time}"
            )
            return ExpressionEvalResult(
                qualified=False,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                provenance=prov,
                progress=progress,
                advanced=False,
                symbol=symbol,
                timeframe=timeframe,
                evaluation_time=T,
                candle_epoch=candle_epoch,
            )

        # eval_candle was validated non-None above
        step_prov = ThenStepProvenance(
            step_index=step_idx,
            evaluation_candle=eval_candle,
            confirmation_time=step_conf,
            evidence=tuple(evidence),
            branch=prov,
        )
        new_committed = progress.committed_steps + (step_prov,)
        next_idx = step_idx + 1
        completed = next_idx >= len(steps)
        new_progress = ThenProgress(
            next_step_index=next_idx if not completed else next_idx,
            committed_steps=new_committed,
            last_qualifying_evaluation_candle=eval_candle,
            last_qualifying_confirmation_time=(
                step_conf
                if step_conf is not None
                else progress.last_qualifying_confirmation_time
            ),
            completed=completed,
        )

        if completed:
            reasons.append(f"THEN sequence completed at step[{step_idx}]")
        else:
            reasons.append(f"THEN advanced to step[{next_idx}]")

        return ExpressionEvalResult(
            qualified=completed,  # overall sequence qualifies only when complete
            reasons=tuple(reasons),
            evidence=tuple(e for s in new_committed for e in s.evidence),
            provenance=prov,
            progress=new_progress,
            advanced=True,
            symbol=symbol,
            timeframe=timeframe,
            evaluation_time=T,
            candle_epoch=candle_epoch,
        )

    @staticmethod
    def _max_confirmation(evidence: Sequence[EvidenceItem]) -> Optional[int]:
        times = [e.confirmation_time for e in evidence if e.confirmation_time is not None]
        if not times:
            return None
        return max(times)

    @staticmethod
    def _is_activation_observation(obs: Observation, T: int) -> bool:
        """True if observation newly associates with evaluation bar T.

        Confirmed: effective_confirmed_at == T.
        Developing is never activation evidence in Phase 7B.
        """
        if obs.is_developing():
            return False
        conf = obs.effective_confirmed_at()
        return conf is not None and conf == T
