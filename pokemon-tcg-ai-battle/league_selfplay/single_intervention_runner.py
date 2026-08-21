from __future__ import annotations

import itertools
import hashlib
import json
import math
import os
import signal
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.run_local_match import import_official_cg, validate_action

from .bootstrap import DriverRegistry
from .action_value import export_action_value_member
from .contracts import MemberId
from .features import INPUT_WIDTH, encode_options
from .single_intervention import (
    CalibrationCell,
    CalibrationKey,
    IncumbentDecision,
    InterventionEnsemble,
    InterventionExample,
    InterventionTracker,
    InterventionUpdateStats,
    calibrated_override_margins,
    centered_label,
    choose_trial_index,
    create_intervention_population,
    eligible_intervention,
    mean_option_scores,
    model_option_scores,
    pretrain_incumbent_population,
    trusted_override,
    update_intervention_population,
)
from .storage import RunStorage, install_cleanup_handlers


TrialSelector = Callable[[np.ndarray, int, np.random.Generator], int]


@dataclass(slots=True)
class PopulationPolicy:
    registry: DriverRegistry
    ensembles: Mapping[MemberId, InterventionEnsemble]
    enabled_members: frozenset[MemberId]
    margin: float = 0.25
    margins: Mapping[MemberId, tuple[float, float]] | None = None
    override_counts: dict[MemberId, int] = field(default_factory=dict)

    def deck(self, member: MemberId) -> list[int]:
        return self.registry.deck(member)

    def decide(
        self,
        member: MemberId,
        observation: Mapping[str, Any],
    ) -> list[int]:
        incumbent = self.registry.action(member, observation)
        if member not in self.enabled_members or not eligible_intervention(
            observation,
            incumbent,
        ):
            return incumbent
        features = encode_options(observation)
        ensemble = self.ensembles[member]
        rows = tuple(
            model_option_scores(
                model,
                features,
                next(model.parameters()).device.type,
            )
            for model in ensemble.models
        )
        margin = (
            self.margin
            if self.margins is None
            else self.margins.get(member, (self.margin, self.margin))
        )
        override = trusted_override(rows, incumbent[0], margin=margin)
        action = incumbent if override is None else [override]
        if override is not None:
            self.override_counts[member] = self.override_counts.get(member, 0) + 1
        return validate_action(action, dict(observation["select"]))

    @property
    def overrides(self) -> int:
        return sum(self.override_counts.values())


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    member: MemberId
    opponent: MemberId
    seat: int
    target_ordinal: int
    features: np.ndarray
    incumbent_index: int
    trial_index: int
    actual_score: float


@dataclass(frozen=True, slots=True)
class ActualGameResult:
    members: tuple[MemberId, MemberId]
    winner: int
    decisions: int
    intervention: InterventionOutcome | None
    control: bool
    incumbent_decisions: dict[MemberId, tuple[IncumbentDecision, ...]]

    def score(self, member: MemberId) -> float:
        if member not in self.members:
            raise ValueError("member did not play this game")
        if self.winner not in (0, 1):
            return 0.5
        return 1.0 if self.members[self.winner] is member else 0.0


@dataclass(frozen=True, slots=True)
class CalibrationBatch:
    cells: dict[CalibrationKey, CalibrationCell]
    decisions: dict[MemberId, tuple[IncumbentDecision, ...]]
    games: int


@dataclass(frozen=True, slots=True)
class InterventionBatch:
    examples: dict[MemberId, tuple[InterventionExample, ...]]
    games: int
    controls: int


@dataclass(frozen=True, slots=True)
class TreeMeasurement:
    files: int
    bytes: int


@dataclass(frozen=True, slots=True)
class ActualComparison:
    candidate_score: float
    incumbent_score: float
    delta: float
    per_member_delta: dict[MemberId, float]
    candidate_games: int
    incumbent_games: int
    same_schedule: bool
    paired_randomness: bool
    candidate_overrides: int


@dataclass(frozen=True, slots=True)
class SingleInterventionDecision:
    passed: bool
    code: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoundSummary:
    round_index: int
    active_members: tuple[MemberId, ...]
    calibration_games: int
    intervention_games: int
    controls: int
    examples_per_member: dict[MemberId, int]
    pretraining: dict[MemberId, InterventionUpdateStats]
    updates: dict[MemberId, InterventionUpdateStats]
    override_margins: dict[MemberId, tuple[float, float]]
    comparison: ActualComparison | None
    promoted_members: tuple[MemberId, ...]


@dataclass(frozen=True, slots=True)
class ProofDependencies:
    game_api_factory: Callable[[Path], Any] = import_official_cg
    registry_factory: Callable[[Path], DriverRegistry] = DriverRegistry.from_project
    verify_engine: bool = True


@dataclass(slots=True)
class SingleInterventionReport:
    decision: SingleInterventionDecision
    engine_sha256: str | None
    rounds: tuple[RoundSummary, ...]
    selection: ActualComparison | None
    screening_confirmation: ActualComparison | None
    confirmation: ActualComparison | None
    promoted_members: tuple[MemberId, ...]
    evaluation_overrides: int
    failures: tuple[str, ...]
    phase_durations_seconds: dict[str, float]
    artifacts_before: TreeMeasurement
    artifacts_after: TreeMeasurement
    storage_root: Path
    storage_quota_bytes: int
    peak_temporary_bytes: int
    raw_replays_written: int
    retained_population: str | None
    device: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def summary(self) -> dict[str, Any]:
        return {
            "decision": self.decision.code,
            "passed": self.decision.passed,
            "promoted_members": [member.value for member in self.promoted_members],
            "selection_delta": None if self.selection is None else self.selection.delta,
            "screening_delta": (
                None
                if self.screening_confirmation is None
                else self.screening_confirmation.delta
            ),
            "confirmation_delta": (
                None if self.confirmation is None else self.confirmation.delta
            ),
            "evaluation_overrides": self.evaluation_overrides,
            "failures": list(self.failures),
        }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _start_failure(start_data: Any) -> RuntimeError:
    return RuntimeError(
        "battle_start failed: "
        f"errorPlayer={getattr(start_data, 'errorPlayer', None)}, "
        f"errorType={getattr(start_data, 'errorType', None)}"
    )


def select_survivors(
    per_member_delta: Mapping[MemberId, float],
) -> tuple[MemberId, ...]:
    return tuple(
        member
        for member in MemberId
        if math.isfinite(float(per_member_delta.get(member, math.nan)))
        and float(per_member_delta[member]) > 0.0
    )


def select_confirmed_survivors(
    candidates: Sequence[MemberId],
    selection: ActualComparison,
    confirmation: ActualComparison,
) -> tuple[MemberId, ...]:
    return tuple(
        member
        for member in candidates
        if all(
            math.isfinite(float(comparison.per_member_delta.get(member, math.nan)))
            and float(comparison.per_member_delta[member]) > 0.0
            for comparison in (selection, confirmation)
        )
    )


def decide_single_intervention_proof(
    *,
    selection: ActualComparison | None,
    confirmation: ActualComparison | None,
    promoted: Sequence[MemberId],
    overrides: int,
    failures: Sequence[str],
) -> SingleInterventionDecision:
    if failures:
        return SingleInterventionDecision(False, "REJECT_FAILURE", tuple(failures))
    promoted = tuple(promoted)
    if not promoted:
        return SingleInterventionDecision(
            False,
            "REJECT_NO_PROMOTION",
            ("no member survived both actual-game selection rounds",),
        )
    if overrides <= 0:
        return SingleInterventionDecision(
            False,
            "REJECT_NO_OVERRIDE",
            ("the learned scorers never changed an evaluation action",),
        )
    comparisons = tuple(
        item for item in (selection, confirmation) if item is not None
    )
    positive_group = (
        len(comparisons) == 2
        and all(
            item.same_schedule
            and not item.paired_randomness
            and math.isfinite(item.delta)
            and item.delta > 0.0
            for item in comparisons
        )
    )
    if not positive_group:
        return SingleInterventionDecision(
            False,
            "REJECT_NO_GROUP_IMPROVEMENT",
            ("selection and fresh confirmation must both improve",),
        )
    regressing = tuple(
        member
        for member in promoted
        if any(
            not math.isfinite(float(item.per_member_delta.get(member, math.nan)))
            or float(item.per_member_delta[member]) <= 0.0
            for item in comparisons
        )
    )
    if regressing:
        return SingleInterventionDecision(
            False,
            "REJECT_MEMBER_REGRESSION",
            (
                "promoted members failed individual confirmation: "
                + ", ".join(member.value for member in regressing),
            ),
        )
    return SingleInterventionDecision(
        True,
        "PASS_SINGLE_INTERVENTION_MAC",
        (),
    )


def decide_group_upgrade_proof(
    *,
    comparison: ActualComparison | None,
    overrides: int,
    expected_games_per_side: int,
    failures: Sequence[str],
) -> SingleInterventionDecision:
    if failures:
        return SingleInterventionDecision(False, "REJECT_FAILURE", tuple(failures))
    if (
        comparison is None
        or comparison.candidate_games != expected_games_per_side
        or comparison.incumbent_games != expected_games_per_side
        or not comparison.same_schedule
        or comparison.paired_randomness
    ):
        return SingleInterventionDecision(
            False,
            "REJECT_INCOMPLETE_GROUP_AUDIT",
            ("the balanced group audit did not reach its frozen sample count",),
        )
    if overrides <= 0:
        return SingleInterventionDecision(
            False,
            "REJECT_NO_OVERRIDE",
            ("the learned population never changed an audit action",),
        )
    if not math.isfinite(comparison.delta) or comparison.delta <= 0.0:
        return SingleInterventionDecision(
            False,
            "REJECT_NO_GROUP_IMPROVEMENT",
            ("the two-round population did not beat the original group",),
        )
    return SingleInterventionDecision(True, "PASS_GROUP_SELFPLAY_MAC", ())


def run_actual_game(
    game_api: Any,
    policy: Any,
    *,
    members: tuple[MemberId, MemberId],
    rng: np.random.Generator,
    experimental_member: MemberId | None = None,
    target_ordinal: int | None = None,
    trial_selector: TrialSelector | None = None,
    capture_incumbent_decisions: bool = False,
    max_steps: int = 2000,
) -> ActualGameResult:
    if members[0] is members[1]:
        raise ValueError("a game requires two different members")
    if experimental_member is not None and experimental_member not in members:
        raise ValueError("experimental member must be one of the players")
    experimenting = experimental_member is not None
    if experimenting and (target_ordinal is None or trial_selector is None):
        raise ValueError("experimental games require a target and trial selector")
    if not experimenting and (target_ordinal is not None or trial_selector is not None):
        raise ValueError("ordinary games cannot have intervention arguments")

    tracker = InterventionTracker(target_ordinal) if experimenting else None
    observation, start_data = game_api.battle_start(
        policy.deck(members[0]),
        policy.deck(members[1]),
    )
    if observation is None:
        raise _start_failure(start_data)

    captured: dict[MemberId, list[IncumbentDecision]] = {
        members[0]: [],
        members[1]: [],
    }
    intervention_data: tuple[int, np.ndarray, int, int] | None = None
    try:
        for decision in range(max_steps):
            current = observation.get("current")
            if not isinstance(current, Mapping):
                raise RuntimeError("engine returned no current state")
            winner = int(current.get("result", -1))
            if winner >= 0:
                outcome = None
                if intervention_data is not None and experimental_member is not None:
                    seat, features, incumbent_index, trial_index = intervention_data
                    opponent = members[1 - seat]
                    actual_score = (
                        0.5
                        if winner not in (0, 1)
                        else (1.0 if winner == seat else 0.0)
                    )
                    outcome = InterventionOutcome(
                        member=experimental_member,
                        opponent=opponent,
                        seat=seat,
                        target_ordinal=int(target_ordinal),
                        features=features,
                        incumbent_index=incumbent_index,
                        trial_index=trial_index,
                        actual_score=actual_score,
                    )
                return ActualGameResult(
                    members=members,
                    winner=winner,
                    decisions=decision,
                    intervention=outcome,
                    control=experimenting and outcome is None,
                    incumbent_decisions={
                        member: tuple(rows) for member, rows in captured.items()
                    },
                )

            seat = int(current["yourIndex"])
            if seat not in (0, 1):
                raise RuntimeError("engine returned invalid current player")
            member = members[seat]
            incumbent = policy.decide(member, observation)
            select = observation.get("select")
            if not isinstance(select, Mapping):
                raise RuntimeError("engine returned no selection")
            incumbent = validate_action(incumbent, dict(select))

            is_eligible = eligible_intervention(observation, incumbent)
            should_intervene = (
                tracker is not None
                and member is experimental_member
                and tracker.consider(observation, incumbent)
            )
            features: np.ndarray | None = None
            if is_eligible and (capture_incumbent_decisions or should_intervene):
                features = encode_options(observation)
            if is_eligible and capture_incumbent_decisions:
                if features is None:
                    raise RuntimeError("eligible decision has no encoded features")
                captured[member].append(
                    IncumbentDecision(
                        member=member,
                        features=features,
                        incumbent_index=incumbent[0],
                    )
                )

            action = incumbent
            if should_intervene:
                if features is None or trial_selector is None:
                    raise RuntimeError("eligible intervention has no encoded features")
                trial_index = int(trial_selector(features, incumbent[0], rng))
                if not 0 <= trial_index < len(features):
                    raise ValueError("trial selector returned an out-of-range action")
                if trial_index == incumbent[0]:
                    raise ValueError("trial selector returned the incumbent action")
                action = validate_action([trial_index], dict(select))
                tracker.mark_used()
                intervention_data = (
                    seat,
                    features,
                    incumbent[0],
                    trial_index,
                )
            observation = game_api.battle_select(action)
        raise RuntimeError(f"match did not finish within {max_steps} decisions")
    finally:
        game_api.battle_finish()


def collect_calibration(
    game_api: Any,
    policy: Any,
    rng: np.random.Generator,
    *,
    max_decisions_per_member: int = 2048,
    max_steps: int = 2000,
    deadline: float | None = None,
) -> CalibrationBatch:
    if max_decisions_per_member <= 0:
        raise ValueError("decision limit must be positive")
    points: dict[CalibrationKey, float] = {}
    games: dict[CalibrationKey, int] = {}
    decisions: dict[MemberId, list[IncumbentDecision]] = {
        member: [] for member in MemberId
    }
    completed = 0
    for focal in MemberId:
        for opponent in MemberId:
            if opponent is focal:
                continue
            for seat in (0, 1):
                members = (focal, opponent) if seat == 0 else (opponent, focal)
                key = CalibrationKey(focal, opponent, seat)
                for _ in range(2):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("calibration exceeded its deadline")
                    result = run_actual_game(
                        game_api,
                        policy,
                        members=members,
                        rng=rng,
                        capture_incumbent_decisions=True,
                        max_steps=max_steps,
                    )
                    points[key] = points.get(key, 0.0) + result.score(focal)
                    games[key] = games.get(key, 0) + 1
                    room = max_decisions_per_member - len(decisions[focal])
                    if room > 0:
                        decisions[focal].extend(
                            result.incumbent_decisions[focal][:room]
                        )
                    completed += 1
    return CalibrationBatch(
        cells={
            key: CalibrationCell(points=points[key], games=games[key])
            for key in points
        },
        decisions={member: tuple(rows) for member, rows in decisions.items()},
        games=completed,
    )


def _balanced_intervention_schedule(
    member: MemberId,
    round_index: int,
    rng: np.random.Generator,
) -> list[tuple[MemberId, int]]:
    opponents = [opponent for opponent in MemberId if opponent is not member]
    cells = list(itertools.product(opponents, (0, 1)))
    schedule = cells * 5
    offset = (tuple(MemberId).index(member) + round_index - 1) % len(cells)
    schedule.extend((cells[offset], cells[(offset + 3) % len(cells)]))
    rng.shuffle(schedule)
    return schedule


def collect_interventions(
    game_api: Any,
    policy: PopulationPolicy,
    calibration: Mapping[CalibrationKey, CalibrationCell],
    round_index: int,
    rng: np.random.Generator,
    *,
    target_members: Sequence[MemberId] = tuple(MemberId),
    trial_ensembles: Mapping[MemberId, InterventionEnsemble] | None = None,
    deadline: float | None = None,
    max_controls_per_example: int = 64,
    max_steps: int = 2000,
) -> InterventionBatch:
    if round_index not in (1, 2):
        raise ValueError("intervention round must be one or two")
    examples: dict[MemberId, list[InterventionExample]] = {
        member: [] for member in MemberId
    }
    games = 0
    controls = 0
    selected_members = tuple(dict.fromkeys(target_members))
    if not selected_members or any(member not in MemberId for member in selected_members):
        raise ValueError("target members must be a non-empty subset of the league")
    scorers = policy.ensembles if trial_ensembles is None else trial_ensembles
    for member in selected_members:
        ensemble = scorers[member]

        def select_trial(
            features: np.ndarray,
            incumbent_index: int,
            local_rng: np.random.Generator,
        ) -> int:
            score_rows = tuple(
                model_option_scores(
                    model,
                    features,
                    next(model.parameters()).device.type,
                )
                for model in ensemble.models
            )
            return choose_trial_index(
                mean_option_scores(score_rows),
                incumbent_index,
                local_rng,
            )

        for opponent, seat in _balanced_intervention_schedule(member, round_index, rng):
            key = CalibrationKey(member, opponent, seat)
            if key not in calibration:
                raise ValueError(f"missing calibration cell {key}")
            for attempt in range(max_controls_per_example + 1):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("intervention collection exceeded its deadline")
                members = (member, opponent) if seat == 0 else (opponent, member)
                target = int(rng.integers(1, 33))
                result = run_actual_game(
                    game_api,
                    policy,
                    members=members,
                    experimental_member=member,
                    target_ordinal=target,
                    trial_selector=select_trial,
                    rng=rng,
                    max_steps=max_steps,
                )
                games += 1
                if result.intervention is None:
                    controls += 1
                    if attempt == max_controls_per_example:
                        raise RuntimeError(
                            f"too many controls while collecting {member.value}"
                        )
                    continue
                outcome = result.intervention
                examples[member].append(
                    InterventionExample(
                        member=member,
                        opponent=opponent,
                        seat=seat,
                        round_index=round_index,
                        target_ordinal=outcome.target_ordinal,
                        features=outcome.features,
                        incumbent_index=outcome.incumbent_index,
                        trial_index=outcome.trial_index,
                        label=centered_label(outcome.actual_score, calibration[key]),
                    )
                )
                break
    if any(len(examples[member]) != 32 for member in selected_members):
        raise RuntimeError("intervention collection did not produce 32 rows per member")
    return InterventionBatch(
        examples={member: tuple(rows) for member, rows in examples.items()},
        games=games,
        controls=controls,
    )


def _clone_population(
    population: Mapping[MemberId, InterventionEnsemble],
    device: str,
) -> dict[MemberId, InterventionEnsemble]:
    clone = create_intervention_population(0, device)
    for member in MemberId:
        for source, destination in zip(
            population[member].models,
            clone[member].models,
            strict=True,
        ):
            destination.load_state_dict(source.state_dict())
    return clone


def _replace_members(
    incumbent: Mapping[MemberId, InterventionEnsemble],
    candidate: Mapping[MemberId, InterventionEnsemble],
    promoted: Sequence[MemberId],
    device: str,
) -> dict[MemberId, InterventionEnsemble]:
    result = _clone_population(incumbent, device)
    for member in promoted:
        for source, destination in zip(
            candidate[member].models,
            result[member].models,
            strict=True,
        ):
            destination.load_state_dict(source.state_dict())
    return result


def _focal_population(
    incumbent: Mapping[MemberId, InterventionEnsemble],
    candidate: Mapping[MemberId, InterventionEnsemble],
    focal: MemberId,
) -> dict[MemberId, InterventionEnsemble]:
    return {
        member: candidate[member] if member is focal else incumbent[member]
        for member in MemberId
    }


def calibrate_population_override_margins(
    population: Mapping[MemberId, InterventionEnsemble],
    decisions: Mapping[MemberId, Sequence[IncumbentDecision]],
    *,
    members: Sequence[MemberId],
    quantile: float = 0.995,
    minimum: float = 0.25,
    batch_size: int = 128,
) -> dict[MemberId, tuple[float, float]]:
    if batch_size <= 0:
        raise ValueError("margin calibration batch size must be positive")
    result: dict[MemberId, tuple[float, float]] = {}
    for member in members:
        member_decisions = list(decisions.get(member, ()))
        if not member_decisions:
            raise ValueError(f"{member.value} has no margin calibration decisions")
        per_decision: list[tuple[np.ndarray, np.ndarray]] = []
        ensemble = population[member]
        for start in range(0, len(member_decisions), batch_size):
            batch = member_decisions[start : start + batch_size]
            maximum = max(len(decision.features) for decision in batch)
            padded = np.zeros((len(batch), maximum, INPUT_WIDTH), dtype=np.float32)
            mask = np.zeros((len(batch), maximum), dtype=bool)
            for row_index, decision in enumerate(batch):
                count = len(decision.features)
                padded[row_index, :count] = decision.features
                mask[row_index, :count] = True
            model_rows: list[np.ndarray] = []
            for model in ensemble.models:
                device = next(model.parameters()).device
                model.eval()
                with torch.no_grad():
                    values, _, _ = model(
                        torch.from_numpy(padded).to(device),
                        torch.from_numpy(mask).to(device),
                    )
                model_rows.append(values.detach().cpu().numpy())
            per_decision.extend(
                (
                    model_rows[0][row_index, : len(decision.features)].astype(
                        np.float64,
                        copy=False,
                    ),
                    model_rows[1][row_index, : len(decision.features)].astype(
                        np.float64,
                        copy=False,
                    ),
                )
                for row_index, decision in enumerate(batch)
            )
        result[member] = calibrated_override_margins(
            per_decision,
            [decision.incumbent_index for decision in member_decisions],
            quantile=quantile,
            minimum=minimum,
        )
    return result


def evaluate_actual_comparison(
    game_api: Any,
    registry: DriverRegistry,
    candidate: Mapping[MemberId, InterventionEnsemble],
    incumbent: Mapping[MemberId, InterventionEnsemble],
    *,
    target_members: Sequence[MemberId],
    incumbent_enabled: Sequence[MemberId],
    candidate_margins: Mapping[MemberId, tuple[float, float]] | None = None,
    incumbent_margins: Mapping[MemberId, tuple[float, float]] | None = None,
    rng: np.random.Generator,
    deadline: float | None = None,
    max_steps: int = 2000,
) -> ActualComparison:
    targets = tuple(dict.fromkeys(target_members))
    if not targets:
        raise ValueError("evaluation requires at least one target member")
    incumbent_enabled_set = frozenset(incumbent_enabled)
    candidate_points: dict[MemberId, float] = {member: 0.0 for member in targets}
    incumbent_points: dict[MemberId, float] = {member: 0.0 for member in targets}
    candidate_games = 0
    incumbent_games = 0
    overrides = 0
    for focal in targets:
        composite_margins = dict(incumbent_margins or {})
        if candidate_margins is not None and focal in candidate_margins:
            composite_margins[focal] = candidate_margins[focal]
        candidate_policy = PopulationPolicy(
            registry=registry,
            ensembles=_focal_population(incumbent, candidate, focal),
            enabled_members=incumbent_enabled_set | {focal},
            margins=composite_margins,
        )
        incumbent_policy = PopulationPolicy(
            registry=registry,
            ensembles=incumbent,
            enabled_members=incumbent_enabled_set,
            margins=incumbent_margins,
        )
        for opponent in MemberId:
            if opponent is focal:
                continue
            for seat in (0, 1):
                members = (focal, opponent) if seat == 0 else (opponent, focal)
                for _ in range(2):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("actual-game evaluation exceeded its deadline")
                    candidate_result = run_actual_game(
                        game_api,
                        candidate_policy,
                        members=members,
                        rng=rng,
                        max_steps=max_steps,
                    )
                    candidate_points[focal] += candidate_result.score(focal)
                    candidate_games += 1
                    incumbent_result = run_actual_game(
                        game_api,
                        incumbent_policy,
                        members=members,
                        rng=rng,
                        max_steps=max_steps,
                    )
                    incumbent_points[focal] += incumbent_result.score(focal)
                    incumbent_games += 1
        overrides += candidate_policy.overrides

    per_member_delta = {
        member: candidate_points[member] / 12.0 - incumbent_points[member] / 12.0
        for member in targets
    }
    candidate_score = sum(candidate_points.values()) / candidate_games
    incumbent_score = sum(incumbent_points.values()) / incumbent_games
    return ActualComparison(
        candidate_score=candidate_score,
        incumbent_score=incumbent_score,
        delta=candidate_score - incumbent_score,
        per_member_delta=per_member_delta,
        candidate_games=candidate_games,
        incumbent_games=incumbent_games,
        same_schedule=True,
        paired_randomness=False,
        candidate_overrides=overrides,
    )


def evaluate_group_actual_comparison(
    game_api: Any,
    registry: DriverRegistry,
    candidate: Mapping[MemberId, InterventionEnsemble],
    original: Mapping[MemberId, InterventionEnsemble],
    *,
    candidate_margins: Mapping[MemberId, tuple[float, float]],
    repetitions: int,
    rng: np.random.Generator,
    deadline: float | None = None,
    max_steps: int = 2000,
) -> ActualComparison:
    if repetitions <= 0:
        raise ValueError("group audit repetitions must be positive")
    candidate_policy = PopulationPolicy(
        registry=registry,
        ensembles=candidate,
        enabled_members=frozenset(MemberId),
        margins=candidate_margins,
    )
    incumbent_policy = PopulationPolicy(
        registry=registry,
        ensembles=original,
        enabled_members=frozenset(),
    )
    candidate_points = {member: 0.0 for member in MemberId}
    incumbent_points = {member: 0.0 for member in MemberId}
    candidate_games = 0
    incumbent_games = 0
    for focal in MemberId:
        for opponent in MemberId:
            if opponent is focal:
                continue
            for seat in (0, 1):
                members = (focal, opponent) if seat == 0 else (opponent, focal)
                for _ in range(repetitions):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("balanced group audit exceeded its deadline")
                    candidate_result = run_actual_game(
                        game_api,
                        candidate_policy,
                        members=members,
                        rng=rng,
                        max_steps=max_steps,
                    )
                    candidate_points[focal] += candidate_result.score(focal)
                    candidate_games += 1
                    incumbent_result = run_actual_game(
                        game_api,
                        incumbent_policy,
                        members=members,
                        rng=rng,
                        max_steps=max_steps,
                    )
                    incumbent_points[focal] += incumbent_result.score(focal)
                    incumbent_games += 1
    games_per_member = 6 * repetitions
    per_member_delta = {
        member: (
            candidate_points[member] - incumbent_points[member]
        ) / games_per_member
        for member in MemberId
    }
    candidate_score = sum(candidate_points.values()) / candidate_games
    incumbent_score = sum(incumbent_points.values()) / incumbent_games
    return ActualComparison(
        candidate_score=candidate_score,
        incumbent_score=incumbent_score,
        delta=candidate_score - incumbent_score,
        per_member_delta=per_member_delta,
        candidate_games=candidate_games,
        incumbent_games=incumbent_games,
        same_schedule=True,
        paired_randomness=False,
        candidate_overrides=candidate_policy.overrides,
    )


def _restricted_comparison(
    comparison: ActualComparison,
    members: Sequence[MemberId],
) -> ActualComparison:
    selected = tuple(members)
    if not selected:
        return ActualComparison(
            candidate_score=0.0,
            incumbent_score=0.0,
            delta=0.0,
            per_member_delta={},
            candidate_games=0,
            incumbent_games=0,
            same_schedule=comparison.same_schedule,
            paired_randomness=comparison.paired_randomness,
            candidate_overrides=comparison.candidate_overrides,
        )
    delta = sum(comparison.per_member_delta[member] for member in selected) / len(
        selected
    )
    return ActualComparison(
        candidate_score=comparison.incumbent_score + delta,
        incumbent_score=comparison.incumbent_score,
        delta=delta,
        per_member_delta={
            member: comparison.per_member_delta[member] for member in selected
        },
        candidate_games=12 * len(selected),
        incumbent_games=12 * len(selected),
        same_schedule=comparison.same_schedule,
        paired_randomness=comparison.paired_randomness,
        candidate_overrides=comparison.candidate_overrides,
    )


def measure_tree(root: Path) -> TreeMeasurement:
    files = 0
    total_bytes = 0
    if root.exists():
        for directory, _, names in os.walk(root):
            for name in names:
                path = Path(directory) / name
                files += 1
                total_bytes += path.stat().st_size
    return TreeMeasurement(files, total_bytes)


def _engine_library(project_root: Path) -> Path:
    return (
        project_root
        / "data/raw/pokemon-tcg-ai-battle/sample_submission/sample_submission"
        / "cg/libcg.dylib"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ENGINE22_SHA256 = "7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30"
TEMP_QUOTA_BYTES = 128 * 1024**2


def _retain_population(
    project_root: Path,
    storage: RunStorage,
    population: Mapping[MemberId, InterventionEnsemble],
    promoted: Sequence[MemberId],
    margins: Mapping[MemberId, tuple[float, float]],
) -> str:
    staged = storage.root / "confirmed-population"
    staged.mkdir()
    for member in promoted:
        for model_index, model in enumerate(population[member].models):
            export_action_value_member(
                model,
                str(staged / f"{member.value}-{model_index}.npz"),
            )
    manifest = {
        "format": "single-intervention-action-value-v1",
        "members": [member.value for member in promoted],
        "models_per_member": 2,
        "input_width": INPUT_WIDTH,
        "override_margins": {
            member.value: list(margins[member]) for member in promoted
        },
    }
    (staged / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination = project_root / "agents/single_intervention_mac_pass"
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing {destination}")
    os.replace(staged, destination)
    return str(destination)


def run_single_intervention_proof(
    project_root: Path,
    *,
    dependencies: ProofDependencies | None = None,
    wall_time_seconds: int = 600,
    seed: int = 20260804,
    initial_population: Mapping[MemberId, InterventionEnsemble] | None = None,
    initial_margins: Mapping[MemberId, tuple[float, float]] | None = None,
) -> SingleInterventionReport:
    if wall_time_seconds <= 0:
        raise ValueError("wall-time limit must be positive")
    project_root = Path(project_root).expanduser().resolve()
    dependencies = ProofDependencies() if dependencies is None else dependencies
    started = time.monotonic()
    deadline = started + wall_time_seconds
    phase_durations: dict[str, float] = {}
    artifacts_before = measure_tree(project_root / "artifacts")
    engine_sha256: str | None = None
    rounds: list[RoundSummary] = []
    selection: ActualComparison | None = None
    screening_confirmation: ActualComparison | None = None
    confirmation: ActualComparison | None = None
    promoted: tuple[MemberId, ...] = ()
    evaluation_overrides = 0
    failures: list[str] = []
    retained: str | None = None
    device = "cpu"
    decision = SingleInterventionDecision(
        False,
        "REJECT_FAILURE",
        ("proof did not finish",),
    )
    registry: DriverRegistry | None = None
    storage_root = (
        Path(tempfile.gettempdir()).resolve()
        / f"pokemon-single-intervention-{uuid.uuid4().hex}"
    )
    storage = RunStorage(
        storage_root,
        quota_bytes=TEMP_QUOTA_BYTES,
        shard_bytes=64 * 1024**2,
        max_pending=2,
    )
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    install_cleanup_handlers(storage)

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("single-intervention proof exceeded 600 seconds")

    def phase(name: str, action: Callable[[], Any]) -> Any:
        check_deadline()
        phase_started = time.monotonic()
        result = action()
        phase_durations[name] = time.monotonic() - phase_started
        check_deadline()
        return result

    try:
        def preflight() -> Any:
            nonlocal engine_sha256, registry
            if dependencies.verify_engine:
                library = _engine_library(project_root)
                if not library.is_file():
                    raise FileNotFoundError(f"missing official Engine22 library: {library}")
                engine_sha256 = _sha256(library)
                if engine_sha256 != ENGINE22_SHA256:
                    raise RuntimeError(
                        f"Engine22 hash mismatch: {engine_sha256}"
                    )
            game_api = dependencies.game_api_factory(project_root)
            registry = dependencies.registry_factory(project_root)
            return game_api

        game_api = phase("preflight", preflight)
        if registry is None:
            raise RuntimeError("registry factory returned no registry")
        if initial_population is None:
            current = create_intervention_population(seed, device)
        else:
            if set(initial_population) != set(MemberId):
                raise ValueError("initial population must contain all four members")
            current = _clone_population(initial_population, device)
        original = _clone_population(current, device)
        if initial_margins is None:
            current_margins = {member: (0.25, 0.25) for member in MemberId}
        else:
            if set(initial_margins) != set(MemberId):
                raise ValueError("initial margins must contain all four members")
            current_margins = {
                member: (
                    float(initial_margins[member][0]),
                    float(initial_margins[member][1]),
                )
                for member in MemberId
            }
        incumbent_enabled: tuple[MemberId, ...] = (
            () if initial_population is None else tuple(MemberId)
        )
        active: tuple[MemberId, ...] = tuple(MemberId)

        for round_index in (1, 2):
            incumbent = _clone_population(current, device)
            incumbent_policy = PopulationPolicy(
                registry=registry,
                ensembles=incumbent,
                enabled_members=frozenset(incumbent_enabled),
                margins=current_margins,
            )
            calibration = phase(
                f"round_{round_index}_calibration",
                lambda: collect_calibration(
                    game_api,
                    incumbent_policy,
                    np.random.default_rng(seed + round_index * 1000 + 1),
                    deadline=deadline,
                ),
            )
            candidate = _clone_population(incumbent, device)
            pretraining = phase(
                f"round_{round_index}_pretrain",
                lambda: pretrain_incumbent_population(
                    candidate,
                    calibration.decisions,
                    device,
                    seed + round_index * 1000 + 2,
                    members=active,
                ),
            )
            intervention_batch = phase(
                f"round_{round_index}_interventions",
                lambda: collect_interventions(
                    game_api,
                    incumbent_policy,
                    calibration.cells,
                    round_index,
                    np.random.default_rng(seed + round_index * 1000 + 3),
                    target_members=active,
                    trial_ensembles=candidate,
                    deadline=deadline,
                ),
            )
            updates = phase(
                f"round_{round_index}_update",
                lambda: update_intervention_population(
                    candidate,
                    intervention_batch.examples,
                    device,
                    seed + round_index * 1000 + 4,
                    members=active,
                ),
            )
            if any(
                not pretraining[member].all_finite
                or not updates[member].all_finite
                or len(intervention_batch.examples[member]) != 32
                for member in active
            ):
                raise RuntimeError("non-finite update or incomplete intervention batch")
            calibrated_margins = phase(
                f"round_{round_index}_margin_calibration",
                lambda: calibrate_population_override_margins(
                    candidate,
                    calibration.decisions,
                    members=active,
                ),
            )
            candidate_margins = dict(current_margins)
            candidate_margins.update(calibrated_margins)
            current = candidate
            current_margins = candidate_margins
            incumbent_enabled = tuple(MemberId)
            rounds.append(
                RoundSummary(
                    round_index=round_index,
                    active_members=tuple(MemberId),
                    calibration_games=calibration.games,
                    intervention_games=intervention_batch.games,
                    controls=intervention_batch.controls,
                    examples_per_member={
                        member: len(intervention_batch.examples[member])
                        for member in MemberId
                    },
                    pretraining=pretraining,
                    updates=updates,
                    override_margins={
                        member: candidate_margins[member]
                        for member in MemberId
                    },
                    comparison=None,
                    promoted_members=tuple(MemberId),
                )
            )

        audit_repetitions = 8
        expected_audit_games = len(MemberId) * (len(MemberId) - 1) * 2 * audit_repetitions
        confirmation = phase(
            "balanced_group_audit",
            lambda: evaluate_group_actual_comparison(
                game_api,
                registry,
                current,
                original,
                candidate_margins=current_margins,
                repetitions=audit_repetitions,
                rng=np.random.default_rng(seed + 10000),
                deadline=deadline,
            ),
        )
        evaluation_overrides = confirmation.candidate_overrides
        decision = decide_group_upgrade_proof(
            comparison=confirmation,
            overrides=evaluation_overrides,
            expected_games_per_side=expected_audit_games,
            failures=failures,
        )
        if decision.passed:
            promoted = tuple(MemberId)
            retained = phase(
                "retain_confirmed_population",
                lambda: _retain_population(
                    project_root,
                    storage,
                    current,
                    promoted,
                    current_margins,
                ),
            )
    except TimeoutError as error:
        failures.append(str(error))
        decision = SingleInterventionDecision(
            False,
            "REJECT_RUNTIME",
            tuple(failures),
        )
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        decision = SingleInterventionDecision(
            False,
            "REJECT_FAILURE",
            tuple(failures),
        )
    finally:
        if registry is not None:
            registry.close()
        cleanup_started = time.monotonic()
        storage.cleanup()
        phase_durations["cleanup"] = time.monotonic() - cleanup_started
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    artifacts_after = measure_tree(project_root / "artifacts")
    if artifacts_after != artifacts_before and decision.passed:
        failures.append("artifacts tree changed during proof")
        decision = SingleInterventionDecision(
            False,
            "REJECT_FAILURE",
            tuple(failures),
        )
    return SingleInterventionReport(
        decision=decision,
        engine_sha256=engine_sha256,
        rounds=tuple(rounds),
        selection=selection,
        screening_confirmation=screening_confirmation,
        confirmation=confirmation,
        promoted_members=promoted,
        evaluation_overrides=evaluation_overrides,
        failures=tuple(failures),
        phase_durations_seconds=phase_durations,
        artifacts_before=artifacts_before,
        artifacts_after=artifacts_after,
        storage_root=storage.root,
        storage_quota_bytes=storage.quota_bytes,
        peak_temporary_bytes=storage.peak_bytes,
        raw_replays_written=0,
        retained_population=retained,
        device=device,
        seed=seed,
    )


def write_report_atomic(report: SingleInterventionReport, path: Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if len(payload.encode("utf-8")) >= 100 * 1024:
        raise ValueError("single-intervention report exceeds 100 KiB")
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
