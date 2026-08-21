from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import MemberId
from .single_intervention import InterventionEnsemble
from .single_intervention_runner import (
    ActualComparison,
    PopulationPolicy,
    run_actual_game,
)


GENOME_WIDTH = len(MemberId) * 2


@dataclass(frozen=True, slots=True)
class MarginGenome:
    """Eight small policy knobs; the action-value models remain immutable."""

    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.scales) != GENOME_WIDTH:
            raise ValueError(f"margin genome requires {GENOME_WIDTH} scales")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.scales):
            raise ValueError("margin scales must be finite and positive")

    @classmethod
    def champion(cls) -> MarginGenome:
        return cls((1.0,) * GENOME_WIDTH)

    @classmethod
    def from_dict(cls, values: Mapping[str, list[float]]) -> MarginGenome:
        scales: list[float] = []
        for member in MemberId:
            pair = values[member.value]
            if len(pair) != 2:
                raise ValueError(f"{member.value} requires two margin scales")
            scales.extend(float(value) for value in pair)
        return cls(tuple(scales))

    def freeze(self, members: tuple[MemberId, ...]) -> MarginGenome:
        scales = list(self.scales)
        for member in members:
            index = tuple(MemberId).index(member) * 2
            scales[index : index + 2] = (1.0, 1.0)
        return MarginGenome(tuple(scales))

    def margins(
        self,
        champion_margins: Mapping[MemberId, tuple[float, float]],
    ) -> dict[MemberId, tuple[float, float]]:
        values: dict[MemberId, tuple[float, float]] = {}
        for index, member in enumerate(MemberId):
            base = champion_margins[member]
            values[member] = (
                float(base[0]) * self.scales[index * 2],
                float(base[1]) * self.scales[index * 2 + 1],
            )
        return values

    def to_dict(self) -> dict[str, list[float]]:
        return {
            member.value: [
                self.scales[index * 2],
                self.scales[index * 2 + 1],
            ]
            for index, member in enumerate(MemberId)
        }


@dataclass(frozen=True, slots=True)
class LeagueScore:
    score: float
    per_member_score: dict[MemberId, float]
    games: int
    focal_overrides: int


def spawn_generation(
    parent: MarginGenome,
    rng: np.random.Generator,
    *,
    size: int = 6,
) -> tuple[MarginGenome, ...]:
    """Create a batch, mixing broad and member-specific mutations."""

    if size < 2:
        raise ValueError("an evolution generation needs at least two challengers")
    parent_values = np.asarray(parent.scales, dtype=np.float64)
    mutations: list[np.ndarray] = []
    mutations.append(parent_values * 0.80)
    mutations.append(parent_values * 1.25)
    while len(mutations) < size:
        child = parent_values * np.exp(rng.normal(0.0, 0.08, GENOME_WIDTH))
        member_index = (len(mutations) - 2) % len(MemberId)
        direction = 0.76 if rng.random() < 0.5 else 1.32
        child[member_index * 2 : member_index * 2 + 2] *= direction
        mutations.append(child)
    return tuple(
        MarginGenome(tuple(float(value) for value in np.clip(child, 0.45, 1.80)))
        for child in mutations
    )


def evaluate_challenger_score(
    game_api: Any,
    registry: Any,
    population: Mapping[MemberId, InterventionEnsemble],
    champion_margins: Mapping[MemberId, tuple[float, float]],
    challenger_margins: Mapping[MemberId, tuple[float, float]],
    *,
    repetitions: int,
    rng: np.random.Generator,
    deadline: float | None = None,
    max_steps: int = 2000,
) -> LeagueScore:
    """Score each challenger member only against frozen V1 opponents."""

    if repetitions <= 0:
        raise ValueError("league repetitions must be positive")
    points = {member: 0.0 for member in MemberId}
    overrides = 0
    games = 0
    for focal in MemberId:
        margins = dict(champion_margins)
        margins[focal] = challenger_margins[focal]
        policy = PopulationPolicy(
            registry=registry,
            ensembles=population,
            enabled_members=frozenset(MemberId),
            margins=margins,
        )
        for opponent in MemberId:
            if opponent is focal:
                continue
            for seat in (0, 1):
                members = (focal, opponent) if seat == 0 else (opponent, focal)
                for _ in range(repetitions):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("evolution screening exceeded its deadline")
                    result = run_actual_game(
                        game_api,
                        policy,
                        members=members,
                        rng=rng,
                        max_steps=max_steps,
                    )
                    points[focal] += result.score(focal)
                    games += 1
        overrides += policy.override_counts.get(focal, 0)
    games_per_member = 6 * repetitions
    return LeagueScore(
        score=sum(points.values()) / games,
        per_member_score={
            member: points[member] / games_per_member for member in MemberId
        },
        games=games,
        focal_overrides=overrides,
    )


def compare_to_frozen_champion(
    game_api: Any,
    registry: Any,
    population: Mapping[MemberId, InterventionEnsemble],
    champion_margins: Mapping[MemberId, tuple[float, float]],
    challenger_margins: Mapping[MemberId, tuple[float, float]],
    *,
    repetitions: int,
    rng: np.random.Generator,
    deadline: float | None = None,
    max_steps: int = 2000,
) -> ActualComparison:
    """Fresh interleaved comparison; only the focal challenger's knobs change."""

    if repetitions <= 0:
        raise ValueError("confirmation repetitions must be positive")
    candidate_points = {member: 0.0 for member in MemberId}
    champion_points = {member: 0.0 for member in MemberId}
    candidate_games = 0
    champion_games = 0
    candidate_overrides = 0
    champion_policy = PopulationPolicy(
        registry=registry,
        ensembles=population,
        enabled_members=frozenset(MemberId),
        margins=champion_margins,
    )
    for focal in MemberId:
        composite = dict(champion_margins)
        composite[focal] = challenger_margins[focal]
        candidate_policy = PopulationPolicy(
            registry=registry,
            ensembles=population,
            enabled_members=frozenset(MemberId),
            margins=composite,
        )
        for opponent in MemberId:
            if opponent is focal:
                continue
            for seat in (0, 1):
                members = (focal, opponent) if seat == 0 else (opponent, focal)
                for _ in range(repetitions):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("evolution confirmation exceeded its deadline")
                    candidate = run_actual_game(
                        game_api,
                        candidate_policy,
                        members=members,
                        rng=rng,
                        max_steps=max_steps,
                    )
                    champion = run_actual_game(
                        game_api,
                        champion_policy,
                        members=members,
                        rng=rng,
                        max_steps=max_steps,
                    )
                    candidate_points[focal] += candidate.score(focal)
                    champion_points[focal] += champion.score(focal)
                    candidate_games += 1
                    champion_games += 1
        candidate_overrides += candidate_policy.override_counts.get(focal, 0)
    games_per_member = 6 * repetitions
    per_member_delta = {
        member: (candidate_points[member] - champion_points[member])
        / games_per_member
        for member in MemberId
    }
    candidate_score = sum(candidate_points.values()) / candidate_games
    champion_score = sum(champion_points.values()) / champion_games
    return ActualComparison(
        candidate_score=candidate_score,
        incumbent_score=champion_score,
        delta=candidate_score - champion_score,
        per_member_delta=per_member_delta,
        candidate_games=candidate_games,
        incumbent_games=champion_games,
        same_schedule=True,
        paired_randomness=False,
        candidate_overrides=candidate_overrides,
    )


def combine_comparisons(
    first: ActualComparison,
    second: ActualComparison,
) -> ActualComparison:
    if first.candidate_games != second.candidate_games:
        raise ValueError("confirmation blocks must have equal size")
    candidate_score = (first.candidate_score + second.candidate_score) / 2.0
    champion_score = (first.incumbent_score + second.incumbent_score) / 2.0
    return ActualComparison(
        candidate_score=candidate_score,
        incumbent_score=champion_score,
        delta=candidate_score - champion_score,
        per_member_delta={
            member: (
                first.per_member_delta[member] + second.per_member_delta[member]
            )
            / 2.0
            for member in MemberId
        },
        candidate_games=first.candidate_games + second.candidate_games,
        incumbent_games=first.incumbent_games + second.incumbent_games,
        same_schedule=first.same_schedule and second.same_schedule,
        paired_randomness=False,
        candidate_overrides=first.candidate_overrides + second.candidate_overrides,
    )


def neutralize_frozen_members(
    comparison: ActualComparison,
    frozen_members: tuple[MemberId, ...],
) -> ActualComparison:
    """Remove deal noise for focal policies known to be byte-identical to V1."""

    deltas = dict(comparison.per_member_delta)
    for member in frozen_members:
        deltas[member] = 0.0
    group_delta = sum(deltas.values()) / len(MemberId)
    return ActualComparison(
        candidate_score=comparison.incumbent_score + group_delta,
        incumbent_score=comparison.incumbent_score,
        delta=group_delta,
        per_member_delta=deltas,
        candidate_games=comparison.candidate_games,
        incumbent_games=comparison.incumbent_games,
        same_schedule=comparison.same_schedule,
        paired_randomness=comparison.paired_randomness,
        candidate_overrides=comparison.candidate_overrides,
    )


def evolution_passes(
    first: ActualComparison,
    second: ActualComparison,
    combined: ActualComparison,
    *,
    minimum_combined_delta: float = 0.02,
    member_floor: float = -0.05,
) -> tuple[bool, str]:
    if first.delta <= 0.0 or second.delta <= 0.0:
        return False, "REJECT_FRESH_BLOCK"
    if combined.delta < minimum_combined_delta:
        return False, "REJECT_SMALL_ADVANTAGE"
    if combined.candidate_overrides <= 0:
        return False, "REJECT_NO_OVERRIDE"
    if any(delta < member_floor for delta in combined.per_member_delta.values()):
        return False, "REJECT_MEMBER_REGRESSION"
    if sum(delta > 0.0 for delta in combined.per_member_delta.values()) < 2:
        return False, "REJECT_NARROW_ADVANTAGE"
    return True, "PASS_MARGIN_EVOLUTION_MAC"
