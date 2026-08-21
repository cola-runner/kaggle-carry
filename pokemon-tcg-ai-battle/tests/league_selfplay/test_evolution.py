from __future__ import annotations

import numpy as np

from league_selfplay.contracts import MemberId
from league_selfplay.evolution import (
    MarginGenome,
    combine_comparisons,
    evolution_passes,
    neutralize_frozen_members,
    spawn_generation,
)
from league_selfplay.single_intervention_runner import ActualComparison


def _comparison(delta: float, per_member: float = 0.03) -> ActualComparison:
    return ActualComparison(
        candidate_score=0.5 + delta,
        incumbent_score=0.5,
        delta=delta,
        per_member_delta={member: per_member for member in MemberId},
        candidate_games=72,
        incumbent_games=72,
        same_schedule=True,
        paired_randomness=False,
        candidate_overrides=10,
    )


def test_generation_is_batched_deterministic_and_bounded() -> None:
    first = spawn_generation(MarginGenome.champion(), np.random.default_rng(7))
    second = spawn_generation(MarginGenome.champion(), np.random.default_rng(7))
    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6
    assert all(0.45 <= value <= 1.80 for genome in first for value in genome.scales)


def test_genome_scales_both_thresholds_for_every_member() -> None:
    base = {member: (2.0, 4.0) for member in MemberId}
    genome = MarginGenome(tuple(float(index + 1) for index in range(8)))
    margins = genome.margins(base)
    assert margins[MemberId.GRIMMSNARL] == (2.0, 8.0)
    assert margins[MemberId.ALAKAZAM] == (14.0, 32.0)


def test_genome_round_trip_and_freeze() -> None:
    original = MarginGenome(tuple(float(index + 1) for index in range(8)))
    restored = MarginGenome.from_dict(original.to_dict())
    assert restored == original
    frozen = restored.freeze((MemberId.LUCARIO, MemberId.ALAKAZAM))
    assert frozen.to_dict()["lucario"] == [1.0, 1.0]
    assert frozen.to_dict()["alakazam"] == [1.0, 1.0]
    assert frozen.to_dict()["grimmsnarl"] == [1.0, 2.0]


def test_pass_requires_two_independent_positive_blocks() -> None:
    first = _comparison(0.03)
    second = _comparison(0.04)
    combined = combine_comparisons(first, second)
    assert evolution_passes(first, second, combined)[0]

    failed = _comparison(-0.01)
    combined_failed = combine_comparisons(first, failed)
    assert evolution_passes(first, failed, combined_failed) == (
        False,
        "REJECT_FRESH_BLOCK",
    )


def test_pass_rejects_member_regression() -> None:
    first = _comparison(0.04)
    second = _comparison(0.04)
    second.per_member_delta[MemberId.GRIMMSNARL] = -0.20
    combined = combine_comparisons(first, second)
    assert evolution_passes(first, second, combined) == (
        False,
        "REJECT_MEMBER_REGRESSION",
    )


def test_frozen_member_noise_is_zeroed_before_gate() -> None:
    comparison = _comparison(0.04)
    comparison.per_member_delta[MemberId.LUCARIO] = -0.20
    adjusted = neutralize_frozen_members(comparison, (MemberId.LUCARIO,))
    assert adjusted.per_member_delta[MemberId.LUCARIO] == 0.0
    assert adjusted.delta == sum(adjusted.per_member_delta.values()) / 4.0
