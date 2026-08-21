from __future__ import annotations

import numpy as np

from league_selfplay.action_value import (
    action_value_parameter_count,
    build_action_value_examples,
    create_action_value_population,
)
from league_selfplay.contracts import GameProvenance, MemberId
from league_selfplay.engine import CompletedGame, TrajectoryStep


def _step(member: MemberId, seat: int, action: int) -> TrajectoryStep:
    return TrajectoryStep(
        member=member,
        seat=seat,
        features=np.zeros((2, 512), dtype=np.float32),
        action=(action,),
        min_count=1,
        max_count=1,
        old_log_probability=0.0,
        old_value=0.0,
        baseline_action=(0,),
        exploration_rate=0.1,
    )


def test_action_value_examples_propagate_result_with_equal_game_weight() -> None:
    first = MemberId.LUCARIO
    second = MemberId.CRUSTLE
    game = CompletedGame(
        provenance=GameProvenance.current_game(first, second),
        winner=0,
        decisions=4,
        steps=[
            _step(first, 0, 0),
            _step(second, 1, 1),
            _step(first, 0, 1),
            _step(second, 1, 0),
        ],
        finished=True,
    )

    examples = build_action_value_examples([game])

    assert [row.target for row in examples[first]] == [1.0, 1.0]
    assert [row.target for row in examples[second]] == [-1.0, -1.0]
    assert sum(row.weight for row in examples[first]) == 1.0
    assert sum(row.weight for row in examples[second]) == 1.0
    assert [row.option_index for row in examples[first]] == [0, 1]


def test_action_value_population_is_small_and_independent() -> None:
    population = create_action_value_population(20260804, "cpu")
    pointers = [next(model.parameters()).data_ptr() for model in population.values()]

    assert len(set(pointers)) == len(MemberId)
    assert action_value_parameter_count(next(iter(population.values()))) < 70_000
