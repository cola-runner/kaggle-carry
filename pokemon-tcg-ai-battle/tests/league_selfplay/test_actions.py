from __future__ import annotations

import math

import numpy as np
import pytest

from league_selfplay.actions import action_log_probability, sample_action


def test_exact_two_samples_distinct_options_without_stop() -> None:
    sample = sample_action(
        np.array([3.0, 2.0, 1.0]),
        stop_logit=100.0,
        min_count=2,
        max_count=2,
        rng=np.random.default_rng(7),
    )

    assert len(sample.indices) == 2
    assert len(set(sample.indices)) == 2


def test_variable_selection_stops_after_the_minimum() -> None:
    sample = sample_action(
        np.array([-9.0, -9.0]),
        stop_logit=30.0,
        min_count=1,
        max_count=2,
        rng=np.random.default_rng(8),
    )

    assert len(sample.indices) == 1


def test_empty_selection_is_legal_when_minimum_is_zero() -> None:
    sample = sample_action(
        np.array([-30.0, -30.0]),
        stop_logit=30.0,
        min_count=0,
        max_count=2,
        rng=np.random.default_rng(9),
    )

    assert sample.indices == ()


def test_exact_action_log_probability_matches_hand_calculation() -> None:
    log_probability = action_log_probability(
        (0, 1),
        np.array([0.0, 0.0]),
        stop_logit=50.0,
        min_count=2,
        max_count=2,
    )

    assert log_probability == pytest.approx(-math.log(2.0))


def test_variable_action_probability_includes_terminal_stop() -> None:
    log_probability = action_log_probability(
        (0,),
        np.array([0.0, 0.0]),
        stop_logit=0.0,
        min_count=1,
        max_count=2,
    )

    assert log_probability == pytest.approx(math.log(0.25))


def test_sampled_probability_recomputes_exactly() -> None:
    logits = np.array([0.5, 1.0, -0.2], dtype=np.float64)
    sample = sample_action(
        logits,
        stop_logit=-0.1,
        min_count=1,
        max_count=3,
        rng=np.random.default_rng(10),
    )

    recomputed = action_log_probability(
        sample.indices,
        logits,
        stop_logit=-0.1,
        min_count=1,
        max_count=3,
    )

    assert recomputed == pytest.approx(sample.log_probability, abs=1e-12)


@pytest.mark.parametrize(
    ("indices", "min_count", "max_count", "message"),
    [
        ((0,), -1, 1, "invalid selection bounds"),
        ((0,), 2, 1, "invalid selection bounds"),
        ((0,), 1, 3, "invalid selection bounds"),
        ((0, 0), 1, 2, "distinct"),
        ((2,), 1, 2, "out of range"),
    ],
)
def test_invalid_actions_are_rejected(
    indices: tuple[int, ...],
    min_count: int,
    max_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        action_log_probability(
            indices,
            np.array([0.0, 0.0]),
            stop_logit=0.0,
            min_count=min_count,
            max_count=max_count,
        )
