from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from league_selfplay.contracts import MemberId
from league_selfplay.residual import DriverBackedActor, mixture_log_probability


PROJECT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT / "tests/rolling/fixtures/visible_observation.json"


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: list[float], stop_logit: float = -20.0) -> None:
        super().__init__()
        self.logits = logits
        self.stop_logit = stop_logit

    def forward(self, features, mask):
        batch, option_count, _ = features.shape
        logits = torch.tensor(self.logits[:option_count], dtype=torch.float32)
        options = logits.repeat(batch, 1)
        stop = torch.full((batch,), self.stop_logit, dtype=torch.float32)
        values = torch.zeros(batch, dtype=torch.float32)
        return options, stop, values


class ExplodingModel(torch.nn.Module):
    def forward(self, features, mask):
        raise AssertionError("untouched driver evaluation must not call the model")


def _observation() -> dict:
    observation = json.loads(FIXTURE.read_text())
    observation["select"]["minCount"] = 1
    observation["select"]["maxCount"] = 1
    observation["select"]["option"] = observation["select"]["option"][:2]
    return observation


def test_zero_exploration_executes_exact_driver_action() -> None:
    actor = DriverBackedActor(
        MemberId.GRIMMSNARL,
        FixedLogitModel([10.0, -10.0]),
        [1] * 60,
        "cpu",
        driver_action=lambda observation: [1],
        exploration_rate=0.0,
    )

    action, step = actor.decide(_observation(), 0, np.random.default_rng(7))

    assert action == [1]
    assert step is not None
    assert step.baseline_action == (1,)
    assert step.exploration_rate == 0.0
    assert step.old_log_probability == pytest.approx(0.0)
    assert actor.counters.driver_actions == 1
    assert actor.counters.exploration_actions == 0


def test_training_exploration_records_finite_mixture_probability() -> None:
    actor = DriverBackedActor(
        MemberId.LUCARIO,
        FixedLogitModel([20.0, -20.0]),
        [2] * 60,
        "cpu",
        driver_action=lambda observation: [1],
        exploration_rate=1.0,
    )

    action, step = actor.decide(_observation(), 0, np.random.default_rng(8))

    assert action == [0]
    assert step is not None
    assert step.action == (0,)
    assert step.baseline_action == (1,)
    assert math.isfinite(step.old_log_probability)
    assert actor.counters.exploration_actions == 1


def test_evaluation_overrides_only_above_margin() -> None:
    weak = DriverBackedActor(
        MemberId.CRUSTLE,
        FixedLogitModel([1.9, 0.0]),
        [3] * 60,
        "cpu",
        driver_action=lambda observation: [1],
        trainable=False,
        overrides_enabled=True,
        override_margin=2.0,
    )
    strong = DriverBackedActor(
        MemberId.CRUSTLE,
        FixedLogitModel([2.1, 0.0]),
        [3] * 60,
        "cpu",
        driver_action=lambda observation: [1],
        trainable=False,
        overrides_enabled=True,
        override_margin=2.0,
    )

    weak_action, weak_step = weak.decide(
        _observation(), 0, np.random.default_rng(9)
    )
    strong_action, strong_step = strong.decide(
        _observation(), 0, np.random.default_rng(9)
    )

    assert weak_action == [1]
    assert strong_action == [0]
    assert weak_step is None and strong_step is None
    assert weak.counters.evaluation_overrides == 0
    assert strong.counters.evaluation_overrides == 1


def test_action_value_override_is_main_phase_only() -> None:
    actor = DriverBackedActor(
        MemberId.LUCARIO,
        FixedLogitModel([0.8, 0.0]),
        [2] * 60,
        "cpu",
        driver_action=lambda observation: [1],
        trainable=False,
        overrides_enabled=True,
        override_margin=0.25,
        override_mode="action_value",
    )
    main = _observation()
    main["select"]["context"] = 0
    followup = _observation()
    followup["select"]["context"] = 7

    main_action, _ = actor.decide(main, 0, np.random.default_rng(11))
    followup_action, _ = actor.decide(
        followup,
        0,
        np.random.default_rng(11),
    )

    assert main_action == [0]
    assert followup_action == [1]
    assert actor.counters.evaluation_overrides == 1


def test_untouched_driver_evaluation_skips_neural_model() -> None:
    actor = DriverBackedActor(
        MemberId.ALAKAZAM,
        ExplodingModel(),
        [4] * 60,
        "cpu",
        driver_action=lambda observation: [1],
        trainable=False,
        overrides_enabled=False,
    )

    action, step = actor.decide(_observation(), 0, np.random.default_rng(10))

    assert action == [1]
    assert step is None


def test_mixture_probability_matches_hand_calculation() -> None:
    value = mixture_log_probability(
        (1,),
        (1,),
        np.array([0.0, 0.0]),
        stop_logit=-20.0,
        min_count=1,
        max_count=1,
        exploration_rate=0.1,
    )

    assert value == pytest.approx(math.log(0.95))
