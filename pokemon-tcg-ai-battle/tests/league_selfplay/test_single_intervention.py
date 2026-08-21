from __future__ import annotations

import numpy as np
import torch

from league_selfplay.action_value import action_value_parameter_count
from league_selfplay.contracts import MemberId
from league_selfplay.single_intervention import (
    CalibrationCell,
    CalibrationKey,
    IncumbentDecision,
    InterventionExample,
    InterventionTracker,
    centered_label,
    calibrated_override_margins,
    choose_trial_index,
    create_intervention_population,
    eligible_intervention,
    mean_option_scores,
    pretrain_incumbent_population,
    trusted_override,
    update_intervention_ensemble,
    update_intervention_population,
)


def _eligible_observation() -> dict:
    return {
        "current": {"yourIndex": 0},
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}, {"type": 14}, {"type": 14}],
        },
    }


def test_calibration_uses_frozen_shrunk_score_and_clipped_label() -> None:
    cell = CalibrationCell(points=2.0, games=2)
    assert cell.expected_score == 0.75
    assert centered_label(1.0, cell) == 0.25
    assert centered_label(0.0, cell) == -0.5


def test_calibration_key_rejects_invalid_seat() -> None:
    try:
        CalibrationKey(MemberId.GRIMMSNARL, MemberId.LUCARIO, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid calibration seat must be rejected")


def test_tracker_allows_exactly_one_targeted_intervention() -> None:
    tracker = InterventionTracker(target_ordinal=2)
    observation = _eligible_observation()
    assert not tracker.consider(observation, [0])
    assert tracker.consider(observation, [0])
    tracker.mark_used()
    assert not tracker.consider(observation, [0])
    assert tracker.eligible_seen == 2


def test_tracker_rejects_invalid_target_and_double_use() -> None:
    for ordinal in (0, 33):
        try:
            InterventionTracker(target_ordinal=ordinal)
        except ValueError:
            pass
        else:
            raise AssertionError("target ordinal outside 1..32 must be rejected")

    tracker = InterventionTracker(target_ordinal=1)
    tracker.mark_used()
    try:
        tracker.mark_used()
    except ValueError:
        pass
    else:
        raise AssertionError("a second intervention must be rejected")


def test_eligibility_requires_nonforced_single_choice_main_action() -> None:
    observation = _eligible_observation()
    assert eligible_intervention(observation, [0])

    for key, value in (
        ("type", 1),
        ("context", 3),
        ("minCount", 0),
        ("maxCount", 2),
    ):
        rejected = _eligible_observation()
        rejected["select"][key] = value
        assert not eligible_intervention(rejected, [0])
    assert not eligible_intervention(observation, [])
    assert not eligible_intervention(observation, [3])


def test_trial_action_is_legal_and_never_the_incumbent() -> None:
    rng = np.random.default_rng(7)
    observed = {
        choose_trial_index(np.asarray([3.0, 2.0, 1.0]), 0, rng)
        for _ in range(100)
    }
    assert observed == {1, 2}


def test_trial_action_rejects_nonfinite_scores_and_invalid_incumbent() -> None:
    rng = np.random.default_rng(7)
    for scores, incumbent in (([1.0, float("nan")], 0), ([1.0], 0), ([1.0, 2.0], 2)):
        try:
            choose_trial_index(np.asarray(scores), incumbent, rng)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid trial selection input must be rejected")


def test_intervention_example_validates_indices_and_label() -> None:
    features = np.zeros((3, 512), dtype=np.float32)
    row = InterventionExample(
        member=MemberId.CRUSTLE,
        opponent=MemberId.ALAKAZAM,
        seat=1,
        round_index=1,
        target_ordinal=4,
        features=features,
        incumbent_index=0,
        trial_index=2,
        label=0.25,
    )
    assert row.trial_index == 2
    assert row.label == 0.25

    try:
        InterventionExample(
            member=MemberId.CRUSTLE,
            opponent=MemberId.ALAKAZAM,
            seat=1,
            round_index=1,
            target_ordinal=4,
            features=features,
            incumbent_index=1,
            trial_index=1,
            label=0.25,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("trial and incumbent must differ")


def _positive_example() -> InterventionExample:
    features = np.zeros((2, 512), dtype=np.float32)
    features[1, 0] = 1.0
    return InterventionExample(
        member=MemberId.LUCARIO,
        opponent=MemberId.CRUSTLE,
        seat=0,
        round_index=1,
        target_ordinal=2,
        features=features,
        incumbent_index=0,
        trial_index=1,
        label=0.5,
    )


def test_population_has_two_independent_small_models_per_member() -> None:
    population = create_intervention_population(20260804, "cpu")
    assert set(population) == set(MemberId)
    for ensemble in population.values():
        assert len(ensemble.models) == 2
        assert (
            sum(action_value_parameter_count(model) for model in ensemble.models)
            < 140_000
        )
        assert (
            ensemble.models[0].layer.weight.data_ptr()
            != ensemble.models[1].layer.weight.data_ptr()
        )


def test_trusted_override_requires_both_gaps_above_margin() -> None:
    assert trusted_override(([0.0, 0.4], [0.0, 0.3]), 0, margin=0.25) == 1
    assert trusted_override(([0.0, 0.4], [0.0, 0.2]), 0, margin=0.25) is None
    assert trusted_override(([0.0, 0.4, 0.5], [0.0, 0.3, 0.1]), 0, margin=0.25) == 1


def test_calibrated_margin_blocks_raw_scale_false_overrides() -> None:
    calibration_scores = [
        (
            np.asarray([0.0, gap], dtype=np.float64),
            np.asarray([0.0, gap + 0.02], dtype=np.float64),
        )
        for gap in np.linspace(-0.2, 0.8, 201)
    ]
    margins = calibrated_override_margins(
        calibration_scores,
        [0] * len(calibration_scores),
        quantile=0.995,
        minimum=0.25,
    )
    assert np.isclose(margins[0], 0.795)
    assert np.isclose(margins[1], 0.815)
    assert trusted_override(([0.0, 0.4], [0.0, 0.4]), 0, margin=margins) is None
    assert trusted_override(([0.0, 1.0], [0.0, 1.0]), 0, margin=margins) == 1


def test_mean_option_scores_averages_two_models() -> None:
    assert np.allclose(
        mean_option_scores(([0.0, 0.4], [0.2, 0.6])),
        np.asarray([0.1, 0.5]),
    )


def test_pairwise_update_moves_positive_trial_above_incumbent() -> None:
    ensemble = create_intervention_population(9, "cpu")[MemberId.LUCARIO]
    rows = [_positive_example() for _ in range(32)]
    before = tuple(
        model.layer.weight.detach().clone() for model in ensemble.models
    )
    stats = update_intervention_ensemble(
        ensemble,
        rows,
        "cpu",
        seed=10,
        epochs=8,
        batch_size=32,
    )
    assert stats.examples == 32
    assert stats.updates == 8
    assert stats.all_finite
    assert all(
        not torch.equal(old, model.layer.weight)
        for old, model in zip(before, ensemble.models, strict=True)
    )
    scores = mean_option_scores(
        tuple(
            model(
                torch.from_numpy(rows[0].features)[None, :, :],
                torch.ones((1, 2), dtype=torch.bool),
            )[0][0]
            .detach()
            .numpy()
            for model in ensemble.models
        )
    )
    assert scores[1] > scores[0]


def _incumbent_decision(member: MemberId) -> IncumbentDecision:
    features = np.zeros((2, 512), dtype=np.float32)
    features[0, 0] = 1.0
    return IncumbentDecision(
        member=member,
        features=features,
        incumbent_index=0,
    )


def test_incumbent_pretraining_updates_both_models_for_every_member() -> None:
    population = create_intervention_population(21, "cpu")
    before = {
        member: tuple(model.layer.weight.detach().clone() for model in ensemble.models)
        for member, ensemble in population.items()
    }
    stats = pretrain_incumbent_population(
        population,
        {
            member: [_incumbent_decision(member) for _ in range(4)]
            for member in MemberId
        },
        "cpu",
        seed=22,
        epochs=2,
    )
    assert set(stats) == set(MemberId)
    assert all(stat.examples == 4 for stat in stats.values())
    assert all(stat.all_finite for stat in stats.values())
    for member, ensemble in population.items():
        assert all(
            not torch.equal(initial, model.layer.weight)
            for initial, model in zip(before[member], ensemble.models, strict=True)
        )


def test_population_update_routes_each_members_own_examples() -> None:
    population = create_intervention_population(31, "cpu")
    rows = {}
    members = tuple(MemberId)
    for index, member in enumerate(members):
        opponent = members[(index + 1) % len(members)]
        base = _positive_example()
        rows[member] = [
            InterventionExample(
                member=member,
                opponent=opponent,
                seat=index % 2,
                round_index=1,
                target_ordinal=2,
                features=base.features.copy(),
                incumbent_index=0,
                trial_index=1,
                label=0.5,
            )
            for _ in range(2)
        ]
    stats = update_intervention_population(
        population,
        rows,
        "cpu",
        seed=32,
        epochs=1,
        batch_size=2,
    )
    assert set(stats) == set(MemberId)
    assert all(stat.examples == 2 for stat in stats.values())
    assert all(stat.updates == 1 for stat in stats.values())
