from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np

from league_selfplay.contracts import MemberId
from league_selfplay.single_intervention_runner import (
    ActualComparison,
    PopulationPolicy,
    ProofDependencies,
    collect_calibration,
    collect_interventions,
    decide_group_upgrade_proof,
    decide_single_intervention_proof,
    run_actual_game,
    run_single_intervention_proof,
    select_confirmed_survivors,
    select_survivors,
)
from league_selfplay.single_intervention import (
    CalibrationCell,
    CalibrationKey,
    create_intervention_population,
)


class FakeBattleGame:
    def __init__(self, decisions: int = 5, winner: int = 0) -> None:
        self.decision_limit = decisions
        self.winner = winner
        self.actions: list[list[int]] = []
        self.starts = 0
        self.finishes = 0
        self._decision = 0

    def _observation(self, result: int = -1) -> dict:
        return {
            "current": {
                "yourIndex": self._decision % 2,
                "result": result,
                "turn": self._decision,
            },
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 14}, {"type": 14}],
            },
        }

    def battle_start(self, deck0: list[int], deck1: list[int]):
        self.starts += 1
        self._decision = 0
        return self._observation(), object()

    def battle_select(self, action: list[int]) -> dict:
        self.actions.append(list(action))
        self._decision += 1
        if self._decision >= self.decision_limit:
            return self._observation(self.winner)
        return self._observation()

    def battle_finish(self) -> None:
        self.finishes += 1


class RecordingPolicy:
    def deck(self, member: MemberId) -> list[int]:
        return [int(tuple(MemberId).index(member))] * 60

    def decide(self, member: MemberId, observation: dict) -> list[int]:
        return [0]

    def action(self, member: MemberId, observation: dict) -> list[int]:
        return self.decide(member, observation)


def test_actual_game_changes_only_target_action_then_returns_to_incumbent() -> None:
    game = FakeBattleGame(decisions=5, winner=0)
    result = run_actual_game(
        game,
        RecordingPolicy(),
        members=(MemberId.GRIMMSNARL, MemberId.LUCARIO),
        experimental_member=MemberId.GRIMMSNARL,
        target_ordinal=2,
        trial_selector=lambda features, incumbent, rng: 1,
        rng=np.random.default_rng(1),
    )
    assert result.intervention is not None
    assert result.intervention.incumbent_index == 0
    assert result.intervention.trial_index == 1
    assert result.intervention.actual_score == 1.0
    assert game.actions == [[0], [0], [1], [0], [0]]
    assert sum(action == [1] for action in game.actions) == 1
    assert game.finishes == 1


def test_game_without_target_is_control_not_training_label() -> None:
    result = run_actual_game(
        FakeBattleGame(decisions=2, winner=1),
        RecordingPolicy(),
        members=(MemberId.CRUSTLE, MemberId.ALAKAZAM),
        experimental_member=MemberId.CRUSTLE,
        target_ordinal=32,
        trial_selector=lambda features, incumbent, rng: 1,
        rng=np.random.default_rng(2),
    )
    assert result.intervention is None
    assert result.control


def test_calibration_runs_exact_schedule_and_freezes_two_games_per_cell() -> None:
    game = FakeBattleGame(decisions=2, winner=0)
    batch = collect_calibration(
        game,
        RecordingPolicy(),
        np.random.default_rng(3),
    )
    assert batch.games == 48
    assert game.starts == 48
    assert game.finishes == 48
    assert len(batch.cells) == 24
    assert all(cell.games == 2 for cell in batch.cells.values())
    assert all(len(rows) <= 2048 for rows in batch.decisions.values())


def test_intervention_collection_gets_32_balanced_real_games_per_member() -> None:
    game = FakeBattleGame(decisions=66, winner=0)
    population = create_intervention_population(4, "cpu")
    policy = PopulationPolicy(
        registry=RecordingPolicy(),  # type: ignore[arg-type]
        ensembles=population,
        enabled_members=frozenset(),
    )
    calibration = {
        CalibrationKey(member, opponent, seat): CalibrationCell(1.0, 2)
        for member in MemberId
        for opponent in MemberId
        if member is not opponent
        for seat in (0, 1)
    }
    batch = collect_interventions(
        game,
        policy,
        calibration,
        1,
        np.random.default_rng(5),
    )
    assert batch.games == 128
    assert batch.controls == 0
    for member, rows in batch.examples.items():
        assert len(rows) == 32
        counts = Counter((row.opponent, row.seat) for row in rows)
        assert sorted(counts.values()) == [5, 5, 5, 5, 6, 6]
        assert all(row.member is member for row in rows)


def test_runner_source_does_not_use_counterfactual_engine_calls() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "league_selfplay"
        / "single_intervention_runner.py"
    ).read_text()
    forbidden = ("search_" + name for name in ("begin", "step", "release", "end"))
    assert not any(name in source for name in forbidden)


def _comparison(
    delta: float,
    member_delta: float | None = None,
    games: int = 48,
) -> ActualComparison:
    per_member = {
        member: delta if member_delta is None else member_delta
        for member in MemberId
    }
    return ActualComparison(
        candidate_score=0.5 + delta,
        incumbent_score=0.5,
        delta=delta,
        per_member_delta=per_member,
        candidate_games=games,
        incumbent_games=games,
        same_schedule=True,
        paired_randomness=False,
        candidate_overrides=3,
    )


def test_round_two_uses_only_positive_round_one_survivors() -> None:
    promoted = select_survivors(
        {
            MemberId.GRIMMSNARL: 0.10,
            MemberId.LUCARIO: 0.0,
            MemberId.CRUSTLE: -0.10,
            MemberId.ALAKAZAM: 0.05,
        }
    )
    assert promoted == (MemberId.GRIMMSNARL, MemberId.ALAKAZAM)


def test_proof_requires_positive_selection_and_confirmation() -> None:
    promoted = (MemberId.GRIMMSNARL,)
    passed = decide_single_intervention_proof(
        selection=_comparison(0.05),
        confirmation=_comparison(0.04),
        promoted=promoted,
        overrides=3,
        failures=(),
    )
    assert passed.code == "PASS_SINGLE_INTERVENTION_MAC"
    rejected = decide_single_intervention_proof(
        selection=_comparison(0.05),
        confirmation=_comparison(-0.01),
        promoted=promoted,
        overrides=3,
        failures=(),
    )
    assert rejected.code == "REJECT_NO_GROUP_IMPROVEMENT"


def test_positive_group_with_one_regressing_member_has_precise_rejection() -> None:
    confirmation = _comparison(0.04)
    confirmation.per_member_delta[MemberId.LUCARIO] = -0.10
    decision = decide_single_intervention_proof(
        selection=_comparison(0.05),
        confirmation=confirmation,
        promoted=(MemberId.GRIMMSNARL, MemberId.LUCARIO),
        overrides=3,
        failures=(),
    )
    assert decision.code == "REJECT_MEMBER_REGRESSION"
    assert "lucario" in decision.reasons[0]


def test_confirmation_rolls_back_only_the_regressing_member() -> None:
    screening = _comparison(0.04)
    screening.per_member_delta[MemberId.LUCARIO] = -0.10
    confirmed = select_confirmed_survivors(
        (MemberId.GRIMMSNARL, MemberId.LUCARIO),
        _comparison(0.05),
        screening,
    )
    assert confirmed == (MemberId.GRIMMSNARL,)


def test_group_proof_uses_large_balanced_audit_not_member_vetoes() -> None:
    comparison = _comparison(0.04, games=192)
    comparison.per_member_delta[MemberId.LUCARIO] = -0.10
    decision = decide_group_upgrade_proof(
        comparison=comparison,
        overrides=3,
        expected_games_per_side=192,
        failures=(),
    )
    assert decision.code == "PASS_GROUP_SELFPLAY_MAC"

    undersized = ActualComparison(
        candidate_score=comparison.candidate_score,
        incumbent_score=comparison.incumbent_score,
        delta=comparison.delta,
        per_member_delta=comparison.per_member_delta,
        candidate_games=48,
        incumbent_games=48,
        same_schedule=True,
        paired_randomness=False,
        candidate_overrides=3,
    )
    assert not decide_group_upgrade_proof(
        comparison=undersized,
        overrides=3,
        expected_games_per_side=192,
        failures=(),
    ).passed


def test_failed_proof_cleans_temp_and_does_not_retain_models(tmp_path: Path) -> None:
    def fail_registry(_root: Path):
        raise RuntimeError("deliberate fake failure")

    report = run_single_intervention_proof(
        tmp_path,
        dependencies=ProofDependencies(
            game_api_factory=lambda _root: FakeBattleGame(),
            registry_factory=fail_registry,
            verify_engine=False,
        ),
        wall_time_seconds=600,
    )
    assert not report.decision.passed
    assert not report.storage_root.exists()
    assert report.artifacts_before == report.artifacts_after
    assert report.retained_population is None
