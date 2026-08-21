from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import sys
from pathlib import Path

from league_selfplay.contracts import MemberId
from league_selfplay.residual_runner import (
    _paired_search_rollout,
    decide_residual_proof,
    decide_survivor_confirmation,
    search_pairing_smoke,
    select_promoted_members,
    training_seat_schedule,
)


PROJECT = Path(__file__).resolve().parents[2]
TEST_PYTHON = sys.executable


@dataclass(frozen=True)
class Update:
    parameter_delta_l2: float = 1.0
    all_finite: bool = True


@dataclass
class Observation:
    current: dict
    select: dict
    logs: list
    search_begin_input: None = None


@dataclass
class SearchState:
    observation: Observation
    searchId: int


class FirstOptionActor:
    def decide(self, observation):
        return [0]


def _updates() -> dict[MemberId, Update]:
    return {member: Update() for member in MemberId}


def test_training_schedule_covers_every_pair_in_both_seats() -> None:
    schedule = training_seat_schedule()

    assert len(schedule) == 12
    assert all(
        (second, first) in schedule
        for first, second in schedule
    )
    assert {
        member: sum(member in game for game in schedule)
        for member in MemberId
    } == {member: 6 for member in MemberId}


def test_gate_requires_a_real_override_and_positive_group_delta() -> None:
    no_override = decide_residual_proof(0.05, 0, _updates(), ())
    regression = decide_residual_proof(-0.01, 4, _updates(), ())
    improvement = decide_residual_proof(0.01, 4, _updates(), ())

    assert no_override.code == "REJECT_NO_OVERRIDE"
    assert regression.code == "REJECT_NO_GROUP_IMPROVEMENT"
    assert improvement.code == "PASS_DRIVER_BACKED_MAC"
    assert improvement.passed is True


def test_gate_rejects_nonfinite_or_missing_updates() -> None:
    missing = _updates()
    missing.pop(MemberId.ALAKAZAM)
    invalid = _updates()
    invalid[MemberId.CRUSTLE] = Update(parameter_delta_l2=0.0)

    assert decide_residual_proof(0.1, 2, missing, ()).code == "REJECT_INVALID_UPDATE"
    assert decide_residual_proof(0.1, 2, invalid, ()).code == "REJECT_INVALID_UPDATE"
    assert decide_residual_proof(0.1, 2, _updates(), ("boom",)).code == "REJECT_FAILURE"


def test_only_strictly_better_members_are_promoted() -> None:
    promoted = select_promoted_members(
        {
            MemberId.GRIMMSNARL: -0.25,
            MemberId.LUCARIO: 0.125,
            MemberId.CRUSTLE: 0.0,
            MemberId.ALAKAZAM: -0.5,
        }
    )

    assert promoted == (MemberId.LUCARIO,)


def test_confirmation_rejects_when_candidate_round_has_no_survivor() -> None:
    decision = decide_survivor_confirmation(
        (),
        group_delta=0.1,
        learned_overrides=2,
        updates=_updates(),
        failures=(),
    )

    assert decision.code == "REJECT_NO_PROMOTION"
    assert decision.passed is False


def test_search_pairing_replays_identical_hidden_games_for_aa_control(
    official_cg: Path,
) -> None:
    code = """
import json
from pathlib import Path
from league_selfplay.residual_runner import search_pairing_smoke

print(json.dumps(search_pairing_smoke(Path.cwd()), sort_keys=True))
"""
    result = subprocess.run(
        [TEST_PYTHON, "-c", code],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    report = json.loads(result.stdout)

    assert report["records"] == 8
    assert report["overrides"] == 0
    assert report["all_paired_equal"] is True
    assert report["delta"] == 0.0


def test_paired_search_scores_two_nonfinishing_identical_branches_as_draws() -> None:
    root = SearchState(
        Observation(
            current={"result": -1, "yourIndex": 0},
            select={"option": [{"type": 1}], "minCount": 1, "maxCount": 1},
            logs=[],
        ),
        1,
    )

    scores = _paired_search_rollout(
        root,
        (FirstOptionActor(), FirstOptionActor()),
        (FirstOptionActor(), FirstOptionActor()),
        member_seat=0,
        start_rng=None,
        final_rng=None,
        search_step=lambda search_id, action: root,
        search_release=lambda search_id: None,
        max_steps=1,
    )

    assert scores == (0.5, 0.5)
