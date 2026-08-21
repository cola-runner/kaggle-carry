from __future__ import annotations

from types import SimpleNamespace

import pytest

from league_selfplay.contracts import (
    GameProvenance,
    GameSource,
    InvalidSelfPlay,
    MemberId,
)
from league_selfplay.evaluation import (
    JudgeResult,
    compare_groups,
    decide_validation,
    validate_training_isolation,
)
from league_selfplay.schedule import JUDGES


def _judge_results(
    rate: float | dict[MemberId, float], repeats_per_key: int = 25
) -> list[JudgeResult]:
    results: list[JudgeResult] = []
    for member in MemberId:
        member_results = [
            (judge, seat)
            for judge in JUDGES
            for seat in (0, 1)
            for _ in range(repeats_per_key)
        ]
        member_rate = rate[member] if isinstance(rate, dict) else rate
        wins = round(member_rate * len(member_results))
        results.extend(
            JudgeResult(member, judge, seat, 1.0 if index < wins else 0.0)
            for index, (judge, seat) in enumerate(member_results)
        )
    return results


def _ancestry(rate: float, games_per_member: int = 100) -> list[JudgeResult]:
    return [
        JudgeResult(
            member,
            "round_0_population",
            index % 2,
            1.0 if index < round(rate * games_per_member) else 0.0,
        )
        for member in MemberId
        for index in range(games_per_member)
    ]


def _finite_updates() -> dict[MemberId, SimpleNamespace]:
    return {
        member: SimpleNamespace(parameter_delta_l2=1.0, all_finite=True)
        for member in MemberId
    }


def test_clear_group_improvement_passes() -> None:
    comparison = compare_groups(
        _judge_results(0.40),
        _judge_results(0.58),
        _ancestry(0.60),
        seed=6,
    )

    decision = decide_validation(comparison, _finite_updates(), [])

    assert comparison.paired_randomness is False
    assert comparison.delta == pytest.approx(0.18)
    assert comparison.confidence_low > 0
    assert decision.code == "PASS_MAC_LEAGUE"
    assert decision.passed is True


def test_equal_group_is_rejected() -> None:
    comparison = compare_groups(
        _judge_results(0.50),
        _judge_results(0.50),
        _ancestry(0.50),
        seed=7,
    )

    decision = decide_validation(comparison, _finite_updates(), [])

    assert decision.code == "REJECT_NO_GROUP_IMPROVEMENT"
    assert decision.passed is False


def test_any_judge_trajectory_is_invalid() -> None:
    records = [
        GameProvenance.current_game(MemberId.GRIMMSNARL, MemberId.LUCARIO),
        GameProvenance.current_game(MemberId.CRUSTLE, MemberId.ALAKAZAM),
        GameProvenance(
            source=GameSource.CURRENT_VS_FIXED,
            actors=(MemberId.GRIMMSNARL.value, JUDGES[0]),
            trajectory_members=(MemberId.GRIMMSNARL,),
            update_members=(MemberId.GRIMMSNARL,),
        ),
    ]

    with pytest.raises(InvalidSelfPlay, match="fixed actors"):
        validate_training_isolation(records)


def test_nonfinite_or_missing_member_update_is_rejected() -> None:
    comparison = compare_groups(
        _judge_results(0.40),
        _judge_results(0.58),
        _ancestry(0.60),
        seed=8,
    )
    updates = _finite_updates()
    updates.pop(MemberId.ALAKAZAM)

    decision = decide_validation(comparison, updates, [])

    assert decision.code == "REJECT_INVALID_UPDATE"


def test_one_member_cannot_be_sacrificed_for_the_group_average() -> None:
    comparison = compare_groups(
        _judge_results(0.40),
        _judge_results(
            {
                MemberId.GRIMMSNARL: 0.75,
                MemberId.LUCARIO: 0.75,
                MemberId.CRUSTLE: 0.75,
                MemberId.ALAKAZAM: 0.25,
            }
        ),
        _ancestry(0.60),
        seed=9,
    )

    decision = decide_validation(comparison, _finite_updates(), [])

    assert comparison.delta > 0.05
    assert comparison.confidence_low > 0
    assert decision.code == "REJECT_MEMBER_COLLAPSE"
