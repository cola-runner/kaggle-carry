from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rolling_policy.constants import EXACT_DECK_FINGERPRINT
from rolling_policy.schema import EpisodeRecord, Split, TeacherSubmission
from rolling_policy.snapshot import (
    assign_split,
    balanced_episode_weights,
    eligible_teachers,
    parse_leaderboard,
    parse_submission_rows,
    rank_ten_threshold,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_rank_ten_threshold_reads_rank_ten_not_row_count() -> None:
    rows = parse_leaderboard(FIXTURES / "leaderboard.csv")
    assert rank_ten_threshold(rows) == 1135.0


def test_rank_ten_threshold_fails_when_rank_is_missing() -> None:
    with pytest.raises(ValueError, match="rank 10"):
        rank_ten_threshold([{"Rank": "9", "Score": "1136.9"}])


def test_chronological_split_boundaries_are_frozen() -> None:
    cutoff = datetime(2026, 7, 28, 16, tzinfo=timezone.utc)
    assert assign_split(cutoff - timedelta(hours=72), cutoff) is Split.TRAIN
    assert assign_split(cutoff - timedelta(hours=24), cutoff) is Split.VALIDATION
    assert assign_split(cutoff - timedelta(hours=12), cutoff) is Split.HOLDOUT
    assert assign_split(cutoff, cutoff) is Split.HOLDOUT
    assert assign_split(cutoff - timedelta(hours=72, microseconds=1), cutoff) is None
    assert assign_split(cutoff + timedelta(microseconds=1), cutoff) is None


def test_parse_submission_rows_preserves_active_submission_score_and_time() -> None:
    rows = parse_submission_rows(
        [
            {
                "id": "55001357",
                "dateSubmitted": "2026-07-26 12:41:05.657000",
                "publicScore": "1153.4",
            }
        ]
    )
    assert rows["55001357"].score == 1153.4
    assert rows["55001357"].submitted_at_utc == datetime(
        2026, 7, 26, 12, 41, 5, 657000, tzinfo=timezone.utc
    )


def _teacher(
    team_id: str,
    submission_id: str,
    score: float,
    *,
    fingerprint: str = EXACT_DECK_FINGERPRINT,
    tracked: bool = True,
) -> TeacherSubmission:
    return TeacherSubmission(
        team_id=team_id,
        team_name=f"team-{team_id}",
        submission_id=submission_id,
        score=score,
        deck_fingerprint=fingerprint,
        tracked_at_cutoff=tracked,
        submitted_at_utc=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )


def test_teacher_eligibility_retains_two_agents_but_counts_team_once() -> None:
    candidates = [
        _teacher("dominic", "55001357", 1153.4),
        _teacher("dominic", "54989332", 1136.8),
        _teacher("liam", "55011514", 1152.1),
        _teacher("insuper", "55035974", 1135.0),
        _teacher("sekkat", "54968369", 1127.9),
        _teacher("wrong-deck", "55000001", 1200.0, fingerprint="0" * 12),
        _teacher("inactive", "55000002", 1200.0, tracked=False),
    ]
    eligible = eligible_teachers(
        candidates,
        rank_ten_score=1135.0,
        active_submission_ids={
            "55001357",
            "54989332",
            "55011514",
            "55035974",
            "54968369",
            "55000001",
            "55000002",
        },
    )
    assert [row.submission_id for row in eligible] == [
        "54989332",
        "55001357",
        "55011514",
        "55035974",
    ]
    assert {row.team_id for row in eligible} == {"dominic", "liam", "insuper"}


def test_teacher_eligibility_fails_below_three_distinct_teams() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        eligible_teachers(
            [
                _teacher("dominic", "55001357", 1153.4),
                _teacher("dominic", "54989332", 1136.8),
                _teacher("liam", "55011514", 1152.1),
            ],
            rank_ten_score=1135.0,
            active_submission_ids={"55001357", "54989332", "55011514"},
        )


def test_direct_teacher_mode_explicitly_allows_one_team() -> None:
    eligible = eligible_teachers(
        [_teacher("16531269", "55002825", 1173.5)],
        rank_ten_score=1135.0,
        active_submission_ids={"55002825"},
        minimum_teacher_teams=1,
    )
    assert [row.submission_id for row in eligible] == ["55002825"]


def test_teacher_eligibility_rejects_nonpositive_team_minimum() -> None:
    with pytest.raises(ValueError, match="minimum_teacher_teams"):
        eligible_teachers(
            [_teacher("16531269", "55002825", 1173.5)],
            rank_ten_score=1135.0,
            active_submission_ids={"55002825"},
            minimum_teacher_teams=0,
        )


def _episode(
    episode_id: str,
    team_id: str,
    seat: int,
    created: datetime,
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=episode_id,
        submission_id="55001357",
        team_id=team_id,
        create_time_utc=created,
        end_time_utc=created + timedelta(minutes=1),
        target_seat=seat,
        split=Split.TRAIN,
        replay_sha256="a" * 64,
    )


def test_episode_weights_balance_team_seat_and_twelve_hour_bucket() -> None:
    start = datetime(2026, 7, 26, tzinfo=timezone.utc)
    episodes = [
        _episode("1", "dominic", 0, start),
        _episode("2", "dominic", 0, start + timedelta(hours=1)),
        _episode("3", "liam", 1, start),
        _episode("4", "liam", 1, start + timedelta(hours=13)),
    ]
    weights = balanced_episode_weights(episodes)
    totals: dict[tuple[str, int, int], float] = defaultdict(float)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for row in episodes:
        bucket = int((row.create_time_utc - epoch).total_seconds() // (12 * 3600))
        totals[(row.team_id, row.target_seat, bucket)] += weights[row.episode_id]
    assert set(totals.values()) == {1.0}
    assert weights["1"] == 0.5
    assert weights["2"] == 0.5
