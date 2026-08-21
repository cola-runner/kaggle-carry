from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rolling_policy.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from rolling_policy.schema import (
    EpisodeRecord,
    Split,
    TeacherSubmission,
    parse_utc_datetime,
)


def test_canonical_json_is_sorted_compact_utf8() -> None:
    value = {"é": "雪", "a": [2, 1]}
    assert canonical_json_bytes(value) == '{"a":[2,1],"é":"雪"}'.encode()


def test_sha256_helpers_return_lowercase_full_digest(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_bytes(b"abc") == expected
    assert sha256_file(path) == expected


def test_parse_utc_datetime_normalizes_naive_kaggle_timestamp() -> None:
    parsed = parse_utc_datetime("2026-07-28 15:18:45.408000")
    assert parsed == datetime(2026, 7, 28, 15, 18, 45, 408000, tzinfo=timezone.utc)


def test_parse_utc_datetime_normalizes_offset_timestamp() -> None:
    parsed = parse_utc_datetime("2026-07-28T23:18:45+08:00")
    assert parsed == datetime(2026, 7, 28, 15, 18, 45, tzinfo=timezone.utc)


def test_teacher_submission_rejects_invalid_identity_and_score() -> None:
    with pytest.raises(ValueError, match="submission_id"):
        TeacherSubmission(
            team_id="team-1",
            team_name="Teacher",
            submission_id="not-numeric",
            score=1158.8,
            deck_fingerprint="596d58fc1fbd",
            tracked_at_cutoff=True,
        )
    with pytest.raises(ValueError, match="score"):
        TeacherSubmission(
            team_id="team-1",
            team_name="Teacher",
            submission_id="55001357",
            score=float("nan"),
            deck_fingerprint="596d58fc1fbd",
            tracked_at_cutoff=True,
        )


def test_episode_record_requires_utc_seat_and_sha256() -> None:
    started = datetime(2026, 7, 28, 15, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="target_seat"):
        EpisodeRecord(
            episode_id="88642720",
            submission_id="55001357",
            team_id="team-1",
            create_time_utc=started,
            end_time_utc=started,
            target_seat=2,
            split=Split.HOLDOUT,
            replay_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="replay_sha256"):
        EpisodeRecord(
            episode_id="88642720",
            submission_id="55001357",
            team_id="team-1",
            create_time_utc=started,
            end_time_utc=started,
            target_seat=0,
            split=Split.HOLDOUT,
            replay_sha256="ABC",
        )


def test_episode_record_round_trip_preserves_typed_values() -> None:
    record = EpisodeRecord(
        episode_id="88642720",
        submission_id="55001357",
        team_id="team-1",
        create_time_utc=datetime(2026, 7, 28, 15, tzinfo=timezone.utc),
        end_time_utc=datetime(2026, 7, 28, 15, 2, tzinfo=timezone.utc),
        target_seat=1,
        split=Split.HOLDOUT,
        replay_sha256="a" * 64,
    )
    assert EpisodeRecord.from_dict(record.to_dict()) == record
