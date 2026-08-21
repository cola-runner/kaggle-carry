from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rolling_policy.constants import EXACT_DECK, EXACT_DECK_FINGERPRINT, deck_fingerprint
from rolling_policy.replays import (
    MirrorReplayError,
    RateLimiter,
    ReplayRateLimitError,
    ReplaySource,
    build_replay_records,
    collect_replay_sources,
    decks_from_replay,
    download_replay,
    replay_episode_id,
    replay_target_seat,
    verify_replay,
)
from rolling_policy.schema import ReplayRecord, Split, TeacherSubmission


FIXTURES = Path(__file__).parent / "fixtures"


def test_replay_decks_are_extracted_from_visualization() -> None:
    episode = json.loads((FIXTURES / "replay_minimal.json").read_text())
    decks = decks_from_replay(episode)
    assert len(decks) == 2
    assert len(decks[0]) == 60
    assert deck_fingerprint(decks[0]) == EXACT_DECK_FINGERPRINT
    assert deck_fingerprint(tuple(reversed(decks[0]))) == EXACT_DECK_FINGERPRINT


def test_replay_episode_id_prefers_official_info() -> None:
    episode = json.loads((FIXTURES / "replay_minimal.json").read_text())
    episode["id"] = "wrong-local-id"
    assert replay_episode_id(episode) == "88642720"


def test_target_seat_requires_team_identity_and_exact_deck() -> None:
    episode = json.loads((FIXTURES / "replay_minimal.json").read_text())
    assert replay_target_seat(
        episode,
        target_fingerprint=EXACT_DECK_FINGERPRINT,
        expected_team_name="Dominic Peel",
    ) == 0
    with pytest.raises(ValueError, match="exact deck"):
        replay_target_seat(
            episode,
            target_fingerprint=EXACT_DECK_FINGERPRINT,
            expected_team_name="Opponent",
        )


def test_mirror_replay_uses_team_identity_instead_of_first_matching_seat() -> None:
    episode = json.loads((FIXTURES / "replay_minimal.json").read_text())
    episode["steps"][0][0]["visualize"][0]["action"][1] = list(EXACT_DECK)
    assert replay_target_seat(
        episode,
        target_fingerprint=EXACT_DECK_FINGERPRINT,
        expected_team_name="Opponent",
    ) == 1
    episode["info"]["TeamNames"] = ["Same Team", "Same Team"]
    with pytest.raises(MirrorReplayError, match="team identity"):
        replay_target_seat(
            episode,
            target_fingerprint=EXACT_DECK_FINGERPRINT,
            expected_team_name="Same Team",
        )


def test_verify_replay_rejects_mismatched_episode_and_malformed_json(tmp_path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_bytes((FIXTURES / "replay_minimal.json").read_bytes())
    with pytest.raises(ValueError, match="episode ID"):
        verify_replay(
            replay,
            expected_episode_id="1",
            exact_deck_fingerprint=EXACT_DECK_FINGERPRINT,
            expected_team_name="Dominic Peel",
        )
    replay.write_text("{")
    with pytest.raises(ValueError, match="valid JSON"):
        verify_replay(
            replay,
            expected_episode_id="88642720",
            exact_deck_fingerprint=EXACT_DECK_FINGERPRINT,
            expected_team_name="Dominic Peel",
        )


def test_download_replay_retries_and_publishes_only_verified_filename(tmp_path) -> None:
    attempts = 0

    def flaky_runner(command: list[str], cwd: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise subprocess.CalledProcessError(1, command)
        output_dir = Path(command[command.index("-p") + 1])
        (output_dir / "episode-88642720-replay.json").write_bytes(
            (FIXTURES / "replay_minimal.json").read_bytes()
        )

    destination = tmp_path / "replays" / "88642720.json"
    result = download_replay(
        "88642720",
        destination,
        retries=4,
        runner=flaky_runner,
        retry_delays=(0.0, 0.0, 0.0, 0.0),
    )
    assert result == destination
    assert attempts == 3
    assert destination.exists()
    assert not list(destination.parent.glob("*.partial"))


def test_shared_rate_limiter_spaces_attempts_across_workers() -> None:
    now = [0.0]
    observed_sleeps: list[float] = []

    def monotonic() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        observed_sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(
        minimum_interval_seconds=1.0,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    limiter.wait()
    limiter.wait()
    limiter.wait()
    assert observed_sleeps == [1.0, 1.0]
    assert now[0] == 2.0


def test_shared_rate_limiter_can_defer_every_worker_after_429() -> None:
    now = [10.0]
    observed_sleeps: list[float] = []

    def monotonic() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        observed_sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(
        minimum_interval_seconds=1.0,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    limiter.defer(30.0)
    limiter.wait()
    assert observed_sleeps == [30.0]
    assert now[0] == 40.0


def test_rate_limit_error_uses_long_retry_delay(tmp_path) -> None:
    attempts = 0
    observed_sleeps: list[float] = []
    observed_global_defers: list[float] = []

    def rate_limited_once(command: list[str], cwd: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ReplayRateLimitError("429 Too Many Requests")
        output_dir = Path(command[command.index("-p") + 1])
        (output_dir / "episode-88642720-replay.json").write_bytes(
            (FIXTURES / "replay_minimal.json").read_bytes()
        )

    destination = tmp_path / "replays" / "88642720.json"
    download_replay(
        "88642720",
        destination,
        retries=2,
        runner=rate_limited_once,
        retry_delays=(1.0, 2.0),
        rate_limit_delays=(30.0, 60.0),
        sleeper=observed_sleeps.append,
        on_rate_limit=observed_global_defers.append,
    )
    assert observed_sleeps == [30.0]
    assert observed_global_defers == [30.0]
    assert attempts == 2


def test_verify_replay_returns_stable_hash_and_target_seat(tmp_path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_bytes((FIXTURES / "replay_minimal.json").read_bytes())
    verified = verify_replay(
        replay,
        expected_episode_id="88642720",
        exact_deck_fingerprint=EXACT_DECK_FINGERPRINT,
        expected_team_name="Dominic Peel",
    )
    assert verified.target_seat == 0
    assert verified.deck_fingerprints[0] == EXACT_DECK_FINGERPRINT
    assert len(verified.replay_sha256) == 64


def test_replay_inventory_record_round_trips_source_identity() -> None:
    record = ReplayRecord(
        snapshot_id="20260728T155631Z",
        episode_id="88642720",
        submission_id="55001357",
        team_id="16514272",
        team_name="Dominic Peel",
        create_time_utc=datetime(2026, 7, 28, 15, tzinfo=timezone.utc),
        end_time_utc=datetime(2026, 7, 28, 15, 2, tzinfo=timezone.utc),
        target_seat=0,
        split=Split.HOLDOUT,
        replay_sha256="a" * 64,
        replay_relpath="replays/88642720.json",
        deck_fingerprint=EXACT_DECK_FINGERPRINT,
    )
    assert ReplayRecord.from_dict(record.to_dict()) == record


def test_collect_replay_sources_filters_window_state_and_type() -> None:
    cutoff = datetime(2026, 7, 28, 16, tzinfo=timezone.utc)
    teacher = TeacherSubmission(
        team_id="16514272",
        team_name="Dominic Peel",
        submission_id="55001357",
        score=1153.4,
        deck_fingerprint=EXACT_DECK_FINGERPRINT,
        tracked_at_cutoff=True,
        submitted_at_utc=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )
    rows = [
        {
            "id": "1",
            "createTime": "2026-07-28 15:00:00",
            "endTime": "2026-07-28 15:01:00",
            "state": "EpisodeState.COMPLETED",
            "type": "EpisodeType.EPISODE_TYPE_PUBLIC",
        },
        {
            "id": "2",
            "createTime": "2026-07-25 15:59:59",
            "endTime": "2026-07-25 16:01:00",
            "state": "EpisodeState.COMPLETED",
            "type": "EpisodeType.EPISODE_TYPE_PUBLIC",
        },
        {
            "id": "3",
            "createTime": "2026-07-28 14:00:00",
            "endTime": "2026-07-28 14:01:00",
            "state": "EpisodeState.RUNNING",
            "type": "EpisodeType.EPISODE_TYPE_PUBLIC",
        },
        {
            "id": "4",
            "createTime": "2026-07-28 14:00:00",
            "endTime": "2026-07-28 14:01:00",
            "state": "EpisodeState.COMPLETED",
            "type": "EpisodeType.EPISODE_TYPE_PRIVATE",
        },
    ]
    sources = collect_replay_sources(
        snapshot_id="20260728T160000Z",
        cutoff_utc=cutoff,
        teachers=[teacher],
        episode_rows_by_submission={"55001357": rows},
    )
    assert [source.episode_id for source in sources] == ["1"]
    assert sources[0].split is Split.HOLDOUT
    assert sources[0].team_name == "Dominic Peel"


def test_build_replay_records_binds_verified_replay_to_each_source(tmp_path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    replay_path = snapshot_dir / "replays" / "88642720.json"
    replay_path.parent.mkdir(parents=True)
    replay_path.write_bytes((FIXTURES / "replay_minimal.json").read_bytes())
    source = ReplaySource(
        snapshot_id="20260728T155631Z",
        episode_id="88642720",
        submission_id="55001357",
        team_id="16514272",
        team_name="Dominic Peel",
        create_time_utc=datetime(2026, 7, 28, 15, tzinfo=timezone.utc),
        end_time_utc=datetime(2026, 7, 28, 15, 2, tzinfo=timezone.utc),
        split=Split.HOLDOUT,
    )
    records = build_replay_records(
        [source],
        replay_path=replay_path,
        snapshot_dir=snapshot_dir,
        exact_deck_fingerprint=EXACT_DECK_FINGERPRINT,
    )
    assert len(records) == 1
    assert records[0].target_seat == 0
    assert records[0].replay_relpath == "replays/88642720.json"
    assert records[0].replay_sha256 == records[0].to_dict()["replay_sha256"]
