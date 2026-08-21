from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import deck_fingerprint
from .hashing import sha256_file
from .schema import ReplayRecord, Split, TeacherSubmission, parse_utc_datetime
from .snapshot import assign_split


class MirrorReplayError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedReplay:
    episode_id: str
    target_seat: int
    deck_fingerprints: tuple[str, str]
    replay_sha256: str


@dataclass(frozen=True, slots=True)
class ReplaySource:
    snapshot_id: str
    episode_id: str
    submission_id: str
    team_id: str
    team_name: str
    create_time_utc: datetime
    end_time_utc: datetime
    split: Split


CommandRunner = Callable[[list[str], Path], None]


class ReplayRateLimitError(RuntimeError):
    pass


class RateLimiter:
    def __init__(
        self,
        minimum_interval_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds cannot be negative")
        self.minimum_interval_seconds = minimum_interval_seconds
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._monotonic()
            delay = max(0.0, self._next_allowed - now)
            if delay:
                self._sleeper(delay)
                now = self._monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.minimum_interval_seconds

    def defer(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("defer seconds cannot be negative")
        with self._lock:
            self._next_allowed = max(
                self._next_allowed,
                self._monotonic() + seconds,
            )


def collect_replay_sources(
    *,
    snapshot_id: str,
    cutoff_utc: datetime,
    teachers: Sequence[TeacherSubmission],
    episode_rows_by_submission: Mapping[str, Sequence[Mapping[str, str]]],
) -> list[ReplaySource]:
    sources: list[ReplaySource] = []
    seen: set[tuple[str, str]] = set()
    for teacher in teachers:
        for row in episode_rows_by_submission.get(teacher.submission_id, ()):
            episode_id = str(row.get("id", "")).strip()
            if not episode_id.isdigit():
                continue
            if row.get("state") != "EpisodeState.COMPLETED":
                continue
            if row.get("type") != "EpisodeType.EPISODE_TYPE_PUBLIC":
                continue
            created = parse_utc_datetime(str(row.get("createTime", "")))
            split = assign_split(created, cutoff_utc)
            if split is None:
                continue
            key = (teacher.submission_id, episode_id)
            if key in seen:
                raise ValueError(
                    f"duplicate episode {episode_id} for submission "
                    f"{teacher.submission_id}"
                )
            seen.add(key)
            sources.append(
                ReplaySource(
                    snapshot_id=snapshot_id,
                    episode_id=episode_id,
                    submission_id=teacher.submission_id,
                    team_id=teacher.team_id,
                    team_name=teacher.team_name,
                    create_time_utc=created,
                    end_time_utc=parse_utc_datetime(str(row.get("endTime", ""))),
                    split=split,
                )
            )
    return sorted(
        sources,
        key=lambda source: (
            source.create_time_utc,
            source.episode_id,
            source.submission_id,
        ),
    )


def decks_from_replay(
    episode: Mapping[str, object],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    steps = episode.get("steps")
    if not isinstance(steps, list):
        raise ValueError("replay has no steps")
    for step in steps[:4]:
        if not isinstance(step, list):
            continue
        for record in step[:2]:
            if not isinstance(record, dict):
                continue
            for visual in record.get("visualize") or []:
                action = visual.get("action") if isinstance(visual, dict) else None
                if (
                    isinstance(action, list)
                    and len(action) >= 2
                    and all(
                        isinstance(deck, list)
                        and len(deck) == 60
                        and all(isinstance(card_id, int) for card_id in deck)
                        for deck in action[:2]
                    )
                ):
                    return tuple(action[0]), tuple(action[1])

    decks: list[tuple[int, ...] | None] = [None, None]
    for step in steps[:8]:
        if not isinstance(step, list):
            continue
        for seat, record in enumerate(step[:2]):
            if not isinstance(record, dict):
                continue
            action = record.get("action")
            if (
                isinstance(action, list)
                and len(action) == 60
                and all(isinstance(card_id, int) for card_id in action)
            ):
                decks[seat] = tuple(action)
    if decks[0] is None or decks[1] is None:
        raise ValueError("replay does not expose both 60-card decks")
    return decks[0], decks[1]


def replay_episode_id(episode: Mapping[str, object]) -> str:
    info = episode.get("info")
    value = info.get("EpisodeId") if isinstance(info, dict) else None
    if value is None:
        value = episode.get("id")
    episode_id = str(value or "")
    if not episode_id.isdigit():
        raise ValueError("replay has no numeric episode ID")
    return episode_id


def replay_team_names(episode: Mapping[str, object]) -> tuple[str, str]:
    info = episode.get("info")
    if not isinstance(info, dict):
        raise ValueError("replay has no info")
    names = info.get("TeamNames")
    if isinstance(names, list) and len(names) >= 2:
        return str(names[0]), str(names[1])
    agents = info.get("Agents")
    if isinstance(agents, list) and len(agents) >= 2:
        agent_names = [
            str(agent.get("Name", "")) if isinstance(agent, dict) else ""
            for agent in agents[:2]
        ]
        if all(agent_names):
            return agent_names[0], agent_names[1]
    raise ValueError("replay has no two-seat team identity")


def replay_target_seat(
    episode: Mapping[str, object],
    target_fingerprint: str,
    expected_team_name: str,
) -> int:
    team_names = replay_team_names(episode)
    matching_team_seats = [
        seat for seat, team_name in enumerate(team_names) if team_name == expected_team_name
    ]
    if len(matching_team_seats) > 1:
        raise MirrorReplayError(
            f"team identity {expected_team_name!r} matches both replay seats"
        )
    if not matching_team_seats:
        raise ValueError(f"team identity {expected_team_name!r} is absent from replay")
    seat = matching_team_seats[0]
    decks = decks_from_replay(episode)
    fingerprints = tuple(deck_fingerprint(deck) for deck in decks)
    if fingerprints[seat] != target_fingerprint:
        raise ValueError(
            f"team {expected_team_name!r} does not use the exact deck: "
            f"{fingerprints[seat]} != {target_fingerprint}"
        )
    return seat


def verify_replay(
    path: Path,
    expected_episode_id: str,
    exact_deck_fingerprint: str,
    expected_team_name: str,
) -> VerifiedReplay:
    try:
        episode = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not valid JSON") from error
    if not isinstance(episode, dict):
        raise ValueError(f"{path} is not a replay object")
    actual_episode_id = replay_episode_id(episode)
    if actual_episode_id != str(expected_episode_id):
        raise ValueError(
            f"replay episode ID {actual_episode_id} != {expected_episode_id}"
        )
    decks = decks_from_replay(episode)
    fingerprints = (
        deck_fingerprint(decks[0]),
        deck_fingerprint(decks[1]),
    )
    target_seat = replay_target_seat(
        episode,
        target_fingerprint=exact_deck_fingerprint,
        expected_team_name=expected_team_name,
    )
    return VerifiedReplay(
        episode_id=actual_episode_id,
        target_seat=target_seat,
        deck_fingerprints=fingerprints,
        replay_sha256=sha256_file(path),
    )


def build_replay_records(
    sources: Sequence[ReplaySource],
    *,
    replay_path: Path,
    snapshot_dir: Path,
    exact_deck_fingerprint: str,
) -> list[ReplayRecord]:
    relative_path = replay_path.relative_to(snapshot_dir).as_posix()
    records: list[ReplayRecord] = []
    for source in sources:
        verified = verify_replay(
            replay_path,
            expected_episode_id=source.episode_id,
            exact_deck_fingerprint=exact_deck_fingerprint,
            expected_team_name=source.team_name,
        )
        records.append(
            ReplayRecord(
                snapshot_id=source.snapshot_id,
                episode_id=source.episode_id,
                submission_id=source.submission_id,
                team_id=source.team_id,
                team_name=source.team_name,
                create_time_utc=source.create_time_utc,
                end_time_utc=source.end_time_utc,
                target_seat=verified.target_seat,
                split=source.split,
                replay_sha256=verified.replay_sha256,
                replay_relpath=relative_path,
                deck_fingerprint=exact_deck_fingerprint,
            )
        )
    return records


def _default_runner(command: list[str], cwd: Path) -> None:
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if process.returncode == 0:
        return
    output = f"{process.stdout}\n{process.stderr}".strip()
    if "429" in output or "Too Many Requests" in output:
        raise ReplayRateLimitError(output[-1000:])
    raise subprocess.CalledProcessError(
        process.returncode,
        command,
        output=process.stdout,
        stderr=process.stderr,
    )


def download_replay(
    episode_id: str,
    destination: Path,
    retries: int = 4,
    *,
    runner: CommandRunner = _default_runner,
    retry_delays: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    rate_limit_delays: Sequence[float] = (30.0, 60.0, 120.0, 240.0),
    sleeper: Callable[[float], None] = time.sleep,
    before_attempt: Callable[[], None] | None = None,
    on_rate_limit: Callable[[float], None] | None = None,
) -> Path:
    if not str(episode_id).isdigit():
        raise ValueError("episode_id must contain decimal digits")
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if before_attempt is not None:
                before_attempt()
            with tempfile.TemporaryDirectory(
                prefix=f"episode-{episode_id}-",
                dir=destination.parent,
            ) as temp_name:
                temp_dir = Path(temp_name)
                command = [
                    "kaggle",
                    "competitions",
                    "replay",
                    str(episode_id),
                    "-p",
                    str(temp_dir),
                    "-q",
                ]
                runner(command, Path.cwd())
                expected = temp_dir / f"episode-{episode_id}-replay.json"
                if not expected.exists():
                    matches = list(temp_dir.glob(f"*{episode_id}*replay*.json"))
                    if len(matches) != 1:
                        raise FileNotFoundError(
                            f"downloaded replay not found for {episode_id}"
                        )
                    expected = matches[0]
                partial = destination.with_suffix(destination.suffix + ".partial")
                os.replace(expected, partial)
                os.replace(partial, destination)
                return destination
        except Exception as error:
            last_error = error
            if attempt >= retries:
                break
            delays = (
                rate_limit_delays
                if isinstance(error, ReplayRateLimitError)
                else retry_delays
            )
            delay = delays[min(attempt, len(delays) - 1)]
            if isinstance(error, ReplayRateLimitError) and on_rate_limit is not None:
                on_rate_limit(delay)
            if delay > 0:
                sleeper(delay)
    assert last_error is not None
    raise last_error
