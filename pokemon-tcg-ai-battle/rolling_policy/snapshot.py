from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .constants import (
    EXACT_DECK_FINGERPRINT,
    HOLDOUT_WINDOW_HOURS,
    MIN_TEACHER_TEAMS,
    SOURCE_WINDOW_HOURS,
    VALIDATION_WINDOW_HOURS,
)
from .schema import EpisodeRecord, Split, TeacherSubmission, parse_utc_datetime


@dataclass(frozen=True, slots=True)
class ActiveSubmission:
    submission_id: str
    score: float
    submitted_at_utc: datetime


def parse_leaderboard(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = [dict(row) for row in csv.DictReader(file)]
    if not rows:
        raise ValueError("leaderboard is empty")
    return rows


def rank_ten_threshold(rows: Sequence[Mapping[str, str]]) -> float:
    scores = []
    for row in rows:
        try:
            rank = int(row.get("Rank", ""))
        except (TypeError, ValueError):
            continue
        if rank == 10:
            try:
                score = float(row.get("Score", ""))
            except (TypeError, ValueError) as error:
                raise ValueError("rank 10 has an invalid score") from error
            if not math.isfinite(score):
                raise ValueError("rank 10 has a non-finite score")
            scores.append(score)
    if not scores:
        raise ValueError("leaderboard has no rank 10 row")
    return min(scores)


def assign_split(create_time_utc: datetime, cutoff_utc: datetime) -> Split | None:
    created = parse_utc_datetime(create_time_utc)
    cutoff = parse_utc_datetime(cutoff_utc)
    if created > cutoff or created < cutoff - timedelta(hours=SOURCE_WINDOW_HOURS):
        return None
    if created >= cutoff - timedelta(hours=HOLDOUT_WINDOW_HOURS):
        return Split.HOLDOUT
    if created >= cutoff - timedelta(
        hours=HOLDOUT_WINDOW_HOURS + VALIDATION_WINDOW_HOURS
    ):
        return Split.VALIDATION
    return Split.TRAIN


def parse_submission_rows(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, ActiveSubmission]:
    parsed: dict[str, ActiveSubmission] = {}
    for row in rows:
        submission_id = str(row.get("id", "")).strip()
        if not submission_id.isdigit():
            continue
        score = float(row.get("publicScore", "nan"))
        if not math.isfinite(score):
            raise ValueError(f"submission {submission_id} has invalid publicScore")
        parsed[submission_id] = ActiveSubmission(
            submission_id=submission_id,
            score=score,
            submitted_at_utc=parse_utc_datetime(str(row.get("dateSubmitted", ""))),
        )
    return parsed


def completed_public_episode_ids(
    rows: Iterable[Mapping[str, str]],
    cutoff_utc: datetime,
) -> set[str]:
    active: set[str] = set()
    for row in rows:
        episode_id = str(row.get("id", "")).strip()
        if not episode_id.isdigit():
            continue
        if row.get("state") != "EpisodeState.COMPLETED":
            continue
        if row.get("type") != "EpisodeType.EPISODE_TYPE_PUBLIC":
            continue
        if assign_split(parse_utc_datetime(str(row.get("createTime", ""))), cutoff_utc):
            active.add(episode_id)
    return active


def eligible_teachers(
    candidates: Sequence[TeacherSubmission],
    rank_ten_score: float,
    active_submission_ids: Collection[str],
    *,
    minimum_teacher_teams: int = MIN_TEACHER_TEAMS,
) -> tuple[TeacherSubmission, ...]:
    if minimum_teacher_teams < 1:
        raise ValueError("minimum_teacher_teams must be positive")
    eligible = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if candidate.deck_fingerprint == EXACT_DECK_FINGERPRINT
                and candidate.tracked_at_cutoff
                and candidate.submission_id in active_submission_ids
                and candidate.score >= rank_ten_score
            ),
            key=lambda candidate: candidate.submission_id,
        )
    )
    if len({candidate.team_id for candidate in eligible}) < minimum_teacher_teams:
        minimum_text = (
            "three" if minimum_teacher_teams == 3 else str(minimum_teacher_teams)
        )
        raise ValueError(
            "fewer than "
            f"{minimum_text} distinct eligible teacher teams"
        )
    return eligible


def balanced_episode_weights(
    episodes: Sequence[EpisodeRecord],
) -> dict[str, float]:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    group_counts: Counter[tuple[str, int, int]] = Counter()
    keys: dict[str, tuple[str, int, int]] = {}
    for episode in episodes:
        bucket = int(
            (episode.create_time_utc - epoch).total_seconds() // (12 * 60 * 60)
        )
        key = (episode.team_id, episode.target_seat, bucket)
        if episode.episode_id in keys:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        keys[episode.episode_id] = key
        group_counts[key] += 1
    return {
        episode_id: 1.0 / group_counts[key]
        for episode_id, key in keys.items()
    }
