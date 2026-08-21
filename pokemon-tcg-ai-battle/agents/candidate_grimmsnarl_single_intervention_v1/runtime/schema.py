from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


_NUMERIC_ID = re.compile(r"^[0-9]+$")
_DECK_FINGERPRINT = re.compile(r"^[0-9a-f]{12}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


def parse_utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise ValueError("timestamp must be a datetime or non-empty ISO string")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_numeric_id(name: str, value: str) -> None:
    if not isinstance(value, str) or _NUMERIC_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must contain decimal digits")


def _require_utc(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class TeacherSubmission:
    team_id: str
    team_name: str
    submission_id: str
    score: float
    deck_fingerprint: str
    tracked_at_cutoff: bool
    submitted_at_utc: datetime | None = None

    def __post_init__(self) -> None:
        _require_text("team_id", self.team_id)
        _require_text("team_name", self.team_name)
        _require_numeric_id("submission_id", self.submission_id)
        if not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if _DECK_FINGERPRINT.fullmatch(self.deck_fingerprint) is None:
            raise ValueError("deck_fingerprint must be 12 lowercase hex characters")
        if not isinstance(self.tracked_at_cutoff, bool):
            raise ValueError("tracked_at_cutoff must be bool")
        if self.submitted_at_utc is not None:
            _require_utc("submitted_at_utc", self.submitted_at_utc)

    def to_dict(self) -> dict[str, object]:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "submission_id": self.submission_id,
            "score": float(self.score),
            "deck_fingerprint": self.deck_fingerprint,
            "tracked_at_cutoff": self.tracked_at_cutoff,
            "submitted_at_utc": (
                self.submitted_at_utc.isoformat().replace("+00:00", "Z")
                if self.submitted_at_utc is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> TeacherSubmission:
        return cls(
            team_id=str(row["team_id"]),
            team_name=str(row["team_name"]),
            submission_id=str(row["submission_id"]),
            score=float(row["score"]),
            deck_fingerprint=str(row["deck_fingerprint"]),
            tracked_at_cutoff=row["tracked_at_cutoff"],
            submitted_at_utc=(
                parse_utc_datetime(row["submitted_at_utc"])
                if row.get("submitted_at_utc")
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    submission_id: str
    team_id: str
    create_time_utc: datetime
    end_time_utc: datetime
    target_seat: int
    split: Split
    replay_sha256: str

    def __post_init__(self) -> None:
        _require_numeric_id("episode_id", self.episode_id)
        _require_numeric_id("submission_id", self.submission_id)
        _require_text("team_id", self.team_id)
        _require_utc("create_time_utc", self.create_time_utc)
        _require_utc("end_time_utc", self.end_time_utc)
        if self.end_time_utc < self.create_time_utc:
            raise ValueError("end_time_utc cannot precede create_time_utc")
        if self.target_seat not in (0, 1):
            raise ValueError("target_seat must be 0 or 1")
        if not isinstance(self.split, Split):
            raise ValueError("split must be a Split")
        if _SHA256.fullmatch(self.replay_sha256) is None:
            raise ValueError("replay_sha256 must be 64 lowercase hex characters")

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "submission_id": self.submission_id,
            "team_id": self.team_id,
            "create_time_utc": self.create_time_utc.isoformat().replace("+00:00", "Z"),
            "end_time_utc": self.end_time_utc.isoformat().replace("+00:00", "Z"),
            "target_seat": self.target_seat,
            "split": self.split.value,
            "replay_sha256": self.replay_sha256,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> EpisodeRecord:
        return cls(
            episode_id=str(row["episode_id"]),
            submission_id=str(row["submission_id"]),
            team_id=str(row["team_id"]),
            create_time_utc=parse_utc_datetime(row["create_time_utc"]),
            end_time_utc=parse_utc_datetime(row["end_time_utc"]),
            target_seat=int(row["target_seat"]),
            split=Split(str(row["split"])),
            replay_sha256=str(row["replay_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    snapshot_id: str
    implementation_started_utc: datetime
    cutoff_utc: datetime
    rank_ten_score: float
    leaderboard_sha256: str
    teacher_candidates_sha256: str
    input_sha256: Mapping[str, str]
    teachers: tuple[TeacherSubmission, ...]
    source_window_start_utc: datetime
    validation_start_utc: datetime
    holdout_start_utc: datetime

    def __post_init__(self) -> None:
        _require_text("snapshot_id", self.snapshot_id)
        for name, value in (
            ("implementation_started_utc", self.implementation_started_utc),
            ("cutoff_utc", self.cutoff_utc),
            ("source_window_start_utc", self.source_window_start_utc),
            ("validation_start_utc", self.validation_start_utc),
            ("holdout_start_utc", self.holdout_start_utc),
        ):
            _require_utc(name, value)
        if self.implementation_started_utc > self.cutoff_utc:
            raise ValueError("implementation_started_utc cannot follow cutoff_utc")
        if not math.isfinite(self.rank_ten_score):
            raise ValueError("rank_ten_score must be finite")
        for name, value in (
            ("leaderboard_sha256", self.leaderboard_sha256),
            ("teacher_candidates_sha256", self.teacher_candidates_sha256),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be 64 lowercase hex characters")
        if not self.input_sha256:
            raise ValueError("input_sha256 must not be empty")
        for name, digest in self.input_sha256.items():
            _require_text("input name", name)
            if _SHA256.fullmatch(digest) is None:
                raise ValueError(f"input_sha256[{name!r}] must be a SHA-256 digest")
        if len({teacher.team_id for teacher in self.teachers}) < 3:
            raise ValueError("teachers must contain at least three distinct teams")

    def to_dict(self) -> dict[str, object]:
        def utc_text(value: datetime) -> str:
            return value.isoformat().replace("+00:00", "Z")

        return {
            "snapshot_id": self.snapshot_id,
            "implementation_started_utc": utc_text(self.implementation_started_utc),
            "cutoff_utc": utc_text(self.cutoff_utc),
            "rank_ten_score": float(self.rank_ten_score),
            "leaderboard_sha256": self.leaderboard_sha256,
            "teacher_candidates_sha256": self.teacher_candidates_sha256,
            "input_sha256": dict(sorted(self.input_sha256.items())),
            "teachers": [teacher.to_dict() for teacher in self.teachers],
            "source_window_start_utc": utc_text(self.source_window_start_utc),
            "validation_start_utc": utc_text(self.validation_start_utc),
            "holdout_start_utc": utc_text(self.holdout_start_utc),
        }


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    snapshot_id: str
    episode_id: str
    submission_id: str
    team_id: str
    team_name: str
    create_time_utc: datetime
    end_time_utc: datetime
    target_seat: int
    split: Split
    replay_sha256: str
    replay_relpath: str
    deck_fingerprint: str

    def __post_init__(self) -> None:
        _require_text("snapshot_id", self.snapshot_id)
        _require_numeric_id("episode_id", self.episode_id)
        _require_numeric_id("submission_id", self.submission_id)
        _require_text("team_id", self.team_id)
        _require_text("team_name", self.team_name)
        _require_utc("create_time_utc", self.create_time_utc)
        _require_utc("end_time_utc", self.end_time_utc)
        if self.end_time_utc < self.create_time_utc:
            raise ValueError("end_time_utc cannot precede create_time_utc")
        if self.target_seat not in (0, 1):
            raise ValueError("target_seat must be 0 or 1")
        if not isinstance(self.split, Split):
            raise ValueError("split must be a Split")
        if _SHA256.fullmatch(self.replay_sha256) is None:
            raise ValueError("replay_sha256 must be 64 lowercase hex characters")
        path = self.replay_relpath
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("replay_relpath must be a safe relative path")
        if _DECK_FINGERPRINT.fullmatch(self.deck_fingerprint) is None:
            raise ValueError("deck_fingerprint must be 12 lowercase hex characters")

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "episode_id": self.episode_id,
            "submission_id": self.submission_id,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "create_time_utc": self.create_time_utc.isoformat().replace("+00:00", "Z"),
            "end_time_utc": self.end_time_utc.isoformat().replace("+00:00", "Z"),
            "target_seat": self.target_seat,
            "split": self.split.value,
            "replay_sha256": self.replay_sha256,
            "replay_relpath": self.replay_relpath,
            "deck_fingerprint": self.deck_fingerprint,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ReplayRecord:
        return cls(
            snapshot_id=str(row["snapshot_id"]),
            episode_id=str(row["episode_id"]),
            submission_id=str(row["submission_id"]),
            team_id=str(row["team_id"]),
            team_name=str(row["team_name"]),
            create_time_utc=parse_utc_datetime(row["create_time_utc"]),
            end_time_utc=parse_utc_datetime(row["end_time_utc"]),
            target_seat=int(row["target_seat"]),
            split=Split(str(row["split"])),
            replay_sha256=str(row["replay_sha256"]),
            replay_relpath=str(row["replay_relpath"]),
            deck_fingerprint=str(row["deck_fingerprint"]),
        )
