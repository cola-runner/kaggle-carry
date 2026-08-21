from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class InvalidSelfPlay(RuntimeError):
    pass


class MemberId(str, Enum):
    GRIMMSNARL = "grimmsnarl"
    LUCARIO = "lucario"
    CRUSTLE = "crustle"
    ALAKAZAM = "alakazam"


class GameSource(str, Enum):
    CURRENT_CURRENT = "current_current"
    CURRENT_HISTORY = "current_history"
    CURRENT_VS_FIXED = "current_vs_fixed"


@dataclass(frozen=True, slots=True)
class FrozenLeagueConfig:
    seed: int = 20260804
    bootstrap_games_per_seat: int = 8
    round_one_games_per_seat: int = 12
    round_two_games_per_seat: int = 12
    history_games_per_seat: int = 2
    judge_games_per_seat: int = 4
    ancestry_games_per_seat: int = 2
    wall_time_seconds: int = 1200
    temp_quota_bytes: int = 512 * 1024**2

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GameProvenance:
    source: GameSource
    actors: tuple[str, str]
    trajectory_members: tuple[MemberId, ...]
    update_members: tuple[MemberId, ...]

    @classmethod
    def current_game(cls, first: MemberId, second: MemberId) -> GameProvenance:
        members = (first, second)
        return cls(
            source=GameSource.CURRENT_CURRENT,
            actors=(first.value, second.value),
            trajectory_members=members,
            update_members=members,
        )

    @classmethod
    def history_game(cls, current: MemberId, snapshot: str) -> GameProvenance:
        return cls(
            source=GameSource.CURRENT_HISTORY,
            actors=(current.value, snapshot),
            trajectory_members=(current,),
            update_members=(current,),
        )


@dataclass(frozen=True, slots=True)
class SelfPlayAudit:
    valid: bool
    code: str
    reasons: tuple[str, ...]
    current_current_games: int
    total_games: int
    updated_members: tuple[MemberId, ...]


def audit_training_batch(
    records: Iterable[GameProvenance],
    expected_members: Iterable[MemberId],
) -> SelfPlayAudit:
    batch = tuple(records)
    expected = frozenset(expected_members)
    reasons: list[str] = []

    fixed_present = any(record.source is GameSource.CURRENT_VS_FIXED for record in batch)
    if fixed_present:
        reasons.append("fixed actors cannot contribute training trajectories")

    current_games = [record for record in batch if record.source is GameSource.CURRENT_CURRENT]
    for record in current_games:
        try:
            actors = tuple(MemberId(actor) for actor in record.actors)
        except ValueError:
            reasons.append("current-current actors must both be current members")
            continue
        if record.trajectory_members != actors or record.update_members != actors:
            reasons.append("current-current games must record and update both participants")

    updated = frozenset(member for record in batch for member in record.update_members)
    missing = sorted(expected - updated, key=lambda member: member.value)
    if missing:
        reasons.append("missing current members: " + ", ".join(member.value for member in missing))

    if len(current_games) * 2 <= len(batch):
        reasons.append("current-current games must be a strict majority")

    valid = not reasons
    return SelfPlayAudit(
        valid=valid,
        code="PASS_SELF_PLAY_AUDIT" if valid else "INVALID_SELF_PLAY",
        reasons=tuple(reasons),
        current_current_games=len(current_games),
        total_games=len(batch),
        updated_members=tuple(sorted(updated, key=lambda member: member.value)),
    )
