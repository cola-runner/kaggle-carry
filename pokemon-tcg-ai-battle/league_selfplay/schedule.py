from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Iterable

from .contracts import FrozenLeagueConfig, MemberId


JUDGES = (
    "agents/public_kiyotah_a_sample_rule_based_agent_mega_lucario_ex_deck",
    "agents/public_biohack44_beating_the_day_2_new",
    "agents/candidate_alakazam_control_hedge_v0",
    "agents/public_mossarimossari_a_sample_rule_based_agent_dragapult_ex_deck",
)


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    phase: str
    actors: tuple[str, str]
    generations: tuple[str, str]
    seats: tuple[int, int]
    ordinal: int

    def __post_init__(self) -> None:
        if self.seats not in ((0, 1), (1, 0)):
            raise ValueError("seats must assign each actor to a different seat")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")

    @property
    def seat_zero_actor(self) -> str:
        return self.actors[self.seats.index(0)]

    @property
    def seat_one_actor(self) -> str:
        return self.actors[self.seats.index(1)]

    def canonical_record(self) -> dict[str, object]:
        return {
            "actors": list(self.actors),
            "generations": list(self.generations),
            "ordinal": self.ordinal,
            "phase": self.phase,
            "seats": list(self.seats),
        }


@dataclass(frozen=True, slots=True)
class LeagueSchedule:
    bootstrap: tuple[ScheduledGame, ...]
    round_one: tuple[ScheduledGame, ...]
    round_two_current: tuple[ScheduledGame, ...]
    round_two_history: tuple[ScheduledGame, ...]
    judges: tuple[ScheduledGame, ...]
    ancestry: tuple[ScheduledGame, ...]

    @property
    def games(self) -> tuple[ScheduledGame, ...]:
        return (
            self.bootstrap
            + self.round_one
            + self.round_two_current
            + self.round_two_history
            + self.judges
            + self.ancestry
        )

    @property
    def bootstrap_count(self) -> int:
        return len(self.bootstrap)

    @property
    def round_one_count(self) -> int:
        return len(self.round_one)

    @property
    def round_two_current_count(self) -> int:
        return len(self.round_two_current)

    @property
    def round_two_history_count(self) -> int:
        return len(self.round_two_history)

    @property
    def judge_count(self) -> int:
        return len(self.judges)

    @property
    def ancestry_count(self) -> int:
        return len(self.ancestry)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            [game.canonical_record() for game in self.games],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def seat_imbalances(self) -> dict[str, int]:
        counts: dict[tuple[str, tuple[str, str], tuple[str, str]], list[int]] = {}
        for game in self.games:
            key = (game.phase, game.actors, game.generations)
            orientation = 0 if game.seats == (0, 1) else 1
            counts.setdefault(key, [0, 0])[orientation] += 1
        return {
            json.dumps(key, separators=(",", ":")): values[0] - values[1]
            for key, values in counts.items()
            if values[0] != values[1]
        }


def _balanced_games(
    phase: str,
    matchups: Iterable[tuple[tuple[str, str], tuple[str, str]]],
    games_per_seat: int,
) -> tuple[ScheduledGame, ...]:
    if games_per_seat < 0:
        raise ValueError("games_per_seat must be non-negative")
    games: list[ScheduledGame] = []
    ordinal = 0
    for actors, generations in matchups:
        for _ in range(games_per_seat):
            for seats in ((0, 1), (1, 0)):
                games.append(ScheduledGame(phase, actors, generations, seats, ordinal))
                ordinal += 1
    return tuple(games)


def _schedule(config: FrozenLeagueConfig, *, dry_run: bool) -> LeagueSchedule:
    def repetitions(value: int) -> int:
        return int(value > 0) if dry_run else value

    members = tuple(member.value for member in MemberId)
    pairs = tuple(itertools.combinations(members, 2))
    bootstrap = _balanced_games(
        "bootstrap",
        ((pair, ("driver", "driver")) for pair in pairs),
        repetitions(config.bootstrap_games_per_seat),
    )
    round_one = _balanced_games(
        "round_1",
        ((pair, ("round_0", "round_0")) for pair in pairs),
        repetitions(config.round_one_games_per_seat),
    )
    round_two_repetitions = (
        3
        if dry_run
        and config.round_two_games_per_seat > 0
        and config.history_games_per_seat > 0
        else repetitions(config.round_two_games_per_seat)
    )
    round_two_current = _balanced_games(
        "round_2_current",
        ((pair, ("round_1", "round_1")) for pair in pairs),
        round_two_repetitions,
    )
    current_history = (
        ((current, historical), ("round_1", "round_0"))
        for current in members
        for historical in members
    )
    round_two_history = _balanced_games(
        "round_2_history",
        current_history,
        repetitions(config.history_games_per_seat),
    )
    judge_matchups = (
        ((member, judge), (generation, "fixed"))
        for generation in ("round_0", "round_2")
        for member in members
        for judge in JUDGES
    )
    judges = _balanced_games(
        "judges",
        judge_matchups,
        repetitions(config.judge_games_per_seat),
    )
    ancestry_matchups = (
        ((final, starting), ("round_2", "round_0"))
        for final in members
        for starting in members
    )
    ancestry = _balanced_games(
        "ancestry",
        ancestry_matchups,
        repetitions(config.ancestry_games_per_seat),
    )
    return LeagueSchedule(
        bootstrap=bootstrap,
        round_one=round_one,
        round_two_current=round_two_current,
        round_two_history=round_two_history,
        judges=judges,
        ancestry=ancestry,
    )


def build_standard_schedule(config: FrozenLeagueConfig) -> LeagueSchedule:
    return _schedule(config, dry_run=False)


def build_dry_run_schedule(config: FrozenLeagueConfig) -> LeagueSchedule:
    return _schedule(config, dry_run=True)
