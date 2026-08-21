from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import MemberId
from .features import encode_options, encode_options_v2


@dataclass(frozen=True, slots=True)
class PairedDecision:
    member: MemberId
    game_id: int
    v1_features: np.ndarray
    v2_features: np.ndarray
    action: tuple[int, ...]
    min_count: int
    max_count: int


@dataclass(frozen=True, slots=True)
class PairedGame:
    game_id: int
    members: tuple[MemberId, MemberId]
    decisions: tuple[PairedDecision, ...]
    finished: bool


def collect_paired_driver_game(
    registry: Any,
    member0: MemberId,
    member1: MemberId,
    game_id: int,
    max_steps: int = 2000,
) -> PairedGame:
    members = (member0, member1)
    observation, start_data = registry.game_api.battle_start(
        registry.deck(member0), registry.deck(member1)
    )
    if observation is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}"
        )
    decisions: list[PairedDecision] = []
    try:
        for _ in range(max_steps):
            current = observation.get("current")
            if not isinstance(current, Mapping):
                raise RuntimeError("engine returned no current state")
            if int(current.get("result", -1)) >= 0:
                return PairedGame(game_id, members, tuple(decisions), True)
            seat = int(current["yourIndex"])
            member = members[seat]
            select = observation.get("select")
            options = select.get("option") if isinstance(select, Mapping) else None
            if isinstance(options, list) and options:
                v1_features = encode_options(observation)
                v2_features = encode_options_v2(observation)
                if len(v1_features) != len(options) or len(v2_features) != len(options):
                    raise ValueError("paired feature rows do not match legal options")
                action = tuple(registry.action(member, observation))
                decisions.append(
                    PairedDecision(
                        member=member,
                        game_id=game_id,
                        v1_features=v1_features,
                        v2_features=v2_features,
                        action=action,
                        min_count=int(select.get("minCount", 0) or 0),
                        max_count=int(select.get("maxCount", 0) or 0),
                    )
                )
            else:
                action = tuple(registry.action(member, observation))
            observation = registry.game_api.battle_select(list(action))
        raise RuntimeError(f"paired driver match did not finish within {max_steps} decisions")
    finally:
        registry.game_api.battle_finish()


def collect_paired_games(
    registry: Any,
    games: int,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[PairedGame, ...]:
    if games <= 0:
        raise ValueError("games must be positive")
    pairs = list(itertools.combinations(MemberId, 2))
    orientations = [orientation for pair in pairs for orientation in (pair, pair[::-1])]
    collected: list[PairedGame] = []
    for game_id in range(games):
        collected.append(
            collect_paired_driver_game(
                registry,
                orientations[game_id % len(orientations)][0],
                orientations[game_id % len(orientations)][1],
                game_id,
            )
        )
        if deadline_check is not None:
            deadline_check()
    return tuple(collected)


def split_games(
    games: Sequence[PairedGame],
    train_games: int,
) -> tuple[tuple[PairedGame, ...], tuple[PairedGame, ...]]:
    batch = tuple(games)
    if not 0 < train_games < len(batch):
        raise ValueError("train_games must leave at least one held-out game")
    game_ids = [game.game_id for game in batch]
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("game IDs must be unique")
    return batch[:train_games], batch[train_games:]


def paired_collection_smoke(project_root: Path) -> dict[str, object]:
    from .bootstrap import DriverRegistry

    registry = DriverRegistry.from_project(project_root)
    try:
        game = collect_paired_driver_game(
            registry,
            MemberId.GRIMMSNARL,
            MemberId.LUCARIO,
            game_id=0,
        )
    finally:
        registry.close()
    return {
        "finished": game.finished,
        "all_actions_valid_for_both": all(
            all(
                0 <= option < len(decision.v1_features) == len(decision.v2_features)
                for option in decision.action
            )
            and decision.min_count <= len(decision.action) <= decision.max_count
            for decision in game.decisions
        ),
        "members": sorted({decision.member.value for decision in game.decisions}),
        "decisions": len(game.decisions),
        "v1_width": game.decisions[0].v1_features.shape[1],
        "v2_width": game.decisions[0].v2_features.shape[1],
        "all_float32": all(
            decision.v1_features.dtype == np.float32
            and decision.v2_features.dtype == np.float32
            for decision in game.decisions
        ),
        "raw_replays_written": 0,
    }
