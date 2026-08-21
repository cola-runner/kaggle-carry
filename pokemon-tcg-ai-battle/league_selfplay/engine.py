from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from scripts.run_local_match import (
    first_result_deck,
    import_official_cg,
    load_agent,
    validate_action,
)

from .actions import sample_action
from .contracts import GameProvenance, GameSource, MemberId
from .features import encode_options
from .model import PolicyValueNet, create_population


@dataclass(slots=True)
class TrajectoryStep:
    member: MemberId
    seat: int
    features: np.ndarray
    action: tuple[int, ...]
    min_count: int
    max_count: int
    old_log_probability: float
    old_value: float
    baseline_action: tuple[int, ...] | None = None
    exploration_rate: float = 1.0
    reward: float = 0.0


@dataclass(slots=True)
class CompletedGame:
    provenance: GameProvenance
    winner: int
    decisions: int
    steps: list[TrajectoryStep]
    finished: bool


@dataclass(slots=True)
class PolicyActor:
    member: MemberId
    model: PolicyValueNet
    deck: list[int]
    device: str
    trainable: bool = True
    generation: str = "current"

    @property
    def name(self) -> str:
        return self.member.value if self.generation == "current" else f"{self.member.value}-{self.generation}"

    def decide(
        self,
        observation: Mapping[str, Any],
        seat: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], TrajectoryStep | None]:
        select = observation.get("select")
        if not isinstance(select, Mapping):
            raise ValueError("observation has no selection")
        options = select.get("option")
        option_count = len(options) if isinstance(options, list) else 0
        min_count = int(select.get("minCount", 0) or 0)
        max_count = int(select.get("maxCount", min_count) or 0)
        if option_count == 0 and min_count == max_count == 0:
            return [], None

        features = encode_options(observation)
        if len(features) != option_count:
            raise ValueError("feature rows do not match legal options")
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.from_numpy(features)[None, :, :].to(self.device)
            mask = torch.ones((1, option_count), dtype=torch.bool, device=self.device)
            option_logits, stop_logits, values = self.model(feature_tensor, mask)
        logits = option_logits[0].detach().cpu().numpy().astype(np.float64)
        stop_logit = float(stop_logits[0].detach().cpu())
        value = float(values[0].detach().cpu())
        sample = sample_action(logits, stop_logit, min_count, max_count, rng)
        action = validate_action(list(sample.indices), dict(select))
        step = TrajectoryStep(
            member=self.member,
            seat=seat,
            features=features,
            action=sample.indices,
            min_count=min_count,
            max_count=max_count,
            old_log_probability=sample.log_probability,
            old_value=value,
        )
        return action, step


def _provenance(actor0: PolicyActor, actor1: PolicyActor, source: GameSource) -> GameProvenance:
    if source is GameSource.CURRENT_CURRENT:
        if not actor0.trainable or not actor1.trainable:
            raise ValueError("current-current games require two trainable actors")
        return GameProvenance.current_game(actor0.member, actor1.member)
    if source is GameSource.CURRENT_HISTORY:
        current = actor0 if actor0.trainable else actor1
        history = actor1 if actor0.trainable else actor0
        if current.trainable == history.trainable:
            raise ValueError("current-history games require exactly one trainable actor")
        return GameProvenance.history_game(current.member, history.name)
    raise ValueError("fixed actors cannot enter training games")


def run_training_game(
    game_api: Any,
    actor0: PolicyActor,
    actor1: PolicyActor,
    source: GameSource,
    rng: np.random.Generator,
    max_steps: int = 2000,
) -> CompletedGame:
    provenance = _provenance(actor0, actor1, source)
    actors = (actor0, actor1)
    observation, start_data = game_api.battle_start(actor0.deck, actor1.deck)
    if observation is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}"
        )
    steps: list[TrajectoryStep] = []
    try:
        for decision in range(max_steps):
            current = observation.get("current")
            if not isinstance(current, Mapping):
                raise RuntimeError("engine returned no current state")
            result = int(current.get("result", -1))
            if result >= 0:
                for seat in (0, 1):
                    seat_steps = [step for step in steps if step.seat == seat]
                    if seat_steps:
                        seat_steps[-1].reward = (
                            0.0
                            if result not in (0, 1)
                            else (1.0 if seat == result else -1.0)
                        )
                return CompletedGame(provenance, result, decision, steps, True)

            seat = int(current["yourIndex"])
            action, step = actors[seat].decide(observation, seat, rng)
            if step is not None and actors[seat].trainable:
                steps.append(step)
            observation = game_api.battle_select(action)
        raise RuntimeError(f"match did not finish within {max_steps} decisions")
    finally:
        game_api.battle_finish()


def _driver_deck(project_root: Path, member: MemberId, relative: str) -> list[int]:
    directory = project_root / relative
    agent = load_agent(directory, f"league_deck_{member.value}")
    return first_result_deck(agent, directory)


def engine_smoke(project_root: Path) -> dict[str, Any]:
    game_api = import_official_cg(project_root)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    population = create_population(20260804, device)
    decks = {
        MemberId.GRIMMSNARL: _driver_deck(
            project_root,
            MemberId.GRIMMSNARL,
            "agents/candidate_grimmsnarl_imitation_full_v2",
        ),
        MemberId.LUCARIO: _driver_deck(
            project_root,
            MemberId.LUCARIO,
            "agents/public_makthanithin_ptcg_mega_lucario_ex_v62",
        ),
    }
    actors = {
        member: PolicyActor(member, population[member], decks[member], device)
        for member in decks
    }
    rng = np.random.default_rng(20260804)
    seat_orders = (
        (MemberId.GRIMMSNARL, MemberId.LUCARIO),
        (MemberId.LUCARIO, MemberId.GRIMMSNARL),
    )
    games = [
        run_training_game(
            game_api,
            actors[first],
            actors[second],
            GameSource.CURRENT_CURRENT,
            rng,
        )
        for first, second in seat_orders
    ]
    return {
        "games": len(games),
        "finished_games": sum(game.finished for game in games),
        "seat_orders": [[first.value, second.value] for first, second in seat_orders],
        "trajectory_members": [
            sorted({step.member.value for step in game.steps}) for game in games
        ],
        "total_steps": sum(len(game.steps) for game in games),
        "all_features_float32": all(
            step.features.dtype == np.float32 for game in games for step in game.steps
        ),
        "all_log_probabilities_finite": all(
            math.isfinite(step.old_log_probability) for game in games for step in game.steps
        ),
        "reward_values_are_terminal": all(
            step.reward in (-1.0, 0.0, 1.0) for game in games for step in game.steps
        ),
        "one_terminal_reward_per_member": all(
            all(step.reward == 0.0 for step in member_steps[:-1])
            and member_steps[-1].reward in (-1.0, 0.0, 1.0)
            for game in games
            for member in (MemberId.GRIMMSNARL, MemberId.LUCARIO)
            for member_steps in ([step for step in game.steps if step.member is member],)
            if member_steps
        ),
        "illegal_actions": 0,
    }
