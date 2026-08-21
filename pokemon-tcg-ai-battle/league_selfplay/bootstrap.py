from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts.run_local_match import (
    first_result_deck,
    import_official_cg,
    load_agent,
    pushd,
    validate_action,
)

from .contracts import GameSource, MemberId
from .engine import PolicyActor, run_training_game
from .features import INPUT_WIDTH, encode_options
from .model import PolicyValueNet, create_population


DRIVER_PATHS = {
    MemberId.GRIMMSNARL: "agents/candidate_grimmsnarl_imitation_full_v2",
    MemberId.LUCARIO: "agents/public_makthanithin_ptcg_mega_lucario_ex_v62",
    MemberId.CRUSTLE: "agents/candidate_crustle_kangaskhan_top1_ranker_v211",
    MemberId.ALAKAZAM: "agents/public_naoto714_alakazam_no_tech_pivot_en",
}


@dataclass(slots=True)
class BootstrapDecision:
    features: np.ndarray
    action: tuple[int, ...]
    min_count: int
    max_count: int


@dataclass(frozen=True, slots=True)
class BootstrapMemberStats:
    decisions: int
    loss_last: float
    parameter_delta_l2: float
    all_finite: bool


@dataclass(frozen=True, slots=True)
class BootstrapStats:
    members: dict[MemberId, BootstrapMemberStats]
    driver_games: int


@dataclass(frozen=True, slots=True)
class DriverGameResult:
    finished: bool
    decisions: dict[MemberId, list[BootstrapDecision]]


@dataclass(frozen=True, slots=True)
class StartGateResult:
    games: int
    teacher_closed: bool


class DriverRegistry:
    def __init__(
        self,
        project_root: Path,
        game_api: Any,
        agents: dict[MemberId, Any],
        directories: dict[MemberId, Path],
        decks: dict[MemberId, list[int]],
    ) -> None:
        self.project_root = project_root
        self.game_api = game_api
        self._agents = agents
        self._directories = directories
        self._decks = decks
        self._closed = False

    @classmethod
    def from_project(cls, project_root: Path) -> DriverRegistry:
        game_api = import_official_cg(project_root)
        agents: dict[MemberId, Any] = {}
        directories: dict[MemberId, Path] = {}
        decks: dict[MemberId, list[int]] = {}
        for index, (member, relative) in enumerate(DRIVER_PATHS.items()):
            directory = project_root / relative
            agent = load_agent(directory, f"league_bootstrap_driver_{index}")
            agents[member] = agent
            directories[member] = directory
            decks[member] = first_result_deck(agent, directory)
        return cls(project_root, game_api, agents, directories, decks)

    @property
    def closed(self) -> bool:
        return self._closed

    def deck(self, member: MemberId) -> list[int]:
        return list(self._decks[member])

    def action(self, member: MemberId, observation: Mapping[str, Any]) -> list[int]:
        if self._closed:
            raise RuntimeError("teacher access closed")
        with pushd(self._directories[member]):
            action = self._agents[member](observation)
        return validate_action(action, observation["select"])

    def close(self) -> None:
        self._closed = True
        self._agents.clear()


def collect_driver_game(
    registry: DriverRegistry,
    member0: MemberId,
    member1: MemberId,
    max_steps: int = 2000,
) -> DriverGameResult:
    members = (member0, member1)
    observation, start_data = registry.game_api.battle_start(
        registry.deck(member0), registry.deck(member1)
    )
    if observation is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}"
        )
    decisions = {member0: [], member1: []}
    try:
        for _ in range(max_steps):
            current = observation.get("current")
            if not isinstance(current, Mapping):
                raise RuntimeError("engine returned no current state")
            if int(current.get("result", -1)) >= 0:
                return DriverGameResult(True, decisions)
            seat = int(current["yourIndex"])
            member = members[seat]
            action = registry.action(member, observation)
            select = observation.get("select")
            options = select.get("option") if isinstance(select, Mapping) else None
            if isinstance(options, list) and options:
                decisions[member].append(
                    BootstrapDecision(
                        features=encode_options(observation),
                        action=tuple(action),
                        min_count=int(select.get("minCount", 0) or 0),
                        max_count=int(select.get("maxCount", 0) or 0),
                    )
                )
            observation = registry.game_api.battle_select(action)
        raise RuntimeError(f"driver match did not finish within {max_steps} decisions")
    finally:
        registry.game_api.battle_finish()


def _action_log_probability_torch(
    action: tuple[int, ...],
    option_logits: torch.Tensor,
    stop_logit: torch.Tensor,
    min_count: int,
    max_count: int,
) -> torch.Tensor:
    selected: list[int] = []
    terms: list[torch.Tensor] = []
    for chosen in action:
        remaining = [index for index in range(len(option_logits)) if index not in selected]
        candidate = option_logits[remaining]
        if len(selected) >= min_count and min_count != max_count:
            candidate = torch.cat((candidate, stop_logit.reshape(1)))
        terms.append(option_logits[chosen] - torch.logsumexp(candidate, dim=0))
        selected.append(chosen)
    if len(selected) < max_count:
        remaining = [index for index in range(len(option_logits)) if index not in selected]
        candidate = torch.cat((option_logits[remaining], stop_logit.reshape(1)))
        terms.append(stop_logit - torch.logsumexp(candidate, dim=0))
    return torch.stack(terms).sum() if terms else option_logits.sum() * 0.0


def _padded_batch(
    decisions: Sequence[BootstrapDecision],
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(len(decision.features) for decision in decisions)
    features = np.zeros((len(decisions), maximum, INPUT_WIDTH), dtype=np.float32)
    mask = np.zeros((len(decisions), maximum), dtype=bool)
    for index, decision in enumerate(decisions):
        count = len(decision.features)
        features[index, :count] = decision.features
        mask[index, :count] = True
    return features, mask


def train_from_decisions(
    population: dict[MemberId, PolicyValueNet],
    decisions: Mapping[MemberId, Sequence[BootstrapDecision]],
    device: str,
    seed: int,
    *,
    epochs: int = 6,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
) -> dict[MemberId, BootstrapMemberStats]:
    rng = np.random.default_rng(seed)
    stats: dict[MemberId, BootstrapMemberStats] = {}
    for member in MemberId:
        member_decisions = list(decisions.get(member, ()))
        if not member_decisions:
            raise ValueError(f"no bootstrap decisions for {member.value}")
        model = population[member]
        policy_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("value_head.")
        ]
        before = [parameter.detach().cpu().numpy().copy() for parameter in policy_parameters]
        optimizer = torch.optim.AdamW(policy_parameters, lr=learning_rate, weight_decay=1e-5)
        losses: list[float] = []
        for _ in range(epochs):
            order = rng.permutation(len(member_decisions))
            for start in range(0, len(order), batch_size):
                batch = [member_decisions[int(index)] for index in order[start : start + batch_size]]
                padded, mask = _padded_batch(batch)
                option_logits, stop_logits, _ = model(
                    torch.from_numpy(padded).to(device),
                    torch.from_numpy(mask).to(device),
                )
                log_probabilities = [
                    _action_log_probability_torch(
                        decision.action,
                        option_logits[index, : len(decision.features)],
                        stop_logits[index],
                        decision.min_count,
                        decision.max_count,
                    )
                    for index, decision in enumerate(batch)
                ]
                loss = -torch.stack(log_probabilities).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_parameters, 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        delta_squared = 0.0
        for initial, parameter in zip(before, policy_parameters, strict=True):
            difference = parameter.detach().cpu().numpy().astype(np.float64) - initial
            delta_squared += float(np.square(difference).sum())
        stats[member] = BootstrapMemberStats(
            decisions=len(member_decisions),
            loss_last=losses[-1],
            parameter_delta_l2=math.sqrt(delta_squared),
            all_finite=all(math.isfinite(loss) for loss in losses),
        )
    return stats


def initialize_population(
    population: dict[MemberId, PolicyValueNet],
    drivers: DriverRegistry,
    games: int,
    device: str,
    seed: int,
    *,
    epochs: int = 6,
) -> BootstrapStats:
    pairs = list(itertools.combinations(MemberId, 2))
    orientations = [oriented for pair in pairs for oriented in (pair, pair[::-1])]
    decisions: dict[MemberId, list[BootstrapDecision]] = {member: [] for member in MemberId}
    for game_index in range(games):
        first, second = orientations[game_index % len(orientations)]
        result = collect_driver_game(drivers, first, second)
        for member in MemberId:
            decisions[member].extend(result.decisions.get(member, ()))
    return BootstrapStats(
        members=train_from_decisions(
            population,
            decisions,
            device,
            seed,
            epochs=epochs,
        ),
        driver_games=games,
    )


def run_start_gate(
    population: dict[MemberId, PolicyValueNet],
    game_api: Any,
    drivers: DriverRegistry,
    members: Sequence[MemberId] = tuple(MemberId),
    seed: int = 20260804,
) -> StartGateResult:
    drivers.close()
    device = next(iter(population.values())).layer_one.weight.device.type
    actors = {
        member: PolicyActor(member, population[member], drivers.deck(member), device)
        for member in members
    }
    pairs = list(itertools.combinations(members, 2))
    rng = np.random.default_rng(seed)
    games = 0
    for pair in pairs:
        for first, second in (pair, pair[::-1]):
            result = run_training_game(
                game_api,
                actors[first],
                actors[second],
                GameSource.CURRENT_CURRENT,
                rng,
            )
            if not result.finished:
                raise RuntimeError("learned-policy start gate did not finish")
            games += 1
    return StartGateResult(games=games, teacher_closed=drivers.closed)


def _synthetic_decisions(seed: int) -> dict[MemberId, list[BootstrapDecision]]:
    rng = np.random.default_rng(seed)
    result: dict[MemberId, list[BootstrapDecision]] = {}
    for member in MemberId:
        member_decisions: list[BootstrapDecision] = []
        for _ in range(12):
            features = rng.normal(0.0, 0.2, size=(3, INPUT_WIDTH)).astype(np.float32)
            member_decisions.append(
                BootstrapDecision(features, (int(features[:, 0].argmax()),), 1, 1)
            )
        result[member] = member_decisions
    return result


def bootstrap_smoke(project_root: Path) -> dict[str, Any]:
    registry = DriverRegistry.from_project(project_root)
    driver_game = collect_driver_game(
        registry,
        MemberId.GRIMMSNARL,
        MemberId.LUCARIO,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    population = create_population(20260804, device)
    stats = train_from_decisions(
        population,
        _synthetic_decisions(20260804),
        device,
        20260804,
        epochs=2,
        batch_size=16,
    )
    start_gate = run_start_gate(
        population,
        registry.game_api,
        registry,
        members=(MemberId.GRIMMSNARL, MemberId.LUCARIO),
    )
    rejected = False
    try:
        registry.action(MemberId.GRIMMSNARL, {})
    except RuntimeError as error:
        rejected = str(error) == "teacher access closed"
    return {
        "driver_game_finished": driver_game.finished,
        "driver_game_members": sorted(
            member.value for member, decisions in driver_game.decisions.items() if decisions
        ),
        "updated_members": sorted(member.value for member in stats),
        "parameter_delta_l2": {
            member.value: member_stats.parameter_delta_l2
            for member, member_stats in stats.items()
        },
        "all_finite": all(member_stats.all_finite for member_stats in stats.values()),
        "start_gate_games": start_gate.games,
        "teacher_closed_before_start_gate": start_gate.teacher_closed,
        "teacher_call_after_close_rejected": rejected,
    }
