from __future__ import annotations

import copy
import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional

from .actions import sample_action
from .contracts import GameProvenance, InvalidSelfPlay, MemberId, audit_training_batch
from .engine import CompletedGame, TrajectoryStep
from .features import INPUT_WIDTH
from .model import PolicyValueNet, create_population


@dataclass(frozen=True, slots=True)
class PPOStats:
    decisions: int
    updates: int
    loss_last: float
    policy_loss_last: float
    value_loss_last: float
    parameter_delta_l2: float
    all_finite: bool


@dataclass(slots=True)
class _TrainingDecision:
    step: TrajectoryStep
    advantage: float
    return_value: float


def compute_gae(
    steps: Sequence[TrajectoryStep],
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(steps), dtype=np.float32)
    gae = 0.0
    for index in range(len(steps) - 1, -1, -1):
        next_value = steps[index + 1].old_value if index + 1 < len(steps) else 0.0
        delta = steps[index].reward + gamma * next_value - steps[index].old_value
        gae = delta + gamma * gae_lambda * gae
        advantages[index] = gae
    values = np.asarray([step.old_value for step in steps], dtype=np.float32)
    return advantages, advantages + values


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    values = np.asarray(advantages, dtype=np.float32)
    if values.size < 2:
        return values.copy()
    deviation = float(values.std())
    if deviation < 1e-6:
        return values.copy()
    return (values - values.mean()) / (deviation + 1e-6)


def _ordered_log_probability(
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


def _mixture_log_probability_torch(
    action: tuple[int, ...],
    baseline_action: tuple[int, ...],
    option_logits: torch.Tensor,
    stop_logit: torch.Tensor,
    min_count: int,
    max_count: int,
    exploration_rate: float,
) -> torch.Tensor:
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("exploration_rate must be between zero and one")
    neural = _ordered_log_probability(
        action,
        option_logits,
        stop_logit,
        min_count,
        max_count,
    )
    negative_infinity = option_logits.new_tensor(-math.inf)
    neural_term = (
        negative_infinity
        if exploration_rate == 0.0
        else neural + math.log(exploration_rate)
    )
    if action != baseline_action:
        return neural_term
    driver_term = (
        negative_infinity
        if exploration_rate == 1.0
        else option_logits.new_tensor(math.log1p(-exploration_rate))
    )
    return torch.logaddexp(driver_term, neural_term)


def _training_decisions(
    games: Sequence[CompletedGame],
) -> dict[MemberId, list[_TrainingDecision]]:
    by_member: dict[MemberId, list[_TrainingDecision]] = {member: [] for member in MemberId}
    for game in games:
        for member in MemberId:
            member_steps = [step for step in game.steps if step.member is member]
            if not member_steps:
                continue
            advantages, returns = compute_gae(member_steps)
            by_member[member].extend(
                _TrainingDecision(step, float(advantage), float(return_value))
                for step, advantage, return_value in zip(
                    member_steps, advantages, returns, strict=True
                )
            )
    for member, decisions in by_member.items():
        normalized = normalize_advantages(
            np.asarray([decision.advantage for decision in decisions], dtype=np.float32)
        )
        for decision, advantage in zip(decisions, normalized, strict=True):
            decision.advantage = float(advantage)
    return by_member


def _padded_steps(
    decisions: Sequence[_TrainingDecision],
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(len(decision.step.features) for decision in decisions)
    features = np.zeros((len(decisions), maximum, INPUT_WIDTH), dtype=np.float32)
    mask = np.zeros((len(decisions), maximum), dtype=bool)
    for index, decision in enumerate(decisions):
        count = len(decision.step.features)
        features[index, :count] = decision.step.features
        mask[index, :count] = True
    return features, mask


def _update_one(
    model: PolicyValueNet,
    decisions: Sequence[_TrainingDecision],
    device: str,
    seed: int,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> PPOStats:
    if not decisions:
        raise InvalidSelfPlay("current member received no decisions")
    rng = np.random.default_rng(seed)
    before = [parameter.detach().cpu().numpy().copy() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    finite = True
    for _ in range(epochs):
        order = rng.permutation(len(decisions))
        for start in range(0, len(order), batch_size):
            batch = [decisions[int(index)] for index in order[start : start + batch_size]]
            padded, mask = _padded_steps(batch)
            option_logits, stop_logits, values = model(
                torch.from_numpy(padded).to(device),
                torch.from_numpy(mask).to(device),
            )
            new_log_probabilities = torch.stack(
                [
                    _ordered_log_probability(
                        decision.step.action,
                        option_logits[index, : len(decision.step.features)],
                        stop_logits[index],
                        decision.step.min_count,
                        decision.step.max_count,
                    )
                    if decision.step.baseline_action is None
                    else _mixture_log_probability_torch(
                        decision.step.action,
                        decision.step.baseline_action,
                        option_logits[index, : len(decision.step.features)],
                        stop_logits[index],
                        decision.step.min_count,
                        decision.step.max_count,
                        decision.step.exploration_rate,
                    )
                    for index, decision in enumerate(batch)
                ]
            )
            old_log_probabilities = torch.tensor(
                [decision.step.old_log_probability for decision in batch],
                dtype=torch.float32,
                device=device,
            )
            advantages = torch.tensor(
                [decision.advantage for decision in batch],
                dtype=torch.float32,
                device=device,
            )
            returns = torch.tensor(
                [decision.return_value for decision in batch],
                dtype=torch.float32,
                device=device,
            )
            ratio = torch.exp(new_log_probabilities - old_log_probabilities)
            surrogate = torch.minimum(
                ratio * advantages,
                torch.clamp(ratio, 0.8, 1.2) * advantages,
            )
            policy_loss = -surrogate.mean()
            value_loss = functional.mse_loss(values, returns)
            log_options = torch.log_softmax(option_logits, dim=1)
            probabilities = torch.softmax(option_logits, dim=1)
            entropy = -(probabilities * log_options).sum(dim=1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            numbers = (
                float(loss.detach().cpu()),
                float(policy_loss.detach().cpu()),
                float(value_loss.detach().cpu()),
                float(gradient_norm.detach().cpu()),
            )
            finite = finite and all(math.isfinite(number) for number in numbers)
            losses.append(numbers[0])
            policy_losses.append(numbers[1])
            value_losses.append(numbers[2])
    delta_squared = 0.0
    for initial, parameter in zip(before, model.parameters(), strict=True):
        difference = parameter.detach().cpu().numpy().astype(np.float64) - initial
        delta_squared += float(np.square(difference).sum())
    return PPOStats(
        decisions=len(decisions),
        updates=len(losses),
        loss_last=losses[-1],
        policy_loss_last=policy_losses[-1],
        value_loss_last=value_losses[-1],
        parameter_delta_l2=math.sqrt(delta_squared),
        all_finite=finite,
    )


def update_population(
    population: dict[MemberId, PolicyValueNet],
    games: Sequence[CompletedGame],
    device: str,
    seed: int,
    *,
    epochs: int = 4,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
) -> dict[MemberId, PPOStats]:
    audit = audit_training_batch(
        [game.provenance for game in games],
        set(MemberId),
    )
    if not audit.valid:
        raise InvalidSelfPlay("; ".join(audit.reasons))
    frozen = _training_decisions(games)
    return {
        member: _update_one(
            population[member],
            frozen[member],
            device,
            seed + index,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        for index, member in enumerate(MemberId)
    }


def _model_hashes(population: dict[MemberId, PolicyValueNet]) -> dict[MemberId, str]:
    result: dict[MemberId, str] = {}
    for member, model in population.items():
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
        result[member] = digest.hexdigest()
    return result


def _synthetic_games(
    population: dict[MemberId, PolicyValueNet],
    seed: int,
) -> list[CompletedGame]:
    rng = np.random.default_rng(seed)
    games: list[CompletedGame] = []
    for game_index, (first, second) in enumerate(itertools.combinations(MemberId, 2)):
        steps: list[TrajectoryStep] = []
        for turn in range(3):
            for seat, member in enumerate((first, second)):
                features = rng.normal(0.0, 0.2, size=(3, INPUT_WIDTH)).astype(np.float32)
                model = population[member]
                with torch.no_grad():
                    tensor = torch.from_numpy(features)[None, :, :]
                    mask = torch.ones((1, len(features)), dtype=torch.bool)
                    option_logits, stop_logits, values = model(tensor, mask)
                sample = sample_action(
                    option_logits[0].numpy(),
                    float(stop_logits[0]),
                    1,
                    1,
                    rng,
                )
                steps.append(
                    TrajectoryStep(
                        member=member,
                        seat=seat,
                        features=features,
                        action=sample.indices,
                        min_count=1,
                        max_count=1,
                        old_log_probability=sample.log_probability,
                        old_value=float(values[0]),
                    )
                )
        winner = game_index % 2
        for seat in (0, 1):
            seat_steps = [step for step in steps if step.seat == seat]
            seat_steps[-1].reward = 1.0 if seat == winner else -1.0
        games.append(
            CompletedGame(
                provenance=GameProvenance.current_game(first, second),
                winner=winner,
                decisions=len(steps),
                steps=steps,
                finished=True,
            )
        )
    return games


def ppo_smoke() -> dict[str, object]:
    population = create_population(20260804, "cpu")
    history = copy.deepcopy(population)
    history_before = _model_hashes(history)
    games = _synthetic_games(population, 20260804)
    stats = update_population(
        population,
        games,
        "cpu",
        20260804,
        epochs=2,
        batch_size=32,
    )
    hand_steps = [
        TrajectoryStep(
            member=MemberId.GRIMMSNARL,
            seat=0,
            features=np.zeros((1, INPUT_WIDTH), dtype=np.float32),
            action=(0,),
            min_count=1,
            max_count=1,
            old_log_probability=0.0,
            old_value=0.0,
            reward=reward,
        )
        for reward in (0.0, 0.0, 1.0)
    ]
    hand_gae, _ = compute_gae(hand_steps)
    return {
        "updated_members": sorted(member.value for member in stats),
        "parameter_delta_l2": {
            member.value: member_stats.parameter_delta_l2
            for member, member_stats in stats.items()
        },
        "all_finite": all(member_stats.all_finite for member_stats in stats.values()),
        "historical_snapshots_unchanged": _model_hashes(history) == history_before,
        "constant_negative_advantages": normalize_advantages(
            np.full(8, -1.0, dtype=np.float32)
        ).tolist(),
        "hand_gae": hand_gae.tolist(),
    }
