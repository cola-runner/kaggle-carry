from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .contracts import InvalidSelfPlay, MemberId, audit_training_batch
from .engine import CompletedGame, TrajectoryStep
from .features import INPUT_WIDTH
from .model import PolicyValueNet


ACTION_VALUE_HIDDEN = 128


class ActionValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(INPUT_WIDTH, ACTION_VALUE_HIDDEN)
        self.option_head = nn.Linear(ACTION_VALUE_HIDDEN, 1)

    def forward(self, features, mask):
        hidden = torch.tanh(self.layer(features))
        option_values = self.option_head(hidden).squeeze(-1).masked_fill(
            ~mask,
            -1e9,
        )
        zeros = torch.zeros(
            features.shape[0],
            dtype=features.dtype,
            device=features.device,
        )
        weights = mask.to(features.dtype)
        state_values = (option_values * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        return option_values, zeros, state_values


def create_action_value_population(
    seed: int,
    device: str,
) -> dict[MemberId, ActionValueNet]:
    torch.manual_seed(seed)
    return {member: ActionValueNet().to(device) for member in MemberId}


def action_value_parameter_count(model: ActionValueNet) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def export_action_value_member(model: ActionValueNet, path: str) -> None:
    np.savez_compressed(
        path,
        w=model.layer.weight.detach().cpu().numpy().T.astype(np.float32),
        b=model.layer.bias.detach().cpu().numpy().astype(np.float32),
        w_option=model.option_head.weight.detach().cpu().numpy()[0].astype(
            np.float32
        ),
        b_option=model.option_head.bias.detach().cpu().numpy().astype(np.float32),
    )


@dataclass(frozen=True, slots=True)
class ActionValueExample:
    features: np.ndarray
    option_index: int
    target: float
    weight: float


@dataclass(frozen=True, slots=True)
class ActionValueStats:
    games: int
    decisions: int
    updates: int
    loss_last: float
    parameter_delta_l2: float
    all_finite: bool


def _eligible(step: TrajectoryStep) -> bool:
    return (
        step.min_count == step.max_count == 1
        and len(step.action) == 1
        and len(step.features) > 1
        and 0 <= step.action[0] < len(step.features)
    )


def build_action_value_examples(
    games: Sequence[CompletedGame],
) -> dict[MemberId, list[ActionValueExample]]:
    result = {member: [] for member in MemberId}
    for game in games:
        if not game.finished:
            continue
        for member in MemberId:
            member_steps = [
                step
                for step in game.steps
                if step.member is member and _eligible(step)
            ]
            if not member_steps:
                continue
            seat = member_steps[0].seat
            target = (
                0.0
                if game.winner not in (0, 1)
                else (1.0 if game.winner == seat else -1.0)
            )
            weight = 1.0 / len(member_steps)
            result[member].extend(
                ActionValueExample(
                    features=step.features,
                    option_index=step.action[0],
                    target=target,
                    weight=weight,
                )
                for step in member_steps
            )
    return result


def _padded(
    rows: Sequence[ActionValueExample],
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(len(row.features) for row in rows)
    features = np.zeros((len(rows), maximum, INPUT_WIDTH), dtype=np.float32)
    mask = np.zeros((len(rows), maximum), dtype=bool)
    for index, row in enumerate(rows):
        count = len(row.features)
        features[index, :count] = row.features
        mask[index, :count] = True
    return features, mask


def _update_one(
    model: PolicyValueNet | ActionValueNet,
    rows: Sequence[ActionValueExample],
    device: str,
    seed: int,
    *,
    games: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> ActionValueStats:
    if not rows:
        raise InvalidSelfPlay("current member received no action-value examples")
    rng = np.random.default_rng(seed)
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(("stop_head.", "value_head."))
    ]
    before = [parameter.detach().cpu().numpy().copy() for parameter in parameters]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=1e-5,
    )
    losses: list[float] = []
    finite = True
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), batch_size):
            batch = [rows[int(index)] for index in order[start : start + batch_size]]
            padded, mask = _padded(batch)
            option_values, _, _ = model(
                torch.from_numpy(padded).to(device),
                torch.from_numpy(mask).to(device),
            )
            indices = torch.tensor(
                [row.option_index for row in batch],
                dtype=torch.long,
                device=device,
            )
            predictions = option_values[
                torch.arange(len(batch), device=device),
                indices,
            ]
            targets = torch.tensor(
                [row.target for row in batch],
                dtype=torch.float32,
                device=device,
            )
            weights = torch.tensor(
                [row.weight for row in batch],
                dtype=torch.float32,
                device=device,
            )
            per_row = functional.smooth_l1_loss(
                predictions,
                targets,
                reduction="none",
            )
            loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            numbers = (
                float(loss.detach().cpu()),
                float(gradient_norm.detach().cpu()),
            )
            finite = finite and all(math.isfinite(number) for number in numbers)
            losses.append(numbers[0])
    delta_squared = 0.0
    for initial, parameter in zip(before, parameters, strict=True):
        difference = parameter.detach().cpu().numpy().astype(np.float64) - initial
        delta_squared += float(np.square(difference).sum())
    return ActionValueStats(
        games=games,
        decisions=len(rows),
        updates=len(losses),
        loss_last=losses[-1],
        parameter_delta_l2=math.sqrt(delta_squared),
        all_finite=finite,
    )


def update_action_values(
    population: Mapping[MemberId, PolicyValueNet | ActionValueNet],
    games: Sequence[CompletedGame],
    device: str,
    seed: int,
    *,
    epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
) -> dict[MemberId, ActionValueStats]:
    audit = audit_training_batch(
        [game.provenance for game in games],
        set(MemberId),
    )
    if not audit.valid:
        raise InvalidSelfPlay("; ".join(audit.reasons))
    examples = build_action_value_examples(games)
    return {
        member: _update_one(
            population[member],
            examples[member],
            device,
            seed + index,
            games=sum(
                any(step.member is member and _eligible(step) for step in game.steps)
                for game in games
            ),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        for index, member in enumerate(MemberId)
    }
