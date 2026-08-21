from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .actions import action_log_probability, greedy_action
from .bootstrap import BootstrapDecision, BootstrapMemberStats, train_from_decisions
from .contracts import MemberId
from .features import INPUT_WIDTH
from .fidelity_data import PairedDecision, PairedGame
from .model import PolicyValueNet, create_population


@dataclass(frozen=True, slots=True)
class FidelityMemberMetrics:
    nontrivial_decisions: int
    forced_decisions: int
    exact_agreement: float
    negative_log_probability: float


@dataclass(frozen=True, slots=True)
class FidelityMetrics:
    nontrivial_decisions: int
    forced_decisions: int
    exact_agreement: float
    negative_log_probability: float
    members: dict[MemberId, FidelityMemberMetrics]


@dataclass(slots=True)
class PairedTrainingResult:
    v1_population: dict[MemberId, PolicyValueNet]
    v2_population: dict[MemberId, PolicyValueNet]
    v1_stats: dict[MemberId, BootstrapMemberStats]
    v2_stats: dict[MemberId, BootstrapMemberStats]


def _bootstrap_decisions(
    games: Sequence[PairedGame],
    version: str,
) -> dict[MemberId, list[BootstrapDecision]]:
    if version not in ("v1", "v2"):
        raise ValueError("version must be v1 or v2")
    result: dict[MemberId, list[BootstrapDecision]] = {
        member: [] for member in MemberId
    }
    attribute = f"{version}_features"
    for game in games:
        for decision in game.decisions:
            result[decision.member].append(
                BootstrapDecision(
                    features=getattr(decision, attribute),
                    action=decision.action,
                    min_count=decision.min_count,
                    max_count=decision.max_count,
                )
            )
    return result


def train_paired_populations(
    train_games: Sequence[PairedGame],
    device: str,
    seed: int,
    *,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
) -> PairedTrainingResult:
    v1_population = create_population(seed, device)
    v2_population = create_population(seed, device)
    v1_stats = train_from_decisions(
        v1_population,
        _bootstrap_decisions(train_games, "v1"),
        device,
        seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    v2_stats = train_from_decisions(
        v2_population,
        _bootstrap_decisions(train_games, "v2"),
        device,
        seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return PairedTrainingResult(
        v1_population,
        v2_population,
        v1_stats,
        v2_stats,
    )


def _padded_features(
    decisions: Sequence[PairedDecision],
    version: str,
) -> tuple[np.ndarray, np.ndarray]:
    attribute = f"{version}_features"
    maximum = max(len(getattr(decision, attribute)) for decision in decisions)
    features = np.zeros((len(decisions), maximum, INPUT_WIDTH), dtype=np.float32)
    mask = np.zeros((len(decisions), maximum), dtype=bool)
    for index, decision in enumerate(decisions):
        matrix = getattr(decision, attribute)
        features[index, : len(matrix)] = matrix
        mask[index, : len(matrix)] = True
    return features, mask


def _member_metrics(
    model: PolicyValueNet,
    decisions: Sequence[PairedDecision],
    version: str,
    device: str,
    batch_size: int,
) -> FidelityMemberMetrics:
    nontrivial = 0
    forced = 0
    agreements = 0
    negative_log_probability = 0.0
    model.eval()
    for start in range(0, len(decisions), batch_size):
        batch = decisions[start : start + batch_size]
        padded, mask = _padded_features(batch, version)
        with torch.no_grad():
            option_logits, stop_logits, _ = model(
                torch.from_numpy(padded).to(device),
                torch.from_numpy(mask).to(device),
            )
        option_numpy = option_logits.detach().cpu().numpy()
        stop_numpy = stop_logits.detach().cpu().numpy()
        for index, decision in enumerate(batch):
            features = getattr(decision, f"{version}_features")
            if len(features) <= 1 and decision.min_count == decision.max_count:
                forced += 1
                continue
            logits = option_numpy[index, : len(features)].astype(np.float64)
            stop_logit = float(stop_numpy[index])
            predicted = greedy_action(
                logits,
                stop_logit,
                decision.min_count,
                decision.max_count,
            )
            agreements += int(predicted == decision.action)
            negative_log_probability -= action_log_probability(
                decision.action,
                logits,
                stop_logit,
                decision.min_count,
                decision.max_count,
            )
            nontrivial += 1
    if nontrivial == 0:
        raise ValueError("held-out member has no non-trivial decisions")
    average_nll = negative_log_probability / nontrivial
    if not math.isfinite(average_nll):
        raise ValueError("held-out negative log probability is not finite")
    return FidelityMemberMetrics(
        nontrivial_decisions=nontrivial,
        forced_decisions=forced,
        exact_agreement=agreements / nontrivial,
        negative_log_probability=average_nll,
    )


def evaluate_population(
    population: Mapping[MemberId, PolicyValueNet],
    held_out_games: Sequence[PairedGame],
    version: str,
    device: str,
    *,
    batch_size: int = 256,
) -> FidelityMetrics:
    if version not in ("v1", "v2"):
        raise ValueError("version must be v1 or v2")
    by_member: dict[MemberId, list[PairedDecision]] = {
        member: [] for member in MemberId
    }
    for game in held_out_games:
        for decision in game.decisions:
            by_member[decision.member].append(decision)
    members = {
        member: _member_metrics(
            population[member],
            by_member[member],
            version,
            device,
            batch_size,
        )
        for member in MemberId
    }
    total_nontrivial = sum(item.nontrivial_decisions for item in members.values())
    total_forced = sum(item.forced_decisions for item in members.values())
    weighted_agreement = sum(
        item.exact_agreement * item.nontrivial_decisions for item in members.values()
    )
    weighted_nll = sum(
        item.negative_log_probability * item.nontrivial_decisions
        for item in members.values()
    )
    return FidelityMetrics(
        nontrivial_decisions=total_nontrivial,
        forced_decisions=total_forced,
        exact_agreement=weighted_agreement / total_nontrivial,
        negative_log_probability=weighted_nll / total_nontrivial,
        members=members,
    )
