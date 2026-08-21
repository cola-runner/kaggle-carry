from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from .action_value import ActionValueNet
from .contracts import MemberId
from .features import INPUT_WIDTH


MAX_TARGET_ORDINAL = 32


@dataclass(frozen=True, slots=True)
class CalibrationKey:
    member: MemberId
    opponent: MemberId
    seat: int

    def __post_init__(self) -> None:
        if self.member is self.opponent:
            raise ValueError("calibration member and opponent must differ")
        if self.seat not in (0, 1):
            raise ValueError("calibration seat must be zero or one")


@dataclass(frozen=True, slots=True)
class CalibrationCell:
    points: float
    games: int

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError("calibration games must be positive")
        if not math.isfinite(self.points) or not 0.0 <= self.points <= self.games:
            raise ValueError("calibration points must be finite and within games")

    @property
    def expected_score(self) -> float:
        return (self.points + 1.0) / (self.games + 2.0)


def centered_label(actual_score: float, cell: CalibrationCell) -> float:
    if not math.isfinite(actual_score) or not 0.0 <= actual_score <= 1.0:
        raise ValueError("actual score must be finite and between zero and one")
    return float(np.clip(actual_score - cell.expected_score, -0.5, 0.5))


def eligible_intervention(
    observation: Mapping[str, Any],
    incumbent_action: Sequence[int],
) -> bool:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return False
    options = select.get("option")
    if not isinstance(options, list) or len(options) <= 1:
        return False
    return (
        int(select.get("type", -1)) == 0
        and int(select.get("context", -1)) == 0
        and int(select.get("minCount", -1)) == 1
        and int(select.get("maxCount", -1)) == 1
        and len(incumbent_action) == 1
        and isinstance(incumbent_action[0], int)
        and 0 <= incumbent_action[0] < len(options)
    )


@dataclass(slots=True)
class InterventionTracker:
    target_ordinal: int
    eligible_seen: int = 0
    used: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.target_ordinal <= MAX_TARGET_ORDINAL:
            raise ValueError("target ordinal must be between 1 and 32")

    def consider(
        self,
        observation: Mapping[str, Any],
        incumbent_action: Sequence[int],
    ) -> bool:
        if self.used or not eligible_intervention(observation, incumbent_action):
            return False
        self.eligible_seen += 1
        return self.eligible_seen == self.target_ordinal

    def mark_used(self) -> None:
        if self.used:
            raise ValueError("intervention already used")
        self.used = True


def choose_trial_index(
    option_scores: np.ndarray | Sequence[float],
    incumbent_index: int,
    rng: np.random.Generator,
) -> int:
    scores = np.asarray(option_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError("trial selection requires at least two one-dimensional scores")
    if not np.isfinite(scores).all():
        raise ValueError("trial scores must be finite")
    if not 0 <= incumbent_index < len(scores):
        raise ValueError("incumbent index is out of range")
    candidates = np.asarray(
        [index for index in range(len(scores)) if index != incumbent_index],
        dtype=np.int64,
    )
    candidate_scores = scores[candidates]
    shifted = candidate_scores - float(candidate_scores.max())
    weights = np.exp(shifted)
    probabilities = weights / weights.sum()
    return int(rng.choice(candidates, p=probabilities))


@dataclass(frozen=True, slots=True)
class InterventionExample:
    member: MemberId
    opponent: MemberId
    seat: int
    round_index: int
    target_ordinal: int
    features: np.ndarray
    incumbent_index: int
    trial_index: int
    label: float

    def __post_init__(self) -> None:
        if self.member is self.opponent:
            raise ValueError("intervention member and opponent must differ")
        if self.seat not in (0, 1):
            raise ValueError("intervention seat must be zero or one")
        if self.round_index not in (1, 2):
            raise ValueError("intervention round must be one or two")
        if not 1 <= self.target_ordinal <= MAX_TARGET_ORDINAL:
            raise ValueError("target ordinal must be between 1 and 32")
        if (
            not isinstance(self.features, np.ndarray)
            or self.features.ndim != 2
            or self.features.shape[1] != INPUT_WIDTH
            or len(self.features) < 2
            or not np.isfinite(self.features).all()
        ):
            raise ValueError("intervention features have invalid shape or values")
        if not 0 <= self.incumbent_index < len(self.features):
            raise ValueError("incumbent index is out of range")
        if not 0 <= self.trial_index < len(self.features):
            raise ValueError("trial index is out of range")
        if self.incumbent_index == self.trial_index:
            raise ValueError("trial and incumbent indices must differ")
        if not math.isfinite(self.label) or not -0.5 <= self.label <= 0.5:
            raise ValueError("intervention label must be finite and clipped")


@dataclass(frozen=True, slots=True)
class IncumbentDecision:
    member: MemberId
    features: np.ndarray
    incumbent_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.features, np.ndarray)
            or self.features.ndim != 2
            or self.features.shape[1] != INPUT_WIDTH
            or len(self.features) < 2
            or not np.isfinite(self.features).all()
        ):
            raise ValueError("incumbent decision features have invalid shape or values")
        if not 0 <= self.incumbent_index < len(self.features):
            raise ValueError("incumbent decision index is out of range")


@dataclass(frozen=True, slots=True)
class _PairwiseExample:
    features: np.ndarray
    incumbent_index: int
    trial_index: int
    label: float


@dataclass(slots=True)
class InterventionEnsemble:
    models: tuple[ActionValueNet, ActionValueNet]

    def __post_init__(self) -> None:
        if len(self.models) != 2 or self.models[0] is self.models[1]:
            raise ValueError("intervention ensemble requires two independent models")


@dataclass(frozen=True, slots=True)
class InterventionUpdateStats:
    examples: int
    updates: int
    loss_last: float
    parameter_delta_l2: tuple[float, float]
    all_finite: bool


def create_intervention_population(
    seed: int,
    device: str,
) -> dict[MemberId, InterventionEnsemble]:
    population: dict[MemberId, InterventionEnsemble] = {}
    for member_index, member in enumerate(MemberId):
        models = []
        for model_index in range(2):
            torch.manual_seed(seed + member_index * 100 + model_index)
            models.append(ActionValueNet().to(device))
        population[member] = InterventionEnsemble((models[0], models[1]))
    return population


def mean_option_scores(
    score_rows: Sequence[Sequence[float] | np.ndarray],
) -> np.ndarray:
    if len(score_rows) != 2:
        raise ValueError("two score rows are required")
    rows = [np.asarray(row, dtype=np.float64) for row in score_rows]
    if (
        rows[0].ndim != 1
        or rows[1].ndim != 1
        or rows[0].shape != rows[1].shape
        or not rows[0].size
        or not all(np.isfinite(row).all() for row in rows)
    ):
        raise ValueError("score rows must have equal finite one-dimensional shape")
    return (rows[0] + rows[1]) / 2.0


def trusted_override(
    score_rows: Sequence[Sequence[float] | np.ndarray],
    incumbent_index: int,
    *,
    margin: float | Sequence[float] = 0.25,
) -> int | None:
    if len(score_rows) != 2:
        raise ValueError("two score rows are required")
    rows = [np.asarray(row, dtype=np.float64) for row in score_rows]
    mean = mean_option_scores(rows)
    margins = (
        (float(margin), float(margin))
        if isinstance(margin, (int, float))
        else tuple(float(value) for value in margin)
    )
    if len(margins) != 2 or any(
        not math.isfinite(value) or value < 0.0 for value in margins
    ):
        raise ValueError("two finite non-negative override margins are required")
    if not 0 <= incumbent_index < len(mean):
        raise ValueError("incumbent index is out of range")
    eligible = [
        index
        for index in range(len(mean))
        if index != incumbent_index
        and all(
            float(row[index] - row[incumbent_index]) > margins[model_index]
            for model_index, row in enumerate(rows)
        )
    ]
    return max(eligible, key=lambda index: float(mean[index])) if eligible else None


def calibrated_override_margins(
    score_rows: Sequence[Sequence[Sequence[float] | np.ndarray]],
    incumbent_indices: Sequence[int],
    *,
    quantile: float = 0.995,
    minimum: float = 0.25,
) -> tuple[float, float]:
    if len(score_rows) != len(incumbent_indices) or not score_rows:
        raise ValueError("calibration scores and incumbents must be non-empty and aligned")
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("calibration quantile must be between zero and one")
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum margin must be finite and non-negative")
    maximum_gaps: list[list[float]] = [[], []]
    for decision_rows, incumbent_index in zip(
        score_rows,
        incumbent_indices,
        strict=True,
    ):
        if len(decision_rows) != 2:
            raise ValueError("each calibration decision requires two model score rows")
        rows = [np.asarray(row, dtype=np.float64) for row in decision_rows]
        mean_option_scores(rows)
        if not 0 <= incumbent_index < len(rows[0]):
            raise ValueError("calibration incumbent index is out of range")
        alternatives = [
            index for index in range(len(rows[0])) if index != incumbent_index
        ]
        if not alternatives:
            raise ValueError("calibration decision requires an alternative")
        for model_index, row in enumerate(rows):
            maximum_gaps[model_index].append(
                max(
                    float(row[index] - row[incumbent_index])
                    for index in alternatives
                )
            )
    return tuple(
        max(
            minimum,
            float(np.quantile(gaps, quantile, method="higher")),
        )
        for gaps in maximum_gaps
    )  # type: ignore[return-value]


def model_option_scores(
    model: ActionValueNet,
    features: np.ndarray,
    device: str,
) -> np.ndarray:
    if (
        not isinstance(features, np.ndarray)
        or features.ndim != 2
        or features.shape[1] != INPUT_WIDTH
        or not len(features)
    ):
        raise ValueError("option features have invalid shape")
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32, copy=False))[None].to(
            device
        )
        mask = torch.ones((1, len(features)), dtype=torch.bool, device=device)
        values, _, _ = model(tensor, mask)
    scores = values[0].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("model produced non-finite option scores")
    return scores


def _padded_examples(
    rows: Sequence[InterventionExample | _PairwiseExample],
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(len(row.features) for row in rows)
    features = np.zeros((len(rows), maximum, INPUT_WIDTH), dtype=np.float32)
    mask = np.zeros((len(rows), maximum), dtype=bool)
    for index, row in enumerate(rows):
        count = len(row.features)
        features[index, :count] = row.features
        mask[index, :count] = True
    return features, mask


def _parameter_snapshot(model: ActionValueNet) -> list[np.ndarray]:
    return [parameter.detach().cpu().numpy().copy() for parameter in model.parameters()]


def _parameter_delta_l2(
    before: Sequence[np.ndarray],
    model: ActionValueNet,
) -> float:
    total = 0.0
    for initial, parameter in zip(before, model.parameters(), strict=True):
        difference = parameter.detach().cpu().numpy().astype(np.float64) - initial
        total += float(np.square(difference).sum())
    return math.sqrt(total)


def update_intervention_ensemble(
    ensemble: InterventionEnsemble,
    examples: Sequence[InterventionExample],
    device: str,
    seed: int,
    *,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
) -> InterventionUpdateStats:
    rows = [row for row in examples if row.label != 0.0]
    if not rows:
        raise ValueError("pairwise update requires non-zero intervention labels")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("training parameters must be positive")
    losses: list[float] = []
    deltas: list[float] = []
    finite = True
    updates_per_model = 0
    for model_index, model in enumerate(ensemble.models):
        before = _parameter_snapshot(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        rng = np.random.default_rng(seed + model_index)
        model.train()
        model_updates = 0
        for _ in range(epochs):
            order = rng.permutation(len(rows))
            for start in range(0, len(order), batch_size):
                batch = [rows[int(index)] for index in order[start : start + batch_size]]
                padded, mask = _padded_examples(batch)
                values, _, _ = model(
                    torch.from_numpy(padded).to(device),
                    torch.from_numpy(mask).to(device),
                )
                row_indices = torch.arange(len(batch), device=device)
                trial_indices = torch.tensor(
                    [row.trial_index for row in batch],
                    dtype=torch.long,
                    device=device,
                )
                incumbent_indices = torch.tensor(
                    [row.incumbent_index for row in batch],
                    dtype=torch.long,
                    device=device,
                )
                labels = torch.tensor(
                    [row.label for row in batch],
                    dtype=torch.float32,
                    device=device,
                )
                gaps = values[row_indices, trial_indices] - values[
                    row_indices,
                    incumbent_indices,
                ]
                weights = torch.maximum(
                    labels.abs(),
                    labels.new_tensor(0.05),
                )
                loss = (
                    functional.softplus(-torch.sign(labels) * gaps) * weights
                ).sum() / weights.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
                optimizer.step()
                numbers = (
                    float(loss.detach().cpu()),
                    float(gradient_norm.detach().cpu()),
                )
                finite = finite and all(math.isfinite(number) for number in numbers)
                losses.append(numbers[0])
                model_updates += 1
        updates_per_model = model_updates
        deltas.append(_parameter_delta_l2(before, model))
    return InterventionUpdateStats(
        examples=len(rows),
        updates=updates_per_model,
        loss_last=losses[-1],
        parameter_delta_l2=(deltas[0], deltas[1]),
        all_finite=finite and all(delta > 0.0 for delta in deltas),
    )


def pretrain_incumbent_population(
    population: Mapping[MemberId, InterventionEnsemble],
    decisions: Mapping[MemberId, Sequence[IncumbentDecision]],
    device: str,
    seed: int,
    *,
    members: Sequence[MemberId] = tuple(MemberId),
    max_decisions: int = 2048,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
) -> dict[MemberId, InterventionUpdateStats]:
    if max_decisions <= 0:
        raise ValueError("maximum incumbent decisions must be positive")
    result: dict[MemberId, InterventionUpdateStats] = {}
    selected_members = tuple(dict.fromkeys(members))
    if not selected_members:
        raise ValueError("incumbent pretraining requires at least one member")
    for member_index, member in enumerate(selected_members):
        member_decisions = list(decisions.get(member, ()))[:max_decisions]
        if not member_decisions:
            raise ValueError(f"{member.value} received no incumbent decisions")
        if any(decision.member is not member for decision in member_decisions):
            raise ValueError("incumbent decision routed to the wrong member")
        rows = [
            _PairwiseExample(
                features=decision.features,
                incumbent_index=decision.incumbent_index,
                trial_index=alternative,
                label=-0.5,
            )
            for decision in member_decisions
            for alternative in range(len(decision.features))
            if alternative != decision.incumbent_index
        ]
        result[member] = update_intervention_ensemble(
            population[member],
            rows,
            device,
            seed=seed + member_index * 100,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    return result


def update_intervention_population(
    population: Mapping[MemberId, InterventionEnsemble],
    examples: Mapping[MemberId, Sequence[InterventionExample]],
    device: str,
    seed: int,
    *,
    members: Sequence[MemberId] = tuple(MemberId),
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
) -> dict[MemberId, InterventionUpdateStats]:
    result: dict[MemberId, InterventionUpdateStats] = {}
    selected_members = tuple(dict.fromkeys(members))
    if not selected_members:
        raise ValueError("intervention update requires at least one member")
    for member_index, member in enumerate(selected_members):
        rows = list(examples.get(member, ()))
        if any(row.member is not member for row in rows):
            raise ValueError("intervention example routed to the wrong member")
        result[member] = update_intervention_ensemble(
            population[member],
            rows,
            device,
            seed=seed + member_index * 100,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    return result
