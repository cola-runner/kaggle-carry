from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ActionSample:
    indices: tuple[int, ...]
    log_probability: float


def _validated_logits(
    option_logits: np.ndarray,
    stop_logit: float,
    min_count: int,
    max_count: int,
) -> np.ndarray:
    logits = np.asarray(option_logits, dtype=np.float64)
    if logits.ndim != 1 or not np.isfinite(logits).all() or not math.isfinite(stop_logit):
        raise ValueError("action logits must be a finite one-dimensional array")
    if min_count < 0 or min_count > max_count or max_count > len(logits):
        raise ValueError("invalid selection bounds")
    return logits


def _log_normalizer(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def action_log_probability(
    indices: tuple[int, ...] | list[int],
    option_logits: np.ndarray,
    stop_logit: float,
    min_count: int,
    max_count: int,
) -> float:
    logits = _validated_logits(option_logits, stop_logit, min_count, max_count)
    action = tuple(indices)
    if not min_count <= len(action) <= max_count:
        raise ValueError("action length is outside selection bounds")
    if len(set(action)) != len(action):
        raise ValueError("action indices must be distinct")
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(logits) for index in action):
        raise ValueError("action index out of range")

    selected: list[int] = []
    log_probability = 0.0
    for chosen in action:
        remaining = [index for index in range(len(logits)) if index not in selected]
        candidate_logits = [float(logits[index]) for index in remaining]
        if len(selected) >= min_count and min_count != max_count:
            candidate_logits.append(float(stop_logit))
        log_probability += float(logits[chosen]) - _log_normalizer(candidate_logits)
        selected.append(chosen)

    if len(selected) < max_count:
        remaining = [index for index in range(len(logits)) if index not in selected]
        candidate_logits = [float(logits[index]) for index in remaining]
        candidate_logits.append(float(stop_logit))
        log_probability += float(stop_logit) - _log_normalizer(candidate_logits)
    return log_probability


def sample_action(
    option_logits: np.ndarray,
    stop_logit: float,
    min_count: int,
    max_count: int,
    rng: np.random.Generator,
) -> ActionSample:
    logits = _validated_logits(option_logits, stop_logit, min_count, max_count)
    selected: list[int] = []

    while len(selected) < max_count:
        remaining = [index for index in range(len(logits)) if index not in selected]
        can_stop = len(selected) >= min_count and min_count != max_count
        candidate_logits = [float(logits[index]) for index in remaining]
        if can_stop:
            candidate_logits.append(float(stop_logit))
        normalizer = _log_normalizer(candidate_logits)
        probabilities = np.exp(np.asarray(candidate_logits, dtype=np.float64) - normalizer)
        choice = int(rng.choice(len(probabilities), p=probabilities))
        if can_stop and choice == len(remaining):
            break
        selected.append(remaining[choice])

    action = tuple(selected)
    return ActionSample(
        indices=action,
        log_probability=action_log_probability(
            action,
            logits,
            stop_logit,
            min_count,
            max_count,
        ),
    )


def greedy_action(
    option_logits: np.ndarray,
    stop_logit: float,
    min_count: int,
    max_count: int,
) -> tuple[int, ...]:
    logits = _validated_logits(option_logits, stop_logit, min_count, max_count)
    selected: list[int] = []
    while len(selected) < max_count:
        remaining = [index for index in range(len(logits)) if index not in selected]
        best = max(remaining, key=lambda index: (float(logits[index]), -index))
        can_stop = len(selected) >= min_count and min_count != max_count
        if can_stop and stop_logit >= float(logits[best]):
            break
        selected.append(best)
    return tuple(selected)
