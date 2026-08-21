from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from scripts.run_local_match import validate_action

from .actions import action_log_probability, greedy_action, sample_action
from .contracts import MemberId
from .engine import TrajectoryStep
from .features import encode_options
from .model import PolicyValueNet


def mixture_log_probability(
    action: tuple[int, ...],
    baseline_action: tuple[int, ...],
    option_logits: np.ndarray,
    stop_logit: float,
    min_count: int,
    max_count: int,
    exploration_rate: float,
) -> float:
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("exploration_rate must be between zero and one")
    neural = action_log_probability(
        action,
        option_logits,
        stop_logit,
        min_count,
        max_count,
    )
    if action != baseline_action:
        return -math.inf if exploration_rate == 0.0 else math.log(exploration_rate) + neural
    driver_term = -math.inf if exploration_rate == 1.0 else math.log1p(-exploration_rate)
    neural_term = -math.inf if exploration_rate == 0.0 else math.log(exploration_rate) + neural
    return float(np.logaddexp(driver_term, neural_term))


@dataclass(slots=True)
class ResidualCounters:
    decisions: int = 0
    driver_actions: int = 0
    exploration_actions: int = 0
    evaluation_overrides: int = 0


@dataclass(slots=True)
class DriverBackedActor:
    member: MemberId
    model: PolicyValueNet
    deck: list[int]
    device: str
    driver_action: Callable[[Mapping[str, Any]], list[int]]
    trainable: bool = True
    generation: str = "current"
    exploration_rate: float = 0.1
    overrides_enabled: bool = False
    override_margin: float = 2.0
    override_mode: str = "policy"
    counters: ResidualCounters = field(default_factory=ResidualCounters)

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be between zero and one")
        if self.override_margin < 0.0:
            raise ValueError("override_margin must be non-negative")
        if self.override_mode not in {"policy", "action_value"}:
            raise ValueError("override_mode must be policy or action_value")

    @property
    def name(self) -> str:
        if self.generation == "current":
            return self.member.value
        return f"{self.member.value}-{self.generation}"

    def decide(
        self,
        observation: Mapping[str, Any],
        seat: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], TrajectoryStep | None]:
        select = observation.get("select")
        if not isinstance(select, Mapping):
            raise ValueError("observation has no selection")
        baseline = tuple(
            validate_action(self.driver_action(observation), dict(select))
        )
        options = select.get("option")
        option_count = len(options) if isinstance(options, list) else 0
        min_count = int(select.get("minCount", 0) or 0)
        max_count = int(select.get("maxCount", min_count) or 0)
        if option_count == 0 and min_count == max_count == 0:
            return list(baseline), None
        self.counters.decisions += 1
        if not self.trainable and not self.overrides_enabled:
            self.counters.driver_actions += 1
            return list(baseline), None

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

        if self.trainable:
            neural = sample_action(logits, stop_logit, min_count, max_count, rng)
            explore = bool(rng.random() < self.exploration_rate)
            action = neural.indices if explore else baseline
            self.counters.exploration_actions += int(explore and action != baseline)
            self.counters.driver_actions += int(action == baseline)
            log_probability = mixture_log_probability(
                action,
                baseline,
                logits,
                stop_logit,
                min_count,
                max_count,
                self.exploration_rate,
            )
            validated = validate_action(list(action), dict(select))
            return validated, TrajectoryStep(
                member=self.member,
                seat=seat,
                features=features,
                action=tuple(validated),
                min_count=min_count,
                max_count=max_count,
                old_log_probability=log_probability,
                old_value=value,
                baseline_action=baseline,
                exploration_rate=self.exploration_rate,
            )

        action = baseline
        if self.overrides_enabled:
            context = int(select.get("context", -1) or 0)
            if (
                self.override_mode == "action_value"
                and context == 0
                and min_count == max_count == 1
                and len(baseline) == 1
            ):
                candidate = (int(np.argmax(logits)),)
                margin = float(logits[candidate[0]] - logits[baseline[0]])
            else:
                candidate = greedy_action(
                    logits,
                    stop_logit,
                    min_count,
                    max_count,
                )
                margin = action_log_probability(
                    candidate,
                    logits,
                    stop_logit,
                    min_count,
                    max_count,
                ) - action_log_probability(
                    baseline,
                    logits,
                    stop_logit,
                    min_count,
                    max_count,
                )
            eligible = self.override_mode == "policy" or (
                context == 0
                and min_count == max_count == 1
                and len(baseline) == 1
            )
            if eligible and candidate != baseline:
                if margin >= self.override_margin:
                    action = candidate
                    self.counters.evaluation_overrides += 1
        self.counters.driver_actions += int(action == baseline)
        return validate_action(list(action), dict(select)), None
