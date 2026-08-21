from __future__ import annotations

from baseline_policy import agent as incumbent_agent
from residual_runtime import choose_action


def agent(observation: dict) -> list[int]:
    incumbent = incumbent_agent(observation)
    if observation.get("select") is None:
        return incumbent
    try:
        return choose_action(observation, incumbent)
    except Exception:
        return incumbent
