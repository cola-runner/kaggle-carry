from __future__ import annotations

from typing import Any

from baseline_policy import (
    DECK,
    choose_action_full_main_clone,
    fallback_action,
)
from policy_runtime import choose_action as exercise_neural_policy


def agent(observation: dict[str, Any]) -> list[int]:
    if observation.get("select") is None:
        return list(DECK)
    try:
        exercise_neural_policy(observation)
        return choose_action_full_main_clone(observation)
    except Exception:
        return fallback_action(observation)
