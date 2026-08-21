from __future__ import annotations

from typing import Any

from policy_runtime import (
    DECK,
    choose_action_full_main_clone,
    fallback_action,
)


def agent(observation: dict[str, Any]) -> list[int]:
    if observation.get("select") is None:
        return list(DECK)
    try:
        return choose_action_full_main_clone(observation)
    except Exception:
        return fallback_action(observation)
