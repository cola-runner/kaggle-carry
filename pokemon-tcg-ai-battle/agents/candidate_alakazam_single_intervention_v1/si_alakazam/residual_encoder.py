from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .extract import sanitize_visible_observation
from .residual_features import visible_action_features, visible_state_features


INPUT_WIDTH = 512


def _dense_hash(features: Mapping[str, float]) -> np.ndarray:
    vector = np.zeros(INPUT_WIDTH, dtype=np.float32)
    for name, raw_value in features.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value == 0.0:
            continue
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        index = number % INPUT_WIDTH
        sign = -1.0 if number >> 63 else 1.0
        vector[index] += np.float32(sign * max(-8.0, min(8.0, value)))
    return vector


def encode_options(raw_observation: Mapping[str, Any]) -> np.ndarray:
    current = raw_observation.get("current")
    seat = int(current.get("yourIndex", 0)) if isinstance(current, Mapping) else 0
    visible = sanitize_visible_observation(raw_observation, seat)
    select = visible.get("select")
    options = select.get("option") if isinstance(select, Mapping) else None
    legal_options = (
        [option for option in options if isinstance(option, Mapping)]
        if isinstance(options, list)
        else []
    )
    if not legal_options:
        return np.zeros((0, INPUT_WIDTH), dtype=np.float32)
    base = visible_state_features(visible)
    return np.stack(
        [
            _dense_hash(
                visible_action_features(visible, option, base_features=base)
            )
            for option in legal_options
        ],
        axis=0,
    ).astype(np.float32, copy=False)
