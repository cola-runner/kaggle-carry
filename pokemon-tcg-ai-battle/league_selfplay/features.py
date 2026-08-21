from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from rolling_policy.extract import sanitize_visible_observation
from rolling_policy.features import visible_action_features, visible_state_features


INPUT_WIDTH = 512
FEATURE_V2_VERSION = 1
NUMERIC_BUCKETS = 128
CATEGORICAL_BUCKETS = INPUT_WIDTH - NUMERIC_BUCKETS


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


def _scaled_numeric(name: str, value: float) -> float:
    if "hp" in name or "damage" in name:
        return float(np.clip(value / 300.0, -2.0, 2.0))
    if "deck_count" in name:
        return float(np.clip(value / 60.0, -2.0, 2.0))
    if name.endswith("step"):
        return float(np.clip(value / 500.0, -2.0, 2.0))
    if name.endswith("turn"):
        return float(np.clip(value / 50.0, -2.0, 2.0))
    if any(part in name for part in ("card_id", "serial", "attack_id")):
        return 0.0
    return float(np.clip(value / 8.0, -2.0, 2.0))


def _dense_hash_v2(features: Mapping[str, float]) -> np.ndarray:
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
        sign = -1.0 if number >> 63 else 1.0
        if name.startswith("n:"):
            index = number % NUMERIC_BUCKETS
            contribution = _scaled_numeric(name, value)
        else:
            index = NUMERIC_BUCKETS + number % CATEGORICAL_BUCKETS
            contribution = float(np.clip(value, -8.0, 8.0))
        vector[index] += np.float32(sign * contribution)
    return vector


def _encode_options(
    raw_observation: Mapping[str, Any],
    dense_encoder: Callable[[Mapping[str, float]], np.ndarray],
) -> np.ndarray:
    current = raw_observation.get("current")
    seat = int(current.get("yourIndex", 0)) if isinstance(current, Mapping) else 0
    visible = sanitize_visible_observation(raw_observation, seat)
    select = visible.get("select")
    options = select.get("option") if isinstance(select, Mapping) else None
    legal_options = [option for option in options if isinstance(option, Mapping)] if isinstance(options, list) else []
    if not legal_options:
        return np.zeros((0, INPUT_WIDTH), dtype=np.float32)
    base = visible_state_features(visible)
    return np.stack(
        [
            dense_encoder(
                visible_action_features(visible, option, base_features=base)
            )
            for option in legal_options
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def encode_options(raw_observation: Mapping[str, Any]) -> np.ndarray:
    return _encode_options(raw_observation, _dense_hash)


def encode_options_v2(raw_observation: Mapping[str, Any]) -> np.ndarray:
    return _encode_options(raw_observation, _dense_hash_v2)
