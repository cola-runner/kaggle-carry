from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np


POLICY_ARRAYS = (
    "w1",
    "b1",
    "w2",
    "b2",
    "w_option",
    "b_option",
    "w_stop",
    "b_stop",
)


def load_policy(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as checkpoint:
        return {
            name: np.asarray(checkpoint[name], dtype=np.float32)
            for name in POLICY_ARRAYS
        }


def numpy_forward(
    features: np.ndarray,
    weights: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 512:
        raise ValueError("features must have shape [options, 512]")
    hidden = np.tanh(matrix @ weights["w1"] + weights["b1"])
    hidden = np.tanh(hidden @ weights["w2"] + weights["b2"])
    option_logits = hidden @ weights["w_option"] + float(weights["b_option"][0])
    pooled = hidden.mean(axis=0) if len(hidden) else np.zeros(928, dtype=np.float32)
    stop_logit = float(pooled @ weights["w_stop"] + float(weights["b_stop"][0]))
    if not np.isfinite(option_logits).all() or not math.isfinite(stop_logit):
        raise ValueError("non-finite policy output")
    return option_logits.astype(np.float32, copy=False), stop_logit
