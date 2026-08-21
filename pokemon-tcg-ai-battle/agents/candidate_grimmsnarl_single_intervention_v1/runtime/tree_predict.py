from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def predict_probability(
    model: Mapping[str, Any],
    features: Mapping[str, float],
) -> float:
    names: Sequence[str] = model["feature_names"]
    score = float(model["baseline"])
    for tree in model["trees"]:
        node_index = 0
        while len(tree[node_index]) > 1:
            feature_index, threshold, left, right, missing_left = tree[node_index]
            value = float(features.get(names[int(feature_index)], 0.0))
            if not math.isfinite(value):
                node_index = int(left if missing_left else right)
            else:
                node_index = int(left if value <= threshold else right)
        score += float(tree[node_index][0])
    calibration = model.get("calibration")
    if isinstance(calibration, Mapping):
        score = (
            float(calibration["slope"]) * score
            + float(calibration["intercept"])
        )
    return _sigmoid(score)
