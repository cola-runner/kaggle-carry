from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from si_grimmsnarl.residual_encoder import encode_options


MEMBER = "grimmsnarl"
_ROOT = Path(__file__).resolve().parent
_MODELS: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None = None
_MARGINS: tuple[float, float] | None = None


def _load() -> tuple[
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    tuple[float, float],
]:
    global _MODELS, _MARGINS
    if _MODELS is None:
        loaded = []
        for index in range(2):
            with np.load(
                _ROOT / "models" / f"single_intervention_{index}.npz",
                allow_pickle=False,
            ) as checkpoint:
                loaded.append(
                    {
                        name: np.asarray(checkpoint[name], dtype=np.float32)
                        for name in ("w", "b", "w_option", "b_option")
                    }
                )
        _MODELS = (loaded[0], loaded[1])
        manifest = json.loads(
            (_ROOT / "models/single_intervention_manifest.json").read_text()
        )
        margins = manifest["override_margins"][MEMBER]
        _MARGINS = (float(margins[0]), float(margins[1]))
    return _MODELS, _MARGINS


def option_scores(observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features = encode_options(observation)
    models, _ = _load()
    rows = []
    for weights in models:
        hidden = np.tanh(features @ weights["w"] + weights["b"])
        scores = hidden @ weights["w_option"] + float(weights["b_option"][0])
        if not np.isfinite(scores).all():
            raise ValueError("non-finite residual scores")
        rows.append(scores.astype(np.float32, copy=False))
    return rows[0], rows[1]


def _eligible(observation: Mapping[str, Any], incumbent: Sequence[int]) -> bool:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return False
    options = select.get("option")
    return (
        isinstance(options, list)
        and len(options) > 1
        and int(select.get("type", -1)) == 0
        and int(select.get("context", -1)) == 0
        and int(select.get("minCount", -1)) == 1
        and int(select.get("maxCount", -1)) == 1
        and len(incumbent) == 1
        and isinstance(incumbent[0], int)
        and 0 <= incumbent[0] < len(options)
    )


def choose_action(
    observation: Mapping[str, Any],
    incumbent: Sequence[int],
) -> list[int]:
    if not _eligible(observation, incumbent):
        return list(incumbent)
    rows = option_scores(observation)
    _, margins = _load()
    incumbent_index = incumbent[0]
    candidates = [
        index
        for index in range(len(rows[0]))
        if index != incumbent_index
        and all(
            math.isfinite(float(row[index] - row[incumbent_index]))
            and float(row[index] - row[incumbent_index]) > margins[model_index]
            for model_index, row in enumerate(rows)
        )
    ]
    if not candidates:
        return list(incumbent)
    chosen = max(
        candidates,
        key=lambda index: float((rows[0][index] + rows[1][index]) / 2.0),
    )
    return [chosen]
