from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from runtime.extract import (
    _source_card,
    option_signature,
    sanitize_visible_observation,
)
from runtime.features import visible_action_features, visible_state_features
from runtime.tree_predict import predict_probability


MAIN_MINIMUM_MARGIN = 0.10619058097192913
DECK_COUNTS = {
    7: 10,
    104: 2,
    112: 4,
    646: 4,
    647: 3,
    648: 3,
    860: 2,
    1079: 3,
    1080: 1,
    1086: 4,
    1097: 3,
    1122: 1,
    1137: 1,
    1152: 4,
    1182: 2,
    1219: 4,
    1227: 4,
    1231: 1,
    1259: 4,
}
DECK = tuple(
    card_id
    for card_id, count in sorted(DECK_COUNTS.items())
    for _ in range(count)
)

_ROOT = Path(__file__).resolve().parent
_MODELS: dict[str, dict[str, Any]] | None = None


def _models() -> dict[str, dict[str, Any]]:
    global _MODELS
    if _MODELS is None:
        _MODELS = {
            name: json.loads((_ROOT / "models" / f"{name}.json").read_text())
            for name in (
                "main_v1",
                "main_v2",
                "followup_v1",
                "followup_v2",
            )
        }
    return _MODELS


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _semantic_top(
    signatures: Sequence[tuple[int, ...]],
    scores: Sequence[float],
) -> tuple[tuple[int, ...] | None, float]:
    by_signature: dict[tuple[int, ...], float] = {}
    for signature, score in zip(signatures, scores, strict=True):
        by_signature[signature] = max(
            by_signature.get(signature, -math.inf),
            float(score),
        )
    ordered = sorted(by_signature.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) < 2 or ordered[0][1] == ordered[1][1]:
        return None, 0.0
    return ordered[0][0], ordered[0][1] - ordered[1][1]


def _clone_choice(
    observation: Mapping[str, Any],
    *,
    mode: str,
    allow_single_fallback: bool = False,
) -> int | None:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return None
    options = select.get("option")
    if not isinstance(options, list) or len(options) < 2:
        return None
    seat = _as_int(
        (observation.get("current") or {}).get("yourIndex"),
        0,
    )
    base = visible_state_features(observation)
    feature_rows = [
        visible_action_features(
            observation,
            option,
            base_features=base,
        )
        for option in options
    ]
    signatures = [
        option_signature(observation, option, seat)
        for option in options
    ]
    models = _models()
    first_scores = [
        predict_probability(models[f"{mode}_v1"], row)
        for row in feature_rows
    ]
    second_scores = [
        predict_probability(models[f"{mode}_v2"], row)
        for row in feature_rows
    ]
    first, first_margin = _semantic_top(signatures, first_scores)
    second, second_margin = _semantic_top(signatures, second_scores)
    chosen = first
    if first is None or first != second:
        chosen = second if allow_single_fallback else None
    elif (
        mode == "main"
        and min(first_margin, second_margin) < MAIN_MINIMUM_MARGIN
    ):
        chosen = second if allow_single_fallback else None
    if chosen is None:
        return None
    return next(
        (
            index
            for index, signature in enumerate(signatures)
            if signature == chosen
        ),
        None,
    )


_PLAY_PRIORITY = {
    1080: 980,
    1086: 930,
    1079: 910,
    1152: 890,
    1122: 870,
    1219: 850,
    1227: 830,
    1231: 810,
    1182: 790,
    1097: 760,
    1137: 740,
    1259: 700,
}
_CARD_PRIORITY = {
    648: 900,
    647: 820,
    646: 760,
    104: 700,
    860: 650,
    112: 600,
    7: 300,
}


def _fallback_score(
    observation: Mapping[str, Any],
    option: Mapping[str, Any],
    index: int,
) -> tuple[float, int]:
    select = observation.get("select") or {}
    context = _as_int(select.get("context"), -1)
    option_type = _as_int(option.get("type"), -1)
    seat = _as_int((observation.get("current") or {}).get("yourIndex"), 0)
    source = _source_card(observation, option, seat)
    card_id = _as_int(source.get("id")) if isinstance(source, Mapping) else 0
    hp = float(source.get("hp") or 0) if isinstance(source, Mapping) else 0.0
    max_hp = (
        float(source.get("maxHp") or hp)
        if isinstance(source, Mapping)
        else hp
    )
    damage = max(0.0, max_hp - hp)

    if context == 0:
        main_priority = {
            10: 1_000,
            7: _PLAY_PRIORITY.get(card_id, 680),
            9: 650,
            8: 600,
            13: 500,
            12: 120,
            14: 0,
        }
        return float(main_priority.get(option_type, 300)), -index
    if option_type == 0:
        return float(_as_int(option.get("number"), 0)), -index
    if option_type == 1:
        return 100.0, -index
    if option_type == 2:
        return 20.0, -index
    if option_type == 3:
        priority = float(_CARD_PRIORITY.get(card_id, 400))
        owner = _as_int(option.get("playerIndex"), seat)
        if context in {13, 14, 15}:
            priority = (1_000.0 if owner != seat else 0.0) - hp + damage
        elif context in {16, 17}:
            priority = (1_000.0 if owner == seat else 0.0) + damage
        elif context in {8, 9, 10}:
            priority = -priority
        elif context in {1, 4}:
            energy = (
                len(source.get("energyCards") or [])
                if isinstance(source, Mapping)
                else 0
            )
            priority += 40.0 * energy + hp
        return priority, -index
    if option_type in {13, 15}:
        return 900.0, -index
    if option_type == 16:
        return 500.0 + _as_int(option.get("specialConditionType")), -index
    return 100.0, -index


def fallback_action(observation: Mapping[str, Any]) -> list[int]:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return []
    options = select.get("option")
    if not isinstance(options, list) or not options:
        return []
    minimum = _as_int(select.get("minCount"), 0)
    maximum = _as_int(select.get("maxCount"), minimum)
    maximum = min(maximum, len(options))
    if maximum <= 0:
        return []
    if minimum == maximum:
        count = minimum
    else:
        context = _as_int(select.get("context"), -1)
        count = minimum if context in {12, 44, 45} else maximum
    ranked = sorted(
        range(len(options)),
        key=lambda index: _fallback_score(
            observation,
            options[index],
            index,
        ),
        reverse=True,
    )
    return sorted(ranked[:count])


def choose_action(raw_observation: Mapping[str, Any]) -> list[int]:
    return _choose_action(raw_observation, full_main_clone=False)


def choose_action_full_main_clone(
    raw_observation: Mapping[str, Any],
) -> list[int]:
    return _choose_action(raw_observation, full_main_clone=True)


def _choose_action(
    raw_observation: Mapping[str, Any],
    *,
    full_main_clone: bool,
) -> list[int]:
    current = raw_observation.get("current")
    seat = (
        _as_int(current.get("yourIndex"), 0)
        if isinstance(current, Mapping)
        else 0
    )
    observation = sanitize_visible_observation(raw_observation, seat)
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return []
    options = select.get("option")
    if not isinstance(options, list) or not options:
        return []
    minimum = _as_int(select.get("minCount"), 0)
    maximum = _as_int(select.get("maxCount"), 0)
    if minimum == maximum == 1 and len(options) > 1:
        mode = "main" if _as_int(select.get("context"), -1) == 0 else "followup"
        chosen = _clone_choice(
            observation,
            mode=mode,
            allow_single_fallback=(full_main_clone and mode == "main"),
        )
        if chosen is not None:
            return [chosen]
    return fallback_action(observation)
