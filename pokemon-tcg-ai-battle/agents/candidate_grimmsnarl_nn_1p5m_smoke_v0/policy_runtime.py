from __future__ import annotations

import copy
import hashlib
import math
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


INPUT_WIDTH = 512
NEURAL_TIEBREAK = 1e-6
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
_MODEL: dict[str, np.ndarray] | None = None
_FORBIDDEN_KEYS = {
    "deckorder",
    "hidden",
    "opponentdeck",
    "opponenthand",
    "opponentprizecards",
    "prizecards",
    "searchbegininput",
    "visualize",
}


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _strip_forbidden(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden(child)
            for key, child in value.items()
            if not any(
                _normalized_key(key) == forbidden
                or _normalized_key(key).startswith(forbidden)
                for forbidden in _FORBIDDEN_KEYS
            )
        }
    if isinstance(value, list):
        return [_strip_forbidden(child) for child in value]
    return copy.deepcopy(value)


def sanitize_visible_observation(
    observation: Mapping[str, Any],
    acting_seat: int,
) -> dict[str, Any]:
    sanitized = _strip_forbidden(dict(observation))
    if not isinstance(sanitized, dict):
        raise ValueError("observation must be an object")
    current = sanitized.get("current")
    players = current.get("players") if isinstance(current, dict) else None
    if isinstance(players, list):
        for seat, player in enumerate(players[:2]):
            if not isinstance(player, dict):
                continue
            player.pop("deck", None)
            prize = player.get("prize")
            if isinstance(prize, list):
                player["prize"] = [None] * len(prize)
            if seat != acting_seat:
                player["hand"] = None
    return sanitized


def _model() -> dict[str, np.ndarray]:
    global _MODEL
    if _MODEL is None:
        with np.load(_ROOT / "model.npz", allow_pickle=False) as checkpoint:
            _MODEL = {
                name: np.asarray(checkpoint[name], dtype=np.float32)
                for name in ("w1", "b1", "w2", "b2", "wout", "bout")
            }
    return _MODEL


def _hash_index(name: str) -> tuple[int, float]:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "little")
    return number % INPUT_WIDTH, (-1.0 if number >> 63 else 1.0)


def _add(vector: np.ndarray, name: str, value: object, scale: float = 1.0) -> None:
    number = _as_float(value)
    if number == 0.0:
        return
    index, sign = _hash_index(name)
    vector[index] += np.float32(sign * max(-8.0, min(8.0, number * scale)))


def _objects(value: object) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _add_cards(vector: np.ndarray, zone: str, cards: object) -> None:
    visible = _objects(cards)
    counts = Counter(_as_int(card.get("id")) for card in visible)
    for card_id, count in counts.items():
        if card_id > 0:
            _add(vector, f"{zone}:card:{card_id}", count)
    for position, card in enumerate(visible[:60]):
        card_id = _as_int(card.get("id"))
        prefix = f"{zone}:slot:{min(position, 7)}:card:{card_id}"
        _add(vector, prefix, 1.0)
        _add(vector, prefix + ":hp", card.get("hp"), 1.0 / 340.0)
        _add(vector, prefix + ":damage", _as_float(card.get("maxHp")) - _as_float(card.get("hp")), 1.0 / 340.0)
        _add(vector, prefix + ":energy", len(_objects(card.get("energyCards"))), 0.25)
        _add(vector, prefix + ":tools", len(_objects(card.get("tools"))), 0.5)
        _add(vector, prefix + ":evolution", len(_objects(card.get("preEvolution"))), 0.5)


def _base_vector(observation: Mapping[str, Any]) -> np.ndarray:
    vector = np.zeros(INPUT_WIDTH, dtype=np.float32)
    current = observation.get("current")
    current = current if isinstance(current, Mapping) else {}
    seat = _as_int(current.get("yourIndex"), 0)
    players = current.get("players")
    players = players if isinstance(players, list) else []
    me = players[seat] if seat < len(players) and isinstance(players[seat], Mapping) else {}
    rival_seat = 1 - seat
    rival = players[rival_seat] if rival_seat < len(players) and isinstance(players[rival_seat], Mapping) else {}

    for name, value, scale in (
        ("turn", current.get("turn"), 1.0 / 20.0),
        ("turn_actions", current.get("turnActionCount"), 1.0 / 20.0),
        ("me_deck", me.get("deckCount"), 1.0 / 60.0),
        ("me_hand", me.get("handCount"), 1.0 / 20.0),
        ("me_prize", len(me.get("prize") or []), 1.0 / 6.0),
        ("rival_deck", rival.get("deckCount"), 1.0 / 60.0),
        ("rival_hand", rival.get("handCount"), 1.0 / 20.0),
        ("rival_prize", len(rival.get("prize") or []), 1.0 / 6.0),
        ("seat", seat + 1, 1.0),
    ):
        _add(vector, name, value, scale)
    for flag in ("energyAttached", "retreated", "stadiumPlayed", "supporterPlayed"):
        _add(vector, "flag:" + flag, float(bool(current.get(flag))))
    for prefix, player in (("me", me), ("rival", rival)):
        _add_cards(vector, prefix + ":active", player.get("active"))
        _add_cards(vector, prefix + ":bench", player.get("bench"))
        _add_cards(vector, prefix + ":discard", player.get("discard"))
    _add_cards(vector, "me:hand", me.get("hand"))
    _add_cards(vector, "stadium", current.get("stadium"))

    logs = _objects(observation.get("logs"))[-32:]
    for offset, log in enumerate(reversed(logs)):
        _add(
            vector,
            f"log:{min(offset, 7)}:{_as_int(log.get('type'), -1)}:{_as_int(log.get('cardId'))}:{_as_int(log.get('playerIndex'), -1)}",
            1.0,
        )
    select = observation.get("select")
    if isinstance(select, Mapping):
        _add(vector, f"context:{_as_int(select.get('context'), -1)}", 1.0)
        _add(vector, f"select_type:{_as_int(select.get('type'), -1)}", 1.0)
        _add(vector, "min_count", select.get("minCount"), 0.25)
        _add(vector, "max_count", select.get("maxCount"), 0.25)
        options = _objects(select.get("option"))
        _add(vector, "option_count", len(options), 1.0 / 60.0)
        for option_type, count in Counter(_as_int(option.get("type"), -1) for option in options).items():
            _add(vector, f"option_type:{option_type}", count, 0.125)
    return vector


def _zone(
    observation: Mapping[str, Any],
    area: object,
    player_index: int,
) -> list[object]:
    current = observation.get("current")
    players = current.get("players") if isinstance(current, Mapping) else None
    player = players[player_index] if isinstance(players, list) and 0 <= player_index < len(players) else None
    area_number = _as_int(area, -1)
    if area_number == 1:
        select = observation.get("select")
        deck = select.get("deck") if isinstance(select, Mapping) else None
        return list(deck or [])
    if area_number == 7:
        return list(current.get("stadium") or []) if isinstance(current, Mapping) else []
    if area_number == 12:
        return list(current.get("looking") or []) if isinstance(current, Mapping) else []
    names = {2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}
    return list(player.get(names.get(area_number, ""), []) or []) if isinstance(player, Mapping) else []


def _source_card(
    observation: Mapping[str, Any],
    option: Mapping[str, Any],
    seat: int,
) -> Mapping[str, Any] | None:
    option_type = _as_int(option.get("type"), -1)
    owner = _as_int(option.get("playerIndex"), seat)
    area = 2 if option_type == 7 else option.get("area")
    cards = _zone(observation, area, owner)
    index = _as_int(option.get("index"), -1)
    card = cards[index] if 0 <= index < len(cards) else None
    return card if isinstance(card, Mapping) else None


def _option_vectors(
    observation: Mapping[str, Any],
    options: list[Mapping[str, Any]],
) -> np.ndarray:
    base = _base_vector(observation)
    vectors = np.repeat(base[None, :], len(options), axis=0)
    current = observation.get("current")
    seat = _as_int(current.get("yourIndex"), 0) if isinstance(current, Mapping) else 0
    for index, option in enumerate(options):
        prefix = "action"
        for field in (
            "type", "area", "index", "playerIndex", "inPlayArea", "inPlayIndex",
            "attackId", "number", "specialConditionType", "energyIndex", "toolIndex",
        ):
            value = _as_int(option.get(field), -1)
            _add(vectors[index], f"{prefix}:{field}:{value}", 1.0)
        source = _source_card(observation, option, seat)
        if source is not None:
            card_id = _as_int(source.get("id"))
            _add(vectors[index], f"action:source:{card_id}", 1.0)
            _add(vectors[index], f"action:source:{card_id}:hp", source.get("hp"), 1.0 / 340.0)
    return vectors


def neural_logits(
    observation: Mapping[str, Any],
    options: list[Mapping[str, Any]],
) -> np.ndarray:
    weights = _model()
    features = _option_vectors(observation, options)
    hidden = np.tanh(features @ weights["w1"] + weights["b1"])
    hidden = np.tanh(hidden @ weights["w2"] + weights["b2"])
    logits = hidden @ weights["wout"] + weights["bout"][0]
    if logits.shape != (len(options),) or not np.isfinite(logits).all():
        raise ValueError("non-finite neural policy output")
    return logits


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
_CARD_PRIORITY = {648: 900, 647: 820, 646: 760, 104: 700, 860: 650, 112: 600, 7: 300}


def _fallback_score(
    observation: Mapping[str, Any],
    option: Mapping[str, Any],
    index: int,
) -> tuple[float, int]:
    select = observation.get("select") or {}
    context = _as_int(select.get("context"), -1)
    option_type = _as_int(option.get("type"), -1)
    current = observation.get("current") or {}
    seat = _as_int(current.get("yourIndex"), 0) if isinstance(current, Mapping) else 0
    source = _source_card(observation, option, seat)
    card_id = _as_int(source.get("id")) if source is not None else 0
    hp = _as_float(source.get("hp")) if source is not None else 0.0
    max_hp = _as_float(source.get("maxHp"), hp) if source is not None else hp
    damage = max(0.0, max_hp - hp)
    if context == 0:
        priority = {10: 1_000, 7: _PLAY_PRIORITY.get(card_id, 680), 9: 650, 8: 600, 13: 500, 12: 120, 14: 0}
        return float(priority.get(option_type, 300)), -index
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
            energy = len(_objects(source.get("energyCards"))) if source is not None else 0
            priority += 40.0 * energy + hp
        return priority, -index
    if option_type in {13, 15}:
        return 900.0, -index
    if option_type == 16:
        return 500.0 + _as_int(option.get("specialConditionType")), -index
    return 100.0, -index


def _selection_count(select: Mapping[str, Any], option_count: int) -> int:
    minimum = max(0, _as_int(select.get("minCount"), 0))
    maximum = min(option_count, _as_int(select.get("maxCount"), minimum))
    if maximum <= 0:
        return 0
    if minimum == maximum:
        return minimum
    context = _as_int(select.get("context"), -1)
    return minimum if context in {12, 44, 45} else maximum


def fallback_action(observation: Mapping[str, Any]) -> list[int]:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return []
    options = _objects(select.get("option"))
    count = _selection_count(select, len(options))
    ranked = sorted(
        range(len(options)),
        key=lambda index: _fallback_score(observation, options[index], index),
        reverse=True,
    )
    return sorted(ranked[:count])


def choose_action(raw_observation: Mapping[str, Any]) -> list[int]:
    current = raw_observation.get("current")
    seat = _as_int(current.get("yourIndex"), 0) if isinstance(current, Mapping) else 0
    observation = sanitize_visible_observation(raw_observation, seat)
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return []
    options = _objects(select.get("option"))
    count = _selection_count(select, len(options))
    if count == 0:
        return []
    logits = neural_logits(observation, options)
    ranked = sorted(
        range(len(options)),
        key=lambda index: (
            _fallback_score(observation, options[index], index)[0],
            float(logits[index]) * NEURAL_TIEBREAK,
            -index,
        ),
        reverse=True,
    )
    return sorted(ranked[:count])

