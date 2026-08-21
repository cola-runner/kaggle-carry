from __future__ import annotations

import math
import re
from hashlib import blake2b
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .extract import (
    _source_card,
    _zone_card,
    assert_visible_only,
    option_signature,
)


_FORBIDDEN_FEATURE_PARTS = (
    "deckorder",
    "hidden",
    "opponentdeck",
    "opponenthand",
    "opponentprize",
    "prizecards",
    "searchbegininput",
    "visualize",
)
FEATURE_SCHEMA_VERSION = 1
HASH_BUCKETS = 384


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def audit_feature_names(names: Iterable[str]) -> tuple[str, ...]:
    forbidden = tuple(
        sorted(
            name
            for name in names
            if any(part in _normalized(name) for part in _FORBIDDEN_FEATURE_PARTS)
        )
    )
    if forbidden:
        raise ValueError(f"forbidden feature names: {', '.join(forbidden)}")
    return ()


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _objects(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _add_card_counts(
    features: dict[str, float],
    prefix: str,
    cards: object,
) -> None:
    counts = Counter(
        _integer(card.get("id"))
        for card in _objects(cards)
        if _integer(card.get("id")) > 0
    )
    for card_id, count in counts.items():
        features[f"count:{prefix}:{card_id}"] = float(min(count, 8))


def _add_board_features(
    features: dict[str, float],
    prefix: str,
    player: Mapping[str, Any],
) -> None:
    board = _objects(player.get("active")) + _objects(player.get("bench"))
    features[f"n:{prefix}_board_count"] = float(len(board))
    features[f"n:{prefix}_bench_count"] = float(len(_objects(player.get("bench"))))
    features[f"n:{prefix}_board_hp"] = sum(
        _number(card.get("hp")) for card in board
    )
    features[f"n:{prefix}_board_damage"] = sum(
        max(0.0, _number(card.get("maxHp")) - _number(card.get("hp")))
        for card in board
    )
    features[f"n:{prefix}_board_energy"] = float(
        sum(len(_objects(card.get("energyCards"))) for card in board)
    )
    features[f"n:{prefix}_board_tools"] = float(
        sum(len(_objects(card.get("tools"))) for card in board)
    )
    features[f"n:{prefix}_evolution_depth"] = float(
        sum(len(_objects(card.get("preEvolution"))) for card in board)
    )
    _add_card_counts(features, f"{prefix}_board", board)
    _add_card_counts(features, f"{prefix}_discard", player.get("discard"))

    active = _objects(player.get("active"))
    if active:
        card = active[0]
        card_id = _integer(card.get("id"))
        if card_id > 0:
            features[f"cat:{prefix}_active={card_id}"] = 1.0
        features[f"n:{prefix}_active_hp"] = _number(card.get("hp"))
        features[f"n:{prefix}_active_damage"] = max(
            0.0,
            _number(card.get("maxHp")) - _number(card.get("hp")),
        )
        features[f"n:{prefix}_active_energy"] = float(
            len(_objects(card.get("energyCards")))
        )


def _compact(features: Mapping[str, float]) -> dict[str, float]:
    compact: dict[str, float] = {}
    for name, value in features.items():
        if name.startswith("n:"):
            compact[name] = float(value)
            continue
        digest = blake2b(name.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        bucket = number % HASH_BUCKETS
        sign = -1.0 if number & (1 << 63) else 1.0
        key = f"h:{bucket:03d}"
        compact[key] = compact.get(key, 0.0) + sign * float(value)
    return compact


def visible_state_features(
    observation: Mapping[str, Any],
) -> dict[str, float]:
    """Build value-model features from an already sanitized observation."""
    assert_visible_only(observation)
    current = observation.get("current")
    current = current if isinstance(current, Mapping) else {}
    seat = _integer(current.get("yourIndex"), 0)
    if seat not in (0, 1):
        seat = 0
    players = current.get("players")
    players = players if isinstance(players, list) else []
    me = players[seat] if seat < len(players) and isinstance(players[seat], Mapping) else {}
    rival_seat = 1 - seat
    rival = (
        players[rival_seat]
        if rival_seat < len(players) and isinstance(players[rival_seat], Mapping)
        else {}
    )

    features: dict[str, float] = {
        "n:step": _number(observation.get("step")),
        "n:turn": _number(current.get("turn")),
        "n:turn_action_count": _number(current.get("turnActionCount")),
        "n:me_deck_count": _number(me.get("deckCount")),
        "n:me_hand_count": _number(me.get("handCount")),
        "n:me_prize_count": float(len(me.get("prize") or [])),
        "n:rival_deck_count": _number(rival.get("deckCount")),
        "n:rival_hand_count": _number(rival.get("handCount")),
        "n:rival_prize_count": float(len(rival.get("prize") or [])),
        f"cat:seat={seat}": 1.0,
    }
    for prefix, player in (("me", me), ("rival", rival)):
        for flag in (
            "asleep",
            "burned",
            "confused",
            "paralyzed",
            "poisoned",
        ):
            features[f"n:{prefix}_{flag}"] = float(bool(player.get(flag)))
    for flag in (
        "energyAttached",
        "retreated",
        "stadiumPlayed",
        "supporterPlayed",
    ):
        features[f"n:{flag}"] = float(bool(current.get(flag)))
    _add_board_features(features, "me", me)
    _add_board_features(features, "rival", rival)
    _add_card_counts(features, "me_hand", me.get("hand"))

    stadium = _objects(current.get("stadium"))
    _add_card_counts(features, "stadium", stadium)
    logs = _objects(observation.get("logs"))
    log_types = Counter(_integer(log.get("type"), -1) for log in logs[-64:])
    log_cards = Counter(
        _integer(log.get("cardId"))
        for log in logs[-64:]
        if _integer(log.get("cardId")) > 0
    )
    for log_type, count in log_types.items():
        features[f"count:recent_log_type:{log_type}"] = float(min(count, 8))
    for card_id, count in log_cards.items():
        features[f"count:recent_log_card:{card_id}"] = float(min(count, 8))

    select = observation.get("select")
    if isinstance(select, Mapping):
        context = _integer(select.get("context"), -1)
        select_type = _integer(select.get("type"), -1)
        features[f"cat:context={context}"] = 1.0
        features[f"cat:select_type={select_type}"] = 1.0
        features["n:min_count"] = _number(select.get("minCount"))
        features["n:max_count"] = _number(select.get("maxCount"))
        options = _objects(select.get("option"))
        features["n:option_count"] = float(len(options))
        option_types = Counter(_integer(option.get("type"), -1) for option in options)
        for option_type, count in option_types.items():
            features[f"count:option_type:{option_type}"] = float(count)

    audit_feature_names(features)
    compact = _compact(features)
    audit_feature_names(compact)
    return compact


def visible_action_features(
    state: Mapping[str, Any],
    option: Mapping[str, Any],
    *,
    base_features: Mapping[str, float] | None = None,
) -> dict[str, float]:
    features = dict(
        visible_state_features(state)
        if base_features is None
        else base_features
    )
    current = state.get("current")
    current = current if isinstance(current, Mapping) else {}
    seat = _integer(current.get("yourIndex"), 0)
    signature = option_signature(state, option, seat)
    names = (
        "type",
        "source_card_id",
        "source_serial",
        "target_card_id",
        "target_serial",
        "attack_id",
        "player",
        "area",
        "index",
        "target_area",
        "target_index",
        "number",
        "special_condition",
        "energy_index",
        "tool_index",
    )
    for name, value in zip(names, signature, strict=True):
        features[f"n:action_{name}"] = float(value)
    source = _source_card(state, option, seat)
    target_owner = _integer(option.get("targetPlayerIndex"), seat)
    target = _zone_card(
        state,
        option.get("inPlayArea"),
        option.get("inPlayIndex"),
        target_owner,
    )

    def add_card_metrics(prefix: str, card: object) -> None:
        if not isinstance(card, Mapping):
            return
        hp = _number(card.get("hp"))
        maximum_hp = _number(card.get("maxHp"))
        features[f"n:action_{prefix}_hp"] = hp
        features[f"n:action_{prefix}_damage"] = max(0.0, maximum_hp - hp)
        features[f"n:action_{prefix}_energy"] = float(
            len(_objects(card.get("energyCards")))
        )
        features[f"n:action_{prefix}_tools"] = float(
            len(_objects(card.get("tools")))
        )
        features[f"n:action_{prefix}_evolution_depth"] = float(
            len(_objects(card.get("preEvolution")))
        )

    add_card_metrics("source", source)
    add_card_metrics("target", target)
    action_categories = {
        f"cat:action_type={signature[0]}": 1.0,
        f"cat:action_source_card={signature[1]}": 1.0,
        f"cat:action_target_card={signature[3]}": 1.0,
        f"cat:action_attack={signature[5]}": 1.0,
        (
            "cat:action_cross="
            f"{signature[0]}:{signature[1]}:{signature[3]}:{signature[5]}"
        ): 1.0,
    }
    for name, value in _compact(action_categories).items():
        features[name] = features.get(name, 0.0) + value
    audit_feature_names(features)
    return features
