from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import canonical_json_bytes, sha256_bytes
from .schema import ReplayRecord


OPTION_SIGNATURE_VERSION = 2
_FORBIDDEN_NORMALIZED = {
    "deckorder",
    "hidden",
    "opponentdeck",
    "opponenthand",
    "opponentprizecards",
    "prizecards",
    "searchbegininput",
    "visualize",
}


@dataclass(slots=True)
class ExtractionRows:
    episodes: list[dict[str, object]]
    decisions: list[dict[str, object]]
    options: list[dict[str, object]]
    hidden: list[dict[str, object]]


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _is_forbidden_key(value: object) -> bool:
    normalized = _normalized_key(value)
    return any(
        normalized == forbidden or normalized.startswith(forbidden)
        for forbidden in _FORBIDDEN_NORMALIZED
    )


def assert_visible_only(value: object, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_key(key):
                raise ValueError(f"forbidden visible key at {path}.{key}")
            assert_visible_only(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_visible_only(child, f"{path}[{index}]")


def _strip_forbidden(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden(child)
            for key, child in value.items()
            if not _is_forbidden_key(key)
        }
    if isinstance(value, list):
        return [_strip_forbidden(child) for child in value]
    return copy.deepcopy(value)


def sanitize_visible_observation(
    observation: Mapping[str, object],
    acting_seat: int,
) -> dict[str, object]:
    if acting_seat not in (0, 1):
        raise ValueError("acting_seat must be 0 or 1")
    sanitized = _strip_forbidden(dict(observation))
    if not isinstance(sanitized, dict):
        raise ValueError("observation must be an object")
    current = sanitized.get("current")
    if isinstance(current, dict):
        players = current.get("players")
        if isinstance(players, list) and len(players) >= 2:
            for seat, player in enumerate(players[:2]):
                if not isinstance(player, dict):
                    continue
                player.pop("deck", None)
                prize = player.get("prize")
                if isinstance(prize, list):
                    player["prize"] = [None] * len(prize)
                if seat != acting_seat:
                    player["hand"] = None
    assert_visible_only(sanitized)
    return sanitized


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: object) -> int:
    return _as_int(card.get("id")) if isinstance(card, dict) else 0


def _card_serial(card: object) -> int:
    return _as_int(card.get("serial")) if isinstance(card, dict) else 0


def _zone(
    observation: Mapping[str, object],
    area: object,
    player_index: int,
) -> list[object]:
    current = observation.get("current")
    players = current.get("players") if isinstance(current, dict) else None
    if not isinstance(players, list) or not 0 <= player_index < len(players):
        return []
    player = players[player_index]
    if not isinstance(player, dict):
        return []
    area_number = _as_int(area, -1)
    names = {
        2: "hand",
        3: "discard",
        4: "active",
        5: "bench",
        6: "prize",
    }
    if area_number == 1:
        select = observation.get("select")
        deck = select.get("deck") if isinstance(select, dict) else None
        return list(deck or [])
    if area_number == 7:
        stadium = current.get("stadium") if isinstance(current, dict) else None
        return list(stadium or [])
    if area_number == 12:
        looking = current.get("looking") if isinstance(current, dict) else None
        return list(looking or [])
    return list(player.get(names.get(area_number, ""), []) or [])


def _zone_card(
    observation: Mapping[str, object],
    area: object,
    index: object,
    player_index: int,
) -> object | None:
    cards = _zone(observation, area, player_index)
    position = _as_int(index, -1)
    return cards[position] if 0 <= position < len(cards) else None


def _source_card(
    observation: Mapping[str, object],
    option: Mapping[str, object],
    acting_seat: int,
) -> object | None:
    option_type = _as_int(option.get("type"), -1)
    owner = _as_int(option.get("playerIndex"), acting_seat)
    area = 2 if option_type == 7 else option.get("area")
    source = _zone_card(observation, area, option.get("index"), owner)
    if option_type in (4, 5, 6) and isinstance(source, dict):
        key = "tools" if option_type == 4 else "energyCards"
        index_key = "toolIndex" if option_type == 4 else "energyIndex"
        attached = list(source.get(key) or [])
        position = _as_int(option.get(index_key), -1)
        if 0 <= position < len(attached):
            return attached[position]
    return source


def option_signature(
    observation: Mapping[str, object],
    option: Mapping[str, object],
    acting_seat: int,
) -> tuple[int, ...]:
    source = _source_card(observation, option, acting_seat)
    target_owner = _as_int(option.get("targetPlayerIndex"), acting_seat)
    target = _zone_card(
        observation,
        option.get("inPlayArea"),
        option.get("inPlayIndex"),
        target_owner,
    )
    return (
        _as_int(option.get("type"), -1),
        _as_int(option.get("cardId")) or _card_id(source),
        _as_int(option.get("serial")) or _card_serial(source),
        _card_id(target),
        _card_serial(target),
        _as_int(option.get("attackId"), -1),
        _as_int(option.get("playerIndex"), acting_seat),
        _as_int(option.get("area"), -1),
        _as_int(option.get("index"), -1),
        _as_int(option.get("inPlayArea"), -1),
        _as_int(option.get("inPlayIndex"), -1),
        _as_int(option.get("number"), -1),
        _as_int(option.get("specialConditionType"), -1),
        _as_int(option.get("energyIndex"), -1),
        _as_int(option.get("toolIndex"), -1),
    )


def selected_signature(
    observation: Mapping[str, object],
    action: Sequence[int],
    acting_seat: int,
) -> tuple[tuple[int, ...], ...]:
    select = observation.get("select")
    options = select.get("option") if isinstance(select, dict) else None
    if not isinstance(options, list):
        raise ValueError("observation has no legal options")
    if len(set(action)) != len(action):
        raise ValueError("action contains duplicate indices")
    signatures = []
    for index in action:
        if not isinstance(index, int) or not 0 <= index < len(options):
            raise ValueError(f"action index out of range: {index}")
        signatures.append(option_signature(observation, options[index], acting_seat))
    return tuple(sorted(signatures))


def extract_episode(
    episode: Mapping[str, object],
    inventory: ReplayRecord,
) -> ExtractionRows:
    steps = episode.get("steps")
    if not isinstance(steps, list):
        raise ValueError("episode has no steps")
    rewards = list(episode.get("rewards") or [])
    rewards.extend([None] * (2 - len(rewards)))
    winner = -1
    if all(isinstance(reward, (int, float)) for reward in rewards[:2]):
        if rewards[0] > rewards[1]:
            winner = 0
        elif rewards[1] > rewards[0]:
            winner = 1

    episode_row: dict[str, object] = {
        "snapshot_id": inventory.snapshot_id,
        "episode_id": inventory.episode_id,
        "submission_id": inventory.submission_id,
        "team_id": inventory.team_id,
        "create_time_utc": inventory.create_time_utc.isoformat().replace(
            "+00:00", "Z"
        ),
        "target_seat": inventory.target_seat,
        "split": inventory.split.value,
        "won": winner == inventory.target_seat if winner >= 0 else None,
        "replay_sha256": inventory.replay_sha256,
    }
    decisions: list[dict[str, object]] = []
    option_rows: list[dict[str, object]] = []
    hidden_rows: list[dict[str, object]] = []
    pending_observation: dict[str, object] | None = None
    pending_step = -1

    for action_step, step in enumerate(steps):
        if not isinstance(step, list) or inventory.target_seat >= len(step):
            continue
        record = step[inventory.target_seat]
        if not isinstance(record, dict):
            continue
        action = record.get("action")
        if (
            pending_observation is not None
            and isinstance(action, list)
            and len(action) != 60
        ):
            select = pending_observation.get("select")
            options = select.get("option") if isinstance(select, dict) else None
            if isinstance(options, list) and options:
                minimum = _as_int(select.get("minCount"), 0)
                maximum = _as_int(select.get("maxCount"), 0)
                if not minimum <= len(action) <= maximum:
                    raise ValueError(
                        f"recorded action length {len(action)} is outside "
                        f"[{minimum}, {maximum}]"
                    )
                semantic_selected = selected_signature(
                    pending_observation,
                    action,
                    inventory.target_seat,
                )
                visible = sanitize_visible_observation(
                    pending_observation,
                    inventory.target_seat,
                )
                visible_sha256 = sha256_bytes(canonical_json_bytes(visible))
                decision_id = (
                    f"{inventory.episode_id}:{inventory.target_seat}:{pending_step}"
                )
                signatures = [
                    option_signature(
                        pending_observation,
                        option,
                        inventory.target_seat,
                    )
                    for option in options
                ]
                unique_option_count = len(set(signatures))
                context = _as_int(select.get("context"), -1)
                decision_row: dict[str, object] = {
                    "snapshot_id": inventory.snapshot_id,
                    "decision_id": decision_id,
                    "episode_id": inventory.episode_id,
                    "submission_id": inventory.submission_id,
                    "team_id": inventory.team_id,
                    "create_time_utc": episode_row["create_time_utc"],
                    "target_seat": inventory.target_seat,
                    "split": inventory.split.value,
                    "root_step": pending_step,
                    "action_step": action_step,
                    "context": context,
                    "select_type": _as_int(select.get("type"), -1),
                    "min_count": minimum,
                    "max_count": maximum,
                    "option_count": len(options),
                    "unique_option_count": unique_option_count,
                    "forced": unique_option_count <= 1,
                    "single_choice_main": (
                        context == 0
                        and minimum == 1
                        and maximum == 1
                        and unique_option_count > 1
                    ),
                    "selected_signature": [
                        list(signature) for signature in semantic_selected
                    ],
                    "won": episode_row["won"],
                    "visible_sha256": visible_sha256,
                    "observation": visible,
                }
                decisions.append(decision_row)
                for option_index, (option, signature) in enumerate(
                    zip(options, signatures, strict=True)
                ):
                    option_rows.append(
                        {
                            "snapshot_id": inventory.snapshot_id,
                            "decision_id": decision_id,
                            "episode_id": inventory.episode_id,
                            "submission_id": inventory.submission_id,
                            "team_id": inventory.team_id,
                            "create_time_utc": episode_row["create_time_utc"],
                            "target_seat": inventory.target_seat,
                            "split": inventory.split.value,
                            "context": context,
                            "option_index": option_index,
                            "selected": option_index in action,
                            "signature_version": OPTION_SIGNATURE_VERSION,
                            "option_signature": list(signature),
                            "option": copy.deepcopy(option),
                        }
                    )
                hidden_rows.append(
                    {
                        "snapshot_id": inventory.snapshot_id,
                        "decision_id": decision_id,
                        "episode_id": inventory.episode_id,
                        "target_seat": inventory.target_seat,
                        "root_step": pending_step,
                        "replay_relpath": inventory.replay_relpath,
                        "replay_sha256": inventory.replay_sha256,
                        "visible_sha256": visible_sha256,
                        "search_begin_input": pending_observation.get(
                            "search_begin_input"
                        ),
                    }
                )
            pending_observation = None
            pending_step = -1

        observation = record.get("observation")
        if not isinstance(observation, dict):
            continue
        current = observation.get("current")
        select = observation.get("select")
        if (
            record.get("status") == "ACTIVE"
            and isinstance(current, dict)
            and _as_int(current.get("yourIndex"), -1) == inventory.target_seat
            and _as_int(current.get("result"), -1) < 0
            and isinstance(select, dict)
            and isinstance(select.get("option"), list)
            and select.get("option")
        ):
            pending_observation = observation
            pending_step = action_step

    return ExtractionRows(
        episodes=[episode_row],
        decisions=decisions,
        options=option_rows,
        hidden=hidden_rows,
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("xb") as file:
        for row in rows:
            file.write(canonical_json_bytes(dict(row)) + b"\n")


def write_extracted_datasets(
    *,
    snapshot_dir: Path,
    episodes: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
    options: Sequence[Mapping[str, object]],
    hidden: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    for collection in (episodes, decisions, options):
        for row in collection:
            assert_visible_only(row)
    public_dir = snapshot_dir / "public"
    hidden_dir = snapshot_dir / "offline_hidden"
    public_dir.mkdir(parents=True, exist_ok=False)
    hidden_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "episodes": public_dir / "episodes.jsonl",
        "decisions": public_dir / "decisions.jsonl",
        "options": public_dir / "options.jsonl",
        "hidden": hidden_dir / "restoration.jsonl",
    }
    _write_jsonl(paths["episodes"], episodes)
    _write_jsonl(paths["decisions"], decisions)
    _write_jsonl(paths["options"], options)
    _write_jsonl(paths["hidden"], hidden)
    return paths
