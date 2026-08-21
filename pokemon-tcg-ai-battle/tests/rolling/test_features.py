from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rolling_policy.extract import sanitize_visible_observation
from rolling_policy.features import (
    audit_feature_names,
    visible_action_features,
    visible_state_features,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _raw() -> dict:
    return json.loads((FIXTURES / "visible_observation.json").read_text())


def test_state_features_ignore_opponent_hidden_cards_and_input_order() -> None:
    first = _raw()
    second = _raw()
    second["current"]["players"][1]["hand"] = [{"id": 1}, {"id": 2}, {"id": 3}]
    second["current"]["players"][1]["prize"] = [{"id": 4}]
    second["search_begin_input"] = "DIFFERENT"
    first_visible = sanitize_visible_observation(first, 0)
    second_visible = sanitize_visible_observation(second, 0)
    assert visible_state_features(first_visible) == visible_state_features(
        dict(reversed(list(second_visible.items())))
    )


def test_state_features_are_finite_and_use_public_hand_count() -> None:
    features = visible_state_features(sanitize_visible_observation(_raw(), 0))
    assert features["n:rival_hand_count"] == 3.0
    assert all(math.isfinite(value) for value in features.values())


def test_feature_audit_rejects_hidden_name_variants() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        audit_feature_names(
            {
                "safe",
                "OpponentHand",
                "opponent-hand.card",
                "visualize.card",
                "deckOrder",
                "prize_cards",
                "search_begin_input",
            }
        )
    assert audit_feature_names({"n:rival_hand_count", "count:rival_discard:7"}) == ()


def test_action_features_distinguish_visible_target_serials() -> None:
    observation = sanitize_visible_observation(_raw(), 0)
    first = {"type": 3, "playerIndex": 1, "area": 5, "index": 0}
    second = {"type": 3, "playerIndex": 1, "area": 5, "index": 1}
    assert visible_action_features(observation, first) != visible_action_features(
        observation,
        second,
    )
    first_features = visible_action_features(observation, first)
    second_features = visible_action_features(observation, second)
    assert first_features["n:action_source_damage"] == 0.0
    assert second_features["n:action_source_damage"] == 30.0


def test_state_features_distinguish_visible_effect_and_context_card() -> None:
    first = _raw()
    first["select"]["context"] = 7
    first["select"]["effect"] = {"id": 1152, "serial": 30}
    first["select"]["contextCard"] = {"id": 1259, "serial": 31}
    different_effect = _raw()
    different_effect["select"]["context"] = 7
    different_effect["select"]["effect"] = {"id": 1219, "serial": 32}
    different_effect["select"]["contextCard"] = {"id": 1259, "serial": 31}
    different_context_card = _raw()
    different_context_card["select"]["context"] = 7
    different_context_card["select"]["effect"] = {"id": 1152, "serial": 30}
    different_context_card["select"]["contextCard"] = {"id": 1086, "serial": 33}

    first_features = visible_state_features(
        sanitize_visible_observation(first, 0)
    )
    assert first_features != visible_state_features(
        sanitize_visible_observation(different_effect, 0)
    )
    assert first_features != visible_state_features(
        sanitize_visible_observation(different_context_card, 0)
    )


def test_action_features_interact_selection_effect_with_option_card() -> None:
    first = _raw()
    first["select"]["context"] = 7
    first["select"]["effect"] = {"id": 1152}
    second = _raw()
    second["select"]["context"] = 7
    second["select"]["effect"] = {"id": 1219}
    option = {"type": 3, "cardId": 647}

    first_features = visible_action_features(
        sanitize_visible_observation(first, 0),
        option,
        base_features={},
    )
    second_features = visible_action_features(
        sanitize_visible_observation(second, 0),
        option,
        base_features={},
    )
    assert first_features != second_features


def test_state_features_capture_first_player_and_recent_log_order() -> None:
    first = _raw()
    first["current"]["firstPlayer"] = 0
    first["current"]["yourIndex"] = 0
    first["logs"] = [
        {"type": 4, "cardId": 1152, "playerIndex": 0},
        {"type": 9, "cardId": 1219, "playerIndex": 0},
    ]
    second = _raw()
    second["current"]["firstPlayer"] = 1
    second["current"]["yourIndex"] = 0
    second["logs"] = list(reversed(first["logs"]))

    assert visible_state_features(
        sanitize_visible_observation(first, 0)
    ) != visible_state_features(
        sanitize_visible_observation(second, 0)
    )


def test_state_features_preserve_per_pokemon_attachments() -> None:
    first = _raw()
    second = _raw()
    board = [
        {
            "id": 676,
            "serial": 40,
            "hp": 90,
            "maxHp": 90,
            "energyCards": [{"id": 6, "serial": 50}],
            "tools": [{"id": 1159, "serial": 60}],
        },
        {
            "id": 673,
            "serial": 41,
            "hp": 80,
            "maxHp": 80,
            "energyCards": [],
            "tools": [],
        },
    ]
    first["current"]["players"][0]["bench"] = board
    second["current"]["players"][0]["bench"] = json.loads(
        json.dumps(board)
    )
    second_board = second["current"]["players"][0]["bench"]
    second_board[0]["energyCards"] = []
    second_board[0]["tools"] = []
    second_board[1]["energyCards"] = [{"id": 6, "serial": 50}]
    second_board[1]["tools"] = [{"id": 1159, "serial": 60}]

    first_features = visible_state_features(
        sanitize_visible_observation(first, 0)
    )
    second_features = visible_state_features(
        sanitize_visible_observation(second, 0)
    )

    assert first_features != second_features
    assert first_features["n:me_bench_0_energy"] == 1.0
    assert first_features["n:me_bench_0_tools"] == 1.0
    assert second_features["n:me_bench_1_energy"] == 1.0
    assert second_features["n:me_bench_1_tools"] == 1.0
