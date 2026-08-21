from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import league_selfplay.features as feature_module


PROJECT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT / "tests/rolling/fixtures/visible_observation.json"


def _observation() -> dict:
    return json.loads(FIXTURE.read_text())


def test_v2_preserves_visible_values_that_v1_clips_together() -> None:
    original = _observation()
    changed = copy.deepcopy(original)
    changed["current"]["players"][0]["active"][0]["hp"] = 120
    changed["current"]["players"][0]["deckCount"] = 9

    np.testing.assert_array_equal(
        feature_module.encode_options(original),
        feature_module.encode_options(changed),
    )
    assert not np.array_equal(
        feature_module.encode_options_v2(original),
        feature_module.encode_options_v2(changed),
    )


def test_v2_distinguishes_visible_source_and_target_card_identity() -> None:
    original = _observation()
    source_changed = copy.deepcopy(original)
    target_changed = copy.deepcopy(original)
    source_changed["current"]["players"][0]["active"][0]["id"] = 999
    target_changed["current"]["players"][1]["bench"][0]["id"] = 998

    encoded = feature_module.encode_options_v2(original)

    assert not np.array_equal(encoded, feature_module.encode_options_v2(source_changed))
    assert not np.array_equal(encoded, feature_module.encode_options_v2(target_changed))


def test_v2_ignores_hidden_payloads_and_opponent_hand() -> None:
    observation = _observation()
    poisoned = copy.deepcopy(observation)
    opponent = 1 - int(poisoned["current"]["yourIndex"])
    poisoned["search_begin_input"] = {
        "deckOrder": [999],
        "opponentHand": [998],
    }
    poisoned["current"]["players"][opponent]["hand"] = [{"id": 997}]
    poisoned["current"]["players"][opponent]["deck"] = [{"id": 996}]

    encoded = feature_module.encode_options_v2(observation)
    poisoned_encoded = feature_module.encode_options_v2(poisoned)

    np.testing.assert_array_equal(encoded, poisoned_encoded)
    assert encoded.dtype == np.float32
    assert encoded.shape == (len(observation["select"]["option"]), 512)
    assert np.isfinite(encoded).all()
