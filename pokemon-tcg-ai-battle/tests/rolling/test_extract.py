from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rolling_policy.extract import (
    assert_visible_only,
    extract_episode,
    option_signature,
    sanitize_visible_observation,
    selected_signature,
    write_extracted_datasets,
)
from rolling_policy.constants import EXACT_DECK_FINGERPRINT
from rolling_policy.schema import ReplayRecord, Split


FIXTURES = Path(__file__).parent / "fixtures"


def _observation() -> dict:
    return json.loads((FIXTURES / "visible_observation.json").read_text())


def test_visible_sanitizer_removes_restoration_and_opponent_hidden_cards() -> None:
    visible = sanitize_visible_observation(_observation(), acting_seat=0)
    assert "search_begin_input" not in visible
    assert "visualize" not in visible
    assert visible["current"]["players"][0]["hand"][0]["id"] == 1182
    assert visible["current"]["players"][0]["prize"] == [None]
    assert visible["current"]["players"][1]["hand"] is None
    assert visible["current"]["players"][1]["prize"] == [None]
    assert_visible_only(visible)


def test_visible_audit_rejects_nested_and_case_varied_hidden_keys() -> None:
    for key in (
        "OpponentHand",
        "opponent-hand",
        "visualize.card",
        "deckOrder",
        "prize_cards",
        "search_begin_input",
    ):
        with pytest.raises(ValueError, match="forbidden"):
            assert_visible_only({"safe": [{key: "secret"}]})


def test_option_signature_keeps_same_card_targets_distinct_by_serial() -> None:
    observation = _observation()
    options = observation["select"]["option"]
    first = option_signature(observation, options[0], acting_seat=0)
    second = option_signature(observation, options[1], acting_seat=0)
    assert first != second
    assert first[2] == 31
    assert second[2] == 32


def test_selected_signature_is_semantic_and_order_independent() -> None:
    observation = _observation()
    observation["select"]["minCount"] = 2
    observation["select"]["maxCount"] = 2
    assert selected_signature(observation, [0, 1], 0) == selected_signature(
        observation, [1, 0], 0
    )


def test_physical_writers_refuse_visible_leakage_and_separate_roots(tmp_path) -> None:
    visible = sanitize_visible_observation(_observation(), acting_seat=0)
    paths = write_extracted_datasets(
        snapshot_dir=tmp_path,
        episodes=[{"episode_id": "1"}],
        decisions=[{"decision_id": "1:0:1", "observation": visible}],
        options=[{"decision_id": "1:0:1", "option_index": 0}],
        hidden=[{"decision_id": "1:0:1", "search_begin_input": "SECRET"}],
    )
    assert paths["episodes"].is_relative_to(tmp_path / "public")
    assert paths["decisions"].is_relative_to(tmp_path / "public")
    assert paths["options"].is_relative_to(tmp_path / "public")
    assert paths["hidden"].is_relative_to(tmp_path / "offline_hidden")
    with pytest.raises(FileExistsError):
        write_extracted_datasets(
            snapshot_dir=tmp_path,
            episodes=[],
            decisions=[],
            options=[],
            hidden=[],
        )
    with pytest.raises(ValueError, match="forbidden"):
        write_extracted_datasets(
            snapshot_dir=tmp_path / "bad",
            episodes=[],
            decisions=[{"search_begin_input": "SECRET"}],
            options=[],
            hidden=[],
        )


def test_extract_episode_pairs_next_step_action_and_marks_main_choice() -> None:
    observation = _observation()
    episode = {
        "info": {"EpisodeId": 88642720},
        "rewards": [1, -1],
        "steps": [
            [
                {
                    "status": "ACTIVE",
                    "action": list(range(60)),
                    "observation": observation,
                    "visualize": [],
                },
                {},
            ],
            [
                {
                    "status": "INACTIVE",
                    "action": [1],
                    "observation": {},
                },
                {},
            ],
        ],
    }
    inventory = ReplayRecord(
        snapshot_id="20260728T160403Z",
        episode_id="88642720",
        submission_id="55001357",
        team_id="16514272",
        team_name="Dominic Peel",
        create_time_utc=datetime(2026, 7, 28, 15, tzinfo=timezone.utc),
        end_time_utc=datetime(2026, 7, 28, 15, 2, tzinfo=timezone.utc),
        target_seat=0,
        split=Split.HOLDOUT,
        replay_sha256="a" * 64,
        replay_relpath="replays/88642720.json",
        deck_fingerprint=EXACT_DECK_FINGERPRINT,
    )
    rows = extract_episode(episode, inventory)
    assert len(rows.decisions) == 1
    assert len(rows.options) == 2
    assert len(rows.hidden) == 1
    assert rows.decisions[0]["context"] == 0
    assert rows.decisions[0]["forced"] is False
    assert rows.decisions[0]["single_choice_main"] is True
    assert rows.options[1]["selected"] is True
    assert rows.options[0]["selected"] is False
    assert rows.hidden[0]["search_begin_input"] == "SECRET-ENGINE-STATE"
    assert_visible_only(rows.decisions[0])
