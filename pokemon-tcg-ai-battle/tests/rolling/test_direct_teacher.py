from __future__ import annotations

import json
from pathlib import Path

import pytest

from rolling_policy.direct_teacher import (
    direct_type_features,
    direct_semantic_signature,
    direct_option_weights,
    eligible_direct_decision,
    load_direct_examples,
    type_routed_prediction,
)
from rolling_policy.extract import option_signature, sanitize_visible_observation
from rolling_policy.imitation import ImitationDecision


FIXTURES = Path(__file__).parent / "fixtures"


def _observation() -> dict:
    raw = json.loads((FIXTURES / "visible_observation.json").read_text())
    raw["select"]["context"] = 0
    raw["select"]["minCount"] = 1
    raw["select"]["maxCount"] = 1
    return sanitize_visible_observation(raw, 0)


def _episode(episode_id: str, split: str) -> dict:
    return {
        "episode_id": episode_id,
        "target_seat": 0,
        "split": split,
        "team_id": "direct-teacher",
        "create_time_utc": "2026-07-29T00:00:00Z",
    }


def _decision(episode_id: str, split: str) -> dict:
    observation = _observation()
    selected = option_signature(observation, observation["select"]["option"][0], 0)
    return {
        "episode_id": episode_id,
        "decision_id": f"{episode_id}:0:1",
        "target_seat": 0,
        "split": split,
        "context": 0,
        "forced": False,
        "single_choice_main": True,
        "min_count": 1,
        "max_count": 1,
        "selected_signature": [selected],
        "observation": observation,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_direct_context_eligibility_is_exact_and_nonforced() -> None:
    assert eligible_direct_decision(
        {
            "context": 0,
            "forced": False,
            "single_choice_main": True,
            "min_count": 1,
            "max_count": 1,
        },
        "main",
    )
    assert eligible_direct_decision(
        {
            "context": 5,
            "forced": False,
            "min_count": 0,
            "max_count": 2,
        },
        "bench",
    )
    assert eligible_direct_decision(
        {
            "context": 7,
            "forced": False,
            "min_count": 0,
            "max_count": 1,
        },
        "search",
    )
    assert not eligible_direct_decision(
        {
            "context": 7,
            "forced": True,
            "min_count": 1,
            "max_count": 1,
        },
        "search",
    )
    assert not eligible_direct_decision(
        {
            "context": 5,
            "forced": False,
            "min_count": 0,
            "max_count": 2,
        },
        "search",
    )


def test_direct_option_weights_support_empty_and_all_selected_sets() -> None:
    assert direct_option_weights(
        [0, 0, 0],
        decision_weight=3.0,
    ) == pytest.approx([1.0, 1.0, 1.0])
    assert direct_option_weights(
        [1, 1],
        decision_weight=3.0,
    ) == pytest.approx([1.5, 1.5])
    assert direct_option_weights(
        [1, 0, 0],
        decision_weight=2.0,
    ) == pytest.approx([1.0, 0.5, 0.5])


def test_main_semantics_merge_interchangeable_copies_but_keep_targets() -> None:
    first_play = (7, 1152, 105, 0, 0, -1, 1, -1, 5, -1, -1, -1, -1, -1, -1)
    second_play = (7, 1152, 106, 0, 0, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1)
    assert direct_semantic_signature(first_play, "main") == (
        direct_semantic_signature(second_play, "main")
    )

    first_target = (8, 7, 63, 112, 75, -1, 1, 2, 1, 5, 0, -1, -1, -1, -1)
    same_target_other_energy = (
        8,
        7,
        64,
        112,
        75,
        -1,
        1,
        2,
        2,
        5,
        0,
        -1,
        -1,
        -1,
        -1,
    )
    different_target = (8, 7, 64, 112, 77, -1, 1, 2, 2, 5, 1, -1, -1, -1, -1)
    assert direct_semantic_signature(first_target, "main") == (
        direct_semantic_signature(same_target_other_energy, "main")
    )
    assert direct_semantic_signature(first_target, "main") != (
        direct_semantic_signature(different_target, "main")
    )


def test_type_features_describe_available_semantic_actions() -> None:
    observation = _observation()
    hand = observation["current"]["players"][0]["hand"]
    hand.append({"id": 1152, "serial": 105, "playerIndex": 0})
    observation["select"]["option"].append(
        {"type": 7, "index": len(hand) - 1}
    )
    features = direct_type_features(observation)
    assert features["available:type=7"] >= 1.0
    assert any(name.startswith("available:action=7:") for name in features)
    assert not any("serial" in name.lower() for name in features)


def test_type_router_masks_unavailable_types_and_then_ranks_within_type() -> None:
    decision = ImitationDecision(
        "routed",
        ((7, 1152), (10, 112, 75), (10, 112, 77)),
        frozenset({(10, 112, 77)}),
    )
    assert type_routed_prediction(
        decision,
        [0.95, 0.4, 0.8],
        classes=[7, 10, 14],
        type_probabilities=[0.1, 0.3, 0.6],
    ) == frozenset({(10, 112, 77)})


def test_loader_skips_holdout_before_reading_teacher_label(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    episodes = tmp_path / "episodes.jsonl"
    holdout = _decision("holdout", "holdout")
    del holdout["selected_signature"]
    _write_jsonl(
        decisions,
        [
            _decision("train", "train"),
            _decision("validation", "validation"),
            holdout,
        ],
    )
    _write_jsonl(
        episodes,
        [
            _episode("train", "train"),
            _episode("validation", "validation"),
            _episode("holdout", "holdout"),
        ],
    )

    examples = load_direct_examples(decisions, episodes, "main")

    assert len(examples["train"]["decisions"]) == 1
    assert len(examples["validation"]["decisions"]) == 1
    assert examples["skipped_holdout_rows_before_label_access"] == 1
    assert examples["train"]["bounds"] == [(1, 1)]
