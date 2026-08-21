from __future__ import annotations

from rolling_policy.branching import (
    BranchLeaf,
    CoinSchedule,
    compare_branch_orders,
    consensus_top_signature,
    is_deterministic_branch_option,
    recorded_next_observation,
    stratified_root_allocation,
    visible_engine_sha256,
)


def _leaf(signature: tuple[int, ...], value: float) -> BranchLeaf:
    return BranchLeaf(
        option_signature=signature,
        leaf_visible_sha256=f"{signature[0]:064x}",
        v1_score=value,
        v2_score=value + 0.1,
        stopped_reason="one_step",
    )


def test_engine_hash_ignores_dataclass_defaults_time_and_card_owner() -> None:
    raw = {
        "remainingOverageTime": 12.3,
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "hand": [{"id": 7, "serial": 3, "playerIndex": 0}],
                    "prize": [],
                },
                {"hand": None, "prize": [None]},
            ],
        },
        "logs": [{"type": 4}],
        "select": {"option": [{"type": 14}]},
    }
    typed_shape = {
        "current": {
            "yourIndex": 0,
            "players": [
                {"hand": [{"id": 7, "serial": 3}], "prize": []},
                {"hand": None, "prize": [None]},
            ],
        },
        "logs": [{"type": 4, "attackId": None, "value": None}],
        "select": {
            "option": [
                {
                    "type": 14,
                    "attackId": None,
                    "playerIndex": None,
                }
            ]
        },
        "search_begin_input": None,
    }
    assert visible_engine_sha256(raw, 0) == visible_engine_sha256(typed_shape, 0)


def test_engine_state_hash_ignores_replay_step_logs_and_copy_serials() -> None:
    first = {
        "step": 10,
        "current": {
            "yourIndex": 0,
            "players": [
                {"hand": [{"id": 7, "serial": 3}], "prize": []},
                {"hand": None, "prize": []},
            ],
        },
        "logs": [{"type": 4, "cardId": 7, "serial": 3}],
        "select": {"option": [{"type": 14}]},
    }
    second = {
        **first,
        "step": 11,
        "logs": [],
        "current": {
            "yourIndex": 0,
            "players": [
                {"hand": [{"id": 7, "serial": 99}], "prize": []},
                {"hand": None, "prize": []},
            ],
        },
    }
    assert visible_engine_sha256(first, 0) == visible_engine_sha256(second, 0)


def test_branch_comparison_keys_by_signature_not_enumeration_order() -> None:
    forward = [_leaf((1,), 0.4), _leaf((2,), 0.7)]
    reverse = [_leaf((2,), 0.7), _leaf((1,), 0.4)]
    assert compare_branch_orders(forward, reverse) == ()


def test_branch_comparison_detects_shared_state_contamination() -> None:
    forward = [_leaf((1,), 0.4), _leaf((2,), 0.7)]
    contaminated = [_leaf((2,), 0.8), _leaf((1,), 0.4)]
    errors = compare_branch_orders(forward, contaminated)
    assert any("v1_score" in error for error in errors)


def test_coin_schedule_is_reproducible_and_bound_to_decision() -> None:
    first = CoinSchedule.from_decision("snapshot", "episode:0:4", count=128)
    second = CoinSchedule.from_decision("snapshot", "episode:0:4", count=128)
    other = CoinSchedule.from_decision("snapshot", "episode:0:5", count=128)
    assert first == second
    assert first != other
    assert len(first.values) == 128


def test_root_allocation_spreads_first_round_across_strata() -> None:
    rows = [
        {"decision_id": f"a{index}", "stratum": "a"} for index in range(4)
    ] + [{"decision_id": f"b{index}", "stratum": "b"} for index in range(4)]
    selected = stratified_root_allocation(
        rows,
        count=2,
        snapshot_id="snapshot",
    )
    assert {row["stratum"] for row in selected} == {"a", "b"}
    assert selected == stratified_root_allocation(
        list(reversed(rows)),
        count=2,
        snapshot_id="snapshot",
    )


def test_recorded_next_observation_uses_active_seat_view() -> None:
    inactive = {"current": {"yourIndex": 0, "turn": 3}}
    active = {"current": {"yourIndex": 1, "turn": 4}}
    step = [
        {"status": "INACTIVE", "observation": inactive},
        {"status": "ACTIVE", "observation": active},
    ]
    assert recorded_next_observation(step) is active


def test_deterministic_branch_boundary_excludes_card_play_and_ability() -> None:
    assert is_deterministic_branch_option({"type": 8})
    assert is_deterministic_branch_option({"type": 9})
    assert is_deterministic_branch_option({"type": 12})
    assert is_deterministic_branch_option({"type": 13})
    assert is_deterministic_branch_option({"type": 14})
    assert not is_deterministic_branch_option({"type": 7})
    assert not is_deterministic_branch_option({"type": 10})


def test_consensus_top_signature_is_order_invariant_and_rejects_ties() -> None:
    signatures = [(1,), (2,), (3,)]
    assert consensus_top_signature(
        signatures,
        [0.1, 0.8, 0.2],
        [0.2, 0.7, 0.3],
    ) == (2,)
    assert consensus_top_signature(
        list(reversed(signatures)),
        [0.2, 0.8, 0.1],
        [0.3, 0.7, 0.2],
    ) == (2,)
    assert (
        consensus_top_signature(
            signatures,
            [0.8, 0.8, 0.2],
            [0.7, 0.6, 0.3],
        )
        is None
    )
