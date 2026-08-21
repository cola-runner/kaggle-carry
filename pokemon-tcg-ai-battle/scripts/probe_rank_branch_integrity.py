from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_policy.branching import (  # noqa: E402
    BranchLeaf,
    CoinSchedule,
    compare_branch_orders,
    is_deterministic_branch_option,
    recorded_next_observation,
    stratified_root_allocation,
    visible_engine_sha256,
)
from rolling_policy.extract import option_signature  # noqa: E402
from rolling_policy.hashing import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from rolling_policy.schema import ReplayRecord, parse_utc_datetime  # noqa: E402


def _card_id(card: object) -> int:
    return int(card.get("id") or 0) if isinstance(card, dict) else 0


def _card_serial(card: object) -> int:
    return int(card.get("serial") or 0) if isinstance(card, dict) else 0


def _cards_signature(cards: object) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(cards, list):
        return ()
    return tuple(
        (_card_id(card), _card_serial(card), int(card.get("hp") or 0))
        for card in cards
        if isinstance(card, dict)
    )


def _state_signature(
    current: dict[str, Any],
    acting_seat: int,
) -> tuple[Any, ...]:
    players = current.get("players") or [{}, {}]
    player_rows = []
    for seat, player in enumerate(players[:2]):
        hand = player.get("hand")
        player_rows.append(
            (
                int(player.get("deckCount") or 0),
                int(player.get("handCount") or 0),
                len(player.get("prize") or []),
                len(player.get("discard") or []),
                _cards_signature(player.get("active")),
                _cards_signature(player.get("bench")),
                (
                    _cards_signature(hand)
                    if seat == acting_seat and hand is not None
                    else ()
                ),
            )
        )
    return (
        int(current.get("turn") or 0),
        int(current.get("turnActionCount") or 0),
        int(current.get("yourIndex") or 0),
        bool(current.get("energyAttached")),
        bool(current.get("supporterPlayed")),
        bool(current.get("stadiumPlayed")),
        _cards_signature(current.get("stadium")),
        tuple(player_rows),
    )


def _visual_frames(episode: dict[str, Any]) -> list[dict[str, Any]]:
    for step in episode.get("steps", []):
        for record in step[:2]:
            frames = record.get("visualize") or []
            if frames:
                return frames
    return []


def _hidden_frame(
    episode: dict[str, Any],
    observation: dict[str, Any],
    acting_seat: int,
) -> dict[str, Any]:
    wanted = _state_signature(observation["current"], acting_seat)
    matches = [
        frame
        for frame in _visual_frames(episode)
        if _state_signature(frame.get("current") or {}, acting_seat) == wanted
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one hidden frame, found {len(matches)}")
    return matches[0]


def _hidden_ids(cards: object) -> list[int]:
    if not isinstance(cards, list):
        return []
    return [_card_id(card) for card in cards if _card_id(card)]


def _hidden_lists(
    frame: dict[str, Any],
    acting_seat: int,
    observation: dict[str, Any],
) -> tuple[list[int], ...]:
    players = (frame.get("current") or {}).get("players") or [{}, {}]
    me = players[acting_seat]
    rival = players[1 - acting_seat]
    visible_players = (observation.get("current") or {}).get("players") or [{}, {}]
    visible_active = visible_players[1 - acting_seat].get("active") or []
    rival_active = []
    if visible_active and visible_active[0] is None:
        rival_active = _hidden_ids(rival.get("active"))[:1]
    return (
        _hidden_ids(me.get("deck")),
        _hidden_ids(me.get("prize")),
        _hidden_ids(rival.get("deck")),
        _hidden_ids(rival.get("prize")),
        _hidden_ids(rival.get("hand")),
        rival_active,
    )


def _load_api(cg_dir: Path) -> Any:
    if not (cg_dir / "cg").is_dir():
        raise ValueError(f"{cg_dir} does not contain cg/")
    sys.path.insert(0, str(cg_dir))
    import cg.api as api  # type: ignore

    return api


def _search_once(
    api: Any,
    raw_root: dict[str, Any],
    hidden: tuple[list[int], ...],
    action: list[int],
    *,
    manual_coin: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_state = None
    next_state = None
    try:
        root_state = api.search_begin(
            api.to_observation_class(raw_root),
            your_deck=hidden[0],
            your_prize=hidden[1],
            opponent_deck=hidden[2],
            opponent_prize=hidden[3],
            opponent_hand=hidden[4],
            opponent_active=hidden[5],
            manual_coin=manual_coin,
        )
        next_state = api.search_step(root_state.searchId, action)
        return asdict(root_state.observation), asdict(next_state.observation)
    finally:
        search_ids = {
            state.searchId
            for state in (root_state, next_state)
            if state is not None
        }
        for search_id in search_ids:
            try:
                api.search_release(search_id)
            except Exception:
                pass
        try:
            api.search_end()
        except Exception:
            pass


def _allocate(
    decisions_path: Path,
    snapshot_id: str,
    count: int,
) -> list[dict[str, Any]]:
    eligible = []
    with decisions_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("split") != "validation" or not row.get(
                "single_choice_main"
            ):
                continue
            created = parse_utc_datetime(row["create_time_utc"])
            four_hour_bucket = int(created.timestamp() // (4 * 60 * 60))
            eligible.append(
                {
                    "decision_id": str(row["decision_id"]),
                    "episode_id": str(row["episode_id"]),
                    "team_id": str(row["team_id"]),
                    "target_seat": int(row["target_seat"]),
                    "create_time_utc": str(row["create_time_utc"]),
                    "root_step": int(row["root_step"]),
                    "action_step": int(row["action_step"]),
                    "option_count": int(row["option_count"]),
                    "stratum": (
                        f"{row['team_id']}:{row['target_seat']}:"
                        f"{four_hour_bucket}"
                    ),
                }
            )
    return [
        dict(row)
        for row in stratified_root_allocation(
            eligible,
            count=count,
            snapshot_id=snapshot_id,
        )
    ]


def _inventory(path: Path) -> dict[tuple[str, int], ReplayRecord]:
    records = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = ReplayRecord.from_dict(json.loads(line))
            key = (record.episode_id, record.target_seat)
            if key in records:
                raise ValueError(f"duplicate inventory episode/seat: {key}")
            records[key] = record
    return records


def _probe_root(
    row: dict[str, Any],
    snapshot_dir: Path,
    inventory: dict[tuple[str, int], ReplayRecord],
    api: Any,
    snapshot_id: str,
) -> dict[str, Any]:
    seat = int(row["target_seat"])
    record = inventory[(str(row["episode_id"]), seat)]
    replay_path = snapshot_dir / record.replay_relpath
    if sha256_file(replay_path) != record.replay_sha256:
        raise ValueError("replay hash mismatch")
    episode = json.loads(replay_path.read_text(encoding="utf-8"))
    raw_root = episode["steps"][int(row["root_step"])][seat]["observation"]
    action_record = episode["steps"][int(row["action_step"])][seat]
    official_action = action_record.get("action")
    if not isinstance(official_action, list):
        raise ValueError("recorded action is missing")
    expected = dict(
        recorded_next_observation(
            episode["steps"][int(row["action_step"])],
        )
    )
    hidden = _hidden_lists(_hidden_frame(episode, raw_root, seat), seat, raw_root)

    restored, reproduced = _search_once(
        api,
        raw_root,
        hidden,
        official_action,
        manual_coin=False,
    )
    root_hash = visible_engine_sha256(raw_root, seat)
    restored_hash = visible_engine_sha256(restored, seat)
    expected_actor = int(expected["current"]["yourIndex"])
    reproduced_actor = int(reproduced["current"]["yourIndex"])
    expected_hash = visible_engine_sha256(expected, expected_actor)
    reproduced_hash = visible_engine_sha256(reproduced, reproduced_actor)

    options = raw_root["select"]["option"]
    deterministic_indices = [
        index
        for index, option in enumerate(options)
        if is_deterministic_branch_option(option)
    ]
    if not deterministic_indices:
        raise ValueError("root has no deterministic branch options")
    forward = []
    reverse = []
    for order, output in (
        (deterministic_indices, forward),
        (reversed(deterministic_indices), reverse),
    ):
        for option_index in order:
            _, leaf = _search_once(
                api,
                raw_root,
                hidden,
                [option_index],
                manual_coin=True,
            )
            leaf_actor = int(leaf["current"]["yourIndex"])
            output.append(
                BranchLeaf(
                    option_signature=option_signature(
                        raw_root,
                        options[option_index],
                        seat,
                    ),
                    leaf_visible_sha256=visible_engine_sha256(leaf, leaf_actor),
                    v1_score=0.0,
                    v2_score=0.0,
                    stopped_reason="one_step",
                )
            )
    order_errors = compare_branch_orders(forward, reverse)
    coins = CoinSchedule.from_decision(snapshot_id, str(row["decision_id"]))
    return {
        **row,
        "root_visible_sha256": root_hash,
        "restored_visible_sha256": restored_hash,
        "recorded_next_visible_sha256": expected_hash,
        "reproduced_next_visible_sha256": reproduced_hash,
        "root_restored": root_hash == restored_hash,
        "recorded_transition_reproduced": (
            expected_actor == reproduced_actor and expected_hash == reproduced_hash
        ),
        "branch_order_invariant": not order_errors,
        "branch_order_errors": list(order_errors),
        "option_signatures": [
            list(leaf.option_signature) for leaf in forward
        ],
        "excluded_stochastic_option_signatures": [
            list(option_signature(raw_root, option, seat))
            for index, option in enumerate(options)
            if index not in deterministic_indices
        ],
        "coin_schedule": list(coins.values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cg-dir", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=12)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    allocation_path = snapshot_dir / "rank_branch_probe_allocation.json"
    report_path = snapshot_dir / "rank_branch_probe_report.json"
    if report_path.exists():
        raise SystemExit("rank branch probe already completed for this snapshot")
    if allocation_path.exists():
        allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
        if (
            allocation.get("snapshot_id") != snapshot["snapshot_id"]
            or allocation.get("requested_roots") != args.roots
        ):
            raise SystemExit("existing allocation does not match this request")
        rows = allocation["roots"]
    else:
        rows = _allocate(
            snapshot_dir / "public" / "decisions.jsonl",
            snapshot["snapshot_id"],
            args.roots,
        )
        allocation = {
            "snapshot_id": snapshot["snapshot_id"],
            "requested_roots": args.roots,
            "allocated_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "roots": rows,
        }
        with allocation_path.open("xb") as file:
            file.write(canonical_json_bytes(allocation) + b"\n")
        print(f"sealed allocation: {allocation_path}", flush=True)

    inventory = _inventory(snapshot_dir / "replay_inventory.jsonl")
    api = _load_api(args.cg_dir.resolve())
    results = []
    for index, row in enumerate(rows, start=1):
        try:
            result = _probe_root(
                row,
                snapshot_dir,
                inventory,
                api,
                snapshot["snapshot_id"],
            )
            result["error"] = ""
        except Exception as error:
            result = {
                **row,
                "root_restored": False,
                "recorded_transition_reproduced": False,
                "branch_order_invariant": False,
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        print(
            f"root={index}/{len(rows)} id={row['decision_id']} "
            f"restore={result['root_restored']} "
            f"replay={result['recorded_transition_reproduced']} "
            f"order={result['branch_order_invariant']} "
            f"error={result['error'] or '-'}",
            flush=True,
        )

    counts = {
        name: sum(bool(row.get(name)) for row in results)
        for name in (
            "root_restored",
            "recorded_transition_reproduced",
            "branch_order_invariant",
        )
    }
    passed = all(value == len(rows) for value in counts.values())
    report = {
        "snapshot_id": snapshot["snapshot_id"],
        "allocation_sha256": sha256_file(allocation_path),
        "requested_roots": args.roots,
        "counts": counts,
        "decision": "PASS" if passed else "REJECT_BRANCH_INFRASTRUCTURE",
        "roots": results,
    }
    with report_path.open("xb") as file:
        file.write(canonical_json_bytes(report) + b"\n")
    print(json.dumps({key: report[key] for key in report if key != "roots"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
