from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import probe_rank_branch_integrity as branch_probe  # noqa: E402
from rolling_policy.branching import (  # noqa: E402
    consensus_top_signature,
    stratified_root_allocation,
)
from rolling_policy.extract import (  # noqa: E402
    option_signature,
    sanitize_visible_observation,
)
from rolling_policy.features import visible_state_features  # noqa: E402
from rolling_policy.hashing import canonical_json_bytes, sha256_file  # noqa: E402
from rolling_policy.schema import parse_utc_datetime  # noqa: E402
from rolling_policy.tree_model import predict_exported_raw  # noqa: E402


PROBE_THRESHOLDS = {
    "minimum_valid_roots": 90,
    "minimum_consensus_rate": 0.70,
    "minimum_consensus_changes": 10,
}
TARGET_OPTION_TYPES = frozenset({8, 9})


def _load_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise ValueError(f"{path} does not contain an object")
    return model


def _allocate(
    decisions_path: Path,
    snapshot_id: str,
    count: int,
) -> list[dict[str, Any]]:
    candidates = []
    with decisions_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if (
                row.get("split") != "validation"
                or not row.get("single_choice_main")
                or len(row.get("selected_signature") or []) != 1
            ):
                continue
            selected_signature = tuple(int(value) for value in row["selected_signature"][0])
            selected_type = selected_signature[0]
            if selected_type not in TARGET_OPTION_TYPES:
                continue
            options = row["observation"]["select"]["option"]
            same_type_count = sum(
                int(option.get("type", -1)) == selected_type for option in options
            )
            if same_type_count < 2:
                continue
            created = parse_utc_datetime(row["create_time_utc"])
            four_hour_bucket = int(created.timestamp() // (4 * 60 * 60))
            candidates.append(
                {
                    "decision_id": str(row["decision_id"]),
                    "episode_id": str(row["episode_id"]),
                    "team_id": str(row["team_id"]),
                    "target_seat": int(row["target_seat"]),
                    "create_time_utc": str(row["create_time_utc"]),
                    "root_step": int(row["root_step"]),
                    "action_step": int(row["action_step"]),
                    "selected_type": selected_type,
                    "selected_signature": list(selected_signature),
                    "same_type_option_count": same_type_count,
                    "stratum": (
                        f"{selected_type}:{row['team_id']}:{row['target_seat']}:"
                        f"{four_hour_bucket}"
                    ),
                }
            )
    return [
        dict(row)
        for row in stratified_root_allocation(
            candidates,
            count=count,
            snapshot_id=f"{snapshot_id}|attach-evolve-signal-v1",
        )
    ]


def _unique_top(
    signatures: list[tuple[int, ...]],
    scores: list[float],
) -> tuple[int, ...] | None:
    maximum = max(scores)
    indices = [index for index, value in enumerate(scores) if value == maximum]
    return signatures[indices[0]] if len(indices) == 1 else None


def _margin(scores: list[float]) -> float:
    ordered = sorted(scores, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0


def _probe(
    row: dict[str, Any],
    snapshot_dir: Path,
    inventory: dict[tuple[str, int], Any],
    api: Any,
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    seat = int(row["target_seat"])
    record = inventory[(str(row["episode_id"]), seat)]
    replay_path = snapshot_dir / record.replay_relpath
    if sha256_file(replay_path) != record.replay_sha256:
        raise ValueError("replay hash mismatch")
    episode = json.loads(replay_path.read_text(encoding="utf-8"))
    raw_root = episode["steps"][int(row["root_step"])][seat]["observation"]
    hidden = branch_probe._hidden_lists(
        branch_probe._hidden_frame(episode, raw_root, seat),
        seat,
        raw_root,
    )
    selected_type = int(row["selected_type"])
    candidate_options = [
        (index, option)
        for index, option in enumerate(raw_root["select"]["option"])
        if int(option.get("type", -1)) == selected_type
    ]
    signatures = [
        option_signature(raw_root, option, seat)
        for _, option in candidate_options
    ]
    if len(set(signatures)) != len(signatures):
        raise ValueError("same-type candidate signatures are not unique")

    scores = {name: [] for name in models}
    leaves = []
    for (option_index, _), signature in zip(
        candidate_options,
        signatures,
        strict=True,
    ):
        _, leaf = branch_probe._search_once(
            api,
            raw_root,
            hidden,
            [option_index],
            manual_coin=True,
        )
        leaf_actor = int(leaf["current"]["yourIndex"])
        if leaf_actor != seat:
            raise ValueError("same-type branch did not return the target-seat view")
        feature_row = visible_state_features(
            sanitize_visible_observation(leaf, seat)
        )
        leaf_scores = {}
        for name, model in models.items():
            value = float(predict_exported_raw(model, [feature_row])[0])
            scores[name].append(value)
            leaf_scores[name] = value
        leaves.append(
            {
                "option_signature": list(signature),
                "scores": leaf_scores,
            }
        )

    v1_top = _unique_top(signatures, scores["v1"])
    v2_top = _unique_top(signatures, scores["v2"])
    consensus = consensus_top_signature(
        signatures,
        scores["v1"],
        scores["v2"],
    )
    teacher = tuple(int(value) for value in row["selected_signature"])
    return {
        **row,
        "valid": True,
        "v1_top_signature": list(v1_top) if v1_top else None,
        "v2_top_signature": list(v2_top) if v2_top else None,
        "consensus_signature": list(consensus) if consensus else None,
        "models_agree": consensus is not None,
        "consensus_matches_teacher": consensus == teacher,
        "consensus_changes_teacher": consensus is not None and consensus != teacher,
        "v1_raw_margin": _margin(scores["v1"]),
        "v2_raw_margin": _margin(scores["v2"]),
        "leaves": leaves,
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cg-dir", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=100)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    allocation_path = snapshot_dir / "attach_evolve_signal_allocation.json"
    report_path = snapshot_dir / "attach_evolve_signal_report.json"
    if report_path.exists():
        raise SystemExit("attach/evolve signal probe already completed")

    if allocation_path.exists():
        allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
        if (
            allocation.get("snapshot_id") != snapshot["snapshot_id"]
            or allocation.get("requested_roots") != args.roots
            or allocation.get("thresholds") != PROBE_THRESHOLDS
        ):
            raise SystemExit("existing signal allocation does not match request")
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
            "thresholds": PROBE_THRESHOLDS,
            "roots": rows,
        }
        with allocation_path.open("xb") as file:
            file.write(canonical_json_bytes(allocation) + b"\n")
        print(f"sealed allocation: {allocation_path}", flush=True)

    models = {
        name: _load_model(snapshot_dir / "value_models" / f"{name}.json")
        for name in ("v1", "v2")
    }
    inventory = branch_probe._inventory(snapshot_dir / "replay_inventory.jsonl")
    api = branch_probe._load_api(args.cg_dir.resolve())
    results = []
    for index, row in enumerate(rows, start=1):
        try:
            result = _probe(row, snapshot_dir, inventory, api, models)
        except Exception as error:
            result = {
                **row,
                "valid": False,
                "models_agree": False,
                "consensus_matches_teacher": False,
                "consensus_changes_teacher": False,
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        if index % 10 == 0 or result["error"]:
            print(
                f"root={index}/{len(rows)} valid={result['valid']} "
                f"agree={result['models_agree']} "
                f"change={result['consensus_changes_teacher']} "
                f"error={result['error'] or '-'}",
                flush=True,
            )

    valid = [row for row in results if row["valid"]]
    consensus_count = sum(bool(row["models_agree"]) for row in valid)
    consensus_changes = sum(
        bool(row["consensus_changes_teacher"]) for row in valid
    )
    consensus_teacher_matches = sum(
        bool(row["consensus_matches_teacher"]) for row in valid
    )
    consensus_rate = consensus_count / len(valid) if valid else 0.0
    passed = (
        len(valid) >= PROBE_THRESHOLDS["minimum_valid_roots"]
        and consensus_rate >= PROBE_THRESHOLDS["minimum_consensus_rate"]
        and consensus_changes >= PROBE_THRESHOLDS["minimum_consensus_changes"]
    )
    summary = {
        "valid_roots": len(valid),
        "model_consensus_roots": consensus_count,
        "model_consensus_rate": consensus_rate,
        "consensus_matches_teacher": consensus_teacher_matches,
        "consensus_changes_teacher": consensus_changes,
    }
    report = {
        "snapshot_id": snapshot["snapshot_id"],
        "allocation_sha256": sha256_file(allocation_path),
        "requested_roots": args.roots,
        "thresholds": PROBE_THRESHOLDS,
        "summary": summary,
        "decision": "PASS" if passed else "REJECT_COUNTERFACTUAL_SIGNAL",
        "roots": results,
    }
    with report_path.open("xb") as file:
        file.write(canonical_json_bytes(report) + b"\n")
    print(
        json.dumps(
            {key: report[key] for key in report if key != "roots"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
