from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_policy.constants import MIN_MAIN_DECISIONS  # noqa: E402
from rolling_policy.extract import (  # noqa: E402
    assert_visible_only,
    extract_episode,
)
from rolling_policy.hashing import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from rolling_policy.schema import ReplayRecord  # noqa: E402


def read_inventory(path: Path) -> list[ReplayRecord]:
    records = [
        ReplayRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("replay inventory is empty")
    return records


def write_row(file: object, row: dict[str, object], *, visible: bool) -> None:
    if visible:
        assert_visible_only(row)
    file.write(canonical_json_bytes(row) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    public_dir = snapshot_dir / "public"
    hidden_dir = snapshot_dir / "offline_hidden"
    report_path = snapshot_dir / "extraction_report.json"
    if public_dir.exists() or hidden_dir.exists() or report_path.exists():
        raise SystemExit("extracted dataset already exists for this snapshot")

    manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    inventory_path = snapshot_dir / "replay_inventory.jsonl"
    inventory_report = json.loads(
        (snapshot_dir / "replay_inventory_report.json").read_text(encoding="utf-8")
    )
    if sha256_file(inventory_path) != inventory_report["inventory_sha256"]:
        raise SystemExit("replay inventory hash mismatch")
    records = read_inventory(inventory_path)

    counts: Counter[str] = Counter()
    split_main: Counter[str] = Counter()
    seen_decisions: set[str] = set()
    with tempfile.TemporaryDirectory(
        prefix="rolling-extract-",
        dir=snapshot_dir,
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary_public = temporary / "public"
        temporary_hidden = temporary / "offline_hidden"
        temporary_public.mkdir()
        temporary_hidden.mkdir()
        paths = {
            "episodes": temporary_public / "episodes.jsonl",
            "decisions": temporary_public / "decisions.jsonl",
            "options": temporary_public / "options.jsonl",
            "hidden": temporary_hidden / "restoration.jsonl",
        }
        with (
            paths["episodes"].open("xb") as episode_file,
            paths["decisions"].open("xb") as decision_file,
            paths["options"].open("xb") as option_file,
            paths["hidden"].open("xb") as hidden_file,
        ):
            for index, inventory in enumerate(records, start=1):
                replay_path = snapshot_dir / inventory.replay_relpath
                if sha256_file(replay_path) != inventory.replay_sha256:
                    raise ValueError(
                        f"replay hash mismatch for {inventory.episode_id}"
                    )
                episode = json.loads(replay_path.read_text(encoding="utf-8"))
                extracted = extract_episode(episode, inventory)
                for row in extracted.episodes:
                    write_row(episode_file, row, visible=True)
                    counts["episodes"] += 1
                for row in extracted.decisions:
                    decision_id = str(row["decision_id"])
                    if decision_id in seen_decisions:
                        raise ValueError(f"duplicate decision_id: {decision_id}")
                    seen_decisions.add(decision_id)
                    write_row(decision_file, row, visible=True)
                    counts["decisions"] += 1
                    if row["single_choice_main"]:
                        counts["single_choice_main"] += 1
                        split_main[str(row["split"])] += 1
                for row in extracted.options:
                    write_row(option_file, row, visible=True)
                    counts["options"] += 1
                for row in extracted.hidden:
                    write_row(hidden_file, row, visible=False)
                    counts["hidden"] += 1
                    if row.get("search_begin_input") is None:
                        counts["missing_search_begin_input"] += 1
                if index % 25 == 0 or index == len(records):
                    print(
                        f"parsed={index}/{len(records)} "
                        f"decisions={counts['decisions']} "
                        f"main={counts['single_choice_main']}",
                        flush=True,
                    )

        os.replace(temporary_public, public_dir)
        os.replace(temporary_hidden, hidden_dir)

    output_paths = {
        "episodes": public_dir / "episodes.jsonl",
        "decisions": public_dir / "decisions.jsonl",
        "options": public_dir / "options.jsonl",
        "hidden": hidden_dir / "restoration.jsonl",
    }
    report = {
        "snapshot_id": manifest["snapshot_id"],
        "inventory_sha256": inventory_report["inventory_sha256"],
        "counts": dict(sorted(counts.items())),
        "single_choice_main_by_split": dict(sorted(split_main.items())),
        "output_sha256": {
            name: sha256_file(path) for name, path in output_paths.items()
        },
        "decision": (
            "PASS"
            if counts["single_choice_main"] >= MIN_MAIN_DECISIONS
            else "REJECT_INSUFFICIENT_DECISIONS"
        ),
        "minimum_single_choice_main": MIN_MAIN_DECISIONS,
    }
    with report_path.open("xb") as file:
        file.write(canonical_json_bytes(report) + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
