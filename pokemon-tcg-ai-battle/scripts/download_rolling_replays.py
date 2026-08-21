from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_policy.constants import (  # noqa: E402
    EXACT_DECK_FINGERPRINT,
    MIN_EPISODES,
)
from rolling_policy.hashing import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from rolling_policy.replays import (  # noqa: E402
    RateLimiter,
    ReplaySource,
    build_replay_records,
    collect_replay_sources,
    download_replay,
)
from rolling_policy.schema import ReplayRecord, TeacherSubmission, parse_utc_datetime  # noqa: E402


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def verify_snapshot_inputs(
    snapshot_dir: Path,
    manifest: dict[str, Any],
) -> None:
    hashes = manifest.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("snapshot input_sha256 is missing")
    for relative, expected in hashes.items():
        path = snapshot_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"snapshot input hash mismatch for {relative}: {actual} != {expected}"
            )


def load_sources(
    snapshot_path: Path,
) -> tuple[dict[str, Any], list[ReplaySource]]:
    manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_dir = snapshot_path.parent
    verify_snapshot_inputs(snapshot_dir, manifest)
    teachers = [
        TeacherSubmission.from_dict(row)
        for row in manifest.get("teachers") or []
    ]
    episode_rows = {
        teacher.submission_id: read_csv_rows(
            snapshot_dir
            / "raw"
            / f"submission-{teacher.submission_id}-episodes.csv"
        )
        for teacher in teachers
    }
    sources = collect_replay_sources(
        snapshot_id=str(manifest["snapshot_id"]),
        cutoff_utc=parse_utc_datetime(manifest["cutoff_utc"]),
        teachers=teachers,
        episode_rows_by_submission=episode_rows,
    )
    return manifest, sources


def write_json_exclusive(path: Path, value: object) -> None:
    with path.open("xb") as file:
        file.write(canonical_json_bytes(value) + b"\n")


def write_inventory_exclusive(
    path: Path,
    records: list[ReplayRecord],
) -> None:
    with path.open("xb") as file:
        for record in records:
            file.write(canonical_json_bytes(record.to_dict()) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--request-interval", type=float, default=1.25)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    snapshot_dir = snapshot_path.parent
    inventory_path = snapshot_dir / "replay_inventory.jsonl"
    report_path = snapshot_dir / "replay_inventory_report.json"
    if inventory_path.exists() and report_path.exists():
        print(f"inventory={inventory_path} status=already_complete")
        return
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if args.request_interval < 0:
        raise SystemExit("--request-interval cannot be negative")

    manifest, sources = load_sources(snapshot_path)
    grouped: dict[str, list[ReplaySource]] = defaultdict(list)
    for source in sources:
        grouped[source.episode_id].append(source)
    if len(grouped) < MIN_EPISODES:
        rejection = {
            "decision": "REJECT_INSUFFICIENT_EPISODES",
            "minimum_unique_episodes": MIN_EPISODES,
            "observed_unique_episodes": len(grouped),
            "observed_source_rows": len(sources),
            "snapshot_id": manifest["snapshot_id"],
        }
        write_json_exclusive(snapshot_dir / "replay_rejection.json", rejection)
        raise SystemExit(
            f"only {len(grouped)} unique completed episodes; need {MIN_EPISODES}"
        )

    replay_dir = snapshot_dir / "replays"
    replay_dir.mkdir(exist_ok=True)
    rate_limiter = RateLimiter(args.request_interval)

    preflight_sources: dict[str, ReplaySource] = {}
    for source in sources:
        previous = preflight_sources.get(source.submission_id)
        if previous is None or source.create_time_utc > previous.create_time_utc:
            preflight_sources[source.submission_id] = source
    print(
        f"sources={len(sources)} unique_replays={len(grouped)} "
        f"preflight={len(preflight_sources)}",
        flush=True,
    )
    for source in sorted(
        preflight_sources.values(),
        key=lambda row: row.submission_id,
    ):
        path = download_replay(
            source.episode_id,
            replay_dir / f"{source.episode_id}.json",
            retries=args.retries,
            before_attempt=rate_limiter.wait,
            on_rate_limit=rate_limiter.defer,
        )
        build_replay_records(
            [source],
            replay_path=path,
            snapshot_dir=snapshot_dir,
            exact_deck_fingerprint=EXACT_DECK_FINGERPRINT,
        )
        print(
            f"preflight_ok submission={source.submission_id} "
            f"episode={source.episode_id}",
            flush=True,
        )

    episode_ids = sorted(
        grouped,
        key=lambda episode_id: (
            grouped[episode_id][0].create_time_utc,
            episode_id,
        ),
    )

    def acquire(episode_id: str) -> Path:
        return download_replay(
            episode_id,
            replay_dir / f"{episode_id}.json",
            retries=args.retries,
            before_attempt=rate_limiter.wait,
            on_rate_limit=rate_limiter.defer,
        )

    completed = 0
    batch_size = max(args.jobs, args.jobs * 4)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for offset in range(0, len(episode_ids), batch_size):
            batch = episode_ids[offset : offset + batch_size]
            futures = {
                executor.submit(acquire, episode_id): episode_id
                for episode_id in batch
            }
            failed_episode = "unknown"
            try:
                for future in as_completed(futures):
                    episode_id = futures[future]
                    failed_episode = episode_id
                    future.result()
                    completed += 1
                    if completed % 25 == 0 or completed == len(episode_ids):
                        print(
                            f"downloaded={completed}/{len(episode_ids)}",
                            flush=True,
                        )
            except Exception as error:
                for future in futures:
                    future.cancel()
                raise RuntimeError(
                    f"replay download failed for {failed_episode}: "
                    f"{type(error).__name__}: {error}"
                ) from error

    records: list[ReplayRecord] = []
    for index, episode_id in enumerate(episode_ids, start=1):
        records.extend(
            build_replay_records(
                grouped[episode_id],
                replay_path=replay_dir / f"{episode_id}.json",
                snapshot_dir=snapshot_dir,
                exact_deck_fingerprint=EXACT_DECK_FINGERPRINT,
            )
        )
        if index % 50 == 0 or index == len(episode_ids):
            print(f"verified={index}/{len(episode_ids)}", flush=True)
    records.sort(
        key=lambda record: (
            record.create_time_utc,
            record.episode_id,
            record.submission_id,
        )
    )

    write_inventory_exclusive(inventory_path, records)
    split_counts = Counter(record.split.value for record in records)
    team_counts = Counter(record.team_id for record in records)
    report = {
        "snapshot_id": manifest["snapshot_id"],
        "source_rows": len(sources),
        "inventory_rows": len(records),
        "unique_replays": len(episode_ids),
        "distinct_teams": len(team_counts),
        "team_rows": dict(sorted(team_counts.items())),
        "split_rows": dict(sorted(split_counts.items())),
        "exact_deck_fingerprint": EXACT_DECK_FINGERPRINT,
        "inventory_sha256": sha256_file(inventory_path),
    }
    write_json_exclusive(report_path, report)
    print(f"inventory={inventory_path}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
