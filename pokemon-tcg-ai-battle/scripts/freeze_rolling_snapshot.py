from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_policy.constants import (  # noqa: E402
    COMPETITION,
    EXACT_DECK_FINGERPRINT,
    HOLDOUT_WINDOW_HOURS,
    MIN_TEACHER_TEAMS,
    SOURCE_WINDOW_HOURS,
    VALIDATION_WINDOW_HOURS,
)
from rolling_policy.hashing import canonical_json_bytes, sha256_file  # noqa: E402
from rolling_policy.schema import (  # noqa: E402
    SnapshotManifest,
    TeacherSubmission,
    parse_utc_datetime,
)
from rolling_policy.snapshot import (  # noqa: E402
    completed_public_episode_ids,
    eligible_teachers,
    parse_leaderboard,
    parse_submission_rows,
    rank_ten_threshold,
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def run_csv_command(command: list[str], output: Path) -> None:
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n"
            f"{process.stderr[-1000:]}"
        )
    output.write_text(process.stdout, encoding="utf-8")


def extract_leaderboard_zip(raw_dir: Path) -> Path:
    archives = sorted(raw_dir.glob("*.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one leaderboard zip, found {len(archives)}")
    with zipfile.ZipFile(archives[0]) as archive:
        csv_members = [
            info
            for info in archive.infolist()
            if not info.is_dir() and PurePosixPath(info.filename).suffix == ".csv"
        ]
        if len(csv_members) != 1:
            raise RuntimeError(
                f"expected one leaderboard CSV in archive, found {len(csv_members)}"
            )
        member = csv_members[0]
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe leaderboard archive member: {member.filename}")
        destination = raw_dir / "leaderboard.csv"
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    return destination


def copy_fixture_inputs(
    input_dir: Path,
    raw_dir: Path,
    candidate_rows: list[dict[str, Any]],
) -> Path:
    leaderboard = input_dir / "leaderboard.csv"
    if not leaderboard.exists():
        raise FileNotFoundError(leaderboard)
    shutil.copy2(leaderboard, raw_dir / "leaderboard.csv")
    for team_id in sorted({str(row["team_id"]) for row in candidate_rows}):
        source = input_dir / f"team-{team_id}-submissions.csv"
        shutil.copy2(source, raw_dir / source.name)
    for row in candidate_rows:
        submission_id = str(row["submission_id"])
        source = input_dir / f"submission-{submission_id}-episodes.csv"
        shutil.copy2(source, raw_dir / source.name)
    return raw_dir / "leaderboard.csv"


def collect_live_inputs(
    competition: str,
    raw_dir: Path,
    candidate_rows: list[dict[str, Any]],
) -> Path:
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "leaderboard",
            competition,
            "-d",
            "-p",
            str(raw_dir),
            "-q",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    leaderboard = extract_leaderboard_zip(raw_dir)
    for team_id in sorted({str(row["team_id"]) for row in candidate_rows}):
        run_csv_command(
            ["kaggle", "competitions", "team-submissions", team_id, "--csv"],
            raw_dir / f"team-{team_id}-submissions.csv",
        )
    for row in candidate_rows:
        submission_id = str(row["submission_id"])
        run_csv_command(
            ["kaggle", "competitions", "episodes", submission_id, "--csv"],
            raw_dir / f"submission-{submission_id}-episodes.csv",
        )
    return leaderboard


def freeze_snapshot(
    *,
    competition: str,
    teacher_candidates_path: Path,
    out_root: Path,
    cutoff_utc: datetime,
    implementation_started_utc: datetime,
    input_dir: Path | None,
    minimum_teacher_teams: int = MIN_TEACHER_TEAMS,
) -> Path:
    if competition != COMPETITION:
        raise ValueError(f"competition must be {COMPETITION!r}")
    cutoff = parse_utc_datetime(cutoff_utc)
    implementation_started = parse_utc_datetime(implementation_started_utc)
    snapshot_id = cutoff.strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = out_root / snapshot_id
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"snapshot already exists: {snapshot_dir}") from error
    raw_dir = snapshot_dir / "raw"
    raw_dir.mkdir()

    config = json.loads(teacher_candidates_path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("teacher candidate config version must be 1")
    if config.get("deck_fingerprint") != EXACT_DECK_FINGERPRINT:
        raise ValueError("teacher candidate config deck fingerprint is not exact")
    candidate_rows = list(config.get("candidates") or [])
    if not candidate_rows:
        raise ValueError("teacher candidate config is empty")

    if input_dir is None:
        leaderboard_path = collect_live_inputs(
            competition, raw_dir, candidate_rows
        )
    else:
        leaderboard_path = copy_fixture_inputs(input_dir, raw_dir, candidate_rows)

    leaderboard_rows = parse_leaderboard(leaderboard_path)
    threshold = rank_ten_threshold(leaderboard_rows)
    leaderboard_teams = {
        str(row.get("TeamId", "")): str(row.get("TeamName", ""))
        for row in leaderboard_rows
    }

    submissions_by_team = {}
    for team_id in sorted({str(row["team_id"]) for row in candidate_rows}):
        path = raw_dir / f"team-{team_id}-submissions.csv"
        submissions_by_team[team_id] = parse_submission_rows(read_csv_rows(path))

    active_submission_ids: set[str] = set()
    teachers: list[TeacherSubmission] = []
    for row in candidate_rows:
        team_id = str(row["team_id"])
        team_name = str(row["team_name"])
        submission_id = str(row["submission_id"])
        if leaderboard_teams.get(team_id) != team_name:
            raise ValueError(
                f"leaderboard team mismatch for {team_id}: "
                f"{leaderboard_teams.get(team_id)!r} != {team_name!r}"
            )
        active_submission = submissions_by_team[team_id].get(submission_id)
        if active_submission is None:
            continue
        episodes_path = raw_dir / f"submission-{submission_id}-episodes.csv"
        episode_ids = completed_public_episode_ids(
            read_csv_rows(episodes_path),
            cutoff,
        )
        if episode_ids:
            active_submission_ids.add(submission_id)
        teachers.append(
            TeacherSubmission(
                team_id=team_id,
                team_name=team_name,
                submission_id=submission_id,
                score=active_submission.score,
                deck_fingerprint=EXACT_DECK_FINGERPRINT,
                tracked_at_cutoff=True,
                submitted_at_utc=active_submission.submitted_at_utc,
            )
        )

    eligible = eligible_teachers(
        teachers,
        rank_ten_score=threshold,
        active_submission_ids=active_submission_ids,
        minimum_teacher_teams=minimum_teacher_teams,
    )
    input_hashes = {
        path.relative_to(snapshot_dir).as_posix(): sha256_file(path)
        for path in sorted(raw_dir.glob("*"))
        if path.is_file()
    }
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        implementation_started_utc=implementation_started,
        cutoff_utc=cutoff,
        rank_ten_score=threshold,
        leaderboard_sha256=sha256_file(leaderboard_path),
        teacher_candidates_sha256=sha256_file(teacher_candidates_path),
        input_sha256=input_hashes,
        teachers=eligible,
        source_window_start_utc=cutoff - timedelta(hours=SOURCE_WINDOW_HOURS),
        validation_start_utc=cutoff
        - timedelta(hours=HOLDOUT_WINDOW_HOURS + VALIDATION_WINDOW_HOURS),
        holdout_start_utc=cutoff - timedelta(hours=HOLDOUT_WINDOW_HOURS),
        minimum_teacher_teams=minimum_teacher_teams,
    )
    snapshot_path = snapshot_dir / "snapshot.json"
    with snapshot_path.open("xb") as file:
        file.write(canonical_json_bytes(manifest.to_dict()) + b"\n")
    return snapshot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--teacher-candidates", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--implementation-started", default=None)
    parser.add_argument(
        "--minimum-teacher-teams",
        type=int,
        default=MIN_TEACHER_TEAMS,
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    cutoff = parse_utc_datetime(args.cutoff) if args.cutoff else now
    implementation_started = (
        parse_utc_datetime(args.implementation_started)
        if args.implementation_started
        else cutoff
    )
    try:
        path = freeze_snapshot(
            competition=args.competition,
            teacher_candidates_path=args.teacher_candidates.resolve(),
            out_root=args.out_root.resolve(),
            cutoff_utc=cutoff,
            implementation_started_utc=implementation_started,
            input_dir=args.input_dir.resolve() if args.input_dir else None,
            minimum_teacher_teams=args.minimum_teacher_teams,
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"snapshot={path}")


if __name__ == "__main__":
    main()
