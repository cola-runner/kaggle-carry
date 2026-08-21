from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"


def _write_live_fixture_tree(root: Path) -> Path:
    root.mkdir()
    (root / "leaderboard.csv").write_bytes((FIXTURES / "leaderboard.csv").read_bytes())
    team_rows = {
        "16514272": (
            "id,dateSubmitted,publicScore\n"
            "55001357,2026-07-26 12:41:05.657000,1153.4\n"
            "54989332,2026-07-26 02:15:56.857000,1136.8\n"
        ),
        "16463316": (
            "id,dateSubmitted,publicScore\n"
            "55011514,2026-07-26 21:41:40.223000,1152.1\n"
            "54994439,2026-07-26 07:18:45.110000,1116.5\n"
        ),
        "16431331": (
            "id,dateSubmitted,publicScore\n"
            "55046753,2026-07-28 06:06:09.773000,1035.3\n"
            "55035974,2026-07-27 18:39:18.597000,1135.0\n"
        ),
        "16561141": (
            "id,dateSubmitted,publicScore\n"
            "54995636,2026-07-26 08:11:51.240000,1117.7\n"
            "54968369,2026-07-25 05:57:07.993000,1127.9\n"
        ),
    }
    for team_id, text in team_rows.items():
        (root / f"team-{team_id}-submissions.csv").write_text(text)
    episode_text = (FIXTURES / "episodes.csv").read_text()
    for submission_id in ("55001357", "54989332", "55011514", "55035974", "54968369"):
        (root / f"submission-{submission_id}-episodes.csv").write_text(episode_text)
    return root


def test_freeze_snapshot_cli_replays_captured_official_inputs(tmp_path) -> None:
    input_dir = _write_live_fixture_tree(tmp_path / "inputs")
    out_root = tmp_path / "snapshots"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/freeze_rolling_snapshot.py"),
        "--competition",
        "pokemon-tcg-ai-battle",
        "--teacher-candidates",
        str(FIXTURES / "teacher_candidates.json"),
        "--out-root",
        str(out_root),
        "--input-dir",
        str(input_dir),
        "--cutoff",
        "2026-07-28T16:00:00Z",
        "--implementation-started",
        "2026-07-28T15:55:00Z",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    snapshots = list(out_root.glob("*/snapshot.json"))
    assert len(snapshots) == 1
    manifest = json.loads(snapshots[0].read_text())
    assert manifest["rank_ten_score"] == 1135.0
    assert {row["team_id"] for row in manifest["teachers"]} == {
        "16514272",
        "16463316",
        "16431331",
    }
    assert len(manifest["teachers"]) == 4
    serialized = snapshots[0].read_text()
    assert "credentials" not in serialized.lower()
    assert str(tmp_path) not in serialized

    repeated = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr


def test_freeze_snapshot_cli_supports_explicit_direct_teacher_mode(
    tmp_path,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    leaderboard_rows = [
        "Rank,TeamId,TeamName,Score",
        "1,16531269,Dries @ Tufa Labs,1173.5",
    ]
    leaderboard_rows.extend(
        f"{rank},{16000000 + rank},team-{rank},{1200 - rank * 7:.1f}"
        for rank in range(2, 11)
    )
    (input_dir / "leaderboard.csv").write_text(
        "\n".join(leaderboard_rows) + "\n"
    )
    (input_dir / "team-16531269-submissions.csv").write_text(
        "id,dateSubmitted,publicScore\n"
        "55002825,2026-07-26 13:51:57.320000,1173.5\n"
    )
    (input_dir / "submission-55002825-episodes.csv").write_bytes(
        (FIXTURES / "episodes.csv").read_bytes()
    )
    candidates = tmp_path / "direct_teacher.json"
    candidates.write_text(
        json.dumps(
            {
                "version": 1,
                "deck_fingerprint": "b8f251a476e7",
                "candidates": [
                    {
                        "team_id": "16531269",
                        "team_name": "Dries @ Tufa Labs",
                        "submission_id": "55002825",
                    }
                ],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/freeze_rolling_snapshot.py"),
            "--teacher-candidates",
            str(candidates),
            "--out-root",
            str(tmp_path / "snapshots"),
            "--input-dir",
            str(input_dir),
            "--cutoff",
            "2026-07-28T16:00:00Z",
            "--implementation-started",
            "2026-07-28T15:55:00Z",
            "--minimum-teacher-teams",
            "1",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        next((tmp_path / "snapshots").glob("*/snapshot.json")).read_text()
    )
    assert [row["submission_id"] for row in manifest["teachers"]] == [
        "55002825"
    ]
