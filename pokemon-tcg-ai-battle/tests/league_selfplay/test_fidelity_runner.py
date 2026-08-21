from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from league_selfplay.contracts import MemberId


PROJECT = Path(__file__).resolve().parents[2]
TEST_PYTHON = sys.executable


def _measure_tree(root: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    if root.exists():
        for directory, _, names in os.walk(root):
            for name in names:
                path = Path(directory) / name
                files += 1
                total_bytes += path.stat().st_size
    return files, total_bytes


def test_dry_fidelity_run_examines_all_members_without_ppo_or_junk(
    tmp_path: Path,
    official_cg: Path,
) -> None:
    artifact_before = _measure_tree(PROJECT / "artifacts")
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            TEST_PYTHON,
            "scripts/run_driver_fidelity.py",
            "--dry-run",
            "--report",
            str(report_path),
        ],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["phases"] == [
        "preflight",
        "collect",
        "split",
        "train",
        "held_out",
        "decision",
        "cleanup",
    ]
    assert report["games"] == {"collected": 24, "held_out": 12, "train": 12}
    assert report["train_game_ids_overlap_held_out"] is False
    expected = {member.value for member in MemberId}
    assert set(report["training"]["v1_updated_members"]) == expected
    assert set(report["training"]["v2_updated_members"]) == expected
    assert set(report["metrics"]["v1"]["members"]) == expected
    assert set(report["metrics"]["v2"]["members"]) == expected
    assert report["ppo_calls"] == 0
    assert report["raw_replays_written"] == 0
    assert report["numpy_parity"]["passed"] is True
    assert report["storage"]["temp_run_exists_after_cleanup"] is False
    assert report["artifacts_before"] == report["artifacts_after"]
    assert _measure_tree(PROJECT / "artifacts") == artifact_before


def test_expired_fidelity_deadline_cleans_before_collecting() -> None:
    code = """
import json
from pathlib import Path
import league_selfplay.fidelity_runner as runner

times = iter((0.0, 361.0, 361.0))
runner.time.monotonic = lambda: next(times, 361.0)
report = runner.run_fidelity(Path.cwd(), games=24, train_games=12, seed=1, epochs=1, wall_time_seconds=360)
print(json.dumps(report.to_dict(), sort_keys=True))
"""
    result = subprocess.run(
        [TEST_PYTHON, "-c", code],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    report = json.loads(result.stdout)

    assert report["decision"]["code"] == "REJECT_DRIVER_FIDELITY"
    assert report["storage"]["temp_run_exists_after_cleanup"] is False
    assert report["games"]["collected"] == 0
