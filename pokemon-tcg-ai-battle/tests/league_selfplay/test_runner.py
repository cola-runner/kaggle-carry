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
    if not root.exists():
        return files, total_bytes
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            files += 1
            total_bytes += path.stat().st_size
    return files, total_bytes


def test_dry_run_executes_every_phase_and_leaves_no_junk(
    tmp_path: Path,
    official_cg: Path,
) -> None:
    artifact_before = _measure_tree(PROJECT / "artifacts")
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            TEST_PYTHON,
            "scripts/run_four_policy_league.py",
            "--dry-run",
            "--report",
            str(report_path),
        ],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["phases"] == [
        "preflight",
        "bootstrap",
        "start_gate",
        "round_1",
        "round_2",
        "judges",
        "ancestry",
        "decision",
        "cleanup",
    ]
    assert report["self_play_audit"]["valid"] is True
    expected = {member.value for member in MemberId}
    assert set(report["round_1"]["updated_members"]) == expected
    assert set(report["round_2"]["updated_members"]) == expected
    assert report["storage"]["raw_replays_written"] == 0
    assert report["storage"]["temp_run_exists_after_cleanup"] is False
    assert report["preflight"]["artifacts_before"] == report["artifacts_after"]
    assert _measure_tree(PROJECT / "artifacts") == artifact_before


def test_expired_deadline_rejects_and_cleans() -> None:
    code = """
import json
from pathlib import Path
from league_selfplay.contracts import FrozenLeagueConfig
from league_selfplay.runner import run_league
from league_selfplay.schedule import build_dry_run_schedule
import league_selfplay.runner as runner

times = iter((0.0, 1201.0, 1201.0))
runner.time.monotonic = lambda: next(times, 1201.0)
config = FrozenLeagueConfig(wall_time_seconds=1200)
report = run_league(config, build_dry_run_schedule(config), Path.cwd())
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

    assert report["decision"]["code"] == "REJECT_RUNTIME"
    assert report["storage"]["temp_run_exists_after_cleanup"] is False
