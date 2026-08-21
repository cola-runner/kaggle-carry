from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
TEST_PYTHON = sys.executable


def test_bootstrap_changes_all_models_then_permanently_closes_teachers(
    official_cg: Path,
) -> None:
    code = """
import json
from pathlib import Path
from league_selfplay.bootstrap import bootstrap_smoke

print(json.dumps(bootstrap_smoke(Path.cwd()), sort_keys=True))
"""
    result = subprocess.run(
        [TEST_PYTHON, "-c", code],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=True,
        timeout=240,
    )
    report = json.loads(result.stdout)

    assert report["driver_game_finished"] is True
    assert report["driver_game_members"] == ["grimmsnarl", "lucario"]
    assert report["updated_members"] == ["alakazam", "crustle", "grimmsnarl", "lucario"]
    assert all(delta > 0 for delta in report["parameter_delta_l2"].values())
    assert report["all_finite"] is True
    assert report["start_gate_games"] == 2
    assert report["teacher_closed_before_start_gate"] is True
    assert report["teacher_call_after_close_rejected"] is True
