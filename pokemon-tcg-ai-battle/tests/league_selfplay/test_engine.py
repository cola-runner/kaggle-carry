from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
TEST_PYTHON = sys.executable


def test_real_engine_records_both_trainable_players_from_both_seats(
    official_cg: Path,
) -> None:
    code = """
import json
from pathlib import Path
from league_selfplay.engine import engine_smoke

print(json.dumps(engine_smoke(Path.cwd()), sort_keys=True))
"""
    result = subprocess.run(
        [TEST_PYTHON, "-c", code],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    report = json.loads(result.stdout)

    assert report["games"] == 2
    assert report["finished_games"] == 2
    assert report["seat_orders"] == [
        ["grimmsnarl", "lucario"],
        ["lucario", "grimmsnarl"],
    ]
    assert report["trajectory_members"] == [
        ["grimmsnarl", "lucario"],
        ["grimmsnarl", "lucario"],
    ]
    assert report["total_steps"] > 0
    assert report["all_features_float32"] is True
    assert report["all_log_probabilities_finite"] is True
    assert report["reward_values_are_terminal"] is True
    assert report["one_terminal_reward_per_member"] is True
    assert report["illegal_actions"] == 0
