from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from league_selfplay.ppo import _mixture_log_probability_torch


PROJECT = Path(__file__).resolve().parents[2]
TEST_PYTHON = sys.executable


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ((1,), math.log(0.95)),
        ((0,), math.log(0.05)),
    ],
)
def test_torch_mixture_probability_matches_behavior_policy(
    action: tuple[int, ...],
    expected: float,
) -> None:
    value = _mixture_log_probability_torch(
        action,
        (1,),
        torch.tensor([0.0, 0.0]),
        torch.tensor(-20.0),
        min_count=1,
        max_count=1,
        exploration_rate=0.1,
    )

    assert float(value) == pytest.approx(expected, abs=1e-6)


def test_ppo_updates_all_current_members_without_mutating_history() -> None:
    code = """
import json
from league_selfplay.ppo import ppo_smoke

print(json.dumps(ppo_smoke(), sort_keys=True))
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

    assert report["updated_members"] == ["alakazam", "crustle", "grimmsnarl", "lucario"]
    assert all(delta > 0 for delta in report["parameter_delta_l2"].values())
    assert report["all_finite"] is True
    assert report["historical_snapshots_unchanged"] is True
    assert report["constant_negative_advantages"] == [-1.0] * 8
    assert report["hand_gae"] == pytest.approx([0.9025, 0.95, 1.0])
