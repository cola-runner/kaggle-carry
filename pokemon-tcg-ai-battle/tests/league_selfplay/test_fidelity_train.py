from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from league_selfplay.actions import greedy_action


PROJECT = Path(__file__).resolve().parents[2]
TEST_PYTHON = sys.executable


def test_greedy_action_obeys_stop_and_selection_bounds() -> None:
    assert greedy_action(np.asarray([3.0, 1.0]), -5.0, 1, 1) == (0,)
    assert greedy_action(np.asarray([1.0, 0.0]), 2.0, 0, 2) == ()
    assert greedy_action(np.asarray([3.0, 2.0, 1.0]), 2.5, 1, 3) == (0,)


def test_metrics_ignore_forced_choices_and_score_real_policy_outputs() -> None:
    code = """
import json
import numpy as np
import torch
from league_selfplay.contracts import MemberId
from league_selfplay.fidelity_data import PairedDecision, PairedGame
from league_selfplay.fidelity_train import evaluate_population
from league_selfplay.model import create_population

population = create_population(7, "cpu")
for model in population.values():
    for parameter in model.parameters():
        parameter.data.zero_()
    model.stop_head.bias.data.fill_(-10.0)

games = []
game_id = 0
for member in MemberId:
    decisions = []
    for offset in range(2):
        features = np.zeros((2, 512), dtype=np.float32)
        decisions.append(PairedDecision(member, game_id, features, features, (0,), 1, 1))
    forced = np.zeros((1, 512), dtype=np.float32)
    decisions.append(PairedDecision(member, game_id, forced, forced, (0,), 1, 1))
    games.append(PairedGame(game_id, (member, member), tuple(decisions), True))
    game_id += 1

metrics = evaluate_population(population, games, "v2", "cpu")
print(json.dumps({
    "nontrivial": metrics.nontrivial_decisions,
    "forced": metrics.forced_decisions,
    "agreement": metrics.exact_agreement,
    "nll": metrics.negative_log_probability,
    "members": sorted(member.value for member in metrics.members),
}, sort_keys=True))
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

    assert report["nontrivial"] == 8
    assert report["forced"] == 4
    assert report["agreement"] == 1.0
    assert report["nll"] == pytest.approx(np.log(2.0), abs=1e-5)
    assert report["members"] == ["alakazam", "crustle", "grimmsnarl", "lucario"]


def test_v2_features_have_exact_torch_numpy_policy_parity() -> None:
    code = """
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from league_selfplay.features import encode_options_v2
from league_selfplay.model import create_population, export_member
from league_selfplay.numpy_runtime import load_policy, numpy_forward
from league_selfplay.contracts import MemberId

observation = json.loads(Path("tests/rolling/fixtures/visible_observation.json").read_text())
features = encode_options_v2(observation)
model = create_population(11, "cpu")[MemberId.GRIMMSNARL]
with torch.no_grad():
    tensor = torch.from_numpy(features)[None, :, :]
    mask = torch.ones((1, len(features)), dtype=torch.bool)
    torch_options, torch_stop, _ = model(tensor, mask)
with tempfile.TemporaryDirectory(prefix="fidelity-parity-") as temporary:
    path = Path(temporary) / "member.npz"
    export_member(model, path)
    numpy_options, numpy_stop = numpy_forward(features, load_policy(path))
print(json.dumps({
    "option_error": float(np.max(np.abs(torch_options[0].numpy() - numpy_options))),
    "stop_error": abs(float(torch_stop[0]) - numpy_stop),
}, sort_keys=True))
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

    assert report["option_error"] < 1e-4
    assert report["stop_error"] < 1e-4
