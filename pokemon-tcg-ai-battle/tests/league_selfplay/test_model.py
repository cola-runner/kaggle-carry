from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from league_selfplay.features import INPUT_WIDTH, encode_options


PROJECT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT / "tests/rolling/fixtures/visible_observation.json"
TEST_PYTHON = sys.executable


def _observation() -> dict:
    return json.loads(FIXTURE.read_text())


def test_option_encoding_has_one_float32_row_per_legal_option() -> None:
    observation = _observation()

    features = encode_options(observation)

    assert features.dtype == np.float32
    assert features.shape == (len(observation["select"]["option"]), INPUT_WIDTH)
    assert np.isfinite(features).all()


def test_option_encoding_ignores_hidden_payloads_and_opponent_hand() -> None:
    observation = _observation()
    poisoned = copy.deepcopy(observation)
    acting_seat = int(poisoned["current"]["yourIndex"])
    opponent = 1 - acting_seat
    poisoned["search_begin_input"] = {"deckOrder": [999], "opponentHand": [998]}
    poisoned["current"]["players"][opponent]["hand"] = [{"id": 997}]
    poisoned["current"]["players"][opponent]["deck"] = [{"id": 996}]

    np.testing.assert_array_equal(encode_options(observation), encode_options(poisoned))


def test_four_models_are_independent_approximately_1p5m_and_export_exactly(tmp_path: Path) -> None:
    code = f"""
import json
from pathlib import Path
from league_selfplay.model import parity_smoke

result = parity_smoke(Path({str(tmp_path)!r}))
print(json.dumps(result, sort_keys=True))
"""
    result = subprocess.run(
        [TEST_PYTHON, "-c", code],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    report = json.loads(result.stdout)

    assert report["members"] == ["alakazam", "crustle", "grimmsnarl", "lucario"]
    assert report["independent_parameter_storage"] is True
    assert report["policy_parameter_count"] == 1_478_370
    assert report["max_option_logit_error"] < 1e-4
    assert report["max_stop_logit_error"] < 1e-4
    assert report["all_finite"] is True
    assert list(tmp_path.iterdir()) == []
