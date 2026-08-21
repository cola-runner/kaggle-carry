from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[2]
AGENT = PROJECT / "agents" / "candidate_grimmsnarl_nn_1p5m_smoke_v0"
FIXTURE = PROJECT / "tests" / "rolling" / "fixtures" / "visible_observation.json"
EXPECTED_PARAMETERS = 1_477_441


def _load_main():
    main_py = AGENT / "main.py"
    if not main_py.exists():
        pytest.fail(f"missing smoke submission entrypoint: {main_py}")
    sys.path.insert(0, str(AGENT))
    try:
        spec = importlib.util.spec_from_file_location("nn_1p5m_smoke_main", main_py)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(AGENT))


def test_model_has_real_1p5m_float_parameters() -> None:
    model_path = AGENT / "model.npz"
    assert model_path.exists(), "the submission must contain a real NumPy checkpoint"
    with np.load(model_path, allow_pickle=False) as model:
        arrays = [model[name] for name in ("w1", "b1", "w2", "b2", "wout", "bout")]
    assert all(array.dtype == np.float32 for array in arrays)
    assert sum(array.size for array in arrays) == EXPECTED_PARAMETERS
    assert model_path.stat().st_size < 8 * 1024**2


def test_agent_imports_and_acts_when_torch_and_sklearn_are_blocked() -> None:
    code = f"""
import builtins, importlib.util, json, pathlib, sys
agent_dir = pathlib.Path({str(AGENT)!r})
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {{'torch', 'sklearn'}}:
        raise AssertionError('forbidden runtime dependency: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, str(agent_dir))
spec = importlib.util.spec_from_file_location('smoke_main', agent_dir / 'main.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
deck = module.agent({{'select': None}})
obs = json.loads(pathlib.Path({str(FIXTURE)!r}).read_text())
action = module.agent(obs)
select = obs['select']
assert len(deck) == 60
assert select['minCount'] <= len(action) <= select['maxCount']
assert len(action) == len(set(action))
assert all(0 <= index < len(select['option']) for index in action)
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=PROJECT)


def test_sixty_option_inference_is_below_one_second() -> None:
    main = _load_main()
    observation = json.loads(FIXTURE.read_text())
    options = observation["select"]["option"]
    observation["select"]["option"] = [dict(options[index % len(options)]) for index in range(60)]
    observation["select"]["minCount"] = 1
    observation["select"]["maxCount"] = 1

    main.agent(observation)
    durations = []
    for _ in range(5):
        started = time.perf_counter()
        action = main.agent(observation)
        durations.append(time.perf_counter() - started)
        assert len(action) == 1
        assert 0 <= action[0] < 60
    assert statistics.median(durations) < 1.0


def test_smoke_submission_preserves_incumbent_fixture_action() -> None:
    def action_from(agent_dir: Path) -> list[int]:
        code = f"""
import importlib.util, json, pathlib, sys
agent_dir = pathlib.Path({str(agent_dir)!r})
sys.path.insert(0, str(agent_dir))
spec = importlib.util.spec_from_file_location('isolated_agent', agent_dir / 'main.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
obs = json.loads(pathlib.Path({str(FIXTURE)!r}).read_text())
print(json.dumps(module.agent(obs)))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=PROJECT,
            text=True,
            capture_output=True,
        )
        return json.loads(result.stdout)

    incumbent = PROJECT / "agents" / "candidate_grimmsnarl_imitation_full_v2"
    assert action_from(AGENT) == action_from(incumbent)


def test_submission_package_is_standalone_and_below_limit(tmp_path: Path) -> None:
    output = tmp_path / "candidate-grimmsnarl-nn-1p5m-smoke-v0.tar.gz"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts" / "package_submission.py"),
            "--agent-dir",
            str(AGENT),
            "--output",
            str(output),
        ],
        check=True,
        cwd=PROJECT,
    )
    assert output.stat().st_size < int(197.7 * 1024**2)
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert {"main.py", "deck.csv", "policy_runtime.py", "model.npz"}.issubset(names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_submission_finishes_kaggle_style_self_validation(
    official_cg: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts" / "run_local_match.py"),
            "--agent0",
            str(AGENT),
            "--agent1",
            str(AGENT),
            "--max-steps",
            "2000",
        ],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "outcome=" in result.stdout
