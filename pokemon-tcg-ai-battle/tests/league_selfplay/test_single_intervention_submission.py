from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from league_selfplay.action_value import ActionValueNet
from league_selfplay.features import encode_options
from league_selfplay.single_intervention import trusted_override


PROJECT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT / "tests/rolling/fixtures/visible_observation.json"
CANDIDATES = {
    "alakazam": PROJECT / "agents/candidate_alakazam_single_intervention_v1",
    "grimmsnarl": PROJECT / "agents/candidate_grimmsnarl_single_intervention_v1",
}
GRIMMSNARL_CONTROL = (
    PROJECT / "agents/candidate_grimmsnarl_single_intervention_control_v1"
)


def _load_runtime(member: str):
    path = CANDIDATES[member] / "residual_runtime.py"
    assert path.is_file(), f"missing submission runtime: {path}"
    for name in tuple(sys.modules):
        if (
            name == "runtime"
            or name.startswith("runtime.")
            or name in {"si_alakazam", "si_grimmsnarl"}
            or name.startswith(("si_alakazam.", "si_grimmsnarl."))
        ):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(CANDIDATES[member]))
    try:
        spec = importlib.util.spec_from_file_location(
            f"{member}_single_intervention_runtime",
            path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(CANDIDATES[member]))


def _torch_scores(member: str, model_index: int, features: np.ndarray) -> np.ndarray:
    source = (
        PROJECT
        / "agents/single_intervention_mac_pass"
        / f"{member}-{model_index}.npz"
    )
    with np.load(source, allow_pickle=False) as checkpoint:
        model = ActionValueNet()
        model.layer.weight.data.copy_(torch.from_numpy(checkpoint["w"].T.copy()))
        model.layer.bias.data.copy_(torch.from_numpy(checkpoint["b"].copy()))
        model.option_head.weight.data.copy_(
            torch.from_numpy(checkpoint["w_option"][None, :].copy())
        )
        model.option_head.bias.data.copy_(
            torch.from_numpy(checkpoint["b_option"].copy())
        )
    with torch.no_grad():
        values, _, _ = model(
            torch.from_numpy(features)[None],
            torch.ones((1, len(features)), dtype=torch.bool),
        )
    return values[0].numpy()


def test_submission_numpy_scores_and_gate_match_training_runtime() -> None:
    observation = json.loads(FIXTURE.read_text())
    features = encode_options(observation)
    manifest = json.loads(
        (PROJECT / "agents/single_intervention_mac_pass/manifest.json").read_text()
    )
    for member in CANDIDATES:
        runtime = _load_runtime(member)
        actual_rows = runtime.option_scores(observation)
        expected_rows = tuple(
            _torch_scores(member, model_index, features)
            for model_index in range(2)
        )
        for actual, expected in zip(actual_rows, expected_rows, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

        incumbent = [0]
        expected_override = trusted_override(
            expected_rows,
            incumbent[0],
            margin=manifest["override_margins"][member],
        )
        expected_action = incumbent if expected_override is None else [expected_override]
        assert runtime.choose_action(observation, incumbent) == expected_action


def test_candidate_agents_run_without_training_dependencies(
    official_cg: Path,
) -> None:
    code = f"""
import builtins, importlib.util, json, os, pathlib, sys
project = pathlib.Path({str(PROJECT)!r})
fixture = json.loads(pathlib.Path({str(FIXTURE)!r}).read_text())
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {{'torch', 'sklearn'}}:
        raise AssertionError('forbidden runtime dependency: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, str(project / 'data/raw/pokemon-tcg-ai-battle/sample_submission/sample_submission'))
for member in ('alakazam', 'grimmsnarl'):
    agent_dir = project / 'agents' / f'candidate_{{member}}_single_intervention_v1'
    for module_name in tuple(sys.modules):
        if module_name in {{'main', 'baseline_policy', 'policy_runtime', 'residual_runtime', 'runtime', 'si_alakazam', 'si_grimmsnarl'}} or module_name.startswith(('runtime.', 'si_alakazam.', 'si_grimmsnarl.')):
            sys.modules.pop(module_name, None)
    sys.path.insert(0, str(agent_dir))
    old = pathlib.Path.cwd()
    os.chdir(agent_dir)
    try:
        spec = importlib.util.spec_from_file_location(member + '_candidate', agent_dir / 'main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        deck = module.agent({{'select': None}})
        action = module.agent(fixture)
        select = fixture['select']
        assert len(deck) == 60
        assert select['minCount'] <= len(action) <= select['maxCount']
        assert len(action) == len(set(action))
        assert all(0 <= index < len(select['option']) for index in action)
    finally:
        os.chdir(old)
        sys.path.remove(str(agent_dir))
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT,
        check=True,
        timeout=90,
    )


def test_candidate_residual_inference_is_fast() -> None:
    observation = json.loads(FIXTURE.read_text())
    for member in CANDIDATES:
        runtime = _load_runtime(member)
        runtime.choose_action(observation, [0])
        started = time.perf_counter()
        for _ in range(20):
            action = runtime.choose_action(observation, [0])
            assert len(action) == 1
        average = (time.perf_counter() - started) / 20
        assert average < 0.05


def test_candidate_archives_are_standalone_and_small(
    tmp_path: Path,
    official_cg: Path,
) -> None:
    for member, agent_dir in CANDIDATES.items():
        output = tmp_path / f"{member}-single-intervention-v1.tar.gz"
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "scripts/package_submission.py"),
                "--agent-dir",
                str(agent_dir),
                "--output",
                str(output),
            ],
            cwd=PROJECT,
            check=True,
            timeout=90,
        )
        assert output.stat().st_size < 10 * 1024**2
        with tarfile.open(output, "r:gz") as archive:
            names = set(archive.getnames())
        required = {
            "main.py",
            "baseline_policy.py",
            "residual_runtime.py",
            "deck.csv",
            "models/single_intervention_0.npz",
            "models/single_intervention_1.npz",
            "models/single_intervention_manifest.json",
            f"si_{member}/residual_encoder.py",
            f"si_{member}/residual_features.py",
        }
        assert required <= names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
        if member == "alakazam":
            assert "cg/game.py" in names


def test_grimmsnarl_control_exactly_matches_the_frozen_incumbent() -> None:
    control_main = GRIMMSNARL_CONTROL / "main.py"
    assert control_main.is_file(), "missing same-time untrained control candidate"
    code = f"""
import importlib.util, json, os, pathlib, sys
project = pathlib.Path({str(PROJECT)!r})
fixture = json.loads(pathlib.Path({str(FIXTURE)!r}).read_text())
trained_dir = project / 'agents/candidate_grimmsnarl_single_intervention_v1'
control_dir = pathlib.Path({str(GRIMMSNARL_CONTROL)!r})

def load(path, name, root):
    for module_name in tuple(sys.modules):
        if module_name in {{'baseline_policy', 'policy_runtime', 'runtime'}} or module_name.startswith('runtime.'):
            sys.modules.pop(module_name, None)
    sys.path.insert(0, str(root))
    old = pathlib.Path.cwd()
    os.chdir(root)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old)
        sys.path.remove(str(root))

incumbent = load(trained_dir / 'baseline_policy.py', 'frozen_incumbent', trained_dir)
control = load(control_dir / 'main.py', 'same_time_control', control_dir)
assert control.agent({{'select': None}}) == incumbent.agent({{'select': None}})
assert control.agent(fixture) == incumbent.agent(fixture)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT,
        check=True,
        timeout=90,
    )


def test_grimmsnarl_control_archive_contains_no_trained_override(
    tmp_path: Path,
) -> None:
    output = tmp_path / "grimmsnarl-single-intervention-control-v1.tar.gz"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/package_submission.py"),
            "--agent-dir",
            str(GRIMMSNARL_CONTROL),
            "--output",
            str(output),
        ],
        cwd=PROJECT,
        check=True,
        timeout=90,
    )
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert {
        "main.py",
        "policy_runtime.py",
        "deck.csv",
        "identity.json",
        "runtime/tree_predict.py",
        "models/main_v2.json",
    } <= names
    assert not any(
        "single_intervention" in name
        or "residual" in name
        or name.startswith("si_")
        for name in names
    )
