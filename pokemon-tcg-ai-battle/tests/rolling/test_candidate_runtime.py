from __future__ import annotations

import csv
import hashlib
import importlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from rolling_policy.constants import EXACT_DECK_FINGERPRINT, deck_fingerprint


PROJECT = Path(__file__).resolve().parents[2]
AGENT = PROJECT / "agents" / "candidate_grimmsnarl_imitation_full_v2"
FIXTURE = Path(__file__).parent / "fixtures" / "visible_observation.json"
MODEL_HASHES = {
    "models/main_v1.json": "c48ebe9d8773ac678a02c0f1811e5746e740e930a8fc6b87df93f6e28599cddc",
    "models/main_v2.json": "4b59938ae8ea9cde7d22545e8ebfb040e5adb7588afd4e32aa1b589963ba3c7f",
    "models/followup_v1.json": "efd6e21b96d0d4402171e4ddfb288f65678744c905cc31a6dba3c62eb7bff489",
    "models/followup_v2.json": "6841940d0a89694806c22bf1aba6aa3669659a54ed80772c95da8f1d82c740f1",
}


def _runtime_modules():
    sys.path.insert(0, str(AGENT))
    try:
        policy = importlib.import_module("policy_runtime")
        candidate_features = importlib.import_module("runtime.features")
        main = importlib.import_module("main")
        return policy, candidate_features, main
    finally:
        sys.path.remove(str(AGENT))


def test_candidate_uses_exact_frozen_deck() -> None:
    rows = list(csv.reader((AGENT / "deck.csv").open()))
    deck = [int(row[0]) for row in rows]
    assert len(deck) == 60
    assert deck_fingerprint(deck) == EXACT_DECK_FINGERPRINT
    policy, _, main = _runtime_modules()
    assert tuple(deck) == policy.DECK
    assert main.agent({"select": None}) == deck


def test_frozen_v2_candidate_keeps_its_original_feature_schema() -> None:
    policy, candidate_features, _ = _runtime_modules()
    assert candidate_features.FEATURE_SCHEMA_VERSION == 1
    assert len(policy._models()) == 4


def test_fallback_promote_preserves_the_more_invested_pokemon() -> None:
    policy, _, _ = _runtime_modules()
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "bench": [
                        {"id": 999, "hp": 200, "energyCards": []},
                        {"id": 999, "hp": 170, "energyCards": [{"id": 7}]},
                    ]
                },
                {},
            ],
        },
        "select": {
            "context": 4,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 3, "playerIndex": 0, "area": 5, "index": 0},
                {"type": 3, "playerIndex": 0, "area": 5, "index": 1},
            ],
        },
    }

    assert policy.fallback_action(observation) == [1]


def test_candidate_returns_legal_fixture_action() -> None:
    policy, _, main = _runtime_modules()
    raw = json.loads(FIXTURE.read_text())
    action = main.agent(raw)
    select = raw["select"]
    assert select["minCount"] <= len(action) <= select["maxCount"]
    assert len(action) == len(set(action))
    assert all(0 <= index < len(select["option"]) for index in action)
    full_action = policy.choose_action_full_main_clone(raw)
    assert select["minCount"] <= len(full_action) <= select["maxCount"]
    assert all(0 <= index < len(select["option"]) for index in full_action)


def test_candidate_model_files_match_frozen_hashes() -> None:
    for relative, expected in MODEL_HASHES.items():
        assert hashlib.sha256((AGENT / relative).read_bytes()).hexdigest() == expected


def test_submission_package_contains_standalone_runtime(tmp_path: Path) -> None:
    output = tmp_path / "candidate.tar.gz"
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
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert set(MODEL_HASHES).issubset(names)
    assert "identity.json" in names
    assert {
        "runtime/__init__.py",
        "runtime/extract.py",
        "runtime/features.py",
        "runtime/hashing.py",
        "runtime/schema.py",
        "runtime/tree_predict.py",
    }.issubset(names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
