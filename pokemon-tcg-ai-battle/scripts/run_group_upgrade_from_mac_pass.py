from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from league_selfplay import single_intervention_runner as runner  # noqa: E402
from league_selfplay.action_value import ActionValueNet, export_action_value_member  # noqa: E402
from league_selfplay.contracts import MemberId  # noqa: E402
from league_selfplay.features import INPUT_WIDTH  # noqa: E402
from league_selfplay.single_intervention import InterventionEnsemble  # noqa: E402


SOURCE_NAME = "single_intervention_mac_pass"
DESTINATION_NAME = "group_upgrade_v2_mac_pass"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path: Path, device: str) -> ActionValueNet:
    model = ActionValueNet().to(device)
    with np.load(path, allow_pickle=False) as checkpoint, torch.no_grad():
        model.layer.weight.copy_(
            torch.from_numpy(np.asarray(checkpoint["w"], dtype=np.float32).T).to(device)
        )
        model.layer.bias.copy_(
            torch.from_numpy(np.asarray(checkpoint["b"], dtype=np.float32)).to(device)
        )
        model.option_head.weight.copy_(
            torch.from_numpy(
                np.asarray(checkpoint["w_option"], dtype=np.float32)[None, :]
            ).to(device)
        )
        model.option_head.bias.copy_(
            torch.from_numpy(np.asarray(checkpoint["b_option"], dtype=np.float32)).to(device)
        )
    return model


def load_mac_pass_population(
    project_root: Path,
    device: str,
) -> tuple[
    dict[MemberId, InterventionEnsemble],
    dict[MemberId, tuple[float, float]],
    dict[str, str],
]:
    source = project_root / "agents" / SOURCE_NAME
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("members") != [member.value for member in MemberId]:
        raise ValueError("source Mac-pass population does not contain all four members")
    population: dict[MemberId, InterventionEnsemble] = {}
    margins: dict[MemberId, tuple[float, float]] = {}
    hashes = {"manifest.json": _sha256(manifest_path)}
    for member in MemberId:
        models = []
        for model_index in range(2):
            filename = f"{member.value}-{model_index}.npz"
            path = source / filename
            models.append(_load_model(path, device))
            hashes[filename] = _sha256(path)
        population[member] = InterventionEnsemble((models[0], models[1]))
        values = manifest["override_margins"][member.value]
        margins[member] = (float(values[0]), float(values[1]))
    return population, margins, hashes


def _strict_group_decision(
    *,
    comparison: runner.ActualComparison | None,
    overrides: int,
    expected_games_per_side: int,
    failures: Sequence[str],
) -> runner.SingleInterventionDecision:
    base = ORIGINAL_DECIDE(
        comparison=comparison,
        overrides=overrides,
        expected_games_per_side=expected_games_per_side,
        failures=failures,
    )
    if not base.passed or comparison is None:
        return base
    regressing = [
        member.value
        for member in MemberId
        if float(comparison.per_member_delta.get(member, float("-inf"))) < -0.05
    ]
    if regressing:
        return runner.SingleInterventionDecision(
            False,
            "REJECT_MEMBER_REGRESSION",
            ("members regressed by more than 0.05: " + ", ".join(regressing),),
        )
    return runner.SingleInterventionDecision(True, "PASS_GROUP_UPGRADE_V2_MAC", ())


def _retainer(
    project_root: Path,
    storage: runner.RunStorage,
    population: Mapping[MemberId, InterventionEnsemble],
    promoted: Sequence[MemberId],
    margins: Mapping[MemberId, tuple[float, float]],
) -> str:
    staged = storage.root / "group-upgrade-v2"
    staged.mkdir()
    for member in promoted:
        for model_index, model in enumerate(population[member].models):
            export_action_value_member(
                model,
                str(staged / f"{member.value}-{model_index}.npz"),
            )
    manifest = {
        "format": "single-intervention-action-value-v2",
        "input_width": INPUT_WIDTH,
        "members": [member.value for member in promoted],
        "models_per_member": 2,
        "override_margins": {
            member.value: list(margins[member]) for member in promoted
        },
        "parent": SOURCE_NAME,
        "parent_sha256": PARENT_HASHES,
        "seed": RUN_SEED,
    }
    (staged / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination = project_root / "agents" / DESTINATION_NAME
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing {destination}")
    os.replace(staged, destination)
    return str(destination)


ORIGINAL_DECIDE = runner.decide_group_upgrade_proof
ORIGINAL_RETAIN = runner._retain_population
INITIAL_POPULATION: dict[MemberId, InterventionEnsemble]
INITIAL_MARGINS: dict[MemberId, tuple[float, float]]
PARENT_HASHES: dict[str, str]
RUN_SEED = 20260807


def main() -> None:
    global INITIAL_POPULATION, INITIAL_MARGINS, PARENT_HASHES, RUN_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "reports/mac_group_upgrade_v2_latest.json",
    )
    parser.add_argument("--wall-time-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=RUN_SEED)
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    RUN_SEED = int(args.seed)
    INITIAL_POPULATION, INITIAL_MARGINS, PARENT_HASHES = load_mac_pass_population(
        project_root,
        "cpu",
    )

    runner.decide_group_upgrade_proof = _strict_group_decision
    runner._retain_population = _retainer
    try:
        report = runner.run_single_intervention_proof(
            project_root,
            wall_time_seconds=args.wall_time_seconds,
            seed=RUN_SEED,
            initial_population=INITIAL_POPULATION,
            initial_margins=INITIAL_MARGINS,
        )
        runner.write_report_atomic(report, args.out)
        print(json.dumps(report.summary(), indent=2, sort_keys=True))
    finally:
        runner.decide_group_upgrade_proof = ORIGINAL_DECIDE
        runner._retain_population = ORIGINAL_RETAIN


if __name__ == "__main__":
    main()
