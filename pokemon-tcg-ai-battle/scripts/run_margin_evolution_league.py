from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from league_selfplay.bootstrap import DriverRegistry  # noqa: E402
from league_selfplay.contracts import MemberId  # noqa: E402
from league_selfplay.evolution import (  # noqa: E402
    MarginGenome,
    combine_comparisons,
    compare_to_frozen_champion,
    evaluate_challenger_score,
    evolution_passes,
    neutralize_frozen_members,
    spawn_generation,
)
from league_selfplay.single_intervention_runner import (  # noqa: E402
    _jsonable,
    import_official_cg,
    measure_tree,
)
from scripts.run_group_upgrade_from_mac_pass import (  # noqa: E402
    SOURCE_NAME,
    load_mac_pass_population,
)


DESTINATION_NAME = "margin_evolution_mac_pass"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(generation: int, index: int, genome: MarginGenome, score: Any) -> dict[str, Any]:
    return {
        "generation": generation,
        "candidate": index,
        "genome": genome.to_dict(),
        "score": score.score,
        "per_member_score": _jsonable(score.per_member_score),
        "games": score.games,
        "focal_overrides": score.focal_overrides,
    }


def _retain(
    project_root: Path,
    genome: MarginGenome,
    champion_margins: dict[MemberId, tuple[float, float]],
    report_path: Path,
) -> str:
    source = project_root / "agents" / SOURCE_NAME
    destination = project_root / "agents" / DESTINATION_NAME
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing {destination}")
    temporary = Path(tempfile.gettempdir()) / f"pokemon-margin-evolution-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for member in MemberId:
            for model_index in range(2):
                filename = f"{member.value}-{model_index}.npz"
                shutil.copy2(source / filename, temporary / filename)
        manifest = {
            "format": "single-intervention-action-value-v1-margin-evolution",
            "parent": SOURCE_NAME,
            "parent_manifest_sha256": _sha256(source / "manifest.json"),
            "members": [member.value for member in MemberId],
            "models_per_member": 2,
            "override_margins": _jsonable(genome.margins(champion_margins)),
            "proof_report": str(report_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return str(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "reports/mac_margin_evolution_latest.json",
    )
    parser.add_argument("--seed", type=int, default=2026080702)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--screening-repetitions", type=int, default=1)
    parser.add_argument("--confirmation-repetitions", type=int, default=3)
    parser.add_argument("--wall-time-seconds", type=int, default=720)
    parser.add_argument("--resume-genome-report", type=Path)
    parser.add_argument(
        "--freeze-member",
        action="append",
        default=[],
        choices=[member.value for member in MemberId],
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    report_path = args.out.expanduser().resolve()
    started = time.monotonic()
    deadline = started + args.wall_time_seconds
    artifacts_before = measure_tree(project_root / "artifacts")
    population, champion_margins, parent_hashes = load_mac_pass_population(
        project_root,
        "cpu",
    )
    game_api = import_official_cg(project_root)
    registry = DriverRegistry.from_project(project_root)
    rng = np.random.default_rng(args.seed)
    parent = MarginGenome.champion()
    hall: tuple[MarginGenome, float] = (parent, 0.5)
    frozen_members = tuple(MemberId(value) for value in args.freeze_member)
    resume_report = None
    if args.resume_genome_report is not None:
        resume_path = args.resume_genome_report.expanduser().resolve()
        resume_report = json.loads(resume_path.read_text(encoding="utf-8"))
        parent = MarginGenome.from_dict(resume_report["selected_genome"]).freeze(
            frozen_members
        )
        hall = (parent, float(resume_report["selected_screening_score"]))
    generations: list[list[dict[str, Any]]] = []
    failures: list[str] = []
    first_confirmation = None
    second_confirmation = None
    combined = None
    retained = None
    code = "REJECT_FAILURE"
    passed = False
    try:
        generation_count = 0 if resume_report is not None else args.generations
        for generation_index in range(1, generation_count + 1):
            candidates = spawn_generation(
                parent,
                rng,
                size=args.population_size,
            )
            rows: list[dict[str, Any]] = []
            scored: list[tuple[MarginGenome, float]] = []
            for candidate_index, genome in enumerate(candidates):
                score = evaluate_challenger_score(
                    game_api,
                    registry,
                    population,
                    champion_margins,
                    genome.margins(champion_margins),
                    repetitions=args.screening_repetitions,
                    rng=rng,
                    deadline=deadline,
                )
                rows.append(_entry(generation_index, candidate_index, genome, score))
                scored.append((genome, score.score))
            generations.append(rows)
            generation_winner = max(scored, key=lambda item: item[1])
            if generation_winner[1] > hall[1]:
                hall = generation_winner
            parent = hall[0]

        challenger = hall[0]
        challenger_margins = challenger.margins(champion_margins)
        first_confirmation = compare_to_frozen_champion(
            game_api,
            registry,
            population,
            champion_margins,
            challenger_margins,
            repetitions=args.confirmation_repetitions,
            rng=np.random.default_rng(args.seed + 100_001),
            deadline=deadline,
        )
        second_confirmation = compare_to_frozen_champion(
            game_api,
            registry,
            population,
            champion_margins,
            challenger_margins,
            repetitions=args.confirmation_repetitions,
            rng=np.random.default_rng(args.seed + 200_001),
            deadline=deadline,
        )
        if frozen_members:
            first_confirmation = neutralize_frozen_members(
                first_confirmation,
                frozen_members,
            )
            second_confirmation = neutralize_frozen_members(
                second_confirmation,
                frozen_members,
            )
        combined = combine_comparisons(first_confirmation, second_confirmation)
        passed, code = evolution_passes(
            first_confirmation,
            second_confirmation,
            combined,
        )
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        code = "REJECT_RUNTIME" if isinstance(error, TimeoutError) else "REJECT_FAILURE"
    finally:
        registry.close()

    payload: dict[str, Any] = {
        "decision": {"passed": passed, "code": code, "failures": failures},
        "seed": args.seed,
        "champion": SOURCE_NAME,
        "champion_hashes": parent_hashes,
        "method": "frozen-champion whole-game margin evolution",
        "resumed_from": (
            None
            if args.resume_genome_report is None
            else str(args.resume_genome_report.expanduser().resolve())
        ),
        "frozen_members": [member.value for member in frozen_members],
        "raw_replays_written": 0,
        "generations": generations,
        "selected_genome": hall[0].to_dict(),
        "selected_screening_score": hall[1],
        "confirmation_first": _jsonable(first_confirmation),
        "confirmation_second": _jsonable(second_confirmation),
        "confirmation_combined": _jsonable(combined),
        "elapsed_seconds": time.monotonic() - started,
        "artifacts_before": _jsonable(artifacts_before),
        "artifacts_after": _jsonable(measure_tree(project_root / "artifacts")),
        "retained_population": None,
    }
    _write_atomic(report_path, payload)
    if passed:
        retained = _retain(
            project_root,
            hall[0],
            champion_margins,
            report_path,
        )
        payload["retained_population"] = retained
        _write_atomic(report_path, payload)
    print(
        json.dumps(
            {
                "decision": code,
                "passed": passed,
                "screening_score": hall[1],
                "combined_delta": None if combined is None else combined.delta,
                "per_member_delta": (
                    None if combined is None else _jsonable(combined.per_member_delta)
                ),
                "retained_population": retained,
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
