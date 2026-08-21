from __future__ import annotations

import json
import os
import shutil
import signal
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from scripts.run_local_match import import_official_cg

from .action_value import (
    ActionValueNet,
    ActionValueStats,
    action_value_parameter_count,
    create_action_value_population,
    export_action_value_member,
    update_action_values,
)
from .bootstrap import DriverRegistry
from .contracts import MemberId
from .residual import DriverBackedActor
from .residual_runner import (
    EXPLORATION_RATE,
    JUDGES,
    TEMP_QUOTA_BYTES,
    EvaluationRecord,
    ResidualComparison,
    ResidualProofDecision,
    ResidualReport,
    ResidualRuntimeExpired,
    ResidualStorageRecord,
    _collect_round,
    _compare,
    _counter_total,
    _driver_actors,
    _evaluate,
    _jsonable,
    decide_survivor_confirmation,
    measure_tree,
    select_promoted_members,
)
from .storage import RunStorage, install_cleanup_handlers


ACTION_VALUE_MARGIN = 0.35


def _stats(stats: Mapping[MemberId, ActionValueStats]) -> dict[str, Any]:
    return {member.value: _jsonable(value) for member, value in stats.items()}


def _retain_population(
    project_root: Path,
    population: Mapping[MemberId, ActionValueNet],
    promoted_members: Sequence[MemberId],
) -> str:
    destination = project_root / "agents/action_value_mac_pass"
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite passing population: {destination}")
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        for member in promoted_members:
            export_action_value_member(
                population[member],
                temporary / f"{member.value}.npz",
            )
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "decision": "PASS_ACTION_VALUE_MAC",
                    "exploration_rate": EXPLORATION_RATE,
                    "override_margin": ACTION_VALUE_MARGIN,
                    "override_mode": "action_value",
                    "promoted_members": [member.value for member in promoted_members],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(destination)


def run_action_value_proof(
    project_root: Path,
    *,
    seed: int = 20260804,
    judge_paths: Sequence[str] = JUDGES,
    wall_time_seconds: int = 360,
) -> ResidualReport:
    project_root = Path(project_root).resolve()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    started = time.monotonic()
    phases = ["preflight"]
    durations: dict[str, float] = {}
    artifacts_before = measure_tree(project_root / "artifacts")
    storage = RunStorage(
        Path(tempfile.gettempdir()).resolve()
        / f"pokemon-action-value-{uuid.uuid4().hex}",
        quota_bytes=TEMP_QUOTA_BYTES,
        shard_bytes=64 * 1024**2,
        max_pending=2,
    )
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    install_cleanup_handlers(storage)
    rounds: dict[str, Any] = {}
    candidate_records: tuple[EvaluationRecord, ...] = ()
    records: tuple[EvaluationRecord, ...] = ()
    candidate_comparison: ResidualComparison | None = None
    comparison: ResidualComparison | None = None
    promoted_members: tuple[MemberId, ...] = ()
    candidate_overrides = 0
    overrides = 0
    explorations = 0
    retained: str | None = None
    failures: list[str] = []
    final_updates: Mapping[MemberId, ActionValueStats] = {}
    decision = ResidualProofDecision(
        False,
        "REJECT_FAILURE",
        ("proof did not run",),
    )
    registry: DriverRegistry | None = None

    def deadline_check() -> None:
        if time.monotonic() - started > wall_time_seconds:
            raise ResidualRuntimeExpired(
                "action-value proof exceeded wall-time budget"
            )

    def begin(name: str) -> float:
        deadline_check()
        phases.append(name)
        return time.monotonic()

    def end(name: str, phase_started: float) -> None:
        durations[name] = time.monotonic() - phase_started
        deadline_check()

    try:
        game_api = import_official_cg(project_root)
        population = create_action_value_population(seed, device)
        rounds["model"] = {
            "parameters_per_member": action_value_parameter_count(
                next(iter(population.values()))
            ),
        }
        registry = DriverRegistry.from_project(project_root)
        actor_groups: list[Mapping[MemberId, DriverBackedActor]] = []

        phase_started = begin("round_1")
        round_one_actors = _driver_actors(
            population,
            registry,
            device,
            trainable=True,
            overrides_enabled=False,
            generation="action_value_round_0",
            override_mode="action_value",
            override_margin=ACTION_VALUE_MARGIN,
        )
        actor_groups.append(round_one_actors)
        round_one_games = _collect_round(
            game_api,
            round_one_actors,
            seed + 10,
            deadline_check,
        )
        round_one_updates = update_action_values(
            population,
            round_one_games,
            device,
            seed + 20,
            epochs=3,
        )
        rounds["round_1"] = {
            "games": len(round_one_games),
            "updates": _stats(round_one_updates),
        }
        end("round_1", phase_started)

        phase_started = begin("round_2")
        round_two_actors = _driver_actors(
            population,
            registry,
            device,
            trainable=True,
            overrides_enabled=False,
            generation="action_value_round_1",
            override_mode="action_value",
            override_margin=ACTION_VALUE_MARGIN,
        )
        actor_groups.append(round_two_actors)
        round_two_games = _collect_round(
            game_api,
            round_two_actors,
            seed + 30,
            deadline_check,
        )
        final_updates = update_action_values(
            population,
            round_one_games + round_two_games,
            device,
            seed + 40,
            epochs=3,
        )
        rounds["round_2"] = {
            "games": len(round_two_games),
            "updates": _stats(final_updates),
        }
        explorations = _counter_total(actor_groups, "exploration_actions")
        end("round_2", phase_started)

        registry.close()
        registry = None
        if device == "mps":
            torch.mps.synchronize()
        for model in population.values():
            model.to("cpu")
        torch.set_num_threads(1)

        phase_started = begin("candidate_judges")
        candidate_records, candidate_overrides = _evaluate(
            project_root,
            game_api,
            population,
            "cpu",
            judge_paths,
            tuple(MemberId),
            seed + 50,
            deadline_check,
            override_mode="action_value",
            override_margin=ACTION_VALUE_MARGIN,
        )
        candidate_comparison = _compare(candidate_records)
        promoted_members = select_promoted_members(
            candidate_comparison.per_member_delta
        )
        end("candidate_judges", phase_started)

        if promoted_members:
            phase_started = begin("confirmation_judges")
            records, overrides = _evaluate(
                project_root,
                game_api,
                population,
                "cpu",
                judge_paths,
                promoted_members,
                seed + 70,
                deadline_check,
                override_mode="action_value",
                override_margin=ACTION_VALUE_MARGIN,
            )
            comparison = _compare(records)
            end("confirmation_judges", phase_started)

        phase_started = begin("decision")
        decision = decide_survivor_confirmation(
            promoted_members,
            comparison.delta if comparison is not None else 0.0,
            overrides,
            final_updates,
            failures,
        )
        if decision.passed:
            decision = ResidualProofDecision(
                True,
                "PASS_ACTION_VALUE_MAC",
                (),
            )
            retained = _retain_population(
                project_root,
                population,
                promoted_members,
            )
        end("decision", phase_started)
    except ResidualRuntimeExpired as error:
        failures.append(str(error))
        decision = ResidualProofDecision(
            False,
            "REJECT_RUNTIME",
            tuple(failures),
        )
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        decision = ResidualProofDecision(
            False,
            "REJECT_FAILURE",
            tuple(failures),
        )
    finally:
        if registry is not None:
            registry.close()
        cleanup_started = time.monotonic()
        storage.cleanup()
        phases.append("cleanup")
        durations["cleanup"] = time.monotonic() - cleanup_started
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    return ResidualReport(
        phases=phases,
        phase_durations_seconds=durations,
        rounds=rounds,
        candidate_comparison=candidate_comparison,
        comparison=comparison,
        candidate_evaluation_records=candidate_records,
        evaluation_records=records,
        promoted_members=promoted_members,
        candidate_overrides=candidate_overrides,
        learned_overrides=overrides,
        training_explorations=explorations,
        decision=decision,
        failures=tuple(failures),
        storage=ResidualStorageRecord(
            root=str(storage.root),
            quota_bytes=storage.quota_bytes,
            bytes_written=storage.bytes_written,
            peak_temporary_bytes=storage.peak_bytes,
            raw_replays_written=0,
            temp_run_exists_after_cleanup=storage.root.exists(),
        ),
        artifacts_before=artifacts_before,
        artifacts_after=measure_tree(project_root / "artifacts"),
        retained_population=retained,
        device=device,
        seed=seed,
    )
