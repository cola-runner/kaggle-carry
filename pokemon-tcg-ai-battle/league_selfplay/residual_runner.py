from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import signal
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.run_local_match import (
    first_result_deck,
    import_official_cg,
    load_agent,
    pushd,
    validate_action,
)

from .bootstrap import DriverRegistry
from .contracts import GameSource, MemberId, audit_training_batch
from .engine import CompletedGame, run_training_game
from .model import PolicyValueNet, create_population, export_member
from .ppo import PPOStats, update_population
from .residual import DriverBackedActor
from .schedule import JUDGES
from .storage import RunStorage, install_cleanup_handlers


EXPLORATION_RATE = 0.1
OVERRIDE_MARGIN = 2.0
TEMP_QUOTA_BYTES = 128 * 1024**2


class ResidualRuntimeExpired(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResidualProofDecision:
    passed: bool
    code: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TreeMeasurement:
    files: int
    bytes: int


@dataclass(frozen=True, slots=True)
class ResidualComparison:
    start_score: float
    final_score: float
    delta: float
    per_member_delta: dict[MemberId, float]
    start_games: int
    final_games: int
    same_schedule: bool
    paired_randomness: bool


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    member: MemberId
    opponent: str
    seat: int
    start_score: float
    final_score: float


@dataclass(frozen=True, slots=True)
class ResidualStorageRecord:
    root: str
    quota_bytes: int
    bytes_written: int
    peak_temporary_bytes: int
    raw_replays_written: int
    temp_run_exists_after_cleanup: bool


@dataclass(slots=True)
class ResidualReport:
    phases: list[str]
    phase_durations_seconds: dict[str, float]
    rounds: dict[str, Any]
    candidate_comparison: ResidualComparison | None
    comparison: ResidualComparison | None
    candidate_evaluation_records: tuple[EvaluationRecord, ...]
    evaluation_records: tuple[EvaluationRecord, ...]
    promoted_members: tuple[MemberId, ...]
    candidate_overrides: int
    learned_overrides: int
    training_explorations: int
    decision: ResidualProofDecision
    failures: tuple[str, ...]
    storage: ResidualStorageRecord
    artifacts_before: TreeMeasurement
    artifacts_after: TreeMeasurement
    retained_population: str | None
    device: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def measure_tree(root: Path) -> TreeMeasurement:
    files = 0
    total_bytes = 0
    if root.exists():
        for directory, _, names in os.walk(root):
            for name in names:
                path = Path(directory) / name
                files += 1
                total_bytes += path.stat().st_size
    return TreeMeasurement(files, total_bytes)


def training_seat_schedule() -> tuple[tuple[MemberId, MemberId], ...]:
    return tuple(
        orientation
        for pair in itertools.combinations(MemberId, 2)
        for orientation in (pair, pair[::-1])
    )


def _valid_updates(updates: Mapping[MemberId, Any]) -> bool:
    if set(updates) != set(MemberId):
        return False
    return all(
        bool(getattr(updates[member], "all_finite", False))
        and math.isfinite(float(getattr(updates[member], "parameter_delta_l2", math.nan)))
        and float(getattr(updates[member], "parameter_delta_l2", 0.0)) > 0.0
        for member in MemberId
    )


def decide_residual_proof(
    group_delta: float,
    learned_overrides: int,
    updates: Mapping[MemberId, Any],
    failures: Sequence[str],
) -> ResidualProofDecision:
    if failures:
        return ResidualProofDecision(False, "REJECT_FAILURE", tuple(failures))
    if not _valid_updates(updates):
        return ResidualProofDecision(
            False,
            "REJECT_INVALID_UPDATE",
            ("all four residual policies must receive positive finite updates",),
        )
    if learned_overrides <= 0:
        return ResidualProofDecision(
            False,
            "REJECT_NO_OVERRIDE",
            ("the learned policy never cleared the override threshold",),
        )
    if not math.isfinite(group_delta) or group_delta <= 0.0:
        return ResidualProofDecision(
            False,
            "REJECT_NO_GROUP_IMPROVEMENT",
            ("the final driver-backed group must beat the untouched drivers",),
        )
    return ResidualProofDecision(True, "PASS_DRIVER_BACKED_MAC", ())


def select_promoted_members(
    per_member_delta: Mapping[MemberId, float],
) -> tuple[MemberId, ...]:
    return tuple(
        member
        for member in MemberId
        if math.isfinite(float(per_member_delta.get(member, math.nan)))
        and float(per_member_delta[member]) > 0.0
    )


def decide_survivor_confirmation(
    promoted_members: Sequence[MemberId],
    group_delta: float,
    learned_overrides: int,
    updates: Mapping[MemberId, Any],
    failures: Sequence[str],
) -> ResidualProofDecision:
    if failures:
        return ResidualProofDecision(False, "REJECT_FAILURE", tuple(failures))
    if not promoted_members:
        return ResidualProofDecision(
            False,
            "REJECT_NO_PROMOTION",
            ("candidate evaluation found no residual member above its driver",),
        )
    return decide_residual_proof(
        group_delta,
        learned_overrides,
        updates,
        failures,
    )


def _driver_actors(
    population: Mapping[MemberId, PolicyValueNet],
    registry: DriverRegistry,
    device: str,
    *,
    trainable: bool,
    overrides_enabled: bool,
    generation: str,
    override_mode: str = "policy",
    override_margin: float = OVERRIDE_MARGIN,
) -> dict[MemberId, DriverBackedActor]:
    actors: dict[MemberId, DriverBackedActor] = {}
    for member in MemberId:
        def driver_action(observation: Mapping[str, Any], selected: MemberId = member) -> list[int]:
            return registry.action(selected, observation)

        actors[member] = DriverBackedActor(
            member=member,
            model=population[member],
            deck=registry.deck(member),
            device=device,
            driver_action=driver_action,
            trainable=trainable,
            generation=generation,
            exploration_rate=EXPLORATION_RATE if trainable else 0.0,
            overrides_enabled=overrides_enabled,
            override_margin=override_margin,
            override_mode=override_mode,
        )
    return actors


def _collect_round(
    game_api: Any,
    actors: Mapping[MemberId, DriverBackedActor],
    seed: int,
    deadline_check: Any,
) -> list[CompletedGame]:
    rng = np.random.default_rng(seed)
    games: list[CompletedGame] = []
    for first, second in training_seat_schedule():
        games.append(
            run_training_game(
                game_api,
                actors[first],
                actors[second],
                GameSource.CURRENT_CURRENT,
                rng,
            )
        )
        deadline_check()
    audit = audit_training_batch(
        [game.provenance for game in games],
        set(MemberId),
    )
    if not audit.valid:
        raise RuntimeError("; ".join(audit.reasons))
    return games


def _counter_total(
    actor_groups: Sequence[Mapping[MemberId, DriverBackedActor]],
    attribute: str,
) -> int:
    return sum(
        int(getattr(actor.counters, attribute))
        for group in actor_groups
        for actor in group.values()
    )


@dataclass(slots=True)
class _FixedActor:
    name: str
    agent: Any
    directory: Path
    deck: list[int]

    def decide(self, observation: Mapping[str, Any]) -> list[int]:
        with pushd(self.directory):
            action = self.agent(dict(observation))
        return validate_action(action, dict(observation["select"]))


def _load_judges(project_root: Path, namespace: str) -> dict[str, _FixedActor]:
    result: dict[str, _FixedActor] = {}
    for index, relative in enumerate(JUDGES):
        directory = project_root / relative
        agent = load_agent(directory, f"residual_{namespace}_judge_{index}")
        result[relative] = _FixedActor(
            relative,
            agent,
            directory,
            first_result_deck(agent, directory),
        )
    return result


def _paired_search_rollout(
    root: Any,
    start_actors: tuple[Any, Any],
    final_actors: tuple[Any, Any],
    member_seat: int,
    start_rng: np.random.Generator,
    final_rng: np.random.Generator,
    search_step: Any,
    search_release: Any,
    max_steps: int = 2000,
) -> tuple[float, float]:
    start_state = root
    final_state = root
    shared = True
    start_score: float | None = None
    final_score: float | None = None
    created: set[int] = set()

    def score(state: Mapping[str, Any]) -> float | None:
        winner = int(state.get("result", -1))
        if winner < 0:
            return None
        if winner not in (0, 1):
            return 0.5
        return 1.0 if winner == member_seat else 0.0

    def decide(
        actors: tuple[Any, Any],
        observation: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> list[int]:
        state = observation.get("current")
        if not isinstance(state, Mapping):
            raise RuntimeError("search engine returned no current state")
        seat = int(state["yourIndex"])
        actor = actors[seat]
        if isinstance(actor, DriverBackedActor):
            action, _ = actor.decide(observation, seat, rng)
            return action
        return actor.decide(observation)

    try:
        for _ in range(max_steps):
            start_observation = asdict(start_state.observation)
            final_observation = (
                start_observation
                if shared
                else asdict(final_state.observation)
            )
            start_current = start_observation.get("current")
            final_current = final_observation.get("current")
            if not isinstance(start_current, Mapping) or not isinstance(final_current, Mapping):
                raise RuntimeError("search engine returned no current state")
            if start_score is None:
                start_score = score(start_current)
            if final_score is None:
                final_score = score(final_current)
            if start_score is not None and final_score is not None:
                return start_score, final_score

            start_action = (
                decide(start_actors, start_observation, start_rng)
                if start_score is None
                else None
            )
            final_action = (
                decide(final_actors, final_observation, final_rng)
                if final_score is None
                else None
            )
            if shared and start_action == final_action:
                next_state = search_step(start_state.searchId, start_action)
                created.add(int(next_state.searchId))
                start_state = final_state = next_state
                continue

            if shared:
                shared = False
            if start_action is not None:
                start_state = search_step(start_state.searchId, start_action)
                created.add(int(start_state.searchId))
            if final_action is not None:
                final_state = search_step(final_state.searchId, final_action)
                created.add(int(final_state.searchId))
        if start_score is None and final_score is None:
            return 0.5, 0.5
        return (
            0.0 if start_score is None else start_score,
            0.0 if final_score is None else final_score,
        )
    finally:
        for search_id in sorted(created, reverse=True):
            try:
                search_release(search_id)
            except Exception:
                pass


def _evaluate(
    project_root: Path,
    game_api: Any,
    population: Mapping[MemberId, PolicyValueNet],
    device: str,
    judge_paths: Sequence[str],
    enabled_members: Sequence[MemberId],
    seed: int,
    deadline_check: Any,
    *,
    override_mode: str = "policy",
    override_margin: float = OVERRIDE_MARGIN,
) -> tuple[tuple[EvaluationRecord, ...], int]:
    from cg.api import (  # type: ignore
        search_begin,
        search_end,
        search_release,
        search_step,
        to_observation_class,
    )

    start_registry = DriverRegistry.from_project(project_root)
    final_registry = DriverRegistry.from_project(project_root)
    try:
        start_actors = _driver_actors(
            population,
            start_registry,
            device,
            trainable=False,
            overrides_enabled=False,
            generation="driver",
            override_mode=override_mode,
            override_margin=override_margin,
        )
        final_actors = _driver_actors(
            population,
            final_registry,
            device,
            trainable=False,
            overrides_enabled=False,
            generation="residual",
            override_mode=override_mode,
            override_margin=override_margin,
        )
        enabled = set(enabled_members)
        for member, actor in final_actors.items():
            actor.overrides_enabled = member in enabled
        start_judges = _load_judges(project_root, f"search_start_{seed}")
        final_judges = _load_judges(project_root, f"search_final_{seed}")
        start_rng = np.random.default_rng(seed)
        final_rng = np.random.default_rng(seed)
        records: list[EvaluationRecord] = []
        overrides = 0
        for member in MemberId:
            for judge_name in judge_paths:
                for seat in (0, 1):
                    start_logical = (start_actors[member], start_judges[judge_name])
                    final_logical = (final_actors[member], final_judges[judge_name])
                    start_pair = (
                        start_logical if seat == 0 else start_logical[::-1]
                    )
                    final_pair = (
                        final_logical if seat == 0 else final_logical[::-1]
                    )
                    observation, start_data = game_api.battle_start(
                        start_pair[0].deck,
                        start_pair[1].deck,
                    )
                    if observation is None:
                        raise RuntimeError(
                            "battle_start failed: "
                            f"errorPlayer={start_data.errorPlayer}, "
                            f"errorType={start_data.errorType}"
                        )
                    try:
                        search_arguments = {
                            "your_deck": list(start_pair[0].deck),
                            "your_prize": [],
                            "opponent_deck": list(start_pair[1].deck),
                            "opponent_prize": [],
                            "opponent_hand": [],
                            "opponent_active": [],
                            "manual_coin": True,
                        }
                        typed_observation = to_observation_class(observation)
                        root = search_begin(
                            typed_observation,
                            **search_arguments,
                        )
                        before = final_actors[member].counters.evaluation_overrides
                        start_score, final_score = _paired_search_rollout(
                            root,
                            start_pair,
                            final_pair,
                            seat,
                            start_rng,
                            final_rng,
                            search_step,
                            search_release,
                        )
                        overrides += (
                            final_actors[member].counters.evaluation_overrides - before
                        )
                    finally:
                        try:
                            search_end()
                        finally:
                            game_api.battle_finish()
                    records.append(
                        EvaluationRecord(
                            member,
                            judge_name,
                            seat,
                            start_score,
                            final_score,
                        )
                    )
                    deadline_check()
        return tuple(records), overrides
    finally:
        start_registry.close()
        final_registry.close()


def search_pairing_smoke(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    population = create_population(20260804, "cpu")
    records, overrides = _evaluate(
        root,
        import_official_cg(root),
        population,
        "cpu",
        JUDGES[:1],
        (),
        20260804,
        lambda: None,
    )
    comparison = _compare(records)
    return {
        "records": len(records),
        "overrides": overrides,
        "all_paired_equal": all(
            record.start_score == record.final_score for record in records
        ),
        "delta": comparison.delta,
    }


def _compare(records: Sequence[EvaluationRecord]) -> ResidualComparison:
    if not records:
        raise ValueError("evaluation records are empty")
    start = np.asarray([record.start_score for record in records], dtype=np.float64)
    final = np.asarray([record.final_score for record in records], dtype=np.float64)
    per_member = {
        member: float(
            np.mean([record.final_score for record in records if record.member is member])
            - np.mean([record.start_score for record in records if record.member is member])
        )
        for member in MemberId
    }
    return ResidualComparison(
        start_score=float(start.mean()),
        final_score=float(final.mean()),
        delta=float(final.mean() - start.mean()),
        per_member_delta=per_member,
        start_games=len(records),
        final_games=len(records),
        same_schedule=True,
        paired_randomness=True,
    )


def _stats(stats: Mapping[MemberId, PPOStats]) -> dict[str, Any]:
    return {member.value: _jsonable(value) for member, value in stats.items()}


def _retain_population(
    project_root: Path,
    population: Mapping[MemberId, PolicyValueNet],
    promoted_members: Sequence[MemberId],
) -> str:
    destination = project_root / "agents/driver_backed_mac_pass"
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite passing population: {destination}")
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        for member in promoted_members:
            export_member(population[member], temporary / f"{member.value}.npz")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "decision": "PASS_DRIVER_BACKED_MAC",
                    "exploration_rate": EXPLORATION_RATE,
                    "override_margin": OVERRIDE_MARGIN,
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


def run_residual_proof(
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
        Path(tempfile.gettempdir()).resolve() / f"pokemon-residual-{uuid.uuid4().hex}",
        quota_bytes=TEMP_QUOTA_BYTES,
        shard_bytes=64 * 1024**2,
        max_pending=2,
    )
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
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
    final_updates: Mapping[MemberId, PPOStats] = {}
    decision = ResidualProofDecision(False, "REJECT_FAILURE", ("proof did not run",))
    registry: DriverRegistry | None = None

    def deadline_check() -> None:
        if time.monotonic() - started > wall_time_seconds:
            raise ResidualRuntimeExpired("driver-backed proof exceeded wall-time budget")

    def begin(name: str) -> float:
        deadline_check()
        phases.append(name)
        return time.monotonic()

    def end(name: str, phase_started: float) -> None:
        durations[name] = time.monotonic() - phase_started
        deadline_check()

    try:
        game_api = import_official_cg(project_root)
        population = create_population(seed, device)
        registry = DriverRegistry.from_project(project_root)
        actor_groups: list[Mapping[MemberId, DriverBackedActor]] = []

        phase_started = begin("round_1")
        round_one_actors = _driver_actors(
            population,
            registry,
            device,
            trainable=True,
            overrides_enabled=False,
            generation="round_0",
        )
        actor_groups.append(round_one_actors)
        round_one_games = _collect_round(
            game_api, round_one_actors, seed + 10, deadline_check
        )
        round_one_updates = update_population(
            population,
            round_one_games,
            device,
            seed + 20,
            epochs=2,
            batch_size=256,
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
            generation="round_1",
        )
        actor_groups.append(round_two_actors)
        round_two_games = _collect_round(
            game_api, round_two_actors, seed + 30, deadline_check
        )
        final_updates = update_population(
            population,
            round_two_games,
            device,
            seed + 40,
            epochs=2,
            batch_size=256,
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
            retained = _retain_population(
                project_root,
                population,
                promoted_members,
            )
        end("decision", phase_started)
    except ResidualRuntimeExpired as error:
        failures.append(str(error))
        decision = ResidualProofDecision(False, "REJECT_RUNTIME", tuple(failures))
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        decision = ResidualProofDecision(False, "REJECT_FAILURE", tuple(failures))
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


def write_report_atomic(report: ResidualReport, path: Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        partial.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
