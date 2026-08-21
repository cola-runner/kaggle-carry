from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import signal
import time
import uuid
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from scripts.run_local_match import (
    first_result_deck,
    import_official_cg,
    load_agent,
    pushd,
    validate_action,
)

from .bootstrap import (
    DriverRegistry,
    initialize_population,
    run_start_gate,
)
from .contracts import (
    FrozenLeagueConfig,
    GameSource,
    InvalidSelfPlay,
    MemberId,
    SelfPlayAudit,
    audit_training_batch,
)
from .engine import CompletedGame, PolicyActor, run_training_game
from .evaluation import (
    GroupComparison,
    JudgeResult,
    LeagueDecision,
    compare_groups,
    decide_validation,
)
from .model import PolicyValueNet, create_population, export_member, policy_parameter_count
from .ppo import PPOStats, update_population
from .schedule import JUDGES, LeagueSchedule, ScheduledGame
from .storage import (
    PendingShardLimit,
    QuotaExceeded,
    RunStorage,
    install_cleanup_handlers,
)


PHASES = (
    "preflight",
    "bootstrap",
    "start_gate",
    "round_1",
    "round_2",
    "judges",
    "ancestry",
    "decision",
    "cleanup",
)


class RuntimeExpired(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TreeMeasurement:
    files: int
    bytes: int


@dataclass(frozen=True, slots=True)
class PreflightRecord:
    config_sha256: str
    schedule_sha256: str
    model_sha256: str
    feature_sha256: str
    device: str
    schedule_counts: dict[str, int]
    artifacts_before: TreeMeasurement
    dependencies_present: bool

    @classmethod
    def freeze(
        cls,
        config: FrozenLeagueConfig,
        schedule: LeagueSchedule,
        project_root: Path,
        device: str,
    ) -> PreflightRecord:
        required = [
            project_root / "league_selfplay/model.py",
            project_root / "league_selfplay/features.py",
            *(project_root / judge for judge in JUDGES),
        ]
        model_path, feature_path = required[:2]
        return cls(
            config_sha256=config.sha256(),
            schedule_sha256=schedule.sha256,
            model_sha256=_file_sha256(model_path),
            feature_sha256=_file_sha256(feature_path),
            device=device,
            schedule_counts={
                "bootstrap": schedule.bootstrap_count,
                "round_one": schedule.round_one_count,
                "round_two_current": schedule.round_two_current_count,
                "round_two_history": schedule.round_two_history_count,
                "judges": schedule.judge_count,
                "ancestry": schedule.ancestry_count,
            },
            artifacts_before=measure_tree(project_root / "artifacts"),
            dependencies_present=all(path.exists() for path in required),
        )


@dataclass(frozen=True, slots=True)
class StorageRecord:
    root: str
    bytes_written: int
    peak_temporary_bytes: int
    raw_replays_written: int
    temp_run_exists_after_cleanup: bool


@dataclass(frozen=True, slots=True)
class FrozenJudgeResults:
    start: tuple[JudgeResult, ...]
    final: tuple[JudgeResult, ...]


@dataclass(slots=True)
class LeagueReport:
    preflight: PreflightRecord
    phases: list[str]
    phase_durations_seconds: dict[str, float]
    bootstrap: Any
    start_gate: Any
    round_one: dict[str, Any]
    round_two: dict[str, Any]
    self_play_audit: dict[str, Any]
    judge_results: FrozenJudgeResults | None
    ancestry_results: tuple[JudgeResult, ...]
    comparison: GroupComparison | None
    decision: LeagueDecision
    storage: StorageRecord
    artifacts_after: TreeMeasurement
    failures: tuple[str, ...]
    retained_population: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        payload["round_1"] = payload.pop("round_one")
        payload["round_2"] = payload.pop("round_two")
        return payload


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


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


def immutable_snapshot(
    population: Mapping[MemberId, PolicyValueNet],
) -> dict[MemberId, PolicyValueNet]:
    snapshot = copy.deepcopy(dict(population))
    for model in snapshot.values():
        model.eval()
        model.requires_grad_(False)
    return snapshot


def _ordered_members(game: ScheduledGame) -> tuple[MemberId, MemberId]:
    logical = (MemberId(game.actors[0]), MemberId(game.actors[1]))
    return logical[game.seats.index(0)], logical[game.seats.index(1)]


def _persist_games(storage: RunStorage, games: Sequence[CompletedGame]) -> list[CompletedGame]:
    if not games:
        return []
    try:
        paths = storage.write_shards(games)
    except PendingShardLimit:
        if len(games) == 1:
            raise
        midpoint = len(games) // 2
        return _persist_games(storage, games[:midpoint]) + _persist_games(
            storage, games[midpoint:]
        )
    return [game for path in paths for game in storage.consume_shard(path)]


def collect_phase(
    scheduled_games: Sequence[ScheduledGame],
    current: Mapping[MemberId, PolicyValueNet],
    storage: RunStorage,
    *,
    game_api: Any,
    decks: Mapping[MemberId, list[int]],
    device: str,
    seed: int,
    deadline_check: Callable[[], None] | None = None,
) -> list[CompletedGame]:
    actors = {
        member: PolicyActor(member, current[member], list(decks[member]), device)
        for member in MemberId
    }
    rng = np.random.default_rng(seed)
    completed: list[CompletedGame] = []
    pending: list[CompletedGame] = []
    for scheduled in scheduled_games:
        first, second = _ordered_members(scheduled)
        pending.append(
            run_training_game(
                game_api,
                actors[first],
                actors[second],
                GameSource.CURRENT_CURRENT,
                rng,
            )
        )
        if len(pending) >= 4:
            completed.extend(_persist_games(storage, pending))
            pending.clear()
        if deadline_check is not None:
            deadline_check()
    completed.extend(_persist_games(storage, pending))
    return completed


def _collect_history_phase(
    scheduled_games: Sequence[ScheduledGame],
    current: Mapping[MemberId, PolicyValueNet],
    history: Mapping[MemberId, PolicyValueNet],
    storage: RunStorage,
    *,
    game_api: Any,
    decks: Mapping[MemberId, list[int]],
    device: str,
    seed: int,
    deadline_check: Callable[[], None] | None = None,
) -> list[CompletedGame]:
    current_actors = {
        member: PolicyActor(member, current[member], list(decks[member]), device)
        for member in MemberId
    }
    history_actors = {
        member: PolicyActor(
            member,
            history[member],
            list(decks[member]),
            device,
            trainable=False,
            generation="round_0",
        )
        for member in MemberId
    }
    rng = np.random.default_rng(seed)
    completed: list[CompletedGame] = []
    pending: list[CompletedGame] = []
    for scheduled in scheduled_games:
        logical: list[PolicyActor] = []
        for actor_name, generation in zip(
            scheduled.actors, scheduled.generations, strict=True
        ):
            member = MemberId(actor_name)
            logical.append(
                current_actors[member]
                if generation == "round_1"
                else history_actors[member]
            )
        actor0 = logical[scheduled.seats.index(0)]
        actor1 = logical[scheduled.seats.index(1)]
        pending.append(
            run_training_game(
                game_api,
                actor0,
                actor1,
                GameSource.CURRENT_HISTORY,
                rng,
            )
        )
        if len(pending) >= 4:
            completed.extend(_persist_games(storage, pending))
            pending.clear()
        if deadline_check is not None:
            deadline_check()
    completed.extend(_persist_games(storage, pending))
    return completed


def collect_round_two(
    schedule: LeagueSchedule,
    current: Mapping[MemberId, PolicyValueNet],
    history: Mapping[MemberId, PolicyValueNet],
    storage: RunStorage,
    *,
    game_api: Any,
    decks: Mapping[MemberId, list[int]],
    device: str,
    seed: int,
    deadline_check: Callable[[], None] | None = None,
) -> list[CompletedGame]:
    current_games = collect_phase(
        schedule.round_two_current,
        current,
        storage,
        game_api=game_api,
        decks=decks,
        device=device,
        seed=seed,
        deadline_check=deadline_check,
    )
    history_games = _collect_history_phase(
        schedule.round_two_history,
        current,
        history,
        storage,
        game_api=game_api,
        decks=decks,
        device=device,
        seed=seed + 1,
        deadline_check=deadline_check,
    )
    return current_games + history_games


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


def _load_judges(project_root: Path) -> dict[str, _FixedActor]:
    judges: dict[str, _FixedActor] = {}
    for index, relative in enumerate(JUDGES):
        directory = project_root / relative
        agent = load_agent(directory, f"league_judge_{index}")
        judges[relative] = _FixedActor(
            name=relative,
            agent=agent,
            directory=directory,
            deck=first_result_deck(agent, directory),
        )
    return judges


def _run_evaluation_game(
    game_api: Any,
    actors: tuple[Any, Any],
    rng: np.random.Generator,
    max_steps: int = 2000,
) -> int:
    observation, start_data = game_api.battle_start(actors[0].deck, actors[1].deck)
    if observation is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}"
        )
    try:
        for _ in range(max_steps):
            current = observation.get("current")
            if not isinstance(current, Mapping):
                raise RuntimeError("engine returned no current state")
            result = int(current.get("result", -1))
            if result >= 0:
                return result
            seat = int(current["yourIndex"])
            actor = actors[seat]
            if isinstance(actor, PolicyActor):
                action, _ = actor.decide(observation, seat, rng)
            else:
                action = actor.decide(observation)
            observation = game_api.battle_select(action)
        raise RuntimeError(f"evaluation match did not finish within {max_steps} decisions")
    finally:
        game_api.battle_finish()


def _score_for_seat(winner: int, seat: int) -> float:
    if winner not in (0, 1):
        return 0.5
    return 1.0 if winner == seat else 0.0


def evaluate_frozen_groups(
    starting: Mapping[MemberId, PolicyValueNet],
    final: Mapping[MemberId, PolicyValueNet],
    scheduled_games: Sequence[ScheduledGame],
    *,
    game_api: Any,
    decks: Mapping[MemberId, list[int]],
    project_root: Path,
    device: str,
    seed: int,
    deadline_check: Callable[[], None] | None = None,
) -> FrozenJudgeResults:
    judges = _load_judges(project_root)
    populations = {"round_0": starting, "round_2": final}
    rng = np.random.default_rng(seed)
    start_results: list[JudgeResult] = []
    final_results: list[JudgeResult] = []
    for scheduled in scheduled_games:
        member = MemberId(scheduled.actors[0])
        judge_name = scheduled.actors[1]
        generation = scheduled.generations[0]
        neural = PolicyActor(
            member,
            populations[generation][member],
            list(decks[member]),
            device,
            trainable=False,
            generation=generation,
        )
        logical = (neural, judges[judge_name])
        actors = (
            logical[scheduled.seats.index(0)],
            logical[scheduled.seats.index(1)],
        )
        winner = _run_evaluation_game(game_api, actors, rng)
        neural_seat = scheduled.seats[0]
        record = JudgeResult(
            member,
            judge_name,
            neural_seat,
            _score_for_seat(winner, neural_seat),
        )
        (start_results if generation == "round_0" else final_results).append(record)
        if deadline_check is not None:
            deadline_check()
    return FrozenJudgeResults(tuple(start_results), tuple(final_results))


def evaluate_ancestry(
    final: Mapping[MemberId, PolicyValueNet],
    starting: Mapping[MemberId, PolicyValueNet],
    scheduled_games: Sequence[ScheduledGame],
    *,
    game_api: Any,
    decks: Mapping[MemberId, list[int]],
    device: str,
    seed: int,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[JudgeResult, ...]:
    rng = np.random.default_rng(seed)
    results: list[JudgeResult] = []
    for scheduled in scheduled_games:
        final_member = MemberId(scheduled.actors[0])
        starting_member = MemberId(scheduled.actors[1])
        logical = (
            PolicyActor(
                final_member,
                final[final_member],
                list(decks[final_member]),
                device,
                trainable=False,
                generation="round_2",
            ),
            PolicyActor(
                starting_member,
                starting[starting_member],
                list(decks[starting_member]),
                device,
                trainable=False,
                generation="round_0",
            ),
        )
        actors = (
            logical[scheduled.seats.index(0)],
            logical[scheduled.seats.index(1)],
        )
        winner = _run_evaluation_game(game_api, actors, rng)
        final_seat = scheduled.seats[0]
        results.append(
            JudgeResult(
                final_member,
                f"round_0:{starting_member.value}",
                final_seat,
                _score_for_seat(winner, final_seat),
            )
        )
        if deadline_check is not None:
            deadline_check()
    return tuple(results)


def _stats_summary(stats: Mapping[MemberId, PPOStats]) -> dict[str, Any]:
    return {
        "updated_members": sorted(member.value for member in stats),
        "members": {
            member.value: _jsonable(member_stats)
            for member, member_stats in stats.items()
        },
    }


def _audit_summary(first: SelfPlayAudit, second: SelfPlayAudit) -> dict[str, Any]:
    return {
        "valid": first.valid and second.valid,
        "code": "PASS_SELF_PLAY_AUDIT"
        if first.valid and second.valid
        else "INVALID_SELF_PLAY",
        "round_one": _jsonable(first),
        "round_two": _jsonable(second),
    }


def _ensure_audit(games: Sequence[CompletedGame]) -> SelfPlayAudit:
    audit = audit_training_batch(
        [game.provenance for game in games],
        set(MemberId),
    )
    if not audit.valid:
        raise InvalidSelfPlay("; ".join(audit.reasons))
    return audit


def _update_failures(
    round_one: Mapping[MemberId, PPOStats],
    round_two: Mapping[MemberId, PPOStats],
) -> list[str]:
    failures: list[str] = []
    for name, stats in (("round_1", round_one), ("round_2", round_two)):
        if set(stats) != set(MemberId):
            failures.append(f"{name} did not update all four members")
        for member, item in stats.items():
            if not item.all_finite or not math.isfinite(item.parameter_delta_l2):
                failures.append(f"{name}/{member.value} produced a non-finite update")
            elif item.parameter_delta_l2 <= 0:
                failures.append(f"{name}/{member.value} did not change")
    return failures


def _retain_passing_population(
    population: Mapping[MemberId, PolicyValueNet],
    project_root: Path,
    preflight: PreflightRecord,
) -> str:
    destination = project_root / "agents/four_policy_league_mac_pass"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite passing population: {destination}")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        checkpoints: dict[str, dict[str, Any]] = {}
        for member in MemberId:
            path = temporary / f"{member.value}.npz"
            export_member(population[member], path)
            checkpoints[member.value] = {
                "file": path.name,
                "sha256": _file_sha256(path),
                "policy_parameters": policy_parameter_count(population[member]),
            }
        manifest = {
            "decision": "PASS_MAC_LEAGUE",
            "config_sha256": preflight.config_sha256,
            "schedule_sha256": preflight.schedule_sha256,
            "model_sha256": preflight.model_sha256,
            "feature_sha256": preflight.feature_sha256,
            "checkpoints": checkpoints,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(destination)


def run_league(
    config: FrozenLeagueConfig,
    schedule: LeagueSchedule,
    project_root: Path,
) -> LeagueReport:
    project_root = Path(project_root).resolve()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    started = time.monotonic()
    phases: list[str] = ["preflight"]
    durations: dict[str, float] = {}
    phase_started = time.monotonic()
    preflight = PreflightRecord.freeze(config, schedule, project_root, device)
    durations["preflight"] = time.monotonic() - phase_started
    storage = RunStorage.create_under_tmp(config)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    install_cleanup_handlers(storage)

    bootstrap_stats: Any = None
    start_gate_result: Any = None
    round_one_summary: dict[str, Any] = {"updated_members": []}
    round_two_summary: dict[str, Any] = {"updated_members": []}
    audit_summary: dict[str, Any] = {"valid": False, "code": "NOT_RUN"}
    judge_results: FrozenJudgeResults | None = None
    ancestry_results: tuple[JudgeResult, ...] = ()
    comparison: GroupComparison | None = None
    retained_population: str | None = None
    failures: list[str] = []
    decision = LeagueDecision(False, "REJECT_FAILURE", ("league did not run",))

    def deadline_check() -> None:
        if time.monotonic() - started > config.wall_time_seconds:
            raise RuntimeExpired("Mac league exceeded its frozen wall-time budget")

    def begin_phase(name: str) -> float:
        deadline_check()
        phases.append(name)
        return time.monotonic()

    def end_phase(name: str, phase_start: float) -> None:
        durations[name] = time.monotonic() - phase_start
        deadline_check()

    try:
        deadline_check()
        if not preflight.dependencies_present:
            raise RuntimeError("preflight dependencies are missing")
        game_api = import_official_cg(project_root)
        population = create_population(config.seed, device)
        drivers = DriverRegistry.from_project(project_root)
        decks = {member: drivers.deck(member) for member in MemberId}
        is_dry = schedule.bootstrap_count <= 12

        phase_start = begin_phase("bootstrap")
        bootstrap_stats = initialize_population(
            population,
            drivers,
            games=schedule.bootstrap_count,
            device=device,
            seed=config.seed,
            epochs=1 if is_dry else 6,
        )
        end_phase("bootstrap", phase_start)

        phase_start = begin_phase("start_gate")
        start_gate_result = run_start_gate(population, game_api, drivers, seed=config.seed)
        if not start_gate_result.teacher_closed:
            raise InvalidSelfPlay("teacher registry remained open")
        starting = immutable_snapshot(population)
        end_phase("start_gate", phase_start)

        phase_start = begin_phase("round_1")
        round_one_games = collect_phase(
            schedule.round_one,
            population,
            storage,
            game_api=game_api,
            decks=decks,
            device=device,
            seed=config.seed + 10,
            deadline_check=deadline_check,
        )
        round_one_audit = _ensure_audit(round_one_games)
        round_one_stats = update_population(
            population,
            round_one_games,
            device,
            config.seed + 20,
            epochs=1 if is_dry else 4,
        )
        round_one_summary = _stats_summary(round_one_stats)
        end_phase("round_1", phase_start)

        phase_start = begin_phase("round_2")
        round_two_games = collect_round_two(
            schedule,
            population,
            starting,
            storage,
            game_api=game_api,
            decks=decks,
            device=device,
            seed=config.seed + 30,
            deadline_check=deadline_check,
        )
        round_two_audit = _ensure_audit(round_two_games)
        round_two_stats = update_population(
            population,
            round_two_games,
            device,
            config.seed + 40,
            epochs=1 if is_dry else 4,
        )
        round_two_summary = _stats_summary(round_two_stats)
        audit_summary = _audit_summary(round_one_audit, round_two_audit)
        final = immutable_snapshot(population)
        end_phase("round_2", phase_start)

        phase_start = begin_phase("judges")
        judge_results = evaluate_frozen_groups(
            starting,
            final,
            schedule.judges,
            game_api=game_api,
            decks=decks,
            project_root=project_root,
            device=device,
            seed=config.seed + 50,
            deadline_check=deadline_check,
        )
        end_phase("judges", phase_start)

        phase_start = begin_phase("ancestry")
        ancestry_results = evaluate_ancestry(
            final,
            starting,
            schedule.ancestry,
            game_api=game_api,
            decks=decks,
            device=device,
            seed=config.seed + 60,
            deadline_check=deadline_check,
        )
        end_phase("ancestry", phase_start)

        phase_start = begin_phase("decision")
        comparison = compare_groups(
            judge_results.start,
            judge_results.final,
            ancestry_results,
            config.seed,
        )
        failures.extend(_update_failures(round_one_stats, round_two_stats))
        decision = decide_validation(
            comparison,
            round_two_stats,
            failures,
        )
        if decision.passed:
            retained_population = _retain_passing_population(
                final,
                project_root,
                preflight,
            )
        end_phase("decision", phase_start)
    except RuntimeExpired as error:
        failures.append(str(error))
        decision = LeagueDecision(False, "REJECT_RUNTIME", tuple(failures))
    except (QuotaExceeded, PendingShardLimit) as error:
        failures.append(str(error))
        decision = LeagueDecision(False, "REJECT_STORAGE", tuple(failures))
    except InvalidSelfPlay as error:
        failures.append(str(error))
        decision = LeagueDecision(False, "INVALID_SELF_PLAY", tuple(failures))
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        decision = LeagueDecision(False, "REJECT_FAILURE", tuple(failures))
    finally:
        cleanup_started = time.monotonic()
        storage.cleanup()
        if "cleanup" not in phases:
            phases.append("cleanup")
        durations["cleanup"] = time.monotonic() - cleanup_started
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    storage_record = StorageRecord(
        root=str(storage.root),
        bytes_written=storage.bytes_written,
        peak_temporary_bytes=storage.peak_bytes,
        raw_replays_written=0,
        temp_run_exists_after_cleanup=storage.root.exists(),
    )
    return LeagueReport(
        preflight=preflight,
        phases=phases,
        phase_durations_seconds=durations,
        bootstrap=bootstrap_stats,
        start_gate=start_gate_result,
        round_one=round_one_summary,
        round_two=round_two_summary,
        self_play_audit=audit_summary,
        judge_results=judge_results,
        ancestry_results=ancestry_results,
        comparison=comparison,
        decision=decision,
        storage=storage_record,
        artifacts_after=measure_tree(project_root / "artifacts"),
        failures=tuple(failures),
        retained_population=retained_population,
    )


def write_report_atomic(report: LeagueReport, path: Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    try:
        with partial.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
