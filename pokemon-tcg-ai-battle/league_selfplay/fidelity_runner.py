from __future__ import annotations

import json
import math
import os
import signal
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bootstrap import DriverRegistry
from .contracts import MemberId
from .fidelity_data import PairedGame, collect_paired_games, split_games
from .fidelity_gate import FidelityDecision, decide_fidelity
from .fidelity_train import (
    PairedTrainingResult,
    evaluate_population,
    train_paired_populations,
)
from .model import export_member
from .numpy_runtime import load_policy, numpy_forward
from .storage import QuotaExceeded, RunStorage, install_cleanup_handlers


PHASES = (
    "preflight",
    "collect",
    "split",
    "train",
    "held_out",
    "decision",
    "cleanup",
)


class FidelityRuntimeExpired(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TreeMeasurement:
    files: int
    bytes: int


@dataclass(frozen=True, slots=True)
class FidelityStorageRecord:
    root: str
    bytes_written: int
    peak_temporary_bytes: int
    temp_run_exists_after_cleanup: bool


@dataclass(slots=True)
class FidelityRunReport:
    phases: list[str]
    phase_durations_seconds: dict[str, float]
    games: dict[str, int]
    decisions: dict[str, int]
    train_game_ids_overlap_held_out: bool
    training: dict[str, Any]
    metrics: dict[str, Any]
    numpy_parity: dict[str, Any]
    decision: FidelityDecision
    failures: tuple[str, ...]
    ppo_calls: int
    raw_replays_written: int
    storage: FidelityStorageRecord
    artifacts_before: TreeMeasurement
    artifacts_after: TreeMeasurement
    device: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


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


def _stats_summary(training: PairedTrainingResult) -> dict[str, Any]:
    return {
        "v1_updated_members": sorted(member.value for member in training.v1_stats),
        "v2_updated_members": sorted(member.value for member in training.v2_stats),
        "v1": _jsonable(training.v1_stats),
        "v2": _jsonable(training.v2_stats),
    }


def _training_failures(training: PairedTrainingResult) -> list[str]:
    failures: list[str] = []
    for version, stats in (("v1", training.v1_stats), ("v2", training.v2_stats)):
        if set(stats) != set(MemberId):
            failures.append(f"{version} did not update all members")
        for member, item in stats.items():
            if not item.all_finite or not math.isfinite(item.parameter_delta_l2):
                failures.append(f"{version}/{member.value} update was non-finite")
            elif item.parameter_delta_l2 <= 0:
                failures.append(f"{version}/{member.value} did not change")
    return failures


def _first_member_decisions(
    games: Sequence[PairedGame],
) -> dict[MemberId, Any]:
    result: dict[MemberId, Any] = {}
    for game in games:
        for decision in game.decisions:
            result.setdefault(decision.member, decision)
    if set(result) != set(MemberId):
        raise ValueError("held-out games do not contain all four members")
    return result


def _owned_file_bytes(storage: RunStorage) -> int:
    return sum(
        path.stat().st_size
        for path in storage.root.iterdir()
        if path.is_file() and path.name != ".league-storage-owner"
    )


def _numpy_parity(
    training: PairedTrainingResult,
    held_out: Sequence[PairedGame],
    storage: RunStorage,
    device: str,
) -> dict[str, Any]:
    decisions = _first_member_decisions(held_out)
    maximum_option_error = 0.0
    maximum_stop_error = 0.0
    all_finite = True
    for member in MemberId:
        model = training.v2_population[member]
        features = decisions[member].v2_features
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(features)[None, :, :].to(device)
            mask = torch.ones((1, len(features)), dtype=torch.bool, device=device)
            torch_options, torch_stop, _ = model(tensor, mask)
        path = storage.root / f"parity-{member.value}.npz"
        try:
            export_member(model, path)
            size = path.stat().st_size
            current_bytes = _owned_file_bytes(storage)
            if current_bytes > storage.quota_bytes:
                raise QuotaExceeded("temporary fidelity parity file exceeded quota")
            storage.bytes_written += size
            storage.peak_bytes = max(storage.peak_bytes, current_bytes)
            numpy_options, numpy_stop = numpy_forward(features, load_policy(path))
        finally:
            path.unlink(missing_ok=True)
        option_error = float(
            np.max(
                np.abs(
                    torch_options[0].detach().cpu().numpy()
                    - numpy_options
                )
            )
        )
        stop_error = abs(float(torch_stop[0].detach().cpu()) - numpy_stop)
        maximum_option_error = max(maximum_option_error, option_error)
        maximum_stop_error = max(maximum_stop_error, stop_error)
        all_finite = all_finite and math.isfinite(option_error) and math.isfinite(stop_error)
    passed = all_finite and maximum_option_error < 1e-4 and maximum_stop_error < 1e-4
    return {
        "passed": passed,
        "max_option_error": maximum_option_error,
        "max_stop_error": maximum_stop_error,
        "all_finite": all_finite,
    }


def run_fidelity(
    project_root: Path,
    *,
    games: int,
    train_games: int,
    seed: int,
    epochs: int,
    wall_time_seconds: int,
) -> FidelityRunReport:
    project_root = Path(project_root).resolve()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    started = time.monotonic()
    phases = ["preflight"]
    phase_durations: dict[str, float] = {}
    preflight_started = time.monotonic()
    artifacts_before = measure_tree(project_root / "artifacts")
    phase_durations["preflight"] = time.monotonic() - preflight_started
    storage_root = (
        Path(tempfile.gettempdir()).resolve()
        / f"pokemon-fidelity-{uuid.uuid4().hex}"
    )
    storage = RunStorage(
        storage_root,
        quota_bytes=128 * 1024**2,
        shard_bytes=64 * 1024**2,
        max_pending=2,
    )
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    install_cleanup_handlers(storage)

    collected: tuple[PairedGame, ...] = ()
    train: tuple[PairedGame, ...] = ()
    held_out: tuple[PairedGame, ...] = ()
    training_summary: dict[str, Any] = {
        "v1_updated_members": [],
        "v2_updated_members": [],
    }
    metrics_summary: dict[str, Any] = {"v1": {}, "v2": {}}
    parity: dict[str, Any] = {"passed": False}
    failures: list[str] = []
    decision = FidelityDecision(
        False,
        "REJECT_DRIVER_FIDELITY",
        ("failure:fidelity run did not finish",),
    )
    overlap = False

    def deadline_check() -> None:
        if time.monotonic() - started > wall_time_seconds:
            raise FidelityRuntimeExpired("fidelity run exceeded six-minute deadline")

    def begin_phase(name: str) -> float:
        deadline_check()
        phases.append(name)
        return time.monotonic()

    def end_phase(name: str, phase_started: float) -> None:
        phase_durations[name] = time.monotonic() - phase_started
        deadline_check()

    try:
        deadline_check()
        phase_started = begin_phase("collect")
        registry = DriverRegistry.from_project(project_root)
        try:
            collected = collect_paired_games(registry, games, deadline_check)
        finally:
            registry.close()
        end_phase("collect", phase_started)

        phase_started = begin_phase("split")
        train, held_out = split_games(collected, train_games)
        train_ids = {game.game_id for game in train}
        held_out_ids = {game.game_id for game in held_out}
        overlap = bool(train_ids & held_out_ids)
        if overlap:
            failures.append("train and held-out game IDs overlap")
        end_phase("split", phase_started)

        phase_started = begin_phase("train")
        training = train_paired_populations(
            train,
            device,
            seed,
            epochs=epochs,
        )
        training_summary = _stats_summary(training)
        failures.extend(_training_failures(training))
        end_phase("train", phase_started)

        phase_started = begin_phase("held_out")
        v1_metrics = evaluate_population(
            training.v1_population,
            held_out,
            "v1",
            device,
        )
        v2_metrics = evaluate_population(
            training.v2_population,
            held_out,
            "v2",
            device,
        )
        metrics_summary = {
            "v1": _jsonable(v1_metrics),
            "v2": _jsonable(v2_metrics),
        }
        parity = _numpy_parity(training, held_out, storage, device)
        if not parity["passed"]:
            failures.append("V2 PyTorch/NumPy parity failed")
        end_phase("held_out", phase_started)

        phase_started = begin_phase("decision")
        decision = decide_fidelity(v1_metrics, v2_metrics, failures)
        end_phase("decision", phase_started)
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        decision = FidelityDecision(
            False,
            "REJECT_DRIVER_FIDELITY",
            tuple(f"failure:{failure}" for failure in failures),
        )
    finally:
        cleanup_started = time.monotonic()
        storage.cleanup()
        phases.append("cleanup")
        phase_durations["cleanup"] = time.monotonic() - cleanup_started
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    return FidelityRunReport(
        phases=phases,
        phase_durations_seconds=phase_durations,
        games={
            "collected": len(collected),
            "train": len(train),
            "held_out": len(held_out),
        },
        decisions={
            "train": sum(len(game.decisions) for game in train),
            "held_out": sum(len(game.decisions) for game in held_out),
        },
        train_game_ids_overlap_held_out=overlap,
        training=training_summary,
        metrics=metrics_summary,
        numpy_parity=parity,
        decision=decision,
        failures=tuple(failures),
        ppo_calls=0,
        raw_replays_written=0,
        storage=FidelityStorageRecord(
            root=str(storage.root),
            bytes_written=storage.bytes_written,
            peak_temporary_bytes=storage.peak_bytes,
            temp_run_exists_after_cleanup=storage.root.exists(),
        ),
        artifacts_before=artifacts_before,
        artifacts_after=measure_tree(project_root / "artifacts"),
        device=device,
        seed=seed,
    )


def write_report_atomic(report: FidelityRunReport, path: Path) -> None:
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
