from __future__ import annotations

import hashlib
import io
import os
import shutil
import signal
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Iterable, Sequence

import numpy as np

from .contracts import GameProvenance, GameSource, MemberId


class QuotaExceeded(RuntimeError):
    pass


class PendingShardLimit(RuntimeError):
    pass


class CorruptShard(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ShardManifest:
    path: Path
    game_count: int
    size_bytes: int
    sha256: str


def _ordered_member_ids(members: Iterable[MemberId]) -> tuple[int, int]:
    indices = [list(MemberId).index(member) for member in members]
    if len(indices) > 2:
        raise ValueError("a game can have at most two trajectory members")
    padded = indices + [-1, -1]
    return padded[0], padded[1]


def _members_from_ids(member_ids: Iterable[int]) -> tuple[MemberId, ...]:
    members = tuple(MemberId)
    result: list[MemberId] = []
    for member_id in member_ids:
        if member_id == -1:
            continue
        if member_id < 0 or member_id >= len(members):
            raise CorruptShard("unknown provenance member id")
        result.append(members[member_id])
    return tuple(result)


def _encoded_shard(games: Sequence[Any]) -> bytes:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    feature_offsets = [0]
    action_offsets = [0]
    step_members: list[int] = []
    step_seats: list[int] = []
    min_counts: list[int] = []
    max_counts: list[int] = []
    old_log_probabilities: list[float] = []
    old_values: list[float] = []
    rewards: list[float] = []
    step_game_ids: list[int] = []

    game_sources: list[str] = []
    game_actor_zero: list[str] = []
    game_actor_one: list[str] = []
    trajectory_member_ids: list[tuple[int, int]] = []
    update_member_ids: list[tuple[int, int]] = []
    winners: list[int] = []
    decisions: list[int] = []
    finished: list[bool] = []

    member_to_index = {member: index for index, member in enumerate(MemberId)}
    for game_id, game in enumerate(games):
        provenance = game.provenance
        game_sources.append(provenance.source.value)
        game_actor_zero.append(provenance.actors[0])
        game_actor_one.append(provenance.actors[1])
        trajectory_member_ids.append(_ordered_member_ids(provenance.trajectory_members))
        update_member_ids.append(_ordered_member_ids(provenance.update_members))
        winners.append(int(game.winner))
        decisions.append(int(game.decisions))
        finished.append(bool(game.finished))
        for step in game.steps:
            array = np.asarray(step.features, dtype=np.float32)
            if array.ndim != 2:
                raise ValueError("trajectory features must be a matrix")
            action = np.asarray(step.action, dtype=np.int32)
            features.append(array)
            actions.append(action)
            feature_offsets.append(feature_offsets[-1] + len(array))
            action_offsets.append(action_offsets[-1] + len(action))
            step_members.append(member_to_index[step.member])
            step_seats.append(int(step.seat))
            min_counts.append(int(step.min_count))
            max_counts.append(int(step.max_count))
            old_log_probabilities.append(float(step.old_log_probability))
            old_values.append(float(step.old_value))
            rewards.append(float(step.reward))
            step_game_ids.append(game_id)

    feature_width = features[0].shape[1] if features else 0
    if any(array.shape[1] != feature_width for array in features):
        raise ValueError("all feature rows must have the same width")
    feature_array = (
        np.concatenate(features, axis=0).astype(np.float32, copy=False)
        if features
        else np.empty((0, feature_width), dtype=np.float32)
    )
    action_array = (
        np.concatenate(actions).astype(np.int32, copy=False)
        if actions
        else np.empty(0, dtype=np.int32)
    )
    arrays = {
        "schema_version": np.asarray([1], dtype=np.int16),
        "features": feature_array,
        "feature_offsets": np.asarray(feature_offsets, dtype=np.int64),
        "actions": action_array,
        "action_offsets": np.asarray(action_offsets, dtype=np.int64),
        "step_members": np.asarray(step_members, dtype=np.int8),
        "step_seats": np.asarray(step_seats, dtype=np.int8),
        "min_counts": np.asarray(min_counts, dtype=np.int16),
        "max_counts": np.asarray(max_counts, dtype=np.int16),
        "old_log_probabilities": np.asarray(old_log_probabilities, dtype=np.float32),
        "old_values": np.asarray(old_values, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "step_game_ids": np.asarray(step_game_ids, dtype=np.int32),
        "game_sources": np.asarray(game_sources, dtype=np.str_),
        "game_actor_zero": np.asarray(game_actor_zero, dtype=np.str_),
        "game_actor_one": np.asarray(game_actor_one, dtype=np.str_),
        "trajectory_member_ids": np.asarray(trajectory_member_ids, dtype=np.int8),
        "update_member_ids": np.asarray(update_member_ids, dtype=np.int8),
        "winners": np.asarray(winners, dtype=np.int8),
        "decisions": np.asarray(decisions, dtype=np.int32),
        "finished": np.asarray(finished, dtype=np.bool_),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _decoded_shard(payload: bytes) -> list[Any]:
    try:
        archive = np.load(io.BytesIO(payload), allow_pickle=False)
        with archive:
            required = {
                "schema_version",
                "features",
                "feature_offsets",
                "actions",
                "action_offsets",
                "step_members",
                "step_seats",
                "min_counts",
                "max_counts",
                "old_log_probabilities",
                "old_values",
                "rewards",
                "step_game_ids",
                "game_sources",
                "game_actor_zero",
                "game_actor_one",
                "trajectory_member_ids",
                "update_member_ids",
                "winners",
                "decisions",
                "finished",
            }
            if set(archive.files) != required:
                raise ValueError("unexpected shard fields")
            arrays = {name: archive[name] for name in required}
    except Exception as error:
        raise CorruptShard("trajectory shard is not a valid league NPZ") from error

    if arrays["schema_version"].tolist() != [1]:
        raise CorruptShard("unsupported trajectory shard version")
    step_count = len(arrays["step_members"])
    if len(arrays["feature_offsets"]) != step_count + 1:
        raise CorruptShard("invalid feature offsets")
    if len(arrays["action_offsets"]) != step_count + 1:
        raise CorruptShard("invalid action offsets")
    if int(arrays["feature_offsets"][-1]) != len(arrays["features"]):
        raise CorruptShard("feature offsets exceed payload")
    if int(arrays["action_offsets"][-1]) != len(arrays["actions"]):
        raise CorruptShard("action offsets exceed payload")
    game_count = len(arrays["winners"])
    game_fields = (
        "game_sources",
        "game_actor_zero",
        "game_actor_one",
        "trajectory_member_ids",
        "update_member_ids",
        "decisions",
        "finished",
    )
    if any(len(arrays[name]) != game_count for name in game_fields):
        raise CorruptShard("inconsistent game metadata lengths")
    step_fields = (
        "step_seats",
        "min_counts",
        "max_counts",
        "old_log_probabilities",
        "old_values",
        "rewards",
        "step_game_ids",
    )
    if any(len(arrays[name]) != step_count for name in step_fields):
        raise CorruptShard("inconsistent trajectory lengths")

    from .engine import CompletedGame, TrajectoryStep

    members = tuple(MemberId)
    games: list[CompletedGame] = []
    for game_id in range(game_count):
        game_steps: list[TrajectoryStep] = []
        for step_id in np.flatnonzero(arrays["step_game_ids"] == game_id):
            feature_start, feature_end = arrays["feature_offsets"][step_id : step_id + 2]
            action_start, action_end = arrays["action_offsets"][step_id : step_id + 2]
            member_index = int(arrays["step_members"][step_id])
            if member_index < 0 or member_index >= len(members):
                raise CorruptShard("unknown member id")
            game_steps.append(
                TrajectoryStep(
                    member=members[member_index],
                    seat=int(arrays["step_seats"][step_id]),
                    features=arrays["features"][feature_start:feature_end].astype(
                        np.float32, copy=True
                    ),
                    action=tuple(
                        int(value)
                        for value in arrays["actions"][action_start:action_end]
                    ),
                    min_count=int(arrays["min_counts"][step_id]),
                    max_count=int(arrays["max_counts"][step_id]),
                    old_log_probability=float(arrays["old_log_probabilities"][step_id]),
                    old_value=float(arrays["old_values"][step_id]),
                    reward=float(arrays["rewards"][step_id]),
                )
            )
        provenance = GameProvenance(
            source=GameSource(str(arrays["game_sources"][game_id])),
            actors=(
                str(arrays["game_actor_zero"][game_id]),
                str(arrays["game_actor_one"][game_id]),
            ),
            trajectory_members=_members_from_ids(
                int(value) for value in arrays["trajectory_member_ids"][game_id]
            ),
            update_members=_members_from_ids(
                int(value) for value in arrays["update_member_ids"][game_id]
            ),
        )
        games.append(
            CompletedGame(
                provenance=provenance,
                winner=int(arrays["winners"][game_id]),
                decisions=int(arrays["decisions"][game_id]),
                steps=game_steps,
                finished=bool(arrays["finished"][game_id]),
            )
        )
    return games


def _estimated_game_bytes(game: Any) -> int:
    feature_bytes = sum(np.asarray(step.features).nbytes for step in game.steps)
    action_bytes = sum(len(step.action) * 4 for step in game.steps)
    return feature_bytes + action_bytes + len(game.steps) * 64 + 1024


class RunStorage:
    def __init__(
        self,
        root: Path,
        *,
        quota_bytes: int,
        shard_bytes: int = 64 * 1024**2,
        max_pending: int = 2,
    ) -> None:
        if quota_bytes <= 0 or shard_bytes <= 0 or max_pending <= 0:
            raise ValueError("storage limits must be positive")
        self.root = Path(root).expanduser().resolve()
        if self.root == self.root.parent:
            raise ValueError("run root cannot be a filesystem root")
        self.quota_bytes = int(quota_bytes)
        self.shard_bytes = int(shard_bytes)
        self.max_pending = int(max_pending)
        self.owner_id = uuid.uuid4().hex
        self._pending: dict[Path, ShardManifest] = {}
        self._next_shard = 0
        self.bytes_written = 0
        self.peak_bytes = 0
        self.root.mkdir(parents=False, exist_ok=False)
        self._marker = self.root / ".league-storage-owner"
        self._marker.write_text(self.owner_id, encoding="utf-8")

    @classmethod
    def create_under_tmp(cls, config: Any) -> RunStorage:
        parent = Path(tempfile.gettempdir()).resolve()
        root = parent / f"pokemon-league-{uuid.uuid4().hex}"
        return cls(
            root,
            quota_bytes=int(config.temp_quota_bytes),
            shard_bytes=64 * 1024**2,
            max_pending=2,
        )

    def __enter__(self) -> RunStorage:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.cleanup()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _disk_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in self.root.iterdir()
            if path.is_file() and path != self._marker
        )

    def _fit_chunks(self, games: Sequence[Any]) -> list[tuple[list[Any], bytes]]:
        rough_chunks: list[list[Any]] = []
        current: list[Any] = []
        estimated = 4096
        target = self.shard_bytes
        for game in games:
            game_bytes = _estimated_game_bytes(game)
            if current and estimated + game_bytes > target:
                rough_chunks.append(current)
                current = []
                estimated = 4096
            current.append(game)
            estimated += game_bytes
        if current:
            rough_chunks.append(current)

        fitted: list[tuple[list[Any], bytes]] = []

        def fit(chunk: list[Any]) -> None:
            payload = _encoded_shard(chunk)
            if len(payload) <= self.shard_bytes:
                fitted.append((chunk, payload))
                return
            if len(chunk) == 1:
                raise QuotaExceeded("one trajectory game exceeds the shard limit")
            midpoint = len(chunk) // 2
            fit(chunk[:midpoint])
            fit(chunk[midpoint:])

        for chunk in rough_chunks:
            fit(chunk)
        return fitted

    def _write_chunks(self, chunks: Sequence[tuple[list[Any], bytes]]) -> list[Path]:
        if self.pending_count + len(chunks) > self.max_pending:
            raise PendingShardLimit("refusing to create a third pending shard")
        paths: list[Path] = []
        for chunk, payload in chunks:
            if self._disk_bytes() + len(payload) > self.quota_bytes:
                raise QuotaExceeded("temporary trajectory quota exceeded")
            path = self.root / f"shard-{self._next_shard:06d}.npz"
            partial = path.with_suffix(".npz.partial")
            self._next_shard += 1
            try:
                with partial.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                if partial.stat().st_size > self.shard_bytes:
                    raise QuotaExceeded("trajectory shard limit exceeded")
                os.replace(partial, path)
                directory_fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                partial.unlink(missing_ok=True)
            manifest = ShardManifest(
                path=path,
                game_count=len(chunk),
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            self._pending[path] = manifest
            self.bytes_written += len(payload)
            self.peak_bytes = max(self.peak_bytes, self._disk_bytes())
            paths.append(path)
        return paths

    def write_shards(self, games: Sequence[Any]) -> list[Path]:
        if not games:
            return []
        return self._write_chunks(self._fit_chunks(list(games)))

    def write_shard(self, games: Sequence[Any]) -> Path:
        if not games:
            raise ValueError("cannot write an empty trajectory shard")
        chunks = self._fit_chunks(list(games))
        if len(chunks) != 1:
            raise ValueError("games require more than one shard")
        return self._write_chunks(chunks)[0]

    def consume_shard(self, path: Path) -> tuple[Any, ...]:
        resolved = Path(path).resolve()
        if resolved.parent != self.root or resolved not in self._pending:
            raise ValueError("path is not a pending shard owned by this run")
        try:
            games = tuple(_decoded_shard(resolved.read_bytes()))
        except Exception as error:
            resolved.unlink(missing_ok=True)
            self._pending.pop(resolved, None)
            if isinstance(error, CorruptShard):
                raise
            raise CorruptShard("could not consume trajectory shard") from error
        resolved.unlink(missing_ok=False)
        self._pending.pop(resolved, None)
        return games

    def cleanup(self) -> None:
        if not self.root.exists():
            self._pending.clear()
            return
        if not self._marker.is_file() or self._marker.read_text(encoding="utf-8") != self.owner_id:
            raise RuntimeError("refusing to clean a run directory without its owner marker")
        shutil.rmtree(self.root)
        self._pending.clear()
        if self.root.exists():
            raise RuntimeError("trajectory run directory survived cleanup")


def install_cleanup_handlers(
    storage: RunStorage,
) -> dict[signal.Signals, Any]:
    handlers: dict[signal.Signals, Any] = {}

    def handler(signum: int, _frame: FrameType | None) -> None:
        try:
            storage.cleanup()
        finally:
            raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handler)
        handlers[signum] = handler
    return handlers


def storage_smoke() -> dict[str, object]:
    from .engine import CompletedGame, TrajectoryStep
    from .features import INPUT_WIDTH

    def compact_games(count: int = 3) -> list[CompletedGame]:
        games: list[CompletedGame] = []
        for game_id in range(count):
            steps = [
                TrajectoryStep(
                    member=member,
                    seat=seat,
                    features=np.full(
                        (2, INPUT_WIDTH), game_id + seat / 10,
                        dtype=np.float32,
                    ),
                    action=(seat,),
                    min_count=1,
                    max_count=1,
                    old_log_probability=-0.5 - seat,
                    old_value=0.25 + seat,
                    reward=1.0 if seat == game_id % 2 else -1.0,
                )
                for seat, member in enumerate(
                    (MemberId.GRIMMSNARL, MemberId.LUCARIO)
                )
            ]
            games.append(
                CompletedGame(
                    provenance=GameProvenance.current_game(
                        MemberId.GRIMMSNARL, MemberId.LUCARIO
                    ),
                    winner=game_id % 2,
                    decisions=len(steps),
                    steps=steps,
                    finished=True,
                )
            )
        return games

    with tempfile.TemporaryDirectory(prefix="league-storage-test-") as temporary:
        parent = Path(temporary)
        normal_root = parent / "normal"
        with RunStorage(
            normal_root,
            quota_bytes=1024**2,
            shard_bytes=32 * 1024,
            max_pending=2,
        ) as storage:
            paths = storage.write_shards(compact_games())
            within_limit = all(path.stat().st_size <= 32 * 1024 for path in paths)
            no_partials = not list(normal_root.glob("*.partial"))
            restored = [
                game for path in paths for game in storage.consume_shard(path)
            ]
            consumed_deleted = not any(path.exists() for path in paths)
            float32_features = all(
                step.features.dtype == np.float32
                for game in restored
                for step in game.steps
            )
        normal_cleaned = not normal_root.exists()

        quota_root = parent / "quota"
        quota_rejected = False
        try:
            with RunStorage(
                quota_root,
                quota_bytes=128,
                shard_bytes=128,
                max_pending=2,
            ) as storage:
                storage.write_shards(compact_games())
        except QuotaExceeded:
            quota_rejected = not quota_root.exists()

        pending_root = parent / "pending"
        pending_rejected = False
        with RunStorage(
            pending_root,
            quota_bytes=1024**2,
            shard_bytes=32 * 1024,
            max_pending=2,
        ) as storage:
            storage.write_shards(compact_games(1))
            storage.write_shards(compact_games(1))
            try:
                storage.write_shards(compact_games(1))
            except PendingShardLimit:
                pending_rejected = storage.pending_count == 2

        corrupt_root = parent / "corrupt"
        corrupt_deleted = False
        with RunStorage(
            corrupt_root,
            quota_bytes=1024**2,
            shard_bytes=32 * 1024,
            max_pending=2,
        ) as storage:
            path = storage.write_shards(compact_games(1))[0]
            path.write_bytes(b"not-an-npz")
            try:
                storage.consume_shard(path)
            except CorruptShard:
                corrupt_deleted = not path.exists()

        sibling = parent / "keep.txt"
        sibling.write_text("keep", encoding="utf-8")
        signal_root = parent / "signal"
        storage = RunStorage(
            signal_root,
            quota_bytes=4096,
            shard_bytes=2048,
            max_pending=2,
        )
        handler = install_cleanup_handlers(storage)[signal.SIGTERM]
        try:
            handler(signal.SIGTERM, None)
        except SystemExit:
            pass
        signal_safe = (
            not signal_root.exists()
            and sibling.read_text(encoding="utf-8") == "keep"
            and parent.exists()
        )

        return {
            "all_shards_within_limit": within_limit,
            "round_trip_games": len(restored),
            "float32_features": float32_features,
            "no_partial_files": no_partials,
            "consumed_deleted_immediately": consumed_deleted,
            "quota_rejected_and_cleaned": quota_rejected,
            "third_pending_rejected": pending_rejected,
            "corrupt_deleted": corrupt_deleted,
            "normal_exit_cleaned": normal_cleaned,
            "signal_cleaned_only_run_root": signal_safe,
        }
