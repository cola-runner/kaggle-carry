from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from rolling_policy.branching import CoinSchedule, visible_engine_sha256

from .contracts import MemberId


MANUAL_COIN_CONTEXT = 46
YES_OPTION_TYPE = 1
NO_OPTION_TYPE = 2
ENGINE22_DYLIB_SHA256 = (
    "7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30"
)
AA_THRESHOLD = 0.95


@dataclass(frozen=True, slots=True)
class AARolloutRecord:
    member: MemberId
    sample_index: int
    immediate_digest_match: bool
    terminal_score_match: bool
    left_result: int | None
    right_result: int | None
    rollout_steps: int
    branch_order: tuple[int, int]
    controlled_coin_choices: int
    intermediate_digest_mismatches: int = 0
    error: str = ""


@dataclass(frozen=True, slots=True)
class AAGateDecision:
    passed: bool
    code: str
    immediate_agreement: float
    terminal_agreement: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AAProbe:
    member: MemberId
    sample_index: int
    decision_id: str
    typed_observation: Any
    hidden_kwargs: Mapping[str, Sequence[int]]
    action: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TreeMeasurement:
    files: int
    bytes: int


@dataclass(frozen=True, slots=True)
class GateDependencies:
    engine_sha256: Callable[[Path], str]
    collect_records: Callable[
        [Path, int, Callable[[], None]],
        Sequence[AARolloutRecord],
    ]


@dataclass(frozen=True, slots=True)
class AAGateReport:
    engine_sha256: str
    expected_engine_sha256: str
    samples_per_member: int
    sample_counts: dict[str, int]
    records: tuple[AARolloutRecord, ...]
    decision: AAGateDecision
    failures: tuple[str, ...]
    artifacts_before: TreeMeasurement
    artifacts_after: TreeMeasurement
    storage_root: Path
    temp_exists_after_cleanup: bool
    elapsed_seconds: float
    wall_time_seconds: int

    def to_dict(self) -> dict[str, Any]:
        value = _jsonable(self)
        if not isinstance(value, dict):
            raise RuntimeError("gate report must serialize to an object")
        return value

    def summary(self) -> dict[str, Any]:
        return {
            "decision": self.decision.code,
            "samples": len(self.records),
            "sample_counts": self.sample_counts,
            "immediate_agreement": self.decision.immediate_agreement,
            "terminal_agreement": self.decision.terminal_agreement,
            "failures": list(self.failures),
            "elapsed_seconds": self.elapsed_seconds,
            "temp_exists_after_cleanup": self.temp_exists_after_cleanup,
        }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value


def _measure_tree(root: Path) -> TreeMeasurement:
    files = 0
    total_bytes = 0
    if root.exists():
        for directory, _, names in os.walk(root):
            for name in names:
                path = Path(directory) / name
                files += 1
                total_bytes += path.stat().st_size
    return TreeMeasurement(files, total_bytes)


def _engine22_sha256(project_root: Path) -> str:
    library = (
        project_root
        / "data/raw/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.dylib"
    )
    if not library.is_file():
        raise FileNotFoundError(f"official Engine 22 library is missing: {library}")
    digest = hashlib.sha256()
    with library.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manual_coin_action(select: Mapping[str, Any], head: bool) -> list[int]:
    if int(select.get("context", -1)) != MANUAL_COIN_CONTEXT:
        raise ValueError("selection is not a manual coin prompt")
    if int(select.get("minCount", -1)) != 1 or int(select.get("maxCount", -1)) != 1:
        raise ValueError("manual coin prompt must select exactly one option")
    options = select.get("option")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError("manual coin prompt has no options")
    wanted = YES_OPTION_TYPE if head else NO_OPTION_TYPE
    matching = [
        index
        for index, option in enumerate(options)
        if isinstance(option, Mapping) and int(option.get("type", -1)) == wanted
    ]
    yes_count = sum(
        isinstance(option, Mapping)
        and int(option.get("type", -1)) == YES_OPTION_TYPE
        for option in options
    )
    no_count = sum(
        isinstance(option, Mapping)
        and int(option.get("type", -1)) == NO_OPTION_TYPE
        for option in options
    )
    if len(matching) != 1 or yes_count != 1 or no_count != 1:
        raise ValueError("manual coin prompt must contain one YES and one NO option")
    return matching


def branch_execution_order(sample_index: int) -> tuple[int, int]:
    if sample_index < 0:
        raise ValueError("sample index must be non-negative")
    return (0, 1) if sample_index % 2 == 0 else (1, 0)


def _plain_observation(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError("search observation must be an object")
    return dict(value)


def _state_digest(observation: Mapping[str, Any]) -> str:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("search observation has no current state")
    return visible_engine_sha256(observation, int(current.get("yourIndex", -1)))


def _result(observation: Mapping[str, Any]) -> int:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("search observation has no current state")
    return int(current.get("result", -1))


def _step_pair(
    states: Sequence[Any],
    action: Sequence[int],
    order: tuple[int, int],
    search_step: Callable[[int, list[int]], Any],
    created: set[int],
) -> list[Any]:
    next_states: list[Any | None] = [None, None]
    for branch in order:
        child = search_step(int(states[branch].searchId), list(action))
        created.add(int(child.searchId))
        next_states[branch] = child
    if next_states[0] is None or next_states[1] is None:
        raise RuntimeError("paired search did not advance both branches")
    return [next_states[0], next_states[1]]


def _rollout_record(
    probe: AAProbe,
    *,
    immediate: bool,
    terminal: bool,
    results: tuple[int | None, int | None],
    steps: int,
    coins: int,
    mismatches: int,
    error: str = "",
) -> AARolloutRecord:
    return AARolloutRecord(
        member=probe.member,
        sample_index=probe.sample_index,
        immediate_digest_match=immediate,
        terminal_score_match=terminal,
        left_result=results[0],
        right_result=results[1],
        rollout_steps=steps,
        branch_order=branch_execution_order(probe.sample_index),
        controlled_coin_choices=coins,
        intermediate_digest_mismatches=mismatches,
        error=error,
    )


def run_aa_rollout(
    probe: AAProbe,
    api: Any,
    downstream_action: (
        Callable[[Mapping[str, Any]], list[int]]
        | Sequence[Callable[[Mapping[str, Any]], list[int]]]
    ),
    *,
    max_steps: int = 2000,
) -> AARolloutRecord:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    order = branch_execution_order(probe.sample_index)
    roots: list[Any] = []
    created: set[int] = set()
    search_started = False
    immediate = False
    steps = 0
    coin_indices = [0, 0]
    mismatches = 0
    if callable(downstream_action):
        downstream_actions = (downstream_action, downstream_action)
    else:
        downstream_actions = tuple(downstream_action)
        if len(downstream_actions) != 2 or not all(
            callable(action) for action in downstream_actions
        ):
            raise ValueError("two downstream action callbacks are required")
    try:
        for _ in range(2):
            search_started = True
            root = api.search_begin(
                probe.typed_observation,
                manual_coin=True,
                **{
                    name: list(values)
                    for name, values in probe.hidden_kwargs.items()
                },
            )
            roots.append(root)
            created.add(int(root.searchId))

        states = _step_pair(
            roots,
            probe.action,
            order,
            api.search_step,
            created,
        )
        steps = 1
        observations = tuple(
            _plain_observation(state.observation) for state in states
        )
        immediate = _state_digest(observations[0]) == _state_digest(observations[1])
        mismatches = int(not immediate)

        schedule = CoinSchedule.from_decision("engine22-aa", probe.decision_id)
        while True:
            observations = tuple(
                _plain_observation(state.observation) for state in states
            )
            results = (_result(observations[0]), _result(observations[1]))
            if results[0] >= 0 and results[1] >= 0:
                return _rollout_record(
                    probe,
                    immediate=immediate,
                    terminal=results[0] == results[1],
                    results=results,
                    steps=steps,
                    coins=sum(coin_indices),
                    mismatches=mismatches,
                )

            if steps >= max_steps:
                return _rollout_record(
                    probe,
                    immediate=immediate,
                    terminal=False,
                    results=tuple(
                        result if result >= 0 else None for result in results
                    ),
                    steps=steps,
                    coins=sum(coin_indices),
                    mismatches=mismatches,
                    error="timeout",
                )

            active = tuple(branch for branch in (0, 1) if results[branch] < 0)
            digests_match = (
                len(active) == 2
                and _state_digest(observations[0])
                == _state_digest(observations[1])
            )
            if len(active) != 2 or not digests_match:
                mismatches += 1

            actions: dict[int, list[int]] = {}
            for branch in active:
                select = observations[branch].get("select")
                if not isinstance(select, Mapping):
                    raise ValueError(
                        "non-terminal search observation has no selection"
                    )
                if int(select.get("context", -1)) == MANUAL_COIN_CONTEXT:
                    coin_index = coin_indices[branch]
                    if coin_index >= len(schedule.values):
                        raise RuntimeError("manual coin schedule exhausted")
                    actions[branch] = manual_coin_action(
                        select,
                        schedule.values[coin_index],
                    )
                    coin_indices[branch] += 1
                else:
                    actions[branch] = downstream_actions[branch](
                        observations[branch]
                    )
            if digests_match and actions.get(0) != actions.get(1):
                return _rollout_record(
                    probe,
                    immediate=immediate,
                    terminal=False,
                    results=(None, None),
                    steps=steps,
                    coins=sum(coin_indices),
                    mismatches=mismatches,
                    error="downstream policy mismatch on identical state",
                )

            next_states = list(states)
            for branch in order:
                if branch not in actions:
                    continue
                child = api.search_step(
                    int(states[branch].searchId),
                    list(actions[branch]),
                )
                created.add(int(child.searchId))
                next_states[branch] = child
            states = next_states
            steps += 1
    except Exception as error:
        return _rollout_record(
            probe,
            immediate=immediate,
            terminal=False,
            results=(None, None),
            steps=steps,
            coins=sum(coin_indices),
            mismatches=mismatches,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        for search_id in sorted(created, reverse=True):
            try:
                api.search_release(search_id)
            except Exception:
                pass
        if search_started:
            api.search_end()


def decide_aa_gate(
    records: Sequence[AARolloutRecord],
    *,
    threshold: float = 0.95,
) -> AAGateDecision:
    if not records:
        raise ValueError("counterfactual gate requires at least one record")
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be finite and in (0, 1]")

    immediate = sum(row.immediate_digest_match for row in records) / len(records)
    terminal = sum(row.terminal_score_match for row in records) / len(records)
    errors = tuple(row.error for row in records if row.error)
    passed = not errors and immediate >= threshold and terminal >= threshold
    reasons: list[str] = list(errors)
    if immediate < threshold:
        reasons.append(
            f"immediate agreement {immediate:.6f} is below {threshold:.6f}"
        )
    if terminal < threshold:
        reasons.append(
            f"terminal agreement {terminal:.6f} is below {threshold:.6f}"
        )
    return AAGateDecision(
        passed=passed,
        code=(
            "PASS_COUNTERFACTUAL_AA"
            if passed
            else "REJECT_COUNTERFACTUAL_ENGINE"
        ),
        immediate_agreement=immediate,
        terminal_agreement=terminal,
        reasons=tuple(reasons),
    )


def _card_ids(cards: object) -> list[int]:
    if not isinstance(cards, list):
        return []
    return [
        int(card.get("id") or 0)
        for card in cards
        if isinstance(card, Mapping) and int(card.get("id") or 0) > 0
    ]


def _exact_hidden_kwargs(
    game_api: Any,
    observation: Mapping[str, Any],
) -> dict[str, list[int]]:
    frames = json.loads(game_api.visualize_data())
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("engine returned no visualization frame")
    frame = frames[-1]
    current = frame.get("current") if isinstance(frame, Mapping) else None
    visible_current = observation.get("current")
    if not isinstance(current, Mapping) or not isinstance(visible_current, Mapping):
        raise RuntimeError("visualization frame has no current state")
    players = current.get("players")
    visible_players = visible_current.get("players")
    if (
        not isinstance(players, list)
        or len(players) < 2
        or not isinstance(visible_players, list)
        or len(visible_players) < 2
    ):
        raise RuntimeError("visualization frame has invalid players")
    seat = int(visible_current.get("yourIndex", -1))
    if seat not in (0, 1):
        raise RuntimeError("observation has invalid acting seat")
    me = players[seat]
    opponent = players[1 - seat]
    if not isinstance(me, Mapping) or not isinstance(opponent, Mapping):
        raise RuntimeError("visualization player state is invalid")
    visible_opponent = visible_players[1 - seat]
    if not isinstance(visible_opponent, Mapping):
        raise RuntimeError("visible opponent state is invalid")
    opponent_active: list[int] = []
    active = visible_opponent.get("active")
    if isinstance(active, list) and active and active[0] is None:
        opponent_active = _card_ids(opponent.get("active"))[:1]
    return {
        "your_deck": _card_ids(me.get("deck")),
        "your_prize": _card_ids(me.get("prize")),
        "opponent_deck": _card_ids(opponent.get("deck")),
        "opponent_prize": _card_ids(opponent.get("prize")),
        "opponent_hand": _card_ids(opponent.get("hand")),
        "opponent_active": opponent_active,
    }


def _clone_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(observation))


def _eligible_probe(observation: Mapping[str, Any]) -> bool:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return False
    options = select.get("option")
    return (
        int(select.get("type", -1)) == 0
        and int(select.get("context", -1)) == 0
        and int(select.get("minCount", -1)) == 1
        and int(select.get("maxCount", -1)) == 1
        and isinstance(options, list)
        and len(options) > 1
    )


def _collect_engine22_records(
    project_root: Path,
    samples_per_member: int,
    deadline_check: Callable[[], None],
) -> tuple[AARolloutRecord, ...]:
    from scripts.run_local_match import import_official_cg

    from .bootstrap import DriverRegistry

    game_api = import_official_cg(project_root)
    from cg import api as search_api  # type: ignore

    pair_schedule = (
        (MemberId.GRIMMSNARL, MemberId.LUCARIO),
        (MemberId.LUCARIO, MemberId.GRIMMSNARL),
        (MemberId.CRUSTLE, MemberId.ALAKAZAM),
        (MemberId.ALAKAZAM, MemberId.CRUSTLE),
    )
    counts = {member: 0 for member in MemberId}
    records: list[AARolloutRecord] = []
    registry = DriverRegistry.from_project(project_root)
    try:
        for members in pair_schedule:
            if all(counts[member] >= samples_per_member for member in members):
                continue
            observation, start_data = game_api.battle_start(
                registry.deck(members[0]),
                registry.deck(members[1]),
            )
            if observation is None:
                raise RuntimeError(
                    "battle_start failed: "
                    f"errorPlayer={start_data.errorPlayer}, "
                    f"errorType={start_data.errorType}"
                )
            history: list[tuple[MemberId, dict[str, Any], list[int]]] = []
            try:
                for step in range(2000):
                    deadline_check()
                    current = observation.get("current")
                    if not isinstance(current, Mapping):
                        raise RuntimeError("engine returned no current state")
                    if int(current.get("result", -1)) >= 0:
                        break
                    seat = int(current.get("yourIndex", -1))
                    if seat not in (0, 1):
                        raise RuntimeError("engine returned invalid acting seat")
                    member = members[seat]
                    action = registry.action(member, observation)
                    root_observation = _clone_observation(observation)
                    if counts[member] < samples_per_member and _eligible_probe(observation):
                        sample_index = len(records)
                        hidden_kwargs = _exact_hidden_kwargs(game_api, observation)
                        probe = AAProbe(
                            member=member,
                            sample_index=sample_index,
                            decision_id=(
                                f"{member.value}:{sample_index}:"
                                f"{int(current.get('turn', 0))}:{step}"
                            ),
                            typed_observation=search_api.to_observation_class(
                                root_observation
                            ),
                            hidden_kwargs=hidden_kwargs,
                            action=tuple(action),
                        )
                        rollout_registries = [
                            DriverRegistry.from_project(project_root)
                            for _ in range(2)
                        ]
                        try:
                            for rollout_registry in rollout_registries:
                                for past_member, past_observation, past_action in (
                                    history
                                    + [(member, root_observation, list(action))]
                                ):
                                    replayed = rollout_registry.action(
                                        past_member,
                                        _clone_observation(past_observation),
                                    )
                                    if replayed != past_action:
                                        raise RuntimeError(
                                            "driver replay mismatch before "
                                            "counterfactual root"
                                        )

                            def make_downstream_action(
                                rollout_registry: DriverRegistry,
                            ) -> Callable[[Mapping[str, Any]], list[int]]:
                                def decide(
                                    branch_observation: Mapping[str, Any],
                                ) -> list[int]:
                                    branch_current = branch_observation.get(
                                        "current"
                                    )
                                    if not isinstance(branch_current, Mapping):
                                        raise RuntimeError(
                                            "counterfactual branch has no "
                                            "current state"
                                        )
                                    branch_seat = int(
                                        branch_current.get("yourIndex", -1)
                                    )
                                    if branch_seat not in (0, 1):
                                        raise RuntimeError(
                                            "counterfactual branch has invalid "
                                            "acting seat"
                                        )
                                    return rollout_registry.action(
                                        members[branch_seat],
                                        branch_observation,
                                    )

                                return decide

                            record = run_aa_rollout(
                                probe,
                                search_api,
                                tuple(
                                    make_downstream_action(rollout_registry)
                                    for rollout_registry in rollout_registries
                                ),
                            )
                        finally:
                            for rollout_registry in rollout_registries:
                                rollout_registry.close()
                        records.append(record)
                        counts[member] += 1
                        deadline_check()

                    history.append((member, root_observation, list(action)))
                    if all(
                        counts[candidate] >= samples_per_member
                        for candidate in MemberId
                    ):
                        break
                    observation = game_api.battle_select(action)
                else:
                    raise RuntimeError("sampling match exceeded 2000 decisions")
            finally:
                game_api.battle_finish()
            if all(counts[member] >= samples_per_member for member in MemberId):
                break
    finally:
        registry.close()
    return tuple(records)


def run_counterfactual_aa_gate(
    project_root: Path | str,
    *,
    samples_per_member: int = 3,
    wall_time_seconds: int = 180,
    dependencies: GateDependencies | None = None,
) -> AAGateReport:
    if samples_per_member <= 0:
        raise ValueError("samples_per_member must be positive")
    if wall_time_seconds <= 0:
        raise ValueError("wall_time_seconds must be positive")
    root = Path(project_root).resolve()
    dependencies = dependencies or GateDependencies(
        engine_sha256=_engine22_sha256,
        collect_records=_collect_engine22_records,
    )
    started = time.monotonic()
    artifacts_before = _measure_tree(root / "artifacts")
    storage_root = Path(tempfile.gettempdir()).resolve() / (
        f"pokemon-counterfactual-aa-{uuid.uuid4().hex}"
    )
    storage_root.mkdir(parents=False, exist_ok=False)
    records: tuple[AARolloutRecord, ...] = ()
    failures: list[str] = []
    engine_sha256 = ""

    def deadline_check() -> None:
        if time.monotonic() - started > wall_time_seconds:
            raise TimeoutError(
                f"counterfactual gate exceeded {wall_time_seconds} seconds"
            )

    try:
        try:
            engine_sha256 = dependencies.engine_sha256(root)
            if engine_sha256 != ENGINE22_DYLIB_SHA256:
                failures.append(
                    "official Engine 22 SHA-256 mismatch: "
                    f"expected {ENGINE22_DYLIB_SHA256}, got {engine_sha256}"
                )
            else:
                records = tuple(
                    dependencies.collect_records(
                        root,
                        samples_per_member,
                        deadline_check,
                    )
                )
                deadline_check()
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
    finally:
        shutil.rmtree(storage_root, ignore_errors=True)

    temp_exists = storage_root.exists()
    if temp_exists:
        failures.append("temporary gate directory still exists after cleanup")

    sample_counts = {
        member.value: sum(record.member is member for record in records)
        for member in MemberId
    }
    for member in MemberId:
        actual = sample_counts[member.value]
        if actual != samples_per_member:
            failures.append(
                f"{member.value} sample count {actual} != {samples_per_member}"
            )

    artifacts_after = _measure_tree(root / "artifacts")
    if artifacts_after != artifacts_before:
        failures.append("artifacts inventory changed during counterfactual gate")

    if records:
        base_decision = decide_aa_gate(records, threshold=AA_THRESHOLD)
    else:
        base_decision = AAGateDecision(
            passed=False,
            code="REJECT_COUNTERFACTUAL_ENGINE",
            immediate_agreement=0.0,
            terminal_agreement=0.0,
            reasons=("no counterfactual records were collected",),
        )
    reasons = tuple(failures) + base_decision.reasons
    passed = base_decision.passed and not failures
    decision = AAGateDecision(
        passed=passed,
        code=(
            "PASS_COUNTERFACTUAL_AA"
            if passed
            else "REJECT_COUNTERFACTUAL_ENGINE"
        ),
        immediate_agreement=base_decision.immediate_agreement,
        terminal_agreement=base_decision.terminal_agreement,
        reasons=reasons,
    )
    return AAGateReport(
        engine_sha256=engine_sha256,
        expected_engine_sha256=ENGINE22_DYLIB_SHA256,
        samples_per_member=samples_per_member,
        sample_counts=sample_counts,
        records=records,
        decision=decision,
        failures=tuple(failures),
        artifacts_before=artifacts_before,
        artifacts_after=artifacts_after,
        storage_root=storage_root,
        temp_exists_after_cleanup=temp_exists,
        elapsed_seconds=time.monotonic() - started,
        wall_time_seconds=wall_time_seconds,
    )
