from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .extract import sanitize_visible_observation
from .hashing import canonical_json_bytes, sha256_bytes


DETERMINISTIC_BRANCH_OPTION_TYPES = frozenset({8, 9, 12, 13, 14})


def is_deterministic_branch_option(option: Mapping[str, Any]) -> bool:
    try:
        option_type = int(option.get("type", -1))
    except (TypeError, ValueError):
        return False
    return option_type in DETERMINISTIC_BRANCH_OPTION_TYPES


def consensus_top_signature(
    signatures: Sequence[tuple[int, ...]],
    v1_scores: Sequence[float],
    v2_scores: Sequence[float],
) -> tuple[int, ...] | None:
    if (
        not signatures
        or len(signatures) != len(v1_scores)
        or len(signatures) != len(v2_scores)
    ):
        raise ValueError("signatures and scores must have equal non-zero length")
    if len(set(signatures)) != len(signatures):
        raise ValueError("option signatures must be unique")

    def unique_top(scores: Sequence[float]) -> int | None:
        values = [float(score) for score in scores]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("scores must be finite")
        maximum = max(values)
        indices = [index for index, value in enumerate(values) if value == maximum]
        return indices[0] if len(indices) == 1 else None

    first = unique_top(v1_scores)
    second = unique_top(v2_scores)
    if first is None or second is None:
        return None
    return signatures[first] if signatures[first] == signatures[second] else None


@dataclass(frozen=True, slots=True)
class CoinSchedule:
    values: tuple[bool, ...]

    @classmethod
    def from_decision(
        cls,
        snapshot_id: str,
        decision_id: str,
        *,
        count: int = 256,
    ) -> CoinSchedule:
        if count <= 0:
            raise ValueError("coin schedule count must be positive")
        values: list[bool] = []
        block = 0
        while len(values) < count:
            digest = hashlib.sha256(
                f"{snapshot_id}|{decision_id}|rank-coins-v1|{block}".encode()
            ).digest()
            for byte in digest:
                for bit in range(8):
                    values.append(bool(byte & (1 << bit)))
                    if len(values) == count:
                        break
                if len(values) == count:
                    break
            block += 1
        return cls(tuple(values))


@dataclass(frozen=True, slots=True)
class BranchLeaf:
    option_signature: tuple[int, ...]
    leaf_visible_sha256: str
    v1_score: float
    v2_score: float
    stopped_reason: str


def recorded_next_observation(
    step: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    active = [
        record["observation"]
        for record in step[:2]
        if record.get("status") == "ACTIVE"
        and isinstance(record.get("observation"), Mapping)
    ]
    if len(active) != 1:
        raise ValueError(
            f"expected one active recorded observation, found {len(active)}"
        )
    return active[0]


def stratified_root_allocation(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    snapshot_id: str,
) -> list[Mapping[str, Any]]:
    if count <= 0:
        raise ValueError("allocation count must be positive")
    if len(rows) < count:
        raise ValueError("fewer eligible rows than requested roots")
    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        decision_id = str(row["decision_id"])
        if decision_id in seen:
            raise ValueError(f"duplicate decision_id: {decision_id}")
        seen.add(decision_id)
        digest = hashlib.sha256(
            f"{snapshot_id}|{decision_id}|rank-root-allocation-v1".encode()
        ).hexdigest()
        grouped[str(row["stratum"])].append((digest, row))
    for candidates in grouped.values():
        candidates.sort(key=lambda item: item[0])

    selected: list[tuple[str, Mapping[str, Any]]] = []
    round_index = 0
    while len(selected) < count:
        round_rows = [
            candidates[round_index]
            for candidates in grouped.values()
            if round_index < len(candidates)
        ]
        if not round_rows:
            break
        round_rows.sort(key=lambda item: item[0])
        selected.extend(round_rows[: count - len(selected)])
        round_index += 1
    if len(selected) != count:
        raise ValueError("could not complete stratified allocation")
    return [row for _, row in selected]


def _normalize_engine_value(value: object) -> object:
    if isinstance(value, Mapping):
        is_card = "id" in value and "serial" in value
        normalized = {}
        for key, child in value.items():
            if key in {
                "logs",
                "remainingOverageTime",
                "search_begin_input",
                "serial",
                "step",
            }:
                continue
            if child is None:
                continue
            if is_card and key == "playerIndex":
                continue
            normalized[str(key)] = _normalize_engine_value(child)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_engine_value(child) for child in value]
    return value


def normalized_visible_engine_observation(
    observation: Mapping[str, Any],
    acting_seat: int,
) -> dict[str, Any]:
    visible = sanitize_visible_observation(observation, acting_seat)
    normalized = _normalize_engine_value(visible)
    if not isinstance(normalized, dict):
        raise ValueError("normalized observation must be an object")
    return normalized


def visible_engine_sha256(
    observation: Mapping[str, Any],
    acting_seat: int,
) -> str:
    normalized = normalized_visible_engine_observation(observation, acting_seat)
    return sha256_bytes(canonical_json_bytes(normalized))


def compare_branch_orders(
    forward: Sequence[BranchLeaf],
    reverse: Sequence[BranchLeaf],
    *,
    tolerance: float = 1e-9,
) -> tuple[str, ...]:
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    errors: list[str] = []

    def keyed(
        rows: Sequence[BranchLeaf],
        label: str,
    ) -> dict[tuple[int, ...], BranchLeaf]:
        result = {}
        for row in rows:
            if row.option_signature in result:
                errors.append(f"{label} duplicate signature {row.option_signature}")
            result[row.option_signature] = row
        return result

    first = keyed(forward, "forward")
    second = keyed(reverse, "reverse")
    for signature in sorted(set(first) | set(second)):
        if signature not in first:
            errors.append(f"forward missing signature {signature}")
            continue
        if signature not in second:
            errors.append(f"reverse missing signature {signature}")
            continue
        left = first[signature]
        right = second[signature]
        if left.leaf_visible_sha256 != right.leaf_visible_sha256:
            errors.append(f"{signature} leaf_visible_sha256 mismatch")
        if left.stopped_reason != right.stopped_reason:
            errors.append(f"{signature} stopped_reason mismatch")
        for field in ("v1_score", "v2_score"):
            left_value = float(getattr(left, field))
            right_value = float(getattr(right, field))
            if (
                not math.isfinite(left_value)
                or not math.isfinite(right_value)
                or abs(left_value - right_value) > tolerance
            ):
                errors.append(f"{signature} {field} mismatch")
    return tuple(errors)
