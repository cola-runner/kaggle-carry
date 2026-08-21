from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    GameProvenance,
    InvalidSelfPlay,
    MemberId,
    audit_training_batch,
)


@dataclass(frozen=True, slots=True)
class JudgeResult:
    member: MemberId
    opponent: str
    seat: int
    score: float

    def __post_init__(self) -> None:
        if self.seat not in (0, 1):
            raise ValueError("seat must be zero or one")
        if self.score not in (0.0, 0.5, 1.0):
            raise ValueError("judge score must be win/draw/loss encoded as 1/0.5/0")


@dataclass(frozen=True, slots=True)
class GroupComparison:
    start_score: float
    final_score: float
    delta: float
    confidence_low: float
    confidence_high: float
    ancestry_score: float
    per_member_delta: dict[MemberId, float]
    start_games: int
    final_games: int
    ancestry_games: int
    bootstrap_resamples: int
    paired_randomness: bool = False


@dataclass(frozen=True, slots=True)
class LeagueDecision:
    passed: bool
    code: str
    reasons: tuple[str, ...]


def validate_training_isolation(records: Sequence[GameProvenance]) -> None:
    audit = audit_training_batch(records, set(MemberId))
    if not audit.valid:
        raise InvalidSelfPlay("; ".join(audit.reasons))


def _scores(results: Sequence[JudgeResult], name: str) -> np.ndarray:
    if not results:
        raise ValueError(f"{name} results are empty")
    present = {result.member for result in results}
    missing = set(MemberId) - present
    if missing:
        names = ", ".join(sorted(member.value for member in missing))
        raise ValueError(f"{name} results are missing members: {names}")
    return np.asarray([result.score for result in results], dtype=np.float64)


def _support(results: Sequence[JudgeResult]) -> Counter[tuple[MemberId, str, int]]:
    return Counter((result.member, result.opponent, result.seat) for result in results)


def _unpaired_interval(
    start: np.ndarray,
    final: np.ndarray,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    chunk_size = 512
    for offset in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - offset)
        final_indices = rng.integers(0, len(final), size=(count, len(final)))
        start_indices = rng.integers(0, len(start), size=(count, len(start)))
        differences[offset : offset + count] = (
            final[final_indices].mean(axis=1) - start[start_indices].mean(axis=1)
        )
    low, high = np.quantile(differences, (0.025, 0.975))
    return float(low), float(high)


def compare_groups(
    start_results: Sequence[JudgeResult],
    final_results: Sequence[JudgeResult],
    ancestry_results: Sequence[JudgeResult],
    seed: int,
    *,
    bootstrap_resamples: int = 20_000,
) -> GroupComparison:
    if bootstrap_resamples < 1000:
        raise ValueError("bootstrap_resamples must be at least 1000")
    if _support(start_results) != _support(final_results):
        raise ValueError("start and final groups must use the same judge/seat schedule")
    start = _scores(start_results, "start")
    final = _scores(final_results, "final")
    ancestry = _scores(ancestry_results, "ancestry")
    confidence_low, confidence_high = _unpaired_interval(
        start, final, seed, bootstrap_resamples
    )
    per_member_delta: dict[MemberId, float] = {}
    for member in MemberId:
        start_member = np.asarray(
            [result.score for result in start_results if result.member is member],
            dtype=np.float64,
        )
        final_member = np.asarray(
            [result.score for result in final_results if result.member is member],
            dtype=np.float64,
        )
        per_member_delta[member] = float(final_member.mean() - start_member.mean())
    start_score = float(start.mean())
    final_score = float(final.mean())
    return GroupComparison(
        start_score=start_score,
        final_score=final_score,
        delta=final_score - start_score,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        ancestry_score=float(ancestry.mean()),
        per_member_delta=per_member_delta,
        start_games=len(start_results),
        final_games=len(final_results),
        ancestry_games=len(ancestry_results),
        bootstrap_resamples=bootstrap_resamples,
        paired_randomness=False,
    )


def _valid_updates(update_stats: Mapping[MemberId, Any]) -> bool:
    if set(update_stats) != set(MemberId):
        return False
    for member in MemberId:
        stats = update_stats[member]
        delta = float(getattr(stats, "parameter_delta_l2", float("nan")))
        if not bool(getattr(stats, "all_finite", False)):
            return False
        if not math.isfinite(delta) or delta <= 0:
            return False
    return True


def decide_validation(
    comparison: GroupComparison,
    update_stats: Mapping[MemberId, Any],
    failures: Sequence[str],
) -> LeagueDecision:
    if failures:
        return LeagueDecision(False, "REJECT_FAILURE", tuple(str(item) for item in failures))
    if not _valid_updates(update_stats):
        return LeagueDecision(
            False,
            "REJECT_INVALID_UPDATE",
            ("all four policies must have positive finite parameter updates",),
        )
    if comparison.delta < 0.05 or comparison.confidence_low <= 0.0:
        return LeagueDecision(
            False,
            "REJECT_NO_GROUP_IMPROVEMENT",
            (
                "judge delta must be at least +0.05 with a positive 95% lower bound",
            ),
        )
    if comparison.ancestry_score < 0.55:
        return LeagueDecision(
            False,
            "REJECT_ANCESTRY",
            ("final population must score at least 0.55 against round 0",),
        )
    collapsed = [
        member.value
        for member, delta in comparison.per_member_delta.items()
        if delta < -0.10
    ]
    if collapsed:
        return LeagueDecision(
            False,
            "REJECT_MEMBER_COLLAPSE",
            ("judge regression below -0.10: " + ", ".join(sorted(collapsed)),),
        )
    return LeagueDecision(True, "PASS_MAC_LEAGUE", ())
