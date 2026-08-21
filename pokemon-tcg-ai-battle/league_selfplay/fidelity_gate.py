from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import MemberId


@dataclass(frozen=True, slots=True)
class FidelityDecision:
    passed: bool
    code: str
    failed_conditions: tuple[str, ...]


def decide_fidelity(
    v1: Any,
    v2: Any,
    failures: Sequence[str],
) -> FidelityDecision:
    failed = [f"failure:{failure}" for failure in failures]
    aggregate_values = (
        float(v1.negative_log_probability),
        float(v2.negative_log_probability),
        float(v1.exact_agreement),
        float(v2.exact_agreement),
    )
    if not all(math.isfinite(value) for value in aggregate_values):
        failed.append("nonfinite_aggregate_metrics")
    if v2.negative_log_probability > 0.90 * v1.negative_log_probability:
        failed.append("aggregate_nll_improvement_below_10_percent")
    if v2.exact_agreement < v1.exact_agreement + 0.05:
        failed.append("aggregate_agreement_improvement_below_5_points")

    if set(v1.members) != set(MemberId) or set(v2.members) != set(MemberId):
        failed.append("member_set_mismatch")
    else:
        improved_members = 0
        for member in MemberId:
            baseline = v1.members[member]
            candidate = v2.members[member]
            if (
                candidate.negative_log_probability
                < baseline.negative_log_probability
                and candidate.exact_agreement > baseline.exact_agreement
            ):
                improved_members += 1
            if (
                candidate.negative_log_probability
                > 1.05 * baseline.negative_log_probability
            ):
                failed.append(f"member_nll_regression:{member.value}")
        if improved_members < 3:
            failed.append("fewer_than_three_members_improved")

    failed_conditions = tuple(dict.fromkeys(failed))
    if failed_conditions:
        return FidelityDecision(
            passed=False,
            code="REJECT_DRIVER_FIDELITY",
            failed_conditions=failed_conditions,
        )
    return FidelityDecision(
        passed=True,
        code="PASS_DRIVER_FIDELITY_V2",
        failed_conditions=(),
    )
