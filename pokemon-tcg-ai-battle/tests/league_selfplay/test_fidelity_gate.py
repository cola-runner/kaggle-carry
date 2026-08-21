from __future__ import annotations

from types import SimpleNamespace

from league_selfplay.contracts import MemberId
from league_selfplay.fidelity_gate import decide_fidelity


def _member(nll: float, agreement: float) -> SimpleNamespace:
    return SimpleNamespace(
        negative_log_probability=nll,
        exact_agreement=agreement,
    )


def _metrics(
    nll: float,
    agreement: float,
    members: dict[MemberId, SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        negative_log_probability=nll,
        exact_agreement=agreement,
        members=members
        or {member: _member(nll, agreement) for member in MemberId},
    )


def test_clear_relative_improvement_passes() -> None:
    decision = decide_fidelity(
        _metrics(1.0, 0.50),
        _metrics(0.80, 0.58),
        [],
    )

    assert decision.passed is True
    assert decision.code == "PASS_DRIVER_FIDELITY_V2"
    assert decision.failed_conditions == ()


def test_one_bad_member_rejects_even_when_group_improves() -> None:
    v1 = _metrics(1.0, 0.50)
    v2_members = {
        member: _member(0.70, 0.64)
        for member in MemberId
    }
    v2_members[MemberId.LUCARIO] = _member(1.10, 0.40)

    decision = decide_fidelity(v1, _metrics(0.80, 0.58, v2_members), [])

    assert decision.passed is False
    assert decision.code == "REJECT_DRIVER_FIDELITY"
    assert "member_nll_regression:lucario" in decision.failed_conditions


def test_only_two_improved_members_is_rejected() -> None:
    v1 = _metrics(1.0, 0.50)
    v2_members = {
        MemberId.GRIMMSNARL: _member(0.70, 0.65),
        MemberId.LUCARIO: _member(0.70, 0.65),
        MemberId.CRUSTLE: _member(1.00, 0.50),
        MemberId.ALAKAZAM: _member(1.00, 0.50),
    }

    decision = decide_fidelity(v1, _metrics(0.80, 0.58, v2_members), [])

    assert decision.passed is False
    assert "fewer_than_three_members_improved" in decision.failed_conditions


def test_external_failure_always_rejects() -> None:
    decision = decide_fidelity(
        _metrics(1.0, 0.50),
        _metrics(0.80, 0.58),
        ["numpy parity failed"],
    )

    assert decision.passed is False
    assert "failure:numpy parity failed" in decision.failed_conditions
