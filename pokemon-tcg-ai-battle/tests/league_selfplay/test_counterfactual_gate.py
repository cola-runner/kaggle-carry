from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from league_selfplay.contracts import MemberId
from league_selfplay.counterfactual_gate import (
    AAProbe,
    ENGINE22_DYLIB_SHA256,
    AARolloutRecord,
    GateDependencies,
    branch_execution_order,
    decide_aa_gate,
    manual_coin_action,
    run_aa_rollout,
    run_counterfactual_aa_gate,
)
from scripts.run_counterfactual_aa_gate import write_report


def _record(
    immediate: bool,
    terminal: bool,
    *,
    error: str = "",
) -> AARolloutRecord:
    return AARolloutRecord(
        member=MemberId.GRIMMSNARL,
        sample_index=0,
        immediate_digest_match=immediate,
        terminal_score_match=terminal,
        left_result=0,
        right_result=0 if terminal else 1,
        rollout_steps=10,
        branch_order=(0, 1),
        controlled_coin_choices=0,
        error=error,
    )


def test_manual_coin_action_maps_schedule_to_yes_and_no_regardless_of_order() -> None:
    select = {
        "context": 46,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": 2}, {"type": 1}],
    }
    assert manual_coin_action(select, True) == [1]
    assert manual_coin_action(select, False) == [0]


def test_manual_coin_action_rejects_non_coin_and_malformed_prompts() -> None:
    non_coin = {
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": 1}, {"type": 2}],
    }
    malformed = {
        "context": 46,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": 1}, {"type": 1}],
    }
    for select in (non_coin, malformed):
        try:
            manual_coin_action(select, True)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid manual-coin prompt must be rejected")


def test_branch_execution_order_alternates() -> None:
    assert branch_execution_order(0) == (0, 1)
    assert branch_execution_order(1) == (1, 0)
    assert branch_execution_order(2) == (0, 1)


def test_gate_requires_at_least_ninety_five_percent_on_both_signals() -> None:
    passing = [_record(True, True) for _ in range(20)]
    passing[-1] = _record(True, False)
    decision = decide_aa_gate(passing)
    assert decision.code == "PASS_COUNTERFACTUAL_AA"
    assert decision.terminal_agreement == 0.95

    passing[-2] = _record(False, True)
    passing[-3] = _record(False, True)
    decision = decide_aa_gate(passing)
    assert decision.code == "REJECT_COUNTERFACTUAL_ENGINE"
    assert decision.immediate_agreement == 0.90
    assert decision.terminal_agreement == 0.95


def test_twelve_sample_gate_rejects_one_mismatch() -> None:
    records = [_record(True, True) for _ in range(12)]
    assert decide_aa_gate(records).passed
    records[-1] = _record(True, False)
    assert not decide_aa_gate(records).passed


def test_any_operational_failure_rejects() -> None:
    records = [_record(True, True), _record(True, True, error="timeout")]
    decision = decide_aa_gate(records)
    assert decision.code == "REJECT_COUNTERFACTUAL_ENGINE"
    assert decision.reasons == ("timeout",)


def test_gate_rejects_empty_records_and_invalid_threshold() -> None:
    for records, threshold in (([], 0.95), ([_record(True, True)], 0.0)):
        try:
            decide_aa_gate(records, threshold=threshold)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid gate input must be rejected")


@dataclass
class _FakeObservation:
    current: dict[str, Any]
    select: dict[str, Any]
    logs: list[dict[str, Any]]
    search_begin_input: str | None = None


@dataclass
class _FakeSearchState:
    observation: _FakeObservation
    searchId: int


class _FakeSearch:
    def __init__(
        self,
        terminal_results: tuple[int, int],
        *,
        mismatch_immediate: bool = False,
    ) -> None:
        self.terminal_results = terminal_results
        self.mismatch_immediate = mismatch_immediate
        self.search_begin_calls = 0
        self.search_end_calls = 0
        self.created_ids: list[int] = []
        self.released_ids: list[int] = []
        self._lineage: dict[int, int] = {}
        self._depth: dict[int, int] = {}
        self._next_id = 1
        self.api = self

    @staticmethod
    def _observation(result: int, marker: int = 0) -> _FakeObservation:
        return _FakeObservation(
            current={
                "yourIndex": 0,
                "result": result,
                "turn": 1,
                "turnActionCount": marker,
                "players": [{"hand": [], "prize": []}, {"hand": None, "prize": []}],
            },
            select={
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 14}],
            },
            logs=[],
        )

    def search_begin(self, observation: Any, **kwargs: Any) -> _FakeSearchState:
        del observation, kwargs
        branch = self.search_begin_calls
        self.search_begin_calls += 1
        search_id = self._next_id
        self._next_id += 1
        self.created_ids.append(search_id)
        self._lineage[search_id] = branch
        self._depth[search_id] = 0
        return _FakeSearchState(self._observation(-1), search_id)

    def search_step(self, search_id: int, action: list[int]) -> _FakeSearchState:
        assert action == [0]
        branch = self._lineage[search_id]
        depth = self._depth[search_id] + 1
        child_id = self._next_id
        self._next_id += 1
        self.created_ids.append(child_id)
        self._lineage[child_id] = branch
        self._depth[child_id] = depth
        marker = branch if self.mismatch_immediate and depth == 1 else depth
        result = self.terminal_results[branch] if depth >= 2 else -1
        return _FakeSearchState(self._observation(result, marker), child_id)

    def search_release(self, search_id: int) -> None:
        self.released_ids.append(search_id)

    def search_end(self) -> None:
        self.search_end_calls += 1


def _probe() -> AAProbe:
    return AAProbe(
        member=MemberId.GRIMMSNARL,
        sample_index=0,
        decision_id="grimmsnarl:0",
        typed_observation=object(),
        hidden_kwargs={
            "your_deck": [],
            "your_prize": [],
            "opponent_deck": [],
            "opponent_prize": [],
            "opponent_hand": [],
            "opponent_active": [],
        },
        action=(0,),
    )


def test_aa_rollout_uses_two_roots_and_releases_every_created_state() -> None:
    fake = _FakeSearch(terminal_results=(0, 0))
    record = run_aa_rollout(_probe(), fake.api, lambda observation: [0])
    assert fake.search_begin_calls == 2
    assert record.immediate_digest_match
    assert record.terminal_score_match
    assert set(fake.created_ids) <= set(fake.released_ids)
    assert fake.search_end_calls == 1


def test_aa_rollout_records_terminal_divergence_instead_of_hiding_it() -> None:
    fake = _FakeSearch(terminal_results=(0, 1))
    record = run_aa_rollout(_probe(), fake.api, lambda observation: [0])
    assert record.immediate_digest_match
    assert not record.terminal_score_match
    assert (record.left_result, record.right_result) == (0, 1)


def test_aa_rollout_continues_after_state_divergence_to_compare_terminal() -> None:
    fake = _FakeSearch(terminal_results=(0, 0), mismatch_immediate=True)
    record = run_aa_rollout(_probe(), fake.api, lambda observation: [0])
    assert not record.immediate_digest_match
    assert record.terminal_score_match
    assert record.intermediate_digest_mismatches >= 1
    assert record.error == ""
    assert set(fake.created_ids) <= set(fake.released_ids)


def _passing_records(samples_per_member: int) -> tuple[AARolloutRecord, ...]:
    records = []
    sample_index = 0
    for member in MemberId:
        for _ in range(samples_per_member):
            records.append(
                AARolloutRecord(
                    member=member,
                    sample_index=sample_index,
                    immediate_digest_match=True,
                    terminal_score_match=True,
                    left_result=0,
                    right_result=0,
                    rollout_steps=12,
                    branch_order=branch_execution_order(sample_index),
                    controlled_coin_choices=1,
                )
            )
            sample_index += 1
    return tuple(records)


def _dependencies(
    records: tuple[AARolloutRecord, ...],
    *,
    engine_sha256: str = ENGINE22_DYLIB_SHA256,
) -> GateDependencies:
    return GateDependencies(
        engine_sha256=lambda project_root: engine_sha256,
        collect_records=lambda project_root, samples, deadline: records,
    )


def test_runner_collects_three_samples_per_member_and_does_not_touch_artifacts(
    tmp_path: Any,
) -> None:
    report = run_counterfactual_aa_gate(
        tmp_path,
        samples_per_member=3,
        wall_time_seconds=180,
        dependencies=_dependencies(_passing_records(3)),
    )
    assert report.sample_counts == {member.value: 3 for member in MemberId}
    assert len(report.records) == 12
    assert report.decision.code == "PASS_COUNTERFACTUAL_AA"
    assert report.artifacts_before == report.artifacts_after
    assert report.temp_exists_after_cleanup is False
    assert not report.storage_root.exists()


def test_report_rejects_missing_samples() -> None:
    records = _passing_records(3)[:-1]
    report = run_counterfactual_aa_gate(
        "/tmp/fake-project",
        samples_per_member=3,
        dependencies=_dependencies(records),
    )
    assert report.decision.code == "REJECT_COUNTERFACTUAL_ENGINE"
    assert any("sample count" in reason for reason in report.decision.reasons)


def test_report_rejects_engine_hash_mismatch_without_collecting() -> None:
    called = False

    def collect(project_root: Any, samples: int, deadline: Any) -> tuple[Any, ...]:
        nonlocal called
        called = True
        return ()

    report = run_counterfactual_aa_gate(
        "/tmp/fake-project",
        dependencies=GateDependencies(
            engine_sha256=lambda project_root: "wrong",
            collect_records=collect,
        ),
    )
    assert report.decision.code == "REJECT_COUNTERFACTUAL_ENGINE"
    assert any("Engine 22" in reason for reason in report.decision.reasons)
    assert not called


def test_cli_report_is_compact_and_contains_no_hidden_state(tmp_path: Any) -> None:
    report = run_counterfactual_aa_gate(
        tmp_path,
        samples_per_member=3,
        dependencies=_dependencies(_passing_records(3)),
    )
    output = tmp_path / "report.json"
    size = write_report(report, output)
    payload = output.read_text(encoding="utf-8")
    assert size < 100 * 1024
    assert "typed_observation" not in payload
    assert "hidden_kwargs" not in payload
    assert "your_deck" not in payload
