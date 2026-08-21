# Pokémon Engine 22 Counterfactual A/A Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a Mac-only gate that accepts terminal counterfactual labels only if two independent Engine 22 rollouts of the same action agree at least 95% of the time.

**Architecture:** A focused `league_selfplay.counterfactual_gate` module owns the report contract, coin scheduling, paired independent-root rollout, real-game state sampling, cleanup, and frozen pass/reject decision. A tiny CLI invokes the module and writes one JSON report. Unit tests use fake observations and fake search functions; the final smoke run uses the four current drivers and the official local Engine 22.

**Tech Stack:** Python 3.12, NumPy, official `cg.game`/`cg.api`, pytest, existing driver loader and visible-state hashing utilities.

## Global Constraints

- Engine: official local Engine 22 only.
- Sample 12 non-forced main decisions, three from each of the four existing drivers.
- Create two fresh `search_begin` roots for every sample; never reuse a mutated root.
- Use `manual_coin=True` and one deterministic schedule for both branches.
- Pass only at 95% or greater exact agreement for both immediate next-state digest and terminal score, with zero runtime, validity, timeout, or randomness-control failures.
- Do not train, export, or submit an agent.
- Do not persist replays, checkpoints, datasets, or raw observations.
- Retain only `reports/mac_counterfactual_aa_engine22_latest.json`, below 100 KiB.
- Use one owned system temporary directory, remove it in all exit paths, and keep `artifacts/` file count and byte size unchanged.
- Enforce a 180-second Mac wall-time limit; timeout rejects the gate.

---

### Task 1: Frozen gate contract and deterministic coin control

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/counterfactual_gate.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_counterfactual_gate.py`

**Interfaces:**
- Consumes: `rolling_policy.branching.CoinSchedule` and `visible_engine_sha256`.
- Produces: `AARolloutRecord`, `AAGateDecision`, `manual_coin_action(select, head)`, `branch_execution_order(index)`, and `decide_aa_gate(records, threshold=0.95)`.

- [ ] **Step 1: Write failing pure-contract tests**

```python
def test_manual_coin_action_maps_schedule_to_yes_and_no_regardless_of_order():
    select = {"context": 46, "minCount": 1, "maxCount": 1,
              "option": [{"type": 2}, {"type": 1}]}
    assert manual_coin_action(select, True) == [1]
    assert manual_coin_action(select, False) == [0]


def test_branch_execution_order_alternates():
    assert branch_execution_order(0) == (0, 1)
    assert branch_execution_order(1) == (1, 0)


def test_gate_requires_at_least_ninety_five_percent_on_both_signals():
    passing = [_record(True, True) for _ in range(20)]
    passing[-1] = _record(True, False)
    assert decide_aa_gate(passing).code == "PASS_COUNTERFACTUAL_AA"
    passing[-2] = _record(False, True)
    assert decide_aa_gate(passing).code == "REJECT_COUNTERFACTUAL_ENGINE"


def test_any_operational_failure_rejects():
    records = [_record(True, True), _record(True, True, error="timeout")]
    assert decide_aa_gate(records).code == "REJECT_COUNTERFACTUAL_ENGINE"
```

- [ ] **Step 2: Run tests and verify the missing module fails**

Run: `python -m pytest tests/league_selfplay/test_counterfactual_gate.py -q`

Expected: collection fails because `league_selfplay.counterfactual_gate` does not exist.

- [ ] **Step 3: Implement the immutable records and gate calculation**

```python
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
    error: str = ""


@dataclass(frozen=True, slots=True)
class AAGateDecision:
    passed: bool
    code: str
    immediate_agreement: float
    terminal_agreement: float
    reasons: tuple[str, ...]


def decide_aa_gate(records, threshold=0.95):
    immediate = sum(row.immediate_digest_match for row in records) / len(records)
    terminal = sum(row.terminal_score_match for row in records) / len(records)
    errors = tuple(row.error for row in records if row.error)
    passed = not errors and immediate >= threshold and terminal >= threshold
    return AAGateDecision(
        passed,
        "PASS_COUNTERFACTUAL_AA" if passed else "REJECT_COUNTERFACTUAL_ENGINE",
        immediate,
        terminal,
        errors,
    )
```

`manual_coin_action` must accept context value `46` only, find exactly one option type `1` (YES) and one option type `2` (NO), and raise on malformed prompts. `branch_execution_order` returns `(0, 1)` for even samples and `(1, 0)` for odd samples.

- [ ] **Step 4: Run the pure tests**

Run: `python -m pytest tests/league_selfplay/test_counterfactual_gate.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/counterfactual_gate.py pokemon-tcg-ai-battle/tests/league_selfplay/test_counterfactual_gate.py
git commit -m "test: freeze counterfactual aa gate"
```

### Task 2: Independent-root rollout with cleanup

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/counterfactual_gate.py`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_counterfactual_gate.py`

**Interfaces:**
- Consumes: two fresh typed root observations, exact hidden lists, one initial action, a shared downstream action callback, `search_begin`, `search_step`, `search_release`, and `search_end`.
- Produces: `run_aa_rollout(...) -> AARolloutRecord`.

- [ ] **Step 1: Add failing fake-engine rollout tests**

```python
def test_aa_rollout_uses_two_roots_and_releases_every_created_state():
    fake = FakeSearch(terminal_results=(0, 0))
    record = run_aa_rollout(_probe(), fake.api, lambda obs: [0])
    assert fake.search_begin_calls == 2
    assert record.immediate_digest_match
    assert record.terminal_score_match
    assert set(fake.created_ids) <= set(fake.released_ids)
    assert fake.search_end_calls == 1


def test_aa_rollout_records_terminal_divergence_instead_of_hiding_it():
    fake = FakeSearch(terminal_results=(0, 1))
    record = run_aa_rollout(_probe(), fake.api, lambda obs: [0])
    assert not record.terminal_score_match
```

The fake API returns dataclass-shaped search states and records creation, step, release, and end calls. It exposes identical first transitions in the first test and distinct terminal results in the second.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/league_selfplay/test_counterfactual_gate.py -q`

Expected: fails because `run_aa_rollout` is not implemented.

- [ ] **Step 3: Implement lockstep independent-root continuation**

```python
def run_aa_rollout(probe, api, downstream_action, *, max_steps=2000):
    roots = [api.search_begin(probe.typed_observation, manual_coin=True,
                              **probe.hidden_kwargs) for _ in range(2)]
    states = list(roots)
    created = {int(root.searchId) for root in roots}
    try:
        states = _step_pair(states, probe.action, probe.branch_order, api.search_step, created)
        immediate = _state_digest(states[0]) == _state_digest(states[1])
        coins = CoinSchedule.from_decision("engine22-aa", probe.decision_id)
        coin_index = 0
        for step in range(1, max_steps + 1):
            left = asdict(states[0].observation)
            right = asdict(states[1].observation)
            if _state_digest(states[0]) != _state_digest(states[1]):
                return _mismatch_record(probe, immediate, step, coin_index)
            results = (_result(left), _result(right))
            if results[0] >= 0 or results[1] >= 0:
                return _terminal_record(probe, immediate, results, step, coin_index)
            select = left["select"]
            action = (
                manual_coin_action(select, coins.values[coin_index])
                if int(select["context"]) == 46
                else downstream_action(left)
            )
            coin_index += int(int(select["context"]) == 46)
            states = _step_pair(states, action, probe.branch_order,
                                api.search_step, created)
        return _error_record(probe, immediate, max_steps, "timeout")
    finally:
        for search_id in sorted(created, reverse=True):
            try:
                api.search_release(search_id)
            except Exception:
                pass
        api.search_end()
```

Before applying one shared downstream action, require the two normalized visible-state digests and legal option payloads to match. A mismatch is recorded immediately and cannot be converted to a draw. All roots and descendants are tracked and released in `finally`.

- [ ] **Step 4: Run focused and existing branching tests**

Run: `python -m pytest tests/league_selfplay/test_counterfactual_gate.py tests/rolling/test_branching.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the rollout core**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/counterfactual_gate.py pokemon-tcg-ai-battle/tests/league_selfplay/test_counterfactual_gate.py
git commit -m "feat: add independent Engine 22 aa rollout"
```

### Task 3: Four-driver sampler, bounded report, and real Mac gate

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/counterfactual_gate.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_counterfactual_aa_gate.py`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_counterfactual_gate.py`
- Create at runtime: `pokemon-tcg-ai-battle/reports/mac_counterfactual_aa_engine22_latest.json`

**Interfaces:**
- Consumes: `DriverRegistry`, `DRIVER_PATHS`, `import_official_cg`, real `visualize_data`, and official `cg.api` search functions.
- Produces: `run_counterfactual_aa_gate(project_root, samples_per_member=3, wall_time_seconds=180) -> AAGateReport` and a CLI with `--project-root` and `--out`.

- [ ] **Step 1: Add failing orchestration tests with fake registry and game**

```python
def test_runner_collects_three_samples_per_member_and_does_not_touch_artifacts(tmp_path):
    report = run_counterfactual_aa_gate(
        tmp_path,
        samples_per_member=3,
        wall_time_seconds=180,
        dependencies=_passing_dependencies(),
    )
    assert report.sample_counts == {member.value: 3 for member in MemberId}
    assert len(report.records) == 12
    assert report.artifacts_before == report.artifacts_after
    assert not Path(report.storage_root).exists()


def test_report_rejects_missing_samples_engine_mismatch_and_oversize_output(tmp_path):
    report = run_counterfactual_aa_gate(
        tmp_path,
        samples_per_member=3,
        dependencies=_dependencies_with_one_engine_failure(),
    )
    assert report.decision.code == "REJECT_COUNTERFACTUAL_ENGINE"
```

The injected dependencies provide small deterministic games and avoid importing `cg` during unit tests.

- [ ] **Step 2: Run orchestration tests and verify they fail**

Run: `python -m pytest tests/league_selfplay/test_counterfactual_gate.py -q`

Expected: fails because the orchestration function and report do not exist.

- [ ] **Step 3: Implement sampling, exact hidden extraction, engine identity, and report**

Use the fixed match schedule `(grimmsnarl, lucario)`, `(lucario, grimmsnarl)`, `(crustle, alakazam)`, `(alakazam, crustle)`. During real play, retain observations only in memory and probe the first three non-forced `MAIN` decisions for each acting member. Extract both players' exact deck, prize, hand, and face-down active IDs from the last `visualize_data()` frame. Prime one fresh downstream driver pair with the in-memory observation history, then use it as the shared action oracle for both independent roots.

The report contains configuration, SHA-256 of the official `cg/libcg.dylib`, per-member sample counts, agreement counts/rates, compact per-sample booleans/results/errors, artifact inventory before/after, temporary-root cleanup status, elapsed seconds, and the frozen decision. It must never contain observations, hidden card IDs, deck lists, action payloads, or coin schedules.

- [ ] **Step 4: Add and test the CLI**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "reports/mac_counterfactual_aa_engine22_latest.json")
    args = parser.parse_args()
    report = run_counterfactual_aa_gate(args.project_root)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if len(payload.encode()) >= 100 * 1024:
        raise SystemExit("report exceeds 100 KiB")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    print(json.dumps(report.summary(), indent=2, sort_keys=True))
```

Run: `python -m pytest tests/league_selfplay/test_counterfactual_gate.py tests/rolling/test_branching.py -q`

Expected: all tests pass.

- [ ] **Step 5: Run the real Engine 22 Mac gate**

Run: `PYTHONDONTWRITEBYTECODE=1 python scripts/run_counterfactual_aa_gate.py`

Expected: completes within 180 seconds, writes exactly one report below 100 KiB, writes no replays/models/datasets, removes its temporary directory, and prints either `PASS_COUNTERFACTUAL_AA` or `REJECT_COUNTERFACTUAL_ENGINE` without interpreting a rejection as a training result.

- [ ] **Step 6: Verify scope and storage**

Run: `python -m pytest tests/league_selfplay/test_counterfactual_gate.py tests/rolling/test_branching.py tests/league_selfplay/test_residual_runner.py -q`

Run: `git status --short && du -h reports/mac_counterfactual_aa_engine22_latest.json`

Expected: tests pass; only the planned code, tests, script, plan, and compact report are new/changed by this work; `artifacts/` is unchanged.

- [ ] **Step 7: Commit implementation and evidence**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/counterfactual_gate.py pokemon-tcg-ai-battle/scripts/run_counterfactual_aa_gate.py pokemon-tcg-ai-battle/tests/league_selfplay/test_counterfactual_gate.py pokemon-tcg-ai-battle/reports/mac_counterfactual_aa_engine22_latest.json
git commit -m "test: gate Engine 22 counterfactual rollouts"
```
