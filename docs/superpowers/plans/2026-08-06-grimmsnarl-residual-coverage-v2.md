# Grimmsnarl Residual Coverage V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a layered Grimmsnarl V2 residual policy that preserves every V1 override and learns additional terminal-outcome-backed interventions from previously untouched online losses.

**Architecture:** A new `coverage_v2` module owns compact replay seed selection, strict counterfactual-label acceptance, V1/V2 layered gating, model loading/training, and three-way evaluation contracts. A single runner downloads official replays into a temporary directory, streams Engine22 rollout aggregates into memory, trains two small V2 scorers, runs a frozen V2/V1/control Mac audit, and retains weights only after every gate passes. Submission packaging is a final conditional task and never runs on a rejected experiment.

**Tech Stack:** Python 3.12, NumPy, PyTorch, official Engine22 `cg.api` search restoration for counterfactual training only, official `cg.game` battle API for evaluation, pytest.

## Global Constraints

- V1 weights, thresholds, and accepted decisions are immutable inputs.
- V2 cannot replace a V1-selected action during the Mac experiment.
- No action is labelled good merely because it came from an online loss; it must win the counterfactual rollout comparison.
- Raw rollouts are streamed into compact aggregate records and deleted after aggregation. Temporary and retained artifacts together must stay below `500 MB` on the Mac.
- The first experiment stays on the Mac. The 1080Ti is used only after the complete Mac gate passes.
- Use Engine22 SHA-256 `7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30`.
- Source online evidence from V1 submission `55255929` and frozen control `55261450`.
- Keep at most one selected uncovered-loss state per episode and at most 24 states in the first Mac run.
- Evaluate at most four root actions per state, including the incumbent.
- Each accepted comparison must complete two rollouts per root action against each of the three non-Grimmsnarl population policies.
- A new preference is accepted only when the alternative is not worse for any continuation policy, is strictly better for at least two policies, and gains at least `0.20` aggregate terminal win rate.
- Do not lower a confidence margin merely to create an override.

---

### Task 1: Compact untouched-loss seed manifest

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/coverage_v2.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_coverage_v2.py`
- Create: `pokemon-tcg-ai-battle/scripts/build_grimmsnarl_coverage_manifest.py`

**Interfaces:**
- Consumes: parsed Kaggle replay dictionaries, target deck fingerprint, loaded control/V1 agents, V1 exported scores and margins.
- Produces: `CoverageSeed`, `ProtectedDecision`, `IncumbentAnchor`, `ReplayCoverage`, `coverage_priority`, `extract_replay_coverage`, and `select_coverage_seeds`.

- [ ] **Step 1: Write failing replay-classification tests**

```python
def test_only_an_untouched_loss_produces_coverage_seeds():
    untouched = _replay(won=False, decisions=[_decision(control=0, v1=0)])
    changed = _replay(won=False, decisions=[_decision(control=0, v1=1)])
    won = _replay(won=True, decisions=[_decision(control=0, v1=0)])
    assert len(extract_replay_coverage(untouched, _agents()).seeds) == 1
    assert extract_replay_coverage(changed, _agents()).seeds == ()
    assert extract_replay_coverage(won, _agents()).seeds == ()
    assert len(extract_replay_coverage(changed, _agents()).protected) == 1
    assert len(extract_replay_coverage(won, _agents()).anchors) == 1


def test_selection_keeps_one_near_margin_state_per_episode():
    rows = [
        _seed("e1", 2, priority=0.4),
        _seed("e1", 8, priority=0.9),
        _seed("e2", 3, priority=0.7),
    ]
    selected = select_coverage_seeds(rows, maximum=24)
    assert [(row.episode_id, row.step_index) for row in selected] == [
        ("e1", 8),
        ("e2", 3),
    ]
```

- [ ] **Step 2: Run the focused tests and confirm the missing module fails**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/league_selfplay/test_coverage_v2.py -q`

Expected: collection fails because `league_selfplay.coverage_v2` does not exist.

- [ ] **Step 3: Implement immutable seed records and deterministic selection**

```python
@dataclass(frozen=True, slots=True)
class CoverageSeed:
    episode_id: str
    step_index: int
    target_seat: int
    opponent_archetype: str
    incumbent_index: int
    candidate_indices: tuple[int, ...]
    priority: float


@dataclass(frozen=True, slots=True)
class ProtectedDecision:
    episode_id: str
    step_index: int
    features: np.ndarray
    control_index: int
    v1_index: int


@dataclass(frozen=True, slots=True)
class IncumbentAnchor:
    episode_id: str
    step_index: int
    features: np.ndarray
    incumbent_index: int
    nearest_alternative: int


@dataclass(frozen=True, slots=True)
class ReplayCoverage:
    episode_id: str
    won: bool
    v1_changed_game: bool
    seeds: tuple[CoverageSeed, ...]
    protected: tuple[ProtectedDecision, ...]
    anchors: tuple[IncumbentAnchor, ...]


def coverage_priority(
    score_rows: tuple[np.ndarray, np.ndarray],
    incumbent_index: int,
    margins: tuple[float, float],
) -> tuple[float, tuple[int, ...]]:
    alternatives = [i for i in range(len(score_rows[0])) if i != incumbent_index]
    ranked = sorted(
        alternatives,
        key=lambda i: (
            min(
                (float(score_rows[0][i] - score_rows[0][incumbent_index]) / margins[0]),
                (float(score_rows[1][i] - score_rows[1][incumbent_index]) / margins[1]),
            ),
            -i,
        ),
        reverse=True,
    )
    return max(0.0, min(1.0, ranked and min(
        (float(score_rows[0][ranked[0]] - score_rows[0][incumbent_index]) / margins[0]),
        (float(score_rows[1][ranked[0]] - score_rows[1][incumbent_index]) / margins[1]),
    ))), tuple(ranked[:3])
```

`extract_replay_coverage` must use the pending-observation alignment already implemented in `scripts/compare_agents_on_official_replays.py`. It rejects mirrored target decks and malformed observations. It records V1-versus-control differences as protected decisions, unchanged eligible decisions from wins as incumbent anchors, and seeds only when the complete loss contains no V1 difference. Seeds, protected decisions, and anchors are limited to `context == 0`, `minCount == maxCount == 1`, multi-option decisions. `select_coverage_seeds` keeps the highest-priority row per episode, sorts by `(-priority, episode_id, step_index)`, and caps at 24.

- [ ] **Step 4: Implement the streaming manifest command**

```python
def main() -> None:
    episode_ids = list_episodes("55255929", 1000)
    with tempfile.TemporaryDirectory(prefix="grim-coverage-manifest-") as tmp:
        rows = []
        for episode_id in episode_ids:
            replay = download_replay_with_timeout(episode_id, Path(tmp), 2, 45)
            rows.extend(extract_replay_coverage_from_path(replay, runtime).seeds)
            replay.unlink(missing_ok=True)
    selected = select_coverage_seeds(rows, maximum=24)
    write_manifest_atomic(selected, args.out)
```

The command writes `reports/grimmsnarl_coverage_v2_manifest.json`, including source submission IDs, Engine22 hash, selected seed metadata, rejected counts, and its own SHA-256. It must never retain replay JSON or observations.

- [ ] **Step 5: Run tests and a read-only manifest smoke**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/league_selfplay/test_coverage_v2.py -q
PYTHONDONTWRITEBYTECODE=1 python scripts/build_grimmsnarl_coverage_manifest.py \
  --limit 8 --maximum 4 \
  --out reports/grimmsnarl_coverage_v2_manifest_smoke.json
```

Expected: tests pass, the smoke manifest contains at most four unique episode IDs, and no replay JSON remains outside the system temporary directory.

- [ ] **Step 6: Commit seed extraction**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/coverage_v2.py \
  pokemon-tcg-ai-battle/tests/league_selfplay/test_coverage_v2.py \
  pokemon-tcg-ai-battle/scripts/build_grimmsnarl_coverage_manifest.py
git commit -m "feat: select untouched Grimmsnarl loss states"
```

### Task 2: Strict Engine22 counterfactual aggregates

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/coverage_v2.py`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_coverage_v2.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_grimmsnarl_coverage_rollouts.py`

**Interfaces:**
- Consumes: Task 1 `CoverageSeed` rows, temporary replay JSON, `probe_official_replay_counterfactuals.repeated_rollouts`, and the three non-Grimmsnarl `DriverRegistry` policies.
- Produces: `PolicyRollout`, `CounterfactualAggregate`, `accept_counterfactual`, and `run_coverage_rollouts`.

- [ ] **Step 1: Add failing acceptance tests**

```python
def test_counterfactual_requires_non_regression_for_every_policy():
    row = _aggregate(
        incumbent=((0, 2), (1, 2), (0, 2)),
        alternative=((1, 2), (2, 2), (0, 2)),
    )
    assert accept_counterfactual(row, minimum_advantage=0.20)
    regressing = _aggregate(
        incumbent=((0, 2), (1, 2), (1, 2)),
        alternative=((1, 2), (2, 2), (0, 2)),
    )
    assert not accept_counterfactual(regressing, minimum_advantage=0.20)


def test_counterfactual_rejects_incomplete_or_weak_rollouts():
    assert not accept_counterfactual(_aggregate_with_one_game(), minimum_advantage=0.20)
    assert not accept_counterfactual(_aggregate_with_advantage(0.10), minimum_advantage=0.20)
```

- [ ] **Step 2: Run tests and verify missing aggregate interfaces fail**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/league_selfplay/test_coverage_v2.py -q`

Expected: import failures for `PolicyRollout`, `CounterfactualAggregate`, and `accept_counterfactual`.

- [ ] **Step 3: Implement strict terminal-outcome acceptance**

```python
@dataclass(frozen=True, slots=True)
class PolicyRollout:
    member: MemberId
    wins: int
    games: int
    mean_steps: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games


@dataclass(frozen=True, slots=True)
class CounterfactualAggregate:
    seed: CoverageSeed
    features: np.ndarray
    trial_index: int
    incumbent: tuple[PolicyRollout, ...]
    alternative: tuple[PolicyRollout, ...]


def accept_counterfactual(row: CounterfactualAggregate, minimum_advantage: float = 0.20) -> bool:
    if any(item.games != 2 for item in row.incumbent + row.alternative):
        return False
    pairs = tuple(zip(row.incumbent, row.alternative, strict=True))
    if any(new.win_rate < old.win_rate for old, new in pairs):
        return False
    if sum(new.win_rate > old.win_rate for old, new in pairs) < 2:
        return False
    old_rate = sum(item.wins for item in row.incumbent) / sum(item.games for item in row.incumbent)
    new_rate = sum(item.wins for item in row.alternative) / sum(item.games for item in row.alternative)
    return new_rate - old_rate >= minimum_advantage
```

- [ ] **Step 4: Implement streamed restored-state rollouts**

For every selected seed, redownload only its episode into a temporary directory, reconstruct the hidden lists with `hidden_frame` and `exact_hidden_lists`, and evaluate the incumbent plus at most three candidate indices. After the root action, the target seat uses V1 and the opponent seat is independently continued by Lucario, Crustle, and Alakazam policies. Call `search_release` and `search_end` in `finally`; unlink the replay after the episode's aggregates are in memory.

```python
for opponent_member in (MemberId.LUCARIO, MemberId.CRUSTLE, MemberId.ALAKAZAM):
    incumbent_runs = repeated_rollouts(
        2,
        typed_obs=typed_obs,
        hidden=hidden,
        root_action=[seed.incumbent_index],
        agent_dirs=_continuation_dirs(v1_dir, registry, opponent_member, seed.target_seat),
        history=history,
        target_seat=seed.target_seat,
        max_steps=800,
    )
```

`run_coverage_rollouts` returns accepted feature matrices in memory so the Task 4
orchestrator can train without an intermediate dataset. The standalone command
writes only `reports/grimmsnarl_coverage_v2_rollouts.json`: seed metadata,
per-policy aggregate counts, accepted preference indices, duration, peak
temporary bytes, and errors; it does not serialize observations or feature
matrices.

- [ ] **Step 5: Run an eight-state Mac signal gate**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_grimmsnarl_coverage_rollouts.py \
  --manifest reports/grimmsnarl_coverage_v2_manifest.json \
  --maximum-states 8 --rollouts-per-policy 2 --max-steps 800 \
  --out reports/grimmsnarl_coverage_v2_rollouts_smoke.json
```

Expected: zero leaked search IDs, temporary usage below 500 MB, and either at least one accepted preference or an explicit `REJECT_NO_COUNTERFACTUAL_SIGNAL`. Do not continue to training on rejection.

- [ ] **Step 6: Commit counterfactual aggregation**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/coverage_v2.py \
  pokemon-tcg-ai-battle/tests/league_selfplay/test_coverage_v2.py \
  pokemon-tcg-ai-battle/scripts/run_grimmsnarl_coverage_rollouts.py
git commit -m "feat: label coverage gaps with terminal rollouts"
```

### Task 3: Layered V2 learner that cannot replace V1 actions

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/coverage_v2.py`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_coverage_v2.py`

**Interfaces:**
- Consumes: accepted Task 2 aggregates, V1 exported NPZ weights, V1 margins, protected V1 replay decisions, and untouched-win incumbent anchors.
- Produces: `load_exported_ensemble`, `build_coverage_examples`, `train_coverage_ensemble`, `layered_override`, and `calibrate_v2_margins`.

- [ ] **Step 1: Add failing preservation and training tests**

```python
def test_layered_gate_never_replaces_a_v1_override():
    assert layered_override(
        v1_rows=([0.0, 1.0, 0.2], [0.0, 0.9, 0.3]),
        v2_rows=([0.0, 0.1, 2.0], [0.0, 0.2, 2.1]),
        incumbent_index=0,
        v1_margin=(0.25, 0.25),
        v2_margin=(0.25, 0.25),
    ) == 1


def test_v2_can_add_an_override_when_v1_abstains():
    assert layered_override(
        v1_rows=([0.0, 0.1], [0.0, 0.1]),
        v2_rows=([0.0, 0.8], [0.0, 0.7]),
        incumbent_index=0,
        v1_margin=(0.25, 0.25),
        v2_margin=(0.25, 0.25),
    ) == 1
```

- [ ] **Step 2: Run tests and verify the layered-gate imports fail**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/league_selfplay/test_coverage_v2.py -q`

Expected: import failures for the Task 3 functions.

- [ ] **Step 3: Load V1 exactly and implement the immutable layered gate**

```python
def layered_override(v1_rows, v2_rows, incumbent_index, *, v1_margin, v2_margin):
    frozen = trusted_override(v1_rows, incumbent_index, margin=v1_margin)
    if frozen is not None:
        return frozen
    return trusted_override(v2_rows, incumbent_index, margin=v2_margin)
```

`load_exported_ensemble` copies `w`, `b`, `w_option`, and `b_option` from the two V1 NPZ files into two `ActionValueNet` instances and verifies score parity within absolute tolerance `1e-5` on a fixture before returning. Clone this ensemble before V2 training; never mutate the V1 object.

- [ ] **Step 4: Build balanced pairwise examples and train two models**

```python
def build_coverage_examples(accepted, protected, untouched_wins):
    positive = [
        _pair(row.features, row.seed.incumbent_index, row.trial_index, +0.5)
        for row in accepted
    ]
    protected_rows = [
        _pair(row.features, row.control_index, row.v1_index, +0.5)
        for row in protected
    ]
    anchors = [
        _pair(row.features, row.incumbent_index, row.nearest_alternative, -0.5)
        for row in untouched_wins
    ]
    target = max(len(positive), len(protected_rows), len(anchors))
    return _cycle_to(positive, target) + _cycle_to(protected_rows, target) + _cycle_to(anchors, target)
```

Train with the existing `update_intervention_ensemble`, independent seeds `20260806` and `20260807`, AdamW `3e-4`, batch 32, and eight epochs. Calibrate only the V2 layer on untouched winning decisions with quantile `0.995` and minimum margin `0.25`.

- [ ] **Step 5: Verify preservation, new coverage, and independent weights**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/league_selfplay/test_coverage_v2.py tests/league_selfplay/test_single_intervention.py -q`

Expected: every protected replay decision returns the V1 action, at least one accepted coverage fixture produces a V2-only override, both parameter deltas are positive, and model tensors do not share storage.

- [ ] **Step 6: Commit the layered learner**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/coverage_v2.py \
  pokemon-tcg-ai-battle/tests/league_selfplay/test_coverage_v2.py
git commit -m "feat: add immutable layered residual training"
```

### Task 4: Bounded Mac V2/V1/control proof

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/coverage_v2.py`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_coverage_v2.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_grimmsnarl_coverage_v2.py`
- Create on pass: `pokemon-tcg-ai-battle/agents/grimmsnarl_coverage_v2_mac_pass/`
- Create at runtime: `pokemon-tcg-ai-battle/reports/mac_grimmsnarl_coverage_v2_latest.json`

**Interfaces:**
- Consumes: Tasks 1-3, `DriverRegistry`, `run_actual_game`, Engine22, and the frozen V1/control policies.
- Produces: `LayeredPopulationPolicy`, `ThreeWayComparison`, `CoverageV2Decision`, `evaluate_three_way_group`, and `run_coverage_v2_proof`.

- [ ] **Step 1: Add failing three-way acceptance tests**

```python
def test_v2_gate_requires_preservation_coverage_and_group_gain():
    decision = decide_coverage_v2(
        preserved=27,
        expected_preserved=27,
        new_overrides=8,
        v2_vs_v1_delta=0.04,
        v2_vs_control_delta=0.08,
        per_member_delta={member: 0.0 for member in MemberId},
        temporary_bytes=12_000_000,
        failures=(),
    )
    assert decision.code == "PASS_GRIMMSNARL_COVERAGE_V2_MAC"


def test_v2_gate_rejects_one_regressing_member_or_quota_breach():
    assert decide_coverage_v2(
        preserved=27,
        expected_preserved=27,
        new_overrides=8,
        v2_vs_v1_delta=0.04,
        v2_vs_control_delta=0.08,
        per_member_delta={MemberId.GRIMMSNARL: -0.06},
        temporary_bytes=12_000_000,
        failures=(),
    ).code == "REJECT_MEMBER_REGRESSION"
    assert decide_coverage_v2(
        preserved=27,
        expected_preserved=27,
        new_overrides=8,
        v2_vs_v1_delta=0.04,
        v2_vs_control_delta=0.08,
        per_member_delta={MemberId.GRIMMSNARL: 0.0},
        temporary_bytes=501 * 1024**2,
        failures=(),
    ).code == "REJECT_STORAGE_QUOTA"
```

- [ ] **Step 2: Run tests and verify missing proof interfaces fail**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/league_selfplay/test_coverage_v2.py -q`

Expected: import failures for Task 4 interfaces.

- [ ] **Step 3: Implement the frozen 192-game-per-arm schedule**

```python
for focal in MemberId:
    for opponent in MemberId:
        if opponent is focal:
            continue
        for seat in (0, 1):
            members = (focal, opponent) if seat == 0 else (opponent, focal)
            for _ in range(8):
                results["v2"].append(run_actual_game(game, v2_policy, members=members, rng=rng))
                results["v1"].append(run_actual_game(game, v1_policy, members=members, rng=rng))
                results["control"].append(run_actual_game(game, control_policy, members=members, rng=rng))
```

The schedule produces 192 completed games per arm and 576 total. Record group score, each focal member's score, override counts split into frozen-V1 and new-V2 layers, same-schedule truth, and `paired_randomness=False`.

- [ ] **Step 4: Implement orchestration, cleanup, and conditional retention**

`run_coverage_v2_proof` verifies the engine hash, measures `artifacts/` before and after, owns one `TemporaryDirectory`, installs cleanup handlers, runs full 24-state counterfactual collection only after the eight-state signal gate, trains V2, replays all protected decisions, runs the three-way audit, and writes one atomic report under 100 KiB. On failure it deletes all checkpoints and returns a rejection report. On pass it atomically creates `agents/grimmsnarl_coverage_v2_mac_pass/` with `v2-0.npz`, `v2-1.npz`, and a manifest containing V1 hashes, V1/V2 margins, seed-manifest hash, rollout-report hash, Engine22 hash, and audit scores.

- [ ] **Step 5: Run the Mac proof**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_grimmsnarl_coverage_v2.py \
  --wall-time-seconds 900 \
  --storage-quota-bytes 524288000 \
  --out reports/mac_grimmsnarl_coverage_v2_latest.json
```

Expected pass conditions: exact V1 preservation, at least one new override, positive V2 delta versus both V1 and control, no per-member delta below `-0.05`, complete 192-game-per-arm schedule, no retained raw replay, and measured storage below 500 MB. A rejection is a valid experiment result and must not create a retained model directory.

- [ ] **Step 6: Run the complete league regression suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  tests/league_selfplay/test_coverage_v2.py \
  tests/league_selfplay/test_single_intervention.py \
  tests/league_selfplay/test_single_intervention_runner.py \
  tests/league_selfplay/test_single_intervention_submission.py \
  tests/league_selfplay/test_action_value.py \
  tests/league_selfplay/test_storage.py -q
```

Expected: all tests pass with no warnings or leaked temporary directories.

- [ ] **Step 7: Commit the Mac proof**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/coverage_v2.py \
  pokemon-tcg-ai-battle/tests/league_selfplay/test_coverage_v2.py \
  pokemon-tcg-ai-battle/scripts/run_grimmsnarl_coverage_v2.py \
  pokemon-tcg-ai-battle/reports/mac_grimmsnarl_coverage_v2_latest.json
git add -f pokemon-tcg-ai-battle/agents/grimmsnarl_coverage_v2_mac_pass
git commit -m "model: retain validated Grimmsnarl coverage v2"
```

If the proof rejects, omit the absent retained directory and commit only the code plus compact rejection report with message `test: record Grimmsnarl coverage v2 rejection`.

### Task 5: Conditional NumPy submission and V1/V2 online comparison

**Files:**
- Create only after Task 4 passes: `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_coverage_v2/`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention_submission.py`
- Create only after Task 4 passes: `pokemon-tcg-ai-battle/submissions/grimmsnarl_coverage_v2_20260806.tar.gz`

**Interfaces:**
- Consumes: frozen V1 candidate runtime and Task 4 retained V2 weights/manifest.
- Produces: a pure-NumPy layered submission in which V1 is evaluated first and V2 is consulted only when V1 abstains.

- [ ] **Step 1: Write a failing NumPy parity and preservation test**

```python
def test_coverage_v2_numpy_runtime_matches_torch_and_preserves_v1():
    runtime = _load_coverage_v2_runtime()
    for observation in _protected_online_observations():
        assert runtime.choose_action(observation, _control_action(observation)) == _v1_action(observation)
    for observation in _accepted_coverage_observations():
        np_rows = runtime.v2_option_scores(observation)
        torch_rows = _torch_v2_scores(observation)
        for actual, expected in zip(np_rows, torch_rows, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
```

- [ ] **Step 2: Run the focused test and verify the missing candidate fails**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/league_selfplay/test_single_intervention_submission.py -k coverage_v2 -q`

Expected: failure because `agents/candidate_grimmsnarl_coverage_v2` does not exist.

- [ ] **Step 3: Build the minimal layered candidate**

Copy the frozen control policy/runtime/deck files from `candidate_grimmsnarl_single_intervention_control_v1`, copy the V1 NumPy weights and feature package from `candidate_grimmsnarl_single_intervention_v1`, and add only the two V2 NPZ files plus manifest. `main.py` obtains the control action and calls `choose_action`; `choose_action` computes V1 rows first, returns any trusted V1 override immediately, and computes V2 rows only if V1 abstains. Catching an inference exception returns the control action.

- [ ] **Step 4: Verify runtime, package, and Engine22 both seats**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/league_selfplay/test_single_intervention_submission.py -q
PYTHONDONTWRITEBYTECODE=1 python scripts/package_submission.py \
  --agent-dir agents/candidate_grimmsnarl_coverage_v2 \
  --output submissions/grimmsnarl_coverage_v2_20260806.tar.gz
python scripts/run_local_match.py \
  --agent0 agents/candidate_grimmsnarl_coverage_v2 \
  --agent1 agents/candidate_grimmsnarl_single_intervention_v1
python scripts/run_local_match.py \
  --agent0 agents/candidate_grimmsnarl_single_intervention_v1 \
  --agent1 agents/candidate_grimmsnarl_coverage_v2
```

Expected: NumPy/Torch parity within `1e-5`, no Torch or sklearn runtime import, archive below 10 MiB, no cache files, and both games finish with legal actions.

- [ ] **Step 5: Commit, then replace the control submission only**

```bash
git add pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_coverage_v2 \
  pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention_submission.py
git add -f pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_coverage_v2/models/*.npz
git commit -m "feat: package Grimmsnarl residual coverage v2"
kaggle competitions submit -c pokemon-tcg-ai-battle \
  -f submissions/grimmsnarl_coverage_v2_20260806.tar.gz \
  -m "Grimmsnarl coverage V2; immutable V1 layer plus terminal-rollout-backed coverage expansion; Mac three-way gate passed"
```

Keep V1 submission `55255929` online. Confirm the new submission appears in `kaggle competitions submissions`, record its reference, and compare V1/V2 only after both have at least 40 identifiable public games. Never submit Task 5 artifacts if Task 4 rejected.
