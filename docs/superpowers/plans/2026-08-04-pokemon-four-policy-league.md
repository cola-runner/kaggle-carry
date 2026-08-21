# Pokémon Four-Policy League Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a disk-bounded Mac validation in which four independent trainable policies complete two synchronous self-play rounds and pass only if the group becomes stronger against frozen external judges and its own starting population.

**Architecture:** Four deck-specific 1.5M policy/value networks share a visible-information feature and legal-action implementation but never share weights. Current-current games record both players and update both models; Round-0 snapshots remain immutable, and fixed drivers disappear after one-time initialization. A frozen schedule, temporary compact shards, external judges, and an auditable decision gate prevent the experiment from drifting back into fixed-opponent training.

**Tech Stack:** Python 3.12, official local `cg` engine, PyTorch 2.12 on Apple MPS, NumPy submission inference, pytest under the project Python 3.14 runtime, macOS `$TMPDIR`.

## Global Constraints

- Members are exactly `grimmsnarl`, `lucario`, `crustle`, and `alakazam`.
- Every member owns an independent policy/value parameter set.
- Strictly more than half of every reinforcement-learning round is current-current self-play, and every current-current game trains both players.
- Every member must receive a finite parameter update in every round.
- Rule drivers provide initialization labels only; teacher access is closed before Round 1.
- Fixed judges never provide labels, trajectories, tuning feedback, or early stopping.
- Terminal reward is exactly win `+1`, draw `0`, loss `-1`; no shaped reward enters optimization.
- Policies receive visible information only and control all single- and multi-option selections after initialization.
- Raw replays are never written; temporary data lives under one run directory in `$TMPDIR` and is removed on every exit path.
- Temporary data is capped at 512 MiB; a compact shard is capped at 64 MiB; at most two unconsumed shards may exist.
- The standard Mac run is capped at 20 minutes and may not change its schedule or thresholds after preflight.
- `pokemon-tcg-ai-battle/artifacts/` must have identical file count and byte size before and after the run.
- No Kaggle submission and no GTX 1080 Ti work occurs in this plan.

---

## File Structure

- `pokemon-tcg-ai-battle/league_selfplay/contracts.py`: member identities, frozen configuration, game-source metadata, and the `INVALID_SELF_PLAY` audit.
- `pokemon-tcg-ai-battle/league_selfplay/actions.py`: ordered legal sampling, multi-selection, learned stop handling, and log-probability recomputation.
- `pokemon-tcg-ai-battle/league_selfplay/features.py`: visible-only 512-wide state/option encoding extracted from the validated submission runtime.
- `pokemon-tcg-ai-battle/league_selfplay/model.py`: independent PyTorch policy/value models and exact NumPy export.
- `pokemon-tcg-ai-battle/league_selfplay/engine.py`: official-engine games that capture both actors' compact trajectories.
- `pokemon-tcg-ai-battle/league_selfplay/bootstrap.py`: one-time driver initialization and the irreversible teacher-close gate.
- `pokemon-tcg-ai-battle/league_selfplay/ppo.py`: GAE, clipped PPO, and simultaneous four-member updates.
- `pokemon-tcg-ai-battle/league_selfplay/schedule.py`: deterministic preflight schedule and immutable configuration hash.
- `pokemon-tcg-ai-battle/league_selfplay/storage.py`: bounded temporary shards, quota checks, and cleanup.
- `pokemon-tcg-ai-battle/league_selfplay/evaluation.py`: external-judge comparison, ancestry cross-play, confidence interval, and pass/reject decision.
- `pokemon-tcg-ai-battle/league_selfplay/runner.py`: phase orchestration and persistent report.
- `pokemon-tcg-ai-battle/scripts/run_four_policy_league.py`: CLI entry point.
- `pokemon-tcg-ai-battle/tests/league_selfplay/`: focused unit, integration, failure, and end-to-end tests.

### Task 1: Make self-play identity and provenance machine-checkable

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/__init__.py`
- Create: `pokemon-tcg-ai-battle/league_selfplay/contracts.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_contracts.py`

**Interfaces:**
- Produces: `MemberId`, `GameSource`, `FrozenLeagueConfig`, `GameProvenance`, `SelfPlayAudit`, and `audit_training_batch(records, expected_members)`.
- Consumes: standard-library dataclasses, enums, and SHA-256 only.

- [ ] **Step 1: Write the failing provenance tests**

```python
def test_current_current_game_updates_both_participants():
    record = GameProvenance.current_game(MemberId.GRIMMSNARL, MemberId.LUCARIO)
    audit = audit_training_batch([record], {MemberId.GRIMMSNARL, MemberId.LUCARIO})
    assert audit.valid
    assert record.update_members == (MemberId.GRIMMSNARL, MemberId.LUCARIO)

def test_fixed_actor_trajectory_is_invalid_self_play():
    record = GameProvenance(
        source=GameSource.CURRENT_VS_FIXED,
        actors=("grimmsnarl", "judge_dragapult"),
        update_members=(MemberId.GRIMMSNARL,),
        trajectory_members=(MemberId.GRIMMSNARL,),
    )
    audit = audit_training_batch([record], set(MemberId))
    assert not audit.valid
    assert audit.code == "INVALID_SELF_PLAY"

def test_all_four_members_must_receive_data():
    records = [GameProvenance.current_game(MemberId.GRIMMSNARL, MemberId.LUCARIO)]
    assert not audit_training_batch(records, set(MemberId)).valid
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_contracts.py`

Expected: FAIL because `league_selfplay.contracts` does not exist.

- [ ] **Step 3: Implement immutable contracts and canonical config hashing**

```python
class MemberId(str, Enum):
    GRIMMSNARL = "grimmsnarl"
    LUCARIO = "lucario"
    CRUSTLE = "crustle"
    ALAKAZAM = "alakazam"

class GameSource(str, Enum):
    CURRENT_CURRENT = "current_current"
    CURRENT_HISTORY = "current_history"
    CURRENT_VS_FIXED = "current_vs_fixed"

@dataclass(frozen=True, slots=True)
class FrozenLeagueConfig:
    seed: int = 20260804
    round_one_games_per_seat: int = 12
    round_two_games_per_seat: int = 12
    history_games_per_seat: int = 2
    judge_games_per_seat: int = 4
    ancestry_games_per_seat: int = 2
    wall_time_seconds: int = 1200
    temp_quota_bytes: int = 512 * 1024**2

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
```

`audit_training_batch` must reject fixed actors, reject a current-current record that does not update and record both participants, reject missing current members, and reject batches where current-current games are half or fewer of all records.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_contracts.py`

Expected: all contract tests PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/__init__.py pokemon-tcg-ai-battle/league_selfplay/contracts.py pokemon-tcg-ai-battle/tests/league_selfplay/test_contracts.py
git commit -m "feat: enforce four-policy self-play provenance"
```

### Task 2: Represent every legal engine selection as one trainable distribution

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/actions.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_actions.py`

**Interfaces:**
- Produces: `ActionSample(indices: tuple[int, ...], log_probability: float)`, `sample_action(option_logits, stop_logit, min_count, max_count, rng)`, and `action_log_probability(indices, option_logits, stop_logit, min_count, max_count)`.
- Consumes: NumPy arrays and engine `minCount`/`maxCount` values.

- [ ] **Step 1: Write failing exact-cardinality, stop, and parity tests**

```python
def test_exact_two_samples_distinct_options_without_stop():
    sample = sample_action(np.array([3.0, 2.0, 1.0]), 100.0, 2, 2, np.random.default_rng(7))
    assert len(sample.indices) == 2
    assert len(set(sample.indices)) == 2

def test_variable_selection_can_stop_after_minimum():
    sample = sample_action(np.array([-9.0, -9.0]), 9.0, 1, 2, np.random.default_rng(8))
    assert len(sample.indices) == 1

def test_sampled_log_probability_recomputes_exactly():
    logits = np.array([0.5, 1.0, -0.2], dtype=np.float64)
    sample = sample_action(logits, -0.1, 1, 3, np.random.default_rng(9))
    recomputed = action_log_probability(sample.indices, logits, -0.1, 1, 3)
    assert recomputed == pytest.approx(sample.log_probability, abs=1e-10)
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_actions.py`

Expected: FAIL because action sampling is absent.

- [ ] **Step 3: Implement ordered sampling without replacement**

At each step, softmax only the unselected options. Add the stop logit only after `min_count` selections. Do not expose stop when `min_count == max_count`; force termination at `max_count`. Store and send the sampled order unchanged so PPO recomputes the probability of the exact engine action rather than an unordered approximation.

```python
@dataclass(frozen=True, slots=True)
class ActionSample:
    indices: tuple[int, ...]
    log_probability: float

def sample_action(
    option_logits: np.ndarray,
    stop_logit: float,
    min_count: int,
    max_count: int,
    rng: np.random.Generator,
) -> ActionSample:
    logits = np.asarray(option_logits, dtype=np.float64)
    if min_count < 0 or min_count > max_count or max_count > len(logits):
        raise ValueError("invalid selection bounds")
    selected: list[int] = []
    log_probability = 0.0
    while len(selected) < max_count:
        remaining = [index for index in range(len(logits)) if index not in selected]
        can_stop = min_count <= len(selected) and min_count != max_count
        candidate_logits = [float(logits[index]) for index in remaining]
        if can_stop:
            candidate_logits.append(float(stop_logit))
        shifted = np.asarray(candidate_logits) - max(candidate_logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        choice = int(rng.choice(len(probabilities), p=probabilities))
        log_probability += float(np.log(probabilities[choice]))
        if can_stop and choice == len(remaining):
            break
        selected.append(remaining[choice])
    return ActionSample(tuple(selected), log_probability)
```

The implementation must raise `ValueError` for negative counts, `min_count > max_count`, or `max_count > option_count`.

- [ ] **Step 4: Run and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_actions.py`

Expected: all action tests PASS.

- [ ] **Step 5: Commit legal action sampling**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/actions.py pokemon-tcg-ai-battle/tests/league_selfplay/test_actions.py
git commit -m "feat: add trainable multi-selection policy"
```

### Task 3: Build four independent 1.5M policy/value models with NumPy parity

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/features.py`
- Create: `pokemon-tcg-ai-battle/league_selfplay/model.py`
- Create: `pokemon-tcg-ai-battle/league_selfplay/numpy_runtime.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_model.py`

**Interfaces:**
- Produces: `encode_options(observation) -> np.ndarray`, `PolicyValueNet`, `create_population(seed, device) -> dict[MemberId, PolicyValueNet]`, `export_member(model, path)`, and `numpy_forward(features, weights)`.
- Consumes: visible observations, the exact 512-wide encoding contract, and `MemberId`.

- [ ] **Step 1: Write failing independence, hidden-data, parameter-count, and export tests**

```python
def test_population_members_do_not_share_parameter_storage():
    population = create_population(20260804, "cpu")
    pointers = [next(model.parameters()).data_ptr() for model in population.values()]
    assert len(set(pointers)) == 4

def test_visible_encoding_ignores_hidden_payloads(visible_observation):
    poisoned = copy.deepcopy(visible_observation)
    poisoned["search_begin_input"] = {"opponentHand": [999]}
    np.testing.assert_array_equal(encode_options(visible_observation), encode_options(poisoned))

def test_each_policy_is_approximately_1p5m_parameters():
    model = create_population(1, "cpu")[MemberId.GRIMMSNARL]
    policy_count = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("value"))
    assert 1_450_000 <= policy_count <= 1_550_000

def test_numpy_and_torch_logits_match(tmp_path):
    assert exported_max_error(tmp_path) < 1e-4
```

- [ ] **Step 2: Run the project tests and Miniforge parity subprocess; verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_model.py`

Expected: FAIL because the feature/model modules are absent.

- [ ] **Step 3: Extract the validated visible encoder and implement the model**

Reuse the semantics already validated in `agents/candidate_grimmsnarl_nn_1p5m_smoke_v0/policy_runtime.py`, but place the maintained implementation in `league_selfplay/features.py`. Do not import agent-private modules at training time.

```python
class PolicyValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_one = nn.Linear(512, 1024)
        self.layer_two = nn.Linear(1024, 928)
        self.option_head = nn.Linear(928, 1)
        self.stop_head = nn.Linear(928, 1)
        self.value_head = nn.Linear(928, 1)

    def forward(self, features, mask):
        hidden = torch.tanh(self.layer_one(features))
        hidden = torch.tanh(self.layer_two(hidden))
        option_logits = self.option_head(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        pooled = masked_mean(hidden, mask)
        return option_logits, self.stop_head(pooled).squeeze(-1), self.value_head(pooled).squeeze(-1)
```

Export policy layers and stop head as float32 NumPy arrays. The training-only value head may remain outside the submission artifact.

- [ ] **Step 4: Run tests under both runtimes and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_model.py`

Run: `python3 -m league_selfplay.model --parity-smoke`

Expected: all tests PASS, MPS is reported available, and max logit error is below `1e-4`.

- [ ] **Step 5: Commit the independent models**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/features.py pokemon-tcg-ai-battle/league_selfplay/model.py pokemon-tcg-ai-battle/league_selfplay/numpy_runtime.py pokemon-tcg-ai-battle/tests/league_selfplay/test_model.py
git commit -m "feat: add independent league policy models"
```

### Task 4: Capture legal trajectories for both players from one engine game

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/engine.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_engine.py`

**Interfaces:**
- Produces: `PolicyActor`, `TrajectoryStep`, `CompletedGame`, and `run_training_game(game_api, actor0, actor1, source, rng, max_steps)`.
- Consumes: `encode_options`, `sample_action`, official `cg`, `GameSource`, and existing `scripts.run_local_match.validate_action`.

- [ ] **Step 1: Write failing two-sided trajectory integration tests**

```python
def test_current_current_game_records_both_members(real_engine, tiny_population):
    game = run_training_game(
        real_engine,
        tiny_population[MemberId.GRIMMSNARL],
        tiny_population[MemberId.LUCARIO],
        GameSource.CURRENT_CURRENT,
        np.random.default_rng(1),
        2000,
    )
    assert game.finished
    assert {step.member for step in game.steps} == {MemberId.GRIMMSNARL, MemberId.LUCARIO}
    assert all(np.isfinite(step.old_log_probability) for step in game.steps)
    assert {step.reward for step in game.steps} <= {-1.0, 0.0, 1.0}

def test_opposite_seat_game_also_finishes(real_engine, tiny_population):
    game = run_training_game(real_engine, tiny_population[MemberId.LUCARIO], tiny_population[MemberId.GRIMMSNARL], GameSource.CURRENT_CURRENT, np.random.default_rng(2), 2000)
    assert game.finished
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m pytest -q tests/league_selfplay/test_engine.py`

If Miniforge has no pytest, run the test through the project pytest and invoke a Miniforge subprocess inside the integration fixture. Expected: FAIL because `engine.py` is absent.

- [ ] **Step 3: Implement one battle owner and two actor recorders**

```python
@dataclass(slots=True)
class TrajectoryStep:
    member: MemberId
    features: np.ndarray
    action: tuple[int, ...]
    old_log_probability: float
    old_value: float
    reward: float = 0.0

@dataclass(slots=True)
class CompletedGame:
    provenance: GameProvenance
    source: GameSource
    actors: tuple[str, str]
    winner: int
    steps: list[TrajectoryStep]
    finished: bool
```

Before each `battle_select`, validate the action. At terminal state assign `+1/-1` to winner/loser steps and `0` to both on a draw. Always call `battle_finish()` in `finally`. Never retain the raw observation after its compact `TrajectoryStep` is constructed.

- [ ] **Step 4: Run and verify GREEN from both seats**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_engine.py`

Expected: both real-engine seat tests PASS with zero illegal actions.

- [ ] **Step 5: Commit two-sided collection**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/engine.py pokemon-tcg-ai-battle/tests/league_selfplay/test_engine.py
git commit -m "feat: collect two-sided league trajectories"
```

### Task 5: Initialize four policies once, then irreversibly close teacher access

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/bootstrap.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_bootstrap.py`

**Interfaces:**
- Produces: `DriverRegistry`, `BootstrapStats`, `initialize_population(population, drivers, games, device, seed)`, and `run_start_gate(population, engine_api, driver_registry)`.
- Consumes: four existing driver directories, `PolicyValueNet`, and compact features only.

- [ ] **Step 1: Write failing teacher-close and one-time initialization tests**

```python
def test_registry_cannot_act_after_close(driver_registry, observation):
    driver_registry.close()
    with pytest.raises(RuntimeError, match="teacher access closed"):
        driver_registry.action(MemberId.GRIMMSNARL, observation)

def test_initialization_changes_all_four_models(population, driver_registry):
    before = parameter_hashes(population)
    stats = initialize_population(population, driver_registry, games=96, device="cpu", seed=3)
    assert set(stats.members) == set(MemberId)
    assert all(parameter_hashes(population)[member] != before[member] for member in MemberId)

def test_start_gate_closes_registry_before_return(population, driver_registry, real_engine):
    result = run_start_gate(population, real_engine, driver_registry)
    assert result.teacher_closed
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_bootstrap.py`

Expected: FAIL because the bootstrap gate is absent.

- [ ] **Step 3: Implement balanced driver initialization**

Map members to these driver directories:

```python
DRIVERS = {
    MemberId.GRIMMSNARL: "agents/candidate_grimmsnarl_imitation_full_v2",
    MemberId.LUCARIO: "agents/public_makthanithin_ptcg_mega_lucario_ex_v62",
    MemberId.CRUSTLE: "agents/candidate_crustle_kangaskhan_top1_ranker_v211",
    MemberId.ALAKAZAM: "agents/public_naoto714_alakazam_no_tech_pivot_en",
}
```

Run the six driver pairings from both seats for eight games per seat, collecting labels for both drivers: 96 games total. Train each model for six shuffled cross-entropy epochs with batch size 128 and learning rate `3e-4`. Report held-out action negative log likelihood before and after. Close the registry before any learned-policy start-gate match.

- [ ] **Step 4: Run and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_bootstrap.py`

Expected: teacher-close, all-four-change, and complete-game gates PASS.

- [ ] **Step 5: Commit one-time initialization**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/bootstrap.py pokemon-tcg-ai-battle/tests/league_selfplay/test_bootstrap.py
git commit -m "feat: add one-time league policy initialization"
```

### Task 6: Update all four policies synchronously with PPO

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/ppo.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_ppo.py`

**Interfaces:**
- Produces: `compute_gae(steps, gamma, gae_lambda)`, `PPOStats`, and `update_population(population, games, device, seed)`.
- Consumes: audited `CompletedGame` batches and exact ordered action log probabilities.

- [ ] **Step 1: Write failing simultaneous-update and immutability tests**

```python
def test_one_round_changes_every_current_member(population, balanced_games):
    before = parameter_hashes(population)
    stats = update_population(population, balanced_games, "cpu", 4)
    assert set(stats) == set(MemberId)
    assert all(stats[member].parameter_delta_l2 > 0 for member in MemberId)
    assert all(stats[member].all_finite for member in MemberId)
    assert all(parameter_hashes(population)[member] != before[member] for member in MemberId)

def test_historical_snapshots_never_change(population, history, round_two_games):
    history_before = parameter_hashes(history)
    update_population(population, round_two_games, "cpu", 5)
    assert parameter_hashes(history) == history_before

def test_constant_negative_advantages_do_not_become_zero():
    np.testing.assert_array_equal(normalize_advantages(np.full(8, -1.0)), np.full(8, -1.0))
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_ppo.py`

Expected: FAIL because PPO is absent.

- [ ] **Step 3: Implement per-member GAE and clipped PPO**

Use `gamma=1.0`, `gae_lambda=0.95`, clip `0.2`, entropy coefficient `0.01`, value coefficient `0.5`, AdamW learning rate `1e-4`, minibatch 512 decisions, four shuffled epochs, and gradient norm cap `1.0`. Build all optimizer inputs from the frozen round batch before applying any member update. Recompute the exact ordered multi-selection log probability from Task 2.

```python
def update_population(
    population: dict[MemberId, PolicyValueNet],
    games: Sequence[CompletedGame],
    device: str,
    seed: int,
) -> dict[MemberId, PPOStats]:
    audit = audit_training_batch([game.provenance for game in games], set(MemberId))
    if not audit.valid:
        raise InvalidSelfPlay(audit.reasons)
    frozen_batches = split_steps_by_current_member(games)
    return {member: update_one(population[member], frozen_batches[member], device, seed) for member in MemberId}
```

- [ ] **Step 4: Run and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_ppo.py`

Expected: four current policies change, history remains byte-identical, and every loss/gradient is finite.

- [ ] **Step 5: Commit synchronous PPO**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/ppo.py pokemon-tcg-ai-battle/tests/league_selfplay/test_ppo.py
git commit -m "feat: update four league policies synchronously"
```

### Task 7: Freeze an exact balanced two-round schedule before results exist

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/schedule.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_schedule.py`

**Interfaces:**
- Produces: `ScheduledGame`, `LeagueSchedule`, `build_standard_schedule(config)`, and `build_dry_run_schedule(config)`.
- Consumes: `FrozenLeagueConfig`, four members, four judge names, and snapshot generation names.

- [ ] **Step 1: Write failing count, seat-balance, and hash tests**

```python
def test_standard_schedule_has_exact_phase_counts():
    schedule = build_standard_schedule(FrozenLeagueConfig())
    assert schedule.bootstrap_count == 96
    assert schedule.round_one_count == 144
    assert schedule.round_two_current_count == 144
    assert schedule.round_two_history_count == 64
    assert schedule.judge_count == 256
    assert schedule.ancestry_count == 64

def test_every_current_pair_is_seat_balanced():
    schedule = build_standard_schedule(FrozenLeagueConfig())
    assert schedule.seat_imbalances() == {}

def test_schedule_hash_changes_when_any_count_changes():
    assert build_standard_schedule(FrozenLeagueConfig()).sha256 != build_standard_schedule(replace(FrozenLeagueConfig(), judge_games_per_seat=5)).sha256
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_schedule.py`

Expected: FAIL because schedule generation is absent.

- [ ] **Step 3: Implement canonical schedule generation**

Round 1 contains six unordered current pairings, 12 games per seat: 144 games. Round 2 repeats 144 current-current games and adds every current member against every Round-0 snapshot for two games per seat: 64 games. Judge evaluation runs both Round 0 and Round 2 groups against four judges for four games per seat per member/judge pair: 256 total. Ancestry evaluation runs every final member against every Round-0 member for two games per seat: 64 games.

The dry schedule keeps all phases and all identities but reduces each positive game count to one game per seat. Serialize phase, actors, generations, seats, and ordinal as canonical JSON before hashing.

- [ ] **Step 4: Run and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_schedule.py`

Expected: exact counts and seat balance PASS.

- [ ] **Step 5: Commit frozen schedules**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/schedule.py pokemon-tcg-ai-battle/tests/league_selfplay/test_schedule.py
git commit -m "feat: freeze balanced league schedules"
```

### Task 8: Bound temporary trajectory storage and prove cleanup on every exit

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/storage.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_storage.py`

**Interfaces:**
- Produces: `RunStorage`, `ShardManifest`, `QuotaExceeded`, `write_shard(games)`, `consume_shard(path)`, `cleanup()`, and `install_cleanup_handlers(storage)`.
- Consumes: compact `CompletedGame` steps and `FrozenLeagueConfig` quotas.

- [ ] **Step 1: Write failing quota, atomicity, and cleanup tests**

```python
def test_shard_never_exceeds_64_mib(tmp_path, compact_games):
    storage = RunStorage(tmp_path, quota_bytes=512 * 1024**2, shard_bytes=64 * 1024**2, max_pending=2)
    paths = storage.write_shards(compact_games)
    assert all(path.stat().st_size <= 64 * 1024**2 for path in paths)

def test_quota_failure_removes_run_directory(tmp_path, compact_games):
    run_root = tmp_path / "league"
    with pytest.raises(QuotaExceeded):
        with RunStorage(run_root, quota_bytes=1024, shard_bytes=512, max_pending=2) as storage:
            storage.write_shards(compact_games)
    assert not run_root.exists()

def test_consumed_shard_is_deleted_immediately(tmp_path, compact_games):
    with RunStorage(tmp_path / "league", quota_bytes=512 * 1024**2, shard_bytes=64 * 1024**2, max_pending=2) as storage:
        path = storage.write_shards(compact_games)[0]
        list(storage.consume_shard(path))
        assert not path.exists()

def test_signal_handler_cleans_only_its_run_directory(tmp_path):
    storage = RunStorage(tmp_path / "league", quota_bytes=4096, shard_bytes=2048, max_pending=2)
    handler = install_cleanup_handlers(storage)[signal.SIGTERM]
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)
    assert not storage.root.exists()
    assert tmp_path.exists()
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_storage.py`

Expected: FAIL because bounded storage is absent.

- [ ] **Step 3: Implement atomic NPZ shards and context cleanup**

Write to `name.partial`, call `os.replace` only after `fsync`, and store only float32 features, integer actions/offsets, member IDs, old log probabilities, old values, rewards, game IDs, and provenance. Refuse a third pending shard. Check total bytes before each write. `__exit__` and installed `SIGINT`/`SIGTERM` handlers must recursively remove only the resolved run directory owned by that storage instance and verify that it no longer exists.

- [ ] **Step 4: Run all failure paths and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_storage.py`

Expected: shard, quota, corrupt-shard, exception, and normal cleanup tests PASS.

- [ ] **Step 5: Commit bounded storage**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/storage.py pokemon-tcg-ai-battle/tests/league_selfplay/test_storage.py
git commit -m "feat: bound and clean league trajectories"
```

### Task 9: Judge group improvement without leaking judges into training

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/evaluation.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_evaluation.py`

**Interfaces:**
- Produces: `JudgeResult`, `GroupComparison`, `LeagueDecision`, `compare_groups(start_results, final_results, ancestry_results, seed)`, and `decide_validation(comparison, update_stats, failures)`.
- Consumes: frozen judge games, ancestry games, update statistics, and self-play audit.

- [ ] **Step 1: Write failing synthetic accept/reject and leakage tests**

```python
def test_clear_group_improvement_passes():
    start_results, final_results = synthetic_scores(start=0.40, final=0.58)
    comparison = compare_groups(start_results, final_results, synthetic_ancestry(0.60), seed=6)
    assert decide_validation(comparison, finite_updates(), []).code == "PASS_MAC_LEAGUE"

def test_equal_group_is_rejected():
    start_results, final_results = synthetic_scores(start=0.50, final=0.50)
    comparison = compare_groups(start_results, final_results, synthetic_ancestry(0.50), seed=7)
    assert decide_validation(comparison, finite_updates(), []).code == "REJECT_NO_GROUP_IMPROVEMENT"

def test_any_judge_trajectory_is_invalid():
    with pytest.raises(InvalidSelfPlay):
        training_batch_with_judge_step()
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_evaluation.py`

Expected: FAIL because evaluation is absent.

- [ ] **Step 3: Implement frozen judges and predeclared thresholds**

Use these judge directories only after both training rounds are frozen:

```python
JUDGES = (
    "agents/public_kiyotah_a_sample_rule_based_agent_mega_lucario_ex_deck",
    "agents/public_biohack44_beating_the_day_2_new",
    "agents/candidate_alakazam_control_hedge_v0",
    "agents/public_mossarimossari_a_sample_rule_based_agent_dragapult_ex_deck",
)
```

Score win/draw/loss as `1/0.5/0`. The local engine is not seedable, so use an unpaired 20,000-resample bootstrap interval. Pass only when aggregate final-minus-start judge score is at least `+0.05`, the 95% lower bound is above zero, final-vs-start ancestry score is at least `0.55`, every member's judge delta is at least `-0.10`, all four update deltas are positive and finite, and failures are empty. Always set `paired_randomness=false` in the report.

- [ ] **Step 4: Run and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_evaluation.py`

Expected: synthetic improvement passes, equal/regressing/leaking populations reject.

- [ ] **Step 5: Commit group evaluation**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/evaluation.py pokemon-tcg-ai-battle/tests/league_selfplay/test_evaluation.py
git commit -m "feat: gate external group improvement"
```

### Task 10: Orchestrate the two-round Mac validation and remove the wrong pilot

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/runner.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_four_policy_league.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_runner.py`
- Delete: `pokemon-tcg-ai-battle/selfplay_pilot/engine_probe.py`
- Delete: `pokemon-tcg-ai-battle/selfplay_pilot/quick_ppo.py`
- Delete: `pokemon-tcg-ai-battle/scripts/run_mac_ppo_feasibility.py`
- Delete: `pokemon-tcg-ai-battle/tests/selfplay_pilot/test_engine_probe.py`
- Delete: `pokemon-tcg-ai-battle/tests/selfplay_pilot/test_quick_ppo.py`
- Delete: `pokemon-tcg-ai-battle/reports/mac_ppo_feasibility_latest.json`

**Interfaces:**
- Produces: `PreflightRecord`, `LeagueReport`, `immutable_snapshot(population)`, `collect_phase(scheduled_games, current, storage)`, `collect_round_two(schedule, current, history, storage)`, `evaluate_frozen_groups(starting, final, scheduled_games)`, `evaluate_ancestry(final, starting, scheduled_games)`, `run_league(config, schedule, project_root) -> LeagueReport`, and CLI modes `--dry-run` and `--standard`.
- Consumes: every preceding module and writes `reports/mac_four_policy_league_latest.json` only.

- [ ] **Step 1: Write the failing end-to-end contract test**

```python
def test_dry_run_executes_every_phase_and_leaves_no_junk(tmp_path, artifact_snapshot):
    result = subprocess.run(
        [MINIFORGE_PYTHON, "scripts/run_four_policy_league.py", "--dry-run", "--report", str(tmp_path / "report.json")],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["phases"] == ["preflight", "bootstrap", "start_gate", "round_1", "round_2", "judges", "ancestry", "decision", "cleanup"]
    assert report["self_play_audit"]["valid"] is True
    assert set(report["round_1"]["updated_members"]) == {member.value for member in MemberId}
    assert set(report["round_2"]["updated_members"]) == {member.value for member in MemberId}
    assert report["storage"]["raw_replays_written"] == 0
    assert report["storage"]["temp_run_exists_after_cleanup"] is False
    assert measure_artifacts(PROJECT / "artifacts") == artifact_snapshot

def test_expired_deadline_rejects_and_cleans(tmp_path, monkeypatch):
    monkeypatch.setattr("league_selfplay.runner.time.monotonic", monotonic_sequence([0.0, 1201.0]))
    report = run_league(replace(FrozenLeagueConfig(), wall_time_seconds=1200), build_dry_run_schedule(FrozenLeagueConfig()), PROJECT)
    assert report.decision.code == "REJECT_RUNTIME"
    assert report.storage.temp_run_exists_after_cleanup is False
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_runner.py`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement strict phase orchestration**

```python
PHASES = (
    "preflight", "bootstrap", "start_gate", "round_1", "round_2",
    "judges", "ancestry", "decision", "cleanup",
)

def run_league(config: FrozenLeagueConfig, schedule: LeagueSchedule, project_root: Path) -> LeagueReport:
    preflight = PreflightRecord.freeze(config=config, schedule=schedule, project_root=project_root)
    storage = RunStorage.create_under_tmp(config)
    with storage:
        population = create_population(config.seed, "mps")
        drivers = DriverRegistry.from_project(project_root)
        bootstrap_stats = initialize_population(
            population=population,
            drivers=drivers,
            games=len(schedule.bootstrap),
            device="mps",
            seed=config.seed,
        )
        start_gate = run_start_gate(population, import_official_cg(project_root), drivers)
        if not start_gate.teacher_closed:
            raise InvalidSelfPlay("teacher registry remained open")
        starting = immutable_snapshot(population)
        round_one_games = collect_phase(schedule.round_one, population, storage)
        audit_or_raise(round_one_games)
        round_one_stats = update_population(population, round_one_games, "mps", config.seed)
        round_one = immutable_snapshot(population)
        round_two_games = collect_round_two(schedule, population, starting, storage)
        audit_or_raise(round_two_games)
        round_two_stats = update_population(population, round_two_games, "mps", config.seed + 1)
        final = immutable_snapshot(population)
        judge_results = evaluate_frozen_groups(starting, final, schedule.judges)
        ancestry_results = evaluate_ancestry(final, starting, schedule.ancestry)
        comparison = compare_groups(judge_results.start, judge_results.final, ancestry_results, config.seed)
        decision = decide_validation(
            comparison,
            {"round_1": round_one_stats, "round_2": round_two_stats},
            failures=[],
        )
    return LeagueReport.from_run(
        preflight=preflight,
        bootstrap=bootstrap_stats,
        start_gate=start_gate,
        round_one=round_one_stats,
        round_two=round_two_stats,
        judge_results=judge_results,
        ancestry_results=ancestry_results,
        comparison=comparison,
        decision=decision,
        storage=storage.final_measurement(),
    )
```

The report must distinguish `PASS_MAC_LEAGUE`, `REJECT_NO_GROUP_IMPROVEMENT`, `INVALID_SELF_PLAY`, `REJECT_RUNTIME`, and `REJECT_STORAGE`. It must record configuration/schedule/model/feature hashes before results, all phase durations, both seat counts, per-member updates, judge data, ancestry data, peak temporary bytes, artifact measurements, and cleanup result.

- [ ] **Step 4: Delete the wrong fixed-opponent pilot files**

Use `apply_patch` to delete only the six listed obsolete source/test/report files. Keep `agents/candidate_grimmsnarl_nn_1p5m_smoke_v0` and `tests/selfplay_pilot/test_1p5m_submission.py` because they prove submission runtime capacity rather than fixed-opponent training.

- [ ] **Step 5: Run the dry league and the complete focused suite**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay tests/selfplay_pilot/test_1p5m_submission.py`

Run: `python3 scripts/run_four_policy_league.py --dry-run --report "$TMPDIR/mac-four-policy-dry.json"`

Expected: all tests PASS; dry run reaches all nine phases; both rounds update all four members; no raw replay or `artifacts/` change occurs; temporary run directory is absent afterward. The dry-run decision itself may reject group improvement because its evaluation sample is intentionally tiny.

- [ ] **Step 6: Commit the runner and wrong-pilot removal**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/runner.py pokemon-tcg-ai-battle/scripts/run_four_policy_league.py pokemon-tcg-ai-battle/tests/league_selfplay/test_runner.py pokemon-tcg-ai-battle/selfplay_pilot pokemon-tcg-ai-battle/scripts/run_mac_ppo_feasibility.py pokemon-tcg-ai-battle/tests/selfplay_pilot pokemon-tcg-ai-battle/reports/mac_ppo_feasibility_latest.json
git commit -m "feat: orchestrate four-policy Mac league"
```

### Task 11: Run the standard Mac league and issue the migration decision

**Files:**
- Create at runtime: `pokemon-tcg-ai-battle/reports/mac_four_policy_league_latest.json`
- Create only on pass: `pokemon-tcg-ai-battle/agents/four_policy_league_mac_pass/` containing four NumPy checkpoints and one manifest.

**Interfaces:**
- Consumes: committed standard runner and frozen standard schedule.
- Produces: one evidence report and either a retained passing population or no model.

- [ ] **Step 1: Record pre-run disk and repository state**

Run: `du -sk artifacts reports 2>/dev/null; git status --short`

Expected: record the existing state without deleting or modifying unrelated user files.

- [ ] **Step 2: Run the standard validation on Apple MPS**

Run: `python3 scripts/run_four_policy_league.py --standard --report reports/mac_four_policy_league_latest.json`

Expected: the command ends within 20 minutes and prints exactly one final decision code. Progress goes to stderr; the JSON report is below 100 KiB.

- [ ] **Step 3: Verify report, cleanup, and no artifact growth**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('reports/mac_four_policy_league_latest.json').read_text())
assert r['self_play_audit']['valid'] is True
assert set(r['round_1']['updated_members']) == {'grimmsnarl', 'lucario', 'crustle', 'alakazam'}
assert set(r['round_2']['updated_members']) == {'grimmsnarl', 'lucario', 'crustle', 'alakazam'}
assert r['storage']['raw_replays_written'] == 0
assert r['storage']['temp_run_exists_after_cleanup'] is False
assert r['storage']['artifacts_unchanged'] is True
assert r['decision'] in {'PASS_MAC_LEAGUE', 'REJECT_NO_GROUP_IMPROVEMENT'}
print(r['decision'])
PY
```

Expected: assertions PASS. `INVALID_SELF_PLAY`, runtime failure, or cleanup failure is reported as a defect, not interpreted as evidence about learning.

- [ ] **Step 4: Apply the migration rule without reinterpretation**

If the report says `PASS_MAC_LEAGUE`, retain the four NumPy checkpoints and authorize a separate GTX 1080 Ti plan. If it says `REJECT_NO_GROUP_IMPROVEMENT`, delete all model checkpoints and keep only the JSON report. Do not submit either result to Kaggle.

- [ ] **Step 5: Commit implementation evidence, excluding runtime models unless passed**

```bash
git add pokemon-tcg-ai-battle/reports/mac_four_policy_league_latest.json
git commit -m "test: record four-policy Mac league result"
```

Only add `agents/four_policy_league_mac_pass/` when the report decision is `PASS_MAC_LEAGUE` and its manifest hashes all four retained checkpoints.
