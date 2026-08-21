# Pokémon Single-Intervention Self-Play Implementation Plan

> **Status (2026-08-04): ACHIEVED.** The final two-round population passed a
> balanced 192-versus-192 Engine22 audit by `+5.99` percentage points. The
> frozen evidence and lessons are recorded in
> `docs/milestones/2026-08-04-pokemon-group-selfplay-mac-pass.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove on Mac that four driver-backed players can improve over two rounds by learning from one randomized action intervention per complete Engine 22 game.

**Architecture:** A pure `single_intervention` module owns calibration, intervention eligibility, pairwise examples, the two-model scorer, and conservative override decisions. A separate runner owns ordinary Engine 22 games, the two-round survivor loop, actual-game selection and confirmation, bounded storage, and reporting. The implementation reuses the existing four drivers, relational features, and approximately 66k-parameter `ActionValueNet`; neither training nor evaluation imports a search API.

**Tech Stack:** Python 3.12, NumPy, PyTorch, official `cg.game` Engine 22 battle API, pytest.

## Global Constraints

- Use official local Engine 22 with SHA-256 `7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30`.
- Four members and decks remain those in `league_selfplay.bootstrap.DRIVER_PATHS`.
- Use two independently seeded approximately 66k-parameter action scorers per member.
- Every intervention game has zero or one intervention; never two.
- Interventions are limited to non-forced single-choice `MAIN` decisions and must differ from the incumbent action.
- Draw the target eligible-decision ordinal uniformly from 1 through 32.
- Calibration per round is 12 member-opponent pairings × 2 seats × 2 repetitions = 48 complete games.
- Each member must collect exactly 32 valid interventions per round, balanced across three opponents and both seats.
- Pairwise intervention training uses AdamW `3e-4`, batch size 32, eight epochs, and labels clipped to magnitude `0.5`.
- A residual override requires both scorer gaps to exceed `0.25`.
- Selection and confirmation each use 12 independent complete games per member for candidate and incumbent comparisons as specified in the design.
- Training and evaluation must not import or call `search_begin`, `search_step`, `search_release`, or `search_end`.
- Mac wall-time is 600 seconds and temporary storage is capped at 128 MiB.
- Do not persist replays, raw observations, hidden cards, datasets, controls, or rejected checkpoints.
- `artifacts/` must have identical file count and byte size before and after.
- Keep one report below 100 KiB; retain model weights only after a fresh confirmation pass.

---

### Task 1: Calibration and one-intervention contract

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/single_intervention.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention.py`

**Interfaces:**
- Consumes: `MemberId`, visible observations, incumbent actions, and NumPy RNG.
- Produces: `CalibrationKey`, `CalibrationCell`, `InterventionTracker`, `InterventionExample`, `eligible_intervention`, `centered_label`, and `choose_trial_index`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_calibration_uses_frozen_shrunk_score_and_clipped_label():
    cell = CalibrationCell(points=2.0, games=2)
    assert cell.expected_score == 0.75
    assert centered_label(1.0, cell) == 0.25
    assert centered_label(0.0, cell) == -0.5


def test_tracker_allows_exactly_one_targeted_intervention():
    tracker = InterventionTracker(target_ordinal=2)
    assert not tracker.consider(_eligible_observation(), [0])
    assert tracker.consider(_eligible_observation(), [0])
    tracker.mark_used()
    assert not tracker.consider(_eligible_observation(), [0])
    assert tracker.eligible_seen == 2


def test_trial_action_is_legal_and_never_the_incumbent():
    rng = np.random.default_rng(7)
    for _ in range(20):
        assert choose_trial_index([3.0, 2.0, 1.0], 0, rng) in {1, 2}
```

- [ ] **Step 2: Run the focused test and verify the missing module fails**

Run: `python -m pytest tests/league_selfplay/test_single_intervention.py -q`

Expected: collection fails because `league_selfplay.single_intervention` does not exist.

- [ ] **Step 3: Implement immutable calibration and intervention records**

```python
@dataclass(frozen=True, slots=True)
class CalibrationKey:
    member: MemberId
    opponent: MemberId
    seat: int


@dataclass(frozen=True, slots=True)
class CalibrationCell:
    points: float
    games: int

    @property
    def expected_score(self) -> float:
        return (self.points + 1.0) / (self.games + 2.0)


@dataclass(slots=True)
class InterventionTracker:
    target_ordinal: int
    eligible_seen: int = 0
    used: bool = False

    def consider(self, observation, incumbent_action) -> bool:
        if self.used or not eligible_intervention(observation, incumbent_action):
            return False
        self.eligible_seen += 1
        return self.eligible_seen == self.target_ordinal

    def mark_used(self) -> None:
        if self.used:
            raise ValueError("intervention already used")
        self.used = True
```

`eligible_intervention` requires `select.type == 0`, `select.context == 0`, `minCount == maxCount == 1`, more than one option, and one valid incumbent index. `centered_label` returns `clip(actual_score - expected_score, -0.5, 0.5)`. `choose_trial_index` removes the incumbent index, applies a stable temperature-1 softmax, and samples only remaining indices.

- [ ] **Step 4: Run the contract tests**

Run: `python -m pytest tests/league_selfplay/test_single_intervention.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/single_intervention.py pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention.py
git commit -m "test: freeze single intervention contract"
```

### Task 2: Two-model pairwise learner and trusted overrides

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/single_intervention.py`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention.py`

**Interfaces:**
- Consumes: `ActionValueNet`, `encode_options`, incumbent calibration decisions, and `InterventionExample` rows.
- Produces: `InterventionEnsemble`, `create_intervention_population`, `pretrain_incumbent_population`, `update_intervention_ensemble`, `update_intervention_population`, `mean_option_scores`, and `trusted_override`.

- [ ] **Step 1: Add failing learner tests**

```python
def test_population_has_two_independent_small_models_per_member():
    population = create_intervention_population(20260804, "cpu")
    assert set(population) == set(MemberId)
    for ensemble in population.values():
        assert len(ensemble.models) == 2
        assert sum(action_value_parameter_count(m) for m in ensemble.models) < 140_000
        assert ensemble.models[0].layer.weight.data_ptr() != ensemble.models[1].layer.weight.data_ptr()


def test_trusted_override_requires_both_gaps_above_margin():
    assert trusted_override(([0.0, 0.4], [0.0, 0.3]), 0, margin=0.25) == 1
    assert trusted_override(([0.0, 0.4], [0.0, 0.2]), 0, margin=0.25) is None


def test_pairwise_update_moves_positive_trial_above_incumbent():
    ensemble = create_intervention_population(9, "cpu")[MemberId.LUCARIO]
    rows = [_positive_example()] * 32
    before = tuple(model.layer.weight.detach().clone() for model in ensemble.models)
    stats = update_intervention_ensemble(ensemble, rows, "cpu", seed=10)
    assert stats.examples == 32
    assert stats.all_finite
    assert all(not torch.equal(old, model.layer.weight) for old, model in zip(before, ensemble.models))
```

- [ ] **Step 2: Run tests and verify missing learner interfaces fail**

Run: `python -m pytest tests/league_selfplay/test_single_intervention.py -q`

Expected: import or assertion failures for the unimplemented ensemble functions.

- [ ] **Step 3: Implement the ensemble and pairwise loss**

```python
@dataclass(slots=True)
class InterventionEnsemble:
    models: tuple[ActionValueNet, ActionValueNet]


def _pairwise_loss(model, batch, device):
    padded, mask = _padded_examples(batch)
    values, _, _ = model(torch.from_numpy(padded).to(device),
                         torch.from_numpy(mask).to(device))
    row = torch.arange(len(batch), device=device)
    trial = values[row, torch.tensor([x.trial_index for x in batch], device=device)]
    incumbent = values[row, torch.tensor([x.incumbent_index for x in batch], device=device)]
    labels = torch.tensor([x.label for x in batch], dtype=torch.float32, device=device)
    signs = torch.sign(labels)
    weights = torch.maximum(labels.abs(), labels.new_tensor(0.05))
    return (functional.softplus(-signs * (trial - incumbent)) * weights).sum() / weights.sum()
```

Create each model with seed `seed + member_index * 100 + model_index`. Pretraining uses the same pairwise form with the incumbent as the positive side for at most 2,048 calibration decisions per member, two epochs. Intervention updates use only non-zero current-round labels, AdamW `3e-4`, batch 32, eight epochs, independent batch-order seeds, gradient clipping at 1.0, and finite/delta statistics.

- [ ] **Step 4: Run learner and existing action-value tests**

Run: `python -m pytest tests/league_selfplay/test_single_intervention.py tests/league_selfplay/test_action_value.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the learner**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/single_intervention.py pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention.py
git commit -m "feat: add conservative intervention learner"
```

### Task 3: Ordinary-game calibration and intervention collection

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/single_intervention_runner.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention_runner.py`

**Interfaces:**
- Consumes: a game object exposing only `battle_start`, `battle_select`, and `battle_finish`; `DriverRegistry`; survivor ensembles; and the pure Task 1/2 interfaces.
- Produces: `PopulationPolicy`, `ActualGameResult`, `InterventionOutcome`, `run_actual_game`, `collect_calibration`, and `collect_interventions`.

- [ ] **Step 1: Write failing fake-engine tests**

```python
def test_actual_game_changes_only_target_action_then_returns_to_incumbent():
    game = FakeBattleGame(decisions=5, winner=0)
    policy = RecordingPolicy()
    result = run_actual_game(
        game,
        policy,
        members=(MemberId.GRIMMSNARL, MemberId.LUCARIO),
        experimental_member=MemberId.GRIMMSNARL,
        target_ordinal=2,
        trial_selector=lambda features, incumbent, rng: 1,
        rng=np.random.default_rng(1),
    )
    assert result.intervention is not None
    assert result.intervention.incumbent_index == 0
    assert result.intervention.trial_index == 1
    assert game.actions == [[0], [0], [1], [0], [0]]
    assert sum(action == [1] for action in game.actions) == 1


def test_game_without_target_is_control_not_training_label():
    result = run_actual_game(
        FakeBattleGame(decisions=2, winner=1),
        RecordingPolicy(),
        members=(MemberId.CRUSTLE, MemberId.ALAKAZAM),
        experimental_member=MemberId.CRUSTLE,
        target_ordinal=32,
        trial_selector=lambda features, incumbent, rng: 1,
        rng=np.random.default_rng(2),
    )
    assert result.intervention is None
    assert result.control
```

The fake game deliberately has no search methods. Add a source assertion that `single_intervention_runner.py` contains none of the four forbidden search API names.

- [ ] **Step 2: Run runner tests and verify the missing module fails**

Run: `python -m pytest tests/league_selfplay/test_single_intervention_runner.py -q`

Expected: collection fails because the runner does not exist.

- [ ] **Step 3: Implement the actual-game loop and adapters**

```python
@dataclass(slots=True)
class PopulationPolicy:
    registry: DriverRegistry
    ensembles: Mapping[MemberId, InterventionEnsemble]
    enabled_members: frozenset[MemberId]

    def decide(self, member, observation):
        incumbent = self.registry.action(member, observation)
        if member not in self.enabled_members:
            return incumbent
        features = encode_options(observation)
        override = trusted_override(
            tuple(model_scores(model, features) for model in self.ensembles[member].models),
            incumbent[0],
            margin=0.25,
        )
        return incumbent if override is None else [override]
```

`run_actual_game` calls `battle_finish` in `finally`, records terminal score as 1/0.5/0 from the experimental member's seat, stores only visible feature tensors and action indices in memory, and rejects invalid actions through the existing `validate_action`. After `tracker.mark_used()`, every later action comes from `PopulationPolicy.decide`.

`collect_calibration` runs the exact 48-game schedule, freezes all 24 member-opponent-seat cells at two games each, and keeps at most 2,048 eligible incumbent decisions per member in memory. `collect_interventions` rotates the six opponent-seat cells so each member gets five examples per cell plus two deterministic extra cells, continuing controls until all members reach 32 valid rows or the shared deadline fires.

- [ ] **Step 4: Run runner, learner, and engine contract tests**

Run: `python -m pytest tests/league_selfplay/test_single_intervention_runner.py tests/league_selfplay/test_single_intervention.py tests/league_selfplay/test_engine.py -q`

Expected: all tests pass and the source search-API prohibition passes.

- [ ] **Step 5: Commit actual-game collection**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/single_intervention_runner.py pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention_runner.py
git commit -m "feat: collect single interventions from actual games"
```

### Task 4: Two-round survivors, fresh confirmation, and Mac proof

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/single_intervention_runner.py`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention_runner.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_single_intervention_selfplay.py`
- Create at runtime: `pokemon-tcg-ai-battle/reports/mac_single_intervention_selfplay_latest.json`

**Interfaces:**
- Consumes: Task 2 populations, Task 3 collection functions, official Engine 22, existing storage measurement and cleanup patterns.
- Produces: `RoundSummary`, `ActualComparison`, `SingleInterventionDecision`, `SingleInterventionReport`, `select_survivors`, `decide_single_intervention_proof`, and `run_single_intervention_proof`.

- [ ] **Step 1: Add failing survivor and orchestration tests**

```python
def test_round_two_uses_only_positive_round_one_survivors():
    promoted = select_survivors({
        MemberId.GRIMMSNARL: 0.10,
        MemberId.LUCARIO: 0.0,
        MemberId.CRUSTLE: -0.10,
        MemberId.ALAKAZAM: 0.05,
    })
    assert promoted == (MemberId.GRIMMSNARL, MemberId.ALAKAZAM)


def test_proof_requires_two_positive_batches_and_no_promoted_regression():
    passed = decide_single_intervention_proof(
        selection=_comparison(0.05, all_promoted_positive=True),
        confirmation=_comparison(0.04, all_promoted_positive=True),
        promoted=(MemberId.GRIMMSNARL,),
        overrides=3,
        failures=(),
    )
    assert passed.code == "PASS_SINGLE_INTERVENTION_MAC"
    rejected = decide_single_intervention_proof(
        selection=_comparison(0.05, all_promoted_positive=True),
        confirmation=_comparison(-0.01, all_promoted_positive=False),
        promoted=(MemberId.GRIMMSNARL,),
        overrides=3,
        failures=(),
    )
    assert rejected.code == "REJECT_NO_GROUP_IMPROVEMENT"


def test_failed_proof_cleans_temp_and_does_not_retain_models(tmp_path):
    report = run_single_intervention_proof(
        tmp_path,
        dependencies=failing_fake_dependencies(),
        wall_time_seconds=600,
    )
    assert not report.decision.passed
    assert not report.storage_root.exists()
    assert report.artifacts_before == report.artifacts_after
    assert report.retained_population is None
```

- [ ] **Step 2: Run orchestration tests and verify missing interfaces fail**

Run: `python -m pytest tests/league_selfplay/test_single_intervention_runner.py -q`

Expected: failures for the unimplemented survivor, decision, and proof runner interfaces.

- [ ] **Step 3: Implement two rounds and actual-game evaluation**

For each round: clone survivor ensembles, collect 48 calibration games, run two-epoch incumbent pretraining, collect 32 interventions per member, apply eight-epoch pairwise updates, then evaluate candidate and incumbent independently against the other three round incumbents in both seats and two repetitions. Promote only strictly positive per-member deltas; restore the pre-round ensemble and incumbent status for every rejected member.

After round 2, evaluate the provisional population and untouched original drivers on a disjoint fresh schedule against the other three untouched drivers. `ActualComparison` records separate candidate/incumbent game counts, group scores, delta, per-member deltas, and `same_schedule=True`, `paired_randomness=False`.

The final decision passes only when selection group delta and confirmation group delta are positive, every promoted member is positive in both, at least one member is promoted, at least one evaluation override occurs, updates are finite, sample counts are exact, and no operational failure exists.

- [ ] **Step 4: Implement bounded storage, report, and CLI**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "reports/mac_single_intervention_selfplay_latest.json")
    args = parser.parse_args()
    report = run_single_intervention_proof(args.project_root)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if len(payload.encode("utf-8")) >= 100 * 1024:
        raise SystemExit("report exceeds 100 KiB")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    print(json.dumps(report.summary(), indent=2, sort_keys=True))
```

Use an owned `pokemon-single-intervention-*` system-temp directory, the existing 128 MiB quota/cleanup handler pattern, a 600-second shared deadline, Engine 22 hash preflight, and `artifacts/` before/after measurement. Rejected runs delete every checkpoint; passing weights move atomically to `agents/single_intervention_mac_pass/`.

- [ ] **Step 5: Run unit and relevant regression tests**

Run: `python -m pytest tests/league_selfplay/test_single_intervention.py tests/league_selfplay/test_single_intervention_runner.py tests/league_selfplay/test_action_value.py tests/league_selfplay/test_residual.py tests/league_selfplay/test_storage.py -q`

Expected: all tests pass.

- [ ] **Step 6: Run the real Mac proof once**

Run: `PYTHONDONTWRITEBYTECODE=1 python scripts/run_single_intervention_selfplay.py`

Expected: completes or rejects at the 600-second boundary, produces one report under 100 KiB, writes no replay/dataset/raw-state files, cleans its temporary directory, leaves `artifacts/` unchanged, and prints exactly one frozen decision code. Do not rerun merely to obtain a more favorable random result.

- [ ] **Step 7: Verify evidence and commit**

Run: `python -m pytest tests/league_selfplay/test_single_intervention.py tests/league_selfplay/test_single_intervention_runner.py tests/league_selfplay/test_action_value.py tests/league_selfplay/test_residual.py tests/league_selfplay/test_storage.py -q`

Run: `git diff --check && du -h reports/mac_single_intervention_selfplay_latest.json`

Expected: tests pass; report is below 100 KiB; no temporary run directory remains; only planned source, tests, CLI, plan, report, and a confirmed passing model directory if applicable are new.

```bash
git add pokemon-tcg-ai-battle/league_selfplay/single_intervention.py pokemon-tcg-ai-battle/league_selfplay/single_intervention_runner.py pokemon-tcg-ai-battle/scripts/run_single_intervention_selfplay.py pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention.py pokemon-tcg-ai-battle/tests/league_selfplay/test_single_intervention_runner.py pokemon-tcg-ai-battle/reports/mac_single_intervention_selfplay_latest.json
git commit -m "test: validate single intervention self-play"
```
