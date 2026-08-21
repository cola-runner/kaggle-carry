# Pokémon Driver Fidelity Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a short Mac experiment that proves whether a lossless 512-wide visible-state encoder lets the existing 1.5M neural policies reproduce their four original drivers better than the current encoder, before any PPO is allowed.

**Architecture:** Collect each driver decision once and encode it through both V1 and V2, then split complete games into identical train and held-out sets. Train two otherwise-identical four-policy populations and compare held-out ordered-action agreement and negative log probability. A pure decision gate rejects the run unless V2 wins the predeclared comparison; the runner writes one small report and never calls PPO.

**Tech Stack:** Python 3.12 Miniforge runtime, NumPy, PyTorch/MPS, official `cg` engine, pytest, existing `league_selfplay` policy/action/storage contracts.

## Global Constraints

- The output remains 512 float32 values per legal option.
- Hidden opponent cards and `search_begin_input` must not affect features.
- The 1,478,370-policy-parameter model and NumPy forward runtime remain unchanged.
- V1 and V2 use identical games, labels, split, model seeds, optimizer, and epochs.
- Collect exactly 96 balanced driver games for the standard run: 72 train and 24 held out by whole game.
- No PPO, terminal reward update, final-judge feedback, or self-play learning is allowed.
- Raw replays are never written; temporary data is capped at 128 MiB and removed on every exit.
- Wall time is six minutes; the only persistent output is `reports/mac_driver_fidelity_latest.json`.
- A failed run retains no model and may not move to the GTX 1080 Ti.

---

### Task 1: Encode visible numeric and categorical information without the V1 clipping failure

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/features.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_features.py`

**Interfaces:**
- Consumes: sanitized observations and `visible_action_features(...)`.
- Produces: `encode_options_v2(raw_observation) -> np.ndarray` and `FEATURE_V2_VERSION = 1`.

- [ ] **Step 1: Write failing distinction and visibility tests**

```python
def test_v2_distinguishes_visible_hp_deck_source_and_target():
    base = fixture_observation()
    hp_changed = visible_copy(base, own_active_hp=120)
    deck_changed = visible_copy(base, own_deck_count=9)
    source_changed = option_copy(base, source_card_id=999)
    target_changed = option_copy(base, target_card_id=998)
    encoded = encode_options_v2(base)
    assert not np.array_equal(encoded, encode_options_v2(hp_changed))
    assert not np.array_equal(encoded, encode_options_v2(deck_changed))
    assert not np.array_equal(encoded, encode_options_v2(source_changed))
    assert not np.array_equal(encoded, encode_options_v2(target_changed))

def test_v2_ignores_hidden_payloads():
    visible = fixture_observation()
    poisoned = poison_hidden_opponent_data(visible)
    np.testing.assert_array_equal(encode_options_v2(visible), encode_options_v2(poisoned))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_features.py`

Expected: FAIL because `encode_options_v2` does not exist.

- [ ] **Step 3: Implement a partitioned 512-wide encoder**

Implement these fixed regions in `features.py`:

```python
FEATURE_V2_VERSION = 1
NUMERIC_BUCKETS = 128
CATEGORICAL_BUCKETS = INPUT_WIDTH - NUMERIC_BUCKETS

def _scaled_numeric(name: str, value: float) -> float:
    if "hp" in name or "damage" in name:
        return np.clip(value / 300.0, -2.0, 2.0)
    if "deck_count" in name:
        return np.clip(value / 60.0, -2.0, 2.0)
    if name.endswith("step"):
        return np.clip(value / 500.0, -2.0, 2.0)
    if name.endswith("turn"):
        return np.clip(value / 50.0, -2.0, 2.0)
    if any(part in name for part in ("card_id", "serial", "attack_id")):
        return 0.0
    return np.clip(value / 8.0, -2.0, 2.0)

def _dense_hash_v2(features: Mapping[str, float]) -> np.ndarray:
    vector = np.zeros(INPUT_WIDTH, dtype=np.float32)
    for name, raw_value in features.items():
        value = float(raw_value)
        if not math.isfinite(value) or value == 0.0:
            continue
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        sign = -1.0 if number >> 63 else 1.0
        if name.startswith("n:"):
            index = number % NUMERIC_BUCKETS
            contribution = _scaled_numeric(name, value)
        else:
            index = NUMERIC_BUCKETS + number % CATEGORICAL_BUCKETS
            contribution = np.clip(value, -8.0, 8.0)
        vector[index] += np.float32(sign * contribution)
    return vector

def encode_options_v2(raw_observation: Mapping[str, Any]) -> np.ndarray:
    return _encode_options(raw_observation, _dense_hash_v2)
```

Refactor the existing `encode_options` through `_encode_options` without changing V1 output.

- [ ] **Step 4: Verify V2 tests and existing feature parity**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_features.py tests/league_selfplay/test_model.py`

Expected: all tests PASS, V1 fixture output remains unchanged, and V2 is finite float32 with shape `[legal_options, 512]`.

- [ ] **Step 5: Commit the encoder**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/features.py pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_features.py
git commit -m "feat: preserve visible decision information"
```

### Task 2: Collect paired V1/V2 labels and split only at game boundaries

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/fidelity_data.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_data.py`

**Interfaces:**
- Consumes: `DriverRegistry`, official observations, `encode_options`, and `encode_options_v2`.
- Produces: `PairedDecision`, `PairedGame`, `collect_paired_driver_game(...)`, `collect_paired_games(...)`, and `split_games(games, train_games) -> tuple[tuple[PairedGame, ...], tuple[PairedGame, ...]]`.

- [ ] **Step 1: Write failing paired-collection and leakage tests**

```python
def test_split_never_places_one_game_on_both_sides():
    games = synthetic_paired_games(12)
    train, held_out = split_games(games, train_games=9)
    assert {game.game_id for game in train}.isdisjoint(
        {game.game_id for game in held_out}
    )
    assert len(train) == 9
    assert len(held_out) == 3

def test_one_engine_decision_has_identical_label_for_v1_and_v2(project_root):
    report = paired_collection_smoke(project_root)
    assert report["finished"] is True
    assert report["same_labels"] is True
    assert report["v1_width"] == report["v2_width"] == 512
    assert report["raw_replays_written"] == 0
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_data.py`

Expected: FAIL because `fidelity_data` is absent.

- [ ] **Step 3: Implement paired collection**

```python
@dataclass(frozen=True, slots=True)
class PairedDecision:
    member: MemberId
    game_id: int
    v1_features: np.ndarray
    v2_features: np.ndarray
    action: tuple[int, ...]
    min_count: int
    max_count: int

@dataclass(frozen=True, slots=True)
class PairedGame:
    game_id: int
    members: tuple[MemberId, MemberId]
    decisions: tuple[PairedDecision, ...]
    finished: bool
```

At every non-empty selection, compute both feature matrices before invoking `battle_select`; attach the one driver action to both matrices. Schedule the twelve ordered pairings cyclically so 72 and 24 games each contain complete balanced cycles.

- [ ] **Step 4: Verify real collection and deterministic split**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_data.py`

Expected: paired labels PASS, train/held-out game IDs do not overlap, and the engine smoke writes no files.

- [ ] **Step 5: Commit paired data collection**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/fidelity_data.py pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_data.py
git commit -m "feat: collect paired driver fidelity data"
```

### Task 3: Train identical populations and score only non-trivial held-out choices

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/actions.py`
- Create: `pokemon-tcg-ai-battle/league_selfplay/fidelity_train.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_train.py`

**Interfaces:**
- Consumes: `PairedGame`, `train_from_decisions`, `create_population`, and exact ordered log probabilities.
- Produces: `greedy_action(...)`, `FidelityMemberMetrics`, `FidelityMetrics`, `train_paired_populations(...)`, and `evaluate_population(...)`.

- [ ] **Step 1: Write failing deterministic-action and metric tests**

```python
def test_greedy_action_obeys_stop_and_selection_bounds():
    assert greedy_action(np.array([3.0, 1.0]), -5.0, 1, 1) == (0,)
    assert greedy_action(np.array([1.0, 0.0]), 2.0, 0, 2) == ()

def test_nontrivial_metrics_ignore_forced_single_option_decisions():
    metrics = fidelity_metric_smoke()
    assert metrics["nontrivial_decisions"] == 8
    assert metrics["forced_decisions"] == 4
    assert metrics["exact_agreement"] == 1.0
    assert metrics["negative_log_probability"] >= 0.0

def test_v2_torch_and_numpy_logits_match_on_held_out_features(tmp_path):
    report = fidelity_numpy_parity_smoke(tmp_path)
    assert report["max_option_error"] < 1e-4
    assert report["max_stop_error"] < 1e-4
    assert report["all_greedy_actions_legal"] is True
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_train.py`

Expected: FAIL because greedy inference and fidelity metrics are absent.

- [ ] **Step 3: Implement greedy ordered selection and identical training**

```python
def greedy_action(option_logits, stop_logit, min_count, max_count):
    selected = []
    while len(selected) < max_count:
        remaining = [i for i in range(len(option_logits)) if i not in selected]
        can_stop = len(selected) >= min_count and min_count != max_count
        candidates = [(float(option_logits[i]), i) for i in remaining]
        if can_stop and stop_logit >= max(score for score, _ in candidates):
            break
        selected.append(max(candidates)[1])
    return tuple(selected)
```

Convert each paired training decision into one V1 and one V2 `BootstrapDecision`. Create both populations with the same seed and call `train_from_decisions` with identical epochs, batch size, learning rate, and member order.

For held-out decisions, report exact ordered agreement and mean negative `action_log_probability`. A decision is non-trivial when it has more than one legal option or permits STOP; all other decisions go only into `forced_decisions`.

For one held-out batch per member, export each policy to a temporary NPZ,
compare PyTorch and NumPy option/STOP logits, verify maximum error below
`1e-4`, and delete every checkpoint before returning.

Use these immutable report types:

```python
@dataclass(frozen=True, slots=True)
class FidelityMemberMetrics:
    nontrivial_decisions: int
    forced_decisions: int
    exact_agreement: float
    negative_log_probability: float

@dataclass(frozen=True, slots=True)
class FidelityMetrics:
    nontrivial_decisions: int
    forced_decisions: int
    exact_agreement: float
    negative_log_probability: float
    members: dict[MemberId, FidelityMemberMetrics]
```

- [ ] **Step 4: Verify metrics and model regression suite**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_train.py tests/league_selfplay/test_actions.py tests/league_selfplay/test_bootstrap.py`

Expected: all tests PASS and all four V1/V2 populations receive finite positive parameter changes.

- [ ] **Step 5: Commit paired training and scoring**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/actions.py pokemon-tcg-ai-battle/league_selfplay/fidelity_train.py pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_train.py
git commit -m "feat: score held-out driver fidelity"
```

### Task 4: Freeze the pass/reject decision independently of training code

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/fidelity_gate.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_gate.py`

**Interfaces:**
- Consumes: aggregate and per-member `FidelityMetrics` for V1 and V2 plus failure strings.
- Produces: `FidelityDecision` and `decide_fidelity(v1, v2, failures)`.

- [ ] **Step 1: Write failing accept/reject tests**

```python
def test_clear_relative_improvement_passes():
    decision = decide_fidelity(
        metrics(nll=1.0, agreement=.50, members=uniform(1.0, .50)),
        metrics(nll=.80, agreement=.58, members=uniform(.80, .58)),
        [],
    )
    assert decision.code == "PASS_DRIVER_FIDELITY_V2"

def test_one_bad_member_rejects_even_when_group_improves():
    v1 = metrics(nll=1.0, agreement=.50, members=uniform(1.0, .50))
    v2 = metrics(nll=.80, agreement=.58, members=with_bad_member(1.10, .40))
    assert decide_fidelity(v1, v2, []).code == "REJECT_DRIVER_FIDELITY"
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_gate.py`

Expected: FAIL because the gate is absent.

- [ ] **Step 3: Implement the exact frozen thresholds**

```python
passes = (
    v2.negative_log_probability <= 0.90 * v1.negative_log_probability
    and v2.exact_agreement >= v1.exact_agreement + 0.05
    and improved_members >= 3
    and all(
        v2.members[m].negative_log_probability
        <= 1.05 * v1.members[m].negative_log_probability
        for m in MemberId
    )
    and not failures
)
```

Return `PASS_DRIVER_FIDELITY_V2` only for `passes`; otherwise return `REJECT_DRIVER_FIDELITY` with machine-readable failed conditions.

- [ ] **Step 4: Verify all gate branches**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_gate.py`

Expected: clear improvements pass; insufficient NLL, insufficient agreement, fewer than three improved members, one regressing member, and external failures reject.

- [ ] **Step 5: Commit the pure gate**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/fidelity_gate.py pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_gate.py
git commit -m "feat: gate driver fidelity before self-play"
```

### Task 5: Orchestrate a bounded Mac fidelity run with no PPO path

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/fidelity_runner.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_driver_fidelity.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_runner.py`

**Interfaces:**
- Consumes: Tasks 1-4, `FrozenLeagueConfig` storage safety, and `DriverRegistry`.
- Produces: `FidelityRunReport`, `run_fidelity(project_root, games, train_games, seed, wall_time_seconds)`, atomic JSON output, CLI `--dry-run` and `--standard`.

- [ ] **Step 1: Write the failing end-to-end contract test**

```python
def test_dry_fidelity_run_never_enters_ppo_or_leaves_junk(tmp_path):
    result = subprocess.run(
        [MINIFORGE, "scripts/run_driver_fidelity.py", "--dry-run", "--report", tmp_path / "report.json"],
        cwd=PROJECT,
        timeout=180,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["phases"] == ["preflight", "collect", "split", "train", "held_out", "decision", "cleanup"]
    assert report["ppo_calls"] == 0
    assert report["raw_replays_written"] == 0
    assert report["train_game_ids_overlap_held_out"] is False
    assert report["temp_run_exists_after_cleanup"] is False
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_runner.py`

Expected: FAIL because the runner and CLI are absent.

- [ ] **Step 3: Implement strict orchestration**

Use phases in this exact order:

```python
PHASES = ("preflight", "collect", "split", "train", "held_out", "decision", "cleanup")
```

Dry mode uses 24 games split 12/12 so both halves contain one complete balanced pairing cycle, and trains for one epoch. Standard mode uses 96 games split 72/24 and six epochs. Both modes create one `RunStorage` under macOS temp with `quota_bytes=128 * 1024**2`, install cleanup handlers, enforce a six-minute deadline at game and phase boundaries, record `ppo_calls=0`, and verify `artifacts/` before/after measurements. Do not import or call `league_selfplay.ppo` from this module.

Write the report through `.partial`, `fsync`, and `os.replace`. Retain no checkpoint in this task even when the relative gate passes; a later development-opponent strength test owns retention.

- [ ] **Step 4: Run the focused suite and dry CLI**

Run: `.venv/bin/python -m pytest -q tests/league_selfplay/test_fidelity_features.py tests/league_selfplay/test_fidelity_data.py tests/league_selfplay/test_fidelity_train.py tests/league_selfplay/test_fidelity_gate.py tests/league_selfplay/test_fidelity_runner.py`

Run: `python3 scripts/run_driver_fidelity.py --dry-run --report "$TMPDIR/mac-driver-fidelity-dry.json"`

Expected: all phases finish, V1/V2 train on the same labels, PPO calls remain zero, and the temporary directory is absent. The dry decision may reject because it has only one balanced held-out cycle.

- [ ] **Step 5: Commit the bounded runner**

```bash
git add pokemon-tcg-ai-battle/league_selfplay/fidelity_runner.py pokemon-tcg-ai-battle/scripts/run_driver_fidelity.py pokemon-tcg-ai-battle/tests/league_selfplay/test_fidelity_runner.py
git commit -m "feat: run bounded driver fidelity gate"
```

### Task 6: Run the standard fidelity experiment and stop or advance

**Files:**
- Create at runtime: `pokemon-tcg-ai-battle/reports/mac_driver_fidelity_latest.json`

**Interfaces:**
- Consumes: committed standard runner.
- Produces: one evidence report and a decision; never a trained checkpoint.

- [ ] **Step 1: Record disk and repository state**

Run: `du -sk artifacts reports 2>/dev/null; git status --short`

Expected: record existing state without modifying unrelated user files.

- [ ] **Step 2: Run the standard six-minute gate**

Run: `python3 scripts/run_driver_fidelity.py --standard --report reports/mac_driver_fidelity_latest.json`

Expected: 96 games collected once, 72/24 whole-game split, both populations trained identically, no PPO call, and one pass/reject result.

- [ ] **Step 3: Verify cleanup and evidence**

Run: `du -sk artifacts reports 2>/dev/null; find "$TMPDIR" -maxdepth 1 -type d -name 'pokemon-fidelity-*' -print`

Expected: `artifacts/` is byte-identical, no fidelity temp directory remains, report is below 100 KiB, and no model directory was created.

- [ ] **Step 4: Commit the evidence report**

```bash
git add pokemon-tcg-ai-battle/reports/mac_driver_fidelity_latest.json
git commit -m "test: record Mac driver fidelity gate"
```

- [ ] **Step 5: Issue the next decision**

If the report code is `PASS_DRIVER_FIDELITY_V2`, design a separate development-opponent strength test before any PPO. If it is `REJECT_DRIVER_FIDELITY`, stop training and use its per-member NLL/agreement evidence to decide whether the next isolated hypothesis is model capacity, feature aliasing, or insufficient state coverage.
