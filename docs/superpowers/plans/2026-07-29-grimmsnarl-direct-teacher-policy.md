# Grimmsnarl Direct-Teacher Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a standalone Grimmsnarl agent that imitates the current rank-3 exact-deck teacher with dedicated policies for main, bench, search, and discard decisions.

**Architecture:** Freeze one verified direct teacher into the existing chronological replay format, add missing effect-aware categorical features, and train two exported option rankers per decision context. A deterministic selector converts averaged semantic option scores into legal single- or multi-option actions, while unsupported contexts retain the existing legal fallback.

**Tech Stack:** Python 3.14, pytest, scikit-learn histogram gradient boosting for training only, pure-Python exported tree inference, Kaggle CLI, official local PTCG engine.

## Global Constraints

- Deck fingerprint remains exactly `b8f251a476e7`.
- Direct teacher is Dries @ Tufa Labs, team `16531269`, submission `55002825`.
- Training features use only information visible to the acting player.
- Training never accesses audit-split labels.
- The existing multi-teacher minimum remains the default; direct-teacher mode must be explicitly requested.
- No second Kaggle submission before a six-hour non-overlapping prospective window passes.
- Candidate runtime has no network or native machine-learning dependency.
- All production behavior changes follow test-first red-green-refactor.

---

### Task 1: Explicit Direct-Teacher Snapshot Mode

**Files:**
- Create: `pokemon-tcg-ai-battle/configs/grimmsnarl_direct_teacher_20260729.json`
- Modify: `pokemon-tcg-ai-battle/rolling_policy/snapshot.py`
- Modify: `pokemon-tcg-ai-battle/scripts/freeze_rolling_snapshot.py`
- Modify: `pokemon-tcg-ai-battle/tests/rolling/test_snapshot.py`
- Modify: `pokemon-tcg-ai-battle/tests/rolling/test_cli.py`

**Interfaces:**
- Consumes: existing `TeacherSubmission` and `eligible_teachers`.
- Produces:
  - `eligible_teachers(..., minimum_teacher_teams: int = MIN_TEACHER_TEAMS)`;
  - `freeze_snapshot(..., minimum_teacher_teams: int = MIN_TEACHER_TEAMS)`;
  - CLI flag `--minimum-teacher-teams`.

- [ ] **Step 1: Write failing eligibility tests**

Add tests proving that a single exact-deck top-10 teacher passes only when
`minimum_teacher_teams=1`, while the default still raises:

```python
def test_direct_teacher_mode_explicitly_allows_one_team() -> None:
    eligible = eligible_teachers(
        [teacher("16531269", "55002825", 1173.5)],
        rank_ten_score=1134.6,
        active_submission_ids={"55002825"},
        minimum_teacher_teams=1,
    )
    assert [row.submission_id for row in eligible] == ["55002825"]
```

- [ ] **Step 2: Run the tests and verify the expected argument failure**

Run:

```bash
cd pokemon-tcg-ai-battle
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_snapshot.py tests/rolling/test_cli.py
```

Expected: failure because `eligible_teachers` and the CLI do not accept the new
argument.

- [ ] **Step 3: Implement the explicit minimum**

Change the function signature to:

```python
def eligible_teachers(
    candidates: Sequence[TeacherSubmission],
    rank_ten_score: float,
    active_submission_ids: Collection[str],
    *,
    minimum_teacher_teams: int = MIN_TEACHER_TEAMS,
) -> tuple[TeacherSubmission, ...]:
```

Reject values below one and compare the eligible team count with this argument.
Add the CLI argument with default `MIN_TEACHER_TEAMS` and pass it into snapshot
freezing. Create the direct-teacher config with only team `16531269`,
submission `55002825`, and fingerprint `b8f251a476e7`.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_snapshot.py tests/rolling/test_cli.py
PYTHONPATH=. .venv/bin/pytest -q tests/rolling
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pokemon-tcg-ai-battle/configs/grimmsnarl_direct_teacher_20260729.json \
  pokemon-tcg-ai-battle/rolling_policy/snapshot.py \
  pokemon-tcg-ai-battle/scripts/freeze_rolling_snapshot.py \
  pokemon-tcg-ai-battle/tests/rolling/test_snapshot.py \
  pokemon-tcg-ai-battle/tests/rolling/test_cli.py
git commit -m "feat: support explicit direct-teacher snapshots"
```

### Task 2: Freeze and Extract the Direct Teacher

**Files:**
- Generate: `pokemon-tcg-ai-battle/artifacts/direct_grimmsnarl/20260729T070000Z/snapshot.json`
- Generate: `pokemon-tcg-ai-battle/artifacts/direct_grimmsnarl/20260729T070000Z/replay_inventory.jsonl`
- Generate: `pokemon-tcg-ai-battle/artifacts/direct_grimmsnarl/20260729T070000Z/public/decisions.jsonl`

**Interfaces:**
- Consumes: Task 1 CLI and direct-teacher config.
- Produces: immutable snapshot path used by every later training command.

- [ ] **Step 1: Freeze the live cutoff**

Run:

```bash
.venv/bin/python scripts/freeze_rolling_snapshot.py \
  --teacher-candidates configs/grimmsnarl_direct_teacher_20260729.json \
  --minimum-teacher-teams 1 \
  --cutoff 2026-07-29T07:00:00Z \
  --implementation-started 2026-07-29T07:00:00Z \
  --out-root artifacts/direct_grimmsnarl
```

Record the printed snapshot path and do not modify its raw inputs.

- [ ] **Step 2: Download and verify replays**

Run:

```bash
.venv/bin/python scripts/download_rolling_replays.py \
  --snapshot artifacts/direct_grimmsnarl/20260729T070000Z/snapshot.json \
  --jobs 8 --retries 5 --request-interval 0.8
```

Expected: at least 500 verified replay rows and exact fingerprint
`b8f251a476e7`.

- [ ] **Step 3: Extract visible decisions**

Run:

```bash
.venv/bin/python scripts/extract_rolling_dataset.py \
  --snapshot artifacts/direct_grimmsnarl/20260729T070000Z/snapshot.json
```

Expected: extraction report `PASS`, at least 5,000 single-choice main
decisions, and no duplicate `decision_id`.

- [ ] **Step 4: Record immutable hashes**

Run:

```bash
jq '{snapshot_id, cutoff_utc, teachers}' artifacts/direct_grimmsnarl/20260729T070000Z/snapshot.json
jq '{counts, output_sha256}' artifacts/direct_grimmsnarl/20260729T070000Z/extraction_report.json
```

Do not commit multi-gigabyte replay artifacts.

### Task 3: Effect-Aware Categorical Features

**Files:**
- Modify: `pokemon-tcg-ai-battle/rolling_policy/features.py`
- Modify: `pokemon-tcg-ai-battle/tests/rolling/test_features.py`

**Interfaces:**
- Consumes: sanitized visible observation and one legal option.
- Produces: unchanged APIs `visible_state_features(...)` and `visible_action_features(...)` with additional sparse keys.

- [ ] **Step 1: Write failing feature tests**

Add a fixture mutation with `select.effect.id=1152`,
`select.contextCard.id=1259`, context 7, and option source card 647. Assert:

```python
assert state["cat:select_effect_card=1152"] == 1.0
assert state["cat:select_context_card=1259"] == 1.0
assert option["cat:action_source_card=647"] == 1.0
assert option["cat:context=7|effect=1152|source=647"] == 1.0
```

Also assert no forbidden hidden keys are introduced.

- [ ] **Step 2: Run the feature test and observe missing-key failure**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_features.py
```

Expected: `KeyError` for `cat:select_effect_card=1152`.

- [ ] **Step 3: Implement sparse categorical keys**

Add visible categorical keys for selection effect, context card, option type,
source card, target card, and attack. Add interactions:

```python
features[f"cat:context={context}|source={source_id}"] = 1.0
features[
    f"cat:context={context}|effect={effect_id}|source={source_id}"
] = 1.0
```

Retain the existing numeric fields for compatibility.

- [ ] **Step 4: Run feature and full tests**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_features.py
PYTHONPATH=. .venv/bin/pytest -q tests/rolling
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pokemon-tcg-ai-battle/rolling_policy/features.py \
  pokemon-tcg-ai-battle/tests/rolling/test_features.py
git commit -m "feat: add effect-aware action features"
```

### Task 4: Structured Semantic Selection

**Files:**
- Modify: `pokemon-tcg-ai-battle/rolling_policy/imitation.py`
- Modify: `pokemon-tcg-ai-battle/tests/rolling/test_imitation.py`

**Interfaces:**
- Produces:
  - `semantic_set_prediction(decision, scores, *, threshold, minimum, maximum) -> frozenset[tuple[int, ...]]`
  - `semantic_set_accuracy(decisions, scores, bounds, *, threshold: float) -> float`
  - `calibrate_semantic_threshold(decisions, scores, bounds) -> float`

`bounds` is a sequence of `(minimum, maximum)` integer tuples aligned one-to-one
with `decisions`.

- [ ] **Step 1: Write failing selector tests**

Cover fixed one-choice, variable zero-to-two, duplicate semantic options, and
minimum clipping:

```python
prediction = semantic_set_prediction(
    decision,
    [0.9, 0.8, 0.2],
    threshold=0.5,
    minimum=1,
    maximum=2,
)
assert prediction == frozenset({(1,), (2,)})
```

- [ ] **Step 2: Run tests and verify missing imports fail**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_imitation.py
```

Expected: collection failure because the new functions do not exist.

- [ ] **Step 3: Implement deterministic semantic selection**

Aggregate duplicate options by maximum score, sort by descending score and
semantic tuple, apply the threshold, and clip the semantic set to the legal
minimum and maximum. Calibrate over thresholds formed from validation scores
plus `0.0`, `0.5`, and `1.0`; break accuracy ties toward the larger threshold.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_imitation.py
PYTHONPATH=. .venv/bin/pytest -q tests/rolling
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pokemon-tcg-ai-battle/rolling_policy/imitation.py \
  pokemon-tcg-ai-battle/tests/rolling/test_imitation.py
git commit -m "feat: select structured semantic action sets"
```

### Task 5: Train Direct-Teacher Context Models

**Files:**
- Create: `pokemon-tcg-ai-battle/rolling_policy/direct_teacher.py`
- Create: `pokemon-tcg-ai-battle/scripts/train_direct_teacher_models.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_direct_teacher.py`

**Interfaces:**
- Produces:
  - `DIRECT_CONTEXTS = {"main": 0, "bench": 5, "search": 7, "discard": 8}`
  - `eligible_direct_decision(row, mode) -> bool`
  - `build_direct_examples(...)`
  - one pair of exported JSON rankers per eligible context;
  - one validation-frozen threshold per context;
  - `direct_teacher_training_report.json`.

- [ ] **Step 1: Write failing eligibility and leakage tests**

Assert that each mode accepts only its context, excludes forced decisions, and
that holdout rows are counted and skipped before reading
`selected_signature`.

- [ ] **Step 2: Run tests and verify module-not-found failure**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_direct_teacher.py
```

Expected: failure because `rolling_policy.direct_teacher` does not exist.

- [ ] **Step 3: Implement direct example loading**

Reuse `visible_state_features`, `visible_action_features`,
`balanced_option_weights`, and episode-balanced weights. Keep source episodes
whole when subsampling each seed. Reject any mode with fewer than 30 validation
decisions.

- [ ] **Step 4: Implement dual model training and reporting**

For each eligible mode:

1. fit two seeded exported classifiers;
2. average their option scores;
3. calibrate a semantic-set threshold on validation only;
4. report exact-set accuracy, first-option baseline, decision counts, export
   parity, feature count, and model hashes.

The training command is:

```bash
.venv/bin/python scripts/train_direct_teacher_models.py \
  --snapshot artifacts/direct_grimmsnarl/20260729T070000Z/snapshot.json
```

- [ ] **Step 5: Run tests and training**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_direct_teacher.py
PYTHONPATH=. .venv/bin/pytest -q tests/rolling
.venv/bin/python scripts/train_direct_teacher_models.py \
  --snapshot artifacts/direct_grimmsnarl/20260729T070000Z/snapshot.json
```

Expected: tests pass and training produces immutable model/report files.

- [ ] **Step 6: Commit**

```bash
git add pokemon-tcg-ai-battle/rolling_policy/direct_teacher.py \
  pokemon-tcg-ai-battle/scripts/train_direct_teacher_models.py \
  pokemon-tcg-ai-battle/tests/rolling/test_direct_teacher.py
git commit -m "feat: train direct-teacher context policies"
```

### Task 6: One-Time Audit Gate

**Files:**
- Create: `pokemon-tcg-ai-battle/scripts/audit_direct_teacher_models.py`
- Modify: `pokemon-tcg-ai-battle/tests/rolling/test_direct_teacher.py`

**Interfaces:**
- Consumes: frozen direct-teacher models and untouched audit rows.
- Produces: `direct_teacher_audit_report.json` exactly once.

- [ ] **Step 1: Write failing audit-seal test**

Assert that a second audit invocation refuses to overwrite the report and that
the training report records skipped audit rows before label access.

- [ ] **Step 2: Run test and observe missing audit function**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_direct_teacher.py
```

- [ ] **Step 3: Implement immutable audit**

Load only split `holdout`, score without fitting, and apply the spec gates:
main 65%, search 55%, bench 55%, discard 40% when at least 30 decisions,
overall 72%, and eight points over the frozen old candidate measurements.

- [ ] **Step 4: Run the audit exactly once**

Run:

```bash
.venv/bin/python scripts/audit_direct_teacher_models.py \
  --snapshot artifacts/direct_grimmsnarl/20260729T070000Z/snapshot.json
```

Expected: report is written once with `PASS` or `REJECT`; do not change gates
after seeing the result.

- [ ] **Step 5: Commit**

```bash
git add pokemon-tcg-ai-battle/scripts/audit_direct_teacher_models.py \
  pokemon-tcg-ai-battle/tests/rolling/test_direct_teacher.py
git commit -m "feat: audit direct-teacher policies once"
```

### Task 7: Standalone Candidate and Safety Gate

**Files:**
- Create: `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_direct_teacher_v3/`
- Modify: `pokemon-tcg-ai-battle/tests/rolling/test_candidate_runtime.py`

**Interfaces:**
- Consumes: passed context models and thresholds.
- Produces: standalone `agent(observation) -> list[int]` and submission archive.

- [ ] **Step 1: Write failing standalone runtime tests**

Parameterize the existing candidate tests for v3. Assert frozen deck, model
hashes, vendored feature parity, legal single/multi selections, no import of
the v2 agent, and presence of all nested model/runtime files in the archive.

- [ ] **Step 2: Run candidate tests and observe missing directory failure**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling/test_candidate_runtime.py
```

- [ ] **Step 3: Build the minimal standalone runtime**

Vendor the effect-aware feature extractor and pure tree predictor. Average the
two context models, convert scores with the frozen semantic selector, and map
semantic choices to deterministic legal indices. Route all unsupported or
failed decisions through the existing legal fallback.

- [ ] **Step 4: Run runtime and package verification**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/rolling
.venv/bin/python scripts/smoke_test_agent.py \
  --agent-dir agents/candidate_grimmsnarl_direct_teacher_v3
.venv/bin/python scripts/package_submission.py \
  --agent-dir agents/candidate_grimmsnarl_direct_teacher_v3 \
  --output submissions/grimmsnarl_direct_teacher_v3.tar.gz
```

- [ ] **Step 5: Run 1,000 official-engine games and latency audit**

Run balanced-seat matrices against the rejected candidate, v11, v168, and the
legal fallback. Require zero crashes, illegal actions, and timeouts, and p99
below 500 ms.

- [ ] **Step 6: Commit**

```bash
git add pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_direct_teacher_v3 \
  pokemon-tcg-ai-battle/tests/rolling/test_candidate_runtime.py
git commit -m "feat: build direct-teacher Grimmsnarl candidate"
```

### Task 8: Prospective Gate and Second-Slot Decision

**Files:**
- Generate: `pokemon-tcg-ai-battle/artifacts/direct_grimmsnarl/20260729T070000Z/prospective_report.json`

**Interfaces:**
- Consumes: episodes created at least six hours after the snapshot cutoff.
- Produces: final immutable `SUBMIT` or `REJECT` verdict.

- [ ] **Step 1: Wait for the fixed non-overlapping cutoff**

Read `snapshot.json.cutoff_utc`, add six hours, and do not fetch labels for
model tuning before that time.

- [ ] **Step 2: Collect at least 40 new teacher episodes**

Use submission `55002825`, require `createTime > snapshot.cutoff_utc`, and
score v3 against the recorded teacher actions without fitting.

- [ ] **Step 3: Apply the prospective gates**

Allow five percentage points below the audit thresholds for sampling noise.
Require zero runtime errors. Write the counts, context accuracies, source IDs,
and hashes into `prospective_report.json`.

- [ ] **Step 4: Protect the second slot**

Only `SUBMIT` authorizes:

```bash
kaggle competitions submit \
  -c pokemon-tcg-ai-battle \
  -f submissions/grimmsnarl_direct_teacher_v3.tar.gz \
  -m "direct-teacher Grimmsnarl v3: context-aware rank-3 imitation"
```

On `REJECT`, retain the archive locally and do not upload it.
