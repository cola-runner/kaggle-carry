# Pokémon TCG Rolling Multi-Teacher Policy Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Within 48 hours, either produce one fresh, leakage-safe Grimmsnarl/Froslass challenger that passes every preregistered feasibility, safety, latency, freshness, and packaging gate, or stop this route with a machine-readable rejection report and no Kaggle submission.

**Architecture:** Freeze one current Kaggle snapshot and a 72-hour replay window from at least three exact-deck top-ten teacher teams. Keep runtime-visible observations and offline hidden engine-restoration material in physically separate trees. Train two independently seeded visible-state value models on an older chronological window, open the newest 12-hour holdout exactly once, and stop unless both models meet the fixed AUC/calibration gates. Only after that, restore real replay roots in the official engine, reject roots that fail replay fidelity or forward/reverse branch invariance, and distill each value model's counterfactual leaf values into its own compact option ranker. The packaged agent may override a deterministic deck-specific fallback only when both rankers choose the same legal option and both estimate at least a `0.10` advantage.

**Tech Stack:** Python 3.14 development environment with Python 3.12 compatibility check, standard library, NumPy, SciPy, scikit-learn for offline training only, pytest, Kaggle CLI, official vendored `cg` engine, JSON/JSONL/CSV, pure-Python exported tree inference in the submission.

## Global Constraints

- Work in the current `/path/to/kaggle-carry/pokemon-tcg-ai-battle` project tree. It is currently untracked by the parent Git repository; stage only files named by the current task and never stage the whole directory.
- Preserve the user's modified root `.gitignore`, `README.md`, `cs2-short/`, all existing agents, all old submissions, and every unrelated artifact.
- Never open, copy, extract, hash, package, stage, or commit `/path/to/kaggle-carry/sensitive-transfer-archive.tar.gz`.
- Raw leaderboard downloads, replay files, visible datasets, hidden restoration data, trained models, reports, and archives live under `pokemon-tcg-ai-battle/artifacts/rolling_grimmsnarl/` and remain outside Git.
- The exact deck is the 60-card list with IDs and counts:

```text
7x10, 112x4, 646x4, 647x3, 648x3, 860x2, 104x2,
1086x4, 1152x4, 1219x4, 1227x4, 1259x4, 1079x3,
1097x3, 1182x2, 1080x1, 1122x1, 1137x1, 1231x1
```

- Teacher eligibility is recomputed at snapshot time: exact deck, currently tracked submission, current score at or above the live rank-ten threshold, a completed public episode in the 72-hour window, and at least three distinct teams.
- The frozen chronological split is `[cutoff-72h, cutoff-24h)` train, `[cutoff-24h, cutoff-12h)` validation, and `[cutoff-12h, cutoff]` unopened holdout. Whole episodes stay in one split.
- Training sampling is balanced by `team_id × target_seat × 12-hour bucket`. Multiple submissions belonging to one team share that team's budget.
- No opponent hidden hand, deck order, prize identities, `visualize`, or future replay field may enter visible rows, features, model files, or runtime code. Hidden data may only restore historical roots offline.
- The holdout may be evaluated once per immutable snapshot. A second run must fail unless it writes a new snapshot ID with a new cutoff.
- Fixed feasibility thresholds must not be weakened after seeing results: at least 3 teams, 500 episodes, 5,000 non-forced `MAIN` one-choice decisions, 50 exact replay roots, 50 branch-order-invariant roots, each value model AUC `>=0.65`, each value model 10-bin equal-frequency ECE `<=0.08`, and both outcome classes in the holdout.
- A counterfactual root is usable only when the recorded action reproduces the replay's next state and forward/reverse legal-option enumeration produces identical per-signature leaf hashes and scores.
- Runtime override requires both distilled rankers to choose the same action and `min(Q1(best)-Q1(fallback), Q2(best)-Q2(fallback)) >= 0.10`.
- Local match outcomes are safety diagnostics only. They cannot promote a candidate or predict Kaggle rating.
- The candidate must finish 1,000 official-engine safety games with zero agent exceptions, invalid actions, engine timeouts attributable to the candidate, or unfinished games, and p99 single-decision latency below 500 ms on one Linux CPU core.
- Packaging is allowed only within 24 hours of the final snapshot. Otherwise rerun snapshot, extraction, training, holdout, branch labeling, and all gates.
- At most one policy candidate is uploaded. A packaging-only replacement after Kaggle validation failure must have identical deck, policy-source, and model hashes.
- Target the one upload no later than 2026-08-04. The competition deadline remains 2026-08-16T23:59:00.
- Immediately before upload, capture the team's current latest-two submission IDs. Kaggle, not this code, determines which older submission leaves the tracked pair by submission order.
- Kaggle chooses all online opponents. No local script may claim to select top-ten opponents or use old scores, old replay outcomes, or old matchup results as promotion evidence.
- The Handheld Fan near-deck variant stays outside training. It may appear only as a safety-path reference.
- The packaged runtime makes no network call and performs no online tree search; all official-engine branching is offline.
- Use test-driven development for every task: add the failing test, run it and observe the named failure, add the minimum implementation, rerun the narrow test, then run the complete rolling-policy suite.
- Commit after every task with the exact message shown. Do not stage unrelated files.

## File Map

### New reusable package

- `pokemon-tcg-ai-battle/rolling_policy/__init__.py` — package version and public imports.
- `pokemon-tcg-ai-battle/rolling_policy/constants.py` — competition slug, exact deck, card IDs, time windows, thresholds, and forbidden feature keys.
- `pokemon-tcg-ai-battle/rolling_policy/hashing.py` — canonical JSON and SHA-256 helpers.
- `pokemon-tcg-ai-battle/rolling_policy/schema.py` — immutable snapshot, teacher, episode, replay, split, model, branch, gate, and submission records.
- `pokemon-tcg-ai-battle/rolling_policy/snapshot.py` — official CSV/JSON parsing, tracked-teacher validation, eligibility, chronological splitting, and balanced sampling weights.
- `pokemon-tcg-ai-battle/rolling_policy/replays.py` — Kaggle episode/replay acquisition, retry, deck extraction, fingerprints, and file hashes.
- `pokemon-tcg-ai-battle/rolling_policy/extract.py` — visible observation sanitizer, exact option signatures, selected actions, outcomes, and separate hidden restoration frames.
- `pokemon-tcg-ai-battle/rolling_policy/features.py` — state/action feature builders and automated leakage audit.
- `pokemon-tcg-ai-battle/rolling_policy/metrics.py` — ROC AUC, ten-bin equal-frequency ECE, class checks, and p99.
- `pokemon-tcg-ai-battle/rolling_policy/tree_model.py` — offline boosted-tree fitting/export and pure-Python exported inference.
- `pokemon-tcg-ai-battle/rolling_policy/branching.py` — official-engine root restore, recorded-next-state fidelity, explicit coin schedule, forced continuation, and order-invariance comparison.
- `pokemon-tcg-ai-battle/rolling_policy/runtime.py` — deterministic fallback, dual-ranker agreement gate, legal action normalization, and timing.
- `pokemon-tcg-ai-battle/rolling_policy/gates.py` — fail-closed feasibility, candidate, freshness, identity, and one-submission decisions.

### New command-line entry points

- `pokemon-tcg-ai-battle/scripts/freeze_rolling_snapshot.py`
- `pokemon-tcg-ai-battle/scripts/download_rolling_replays.py`
- `pokemon-tcg-ai-battle/scripts/extract_rolling_dataset.py`
- `pokemon-tcg-ai-battle/scripts/train_rolling_value_models.py`
- `pokemon-tcg-ai-battle/scripts/audit_rolling_holdout.py`
- `pokemon-tcg-ai-battle/scripts/probe_rolling_branch_integrity.py`
- `pokemon-tcg-ai-battle/scripts/label_rolling_counterfactuals.py`
- `pokemon-tcg-ai-battle/scripts/train_rolling_option_rankers.py`
- `pokemon-tcg-ai-battle/scripts/materialize_rolling_candidate.py`
- `pokemon-tcg-ai-battle/scripts/run_rolling_feasibility_gate.py`
- `pokemon-tcg-ai-battle/scripts/run_rolling_candidate_gate.py`
- `pokemon-tcg-ai-battle/scripts/submit_rolling_candidate_once.py`
- `pokemon-tcg-ai-battle/scripts/report_rolling_online_result.py`
- `pokemon-tcg-ai-battle/configs/rolling_grimmsnarl_teacher_candidates_20260728.json`

### New candidate

- `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/main.py`
- `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/deck.csv`
- `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/policy_runtime.py`
- `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/features.py`
- `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/ranker_v1.json`
- `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/ranker_v2.json`
- `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/identity.json`

### Tests and fixtures

- `pokemon-tcg-ai-battle/tests/rolling/fixtures/{leaderboard.csv,teacher_candidates.json,episodes.csv,replay_minimal.json,visible_observation.json,hidden_frame.json}`
- `pokemon-tcg-ai-battle/tests/rolling/test_{constants,snapshot,replays,extract,features,metrics,tree_model,branching,runtime,gates,cli}.py`
- `pokemon-tcg-ai-battle/requirements-train.txt`

### Existing infrastructure to reuse without redesign

- `pokemon-tcg-ai-battle/scripts/analyze_submission_replays.py`
- `pokemon-tcg-ai-battle/scripts/probe_official_replay_counterfactuals.py`
- `pokemon-tcg-ai-battle/scripts/run_local_match.py`
- `pokemon-tcg-ai-battle/scripts/package_submission.py`
- `pokemon-tcg-ai-battle/scripts/pre_submit_audit.py`
- `pokemon-tcg-ai-battle/agents/baselines/v11_hammer_metal_from_submission/cg/`

---

## Task 1: Bootstrap the Isolated Rolling-Policy Package and Exact Deck Contract

**Files:**

- Create: `pokemon-tcg-ai-battle/requirements-train.txt`
- Create: `pokemon-tcg-ai-battle/rolling_policy/{__init__,constants,hashing,schema}.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_constants.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_schema.py`

**Interfaces:**

```python
EXACT_DECK_COUNTS: dict[int, int]
EXACT_DECK: tuple[int, ...]
EXACT_DECK_FINGERPRINT: str

def canonical_json_bytes(value: object) -> bytes: ...
def sha256_bytes(value: bytes) -> str: ...
def sha256_file(path: Path) -> str: ...

class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"

@dataclass(frozen=True, slots=True)
class TeacherSubmission:
    team_id: str
    team_name: str
    submission_id: str
    score: float
    deck_fingerprint: str
    tracked_at_cutoff: bool

@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    submission_id: str
    team_id: str
    create_time_utc: datetime
    end_time_utc: datetime
    target_seat: int
    split: Split
    replay_sha256: str
```

- [ ] Add tests proving `EXACT_DECK` has exactly 60 integers, the required counts, no accidental Mega Froslass (`861`), and a stable order-insensitive fingerprint. Test canonical JSON key ordering, compact separators, UTF-8, and lowercase 64-character SHA-256.

```python
from collections import Counter

from rolling_policy.constants import EXACT_DECK, EXACT_DECK_COUNTS


def test_exact_grimmsnarl_deck_is_frozen() -> None:
    assert len(EXACT_DECK) == 60
    assert Counter(EXACT_DECK) == Counter(EXACT_DECK_COUNTS)
    assert EXACT_DECK_COUNTS[860] == 2
    assert EXACT_DECK_COUNTS[104] == 2
    assert 861 not in EXACT_DECK
```

- [ ] Run the narrow tests and observe `ModuleNotFoundError: rolling_policy`.

```bash
cd /path/to/kaggle-carry/pokemon-tcg-ai-battle
.venv/bin/python -m pytest tests/rolling/test_constants.py tests/rolling/test_schema.py -q
```

- [ ] Create `requirements-train.txt` with explicit compatible lower bounds:

```text
numpy>=2.0
scipy>=1.13
scikit-learn>=1.7
pytest>=8.4
```

- [ ] Install the training-only dependencies, then implement the constants, canonical hashing, strict UTC-aware dataclasses, enum parsing, and JSON serialization. Keep training dependencies out of every module imported by the packaged candidate.

```bash
.venv/bin/python -m pip install -r requirements-train.txt
```

- [ ] Rerun the narrow tests, then the rolling suite.

```bash
.venv/bin/python -m pytest tests/rolling/test_constants.py tests/rolling/test_schema.py -q
.venv/bin/python -m pytest tests/rolling -q
```

- [ ] Commit only the task files.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/requirements-train.txt \
  pokemon-tcg-ai-battle/rolling_policy/__init__.py \
  pokemon-tcg-ai-battle/rolling_policy/constants.py \
  pokemon-tcg-ai-battle/rolling_policy/hashing.py \
  pokemon-tcg-ai-battle/rolling_policy/schema.py \
  pokemon-tcg-ai-battle/tests/rolling/test_constants.py \
  pokemon-tcg-ai-battle/tests/rolling/test_schema.py
git -C /path/to/kaggle-carry commit -m "feat: freeze rolling Grimmsnarl policy contracts"
```

## Task 2: Freeze One Immutable Live Snapshot and Enforce Teacher Eligibility

**Files:**

- Create: `pokemon-tcg-ai-battle/rolling_policy/snapshot.py`
- Create: `pokemon-tcg-ai-battle/scripts/freeze_rolling_snapshot.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/fixtures/{leaderboard.csv,teacher_candidates.json,episodes.csv}`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_snapshot.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_cli.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    snapshot_id: str
    implementation_started_utc: datetime
    cutoff_utc: datetime
    rank_ten_score: float
    leaderboard_sha256: str
    teacher_candidates_sha256: str
    teachers: tuple[TeacherSubmission, ...]
    source_window_start_utc: datetime
    validation_start_utc: datetime
    holdout_start_utc: datetime

def parse_leaderboard(path: Path) -> list[dict[str, str]]: ...
def rank_ten_threshold(rows: Sequence[Mapping[str, str]]) -> float: ...
def assign_split(create_time_utc: datetime, cutoff_utc: datetime) -> Split | None: ...
def eligible_teachers(
    candidates: Sequence[TeacherSubmission],
    rank_ten_score: float,
    active_submission_ids: Collection[str],
) -> tuple[TeacherSubmission, ...]: ...
def balanced_episode_weights(episodes: Sequence[EpisodeRecord]) -> dict[str, float]: ...
```

- [ ] Add failing tests for rank-ten extraction; ISO timestamp normalization; boundary behavior at 72h, 24h, 12h, and cutoff; rejection of future episodes; exact-deck filtering; `score >= rank10`; tracked/current-episode filtering; team deduplication; the three-team minimum; and equal total weight per `team × seat × 12h bucket`.

```python
from datetime import datetime, timedelta, timezone

from rolling_policy.schema import Split
from rolling_policy.snapshot import assign_split


def test_chronological_split_boundaries_are_frozen() -> None:
    cutoff = datetime(2026, 7, 28, 16, tzinfo=timezone.utc)
    assert assign_split(cutoff - timedelta(hours=72), cutoff) is Split.TRAIN
    assert assign_split(cutoff - timedelta(hours=24), cutoff) is Split.VALIDATION
    assert assign_split(cutoff - timedelta(hours=12), cutoff) is Split.HOLDOUT
    assert assign_split(cutoff, cutoff) is Split.HOLDOUT
    assert assign_split(cutoff + timedelta(microseconds=1), cutoff) is None
```

- [ ] Run the tests and observe missing snapshot APIs.

```bash
.venv/bin/python -m pytest tests/rolling/test_snapshot.py tests/rolling/test_cli.py -q
```

- [ ] Implement `freeze_rolling_snapshot.py` as an allocation-only collector. It must:

  1. create a UTC cutoff once;
  2. run `kaggle competitions leaderboard pokemon-tcg-ai-battle -d -p "${snapshot_dir}/raw" -q`;
  3. read a reviewed candidate seed containing the five known 2026-07-28 exact-deck submissions `55001357`, `54989332`, `55011514`, `55035974`, and `54968369`, with their team identities;
  4. query `kaggle competitions episodes "${submission_id}" --csv`;
  5. treat a candidate as currently tracked only when the official endpoint returns a completed public episode inside the source window;
  6. retain only candidates whose reviewed current score is at least the newly downloaded rank-ten threshold;
  7. fail if fewer than three distinct teams remain;
  8. write raw inputs first and `snapshot.json` last using exclusive creation;
  9. refuse to overwrite an existing snapshot ID.

- [ ] Require the candidate seed to bind each score and tracked-submission claim to a captured official response hash. A human-readable name alone is never an identity key; `team_id` and `submission_id` are mandatory.

- [ ] Rerun tests and a CLI fixture test that patches the subprocess runner. Assert the manifest contains no absolute credential path and every input has a hash.

- [ ] Run the live collector once into a new timestamped directory.

```bash
.venv/bin/python scripts/freeze_rolling_snapshot.py \
  --competition pokemon-tcg-ai-battle \
  --teacher-candidates configs/rolling_grimmsnarl_teacher_candidates_20260728.json \
  --out-root artifacts/rolling_grimmsnarl
```

Expected: one immutable `snapshot.json`, at least three team IDs, and no model training yet. If the current threshold has risen above all but two teams, stop the entire route.

- [ ] Bind all later commands to the newly created snapshot without hand-editing a path.

```bash
export SNAPSHOT_PATH="$(
  find artifacts/rolling_grimmsnarl -mindepth 2 -maxdepth 2 \
    -name snapshot.json -type f -print | sort | tail -1
)"
export SNAPSHOT_ID="$(basename "$(dirname "$SNAPSHOT_PATH")")"
test -f "$SNAPSHOT_PATH"
```

- [ ] Commit the implementation, tests, fixtures, and reviewed candidate config. Do not commit the generated live snapshot.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/configs/rolling_grimmsnarl_teacher_candidates_20260728.json \
  pokemon-tcg-ai-battle/rolling_policy/snapshot.py \
  pokemon-tcg-ai-battle/scripts/freeze_rolling_snapshot.py \
  pokemon-tcg-ai-battle/tests/rolling/fixtures/leaderboard.csv \
  pokemon-tcg-ai-battle/tests/rolling/fixtures/teacher_candidates.json \
  pokemon-tcg-ai-battle/tests/rolling/fixtures/episodes.csv \
  pokemon-tcg-ai-battle/tests/rolling/test_snapshot.py \
  pokemon-tcg-ai-battle/tests/rolling/test_cli.py
git -C /path/to/kaggle-carry commit -m "feat: freeze eligible rolling teacher snapshots"
```

## Task 3: Download, Verify, Hash, and Inventory Current Replays

**Files:**

- Create: `pokemon-tcg-ai-battle/rolling_policy/replays.py`
- Create: `pokemon-tcg-ai-battle/scripts/download_rolling_replays.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/fixtures/replay_minimal.json`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_replays.py`

**Interfaces:**

```python
def decks_from_replay(episode: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
def replay_target_seat(episode: Mapping[str, object], target_fingerprint: str) -> int: ...
def replay_episode_id(episode: Mapping[str, object]) -> str: ...
def verify_replay(
    path: Path,
    expected_episode_id: str,
    exact_deck_fingerprint: str,
) -> ReplayRecord: ...
def download_replay(
    episode_id: str,
    destination: Path,
    retries: int = 4,
) -> Path: ...
```

- [ ] Add failing tests proving deck order does not change fingerprint, exactly one target seat is required, a mirror with the target deck in both seats is marked separately rather than silently assigned, mismatched episode IDs fail, malformed/partial JSON fails without entering the manifest, and replay hashes are stable.

- [ ] Add a retry test where attempts 1 and 2 fail, attempt 3 succeeds, and only the final atomically renamed JSON appears in `replays/`.

- [ ] Run the narrow tests and observe missing replay APIs.

```bash
.venv/bin/python -m pytest tests/rolling/test_replays.py -q
```

- [ ] Implement acquisition by adapting the proven episode/replay subprocess logic from `scripts/analyze_submission_replays.py`. Download to `${episode_id}.partial`, parse and verify, then rename to `${episode_id}.json`. Never retain unverified replay files under the final directory.

- [ ] Write `replay_inventory.jsonl` sorted by `(create_time_utc, episode_id)`. Bind every row to snapshot ID, teacher team, submission, target seat, split, deck fingerprint, and replay SHA-256.

- [ ] Rerun tests, then download every eligible episode from the immutable snapshot.

```bash
.venv/bin/python scripts/download_rolling_replays.py \
  --snapshot "$SNAPSHOT_PATH" \
  --jobs 8 \
  --retries 4
```

Expected: at least 500 verified episodes. Otherwise write `REJECT_INSUFFICIENT_EPISODES` and stop.

- [ ] Commit only source, fixture, and tests.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/rolling_policy/replays.py \
  pokemon-tcg-ai-battle/scripts/download_rolling_replays.py \
  pokemon-tcg-ai-battle/tests/rolling/fixtures/replay_minimal.json \
  pokemon-tcg-ai-battle/tests/rolling/test_replays.py
git -C /path/to/kaggle-carry commit -m "feat: inventory exact-deck rolling replays"
```

## Task 4: Extract Visible Decisions and Hidden Restoration Frames into Separate Trees

**Files:**

- Create: `pokemon-tcg-ai-battle/rolling_policy/extract.py`
- Create: `pokemon-tcg-ai-battle/scripts/extract_rolling_dataset.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/fixtures/{visible_observation.json,hidden_frame.json}`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_extract.py`

**Interfaces:**

```python
def sanitize_visible_observation(
    observation: Mapping[str, object],
    acting_seat: int,
) -> dict[str, object]: ...
def option_signature(
    observation: Mapping[str, object],
    option: Mapping[str, object],
    acting_seat: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int]: ...
def selected_signature(
    observation: Mapping[str, object],
    action: Sequence[int],
    acting_seat: int,
) -> tuple[tuple[int, int, int, int, int, int, int, int, int, int, int], ...]: ...
def extract_episode(
    replay_path: Path,
    inventory: ReplayRecord,
) -> tuple[list[VisibleDecisionRow], list[HiddenRestorationRow]]: ...
```

The option signature fields are:

```text
type, source_card_id, source_serial, attack_id, target_card_id,
target_serial, area, index, in_play_area, in_play_index, number
```

If the engine exposes an additional target/effect discriminator, append it to the signature version and fail old readers rather than silently colliding.

- [ ] Add failing tests showing two same-ID targets with different serials remain distinct; official selected actions are matched semantically; a pending observation pairs only with the next action from the same seat; and non-`MAIN`, forced, multi-choice, and single-choice flags are correctly recorded.

- [ ] Add a recursive fixture test proving the visible output contains none of:

```python
{
    "visualize",
    "opponent_hand",
    "opponent_deck",
    "opponent_prize_cards",
    "deck_order",
    "hidden",
    "search_begin_input",
}
```

- [ ] Add a filesystem test proving visible rows are written only below `public/`, hidden restoration rows only below `offline_hidden/`, and neither writer accepts the other's destination.

- [ ] Run the tests and observe missing extractor behavior.

```bash
.venv/bin/python -m pytest tests/rolling/test_extract.py -q
```

- [ ] Implement the extractor by reusing observation/action pairing and state parsing from `scripts/extract_official_option_dataset.py`, but remove its agent dependency and replace the old lossy option signature with the serial-aware signature above.

- [ ] Write:

```text
public/episodes.jsonl
public/decisions.jsonl
public/options.jsonl
offline_hidden/restoration.jsonl
extraction_report.json
```

Every row must retain `snapshot_id`, `team_id`, `submission_id`, `episode_id`, `create_time_utc`, `target_seat`, `split`, and source replay hash. Hidden rows additionally bind the matching visible decision hash.

- [ ] Rerun tests and live extraction.

```bash
.venv/bin/python scripts/extract_rolling_dataset.py \
  --snapshot "$SNAPSHOT_PATH"
```

Expected: at least 5,000 non-forced `MAIN`, `minCount=1`, `maxCount=1` decisions. Otherwise write `REJECT_INSUFFICIENT_DECISIONS` and stop.

- [ ] Commit source and tests, not extracted data.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/rolling_policy/extract.py \
  pokemon-tcg-ai-battle/scripts/extract_rolling_dataset.py \
  pokemon-tcg-ai-battle/tests/rolling/fixtures/visible_observation.json \
  pokemon-tcg-ai-battle/tests/rolling/fixtures/hidden_frame.json \
  pokemon-tcg-ai-battle/tests/rolling/test_extract.py
git -C /path/to/kaggle-carry commit -m "feat: separate visible and restoration replay data"
```

## Task 5: Build Leakage-Safe Features, Metrics, and Two Visible-State Value Models

**Files:**

- Create: `pokemon-tcg-ai-battle/rolling_policy/features.py`
- Create: `pokemon-tcg-ai-battle/rolling_policy/metrics.py`
- Create: `pokemon-tcg-ai-battle/rolling_policy/tree_model.py`
- Create: `pokemon-tcg-ai-battle/scripts/train_rolling_value_models.py`
- Create: `pokemon-tcg-ai-battle/scripts/audit_rolling_holdout.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_features.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_metrics.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_tree_model.py`

**Interfaces:**

```python
def visible_state_features(row: Mapping[str, object]) -> dict[str, float]: ...
def visible_action_features(
    state: Mapping[str, object],
    option: Mapping[str, object],
) -> dict[str, float]: ...
def audit_feature_names(names: Collection[str]) -> tuple[str, ...]: ...
def roc_auc(y_true: Sequence[int], probabilities: Sequence[float]) -> float: ...
def equal_frequency_ece(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    bins: int = 10,
) -> float: ...

@dataclass(frozen=True, slots=True)
class ExportedTreeModel:
    model_version: int
    feature_names: tuple[str, ...]
    baseline: float
    trees: tuple[tuple[tuple[float, ...], ...], ...]

def fit_exported_classifier(
    rows: Sequence[Mapping[str, object]],
    labels: Sequence[int],
    weights: Sequence[float],
    random_seed: int,
) -> tuple[ExportedTreeModel, dict[str, object]]: ...
def predict_exported(
    model: ExportedTreeModel,
    feature_rows: Sequence[Mapping[str, float]],
) -> list[float]: ...
```

- [ ] Add failing feature tests proving the same visible observation yields identical features despite different opponent hidden hands, deck orders, prizes, and `visualize`; target serial changes action features; input key order does not change features; and every feature has a finite numeric value.

- [ ] Add an adversarial audit test for case, punctuation, and nesting variants such as `OpponentHand`, `opponent-hand`, `visualize.card`, `deckOrder`, `prize_cards`, and `search_begin_input`.

- [ ] Add metric tests with known AUC/ECE values, deterministic equal-frequency tie handling, exactly ten nonempty bins when sample size permits, and hard failure when `y_true` lacks wins or losses.

```python
import pytest

from rolling_policy.metrics import equal_frequency_ece, roc_auc


def test_holdout_without_both_classes_fails() -> None:
    with pytest.raises(ValueError, match="both outcome classes"):
        roc_auc([1, 1, 1], [0.6, 0.7, 0.8])
    with pytest.raises(ValueError, match="both outcome classes"):
        equal_frequency_ece([0, 0, 0], [0.1, 0.2, 0.3], bins=10)
```

- [ ] Add exporter parity tests comparing scikit-learn decision scores and pure-Python exported scores within `1e-6` on at least 200 rows.

- [ ] Run the narrow tests and observe missing implementations.

```bash
.venv/bin/python -m pytest \
  tests/rolling/test_features.py \
  tests/rolling/test_metrics.py \
  tests/rolling/test_tree_model.py -q
```

- [ ] Implement visible state features from acting hand, both public boards/discards, prize counts, energy/tools/evolutions, public reveals/logs, stadium, turn, seat, damage, and action flags. Never pass raw observation dictionaries into model fitting.

- [ ] Fit `V1` with seed `1701` and `V2` with seed `2909`. Use only train rows for fitting and validation rows for fixed hyperparameter selection. Use the frozen balanced episode weights; do not duplicate individual decisions to simulate balance.

- [ ] Export models and a sealed `holdout_plan.json` before reading holdout labels. The plan binds snapshot, feature schema, model hashes, AUC/ECE thresholds, and a `holdout_opened=false` marker.

- [ ] Implement `audit_rolling_holdout.py` to use exclusive creation for `holdout_report.json`; refuse a second evaluation for the same snapshot/model hashes. It opens holdout once, checks both classes, evaluates both models, writes bin details, and returns nonzero if either model misses either threshold.

- [ ] Rerun tests, train on the live snapshot, and open the holdout once.

```bash
.venv/bin/python scripts/train_rolling_value_models.py \
  --snapshot "$SNAPSHOT_PATH"
.venv/bin/python scripts/audit_rolling_holdout.py \
  --snapshot "$SNAPSHOT_PATH"
```

Expected: each model has AUC at least `0.65`, ECE at most `0.08`, no forbidden feature, and both classes. Any failure stops the route with no tuning against holdout.

- [ ] Commit source and tests only.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/rolling_policy/features.py \
  pokemon-tcg-ai-battle/rolling_policy/metrics.py \
  pokemon-tcg-ai-battle/rolling_policy/tree_model.py \
  pokemon-tcg-ai-battle/scripts/train_rolling_value_models.py \
  pokemon-tcg-ai-battle/scripts/audit_rolling_holdout.py \
  pokemon-tcg-ai-battle/tests/rolling/test_features.py \
  pokemon-tcg-ai-battle/tests/rolling/test_metrics.py \
  pokemon-tcg-ai-battle/tests/rolling/test_tree_model.py
git -C /path/to/kaggle-carry commit -m "feat: train dual leakage-safe state value models"
```

## Task 6: Prove Exact Replay Restoration and Forward/Reverse Branch Invariance

**Files:**

- Create: `pokemon-tcg-ai-battle/rolling_policy/branching.py`
- Create: `pokemon-tcg-ai-battle/scripts/probe_rolling_branch_integrity.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_branching.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class CoinSchedule:
    values: tuple[bool, ...]

@dataclass(frozen=True, slots=True)
class BranchLeaf:
    option_signature: tuple[int, ...]
    leaf_visible_sha256: str
    v1_probability: float
    v2_probability: float
    stopped_reason: str

def restore_root(
    replay: Mapping[str, object],
    hidden_row: Mapping[str, object],
    official_game: object,
) -> dict[str, object]: ...
def reproduce_recorded_next_state(
    root: Mapping[str, object],
    recorded_action: Sequence[int],
    expected_next_visible_sha256: str,
) -> bool: ...
def branch_root(
    root: Mapping[str, object],
    option_order: Sequence[int],
    coin_schedule: CoinSchedule,
    v1: ExportedTreeModel,
    v2: ExportedTreeModel,
) -> tuple[BranchLeaf, ...]: ...
def compare_branch_orders(
    forward: Sequence[BranchLeaf],
    reverse: Sequence[BranchLeaf],
    tolerance: float = 1e-9,
) -> tuple[str, ...]: ...
```

- [ ] Add failing tests proving every branch begins from an independent root; recorded next-state mismatch rejects the root; the same coin schedule is consumed identically; leaf comparison keys by semantic option signature rather than list index; reverse enumeration yields equal leaf hashes/scores; and an intentional shared-engine-state bug is detected.

- [ ] Run the narrow tests and observe missing branching behavior.

```bash
.venv/bin/python -m pytest tests/rolling/test_branching.py -q
```

- [ ] Implement restoration by extracting and tightening the proven `hidden_frame`, `exact_hidden_lists`, `search_begin`, state-signature, and finish lifecycle logic in `scripts/probe_official_replay_counterfactuals.py`. Do not import an old candidate policy.

- [ ] Generate each manual coin schedule from `SHA256(snapshot_id | decision_id | "coins-v1")`, store the complete schedule in the integrity report, and bind forward/reverse runs to the same schedule.

- [ ] Stop a branch at terminal result, the next non-forced `MAIN` choice, or turn boundary. Resolve only forced intervening selections with a stable semantic ordering. Reject, rather than guess, any unsupported non-forced continuation before the stop condition.

- [ ] Sample 50 roots stratified by team, seat, and time bucket before probing. Write the root allocation before outcomes and never replace failed roots with easier ones.

- [ ] Rerun tests and the live integrity probe.

```bash
.venv/bin/python scripts/probe_rolling_branch_integrity.py \
  --snapshot "$SNAPSHOT_PATH" \
  --roots 50 \
  --cg-dir agents/baselines/v11_hammer_metal_from_submission
```

Expected: `50/50` recorded transitions and `50/50` forward/reverse checks pass. Any failure stops the route as an infrastructure rejection.

- [ ] Commit source and tests.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/rolling_policy/branching.py \
  pokemon-tcg-ai-battle/scripts/probe_rolling_branch_integrity.py \
  pokemon-tcg-ai-battle/tests/rolling/test_branching.py
git -C /path/to/kaggle-carry commit -m "feat: verify replay branch integrity"
```

## Task 7: Generate Counterfactual Labels and Train Two Distilled Option Rankers

**Files:**

- Create: `pokemon-tcg-ai-battle/scripts/label_rolling_counterfactuals.py`
- Create: `pokemon-tcg-ai-battle/scripts/train_rolling_option_rankers.py`
- Modify: `pokemon-tcg-ai-battle/rolling_policy/tree_model.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_counterfactual_training.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class CounterfactualOptionRow:
    decision_id: str
    option_signature: tuple[int, ...]
    selected_by_teacher: bool
    fallback_selected: bool
    v1_leaf_probability: float
    v2_leaf_probability: float
    fidelity_passed: bool
    order_invariant: bool

def fit_exported_regressor(
    feature_rows: Sequence[Mapping[str, float]],
    targets: Sequence[float],
    weights: Sequence[float],
    random_seed: int,
) -> tuple[ExportedTreeModel, dict[str, object]]: ...
```

- [ ] Add failing tests proving failed or order-dependent roots never reach training; teacher selection is retained as diagnostic data but is not the label; `Q1` targets only V1 leaf probabilities and `Q2` targets only V2; whole decisions stay in one chronological split; and exported regressor parity is within `1e-6`.

- [ ] Add a test showing a behavior clone that perfectly reproduces `selected_by_teacher` cannot pass the ranker objective unless it also orders the counterfactual leaf values correctly.

- [ ] Run the narrow tests and observe missing training APIs.

```bash
.venv/bin/python -m pytest tests/rolling/test_counterfactual_training.py -q
```

- [ ] Label every eligible train/validation `MAIN` one-choice root using the exact same fidelity and order checks as Task 6. Do not branch holdout roots. Store forward and reverse leaf evidence and keep only passing rows in `public/counterfactual_options.jsonl`.

- [ ] Train `Q1` with seed `3701` on V1 leaf probabilities and `Q2` with seed `4909` on V2 leaf probabilities. Select fixed tree size/depth from train/validation only. Report option-order correlation, mean absolute error, teacher agreement, fallback agreement, dual-model agreement, and the fraction of decisions eligible for a `0.10` override.

- [ ] Require nonzero override coverage on validation and require each ranker to beat the deterministic fallback on counterfactual regret. These are diagnostics that may veto an obviously inert/broken ranker; they do not replace the earlier holdout gates.

- [ ] Rerun tests, then label and train.

```bash
.venv/bin/python scripts/label_rolling_counterfactuals.py \
  --snapshot "$SNAPSHOT_PATH" \
  --cg-dir agents/baselines/v11_hammer_metal_from_submission
.venv/bin/python scripts/train_rolling_option_rankers.py \
  --snapshot "$SNAPSHOT_PATH"
```

- [ ] Commit source and tests.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/rolling_policy/tree_model.py \
  pokemon-tcg-ai-battle/scripts/label_rolling_counterfactuals.py \
  pokemon-tcg-ai-battle/scripts/train_rolling_option_rankers.py \
  pokemon-tcg-ai-battle/tests/rolling/test_counterfactual_training.py
git -C /path/to/kaggle-carry commit -m "feat: distill counterfactual option rankers"
```

## Task 8: Build the Deterministic Grimmsnarl Runtime with a Dual-Model Override Gate

**Files:**

- Create: `pokemon-tcg-ai-battle/rolling_policy/runtime.py`
- Create: `pokemon-tcg-ai-battle/scripts/materialize_rolling_candidate.py`
- Create: `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/{main,policy_runtime,features}.py`
- Create: `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/deck.csv`
- Generated: `pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/{ranker_v1,ranker_v2,identity}.json`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_runtime.py`

**Interfaces:**

```python
def fallback_indices(observation: Mapping[str, object]) -> list[int]: ...
def choose_dual_ranker_action(
    observation: Mapping[str, object],
    q1: ExportedTreeModel,
    q2: ExportedTreeModel,
    minimum_advantage: float = 0.10,
) -> tuple[list[int], str]: ...
def normalize_legal_action(
    indices: Sequence[int],
    minimum: int,
    maximum: int,
    option_count: int,
) -> list[int]: ...
```

Runtime decision reasons are a closed set:

```text
forced, fallback_non_main, fallback_multi_choice, fallback_low_margin,
fallback_disagreement, fallback_error, model_override
```

- [ ] Add failing tests for deck probe, empty selection, forced single and forced multi-choice, `minCount=0`, same-card different-target legality, non-`MAIN` fallback, model agreement, model disagreement, exactly `0.10`, just below `0.10`, model exception, NaN score, deterministic repeatability, and deadline fallback.

```python
def test_override_requires_both_models_and_minimum_margin() -> None:
    action, reason = choose_dual_ranker_action(
        observation=two_option_main_observation(),
        q1=fake_ranker([0.20, 0.31]),
        q2=fake_ranker([0.40, 0.50]),
        minimum_advantage=0.10,
    )
    assert action == [1]
    assert reason == "model_override"
```

- [ ] Run the narrow tests and observe missing runtime behavior.

```bash
.venv/bin/python -m pytest tests/rolling/test_runtime.py -q
```

- [ ] Implement the fallback as a small explicit Grimmsnarl policy, not a copied old leaderboard driver:

  - setup prioritizes one Marnie's Impidimp active, then Munkidori/Snorunt/extra Impidimp on bench;
  - search targets missing evolution chain pieces, Rare Candy, Darkness Energy for Munkidori, and draw/search supporters;
  - recovery prioritizes Grimmsnarl chain, Munkidori, then Darkness Energy;
  - main action priority is legal evolve/rare-candy setup, enable unused Munkidori, attach needed energy, beneficial ability, supporter/search, attack, then end;
  - damage-counter placement prefers legal knockouts, then highest-value damaged target;
  - all unsupported contexts use stable semantic option ordering and satisfy exact min/max counts.

- [ ] Keep all inference code pure Python. The candidate must not import NumPy, SciPy, scikit-learn, Kaggle, `rolling_policy`, or any local absolute path.

- [ ] Materialize candidate models only from passing Task 7 outputs. `identity.json` binds snapshot cutoff/hash, exact deck fingerprint, feature schema hash, Q1/Q2 hashes, source hashes, override margin, and official engine hash.

- [ ] Rerun tests and run the existing deck/runtime smoke.

```bash
.venv/bin/python scripts/materialize_rolling_candidate.py \
  --snapshot "$SNAPSHOT_PATH" \
  --agent-dir agents/candidate_grimmsnarl_rolling_value
.venv/bin/python -m pytest tests/rolling/test_runtime.py -q
.venv/bin/python scripts/run_local_match.py \
  --agent0 agents/candidate_grimmsnarl_rolling_value \
  --agent1 agents/baselines/v11_hammer_metal_from_submission \
  --max-steps 3000
```

- [ ] Commit candidate source/deck and materializer. Do not commit snapshot-bound generated model or identity JSON; the archive/report hashes bind them.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/rolling_policy/runtime.py \
  pokemon-tcg-ai-battle/scripts/materialize_rolling_candidate.py \
  pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/main.py \
  pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/policy_runtime.py \
  pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/features.py \
  pokemon-tcg-ai-battle/agents/candidate_grimmsnarl_rolling_value/deck.csv \
  pokemon-tcg-ai-battle/tests/rolling/test_runtime.py
git -C /path/to/kaggle-carry commit -m "feat: build dual-gated Grimmsnarl runtime"
```

## Task 9: Produce the Fixed 48-Hour Feasibility Verdict

**Files:**

- Create: `pokemon-tcg-ai-battle/rolling_policy/gates.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_rolling_feasibility_gate.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_gates.py`

**Interfaces:**

```python
class GateDecision(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"

@dataclass(frozen=True, slots=True)
class FeasibilityEvidence:
    distinct_teacher_teams: int
    completed_episodes: int
    eligible_main_decisions: int
    fidelity_passes: int
    order_invariance_passes: int
    v1_auc: float
    v1_ece: float
    v2_auc: float
    v2_ece: float
    holdout_has_both_classes: bool
    forbidden_features: tuple[str, ...]

def decide_feasibility(evidence: FeasibilityEvidence) -> GateDecision: ...
```

- [ ] Add one parameterized failing test for every threshold at exact pass value and just below/above fail value. Prove NaN, missing report, hash mismatch, second holdout opening, unknown schema version, and forbidden feature all reject.

- [ ] Run the tests and observe missing gate behavior.

```bash
.venv/bin/python -m pytest tests/rolling/test_gates.py -q
```

- [ ] Implement a fail-closed aggregator that reads only hash-bound reports from Tasks 2–7, verifies all share one snapshot ID, checks the implementation-start timestamp is within 48 hours, and writes exactly one `feasibility_verdict.json` using exclusive creation.

- [ ] Do not add command-line threshold overrides. Thresholds come only from versioned constants tested in Task 1.

- [ ] Rerun tests and produce the live verdict.

```bash
.venv/bin/python scripts/run_rolling_feasibility_gate.py \
  --snapshot "$SNAPSHOT_PATH"
```

If the verdict is `REJECT`, stop implementation, do not package, and hand back the exact failed checks. If `PASS`, continue.

- [ ] Commit source and tests.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/rolling_policy/gates.py \
  pokemon-tcg-ai-battle/scripts/run_rolling_feasibility_gate.py \
  pokemon-tcg-ai-battle/tests/rolling/test_gates.py
git -C /path/to/kaggle-carry commit -m "feat: enforce rolling feasibility stop"
```

## Task 10: Run 1,000 Safety Games, Linux p99, Package, and Audit

**Files:**

- Create: `pokemon-tcg-ai-battle/scripts/run_rolling_candidate_gate.py`
- Modify: `pokemon-tcg-ai-battle/scripts/package_submission.py`
- Modify: `pokemon-tcg-ai-battle/scripts/pre_submit_audit.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_candidate_gate.py`
- Modify: `pokemon-tcg-ai-battle/tests/rolling/test_gates.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class SafetySummary:
    scheduled_games: int
    completed_games: int
    invalid_actions: int
    agent_exceptions: int
    candidate_timeouts: int
    unfinished_games: int
    decision_latency_p99_ms: float
    linux_cpu_cores: int

def decide_candidate_gate(
    feasibility: GateDecision,
    safety: SafetySummary,
    archive_audit_passed: bool,
    snapshot_age_hours: float,
) -> GateDecision: ...
```

- [ ] Add failing tests proving 999 games reject; any single error/timeout/unfinished game rejects; exactly 500 ms rejects because the requirement is below 500; macOS latency evidence rejects; more than one CPU core rejects; stale snapshot rejects; and a local win rate field is ignored.

- [ ] Add package tests proving `ranker_v1.json`, `ranker_v2.json`, and `identity.json` are included; raw value models, replay files, hidden restoration data, credentials, absolute paths, `.venv`, caches, and training modules are excluded.

- [ ] Run the narrow tests and observe failure because the existing packager's extra-file allowlist omits the new files.

```bash
.venv/bin/python -m pytest tests/rolling/test_candidate_gate.py tests/rolling/test_gates.py -q
```

- [ ] Extend `package_submission.py` only enough to accept an explicit safe extra-file allowlist from `identity.json`. Extend `pre_submit_audit.py` with a rolling identity audit that verifies archive file hashes, exact deck fingerprint, no forbidden strings, pure-Python imports, and snapshot freshness.

- [ ] Build a safety pool from current downloaded opponent deck fingerprints, materialized with the existing generic public/rule drivers only to exercise legality and runtime paths. Balance both seats. Run exactly 1,000 completed scheduled assignments; an infrastructure crash resumes the same assignment ID and may not silently replace it.

- [ ] Run latency inside a Linux container pinned to one CPU:

```bash
docker run --rm --cpus=1 \
  -v "$PWD:/work" -w /work \
  python:3.12-slim \
  python scripts/run_rolling_candidate_gate.py \
    --snapshot "$SNAPSHOT_PATH" \
    --candidate agents/candidate_grimmsnarl_rolling_value \
    --safety-games 1000 \
    --latency-only
```

- [ ] Run the complete safety gate on available Macs/1080 Ti host in parallel assignment shards, merge only by immutable assignment ID, and treat the GPU as optional because runtime inference is CPU-only.

```bash
.venv/bin/python scripts/run_rolling_candidate_gate.py \
  --snapshot "$SNAPSHOT_PATH" \
  --candidate agents/candidate_grimmsnarl_rolling_value \
  --safety-games 1000 \
  --jobs 8
```

- [ ] Package and run existing plus rolling audits only after safety and Linux latency pass.

```bash
.venv/bin/python scripts/package_submission.py \
  --agent-dir agents/candidate_grimmsnarl_rolling_value \
  --output "submissions/grimmsnarl_rolling_${SNAPSHOT_ID}.tar.gz"
.venv/bin/python scripts/pre_submit_audit.py \
  --candidate "submissions/grimmsnarl_rolling_${SNAPSHOT_ID}.tar.gz" \
  --skip-gate \
  --out-dir "artifacts/rolling_grimmsnarl/${SNAPSHOT_ID}/pre_submit_audit" \
  --keep-out-dir
```

- [ ] Hash the final archive and write `candidate_verdict.json`. No archive is promotable without `PASS`.

- [ ] Rerun the full rolling suite and existing smoke tests.

```bash
.venv/bin/python -m pytest tests/rolling -q
.venv/bin/python scripts/smoke_test_agent.py
```

- [ ] Commit only implementation/tests; never commit the generated archive or reports.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/scripts/run_rolling_candidate_gate.py \
  pokemon-tcg-ai-battle/scripts/package_submission.py \
  pokemon-tcg-ai-battle/scripts/pre_submit_audit.py \
  pokemon-tcg-ai-battle/tests/rolling/test_candidate_gate.py \
  pokemon-tcg-ai-battle/tests/rolling/test_gates.py
git -C /path/to/kaggle-carry commit -m "feat: gate and package rolling challenger"
```

## Task 11: Enforce Freshness, Submit Exactly Once, and Measure Official Episodes

**Files:**

- Create: `pokemon-tcg-ai-battle/scripts/submit_rolling_candidate_once.py`
- Create: `pokemon-tcg-ai-battle/scripts/report_rolling_online_result.py`
- Create: `pokemon-tcg-ai-battle/tests/rolling/test_submission_guard.py`
- Modify: `pokemon-tcg-ai-battle/tests/rolling/test_cli.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class SubmissionIntent:
    competition: str
    archive_sha256: str
    deck_sha256: str
    policy_sha256: str
    ranker_v1_sha256: str
    ranker_v2_sha256: str
    snapshot_id: str
    snapshot_age_hours: float
    message: str

def authorize_submission(
    intent: SubmissionIntent,
    candidate_verdict: GateDecision,
    prior_intents: Sequence[SubmissionIntent],
) -> GateDecision: ...
```

- [ ] Add failing tests proving stale snapshots reject; a missing candidate `PASS` rejects; a changed policy/deck/model rejects packaging-only retry; a second distinct policy candidate rejects; the same archive cannot be uploaded twice; and a validation-only retry is allowed only after a recorded Kaggle validation error.

- [ ] Run the narrow tests and observe missing guard behavior.

```bash
.venv/bin/python -m pytest tests/rolling/test_submission_guard.py tests/rolling/test_cli.py -q
```

- [ ] Implement a dry-run-by-default submission guard. It verifies the live current time, archive and component hashes, candidate verdict, and existing `submission_intent.jsonl`. `--execute` writes the intent before invoking Kaggle and appends the returned submission ID/status afterward. A failed network call preserves the intent and requires explicit `--resume-intent`.

- [ ] Immediately before upload, rerun the current leaderboard download, query the team's current submissions, record the new rank-ten threshold and latest-two IDs, and state which older ID Kaggle will displace by submission order. Do not change teachers/models unless snapshot age exceeds 24 hours; if it does, reject and return to Task 2.

- [ ] Record a schedule miss if execution reaches 2026-08-05 without a passing archive. Do not compensate by weakening gates or introducing a second candidate; snapshot freshness and the official competition deadline remain independently enforced.

- [ ] Run the dry run:

```bash
.venv/bin/python scripts/submit_rolling_candidate_once.py \
  --snapshot "$SNAPSHOT_PATH" \
  --archive "submissions/grimmsnarl_rolling_${SNAPSHOT_ID}.tar.gz"
```

Expected: a complete intent and exact Kaggle command, but no upload.

- [ ] If and only if every prior gate is `PASS` and the snapshot is fresh, execute exactly once:

```bash
.venv/bin/python scripts/submit_rolling_candidate_once.py \
  --snapshot "$SNAPSHOT_PATH" \
  --archive "submissions/grimmsnarl_rolling_${SNAPSHOT_ID}.tar.gz" \
  --execute
```

- [ ] Poll only the submitted candidate's validation state and official episode list. Do not submit another policy while it accumulates evidence. After at least 200 completed public episodes, generate seat, opponent, matchup, validation-error, and current rank-ten comparison reports.

```bash
.venv/bin/python scripts/report_rolling_online_result.py \
  --snapshot "$SNAPSHOT_PATH" \
  --minimum-completed-episodes 200
```

- [ ] Add Python 3.12 compatibility and final archive import checks. A compatibility failure after upload is reported, never hidden.

- [ ] Rerun all rolling tests and scan the implementation plan and source for placeholders.

```bash
.venv/bin/python -m pytest tests/rolling -q
rg -n 'TODO|FIXME|placeholder|NotImplementedError|pass[[:space:]]*(#.*)?$' \
  rolling_policy scripts agents/candidate_grimmsnarl_rolling_value
```

Expected: all tests pass and the placeholder scan is empty except any test fixture explicitly asserting rejection of those strings.

- [ ] Commit source and tests.

```bash
git -C /path/to/kaggle-carry add \
  pokemon-tcg-ai-battle/scripts/submit_rolling_candidate_once.py \
  pokemon-tcg-ai-battle/scripts/report_rolling_online_result.py \
  pokemon-tcg-ai-battle/tests/rolling/test_submission_guard.py \
  pokemon-tcg-ai-battle/tests/rolling/test_cli.py
git -C /path/to/kaggle-carry commit -m "feat: guard one fresh rolling submission"
```

## Final Verification Checklist

- [ ] Confirm every approved-design deliverable maps to an implementation artifact:

| Design deliverable | Implementation artifact |
|---|---|
| Fresh snapshot | `snapshot.json` |
| Replay inventory and hashes | `replay_inventory.jsonl` |
| Visible state/options | `public/{episodes,decisions,options}.jsonl` |
| Offline restoration | `offline_hidden/restoration.jsonl` |
| Value calibration | `holdout_report.json` |
| Branch integrity | `branch_integrity_report.json` |
| Counterfactual labels | `public/counterfactual_options.jsonl` |
| Candidate and models | agent directory plus `identity.json` |
| Feasibility decision | `feasibility_verdict.json` |
| Safety/latency/package audit | `candidate_verdict.json` and audit directory |
| One official submission | `submission_intent.jsonl` plus Kaggle submission ID |
| Post-submission report | `online_report.json` after 200 episodes |

- [ ] Confirm no feature/model/runtime type mismatch: feature schema hash is identical across V1/V2 training, Q1/Q2 training, candidate materialization, archive identity, and runtime.
- [ ] Confirm no placeholder or omitted file named by a commit command.
- [ ] Confirm no task stages the credential archive, raw artifacts, old agents, unrelated root files, or the whole untracked project.
- [ ] Confirm rejection paths stop before the next irreversible step: snapshot failure before download, data failure before holdout, holdout failure before branching, branch failure before candidate, candidate failure before package, and freshness failure before upload.
- [ ] Confirm the final handoff reports observed facts only: gate results, hashes, submission ID/status, completed official episodes, and live threshold. It must not claim a likely Kaggle rating from local games.
