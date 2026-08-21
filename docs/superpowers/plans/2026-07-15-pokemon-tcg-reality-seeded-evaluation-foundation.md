# Pokémon TCG Reality-Seeded Evaluation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a trustworthy Gate A/Gate B decision instrument: two source-bound exact-60 tournament decks, code-and-deck policy identities, a preregistered randomized-block full-game runner with an intention-to-treat ledger, and a calibrated matched-determinization decision-root evaluator.

**Architecture:** Start from the already hardened `codex/ptcg-evaluation-foundation` branch in a new worktree, but treat the current untracked Pokémon project as read-only source material. Shared library modules own schemas, hashing, validation, allocation, statistics, engine loading, and worker isolation. Thin CLIs create immutable manifests, execute only those manifests, and analyze only complete ledgers. Raw card data, decision roots, outcomes, and vendored `cg` binaries remain outside Git; committed reports bind them by SHA-256. Every promotion path dispatches on evidence kind and fails closed.

**Tech Stack:** Python 3.12+, standard library, pytest 8.4.2, official vendored `cg` API/native library, JSON/JSONL/TSV, Git worktrees.

## Global Constraints

- Work in a new isolated worktree/branch created from `main`; do not edit the dirty main worktree or its untracked `pokemon-tcg-ai-battle/` tree.
- Set `PTCG_WORKTREE_ROOT=/path/to/kaggle-carry/.worktrees/ptcg-reality-seeded-evaluation`. After Task 1's merge, run every Task 2–17 Python/script command from `$PTCG_WORKTREE_ROOT/pokemon-tcg-ai-battle`; run every repository-level Git command as `git -C "$PTCG_WORKTREE_ROOT" ...`. Never rely on an implicit current directory.
- Merge or cherry-pick the tested foundation history into that isolated branch. Do not overwrite the current untracked scripts with the old branch snapshot.
- Never open, copy, extract, hash, package, stage, or commit `sensitive-transfer-archive.tar.gz`. Do not run a recursive command whose input includes it.
- Never call Kaggle submission APIs. This plan ends at local `PASS`, `HOLD`, or `REJECT` reports and small diagnostic engine smokes.
- Raw competition data, official episodes, decision roots, outcome ledgers, and `cg` binaries stay outside Git. Only source manifests, schemas, tests, and hash-bound validation reports are committed.
- `unpaired_full_game` and legacy `paired_counterfactual` evidence may veto but can never promote. In this plan only analyzer-attested `matched_determinization` and `randomized_block_full_game` evidence may support a positive decision. `official_online` is a reserved enum value and remains non-promotable until a later sealed-online protocol and analyzer are implemented.
- An agent error, invalid action, or agent timeout is an intention-to-treat loss for that agent. An infrastructure failure is resumable only with the same immutable assignment ID. Ambiguous ownership is `HOLD`, never deletion.
- A policy experiment must keep deck and engine hashes fixed. A deck experiment must keep policy and engine hashes fixed. A comparison changing both is labeled `artifact` and cannot attribute either change.
- Use `PTCG_CARD_DATA` for `EN_Card_Data.csv` and `PTCG_CG_DIR` for the parent directory that contains the official `cg/` package. No runtime discovery may silently choose a different engine.
- Use test-driven development: add the named failing test, run it and observe the stated failure, add the minimum implementation, rerun the narrow test, then run the whole suite.
- Commit after every task with the exact commit message shown. Do not stage unrelated files.

## Scope Decomposition and Stop Points

This plan has two sequential, independently reviewable phases because the randomized full-game system and the decision-root simulator are separate subsystems.

1. **Phase I — exact-deck and randomized full-game foundation (Tasks 1–10):** integrate the hardened foundation, freeze exact decks, inventory policies, create balanced allocations, execute immutable assignments, and analyze a complete ITT ledger. Its successful terminal status is `FOUNDATION_PASS`, not Gate A `PASS`; full Gate A remains `HOLD_POLICY_CONTRACT`/`PACKAGE_NOT_BUILT` until the later target-policy and packaging plans are complete.
2. **Phase II — Gate B decision-root calibration (Tasks 11–17):** define root artifacts, preserve real episode-persistent policy behavior, run matched determinizations with search enabled, and pass the calibration gate. A failure produces `HOLD`; it does not weaken the gate.

Target Dragapult/Clefairy policy construction, learned models, the runtime verifier, a 2,000-game Gate C2/Gate E experiment, the conditional specialist, packaging for submission, and online uploads are deliberately excluded. They receive later plans only after this foundation passes.

## File Map

### Existing foundation files to retain and extend

- `.gitignore` — exclude local environments, raw evidence, ledgers, and credential-bearing files.
- `pokemon-tcg-ai-battle/pyproject.toml` — Python package and pytest configuration.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/evidence.py` — evidence kinds and strict row parsing.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/manifest.py` — canonical artifact hashing and append-only decision records.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_stats.py` — grouped statistical primitives.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/archive_safety.py` — credential/path/archive safety.
- `pokemon-tcg-ai-battle/scripts/run_local_match.py` — low-level official-engine match primitive.
- `pokemon-tcg-ai-battle/scripts/package_submission.py` — hardened deterministic packaging; retained but not invoked by this plan.
- `pokemon-tcg-ai-battle/tests/test_{evidence,manifest,paired_stats,archive_safety,unpaired_gate}.py` — 291-test regression foundation.

### New or materially extended Phase I files

- `pokemon-tcg-ai-battle/src/ptcg_hybrid/promotion.py` — one fail-closed evidence-to-decision dispatcher.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/decks.py` — strict deck/source/card-data schemas, fingerprints, legality, and mutation distance.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/engine.py` — explicit official `cg` loading and engine identity.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/agent_identity.py` — artifact, policy, deck, and engine hashes plus policy-family deduplication.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/policy_worker.py` — one persistent subprocess per `policy × episode` with attributable timeouts.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/full_game.py` — experiment manifest, balanced assignments, ITT outcome ledger, and randomization analysis.
- `pokemon-tcg-ai-battle/tests/fixtures/card_data_minimal.csv` — tracked deterministic card-data fixture covering both exact decks and duplicate-ID multi-effect rows; unit tests never depend on an external fixture.
- `pokemon-tcg-ai-battle/scripts/validate_exact_deck.py` — card/effect/engine validation CLI.
- `pokemon-tcg-ai-battle/scripts/inventory_policy_families.py` — deterministic policy-family inventory CLI.
- `pokemon-tcg-ai-battle/scripts/create_randomized_full_game_manifest.py` — allocation-only CLI.
- `pokemon-tcg-ai-battle/scripts/run_randomized_full_games.py` — execution-only CLI; no aggregate outcomes.
- `pokemon-tcg-ai-battle/scripts/analyze_randomized_full_games.py` — completion and block-randomization analysis CLI.
- `pokemon-tcg-ai-battle/decks/dragapult_dudunsparce_prague2026/{source.json,mapping.tsv,deck.csv,validation.json}` — primary exact transfer.
- `pokemon-tcg-ai-battle/decks/lillie_clefairy_naic2026/{source.json,mapping.tsv,deck.csv,validation.json}` — independent exact transfer.
- `pokemon-tcg-ai-battle/tests/test_{promotion,decks,engine,agent_identity,policy_worker,full_game}.py` — Phase I contract tests.

### New Phase II files

- `pokemon-tcg-ai-battle/src/ptcg_hybrid/root_schema.py` — immutable decision roots and inner/outer seed namespaces.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/search_audit.py` — bound search delegate and lifecycle evidence.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/continuations.py` — explicit search-free calibration continuation adapter.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_evaluator.py` — one-root, two-action, round-robin matched-determinization evaluator.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/calibration.py` — identity, antisymmetry, order, positive, negative, null, and failure-closure gates.
- `pokemon-tcg-ai-battle/agents/controls/archaludon_v168_measurement/{main.py,deck.csv}` — hash-bound executable evaluator-calibration control, with no vendored engine binaries.
- `pokemon-tcg-ai-battle/agents/controls/archaludon_v168_measurement.control.json` — non-executable sibling attestation, outside the agent directory to avoid executable-identity self-reference.
- `pokemon-tcg-ai-battle/scripts/capture_decision_roots.py` — hash-bound root/prefix capture CLI.
- `pokemon-tcg-ai-battle/scripts/create_outer_determinizations.py` — post-action outer-seed and hidden-state allocation CLI.
- `pokemon-tcg-ai-battle/scripts/create_null_calibration_inputs.py` — pre-outcome independent simulated/duplicated null allocation CLI.
- `pokemon-tcg-ai-battle/scripts/run_simulated_null_calibration.py` — deterministic exchangeable-null ledger generator bound to the frozen null inputs.
- `pokemon-tcg-ai-battle/scripts/run_matched_root_evaluation.py` — outer-namespace paired evaluation CLI.
- `pokemon-tcg-ai-battle/scripts/calibrate_paired_evaluator.py` — Gate B report CLI.
- `pokemon-tcg-ai-battle/scripts/create_evaluator_calibration_manifest.py` — pre-outcome calibration allocation CLI.
- `pokemon-tcg-ai-battle/tests/test_{root_schema,search_audit,paired_evaluator,calibration}.py` — Phase II unit/integration contracts.

---

## Task 1: Create the Isolated Integration Branch and Preserve the 291-Test Foundation

**Files:**

- Modify: `.gitignore`
- Review: `pokemon-tcg-ai-battle/pyproject.toml`
- Review: `pokemon-tcg-ai-battle/src/ptcg_hybrid/*.py`
- Review: `pokemon-tcg-ai-battle/scripts/{evaluate_pool_from_csv,run_candidate_gate,run_local_match,package_submission}.py`
- Review: `pokemon-tcg-ai-battle/tests/*.py`

**Interfaces:** No public API change in this task. The observable contract is the existing 291-test suite.

- [ ] From the clean repository root, use `superpowers:using-git-worktrees` and create a worktree from `main` named `.worktrees/ptcg-reality-seeded-evaluation` on branch `codex/ptcg-reality-seeded-evaluation`.

```bash
export PTCG_WORKTREE_ROOT=/path/to/kaggle-carry/.worktrees/ptcg-reality-seeded-evaluation
git worktree add "$PTCG_WORKTREE_ROOT" -b codex/ptcg-reality-seeded-evaluation main
```

Expected: a new clean worktree. The dirty main worktree remains unchanged.

- [ ] In every shell used by this plan, bind the read-only source tree, raw competition card data, and the explicit vendored official engine to these exact paths. Refuse to continue if any required input is absent; never replace them with runtime discovery.

```bash
export PTCG_WORKTREE_ROOT=/path/to/kaggle-carry/.worktrees/ptcg-reality-seeded-evaluation
export PTCG_SOURCE_ROOT=/path/to/kaggle-carry/pokemon-tcg-ai-battle
export PTCG_CARD_DATA="$PTCG_SOURCE_ROOT/data/raw/pokemon-tcg-ai-battle/EN_Card_Data.csv"
export PTCG_CG_DIR="$PTCG_SOURCE_ROOT/agents/baselines/v11_hammer_metal_from_submission"
test -d "$PTCG_SOURCE_ROOT"
test -f "$PTCG_CARD_DATA"
test -d "$PTCG_CG_DIR/cg"
```

Expected: all checks succeed. `PTCG_SOURCE_ROOT` is read-only source material; all new tracked files are created only inside the isolated worktree.

- [ ] In the new worktree, merge the completed foundation branch, then inspect only tracked conflicts.

```bash
git -C "$PTCG_WORKTREE_ROOT" merge --no-ff codex/ptcg-evaluation-foundation -m "Merge hardened PTCG evaluation foundation"
git -C "$PTCG_WORKTREE_ROOT" status --short
```

Expected: the merge completes without reading or overwriting the main worktree's untracked project. If a tracked conflict occurs, resolve it by retaining the hardened library/tests and the newer approved design docs; do not copy the whole untracked project.

- [ ] Create a clean Python 3.12 environment and install the package development dependencies.

```bash
cd "$PTCG_WORKTREE_ROOT/pokemon-tcg-ai-battle"
python3 --version
python3 -m venv .venv312
.venv312/bin/python -m pip install -e '.[dev]'
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m pytest -q -p no:cacheprovider
```

Expected: Python is at least 3.12 and `291 passed`. The already observed Python 3.14 result is valid for development; an exact Python 3.12 compatibility run remains a closing requirement in Task 17.

- [ ] Extend the merged root `.gitignore` with `**/.venv*/` and `pokemon-tcg-ai-battle/artifacts/`. Verify a dummy path under each rule is ignored with `git check-ignore`; do not create a credential-shaped test file.

- [ ] Confirm the credential archive is outside the candidate worktree and no ignored/untracked file was staged.

```bash
git -C "$PTCG_WORKTREE_ROOT" status --short
git -C "$PTCG_WORKTREE_ROOT" diff --cached --name-only
```

Expected: no credential archive path and no unrelated file.

- [ ] Commit the ignore rules and any tracked merge-resolution changes; retain all unrelated main-worktree files outside this branch.

```bash
git -C "$PTCG_WORKTREE_ROOT" add .gitignore
git -C "$PTCG_WORKTREE_ROOT" commit -m "chore: integrate hardened PTCG evaluation foundation"
```

## Task 2: Centralize Evidence Kinds and Fail-Closed Promotion

**Files:**

- Modify: `pokemon-tcg-ai-battle/src/ptcg_hybrid/evidence.py`
- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/promotion.py`
- Modify: `pokemon-tcg-ai-battle/scripts/run_candidate_gate.py`
- Create from reviewed read-only source if absent: `pokemon-tcg-ai-battle/scripts/repeat_candidate_gate.py`
- Create from reviewed read-only source if absent: `pokemon-tcg-ai-battle/scripts/summarize_action_diff_gate.py`
- Create from reviewed read-only source if absent: `pokemon-tcg-ai-battle/scripts/pre_submit_audit.py`
- Create: `pokemon-tcg-ai-battle/tests/test_promotion.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_evidence.py`

**Interfaces:**

```text
class EvidenceKind(StrEnum):
    UNPAIRED_FULL_GAME = "unpaired_full_game"
    PAIRED_COUNTERFACTUAL = "paired_counterfactual"
    MATCHED_DETERMINIZATION = "matched_determinization"
    RANDOMIZED_BLOCK_FULL_GAME = "randomized_block_full_game"
    OFFICIAL_ONLINE = "official_online"

class Decision(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"
    HOLD = "HOLD"
    HOLD_UNPAIRED = "HOLD_UNPAIRED"

class AnalyzerId(StrEnum):
    MATCHED_ROOT_V1 = "matched_root_v1"
    RANDOMIZED_BLOCK_V1 = "randomized_block_v1"

@dataclass(frozen=True, slots=True)
class AnalyzerAttestation:
    evidence_kind: EvidenceKind
    analyzer_id: AnalyzerId
    manifest_sha256: str
    report_sha256: str
    complete: bool
    proposed_decision: Decision

eligible_to_promote(attestation: AnalyzerAttestation) -> bool
require_promotable(attestation: AnalyzerAttestation) -> Decision
```

- [ ] Add failing tests proving both legacy evidence kinds turn an attempted `PASS` into `HOLD_UNPAIRED`; incomplete attestations and analyzer/evidence mismatches return `HOLD`; unknown analyzer IDs raise; `official_online` remains `HOLD`; malformed/uppercase/short/nonhex digests fail; and each of the two new local evidence kinds may reach `PASS` only through its named analyzer with strict lowercase 64-character SHA-256 manifest/report hashes.

- [ ] Add subprocess regression cases for all four legacy promotion CLIs. Feed each an otherwise favorable unpaired/action-agreement fixture and assert it cannot print `PASS`/`SUBMIT_CANDIDATE` and exits nonzero with `HOLD_UNPAIRED`.

```python
import pytest

from ptcg_hybrid.evidence import EvidenceKind
from ptcg_hybrid.promotion import (
    AnalyzerAttestation,
    AnalyzerId,
    Decision,
    require_promotable,
)


@pytest.mark.parametrize(
    "kind",
    [
        EvidenceKind.UNPAIRED_FULL_GAME,
        EvidenceKind.PAIRED_COUNTERFACTUAL,
    ],
)
def test_nonprospective_evidence_cannot_promote(kind: EvidenceKind) -> None:
    attestation = AnalyzerAttestation(
        evidence_kind=kind,
        analyzer_id=AnalyzerId.RANDOMIZED_BLOCK_V1,
        manifest_sha256="a" * 64,
        report_sha256="b" * 64,
        complete=True,
        proposed_decision=Decision.PASS,
    )
    assert require_promotable(attestation) is Decision.HOLD_UNPAIRED


@pytest.mark.parametrize(
    "kind_and_analyzer",
    [
        (EvidenceKind.MATCHED_DETERMINIZATION, AnalyzerId.MATCHED_ROOT_V1),
        (EvidenceKind.RANDOMIZED_BLOCK_FULL_GAME, AnalyzerId.RANDOMIZED_BLOCK_V1),
    ],
)
def test_eligible_evidence_preserves_analyzer_decision(
    kind_and_analyzer: tuple[EvidenceKind, AnalyzerId],
) -> None:
    kind, analyzer = kind_and_analyzer
    attestation = AnalyzerAttestation(
        evidence_kind=kind,
        analyzer_id=analyzer,
        manifest_sha256="a" * 64,
        report_sha256="b" * 64,
        complete=True,
        proposed_decision=Decision.PASS,
    )
    assert require_promotable(attestation) is Decision.PASS


def test_rejection_is_never_weakened() -> None:
    attestation = AnalyzerAttestation(
        evidence_kind=EvidenceKind.UNPAIRED_FULL_GAME,
        analyzer_id=AnalyzerId.RANDOMIZED_BLOCK_V1,
        manifest_sha256="a" * 64,
        report_sha256="b" * 64,
        complete=True,
        proposed_decision=Decision.REJECT,
    )
    assert require_promotable(attestation) is Decision.REJECT
```

- [ ] Run the narrow tests and observe failure because the new enum values/module do not exist.

```bash
.venv312/bin/python -m pytest tests/test_evidence.py tests/test_promotion.py -q
```

- [ ] Implement the enum extension and central dispatcher. Preserve the existing member name and serialized value `PAIRED_COUNTERFACTUAL = "paired_counterfactual"`; do not silently rename old data. Extend strict row validation so `matched_determinization` requires `paired=1` and `randomized_block_full_game` requires `paired=0`.

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from ptcg_hybrid.evidence import EvidenceKind


class Decision(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"
    HOLD = "HOLD"
    HOLD_UNPAIRED = "HOLD_UNPAIRED"


class AnalyzerId(StrEnum):
    MATCHED_ROOT_V1 = "matched_root_v1"
    RANDOMIZED_BLOCK_V1 = "randomized_block_v1"


@dataclass(frozen=True, slots=True)
class AnalyzerAttestation:
    evidence_kind: EvidenceKind
    analyzer_id: AnalyzerId
    manifest_sha256: str
    report_sha256: str
    complete: bool
    proposed_decision: Decision


_PROMOTABLE_ANALYZERS = {
    EvidenceKind.MATCHED_DETERMINIZATION: AnalyzerId.MATCHED_ROOT_V1,
    EvidenceKind.RANDOMIZED_BLOCK_FULL_GAME: AnalyzerId.RANDOMIZED_BLOCK_V1,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _validate_attestation(attestation: AnalyzerAttestation) -> None:
    if type(attestation.evidence_kind) is not EvidenceKind:
        raise TypeError("unknown evidence kind")
    if type(attestation.analyzer_id) is not AnalyzerId:
        raise TypeError("unknown analyzer id")
    if type(attestation.proposed_decision) is not Decision:
        raise TypeError("unknown decision")
    if type(attestation.complete) is not bool:
        raise TypeError("complete must be bool")
    for digest in (attestation.manifest_sha256, attestation.report_sha256):
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            raise ValueError("invalid sha256")


def eligible_to_promote(attestation: AnalyzerAttestation) -> bool:
    _validate_attestation(attestation)
    return (
        attestation.complete
        and _PROMOTABLE_ANALYZERS.get(attestation.evidence_kind)
        is attestation.analyzer_id
    )


def require_promotable(attestation: AnalyzerAttestation) -> Decision:
    _validate_attestation(attestation)
    if attestation.proposed_decision is not Decision.PASS:
        return attestation.proposed_decision
    if eligible_to_promote(attestation):
        return Decision.PASS
    if attestation.evidence_kind in {
        EvidenceKind.UNPAIRED_FULL_GAME,
        EvidenceKind.PAIRED_COUNTERFACTUAL,
    }:
        return Decision.HOLD_UNPAIRED
    return Decision.HOLD
```

- [ ] Route every present promotion surface through an `AnalyzerAttestation` and `require_promotable`: `run_candidate_gate.py`, `repeat_candidate_gate.py`, `summarize_action_diff_gate.py`, and `pre_submit_audit.py`. Selectively port the latter three from their exact read-only `$PTCG_SOURCE_ROOT/scripts/` files if the foundation merge lacks them; do not copy unrelated scripts. Historical repeats, action agreement, unpaired deltas, and pre-submit summaries must terminate as `HOLD_UNPAIRED`/nonzero and can never emit `PASS` or `SUBMIT_CANDIDATE`. `Decision.PASS` belongs only to analyzer reports and is not serialized into the existing `manifest.ExperimentRecord`; keep that module's existing allowed decision vocabulary unchanged.

- [ ] While selectively porting `pre_submit_audit.py`, remove its unavailable top-level imports from `analyze_submission_replays` and `mine_episode_replays`. Implement the tiny strict deck-line reader and legacy diagnostic fingerprint locally (or with already merged hardened primitives) and add an import-smoke test; do not pull either large replay-mining script into the foundation merely to satisfy those two helpers.

- [ ] Run all tests.

```bash
.venv312/bin/python -m pytest -q -p no:cacheprovider
```

Expected: all tests pass, and no unpaired CLI can print a promotable `PASS`.

- [ ] Commit.

```bash
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/evidence.py \
  pokemon-tcg-ai-battle/src/ptcg_hybrid/promotion.py \
  pokemon-tcg-ai-battle/scripts/run_candidate_gate.py \
  pokemon-tcg-ai-battle/scripts/repeat_candidate_gate.py \
  pokemon-tcg-ai-battle/scripts/summarize_action_diff_gate.py \
  pokemon-tcg-ai-battle/scripts/pre_submit_audit.py \
  pokemon-tcg-ai-battle/tests/test_evidence.py \
  pokemon-tcg-ai-battle/tests/test_promotion.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: centralize fail-closed PTCG evidence decisions"
```

## Task 3: Implement Strict Exact-60 and Mutation Contracts

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/decks.py`
- Create: `pokemon-tcg-ai-battle/tests/test_decks.py`
- Create: `pokemon-tcg-ai-battle/tests/fixtures/card_data_minimal.csv`

**Interfaces:**

```text
@dataclass(frozen=True, slots=True)
class SourceCard:
    count: int
    name: str
    printing: str
    source_card_url: str
    source_effect_sha256: str
    source_semantics: dict[str, object]

@dataclass(frozen=True, slots=True)
class DeckSource:
    schema_version: int
    deck_id: str
    transfer_kind: Literal["exact", "adapted"]
    source_url: str
    event: str
    event_date: str
    player: str
    placing: int
    format: str
    category_counts: dict[str, int]
    published_cards: tuple[SourceCard, ...]

@dataclass(frozen=True, slots=True)
class DeckEntry:
    count: int
    card_id: int
    source_name: str
    source_printing: str
    mapped_name: str
    mapped_printing: str
    equivalence: Literal["exact_print", "reviewed_rule_equivalent"]
    review_note: str

@dataclass(frozen=True, slots=True)
class CardRecord:
    card_id: int
    card_name: str
    rows_sha256: str
    printings: tuple[str, ...]
    stage_or_type: tuple[str, ...]
    rules: tuple[str, ...]
    categories: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class DeckValidation:
    deck_id: str
    source_capture_sha256: str
    ordered_file_sha256: str
    multiset_sha256: str
    mapping_sha256: str
    card_data_sha256: str
    category_counts: dict[str, int]
    card_records: tuple[CardRecord, ...]
    errors: tuple[str, ...]

load_source(path: Path) -> DeckSource
load_mapping(path: Path) -> tuple[DeckEntry, ...]
load_deck(path: Path) -> tuple[int, ...]
load_card_data(path: Path) -> dict[int, CardRecord]
multiset_sha256(deck: Sequence[int]) -> str
validate_exact_deck(deck_dir: Path, card_data_path: Path) -> DeckValidation
mutation_distance(base: Sequence[int], candidate: Sequence[int]) -> tuple[int, int]
require_allowed_mutation(base: Sequence[int], candidate: Sequence[int], limit: int = 4) -> None
```

- [ ] Create `tests/fixtures/card_data_minimal.csv` with the exact competition rows needed by both committed decks plus deliberate same-ID multi-effect rows. Write failing unit tests against a copied/mutated version of that tracked fixture for strict headers, exact expansion of mapping counts, 59/61-card rejection, unknown IDs, name mismatch, independent source-capture/mapping mismatch, source expansion/collection mismatch, full multi-row effect hashing, Basic Energy copy exemption, ACE SPEC limit, at least one Basic Pokémon, source category totals, deterministic ordered/multiset SHA-256, and `(removed, added) <= (4, 4)` mutation enforcement. Consistent duplicate Card-ID rows with the same name/printing are valid and all contribute to the effect hash; only inconsistent duplicate identity fields are rejected. No committed unit test may depend on an external `card_data_path` fixture or silently skip when raw Kaggle data is unavailable.

```python
def test_mutation_distance_counts_copies_not_unique_names() -> None:
    base = tuple(range(1, 61))
    candidate = base[:-4] + (100, 100, 101, 101)
    assert mutation_distance(base, candidate) == (4, 4)
    require_allowed_mutation(base, candidate)


def test_five_copy_mutation_is_rejected() -> None:
    base = tuple(range(1, 61))
    candidate = base[:-5] + (100, 100, 101, 101, 102)
    with pytest.raises(ValueError, match="removed=5, added=5"):
        require_allowed_mutation(base, candidate)
```

- [ ] Run the new test and observe import failure.

```bash
.venv312/bin/python -m pytest tests/test_decks.py -q
```

- [ ] Implement strict parsers. `mapping.tsv` must have exactly these columns in this order:

```text
count	card_id	source_name	source_printing	mapped_name	mapped_printing	equivalence	review_note
```

Reject duplicate columns, extra columns, blank values, nonpositive counts/IDs, duplicate mapped IDs, and `reviewed_rule_equivalent` rows without a nonempty review note. Parse `source.json` independently, require `schema_version == 1`, hash its full canonical representation as `source_capture_sha256`, and require the mapping's `(count, source_name, source_printing)` multiset to equal `published_cards`; a mapping can never certify its own source facts. Collapse repeated card-data rows by ID only after verifying that every repeated row has the same card name and printing. Hash the canonical JSON representation of all sorted rows for that ID so multi-attack Pokémon are not truncated. Validate `mapped_printing` against the competition CSV. `exact_print` requires both source/mapped name and source/mapped printing to match; any legal reprint or normalized-name mapping requires `reviewed_rule_equivalent` plus an explicit complete-effect review note. Every `SourceCard` must contain its own card-page URL, source-effect SHA-256, and structured source semantics covering every engine-relevant field; free-text notes alone never establish equivalence.

- [ ] Implement deck rules and hashing. Use full 64-character SHA-256; keep the old 12-character SHA-1 fingerprint only as a reported legacy alias, never as an identity key.

```python
def multiset_sha256(deck: Sequence[int]) -> str:
    counts = Counter(deck)
    payload = "\n".join(f"{card_id}:{counts[card_id]}" for card_id in sorted(counts))
    return hashlib.sha256(b"ptcg-deck-multiset:v1\0" + payload.encode("ascii")).hexdigest()


def mutation_distance(
    base: Sequence[int], candidate: Sequence[int]
) -> tuple[int, int]:
    base_counts = Counter(base)
    candidate_counts = Counter(candidate)
    removed = sum((base_counts - candidate_counts).values())
    added = sum((candidate_counts - base_counts).values())
    return removed, added
```

- [ ] Run the narrow and full suites.

```bash
.venv312/bin/python -m pytest tests/test_decks.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
```

- [ ] Commit.

```bash
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/decks.py \
  pokemon-tcg-ai-battle/tests/test_decks.py \
  pokemon-tcg-ai-battle/tests/fixtures/card_data_minimal.csv
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: enforce source-bound exact-60 deck contracts"
```

## Task 4: Materialize the Two Tournament Deck Manifests

**Files:**

- Create: `pokemon-tcg-ai-battle/decks/dragapult_dudunsparce_prague2026/source.json`
- Create: `pokemon-tcg-ai-battle/decks/dragapult_dudunsparce_prague2026/mapping.tsv`
- Create: `pokemon-tcg-ai-battle/decks/dragapult_dudunsparce_prague2026/deck.csv`
- Create: `pokemon-tcg-ai-battle/decks/lillie_clefairy_naic2026/source.json`
- Create: `pokemon-tcg-ai-battle/decks/lillie_clefairy_naic2026/mapping.tsv`
- Create: `pokemon-tcg-ai-battle/decks/lillie_clefairy_naic2026/deck.csv`
- Modify: `pokemon-tcg-ai-battle/tests/test_decks.py`

**Interfaces:** Both directories satisfy `validate_exact_deck()` with no errors against the tracked fixture before engine validation is added. Each independently captured `source.json` is exactly the Task 3 `DeckSource` schema and contains the published list's structured `published_cards`; `mapping.tsv` is a separately authored mapping that must cross-check it.

- [ ] Add failing golden tests that locate both committed deck directories, assert 60 cards, assert the published category totals, and prove the existing Clefairy candidate substitution is not equal to the exact baseline.

```python
@pytest.mark.parametrize(
    ("name", "expected_counts"),
    [
        (
            "dragapult_dudunsparce_prague2026",
            {"pokemon": 19, "trainer": 32, "energy": 9},
        ),
        (
            "lillie_clefairy_naic2026",
            {"pokemon": 22, "trainer": 28, "energy": 10},
        ),
    ],
)
def test_committed_exact_decks_match_source_totals(
    name: str, expected_counts: dict[str, int]
) -> None:
    report = validate_exact_deck(DECKS / name, FIXTURES / "card_data_minimal.csv")
    assert report.errors == ()
    assert report.category_counts == expected_counts
```

- [ ] Add Prague `source.json` using these frozen facts: Mateusz Łaszkiewicz, first place, Regional Prague, 2026-04-25, `Temporal Forces - Perfect Order`, source `https://limitlesstcg.com/tournaments/539/decklists`, exact transfer, 19/32/9.

- [ ] Add Prague `mapping.tsv` with these exact counts and competition IDs. Use the source printing shown below; lines without an arrow use the same source/mapped name and printing, while an arrow records a competition reprint. `mapped_name` must exactly match the competition card data spelling.

```text
4  119   Dreepy                  TWM 128
4  120   Drakloak                TWM 129
3  121   Dragapult ex            TWM 130
2  305   Dunsparce               JTG 120
2  66    Dudunsparce             TEF 129
1  306   Dudunsparce ex          JTG 121
2  112   Munkidori               TWM 95
1  235   Budew                   PRE 4
4  1227  Lillie's Determination  MEG 119
3  1182  Boss's Orders           MEG 114  ->  Boss’s Orders  PAL 172
3  1198  Crispin                 SCR 133
2  1210  Brock's Scouting        JTG 146
1  1228  Acerola's Mischief      MEG 113
4  1086  Buddy-Buddy Poffin      TEF 144
4  1152  Poké Pad                POR 81
4  1121  Ultra Ball              SVI 196
2  1122  Pokégear 3.0            SVI 186
2  1097  Night Stretcher         SFA 61
1  1159  Hero's Cape             TEF 152
2  1260  Risky Ruins             MEG 127
4  5     Basic {P} Energy        SVE 5
3  2     Basic {R} Energy        SVE 2
2  7     Basic {D} Energy        SVE 7
```

For the `Boss's Orders` MEG 114 to `Boss’s Orders` PAL 172 reprint, use `reviewed_rule_equivalent` and record the complete-effect comparison. For punctuation-only differences (`Brock's`/`Brock’s`, `Hero's`/`Hero’s`), use `reviewed_rule_equivalent` and state that normalization is the only difference. All exact source prints use `exact_print`.

- [ ] Expand Prague `deck.csv` in mapping order, one integer ID per line, exactly 60 lines.

- [ ] Promote the existing NAIC evidence into the strict schema rather than using `agents/candidate_lillie_clefairy/deck.csv`. Freeze James Kowalski, first place, NAIC 2026, source `https://limitlesstcg.com/decks/list/28249`, format `Temporal Forces - Chaos Rising`, exact transfer, 22/28/10. Preserve the source's one `Chien-Pao (209)` and two `Lillie's Pearl (1172)`. Correct its Boss source record to MEG 114 mapped to the rule-equivalent PAL 172 competition ID, and retain the explicit Telepathic/Telepath Psychic Energy name-equivalence review.

- [ ] Author the NAIC `source.json` independently from the published page, then author `mapping.tsv` from this complete frozen 24-row conversion. The left side is the source printing; an arrow identifies the reviewed competition printing/name. Basic Energy source pages and effect hashes must still be captured even when the competition printing is SVE. `Telepathic Psychic Energy` POR 88 to raw-ID `19` (`Telepath Psychic Energy` POR 87) is an explicit reviewed reprint/name mapping, not an exact-name shortcut.

```text
4  756   Mega Kangaskhan ex              MEG 104
4  1071  Meowth ex                       POR 62
4  272   Lillie's Clefairy ex            JTG 56
3  184   Latias ex                       SSP 76
2  108   Wellspring Mask Ogerpon ex      TWM 64
2  140   Fezandipiti ex                  SFA 38
1  791   Moltres                         PFL 14
1  209   Chien-Pao                       SSP 56
1  979   Koraidon ex                     ASC 121
4  1198  Crispin                         SCR 133
3  1182  Boss's Orders                   MEG 114  ->  Boss’s Orders                 PAL 172
2  1188  Ciphermaniac's Codebreaking     TEF 145
1  1205  Cyrano                          SSP 170
4  1121  Ultra Ball                      MEG 131  ->  Ultra Ball                    SVI 196
4  1102  Dusk Ball                       SSP 175
3  1146  Wondrous Patch                  PFL 94
1  1088  Prime Catcher                   TEF 157
2  1172  Lillie's Pearl                  JTG 151
4  1250  Area Zero Underdepths           SCR 131
4  5     Basic {P} Energy                published basic Psychic  ->  Basic {P} Energy  SVE 5
2  3     Basic {W} Energy                published basic Water    ->  Basic {W} Energy  SVE 3
2  6     Basic {F} Energy                published basic Fighting ->  Basic {F} Energy  SVE 6
1  19    Telepathic Psychic Energy       POR 88   ->  Telepath Psychic Energy      POR 87
1  2     Basic {R} Energy                published basic Fire     ->  Basic {R} Energy  SVE 2
```

- [ ] Run both committed golden validations against the tracked minimal card-data fixture.

```bash
.venv312/bin/python -m pytest tests/test_decks.py -q
```

Expected: the tracked-fixture golden tests pass. Task 5's explicit CLI performs the required full raw-data integration validation and may neither skip nor fall back to the fixture. The legacy Clefairy candidate differs by one removed and one added copy and cannot be called exact.

- [ ] Commit.

```bash
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/decks \
  pokemon-tcg-ai-battle/tests/test_decks.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "data: freeze Prague and NAIC exact-60 tournament decks"
```

## Task 5: Bind Deck Validation to the Explicit Official Engine

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/engine.py`
- Create: `pokemon-tcg-ai-battle/scripts/validate_exact_deck.py`
- Create: `pokemon-tcg-ai-battle/tests/test_engine.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_decks.py`
- Create: both `decks/*/validation.json` files after the CLI passes.

**Interfaces:**

```text
@dataclass(frozen=True, slots=True)
class EngineIdentity:
    python_sources_sha256: str
    api_sha256: str
    native_library_name: str
    native_library_sha256: str
    platform: str
    python_version: str

@dataclass(frozen=True, slots=True)
class OfficialCG:
    game: ModuleType
    api: ModuleType
    identity: EngineIdentity

@dataclass(frozen=True, slots=True)
class EffectEquivalence:
    card_id: int
    source_effect_sha256: str
    source_semantics_sha256: str
    engine_card_data_sha256: str
    engine_attacks_sha256: str
    engine_semantics_sha256: str
    equivalent: bool

load_official_cg(cg_dir: Path) -> OfficialCG
engine_identity(cg_dir: Path) -> EngineIdentity
validate_engine_deck(cg: OfficialCG, deck: Sequence[int]) -> None
compare_source_to_engine_effects(cg: OfficialCG, source: DeckSource, mapping: Sequence[DeckEntry]) -> tuple[EffectEquivalence, ...]
```

- [ ] Write failing tests proving an absent `cg` directory, a directory without the platform-native library, stale `cg` and `cg.*` modules being fully cleared before import, `battle_start()` returning `None` instead of `(observation, start_data)`, and a tuple whose observation is `None` all fail closed. Add effect tests proving a free-text review note cannot hide a changed attack cost, damage/effect, ability, weakness/resistance/retreat, trainer text, energy text, prize/rule marker, or other engine-relevant field. Use fake modules for unit tests; do not require native binaries in pytest.

- [ ] Implement an explicit loader that may be called only from an already spawned worker; the loader itself never spawns and never attempts to return `ModuleType` objects across a process boundary. It must resolve the directory, verify `cg/__init__.py`, choose `libcg.dylib` on Darwin, `libcg-arm64.so` on Linux arm64, and `libcg.so` on other Linux x86-64, delete every `sys.modules` key equal to `cg` or beginning with `cg.`, prepend only the requested parent, import `cg.game` and `cg.api`, and verify both module files are under that directory. Hash exactly the allowlisted Python sources `{__init__.py, api.py, game.py, sim.py, utils.py}` plus only the selected current-platform native binary; exclude caches, generated files, Windows binaries, and native binaries for other platforms. Return `OfficialCG(game, api, identity)` within that worker. Agent-vendored `cg` comparisons use the allowlisted Python-source identity, while every runtime identity always binds the selected native-library hash explicitly.

- [ ] Implement engine legality in that worker with `started = False`; call `observation, start_data = cg.game.battle_start(deck, deck)`, validate the return shape and `observation is not None`, then set `started = True`. Retain `start_data` only for diagnostics and call `cg.game.battle_finish()` in `finally` only when `started` is true. Treat an invalid return shape, engine error, or null observation as validation failure; tests must prove `battle_finish()` is neither skipped after a valid start nor called against an invalid native battle pointer.

- [ ] Implement structured full-effect equivalence using `cg.api.all_card_data()` and `cg.api.all_attack()`. Canonicalize every engine-relevant field—identity/type, HP/stage/evolution, ability text and conditions, attack energy/cost/damage/effects, trainer/energy effects, weakness/resistance/retreat, and prize/rule markers—then bind the source-effect hash, engine card-data hash, engine attack hash, and both canonical semantic hashes. An exact print or reviewed reprint passes only when the structured semantics are equal after explicitly allowlisted punctuation/name normalization; a free-text note never overrides a mismatch.

- [ ] Implement the CLI. It reads only `--deck-dir`, `--card-data`, `--cg-dir`, and `--out`; the parent parses files, then starts one spawn worker that loads/uses `OfficialCG` and returns only JSON-serializable identity, legality, and effect records—never module objects. The parent writes canonical JSON atomically, omits absolute paths, includes `source_capture_sha256`, all deck/card/engine hashes, every unique mapped card's `EffectEquivalence`, and `engine_legal: true`, and exits nonzero on missing raw data, missing engine data, worker failure, or any mismatch. This is the mandatory non-skippable full-data integration check; it never substitutes `tests/fixtures/card_data_minimal.csv` for `--card-data`.

```bash
.venv312/bin/python scripts/validate_exact_deck.py \
  --deck-dir decks/dragapult_dudunsparce_prague2026 \
  --card-data "$PTCG_CARD_DATA" \
  --cg-dir "$PTCG_CG_DIR" \
  --out decks/dragapult_dudunsparce_prague2026/validation.json

.venv312/bin/python scripts/validate_exact_deck.py \
  --deck-dir decks/lillie_clefairy_naic2026 \
  --card-data "$PTCG_CARD_DATA" \
  --cg-dir "$PTCG_CG_DIR" \
  --out decks/lillie_clefairy_naic2026/validation.json
```

Expected: each command prints one line containing `status=PASS`, an ordered hash, a multiset hash, and the native library hash. If either fails, stop Phase I with decision `HOLD` and reason code `HOLD_DECK_MAPPING`.

- [ ] Re-run the CLI once and prove byte-identical output.

```bash
DRAG_VALIDATION_SHA256=$(shasum -a 256 decks/dragapult_dudunsparce_prague2026/validation.json | awk '{print $1}')
CLEFAIRY_VALIDATION_SHA256=$(shasum -a 256 decks/lillie_clefairy_naic2026/validation.json | awk '{print $1}')

.venv312/bin/python scripts/validate_exact_deck.py \
  --deck-dir decks/dragapult_dudunsparce_prague2026 \
  --card-data "$PTCG_CARD_DATA" \
  --cg-dir "$PTCG_CG_DIR" \
  --out decks/dragapult_dudunsparce_prague2026/validation.json

.venv312/bin/python scripts/validate_exact_deck.py \
  --deck-dir decks/lillie_clefairy_naic2026 \
  --card-data "$PTCG_CARD_DATA" \
  --cg-dir "$PTCG_CG_DIR" \
  --out decks/lillie_clefairy_naic2026/validation.json

test "$(shasum -a 256 decks/dragapult_dudunsparce_prague2026/validation.json | awk '{print $1}')" = "$DRAG_VALIDATION_SHA256"
test "$(shasum -a 256 decks/lillie_clefairy_naic2026/validation.json | awk '{print $1}')" = "$CLEFAIRY_VALIDATION_SHA256"
```

- [ ] Run all tests and commit.

```bash
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/engine.py \
  pokemon-tcg-ai-battle/scripts/validate_exact_deck.py \
  pokemon-tcg-ai-battle/tests/test_engine.py \
  pokemon-tcg-ai-battle/tests/test_decks.py \
  pokemon-tcg-ai-battle/decks/dragapult_dudunsparce_prague2026/validation.json \
  pokemon-tcg-ai-battle/decks/lillie_clefairy_naic2026/validation.json
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: validate exact decks in the official engine"
```

## Task 6: Define Canonical Agent Identity and Policy-Family Deduplication

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/agent_identity.py`
- Create: `pokemon-tcg-ai-battle/scripts/inventory_policy_families.py`
- Create: `pokemon-tcg-ai-battle/tests/test_agent_identity.py`

**Interfaces:**

```text
@dataclass(frozen=True, slots=True)
class AgentIdentity:
    label: str
    artifact_sha256: str
    policy_sha256: str
    deck_sha256: str
    deck_multiset_sha256: str
    engine_sha256: str

@dataclass(frozen=True, slots=True)
class PolicyFamily:
    family_id: str
    policy_sha256: str
    members: tuple[AgentIdentity, ...]

identify_agent(label: str, agent_dir: Path, cg_dir: Path) -> AgentIdentity
deduplicate_policy_families(agents: Sequence[AgentIdentity]) -> tuple[PolicyFamily, ...]
validate_one_axis(experiment_kind: Literal["policy", "deck", "artifact"], control: AgentIdentity, candidate: AgentIdentity) -> None
```

`artifact_sha256` is the domain-separated hash of `(policy_sha256, deck_sha256, engine_sha256)` and therefore represents the evaluated artifact rather than an arbitrary directory layout. `deck_sha256` hashes the ordered `deck.csv`. `policy_sha256` hashes all executable policy/model/helper files except `deck.csv`, cache files, and the `cg/` directory. The measurement-control attestation is a sibling file outside the agent directory and receives its own `control_attestation_sha256` after it records the already frozen artifact identity, avoiding a self-referential hash without a generic identity exclusion. `engine_sha256` is the canonical identity of the explicit engine's allowlisted Python sources plus selected current-platform native binary. If the agent directory contains its own `cg/`, compare only any allowlisted Python sources it actually vendors against the explicit engine; an API-only vendored package is valid, but any conflicting source fails. Ignore agent-vendored native binaries for runtime identity and always bind/load the native binary from explicit `cg_dir`. Reject symlinks, hard links, devices, unexpected nested directories, and mutable files by reusing the hardened descriptor-bound hashing rules.

- [ ] Write failing tests proving two copied agents with identical policy and deck collapse to one family; same policy/different deck stays one policy family but has different artifact identity; different policy/same deck creates two families; no file inside an agent directory is silently excluded as an attestation; the external sibling attestation has an independently verified hash; an API-source-only vendored `cg/` is accepted when its present sources match while any source mismatch fails; agent-vendored native files never replace the explicit runtime native identity; an opponent zoo rejects duplicate `(policy_sha256, deck_multiset_sha256)` pairs; and `validate_one_axis` enforces the inverse freeze contracts.

- [ ] Implement canonical identities and family IDs as full SHA-256 values with domain separation. Do not use names, archetype labels, replay fingerprints, or ratings as identity.

- [ ] Implement the inventory CLI with repeated `--agent label=path`, required `--cg-dir`, and `--out`. Output canonical JSON sorted by family ID and member artifact hash. Store logical labels and identities separately from a `locations` map; locations are runtime resolution metadata and are excluded from identity hashes. Include eligibility fields:

```json
{
  "policy_contract": "missing",
  "same_deck_eligible": false,
  "reason": "deck-specific policy contract and fixtures are not yet frozen"
}
```

Existing Dragapult templates remain inventory candidates, not faithful policies, until the later target-policy plan supplies that contract.

- [ ] Run the inventory against at most three code-hash-distinct Dragapult families from the read-only current project and the recovered v168 measurement control. Store the resulting inventory outside Git under `artifacts/inventory/`; record its SHA-256 in the task notes.

```bash
.venv312/bin/python scripts/inventory_policy_families.py \
  --agent "phantom=$PTCG_SOURCE_ROOT/agents/public_phantom_dive" \
  --agent "tempo=$PTCG_SOURCE_ROOT/agents/public_seokjeongeum_top_dragapult_ex_tempo_control_agent" \
  --agent "advanced=$PTCG_SOURCE_ROOT/agents/public_rauffauzanrambe_pokemon_ai_battle_best_ptcg_advanced" \
  --agent "archaludon-v168-source=$PTCG_SOURCE_ROOT/agents/baselines/v168_current_meta_recovered" \
  --cg-dir "$PTCG_CG_DIR" \
  --out artifacts/inventory/initial-policy-families.json
```

Expected: duplicates are reported, no more than three distinct Dragapult families remain, and none is marked faithful merely because its deck was replaced. This inventory identifies the read-only Archaludon source only; it is never used as the executed measurement-control identity after Task 12 materializes the tracked copy.

- [ ] Run tests and commit code/tests only.

```bash
.venv312/bin/python -m pytest tests/test_agent_identity.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/agent_identity.py \
  pokemon-tcg-ai-battle/scripts/inventory_policy_families.py \
  pokemon-tcg-ai-battle/tests/test_agent_identity.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: inventory code-and-deck-distinct policy families"
```

## Task 7: Build Episode-Persistent, Timeout-Attributable Policy Workers

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/policy_worker.py`
- Create: `pokemon-tcg-ai-battle/tests/test_policy_worker.py`

**Interfaces:**

```text
class WorkerFailureKind(StrEnum):
    IMPORT_ERROR = "import_error"
    DECK_ERROR = "deck_error"
    INVALID_ACTION = "invalid_action"
    AGENT_EXCEPTION = "agent_exception"
    AGENT_TIMEOUT = "agent_timeout"
    PROTOCOL_ERROR = "protocol_error"

@dataclass(frozen=True, slots=True)
class WorkerReply:
    action: tuple[int, ...] | None
    deck: tuple[int, ...] | None
    failure: WorkerFailureKind | None
    detail: str
    search_audit: dict[str, object] | None

class PolicyClient:
    __init__(agent_dir: Path, label: str, cg_dir: Path, timeout: float)
    start() -> None
    request_deck() -> WorkerReply
    act(observation: dict[str, object]) -> WorkerReply
    close() -> None
```

- [ ] Write fixture agents inside `tmp_path` for: persistent counter state, import error, malformed deck, exception on the second action, invalid action, and infinite loop. Tests must prove one worker persists state across an episode, a fresh worker resets it, timeout kills only the responsible worker, and every failure has exactly one owner.

- [ ] Implement a `multiprocessing` child with a small explicit protocol: `deck`, `act`, `close`. Load the explicitly requested `cg` package before importing the agent so `from cg.api import ...` cannot bind a different vendored engine. Import the agent once, preserve its module state, change working directory only around calls, and return JSON-serializable replies. The parent validates actions against the supplied selection before accepting them.

- [ ] Use `multiprocessing.get_context("spawn")` on every platform. Do not fork a process after the native engine has initialized; Mac and Linux must exercise the same worker lifecycle.

- [ ] Do not implement random fallback. A failed call returns a typed failure; the caller assigns the loss or `HOLD` according to the experiment contract.

- [ ] Run tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_policy_worker.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/policy_worker.py \
  pokemon-tcg-ai-battle/tests/test_policy_worker.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: isolate episode-persistent policy workers"
```

## Task 8: Define and Generate Balanced Randomized Full-Game Manifests

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/full_game.py`
- Create: `pokemon-tcg-ai-battle/scripts/create_randomized_full_game_manifest.py`
- Create: `pokemon-tcg-ai-battle/tests/test_full_game.py`

**Interfaces:**

```text
class ExperimentKind(StrEnum):
    POLICY = "policy"
    DECK = "deck"
    ARTIFACT = "artifact"

@dataclass(frozen=True, slots=True)
class GameAssignment:
    game_id: str
    block_id: str
    batch: int
    order: int
    arm: str
    opponent_label: str
    opponent_artifact_sha256: str
    seat: int

@dataclass(frozen=True, slots=True)
class PowerSpec:
    development_dataset_sha256: str
    method: str
    alpha: float
    power: float
    minimum_detectable_effect: float
    required_games_per_arm: int
    chosen_games_per_arm: int

@dataclass(frozen=True, slots=True)
class AnalysisSpec:
    confidence: float
    resamples: int
    weighting: Literal["equal_assignment"]
    minimum_opponents: int
    minimum_games_per_arm: int
    sufficient_stratum_games_per_arm: int
    minimum_positive_strata_fraction: float
    maximum_stratum_regression: float

@dataclass(frozen=True, slots=True)
class Contrast:
    name: str
    candidate_arm: str
    control_arm: str

@dataclass(frozen=True, slots=True)
class FullGameManifest:
    schema_version: int
    experiment_id: str
    experiment_kind: ExperimentKind
    evidence_kind: EvidenceKind
    scheduler_seed: int
    analysis_seed: int
    arms: tuple[AgentIdentity, ...]
    contrasts: tuple[Contrast, ...]
    opponents: tuple[AgentIdentity, ...]
    diagnostic: bool
    power_spec: PowerSpec | None
    analysis_spec: AnalysisSpec
    games_per_arm_per_opponent_seat_batch: int
    analysis_command: tuple[str, ...]
    assignments: tuple[GameAssignment, ...]

build_balanced_manifest(arms: Sequence[AgentIdentity], contrasts: Sequence[Contrast], opponents: Sequence[AgentIdentity], experiment_id: str, experiment_kind: ExperimentKind, batches: int, games_per_arm_per_opponent_seat_batch: int, scheduler_seed: int, analysis_seed: int, diagnostic: bool, power_spec: PowerSpec | None, analysis_spec: AnalysisSpec) -> FullGameManifest
validate_full_game_manifest(manifest: FullGameManifest) -> None
```

- [ ] Write failing tests for deterministic allocation, a changed scheduler seed changing order but not counts, equal allocation for every arm inside every `opponent × seat × batch` block, unique immutable IDs, both seats, batch-monotone global `order`, code-and-deck opponent deduplication, valid named contrasts, fixed total N, one-axis validation, and rejection of a hand-edited or missing assignment. Include a three-arm artifact manifest test so later Gate E can compare all finalists against common opponents/blocks without changing this schema. A production manifest must bind a development-data hash and power calculation with chosen N at least the calculated N and at least 2,000 games per arm; only an explicit diagnostic manifest may omit it and such a manifest is permanently non-promotable.

- [ ] Generate game IDs from the canonical experiment fields and ordinal using SHA-256; do not derive them from outcomes or filesystem paths. Allocation is explicitly batch-major: for each ascending batch, construct all balanced assignments across every opponent/seat block, use the single `random.Random(scheduler_seed)` stream to shuffle that batch's complete assignment list, append it, then assign global `order`. Never interleave a later batch ahead of an earlier one. The runner consumes this order without re-sorting, so finishing batch `n` before dispatching `n+1` is consistent with the manifest.

- [ ] Implement strict canonical JSON serialization. The CLI accepts identities from the Task 6 inventory, never reinterprets names as hashes, and writes atomically. Derive `analysis_command` from one constant template owned by the manifest CLI—`analyze_randomized_full_games.py --manifest {manifest} --ledger {ledger} --out {report}`—and provide no flag or API argument that can override it. The CLI prints counts and hashes only; no outcomes exist at this stage.

- [ ] Build a separate diagnostic inventory from four simple external agents, then run a small manifest-only artifact smoke with two arms and two distinct opponents. Inspect balance using the validator, not an ad-hoc spreadsheet.

```bash
.venv312/bin/python scripts/inventory_policy_families.py \
  --agent "control=$PTCG_SOURCE_ROOT/agents/rule_starter" \
  --agent "candidate=$PTCG_SOURCE_ROOT/agents/sample_lucario" \
  --agent "opponent-a=$PTCG_SOURCE_ROOT/agents/sample_dragapult" \
  --agent "opponent-b=$PTCG_SOURCE_ROOT/agents/sample_iono" \
  --cg-dir "$PTCG_CG_DIR" \
  --out artifacts/inventory/diagnostic.json

.venv312/bin/python scripts/create_randomized_full_game_manifest.py \
  --experiment-id diagnostic-runner-v1 \
  --kind artifact \
  --inventory artifacts/inventory/diagnostic.json \
  --arm control \
  --arm candidate \
  --contrast candidate-v-control=candidate,control \
  --opponent opponent-a \
  --opponent opponent-b \
  --batches 2 \
  --games-per-arm-per-opponent-seat-batch 1 \
  --scheduler-seed 20260715 \
  --analysis-seed 9152026 \
  --diagnostic \
  --out artifacts/manifests/diagnostic-runner-v1.json
```

Expected: `2 arms × 2 opponents × 2 seats × 2 batches = 16` assignments and perfect within-block balance.

- [ ] Run tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_full_game.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/full_game.py \
  pokemon-tcg-ai-battle/scripts/create_randomized_full_game_manifest.py \
  pokemon-tcg-ai-battle/tests/test_full_game.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: preregister balanced randomized full-game assignments"
```

## Task 9: Execute Immutable Assignments into an ITT Ledger

**Files:**

- Modify: `pokemon-tcg-ai-battle/src/ptcg_hybrid/full_game.py`
- Modify: `pokemon-tcg-ai-battle/scripts/run_local_match.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_randomized_full_games.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_full_game.py`

**Interfaces:**

```text
class OutcomeStatus(StrEnum):
    TERMINAL = "terminal"
    AGENT_FAILURE = "agent_failure"
    INFRASTRUCTURE_INTERRUPTION = "infrastructure_interruption"
    AMBIGUOUS = "ambiguous"

@dataclass(frozen=True, slots=True)
class GameOutcome:
    game_id: str
    assignment_sha256: str
    arm: str
    opponent_artifact_sha256: str
    seat: int
    batch: int
    attempt: int
    status: OutcomeStatus
    assigned_value: float | None
    terminal_result: int | None
    failure_owner: Literal["assigned", "opponent", "infrastructure", "ambiguous"] | None
    failure_kind: str | None
    decisions: int
    started_at: str
    finished_at: str

reconcile_ledger(manifest: FullGameManifest, rows: Sequence[GameOutcome]) -> dict[str, GameOutcome]
```

- [ ] Add failing tests that prove: terminal win/draw/loss maps to `1/0.5/0`; assigned-agent crash/invalid action/timeout maps to `0`; opponent failure maps to `1`; infrastructure interruption has no value and may be retried only with the same ID; a second conflicting terminal row is `AMBIGUOUS`; duplicate successful rows, unknown IDs, changed arm/seat/opponent, skipped assignments, and selective deletion force `HOLD`.

- [ ] Refactor `run_local_match.py` to use two `PolicyClient` workers while the controller owns the official engine. One fresh pair of workers is created per game. Return a structured result with failure ownership; keep the old sentinel CLI format as a backward-compatible adapter for unpaired diagnostics.

- [ ] Implement the randomized runner as a pure manifest consumer. It accepts repeated `--agent-dir label=path` runtime resolutions, rejects missing/extra labels, validates every resolved artifact against the manifest identity before starting, executes only missing or infrastructure-interrupted assignments, writes append-only canonical JSONL, flushes and `fsync`s every row, and prints only progress counts such as `completed=8/16`; it must not print arm win rates or treatment effects. This is **procedurally sealed**, not cryptographically hidden: workers never render outcomes, only the single coordinator owns the raw ledger during execution, the coordinator atomically publishes a readable aggregate only after fixed-N reconciliation, and the analyzer refuses every partial ledger. Document that filesystem administrators could still inspect raw bytes; the integrity guarantee comes from the preregistered manifest and complete ITT reconciliation.

- [ ] Add `--jobs`, but have each job execute a complete game in its own process. The coordinator alone appends the ledger so rows cannot interleave. Preserve the manifest's randomized order when dispatching. Finish and reconcile every assignment in execution batch `n` before dispatching batch `n+1`; parallelism is allowed only within a batch so the preregistered batch block remains meaningful.

- [ ] Run unit tests, then a 16-game diagnostic engine smoke using simple local agents. This is a runner check, not promotion evidence.

```bash
.venv312/bin/python scripts/run_randomized_full_games.py \
  --manifest artifacts/manifests/diagnostic-runner-v1.json \
  --cg-dir "$PTCG_CG_DIR" \
  --agent-dir "control=$PTCG_SOURCE_ROOT/agents/rule_starter" \
  --agent-dir "candidate=$PTCG_SOURCE_ROOT/agents/sample_lucario" \
  --agent-dir "opponent-a=$PTCG_SOURCE_ROOT/agents/sample_dragapult" \
  --agent-dir "opponent-b=$PTCG_SOURCE_ROOT/agents/sample_iono" \
  --ledger artifacts/outcomes/diagnostic-runner-v1.jsonl \
  --jobs 2 \
  --agent-timeout 10 \
  --game-timeout 180
```

Expected: all 16 immutable IDs have a final ITT row, both seats run, and no aggregate result is printed. Any native/engine interruption is retained and rerun under the same ID; any ambiguity stops analysis.

- [ ] Run all tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_policy_worker.py tests/test_full_game.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/full_game.py \
  pokemon-tcg-ai-battle/scripts/run_local_match.py \
  pokemon-tcg-ai-battle/scripts/run_randomized_full_games.py \
  pokemon-tcg-ai-battle/tests/test_full_game.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: record full-game outcomes in an immutable ITT ledger"
```

## Task 10: Analyze Complete Randomized Blocks and Close Phase I

**Files:**

- Modify: `pokemon-tcg-ai-battle/src/ptcg_hybrid/full_game.py`
- Create: `pokemon-tcg-ai-battle/scripts/analyze_randomized_full_games.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_full_game.py`
- Modify: `pokemon-tcg-ai-battle/README.md`

**Interfaces:**

```text
@dataclass(frozen=True, slots=True)
class RandomizedEffect:
    contrast: str
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    resamples: int
    seed: int
    by_opponent_seat: dict[str, float]

block_randomization_effect(manifest: FullGameManifest, outcomes: Mapping[str, GameOutcome], contrast: Contrast) -> RandomizedEffect
randomized_gate(manifest: FullGameManifest, outcomes: Mapping[str, GameOutcome], contrast: Contrast, effect: RandomizedEffect) -> Decision
```

- [ ] Add failing tests for the weighted mean of each named within-block candidate-minus-control contrast, deterministic balanced-label permutations, antisymmetry when contrast arms swap, common-manifest three-arm contrasts, per-opponent/seat strata, incomplete fixed N, fewer than 20 distinct opponents, fewer than 2,000 games per arm, a >5-point sufficiently sampled regression, and a manifest with `diagnostic: true` never returning `PASS`. Prove there is no analyzer `--diagnostic` flag and no runtime `resamples`/`diagnostic` function argument.

- [ ] Implement the interval from the centered balanced-label randomization distribution: within every block, permute labels while preserving its assigned arm counts; calculate the treatment estimate for each permutation; return `estimate - q97.5(permuted)` and `estimate - q2.5(permuted)`. Read confidence, resample count, weighting, and thresholds only from `manifest.analysis_spec`, and use only `manifest.analysis_seed` with stable quantile interpolation. The analyzer recomputes and verifies the constant `analysis_command` template and refuses a manifest containing any alternate command.

- [ ] Implement Gate C2-shaped validation without running Gate C2: a production `PASS` requires the spec's fixed N, at least 20 distinct opponent policies, positive lower bound, positive point estimate in at least 70% of sufficiently sampled opponent-seat strata, no sufficiently sampled regression worse than 0.05, and zero selective loss. An `artifact` experiment can report comparisons but cannot attribute/promote deck or policy.

- [ ] Serialize the report through an `AnalyzerAttestation` with analyzer ID `randomized_block_v1`, exact manifest/report-payload hashes, reconciled completeness, and the analyzer's proposed decision. Only `require_promotable(attestation)` may write the final decision; a hand-authored report or generic evidence-kind string cannot promote.

- [ ] Analyze the 16-game smoke in diagnostic mode.

```bash
.venv312/bin/python scripts/analyze_randomized_full_games.py \
  --manifest artifacts/manifests/diagnostic-runner-v1.json \
  --ledger artifacts/outcomes/diagnostic-runner-v1.jsonl \
  --out artifacts/reports/diagnostic-runner-v1.json
```

Expected: a deterministic report with evidence kind `randomized_block_full_game`, complete balance/ITT checks, decision `HOLD`, and reason code `HOLD_DIAGNOSTIC`, never `PASS`.

- [ ] Update README commands and explicitly state that old pool win rates, action agreement, and `imap_unordered` execution are diagnostic only.

- [ ] Run all tests. Phase I earns only `FOUNDATION_PASS` if the suite, both exact-deck engine validations, manifest balance test, runner smoke, and diagnostic analysis all pass. Record full Gate A as `HOLD_POLICY_CONTRACT` and `PACKAGE_NOT_BUILT`; neither condition is waived by foundation success.

```bash
.venv312/bin/python -m pytest -q -p no:cacheprovider
```

- [ ] Commit.

```bash
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/full_game.py \
  pokemon-tcg-ai-battle/scripts/analyze_randomized_full_games.py \
  pokemon-tcg-ai-battle/tests/test_full_game.py \
  pokemon-tcg-ai-battle/README.md
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: add fail-closed block-randomization analysis"
```

---

## Task 11: Define Immutable Decision Roots and Seed Namespaces

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/root_schema.py`
- Create: `pokemon-tcg-ai-battle/tests/test_root_schema.py`

**Interfaces:**

```text
class SeedNamespace(StrEnum):
    INNER = "inner"
    OUTER = "outer"

@dataclass(frozen=True, slots=True)
class Determinization:
    namespace: SeedNamespace
    seed: int
    determinization_id: str
    root_id: str
    root_sha256: str
    deck_sha256: str
    engine_state_sha256: str
    ordinal: int
    determinization_sha256: str
    your_deck: tuple[int, ...]
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...]
    manual_coin: bool
    continuation_family: str
    continuation_sha256: str
    coin_tape_id: str
    coin_tape: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class PrefixFrame:
    ordinal: int
    observation: dict[str, object]
    observation_sha256: str

@dataclass(frozen=True, slots=True)
class DecisionRoot:
    schema_version: int
    root_id: str
    deck_namespace: str
    source_episode_id: str
    step: int
    seat: int
    turn: int
    archetype: str
    collection_batch: str
    raw_observation: dict[str, object]
    active_prefix: tuple[PrefixFrame, ...]
    legal_actions: tuple[tuple[int, ...], ...]
    semantic_actions: tuple[str, ...]
    agent_sha256: str
    deck_sha256: str
    opponent_sha256: str
    cg_api_sha256: str
    native_sha256: str
    platform: str
    engine_state_sha256: str
    diagnostic_only: bool

@dataclass(frozen=True, slots=True)
class RootManifest:
    schema_version: int
    engine_identity_sha256: str
    measurement_control_artifact_sha256: str
    control_attestation_sha256: str | None
    roots: tuple[DecisionRoot, ...]
    manifest_sha256: str

@dataclass(frozen=True, slots=True)
class DeterminizationManifest:
    schema_version: int
    namespace: SeedNamespace
    allocation_seed: int
    root_manifest_sha256: str
    root_capture_report_sha256: str
    root_selection_manifest_sha256: str | None
    control_attestation_sha256: str | None
    action_manifest_sha256: str | None
    determinizations: tuple[Determinization, ...]
    manifest_sha256: str

derive_seed(namespace: SeedNamespace, allocation_seed: int, root_id: str, ordinal: int) -> int
validate_root(root: DecisionRoot) -> None
validate_root_manifest(manifest: RootManifest) -> None
validate_determinization_manifest(root_manifest: RootManifest, determinization_manifest: DeterminizationManifest) -> None
validate_disjoint_namespaces(inner: Sequence[Determinization], outer: Sequence[Determinization]) -> None
```

- [ ] Write failing tests for missing/blank `search_begin_input`, nonchronological/missing active prefix, wrong seat, forced or multi-choice root admission, duplicate root IDs, invalid hidden-list lengths, `manual_coin=False`, duplicate determinization IDs, seed reuse across namespaces, deck-hash crossing, orphaned root IDs, root/deck/engine-state hash crossing, noncontiguous or reused ordinals, missing capture-report binding, tampered determinization hashes, and deterministic domain-separated seed derivation where changing the recorded allocation seed changes outputs.

- [ ] Implement canonical JSON hashing for roots and determinizations. `root_sha256` is the domain-separated canonical hash of the complete referenced root; `determinization_id` is derived only from namespace/root/ordinal; `determinization_sha256` is the canonical hash of every determinization field except itself. A determinization is an input state, not a comparison pair: `pair_id` is created later from a frozen calibration role and action assignment. `validate_root_manifest()` recomputes every root and manifest hash. `validate_determinization_manifest()` requires every determinization to reference exactly one root in the bound manifest and to repeat that root's `root_id`, `root_sha256`, `deck_sha256`, and `engine_state_sha256` exactly. Roots with no usable search input may be serialized only with `diagnostic_only: true`; the paired evaluator must reject them.

- [ ] Keep multiple roots linked by `source_episode_id`. Statistical code must never infer independent samples from distinct root IDs in the same episode.

- [ ] Run tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_root_schema.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/root_schema.py \
  pokemon-tcg-ai-battle/tests/test_root_schema.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: define hash-bound decision-root evidence"
```

## Task 12: Audit Real Search-Enabled Policy Calls and Capture Roots

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/search_audit.py`
- Modify: `pokemon-tcg-ai-battle/src/ptcg_hybrid/root_schema.py`
- Modify: `pokemon-tcg-ai-battle/src/ptcg_hybrid/policy_worker.py`
- Create: `pokemon-tcg-ai-battle/scripts/capture_decision_roots.py`
- Create: `pokemon-tcg-ai-battle/scripts/verify_measurement_control.py`
- Create: `pokemon-tcg-ai-battle/agents/controls/archaludon_v168_measurement/main.py`
- Create: `pokemon-tcg-ai-battle/agents/controls/archaludon_v168_measurement/deck.csv`
- Create: `pokemon-tcg-ai-battle/agents/controls/archaludon_v168_measurement.control.json`
- Create: `pokemon-tcg-ai-battle/tests/test_search_audit.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_root_schema.py`
- Create: `pokemon-tcg-ai-battle/tests/test_measurement_control.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_policy_worker.py`

**Interfaces:**

```text
@dataclass(frozen=True, slots=True)
class SearchAudit:
    available: bool
    eligible: bool
    status: Literal["triggered", "not_triggered", "error"]
    begin_calls: int
    step_calls: int
    end_calls: int
    exception: str | None
    action_source: str

class SearchDelegate:
    install(module: ModuleType) -> None
    snapshot() -> SearchAudit

@dataclass(frozen=True, slots=True)
class SearchAuditRecord:
    root_id: str
    root_sha256: str
    action_sha256: str
    measurement_control_artifact_sha256: str
    control_attestation_sha256: str
    audit: SearchAudit
    record_sha256: str

@dataclass(frozen=True, slots=True)
class SearchAuditManifest:
    schema_version: int
    root_manifest_sha256: str
    measurement_control_artifact_sha256: str
    control_attestation_sha256: str
    records: tuple[SearchAuditRecord, ...]
    manifest_sha256: str

@dataclass(frozen=True, slots=True)
class CalibrationAction:
    root_id: str
    root_sha256: str
    kind: Literal["control", "identity", "positive", "negative"]
    action: tuple[int, ...]
    action_sha256: str
    search_audit_record_sha256: str

@dataclass(frozen=True, slots=True)
class ActionManifest:
    schema_version: int
    root_manifest_sha256: str
    measurement_control_artifact_sha256: str
    control_attestation_sha256: str
    search_audit_manifest_sha256: str
    actions: tuple[CalibrationAction, ...]
    manifest_sha256: str

class RootCaptureStatus(StrEnum):
    PASS = "PASS"
    HOLD = "HOLD"

@dataclass(frozen=True, slots=True)
class RootCaptureReport:
    schema_version: int
    status: RootCaptureStatus
    scheduler_seed: int
    max_games: int
    max_decisions: int
    observed_counts: dict[str, int]
    root_manifest_sha256: str
    action_manifest_sha256: str
    search_audit_manifest_sha256: str
    reason_codes: tuple[str, ...]
    report_sha256: str
```

- [ ] Write fake-agent tests proving the delegate wraps the module's already-bound `search_begin`, `search_step`, and `search_end`; an eligible trigger must record real begin and step calls; an ineligible/manual calibration action is `not_triggered`; an exception is `error`; and the worker returns the single action from the real final entry point without recalling the root. Add manifest tests proving every action references exactly one audit record with matching root/action/measurement hashes, no audit record is reused or orphaned, and a tampered/replayed audit hash fails.

- [ ] Extend `PolicyClient` with `replay_prefix_and_act(prefix, root)`. It starts from one deck call, replays that seat's active observations in chronological order, then invokes the root once. One worker is still exactly one `policy × episode`.

- [ ] Replace `WorkerReply.search_audit` with the forward annotation `"SearchAudit | None"` now that `search_audit.py` exists. Non-search calls return `None`; search-enabled calls return the typed immutable audit, never an unvalidated dictionary.

- [ ] Implement the capture CLI around real local games. It records non-forced, single-choice main-phase roots, full active prefixes, semantic legal actions, engine/card/agent hashes, and exact hidden lists derived from official engine visualization. With `--calibration`, it writes a pre-outcome `ActionManifest` containing the real search-enabled control action, any legal one-step immediate-winning action, and a verified inferior legal action, plus a separate `SearchAuditManifest`. Every `SearchAuditRecord` binds one root, one action hash, and the actual measurement-control artifact; manually defined calibration actions receive their own explicit `not_triggered` record and can never borrow the control action's triggered audit. After atomically finalizing those three outputs, write a hash-bound `RootCaptureReport` last. Downstream CLIs require report status `PASS` and exact referenced hashes; a budget-exhausted `HOLD` report preserves the partial evidence for diagnosis but cannot be used to allocate outer determinizations.

- [ ] Freeze recovered v168 before editing: run Task 6's identity tool on the current read-only directory, capture a golden prefix/action corpus outside Git, and record both hashes. Materialize only reviewed `main.py` and `deck.csv` under `agents/controls/archaludon_v168_measurement/`; never copy its `cg/`, caches, or unrelated files. The sibling `agents/controls/archaludon_v168_measurement.control.json` records the source identity, frozen identity, deck hash, golden-corpus hash, and explicitly states that no historical rating is inherited.

- [ ] If the search audit proves a successful path omits `search_end()`, make only the lifecycle refactor in the tracked measurement copy. `verify_measurement_control.py` must replay the frozen corpus through source and tracked copies in fresh workers and require byte-equal semantic actions at every root. Any action drift rejects the refactor; the source copy remains the control and calibration records the lifecycle failure as `HOLD`.

- [ ] After materialization and any accepted lifecycle-only refactor, rerun Task 6's identity tool on the tracked copy and write a dedicated inventory under `artifacts/inventory/measurement-control.json` with label `archaludon-v168-measurement`. Then finalize the non-executable sibling attestation with that post-refactor artifact hash and the frozen golden hash, and compute its independent `control_attestation_sha256`. This avoids self-reference because the sidecar is outside the agent directory. Statically reject any policy reference to the sidecar basename and have the verifier fail if the worker opens that exact path; after finalization, replay all 128 golden prefixes again before root capture. The sidecar hash is bound by every root/action/audit manifest, outer determinization, pair assignment, and calibration manifest/report.

- [ ] Capture a small canary in both seats and verify same-build cross-process replay returns the same control action and search-audit state. A mismatch stops Phase II with decision `HOLD` and reason code `HOLD_ROOT_PORTABILITY`.

```bash
.venv312/bin/python scripts/capture_decision_roots.py \
  --agent "$PTCG_SOURCE_ROOT/agents/baselines/v168_current_meta_recovered" \
  --opponent "$PTCG_SOURCE_ROOT/agents/sample_lucario" \
  --cg-dir "$PTCG_CG_DIR" \
  --both-seats \
  --golden-only \
  --scheduler-seed 2026071500 \
  --max-games 512 \
  --max-decisions 50000 \
  --golden-prefix-count 128 \
  --golden-out artifacts/calibration/archaludon-v168-golden.jsonl

GOLDEN_SHA256=$(shasum -a 256 artifacts/calibration/archaludon-v168-golden.jsonl | awk '{print $1}')
test "${#GOLDEN_SHA256}" -eq 64

.venv312/bin/python scripts/verify_measurement_control.py \
  --source-agent "$PTCG_SOURCE_ROOT/agents/baselines/v168_current_meta_recovered" \
  --frozen-agent agents/controls/archaludon_v168_measurement \
  --golden-prefixes artifacts/calibration/archaludon-v168-golden.jsonl \
  --expected-golden-sha256 "$GOLDEN_SHA256" \
  --cg-dir "$PTCG_CG_DIR" \
  --out artifacts/calibration/archaludon-v168-control-verification.json

.venv312/bin/python scripts/inventory_policy_families.py \
  --agent "archaludon-v168-measurement=$PTCG_WORKTREE_ROOT/pokemon-tcg-ai-battle/agents/controls/archaludon_v168_measurement" \
  --cg-dir "$PTCG_CG_DIR" \
  --out artifacts/inventory/measurement-control.json

CONTROL_ATTESTATION=agents/controls/archaludon_v168_measurement.control.json
test -f "$CONTROL_ATTESTATION"
CONTROL_ATTESTATION_SHA256=$(shasum -a 256 "$CONTROL_ATTESTATION" | awk '{print $1}')
test "${#CONTROL_ATTESTATION_SHA256}" -eq 64

.venv312/bin/python scripts/verify_measurement_control.py \
  --source-agent "$PTCG_SOURCE_ROOT/agents/baselines/v168_current_meta_recovered" \
  --frozen-agent agents/controls/archaludon_v168_measurement \
  --golden-prefixes artifacts/calibration/archaludon-v168-golden.jsonl \
  --expected-golden-sha256 "$GOLDEN_SHA256" \
  --forbid-read-path "$CONTROL_ATTESTATION" \
  --cg-dir "$PTCG_CG_DIR" \
  --out artifacts/calibration/archaludon-v168-post-attestation-verification.json

.venv312/bin/python scripts/capture_decision_roots.py \
  --agent agents/controls/archaludon_v168_measurement \
  --opponent "$PTCG_SOURCE_ROOT/agents/sample_lucario" \
  --inventory artifacts/inventory/measurement-control.json \
  --measurement-control-label archaludon-v168-measurement \
  --control-attestation "$CONTROL_ATTESTATION" \
  --cg-dir "$PTCG_CG_DIR" \
  --both-seats \
  --calibration \
  --scheduler-seed 2026071502 \
  --max-games 20000 \
  --max-decisions 2000000 \
  --minimum-positive-episodes 50 \
  --minimum-identity-episodes 50 \
  --minimum-duplicated-null-episodes 3000 \
  --roots-out artifacts/calibration/archaludon-v168-roots.jsonl \
  --actions-out artifacts/calibration/archaludon-v168-actions.json \
  --search-audits-out artifacts/calibration/archaludon-v168-search-audits.json \
  --report-out artifacts/calibration/archaludon-v168-root-capture-report.json
```

Expected: the pre-refactor golden corpus contains exactly 128 prefixes and its captured hash is frozen before any edit. The calibration capture produces at least 50 distinct positive-control episodes, 50 separate identity episodes, and 3,000 additional distinct duplicated-null episodes (60 experiments × 50 episodes), plus one-to-one audit records, all bound to the post-refactor `archaludon-v168-measurement` artifact identity. The scheduler seed freezes seat/opponent dispatch even though the native engine exposes no full RNG seed; content hashes bind the resulting evidence. The CLI always terminates at the fixed game/decision budget and writes a complete `HOLD` capture report on exhaustion, with `INSUFFICIENT_POSITIVE_CONTROL_ROOTS`, `INSUFFICIENT_IDENTITY_ROOTS`, and/or `INSUFFICIENT_DUPLICATED_NULL_ROOTS`; it never loops indefinitely or silently returns a partial success.

- [ ] Run tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_search_audit.py tests/test_root_schema.py tests/test_measurement_control.py tests/test_policy_worker.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/search_audit.py \
  pokemon-tcg-ai-battle/src/ptcg_hybrid/root_schema.py \
  pokemon-tcg-ai-battle/src/ptcg_hybrid/policy_worker.py \
  pokemon-tcg-ai-battle/scripts/capture_decision_roots.py \
  pokemon-tcg-ai-battle/scripts/verify_measurement_control.py \
  pokemon-tcg-ai-battle/agents/controls/archaludon_v168_measurement \
  pokemon-tcg-ai-battle/agents/controls/archaludon_v168_measurement.control.json \
  pokemon-tcg-ai-battle/tests/test_search_audit.py \
  pokemon-tcg-ai-battle/tests/test_root_schema.py \
  pokemon-tcg-ai-battle/tests/test_measurement_control.py \
  pokemon-tcg-ai-battle/tests/test_policy_worker.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: capture search-audited episode-persistent roots"
```

## Task 13: Implement One-Root Matched-Determinization Branching

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/continuations.py`
- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_evaluator.py`
- Create: `pokemon-tcg-ai-battle/tests/test_paired_evaluator.py`

**Interfaces:**

```text
class ContinuationAdapter(Protocol):
    name: str
    sha256: str

    choose(observation: object, coin_cursor: CoinCursor) -> tuple[int, ...]

class CalibrationFirstLegalV1:
    name: Literal["calibration_first_legal_v1"]
    sha256: str
    choose(observation: object, coin_cursor: CoinCursor) -> tuple[int, ...]

@dataclass(frozen=True, slots=True)
class BranchOutcome:
    action: tuple[int, ...]
    terminal_value: float | None
    truncated_value: float | None
    transitions: int
    complete: bool
    error: str | None

@dataclass(frozen=True, slots=True)
class MatchedPairOutcome:
    root_id: str
    pair_id: str
    episode_id: str
    archetype: str
    seat: int
    order: Literal["control_first", "candidate_first"]
    control: BranchOutcome
    candidate: BranchOutcome

@dataclass(frozen=True, slots=True)
class PairAssignment:
    pair_id: str
    assignment_sha256: str
    calibration_role: Literal["identity", "positive_negative", "duplicated_null", "target"]
    null_experiment_id: str | None
    canary: bool
    outer_manifest_sha256: str
    calibration_manifest_sha256: str | None
    control_attestation_sha256: str | None
    root_id: str
    root_sha256: str
    determinization_id: str
    determinization_sha256: str
    action_manifest_sha256: str
    control_action_sha256: str
    candidate_action_sha256: str
    our_adapter_sha256: str
    opponent_adapter_sha256: str
    order: Literal["control_first", "candidate_first"]

@dataclass(frozen=True, slots=True)
class PairAssignmentManifest:
    schema_version: int
    experiment_id: str
    root_manifest_sha256: str
    outer_manifest_sha256: str
    action_manifest_sha256: str
    calibration_manifest_sha256: str | None
    null_input_manifest_sha256: str | None
    control_attestation_sha256: str | None
    assignments: tuple[PairAssignment, ...]
    manifest_sha256: str

class PairLedgerStatus(StrEnum):
    TERMINAL = "terminal"
    ERROR = "error"
    INFRASTRUCTURE_INTERRUPTION = "infrastructure_interruption"
    AMBIGUOUS = "ambiguous"

@dataclass(frozen=True, slots=True)
class PairLedgerRow:
    pair_id: str
    assignment_sha256: str
    attempt: int
    status: PairLedgerStatus
    outcome: MatchedPairOutcome | None
    failure_kind: str | None
    failure_detail: str | None
    started_at: str
    finished_at: str

evaluate_pair(game_api: ModuleType, root: DecisionRoot, determinization: Determinization, control_action: tuple[int, ...], candidate_action: tuple[int, ...], our_adapter: ContinuationAdapter, opponent_adapter: ContinuationAdapter, assignment: PairAssignment, *, transition_budget: int) -> MatchedPairOutcome
reconcile_pair_ledger(manifest: PairAssignmentManifest, rows: Sequence[PairLedgerRow]) -> dict[str, PairLedgerRow]
```

- [ ] Build a fake branching engine and write failing tests proving exactly one `search_begin`, two children from the same root, identical ordered hidden lists, `manual_coin=True`, independent coin cursors both starting at zero, assignment-bound alternating initial branch order, one transition per live branch per round, equal transition budgets, terminal draw retained as `0.5`, truncated value kept separately, and `search_end()` in `finally` on every path. Add ledger tests rejecting missing, unknown, duplicate, hash-mismatched, out-of-order retry, and conflicting rows.

- [ ] Implement `CalibrationFirstLegalV1` as a calibration-only adapter: consume the next immutable tape value for an engine coin selection; otherwise return the lexicographically first legal selection satisfying `minCount/maxCount`. It must import no `search_*` symbol and must be labeled ineligible for target-deck promotion evidence.

- [ ] Add a guard adapter whose `choose()` raises if any `search_*` function is touched. This proves continuation adapters are explicitly search-free and cannot recursively call the shipped final agent.

- [ ] Implement `CoinCursor` over an immutable tuple of `0/1` values. A tape exhaustion or malformed coin selection makes the pair incomplete; it never samples a new random value.

- [ ] Preserve both branches when one fails. Do not calculate a terminal-only delta here; downstream sensitivity analysis owns incomplete evidence.

- [ ] Derive each `pair_id` from the experiment ID, role, optional null-experiment ID, determinization ID, action hashes, adapter hashes, and order; a `Determinization` never supplies its own pair ID. Do not include the later calibration-manifest hash in `pair_id`, which lets the pre-outcome calibration design list exact pair IDs without a content-hash cycle. After that design freezes, canonically hash each `PairAssignment` from its complete payload—including the calibration-manifest hash but excluding `assignment_sha256`; require every determinization in the bound pre-registered root-selection/outer manifest to occur in exactly one assignment, while roots explicitly excluded before outer allocation have no determinization at all. The negative-control statistic is the declared arm reversal of the same `positive_negative` pair rather than a second execution. Validate all root/determinization/action/adapter/order and outer/calibration-manifest bindings before starting a worker. The coordinator alone appends `PairLedgerRow` records and calls `reconcile_pair_ledger()`. A worker crash, timeout, malformed branch, or search failure produces a final `ERROR` row and therefore downstream `HOLD`; it is never silently dropped or rerun. An `ERROR` row retains any partial `MatchedPairOutcome`, including the successful sibling branch, so later sensitivity summaries do not selectively discard it even though calibration cannot pass. Only `INFRASTRUCTURE_INTERRUPTION` may advance `attempt` under the same pair/assignment hashes. Once a pair has `TERMINAL`, `ERROR`, or `AMBIGUOUS` status, any later attempt is a reconciliation failure and the experiment remains `HOLD`.

- [ ] Run tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_paired_evaluator.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_evaluator.py \
  pokemon-tcg-ai-battle/src/ptcg_hybrid/continuations.py \
  pokemon-tcg-ai-battle/tests/test_paired_evaluator.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: evaluate actions with matched official-engine branches"
```

## Task 14: Upgrade Paired Statistics to Episode Clusters and Incomplete Bounds

**Files:**

- Modify: `pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_stats.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_paired_stats.py`

**Interfaces:**

```text
@dataclass(frozen=True, slots=True)
class RootEffect:
    episode_id: str
    root_id: str
    stratum: str
    complete_deltas: tuple[float, ...]
    candidate_incomplete: int
    control_incomplete: int

@dataclass(frozen=True, slots=True)
class SensitivityEstimate:
    observed: EffectEstimate
    worst_case: EffectEstimate
    best_case: EffectEstimate
    completion_by_action: dict[str, float]

episode_clustered_bootstrap(roots: Sequence[RootEffect], *, confidence: float, resamples: int, seed: int) -> SensitivityEstimate
```

- [ ] Add failing tests that average determinizations within root, keep roots from one episode in one bootstrap cluster, stratify by `archetype|seat`, preserve draws/continuous values, produce antisymmetric estimates on arm swap, and assign incomplete candidate/control branches respectively to worst/best legal values for sensitivity bounds.

- [ ] Implement hierarchical aggregation in this order only: determinization to root, root to source episode cluster, then stratified episode bootstrap. Report effective episode and root counts separately.

- [ ] Keep the existing `stratified_paired_bootstrap()` compatibility function for old tests, but do not use it for Gate C1 or calibration.

- [ ] Run tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_paired_stats.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_stats.py \
  pokemon-tcg-ai-battle/tests/test_paired_stats.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: cluster paired action evidence by source episode"
```

## Task 15: Add the Outer-Namespace Evaluation CLI

**Files:**

- Create: `pokemon-tcg-ai-battle/scripts/create_outer_determinizations.py`
- Create: `pokemon-tcg-ai-battle/scripts/create_matched_pair_assignments.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_matched_root_evaluation.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_paired_evaluator.py`

**Interfaces:** The outer-allocation CLI consumes a frozen root manifest and frozen control/candidate action manifest, then writes an `OUTER` determinization manifest. A separate pre-outcome assignment CLI consumes the bound root/action/outer manifests, optional calibration manifest, and exact adapter identities and writes a `PairAssignmentManifest`. The evaluation CLI consumes that immutable assignment manifest and produces append-only `PairLedgerRow` records plus a reconciliation-only summary. No CLI generates actions, assignments, and outcomes from the same seed namespace or phase.

- [ ] Add CLI tests proving allocation happens after the action manifest freezes; a missing/`HOLD`/hash-mismatched root-capture report is rejected; the recorded allocation seed deterministically controls hidden-card permutations and coin tapes; every output namespace is `OUTER`; generic assignment creation rejects a missing experiment ID/role; assignments freeze before outcomes and bind every required hash; and evaluation rejects inner seeds, seed overlap, action files written after outer outcomes, deck-hash mismatch, missing root/prefix/search input, disabled search on an eligible root, non-distinct actions where a comparison is required, changed adapters, unknown/missing/duplicate/conflicting pair rows, and a partial output presented as complete.

- [ ] Implement `create_outer_determinizations.py` with `--roots`, `--actions`, required `--root-capture-report`, `--per-root`, `--seed`, `--continuation-family`, `--continuation-sha256`, and `--out`. Require the capture report status `PASS` and exact root/action hashes. Store `--seed` as `allocation_seed`, call `derive_seed(OUTER, allocation_seed, root_id, ordinal)`, permute only hidden cards consistent with the captured root, generate immutable SHA-256-derived coin tapes, and bind the report/root/action hashes. Refuse to overwrite an existing manifest.

- [ ] Implement `create_matched_pair_assignments.py` with `--roots`, `--actions`, `--outer-determinizations`, optional `--calibration-manifest`, exact `--our-adapter-sha256` and `--opponent-adapter-sha256`, and `--out`. Outside calibration mode it requires explicit `--experiment-id` and `--role identity|target`; in calibration mode Task 16 will derive both from the bound manifests and reject overrides. Store the experiment ID in `PairAssignmentManifest`, validate all currently available upstream manifests, freeze balanced branch order, write canonical official-engine assignments atomically, and refuse overwrite. Task 16 extends this CLI only after `NullInputManifest` exists.

- [ ] Implement the runner with `--roots`, `--actions`, `--outer-determinizations`, `--pair-assignments`, optional `--calibration-manifest`, `--cg-dir`, `--pairs-out`, `--summary-out`, `--transition-budget`, and `--jobs`. It rehashes every supplied artifact before opening the append-only ledger and never synthesizes or edits an assignment at execution time.

- [ ] Alternate branch order by determinization ordinal, isolate each root evaluation in a process because the native search arena is global, and have the coordinator append pair outcomes.

- [ ] Reconcile before rendering any summary. Missing, unknown, duplicate, conflicting, `ERROR`, or `AMBIGUOUS` rows make the summary incomplete with decision `HOLD`; terminal rows are immutable and never rerun. Only an infrastructure-interrupted row can be retried with the same assignment and next attempt number.

- [ ] Run tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_paired_evaluator.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/scripts/run_matched_root_evaluation.py \
  pokemon-tcg-ai-battle/scripts/create_outer_determinizations.py \
  pokemon-tcg-ai-battle/scripts/create_matched_pair_assignments.py \
  pokemon-tcg-ai-battle/tests/test_paired_evaluator.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: run frozen outer-namespace root evaluations"
```

## Task 16: Implement the Complete Gate B Calibration Harness

**Files:**

- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/calibration.py`
- Create: `pokemon-tcg-ai-battle/scripts/create_evaluator_calibration_manifest.py`
- Create: `pokemon-tcg-ai-battle/scripts/create_null_calibration_inputs.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_simulated_null_calibration.py`
- Create: `pokemon-tcg-ai-battle/scripts/create_calibration_root_selection.py`
- Modify: `pokemon-tcg-ai-battle/scripts/create_outer_determinizations.py`
- Modify: `pokemon-tcg-ai-battle/scripts/create_matched_pair_assignments.py`
- Modify: `pokemon-tcg-ai-battle/scripts/run_matched_root_evaluation.py`
- Modify: `pokemon-tcg-ai-battle/src/ptcg_hybrid/root_schema.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_root_schema.py`
- Modify: `pokemon-tcg-ai-battle/tests/test_paired_evaluator.py`
- Create: `pokemon-tcg-ai-battle/scripts/calibrate_paired_evaluator.py`
- Create: `pokemon-tcg-ai-battle/tests/test_calibration.py`

**Interfaces:**

```text
class CalibrationCheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"

@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    name: str
    status: CalibrationCheckStatus
    metrics: dict[str, float | int | str]
    reason: str

@dataclass(frozen=True, slots=True)
class CalibrationReport:
    evidence_kind: EvidenceKind
    decision: Decision
    checks: tuple[CalibrationCheck, ...]

@dataclass(frozen=True, slots=True)
class SelectedCalibrationRoot:
    root_id: str
    root_sha256: str
    source_episode_id: str
    role: Literal["identity", "positive_negative", "duplicated_null"]
    selection_ordinal: int

@dataclass(frozen=True, slots=True)
class ExcludedCalibrationRoot:
    root_id: str
    root_sha256: str
    source_episode_id: str
    reason_code: str

@dataclass(frozen=True, slots=True)
class CalibrationRootSelectionManifest:
    schema_version: int
    root_capture_report_sha256: str
    root_manifest_sha256: str
    action_manifest_sha256: str
    search_audit_manifest_sha256: str
    selection_seed: int
    selected: tuple[SelectedCalibrationRoot, ...]
    excluded: tuple[ExcludedCalibrationRoot, ...]
    manifest_sha256: str

@dataclass(frozen=True, slots=True)
class CalibrationAnalysisSpec:
    confidence: float
    branch_order_margin: float
    bootstrap_resamples: int
    calibration_analysis_seed: int
    roots_per_experiment: int
    determinizations_per_root: int
    null_false_positive_limit: float
    our_adapter_sha256: str
    opponent_adapter_sha256: str
    pair_order_seed: int
    null_model_id: Literal["dirichlet_categorical_episode_v1"]
    null_model_parameters: dict[str, object]
    null_model_golden_moments: dict[str, float]
    analyzer_spec_sha256: str

@dataclass(frozen=True, slots=True)
class NullExperimentSpec:
    experiment_id: str
    kind: Literal["simulated", "duplicated"]
    pair_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    allocation_seed: int
    analysis_seed: int
    roots_per_experiment: int
    determinizations_per_root: int
    analyzer_spec_sha256: str

@dataclass(frozen=True, slots=True)
class SimulatedNullAssignment:
    pair_id: str
    experiment_id: str
    episode_id: str
    synthetic_root_id: str
    ordinal: int
    control_action_sha256: str
    candidate_action_sha256: str
    our_adapter_sha256: str
    opponent_adapter_sha256: str
    order: Literal["control_first", "candidate_first"]
    assignment_sha256: str

@dataclass(frozen=True, slots=True)
class DuplicatedNullAssignment:
    pair_id: str
    experiment_id: str
    episode_id: str
    root_id: str
    root_sha256: str
    determinization_id: str
    determinization_sha256: str
    action_sha256: str
    our_adapter_sha256: str
    opponent_adapter_sha256: str
    order: Literal["control_first", "candidate_first"]
    assignment_sha256: str

@dataclass(frozen=True, slots=True)
class NullInputManifest:
    schema_version: int
    experiment_id: str
    root_selection_manifest_sha256: str
    analysis_spec: CalibrationAnalysisSpec
    simulated_assignments: tuple[SimulatedNullAssignment, ...]
    duplicated_assignments: tuple[DuplicatedNullAssignment, ...]
    experiments: tuple[NullExperimentSpec, ...]
    generator_sha256: str
    manifest_sha256: str

@dataclass(frozen=True, slots=True)
class SimulatedNullLedgerRow:
    pair_id: str
    assignment_sha256: str
    null_input_manifest_sha256: str
    calibration_manifest_sha256: str
    control_value: float
    candidate_value: float
    row_sha256: str

@dataclass(frozen=True, slots=True)
class CalibrationManifest:
    schema_version: int
    experiment_id: str
    created_before_outcomes: str
    engine_sha256: str
    measurement_control_artifact_sha256: str
    control_attestation_sha256: str
    root_capture_report_sha256: str
    root_selection_manifest_sha256: str
    root_manifest_sha256: str
    action_manifest_sha256: str
    outer_determinization_sha256: str
    search_audit_manifest_sha256: str
    null_input_manifest_sha256: str
    analyzer_spec_sha256: str
    our_adapter_sha256: str
    opponent_adapter_sha256: str
    pair_order_seed: int
    identity_pair_ids: tuple[str, ...]
    positive_pair_ids: tuple[str, ...]
    negative_pair_ids: tuple[str, ...]
    null_experiments: tuple[NullExperimentSpec, ...]
    branch_order_margin: float
    confidence: float

@dataclass(frozen=True, slots=True)
class CalibrationEvidenceBundle:
    manifest: CalibrationManifest
    root_capture_report: RootCaptureReport
    root_manifest: RootManifest
    root_selection_manifest: CalibrationRootSelectionManifest
    action_manifest: ActionManifest
    outer_manifest: DeterminizationManifest
    null_input_manifest: NullInputManifest
    pair_assignment_manifest: PairAssignmentManifest
    reconciled_pair_ledger: dict[str, PairLedgerRow]
    simulated_null_ledger: tuple[SimulatedNullLedgerRow, ...]
    search_audit_manifest: SearchAuditManifest
    engine_identity: EngineIdentity
    measurement_control_identity: AgentIdentity

calibrate_evaluator(bundle: CalibrationEvidenceBundle) -> CalibrationReport
```

- [ ] Write failing tests for every gate and exact threshold:

  - identity executes both branches and has exactly zero delta;
  - swapping arms negates the estimate;
  - the 95% branch-order interaction interval lies inside `[-0.005, +0.005]`;
  - at least 50 independent-episode positive-control roots have lower bound above zero;
  - reversed negative control has upper bound below zero;
  - at least 60 simulated and 60 duplicated null experiments each produce zero `PASS` results; the exact one-sided 95% binomial upper bound is below 0.05 both per kind and jointly;
  - inner/outer namespaces are disjoint and actions predate outer outcomes;
  - every eligible trigger has begin/step search calls; ineligible roots are `not_triggered`;
  - timeout, malformed action, absent search input, incomplete pair, or worker crash yields `HOLD`/`ERROR`, never `PASS`.

  Null tests also require exactly 50 episode clusters × 8 pair assignments in every experiment, the identical `CalibrationAnalysisSpec` used by the real positive-control analyzer, disjoint pair/episode membership, deterministic generation from frozen seeds, exact reconciliation of both null ledgers, and rejection of any reduced-sample null designed to make `PASS` artificially difficult. Positive, negative, identity, and branch-order intervals read `bootstrap_resamples` and domain-separated seeds derived from `calibration_analysis_seed`; no runtime seed or resample override exists.

- [ ] Implement `create_calibration_root_selection.py` before any outer determinization exists. It requires a `PASS` `RootCaptureReport`, freezes exactly one root per source episode using `selection_seed`, assigns 50 episodes to `identity`, 50 disjoint episodes to `positive_negative`, and 3,000 disjoint episodes to `duplicated_null`, and lists every remaining captured root exactly once in `excluded` with a stable reason code such as `EXTRA_ROOT_SAME_EPISODE`, `ROLE_QUOTA_FILLED`, or `DIAGNOSTIC_ONLY`. Selected plus excluded root IDs must exactly partition the capture's root manifest. A changed selection, missing exclusion, or outcome-aware input is an `ERROR`.

- [ ] Extend `create_outer_determinizations.py` with required `--root-selection` in calibration mode. Validate its capture report/root/action/audit bindings and generate 8 determinizations only for the 3,100 selected roots; bind both selection and capture-report hashes in `DeterminizationManifest`. No excluded or surplus root receives an outer determinization.

- [ ] Implement `create_null_calibration_inputs.py` as the pre-outcome null generator. It freezes both adapter SHA-256 values, one `pair_order_seed`, and one `CalibrationAnalysisSpec` before it derives any pair ID; all later assignment CLI values must equal this spec. It then creates exactly 60 simulated and 60 duplicated `NullExperimentSpec` records; 60 is the minimum because zero successes in only 50 trials has a one-sided 95% exact binomial upper bound above 0.05. Every experiment has 50 unique episode clusters and 8 unique pairs per episode, for 400 pairs per experiment. The duplicated specs partition 3,000 real selected null episodes and their 24,000 outer determinizations into explicit `DuplicatedNullAssignment` records without overlap with positive/identity episodes. The simulated specs create 3,000 synthetic episode IDs and 24,000 explicit `SimulatedNullAssignment` records. Pair IDs and episode IDs are disjoint across all specs and kinds; each spec has domain-separated allocation/analysis seeds and the same analyzer-spec hash. The generator source hash is bound, and reuse, overlap, reduced sample size, changed adapters/order, or post-outcome allocation is an `ERROR`.

- [ ] Freeze the nondegenerate simulated model exactly as `dirichlet_categorical_episode_v1`: for each episode draw category probabilities over outcome values `(0.0, 0.5, 1.0)` from `Dirichlet(alpha=(2.0, 1.0, 2.0))`, then draw control and candidate outcomes exchangeably and conditionally independently for each determinization. The canonical golden moments are mean `0.5`, draw mass `0.2`, marginal outcome variance `0.2`, episode covariance `1/30`, episode ICC `1/6`, and mean paired delta `0.0`. Unit tests require a fixed-seed golden ledger hash plus Monte Carlo moments within preregistered tolerances; a constant/equal-pair or zero-variance implementation fails.

- [ ] Implement `run_simulated_null_calibration.py` as a pure `NullInputManifest + CalibrationManifest` consumer. For each simulated assignment it uses only that spec's allocation seed to draw exchangeable zero-effect control/candidate outcomes from the frozen categorical null model, writes a complete canonical `SimulatedNullLedgerRow` set, and reveals no per-experiment result. Analysis later uses only the distinct analysis seed. It refuses overwrite, partial output, a changed generator hash, or a simulated pair ID not present in the frozen spec.

- [ ] Extend `create_matched_pair_assignments.py` with required `--null-inputs` in calibration mode. It derives `experiment_id` and every role from the calibration/null manifests, requires the supplied adapter hashes to equal `CalibrationAnalysisSpec`, reads branch order only from its frozen `pair_order_seed`, materializes exactly the duplicated-null pair IDs planned in `NullInputManifest`, binds its hash, and excludes simulated assignments from the official-engine manifest. Mark exactly the first preregistered identity assignment `canary: true`; all others are false. Tests prove every duplicated planned pair appears once, every simulated pair appears zero times, exactly one identity canary exists, and any later experiment/role/adapter/order change fails before output.

- [ ] In calibration mode, the runner executes the single manifest-marked identity canary first. Both branches must execute from the real official-engine root and produce exact zero delta. On success, the same invocation continues with the remaining immutable assignments and never reruns the canary; on failure it retains the canary ledger row, stops before other pairs, and emits `HOLD_ROOT_PORTABILITY`. This is a protocol-defined first assignment, not an outcome-selected subset.

- [ ] Implement the remaining pre-outcome calibration-manifest CLI. It verifies that each positive-control action terminates as an immediate win in a one-step engine probe, defines the negative control as the arm reversal of those same pairs, defines identity with two executions of the same legal action, consumes the frozen `NullInputManifest`, and copies its exact null specs and analysis thresholds. Freeze balanced branch order and every exact threshold shown above before any official or simulated outcome ledger is opened.

- [ ] Implement `CalibrationEvidenceBundle` validation before calculating a statistic: recompute every root/action/audit/outer/null-input/calibration/assignment/simulated-row hash; re-identify the explicit engine and measurement-control artifact supplied to the CLI; require each official assignment and simulated row to bind this calibration manifest; reconcile exact official and simulated pair membership; enforce the one-action/one-`SearchAuditRecord` mapping; reject missing, unknown, duplicate, conflicting, incomplete, or `ERROR` official rows; and validate all null independence/sample-size rules. The calibrator accepts no loose row sequences or runtime confidence override.

- [ ] Implement a single conjunction gate: all checks must have `CalibrationCheckStatus.PASS`. A scientific threshold miss is `FAIL`; malformed evidence, worker failure, hash mismatch, or evaluator exception is `ERROR`. Either status yields report decision `HOLD`; there is no `Decision.ERROR`. Do not average checks, waive a failed canary, or let Archaludon outcomes promote a target policy.

- [ ] Implement the CLI so every input file and engine/agent/adapter artifact is hash-bound in the report. Construct an `AnalyzerAttestation` with analyzer ID `matched_root_v1`, the calibration-manifest hash, canonical pre-attestation report-payload hash, completeness, and proposed decision, then call `require_promotable(attestation)` before serializing the final report envelope.

- [ ] Create the calibration manifest before outcomes, run synthetic tests, then run the real Archaludon measurement-control calibration. The real command uses at least 50 positive/negative episode roots and 120 null experiments; output stays outside Git.

```bash
CONTINUATION_SHA256=$(.venv312/bin/python -c 'from pathlib import Path; from ptcg_hybrid.manifest import sha256_path; print(sha256_path(Path("src/ptcg_hybrid/continuations.py")))')

.venv312/bin/python scripts/create_calibration_root_selection.py \
  --roots artifacts/calibration/archaludon-v168-roots.jsonl \
  --actions artifacts/calibration/archaludon-v168-actions.json \
  --search-audits artifacts/calibration/archaludon-v168-search-audits.json \
  --root-capture-report artifacts/calibration/archaludon-v168-root-capture-report.json \
  --identity-episodes 50 \
  --positive-episodes 50 \
  --duplicated-null-episodes 3000 \
  --selection-seed 2026071505 \
  --out artifacts/calibration/archaludon-v168-root-selection.json

.venv312/bin/python scripts/create_outer_determinizations.py \
  --roots artifacts/calibration/archaludon-v168-roots.jsonl \
  --actions artifacts/calibration/archaludon-v168-actions.json \
  --root-capture-report artifacts/calibration/archaludon-v168-root-capture-report.json \
  --root-selection artifacts/calibration/archaludon-v168-root-selection.json \
  --per-root 8 \
  --seed 2026071501 \
  --continuation-family calibration_first_legal_v1 \
  --continuation-sha256 "$CONTINUATION_SHA256" \
  --out artifacts/calibration/archaludon-v168-outer.json

.venv312/bin/python scripts/create_null_calibration_inputs.py \
  --experiment-id archaludon-v168-calibration-v1 \
  --roots artifacts/calibration/archaludon-v168-roots.jsonl \
  --actions artifacts/calibration/archaludon-v168-actions.json \
  --outer-determinizations artifacts/calibration/archaludon-v168-outer.json \
  --root-selection artifacts/calibration/archaludon-v168-root-selection.json \
  --simulated-experiments 60 \
  --duplicated-experiments 60 \
  --roots-per-experiment 50 \
  --determinizations-per-root 8 \
  --allocation-seed 2026071503 \
  --null-analysis-master-seed 2026071504 \
  --calibration-analysis-seed 2026071507 \
  --bootstrap-resamples 20000 \
  --confidence 0.95 \
  --branch-order-margin 0.005 \
  --null-false-positive-limit 0.05 \
  --our-adapter-sha256 "$CONTINUATION_SHA256" \
  --opponent-adapter-sha256 "$CONTINUATION_SHA256" \
  --pair-order-seed 2026071506 \
  --null-model-id dirichlet_categorical_episode_v1 \
  --out artifacts/calibration/archaludon-v168-null-inputs.json

.venv312/bin/python scripts/create_evaluator_calibration_manifest.py \
  --roots artifacts/calibration/archaludon-v168-roots.jsonl \
  --actions artifacts/calibration/archaludon-v168-actions.json \
  --outer-determinizations artifacts/calibration/archaludon-v168-outer.json \
  --root-capture-report artifacts/calibration/archaludon-v168-root-capture-report.json \
  --root-selection artifacts/calibration/archaludon-v168-root-selection.json \
  --cg-dir "$PTCG_CG_DIR" \
  --inventory artifacts/inventory/measurement-control.json \
  --measurement-control-label archaludon-v168-measurement \
  --control-attestation agents/controls/archaludon_v168_measurement.control.json \
  --search-audits artifacts/calibration/archaludon-v168-search-audits.json \
  --null-inputs artifacts/calibration/archaludon-v168-null-inputs.json \
  --out artifacts/calibration/archaludon-v168-manifest.json

.venv312/bin/python scripts/create_matched_pair_assignments.py \
  --roots artifacts/calibration/archaludon-v168-roots.jsonl \
  --actions artifacts/calibration/archaludon-v168-actions.json \
  --outer-determinizations artifacts/calibration/archaludon-v168-outer.json \
  --root-selection artifacts/calibration/archaludon-v168-root-selection.json \
  --calibration-manifest artifacts/calibration/archaludon-v168-manifest.json \
  --null-inputs artifacts/calibration/archaludon-v168-null-inputs.json \
  --our-adapter-sha256 "$CONTINUATION_SHA256" \
  --opponent-adapter-sha256 "$CONTINUATION_SHA256" \
  --out artifacts/calibration/archaludon-v168-pair-assignments.json

.venv312/bin/python scripts/run_simulated_null_calibration.py \
  --null-inputs artifacts/calibration/archaludon-v168-null-inputs.json \
  --calibration-manifest artifacts/calibration/archaludon-v168-manifest.json \
  --out artifacts/calibration/archaludon-v168-simulated-null.jsonl

.venv312/bin/python scripts/run_matched_root_evaluation.py \
  --roots artifacts/calibration/archaludon-v168-roots.jsonl \
  --actions artifacts/calibration/archaludon-v168-actions.json \
  --outer-determinizations artifacts/calibration/archaludon-v168-outer.json \
  --pair-assignments artifacts/calibration/archaludon-v168-pair-assignments.json \
  --calibration-manifest artifacts/calibration/archaludon-v168-manifest.json \
  --cg-dir "$PTCG_CG_DIR" \
  --pairs-out artifacts/calibration/archaludon-v168-pairs.jsonl \
  --summary-out artifacts/calibration/archaludon-v168-pair-summary.json \
  --transition-budget 2000 \
  --jobs 1
```

```bash
.venv312/bin/python scripts/calibrate_paired_evaluator.py \
  --roots artifacts/calibration/archaludon-v168-roots.jsonl \
  --actions artifacts/calibration/archaludon-v168-actions.json \
  --root-capture-report artifacts/calibration/archaludon-v168-root-capture-report.json \
  --root-selection artifacts/calibration/archaludon-v168-root-selection.json \
  --outer-determinizations artifacts/calibration/archaludon-v168-outer.json \
  --calibration-manifest artifacts/calibration/archaludon-v168-manifest.json \
  --pair-assignments artifacts/calibration/archaludon-v168-pair-assignments.json \
  --pairs artifacts/calibration/archaludon-v168-pairs.jsonl \
  --null-inputs artifacts/calibration/archaludon-v168-null-inputs.json \
  --simulated-null-ledger artifacts/calibration/archaludon-v168-simulated-null.jsonl \
  --search-audits artifacts/calibration/archaludon-v168-search-audits.json \
  --cg-dir "$PTCG_CG_DIR" \
  --inventory artifacts/inventory/measurement-control.json \
  --measurement-control-label archaludon-v168-measurement \
  --control-attestation agents/controls/archaludon_v168_measurement.control.json \
  --out artifacts/calibration/archaludon-v168-report.json
```

Expected: either a fully supported `PASS` or an attributable `HOLD` naming the failed check. A `HOLD` ends learned-policy work but still permits later deterministic exact-deck policy work.

- [ ] Run all tests and commit.

```bash
.venv312/bin/python -m pytest tests/test_calibration.py tests/test_root_schema.py tests/test_paired_evaluator.py -q
.venv312/bin/python -m pytest -q -p no:cacheprovider
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/src/ptcg_hybrid/calibration.py \
  pokemon-tcg-ai-battle/src/ptcg_hybrid/root_schema.py \
  pokemon-tcg-ai-battle/scripts/create_calibration_root_selection.py \
  pokemon-tcg-ai-battle/scripts/create_outer_determinizations.py \
  pokemon-tcg-ai-battle/scripts/create_null_calibration_inputs.py \
  pokemon-tcg-ai-battle/scripts/run_simulated_null_calibration.py \
  pokemon-tcg-ai-battle/scripts/create_matched_pair_assignments.py \
  pokemon-tcg-ai-battle/scripts/run_matched_root_evaluation.py \
  pokemon-tcg-ai-battle/scripts/create_evaluator_calibration_manifest.py \
  pokemon-tcg-ai-battle/scripts/calibrate_paired_evaluator.py \
  pokemon-tcg-ai-battle/tests/test_calibration.py \
  pokemon-tcg-ai-battle/tests/test_root_schema.py \
  pokemon-tcg-ai-battle/tests/test_paired_evaluator.py
git -C "$PTCG_WORKTREE_ROOT" commit -m "feat: enforce the matched-evaluator calibration gate"
```

## Task 17: Perform the Foundation Acceptance Audit

**Files:**

- Modify: `pokemon-tcg-ai-battle/README.md`
- Create: `pokemon-tcg-ai-battle/docs/evaluation-foundation-acceptance.md`
- Review: every file named in this plan.

**Interfaces:** The acceptance document records commands, Git revision, Python/platform, external data/engine hashes, exact-deck hashes, inventory hash, test result, runner smoke report hash, calibration report hash, Phase I `FOUNDATION_PASS`/`HOLD`, full Gate A hold reasons, and Phase II calibration `PASS`/`HOLD`. It contains no raw competition data or absolute credential paths.

- [ ] Run the clean active Python 3.12+ suite with bytecode/cache disabled, then run the same suite once under exact Python 3.12 on the second Mac, 1080 Ti host, or Colab. Record both interpreters and hashes.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m pytest -q -p no:cacheprovider
python3.12 -m venv .venv312-min
.venv312-min/bin/python -m pip install -e '.[dev]'
PYTHONDONTWRITEBYTECODE=1 .venv312-min/bin/python -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] Re-run both exact-deck validators and compare their output byte-for-byte with the committed reports.

- [ ] Re-run policy inventory twice and confirm canonical output hashes match. Confirm the Clefairy exact multiset contains `209` once and `1172` twice; reject the existing mutated candidate as baseline.

- [ ] Reconcile and reanalyze the already completed randomized 16-game diagnostic ledger using its frozen manifest; do not rerun any `TERMINAL` assignment. Confirm both seats, balanced blocks, no selective deletion, manifest-derived diagnostic decision `HOLD`, and reason code `HOLD_DIAGNOSTIC`. If a fresh runner smoke is genuinely required, create a new experiment ID, manifest, ledger, and report; never reuse `diagnostic-runner-v1` or its terminal IDs.

- [ ] Reconcile and reanalyze the existing full calibration ledger, including its manifest-marked first identity canary; do not rerun any pair with `TERMINAL`, `ERROR`, or `AMBIGUOUS` status. Infrastructure-only resumptions retain the same assignment hash and next attempt. If a new smoke is needed, preregister a new calibration experiment ID and new output paths. Record `PASS` or the exact `HOLD_*` reason without editing thresholds after seeing the result.

- [ ] Inspect the repository for forbidden artifacts and placeholders.

```bash
git -C "$PTCG_WORKTREE_ROOT" status --short
git -C "$PTCG_WORKTREE_ROOT" diff --check
PLACEHOLDER_PATTERN='/absolute'"/path"
PLACEHOLDER_WORDS='TO'"DO|TB"'D|FI'"XME"
CREDENTIAL_PATTERN='pokemon-tcg-ai-battle-transfer-with-credentials'".tar.gz"
SUBMISSION_PATTERN='kagglehub.*sub'"mit|competitions sub"'missions'
if rg -n "${PLACEHOLDER_WORDS}|${PLACEHOLDER_PATTERN}|${CREDENTIAL_PATTERN}|${SUBMISSION_PATTERN}" \
  "$PTCG_WORKTREE_ROOT/pokemon-tcg-ai-battle"; then exit 1; fi
if rg -n "${PLACEHOLDER_WORDS}|${PLACEHOLDER_PATTERN}" \
  "$PTCG_WORKTREE_ROOT/docs/superpowers/plans"; then exit 1; fi
```

Expected: both `rg` checks return no match. The plan document's explicit credential-archive safety wording is deliberately outside the second pattern and must not cause any filesystem action; the implementation tree may contain neither that path nor a submission call.

- [ ] Request a code review with `superpowers:requesting-code-review`. Resolve only evidence-backed findings and rerun the affected narrow tests plus the full suite. For every accepted code finding, explicitly enumerate only the affected tracked paths in `git -C "$PTCG_WORKTREE_ROOT" add ...` and create a separate `fix: resolve evaluation foundation review findings` commit before the acceptance-doc commit; never use `git add -A`, never include README/acceptance drafts in that fix commit, and never leave review code changes unstaged. Immediately before the final documentation commit, `git status --short` may list only `pokemon-tcg-ai-battle/README.md` and `pokemon-tcg-ai-battle/docs/evaluation-foundation-acceptance.md`.

- [ ] Commit acceptance documentation.

```bash
git -C "$PTCG_WORKTREE_ROOT" add pokemon-tcg-ai-battle/README.md \
  pokemon-tcg-ai-battle/docs/evaluation-foundation-acceptance.md
git -C "$PTCG_WORKTREE_ROOT" commit -m "docs: record PTCG evaluation foundation acceptance"
```

- [ ] Use `superpowers:verification-before-completion`, then report the phase decisions without overstating competition strength:

  - Phase I `FOUNDATION_PASS` means exact deck identity and prospective full-game measurement machinery are trustworthy; full Gate A remains `HOLD_POLICY_CONTRACT`/`PACKAGE_NOT_BUILT`, and neither deck is thereby strong.
  - Phase II calibration `PASS` means only the generic paired-evaluator instrument passed on the Archaludon measurement control. Dragapult/Clefairy artifact identity, antisymmetry, search, portability, and failure canaries remain mandatory in the later target-policy plan; no target deck or learned policy is promoted here.
  - Any `HOLD` names the failed contract and blocks only dependent work.
