# Pokémon TCG Evaluation Integrity Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tracked, tested evaluation foundation that labels current local full-game pools as unpaired diagnostics, prevents them from promoting candidates, calculates grouped paired-bootstrap intervals for future counterfactual evidence, provides a deterministic experiment-manifest writer, and rejects credential-bearing submission artifacts.

**Architecture:** A small `ptcg_hybrid` Python package owns evidence types, paired statistics, experiment hashing, and archive safety. Existing command-line scripts remain orchestration layers: they emit explicit evidence metadata and consume the package rather than embedding statistical policy. This is Plan 1 of four independently testable subprojects; proposal learning, paired replay counterfactual execution, and the online verifier receive separate plans after this foundation passes.

**Tech Stack:** Python 3.12, Python standard library, pytest 8.4.2, existing official `cg` runtime for later integration smoke tests.

## Global Constraints

- Do not read, print, copy, commit, or package any credential value.
- Keep local transfer archives and credentials outside the repository and include them in pre-publication secret scans.
- Raw competition data, downloaded agents, generated datasets, trained artifacts, and submission archives remain outside Git.
- An unpaired full-game pool may report diagnostics but must never return a promotion status of `PASS`.
- Paired intervals group repeated rollouts by independent decision ID before bootstrapping.
- Missing metadata, malformed rows, runtime errors, or unknown evidence kinds fail closed.
- No Kaggle submission is made by this plan.
- Preserve unrelated workspace modifications.
- Use a worktree created with `superpowers:using-git-worktrees` before executing Task 1.

## Scope Boundary

This plan implements the first independently useful subsystem from the approved
design. It deliberately does not train a new model, build the belief-state
sampler, alter an agent's decisions, or submit to Kaggle. Those changes depend
on the evidence contracts produced here.

## File Map

- `pokemon-tcg-ai-battle/pyproject.toml`: project metadata and pytest discovery.
- `pokemon-tcg-ai-battle/README.md`: current evaluation semantics and commands.
- `pokemon-tcg-ai-battle/scripts/evaluate_pool_from_csv.py`: emits explicit unpaired evidence metadata.
- `pokemon-tcg-ai-battle/scripts/run_candidate_gate.py`: diagnostic summary that refuses to promote unpaired evidence.
- `pokemon-tcg-ai-battle/scripts/run_local_match.py`: imported baseline match runner, unchanged in this plan.
- `pokemon-tcg-ai-battle/scripts/run_v11_v19_calibration_gate.py`: imported source of the current risk-pool definition.
- `pokemon-tcg-ai-battle/scripts/package_submission.py`: invokes archive safety before and after packaging.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/evidence.py`: evidence-kind and trial schemas.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_stats.py`: grouped stratified bootstrap.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/manifest.py`: deterministic hashes and JSONL experiment records.
- `pokemon-tcg-ai-battle/src/ptcg_hybrid/archive_safety.py`: source-tree and tar secret detection.
- `pokemon-tcg-ai-battle/tests/`: unit and CLI-contract tests.

---

### Task 1: Import the repo-owned baseline and establish typed evidence

**Files:**
- Modify: `.gitignore`
- Create: `pokemon-tcg-ai-battle/pyproject.toml`
- Create: `pokemon-tcg-ai-battle/README.md` from the existing workspace snapshot
- Create: `pokemon-tcg-ai-battle/requirements.txt` from the existing workspace snapshot
- Create: `pokemon-tcg-ai-battle/scripts/evaluate_pool_from_csv.py` from the existing workspace snapshot
- Create: `pokemon-tcg-ai-battle/scripts/run_candidate_gate.py` from the existing workspace snapshot
- Create: `pokemon-tcg-ai-battle/scripts/run_local_match.py` from the existing workspace snapshot
- Create: `pokemon-tcg-ai-battle/scripts/run_v11_v19_calibration_gate.py` from the existing workspace snapshot
- Create: `pokemon-tcg-ai-battle/scripts/package_submission.py` from the existing workspace snapshot
- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/__init__.py`
- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/evidence.py`
- Test: `pokemon-tcg-ai-battle/tests/test_evidence.py`

**Interfaces:**
- Consumes: the approved design and the current untracked project snapshot at `/path/to/kaggle-carry/pokemon-tcg-ai-battle`.
- Produces: `EvidenceKind`, `TrialOutcome`, and `parse_trial_row(row: Mapping[str, str]) -> TrialOutcome`.

- [ ] **Step 1: Copy only the existing repo-owned entry points into the isolated worktree**

Run from the worktree root:

```bash
SOURCE=/path/to/kaggle-carry/pokemon-tcg-ai-battle
mkdir -p pokemon-tcg-ai-battle/scripts pokemon-tcg-ai-battle/src/ptcg_hybrid pokemon-tcg-ai-battle/tests
cp "$SOURCE/README.md" pokemon-tcg-ai-battle/README.md
cp "$SOURCE/requirements.txt" pokemon-tcg-ai-battle/requirements.txt
cp "$SOURCE/scripts/evaluate_pool_from_csv.py" pokemon-tcg-ai-battle/scripts/evaluate_pool_from_csv.py
cp "$SOURCE/scripts/run_candidate_gate.py" pokemon-tcg-ai-battle/scripts/run_candidate_gate.py
cp "$SOURCE/scripts/run_local_match.py" pokemon-tcg-ai-battle/scripts/run_local_match.py
cp "$SOURCE/scripts/run_v11_v19_calibration_gate.py" pokemon-tcg-ai-battle/scripts/run_v11_v19_calibration_gate.py
cp "$SOURCE/scripts/package_submission.py" pokemon-tcg-ai-battle/scripts/package_submission.py
```

Expected: exactly the seven listed Python/metadata files are copied; no agents,
data, `.venv`, submissions, or credentials are copied.

- [ ] **Step 2: Add project and pytest configuration**

Create `pokemon-tcg-ai-battle/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "pokemon-tcg-hybrid"
version = "0.1.0"
requires-python = ">=3.12"

[project.optional-dependencies]
dev = ["pytest==8.4.2"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src", "scripts"]
testpaths = ["tests"]
addopts = "-ra"
```

Create `pokemon-tcg-ai-battle/src/ptcg_hybrid/__init__.py`:

```python
"""Shared evaluation and runtime-safety primitives for the PTCG agent."""
```

Append these development-artifact rules to the root `.gitignore`:

```gitignore
**/.pytest_cache/
**/*.egg-info/
```

- [ ] **Step 3: Write the failing evidence-schema tests**

Create `pokemon-tcg-ai-battle/tests/test_evidence.py`:

```python
from __future__ import annotations

import pytest

from ptcg_hybrid.evidence import EvidenceKind, parse_trial_row


def test_parse_unpaired_trial_row() -> None:
    trial = parse_trial_row(
        {
            "agent": "candidate",
            "opponent": "alakazam",
            "seat": "1",
            "game_id": "7",
            "win": "1",
            "result": "1",
            "status": "ok",
            "evidence_kind": "unpaired_full_game",
            "paired": "0",
        }
    )
    assert trial.agent == "candidate"
    assert trial.trial_id == "alakazam|seat=1|trial=7"
    assert trial.value == 1.0
    assert trial.kind is EvidenceKind.UNPAIRED_FULL_GAME
    assert trial.is_paired is False


def test_parse_trial_rejects_unknown_evidence_kind() -> None:
    with pytest.raises(ValueError, match="unknown evidence_kind"):
        parse_trial_row(
            {
                "agent": "candidate",
                "opponent": "alakazam",
                "seat": "0",
                "game_id": "0",
                "win": "0",
                "status": "ok",
                "evidence_kind": "looks_good",
                "paired": "1",
            }
        )


def test_parse_trial_rejects_paired_claim_for_unpaired_kind() -> None:
    with pytest.raises(ValueError, match="cannot be marked paired"):
        parse_trial_row(
            {
                "agent": "candidate",
                "opponent": "alakazam",
                "seat": "0",
                "game_id": "0",
                "win": "0",
                "status": "ok",
                "evidence_kind": "unpaired_full_game",
                "paired": "1",
            }
        )
```

- [ ] **Step 4: Run the tests to verify they fail**

Run:

```bash
cd pokemon-tcg-ai-battle
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest tests/test_evidence.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ptcg_hybrid.evidence'`.

- [ ] **Step 5: Implement the evidence schema**

Create `pokemon-tcg-ai-battle/src/ptcg_hybrid/evidence.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class EvidenceKind(StrEnum):
    UNPAIRED_FULL_GAME = "unpaired_full_game"
    PAIRED_COUNTERFACTUAL = "paired_counterfactual"
    OFFICIAL_ONLINE = "official_online"


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    agent: str
    opponent: str
    seat: int
    trial_id: str
    value: float
    status: str
    kind: EvidenceKind
    is_paired: bool

    @property
    def stratum(self) -> str:
        return f"{self.opponent}|seat={self.seat}"


def parse_trial_row(row: Mapping[str, str]) -> TrialOutcome:
    raw_kind = row.get("evidence_kind", "")
    try:
        kind = EvidenceKind(raw_kind)
    except ValueError as exc:
        raise ValueError(f"unknown evidence_kind: {raw_kind!r}") from exc

    paired_text = row.get("paired", "")
    if paired_text not in {"0", "1"}:
        raise ValueError(f"paired must be 0 or 1, got {paired_text!r}")
    is_paired = paired_text == "1"
    if kind is EvidenceKind.UNPAIRED_FULL_GAME and is_paired:
        raise ValueError("unpaired_full_game cannot be marked paired")
    if kind is EvidenceKind.PAIRED_COUNTERFACTUAL and not is_paired:
        raise ValueError("paired_counterfactual must be marked paired")

    seat = int(row["seat"])
    if seat not in {0, 1}:
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    opponent = row["opponent"]
    game_id = row["game_id"]
    value = float(row["win"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"win/value must be in [0, 1], got {value}")
    return TrialOutcome(
        agent=row["agent"],
        opponent=opponent,
        seat=seat,
        trial_id=f"{opponent}|seat={seat}|trial={game_id}",
        value=value,
        status=row.get("status", ""),
        kind=kind,
        is_paired=is_paired,
    )
```

- [ ] **Step 6: Run the tests and compile the imported scripts**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence.py -v
.venv/bin/python -m py_compile scripts/evaluate_pool_from_csv.py scripts/run_candidate_gate.py scripts/run_local_match.py scripts/run_v11_v19_calibration_gate.py scripts/package_submission.py
```

Expected: `3 passed`; all five scripts compile with no output.

- [ ] **Step 7: Commit the baseline and evidence contract**

```bash
git add .gitignore pokemon-tcg-ai-battle/pyproject.toml pokemon-tcg-ai-battle/README.md pokemon-tcg-ai-battle/requirements.txt pokemon-tcg-ai-battle/scripts pokemon-tcg-ai-battle/src/ptcg_hybrid/__init__.py pokemon-tcg-ai-battle/src/ptcg_hybrid/evidence.py pokemon-tcg-ai-battle/tests/test_evidence.py
git commit -m "test: establish PTCG evaluation evidence contract"
```

Expected: the commit contains only the imported entry points, package scaffold,
and evidence tests; generated installation metadata such as `*.egg-info` is not
staged.

---

### Task 2: Implement grouped stratified paired bootstrap statistics

**Files:**
- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_stats.py`
- Test: `pokemon-tcg-ai-battle/tests/test_paired_stats.py`

**Interfaces:**
- Consumes: `PairedObservation(group_id, stratum, candidate_value, control_value)`.
- Produces: `EffectEstimate` and `stratified_paired_bootstrap(observations, *, confidence, resamples, seed)`.

- [ ] **Step 1: Write failing tests for positive, zero, grouped, and invalid evidence**

Create `pokemon-tcg-ai-battle/tests/test_paired_stats.py`:

```python
from __future__ import annotations

import pytest

from ptcg_hybrid.paired_stats import PairedObservation, stratified_paired_bootstrap


def obs(group: str, stratum: str, candidate: float, control: float) -> PairedObservation:
    return PairedObservation(group, stratum, candidate, control)


def test_constant_positive_effect_has_exact_interval() -> None:
    rows = [obs(f"g{i}", "alakazam|seat=0", 1.0, 0.5) for i in range(20)]
    estimate = stratified_paired_bootstrap(rows, resamples=500, seed=7)
    assert estimate.groups == 20
    assert estimate.delta == pytest.approx(0.5)
    assert estimate.ci_low == pytest.approx(0.5)
    assert estimate.ci_high == pytest.approx(0.5)


def test_repeated_rollouts_are_aggregated_by_decision_group() -> None:
    rows = [
        obs("d1", "mirror|seat=0", 1.0, 0.0),
        obs("d1", "mirror|seat=0", 0.0, 0.0),
        obs("d2", "mirror|seat=0", 0.0, 1.0),
    ]
    estimate = stratified_paired_bootstrap(rows, resamples=500, seed=11)
    assert estimate.groups == 2
    assert estimate.delta == pytest.approx(-0.25)


def test_strata_are_resampled_at_their_original_sizes() -> None:
    rows = [
        obs("a1", "a", 1.0, 0.0),
        obs("a2", "a", 1.0, 0.0),
        obs("a3", "a", 1.0, 0.0),
        obs("b1", "b", 0.0, 1.0),
    ]
    estimate = stratified_paired_bootstrap(rows, resamples=500, seed=13)
    assert estimate.groups == 4
    assert estimate.delta == pytest.approx(0.5)


def test_conflicting_strata_for_same_group_are_rejected() -> None:
    with pytest.raises(ValueError, match="multiple strata"):
        stratified_paired_bootstrap(
            [obs("d1", "a", 1.0, 0.0), obs("d1", "b", 1.0, 0.0)],
            resamples=100,
        )


def test_out_of_range_value_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        stratified_paired_bootstrap([obs("d1", "a", 1.2, 0.0)], resamples=100)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_paired_stats.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ptcg_hybrid.paired_stats'`.

- [ ] **Step 3: Implement the grouped bootstrap**

Create `pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_stats.py`:

```python
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PairedObservation:
    group_id: str
    stratum: str
    candidate_value: float
    control_value: float


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    groups: int
    delta: float
    ci_low: float
    ci_high: float
    confidence: float
    resamples: int
    seed: int


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _grouped_differences(
    observations: Iterable[PairedObservation],
) -> dict[str, list[float]]:
    values_by_group: dict[str, list[float]] = defaultdict(list)
    stratum_by_group: dict[str, str] = {}
    for row in observations:
        for name, value in (
            ("candidate_value", row.candidate_value),
            ("control_value", row.control_value),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        previous = stratum_by_group.setdefault(row.group_id, row.stratum)
        if previous != row.stratum:
            raise ValueError(f"group {row.group_id!r} appears in multiple strata")
        values_by_group[row.group_id].append(row.candidate_value - row.control_value)
    if not values_by_group:
        raise ValueError("at least one paired observation is required")

    by_stratum: dict[str, list[float]] = defaultdict(list)
    for group_id, values in values_by_group.items():
        by_stratum[stratum_by_group[group_id]].append(sum(values) / len(values))
    return dict(by_stratum)


def stratified_paired_bootstrap(
    observations: Iterable[PairedObservation],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260714,
) -> EffectEstimate:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    by_stratum = _grouped_differences(observations)
    groups = sum(len(values) for values in by_stratum.values())
    delta = sum(sum(values) for values in by_stratum.values()) / groups

    rng = random.Random(seed)
    bootstraps: list[float] = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for values in by_stratum.values():
            total += sum(rng.choice(values) for _ in range(len(values)))
            count += len(values)
        bootstraps.append(total / count)
    bootstraps.sort()
    alpha = (1.0 - confidence) / 2.0
    return EffectEstimate(
        groups=groups,
        delta=delta,
        ci_low=_percentile(bootstraps, alpha),
        ci_high=_percentile(bootstraps, 1.0 - alpha),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
```

- [ ] **Step 4: Run paired-statistics and evidence tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_paired_stats.py tests/test_evidence.py -v
```

Expected: `8 passed`.

- [ ] **Step 5: Commit the statistical primitive**

```bash
git add pokemon-tcg-ai-battle/src/ptcg_hybrid/paired_stats.py pokemon-tcg-ai-battle/tests/test_paired_stats.py
git commit -m "feat: add grouped paired bootstrap statistics"
```

---

### Task 3: Mark local pool games unpaired and make the gate fail closed

**Files:**
- Modify: `pokemon-tcg-ai-battle/scripts/evaluate_pool_from_csv.py:36-176`
- Modify: `pokemon-tcg-ai-battle/scripts/run_candidate_gate.py:86-173`
- Test: `pokemon-tcg-ai-battle/tests/test_unpaired_gate.py`

**Interfaces:**
- Consumes: `EvidenceKind.UNPAIRED_FULL_GAME` and existing evaluation CSV rows.
- Produces: CSV columns `evidence_kind=unpaired_full_game` and `paired=0`; `gate_status(...) -> str` returning `HOLD_UNPAIRED` or a fail-closed status. This legacy full-game script never returns `PASS`.

- [ ] **Step 1: Write failing tests for CSV metadata and gate status**

Create `pokemon-tcg-ai-battle/tests/test_unpaired_gate.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path

from run_candidate_gate import gate_status, summarize_results


def write_rows(path: Path) -> None:
    fieldnames = [
        "agent", "opponent", "seat", "game_id", "win", "result",
        "decisions", "status", "stderr_tail", "evidence_kind", "paired",
    ]
    rows = [
        ["baseline", "opp", 0, 0, 0, 1, 20, "ok", "", "unpaired_full_game", 0],
        ["candidate", "opp", 0, 0, 1, 0, 20, "ok", "", "unpaired_full_game", 0],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def test_unpaired_results_are_diagnostic_only(tmp_path: Path) -> None:
    result_path = tmp_path / "results.csv"
    write_rows(result_path)
    totals, errors, evidence_kinds = summarize_results(result_path)
    assert totals["candidate"] == (1, 1)
    assert errors == 0
    assert evidence_kinds == {"unpaired_full_game"}
    assert gate_status(evidence_kinds, errors=0, total=1) == "HOLD_UNPAIRED"


def test_legacy_gate_rejects_non_pool_evidence() -> None:
    assert gate_status({"paired_counterfactual"}, errors=0, total=30) == "FAIL_UNSUPPORTED_EVIDENCE"
    assert gate_status(set(), errors=0, total=30) == "FAIL_UNSUPPORTED_EVIDENCE"


def test_unpaired_gate_fails_when_runtime_errors_exist() -> None:
    assert gate_status({"unpaired_full_game"}, errors=1, total=30) == "FAIL"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_unpaired_gate.py -v
```

Expected: import fails because `gate_status` does not exist and
`summarize_results` returns two values.

- [ ] **Step 3: Add explicit evidence metadata to every evaluator row**

In `scripts/evaluate_pool_from_csv.py`, import the enum:

```python
from ptcg_hybrid.evidence import EvidenceKind
```

Add these keys to every dictionary returned by `run_one`, including error and
parse-error paths:

```python
"evidence_kind": EvidenceKind.UNPAIRED_FULL_GAME.value,
"paired": 0,
```

Replace the writer fieldnames with:

```python
fieldnames = [
    "agent", "opponent", "seat", "game_id", "win", "result",
    "decisions", "status", "stderr_tail", "evidence_kind", "paired",
]
writer = csv.DictWriter(file, fieldnames=fieldnames)
```

Immediately after parsing arguments, print the contract warning:

```python
print(
    "evidence=unpaired_full_game: game_id labels repetitions but does not "
    "seed the official engine; results are diagnostic and cannot promote",
    file=sys.stderr,
)
```

- [ ] **Step 4: Make candidate-gate status depend on evidence kind**

In `scripts/run_candidate_gate.py`, replace `summarize_results` with:

```python
def summarize_results(
    path: Path,
) -> tuple[dict[str, tuple[int, int]], int, set[str]]:
    rows = list(csv.DictReader(path.open()))
    totals: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors = 0
    evidence_kinds: set[str] = set()
    for row in rows:
        evidence_kinds.add(row.get("evidence_kind", ""))
        if row.get("status") != "ok":
            errors += 1
            continue
        agent = row["agent"]
        totals[agent][0] += int(row["win"])
        totals[agent][1] += 1
    return (
        {agent: (wins, total) for agent, (wins, total) in totals.items()},
        errors,
        evidence_kinds,
    )
```

Add the pure gate policy:

```python
def gate_status(
    evidence_kinds: set[str],
    *,
    errors: int,
    total: int,
) -> str:
    if errors or total <= 0:
        return "FAIL"
    if evidence_kinds == {"unpaired_full_game"}:
        return "HOLD_UNPAIRED"
    return "FAIL_UNSUPPORTED_EVIDENCE"
```

Change `print_gate_summary` to accept `evidence_kinds: set[str]`, calculate the
existing diagnostic interval, then use:

```python
status = gate_status(
    evidence_kinds,
    errors=errors,
    total=total,
)
passed[name] = False
```

Write `status` into the summary row and printed output instead of deriving it
from the independent-proportion z-score. Keep `z` and the old interval only as
diagnostic columns named `diagnostic_unpaired_z`,
`diagnostic_unpaired_ci95_low`, and `diagnostic_unpaired_ci95_high`.

Update the main loop to unpack and pass evidence kinds:

```python
totals, errors, evidence_kinds = summarize_results(result_csv)
```

The overall row is `HOLD_UNPAIRED` only when every component has that status;
otherwise it is `FAIL`. Both statuses exit nonzero. Paired evidence is consumed
by the dedicated paired-counterfactual CLI in the next subproject, where
`stratified_paired_bootstrap` is mandatory.

- [ ] **Step 5: Run unit tests and a synthetic summary check**

Run:

```bash
.venv/bin/python -m pytest tests/test_unpaired_gate.py tests/test_evidence.py tests/test_paired_stats.py -v
.venv/bin/python -m py_compile scripts/evaluate_pool_from_csv.py scripts/run_candidate_gate.py
```

Expected: `11 passed`; both scripts compile. No engine matches are required for
this task.

- [ ] **Step 6: Commit the fail-closed gate**

```bash
git add pokemon-tcg-ai-battle/scripts/evaluate_pool_from_csv.py pokemon-tcg-ai-battle/scripts/run_candidate_gate.py pokemon-tcg-ai-battle/tests/test_unpaired_gate.py
git commit -m "fix: prevent unpaired PTCG pools from promoting agents"
```

---

### Task 4: Record deterministic experiment manifests

**Files:**
- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/manifest.py`
- Test: `pokemon-tcg-ai-battle/tests/test_manifest.py`

**Interfaces:**
- Consumes: an `ExperimentRecord` and file/directory artifact paths.
- Produces: `sha256_path(path: Path) -> str` and `append_record(path: Path, record: ExperimentRecord) -> None`.

- [ ] **Step 1: Write failing hash and JSONL tests**

Create `pokemon-tcg-ai-battle/tests/test_manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ptcg_hybrid.manifest import ExperimentRecord, append_record, sha256_path


def test_directory_hash_is_content_and_path_stable(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "nested").mkdir(parents=True)
        (root / "b.txt").write_text("two")
        (root / "nested/a.txt").write_text("one")
    assert sha256_path(left) == sha256_path(right)
    (right / "nested/a.txt").write_text("changed")
    assert sha256_path(left) != sha256_path(right)


def test_append_record_writes_one_canonical_json_object(tmp_path: Path) -> None:
    output = tmp_path / "experiments.jsonl"
    record = ExperimentRecord(
        experiment_id="eval-001",
        created_at="2026-07-14T00:00:00Z",
        git_commit="a" * 40,
        decision="HOLD_UNPAIRED",
        control={"name": "v11", "sha256": "b" * 64},
        candidates=[{"name": "v2", "sha256": "c" * 64}],
        dataset_sha256="d" * 64,
        command=["python", "scripts/run_candidate_gate.py"],
        metrics={"delta": 0.01},
    )
    append_record(output, record)
    rows = output.read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["experiment_id"] == "eval-001"


def test_invalid_decision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decision"):
        append_record(
            tmp_path / "experiments.jsonl",
            ExperimentRecord(
                experiment_id="eval-002",
                created_at="2026-07-14T00:00:00Z",
                git_commit="a" * 40,
                decision="SHIP_IT",
                control={},
                candidates=[],
                dataset_sha256="d" * 64,
                command=[],
                metrics={},
            ),
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_manifest.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ptcg_hybrid.manifest'`.

- [ ] **Step 3: Implement deterministic hashes and the manifest record**

Create `pokemon-tcg-ai-battle/src/ptcg_hybrid/manifest.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ALLOWED_DECISIONS = {"REJECT", "HOLD", "HOLD_UNPAIRED", "PACKAGE", "REQUEST_SUBMISSION"}


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    created_at: str
    git_commit: str
    decision: str
    control: dict[str, Any]
    candidates: list[dict[str, Any]]
    dataset_sha256: str
    command: list[str]
    metrics: dict[str, Any]


def sha256_path(path: Path) -> str:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def append_record(path: Path, record: ExperimentRecord) -> None:
    if record.decision not in ALLOWED_DECISIONS:
        raise ValueError(f"unsupported decision: {record.decision!r}")
    if len(record.git_commit) != 40:
        raise ValueError("git_commit must be a 40-character commit hash")
    if len(record.dataset_sha256) != 64:
        raise ValueError("dataset_sha256 must be a 64-character SHA-256")
    payload = asdict(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
```

- [ ] **Step 4: Run the manifest tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_manifest.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the experiment manifest primitive**

```bash
git add pokemon-tcg-ai-battle/src/ptcg_hybrid/manifest.py pokemon-tcg-ai-battle/tests/test_manifest.py
git commit -m "feat: record reproducible PTCG experiments"
```

---

### Task 5: Reject credential-bearing source trees and archives

**Files:**
- Create: `pokemon-tcg-ai-battle/src/ptcg_hybrid/archive_safety.py`
- Modify: `pokemon-tcg-ai-battle/scripts/package_submission.py:1-87`
- Modify: `.gitignore`
- Test: `pokemon-tcg-ai-battle/tests/test_archive_safety.py`

**Interfaces:**
- Consumes: an agent directory or generated `.tar.gz`.
- Produces: `assert_safe_tree(path: Path) -> None` and `assert_safe_tar(path: Path) -> None`, raising `UnsafeArtifactError` with names only and never credential contents.

- [ ] **Step 1: Write failing source-tree and tar safety tests**

Create `pokemon-tcg-ai-battle/tests/test_archive_safety.py`:

```python
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from ptcg_hybrid.archive_safety import UnsafeArtifactError, assert_safe_tar, assert_safe_tree


def test_tree_rejects_credentials_filename_without_reading_it(tmp_path: Path) -> None:
    secret = tmp_path / ".kaggle/credentials.json"
    secret.parent.mkdir()
    secret.write_text("DO_NOT_READ")
    with pytest.raises(UnsafeArtifactError, match="credentials.json"):
        assert_safe_tree(tmp_path)


def test_tar_rejects_credentials_member(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"DO_NOT_READ"
        info = tarfile.TarInfo(".kaggle/credentials.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(UnsafeArtifactError, match="credentials.json"):
        assert_safe_tar(archive)


def test_safe_submission_archive_passes(tmp_path: Path) -> None:
    archive = tmp_path / "safe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in (("main.py", b"def agent(obs): return []"), ("deck.csv", b"1\n")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    assert_safe_tar(archive)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_archive_safety.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ptcg_hybrid.archive_safety'`.

- [ ] **Step 3: Implement name-only secret detection**

Create `pokemon-tcg-ai-battle/src/ptcg_hybrid/archive_safety.py`:

```python
from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath


FORBIDDEN_NAMES = {
    ".env",
    "credentials.json",
    "kaggle.json",
    "id_rsa",
    "id_ed25519",
}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


class UnsafeArtifactError(ValueError):
    pass


def _unsafe_name(name: str) -> bool:
    path = PurePosixPath(name)
    lowered_parts = [part.lower() for part in path.parts]
    basename = lowered_parts[-1] if lowered_parts else ""
    return (
        path.is_absolute()
        or ".." in path.parts
        or basename in FORBIDDEN_NAMES
        or Path(basename).suffix in FORBIDDEN_SUFFIXES
        or ".kaggle" in lowered_parts
    )


def assert_safe_tree(path: Path) -> None:
    unsafe = [item.relative_to(path).as_posix() for item in path.rglob("*") if _unsafe_name(item.relative_to(path).as_posix())]
    if unsafe:
        raise UnsafeArtifactError(f"unsafe credential-bearing paths: {sorted(unsafe)!r}")


def assert_safe_tar(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        unsafe = [member.name for member in archive.getmembers() if _unsafe_name(member.name)]
    if unsafe:
        raise UnsafeArtifactError(f"unsafe credential-bearing archive members: {sorted(unsafe)!r}")
```

The implementation intentionally checks names and metadata only. It never
opens or prints credential file contents.

- [ ] **Step 4: Integrate safety into submission packaging**

At the top of `scripts/package_submission.py`, add:

```python
from ptcg_hybrid.archive_safety import assert_safe_tar, assert_safe_tree
```

At the start of `build_archive`, before reading `main.py`, add:

```python
assert_safe_tree(agent_dir)
```

After the `with tarfile.open(...)` block completes, before printing success,
add:

```python
assert_safe_tar(output)
```

The packager remains allowlist-based; this task adds a fail-closed safety check
without changing the current submission file set.

- [ ] **Step 5: Ignore credential-bearing archives and standard token files**

Append these exact lines to the root `.gitignore` in the isolated worktree:

```gitignore

# Credential-bearing transfer bundles and Kaggle tokens must never enter Git.
**/*credentials*.tar.gz
**/credentials.json
**/kaggle.json
```

Do not delete, move, open, or stage the existing transfer archive.

- [ ] **Step 6: Run safety and packaging tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_archive_safety.py -v
.venv/bin/python -m py_compile scripts/package_submission.py
```

Expected: `3 passed`; packager compiles without output.

- [ ] **Step 7: Commit safety enforcement**

```bash
git add .gitignore pokemon-tcg-ai-battle/src/ptcg_hybrid/archive_safety.py pokemon-tcg-ai-battle/scripts/package_submission.py pokemon-tcg-ai-battle/tests/test_archive_safety.py
git commit -m "security: block credentials from PTCG artifacts"
```

---

### Task 6: Document the corrected evidence semantics and verify the subsystem

**Files:**
- Modify: `pokemon-tcg-ai-battle/README.md`
- Test: all tests under `pokemon-tcg-ai-battle/tests/`

**Interfaces:**
- Consumes: all Task 1-5 commands and evidence contracts.
- Produces: a documented, green foundation ready for the paired replay-counterfactual plan.

- [ ] **Step 1: Replace stale status language with the current evidence contract**

Add this section immediately after the opening paragraph of
`pokemon-tcg-ai-battle/README.md`:

```markdown
## Evaluation Evidence Contract

Local full-game pools are **unpaired diagnostics**. The official `battle_start`
API does not accept a seed or explicit initial hidden state, so the same
`game_id` does not give control and candidate the same draw order. These pools
may detect crashes, timeouts, and large regressions, but they cannot promote a
candidate.

`scripts/evaluate_pool_from_csv.py` writes
`evidence_kind=unpaired_full_game,paired=0`. Consequently,
`scripts/run_candidate_gate.py` reports `HOLD_UNPAIRED` even when its diagnostic
win-rate delta is positive. Promotion evidence must come from grouped paired
counterfactuals or a documented official online experiment.

Run the unit contract:

```bash
.venv/bin/python -m pytest tests -v
```
```

Remove the obsolete claim that local 55% win rate is sufficient for a Kaggle
submission. Keep setup and packaging commands that are still correct.

- [ ] **Step 2: Run the complete foundation test suite**

Run:

```bash
cd pokemon-tcg-ai-battle
.venv/bin/python -m pytest tests -v
```

Expected: `17 passed` from evidence (3), paired statistics (5), unpaired gate
(3), manifest (3), and archive safety (3).

- [ ] **Step 3: Run static and repository checks**

Run from the repository root:

```bash
pokemon-tcg-ai-battle/.venv/bin/python -m py_compile \
  pokemon-tcg-ai-battle/scripts/evaluate_pool_from_csv.py \
  pokemon-tcg-ai-battle/scripts/run_candidate_gate.py \
  pokemon-tcg-ai-battle/scripts/run_local_match.py \
  pokemon-tcg-ai-battle/scripts/run_v11_v19_calibration_gate.py \
  pokemon-tcg-ai-battle/scripts/package_submission.py \
  pokemon-tcg-ai-battle/src/ptcg_hybrid/*.py
git diff --check
git status --short
```

Expected: compilation and `git diff --check` produce no output. `git status`
shows only the intended README change before the final commit; no data,
credential archive, agent pool, `.venv`, egg-info, or test cache is staged.

- [ ] **Step 4: Commit the evidence documentation**

```bash
git add pokemon-tcg-ai-battle/README.md
git commit -m "docs: explain PTCG evaluation evidence levels"
```

- [ ] **Step 5: Record the handoff evidence**

Run:

```bash
git log --oneline -6
git status --short
```

Expected: six task commits or fewer if a reviewer requested logically combined
commits, a clean implementation worktree, and no Kaggle submission.

## Plan Self-Review Mapping

- Design Goals and repository boundary: Tasks 1 and 6.
- Grouped paired evidence and confidence intervals: Task 2.
- Fail-closed treatment of current unpaired pools: Task 3.
- Reproducible source/data/artifact records: Task 4.
- Credential and archive safety: Task 5.
- Runtime legality, belief-state search, proposal training, and online
  challenger behavior are intentionally deferred to their dedicated plans; no
  claim in this foundation depends on them already existing.
