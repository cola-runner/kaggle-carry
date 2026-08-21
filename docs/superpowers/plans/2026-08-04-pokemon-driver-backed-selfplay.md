# Driver-Backed Self-Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded Mac proof in which four neural residual policies explore and learn around their original rule drivers without replacing the drivers by imitation.

**Architecture:** A `DriverBackedActor` obtains the original driver's action first, then uses a 10% neural-policy mixture during training. PPO evaluates the same mixture probability. At evaluation, the original action remains the default and the model overrides it only when the neural log-probability margin is at least 2.0. A separate small runner performs two current-current rounds and compares untouched drivers with final residual actors on one frozen judge schedule.

**Tech Stack:** Python 3.12, NumPy, PyTorch/MPS, official local `cg` engine, pytest.

## Global Constraints

- Keep all four original drivers and decks unchanged.
- Do not bootstrap by imitation.
- Do not write raw replays.
- Cap owned temporary storage at 128 MiB and remove it after every exit path.
- Retain one compact JSON report; retain checkpoints only after a passing gate.
- Run on the Mac before using the 1080Ti.

---

### Task 1: Driver-backed behavior policy

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/residual.py`
- Modify: `pokemon-tcg-ai-battle/league_selfplay/engine.py`
- Test: `pokemon-tcg-ai-battle/tests/league_selfplay/test_residual.py`

**Interfaces:**
- Consumes: `PolicyValueNet`, `sample_action`, `greedy_action`, `action_log_probability`, and a callable `driver_action(observation) -> list[int]`.
- Produces: `mixture_log_probability(...) -> float`, `ResidualDecision`, `ResidualCounters`, and `DriverBackedActor.decide(...)`.

- [ ] Write tests proving exploration rate 0 returns the exact driver action, training trajectories store the baseline action and finite mixture log probability, and evaluation overrides only above a 2.0 margin.
- [ ] Run `pytest tests/league_selfplay/test_residual.py -q` and verify it fails because `residual.py` does not exist.
- [ ] Implement the mixture probability as `log((1-epsilon) * I[action==driver] + epsilon * neural_probability)` and add `baseline_action` plus `exploration_rate` to `TrajectoryStep`.
- [ ] Implement `DriverBackedActor`: exact driver fallback, bounded training exploration, thresholded evaluation override, legal-action validation, and override/exploration counters.
- [ ] Run the focused test and existing engine/PPO tests.
- [ ] Commit with `feat: keep drivers inside self-play policy`.

### Task 2: PPO mixture update

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/ppo.py`
- Test: `pokemon-tcg-ai-battle/tests/league_selfplay/test_ppo.py`

**Interfaces:**
- Consumes: `TrajectoryStep.baseline_action`, `TrajectoryStep.exploration_rate`.
- Produces: `_mixture_log_probability_torch(...) -> torch.Tensor`, used for every residual trajectory during PPO.

- [ ] Add a failing parity test comparing the Torch mixture log probability with the NumPy implementation for both driver and non-driver actions.
- [ ] Run the new test and verify failure.
- [ ] Implement the differentiable mixture log probability with `torch.logaddexp` for the driver action and `log(epsilon) + neural_log_probability` otherwise; preserve legacy behavior when no baseline is stored.
- [ ] Run `pytest tests/league_selfplay/test_ppo.py tests/league_selfplay/test_residual.py -q`.
- [ ] Commit with `feat: train residual policy with behavior mixture`.

### Task 3: Bounded Mac residual proof

**Files:**
- Create: `pokemon-tcg-ai-battle/league_selfplay/residual_runner.py`
- Create: `pokemon-tcg-ai-battle/scripts/run_driver_backed_league.py`
- Create: `pokemon-tcg-ai-battle/tests/league_selfplay/test_residual_runner.py`

**Interfaces:**
- Consumes: `DriverRegistry`, `DriverBackedActor`, `run_training_game`, `update_population`, the existing four judges, and `RunStorage`.
- Produces: `run_residual_proof(project_root, ...) -> ResidualReport` and an atomic JSON report writer.

- [ ] Write a failing runner test using small fakes that requires two update rounds, identical start/final judge support, nonzero learned overrides, zero raw replays, 128 MiB quota, and cleanup after rejection.
- [ ] Run the focused test and verify failure.
- [ ] Implement a 12-game first round, 12-game second round, and 64-game start/final judge comparison. Starting actors always execute their drivers; final actors use thresholded residual overrides. Reject unless group delta is positive, at least one override occurs, all four updates are finite, and all safety gates pass.
- [ ] Add `--dry-run` and `--standard` entry points; both write only the requested report.
- [ ] Run `pytest tests/league_selfplay/test_residual_runner.py tests/league_selfplay/test_residual.py tests/league_selfplay/test_ppo.py -q`.
- [ ] Commit with `feat: add bounded driver-backed Mac proof`.

### Task 4: Real Mac proof and decision

**Files:**
- Create on run: `pokemon-tcg-ai-battle/reports/mac_driver_backed_latest.json`

**Interfaces:**
- Consumes: `scripts/run_driver_backed_league.py --standard`.
- Produces: one evidence report containing baseline score, final score, delta, per-member deltas, override count, update validity, runtime, and storage cleanup measurements.

- [ ] Measure `artifacts/` before the run and ensure the runner records the same measurement afterward.
- [ ] Run `python scripts/run_driver_backed_league.py --standard --report reports/mac_driver_backed_latest.json` with a six-minute wall-time limit.
- [ ] Verify no `pokemon-residual-*` directory remains, no raw replay exists, and the report is below 100 KiB.
- [ ] If the gate passes, retain the four residual checkpoints; if it rejects, retain no checkpoint and use the report to choose the next single hypothesis.
- [ ] Run the complete focused league suite and commit the report with `test: record driver-backed Mac proof`.

### Task 5: Survivor promotion with fresh confirmation

**Files:**
- Modify: `pokemon-tcg-ai-battle/league_selfplay/residual_runner.py`
- Modify: `pokemon-tcg-ai-battle/tests/league_selfplay/test_residual_runner.py`
- Update on run: `pokemon-tcg-ai-battle/reports/mac_driver_backed_latest.json`

**Interfaces:**
- Consumes: per-member candidate deltas from the first frozen judge schedule.
- Produces: `select_promoted_members(...) -> tuple[MemberId, ...]` and a fresh confirmation comparison of the hybrid population.

- [ ] Write a failing test proving only strictly positive members are promoted and that an empty promotion set rejects.
- [ ] Implement hybrid evaluation: promoted members enable residual overrides; every other member executes its untouched driver.
- [ ] Run a second fresh judge schedule and make the proof decision depend only on its group delta and override count.
- [ ] Verify cleanup, focused tests, and a real Mac standard report; retain only confirmed promoted checkpoints.
