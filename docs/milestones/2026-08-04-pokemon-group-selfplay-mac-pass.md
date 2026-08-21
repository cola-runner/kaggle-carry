# Milestone: Two-Round Group Self-Play Passed on Mac

**Date:** 2026-08-04  
**Decision:** `PASS_GROUP_SELFPLAY_MAC`  
**Meaning:** For the first time in this project, a population trained from its
own complete Engine22 games beat the original driver population under a frozen,
large-sample local gate.

## Frozen experiment

- Population: Grimmsnarl, Lucario, Crustle, and Alakazam.
- Engine: official Engine22, SHA-256
  `7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30`.
- Training: two complete rounds; every member received 32 valid randomized
  single-action interventions per round.
- Causal constraint: at most one experimental action per game; all later
  actions returned to the current incumbent policy.
- Learner: two independent approximately 66k-parameter action-value scorers
  per member, updated with centered terminal outcomes.
- Safety gate: per-model override margins were calibrated from the current
  round's driver decisions. A runtime override required both models to clear
  their own margin.
- No counterfactual search APIs were used for training or evaluation.
- Evaluation: all four members completed both rounds. The final audit used 192
  candidate games and 192 untouched-driver games on the same balanced matchup
  and seat schedule.

## Result

| Population | Audit score | Games |
|---|---:|---:|
| Two-round candidate | 51.82% | 192 |
| Original drivers | 45.83% | 192 |
| Delta | **+5.99 percentage points** | 384 total |

Per-member audit deltas:

- Alakazam: `+10.42 pp`
- Grimmsnarl: `+8.33 pp`
- Lucario: `+5.21 pp`
- Crustle: `0.00 pp`

The candidate made 259 learned overrides in its 192 audit games. Across the
whole run, Engine22 completed 766 games: 96 calibration games, 286 intervention
games, and 384 audit games. Training produced 256 valid intervention examples
and 30 control games. Total measured phase time was 588.74 seconds.

## What changed the outcome

Two earlier gates exposed mistakes in the evaluation design:

1. A fixed raw-score margin of `0.25` had no consistent meaning across neural
   scorers. It allowed 573 overrides in 48 games and produced a `-20.83 pp`
   group regression.
2. Calibrating margins fixed the override explosion, but eliminating a member
   after only 12 candidate and 12 incumbent games was unstable. Engine22's
   normal `battle_start(deck0, deck1)` API exposes no seed, so Python seeds
   cannot pair the shuffles. Repeating that small gate changed the first-round
   conclusion even with the same configured seed.

The successful protocol therefore kept all four members for both training
rounds and spent the evaluation budget once, on a balanced 192-versus-192
group audit. This is the project-management lesson to preserve: **do not use
tiny noisy batches to terminate population members early.**

## Reproducible evidence

- Report:
  `pokemon-tcg-ai-battle/reports/mac_single_intervention_selfplay_latest.json`
- Confirmed weights and calibrated margins:
  `pokemon-tcg-ai-battle/agents/single_intervention_mac_pass/`
- Runner:
  `pokemon-tcg-ai-battle/league_selfplay/single_intervention_runner.py`
- Command:

  ```bash
  cd pokemon-tcg-ai-battle
  PYTHONDONTWRITEBYTECODE=1 python scripts/run_single_intervention_selfplay.py
  ```

- Relevant commits:
  - `87e039c` — large balanced group audit
  - `051dd3e` — passing Engine22 evidence and retained manifest
  - `8c05067` — confirmed model weights

The run left `artifacts/` unchanged, removed its owned temporary directory,
wrote no replay or raw-observation dataset, and retained only an 8.3 KiB report
plus approximately 1.96 MiB of confirmed weights.

## Scope and next gate

This proves the local group-upgrade mechanism is viable. It does **not** prove
a Kaggle leaderboard gain or statistical certainty for every individual deck.
The next gate is to load these weights and calibrated margins inside a legal,
submission-sized driver, verify both seats and runtime limits locally, and only
then consider a Kaggle submission.
