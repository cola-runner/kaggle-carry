# Pokémon TCG Live-Leader Policy Improvement Design

Date: 2026-07-23

Status: Superseded on 2026-07-28 by
`2026-07-28-pokemon-rolling-multiteacher-policy-improvement-design.md`

## Objective

Build exactly one new Kaggle challenger from the current ladder, not from
historical scores. Keep the current v11 submission as the control and replace
the failed v234 slot only if the new archive passes the offline safety and
measurement gates in this document.

The challenger will use the exact 60-card deck from the current rank-one
submission, but it will not try to copy that submission's actions directly.
It will learn which legal action has the best downstream result by replaying
current ladder states through the official engine.

## Current Evidence

The live snapshot taken on 2026-07-23 identified:

- Team: `Luca`
- Active submission: `54863653`
- Current observed rating: `1221.8`
- Listed official episodes: `386`
- Submission time: `2026-07-20T22:53:08Z`
- Latest checked episode: `87610310`

The exact deck in the latest official replay is:

| Count | Card |
|---:|---|
| 10 | Basic Darkness Energy |
| 4 | Munkidori |
| 4 | Marnie's Impidimp |
| 3 | Marnie's Morgrem |
| 3 | Marnie's Grimmsnarl ex |
| 2 | Snorunt |
| 2 | Froslass |
| 4 | Buddy-Buddy Poffin |
| 4 | Poké Pad |
| 4 | Team Rocket's Petrel |
| 4 | Lillie's Determination |
| 4 | Spikemuth Gym |
| 3 | Rare Candy |
| 3 | Night Stretcher |
| 2 | Boss's Orders |
| 1 | Unfair Stamp |
| 1 | Pokégear 3.0 |
| 1 | Tool Scrapper |
| 1 | Dawn |

Fresh behavior cloning is already contradicted by current official evidence.
The public Rmy Grimmsnarl replay hybrid was trained from 125 recent replays but
its submission `54911691` reached only `743.5`, while Rmy's two current
submissions were observed at `1113.8` and `1109.9`. Therefore action imitation
is not the primary policy or promotion signal.

## Scope

The implementation will:

1. Freeze a reproducible 2026-07-22 through 2026-07-23 live-data snapshot.
2. Download official replays for the current rank-one submission and a small
   set of other current high-rating submissions needed to train a visible-state
   value model.
3. Extract exact visible observations, legal options, actual actions, results,
   and offline-only engine restoration data.
4. Use the official engine to branch current real ladder states at important
   single-choice decisions.
5. Train a deck-specific action ranker from branch outcomes.
6. Package one safe challenger and, only if all gates pass, submit it once in
   place of v234.

The implementation will not:

- use v168, v11's historical `1032.0`, or other old ratings as training proof;
- claim that public agents are the current top-ten policies;
- attempt to select Kaggle opponents;
- use local tournament win rate as a Kaggle score prediction;
- train a direct behavior clone as the final driver;
- implement full MCTS in the first challenger;
- create or submit multiple competing candidates.

## Architecture

### 1. Live Snapshot Collector

The collector records the leaderboard timestamp, team IDs, submission IDs,
ratings, episode IDs, replay hashes, and exact deck lists. The snapshot is
immutable after training begins. Every derived dataset stores the source
submission ID and episode ID.

Only episodes created on 2026-07-22 or 2026-07-23 are eligible for model
training or evaluation. Older code may be reused as infrastructure, but older
game outcomes and ratings may not be used as evidence.

### 2. Replay-State Extractor

For every decision, the extractor stores:

- the acting player's legal observation only;
- every legal option, including target serial, area, index, card ID, attack ID,
  effect, and selection context;
- the action taken and final game result;
- the episode time, player seat, opponent, and deck fingerprint;
- the replay's engine restoration payload in a separate offline-only record.

`visualize.current` and hidden hands or decks are forbidden as model features.
They may be used only to reproduce an exact historical state inside the
offline official engine.

Episodes are split chronologically. No decision from one episode may appear in
more than one of train, validation, and final holdout.

### 3. Visible-State Value Model

A small model `V(s)` predicts the final win probability from information that
would have been visible to the acting player at state `s`.

Training examples come from both seats of current high-rating official games.
The model uses card IDs, public board state, hand contents for the acting
player, prizes remaining, energy, damage, discard, turn flags, recently
revealed opponent cards, and seat. It must not use the opponent's hidden hand,
deck order, prizes, or any replay-only visualization field.

The final holdout is the newest chronological slice and remains unopened until
model selection is complete.

### 4. Official-Engine Branch Labeler

For each important `MAIN`, single-choice decision made by the rank-one deck:

1. Restore the exact historical state in the official engine.
2. Verify that applying the recorded action reproduces the next replay state.
3. Apply each alternative legal action from an independent copy of the same
   root state.
4. Use the same fixed deck order and the same explicit manual coin schedule
   for every branch from that root.
5. Resolve forced follow-up selections and stop at the next meaningful main
   decision or turn boundary.
6. Score each resulting visible state with `V(s)`.

Branches are evaluated in both forward and reversed option order. Any result
that changes with branch order is discarded as an infrastructure error.

This produces a direct action-value training set. The label is the downstream
value of an action, not whether the rank-one player selected it.

### 5. Runtime Driver

The submitted agent contains:

- a deterministic legality and forced-selection safety shell;
- a compact option-ranking model using only the legal runtime observation;
- a confidence gate that keeps the safe action unless the proposed override
  has a material and independently confirmed value advantage;
- a deterministic fallback for missing features, model errors, or low time.

The agent does not need replay visualization or internet access at runtime. It
does not run full MCTS. Setup, forced selections, and unsupported contexts stay
under the safety shell.

## Gates

No Kaggle submission is made unless all gates pass:

1. **Snapshot integrity:** every training row resolves to a frozen current
   episode and deck fingerprint.
2. **Replay fidelity:** the recorded root action reproduces the next replay
   state for every state admitted to branch training.
3. **Branch independence:** forward and reverse branch order produce the same
   result.
4. **No hidden leakage:** an automated feature audit finds no replay-only or
   opponent-hidden field in model inputs or the packaged runtime.
5. **Value quality:** on the unopened chronological holdout, `V(s)` must have
   ROC AUC of at least `0.65` and expected calibration error no worse than
   `0.08`.
6. **Conservative overrides:** an action may replace the safety-shell action
   only when two independently trained value models agree and the smaller
   estimated advantage is at least `0.10`.
7. **Runtime safety:** 1,000 official-engine smoke games complete with zero
   illegal actions, agent exceptions, or timeouts. These games validate safety
   only and are not a rating prediction.
8. **Runtime budget:** single-decision p99 latency is below `500 ms` on one
   Linux CPU core.

Failure of any gate ends this candidate without consuming the v234 slot.

## Online Experiment

The two tracked slots are used as follows:

- Control: current v11 submission `54600199`
- Challenger slot: replace v234 submission `54592322`

Exactly one challenger archive is submitted. Kaggle chooses all opponents.
There is no claim that the control and challenger face identical opponents.
The official skill rating is the decision metric.

The challenger is allowed to collect at least 200 completed public episodes
before a verdict unless it errors. Its result is reported against:

- the control's current official rating;
- the live rank-ten threshold at verdict time;
- its own matchup and seat splits from official episodes.

No code change or second challenger is introduced while this measurement is in
progress.

## Deliverables

- immutable live-snapshot manifest;
- downloaded replay index with hashes;
- leakage-safe state and option datasets;
- value-model and branch-label training reports;
- packaged agent source, model, exact deck, and official engine;
- pre-submit audit report;
- one Kaggle submission reference if and only if all gates pass;
- official post-submission episode report.

## Known Limitation

The rank-one source code and model weights are private. This design improves a
policy from the consequences of current real states; it does not claim to
recover Luca's exact internal algorithm. The only authoritative test of the
finished challenger remains Kaggle's automatic matchmaking.
