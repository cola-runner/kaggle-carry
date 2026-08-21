# Pokémon Single-Intervention Self-Play Design

## Outcome

Test whether four driver-backed players can improve as a population using only
complete Engine 22 games. Training and evaluation must not call `search_begin`,
fork a state, or treat simulated branch outcomes as action labels.

The Mac proof passes only if two training rounds produce a population that
beats the untouched four-driver population in both a selection batch and a
fresh confirmation batch. A failed proof retains no model and authorizes no
Kaggle submission.

## Why the Previous Method Failed

The previous residual PPO run made about 259 exploratory changes in 24 games,
so one terminal result was attributed to roughly ten changed decisions per
game. Its GAE setting also strongly discounted early decisions in games that
often contain more than one hundred choices. The later A/A gate showed that
independent terminal search branches agree on the winner only 8 of 12 times.

The new experiment removes both sources of ambiguity: one intervention per
game and no search branches.

## Player and Model

- Keep the four existing decks and drivers unchanged as incumbents.
- Reuse the current visible relational option features.
- Use two independently seeded copies of the existing approximately
  66-thousand-parameter action scorer per member rather than a Transformer or
  recurrent model for this feasibility proof.
- The learned component is residual: the driver action remains the default.
- A learned action may replace the driver action only when both scorers prefer
  it to the driver by a logit margin greater than `0.25`. Otherwise the driver
  acts.

Before each round's interventions, select at most 2,048 incumbent decisions per
member from calibration and train both scorers for two epochs to rank the
incumbent action above every other legal action. This avoids obviously
nonsensical alternatives. The initialization only shapes exploration; it is
not accepted as evidence of improvement and cannot replace the incumbent by
itself.

## Real-Game Credit Assignment

Each intervention game designates one experimental member. Until intervention,
both players follow their current incumbent policies. At the start of a game,
draw one target eligible-decision ordinal uniformly from 1 through 32. If the
game reaches that non-forced `MAIN` decision, the experimental member samples
exactly one legal action different from its incumbent action. From the next
decision until the game ends, both incumbents handle the changed state normally.

No second intervention is allowed in that game. Games in which no eligible
decision is reached are recorded as controls and do not become action labels.

At the start of each round, use the 12 logical member-opponent pairings, both
seats, and two repetitions: 48 calibration games total. Freeze the incumbent
score for each member-opponent-seat cell as `(points + 1) / (games + 2)`, which
shrinks its two-game sample toward 0.5. The intervention label is:

`actual terminal score - calibrated incumbent score`

The scorer learns a pairwise preference between the tried action and the
incumbent action. Positive labels raise the tried action relative to the
incumbent; negative labels lower it. Label magnitude is clipped to `0.5`, and
matchup, seat, target ordinal, and round are retained for audit. Raw
observations and hidden cards are never persisted.

The alternative is sampled at temperature `1.0` from the mean score of the two
models after removing the incumbent action. For each non-zero label `y`, both
models minimize `softplus(-sign(y) * (trial_score - incumbent_score))`, weighted
by `max(abs(y), 0.05)`. Use AdamW at `3e-4`, a batch size of 32, and eight
epochs. The two models receive different initialization and batch-order seeds;
all other inputs are identical.

## Two Population Rounds

Round 1 starts from the untouched drivers. Round 2 starts each member from its
round-1 survivor: a promoted residual if it passed selection, otherwise the
untouched driver. Every member must receive 32 valid single-intervention games
per round, balanced across the other three population members and both seats.

After each round, train only from that round's interventions plus the member's
bounded in-memory calibration examples. A member with missing samples,
non-finite updates, no action different from its incumbent, or an illegal
action is automatically rolled back.

## Actual-Game Promotion Gate

Evaluation uses ordinary `battle_start`/`battle_select` games with exploration
disabled. Search APIs are forbidden.

For each member, compare its candidate and incumbent against the other three
round incumbents, both seats, and two repetitions using independent complete
games. Selection therefore uses 12 games per member for each side. Only
members with positive candidate-minus-incumbent score enter the provisional
final population; all others roll back.

A fresh schedule compares each provisional member and its untouched original
driver against the other three untouched original drivers, both seats, and two
repetitions. The proof requires:

- positive group delta in selection;
- positive group delta again in fresh confirmation;
- every promoted member positive in both batches;
- at least one promoted member and at least one learned override;
- zero engine, validity, timeout, storage, or cleanup failures.

The Mac proof is a directional feasibility gate, not a statistical leaderboard
claim. A pass authorizes a larger GTX 1080 Ti run with enough games for formal
confidence intervals.

## Bounded Execution and Storage

- Official local Engine 22 SHA-256 must match the already validated binary.
- Hard Mac wall-time limit: 10 minutes.
- At most 32 valid interventions per member per round.
- No replays, raw state dumps, datasets, or rejected checkpoints are written.
- Transient tensors live in one owned system-temporary directory capped at
  128 MiB and are deleted on success, rejection, exception, signal, or timeout.
- `artifacts/` must have identical file count and byte size before and after.
- Retain one JSON report below 100 KiB. Retain model weights only after a fresh
  confirmation pass.

## Required Tests

- A game cannot contain more than one intervention.
- The intervention action must be legal and differ from the incumbent action.
- Post-intervention actions return to the incumbent policy.
- Terminal credit is attached only to the designated intervention.
- Matchup and seat calibration produce the frozen centered label.
- Round-2 membership uses promoted survivors with per-member rollback.
- Promotion and confirmation consume disjoint actual games and never import or
  call a search API.
- Timeout, exception, rejection, and signal paths remove the owned temporary
  directory without changing `artifacts/`.
