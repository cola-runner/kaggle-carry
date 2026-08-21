# Pokémon Four-Policy League Mac Validation

## Goal

Use one Mac to answer one question: can four independently trainable Pokémon
TCG policies become stronger as a group after two rounds of playing and
learning from one another?

This validation does not target the Kaggle leaderboard. Passing allows the
same league design to move to the GTX 1080 Ti. Failing stops that move and
produces a diagnostic report.

## Non-Negotiable Self-Play Contract

The league has four independently trainable policy/value models:

- Grimmsnarl;
- Mega Lucario;
- Crustle;
- Alakazam.

A run counts as self-play only when all of the following are true:

- current trainable policies play one another for most training games;
- a game between two current policies produces trajectories and updates for
  both participants;
- all four current policies receive an update in every league round;
- frozen historical policies may play but may not update;
- fixed rule-based agents never supply reinforcement-learning trajectories;
- driver imitation is disabled before the first league game.

If any condition is false, the runner must reject the experiment as
`INVALID_SELF_PLAY` rather than report a training result.

## Architecture

Each participant owns a separate approximately 1.5-million-parameter neural
policy/value model. The models share one visible-information feature contract
and one NumPy-compatible inference implementation, but do not share weights.
Learning one deck therefore cannot directly overwrite another deck's policy.

The policy controls every legal selection after league play begins. It samples
distinct options without replacement and supports a learned stop decision for
variable-cardinality selections. A legal fallback may prevent an engine crash,
but using it marks the validation as failed; a rule-based teacher may not
silently take control.

## One-Time Start

The four existing drivers provide compact action examples only to initialize
their corresponding neural policies. This stage ends before self-play and is
not counted as a league round.

The start gate checks that every model can finish complete games from both
seats without illegal actions. Once the gate opens, driver labels and teacher
actions are unavailable to training.

## Two-Round League

Round 1 freezes the four starting policies, schedules balanced cross-deck
matches from both seats, records both sides of every game, and then updates all
four policies simultaneously.

Round 2 freezes the four Round-1 policies. Most games again match current
policies against one another and update both sides. A smaller portion matches
current policies against frozen earlier snapshots to expose forgetting and
cyclic strategies. After the entire round, all four current policies update
simultaneously.

The only optimization reward is the terminal result: win `+1`, draw `0`, loss
`-1`. Prize difference, damage, and board material may be diagnostics but may
not alter the reward.

The runner chooses exact game counts from a fixed Mac resource budget during
preflight. It records and freezes the schedule before training starts; counts
cannot be changed after any evaluation result is observed.

## Proving Group Improvement

Internal league win rate cannot prove group improvement because every internal
win is another member's loss. The validation therefore freezes the starting
group and compares both the starting and final groups against the same external
judge pool.

The judge pool contains four agents and deck styles that provide no training
games, labels, tuning feedback, or early-stopping signals. The final group must
also play the frozen starting group directly.

The league passes only when:

- the final group's aggregate external-judge score improves with positive
  statistical evidence;
- the final group defeats the starting group head-to-head;
- no individual policy suffers a major external-judge collapse;
- all four policy parameter sets changed through finite updates;
- no illegal action, crash, hidden-information access, or teacher takeover
  occurred.

Exact numerical thresholds and sample counts are implementation decisions.
They must be declared in the frozen preflight configuration and included in
the report, never selected after seeing results.

## Storage and Failure Safety

Raw replays are never written. Compact trajectory shards live only under one
run-specific directory in macOS `$TMPDIR`, are consumed promptly, and are
deleted on success, rejection, exception, signal, or timeout.

The runner enforces hard limits for time, temporary disk, shard size, and
queued shards. A failed run retains no trained model. Persistent output is one
small JSON report containing the frozen schedule, cross-play results, judge
comparison, update statistics, cleanup verification, and final decision.

The existing `artifacts/` directory must have identical file count and byte
size before and after the run.

## Validation Evidence

Before the two league rounds, automated tests must prove:

- a current-current game records legal trajectories for both players;
- one simultaneous update changes all four policies;
- a fixed rule agent cannot enter the reinforcement-learning data path;
- teacher code cannot be called after the start gate;
- historical snapshots remain immutable;
- PyTorch and NumPy rank legal actions consistently;
- forced worker, numerical, quota, and timeout failures remove temporary data;
- the final decision rejects a synthetic non-improving population and accepts
  a synthetic improving population.

## Out of Scope

- changing any deck list;
- Kaggle submission or active-submission changes;
- copying a champion's hidden strategy;
- optimizing one policy against a fixed opponent set;
- moving training to the GTX 1080 Ti before the Mac validation passes.
