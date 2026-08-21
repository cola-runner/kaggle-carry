# Grimmsnarl Direct-Teacher Context Policy Design

## Objective

Replace the rejected multi-teacher behavior clone with a policy trained on the
current strongest verified driver of the exact 60-card Grimmsnarl deck.

The deck remains frozen at fingerprint `b8f251a476e7`. The primary teacher is
Dries @ Tufa Labs, submission `55002825`, which was rank 3 at `1173.5` when
selected. The first online candidate, submission `55069325`, is retained only
as a negative control; its latest observed score was `685.3`.

No second Kaggle submission is allowed until a later, non-overlapping replay
window passes the gates in this document.

## Evidence Behind the Change

On replays created after the original snapshot cutoff:

- the rejected candidate won 17 of 37 identifiable non-mirror games;
- the direct teacher won 44 of 73 identifiable non-mirror games;
- the rejected candidate matched the direct teacher on 63.8% of non-forced
  decisions;
- agreement was 58.6% for main actions, 36.9% for bench placement, 26.1% for
  card search, and 0% for discard choices.

The exact deck is therefore still competitive. The failure is in policy
representation and teacher selection, not deck construction or engine
compatibility.

## Considered Approaches

### 1. Direct-teacher, context-specific models — selected

Train separate policies for main actions and the three failed follow-up
contexts. Add the identity of the effect that caused a selection, categorical
card identities, and effect-option interactions.

This directly addresses the measured failure while preserving the existing
legal-action fallback for unsupported cases.

### 2. Hard-coded search and discard tables

This is fast and transparent but cannot condition on board, hand, prizes,
matchup, or turn state. It would repeat the stale-rule failure that the rolling
teacher route was intended to avoid.

### 3. Monte Carlo tree search

This remains unsuitable for the immediate recovery because stochastic engine
branches are not order-independent, while the current failure is already
localized to observable choice semantics. It adds compute and correctness risk
without first fixing the missing information supplied to the policy.

## Data Contract

Create an immutable direct-teacher snapshot from submission `55002825`.

- Verify the teacher by both team name and exact deck fingerprint.
- Use only completed public episodes no newer than the frozen cutoff.
- Retain the existing 72-hour rolling source window.
- Split chronologically:
  - train: oldest 48 hours;
  - validation: following 12 hours;
  - audit: newest 12 hours.
- Never select hyperparameters using the audit split.
- Treat the already inspected audit window as diagnostic rather than a final
  blind test.
- Reserve all episodes after the new snapshot cutoff for the final prospective
  gate.

Replay identity is `episode_id:teacher_seat`; exact-deck mirrors must use the
teacher name to choose the correct seat.

## Feature Contract

All features must be visible to the acting player.

Add the following to the existing visible state and option features:

- categorical selection context and selection type;
- `select.effect` card identity;
- `select.contextCard` card identity;
- categorical option type, source card, target card, and attack identity;
- interaction keys for context × effect × option source;
- interaction keys for context × option source;
- current hand, board, prize, turn, and opponent visible board features already
  present in the runtime.

Card IDs must not be represented only as ordered numeric values. The runtime
and training feature functions must remain byte-for-byte equivalent.

No replay rewards, hidden hands, deck order, future actions, or restoration
state may enter the policy features.

## Model and Runtime Design

Train two independently seeded exported tree rankers for each supported mode:

- `main` for context 0;
- `bench` for context 5;
- `search` for context 7;
- `discard` for context 8.

For single-choice decisions, average the two rankers by semantic option and
choose the highest unique option.

For multi-choice decisions:

1. average the two option scores;
2. use a validation-frozen threshold for that context;
3. select all semantic options above the threshold;
4. clip the result to `minCount..maxCount` using score order;
5. map semantic choices back to legal option indices deterministically.

Unsupported contexts continue through the previous legal fallback. Forced
choices remain deterministic and do not use a model.

The new runtime must catch model errors and return a legal fallback action. It
must not import project training code, use network access, or depend on native
machine-learning libraries.

## Gates

Validation and audit reports must show:

- no holdout/audit label access during training;
- main exact semantic accuracy at least 65%;
- search exact-set accuracy at least 55%;
- bench exact-set accuracy at least 55%;
- discard exact-set accuracy at least 40%, unless fewer than 30 eligible audit
  decisions exist, in which case discard remains on fallback;
- overall non-forced semantic exact-set accuracy at least 72%;
- improvement of at least 8 percentage points over the rejected candidate on
  the same direct-teacher decisions.

Runtime gates:

- exact frozen deck fingerprint;
- zero illegal actions, crashes, and timeouts in 1,000 official-engine games;
- p99 decision latency below 500 ms;
- the packaged archive must run independently.

Prospective gate:

- wait until at least six hours after the new snapshot cutoff;
- collect at least 40 new direct-teacher public episodes;
- do not tune on them;
- require the same context gates, allowing a five-point sampling tolerance;
- do not use the second Kaggle submission if the prospective gate fails.

## Outputs

- immutable direct-teacher snapshot and replay inventory;
- context-specific training and audit reports;
- standalone candidate directory with frozen model hashes;
- independently verified submission archive;
- prospective comparison report with a final `SUBMIT` or `REJECT` decision.
