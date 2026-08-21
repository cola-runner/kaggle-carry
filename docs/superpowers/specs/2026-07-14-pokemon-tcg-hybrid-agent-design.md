# Pokémon TCG Hybrid Agent Design

Date: 2026-07-14

Status: Proposed for implementation

Primary objective: finish in the Strategy Category top 8, using a Simulation
Category top-10 position as the hard intermediate milestone.

## 1. Context

The Simulation Category is a partially observable stochastic game rather than
a conventional supervised-learning task. The agent sees its own hand, the
public board, logs, and legal options, but not the opponent's hand or either
deck order. Draws and coin flips add randomness, actions affect many future
turns, and the official rating uses only game outcomes.

The current workspace contains three important lessons:

1. A historical score is not a stationary baseline. As observed on 2026-07-13,
   the byte-identical v11 control scored 1032.0 historically but 799.2 after a
   current-meta resubmit.
2. Action imitation is not a reliable objective. v234 matched 82.1% of sampled
   non-forced replay actions yet scored 727.0 online.
3. Existing local pools are useful regression guards but can reverse the
   official ordering. They do not provide a trustworthy promotion signal for
   small win-rate differences.

The first implementation will therefore target one deck-specific pilot,
initially Archaludon/Cinderace because it has the deepest rule baseline and
official replay corpus in this workspace. A second deck is selected later from
the current meta rather than being fixed in this design.

## 2. Goals and non-goals

### Goals

- Optimize expected game outcome, not expert-action agreement.
- Preserve a deterministic, legal rule policy as the runtime safety baseline.
- Let a learned model propose a small number of potentially better actions.
- Verify proposed overrides with paired, bounded official-engine search.
- Model hidden information as a belief over plausible states and opponent
  archetypes instead of one guessed deck.
- Make every experiment reproducible from a source revision, data snapshot,
  configuration, artifact hash, and result record.
- Keep one current-meta control active while using the second active slot for
  a challenger.
- Produce defensible evidence for the Strategy Category report.

### Non-goals

- Training one universal policy for every legal deck in the first iteration.
- Replacing the rule policy with an end-to-end model.
- Treating local weighted win rate, replay agreement, or one-ply board value as
  a direct estimate of Kaggle rating.
- Running unbounded MCTS at every decision.
- Automatically submitting an agent without explicit user approval.

## 3. Decision architecture

The runtime principle is:

> Rules provide safety, the model proposes, and search verifies.

### 3.1 Deck-specific rule pilot

The rule pilot owns all mandatory selection contexts, legality handling, setup
sequencing, and the deck's basic win condition. It returns the default action
for every observation and remains callable independently of the learned model
and search API.

The initial experimental control is the byte-verified v11 submission. The
initial candidate policy base is the recovered v168 behavior because it is
more recent and already underlies the value-search prototype. Phase-one
evaluation must compare both on the same current replay window before v168 is
accepted as the candidate base.

### 3.2 Outcome-supervised proposal model

The proposal model ranks legal main-phase actions. It is not authorized to
return the final runtime action by itself.

Training rows are grouped by decision. Targets retain terminal outcomes and
continuous cutoff returns instead of silently dropping non-binary rollouts.
The training objective is pairwise or listwise within a decision so that a
state with many legal actions does not receive disproportionate weight.

Features include:

- public board, prizes, turn, seat, hand and deck counts;
- energy, damage, tools, evolution lines, and attack readiness;
- visible opponent cards and archetype posterior;
- rule rank and rule explanation category;
- one-step transition deltas after applying the candidate action;
- whether the action creates or destroys next-turn attack, recovery, retreat,
  or knockout routes;
- matchup interactions rather than only global hashed features.

The primary offline model metric is top-k proposal recall of the best tested
action and decision-level regret on opponent-and-time holdouts. Classification
accuracy and replay agreement are diagnostic metrics only.

To advance from model research to verifier integration, top-2 proposal recall
must improve by at least five percentage points over the rule policy's top two
actions on the frozen opponent-and-time holdout, and the 95% grouped-bootstrap
upper bound for proposal regret must be below the rule policy's mean regret.

### 3.3 Belief state and opponent model

Visible opponent cards update a posterior over current meta deck fingerprints
and archetypes. Each search determinization samples a deck hypothesis and then
samples hidden hand, deck, and prize cards consistent with public information.

The initial opponent-policy library contains lightweight deck-specific pilots
for the most frequent current archetypes. Unknown or low-confidence opponents
use a mixture of compatible policies. If neither the hidden-state sampler nor
opponent policy is trustworthy enough, runtime search declines to override the
rule action.

This is still an approximation: determinization can suffer from strategy
fusion, and public deck identity does not reveal private policy. Confidence
gating and conservative fallback are therefore part of the algorithm, not
only engineering safeguards.

### 3.4 Paired bounded verifier

Search considers a deliberately small action set: the rule action plus at most
two distinct model proposals. It runs only for high-leverage, single-choice
main decisions where the model and rule disagree.

For each sampled hidden state, all candidate actions start from the same root
and use the same determinization. Candidate evaluation order alternates or is
randomized, and each candidate receives the same transition or time budget.
This removes the current bias where the first branch can consume the shared
deadline.

For action `a`, the verifier estimates a bounded value:

```text
Q(a) = mean_k rollout_return(a, hidden_state_k)
```

It records paired differences against the rule action. An override is allowed
only when all conditions hold:

- at least the configured minimum number of paired samples completed;
- the lower confidence bound of the paired value difference exceeds the
  configured margin;
- no legality, search, model, or deadline error occurred;
- enough per-game time remains for the hard safety reserve.

Otherwise the rule action wins ties and uncertainty.

### 3.5 Runtime orchestrator

The orchestrator performs the following sequence:

```text
observation
  -> validate/select context
  -> rule action
  -> check search trigger and remaining budget
  -> model proposals
  -> belief sampling and paired verifier
  -> confidence gate
  -> validated final action or rule fallback
```

It emits compact local diagnostics but does not depend on writable storage or
network access during Kaggle execution.

## 4. Data and training flow

### 4.1 Data sources

- current official episodes for the live control and challengers;
- daily top-episode exports;
- exact public opponent policies where available;
- local official-engine self-play and counterfactual branches;
- deterministic replay fixtures for regression tests.

Every dataset receives a snapshot manifest containing collection time, episode
IDs, target submission IDs, deck fingerprints, source URLs or commands, and a
content hash. Raw competition data and credentials remain outside Git.

### 4.2 Split policy

Random row splits are prohibited. Evaluation uses all of the following:

- whole-game grouping so decisions from one game never cross splits;
- opponent-policy or opponent-team holdout;
- archetype-stratified reporting;
- forward time holdout representing the newest meta window;
- seat-stratified metrics.

This tests generalization to both unseen policy behavior and meta drift.

### 4.3 Counterfactual generation

Counterfactual data is generated only at non-forced, high-leverage decisions.
The rule action and candidate actions share the same sampled hidden state.
Terminal results remain `0`, `0.5`, or `1`; truncated rollouts retain their
bounded continuous estimate and an explicit `truncated` flag.

Rows from the same decision are kept together and reweighted as one decision
group. Training does not pretend correlated rollouts are independent games.

## 5. Evaluation and promotion gates

### Gate 0: security, package, and runtime

- no credentials, tokens, network calls, or unexpected files in the archive;
- exactly 60 legal cards and a valid deck fingerprint;
- source and archive hashes match;
- compilation, model loading, action validation, and both-seat smoke pass;
- missing-model, missing-search, and timeout paths return the rule action.

### Gate 1: deterministic replay regression

- forced and auxiliary selection behavior remains byte-for-byte equivalent to
  the rule pilot unless the change is explicitly in scope;
- all intended main-action differences are recorded with reason codes;
- no action changes occur outside registered trigger families.

### Gate 2: paired counterfactual evidence

- at least 200 independent decision groups across at least five current
  archetypes for a broad promotion claim;
- a focused matchup claim requires at least 30 independent decision groups
  from at least two opponent policies in that matchup, and zero triggers
  outside its declared predicate on the frozen broad replay fixture;
- the 95% paired-bootstrap lower bound for the proposed override is above zero;
- no core archetype with at least 30 decision groups has a paired point-estimate
  regression greater than five percentage points. Less-covered archetypes are
  labeled unknown and cannot support a broad promotion claim.

### Gate 3: current-meta local guard

The candidate and control are evaluated by seat and opponent using common
initial hidden states wherever the engine API permits. Where exact full-game
pairing is unavailable, the result is labeled unpaired and cannot independently
promote a candidate. Pool results are reported per opponent and seat, not only
as one aggregate binomial proportion.

### Gate 4: online active-control experiment

Only one variable changes between control and challenger. One of the two active
Kaggle slots remains the current control; the other holds the challenger. A
third upload is not made until the effect of retiring the older slot is
explicitly reviewed.

The online report uses raw episodes, opponent rating/archetype, seat, and time
window. Promotion requires no runtime failures, at least 100 completed
episodes, and a 95% opponent-adjusted bootstrap interval whose lower bound is
non-negative. If the scheduler does not produce 100 episodes, the result stays
`HOLD`; a dynamic displayed rating alone is not sufficient evidence.

The two slots are a deployment hedge, not an ensemble: Kaggle does not choose
between our agents per match or combine their ratings. The anchor preserves a
known live reference while the challenger tests a specialized response to the
current field.

## 6. Failure handling and budgets

Runtime behavior is fail-closed:

- invalid observation or unsupported selection context: use the rule pilot;
- missing or corrupt model: disable proposals for the rest of the episode;
- unavailable search API: use the rule pilot;
- unknown opponent or inconsistent hidden-state reconstruction: skip override;
- search exception or incomplete paired samples: discard the search result;
- low remaining time: permanently disable nonessential search for the episode;
- invalid proposed action: reject it before engine submission;
- repeated verifier failures: trip an episode-local circuit breaker.

Search has both a per-decision deadline and a per-game reserve. Concrete values
are benchmarked in Phase 1; the current 0.12-second prototype is a starting
measurement, not a fixed requirement. The shipped configuration must keep a
large margin below the official 600-second player budget.

## 7. Test strategy

### Unit tests

- legal-action normalization and fallback;
- feature determinism and model export parity;
- hidden-card accounting and belief normalization;
- paired candidate scheduling and equal-budget accounting;
- continuous target retention and decision-group weighting;
- confidence-bound and promotion calculations;
- experiment-manifest and archive secret scanning.

### Integration tests

- replay a frozen set of official observations through control and candidate;
- branch multiple actions from an identical search root and determinization;
- run complete matches from both seats against representative archetypes;
- load the extracted submission in an isolated Kaggle-like directory;
- inject missing files, malformed model data, search exceptions, and deadline
  exhaustion and verify rule fallback.

### Statistical tests

- synthetic paired outcomes with known positive, zero, and negative effects;
- bootstrap coverage and grouping by decision rather than rollout row;
- invariance to candidate evaluation order;
- separate reporting for seat, archetype, opponent, and time window.

## 8. Reproducibility and repository structure

Repo-owned source, tests, configurations, and small fixtures become tracked.
Downloaded agents, raw episodes, generated datasets, trained artifacts, and
submission archives remain ignored and reproducible from manifests.

Each experiment record contains at minimum:

- experiment ID and timestamp;
- Git commit;
- control and candidate identifiers;
- source, deck, model, and archive hashes;
- dataset snapshot hash and split definition;
- exact command and configuration;
- random seeds or determinization IDs;
- per-opponent and per-seat results with confidence intervals;
- Kaggle submission reference and online episode window when applicable;
- decision: reject, hold, package, or request submission approval.

The stale project README is updated as part of implementation, but unrelated
workspace changes are preserved.

## 9. Security prerequisite

Local transfer archives and credential files must not be committed or
transferred. Before publication, the repository is checked for credential
files and common secret formats; authentication remains in user-level tooling
outside the project tree.

## 10. Implementation phases

1. **Foundation:** security guard, tracked source boundary, experiment manifest,
   tests, and paired evaluation primitives.
2. **Current baseline:** refresh current replays; compare v11, recovered v168,
   and value-search v2 under the corrected evaluator; select one candidate base.
3. **Proposal model:** retain continuous outcomes, add transition and matchup
   features, use grouped opponent/time splits, and measure proposal regret.
4. **Verifier:** add archetype belief, deck-specific opponent policies,
   equal-budget paired search, confidence gating, and circuit breakers.
5. **First challenger:** package only after Gates 0-3 pass; request explicit
   approval before Kaggle submission; evaluate beside the live control.
6. **Second deck and strategy evidence:** choose a complementary current-meta
   deck from official matchup evidence, repeat the same pilot pipeline, and
   maintain a report-ready experiment narrative.

## 11. Success criteria

The technical design is successful when:

- local promotion decisions no longer depend on unpaired aggregate noise;
- learned proposals reduce decision-level regret on opponent-and-time holdouts;
- runtime overrides occur only with paired positive evidence and never create
  invalid actions or timeouts;
- an online challenger beats or safely replaces the current live control under
  a documented current-meta comparison;
- the team reaches the moving Simulation top-10 threshold and has a complete,
  reproducible Model/Deck/Report case for Strategy top-8 evaluation.

The last criterion is an objective, not a guaranteed outcome. Negative results
and rejected candidates remain part of the evidence trail.

## 12. External references

- Simulation evaluation:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/evaluation
- Strategy evaluation:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/overview/evaluation
- Official challenge site:
  https://ptcg-abc.pokemon.co.jp/
- CABT SDK documentation:
  https://matsuoinstitute.github.io/cabt/
