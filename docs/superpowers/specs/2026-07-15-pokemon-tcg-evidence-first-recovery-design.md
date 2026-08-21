# Pokémon TCG Evidence-First Recovery Campaign

Date: 2026-07-15
Status: Approved; amended with the reality-seeded deck decision
Decision owner: Codex, with the user retaining approval over external Kaggle submissions

This document supersedes the execution strategy in
`2026-07-14-pokemon-tcg-hybrid-agent-design.md`. The earlier document remains a
useful architecture reference, but its permanent-online-control assumption is
invalid under Kaggle's latest-two-active-submissions rule.

## 1. Decision

The campaign will use an evidence-first policy process on top of
reality-seeded decks. Real tournament results supply the deck priors; the
competition engine decides whether a faithful transfer or a small adaptation
survives.

- Measurement route: keep the recovered Archaludon family only as the
  instrumentation-rich control for evaluator calibration. Its historical
  rating and engineering maturity do not make it the default competition deck.
- Primary competition route: freeze the exact 60-card Dragapult/Dudunsparce
  list that won Regional Prague on 2026-04-25, then select and improve a
  faithful deck-specific policy without changing that deck.
- Independent challenger: freeze James Kowalski's exact 60-card Lillie's
  Clefairy list that won NAIC 2026. It begins with its own deterministic policy
  and receives no evidence from a different Clefairy list or policy template.
- Conditional specialist: Crustle/Mega Kangaskhan remains a later anti-meta
  option. Its published list is not an exact transfer because one card is
  absent from the competition pool, so it cannot preempt the two exact-transfer
  candidates.
- Policy improvement route: after a same-deck deterministic control freezes, a
  learned model may propose one alternative and an equal-budget official-engine
  verifier may override only through the gates in this document.
- Rejected deck route: unconstrained deck construction from zero. Deck search
  is reduced to preregistered replacements around a proven exact-60 seed.
- Rejected modeling route: a from-scratch end-to-end AlphaZero/MuZero campaign.
  Self-play and outcome supervision are retained, but the network does not own
  the complete policy during this competition window.
- Rejected artifact: the current `value-search-v2` is not submitted or patched
  in place. Its useful pieces may be reused only behind new evaluation
  contracts.

The objective is to maximize the probability of finishing with two defensible,
strong submissions by 2026-08-10, leaving six days before the official
2026-08-16 deadline and the post-deadline convergence period. Top 10 remains a
target, not a promised result. The two artifacts may share a deck only if a
same-deck policy improvement passes; diversity is useful but cannot replace
evidence.

## 2. Why this is the chosen route

The observed failures are measurement failures before they are modeling
failures:

1. The byte-identical v11 scored 1032.0 historically and about 799.4 after a
   current-meta resubmission. Historical ratings are not stationary controls.
2. Nine local pools reversed the official v11/v19 ordering. Unpaired aggregate
   local win rate cannot promote a candidate.
3. v234 reached about 82.1% replay action agreement without becoming a strong
   online agent. Imitation accuracy is not the competition objective.
4. The latest `value-search-v2` run produced only three net wins across 2,340
   unpaired games, with a risk-pool regression and seat-dependent gains.
5. Its generated continuous values and draws are discarded by the current
   training loader, training and runtime use different opponent policies, and
   its verifier gives the first branch more of the shared deadline.
6. Copying a strong list is not sufficient: the prior Team Rocket attempt used
   a high-ranked exact deck, public episodes, and about 82.1% action imitation,
   yet produced a weak online result. A tournament deck saves deck-discovery
   time; it does not supply a faithful agent policy.
7. Conversely, the exact NAIC Clefairy list has no exact-match games in the
   captured Kaggle replay manifests. Nearby Clefairy lists and synthetic policy
   templates cannot be treated as evidence for it.
8. Dragapult has both repeated real-tournament success in the closest available
   card format and an existing competition-native policy family. The captured
   public Kaggle list is not the Prague exact-60 and its old replay outcomes are
   confounded, but they establish that a deployable Dragapult policy is a lower
   risk starting point than a policy-free archetype.

More compute applied to those contracts would amplify bias. The first product
of this campaign is therefore a calibrated decision instrument, not a larger
model.

## 3. Scope and non-goals

### In scope

- Review and integrate the existing evaluation-foundation worktree.
- Build a real decision-root paired evaluator with search enabled.
- Freeze a reproducible Archaludon measurement control for evaluator
  calibration.
- Materialize source-bound exact-60 manifests for Prague Dragapult/Dudunsparce
  and NAIC Lillie's Clefairy, then validate every mapped card in the competition
  engine.
- Freeze a faithful deterministic control for each deck before improving it.
- Compare deck changes with policy held constant and policy changes with deck
  held constant.
- Generate outcome-supervised, on-policy counterfactual data.
- Train a compact proposal ranker that can be exported to dependency-free
  runtime inference.
- Implement equal-budget, margin-gated verification with fail-closed rule
  fallback.
- Run the exact NAIC Clefairy challenger independently from Dragapult.
- Admit an adapted specialist only through the frozen criteria in Section 12.
- Run one planned control/challenger comparison cohort, followed only by the
  fixed final-slot adjustment in Section 13.

### Out of scope

- A universal policy for arbitrary decks.
- Random or evolutionary deck generation from the full card pool.
- Calling a modified or merely similar list an exact tournament transfer.
- Transferring action evidence, model evidence, or win rates across decks.
- A neural network that directly returns every runtime action.
- Full belief-state research before the simpler verifier proves useful.
- Treating public rating, replay agreement, board score, or unpaired pool win
  rate as a sufficient promotion signal.
- Automatic Kaggle submission.
- Repeatedly modifying a candidate after inspecting the same shadow holdout.

The next implementation plan covers the source-bound exact-60 manifests and
card/effect validation, policy-family inventory, evaluation foundation,
calibrated decision-root evaluator, and preregistered randomized full-game
runner. Target-policy construction, model/runtime work, conditional specialist,
and online-cohort work receive separate plans only after that foundation passes.
This keeps the first implementation independently testable and prevents a
speculative policy or model from defining its own test.

## 4. Campaign architecture

```text
official episodes + local on-policy games + fixed opponent zoo
                         |
                         v
              versioned root-state snapshots
                         |
                         v
       paired decision evaluator and calibration controls
                         |
             +-----------+-----------+
             |                       |
             v                       v
   target-deck policy work    reality-seeded deck track
             |                       |
             v                       v
 equal-budget runtime verifier  deterministic rule pilot
             |                       |
             +-----------+-----------+
                         |
                         v
       package/runtime gates -> online cohort -> minimal slot transition
```

Each box has a single contract. Data collection does not calculate promotion;
the evaluator does not train a model; the model cannot return a final action;
the verifier cannot package or submit; packaging cannot call Kaggle.

## 5. Frozen controls and competition baselines

### 5.1 Archaludon measurement control

Archaludon remains the evaluator-calibration control because the workspace has
the deepest rule policy, replay corpus, and functioning official-engine search
for that deck family. It is no longer the presumed competition primary.

The frozen measurement control starts from the current recovered-v168 directory
and deck, captured under a new content hash. It is not claimed to be
byte-identical to historical v168 and does not inherit the historical score. A
minimal search-lifecycle refactor is allowed because the recovered control does
not guarantee a clean `search_end()` boundary on every successful path. The
refactor must preserve semantic actions on a frozen golden corpus; any action
drift fails the control freeze.

Passing calibration on Archaludon proves that the instrument can detect known
effects. It supplies zero promotion evidence for a Dragapult or Clefairy policy.
Target-deck gates must use target-deck roots, target-deck controls, and
target-deck full games.

### 5.2 Primary exact-60: Prague Dragapult/Dudunsparce

The source of truth is Mateusz Łaszkiewicz's first-place Regional Prague list
from 2026-04-25: 19 Pokemon, 32 Trainer cards, and 9 Energy cards. It was played
in the TEF-POR format that most closely matches the frozen competition card
pool. Research found a competition-card mapping for all 60 cards; Gate A must
independently verify the IDs, printed effects used by the engine, legality, and
the normalized multiset before policy testing begins.

The deck manifest binds the source URL, a captured source record, normalized
card names and counts, competition card IDs, a 60-line canonical deck file, and
both ordered-file and multiset hashes. At most three code-hash-distinct existing
Dragapult policy families enter a preregistered same-deck screening tournament.
Merely copying the exact deck into an archetype template does not create a
faithful policy. Here, `faithful` means the policy binds the exact deck hash,
contains no rule for an absent card, passes scripted fixtures for every
deck-specific attack, Ability, and selection context it can invoke, and
completes both-seat games without falling through to an unrelated archetype's
logic. The selected deterministic policy becomes the candidate primary
competition control only after it completes both seats without invalid actions,
timeouts, or deck-policy contradictions; Gate E decides whether it becomes a
finalist.

### 5.3 Independent exact-60: NAIC Lillie's Clefairy

The source of truth is James Kowalski's first-place NAIC 2026 list: 22 Pokemon,
28 Trainer cards, and 10 Energy cards. Research found a full competition-card
mapping, but the exact list has no exact-match evidence in the captured Kaggle
replay manifests. It therefore begins as a strong real-world prior, not as a
locally proven deck. NAIC also included the newer Chaos Rising set while the
competition pool stops at Perfect Order; the chosen 60 cards map, but the event's
matchup distribution is not assumed to transfer.

The workspace's existing `candidate_lillie_clefairy/deck.csv` is explicitly not
the baseline: it replaces the source list's one Chien-Pao with another Lillie's
Pearl. The candidate must point to a byte-verified copy of the exact source deck
before any result is accepted. Clefairy receives its own policy, roots, model,
runtime report, and full-game manifest. Dragapult or synthetic big-basic results
have zero promotion weight for it.

### 5.4 Same-deck fail-closed action contract

For either competition deck, the challenger computes and stores its frozen
same-deck rule action before invoking any new model or verifier. Model,
hidden-card accounting, search, or deadline failure returns that stored action.
No exception path may sample a random legal action.

v11 remains the live-meta anchor, package/runtime regression guard, and final
fallback. v161, historical v168, and public policies may propose implementation
ideas only after their source is hashed; their old ratings and different-deck
outcomes are never inherited.

## 6. Root-state and evidence contracts

### 6.1 Root collection

Decision roots come from separately labeled sources:

- Archaludon measurement-control games used only for evaluator calibration;
- exact-60 Dragapult control games;
- exact-60 Dragapult challenger games, preventing baseline-only state coverage;
- exact-60 Clefairy control and challenger games in a separate deck namespace;
- conditional-specialist games used only for meta coverage until that deck
  independently qualifies; and
- recent official observations used as policy/regression fixtures.

Primary paired evidence requires a valid `search_begin_input`. Local roots have
this opaque official-engine state string and are evaluated immediately or saved
outside Git with competition-data controls. Public replay observations normally
do not preserve it; such observations may test policy behavior or seed a local
scenario family, but they cannot enter Gate C unless `search_begin` is actually
replayed successfully from the recorded root.

Only non-forced, single-choice main-phase decisions enter model evaluation.
Forced selections and auxiliary contexts remain deterministic regression
fixtures.

Every root artifact, stored outside Git, contains:

- the raw observation including the complete `search_begin_input`;
- that seat's ordered active-observation prefix from deck initialization through
  the root;
- legal options plus their semantic action mapping;
- source episode, step, seat, turn, and collection batch;
- agent, deck, opponent, cg API, native binary, and platform hashes;
- for every matched determinization, the ordered `your_deck`, `your_prize`,
  `opponent_deck`, `opponent_prize`, `opponent_hand`, and `opponent_active`
  lists;
- `manual_coin`, continuation adapter/version, Python seed, engine-state hash,
  determinization hash, RNG-tape ID, and unique `pair_id`.

Multiple roots from one game remain linked to the same episode cluster.
`search_begin_input` portability is guaranteed only for the same cg build and
platform until an explicit cross-platform canary passes.

### 6.2 Preserving actual policy behavior

Control and challenger are called in isolated, episode-persistent workers: one
worker per `policy × episode`. Each worker performs one deck call, then receives
that seat's active observations in chronological order. The root action is the
single return value from the real final entry point; it is not obtained by
re-calling the root after diagnostics. Starting midway requires replaying the
complete prefix.

Search remains enabled, and module-global random draws are replaced in the new
challenger with an explicit `random.Random` derived from canonical opening JSON,
agent hash, deck hash, and SHA-256. A transparent delegate around the module's
bound `search_begin`, `search_step`, and `search_end` records availability,
trigger eligibility, call counts, exceptions, deadlines, and final action
source. An eligible trigger must show real begin/step activity; an ineligible
root records `not_triggered`. Merely observing `SEARCH_OK=True` is not proof.

A root that cannot recreate this process is diagnostic-only. A paired test that
disables `SEARCH_OK` does not represent the shipped candidate and is rejected.

### 6.3 Paired action evaluation

For every root and hidden-state determinization:

1. Create one official-engine search root.
2. Apply the control and challenger actions from that same root.
3. Advance branches round-robin with the same transition count as the fairness
   budget. Wall time is only a verifier safety ceiling.
4. Alternate branch order across determinizations.
5. Use the same matched determinization: one search root, identical ordered
   hidden-card lists, `manual_coin=True` with one deterministic coin tape, and
   the same seed for a search-free continuation adapter. The tape is immutable;
   each branch owns an independent cursor initialized at position zero.
6. Record terminal win/draw/loss when reached.
7. Keep truncated estimates in a separate evidence field; they do not count as
   terminal outcomes or primary promotion evidence until calibration proves
   their ordering against a terminal subset.

The candidate's own model and runtime verifier may use only the inner
determinization namespace. Gate C evaluation uses a disjoint outer namespace
that is generated after the candidate action is frozen. Seed overlap is a hard
schema error. This prevents the verifier from selecting and grading an action
on the same random futures.

The final search-enabled agent is used only to select the root action in its own
worker. Search-state child observations do not carry `search_begin_input`, and
`search_end()` owns a process-global arena, so continuation evaluation must not
call the final agent recursively. Each continuation family is an explicit pair
`(our_search_free_adapter, opponent_search_free_adapter)` and may not touch any
`search_*` function.

Rollouts are repeated under the frozen runtime continuation family and one
confirmatory family hidden from training and development. A gain that exists
only under the runtime family is not eligible for promotion. The API exposes no
native RNG clone, so this evidence is named `matched_determinization`, not a
claim that every engine random draw is paired.

Repeated determinizations are averaged within a decision root. No pair may
cross a deck hash or use an action proposed for another deck. If any branch of
a pair fails to complete, runtime rejects the whole decision; offline marks the
root incomplete, reports action-specific completion, and applies worst/best-case
sensitivity bounds. A terminal-only subset cannot promote when completion is
action-dependent. Statistical resampling clusters by source episode, then
stratifies by opponent archetype and seat, so multiple roots or rollouts from
one game are not treated as independent games.

Paired action evidence estimates the local effect of an action under fixed
continuation policies. It does not claim that a policy which triggers many times
per game has a higher full-game win rate.

### 6.4 Prospective randomized full-game evaluation

Policy-level evidence uses a separate evidence kind,
`randomized_block_full_game`. It does not pretend that two games share an engine
seed. Instead, before outcomes exist, a manifest fixes:

- the control and candidate hashes;
- a code-and-deck-hash-deduplicated opponent zoo;
- opponent, seat, and execution-batch blocks;
- equal control/candidate allocation inside every block;
- randomized interleaving order and a fixed scheduler seed;
- the total sample size and analysis command.

Each game resets all agent and opponent module state. Outcomes remain sealed
until the complete manifest has run. Every started assignment remains in the
intention-to-treat ledger. An infrastructure-interrupted game is resumed only
under the same immutable game ID and assignment; ambiguity or selective loss is
`HOLD`. Agent crash, invalid action, or timeout is a loss, not a deleted row.
The primary analysis is a block-randomization estimate. Opponent-policy and seat
strata are also reported separately.

Historical or agent-by-agent batches remain `unpaired_full_game` and can only
veto. A prospective randomized experiment may support promotion for the frozen
opponent zoo, but it still cannot prove transport to the changing Kaggle meta;
that uncertainty is handled by the online cohort.

A deck experiment freezes one policy hash for both arms. A policy experiment
freezes one deck hash for both arms. A manifest that changes both is an
artifact-level exploratory comparison and cannot attribute or promote either
the deck change or the policy change.

## 7. Evaluator calibration gate

Model work cannot begin until all calibration checks pass:

- Identity: identical actions still execute both branches and produce exactly
  zero paired difference.
- Antisymmetry: swapping candidate and control negates the reported effect.
- Branch-order invariance: the 95% interval for the order interaction lies
  wholly inside `[-0.005, +0.005]`.
- Positive control: on at least 50 independent episode roots, an immediate
  winning action has a 95% episode-clustered lower confidence bound above zero
  against an intentionally inferior legal action.
- Negative control: the reversed comparison has an upper confidence bound below
  zero.
- Null calibration: at least 100 independent simulated and duplicated null
  experiments produce zero `PASS` results, putting the one-sided 95% binomial
  upper bound below 5%.
- Outer-evaluation proof: manifests reject any inner/outer determinization seed
  overlap and prove that outer outcomes were generated only after actions froze.
- Search-on proof: both workers preserve search availability; every eligible
  trigger records real begin/step calls, while an ineligible root is explicitly
  `not_triggered`.
- Failure closure: timeout, malformed action, missing search input, incomplete
  pairs, or worker crash produces `ERROR` or `HOLD`, never `PASS`.

If this gate has not passed by 2026-07-19, learned-policy work stops and all
remaining effort moves to the two reality-seeded deterministic policies and
rule-level fixes. Archaludon roots may satisfy the instrument's generic
calibration checks, but every target deck must also pass identity,
antisymmetry, search-on, and failure-closure canaries on its own roots before
its paired effects can enter Gate C.

## 8. Dataset and split policy

Rows are never randomly split or pooled across deck hashes. Before the first
model fit, data are partitioned first by target deck, then grouped in the order
`policy/template hash -> team/source -> episode`, and assigned by a
forward-time cutoff to immutable partitions:

- 60% train;
- 20% development, used for model and trigger choices;
- 20% shadow test, forward in time and containing held-out opponent policies.

All decisions and determinizations from one episode stay together. Reports are
separate for source, archetype, opponent policy, seat, and time window.

Each admitted deck receives at most one preregistered candidate-policy family
and exactly one confirmatory shadow test. Its family ID binds the deck and
policy hashes, feature schema, trigger predicates, ranker, verifier, runtime
margin, continuation families, belief prior, tree shape, and every tuned
hyperparameter. After that deck's shadow result is read, any version inheriting
a design choice from the result is exploratory and cannot receive another
nominal 95% confirmatory test in this campaign.

Teacher/tree comparisons and all model selection occur on train, development,
or a separate selection holdout. Public top-episode exports are
selection-biased and have zero Gate C weight; they are discovery/training data
only. At least one continuation-policy family is unused by training and
development and remains sealed until Gate C.

## 9. Proposal model

The first competition model, if the evaluator and deterministic baseline pass,
targets the exact-60 Dragapult control. A Clefairy model is a separate family
and is attempted only after its deterministic policy clears the full-game
screen. No training row, target, feature prior, or shadow result crosses deck
hashes.

The model is a compact pairwise/listwise gradient-boosted tree ranker over
explicit state-action features. It is exported as
`proposal_model.json` and evaluated by a scorer that depends only on the Python
standard library. The complete agent still vendors the native `cg` runtime.
LightGBM or an equivalent trainer is a training dependency only.

The model receives:

- public board, prizes, turn, seat, hand/deck counts, and remaining time;
- visible card identities and an archetype posterior computed only from cards
  visible at runtime and a prior frozen before the shadow test;
- attack readiness, knockout/recovery/retreat routes, energy and evolution
  structure;
- rule rank and rule reason family;
- matchup interactions defined before the shadow test is opened.

Targets are paired action preferences and advantages within one root. Draws and
finite continuous values are retained with explicit kinds; the loader may not
silently coerce or discard them. Uncalibrated truncated values are diagnostic or
lower-weight training evidence, never primary test labels.

The export schema fixes feature names/order, schema version, numeric splits,
missing/default direction, tree and leaf weights, float precision, stable
tie-breaking, and model hash. Training and runtime scorers must match on a
golden corpus before packaging, and the packager explicitly allowlists this one
model filename.

The proposal model returns one distinct legal option from the top six
rule-ranked actions. Engine transition features are excluded from v1 so model
scoring cannot consume hidden-state or search budget before verification. The
model cannot directly override forced contexts, auxiliary selections, or the
default rule action.

The 1080 Ti and Colab may train neural teacher experiments after the tree
baseline exists. Teacher selection occurs only on development/selection data;
the selected family still receives exactly one confirmatory shadow. A teacher
is shipped only if it can be distilled into the same runtime contract. Neural
research cannot delay the tree candidate or the online calendar.

## 10. Runtime verifier

At an eligible disagreement, the verifier evaluates the target deck's frozen
same-deck rule action and one target-deck proposal. It uses inner determinization
seeds derived from a stable
observation hash plus a per-episode salt, making local reproduction possible
without relying on module-global random state. Statistical promotion happens
offline; runtime uses a fixed conservative filter rather than repeatedly running
low-powered hypothesis tests throughout a game.

An override requires all of the following:

- every action is legal and distinct;
- both actions complete exactly eight matched samples; any incomplete pair
  rejects the override;
- paired evaluation order alternates;
- all eight samples use the frozen search-free runtime continuation family;
- all eight proposal-minus-rule differences are strictly positive and their
  mean is at least the runtime margin frozen from development, never below 0.03
  on the normalized `[0, 1]` value scale;
- the proposal's trigger family passed the frozen outer-evaluation gate before
  packaging;
- no model, search, deadline, or hidden-card accounting error occurs;
- at least 120 seconds of the official per-player game allowance remains; if a
  reliable remaining-time value is unavailable, search is disabled.

Runtime leaf evaluation is versioned as `leaf_value_v1`:

```text
terminal = win 1.0, draw 0.5, loss 0.0
raw = 0.78 * prize_advantage
      + 0.18 * board_hp_energy_advantage
      + 0.04 * deck_count_advantage
leaf_value_v1 = 0.5 + 0.5 * clamp(raw, -0.99, 0.99)
```

The advantages use the exact existing-v2 normalization: prize difference
divided by 6; summed HP plus 22 per attached energy divided by 1800; and deck
count difference divided by 60. Development calibration occurs before the
candidate family freezes. If it fails, that family is terminal/forced-win-only;
if it passes, the frozen family enables `leaf_value_v1` before shadow is opened.
The sealed Gate C continuation family then confirms non-inferiority; failure
rejects the family rather than silently changing it and testing again.

Calibration requires at least 1,000 truncated states from 200 complete games on
an opponent/time holdout, a pairwise-concordance/AUC 95% lower bound above 50%
for ordering two states by their eventual terminal outcome, and Brier score no
worse than the constant outcome-rate predictor. Failure leaves runtime in
terminal/forced-win-only mode.

Transitions are allocated round-robin. Transition count is the fairness budget;
`time.perf_counter()` supplies only the outer wall-clock stop. Terminal branches
may stop early. At deadline, fewer than eight complete pairs returns the rule
action. Search cleanup runs in `finally` and releases the whole arena.

Ties and uncertainty return the rule action. At most four learned overrides are
allowed per game. One verifier failure disables learned overrides for the
current decision; three failures disable them for the episode. Search time is
measured on at least 1,000 triggered decisions and 100 complete games in a Linux
x86 single-core constrained profile, with cold- and warm-start results reported
separately. Shipping requires an upper confidence bound for p99 below 400 ms,
p99 cumulative search below 30 seconds per complete game, and an upper 95% bound
on runtime error rate below 0.5%. Mac numbers are development diagnostics only.
Failing any target rejects the candidate.

Full archetype belief search is deferred. The first verifier uses a frozen
mixture over compatible deck hypotheses derived only from visible cards. Low
posterior confidence skips the override.

## 11. Offline promotion gates

### Gate A: foundation integrity

- evaluation-foundation tests pass from a clean environment;
- source, dataset, agent, model, and package hashes are recorded;
- every exact-transfer deck matches its source name/count manifest, competition
  card-ID mapping, ordered hash, multiset hash, and 60-card engine-legality
  report; adapted decks carry a distinct source and mutation manifest;
- the packaged deck hash matches the experiment, and the policy compatibility
  check finds no deck-specific rule that references an absent card;
- a deck experiment has one policy hash across arms and a policy experiment has
  one deck hash across arms;
- historical/nonrandom unpaired full games can only return `FAIL` or
  `HOLD_UNPAIRED`;
- `randomized_block_full_game` requires a frozen balanced allocation manifest
  and rejects missing, duplicate, or selectively completed blocks;
- package scanning and deterministic rebuild pass.

### Gate B: calibrated paired evaluator

All checks in Section 7 pass. This gate is a hard dependency for every later
`PASS`.

### Gate C1: frozen decision-root shadow evidence

- effective cluster count is fixed from development-set variance and episode
  intraclass correlation before opening shadow, with a hard minimum of 250 roots
  from 100 source episodes; no interim confidence checks are allowed;
- both seats and at least five archetypes represented;
- target weights for source, seat, archetype, and opponent policy are frozen in
  the shadow manifest;
- all reported effects use the disjoint outer determinization namespace;
- episode-clustered 95% lower confidence bound of paired advantage above zero;
- mean paired advantage at least 0.02;
- the sealed continuation family's 95% lower bound is above the preregistered
  non-inferiority margin of -0.01;
- no archetype with at least 30 roots from 15 episodes and three policy hashes
  has a point regression worse than 0.03; lower-support archetypes cannot enable
  runtime override triggers;
- worst/best-case incomplete-rollout sensitivity still has a positive lower
  bound;
- the candidate was frozen before opening the shadow set.

Action agreement and top-k recall are reported but cannot satisfy Gate C1.

### Gate C2: preregistered policy-level evidence

- a fixed sample size derived from development variance and power, never fewer
  than 2,000 randomized full games per agent;
- at least 20 code-and-deck-hash-distinct opponent policies, both seats, and
  balanced control/candidate allocation within every block;
- source/opponent/seat weights and the analysis are frozen before execution;
- no outcome inspection before the fixed manifest completes;
- 95% block-randomization lower confidence bound above zero for the frozen-zoo
  average treatment effect;
- positive point estimate in at least 70% of sufficiently sampled
  opponent-by-seat strata, where sufficient means at least 40 assigned games
  per agent;
- no stratum with at least 40 assigned games per agent has a candidate
  regression larger than five percentage points;
- zero selective block loss; invalid actions, crashes, and timeouts stay in the
  intention-to-treat analysis as losses for the responsible agent.

C2 measures the complete shipped policy, including repeated triggers and
on-policy states. It is required because C1's one-root action value cannot prove
a full-game policy improvement.

### Gate D: runtime and behavioral safety

- zero invalid actions, agent errors, or timeouts in both-seat stress tests;
- no changes in forced or auxiliary contexts;
- search-on behavior confirmed in the packaged artifact;
- p99 time budgets pass on the constrained profile;
- risk and broad unpaired pools show no regression larger than five percentage
  points in either seat. Because they are unpaired, they may veto but never
  promote.

### Gate E: complete-artifact finalist evidence

Gate E compares frozen deck-policy artifacts, not individual deck or policy
causes. All eligible artifacts run in one prospective
`randomized_block_full_game` manifest against the same code-and-deck-deduplicated
opponent zoo, seats, execution batches, and fixed scheduler allocation. It uses
at least 2,000 games per artifact and includes byte-exact v11 and the exact-60
Dragapult control as anchors.

- The exact-60 Dragapult control qualifies only if its 95% block-randomization
  lower bound versus v11 is above zero.
- Any alternate finalist must have a lower bound above zero versus v11 and a
  lower bound above -0.02 versus the exact-60 Dragapult control.
- Runtime failures remain losses, both-seat and opponent strata are reported,
  and no sufficiently sampled stratum may regress by more than five percentage
  points versus each applicable anchor.
- Among qualifying alternatives, the online challenger is ordered by the 95%
  lower bound versus the Dragapult control. If bounds differ by less than 0.01,
  prefer a different exact-transfer deck over a same-deck artifact; otherwise
  use the larger bound.

A deterministic baseline needs A, D, and E to become a finalist. Any changed
decision rule or learned policy additionally needs B, C1, and C2. A deck
mutation additionally needs the one-axis contract in Section 12.3. Only the
applicable complete gate set authorizes building an archive for human
submission review.

## 12. Reality-seeded deck track

### 12.1 Evidence hierarchy

Deck choice is not a free-form optimizer and does not blindly copy the latest
winner. Evidence is applied in this order:

1. the published 60 cards can be mapped to the frozen competition pool and the
   engine implements every effect the policy depends on;
2. the real event used a format close to the competition pool;
3. the archetype reproduced across large events or has strong matchup evidence,
   rather than relying on one champion's finish;
4. a faithful competition policy can operate it within the runtime budget; and
5. prospective local full games support the complete deck-policy artifact.

Real results create a prior, not a promotion. Captured Kaggle replay outcomes
also create only a prior because opponent strength, policy code, time, and deck
are confounded. Synthetic replay agents produced by replacing a template's deck
file are never counted as independent faithful policies.

The frozen initial roster is:

1. Prague Dragapult/Dudunsparce exact-60: primary competition control;
2. NAIC Lillie's Clefairy exact-60: independent challenger; and
3. Crustle/Mega Kangaskhan: conditional specialist only after one exact-transfer
   track is eliminated or frozen. Its missing Pokemon Center Lady requires a
   declared replacement, so every result is labeled `adapted`, never `exact`.

No fourth deck and no from-zero deck generator is admitted before one of these
slots closes. This prevents deck breadth from consuming policy and evaluation
depth.

### 12.2 Exact-transfer contract

An exact transfer has a source record, canonical name/count manifest,
competition-ID mapping, ordered deck hash, multiset hash, and engine-legality
report. The policy package must contain the same 60-card multiset. Any silent
change, name-only approximate mapping, unsupported effect, or hash mismatch is
`ERROR`, not an adaptation.

`Exact` refers to the same card-name/count multiset and rule-equivalent effects,
not collector artwork. A competition-provided legal reprint may replace the
source printing only when its name and complete engine-relevant effect are
equivalent; the source printing and mapped ID are both recorded. Any functional
difference makes the deck `adapted`.

The exact source deck remains immutable even after an adapted deck wins. Both
are retained so every comparison can be reproduced. In particular, the current
Clefairy candidate's Chien-Pao-to-Lillie's-Pearl substitution is a mutation and
cannot serve as the NAIC baseline.

### 12.3 Controlled local adaptation

The exact-60 deck and a faithful deterministic policy must first pass legality,
both-seat smoke, and runtime gates. Only then may local matchup evidence open a
deck mutation. Each mutation manifest:

- freezes one policy hash in both arms;
- names one failure hypothesis and the opponent/archetype stratum it targets;
- replaces at most four cards, meaning no more than four removed copies and
  four added copies while the deck remains exactly 60 cards;
- changes no policy rule, model, search budget, or opponent allocation;
- uses randomized interleaving and a fixed sample size chosen before outcomes;
- reports every other stratum so a targeted gain cannot hide a broad collapse;
  and
- receives only one confirmatory shadow comparison after development.

Policy work uses the inverse contract: freeze the deck hash and change only the
policy. A run that changes both axes is exploratory and can generate a new
hypothesis, but it cannot promote either change. Card substitutions that merely
make an existing policy easier to execute still count as deck mutations.

### 12.4 Advancement and fallback

Dragapult is the first competitive control because it combines exact format
compatibility, repeated real-event success, and lower policy cold-start risk.
Clefairy can become the independent final-slot challenger only after
its exact-60 artifact passes A, D, and E; any changed decision rule additionally
passes B, C1, and C2. A seed-deck mutation is eligible only after its one-axis
comparison and Gate E. Crustle is opened only if its anti-meta coverage is still
needed and there is time to validate the declared replacement.

No deck inherits Archaludon calibration, another archetype's replay win rate,
or a human tournament finish as Gate C evidence. If exact Dragapult fails Gate E
by 2026-07-24, v11 remains the anchor and there is no online cohort. If
Dragapult passes but no alternate qualifies, v11 is its second-slot hedge. If
neither new artifact qualifies, the campaign reports that outcome and does not
invent a weak second deck to fill a slot.

## 13. Online cohort protocol

Kaggle keeps only the two latest submissions active. Therefore there is no
permanent online anchor across multiple challengers.

The online cohort is a catastrophic/runtime/meta sanity gate, not a powered test
of the offline +0.02 target. The control is the Gate-E-passing exact-60
Dragapult artifact. The challenger is the highest-ranked Gate-E alternative:
exact Clefairy, a passing seed-deck mutation, a same-deck policy improvement, or
the conditional adapted specialist. Gate E's near-tie rule prefers exact
Clefairy or another different exact-transfer deck without ignoring a material
lower-bound difference. If no alternative qualifies, there is no cohort.
Archaludon cannot occupy either cohort slot merely because it calibrated the
evaluator.

When the two arms use different decks, the online estimate is explicitly the
effect of the complete artifacts in the observed meta. It cannot attribute the
difference to deck or policy. Causal deck and policy claims remain governed by
the one-axis-at-a-time local experiments. The cohort is submitted in a
deliberate order:

1. Build and record exact control and challenger archives.
2. Submit the fresh challenger first and the fresh control second, back-to-back
   in the same time window. The control is therefore the newer retained slot.
3. Do not submit a third artifact while the cohort is being measured.
4. Both submissions must be validation-complete and active by 2026-07-30 23:59
   UTC; otherwise use the no-cohort branch. The analysis window begins only
   after both artifacts are active and closes at 2026-08-07 23:59 UTC. Do not
   inspect interim outcomes. Target at least 300 completed episodes per
   artifact; fewer than 300 at cutoff is automatically `HOLD_UNIDENTIFIED`.
5. Compare raw outcomes with seat, opponent rating/archetype, and time-window
   adjustment. Analysis requires covariate overlap and clusters resampling by
   opponent submission/team and calendar day. Opponent rating means its
   event-time snapshot; if unavailable, no later rating is backfilled. Displayed
   rating is secondary; lack of overlap is `HOLD_UNIDENTIFIED`, not permission to
   extrapolate.

Before either upload, the cohort manifest freezes this estimator:

- outcome is win `1`, draw `0.5`, loss `0`;
- include every completed non-validation episode whose start time is inside the
  sealed window; agent error, invalid action, and timeout remain losses;
- deduplicate only exact episode IDs; a missing replay with a known official
  result remains included with `unknown` covariates;
- an episode with missing arm or outcome is never silently dropped and makes the
  cohort unidentified if more than 5% of either arm is affected;
- event-time rating comes only from an explicitly event-time field captured in
  the raw episode at ingestion; otherwise its value is `unknown`;
- common-support cells are
  `seat × frozen_archetype × 100-point event-rating bin × 48-hour window block`,
  with explicit `unknown` levels;
- a cell enters standardization only with at least five episodes per arm; the
  cohort additionally requires at least 60% episode coverage in each arm, 12
  common cells, 20 distinct opponent-submission clusters, and five UTC-day
  clusters;
- the adjusted effect is the pooled-cell-frequency standardized difference:
  `sum_cell weight_cell * (mean_challenger - mean_control)`, where weights use
  pooled arm counts and never outcomes;
- uncertainty uses 20,000 two-way cluster Bayesian-bootstrap resamples with
  independent positive Exponential(1) weights for opponent-submission and
  UTC-day clusters; the weights' product applies to each episode, the seed is
  derived from the frozen cohort manifest SHA-256, and one-sided percentile 95%
  bounds drive the decision.

The frozen archetype map uses only information visible in the official episode
record and is versioned in the manifest. Any estimator exception, failed
overlap threshold, insufficient bootstrap support, or unavailable required key
is `HOLD_UNIDENTIFIED`.

Decision rules are evaluated in this strict priority order:

1. Candidate-attributable runtime failure -> `REJECT`.
2. Fewer than 300 episodes in either arm, excessive essential missingness,
   estimator failure, or failed overlap -> `HOLD_UNIDENTIFIED`.
3. Adjusted effect at least +0.03 and one-sided 95% lower bound above zero ->
   `PROMOTE`.
4. Adjusted effect at most -0.03 and one-sided 95% upper bound below zero ->
   `REJECT`.
5. Remaining adjusted point estimate at least zero -> `HOLD_POSITIVE`.
6. Remaining adjusted point estimate below zero -> `HOLD_NEGATIVE`.

Final slots follow a minimum-upload decision table:

- `PROMOTE` or `HOLD_POSITIVE`: make no more upload. Preserve challenger and
  control, including both accumulated ratings. The control is the second-slot
  hedge against residual uncertainty.
- `REJECT`, `HOLD_NEGATIVE`, or `HOLD_UNIDENTIFIED`: submit only the next
  offline-audited finalist, or v11 if none exists. Because challenger was older,
  it is evicted and the measured control keeps its rating; final slots are
  control + hedge.
- no cohort: inspect the ordered active-slot state and upload only a missing
  final artifact. Never reset an already selected incumbent merely to create a
  visually symmetric pair.

If no alternate finalist passed, byte-exact v11 is the audited hedge. All
planned slot changes finish by 2026-08-10. There is no automatic late
replacement: before any validation-failure recovery, the exact latest-two
transition and rating reset must be written down and explicitly approved. The
official submission deadline is 2026-08-16 23:59 UTC; the final active pair is
expected to continue playing until approximately 2026-08-31 or convergence.

## 14. Calendar and automatic fallbacks

| Date | Deliverable | Automatic fallback if missed |
|---|---|---|
| Jul 15-17 | Source-bound exact-60 manifests, card/effect validation, and policy-family deduplication | Keep decks immutable and use v11 while mapping is repaired |
| Jul 15-19 | Foundation, calibrated root evaluator, randomized-runner contract | Stop learned policy; deterministic reality-seeded tracks + v11 only |
| Jul 18-23 | Dragapult deterministic policy screen, Clefairy deterministic baseline, root snapshot, frozen zoo and A/B manifests | Dragapult exact-60 control only; Clefairy remains unqualified |
| Jul 24-30 | Target-deck proposal ranker/verifier where justified, artifact full-game gates, Gates C1-C2-D | Reject learned work; use best passing deterministic artifact |
| By Jul 30 23:59 UTC | Challenger-first/control-second cohort if all offline gates pass | No cohort; preserve current agents while hedge is finalized |
| Aug 7 23:59 UTC | Close the sealed online window and apply the decision rule | Insufficient episodes/overlap is HOLD_UNIDENTIFIED |
| Aug 8-9 | Online decision and hedge audit | v11 is the hedge |
| Aug 10 | Apply the fixed slot decision table | Later changes require explicit slot-transition review |
| Aug 11-16 | Monitor errors and collect evidence; no tuning uploads | Keep final pair active |

The calendar is a scope control. A late model does not steal convergence time
from a verified deterministic agent.

## 15. Compute allocation

- Mac 1: evaluator calibration, Dragapult policy screening, and Dragapult roots.
- Mac 2: exact Clefairy policy games, opponent-zoo games, and independent
  reproduction of Dragapult results.
- 1080 Ti: optional neural teacher and batched feature experiments; it is not a
  dependency for the tree ranker.
- Colab: burst ablations and reproducibility checks; no irreplaceable state lives
  only in a temporary session.

Jobs write immutable shards and manifests. Shards merge only after schema,
hash, and duplicate checks. A worker failure cannot silently reduce a stratum;
missing work is reported and leaves the gate at `HOLD`.

## 16. Test and failure strategy

### Unit and statistical tests

- exact-60 source normalization, card-ID mapping, multiset hashing, and legality;
- failure on silent deck substitution and enforcement of the four-card mutation
  distance;
- manifest rejection when a deck experiment changes policy or a policy
  experiment changes deck;
- evidence-kind parsing and fail-closed promotion;
- episode-clustered, stratified paired bootstrap;
- prospective block allocation, balance, fixed-N completion, and randomization
  inference;
- identity, antisymmetry, branch-order, positive, negative, and null controls;
- inner/outer determinization namespace separation;
- fixed-N and incomplete-pair sensitivity bounds;
- continuous/draw target retention;
- `leaf_value_v1` terminal-holdout calibration;
- tree export parity with the training implementation;
- deterministic seed derivation and hidden-card accounting;
- confidence and online cohort decision rules.

### Integration tests

- source deck -> competition IDs -> engine deck initialization for all 60 cards;
- exact Dragapult and Clefairy complete-game smoke tests in both seats;
- policy/deck compatibility tests that reject rules referring to absent cards;
- episode-persistent search-enabled policy workers on the same recorded prefix,
  including mutable-state replay and search delegate-spy assertions;
- same-build cross-process root restoration plus explicit cross-platform canary;
- multiple actions branched from one search root and determinization;
- equal transitions and alternating branch order;
- proof that search-free continuation adapters never touch the outer search
  arena;
- preregistered interleaving of complete control/candidate games inside each
  opponent-by-seat block;
- both-seat complete games against representative archetypes;
- packaged agent in a network-disabled, read-only, constrained environment;
- missing/corrupt model, absent search input, exceptions, and deadline exhaustion
  all returning the rule action.

### Security and reproducibility

The credential-bearing root archive is never opened, copied, packaged, or
committed. Before any push that could expose the repository, the Kaggle token is
rotated and the archive is removed or relocated by explicit user action.

Every decision record binds the Git revision, agent/deck/model/archive hashes,
dataset and split manifests, exact command, seeds, per-stratum results, errors,
and final `REJECT`, `HOLD`, `PACKAGE`, or `REQUEST_SUBMISSION` state.

## 17. Definition of success

The campaign has succeeded technically when it produces two valid final
artifacts through the declared gates, with no candidate promoted by a known
invalid proxy. The intended anchor is exact-60 Dragapult; if it fails Gate E,
there is no online cohort and v11 remains the anchor. The second finalist is the
highest Gate-E-ranked qualifying alternative, with the near-tie preference for
exact Clefairy or another deck-diverse exact transfer. If none qualifies, v11
fills the fallback slot only once and the campaign reports that it did not
produce two new finalists. It has succeeded competitively only if the resulting
agents win enough official games to improve the final standing.

No design can guarantee that second outcome. This design guarantees that a
failure becomes attributable and actionable: evaluator failure, model failure,
runtime failure, meta/deck failure, or online uncertainty. That is the condition
required to stop the current cycle of unexplained regressions.

## 18. References

- Simulation evaluation:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/evaluation
- Latest-two-active and post-deadline protocol:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/714189
- Matchmaking update:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/716045
- Submission guide:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/how-to-submit-to-this-competition
- CABT API:
  https://matsuoinstitute.github.io/cabt/
- 2026 Standard rotation:
  https://www.pokemon.com/uk/pokemon-news/2026-pokemon-tcg-standard-format-rotation-announcement
- Regional Prague exact Dragapult/Dudunsparce list:
  https://limitlesstcg.com/tournaments/539/decklists
- Los Angeles Regional Dragapult/Dudunsparce result:
  https://www.ptcgstats.com/p/2026-los-angeles-regional.html
- NAIC 2026 exact Lillie's Clefairy list:
  https://limitlesstcg.com/decks/list/28249
- NAIC 2026 field statistics:
  https://limitlesstcg.com/tournaments/518/statistics
- Japan Championships 2026 winner interview and deck-choice rationale:
  https://www.pokemon-card.com/info/005538.html
