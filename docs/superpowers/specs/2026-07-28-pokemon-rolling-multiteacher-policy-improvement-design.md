# Pokémon TCG Rolling Multi-Teacher Policy Improvement Design

Date: 2026-07-28

Status: Approved for implementation planning

Supersedes:
`2026-07-23-pokemon-live-leader-policy-improvement-design.md`

## Decision

Build exactly one challenger using the currently dominant
Grimmsnarl/Froslass deck. Do not bind the project to one leaderboard player.
Train from multiple current high-rating submissions that use the same exact
deck, and refresh the source window whenever the snapshot becomes stale.

The driver will learn the downstream value of legal actions in current real
ladder states. Direct action imitation remains auxiliary evidence, not the
policy objective or the promotion gate.

## 2026-07-28 Live Snapshot

The official leaderboard snapshot is
`pokemon-tcg-ai-battle-publicleaderboard-2026-07-28T14:40:57.csv`.

- Competition submission deadline: `2026-08-16T23:59:00`
- Rank-one team: `Dominic Peel`
- Rank-one score: `1158.8`
- Rank-ten threshold: `1133.4`
- Former rank-one team `Luca`: rank `61`, score `1047.1`
- Team `ezreal77`: rank `1592`, score `756.0`

The team's two currently tracked submissions are:

| Submission | Submitted UTC | Current score | Order |
|---|---|---:|---|
| `54592322` (v234) | 2026-07-12 02:44 | 756.0 | older |
| `54600199` (v11) | 2026-07-12 07:58 | 739.4 | newer |

Kaggle tracks the latest two submissions. One new challenger will therefore
make the tracked pair `v11 + challenger`; it will automatically remove the
older v234 from the tracked pair. This is determined by submission order, not
by score and not by a selectable slot.

## Current Deck Evidence

Nine of the current top ten teams were inspected through an official replay
from their higher-scoring tracked submission:

- Four use the exact same Grimmsnarl/Froslass 60-card list.
- One uses the same core with a two-card Handheld Fan substitution.
- The remaining observed decks are Crustle, Festival Grounds, Raging
  Bolt/Area Zero, and Alakazam.

The exact-deck current teacher submissions are:

| Rank | Team | Submission | Score |
|---:|---|---:|---:|
| 1 | Dominic Peel | `55001357` | 1158.8 |
| 1 | Dominic Peel, second tracked agent | `54989332` | 1147.4 |
| 4 | LiamK | `55011514` | 1148.8 |
| 9 | insuperabilehart | `55035974` | 1135.0 |
| 10 | A. R. SEKKAT | `54968369` | 1133.4 |

The near-deck current reference is:

| Rank | Team | Submission | Score | Difference |
|---:|---|---:|---:|---|
| 7 | Dries @ Tufa Labs | `54986389` | 1140.9 | two Handheld Fans replace Pokégear 3.0 and Tool Scrapper |

The exact 60-card teacher deck remains:

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

This evidence keeps the deck choice but invalidates the former single-teacher
assumption. Luca is no longer a current teacher. Dominic is not made a
permanent replacement teacher; the eligible teacher set is recomputed from
every fresh snapshot.

## Freshness Contract

Every build starts by creating a new official snapshot with a single UTC
cutoff time.

Teacher eligibility requires:

1. the exact 60-card deck fingerprint;
2. a tracked submission at snapshot time;
3. a score at or above the live rank-ten threshold;
4. at least one completed episode inside the rolling source window.

The source window is the 72 hours ending at the snapshot cutoff. The newest
12 hours form the untouched chronological holdout; the preceding 60 hours are
available for training and validation. Episodes are split as whole episodes,
never as individual decisions.

The near-deck variant is not mixed into the exact-deck teacher data. It is used
only as a robustness reference and possible future deck ablation.

Training examples are balanced by team, game seat, and 12-hour time bucket.
Multiple qualifying submissions from one team share one team-level sampling
budget, so Dominic's two agents cannot dominate the dataset.

If fewer than three distinct eligible exact-deck teacher teams remain, this
design stops and requires a new deck decision. If more than 24 hours pass
between the final snapshot and archive packaging, the snapshot, extraction,
training, and gates must be rerun from the newest rolling window.

Older source code may be reused as infrastructure. Older ratings, replay
outcomes, and matchup results may not be used as promotion evidence.

## Scope

The implementation will:

1. freeze the fresh leaderboard, submission, episode, replay, and deck
   manifests with hashes;
2. download current eligible exact-deck episodes;
3. extract leakage-safe visible observations and legal options;
4. restore current real states in the official engine;
5. estimate the downstream result of alternative legal actions;
6. train one compact deck-specific option ranker;
7. package and submit one challenger only if every gate passes.

The implementation will not:

- select or claim to select Kaggle opponents;
- treat local public agents as current top-ten policies;
- use v168, the historical v11 score, or the old Luca score as evidence;
- train a direct behavior clone as the final driver;
- use a local tournament score as a Kaggle rating prediction;
- implement full MCTS in the first challenger;
- upload a portfolio of experimental candidates.

## Architecture

### 1. Rolling Snapshot Collector

The collector records:

- snapshot cutoff and leaderboard file hash;
- rank-ten threshold;
- team, submission, score, and submission-time fields;
- episode IDs and creation times;
- replay file hashes;
- exact deck lists and fingerprints.

The manifest is immutable after training starts. Every derived row retains its
source submission ID, episode ID, timestamp, and deck fingerprint.

### 2. Leakage-Safe State Extractor

For decisions made by an eligible exact-deck agent in either game seat, the
extractor stores:

- the observation visible to that acting player;
- all legal options, including target serial, area, index, card ID, attack ID,
  effect, and selection context;
- the selected action and final result;
- the opponent's publicly revealed cards and deck fingerprint;
- offline-only engine restoration material in a physically separate dataset.

Opponent hidden hands, deck order, prizes, and replay visualization fields are
forbidden as model inputs. Offline hidden data may only restore the exact
historical engine state used to generate counterfactual labels.

### 3. Visible-State Value Model

Two independently initialized small models, `V1(s)` and `V2(s)`, predict final
win probability from runtime-visible information:

- acting-player hand, active, bench, discard, prizes remaining, and energy;
- opponent public board, discard, prizes remaining, and revealed cards;
- turn, seat, action flags, stadium, damage, and recent public logs.

The models are selected using the training and validation portions only. The
newest 12-hour holdout stays unopened until the 48-hour feasibility decision.
Expected calibration error uses ten equal-frequency bins. A holdout lacking
either wins or losses fails the gate instead of producing a metric.

### 4. Official-Engine Branch Labeler

For important `MAIN`, single-choice decisions:

1. restore the exact current historical state;
2. verify that the recorded action reproduces the replay's next state;
3. branch every eligible legal action from an independent root;
4. use identical deck order and an explicit manual coin schedule per branch;
5. resolve forced follow-ups and stop at the next meaningful main decision or
   turn boundary;
6. score each visible leaf with both value models.

Branch enumeration is repeated in reverse option order. Order-dependent roots
are rejected as infrastructure failures.

The training target is downstream action value. Teacher action agreement is
reported for diagnosis only.

### 5. Runtime Driver

The packaged agent contains:

- deterministic setup, forced-selection, and legality handling;
- the compact option ranker;
- a confidence gate;
- a deterministic fallback for unsupported contexts, model errors, or low
  remaining time.

The ranker may override the fallback only when both value models prefer the
same action and the smaller estimated advantage over fallback is at least
`0.10`. The runtime uses only the legal observation and makes no network call.

## 48-Hour Feasibility Stop

Within 48 hours of implementation start, all of the following must hold:

1. at least three distinct eligible exact-deck teacher teams are present;
2. at least 500 current completed episodes and 5,000 non-forced single-choice
   `MAIN` decisions are extracted;
3. 50 sampled replay roots reproduce their recorded next state exactly;
4. the same 50 roots are invariant to forward versus reverse branch order;
5. the unopened holdout gives each value model ROC AUC at least `0.65` and
   expected calibration error at most `0.08`;
6. automated feature audit finds no opponent-hidden or visualization-only
   model input.

Any failure ends this route immediately. No Kaggle submission is made and no
threshold is weakened after seeing the result.

## Candidate Gates

After feasibility passes:

1. all branch-training roots must pass replay fidelity and branch-order checks;
2. candidate overrides require the two-model agreement and `0.10` margin;
3. 1,000 official-engine safety games must finish with zero illegal actions,
   agent exceptions, or timeouts;
4. single-decision p99 latency must stay below `500 ms` on one Linux CPU core;
5. the packaged archive must pass the existing pre-submit legality, dependency,
   and hidden-data audits.

The 1,000 games are a safety test only. Their win rate is not a Kaggle rating
estimate and cannot promote the candidate by itself.

## Schedule and Online Experiment

- Implementation starts after written-spec review.
- The 48-hour feasibility verdict is the first hard deadline.
- A passing archive targets submission no later than 2026-08-04.
- The competition submission deadline is 2026-08-16T23:59:00.

Exactly one new policy candidate is submitted. A packaging-only resubmission is
allowed after a validation error, but its policy, deck, and model hashes must
remain unchanged.

Kaggle chooses every opponent. After validation, the challenger is allowed at
least 200 completed public episodes before a verdict. It is compared with the
live rank-ten threshold and its own official matchup and seat splits. No
second policy candidate is introduced while this measurement is running.

## Deliverables

- fresh rolling snapshot manifest;
- replay inventory and file hashes;
- leakage-safe state, option, and offline restoration datasets;
- value-model calibration and holdout report;
- branch-label integrity report;
- packaged agent, model, exact deck, and official engine;
- pre-submit audit report;
- one Kaggle submission reference if and only if all gates pass;
- official post-submission episode report.

## Known Limitations

The current top agents' source code and model weights are private. Multiple
high-scoring agents using the same deck reduce dependence on one teacher but
do not reveal their algorithms. Counterfactual labels also depend on the
quality of the visible-state value models. The 48-hour stop exists to reject
the route before those assumptions consume the remaining competition time.

The only authoritative test of a passing archive remains Kaggle's automatic
matchmaking.
