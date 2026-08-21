# Grimmsnarl Residual Coverage V2 Design

Date: 2026-08-06

## Objective

Increase the trained Grimmsnarl driver's intervention coverage without weakening
the high-confidence behavior already demonstrated by V1. V2 is an additive
residual policy, not a replacement driver and not a full-policy retrain.

## Evidence and interpretation

The online V1 submission (`55255929`) and same-deck frozen control
(`55261450`) isolate the residual policy as the only implementation difference.
At the latest snapshot, V1 scored `678.0` and the control scored `618.6`, but
their opponent-archetype-standardized win rates were effectively tied. The
useful signal is conditional: V1 changed actions in 18 of 55 identifiable
games, going 13-5 in those games, while games without an intervention went
11-26. In the latest 12-game slice, intervention games went 2-1 and untouched
games went 1-8.

This does not prove that each changed action caused a win. It does show that
the current ensemble gate is selective rather than globally destructive. The
next experiment therefore targets coverage in untouched losses while treating
V1's existing intervention decisions as protected behavior.

## Proposed experiment

1. Reconstruct decision observations from V1's online losses in which V1 never
   overrode the frozen driver.
2. At eligible main-phase decisions, evaluate the incumbent and alternative
   legal actions with Engine22 rollouts against the existing four-policy local
   population. Outcomes are measured to terminal state; immediate board score
   is not used as the training target.
3. Retain a candidate preference only when repeated rollouts show a stable
   advantage over the incumbent across seeds and opponent members. Ambiguous
   decisions remain unchanged.
4. Train two independent small residual value models on the accepted new
   preferences, balanced with existing V1 intervention examples and untouched
   control examples.
5. Build V2 as a layered gate: V1 decisions remain frozen; the new models may
   intervene only on previously uncovered decisions when both agree and clear
   a separately calibrated margin.

## Safety boundaries

- V1 weights, thresholds, and accepted decisions are immutable inputs.
- V2 cannot replace a V1-selected action during the Mac experiment.
- No action is labelled good merely because it came from an online loss; it
  must win the counterfactual rollout comparison.
- Raw rollouts are streamed into compact aggregate records and deleted after
  aggregation. Temporary and retained artifacts together must stay below
  `500 MB` on the Mac.
- The first experiment stays on the Mac. The 1080Ti is used only after the
  complete Mac gate passes.

## Mac acceptance gate

V2 advances only if all of the following hold on frozen seeds and schedules:

- exact preservation of every replayable V1 intervention decision;
- at least one new trusted intervention in previously untouched loss states;
- positive group win-rate delta versus V1 and the frozen control;
- no member-level catastrophic regression greater than 5 percentage points;
- both independent V2 models agree on every new submitted override;
- full Engine22 legality, both-seat completion, runtime, dependency, and
  package-size checks pass.

The experiment is rejected if no new decision clears the gate. Lowering the
margin merely to create more interventions is not allowed.

## Online decision

If the Mac gate passes, package V2 and replace the control submission while V1
remains online. V1 versus V2 then becomes the next same-deck online comparison.
If the gate fails, retain V1 and spend no Kaggle submission.

## Deliverables

- a compact manifest of untouched-loss decision seeds;
- counterfactual aggregate labels without retained raw rollout files;
- two V2 residual checkpoints and calibrated margins;
- a reproducible V1/control/V2 evaluation report;
- a submission archive only when the Mac acceptance gate passes.
