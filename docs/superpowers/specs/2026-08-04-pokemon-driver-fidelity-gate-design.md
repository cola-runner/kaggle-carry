# Pokémon Driver Fidelity Gate

## Goal

Before any further self-play, prove that each 1.5-million-parameter neural
policy has learned enough of its original driver to be a credible starting
player. A failed fidelity run stops before PPO and retains no model.

## Evidence and Root-Cause Hypothesis

The original four drivers scored `0.562` against the frozen development pool.
The neural policies scored `0.254` immediately after imitation and `0.227`
after two self-play rounds. The first failure therefore occurs at the boundary
between visible game state and driver imitation, before self-play.

The current encoder sends every numeric feature through one global `[-8, 8]`
clip and then hashes all feature types into the same 512 positions. Important
values such as HP, deck size, turn, and action identifiers lose scale or
collide. The first hypothesis is that this representation prevents the student
from distinguishing decisions the driver can distinguish.

## Considered Approaches

1. **Increase imitation games with the current encoder.** Fast to launch, but
   it feeds more examples through a representation already known to discard
   information. Rejected.
2. **Replace only the encoder and compare it against the old encoder on exactly
   the same decisions.** This isolates one variable, keeps the validated 1.5M
   model and submission runtime, and gives a short Mac result. Selected.
3. **Replace the policy with a Transformer.** It may eventually be useful, but
   it changes representation, architecture, runtime, and training at once.
   Rejected for this diagnostic.

## Encoder V2

The output remains 512 float32 values per legal option, so the policy model and
NumPy inference contract do not change.

- bounded game quantities receive value-specific normalization instead of a
  global clip;
- HP/damage, deck size, turn/step, prize count, energy count, and selection
  bounds occupy separate deterministic numeric regions;
- categorical state and action signals occupy separate hash regions;
- action identity, source card, target card, attack, and context remain
  distinguishable;
- hidden information remains forbidden and the existing hidden-payload test
  continues to apply.

Automated tests must show that changing HP, deck count, action source, or target
changes the encoded option while poisoning hidden opponent data does not.

## Paired Fidelity Experiment

Collect 96 balanced driver-versus-driver games from both seats. Each observed
decision is encoded by both V1 and V2 at collection time, giving the two
encoders identical observations and labels. Split by whole games into 72
training games and 24 held-out games; decisions from one game may not cross the
split.

Train two fresh four-policy populations with identical seeds, optimizer,
epochs, and model architecture. The only changed variable is the encoder.
Drivers are then closed. No PPO, reward learning, judge result, or self-play
update is allowed in this experiment.

Held-out evaluation uses exact ordered action agreement and negative log
probability on non-trivial choices. Forced single-option decisions are reported
but cannot make the gate pass.

## Frozen Decision

`PASS_DRIVER_FIDELITY_V2` requires all of the following:

- V2 aggregate held-out negative log probability improves by at least 10%
  relative to V1;
- V2 aggregate exact agreement improves by at least 5 percentage points;
- at least three of four members improve on both measures;
- no member's held-out negative log probability regresses by more than 5%;
- all features, losses, gradients, and exported NumPy logits are finite;
- no illegal action, hidden-information dependency, or split leakage occurs.

Failure produces `REJECT_DRIVER_FIDELITY` and no model. Passing permits a
separate development-opponent strength test; it does not yet permit PPO or a
move to the GTX 1080 Ti.

## Storage and Time

Raw replays are never written. Paired decisions live only in a run-specific
macOS temporary directory under a 128 MiB quota and are removed on success,
failure, exception, signal, or timeout. The Mac wall-time limit is six minutes.
The only persistent output is
`reports/mac_driver_fidelity_latest.json`, expected to remain below 100 KiB.

The existing `artifacts/` directory must have identical file count and byte
size before and after the run.

## Next Step After a Pass

Use new development opponents, separate from the already inspected final
judges, to compare the learned students against their drivers. Only students
that preserve most driver strength can enter a future conservative self-play
league with per-member rollback.
