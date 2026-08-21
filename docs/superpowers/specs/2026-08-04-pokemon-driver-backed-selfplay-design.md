# Driver-backed self-play design

## Outcome

Replace lossy driver imitation with a residual policy that keeps each original
driver as the default action. The learned policy may override the driver only
when its learned advantage clears a fixed confidence threshold.

## Architecture

- Keep the four existing drivers and decks unchanged.
- At every decision, record the driver's legal action as the baseline.
- The neural policy scores legal actions and learns an advantage relative to
  that baseline from self-play returns.
- During training, bounded exploration occasionally tries a non-driver action.
- During evaluation and export, use the driver action unless the learned
  override clears the confidence threshold.
- Reject the trained population unless it improves the fixed group evaluation;
  rejected models are deleted and the original drivers remain the output.

## Mac proof

Run four policies for two short synchronous rounds. Compare the final
driver-backed population with the untouched four drivers against the same
frozen judge schedule. This is a feasibility test, not a leaderboard claim.

The proof passes only when the learned component actually overrides at least
one non-forced action, the group point estimate improves, and no runtime,
validity, storage, or cleanup gate fails.

Candidate evaluation promotes members independently: a residual member is
eligible only when it beats its own untouched driver. A second, fresh judge
schedule then evaluates the promoted residual members together with original
drivers for every non-promoted member. Only this confirmation result may pass
the proof; candidate-selection games cannot also serve as confirmation games.

## Storage and safety

- Do not write raw replays.
- Keep transient data in an owned temporary directory capped at 128 MiB.
- Delete rejected checkpoints and the temporary directory.
- Retain one compact JSON report only.

## Scope

This proof reuses the current 1.5M-parameter network and current visible
features. Richer state encoding and large-scale training are explicitly
deferred until the driver-backed control path demonstrates that self-play can
improve without destroying the starting policy.
