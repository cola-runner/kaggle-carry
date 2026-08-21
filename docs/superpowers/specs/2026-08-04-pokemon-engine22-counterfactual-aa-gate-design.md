# Pokémon Engine 22 Counterfactual A/A Gate

## Goal

Before generating any self-play labels, verify that Engine 22 can evaluate the
same action from the same decision state reproducibly. A failed gate stops the
counterfactual-training route; it does not train or submit an agent.

## Probe

- Collect 8 to 12 non-forced decision states from real games played by the four
  existing drivers.
- For every sampled state, create two independent fresh search roots.
- Apply the same chosen legal action to both roots.
- Keep deck order, hidden information, downstream driver policies, and all
  manually controlled random choices identical between the pair.
- Continue both branches to the end of the game and compare their terminal
  scores. Also record the immediate next-state digest to distinguish an
  unstable first transition from unstable later play.

The probe must not reuse a mutated search root across samples. Branch order is
alternated so a global-state leak cannot consistently favour one side.

## Frozen Decision

`PASS_COUNTERFACTUAL_AA` requires at least 95% exact paired agreement in both
terminal score and immediate next-state digest, with no illegal actions,
timeouts, engine errors, or uncontrolled-randomness warnings.

Anything below the threshold returns `REJECT_COUNTERFACTUAL_ENGINE`. In that
case, terminal branch outcomes cannot be used as action labels; the next
candidate route is learning from actual complete games without search forks.

A pass only authorizes a separate A/B action-value experiment. It does not
prove that self-play improves the population and does not authorize a Kaggle
submission.

## Storage and Runtime

- Engine: official local Engine 22 only.
- No replay, checkpoint, dataset, or raw state dump is persisted.
- Temporary state is kept in one owned system-temp directory and always
  removed.
- The only persistent output is one compact JSON report under `reports/`, with
  a target size below 100 KiB.
- The Mac run has a short fixed wall-time limit; timeout is a rejection, not a
  reason to increase compute.
