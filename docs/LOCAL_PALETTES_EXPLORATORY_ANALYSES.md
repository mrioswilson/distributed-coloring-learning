# Analyses after the primary test (22 October 2026)

The frozen protocol and all primary outcomes remain unchanged. These analyses
are explicitly exploratory and do not select new weights or replace negatives.

The 18 heterogeneous-state progress comparisons include a largest discrepancy:
DeepSets-s41000 on the Transformer-generated n=640 state predicts about 165.35
confirmations from estimated marginals, versus 162.86 observed in 256 trials.
The difference is about four **observation-only** Monte Carlo standard errors;
that denominator omits uncertainty in the predicted marginals. Archived
participation counts are unusually low in that trial bank (about .49384,
versus the independently estimated .49936). Reusing the trial bank's joint
offer marginals gives about 163.40 expected confirmations, but this is a
diagnostic calculation on reused observations, not independent validation.

`python -m dcl.localpalettes.progress_replication` selects the largest such
absolute residual by a fixed rule and generates fresh estimation and test
streams: 16,384 candidate draws per pending vertex, 1,048,576 participation
draws for the state-independent pending-node head, and 1,024 full one-round
trials on the same frozen state. The original 256-trial results remain intact.
Its GPU time is added to the same twelve-hour ceiling. Selection is post hoc;
neither a small residual nor nonrejection is a proof of the probability model.

The independent replication predicted 165.48 versus 164.83 observed confirmations
(observation-only standard error 0.295). Exact and learned COMMIT counts agreed.
The remaining residual is not hidden, and predicted-marginal uncertainty is not
included in that standard error. It does not refute the algebraic identity, nor
does this finite check prove every numerical/sampling hypothesis universally.

Further post-test readouts use the already archived data only:

* The forbidden-proposal audit identifies 13 model-state estimates each for
  DeepSets and Transformer, despite perfect full-graph completion in the primary
  cohort. These are valid local-list views and proposal errors, not observed
  violations of their certified COMMIT channel. Counterexamples are exported.
* Singleton-null product laws have the exact formula stated in the theory note.
  Estimates and simultaneous binomial bounds cover all 1,015 recorded singleton
  model-state laws. Values for m>1 are analytic predictions, not additional B3
  test trials or a retrospective calibration of the original classifiers.
* `classifier-summary.csv` and paired success discordances summarize the primary
  test without changing it. A sample-size flag that fails the joint observed
  power/FPR criterion is not automatically a low-power result: FPR fluctuates.
* The primary protocol's shorthand pilot namespace is implemented by the hash
  prefix `local-palette-v1|pilot`, separate from
  `local-palette-v1|confirmatory`. No pilot outcome selected a test graph.
