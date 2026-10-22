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
