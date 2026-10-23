# Additional read-only trajectory storage

This instrumentation addendum is made during the main confirmation sampling,
before any mechanistic trajectory bank is generated. The complete-law test,
its inference dependencies, events and sample allocation remain frozen at
5633b547682b129bb0f0baa81f6ad9fe1012eb25.

The planned eight-run diagnostic trajectories will additionally store each
round's per-node candidate, participation flag, COMMIT flag and exact available
count. These are read-only records of already fixed decisions. Their purpose
is to permit stratification of the planned proposal/confirmation accounting
by candidate position and availability without rerunning trajectories or
treating dependent vertices as independent samples. Whole-run summaries retain
the run as the unit of replication. Existing action-equivalence regression
tests must pass after the storage addition.

This is exploratory instrumentation, not a change to any confirmatory
hypothesis or to the original policy/adapter/sampler. No observer output
is passed to participation, proposals or COMMIT. The initial-scorer
real-arithmetic reference integration is likewise a separate exploratory
CPU diagnostic and is not used for the primary test or sample allocation.
