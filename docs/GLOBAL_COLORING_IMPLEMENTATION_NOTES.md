# Implementation and scope notes

The statistical protocol was frozen at commit 80195c5, before every
confirmatory sample. Sampling/analysis implementation follows that commit;
this file does not alter any hypothesis, graph, sample size, kernel, cutoff,
control, seed namespace, selection criterion or multiple-testing family.

The sampler groups independent copies of the same graph in disjoint unions
with up to 8192 vertices per batch. This is an inference-memory choice, not
a graph feature. It retains a separate PCG64 state per execution and draws
fresh independent uniforms for each node and round. Every batch creates new
colors and a new LocalPaletteAdapter. The before-freeze cost pilot used
1024 vertices per batch; larger batches amortize fixed kernel-launch overhead.
Float32 reductions may depend on batch shape. Historical adapter tests and
fifteen-checkpoint CPU integration probes check action agreement in sampled
states, not every possible numerical boundary. The realized law being tested
is the explicitly archived float32 implementation and batching scheme.

The queue charges elapsed process time while holding the GPU lock, including
teacher/CPU evaluation and overhead; this is conservative relative to time
actively computing on GPU. Analysis occurs in a separate CPU process. If a
coordinator is interrupted, recovery conservatively charges time since the
last active receipt, including downtime. The queue cannot silently reset the
budget. A timer interrupts sampling near the ceiling; an interrupted batch
is retained as a technical error rather than replaced with successful runs.

All exact laws use Fraction arithmetic, not neural Monte Carlo. For the MMD
analysis, a precomputed-kernel entry point avoids recomputing kernels in
power resampling and is unit-tested against the direct unequal-sample U
formula. No primary statistic was selected using observed p-values.

The power experiment is the prospectively declared conditional empirical-bank
resampling procedure. Its results are explicitly not population power with
full bank uncertainty. This limitation is retained in plots and manuscript.
No universal or architecture-population equivalence is claimed.

The contextual comparison to archived candidate laws is post-hoc and is
not a new confirmatory claim. It matches only the identical all-free center
state of the degree-200 star (palette 201 at the center, 2 at leaves) and
keeps each checkpoint separate. It does not substitute proposal-law tests
for final-vector tests or estimate every conditional law along trajectories.

The frozen list has 53 graph entries and 51 unique labeled edge hashes:
`star3`/`star-d3` and `wheel-d3`/`clique-d3` are duplicate fixed graphs under
two family entries. Each entry has fresh banks and remains in the prospective
795-test Holm family. They are not independent graph realizations; retaining
rather than deleting them avoids changing the family after evaluation begins.
All summaries identify entries and the report discloses this duplication.

The exact teacher reference is its eventual absorbing law. The primary
empirical test compares conditional completion-by-128 laws. On a small graph,
each still-pending vertex participates and sees all its neighbors
nonparticipating with probability at least 2^(-d(v)-1), then necessarily
commits. Iterating the conditional bound and taking a union bound gives
Pr(teacher cutoff by R) <= n*(1-2^(-Delta-1))^R, capped at one. By decomposing
the eventual law according to completion by R, its TV distance to the
completion-by-R conditional law is at most this tail probability. The
small-TV table reports this conservative reference discrepancy bound;
its TV intervals directly target the exact eventual teacher reference.
For degree-three four-node graphs this bound is about 0.001, below the
reported Monte Carlo uncertainty; absence of observed cutoffs alone is
never used to assert the two mathematical laws are identical.

The bipartite entry `bipartite-n256-d3` has actual maximum degree 4: a
right-side vertex receives four independently sampled left neighbors, even
though the designated hub has the target degree 3. The frozen manifest
correctly records delta=4 and the exact edges. The protocol's degree-3
wording describes the generator target, not this measured maximum. This
entry is retained, not regenerated, and all tables use actual delta=4.
The other target-degree suffixes match their actual maxima. No hypothesis
or observation is removed because of this design discrepancy.
