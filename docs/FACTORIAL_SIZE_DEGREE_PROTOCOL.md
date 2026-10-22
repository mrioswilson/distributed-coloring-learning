# Factorial size–degree study: prospective protocol

This protocol and the exact design in `results/ucgnn-size-degree/freeze.json`
must be committed before any confirmation bank is generated. Development
pilot outcomes are not used to choose cells, checkpoints, events or test
components. The parent study is commit
`74134d865db4f142bfeea49b091200588d1f2803`; its results are preserved.

## Scientific object and scope

For a fixed labeled simple connected graph G, compare the exact local Luby
teacher with each of the fifteen historical epoch-120 checkpoints: five
NORMALIZED, five DeepSets, five LocalPaletteTransformer, training seeds
41000–41004. SHA256 digests and original paths are archived. No training,
checkpoint averaging, global-degree input, new action shield or remote push.

The primary observation is one complete vector (c_1,...,c_n), with vertex
and color labels retained. Nodes in a run are not independent observations.
Local lists are precisely 0,...,d(v). Participation, proposal, floating point
conversion and COMMIT are the original frozen numerical implementations.
The teacher uses the exact local availability/rank rule, not a learned proxy.

All runs stop at round 128. Status codes 0,1,2,3 mean proper completion,
improper completion, cutoff, technical error. Nothing is silently retried or
discarded as a successful run. The principal distributional estimand is the
law conditional on proper completion within 128 rounds. We also report each
status frequency and compare success probabilities separately. If any status
is nonzero, retain the status-law permutation and coarsened unconditional
kernel tests; their p-values are descriptive secondary checks, not extra
primary rejections. No conclusion about the unrestricted eventual law follows
solely from a finite-cutoff conditional comparison.

## Graph factorial and cost-only design rule

Main matrix: n in {128,256,512,1024}, degree in {3,10,50,100}, exactly three
new independent connected random regular graphs in every one of the 16 cells.
NetworkX `random_regular_graph` with independent namespaced seeds is retried
only for connectivity, at most 100 attempts. The generator need not be an
exact uniform sampler of all regular graphs. These 48 fixed realizations are
the target cohort; three graphs do not establish universal population trends.

The secondary heterogeneous sequence has approximately 1% degree-delta hubs,
35% degree max(2,floor(delta/3)), and the rest degree two. One middle degree
is adjusted by one for parity when necessary. Labels are independently
shuffled. Havel–Hakimi realizes the graphical sequence, cycle-edge switches
connect components while preserving degrees, and 10|E| attempted connected
double-edge swaps randomize it. No claim of uniformity or mixing is made.
Record achieved degrees, successful swaps, mean, variance, histogram, near-hub
fraction and local-list histogram. It is a particular heterogeneous family.

One independent irregular graph per cell is the preferred secondary design.
Before freezing, use fresh 128-run timing pilots on all regular and irregular
cells and the first checkpoint of each architecture. Candidate secondary
designs, in fixed order, are: all sixteen cells; n={128,1024} crossed with all
four degrees; n={128,1024} crossed with degree={10,100}. Select the first for
which charged pilots + 1.25 times the predicted complete confirmation cost +
900 seconds for mechanistic work is at most 43,200 GPU-process seconds.
The exact selected subset, forecasts and all alternatives are archived. The
48-graph principal matrix is never pruned by significance. If even this
cost rule cannot fit the principal cohort, stop before confirmation and make
a prospective revision rather than silently launching an infeasible plan.

An additional stratum uses exactly the archived regular-n512-d100 edge list
with all fifteen checkpoints. Its new seeds and new outcomes are independent
of historical banks. Historical last-250 event counts and source hashes are
reproduced separately and never included as fresh data.

Audit graph size, every degree, connectedness, loops, directed edge uniqueness,
labeled edge hashes and duplicates against all new/pilot/historical graphs.
Freeze graph files; do not regenerate after reading confirmation outcomes.

## Samples, independence and resources

For every retained graph and each of the sixteen generators, attempt exactly
625 independent runs. The first 500 are the fixed-size confirmation sample;
the final 125 are an independent explanation holdout. The teacher bank is
shared across the fifteen comparisons on that graph and is counted only
once when constructing architecture-level effect estimates and uncertainty.
Training replicas remain separate throughout.

The principal omnibus uses floor(m_valid/2) successful rows from each 500-run
prefix for selection and the rest for evaluation. Their order is the fixed
run index, never sorted by outcomes. Ancillary identifiers may establish this
same order after file shuffling; vertex columns are never canonicalized.

Sampling reuses `dcl.globaldist.sampling.sample` unchanged, with a separate
PCG64 seed per complete run derived by the original SHA256 function from
stage `factorial-v1/confirmation`, fixed graph ID, generator and run index.
Pilot/cost/mechanism stages have distinct namespaces. Record every seed,
initial RNG metadata and final-state digest; check collisions internally and
against both historical studies. Stateless graph copies are disjoint unions;
verify action equivalence of read-only instrumentation against the sampler.

The persistent single-GPU lock excludes concurrent study workers. Charge
initialization, weight loading, pilots and every task to one 12-hour ledger;
charge interrupted unreceipted active wall time conservatively on recovery.
A BaseException alarm bypasses the sampler's ordinary error handler and
stops before the hard cap. Banks and receipts are written atomically, and
completed banks are never regenerated. Principal tasks precede historical
replication and secondary tasks. Reserve 900 seconds for read-only mechanistic
work. CPU permutation, graph, uncertainty and plotting jobs consume no GPU.
No foreign process is stopped. No optional sample expansion or interim
significance stopping is permitted.

## Unchanged omnibus V2 and baseline

Reuse the robust study's exact `stats.py`, source dependencies and hashes.
Its supervised selection is confined to the selection half: four local,
four pair/triple interaction and four structural witnesses, a dense marginal
witness, and four global kernel witnesses with gamma=(.25,1,4,16). Pair
regions include the original fixed graph/seeded groups and top-eight raw and
median-binary covariance-change pairs from the selection half. No new scan
or graph-size-specific tuning is introduced. Triples remain limited; remote
unknown higher-order alternatives are a known blind spot.

Pooled evaluation standardization ignores source labels. The maximum jointly
combines witness mean discrepancies and the characteristic full-vector kernel
exp(-d_H/n) U statistic. The kernel U estimate retains negative values.
All evaluation-label permutations keep selection fixed and permute complete
rows jointly across components. This preserves conditional exchangeability
under equal generator laws; it makes no within-coloring independence claim.

Use B=19,999 Monte Carlo permutations, inclusive ties with the frozen numerical
tolerance, and p=(1+exceedances)/(B+1). The minimum is 0.00005, below 0.05/720
for the largest primary family. Retain the old full-sample gamma-one unbiased
MMD on the identical 500-run successful prefixes as a separate secondary
method family. Neither bandwidth nor gamma-one normalization is changed.

The original conservative futility shortcut may return p=1 after at least
20 exceedances of every component. The corresponding full-B p is at least
21/20,000. Apply the archived optimistic-Holm refinement: any shortened test
that could affect the full-family rejection set is replayed from the same
permutation seed through all B draws. This yields exactly the full-B Holm
rejection set. Report conservative p=1 values as such, not accurate tail
estimates. No parametric tail extrapolation.

Before confirmation, require exact reproduction of archived V2 statistics and
p-values, frozen-source and checkpoint hashes, regression tests, and an
independent transfer check. The transfer check uses 200 null trials for each
of three explicitly finite empirical laws, 20 point-mass null trials, and
30 trials each of the mandatory localized star and diffuse alternatives at
the strict 0.05/720 threshold. All positive-control pointwise exact binomial
95% power lower bounds must exceed .80. These checks supplement, rather than
replace, the prior large calibration study, including its documented failures.

## Multiplicity

Primary family: all 48 x 15 = 720 graph/checkpoint complete-law hypotheses,
Holm FWER .05. Secondary irregular family: 15 times the frozen irregular graph
count, Holm .05. Independent historical-graph replication family: 15, Holm
.05. Old MMD has its own corresponding method families. Predefined events,
completion rates and selected holdout witnesses each have separate families
within each graph stratum. Event families contain two tests per comparison;
the selected witness family contains one per comparison. P-values are always
reported raw and adjusted. These separate families do not provide a combined
.05 FWER over their union.

## Preregistered effects and factorial inference

Two primary Bernoulli observables on a completed coloring:

* A={U(X)<=delta}, with U the number of distinct colors;
* T={color delta occurs}.

A indicates that at least one palette color is unused; it does not imply that
the top color specifically is unused. T measures that distinct question.
Estimate both event probabilities separately for each graph and checkpoint.
For regular graphs degree also sets every local-list size K=delta+1. This
factorial cannot identify separate causal degree and palette-size effects.

The elementary model is a saturated stratified binomial model: one free event
probability for each independent graph/generator bank, conditional on its
valid completion count. Graph/checkpoint effects are not assumed equal.
For architecture summaries use equal weights over the five fixed checkpoints
and three fixed graphs in each cell, sharing the same teacher term once.
This estimates a fixed-cohort risk difference D(n,d)=mean(P_M(A)-P_L(A));
it is not an estimate of a population of infinitely many trained models.

For each of two events and 18 groups (15 individual checkpoints plus three
architecture means), preregister 50 contrasts: sixteen cell differences,
twelve adjacent-n changes at fixed degree, twelve adjacent-degree changes at
fixed n, nine adjacent difference-in-differences and one four-corner contrast.
There are 1,800 contrasts total. Interaction is on the risk-difference scale,
not an invariant causal interaction or a post-selected logit interaction.

Construct simultaneous CP intervals for every graph/generator event p with
total failure budget .025. Propagate these linearly with coefficient signs.
Also use independent-run weighted Hoeffding intervals for all 1,800 contrasts
with total budget .025: if bank weights are w_i and valid counts N_i, radius
is sqrt(log(2/alpha_c)*sum(w_i^2/N_i)/2), alpha_c=.025/1800. Combine duplicate
teacher weights before calculating the radius. Intersect both intervals,
record any conflicting bounds. A union bound gives simultaneous coverage at
least .95 for all registered contrasts under the fixed-bank model. No graph
or checkpoint is mistaken for an additional independent coloring run.

For a fixed graph, any event-probability difference interval [l,u] gives
TV(P_L,P_M)>=max(0,l,-u), with the elementary simultaneous coverage above.
This is a lower bound only. Comparing lower bounds at different graph sizes
does not prove that their full-vector TV distances are ordered. Cross-size
claims concern the same interpretable events and structural functions, not
TV between spaces of different-dimensional vectors.

Show deterministic leave-one-graph and leave-one-checkpoint sensitivity ranges,
individual graph points and all five training replicas. They are sensitivity
analyses, not population confidence intervals. No monotonicity is assumed.

## Structural distributions and explanatory holdout

Archive per-run distinct count, maximum color, top-color indicator, U/K,
largest class fraction, class entropy/log(K), mean c_v/max(d(v),1), per-color
frequencies and equality frequency over all ordered nonadjacent pairs.
Empirical projected Wasserstein distances use these bounded normalized
features. Their DKW confidence intervals are pointwise descriptive 95%
intervals, not multiplicity-adjusted evidence for the principal conclusions.
Record gamma-one MMD U, biased norm and a conservative population-norm bound;
MMD values are not a universal cross-graph distance scale.

For each graph/checkpoint choose exactly one observable event from V2's
selected descriptors using only the first-500 bank. Estimate and validate it
on the final 125 independent attempts. Fisher tests use Holm in the separate
witness family; CP bounds simultaneously cover both generators' probabilities
over all selected events. A selected event can be validated even if the
global test did not reject, but is never added to the primary rejection count.
All per-color appearance probabilities are also measured prospectively with
a separate simultaneous CP family over all colors, graphs and generators;
do not give post-hoc selected-color intervals a naive confirmatory reading.

## Mechanistic and exploratory analysis

For the first graph in every regular cell and the historical target graph,
collect eight fresh trajectories per generator with read-only proposal,
participation, exact-availability reference and COMMIT observers. No diagnostic
changes participation, candidates or COMMIT. Retain whole-run trajectories
and per-run color counts, rather than declaring nodes independent replicates.

Probe common states from the first independent Luby diagnostic trajectory at
rounds 0,2,4,8, taking the first uncolored labeled vertex when present. Compare
every checkpoint on the identical state. Midpoint uniform grids of 1,024 and
4,096 points approximate candidate masses; report their discrepancy and do
not claim an exact integration error guarantee. Relate candidate top-color
mass, same-state teacher availability, accepted-color counts and independent
confirmation-bank final usage. Association and this temporal accounting alone
do not identify a causal intervention effect.

Any regressions comparing log(n), log(K), log(n/K), density or interactions
are explicitly exploratory. Use binomial counts and checkpoint intercepts,
with graph-replicate held-out predictive comparison where feasible; report
all specified candidate forms, not just the most attractive curve. Do not
substitute their fitted p-values for the registered simultaneous contrasts.

## Required reporting and limitations

Produce the nine requested figures from archived data, P1–P10 answers, exact
historical reproduction versus independent replication, and a manuscript
section entitled “How Graph Size and Degree Affect Distributional
Generalization”. Preserve historical sections, compile and inspect the PDF.
Report nulls, nonrejections, cutoffs and incomplete tasks without reinterpretation
as equality. Only graph-specific proof or suitable equivalence inference can
establish equality or approximate equivalence. Prior small-graph TV results
remain confined to their previous conditional estimands. This experiment
does not certify universal detectability, causal degree effects, or universal
neural equivalence/correctness outside the training domain.
