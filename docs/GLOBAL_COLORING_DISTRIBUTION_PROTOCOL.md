# Global coloring distributions — prospective protocol v1

Frozen before confirmatory runs. Base commit 25cbeb5 (local-palette extension).
The exact checkpoint list, graph list, pilot, source/environment hashes and
this file's hash are recorded in results/ucgnn-global-distribution/freeze.json.
The Git commit containing this protocol precedes every confirmatory sample.
No training, tuning of weights, model selection or remote push is authorized.

## Scientific question and observations

For every fixed labeled graph G and each frozen checkpoint theta, test
H0: P_L^G(. | valid completion by round 128) = P_theta^G(. | valid completion
by round 128), versus inequality. An observation is an entire final vector.
Separate algorithm success and cutoff/invalid/technical frequencies accompany
all conditional comparisons. Complete vectors retain both vertex and color
labels. No checkpoint pooling and no cross-graph pooling of vectors.

Reuse epoch-120 checkpoints normalized, deepsets and transformer, training
seeds 41000,...,41004, exclusively from ucgnn-paper-study. NORMALIZED is the
recent E1 architecture; historical E1 seeds 0,...,4 are excluded. The 15 file
hashes must match the manifest before inference and in the final audit.

Local palettes K_v=d(v)+1 only. The existing LocalPaletteAdapter queries each
receiver's own colors; global maximum degree is never a model feature.
Learned participation, candidate and commit heads remain frozen, with no
oracle safety shield. Permanent confirmed colors and part/pending gating
are the historical structural rules. Luby uses participation probability 1/2,
uniform sorted available colors and exact simultaneous conflict rejection.
Teacher uniforms are float64 NumPy PCG64; network inputs convert the same
uniform distribution to float32 with values rounding to 1 clipped to the
preceding float. These are different numerical realizations and this fact
is retained in interpretation. TF32 is off; deterministic Torch operations;
eval/inference mode, dropout configured zero. Exact teacher mathematics and
finite-precision neural observations are distinguished.

## Graph universe

53 fixed graphs, not an i.i.d. graph-population sample:
* 7 exactly enumerable: edge, paths of 3 and 4 vertices, triangle, 4-cycle,
  3-leaf star and paw (triangle with pendant vertex).
* Random regular graphs at n=256 and n=512, each at actual maximum degree
  2,3,5,10,20,50,100,200 (16 graphs). One graph per cell, fixed named seed.
* Stars of each of these eight degrees (n=Delta+1).
* Path and cycle with 256 vertices.
* Random irregular, irregular bipartite and hub-rooted tree at n=256 and
  degrees 3,20,100,200 (12 graphs).
* Wheel/hub and clique at degrees 3,20,100,200 (8 graphs).

The graph generator is src/dcl/globaldist/core.py. Random irregular graphs
start with G(n,0.7 Delta/(n-1)), replace node 0's edges by a Delta-star and
cap other degrees by deleting non-hub edges. Bipartite graphs have parts
128/128 except degree 200 (56/200); node 0 has exactly Delta neighbors and
other left nodes sample floor(Delta/3) neighbors, at least one. The tree
attaches a Delta-star and extends each branch by a deterministic parent rule.
All labeled edges, actual degrees, degree CVs, seeds (null for deterministic
graphs) and SHA256 including n are archived. A dense clique necessarily has
n=Delta+1. Paths/cycles cannot realize Delta>2. Stars confound size and degree;
only the regular fixed-size grid supports controlled descriptive degree
comparisons. No inference over graph seeds or all graph families is intended.

## Pilot and resource allocation

An independent pilot used seed-41000 policies on path3, degree-20 regular,
degree-200 n=512 regular, degree-200 star and degree-100 random graphs, 12
runs each, plus reset/disjoint-batch checks. All are separate from confirmation.
Archive full pilot timings and null/positive controls. The slowest measured
cell is approximately 5.5 seconds per 12 runs, for heterogeneous degree-100
graphs. A conservative planning envelope of four times this cost for every
large-graph cell plus measured high-degree scaling fits the 12-hour process
budget. Choose m=250 per algorithm and graph for every primary contrast.
This small balanced confirmatory core prioritizes full completion. It does
not promise high power for subtle differences. Additional banks below cannot
change the primary sample size or kernel. GPU process time includes all
sampling, pilot and overhead while holding the evaluation lock. Stop before
43,200 seconds accumulated; preserve all incomplete tasks and data.

## Primary tests and multiplicity

795 planned hypotheses = 53 graphs * 15 checkpoints, fixed before testing.
For every graph, reuse one independent m=250 teacher bank across the 15
contrasts; each network bank has its own independent named RNG stream.
No common random tapes between algorithms. Draws within every bank have
separate seeds per execution. Kernel exp(-d_H(x,y)/n), gamma=1, no scale
selection. U-statistic MMD; B=19,999 random fixed-size label permutations,
p=(1+# U_perm>=U_obs)/(B+1), conservative ties. Familywise alpha=0.05 with
Holm over all 795 hypotheses. If a cell has fewer than 2 successes in either
bank, mark untestable, p=1 for multiplicity bookkeeping, and retain failures.
Do not replace failures by extra successful runs. Report actual successful n.

Report U, nonnegative biased empirical MMD norm, conservative pointwise 95%
population-MMD interval (RKHS mean concentration), p, adjusted p, nominal and
familywise decision. Negative U is allowed. Tolerance for optional approximate
equivalence: population MMD <=0.02, claimed only if its interval upper bound
is below 0.02. These intervals are not simultaneous. No TV claims on large
spaces and no universal normalization across different graphs.

If any invalid/cutoff/technical outputs occur, separately test the four-category
unconditional status distribution by a permutation difference-of-proportions
squared statistic, and a kernel on valid vectors plus three orthogonal failure
atoms. The latter coarsens invalid/cutoff final partial vectors to their category;
it is explicitly not the full law of every internal partial state. These tests
are secondary with their own Holm correction and cannot establish full-law
equality. Archive all final partial/invalid vectors for finer reanalysis.

## Exact small-graph estimation

Separately enumerate valid support and compute rational absorbing teacher laws
for all seven small graphs. Draw fresh m=2000 per generator for distribution
plots and TV estimates. Compare teacher MC against exact probabilities, report
TV to the uniform reference, and exact rational laws. Neural TV confidence
intervals use a finite-support concentration bound, not sparse chi-squared.
No use of these estimation banks in confirmatory MMD tests.

## Controls

Pilot: 100 independent exact-teacher null pairs and 100 known distinct point
mass valid-coloring pairs on path3, m=100, B=199, alpha=0.05.
Final controls: 100 independent Luby/Luby pairs, split equally between path3
and regular-n256-d100, m=100, B=199. Same-checkpoint control: every one of the
15 checkpoints, 30 independent pairs on path3, m=100, B=199. These controls
use new banks, disjoint runs for each pair. Report binomial Wilson intervals
and rates, not a pass/fail selection of models. Positive control: point masses
at two different valid path3 vectors, 100 pairs, same calibration. Compatibility
checks use a 99% binomial acceptance interval for the nominal null rate; a
failure triggers investigation before interpreting primary tests, not deletion.

## Secondary power and classifier analyses

Independent dedicated power banks m=2000 per generator on path3, star-d200
and regular-n256-d100 for seed-41000 of each architecture. Estimate conditional
empirical-bank power at m=100,250,500,1000,2000 by 30 independent with-replacement
resampling experiments per cell, B=199, nominal alpha=0.05. This estimates the
power against empirical bank distributions; repetitions are independent only
conditional on the archived banks. Wilson intervals quantify resampling Monte
Carlo uncertainty, not population-bank uncertainty. Report this limitation
and do not call these independent population-power trials. No retrospective
choice of cases or favorable m. No multiplicity-adjusted power is claimed.

A categorical naive Bayes source classifier uses dedicated 300-run banks per
algorithm on the same three power graphs for seed-41000 of each architecture.
Runs 0:100 training, 100:200 validation, 200:300 independent test. Compare
Laplace smoothing 0.5 and 1 on validation only; fixed 0.5 tie break. Features
are original vertex-specific categorical colors, with conditional-independence
assumption used only for classification, never for the main test. Report
balanced test accuracy and conservative simultaneous class-wise Wilson 97.5%
intervals averaged to a 95% accuracy interval. Conditional on frozen predictions,
permute balanced test source labels 9999 times for significance; train and
validation data never enter this permutation. Holm across nine classifier
hypotheses, secondary to the primary MMD family.

## Structural and across-checkpoint analyses

For every graph and checkpoint report number of used colors, sorted class
sizes, per-vertex/degree color frequencies, co-color probabilities for a fixed
set of up to 256 nonadjacent pairs, and a partition-encoding diversity diagnostic.
These are secondary descriptive functions of full vectors, not replacements
for the global test. Do not pool networks into a mixture. Plot all five
policies separately and describe finite-cohort variation; no population claim
from five seeds. Shared graphs are blocks. Descriptive fixed-n regular curves
separate n, family and Delta. Do not fit an unplanned population regression.
Previously reported proposal differences are context, not fresh evidence about
conditional proposals on the states seen here. P8 is not answered by borrowing
a different graph's proposal test.

## Persistence, provenance and stopping

The queue records every planned bank, atomic archive, SHA256 and receipt.
Each run seed is hash-derived from stage, graph, generator and run index;
graph, training, execution, permutation and resampling namespaces differ.
PCG64 initial states are reproducible from the archived seeds and version;
record the initial state of every bank's first stream and final states hash.
Reset creates new color arrays and adapter per batch, no retained hidden state.
Unit and integration tests cover fresh reset, repeat replay, batch grouping,
exact-teacher MC, strict positive definiteness, null calibration, known
alternatives, actual checkpoint identity, model eval and local palettes.
Technical errors remain explicit. An exhausted budget never licenses selecting
only significant graphs or treating pending hypotheses as successful tests.
Do not modify this document after freeze; append prospective deviations in a
separate file and label any affected analyses. Historical data stay untouched.

Deliver reports, complete data manifests, six figures, per-graph/checkpoint
CSV, theory, this protocol, reproduction instructions, an updated provisional
manuscript section and compiled PDF. Audit all requirements and limitations.
