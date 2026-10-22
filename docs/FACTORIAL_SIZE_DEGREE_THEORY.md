# What size and degree can identify

Let P^G_M be the law of the complete **labeled** coloring vector for fixed G
and fixed policy weights. In this study its operational estimand is
P^G_M(. | proper completion by round 128). Each random run is an observation;
the n vertex colors in a run can have arbitrary dependence. Status laws are
retained separately. No canonicalization identifies vectors up to color or
vertex permutations.

## Exact symmetry of the ideal teacher on regular graphs

On a delta-regular graph every palette is the same K=delta+1 colors. Under
ideal independent continuous randomness, any global permutation sigma of the
color names commutes with a teacher transition in distribution. Uniform choice
from available colors pushes forward to uniform choice from their permuted
set; participation is color independent; COMMIT depends only on equality of
colors. Starting from the colorless state, induction proves color-permutation
invariance at every round. Proper completion by round 128 is invariant too,
so the conditional completed law has the same invariance.

Consequently, for every vertex v and color c,

    P_L(c_v=c | complete)=1/K,
    E_L[|class(c)| | complete]=n/K,
    E_L[U | complete]=K P_L(color c occurs | complete).

These identities need no independence between vertices, vertex transitivity,
or uniform distribution over proper colorings. They do **not** prove uniformity
of the full coloring law on a general regular graph. The numerical rank
implementation uses finite precision; the ideal theorem should not be called
a bitwise floating-point symmetry certificate. Empirical symmetry diagnostics
are reported as diagnostics, with this distinction retained.

For heterogeneous local prefix palettes, arbitrary global color permutations
need not preserve all lists. The proof applies only to permutations stabilizing
every local list. In particular, maximum-degree colors are available at fewer
vertices. Regular and irregular results must therefore be stratified.

## Why proposals can have no n dependence while occupancy does

All three studied policies use MAX neighbor aggregation. In the all-uncolored
state, every incoming first-stage message has the same finite message type.
At a nonisolated vertex MAX therefore returns that same vector regardless of
the number of neighbors. At fixed K, its deterministic first-stage state and
candidate function of private r_c are independent of n and of the particular
regular edge list. This statement follows from the frozen adapter's finite
message table, not from failure to reject a statistical null. It applies to
the all-uncolored state; mixed states can differ through their message types.

Changing delta on regular graphs changes K. The normalized E1 prefix intervals
then shrink in width roughly as 1/K, while the fitted neural scorer remains
fixed. Whether that produces top-color underuse is measured, not assumed.
Candidate-grid probes compare every checkpoint on identical teacher states;
the grids are approximations, with coarse/fine discrepancies reported.

Even a fixed per-node bias can affect an entire-vector occupancy event
differently at different n. In the **independent categorical occupancy toy**,
if a color has probability q, the probability it is absent is (1-q)^n.
Against a uniform reference it differs by

    |(1-q)^n - (1-1/K)^n|.

This is an explicit event lower bound on TV for that toy model. Actual proper
colorings have dependent colors; this formula is not substituted for measured
graph probabilities and is not a causal model fitted to the final data. It
illustrates why the same local discrepancy need not yield a constant global
event difference as n changes. Both occupancy saturation and palette size
can matter, so monotonicity of any discrepancy is not automatic.

## Identified factorial effects

The confirmatory response is an event probability, not a rejection indicator.
For a fixed architecture a, define

    D_a(n,d)= (1/(3*5)) sum_{g,theta} [p_{a,theta,g}-p_{L,g}].

The graph sum is over the three preregistered realizations of that cell. The
teacher term is shared across all five policies, not five independent banks.
Adjacent-n, adjacent-degree and difference-in-differences contrasts of D are
defined prospectively. A nonzero interaction on this probability scale is
an interaction of observed fixed-cohort risk differences. It is not proof of
a unique mechanistic or causal interaction under arbitrary reparameterization.

The saturated binomial model gives one probability per fixed bank. Whole-run
independence, conditional on valid counts, permits exact Clopper–Pearson
coverage. The weighted Hoeffding bound for a contrast sum w_i p_i is

    P(|sum_i w_i(p_hat_i-p_i)| >= t)
       <= 2 exp(-2 t^2 / sum_i w_i^2/N_i).

This follows by applying the bounded independent-variable inequality to the
N_i Bernoulli observations with weights w_i/N_i. First combine any repeated
bank coefficients: the shared teacher must not be counted as independent
replications. With the protocol's two union-bound allocations, the intersected
intervals simultaneously cover all 1,800 contrasts with probability at least
.95. Correlation between different contrasts is allowed by the union bound.
Conditioning on independent per-bank completion counts preserves the argument
for the conditional laws. No graph-population independence approximation is
needed for these fixed-graph estimands.

Three graph draws and five training seeds are small scientific cohorts. Their
leave-one-out ranges measure sensitivity, not the uncertainty of an entire
unobserved population. Exploratory binomial regression predictions are not a
replacement for the simultaneous fixed-cohort inference.

## Complete-vector distances and projections

For an event A on one fixed graph,

    TV(P^G_L,P^G_M) >= |P^G_L(A)-P^G_M(A)|.

Simultaneous event probability intervals thus yield simultaneous graph-specific
TV **lower** bounds. Two lower bounds, on the same or different graphs, do
not order the unknown true TV distances. Across different n the vector spaces
also differ. We compare the meaning of U<=delta, top-color usage, normalized
occupancy and other functions rather than taking a cross-dimensional TV.

The gamma-one Hamming kernel is strictly positive definite on the finite
categorical product and hence characteristic on a subset of valid labeled
colorings. Its population MMD separates unequal fixed-space laws, but finite
sample power and scaling with n remain separate issues. V2 retains this
component and supplements it with holdout-calibrated local and structural
witnesses; its known high-order and rare-event weaknesses remain. The old
star failure and prior validation are preserved, not reinterpreted as an
invalid Holm correction. Small MMD never becomes a useful TV upper bound
without an explicit quantitative graph-specific inequality.

Projected Wasserstein distances concern normalized structural functions only;
they can be zero for different complete laws. For bounded functions in [0,1],
W1 equals the integrated CDF discrepancy. Two DKW CDF bounds and the triangle
inequality yield the pointwise confidence bands used here. They are separate
from the multiplicity-controlled primary factorial statements.

## Mechanism and causality

Candidate q_M(c|S), authorization probabilities and P^G_M(X) are different
objects. Read-only records can connect a same-state proposal bias to accepted
colors and final color classes along actual paths. They do not prove that
changing that bias alone would change the terminal law by a specified amount.
Mixed-state availability, simultaneous conflicts and repeated rounds can
amplify or offset a proposal difference. No observational correlation is
called a causal mediation estimate.

A separate prospectively documented candidate-channel intervention goes
beyond these observational records. It retains learned participation and
COMMIT, replaces only candidates by exact local teacher proposals, and never
feeds its paired outputs into the original independent two-sample tests.
Its coupling bounds and direct original-versus-intervened event effects are
derived in `FACTORIAL_CANDIDATE_INTERVENTION_PLAN.md`. Such a controlled
computational change can establish an effect of that channel for the fixed
policy and graph, without identifying complete mediation or the separate
causal effects of degree and palette size. A loose coupling cannot prove
residual discrepancy from a large TV upper bound.

On regular graphs delta and K are deterministically linked. This design
controls observed n versus observed delta/K but cannot separate their causal
roles. Correctness of all recorded colorings, equality by a precise symmetry
theorem, statistically supported approximate equivalence, and failure to
reject equality remain four different kinds of statement. Only the first
three have positive evidentiary content of their stated scope.
## Numerical score reference beyond midpoint grids

An additional exploratory diagnostic integrates the one-hidden-ReLU scorer
for E1 and DeepSets. Its frozen first-stage float32 features and weights are
promoted to float64; as a function of the private scalar r, each color score
is piecewise affine. ReLU breakpoints partition [0,1], and the upper envelope
of the affine color scores gives the winning-color masses. Tests include
nonmonotone winner regions, ties and an independent dense-grid comparison.
Transformer attention is not piecewise affine in r and is excluded here.

This is a numerical reference for a real-arithmetic surrogate, not a certified
probability law for the GPU float32 rollout. In particular, a zero integrated
reference mass or a positive reference dominance gap must not be presented
as a proof that the operational policy can never choose that color in every
reachable state. It addresses the initial frozen state only. The reference
and grid diagnostics are archived separately and do not select a test,
sample size, checkpoint or confirmatory hypothesis.

## Methodological references

The coverage derivations above are given explicitly rather than importing
guarantees from unrelated settings. Relevant original references are
[Clopper and Pearson (1934)](https://academic.oup.com/biomet/article-abstract/26/4/404/291538)
for binomial confidence limits,
[Hoeffding (1963)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830)
for bounded independent sums, and
[Massart (1990)](https://projecteuclid.org/euclid.aop/1176990746)
for the sharp DKW constant. No asymptotic graph-population guarantee is inferred
from these finite independent-run bounds.
## A sufficient power guarantee for event contrasts

This calculation concerns the registered Bernoulli event contrasts, not
omnibus power or a uniform guarantee in complete-vector TV. With independent
bank counts N_i, combined weights w_i, S=sum(w_i^2/N_i), and a simultaneous
Hoeffding radius R=sqrt(S log(2/alpha_c)/2), a true contrast delta>R satisfies

    P(the Hoeffding interval excludes zero)
      >= 1 - exp(-2(delta-R)^2/S).

Indeed, failure to exclude zero requires the empirical contrast to be at most
R; apply the one-sided bounded-sum inequality at delta-R. The symmetric result
holds for negative delta. Intersecting with the CP interval cannot revoke an
exclusion by the Hoeffding interval. These power statements are conditional
on the valid counts, not a guarantee of obtaining those counts by R128.

For equal N_i=m on the regular cohort, sum(w_i^2) is .4 for an architecture
cell, 2/3 for a single-checkpoint cell, .8 for an architecture adjacent change,
and 1.6 for a four-cell interaction. The shared teacher accounts for 1/3 of
the .4 cell value; treating it as five independent teachers would be wrong.
Thus delta >= R+sqrt(S log(1/(1-beta))/2) suffices for power beta. The archived
table evaluates the formula for m=100,250,500,1000,2000 and beta=.8,.9, using
alpha_c=.025/1800. It is deliberately conservative, does not exploit favorable
event variances or the additional CP bound, and never justifies a TV upper
bound after a nonrejection. Sample allocation remains the frozen fixed design.
## A list-capacity limitation in the secondary cohort

For prefix local palettes, every assignment respecting its lists, proper or
not, obeys the deterministic upper bound

    U(X) <= min_{0<=t<=delta+1} [t + #{v: d(v)>=t}].

For a fixed t, at most t used colors are below t; each color at least t must
be assigned to a vertex whose degree is at least t, and each such vertex
supplies only one terminal color. Minimize these valid bounds over t.

If this upper bound is at most delta, A={U<=delta} is identically true under
every completed generator. Its observed zero difference cannot diagnose
similarity of the complete laws. For example, ten degree-100 hubs, 358
degree-33 vertices and 656 degree-two vertices have U<=34+10=44, even though
the maximum palette has 101 colors. The graph may still admit substantial
differences detectable through other labeled events or the complete test.

This calculation is archived for every graph and is independent of algorithm
outcomes. It does not change the frozen tests, event definitions, graph files
or external families. In particular, the secondary A-matrix is annotated
where the event is forced by the local lists. Comparing its zeros with the
regular matrix would otherwise confound a support restriction with algorithmic
agreement. Maximum degree alone is not an effective global palette capacity
in a heterogeneous family.
