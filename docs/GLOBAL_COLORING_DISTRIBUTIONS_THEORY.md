# Complete coloring laws: definitions and proofs

This document applies to a fixed, labeled finite graph, not a population of
unlabeled graphs. The elementary observation is a complete independent run.
These are standard Markov-chain and kernel arguments, not claims of new theorems.

## A. Random protocols and absorption

Let C_v={0,...,d(v)} and S=product_v({-1} union C_v). A round is a
measurable deterministic function F(s,U), where U has independent private
uniform participation and proposal coordinates for every vertex. Frozen
weights are constants. Fresh independent U_t give a time-homogeneous Markov
kernel T(s,t)=Pr(F(s,U)=t). Fixed colors remain fixed. Legal complete states are
absorbing. The output law is the pushforward of the entire random trajectory,
not the law of any single proposal q(c|s).

For a cutoff R, retain every complete vector, invalid-complete category,
cutoff category, and technical-error category. The principal law is the
conditional law given valid completion by R. Its identity does not imply
identity of the unconditional law; success probabilities are separate
estimands. Cutoff observations do not prove eventual nontermination.
Implementation errors have an operational frequency, not an intrinsic
mathematical protocol probability without an execution-environment model.

For the exact teacher, every pending vertex has at least one available color:
at most d(v) fixed neighbors exclude colors from d(v)+1 slots. With r pending
vertices, the event that precisely one participates has probability r/2^r,
and makes progress. Thus Pr(no progress for k rounds | state) <=(1-r/2^r)^k.
There are at most n strict increases of the fixed set. Induction on the number
pending establishes almost sure absorption in a proper coloring. This argument
does not certify learned progress or safety.

The exact calculation stores rational transition masses. Proposals of a
nonparticipating vertex integrate out, leaving outcomes '-1' of probability
1/2 and each available participating color of probability 1/(2|A_v|).
Enumerate their Cartesian product, apply simultaneous conflict rejection,
and merge identical next states. Every non-self transition fixes more nodes.
Writing a_s=T(s,s), the absorbing law satisfies

    P_s(x) = sum_{t != s} T(s,t) P_t(x)/(1-a_s).

Backward recursion over the number pending is finite and unique; a_s<1 by
the progress bound. This equals the usual absorbing-chain solution while
avoiding inversion of a large rational matrix. The archived rational masses
sum exactly to one. The code separately enumerates the entire valid support.

## B. Local uniformity does not imply global uniformity

Take the three-vertex path, local palettes (2,3,2), independent participation
probability 1/2 and uniform available proposals. The six complete proper
vectors have probabilities

    (0,1,0), (1,0,1): 343/1628 each;
    (0,2,0), (1,2,1): 205/1628 each;
    (0,2,1), (1,2,0): 133/814 each.

These fractions follow by the finite recursion above, with exact unit tests
and an archived transition-law computation. They sum to one and differ from
1/6. This provides an explicit counterexample: proposal uniformity does not
imply solution uniformity. The two endpoints have lists lacking color 2, so
arbitrary global color renaming is not a symmetry of this sample space.
The uniform law on the six valid vectors is a separate mathematical reference.

## C. Sufficient conditions for equal final laws

Suppose two protocols start in the same state and have the same transition
kernel on every reachable state, including absorption/failure behavior.
For every finite trajectory, its probability is the product of the initial
mass and the successive transition probabilities. These products coincide;
summing over trajectories gives equal finite-time laws. Taking increasing
unions of events 'absorbed at x by time R' gives equal eventual absorbing
masses by continuity of probability. This is sufficient, not necessary:
different transition kernels can have the same absorbing law (for example,
adding an independent holding probability preserves the law).

Equality of candidate marginals alone is insufficient: participation,
candidate/participation dependence, commit rules and joint vertex laws enter
T. Identical deterministic round functions applied to identically distributed
private uniforms are another sufficient condition. Matching a finite set of
sampled round actions is not a proof of kernel equality.

## D. Kernel, MMD, permutation calibration and uncertainty

Fix gamma=1 prospectively. For labeled vectors use
k(x,y)=exp(-gamma d_H(x,y)/n). Let a=exp(-gamma/n), 0<a<1. On a coordinate
with K values its matrix is (1-a)I+a11^T. Its eigenvalues are 1-a (multiplicity
K-1) and 1+(K-1)a, all strictly positive (K=1 has eigenvalue 1).
On the full product of lists the Gram matrix is the Kronecker product of
these positive definite matrices, so all eigenvalues are positive. Restricting
to proper vectors gives a principal submatrix, still strictly positive definite:
extend a nonzero coefficient vector by zeros and use positivity of the full
quadratic form. Therefore MMD^2=(p-q)^T K(p-q)>0 for any distinct laws.
The kernel is characteristic on this finite sample space. It does not require
vertex independence within a vector. Equality under all coordinate marginals
is insufficient for equality under this kernel.

The implemented unbiased two-sample estimator is

    U = sum_{i!=j} k(X_i,X_j)/(m(m-1))
      + sum_{i!=j} k(Y_i,Y_j)/(l(l-1))
      - 2 sum_{i,j} k(X_i,Y_j)/(ml).

Independence of complete runs proves unbiasedness term by term. U may be
negative. The displayed nonnegative empirical effect is the norm of the
difference of empirical RKHS means, the square root of the biased estimator.
Bounded-kernel laws of large numbers make both estimators consistent. Since
there are finitely many possible vectors, empirical frequencies converge;
under an alternative the statistic tends to a positive number. Under random
permutations both groups converge to the pooled distribution, giving a
consistent test as samples and permutation resolution grow. No finite sample
size guarantees useful power against every alternative.

Under the null, pooled complete observations are exchangeable. For each
fixed pooled dataset, the original grouping and uniformly random groupings
have exchangeable statistics. Hence (1+# permuted U >= observed U)/(B+1)
is super-uniform, including ties conservatively. Every grouping preserves
sample sizes; no within-vector coordinate is permuted. Teacher reuse between
checkpoint contrasts creates dependent p-values but does not invalidate any
one test or Holm's familywise control, which holds under arbitrary dependence.
Conditioning on independent successes likewise preserves exchangeability of
the conditional laws; it does not test success rates. B=19999 gives resolution
0.00005, below 0.05/795 for the fixed primary family of 53*15 hypotheses.

For uncertainty, ||phi(x)||=1 since k(x,x)=1. For m independent draws,
E||mu_hat-mu||<=1/sqrt(m), by Jensen and the RKHS variance bound. Changing
one observation changes this norm by at most 2/m. McDiarmid's inequality gives
with probability at least 1-delta,
||mu_hat-mu|| <= 1/sqrt(m)+sqrt(2 log(1/delta)/m).
Use delta=0.025 in each of the two samples and the triangle/reverse triangle
inequalities to obtain the reported conservative 95% interval for population
MMD. Clip it to [0,sqrt(2)], since k is nonnegative. These are pointwise
intervals, not simultaneous architecture or graph-population intervals.
Approximate equivalence at preregistered tolerance epsilon=0.02 is asserted
only if this upper endpoint is below epsilon. Wide intervals are retained.

On a small graph with s possible vectors and exact teacher law p, plug-in TV
has upward estimation noise. Let q_hat be the empirical neural law from m
successful independent samples. TV(q_hat,q)=max_A(q_hat(A)-q(A)). For each
of 2^s subsets, Hoeffding's inequality bounds the upper tail by exp(-2mt^2).
A union bound yields radius sqrt((s log 2+log(1/alpha))/(2m)). The triangle
inequality for TV then gives a valid interval about TV(p,q_hat) for TV(p,q).
This remains valid despite sparse cells and may be conservative. No sparse
chi-squared approximation is used. Uniform-reference and teacher-MC TVs are
reported separately. Confidence intervals do not remove finite-support bias.

## E. Interpretation and graph comparisons

Mathematical equality requires a proof, approximate equivalence an appropriate
upper bound below a prespecified tolerance, and rejection an appropriately
calibrated test. Nonrejection means insufficient evidence at the available
sample size, kernel and multiplicity correction. A test can miss real
alternatives. A positive control checks only its specified alternative;
repeated null controls have binomial uncertainty and cannot certify universal
calibration.

Normalized Hamming does not make MMD a universal comparable effect across
n, degree, family or palette sizes: the kernel feature geometry and spectrum
change. Degree comparisons are therefore shown within fixed n and regular
family, and are descriptive finite-battery contrasts. Training replicas are
five fixed policies, never mixed. The same graphs are shared blocks; effects
are tabulated per graph and checkpoint. No graph-population regression or
pseudo-independent graph-by-checkpoint confidence interval is claimed.
Spatial diagnostics and color-class counts are deterministic functions of
complete vectors: they may discard distinguishing information. Restricted
growth encodings describe unlabeled partitions, not an arbitrary color-name
group action preserving heterogeneous lists. The main hypothesis retains
original vertex and color labels throughout.

Primary verified reference: Gretton et al. (2012), *A Kernel Two-Sample Test*,
JMLR 13:723–773, https://www.jmlr.org/papers/v13/gretton12a.html . MMD and
kernel two-sample testing are established methods; the finite-space kernel
proof and the explicit teacher recursion above specialize them to this study.

## F. A localized alternative can have small kernel effect at large TV

Suppose P=p_v product R and Q=q_v product R, with independent coordinate v
and remaining coordinates. Factor the product kernel into its v and remaining
parts. Expanding MMD gives

    MMD_k^2(P,Q) = E_{Z,Z'~R} k_{-v}(Z,Z') (1-a)||p_v-q_v||_2^2,
    a=exp(-gamma/n).

Indeed the one-coordinate Gram matrix is (1-a)I+a11^T; the 11^T term
vanishes because the entries of p_v-q_v sum to zero. Thus 1-a is of order
1/n even when TV(p_v,q_v) is large. This statement assumes the displayed
factorization; it is not assumed for the learned laws.

There is a valid-coloring example: on a degree-D star, leaves are independent
uniform {0,1}, while the center is independent and takes colors in {2,...,D}.
All outputs satisfy the local lists and all edges. At D=200 let p be uniform
on the 199 high colors and q uniform on the first 100 of them. Their global
TV is 99/199, by direct summation (the common leaf law factors out).
Their exact MMD squared at gamma=1 is

    ((1+a)/2)^200 (1-a) (1/100 - 1/199), a=exp(-1/201).

This is strictly positive but small. Characteristicness does not guarantee
high finite-sample power for this alternative, nor make MMD a universal TV
surrogate across graph sizes. It motivates the explicitly supplementary
localized positive-control diagnostic without changing the primary kernel.

For numerical ties the software uses a conservative guard
1e-14*min(1,500/(m+l)), which is 1e-14 throughout the primary family.
This threshold tends to zero as samples increase; the above ideal-arithmetic
consistency argument therefore also holds with this conservative guard.
Floating-point resolution itself remains a practical limitation.

The exact teacher has full support on C(G): for any proper x, the event
that all vertices participate initially and each proposes x_v has probability
2^(-n) product_v (d(v)+1)^(-1)>0. Since x is proper, all confirmations succeed
in that first round. This proves positivity of every valid final coloring,
not uniformity of those probabilities or full support for a learned policy.
