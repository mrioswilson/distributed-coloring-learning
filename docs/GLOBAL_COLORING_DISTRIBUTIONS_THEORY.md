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

## G. Symmetry forces identical conditional solution laws on cliques

This is a standard finite group-invariance argument, not an original theorem.
Let a finite group H of vertex automorphisms preserve every local list, and
suppose the round function is H-equivariant when colors and private uniforms
are permuted together. Suppose private uniforms are iid across vertices and
rounds, and the initial uncolored state is invariant. Induction on rounds
shows that the trajectory law is H-invariant: the transformed uniforms have
the same law, and equivariance transforms every successive state. The event
E_R='valid completion by round R' is invariant. If Pr(E_R)>0, conditioning
therefore leaves the output law H-invariant. The same argument holds for
eventual valid completion by a countable union over R.

If H acts transitively on C(G), all valid vectors have equal conditional
probability: for x,y choose h with hx=y and use invariance. Their masses sum
to one, so each is 1/|C(G)|. Vertex-transitivity alone is not sufficient;
transitivity on the entire coloring space is the required hypothesis.

For G=K_n and local lists C_v={0,...,n-1}, every proper coloring is a
bijection from vertices to colors. The vertex automorphism group S_n acts
transitively on these n! bijections. Consequently any such equivariant
protocol, including Luby and each of the three selected shared-weight
architectures, has conditional output law Uniform(C(K_n)). It need not have
uniform proposals, equal round counts or equal success probability. Failure
masses can distinguish unconditional laws even when conditional laws agree.

For the selected architectures, shared row-wise MLPs, neighbor MAX, shared
participation/commit rules and per-node palette attention all commute with
vertex relabeling in real arithmetic. Positions c/(K_v-1) order colors but
are not vertex identifiers. Permanent-state updates and simultaneous commit
gating also commute with relabeling. Therefore the architectural symmetry
condition holds for every fixed checkpoint in the ideal mathematical model;
no training-seed mixture is needed. Floating-point batch kernels are audited
numerically, not universally certified bit-for-bit equivariant on every
possible input, so the ideal result must be distinguished from a numerical
symmetry certificate for the particular hardware implementation.

On an initially uncolored clique K_201, every node's first-message MAX is
the pending-neighbor message. Its own palette is 201, so its proposal head
has exactly the same deterministic first-stage input as the center of the
archived all-free degree-200 star witness, regardless of its neighbors'
palette lengths. Thus the earlier K=201 all-free proposal-law discrepancies
apply to clique initial proposals too, while the conditional ideal complete
law is nevertheless uniform by symmetry. This gives a precise mechanism
by which proposal imitation and final-law equivalence diverge. It does not
claim arbitrary graphs have uniform Luby laws; path3 is a counterexample.

The four-cycle supplies a homogeneous-list counterexample too: all lists
are {0,1,2}. Exact recursion gives probability 35051/859584 to each of the
six two-color alternating vectors, and 36071/573056 to each of the twelve
three-color vectors. They sum to one; TV from Uniform(C(C4)) is
38111/429792, approximately 0.0886731. Thus teacher nonuniformity is not
only a consequence of heterogeneous local lists. The vertex/color symmetries
have two orbits here, unlike the transitive clique case.

In the localized-star construction, let the center's alternative support
have h=ceil((D-1)/2) of the D-1 high colors. TV=1-h/(D-1) tends to 1/2,
while the exact MMD squared is

    ((1+a)/2)^D (1-a) (1/h-1/(D-1)), a=exp(-1/(D+1)).

Since ((1+a)/2)^D tends to exp(-1/2), 1-a is asymptotic to 1/D, and
1/h-1/(D-1) is asymptotic to 1/D, MMD squared is asymptotic to
exp(-1/2)/D^2. This compares a sequence of different graph spaces, not a
failure of characteristicness for any fixed graph. It explicitly shows why
normalized-Hamming MMD cannot be treated as a universal degree-independent
surrogate for TV, and why sample complexity can increase with degree for
localized alternatives whose TV does not shrink.

For equal source priors, a classifier selecting Q on a measurable set A has
balanced accuracy (1+Q(A)-P(A))/2. Maximizing over A gives (1+TV(P,Q))/2.
For the known localized D=200 alternative, the set 'center in {2,...,101}'
attains balanced accuracy 149/199, approximately 0.7487. Thus the observed
MMD sensitivity limitation is not an information-theoretic impossibility of
distinguishing those two artificial generators when their laws are known.

## H. Elementary concentration and multiplicity details

For completeness, the concentration steps above can be derived without
assuming a special infinite-dimensional theorem. On a finite space, take
the columns of K^(1/2) as Euclidean feature vectors; their inner products
are k and squared norms are one.

**Bounded-variable exponential lemma.** If Z is in [a,b], let
F(t)=log E exp(tZ). Its second derivative is the variance under exponential
tilting, bounded by (b-a)^2/4: the squared distance from the interval midpoint
is at most that number, and variance is no larger than the mean squared
distance from any fixed point. Since F(0)=0 and F'(0)=EZ, integrating this
second-derivative bound yields

    E exp(t(Z-EZ)) <= exp(t^2(b-a)^2/8).

The same integration applies to negative t. This is the usual bounded-variable
exponential lemma, not a new result.

**Bounded differences.** For independent X_1,...,X_m and a real function f
whose value changes by at most c_i when only X_i changes, use the Doob
martingale M_i=E[f|X_1,...,X_i]. Conditional on the past, the martingale
difference has range length at most c_i: integrating over the independent
remaining variables preserves the bounded difference as X_i varies.
The preceding conditional exponential bound, iterated over i, gives

    E exp(t(f-Ef)) <= exp(t^2 sum_i c_i^2/8).

Chernoff's inequality and t=4u/sum_i c_i^2 give

    Pr(f-Ef >= u) <= exp(-2u^2/sum_i c_i^2).

For f=||mu_hat-mu|| and ||phi(x)||=1, c_i=2/m. Also, expanding the squared
norm and using independence to cancel cross terms gives
E||mu_hat-mu||^2=(E||phi||^2-||mu||^2)/m<=1/m. Jensen's inequality yields
the stated expectation bound, and the bounded-difference tail yields the
MMD confidence radius used in the code. Applying bounded differences to
an empirical subset probability with c_i=1/m gives the Hoeffding bound used
in the TV argument. Thus both reported concentration intervals follow from
the displayed elementary argument.

**Finite-space consistency.** Applying the same indicator bound to every
point of the finite sample space gives convergence of empirical frequencies
in probability (and almost surely by summability and Borel--Cantelli).
MMD squared is a finite quadratic polynomial in those frequencies; its
biased version converges to the population value, and the unbiased version
differs by a term tending to zero. Conditional on a pooled finite sample,
a uniformly permuted group's frequency at any point is hypergeometric,
with variance at most 1/(4m); the complementary group has the analogous
bound 1/(4l). Both groups therefore approach the pooled frequencies, uniformly
in the underlying pooled composition, as m,l grow. Their permuted MMD squared
tends to zero in probability. Under an alternative, the observed statistic
tends to its strictly positive population value. The probability that a
random permutation exceeds it tends to zero. For fixed B, a union bound
shows that all B permutations fall below it with probability tending to one;
the limiting Monte Carlo p is 1/(B+1). This gives consistency whenever the
fixed test threshold exceeds that resolution. The primary B=19999 meets
0.05/795. With B tending to infinity, Markov's inequality applied to the
fraction of exceedances also gives consistency without needing every
permutation to be small. A tie guard tending to zero does not change the
positive-gap argument in ideal arithmetic.

**Holm under arbitrary dependence.** Let m_0 be the number of true nulls in
a family of M hypotheses. If any true null is rejected, consider the first
true null in the ordered step-down procedure. At that point at least m_0
hypotheses remain, so its p is at most alpha/m_0. Consequently the event of
any false rejection is contained in {minimum true-null p <= alpha/m_0}.
A union bound over the m_0 super-uniform true-null p-values bounds its
probability by alpha. No independence across shared-graph or shared-teacher
contrasts is required. This justifies the implemented cumulative-max adjusted
p-values and the preserved 795-hypothesis family, including graph aliases.

## I. Labeled vectors, color counts and partitions differ

On path3 with lists (2,3,2), both x=(0,1,0) and y=(1,2,1) are proper.
Point masses at x and y have TV=1 and positive complete-vector MMD, but
identical distributions of used-color count (two), sorted class sizes (2,1),
all co-color pair indicators, and the induced partition {{0,2},{1}}.
These are precisely the two valid vectors used in the easy positive control.
Thus even the entire unlabeled partition law need not determine the labeled
coloring law. An unrestricted global permutation taking 1 to 2 does not
preserve every endpoint list, so quotienting by arbitrary color permutations
is not automatically a group action on the declared local-list space.
The secondary restricted-growth representation describes a partition map;
it does not change the principal labeled-vector hypothesis.
