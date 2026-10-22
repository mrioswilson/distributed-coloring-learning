# Robust complete-coloring tests: definitions, guarantees and limits

These are standard finite-space probability and permutation arguments,
adapted explicitly to this experiment. They are not claims of new general
theorems or a universally powerful finite-sample test.

## 1. The probability space

Fix a labeled finite graph G. Vertex v has C_v={0,...,d(v)}. The sample
space C(G) comprises vectors in the product of these lists satisfying every
edge. A protocol with frozen weights maps an infinite private-randomness
tape into a trajectory. Its conditional law given valid completion by R=128
is P^G_R. Invalid-complete, cutoff and technical-error frequencies are
different operational outcomes and must be retained. Equal conditional
solution laws do not imply equal success or stopping-time laws.

Every row is a whole independent protocol execution. Vertices within a row
are dependent. Statistical permutations act on row source labels, never on
vertex columns or color names. Ancillary run identifiers define a fixed
selection/evaluation split before values are inspected. Reordering rows
together with their identifiers leaves the implemented pipeline unchanged.
Sorting colorings by their values to define the split would generally be
invalid and is not done. Dynamic protocol state and adapter caches reset
between independent disjoint-union batches, with a fresh RNG stream per run.

## 2. Clique equality: theorem versus implementation probes

On K_n all local lists are {0,...,n-1}. Assume an uncolored initial state,
a deterministic round map equivariant to vertex permutations, iid private
randomness with a permutation-invariant law, and positive probability of
proper completion (possibly by a fixed cutoff). Induction over rounds gives
vertex-invariance of the trajectory law: apply a permutation to the state
and tape and use equivariance at each step. The proper-completion event is
invariant, so conditioning preserves invariance. Vertex permutations act
transitively on the n! proper complete vectors, since every such vector is
a bijection between vertices and colors. All conditional masses are equal
and sum to one. This proves uniformity without color-permutation symmetry
of the proposal head. It also proves equality between any two such protocols.

The fifteen selected architectures have shared vertex rules and invariant
neighborhood aggregation in ideal real arithmetic. Prefix color features
do not assign vertex identities. Floating-point reductions receive numerical
equivariance checks, not a theorem for every hardware input. Simulated
uniform-clique controls exercise a known null law; actual clique comparisons
test these implementations. Nonrejection never proves the theorem's premises.

## 3. Why the old characteristic kernel can fail

For fixed n and gamma>0 let a=exp(-gamma/n). The single-coordinate kernel
has Gram matrix (1-a)I+a11^T. Its eigenvalues are 1-a and 1+(K-1)a, all
positive. The full product kernel exp(-gamma d_H/n) is the Kronecker product
of these matrices. Restricting to C(G) is a principal submatrix, hence
positive definite. For distinct probability vectors p,q,
MMD^2=(p-q)^T K(p-q)>0. Thus the old kernel identifies every law on this fixed
finite space and supports consistency, without independent vertex colors.

Consider a star with D=200 leaves. Both laws have iid uniform binary leaves.
Its center is independent, uniform on 199 colors under A and a 100-color
subset under B. Every coloring is proper with local lists. The common leaf
law factors out. With a=exp(-1/201), b=(1+a)/2,

    TV(A,B)=99/199,
    MMD^2=b^200 (1-a) (1/100-1/199)=0.0000150213168648.

To see the center factor, its kernel is a+(1-a)1[c=c']; the constant part
cancels from MMD, leaving (1-a) sum_c(p_c-q_c)^2. The leaf kernel expectation
is b^200. Under A/A the row expectation mu=b^200[a+(1-a)/199] is constant,
and zeta2=E[k^2]-mu^2, where

    E[k^2]=((1+a^2)/2)^200 [a^2+(1-a^2)/199].
    Var_0(U)=[4/(m(m-1))+4/m^2] zeta2.

Proof: center the kernel by its constant row mean. Each within-sample
degenerate U statistic has variance 2 zeta2/(m(m-1)); the cross average has
variance zeta2/m^2. All cross-covariances vanish by conditional centering.
Adding the two within terms and minus twice the cross term gives the formula.
At m=250, exact null SD is about .00024129, sixteen times the signal. The
mean signal/SD is only .06225, and even m=2000 gives .4985. The archived
100-trial replay reproduces all original p-values exactly. This failure is
already present at nominal .05, before external Holm correction.

For growing analogous stars the center L2 squared difference is O(D^-1),
1-a=O(D^-1), and b^D stays bounded away from zero. Thus the signal is
O(D^-2). Expanding the leaf second moment gives zeta2=O(D^-1), and null SD
O(1/(m sqrt(D))). A signal/SD argument alone therefore suggests m on the
order of D^(3/2), with potentially large constants. The recorded planning
approximation m≈9,977 for signal/SD≈2.487 is not a power theorem or p-value
approximation. Under-alternative variance, maximum aggregation and external
multiplicity can require more samples.

In contrast, the named event center<=101 has probabilities 100/199 and 1.
It retains the full .4975 probability difference and ignores irrelevant
leaves. Learning a similar event on independent selection data can have
useful finite-sample power while retaining conditional calibration.

The remaining remote-triple limitation also has an explicit calculation.
Keep the center fixed at2, P uniform on D binary leaves, and Q conditioned
on even parity of a specified r-leaf subset. Then TV=.5, and the Boolean
parity vector is an eigenvector of the product Gram matrix, giving

    MMD^2=((1-a)/2)^r ((1+a)/2)^(D-r).

For D=200 and r=3 this is 9.36528475045e-9. The null SD at m=250 is about
.000242473, yielding signal/SD3.86e-5. A known parity event has probabilities
.5 and1 and is easy to test; all marginal and pair laws coincide. Thus an
all-pair search cannot represent that signal unless a suitable triple is
already included. This is a representation/power failure, not statistical
impossibility of distinguishing the two specified laws. A third-order search
or a prespecified relevant triple would be needed to expose that witness;
the final instrument does not claim to perform that search. The analogous
diffuse .55-versus-.5 binary-leaf alternative has squared kernel effect
about .00303445, despite a global TV of similar order. Kernel geometry and
noise structure, rather than TV alone, determine these finite power gaps.

## 4. Honest omnibus construction

For each generator, split total m into s=floor(m/2) selection rows and
t=m-s evaluation rows. On selection rows only, fit signed categorical
contrasts for individual vertices, predefined graph-based/random pairs and
triples; select four local, four interaction and four structural candidates.
Version two adds the eight largest covariance-change pairs in each of raw
color and median-binary selection representations before that four-candidate
interaction selection. The search covers all vertex pairs but uses only
selection labels; the conditional calibration proof is unchanged. It does
not search all triples, and remote third-order alternatives can remain hidden.
Screening uses only two covariance representations; it does not guarantee
selection of every pair with a changed categorical joint law. An alternative
preserving both screened covariances may also remain poorly represented.
Structural candidates include used-color counts, class concentration/entropy,
degree-wise means/parity, and predefined nonadjacent co-color events. Add
a dense marginal score and four empirical global-kernel witness functions
at gamma=.25,1,4,16. The exact graph-seeded region construction and finite
threshold grids are implementation constants, frozen before confirmation.
This family cannot represent every localized high-order alternative.

Explicitly, let N_c(x)=#{v:c_v=c} and p_c=N_c/n. The reported observables
are used_colors=#{c:N_c>0}, largest_class_fraction=max_c p_c,
class_entropy=-sum_c p_c ln(p_c) (natural logarithms, with 0 ln0=0),
and mean_color=(1/n)sum_v c_v. Degree-group summaries apply to the named
fixed vertex group, and a co-color feature is 1[c_u=c_v]. A threshold event
is the preimage {x:h(x)<=t}. For integer used-color counts, displaying
floor(t) instead of a fractional quantile threshold defines exactly the
same event; the original numerical descriptor remains in the raw result.

Evaluate these functions h_j on held-out pooled rows. Center and scale each
by its pooled SD, without using evaluation source labels; constant features
are zero. Define Z_j=|mean_X h_j-mean_Y h_j|/sqrt(1/t_X+1/t_Y).
Also compute the original unbiased full-vector MMD U on evaluation rows,
including negative values, with gamma=1 and a strictly positive pooled
Frobenius scale s_K. In code, with zero-diagonal K and H=I-11^T/N,

    s_K=max(sqrt(2)(1/t_X+1/t_Y)||HKH||_F/N, 1e-15).
    T=max(max_j Z_j, U/s_K).

This scale is a label-invariant normalization, not an asserted exact null
standard deviation. Its precise constant does not affect the permutation
level theorem. The complete characteristic-kernel term is always retained
with coefficient one; structural projections never replace labeled vectors.
Selection observations count toward total m in efficiency comparisons.

## 5. Conditional permutation validity and selection

Under H0 evaluation rows are iid from one law, independently of selection.
Condition on all selection data and the multiset of pooled evaluation rows.
The fixed-size allocation of their source labels is uniform. All selected
functions and pooled scales are now fixed. Therefore computing T for every
uniform label allocation gives its exact conditional null distribution,
including its maximization over components. The ordinary inclusive-tail
permutation rank is superuniform: among a finite list of allocations, at
most an alpha fraction have an upper-tail rank <=alpha. Integrating over
the conditioning variables proves unconditional level <=alpha.

For B independent uniform random allocations, include the observed
allocation through the plus-one rule p=(1+sum_b 1[T_b>=T_obs])/(B+1).
Conditional exchangeability of the B+1 statistics yields the same rank bound;
ties can only increase inclusive-tail p-values. The implementation uses an
additional nonnegative numerical tolerance, counting more exceedances,
which is conservative relative to the exact comparison. The maximum is
recomputed for every allocation. Choosing a region using evaluation labels
and then permuting only that selected region without repeating selection
would not satisfy this proof. Our supervised choices are all independent
of evaluation labels. No neural model is trained for the statistical test.

An optional computational futility rule pre-generates conceptually B
allocations, stops after a specified number of exceedances and returns p=1.
This p is pointwise at least the complete plus-one p, under the same random
tape, so remains superuniform. It can sacrifice power and cannot increase it.
The result records its p-kind, used permutations and exceedances. It is not
presented as a precise estimate of the full permutation tail. Strong results
receive all B allocations; no parametric tail extrapolation is used.

For final family decisions, each censored full-B p has a deterministic
lower bound (L+1)/(B+1) when at least L exceedances were observed. Apply Holm
to all these optimistic bounds. Recompute the conceptual full-B p, with
the same RNG tape, for every censored cell rejected by that optimistic Holm.
Every unrefined cell is nonrejected even with optimistic bounds, hence is
nonrejected for the true full-B values by Holm monotonicity. All true full-B
rejections consequently have known exact full-B values. Increasing the
remaining nonrejected values to one cannot change the rejected sorted prefix.
Thus the final rejection set equals complete fixed-B Holm computation.
The reported p-values remain individually valid upper p-values; only
nonrejected censored cells have an unevaluated exact full-B tail.

## 6. Fixed-space consistency and its limits

The retained kernel is characteristic, so on a fixed alternative its
population MMD squared is positive. In exact real arithmetic the separate
gamma-1 empirical kernel witness converges on every point of the finite
space to h(z)=E_P k(z,X)-E_Q k(z,Y). Its expectation difference is exactly
MMD squared, and its variance under the pooled mixture is positive: a
constant on that support could not have different P and Q expectations.
The held-out normalized mean difference therefore grows at rate sqrt(t),
by finite-support laws of large numbers, even if a fixed positive floor
eventually dominates the separately retained U-component's normalization.
No supervised feature-selection success is needed for this argument.

Under uniform allocations, a centered pooled-standardized feature has
conditional squared-Z expectation N/(N-1), by the variance formula for
sampling without replacement. If its pooled SD is too small and the code
leaves it unscaled, this bound is smaller. For a nondegenerate finite pooled
law, ||HKH||_F is O(N) with positive limiting normalized norm, and the
conditional U variance is of order the squared unfloored s_K. Taking a
larger scale via a numerical floor only reduces that standardized variance.
Constant kernels have U identically zero. Thus every permutation component
is tight. A union bound over the fixed finite number of components makes
their permutation maximum tight, despite selection functions changing with
the selection data. The observed T diverges and eventually exceeds its
permutation critical value. With fixed B and 1/(B+1)<=alpha, all finitely
many sampled permutation statistics fall below T with probability tending
to one. This proves fixed-space consistency in real arithmetic. A fixed
external Holm family also preserves it if its first threshold resolves
that minimum p. Finite machine precision is not an asymptotic theorem, and
no dimension-uniform sample complexity follows.

## 7. Other methods and classifier calibration

The multiscale benchmarks recompute the max or log-mean-exponential of
pooled-normalized global MMD statistics in every permutation. Their own
conditional-rank proof gives level control. They are inspired adaptations,
not claims to reproduce MMDAgg or MMD-FUSE exactly. MMDAgg's Sobolev/density
minimax conclusions do not transfer to our discrete graph-specific supports.
MMD-FUSE motivates permutation-invariant choices and aggregation; its specific
normalizations and power theorems are not automatically those of our tests.

Classifiers use three distinct blocks: training, validation and independent
evaluation. Training learns regularized logistic or nonlinear Gaussian
naive-Bayes scores on local/structural features; validation selects the
candidate. Conditional on these blocks and fixed test predictions, the
number of positive predictions assigned to source X under H0 is
Hypergeometric(N,K,t_X). Its exact upper tail calibrates test accuracy
without refitting, because test rows were not used to train or select.
Class-specific Clopper-Pearson intervals with Bonferroni combination give
a conservative balanced-accuracy interval. Testing on training rows would
invalidate this justification.

On small enumerated supports, a likelihood-ratio statistic is calibrated
by the exact conditional multivariate-hypergeometric count law (or equivalent
uniform row allocations). No asymptotic chi-square reference is used.

Extreme exact hypergeometric probabilities can underflow to zero in float
arithmetic. Valid reported probabilities replace those numerical zeros by
a conservative finite-population Hoeffding upper bound. For m observations
per test class and balanced accuracy A, correct_count=m-K+2TP and
E_0(TP)=K/2, so Pr_0(TP-K/2>=m(A-.5))<=exp(-2m(A-.5)^2), uniformly over K.
The audit exports these positive upper probabilities and verifies that every
recorded .05 decision is unchanged. Raw underflows remain identified in the
archive. The main m=500 classifiers have representable tails. This numerical
postprocessing changes no feature, statistic, selection, or primary test.

For completeness, the without-replacement exponential bound follows from
elementary symmetric means. The mgf of drawing m entries without replacement
from fixed nonnegative values x_i=exp(lambda b_i), b_i in {0,1}, is the
normalized degree-m elementary symmetric polynomial. At fixed sum, replacing
any two x,y by their average leaves its terms independent of x,y and linear
in x+y unchanged, and increases the remaining xy times a nonnegative
coefficient. Repeated averaging and continuity therefore bound this mgf by
the mth power of the arithmetic mean, the binomial mgf. For the centered
Bernoulli log-mgf f(lambda), f(0)=f'(0)=0 and
f''(lambda)=p_lambda(1-p_lambda)<=1/4. Integrating gives
f(lambda)<=lambda^2/8. The exponential Markov inequality, minimized at
lambda=4d/m, gives Pr(TP-E TP>=d)<=exp(-2d^2/m). Substituting
d=m(A-.5) proves the reported upper bound. These are known concentration
arguments, not a new theorem.

## 8. TV inference, uncertainty and multiplicity

For known reference p on S valid vectors, use empirical q from M independent
valid neural completions. For every subset A, Hoeffding bounds
Pr(|q(A)-p_true(A)|>r)<=2 exp(-2Mr^2). Because TV is the largest subset
probability difference, a union bound (or one-sided subset/complement
pairing) gives Pr(TV(q,p_true)>r)<=2^S exp(-2Mr^2).
Thus r=sqrt((S log2+log(1/alpha))/(2M)) yields a conservative interval
[max(0,TV(q,p)-r),min(1,TV(q,p)+r)]. The triangle inequality supplies both
endpoints. Union-bound alpha over the prespecified small-graph/checkpoint
family for simultaneous coverage. Add the teacher cutoff/eventual-law
bound delta<=n(1-2^(-Delta-1))^128 where the exact reference is eventual.
To derive delta, any pending vertex fixes with probability at least
(1/2)2^-d(v), by participating while all pending neighbors do not. Iterated
conditional failure and a union bound give the stated tail. Coupling an
eventual output with completion by R gives the same TV upper bound.

The exact rational reference uses ideal continuous proposals; execution uses
float64 uniforms and rounded floor(r*k). Under the standard uniform 53-bit
grid and IEEE nearest-rounding model, grid counting contributes at most
k/(2*2^53) proposal TV. Rank multiplication can move at most two grid
positions per integer boundary, giving a conservative combined bound
4k/2^53. Union over R rounds and vertices gives
epsilon_fp=4 R sum_v K_v/2^53 for the finite-path laws. Conditioning on
success with probability at least 1-delta contributes at most
2 epsilon_fp/(1-delta), by the total-variation conditioning inequality.
The small-graph intervals include this additional conservative bound, about
10^-12 or smaller here. It does not certify a neural real-arithmetic law;
the neural estimand is its actual frozen numerical implementation.

An event chosen on primary data and then estimated on a wholly independent
explanation bank has conditional binomial sampling laws. Simultaneous
Clopper-Pearson bounds on its two probabilities, unioned over all planned
graph/checkpoint cells, provide valid difference intervals despite primary
selection. If explanation observations were reused to choose the event,
those intervals would be naive and are not allowed.

Every such event additionally gives TV(P,Q)>=|P(A)-Q(A)|. Transforming its
simultaneous probability-difference interval [l,u] into max(0,l,-u) therefore
gives a simultaneous lower bound on complete-vector TV. This remains useful
on large spaces without enumerating them; it supplies no TV upper bound.

For an arbitrary dependence structure among valid primary p-values, Holm
controls FWER: if any true null is rejected, the first rejected true null
has raw p<=alpha/M0, since at least M0 true nulls remain. The union bound
over those M0 nulls gives probability <=alpha. Shared teacher banks,
nested samples and repeated graph aliases do not require independent tests.
They do restrict population interpretations and must be labeled explicitly.

For large spaces, TV<=sqrt(S)/(2 sqrt(lambda_min(K))) MMD follows from
||p-q||_1<=sqrt(S)||p-q||_2 and the minimum-eigenvalue bound on the quadratic
form. The full-product lower eigenvalue is at least the product of coordinate
(1-a) terms; the resulting TV constant can be enormous. Without a useful
graph-specific constant, a small MMD supplies no useful TV equivalence claim.

## 9. Fundamental finite-sample limits

If Q=(1-epsilon)P+epsilon R, couple each Q draw to a P draw until a rare
replacement occurs. With m Q draws, the probability of any replacement is
1-(1-epsilon)^m. On its complement the joint samples have the null law,
so every level-alpha test has power at most
1-(1-alpha)(1-epsilon)^m. Our rare-center control realizes this with disjoint
center colors and identical leaves. No honest test can guarantee 80% power
at m=100 when epsilon=.001; the upper bound is about .1404.

A stronger limitation involves huge unknown supports even when TV=1.
On S even valid vectors choose an unknown uniformly random half H. Let
P_H be uniform on H and Q_H uniform on its complement; their TV is one.
Compare this mixture of alternatives to the null P=Q=Uniform(S). Conditional
on all 2m observed vectors being distinct, both mixture and null assign
the same uniform distribution to every ordered distinct tuple, by support
permutation symmetry. Under the null collision probability is at most
choose(2m,2)/S; under the alternative it is at most 2m(m-1)/S, since only
within-source collisions are possible, on supports of size S/2. Coupling
the two conditional-distinct laws bounds the TV between the mixture and
null experiments by the sum of these collision probabilities, O(m^2/S).
Consequently the average alternative power of any level-alpha test is at
most alpha+O(m^2/S), so some TV=1 alternatives have low power when S is
large. A star with fixed center 2 and D binary leaves supplies S=2^D valid
colorings. Structure-specific witnesses help only when the relevant
structure is represented or can be learned from available data.

## Verified primary references

- Schrab et al. (2023), [MMD Aggregated Two-Sample Test](https://www.jmlr.org/papers/v24/21-1289.html), JMLR 24(194):1–81.
- Biggs, Schrab and Gretton (2023), [MMD-FUSE](https://proceedings.neurips.cc/paper_files/paper/2023/file/edd00cead3425393baf13004de993017-Paper-Conference.pdf), NeurIPS 36.
- Hemerik and Goeman (2018), [Exact testing with random permutations](https://link.springer.com/article/10.1007/s11749-017-0571-1), TEST 27:811–825.
- Phipson and Smyth (2010), [Permutation p-values should never be zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf), Statistical Applications in Genetics and Molecular Biology 9(1), article 39.
