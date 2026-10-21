# Safety, progress and representation: explicit statements

These are mathematical statements about specified policies. A learned
checkpoint satisfies a hypothesis only where it has been separately verified.
The statements do not follow from a collection of successful rollouts.

## Distributed model and conditional probability

Let G be a finite simple undirected graph with maximum degree at most Delta,
and let a common palette have K >= Delta+1 colors. A state is a proper partial
coloring; permanent colors never change. Write U_v for the uncolored neighbors
of an uncolored v, d_U(v)=|U_v|, A_v for colors not fixed by a neighbor, and
k_v=|A_v|. Since at most deg(v)-d_U(v) distinct colors are fixed nearby,

    k_v >= K - (deg(v)-d_U(v)) >= d_U(v)+1.                    (1)

Condition on the complete current state and any past randomness. Each node
uses fresh independent private random draws. Participation B_v has probability
p_v; candidate C_v has law q_v on the palette. Within each node B_v and C_v
are independent, and the pairs (B_u,C_u) are independent across nodes.
These assumptions hold for the implemented inference interface: first-stage
messages contain colors/uncolored status only; the candidate uses the local
r_c; participation uses local r_p only. Self-attention across a node's colors
does not itself couple the private draws of different nodes. A common random
draw, dependence on neighbors' draws or additional shared latent randomness
would require a different calculation.

For EXACT conflict rejection, v confirms iff it participates, C_v is in A_v,
and no uncolored neighbor participates with the same candidate. Conditioning
on C_v=c and using the stated independence gives

    P(v confirms | state)
      = p_v sum_{c in A_v} q_v(c) product_{u in U_v}(1-p_u q_u(c)).    (2)

There is no assumption that confirmation events at different nodes are
independent. They generally are dependent. Equation (2) need not hold for
an inaccurate learned COMMIT, even if all candidate draws are independent.
If participation and candidate are coupled, replace p_v q_v(c) by
P(B_v=1,C_v=c), and replace each p_u q_u(c) in the product similarly.

## Proposition 1: a sufficient anti-concentration condition for progress

Suppose the state is proper, K>=Delta+1, exact conflict rejection is used,
and the conditional independence assumptions above hold. Let
0<p_min<=p_v<=p_max<1 for every pending node. If, for a particular v,

    supp(q_v) subset A_v,    max_c q_v(c) <= beta/k_v,

then

    P(v confirms | state)
      >= p_v (1-p_max)^[beta d_U(v)/k_v]
      >= p_min (1-p_max)^beta.                               (3)

Only v's concentration bound is needed for this individual-node conclusion;
neighbors' candidate laws may be nonuniform, including forbidden proposals.
They still obey the independence and participation assumptions.

Proof. For 0<=x<=p_max, concavity of log gives its chord bound

    log(1-x) >= (x/p_max) log(1-p_max).

Let Lambda_c=sum_{u in U_v}p_u q_u(c). The product in (2) is at least
exp(a Lambda_c), where a=log(1-p_max)/p_max<0. Jensen's inequality for the
convex exponential and q_v's support inside A_v imply

    sum_c q_v(c) exp(a Lambda_c) >= exp(a sum_c q_v(c)Lambda_c).

The expectation satisfies

    sum_c q_v(c)Lambda_c
      = sum_u p_u sum_c q_v(c)q_u(c)
      <= (beta/k_v) sum_u p_u sum_c q_u(c)
      <= beta p_max d_U(v)/k_v.

Because a is negative, applying this upper bound inside exp(a·) gives the
first inequality in (3). Equation (1), and 0<1-p_max<1, give the second.
This proves the proposition. For an isolated node the empty product is one,
and the same proof gives the stronger bound P=p_v. QED.

A simpler union bound also yields

    P(v confirms|state) >= p_v max(0,1-beta p_max d_U(v)/k_v).

The exponential bound stays positive for every fixed finite beta, whereas
the union bound is useful when beta p_max is small. Neither bound is asserted
for our learned policies in all states: finite-grid estimates of max q are
not uniform bounds over all configurations or all future palettes.

## Corollary: conditional geometric tails and termination

If all hypotheses of Proposition 1 hold at EVERY reachable proper state for
EVERY uncolored node with the same constants, set
rho=p_min(1-p_max)^beta. Exact rejection preserves properness (proof below).
For any initially pending node, the conditional chance of staying pending
one more round is at most 1-rho. Iterated conditional expectation therefore
gives P(T_v>t)<=(1-rho)^t, without temporal independence. Union bound gives

    P(T>t) <= min(1,N(1-rho)^t).

Hence termination is almost sure and
E[T] <= ceil(log(N)/(-log(1-rho))) + 1/rho (with the N=1 case interpreted
directly). This is a theorem for policies satisfying uniform hypotheses,
not a termination certificate for E1 or the new trained checkpoints.

## Counterexamples to stronger claims

Safety alone gives no positive progress lower bound. In an all-free clique
on K=d+1 nodes, let every node propose color 0 and participate independently
with p=1/2. Every proposal is available and COMMIT is exact. A node confirms
only if it is the unique participant, so its probability is 2^-K and
P(any confirmation)=K/2^K. The mean time to the FIRST confirmation is
2^K/K. Thus a safe policy can advance exponentially slowly. With p=1 the
state never changes; a policy that always waits is another counterexample.

Uniform marginal candidates alone do not suffice without private independent
draws. On the same clique, let all nodes use one common uniform color, with
independent participation p=1/2. Every marginal q is uniform, but the same
exponential first-confirmation law holds. This refutes applying (2) solely
from marginal histograms. Conversely anti-concentration is sufficient, not
necessary: neighboring nodes proposing distinct fixed colors can progress
well despite deterministic individual candidate laws.

## Proposition 2: finite-type MAX reduction of COMMIT

Assume legal inputs, nonnegative deterministic shared message function phi_2,
componentwise MAX with zero for an empty neighborhood, and the exact UCGNN
candidate-slot gather. For an uncolored participating center proposing legal
color c its own slot input is f=[1,0,1,1]. Every neighbor's slot input belongs
to the following six types [uncolored,owns_c,participates,offers_c]:

    t0=[0,0,0,0]  fixed other color
    t1=[0,1,0,0]  fixed color c
    t2=[1,0,0,0]  waiting, not offering c (or no candidate)
    t3=[1,0,0,1]  waiting, offering c
    t4=[1,0,1,0]  active, not offering c (or no candidate)
    t5=[1,0,1,1]  active, offering c.

The implementation gates participation by uncolored status, and forces
colored nodes' candidates to -1. Thus no other neighbor type is legal.
For a type-set S subset {t0,...,t5}, let R_S=max({0} union {phi_2(t):t in S})
componentwise, and let

    F(S)=commit_head([state2([f,R_S]),1,1,1]).

If sign(F(S)) is positive exactly when neither t1 nor t5 belongs to S,
for all 64 subsets S (strict negative margin for rejection), then the learned
COMMIT authorizes this center exactly according to conflict rejection for
arbitrary degree and palette, in real arithmetic. In particular it is safe.

Proof. Replace each neighbor by its legal type. MAX is commutative,
associative and idempotent, so multiplicities do not change R_S. Zero does
not change a nonempty maximum because messages are nonnegative. The own
candidate has exactly one nonzero offer slot; gathering eliminates every
other palette slot. Consequently the actual logit is F(S), independent of
the number of neighbors or other palette slots. A fixed-color conflict is
equivalent to t1 in S; a simultaneous participating-proposal conflict is
equivalent to t5 in S. The assumed sign table therefore gives precisely the
correct rule. Gating prevents executions at a colored, nonparticipating or
no-candidate center, regardless of its raw logit. QED.

Every subset is realizable as a star with one leaf per present type and at
least two colors. Fixed leaves may share a color because they are not adjacent.
For K=1 some subsets are not realizable, but verifying the same superset of
conditions remains sufficient. For a graph requiring Delta>=the star degree,
one can declare a larger palette; the type semantics and weights are unchanged.

Properness preservation follows by induction: an execution cannot equal a
previously fixed neighbor (t1 excluded), and two adjacent executions cannot
propose equal colors (t5 excluded). This yields safety of a full trajectory
from a proper initial state, with NO inference of termination.

For SUM, representations vary with the vector of type counts; for MEAN they
vary with type proportions. Neither has a 64-case reduction for arbitrary
degree. Duplication tests are diagnostics for these treatments, not proofs.

## Numerical certification scope

Float32-stored weights are exact binary rational numbers interpreted as real
coefficients in Proposition 2. `dcl.paperstudy.certification` propagates
outward-rounded float64 intervals through all 64 type sets, MAX and affine/
ReLU layers. Every multiplication and sequential addition is enclosed with
`nextafter`. A strict output sign interval certifies the real network under
IEEE float64 interval arithmetic, as implemented on the recorded platform.

A second enclosure adds the standard gamma_(n+2) affine error allowance for
float32 dot products with n terms, including bias, plus a conservative
subnormal allowance, assuming no overflow, IEEE rounding and no TF32 or other
lower-precision multiplication. ReLU and MAX add no error. This is a
conditional floating-point implementation certificate under that arithmetic
model, not an unconditional certificate of arbitrary CUDA/BLAS kernels.
Batch shape can change reduction order; the n-term bound permits any standard
ordering/FMA implementation with the stated error bound. Actual CPU/CUDA
comparisons and +/-1e-6 input perturbations are reported separately.

## Position-free palette limitation

Suppose a shared deterministic slot scorer is permutation-equivariant across
colors, all available slots have identical inputs, and the only global
context is an invariant pooled vector. Conditional on a fixed scalar r, every
available slot has the same score. It cannot represent an ordered teacher
that selects different available ranks for different r. Index-based argmax
resolves ties externally and may collapse onto the lowest available color.
Functional color positions break this symmetry in our principal prefix-free
baselines. Their remaining errors cannot be attributed to this impossibility
without checking the actual inputs and learned function.

## Alternative teacher law

For sorted nonempty A, choose a uniformly random start from {0,...,K-1} and
scan cyclically to the first member of A. For c in A let gap(c) be the number
of cyclic start positions strictly after the previous available color and
up to and including c. Then q(c)=gap(c)/K, since exactly those starts lead to
c. The positive gaps partition K start positions. Singleton A has gap K.
This differs from uniform over A when gaps differ. Exact conflict rejection
still preserves properness. However, constant beta in Proposition 1 need
not hold uniformly: a large gap can put most mass on one color. The pilot
tests candidate imitation and progress separately from this safety argument.
