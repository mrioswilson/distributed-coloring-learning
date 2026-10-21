# Local lists, conditional laws, safety and progress

All propositions below concern exact protocols under the stated assumptions.
No proposition certifies an arbitrary neural checkpoint. Numerical probes,
empirical distributions, and floating-point execution are separate evidence.

## 1. Model and existence

Let G be a finite simple undirected graph, d_v its degree, L_v={0,...,d_v},
and S a proper partial coloring in which each fixed color belongs to its own
node's list. Let t_v be the number of pending neighbors of pending v and
A_v=L_v minus its fixed neighbors' colors, k_v=|A_v|.

**Proposition 1.** k_v ≥ t_v+1.

**Proof.** There are d_v−t_v fixed neighbors, each of which excludes at most
one color in L_v; repeated colors and colors outside L_v exclude fewer.
Thus k_v≥d_v+1−(d_v−t_v)=t_v+1. This does not require neighbors to share L_v. ∎

The same counting argument applies to any lists with |L_v|≥d_v+1, not just
prefix lists. The neural representation here supports prefix lists only;
arbitrary list masks would be a separate implementation experiment.

## 2. Exact simultaneous protocol and safety

At each round pending v independently samples participation B_v~Bernoulli(p_v)
and candidate C_v with law q_v on A_v. Draws for distinct pending nodes are
mutually independent conditional on S. Candidate and participation draws at a
node are independent. A node fixes C_v exactly when B_v=1, no fixed neighbor has
C_v, and no participating pending neighbor offers C_v. Fixed colors never change.

**Proposition 2 (safety).** Proper partial list-colorings are invariant.

**Proof.** Old–old edges remain proper. A new–old edge is proper by the explicit
fixed-neighbor test. If adjacent pending vertices offered the same color and
both participated, both reject, so no new–new edge is monochromatic. Candidates
are in their own lists. These exhaust edge types, proving the invariant. ∎

Independent randomness is unnecessary for this safety argument; exact rejection
and list membership are necessary. Learned availability and learned COMMIT do
not automatically satisfy them. A correct final coloring can coexist with an
earlier exact-rule-violating authorization if the other endpoint did not commit;
the study records both events separately.

## 3. Conditional progress and termination

Define q_u(c)=0 outside A_u. Under the conditional independence above,

    P_v(S)=p_v Σ_{c∈A_v} q_v(c) Π_{u∈N(v), pending}(1−p_u q_u(c)).        (1)

**Proof.** Condition on v's participating and proposing c. Every pending neighbor
avoids the joint event (participating, proposing c) with probability 1−p_u q_u(c).
Conditional private independence makes their probabilities multiply; average
over c and multiply by p_v. ∎

**Proposition 3 (local uniform teacher).** If p_v=1/2 for all pending nodes,
q_v is uniform on A_v and choices obey the stated independence, P_v(S)≥1/4.

**Proof.** For x_i∈[0,1], Π_i(1−x_i)≥1−Σ_i x_i (induction). Consequently,

    P_v ≥ (1/(2k_v)) Σ_{c∈A_v}[1−(1/2)Σ_u q_u(c)]
        = 1/2 − (1/(4k_v))Σ_u Σ_{c∈A_v}q_u(c)
        ≥ 1/2 − t_v/(4k_v) ≥ 1/4.

The inner sum is at most one, including when A_u and A_v differ. The last step
uses Proposition 1. In fact t_v/(t_v+1)<1 for finite t_v. Uniformity of neighbors
is not needed for this one-node bound; uniformity at every pending node makes
the bound apply to every node. No maximum degree is known or used by a node. ∎

**Corollary 4.** Starting from a proper partial coloring, after t rounds,
Pr(T_final>t)≤min(1,n(3/4)^t). Termination is almost sure, and
E T_final≤r+4n(3/4)^r for any nonnegative integer r, hence O(log n).

**Proof.** Conditional on any history for which v remains pending, its chance
of remaining pending in the next round is ≤3/4. The tower property yields
Pr(v pending at t)≤(3/4)^t. Union bound over vertices gives the tail; it does
not presume independence of different nodes' survival events. Sum the tail,
using 1 for t<r and n(3/4)^t for t≥r, to obtain the expectation bound. ∎

Here rounds are complete two-communication rounds, measured after permanent
updates. A 256-round cutoff is a finite operational choice, not a failure of
almost-sure termination, and not a wall-clock timeout.

## 4. How different may candidate laws be?

**Proposition 5 (TV perturbation at a vertex).** Keep the neighbors' independent
action laws fixed. With exact COMMIT and q_v supported on A_v, write
ε_v=TV(q_v,Unif(A_v)). If all neighbors have p_u≤p_max, then

    P_v ≥ p_v max(0,1−p_max t_v/k_v−ε_v).

**Proof.** g(c)=Π_u(1−p_u q_u(c)) takes values in [0,1]. For any such g,
|E_q g−E_uniform g|≤TV(q,uniform), by the positive/negative parts of the signed
measure (or the layer-cake representation). The uniform expectation is at least
1−p_max t_v/k_v by the preceding union bound. Multiply by p_v and use P_v≥0. ∎

Thus p=1/2, ε<1/2 gives P_v≥1/4−ε/2 without requiring the neighbors to be uniform.
This is a sufficient condition, not a necessary one. It makes explicit that
small detectable distributional changes need not destroy progress.

**Proposition 6 (diffuse component, potentially large TV).** Suppose a subset
D_v⊂A_v has q_v-mass η_v>0, and q_v(c)≤β/k_v on D_v. With independent decisions,
p_u≤p_max<1, exact COMMIT and k_v≥t_v+1,

    P_v ≥ p_v η_v (1−p_max)^{β t_v/(η_v k_v)}.

**Proof.** Concavity of log(1−p_max x) on [0,1] puts it above its endpoint chord:
log(1−p_u x)≥x log(1−p_max). Thus g(c)≥(1−p_max)^{Σ_u q_u(c)}.
Restrict to D_v and apply Jensen to the convex exponential, weighted by
q_v(c)/η_v. Its mean exponent is
Σ_u Σ_{c∈D_v}q_v(c)q_u(c)/η_v≤β t_v/(η_v k_v).
Since the base lies in (0,1), this upper bound on the exponent is a lower
bound on the result. Multiplying by p_vη_v proves the claim. ∎

For q_v=(1−ε)Unif(A_v)+ε δ_{a_v}, choose D_v=A_v\{a_v} (k_v>1):
η_v=(1−ε)(k_v−1)/k_v and β=1−ε. TV=ε(1−1/k_v) can approach ε, while for fixed
ε<1 and k_v≥2 the bound remains a positive degree-independent constant.
Since t_v≤k_v−1, the mixture's exponent is at most one, hence at p_v=p_max=1/2
and k_v≥2 this also gives the explicit uniform bound P_v≥(1−ε)/8.
This condition does not certify checkpoints unless a uniform diffuse mass bound
is itself established beyond the sampled states.

**Counterexample (TV/diversity alone do not order progress).** In the all-free
K-clique with K colors, deterministic candidates have TV=1−1/K and entropy zero
at each vertex. If all vertices choose the same color, P_v=2^(−K) at p=1/2.
If the deterministic candidates form a proper coloring, P_v=1/2. These laws
have identical individual distances/diversity but radically different joint
overlap. The second policy is a state-specific mathematical example, not an
anonymous identifier-free neural implementation claim. Cross-policy correlation
between measured entropy and rounds alone cannot establish causality.

## 5. Fidelity is not equality in law

For fixed S, let f_L(r)=a_{floor(kr)} on ordered available colors a_0,...,a_{k−1}.
Define f_M(r)=a_{(floor(kr)+1) mod k}, k≥2. Both have the uniform law because
each color has a preimage interval of length 1/k, but f_L(r)≠f_M(r) for all r∈[0,1).
Shared-r fidelity is zero and TV is zero. Therefore no test based only on any
number of independent candidate observations can distinguish them at that S.
For k=1 all valid deterministic candidate maps coincide.

More generally a measure-preserving reparameterization of r can change the
coupling to the teacher without changing candidate laws. Conversely, equality
at finitely many quadrature points does not imply equality almost everywhere.

## 6. Transition kernels and trajectories

**Proposition 7.** Suppose algorithms A and B have the same conditional joint law
μ_S of *all relevant actions* at every possible protocol state S, the same
update map U(S,a), and the same distribution of the initial state. Then they
have identical transition kernels and finite trajectory laws.

**Proof.** For each measurable next-state set D,
K_A(S,D)=∫1[U(S,a)∈D]μ_S(da)=K_B(S,D). In this finite-state coloring setting all
sets are measurable. For any cylinder S_0∈D_0,...,S_t∈D_t its probability is
the initial measure integrated against the same sequence of kernels, equal by
induction. Cylinder equality establishes the finite trajectory claim, and the
usual countable product construction also gives equality of infinite path laws. ∎

Sufficiency: identical per-node joint (B,C) laws and private independence across
nodes plus exact same COMMIT/update produce identical joint action laws. If
COMMIT is learned, its conditional behavior is also part of what must agree.
Marginal candidate equality alone is insufficient. For adjacent nodes with two
colors and participation one, shared identical uniform candidates always collide;
independent uniform candidates collide with probability 1/2. Each node has the
same marginal law. Within-node participation coupling also matters: choose a
uniform binary candidate, let B=1 iff C=0 versus B independent Bernoulli(1/2).
Participation and candidate marginals agree, while the participating-candidate
law differs. Hence neither the joint action law nor transition kernels follow
from those marginals alone.

## 7. Conditional statistical distinguishability

Let P and Q be known categorical candidate laws at the *same fixed state*.
The observer gets m IID draws from P^m or Q^m with equal priors, hidden randomness.

**Proposition 8.** The optimal balanced accuracy is
(1+TV(P^m,Q^m))/2 and the minimum error is (1−TV(P^m,Q^m))/2.

**Proof.** For a deterministic classifier declaring Q on event D, accuracy is
(P^m(D^c)+Q^m(D))/2=1/2+(Q^m(D)−P^m(D))/2. Maximize over D: the maximizing
event consists of outcomes where Q^m has larger probability. This is a
likelihood-ratio test (with arbitrary tie decision); the supremum is TV.
Randomization cannot improve the supremum. ∎

Writing δ=TV(P,Q), ρ=Σ_c sqrt(P(c)Q(c)), valid product-law bounds are

    max(δ,1−ρ^m) ≤ TV(P^m,Q^m)
        ≤ min(1−(1−δ)^m, sqrt(1−ρ^(2m))).

**Proof.** Projection to one coordinate cannot increase TV, giving the δ lower
bound. A maximal single-draw coupling agrees with probability 1−δ; independent
couplings over m coordinates disagree with probability 1−(1−δ)^m, giving an upper
bound. Affinity factorizes to ρ^m. For arbitrary finite laws p,q,
1−Σsqrt(pq)≤TV because min(p,q)≤sqrt(pq). Also Cauchy–Schwarz applied to
|p−q|=|sqrt(p)−sqrt(q)|(sqrt(p)+sqrt(q)) gives TV≤sqrt(1−ρ²).
Apply to product laws. ∎

If P=Q, optimal accuracy is 1/2 for every m. If P≠Q, choose a category whose
probability differs; its empirical frequency consistently separates the laws,
so optimal accuracy tends to one. This does not give a state-independent finite
sample requirement: differences can be arbitrarily small.

When Q is estimated, a plug-in likelihood ratio is not the exact Bayes test.
Estimation, threshold calibration and test observations must be disjoint.
An observed category outside the known teacher support has infinite likelihood
ratio under that exact null. An unobserved model category does not have certified
probability zero; smoothing must be reported. Product bounds using estimated
laws alone describe surrogates. For any Qhat, coupling gives
|TV(P^m,Q^m)−TV(P^m,Qhat^m)|≤min(1,m TV(Q,Qhat)); this often becomes vacuous at
large m and must not be concealed. A one-draw Bayes interval is obtained directly
from the interval for TV(P,Q).

## 8. Transfer of a certified MAX COMMIT channel to local lists

**Proposition 9 (conditional certificate transfer).** Suppose a frozen model's
COMMIT pathway has the exact module structure audited here: its selected-slot
state sees [u,own,B,offer], neighbor messages are shared MLPs on those four bits,
neighbor reduction is componentwise MAX of nonnegative messages (empty zero),
and neither K, position, nor first-stage availability enters COMMIT. Suppose
its logit is strictly negative for every finite set of legal neighbor types
containing fixed-same or pending-participating-same. Then, in real arithmetic,
permanent decisions under the receiver-query adapter preserve properness for
arbitrary prefix-list sizes, provided own proposals are in their lists.

**Proof.** At a pending participating receiver's own candidate c, its own bits
are [1,0,1,1]. Every neighbor's bits are one of
[0,0,0,0], [0,1,0,0], [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1].
A neighbor whose list excludes c can produce only types with own=offer=0;
this introduces no new type. Each finite multiset reduces to the same MAX as
its set of present types, one of 64 sets including the empty set. A fixed-same
or participating-same neighbor therefore implies a negative logit and rejection
by hypothesis, regardless of degree, other nodes' K, or duplicate messages.
The own-list check and the edge argument of Proposition 2 prove invariance. ∎

This proposition transfers a **verified sign property**, not correctness from
the mere name MAX. Previous interval certificates may be reused only after
matching checkpoint hashes and module definitions. Float32 conclusions require
the additional documented affine-error bounds, no overflow/TF32, correct gather
and strict margins. They are conditional on the implementation's numerical
error model, not an unconditional GPU guarantee. SUM/MEAN do not reduce to 64
sets and this proposition does not apply to them. Nothing here proves neural
progress or termination; RAW can remain safe and fail to advance.

## 9. Honest uncertainty and precision

For N IID categorical draws from Q on K colors, the subset characterization of
TV and Hoeffding plus a union bound over at most 2^K subsets give

    Pr(TV(Qhat,Q)≥ε) ≤ 2^K exp(−2N ε²).

Indeed TV(Qhat,Q)=max_D(Qhat(D)−Q(D)), and each subset indicator is Bernoulli.
Thus r=sqrt((K log2+log(1/α))/(2N)) is a valid radius, clipped at 1. The reverse
triangle inequality yields |TV(Qhat,P)−TV(Q,P)|≤r. This conservative pointwise
bound supports a prespecified equivalence tolerance, without interpreting
nonrejection as equality. It applies to IID Monte Carlo, not deterministic grids.
Finite quadrature can miss arbitrarily narrow decision intervals.

The continuous mathematical uniform and its clipped float32 conversion are
distinct input laws. Rank floor(k r) can differ at interval boundaries; it is
incorrect to attribute such differences to learning. Our numerical candidate
law is induced by a float64 draw followed by the documented conversion. The
mathematical reference teacher is exactly uniform over A. Boundary probes and
separate teacher-precision disagreement record the distinction. Statements
about mathematical real-valued neural evaluation are not float32 certificates.

The clipped conversion has |r32−r|≤2^(−24) on [0,1). The sorted uniform teacher
can change its rank only within that distance of one of its k−1 internal
boundaries. For continuous uniform r, a union bound therefore gives teacher
coupling disagreement at most min(1,2(k−1)2^(−24)), and the same upper bound on
the TV between the two numerical teacher laws. At k=501 this is less than
0.000060. This bound concerns conversion of the teacher input, not perturbations
of neural decision boundaries.

For the uncertain product-law calculation, TV(Q,Qhat)≤r implies H(Q,Qhat)≤sqrt(r).
The Hellinger triangle inequality bounds H(P,Q) by H(P,Qhat)±sqrt(r), clipped
to [0,1]. Converting back to affinity bounds and inserting their proper endpoints
in Proposition 8 gives valid (often broad) product-TV/Bayes-accuracy intervals
on that same pointwise confidence event.
