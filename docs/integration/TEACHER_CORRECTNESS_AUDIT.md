# Native teacher correctness, guarantees and scope

The object audited is the exact program in
`src/dcl/multiteacher/programs.py`, originating at Pedro's commit
`6836d85566d94710ef5bdea05e10702252520702`. The conditional arguments below
concern its stated mathematical rules. They are not formal verification of
NumPy, the PRNG, every floating-point execution or a learned checkpoint.

## Assumptions and evidence levels

G is finite, simple, undirected, with labeled vertices 0..n-1,
1≤n≤65,536 and maximum degree≤3. Initial permanent colors are all pending;
any partial-coloring safety argument assumes an already proper partial
coloring. The common palette is{0,1,2,3}. ID-dependent teachers receive
distinct fixed IDs in 0..65,535. Native phase and schedule information is
supplied, not learned. Randomized guarantees use fresh mutually independent
private draws in the ideal mathematical model.

The implementation's float 32 random ranks approximate continuous uniform
ranks. When the number of free colors is 3, a finite binary grid cannot give
exactly equal rank masses. Both continuous and independent finite-grid draws
have the necessary positive-progress property here, but exact numerical
laws are specified separately. A seeded PRNG experiment does not itself
prove an infinite-time probabilistic theorem.

Four levels are kept separate: (i) conditional mathematical proof of the
defined exact rules, (ii) agreement with an independent scalar specification
on an explicit finite domain, (iii) finite random execution evidence,
(iv) learned-policy correctness not established outside separately verified
hypotheses. The first three do not imply the fourth.

## Safety for permanent random-trial colors

Let A_v be colors not fixed by neighbors and d_U(v) the number of pending
neighbors. At most d(v)-d_U(v) distinct colors are forbidden, so
|A_v|≥4-d(v)+d_U(v)≥d_U(v)+1. Thus a pending vertex always has an available
color. LUBY and JOHANSSON choose only from A_v. A confirmation excludes an
equal candidate at every simultaneously participating pending neighbor.
Hence a new fixed color conflicts neither with an old fixed neighbor nor
with another simultaneous fixation. Induction proves palette membership,
properness and permanence from a proper partial state.

LUBY participation is Bernoulli 1/2, independently of the candidate and all
other nodes. Conditional on state, for any independent neighbor proposal laws
q_u and center law q_v,

    P(v commits) = p_v sum[c in A_v] q_v(c) product[u in U_v](1-p_u q_u(c)).

For uniform q_v, Jensen and log(1-x)≥2x log(1/2),0≤x≤1/2, give

    P(v commits) ≥ (1/2) (1/2)^[d_U(v)/|A_v|] ≥ 1/4.

Indeed the q_v-average of sum_u(1/2)q_u(c) is at most d_U(v)/(2|A_v|).
Applying the convex exponential to the preceding logarithmic lower bound
gives the displayed result. This also works for local lists of d(v)+1
colors, using the same availability inequality. It does not justify arbitrary
learned candidate or COMMIT rules.

For JOHANSSON, every pending vertex participates. Each pending neighbor u of
v has at least two available colors, because v is pending. Thus q_u(c)≤1/2
for every c under ideal uniform proposals. Every term's collision-free
product is at least 2^-d_U(v)≥1/8. Summing the center proposal law yields a
self-contained progress bound 1/8 for this degree-three domain. The published
Johansson analysis gives a stronger general guarantee; the conservative
bound here suffices without importing its full proof.

If every pending vertex has conditional progress probability at leastρ>0
at every reachable state, iterated conditional expectation gives
P(T_v>t)≤(1-ρ)^t, without independence of neighboring success events.
A union bound gives P(T>t)≤min(1,n(1-ρ)^t), almost-sure termination and
expected O(log n) protocol rounds. A finite cutoff still permits incomplete
outcomes; it is never a deterministic termination guarantee.

## JP: distinct priorities and a finite cap

Adjacent pending vertices cannot both be strict local ID maxima. Therefore
simultaneous participants form an independent set. Their smallest available
colors avoid fixed neighbors. At least the globally highest pending ID is
eligible whenever a vertex remains, so at least one vertex is fixed each
round. The exact uncapped rule terminates in at most n native rounds, with
proper permanent colors. This bound requires distinct priorities.

The implementation caps JP at 128. On path 129 with IDs increasing along the
path, exactly one endpoint of the pending path is eligible each round.
After 128 rounds one vertex remains. The integration regression test verifies
the recorded incomplete result. This is a valid cutoff outcome, not a
teacher-safety failure or a reason to count an incomplete coloring as success.

For a fixed priority order, synchronous JP yields the same final first-fit
coloring as descending-priority sequential greedy: induction over the
priority dependency DAG fixes a vertex only after all larger adjacent
priorities, with the same occupied colors. Its law under random IDs need
not equal LUBY's law. It does equal ascending-order GREEDY's unrestricted
final-coloring law when the ID order is uniform: descending and ascending
orders are both uniform permutations. This does not say their common-ID
executions agree, or their round/step laws agree. It also does not transfer
automatically to JP conditioned on a restrictive 128-round cutoff. A finite
24-permutation path 4 regression supplements the elementary universal argument.

## Repeated ColorTrials

Only pending vertices with c∈A_v compete for c. A vertex may retain c only
if its preference exceeds strictly every competing neighbor's preference
for c. Two adjacent vertices cannot both strictly exceed each other, so
simultaneous equal-color confirmation is impossible. Old fixed colors are
excluded by availability. Ties cause waiting and are not arbitrarily broken.

Fix any available color at v. The event that v's preference is 3 and every
eligible neighbor's preference is at most 2 has probability at least
(1/4)(3/4)^3=27/256. On that event v wins some color, even if it ultimately
keeps a smaller winning color. Independent repeated preferences therefore
give the same geometric-tail argument withρ=27/256, within degree≤3.

This proof concerns the repeated discrete-preference subroutine, with known
global bound 3. It does not transfer the faster complexity guarantees of
Schneider–Wattenhofer's full multi-trial/RankingTrials algorithm. Their
published subroutine uses a neighborhood degree bound; the benchmark
explicitly substitutes the constant global bound.

## BE_BASE_D3: polynomial initialization and KW

Initially unique IDs are a proper auxiliary coloring with values below 65,536.
Represent each value in base p as a polynomial. Distinct values in the
declared range give distinct polynomials. Over a prime field, a nonzero
degree-r difference has at most r roots. With at most three neighbors and
p>3r, some point x makes the center polynomial differ from every neighbor
at that point. Outputting(x,p_v(x)) preserves properness: equal output
pairs would require equal values at the same point on an edge.

For bound 65,536, p 13 and degree≤4 satisfy 13>12; the next bound is 169.
For bound 169, p 7 and degree≤2 satisfy 7>6; the next bound is 49. The two
implemented Linial steps therefore produce a proper auxiliary 49-coloring.
The supplied polynomial evaluation is exact integer arithmetic at these
small bounds. Its result is not a learned arithmetic capability.

For a proper auxiliary coloring in{0,...,B-1}, set q=ceil(4B/5).
Vertices with old auxiliary a<q retain it. Each a≥q receives its own
disjoint block of four choices

    {4(B-1-a),...,4(B-1-a)+3}.

There are B-q such blocks, entirely below q because 4(B-q)≤q. At most
three neighbor auxiliary colors can block choices, so a free choice exists.
Two adjacent recoloring vertices had different old auxiliaries, hence
disjoint blocks. Recoloring vertices avoid retained neighbor colors too.
Thus the simultaneous reduction is proper and remains below q.

For every integer B>4, ceil(4B/5)<B. Starting at 49 the deterministic bound
sequence reaches 4 after 13 KW steps; together with the two Linial steps,
the native program uses 15 steps. This is the degree-three base construction,
not the high-degree recursive Barenboim–Elkin algorithm or its optimized
initialization complexity.

## Speculative GM and sequential greedy

GM partitions a random ID order into at most four balanced blocks. At each
assignment position, one vertex per block chooses first fit against all
previously assigned colors. A conflict after this phase can therefore occur
only between vertices assigned at the same position. Detection marks the
lower-ID endpoint of every equal-color edge at that position; every
conflicting edge has at least one marked endpoint. The unmarked subgraph is
proper.

Repair visits marked vertices one at a time and assigns a color absent from
all currently colored neighbors. Four colors suffice at degree≤3. For any
edge incident to repaired vertices, its last repaired endpoint avoids the
other endpoint's current final color. Edges with no repaired endpoint were
already proper. This proves final properness and termination, while allowing
intermediate conflicts and changes to assigned colors. If w is the maximum
block length and m the number of marked vertices, the schedule uses 2w+m
native steps. It is not an irreversible-color protocol or an anonymous LOCAL
round bound. Its supplied block schedule is part of the baseline cost.

Sequential greedy writes one vertex per ID-order step. At most three colors
are forbidden, so first fit exists. The last written endpoint argument proves
properness, and the fixed schedule finishes in n steps. No neighborhood
message-learning theorem is needed to solve coloring with this exact rule.

## Finite independent audit, defect and repair

`python -m dcl.integration.teacher_audit` independently implements every local
choice with scalar neighbor loops and integer polynomial evaluation. Across
all 103 unlabeled graphs with 1–6 vertices and degree≤3, two specified tapes
per graph and all seven teachers give 1,442 successful complete runs and
7,641 matching transition-choice comparisons. No statement of exhaustive
random tapes, IDs, or all adversarial states follows. Eight explicit-tape
LUBY checks additionally match the canonical random-trial implementation on
path 8, star 3, K4 and a fresh cubic 40 graph, at every state transition.

The original constructor did not reject directed graphs, parallel-edge graphs
or loops. Twenty-one regression cases failed before adding the domain check
and pass after it; the separate JP-cap case passes both versions. XML evidence
is preserved as `results/integration/teacher-guard-before.xml` and
`teacher-guard-after.xml`. The repair restricts inputs to the already declared
domain. It changes no mathematical rule or historical supported graph.
No published result is invalidated on this evidence. An external-domain guard
defect must not be inflated into a claim that Pedro's reported teachers were
unsafe on their benchmark.

Source hashes, representative native traces, explicit participation/candidate/
COMMIT tapes and finite-domain counts are in `results/integration/teachers/`.
The universal proof obligation remains conditional on the assumptions stated
above. Learned-policy safety, termination, full distributed preprocessing and
behavior outside degree 3 are separate open questions.

## Primary references and attribution limits

* [Johansson, 1999](https://www.sciencedirect.com/science/article/abs/pii/S0020019099000642): random-trial coloring, including the distinction from the coin-participation variant.
* [Luby, 1993](https://www.sciencedirect.com/science/article/pii/002200009390033S): parallel coloring/derandomization context; this repository does not implement the entire derandomization construction.
* [Jones–Plassmann, 1993](https://doi.org/10.1137/0914041): priority-based parallel coloring context; the benchmark is explicitly synchronous.
* [Linial, 1992](https://courses.csail.mit.edu/6.852/08/papers/Linial.pdf), section 4: polynomial color reduction.
* [Kuhn–Wattenhofer, 2006](https://ac.informatik.uni-freiburg.de/publications/publications/podc06.pdf), Theorem 5.1: disjoint-block iterative reduction.
* [Barenboim–Elkin, arXiv version 2](https://arxiv.org/pdf/0812.1379v2), Algorithm 2 base case: use of KW.
* [Schneider–Wattenhofer, 2010](https://tik-old.ee.ethz.ch/file/22a700b74fc2f37acb56d690e62c86cf/podcfp107_schneider_188.pdf): ColorTrials, rather than the full fast algorithm.
* [Gebremedhin–Manne, 2000](https://eecs.wsu.edu/~assefaw/publications/Concurrency12-2000.pdf), Algorithm 1: synchronized assignment, detection and sequential repair.

Primary sources were inspected on 23 October 2026. The proofs above spell out
the actual coded specializations; bibliography membership alone does not
certify equivalence to every version of the published algorithms.
