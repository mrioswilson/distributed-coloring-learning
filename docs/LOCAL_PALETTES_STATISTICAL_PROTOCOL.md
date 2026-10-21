# Preregistered local-palette and distribution study, v1

Date: 21 October 2026. This document is committed before confirmatory evaluation.
Source base 1d8857b. No training, selection, tuning or checkpoint replacement.
Twelve GPU-process hours maximum (conservative wall duration of CUDA workers,
including startup). Single global GPU lock; no foreign process terminated.

## Questions and fixed models

A: Can the 30 frozen epoch-120 networks (six treatments × seeds 41000–41004)
produce a proper complete coloring using only own K=d(v)+1? Compare each network
on paired graph/tape cases under GLOBAL, LOCAL, OVERSIZED-1.5 and OVERSIZED-2.
Own palette is allowed input; neither graph maximum nor storage width is a model
feature. The adapter must pass invariance and historical-forward checks before
the main test. Exact uniform teacher runs all regimes; exact cyclic candidate
teacher is a secondary distribution-law control. Historical E1 is contextual,
not pooled as five more independent replicas. M6B is dimension-incompatible.

B1: shared-r fidelity is not law equality. B2: estimate conditional candidate laws,
TV, JS (natural logs), Hellinger, entropy, exp(entropy), 1/Σq², maximum mass,
collision Σq², forbidden mass, absolute insertion-rank error (allowed draws only).
B3: identify teacher versus network from candidates with hidden randomness at
fixed state. Main balanced accuracy uses equal priors, log-likelihood threshold
zero; separately calibrate a level .05 test and report power and actual test FPR.
Candidate law similarity is not presumed to imply faster coloring.

## Cohorts, pairing, and randomness

New seed namespace `local-palette-v1`, hash-derived 63-bit PCG64 seeds. Pilot
namespace `local-palette-pilot-v1`; separate states/graphs. Graphs are generated
only after this commit, identities and edge hashes stored before evaluations.
Eight families REGULAR, IRREGULAR (capped Erdos–Renyi), HETEROGENEOUS (capped
preferential attachment), STAR (one/two disjoint stars), BIPARTITE, HUB (wheel),
DENSE (clique/diluted clique), LOW_HIGH (hub plus low-degree forest), at target
degrees 3,10,50,200, two graph specifications each. Base n=max(80,2d+2), except
STAR/DENSE/BIPARTITE/HUB where the natural size is stored. Additional n=640
IRREGULAR/HETEROGENEOUS/LOW_HIGH at target degrees 10,50, two specs each:
**76 graph specifications**, two tapes each, four regimes, **608 cases per
network or teacher; 18,848 total**. Deterministic/isomorphic structured graphs
are not advertised as independent random graph draws. Actual maximum degree is
recorded and defines global palettes. Oversized uses ceil(factor × actual Δ)+1.
256 rounds is an algorithmic cutoff, not technical timeout. All cells attempted,
including failures. Regular GLOBAL/LOCAL are a paired equivalence control.

Store graph edge hashes and compare against preceding paper/extreme graph
manifests when those hashes are available; independent seed generation is the
independence guarantee, not a claim that graph isomorphism never recurs.
Tapes use identical per-node/per-round rp,rc across policies and regimes, always
drawn for fixed nodes too. Different rounds and nodes use independent draws.
Numerical model inputs are float32 as described by the architecture audit.

## Conditional states

K=4,11,51,101,201,501; patterns: ALL_FREE; SINGLE at first, quarter, middle,
three-quarter and last slot (deduplicate positions); TWO at adjacent start,
widely separated endpoints, adjacent middle; THREE spread over the palette;
BLOCK at start, middle, end (length max(1,K//4)); ALTERNATING even; SCATTERED
deterministic hash-generated min(7,K) colors. Also A={1,6,10} at K=11 and
A={0,7,19,63} at K=64. Deduplicate identical sets within K.
For each, central d=K−1, one fixed neighbor per unavailable color, |A|−1 pending
neighbors. A local-list witness extends each high-color fixed neighbor with
private pending leaves so its own degree permits its color. Neighbor K values
and witness metadata are stored; only their legal state bits enter messages.
This distinguishes realizable local first-neighborhoods from invalid synthetic
states. Autonomous contexts: up to six first nontrivial partial states from
new teacher/model pilots (explicitly exploratory), and six confirmatory LOCAL
states from the lexically first HETEROGENEOUS graphs, first pending node after
round 1 under normalized/deepsets/transformer seed41000. Extract by deterministic
replay and freeze before evaluating alternative policies on them. Provenance
includes graph, regime, tape, round, generator, node and arrays. Results on
these trajectory-selected states are secondary, not a uniform state sample.

## Distribution sampling and inference

Every network/state gets 2,049 deterministic midpoint quadrature points plus
float32 boundary/nextafter probes. These are numerical diagnostics, with **no
binomial confidence intervals**. Independent estimation MC N=32,768 and independent
test MC N=8,192 per state. B1 uses shared converted rc against teacher, with
the same test draws used for paired distribution contrasts (not independent
observations between policies). Store estimation/test counts separately.
The float64 versus converted-float32 teacher disagreement is reported separately.
For B2, provide a distribution-free 95% TV uncertainty radius
sqrt((K log 2 + log(1/.05))/(2N)) (cap at 1), using the multinomial subset-union
bound; interval for true TV is empirical TV ± radius, clipped to [0,1].
Approximate equivalence tolerance **TV ≤ .05**; declare only when the upper bound
is at most .05. A bound including .05 is inconclusive, never equality.
These are pointwise intervals; no simultaneous universal-state claim.
Fidelity, forbidden-mass and B3 rates have Wilson 95% intervals; zero observed
forbidden counts additionally get the exact one-sided 95% upper bound.

B3 subset fixed in advance: ALL_FREE K=4,51,201; SINGLE-middle K=51;
THREE-spread K=51; ALTERNATING K=201, for **all 30 networks**. N=256 independent
trials per source at maximum m=2000, use nested prefixes m=1,5,20,100,500,2000.
This pairs points on each curve, not 6 independent trial cohorts. All model
candidate draws are fresh forward evaluations, never resampling the fitted law.
Model probabilities use estimation counts with Jeffreys pseudocount **.5 per
one of K slots**, including unavailable colors. No empirical zero is declared
an exact zero. Teacher probabilities are analytically known; an unavailable
observation has infinite evidence against this teacher. Threshold calibration
uses 1,024 independent exact-teacher trials per m (separate RNG stream), choosing
the conservative 95th percentile and rejecting **strictly above** it. Ties and
their resulting conservatism are reported. The balanced-accuracy threshold 0
uses random 50/50 ties from a separate seed. Test H0 and H1 draws are independent
of estimation and calibration. Wilson intervals apply to source-specific power
and FPR; balanced-accuracy interval is obtained by averaging their separate
interval endpoints (conservative via 97.5% intervals for each component).
Power ≥ .8 at observed FPR ≤ .05 is descriptive; a stronger certified statement
requires the corresponding interval bounds and is labeled separately.

Report empirical plug-in likelihood ratios as **estimated-distribution tests**,
not exact Bayes rules. Known-law controls: identical uniform versus permuted
quantile (B1 zero, B2 zero, Bayes accuracy .5), and two-category biased laws
(exact multinomial/binomial product TV). For learned laws, give one-draw TV-based
Bayes intervals and product-law coupling/Hellinger bounds with uncertainty;
point estimates of product bounds are explicitly surrogate calculations.
No inference of indistinguishability from low empirical test power.

## Progress intervention and primary outcomes

Primary A outcome proper complete coloring, then rounds (censored at 256),
unsafe exact-rule-violating confirmations, final edge conflicts, out-of-list
proposals, forbidden proposals, pending counts. Per-round participation,
commits, safe proposals, simultaneous conflicts, candidate diversity, memory,
time, and failed final states are retained. Learned actions are never shielded.
Distinguish implementation error, unsafe decisions, progress cutoff, success,
OOM/technical failure, and unexecuted cells. Continue after individual failures.

Controlled progress: all-free cliques at K=4,11,51,201, exact COMMIT and p=.5;
estimate q for the same state, evaluate P_v=.5 Σq_v(c) Π_u(1−.5q_u(c)),
compare uniform, learned, and a 50:50 mixture with uniform (explicit policy
intervention, not a historical network). On six frozen heterogeneous LOCAL
states evaluate all pending-node marginals and exact-COMMIT expected progress
using independent MC draws, and compare one-round empirical exact and learned
COMMIT. These are controlled distributional interventions, not causal claims
from autonomous round-count correlations.

Aggregate by family/actual degree/n/regime/model seed; paired contrasts use the
same graph/tapes and report per-network results. Graph-cluster bootstrap 2,000
replicates for success and capped-round differences, resampling graph identities
and carrying all seeds/tapes; five network means are separately displayed.
Families of deterministic graphs are descriptive. No pooling 5 networks × shared
graphs as independent graph draws. Exploratory multiple-state searches are
marked; no retrospective checkpoint selection or silent coverage reduction.

## Pilots, budget, stopping, reproducibility

First test invariance, legal/out-of-list states, exact teacher, LRT controls and
resume. Pilot six architectures seed41000 on six separate small graphs, and
candidate speed at K=51/201/501. Do not tune weights. If predicted main runtime
exceeds 12h, queue A before B; complete cells until budget, explicitly retain
pending cells. No architecture-specific reduced test to hide negatives. OOM is
technical, not coloring failure; retry once at half chunk size, retain both logs.
Per-case atomic output enables resume. Workers stop before another task if budget
is spent; task-level 30min wall cap and global 12h cap. Interrupted tasks remain
recoverable; no continuity claimed across reboot without relaunching resume.

Checkpoint SHA256, protocol commit/hash, graph/state/tape seeds and hashes,
environment versions, source hash, command and GPU costs are in manifests.
Coordinator runs in tmux `ucgnn-local-palette-study`; commands status/run/resume/
audit/report are documented at repository root. No remote push.
