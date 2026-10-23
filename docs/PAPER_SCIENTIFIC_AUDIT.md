# Scientific audit handoff: learning distributed algorithms from teachers

This is a critical-review dossier, not a declaration that every historical
result was independently reproduced. The integrated English manuscript is
`paper/PAPER_DRAFT.md`, generated from its versioned template and data;
`paper/ucgnn/main.tex` and `main.pdf` are its compiled counterparts.
`results/integration/evidence-ledger.json` maps twelve principal claim groups
to sources, configurations, checkpoint identities, confidence and limitations.

## 1. Problem and scientific interest

We train local neural policies from graph-coloring algorithms. A policy may
imitate actions on teacher states, reproduce transitions, generate valid
solutions autonomously, preserve an appropriate execution cost, or reproduce
the teacher's full solution law. These are separate research targets. A
large set of correct colorings leaves room for many probability laws.

The contribution candidate is an integrated, auditable account of these
distinctions across teachers, information interfaces and palette architectures.
Neither task solvability by neural networks nor the general observation that
imitation can suffer distribution shift is claimed as new.

## 2. Relevant literature and novelty boundary

Verified primary sources cover neural algorithmic reasoning and CLRS,
distributed locality/GNN expressiveness, Deep Sets, attention, imitation
learning, randomized/priority/auxiliary/speculative coloring, MMD,
permutation calibration and Holm correction. New teacher bibliography:
`paper/ucgnn/integration-references.bib`; verification record:
`integration-bibliography-verification.json`. Earlier bibliography and its
dated verification remain preserved separately.

The literature supports the algorithm definitions and methodological context.
It does not certify our learned models. The native ColorTrials teacher is a
repeated preference subroutine, BE is only the degree-three base case, JP is
a synchronous variant, and GM retains a supplied block schedule. Their full
original-paper complexity bounds must not be inherited silently. A broader
novelty review and venue selection remain editorial work; no accepted
publication or exhaustive literature search is claimed.

## 3. Reconstruction of both research lines

Our preserved worktree branches study degree/palette transfer, aggregation
and normalization controls, local safety/progress, full labeled output laws,
robust tests and factorial size/degree effects. Pedro's three commits at
`fc756cef`, `80f486eb`, `6836d855` add seven native teachers, a direct-action
GNN, five newly retrained E1s, behavioral/distributional comparisons, and a
native E1 follow-up. Their history is merged without updating main.

The substantive reconstruction is
`docs/integration/PEDRO_RESEARCH_RECONSTRUCTION.md`. Its source manifest
records exact commits and original checkpoint hashes. Original Pedro sources
passed 309 tests; the merged baseline passed 483 before integration tests.
These are technical checks, not reproduction of his missing trained models.

Five retrained E1 and 35 native E1 weight files are missing. Their hashes do
not occur in the locally preserved checkpoint inventory; original coloring
banks and source pools are not tracked or accessible. The original runtime
path `/workspace/distributed-coloring-learning/data/` is absent. Consequently
Pedro's numerical results are reported archival evidence checked against
tables, not independently regenerated results. A location request was made;
work independent of that dependency continued.

## 4. Teachers, guarantees and actual implementation scope

`TEACHER_CORRECTNESS_AUDIT.md` states full hypotheses and conditional proofs.
Native programs support simple undirected graphs, degree≤3, global K4,
contiguous labels, n≤65,536 and distinct IDs where required. Initial states
are all pending; proper-state invariants are explicit. Exact-teacher
guarantees are not learned-policy guarantees.

| Teacher | Correctness argument | Cost / limitation |
|---|---|---|
| LUBY | Available proposal and equal-offer rejection; ideal conditional progress≥1/4 | Almost-sure uncapped termination, not deterministic completion by 128 |
| JOHANSSON | Same safety; a sufficient degree-three progress bound≥1/8 | Always active; common K4 differs from the smallest local lists |
| JP | Adjacent strict ID maxima cannot coexist; a pending maximum always exists | At most n uncapped rounds; monotone path 129 fails the 128 cap |
| COLOR_TRIALS | Strict preference winners cannot conflict; positive progress≥27/256 | Fixed global degree bound; no RankingTrials |
| BE_BASE_D3 | Polynomial root bound and disjoint KW recoloring blocks | Two Linial plus 13 KW steps; supplied arithmetic and phase clock |
| SPECULATIVE_GM | Same-slot conflict cover followed by sequential repair | Intermediate conflicts allowed; supplied four-block schedule |
| GREEDY | Four-color availability and last-written endpoint induction | Sequential supplied order, n steps |

The independent scalar audit compares 7,641 transitions over 103 unlabeled
graphs of 1–6 vertices and degree≤3, two specified tapes each, giving 1,442
successful runs. Eight explicit-array LUBY differentials match every state,
participation, available count, proposal, confirmation and update. This is
not exhaustive over all IDs, tapes, partial states or graph sizes.

The newly found input-domain guard defect accepted directed/multigraph/loop
inputs. Twenty-one saved failures pass after the guard repair. Supported
historical graph behavior is unchanged; no published benchmark number is
invalidated by this evidence. Pedro's earlier KW second-message repair is
preserved and attributed as his pre-final-cohort correction.

## 5. Architectures and what is supplied

Canonical E1 has 16,035 parameters, hidden/message dimensions 64/32, shared
node/color MLPs, MAX aggregation, learned availability, ordered normalized
prefixes and three learned decisions. DeepSets uses shared slot functions
and mean palette context. LocalPaletteTransformer uses attention within a
single node's palette. Both retain learned graph communication and COMMIT;
they are not non-GNN graph baselines.

The native interfaces retain teacher-specific context. Four teachers add
3,361 parameters to a large E1 core. Polynomial evaluation, native phases,
sequential writable schedules, palette-range masks and pending-node gating
are supplied. Available colors and valid confirmations are not imposed as an
oracle shield in autonomous learned inference. ColorTrials reads its final
candidate after the second exchange. BE predicts a valid field point whose
polynomial value is computed explicitly. These changes are scientifically
material and appear in the paper.

The integration reuses these native interfaces with all three palette
architectures. It does not adapt native teachers silently to local lists or
higher degrees. Canonical checkpoint and source compatibility remain in
the existing `src/dcl/` package.

## 6. Small models, baselines and necessity claims

The new small normalized model uses dimensions 8/4 and has 551 parameters
for LUBY/JOHANSSON/GREEDY,1,560 when native context is needed. The
no-communication control retains the larger parameter count while erasing
received states, relations and masks. Exact teachers require zero trainable
parameters. The older 36,719-parameter direct GNN is a different package
of representation, loss and architecture, not a one-factor control.

Small LUBY and greedy complete 1,280/1,280 attempts each across two seeds.
The small ColorTrials seeds split 640/640 versus 0/640. No-communication
models complete 38/8,960 attempts in total. The large core is therefore not
empirically necessary for the successful finite small-model cells. Received
information is useful under this training/interface design, while an exact
handwritten rule can replace learned message computation. No universal GNN
necessity, minimum parameter count, or impossibility for a failed small
network follows. Higher-degree transfer of these small checkpoints is not
established by the native benchmark.

## 7. Experimental evidence and exact source locations

| Evidence | Source and configuration | Checkpoints / independence | Result and confidence |
|---|---|---|---|
| Full-budget architecture study | `results/ucgnn-paper-study/FINAL_REPORT.md`; paper protocol and `paperstudy/config.py` | Five training seeds 41000–41004 per architecture; final specifications | E1/DeepSets/Transformer each 1,520/1,520; finite fixed-cohort evidence |
| Pedro native E1 follow-up | `results/multiteacher-e1/2026-10-22/architecture-comparison.csv`; native E1 config |35 unavailable original checkpoints, shared pools | Five teachers 1,980/1,980; JP1,631 and GM1,951; raw reproduction pending |
| New balanced learning | `results/integration/multiteacher/{training,functional}.json`; balanced config and frozen cost design |70 final checkpoints, two seeds,24 training graphs |49,280 new runs including exact references; all failures preserved |
| Historical full-law effects | `results/ucgnn-size-degree/main.json`; frozen factorial protocol |15 epoch 120 checkpoints,48 new regular graphs |115/170/180 V2 rejections per 240 architecture comparisons |
| Matched procedures | `results/integration/statistics/matched-archives.json` | Six identical archived 500-row banks, digests included | Three low-degree nonrejections and three high-degree rejections for both; V2 replay exact |
| New laws | `new-study-global.json` in the same statistics directory |350 planned comparisons; each model's own teacher |227 testable,6 rejections; conditional laws, no equality inference |

Historical full-budget negative controls are retained: raw endpoints lose
progress in 346 cases, SUM stalls in 1,196 and MEAN gives 839 improper final
colorings. Their roles differ and they must not be collapsed into one generic
failure metric. All 85 selected historical paper-study checkpoints were
authenticated and published in the recovery release, including nonpromoted
comparators. The 15 frozen global-law models remain separately identified.

## 8. Generalization axes and limitations

Local distributional evidence also includes the descriptive
`results/integration/multiteacher/local-action-marginals.json`: 420
model/phase/component records for all 70 policies on their unchanged validation
roots/private draws. Teacher/model histograms and paired mismatch counts are
retained for native actions, participation, candidates and COMMIT under teacher
or own proposals. Native error totals replay the original records exactly.
These are empirical common-state marginals; root dependence and validation
reuse preclude treating them as new population-law confirmation.

The historical functional cohort covers degrees through 200 and size transfer
through 640; the factorial covers n 128/256/512/1024 at degrees 3/10/50/100.
Local palettes are exactly d(v)+1. The factorial's three graph realizations
per cell and five trainings per architecture are a fixed cohort, not a
population confidence sample. Degree and palette size change together in
regular graphs. The secondary irregular construction is specific, not a
uniform sample of all degree sequences or graphs.

The new native study tests five fixed graph specifications at sizes 4–160,
degree≤3 and global K4, with two initialization replicas. It measures
native-interface transfer and does not validate local-palette/high-degree
behavior of all seven teachers. The earlier native follow-up through
N10,240 reuses known cases and is exploratory. These three scopes are not
interchanged in the article.

## 9. Complete output laws, power and reconciliation

`GLOBAL_DISTRIBUTION_RECONCILIATION.md` reconstructs both estimands, original
results, code/mathematical audits, matched tests, defects and open differences.
Our observations are labeled complete vectors conditional on proper completion
by 128 rounds. Pedro's behavior experiment includes joint terminal status at
256 and a color-name quotient, degree≤3, different seeds/weights and
stratified mixture MMD. His 0/100 findings are not directly contrary to our
high-degree graph/checkpoint rejections.

On matched archives both tests agree. On fresh center-only star alternatives
with TV99/199, mixture MMD detects 0/20 at m 256 and 3/20 at m 500, while V2
detects 20/20 at both. Exact pointwise 95% power intervals are[0,.1684],
[.0321,.3789], and[.8316,1]. These are nominal.05 controls with 999
permutations, not guarantees for the original 720-test family. Twenty null
trials per size yield 0/20 and 1/20 for the mixture, and 1/20 at both sizes
for V2; uncertainty is wide.

The native integration exposed an allocator assumption: paths have degree 2
but global K4. A call failed before a p-value because only three histogram
bins were allocated. The declared-alphabet adapter retains actual degrees,
all witness definitions and permutation logic in an isolated import of the
unchanged V2 source. Its addendum records the failure and regression. No
old local-list result is invalidated. The conditional exchangeability proof
still applies; no finite-power guarantee is automatically inherited.

Six new rejections occur only in small LUBY s 73001 and ColorTrials s 73000,
three graphs each. Both checkpoints succeed 640/640 functionally; all six
raw p-values are.00005 and Holm values.0175 across 350 planned hypotheses.
This independently supports functional/distributional separation under two
teachers. Large-model nonrejections in this low-degree cohort are retained
and do not prove equality. The 123 untestable cases are explicitly unavailable.

The original mixture statistic matches unbiased MMD and uses a valid
plus-one permutation form. No new level-inflation defect was found. The
actual cause of each of Pedro's original nonrejections remains partly open
until his original weights/banks can be tested. A synthetic power gap is not
a demonstrated explanation of every neural comparison.

## 10. Mathematical claims and proof status

These are conditional mathematical statements with proofs in the manuscript
and teacher audit, rather than theorems inferred from simulation counts:

1. **Safety versus progress.** From a proper partial coloring, available
   candidates and exact rejection of simultaneous equal neighboring offers
   preserve properness. This gives no uniform progress: on an all-free
   K-clique, always proposing zero with independent participation 1/2 advances
   only with probability K/2^K per round; always-WAIT never terminates.
   Status: proved induction and counterexamples for these explicit policies.
2. **Exact teacher guarantees.** Under the simple-undirected degree-three,
   four-color, distinct-ID and supplied-schedule assumptions, Luby, Johansson
   and repeated ColorTrials have safe conditional vertex progress at least
   1/4, 1/8 and 27/256 with fresh independent ideal draws. Thus
   P(T>t) is at most min(1,n(1-rho)^t), giving almost-sure termination
   and expected O(log n) native rounds. JP safely terminates within n uncapped
   rounds; the 128 cap is separate. Two Linial plus 13 KW reductions finish
   in 15 steps. Greedy takes n writes; GM finishes properly in 2w+m supplied
   steps, where w is the longest block and m the marked repair count.
   Status: conditional proofs, not verification of every numerical backend
   or learned model. GM explicitly permits intermediate conflicts.
3. **Canonical MAX COMMIT sufficiency.** With nonnegative deterministic
   shared messages, zero empty aggregation, legal candidate gathering and
   pending/participation gating, the canonical candidate-slot interface has
   six legal neighbor types. Correct strict COMMIT signs on all 64 subsets
   suffice at arbitrary multiplicities/degrees in real arithmetic, because
   MAX depends only on the type set. Status: sufficient conditional proof,
   not a necessary architecture characterization. Numerical assumptions,
   continuous native context, candidate diversity and termination are separate.
4. **Scoped information limitation.** A deterministic own-state/private-draw
   policy cannot reproduce every arbitrary-state first-free choice: a fixed
   neighbor of color zero versus one gives identical own inputs and different
   correct outputs. In the memoryless current-state random-trial interface,
   some COMMIT decisions also depend on a distance-two fixed color through
   a neighbor's proposal; Proposition 4 gives explicit path states/draws.
   Status: proved interface-specific indistinguishability examples, not a
   lower bound for every stateful distributed protocol.
5. **Scoped law equalities.** On K_n with exactly n colors, a vertex-equivariant
   protocol with exchangeable private inputs and positive proper-completion
   probability has uniform conditional law on all n! bijective colorings.
   Uncapped synchronous JP with fixed distinct IDs also equals descending-ID
   sequential first fit. A uniform order and its reversal have the same law,
   so unrestricted JP and ascending random-order greedy have equal final laws.
   Status: proved symmetry/order statements. They do not equate trajectories,
   costs, other teachers, or JP under restrictive cutoff conditioning.
6. **Event/coupling bounds.** For any fixed event A, TV(P,Q) is at least
   |P(A)-Q(A)|. An interval [l,u] covering this difference supplies the lower
   bound max(0,l,-u); simultaneous intervals preserve their joint coverage.
   For any valid coupling (X,Y), TV(P,Q) is at most P(X differs from Y).
   Status: proved inequalities; empirical certificates require their stated
   coverage assumptions. A poor coupling's large disagreement gives no lower
   bound, and a nonrejection gives no equality statement.

The paper and teacher audit give conditional proofs of safety/termination
for the defined exact rules, a finite-type MAX COMMIT sufficient certificate,
scoped own-state/radius-one information limitations, clique conditional-law
uniformity, observable TV lower bounds and coupling TV upper bounds. They
also explain unrestricted JP/random-order-greedy output-law equality under
uniform ID orders, while their shared-ID trajectories and native costs differ.
These are elementary scoped arguments, not claims of new general coloring
algorithms or automatic correctness of neural models.

The historical theory documents give the full progress/anti-concentration
conditions, interval arithmetic assumptions, permutation/Holm proofs and
factorial confidence construction. In particular, a valid local COMMIT
certificate does not prove candidate diversity or termination. Real-arithmetic
equivariance and finite numerical probes are distinct. A seeded PRNG trace
is not itself a universal infinite-time termination proof.

## 11. Threats to validity and possible weaknesses

1. Missing original collaborator artifacts prevent exact historical replay.
2. Two new initialization seeds and 24 training graphs give limited robustness
   evidence; a 500-update failure does not establish insufficient capacity.
3. Teacher-specific action alphabets, trajectories and supplied context differ.
   The same update count is not identical statistical difficulty or compute.
4. Native schedules and polynomial/equality features provide substantial
   computation. Calling the resulting policy an end-to-end discovered
   distributed algorithm would overstate the experiment.
5. Native status 1 includes early terminal collisions even when vertices remain
   pending; status 2 includes incomplete native schedules. These differ from
   the historical complete/cutoff vocabulary. Raw failure labels are preserved.
6. The new law tests have small successful subsets in some cells and no broad
   native-alphabet power certification. Nonrejections are weak evidence.
7. Historical graph-family/cohort reuse and shared teacher banks require blocked
   accounting; repeated tapes/checkpoints are not independent graph draws.
8. Global test families, event intervals and explanatory families are separate.
   No global 5% guarantee covers their uncorrected union.
9. Comparing observable lower bounds does not order TV across graph sizes.
   Candidate interventions remain exploratory and do not establish general
   causal mediation of every observed output-law effect.
10. Hardware/numerical representation changes can matter near decision margins.
    Conditional IEEE error bounds do not cover arbitrary GPU implementations.

## 12. Strongest narrative

The article should organize evidence around what it means to learn a
distributed algorithm: local actions, transitions, validity, cost and solution
law. The teacher specifies the available information; architecture and
supplied mechanisms determine which computation is learned. Small controls
show that a large GNN is not automatically necessary. Both historical and
new two-teacher examples show that successful coloring does not guarantee
the teacher's output law. Clique symmetry, JP/greedy equivalence and reported
coupling certificates prevent turning that statement into universal inequality.

## 13. Candidate contributions, classified

**Conditionally demonstrated mathematically:** safety/progress results for
the explicit exact rules; scoped MAX certificate; information counterexamples;
symmetry and coupling/event bounds; permutation/Holm guarantees under their
assumptions. External proof review remains appropriate.

**Experimentally supported:** integrated 70-model native interface; small
finite-cohort sufficiency; the 6 new conditional-law rejections; matched-bank
method agreement; the finite-law sensitivity gap; historical degree/palette
functional transfer and event differences with their preserved protocols.

**Plausible but not established:** universal architectural advantages,
parameter minimality, arbitrary-teacher learning from one raw interface,
general monotone degree/size TV trends, universal learned-policy termination,
or a complete causal account of the original cross-researcher discrepancy.

## 14. Open scientific questions

Which additional training coverage stabilizes the small ColorTrials control?
Would full matched budgets change JP/BE architecture comparisons? Which
teacher-specific information could be removed without replacing it by hidden
rules? Can progress be certified uniformly for any learned candidate policy?
How do output-law differences change beyond this fixed graph/cohort sample?
What would the frozen omnibus find on Pedro's original E1 banks, and are his
reported coupling certificates reproduced from the raw paired outcomes?

## 15. Editorial decisions before submission

Agree on author list/order, venue and licensing. Decide how much historical
mechanism/factorial detail belongs in the main paper versus supplements.
Preserve negative data whichever presentation is chosen. Strengthen the new
multi-teacher evidence with additional initialization/data replicas and an
explicitly matched larger budget if broad architecture conclusions are
desired. Do not present historical unavailability as completed replication.

## 16. Priorities for an independent reviewer

First, recover and authenticate Pedro's 40 original weight files and raw banks;
apply both procedures to identical rows under matched conditioning and
multiplicity. Second, check every theorem's assumptions against code and
confirm that native context does not add undeclared information. Third,
review the six new rejections, their success counts, valid sample splits,
permutation seeds and 350-member Holm family from the raw release.
Fourth, challenge necessity/minimality and teacher-difficulty language using
the small/no-message controls and unequal computational interfaces.
Fifth, assess novelty and determine which expanded experiments would change
the central conclusion rather than merely add more runs.

Reproducible material and publication manifests are linked from the README.
The final Git/release verification is recorded separately so this scientific
assessment does not infer completion from an engineering milestone.
