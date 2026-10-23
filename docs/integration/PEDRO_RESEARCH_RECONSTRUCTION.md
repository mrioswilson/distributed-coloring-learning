# Pedro research reconstruction

## Sources, chronology, and evidence status

The collaborator work is accessible in `origin/main`, ending at
`6836d85566d94710ef5bdea05e10702252520702` (22 October 2026 UTC). Three commits
are preserved as ancestors of the integration branch:

| Commit | Contribution |
|---|---|
| `fc756cef06d1d1ba9bf41bd3b73c4d727d188e7a` | Seven native coloring teachers and a common direct-action GNN |
| `80f486ebaeb269f094c1f267326a4628ee57b359` | Five independently retrained canonical E1 models and behavioral/distributional comparisons |
| `6836d85566d94710ef5bdea05e10702252520702` | Seven-teacher E1 follow-up, adapters, component audit and PDF report |

`PEDRO_SOURCE_MANIFEST.json` records exact source hashes, original model
digests and missing runtime paths. The original sources were tested in a
detached worktree: 309 tests passed. The merged baseline passed 483 tests.
Those tests independently reproduce technical checks, not the unavailable
historical training runs.

All published source, plans, aggregate tables, training histories and reports
were obtained. None of the five retrained E1 weight hashes or 35 native E1
weight hashes occurs in the locally inventoried checkpoint collection. The
recorded `/workspace/distributed-coloring-learning/data/` directory does not
exist here. The original raw coloring banks and native source pools are also
unavailable. The original branch has no separate recovery release supplying
these artifacts. A text request for their location was made while independent
work continued. Reproduction of those exact trained policies is therefore
open; new training must not be called their reproduction.

## Research questions and scientific objects

Pedro asks whether the teacher affects the difficulty of learning local
decisions and whether high local imitation accuracy survives autonomous
execution on larger graphs. He also separates agreement under shared random
inputs from equality of final-output probability laws.

His native benchmark deliberately includes heterogeneous computational models:
anonymous randomized parallel trials, fixed-ID priority decisions, color-wise
preferences, auxiliary deterministic color reduction, speculative repair,
and supplied sequential schedules. Thus it studies learning decisions inside
specified interpreters, not seven interchangeable anonymous LOCAL algorithms.
Part of the schedule and arithmetic is already provided by the representation.

The separate E1/Luby behavior experiment observes full labeled terminal color
vectors and their status. It additionally observes a quotient under global
color-name permutations. Neither observation is an individual-node candidate.

## Seven teacher definitions

Implementation: `src/dcl/multiteacher/programs.py`. The supported graph domain
is finite simple undirected graphs, contiguous labels, maximum degree at most
three, 1 through 65,536 vertices, global colors 0–3. Distinct IDs are sampled
without replacement from 0–65,535. IDs are available only where the interface
permits them. The repaired constructor enforces the simple-undirected domain.
The exact scope and conditional proofs are in `TEACHER_CORRECTNESS_AUDIT.md`.

| Native name | Actual implemented decision | Supplied information / limitation |
|---|---|---|
| LUBY | Pending nodes participate with probability 1/2, choose an available color by random rank, reject simultaneous equal neighboring offers | Common K4; this is the stated randomized trial rule, not the entire derandomization algorithm |
| JOHANSSON | The same random-rank proposal with participation 1 | Common K4 even at lower-degree vertices; larger initial palettes are permitted by the original analysis, but differ from initializing each list to d(v)+1 |
| JP | Pending local ID maxima choose the smallest free color | Distinct fixed IDs; a synchronous priority variant |
| COLOR_TRIALS | Independent discrete preferences 0–3 per free color; retain the lowest color with a strict winning preference against every eligible neighbor | Repeated ColorTrials with a supplied global degree bound; no RankingTrials implementation |
| BE_BASE_D3 | Two polynomial Linial reductions, fields 13 then 7, followed by KW reduction from 49 colors to 4 | Only the degree-three base case; exact local polynomial arithmetic and phase clock are supplied |
| SPECULATIVE_GM | Four balanced ID-order blocks; simultaneous first-fit assignment by block position, same-position conflict marking, sequential repair | Synchronized block-partition variant; PRAM-style schedule, not anonymous LOCAL execution |
| GREEDY | First fit in increasing-ID order | One supplied writable vertex at each sequential step |

The first four have a 128-step evaluation cap. JP can require more than 128
steps on a valid 129-vertex monotone-priority path; a regression test retains
this example. BE, GM and greedy follow their native schedules instead. A
native sequential step is not interchangeable with a two-message parallel
round, and neither is a training wall-clock second.

The random-trial teacher is mathematically defined using fresh independent
private uniforms. The native interpreter generates float 32 uniforms; the
canonical E1 behavior experiment generates float 64 tapes and converts the
model inputs to float 32. These numerical laws are distinct representations
of the ideal teacher. Differential checks use identical explicit arrays, not
an assumption that equal numerical seeds imply equal random tapes.

## Direct-action GNN study

Plan: `configs/multiteacher/comparison.json`; results:
`results/multiteacher/2026-10-20/`.

`LocalGNN` has 36,719 parameters and two message stages, each combining MAX
and SUM neighbor summaries. Inputs include 42 local features and 19 relational
features: pending/fixed colors, native permissions and phase, permitted IDs
and clocks, private draws/preferences, auxiliary values, polynomial evaluations
and equality comparisons. It predicts WAIT/color, an encoded auxiliary pair,
or a repair mark according to the supplied phase. Polynomial equality and
schedule permissions are provided features, not learned algorithmic results.

The training pool consists of 800 regular N40/degree 3 graphs, three native
trajectories per graph; validation has 50 independent graphs and four common
N40/N160 rollout monitors. All teachers share graph and tape identities, but
their trajectories and eligible root decisions differ. Each of five seeds
sees the same teacher-specific pool. Seeds therefore replicate initialization
and sampling, not independently generated datasets.

Every model receives 17,760 Adam updates, learning rate.001, batch 1,680 root
decisions:29,836,800 supervised decisions. Direct action/color/auxiliary
classification uses cross-entropy, and mark prediction uses binary
cross-entropy. Sampling is with replacement from eligible native decisions.
The final planned update is selected. Teacher forcing supplies trajectory
states during training; learned rollouts evolve autonomously at evaluation.
This differs from on-policy DAgger and from counterfactual E1 supervision.

## Native E1 follow-up

Plan and exact adaptations: `docs/MULTITEACHER_E1.md`,
`configs/multiteacher_e1/comparison.json`; outputs:
`results/multiteacher-e1/2026-10-22/`.

The follow-up retains the canonical 16,035-parameter E1 core, including MAX,
shared node/color state, learned availability, ordered normalized prefixes,
separate participation, candidate and COMMIT modules. Luby/Johansson/greedy
need no context extension. JP, ColorTrials, BE and GM add 3,361 context
parameters, totaling 19,396. These are explicitly extended E1 models.

ColorTrials requires learned availability/preferences to be exchanged before
the final candidate is chosen. BE-Linial predicts a field point; the supplied
adapter computes its polynomial value. This replaces the earlier 169-class
joint auxiliary choice with a different constrained representation. KW's
receiver-relative auxiliary occupancy enters both message stages. GM learns
marks and repair choices while retaining the supplied schedule. These changes
preclude interpreting before/after differences as the causal effect of the
E1 architecture alone.

The same immutable native pools, five seeds and 17,760-update budget are used.
The first 1,480 updates use teacher COMMIT proposals. Subsequently the loss is
participation BCE + candidate CE +.5 own-COMMIT BCE +.5 counterfactual-COMMIT
BCE. Every COMMIT label is recomputed for the proposals actually queried.
Star-local counterfactual conflicts use probabilities.17 and.04; ColorTrials
and KW additionally receive changed root queries. These are not identical
sampling distributions to full-graph E1 counterfactual training.

The exact rules compute labels and diagnostics. In learned inference they
do not mask occupied colors, veto unsafe actions or replace difficult
decisions. Native phase/schedule permissions remain supplied by the
interpreter. Learned collisions, bad auxiliary choices and incomplete outputs
are retained. A previous KW missing-occupancy adapter error was documented
and repaired by Pedro before the final BE training cohort; this integration
does not rediscover that historical repair as a new scientific result.

## Published functional results, retained with their provenance

Source: `architecture-comparison.csv`. Denominator 1,980 per teacher is five
initializations times 396 already-published graph/tape cases. It is not 1,980
independent graphs. Evaluation spans regular/irregular N40–10,240 and
structured controls. The E1 follow-up is exploratory because it reuses those
known test cases.

| Teacher | Direct GNN successes /1,980 | E1/adapted successes /1,980 |
|---|---:|---:|
| LUBY | 1,857 | 1,980 |
| JOHANSSON | 1,919 | 1,980 |
| JP | 1,635 | 1,631 |
| COLOR_TRIALS | 1,980 | 1,980 |
| BE_BASE_D3 | 872 | 1,980 |
| SPECULATIVE_GM | 1,975 | 1,951 |
| GREEDY | 1,842 | 1,980 |

Local action accuracy can be nearly perfect while autonomous success falls:
the native E1 JP average local action accuracy is.999735, yet 1,631/1,980
rollouts succeed. Marks and WAIT class imbalance require phase/component
counts; `unsafe_own_commits` denotes violation of each teacher's native rule,
not the same physical collision event for all teachers. The project retains
`teacher-task-baselines.csv`, phase metrics, all failed cases and paired-cost
tables. Runtime comparisons use successful paired cases within one teacher.

## E1/Luby behavior and global laws

Source: `src/dcl/e1_behavior.py`, `behavior_statistics.py`,
`results/e1-retrained/2026-10-21/PROTOCOL.md`, `manifest.json`,
`distribution-tests.csv`, `by-graph.csv`, `key-results.json`.

Five new canonical E1 models use seeds 0–4,800 cubic N40 graphs, three tapes,
120 epochs, Adam.001, batch 128 and ten warmup epochs. Training is CPU and
uses the original canonical modules. These are different weights from both
the historical E1 seed 0–4 models and our paper-study seeds 41000–41004.

The test contains 80 new regular/irregular degree-at-most-three graphs at
N8,12,40,160,640 plus path 8/cycle 8/K4 controls. Per graph,256 independent A
tapes generate Luby references; independent B tapes generate Luby and all
five models. Sharing B supplies a valid coupling without making A/B a
paired two-sample test. Collisions and partial outcomes remain in a joint
terminal-color/status law at cutoff 256.

The kernel is the mean of exp(-gamma normalized Hamming distance),
gamma 1,4,16, multiplied by terminal-status equality. The unbiased MMD
statistic is calibrated by 3,999 balanced whole-row permutations within each
graph. Graph-specific statistics and independent permutation draws are
averaged within family/size strata. Holm covers 100 primary stratum/model/
representation hypotheses: labeled colors and the color-name quotient.

The published outcome is 0/100 rejections,102,400/102,400 successful learned
random-graph runs, and 21 simultaneous TV≤.05 certificates across labeled
and quotient graph/model pairs. The latter use a Bonferroni-adjusted exact
binomial upper bound on shared-tape disagreement, not nonrejection. Ten
certificates concern labeled laws and eleven concern quotient laws; they
must not be described as 21 labeled-law certificates. A large coupling
disagreement is not a TV lower bound.

The underlying colorings and weights are absent, so these numerical results
remain reported historical evidence, independently checked against published
tables but not independently regenerated. The reconciliation document
separates that fact from our new matched-bank and independent-power results.

## Integration and open questions

Our historical work emphasizes degree/palette extrapolation, conditional
safety and progress, and full labeled local-list output laws. Pedro adds
teacher diversity, native multi-phase decisions, controlled supplied context,
and explicit coupling-based approximate-equivalence certificates. Both show
why local accuracy, functional validity and distributional fidelity require
different evidence. Neither establishes universal correctness of a trained
network or a universal architectural winner.

The integration implements all seven native interfaces for E1, DeepSets and
Transformer, plus small/no-communication controls, in a separate balanced
short-budget study. Its original checkpoints and fresh banks are provided.
It does not silently expand the native teachers beyond degree 3/global K4.
The outstanding highest-value historical dependency is Pedro's original
runtime archive, whose exact missing model digests are recorded alongside
this reconstruction.
