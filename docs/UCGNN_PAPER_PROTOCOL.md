# UCGNN paper study: prospective protocol v1

Registered on 2026-10-21. This document must be committed before training and
before generating or reading the new confirmatory cohort. The coordinator
records the commit and SHA256; a later amendment cannot silently replace it.
Historical source: main HEAD 708bb57 and extreme HEAD 2925197, plus the hashed
snapshot in `docs/historical/source-snapshot.json`. Historical checkpoints,
graphs and reports remain read-only in their original directories.

## Questions and falsifiable hypotheses

The estimand is the effect of a programmed intervention under the fixed
training procedure below, not the impossibility of training another design.

* HN: changing ONLY learned-prefix endpoints from normalized to raw reduces
  out-of-degree success or proposal diversity, while preserving in-degree
  learning. A null or opposite result refutes necessity under this procedure.
* HM: replacing neighbor MAX by SUM or MEAN harms transfer because multiplicity
  or composition changes the received representations. MAX's invariance is
  architectural; better learned safety or progress remains an empirical claim.
* HP: an ordered cumulative sum helps degree transfer beyond a generic shared
  palette processor. Compare DeepSets and local palette attention without any
  cumulative-sum input. These comparisons change more than one operation and
  cannot identify the isolated causal effect of a prefix alone.
* HC: Own-only COMMIT can miss fixed-color conflicts hidden by safe proposals.
  Compare existing E1/C1 and one new paired Own-only training; distinguish
  proposal validity, COMMIT safety, progress and graph success.
* HQ: a frozen monotone calibration of E1-0's rank quantile predicts choices
  on held-out K and availability patterns with mean absolute rank error <=2
  and forbidden probability <=0.001. A descriptive approximation, not identity.
* HT: attention trained with K=4 can transfer to K=11,21,51,101,201. No favorable
  direction is assumed. Technical OOM/timeout differs from algorithm failure.

## Treatments, inputs and isolation

Six primary treatments: `normalized`, `raw_endpoints`, `sum`, `mean`,
`deepsets`, `transformer`. Two auxiliary pilots: `own_only`, `cyclic_teacher`.
The original `UniversalColorGNN` source is unchanged.

The first four have exactly the original tensor dimensions, modules, parameter
names and count (16,035). `raw_endpoints` supplies (alpha,S,S+alpha,r) in the
same four scorer slots as (alpha,S/D,(S+alpha)/D,r); D=max(sum alpha,0.5).
It has no extra total-mass input. Tensor-by-tensor initialization, training
examples, minibatch permutations, counterfactual RNG and optimizer match
`normalized` within each seed. SUM and MEAN replace the reduction in BOTH
communications; all other operations match. Isolated nodes receive zero;
MEAN divides by actual in-degree, clipped at one. SUM receives no scaling.
MAX retains nonnegative messages and zero for empty neighborhoods.

DeepSets and LocalPaletteTransformer use shared node-color messages, two
neighbor communications, learned availability and COMMIT, and the same local
colors/random draws. Palette processing is within a node. They receive
functional position c/max(K-1,1), 1/K and r_c, but NO ordered cumulative sums,
oracle availability, shield or global node attention. DeepSets mean-pools
learned slot features; attention uses a shared projection and one local
self-attention block. Padding is masked with each node's own K. Report actual
parameter counts; target roughly 16k, accept 12k--24k without post-test tuning.
An optional position-free diagnostic has indistinguishable available slots:
exact ordered teacher imitation is impossible there in general. Argmax
tie-breaking by index is not learned diversity.

## Training, selection and promotion

New independent training seeds: **41000,41001,41002,41003,41004**. Graph and tape
streams are the existing named M6B streams indexed by these new seeds; they
do not reuse historical replica seeds 0--4. Within a seed all primary models
see identical stored teacher trajectories: 800 connected REGULAR graphs,
N=40, Delta=3, K=4, three tapes/graph. Validation: 200 new N=40 graphs, mixed
REGULAR/IRREGULAR, one tape, plus 4 rollouts/family at N=40/160, Delta=3.
Validation never includes a held-out degree. Fingerprint disjointness is
checked. Adam lr=0.001, batch 128 round examples, 120 epochs, warmup=10;
loss P+C+0.5 COMMIT-own+0.5 COMMIT-counterfactual. No scheduler or search.
Own-only uses own proposals in both COMMIT halves after warmup.

Run seed 41000 for every primary treatment FIRST as the learning/stability
pilot at the full fixed budget. This seed counts as the first of five, not an
additional independent replicate. Promote automatically to seeds 41001--41004
if all epochs are finite, participation accuracy >=0.98, candidate accuracy
>=0.90 and own-COMMIT accuracy >=0.95 on final in-degree local validation.
These are learning/technical gates, never tests of high-degree success.
If a gate fails, keep the negative pilot and diagnose it; do not tune it until
it passes. All primary promotion decisions precede confirmatory evaluation.
Own-only and the alternative teacher are one-seed exploratory studies.

**Primary checkpoint: epoch 120**, regardless of validation rank. Keep epochs
1,10,20,40,60,80,100,120 and resumable last including RNG and optimizer.
Separately retain validation-best under the historical lexicographic rule:
rollout success, fewer unsafe/state, fewer rounds, teacher action fidelity,
local loss (first checkpoint on ties). Evaluate best on the same held-out
cohort but label it a separate selection policy; never pool with epoch 120.

## Budget and runtime

Up to **36 accumulated active GPU process hours**, including training,
validation and heavy evaluation; conservative wall accounting per child.
One GPU worker at a time under the existing global flock. Persistent tmux
session `ucgnn-paper-study`; JSON task manifest; atomic per-epoch checkpoints
and per-case results. Resume repeats only an incomplete epoch/case. A source
hash mismatch refuses exact training resume. No other services are stopped.
GPU allocator cap 4 GiB with >=1 GiB external reserve, min 10 GiB disk,
2 GiB available RAM. Wait for external resource contention without consuming
active budget. A worker records OOM/timeout; programming errors halt the queue.
Training per-run timeout 2400 s, final case 180 s, diagnostics 1800 s.
Reserve roughly 12h training, 16h evaluation, 6h mechanisms, 2h recovery.
Unused allocations may be used by later predeclared tasks within the total.

## Independent confirmatory graphs

Namespace **202610210211**, split `paper_confirm_v1`; seeds derived by SHA256
of (protocol,namespace,degree,N,family,index,stream). These graphs are generated
only after the protocol commit. Same instances/tapes for every model. Exclude
fingerprints/seeds used in historical training/validation/evaluation or any
new train/validation split. The old extreme cohort is historical evidence.

Fixed matrix (all degrees are declared maximum, K=Delta+1):

| Tier | Delta | N | Family | Graphs per cell | Tapes |
|---|---|---|---|---:|---:|
| Core | 3,5,10,15,20,50,100 |160|REGULAR,IRREGULAR,HETEROGENEOUS|8|1|
| Core |200|400|same three|8|1|
| Size |3,10,20,50,100,200|640|same three|4|1|
| Structured |3,5,10,15,20,50,100,200|Delta+1|CLIQUE|1 deterministic graph|3|
| Structured |same eight|max(160,2 Delta)|BIPARTITE circulant|2|1|

Total 304 graph-tape cases/model, including repeated tapes of deterministic
cliques (NOT independent graphs). Max 256 rounds, continue other cases after
a failure and continue the full rollout after an unsafe commit. Record both
first unsafe event and all subsequent behavior. At 8 graphs, uncertainty is
large: this is a feasible stratified first study, not a high-precision claim.
Historical E1 five replicas, A1/C1 one each, and M6B specialists at degrees
3,5,10 are reference evaluations with epoch 120 on the compatible subset;
M6B is never padded to an untrained palette. Secondary best-policy evaluations
use the core tier, and can remain explicitly pending if budget is exhausted.

## Metrics and statistics

Primary: complete proper coloring by 256 rounds, per architecture/training
seed/family/degree/N. Secondary: unsafe COMMIT, fixed/simultaneous conflicts,
forbidden proposal, exact candidate/action fidelity, valid commits rejected,
participation, rounds, pending nodes, wall time and peak CUDA allocation.
Retain per-round proposals, safe proposals, executions, collisions, time to
first/next execution, and candidate histograms for failures and progress probes.
Never treat OOM/timeout/unexecuted as a wrong coloring; report their counts and
the planned denominator. A round cap is an algorithmic nontermination outcome.

Report Clopper--Pearson 95% intervals per fixed network and random-graph cell;
do not apply them to pooled executions of five networks on shared graphs.
Report all five seed rates and paired architecture differences. Hierarchical
paired bootstrap: sample training-seed pairs and graph IDs within strata,
preserving common tapes and all paired model observations (2000 draws,
bootstrap seed 20261021). With five networks, inference over training seeds is
limited. Cliques are deterministic stress tests with repeated tapes, reported
separately. HN and HM are three prespecified contrasts (normalized vs raw,
SUM, MEAN); provide effect intervals, and Holm-adjust any claimed hypothesis
test p-values. Other comparisons are exploratory; no superiority claim from
selective significant cells. No pooling of heterogeneous families into a
binomial confidence interval. Negative pilots remain in the report.

## Frozen mechanistic diagnostics

Candidate calibration: E1-0 only, fit a 257-point monotone linear interpolation
of chosen available rank/k at midpoint r values using K=101 all-free star.
Save and hash BEFORE evaluating held-out states; no refitting. Prediction:
floor(k*g(r)), clipped to [0,k-1], mapped to sorted available colors. Held-out
K=17,63,251,997; k in {1,2,4,ceil(K/4),ceil(K/2),K}; patterns LOW,HIGH,
SCATTERED,ALTERNATING,EXTREMES, deterministic SHA seeds. Use 2049 uniform
midpoints and float32 predecessor/value/successor of j/k boundaries.
Probability metrics use midpoints ONLY; boundaries are a separate stress set.
Measure rank error, TV to uniform, entropy, exp(entropy), max mass, forbidden
rate, pair collision sum(q^2), monotonicity and calibration prediction error.
New models get identical distribution probes at K=4,11,21,51,101,201, and
held-out states as budget permits; attention limit is reported explicitly.

COMMIT: enumerate all six legal neighbor types in the own candidate slot and
64 sets for own (uncolored,participating,offering); also own wait/no-offer and
colored cases. Check realizable stars, duplicated types, colors at endpoints,
float64 vs float32, perturbations +/-1e-6 of pooled messages, margins and a
Lipschitz bound through affine/ReLU layers. Exact real-arithmetic conclusions
must have explicit sign-margin assumptions; float samples are not certificates.

Progress: homogeneous all-free states and valid partially colored clique cores
with fixed leaves, pending-core degree in {3,10,20,50}, available colors in
{2,4,11,21,51} when K=core_degree+fixed_count+1 permits them. Use 16 tapes/state,
256 rounds; record state construction and conditional independent-choice
prediction p_v sum_c q_v(c) product_u(1-p_u q_u(c)). Independence conditional
on fixed state and independent per-node draws must be stated. Reject stronger
claims using concentrated/correlated candidate counterexamples.

## Secondary teacher and RL

Alternative teacher starts at floor(K*r_c) and scans cyclically for first
available color. Participation and exact conflict rejection are unchanged.
For nonempty A, P(c)=cyclic gap from previous available color to c divided by
K. Test this law and safety before training one fresh seed 41005; no previous
weights. Same optimizer/120 epochs/800 graphs/three tapes; record differing
trajectory lengths and updates, so this is not an isolated equal-update
causal comparison. It runs after primary training and evaluation are secured.
Teacher-free RL is a separate written design; no large RL training here.

## Release requirements

Generate CSVs, tables and figures programmatically; retain exact commands,
source/checkpoint hashes, cohort IDs, technical failures and pending tasks.
Manuscript separates incorporated operations, measurements, hypotheses and
proved conditional statements. Verified bibliographic primary sources only.
Local coherent commits; no remote push. A persistent process survives client
disconnects, not container restart; explicit resume command is required.
