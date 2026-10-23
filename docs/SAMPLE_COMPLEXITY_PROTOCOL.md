# Sample complexity and empirical learnability: prospective protocol

Protocol `sample-complexity-native-v1`, namespace `20261023-sc-independent-v1`.
Configuration: `configs/sample-complexity.json`. Executable preregistration,
source hashes, cost design and exact-teacher validation are exported under
`results/sample-complexity/`. Original 70-model data/checkpoints and all prior
experiments remain unchanged in separate directories and Git history.

## Scope and hypotheses

The experiment asks how many **acquired node/action-component labels** allow
a policy to complete an explicitly defined finite benchmark without a single
functional failure. It does not prove correctness for arbitrary graphs or
tapes, asymptotic sample complexity, minimum network size, or distributional
equivalence. Main teachers are the actual seven native K4/degree-three programs,
with regimes A/B defined in `TEACHER_MODEL_INFORMATION_COMPATIBILITY.md`.

Confirmatory design hypotheses: teacher/architecture packages can differ in
the first development-perfect, finally confirmed budget (H1/H2); an empirical
functional PASS is logically separate from action, time and law matching (H6).
The order-interface, optimization, capacity, communication and one-aggregation
interactive analyses are exploratory (H3–H5), with fixed cells chosen before
main training. No architecture or difficult teacher is dropped by pilot accuracy.

## Observation and accounting

A supervised query returns one component for a specified teacher, trajectory,
global/native state, node and action component. Normal color phases acquire
three stored labels per root: participation, proposal, teacher confirmation.
They are the teacher's actual decomposition on its state, including a
well-defined proposal for nonparticipating nodes. BE's Linial phase acquires
one point label, whose polynomial value is a supplied local decoder. GM's
detection phase acquires one mark label. Availability is a learned latent
quantity with no separate target or query. No fresh own-proposal or
counterfactual confirmations are generated during optimization.

Immutable caches carry explicit per-example trajectory, node, native step,
global-state hash, label and phase. Component bundles are never partially
queried. Actual acquired prefixes can fall at most two labels below B; reports
include both maximum budget and exact query count. Every bundle's contribution
is recoverable from its phase. The complete teacher simulation needed to create
states is separately counted as simulator steps, including actions not acquired
as supervised examples at the final truncated state. This study's primary cost
is supervised information delivered to the learner, not all computation of the
simulator or the economic cost of a teacher invocation.

One max-budget library per teacher/seed is generated and hashed once, shared
across architectures. Each model only accesses its prescribed nested prefix.
The experimental acquisition total counts that cached library once; each
model's supervised-access budget counts its prefix. No final labels enter a
cache. The caches contain teacher states, never final states. Replayed labels
do not increase the query budget. Query types have unequal semantics/cost;
component counts and simulator costs remain separate.

Each training record includes label/component counts, independent trajectories,
graphs, distinct local/global observations, prefix/cache bytes and hashes,
presentations/reuse counts, effective passes, updates, CPU/wall time, zero GPU
time, trainable parameters and RNG states in the checkpoint. Sampling with
replacement has no completed-epoch interpretation: `epochs=null` is explicit.
Local-state identity includes the actual random inputs. Trajectories are
correlation clusters; root labels are not independent experimental replicas.
Distinct-local-state counts refer to stored rooted observation tensors, rather
than a minimal quotient after removing every feature ignored by a learned core.
Full FLOPs are unmeasured and marked missing, never synthesized from time.
A separate post hoc Torch profiler calibration estimates supported dominant
forward/backward arithmetic by phase and exact update counts; it explicitly
excludes encoder, interpreter, optimizer and unsupported elementwise/reduction
operations. `flops-estimates.json` is an estimate of those operations, not a
complete hardware instruction or total-FLOP measurement.

## Seeds, splits, families and independence

Five training seeds: 83101, 83102, 83103, 83104, 83105. Each controls new neural
initialization, independently generated training graphs/tapes and minibatches;
the modules have zero dropout. Python/NumPy/Torch states and the batch RNG are
saved. Every budget starts from a new construction of the same seed's initial
weights, without a previous-budget checkpoint. Nested datasets and matched
initialization within a seed support paired curves; budgets are not independent
training replicas. Architectures share that seed's teacher library.

Training uses n=8,12,16,24 and a predetermined rotation of cubic, irregular,
bipartite, path, cycle and symmetric ladder families. Development uses 28
executions per seed, n=28,36. Final confirmation uses 1,000 executions per seed,
n=32,48,64. The seven evaluation strata rotate path, cycle, cubic, irregular,
bipartite, circular ladder symmetry and disconnected K4 components. K4 unions
are adversarial four-color-necessary cases and valid disconnected native
inputs. The first six strata get 143 final runs and K4 gets 142. All graph
generation seeds, relabelings, edge lists, fingerprints and tape/ID seeds are
preregistered. Each run has a fresh independent random namespace; deterministic
families can recur as isomorphism classes within a split. Vertex-count sets are
disjoint across splits, proving no cross-split graph identity even up to
isomorphism. Evaluation runs are independent of training, and random tapes/IDs
are fresh per case. This is a finite stratified distribution, not all graphs.

## Native clocks and prospective horizons

| Teacher | One native step | Deadline H(n) | Exact justification |
|---|---|---|---|
| LUBY | Full participation/proposal/confirmation trial | ceil(log(10^-12/n)/log(3/4)) | Ideal conditional per-node progress ≥1/4; union-tail bound |
| JOHANSSON | Full always-active trial | ceil(log(10^-12/n)/log(7/8)) | Conservative progress ≥1/8 on this domain |
| COLOR_TRIALS | Full preference subroutine | ceil(log(10^-12/n)/log(229/256)) | Conservative progress ≥27/256 |
| JP | Simultaneous strict-priority-maxima step | n | At least one new fixed vertex |
| BE_BASE_D3 | One Linial or KW reduction | 15 | Two initialization plus thirteen reductions |
| SPECULATIVE_GM | One synchronized assign/detect slot, or serial repair | 2 ceil(n/min(4,n))+n | At most n marked repairs |
| GREEDY | One scheduled vertex write | n | Exactly n writes |

Randomized deadlines are conservative *evaluation deadlines*, not deterministic
termination proofs; mathematical tail statements assume ideal independent
randomness. The finite float32/PRNG implementation is also checked empirically.
No student's failure changes a deadline. Exact teachers must complete every
development and final case under these deadlines before preregistration closes.
A failed exact tape stops the protocol: it cannot be replaced by a successful
one. All native steps and within-teacher comparisons to the exact program are
retained. Cross-teacher native step counts are not uniform communication costs.

## Optimization and supervised budgets

The full initial grid is 1,000; 3,000; 10,000; 30,000; 100,000; 300,000; 1,000,000.
Each teacher × three architectures × five seeds receives the same selected
grid and 500 Adam updates, learning rate .001, batch 64. Updates cycle uniformly
through phases observed in the acquired prefix; within a phase, stored root
bundles are sampled with replacement. This predeclared strategy exposes rare
phases without final-guided resampling. It does not guarantee every acquired
label is used; actual visited counts are reported. The same loss weights and
interface apply to all three architectures. No early stopping or intermediate
checkpoint selection uses final information.

For fixed B=3,000, NORMALIZED LUBY/JP/BE controls use 1,500 instead of 500 updates
for all five seeds. Small and no-message controls use the same B and 500 updates
on those teachers. Thus supervision, optimization and capacity are not
automatically identified as the cause of any failure. Parameter counts and
time are separate from update counts.

## Pilot, resource selection and freeze chronology

The isolated pilot uses seed 99101 and a distinct split, 600 acquired labels,
12 updates, and seven timing cases for each of 21 teacher/architecture packages.
Its CPU implementation consumes zero GPU-process hours, below the allowed
12-hour pilot ceiling. Technical tests check counts, repeatability, graph/tape
separation, ID transforms, exact/native invariants, missing-case rejection and
the strict gate. Pilot accuracies are never used to select main cells.

An initial conservative 30,000 cap was expanded **before main training** after
timing showed the full grid feasible. Its superseded timing design is retained
under `pre-freeze/`. Three CPU workers reduce memory pressure; the prospective
total allocation is 7,200 wall seconds, with 1,200 reserved for confirmation
and 600 for controls. The selection uses a factor-two timing allowance and
records estimates for every grid prefix, queries, datasets, RAM and disk.
Final exact trajectories already checked during the resource adjustment are
reused unchanged; no student had accessed any confirmation result. The final
design and source/config hashes are frozen only after exact validation. Runtime
and any resource-limit deviations must be disclosed in the results.
This is a local prospective freeze with executable hash checks, not an
independently timestamped external preregistration. The frozen artifacts were
committed publicly after main training had started; that distinction is retained
for audit rather than claiming an earlier external registration.

## Strict PASS and candidate selection

For a fixed policy and its frozen E_s, PASS=1 iff **every** execution has a
proper final coloring, palette membership, legal protocol invariants,
termination by its deadline, no technical error, no cutoff and no incorrect
terminal state. GM temporary speculative conflicts are legal; BE improper
auxiliary colorings are immediate invariant failures. Unsafe actions are
applied and recorded as failures. No correctness filter, external repair,
failed-case replacement or success-conditioned resampling is used.

All requested case IDs must appear exactly once. Missing/duplicated rows fail
the gate. Five-seed PASS requires all five policies to pass all 1,000 final
executions: 5,000 successes, with **five separate policy summaries**. Always
report 0/5 through 5/5, final attempted/missing status, and failure types.
An unattempted seed is not an observed failing execution. A 95% functional
threshold is never used.

Train/development curves are examined in grid order. Select the first budget
at which all five seeds have 28/28 development executions, freeze the five
checkpoint hashes and configuration, then confirm them **once**. Continue
recording the predeclared development grid, including later regressions;
those curves do not change the first candidate. Before any final execution,
the candidate file is immutable and a consumed-attempt marker is written.
An interrupted final attempt cannot automatically retry. A failure is NOT
CERTIFIED with all rows retained. No later budget is selected using that final
failure. A new certification protocol would need independent final evaluation.

The reported B* is the smallest **validated** grid budget under this candidate
procedure. Earlier development failures are not independently final-tested
policies. If no candidate passes, report an unachieved protocol threshold
B*>B_max, explicitly scoped to this optimization/selection procedure; this is
not mathematical impossibility, true monotone threshold estimation, or proof
that all untested policies below B_max fail. Missing budgets are never filled.

## Statistical interpretation

For zero failures in 1,000 independent runs of one frozen policy, the one-sided
95% binomial lower bound is 0.05^(1/1000)=0.9970087504549047. Report separately
per policy, never using n=5,000 for a single policy. With fixed graph-family
strata, execution probabilities can differ: the same zero-failure lower bound
also conservatively bounds the allocation-weighted mean success probability
by independence and the arithmetic–geometric mean inequality. An ordinary
homogeneous-binomial model for all strata is not assumed to be literally true.
This is not a guarantee over all graphs or a simultaneous 95% guarantee over
all policies/cells. The five seeds are training replicas; execution rows
estimate each fixed policy's performance and cannot inflate that replication.

Plots show all five unsmoothed development curves against log B and an explicit
100% boundary. Fidelity/transition counts, native steps, failures, family/size
results and exact-teacher efficiency are distinct outcomes. Nonmonotonicity
and right-censored, unachieved protocol thresholds remain visible. Five seeds
provide limited information about the distribution of possible trainings.

## Separate interactive and distributional experiments

The one-aggregation DAgger-like control uses NORMALIZED LUBY/JP/BE, B=3,000,
five seeds, and the same total 500 updates: 250 on 1,500 passive labels,
acquire up to 1,500 new labels on that frozen warm policy's states, then 250
updates on the aggregate. Adam is reset at aggregation and this difference is
disclosed. All newly acquired components count, including from failed learner
trajectories. Queries require proper partial colorings; BE additionally
requires proper bounded auxiliary colors. Illegal states terminate that
trajectory without a teacher query or repair and are recorded; new trajectories
are new training acquisitions, not replacements of evaluation failures.
GM is excluded because arbitrary speculative history admissibility is not
proved. These controls use development only and cannot claim final PASS.

Global-law analysis is separate from the functional gate. The existing robust
teacher-specific native law test remains preserved; using it on a new cohort
requires a separately documented hypothesis family, valid status encoding and
teacher-specific reference banks. Nonrejection never proves equality. No
distributional conclusion for the new query-budgeted cohort is inferred from
historical tests or from this functional confirmation.

The separate `sample-complexity-law.json` supplement fixes two small graphs,
128 independent trajectories per bank, 9,999 permutations and a complete
210-hypothesis Holm family. Checkpoints are selected by first common perfect
development budget, otherwise the maximum grid budget, without consulting final
confirmation. The native K4 robust allocator and split selection/evaluation
instrument are reused unchanged; fewer than four successful rows gives p=1.
Futility sets all reported p-values to one unless the full permutation budget
is completed. It cannot generate a rejection.

The small-model supplement in `sample-complexity-capacity.json` tests all seven
budgets for LUBY/JP/BE at the same 500 updates and five seeds. Its B=3,000
checkpoints are the existing capacity controls, reused with their identities.
Other budgets start afresh. It selects one development candidate per teacher
and uses an independent namespace for supplemental 1,000-case-per-policy final
batteries, validating those tapes with exact teachers first. It is exploratory,
has separate frozen checkpoints/consumption markers and never revises a primary
result or the main final battery.

After the development-only optimization control, an additional exploratory
protocol (`sample-complexity-optimization.json`) freezes both existing 500- and
1,500-update NORMALIZED checkpoints at B=3,000 for **all three** previously
chosen control teachers and five seeds. It verifies identical initialization
and label-prefix hashes, then uses a new independent final namespace. Both
arms see the same 1,000 cases per seed, exact-validated first. All outcomes and
paired seed summaries remain available. Its consumption markers allow one
attempt; no weights, budgets, primary statuses or earlier final batteries are
changed. A successful control validates that fixed optimizer/label point, not
a newly estimated minimum B* or a repair of the primary certification.

## Reproduction

For a new scientific replication, use the documented Torch/NumPy environment
and an isolated checkout/namespace:

```bash
python scripts/replicate-sample-complexity.py --directory /absolute/path/to/new-replication --namespace independent-replication-01
```

The wrapper preserves the published checkout and frozen exports. It runs the
stages below in the **new** experiment. For the published cohort, fetch the
runtime and run verification/reporting instead of overwriting pilot/design:

```bash
python -m pytest -q tests/test_samplecomplexity.py
PYTHONPATH=src python -m dcl.samplecomplexity.runner pilot
PYTHONPATH=src python -m dcl.samplecomplexity.runner design
PYTHONPATH=src python -m dcl.samplecomplexity.runner validate
PYTHONPATH=src python -m dcl.samplecomplexity.runner train
PYTHONPATH=src python -m dcl.samplecomplexity.runner controls
PYTHONPATH=src python -m dcl.samplecomplexity.runner confirm
PYTHONPATH=src python -m dcl.samplecomplexity.optimization
PYTHONPATH=src python -m dcl.samplecomplexity.capacity
PYTHONPATH=src python -m dcl.samplecomplexity.law
PYTHONPATH=src python -m dcl.samplecomplexity.diagnostics
python scripts/estimate-sample-complexity-flops.py
PYTHONPATH=src python -m dcl.samplecomplexity.report
```

Frozen results are replayed from their hashes rather than overwritten. A fresh
experimental replication should use a distinct namespace/root and independently
frozen confirmation battery; it must not change this consumed preregistration.
