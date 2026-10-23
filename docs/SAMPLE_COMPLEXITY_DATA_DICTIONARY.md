# Controlled-supervision data dictionary

All paths below are relative to `results/sample-complexity/` unless stated
otherwise. Main training runtime is separate under `data/sample-complexity-v1/`.
Published frozen files describe the completed experiment; a new replication
uses an isolated checkout and independent namespace.

| Artifact | Meaning and units |
|---|---|
| `design.json`, `preregistration.json` | Configuration, selected budget grid, software/source hashes, frozen battery identities and pilot resource estimates. This is a local prospective freeze, not external timestamp registration. |
| `batteries/{development,final}-s*.json` | Case IDs, families, sizes, exact edge lists, graph fingerprints, graph seeds and independent private-tape/ID seeds. Case IDs are scoped by experiment namespace, split and training seed. |
| `data/TEACHER/s*.json` | Max-budget teacher-library manifests: component acquisitions, root bundles, trajectories, distinct observations, graph identities, simulator steps, RNG states and compressed-data digest. The library is acquired once and shared across architectures/budget prefixes. |
| `training.json`, `training/TEACHER/ARCH/s*-b*.json` | One record per primary teacher × architecture × seed × maximum label budget: exact acquired prefix, reuse, optimizer updates, initial/final weight hashes, checkpoint digest, CPU/wall seconds, parameters, development summary and original prefix-fidelity diagnostic. |
| `frozen-candidates.json` | First common development-perfect budget and five frozen checkpoint digests per package. `budget=null` means no candidate. |
| `consumption/*.json` | A final attempt begins before any student final result is read. `complete=false` prevents automatic retry after interruption; completed markers retain the original result. |
| `final/TEACHER/ARCH/s*.json` | Exactly 1,000 individual records per attempted policy, including failures. There is no such file for an unattempted candidate. |
| `exact/TEACHER/{development,final}-s*.json` | Exact teacher validation on all prescribed cases and original deadlines. No difficult tape is replaced. |
| `certification.json`, `results-table.{json,csv}` | Strict five-seed classification, validated/candidate/max budgets, actual per-seed queries, failures, compute, information regime, first development passes and regressions. NOT CERTIFIED includes failed confirmation and absence of a development candidate. |
| `controls.json` | Development-only fixed-label optimizer, small/no-message and interactive controls. Interactive records include 250+250 updates, actual query counts and failed acquisition trajectories. |
| `optimization-confirmation/` | Additional exploratory fixed-B paired final protocol: both 500/1,500-update arms, matched initial/prefix hashes, independent batteries, exact references, one-attempt markers and every student run. It cannot alter primary results. |
| `capacity/` | Seven-budget small-model curves, including identity-preserved reuse of B=3,000 controls, independent exact-validated final batteries and candidate/confirmation status. No candidate means no student final attempt. |
| `law/` | Independent 128-trajectory banks on two fixed small graphs, 210 planned hypotheses, native K4 robust tests, 9,999-permutation limit, conservative futility, complete-family Holm adjustment and raw statuses. These are conditional-on-success laws. |
| `full-fidelity.json` | Every primary checkpoint on complete development teacher trajectories. Records action agreement, actual-state transition agreement and phase/bound strata. Evaluation reference labels do not enter training budgets. |
| `difficult-teacher-diagnostics.json` | Complete common-state phase/component audit at the maximum budget for JP/BE/GM, plus first autonomous development failure phases for all main budgets. |
| `structural-failure-witnesses.json` | Post hoc analysis of frozen GM candidates: original failed final cases, degrees of pending nodes, an independent singleton diagnostic, and reconstructed training presentations verified against the recorded visit hash. No final model is changed or rerun. |
| `generalization.json`, `native-efficiency.json` | Per-family/size/seed outcomes and within-teacher paired native-step ratios. Failed attempted steps and successful conditional efficiency are separate. Nominal exchange-stage multipliers are not actual network latency/bit counts. |
| `model-inventory.json` | Parameters, dimensions, graph aggregation, communication depth, palette mechanism, coded rules, learned decisions and cost-source limitations. |
| `flops-estimates.json` | Post hoc supported-operation profiler estimates by phase and exact update counts. This Torch build counts addmm/mm/add/mul; unreported operations, optimizer, encoding and interpretation are omitted. It is not total FLOPs. Measured CPU time is separate. |
| `figures/`, `figure-manifest.json` | Unsmoothed, individually drawn seed curves on a logarithmic query axis; success/failure frontiers, fidelity, attempted steps and capacity comparison. Full-fidelity and training input hashes are retained. |
| `runtime-manifest.json`, `runtime-release.json` | Every released runtime member's path, byte count and SHA-256, plus archive length/digest/URL. Restoring refuses conflicting files. Historical 70-model checkpoints are excluded. |

## Query and replay fields

A **query** is one acquired label for a specified node, state and component.
Normal color phases store three components: participation, proposal and teacher
confirmation. Linial point and GM conflict mark each cost one. Availability
has no separate supervised label. Proposals for nonparticipating/retained
vertices are well-defined decomposition targets and are counted if acquired;
their difficulty/cost is not assumed equal to another component's.

The data cache contains a maximum-budget library; a model only uses its
chronological prefix. `budget` is the maximum allowed acquisition. `queries`
is the exact sum of acquired components, up to two below the maximum because
component bundles are not split. `by_component` must sum to `queries`.
`examples` counts root bundles, not labels or independent trajectories.
`component_presentations` counts replays, not new acquisitions. A label
repeated ten times is still one acquired label.

`acquired_examples_ever_used` and `repeated_example_presentations` expose how
much of the acquired library reached gradient updates. Large libraries at
500 fixed updates can contain many unused labels. `parameter_updates` is
optimizer steps; `epochs=null` explicitly records sampling with replacement,
and `effective_passes` is presentations divided by root-bundle count.

Trajectories can end at a budget prefix; `completed=false` is not an
evaluation cutoff or a success. Their label observations remain a correlated
trajectory cluster. Distinct-local counts refer to stored rooted tensors,
including actual private inputs, rather than a minimal quotient after deleting
unused channels. Distinct-global counts refer to stored native states, with
trajectory/graph identities available separately.

Runtime teacher caches contain `a0..a3` rooted node/neighbor/relation/mask
arrays, `labels`, `trajectory`, `node`, `native_step` and `global_state`.
Teacher action-component labels are `(phase, participation, proposal,
confirmation, mark, Linial point)`. Phases 0..5 are COLOR, ASSIGN, DETECT,
REPAIR, LINIAL, KW. Native color **action codes** are 0=WAIT and 1..4=write
permanent color 0..3; they must not be confused with actual color labels.
Native auxiliary action pairs are encoded with `13*x+y`; the interpreter
updates auxiliary color to `field*x+y`. GM mark actions are Boolean.

## Individual functional result fields

`proper`, `palette`, `invariants`, `terminated`, `no_technical_error`,
`no_cutoff` and `terminal_correct` must **all** be true for a run to pass.
`case_id` must appear exactly once in its prescribed battery. Palette checks
allow pending -1 internally; terminal correctness additionally excludes every
pending vertex. `colors` is the final permanent-color vector. `native_steps`
and `horizon` use the teacher-specific unit in the protocol.

`failure` can record invariant, collision, palette, incomplete, cutoff or
technical error; a technical exception is also retained as text. An incomplete
scheduled GM terminal state need not be a cutoff. GM intermediate speculative
conflicts are legal; BE's bounded proper auxiliary coloring is checked after
each reduction. No failure is repaired or replaced. Main raw rows retain final
colors and flags, not every intermediate student state; frozen code, checkpoint
and tapes support a deterministic audit replay. Such a replay is not a new
independent trial or another model-selection/confirmation attempt.

`seeds_pass` counts complete policy-level passes, never average success.
The lower one-sided 95% bound after 1,000/1,000 is 0.9970087504549047 for that
policy and declared independent evaluation distribution. Five policies are
not pooled into a 5,000-run interval for one policy. No-candidate rows have
zero confirmed seeds and no observed final outcomes.

## Full transition fidelity

Action-fidelity denominators count native action decisions (WAIT/color,
auxiliary pair or mark), rather than the three-component supervision query
count. Internal participation/proposal heads have additional component
diagnostics. These denominators must not be equated with training acquisitions.

The full audit compares every permitted root's actually updated state channel.
Linial decodes the pair using the active prime field. KW computes either the
old auxiliary value or its block-offset write. GM repair and ordinary color
steps compute either the old color or the predicted write; detection updates
the mark. Equal updated channels from a common teacher state give the same
remaining interpreter/clock/scheduler/RNG state. Consequently redundant writes
can preserve transitions despite action disagreement. Tests compare these
cases against the real interpreter. The earlier 256-decision prefix proxy
counts action-identical transitions and is preserved as such.

No new-cohort distributional equality is inferred from nonrejection. Neither
frozen empirical success nor a fidelity curve is a universal mathematical
correctness, learnability or sample-complexity theorem.
