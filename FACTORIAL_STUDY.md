# Factorial coloring-distribution study

Worktree: `/workspaces/AlgoLearning/ucgnn-size-degree-study`.
Branch: `research/ucgnn-size-degree`. No training and no remote push.
The prior robust study remains unchanged in its own worktree.

The protocol was committed at `5633b547682b129bb0f0baa81f6ad9fe1012eb25`
before generating confirmation banks. Read
`docs/FACTORIAL_SIZE_DEGREE_PROTOCOL.md`, `results/ucgnn-size-degree/freeze.json`
and `protocol-commit.json` for the exact design and timing proof.

Main cohort: 16 size/degree cells, three independent regular graphs each,
15 fixed checkpoints, 720 hypotheses. Separate secondary cohort: eight
irregular graphs and 120 hypotheses. Independent historical-graph replication:
15 hypotheses. Each of 16 generators attempts 625 runs per graph: first500
for the test and preregistered events, last125 for selected-witness estimation.
The omnibus V2 and original policies are frozen; no canonicalization, global
degree input or new safety rule is introduced. Primary laws condition on
proper completion by round128. All statuses are preserved.

## Live state and resumption

Persistent runtime is `data/ucgnn-size-degree/`. Bank receipts, queues and
`budget.json` make resumption idempotent and preserve the 12 GPU-process-hour
cap. The shared GPU lock prevents duplicate study workers. The coordinator
resumes confirmation then read-only diagnostic trajectories:

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python
$PYTHON scripts/factorial-gpu-workflow.py
```

Do not launch this concurrently with the existing
`ucgnn-size-degree-gpu` tmux session. CPU contrast analysis can run separately:

```bash
$PYTHON -m dcl.sizedegree.analysis --watch --workers 4
```

The CPU watcher waits for all855 comparisons before adjusting each external
family and writing final event and factorial summaries. `gpu-workflow.json`
records completion or budget incompleteness. A physical reboot requires an
explicit resume; reboot scheduling is not claimed.

A separate exploratory candidate-channel intervention was preregistered in
`docs/FACTORIAL_CANDIDATE_INTERVENTION_PLAN.md` at commit
`8d538766cd038190e38c1d7305c982124c20aa71`, before any intervention banks.
Its worker waits for the original GPU workflow, uses the same lock and total
12-hour ledger, and chooses its fixed sample count only from a timing pilot:

```bash
$PYTHON -m dcl.sizedegree.intervention worker
```

Do not duplicate the existing `ucgnn-size-degree-intervention` session. Only
the candidate channel is replaced; learned participation and COMMIT are
unchanged. Paired intervention/teacher draws are never included in the
original independent two-sample tests. A coupling can bound TV from above
for this intervened policy only; it does not establish equivalence for the
unmodified checkpoint. Missing or budget-skipped cells remain explicit.

## Reproduction and derived outputs

After all sampling and contrast jobs complete:

```bash
$PYTHON -m dcl.sizedegree.intervention analyze
$PYTHON -m dcl.sizedegree.precision
$PYTHON -m dcl.sizedegree.capacity
$PYTHON -m dcl.sizedegree.exploratory
$PYTHON -m dcl.sizedegree.audit
$PYTHON -m dcl.sizedegree.summaries
$PYTHON -m dcl.sizedegree.figures
$PYTHON -m dcl.sizedegree.report
/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic paper/ucgnn/main.tex
```

`results/ucgnn-size-degree/` contains all raw/adjusted p-values, events and
simultaneous bounds, 1,800 contrasts, same-sample old/new comparisons,
per-run structural metrics, held-out witnesses, per-color profiles and
proposal/COMMIT records. `figures.py` generates the nine requested PDF/PNG
figures, additional diagnostics and a 16-page full-cohort occupancy PDF.
The report explicitly answers P1–P10; the new provisional manuscript section
preserves all historical sections. Full interpretation follows the completed
audit, not an intermediate count of rejections.

In a delivered archive, raw original algorithm banks, the new raw banks,
all15 exact checkpoints, source history and document compiler are included.
`scripts/replay-factorial-study.py` relocates only path bindings in memory.
For a freshly unpacked source directory without Git metadata, initialize
metadata from the local bundle once, then run CPU audit/reproduction:

```bash
$PYTHON scripts/replay-factorial-study.py audit --restore-git-metadata
$PYTHON scripts/replay-factorial-study.py analysis --workers 4
```

The original frozen JSON files and source remain byte-identical. To recompute
all Monte Carlo tests, use an isolated copy and move only that copy's generated
`results/ucgnn-size-degree/tests/` out of the way before invoking analysis.
Do not remove raw banks, protocol metadata or historical results. Reproduction
may change timestamps and measured costs; fixed seed statistics and decisions
are the quantities audited for equality. Environment versions and every input
digest are recorded in the delivery manifest.

The CP/Hoeffding event-contrast guarantees concern independent complete runs
on the fixed cohort. They do not establish a population trend from three
graphs or five trainings. Degree and palette size are confounded in regular
graphs, grid candidate masses are numerical diagnostics, and nonrejections
never establish equality or a TV upper bound.
