# Local palettes and statistical distinguishability

Worktree `/workspaces/AlgoLearning/ucgnn-local-palette-study`, branch
`research/ucgnn-local-palettes`. This extension reuses the 30 paper-study
epoch-120 checkpoints; it does **not** retrain or overwrite historical results.
No remote publication is authorized for this extension.

Start with [the architecture audit](docs/LOCAL_PALETTES_ARCHITECTURE_AUDIT.md)
and [the preregistered protocol](docs/LOCAL_PALETTES_STATISTICAL_PROTOCOL.md).
Current progress is in [STATUS.md](STATUS.md). New results live under
`results/ucgnn-local-palettes/`, separate from preceding study data.

Commands (from this worktree; implementation follows the protocol commit):

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python
$PY -m dcl.localpalettes status
$PY -m dcl.localpalettes run
$PY -m dcl.localpalettes resume
$PY -m dcl.localpalettes audit
$PY -m dcl.localpalettes report
```

`run`/`resume` are synchronous resumable coordinators. The actual persistent
launch was verified with per-task receipts and 152 exact-teacher cases before
accepting the queue. Two tmux windows run the coordinator and final-stage
supervisor. Inspect them with:

```bash
tmux list-windows -t ucgnn-local-palette-study
tail -n 12 data/ucgnn-local-palettes/coordinator.log
tail -n 12 data/ucgnn-local-palettes/final-stage.log
```

The supervisor waits for the initial coordinator to exit, runs the separately
preregistered five original-E1 law tasks within the same resource ceiling,
audits checkpoints, generates CSV/tables/figures/compressed records, reruns
tests after releasing the GPU lock, and compiles LaTeX. Its durable command is
`bash scripts/local-palette-final-stage.sh COORDINATOR_PID`. It does not publish.
After a container reboot, explicitly run `resume`, then:

```bash
$PY -m dcl.localpalettes.finalize
$PY -m pytest -q
/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic -X compile --keep-logs paper/ucgnn/main.tex
```

Incomplete task outputs resume at the next missing graph/state. Workers inherit
the GPU flock, preserving exclusion if their parent exits. Recovery conservatively
charges an interrupted task's duration (up to its 30-minute cap), rather than
resetting the 12-hour budget. Failed technical tasks remain explicit for inspection.
Initial full-suite tests that touched the held GPU lock were deferred; their
original XML is preserved, and `all-tests.xml` is replaced by the final run.

`results/ucgnn-local-palettes/FINAL_REPORT.md` is generated after the full queue;
`REPORT.md` may be generated while it is partial. `candidate-counts.npz` and
`graph-records.jsonl.gz` are versionable compact data exports. Large raw independent
classifier trial arrays stay in runtime with their hashes in
`candidate-array-manifest.json`; their named RNG seeds reproduce them.

Reboot requires explicit resume.
The historical [E1 entry](E1.md) remains available and unchanged.
