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
launch is documented in STATUS once verified; reboot requires explicit resume.
The historical [E1 entry](E1.md) remains available and unchanged.
