# UCGNN paper study — completed

Updated UTC: 2026-10-21T10:59:59.302951+00:00

All **157/157 tasks completed**. Accumulated active GPU-process hours:
**6.1717 / 36**. No study GPU worker or tmux session remains.

Protocol commit `023deaf`, unchanged SHA256:
`6bb9af85893a7c48c41362a875cbbc0cd1f0d38b989785c0d931f68510f2087a`.

- 32 new networks: 30 primary (six treatments × five seeds), own-only and cyclic teacher.
- 85 checkpoint views, 18,363 graph evaluations, 3,520 controlled progress rollouts.
- Zero unexecuted evaluation cells, OOMs or evaluation timeouts.
- NORMALIZED / DeepSets / Transformer: each 1,520/1,520 correct primary executions.
- RAW: 1,174/1,520; SUM: 324/1,520; MEAN: 681/1,520. Negatives preserved.
- Full suite: 259 passed. Initial manuscript compiled; scientific limitations explicit.
- Historical sources/checkpoints unchanged; local commits only, no remote push.

Final report: `results/ucgnn-paper-study/FINAL_REPORT.md`.
Machine coverage audit: `results/ucgnn-paper-study/completion-audit.json`.
English manuscript: `paper/ucgnn/main.tex`, `paper/ucgnn/main.pdf`.
The earlier partial handoff remains in `results/ucgnn-paper-study/CLOSEOUT.md`.

```bash
bash experiments/ucgnn-paper-study status
env PYTHONPATH=src OMP_NUM_THREADS=1 /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m dcl.paperstudy.audit
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m dcl.paperstudy.finalize --once
```

`resume` recovers genuinely incomplete tasks after interruption. This manifest
is complete, so invoking it does not start additional training. A fresh-run
reproduction worktree and exact commands are documented in `FINAL_REPORT.md`.
No automatic recovery after a host/container restart is claimed.
