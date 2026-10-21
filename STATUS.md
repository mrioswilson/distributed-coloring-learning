# UCGNN paper study

Updated UTC: 2026-10-21T03:27:10.120325+00:00

Protocol frozen at `023deaf`; SHA256 `6bb9af85893a7c48c41362a875cbbc0cd1f0d38b989785c0d931f68510f2087a`.
Queue: **running**; accumulated active GPU process hours: 1.0136 / 36.
Task states: `{"complete": 12, "pending": 18, "running": 1}`.

Running `train-raw_endpoints-s41002` (pid 3771848); log `data/ucgnn-paper-study/logs/train-raw_endpoints-s41002-attempt1.txt`.
Progress: `{
  "complete": false,
  "elapsed_seconds": 204.04355476074852,
  "epoch": 95,
  "run": "raw_endpoints-s41002",
  "updates": 14060
}`

Historical directories are read-only. One GPU worker under global flock;
RTX 4060 8 GiB; process allocator cap 4 GiB; no external services stopped.

```bash
tmux attach -t ucgnn-paper-study
/workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m json.tool data/ucgnn-paper-study/manifest.json
tail -n 15 data/ucgnn-paper-study/logs/*.txt
bash experiments/ucgnn-paper-study resume
```

tmux survives disconnection. After a container restart run the resume command;
there is no claim of automatic host/container reboot recovery.

Training pilots and all promotion decisions precede confirmatory evaluation.
Final results, mathematical analysis and manuscript status are in
`results/ucgnn-paper-study/REPORT.md` when generated.
