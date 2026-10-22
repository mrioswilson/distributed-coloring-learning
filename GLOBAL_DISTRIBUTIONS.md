# Complete distributed-coloring laws

Worktree: `/workspaces/AlgoLearning/ucgnn-global-distribution-study`.
Branch: `research/ucgnn-global-distribution`. Base: 25cbeb5. Protocol: 80195c5.
No new training and no remote push.

Read `docs/GLOBAL_COLORING_DISTRIBUTION_PROTOCOL.md`,
`docs/GLOBAL_COLORING_DISTRIBUTIONS_THEORY.md` and the final report in
`results/ucgnn-global-distribution/FINAL_REPORT.md`.

## Reproduce or resume

Use the original fifteen frozen files recorded in
`results/ucgnn-global-distribution/checkpoints.json`. They remain in the
paper-study worktree; the archive does not redistribute neural checkpoints.
Verify hashes before use. No weight selection or training is required.

From this worktree:

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python
$PY -m pytest -q tests/test_globaldist.py tests/test_globaldist_integration.py
$PY -m dcl.globaldist.run
$PY -m dcl.globaldist.supplement
$PY -m dcl.globaldist.analyze
$PY -c "from dcl.globaldist.analyze import independent_power; independent_power()"
$PY -c "from dcl.globaldist.analyze import localized_control; localized_control()"
$PY -m dcl.globaldist.audit
$PY -m dcl.globaldist.report
$PY -m pytest -q tests/test_globaldist.py tests/test_globaldist_integration.py tests/test_localpalettes.py tests/test_graphs.py tests/test_oracle.py
/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic -X compile --keep-logs paper/ucgnn/main.tex
```

`run` is synchronous and reuses completed, hashed bank receipts. It takes a
nonblocking GPU lock shared with the previous project studies. It never
kills other services. The immutable `freeze.json` guards the protocol and
exact graph/checkpoint lists. Do not invoke `freeze` again in the completed
study. An explicit fresh study directory is needed for a wholly independent
replication; copy the source and immutable manifests and start with no queue
or bank files. Versioned named seeds reproduce this realization. Changing
sample sizes, kernels or graphs requires a prospectively separate protocol.

## Data and interpretation

`data/ucgnn-global-distribution/banks/*.npz`: rows are independent complete
runs, columns are labeled vertices. Arrays: colors (including partial vectors
on failures), status (0 valid, 1 invalid complete, 2 cutoff, 3 technical),
rounds, execution seeds, elapsed seconds, error descriptions, first initial
PCG64 state and SHA256 of final RNG states. Local lists are `degrees+1` in
the immutable graph manifest. Graph edges are both directed incidences;
hashes also include vertex count. One bank is shared only within the same
graph/generator/stage, never across different probability spaces.

`queue.json` and receipts are atomic and persistent. After an interruption,
`run` resumes missing banks; conservatively charged downtime can exhaust the
budget, in which case pending cells remain explicit. No automatic reboot
restart is promised. The final archived state has no pending work.

`results/ucgnn-global-distribution/primary.json` retains actual sample counts,
U-statistic, empirical MMD norm, conservative population-MMD CI, p, Holm p,
checkpoint, graph, size and degree. CSV tables are compact exports; JSON
retains list-valued intervals and status counts. `rejected.csv` and
`not-rejected.csv` retain both outcomes. `exact-*.json`, `small-tv.csv`,
`power.csv`, `classifier.json`, `structural.json`, `controls.json`,
`success.json`, `audit.json` and six main plus two supplementary PDF/PNG figures are regenerated from
raw archived arrays. Classifier and power banks have distinct RNG namespaces.

Every checkpoint is analyzed separately. Architecture summaries aggregate
test results and effects, never vectors from different policies. Graphs are
fixed blocks, not independent replicas of graph populations. Nonrejection
is not equality. Power estimates are conditional on empirical banks and
have limited scope. Large-graph MMD is not exact TV.

The regression command above is read only. Several unrelated historical tests
exercise training/resume by fitting temporary neural models; they are excluded
to honor the no-training instruction. The changed modules and reused local
palette adapter receive the unit/integration coverage listed above.

Prospective supplements are documented separately in
`docs/GLOBAL_COLORING_POWER_SUPPLEMENT.md` and
`docs/GLOBAL_COLORING_LOCALIZED_CONTROL_SUPPLEMENT.md`; they leave the
primary 795-hypothesis family unchanged. Figure 3b and
`independent-power.csv` contain true independent repeated-experiment power
on path3. `localized-positive-control.csv` records a known-TV star
alternative whose kernel effect is analytically small.

Figure 3c and independent-power.csv also contain independent population-power
experiments at m=100,250 on star-d200 and regular-n256-d100 for seed 41000.
Each point uses twenty disjoint trials; intervals are pointwise and nominal,
with no claim about the Holm-adjusted primary family or all training seeds.
