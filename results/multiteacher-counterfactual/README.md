# Reproduction and evidence boundaries

The Spanish P1–P14 report is `docs/MULTITEACHER_COUNTERFACTUAL_RESULTS.md`.
This directory preserves all new experimental runtime. Large raw rows,
checkpoint payloads and arrays are supplied in the authenticated archive;
summary results, frozen plans, source hashes, test logs and figures are also
versioned. Nothing replaces `results/sample-complexity/` or earlier studies.

The nine representative CF packages are in `plans/successful-policies.json`:
BE/ColorTrials at4000 and GM at500, five independent training seeds each.
Every policy has1000/1000 native final successes. Their combination with the
twelve historical certifications is21/21 coverage, with protocols separated.
Matched controls also close all nine. All195000 final executions, including
21 failed executions at earlier points, remain available. There are170
distinct final checkpoints; evaluations repeated on new batteries do not
create new training replicas.

## Restore

The delivery archive contains `source.bundle`, this entire runtime directory,
an inventory with every file's SHA-256, and the current paper/report. A local
Git clone from the bundle restores the complete source history, including
the `61f39760` baseline and the research branch, without a network push:

```bash
git clone /path/to/source.bundle research
cd research
# Copy/extract the archive's results/multiteacher-counterfactual/ into results/.
python -m venv .venv
.venv/bin/pip install -e '.[dev,study,paper]'
```

`environment.json` records the exact tested environment (Python3.11.2,
Torch2.11.0+cu128, NumPy2.4.6, NetworkX3.6.1). Bitwise training replay is
verified in this CPU environment; different library/hardware kernels can
change numerical rounding. For the preceding sample-complexity runtime, use
the preserved `scripts/fetch-sample-complexity.py` and its public release.
The 21 authenticated early M6 artifacts are also copied byte for byte under
`historical-recovery/m6-originals/` in this archive. Pedro's40 missing original
checkpoints are not imputed or supplied as new weights.

Historical replays accept relocated runtime roots and a fresh output directory:

```bash
PYTHONPATH=src python scripts/audit-multicf-history.py replay-m6 --historical-root results/multiteacher-counterfactual/historical-recovery/m6-originals --output-dir work/m6-new-replay
PYTHONPATH=src python scripts/replicate-be-multicf.py --original-runtime data/sample-complexity-v1 --output work/be-new-replay
```

## Validate and rebuild without training

Run commands from the repository root with the tested environment:

```bash
PYTHONPATH=src python scripts/verify-multicf.py
PYTHONPATH=src python scripts/report-multicf.py
PYTHONPATH=src python scripts/plot-multicf.py
PYTHONPATH=src python scripts/build-multicf-paper.py --compiler /path/to/verified/tectonic
PYTHONPATH=src python -m pytest -q
```

`verification.json` recomputes every gate, validates frozen model/battery/raw
row hashes, verifies all stored counterfactual/native component labels and
student-specific query caches, and checks unchanged teacher/inference files.
It authenticates original M6 files through their local path or archived copy.
The final suite passed668 cases. `paper-build.json` records the pinned
Tectonic0.17.0 hash, PDF hash, undefined-reference/glyph/overflow checks and
the precise visual-review scope. `pre-counterfactual-main.tex/pdf` preserves
the preceding paper byte for byte.

## New training and exact restart

Use a new checkout/output location for a new scientific replication. A
namespace denotes an immutable complete configuration; changing training
sizes, state domains, sampling or final policy requires a fresh namespace
and appropriate directory. All main/supervision/coverage/own-proposal runs
save authoritative `last.pt` every250 updates and frozen intermediate
checkpoints. Payloads include Adam, consumed RNG states, label visits and
query caches. Repeating the same training command resumes exactly when the
recorded source and configuration match, and refuses a changed identity.

The executed primary commands were:

```bash
PYTHONPATH=src python -m dcl.multicf.runner pilot
PYTHONPATH=src python -m dcl.multicf.runner train --teachers BE_BASE_D3 SPECULATIVE_GM --methods counterfactual matched passive
PYTHONPATH=src python -m dcl.multicf.runner train --teachers COLOR_TRIALS --budget 30000 --methods counterfactual matched passive
PYTHONPATH=src python scripts/confirm-multicf.py screen --teachers BE_BASE_D3 COLOR_TRIALS --plan be-ct
PYTHONPATH=src python scripts/confirm-multicf.py freeze --plan be-ct
PYTHONPATH=src python scripts/confirm-multicf.py run --plan be-ct
PYTHONPATH=src python scripts/confirm-multicf.py screen --teachers SPECULATIVE_GM --plan gm
PYTHONPATH=src python scripts/confirm-multicf.py freeze --plan gm
PYTHONPATH=src python scripts/confirm-multicf.py run --plan gm
```

Official final attempts use `confirm-multicf.py` and the300-case screen.
The generic runner's earlier `certify` helper belongs to the superseded
60-case pilot selection and was not used for the reported confirmations.
Consumption markers prevent automatic reattempts of incomplete or failed
final cohorts. Completed attempts return their preserved records. A new
method/final-guided follow-up must use a new frozen namespace; never delete
a marker or overwrite a failed candidate.

## Follow-ups and supplements

`be-cf-optimization-multicf.py` and `ct-cf-optimization-multicf.py` each have
`freeze` and `run` stages. They use existing1500/4000 checkpoints on independent
new batteries, without new training labels. Read
`audit/ct-followup-rationale-erratum.json`: the CT plan's frozen `reason`
mistakenly calls its initial E1 failure1500; that failure was500. The actual
follow-up always compares1500/4000. Its historical source/plan is preserved
and the erratum does not change any checkpoints, seeds, rows or results.

Additional reproducible scripts are `supervision-curves-multicf.py`,
`information-multicf.py`, `replay-information-multicf.py`,
`own-proposal-pilot-multicf.py`, `diagnose-all-multicf.py`,
`stress-multicf-oracles.py`, `audit-shared-heads-multicf.py`,
`regress-gm-multicf.py`, `posthoc-multicf-failures.py`, and
`generalize-multicf.py {laws,transfer} --plan successful-policies`.
The coverage control uses the runner's `--no-coverage --methods matched`
at1500 updates and500/1500 checkpoints. Exact commands and identities remain
in the run configuration/record files. Twenty information replays reproduce
the original ablation weights while retaining intermediate Adam/RNG states.

Labels acquired, model access, label presentations, updates, simulation and
process times are separate in `compute-cost.json`. Prefixes reuse labels;
own queries are not free. Shared architecture heads and paired seeds are
correlated comparisons. Law/transfer results never enter training or model
selection. PASS is finite functional evidence, not universal correctness or
identity of trajectories or marginal coloring laws.
